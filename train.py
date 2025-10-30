import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from copy import deepcopy
from dataclasses import asdict
import os
from utils import Config
from Transformers import GNNTransformer


def prepare_targets_as_deltas(y_abs, last_pos, player_mask):
    """
    Convert absolute target positions to deltas relative to previous frame,
    using last_pos from the input sequence as the reference for the first frame.
    This should only be done for valid players not padded players

    Parameters:
        y_abs (torch.Tensor or np.ndarray): Absolute target positions (B, N, T_out, num_features)
        last_pos (torch.Tensor or np.ndarray): Last known positions from input (B, N, 2)
        player_mask (torch.Tensor or np.ndarray): Player validity mask (B, N)
        
    Returns:
        np.ndarray: Target deltas (B, N, T_out, num_features)
    """
    # Convert to numpy if needed
    if isinstance(y_abs, torch.Tensor):
        y_abs = y_abs.detach().cpu().numpy()
    if isinstance(last_pos, torch.Tensor):
        last_pos = last_pos.detach().cpu().numpy()
    if isinstance(player_mask, torch.Tensor):
        player_mask = player_mask.detach().cpu().numpy()
    
    B, N, T, F = y_abs.shape
    deltas = np.zeros_like(y_abs, dtype=np.float32)

    valid_mask = player_mask.astype(bool)

    # First frame delta relative to last input position for valid players
    deltas[valid_mask, 0, 0:2] = (
        y_abs[valid_mask, 0, 0:2] - last_pos[valid_mask, 0:2]
    )

    # Subsequent deltas relative to previous target frame
    if T > 1:
        deltas[valid_mask, 1:, 0:2] = (
            y_abs[valid_mask, 1:, 0:2] - y_abs[valid_mask, :-1, 0:2]
        )
    # copy over any remaining non-positional features unchanged
    if F > 2:
        deltas[..., 2:] = y_abs[..., 2:]

    return deltas

def reconstruct_absolute_from_deltas(last_pos, deltas, player_mask):
    """
    Reconstruct absolute target positions from frame-to-frame deltas.

    Parameters:
        last_pos (torch.Tensor or np.ndarray): Last known position from input (B, N, 2)
        deltas (torch.Tensor or np.ndarray): Predicted or true deltas (B, N, T_out, num_features)
        player_mask (torch.Tensor or np.ndarray): Player validity mask (B, N)
        
    Returns:
        np.ndarray: Absolute positions (B, N, T_out, num_features)
    """
    # Convert to numpy if necessary
    if isinstance(deltas, torch.Tensor):
        deltas = deltas.detach().cpu().numpy()
    if isinstance(last_pos, torch.Tensor):
        last_pos = last_pos.detach().cpu().numpy()
    if isinstance(player_mask, torch.Tensor):
        player_mask = player_mask.detach().cpu().numpy()
    
    B, N, T, F = deltas.shape
    abs_pos = np.zeros_like(deltas, dtype=np.float32)
    valid_mask = player_mask.astype(bool)

    for b in range(B):
        for n in range(N):
            if not valid_mask[b, n]:
                continue    # skip padded players
            abs_pos[b, n, 0, :2] = last_pos[b, n, :2] + deltas[b, n, 0, :2]
            for t in range(1, T):
                abs_pos[b, n, t, :2] = abs_pos[b, n, t-1, :2] + deltas[b, n, t, :2]

    # copy over any remaining non-positional features unchanged
    if F > 2:
        abs_pos[..., 2:] = deltas[..., 2:]

    return abs_pos

def masked_mse_loss(predictions, targets, mask):
    """
    Compute masked Mean Squared Error loss

    Parameters:
        predictions (torch.Tensor): Predicted values (B, N, T_out, F) or (S, T_out, F)
        targets (torch.Tensor): Target values (B, N, T_out, F) or (S, T_out, F)
        mask (torch.Tensor): (B, N, T_out) -> True where valid output

    Returns:
        loss (torch.Tensor): Scalar loss value
    """
    # squared errors
    se = (predictions - targets).pow(2).sum(dim=-1)  # (..., T)
    # apply mask
    masked_se = se * mask
    denom = mask.sum()
    if denom <= 0:
        # no valid frames in batch, don't divide by zero
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    return masked_se.sum() / denom

def masked_FDE_loss(predictions, targets, mask):
    """
    Compute masked Final Displacement Error loss

    Parameters:
        predictions (torch.Tensor): Predicted values (B, N, T_out, F) or (S, T_out, F)
        targets (torch.Tensor): Target values (B, N, T_out, F) or (S, T_out, F)
        mask (torch.Tensor): (B, N, T_out) -> True where valid output

    Returns:
        loss (torch.Tensor): Scalar loss value
    """
    device = predictions.device
    B, N, T_out, F = predictions.shape
    
    # obtain last valid position
    valid_counts = mask.sum(dim=-1)     # (B, N)
    last_idx = (valid_counts - 1).clamp(min=0)  # avoid negatives if 0 output

    # obtain predictions and targets
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, N)
    player_idx = torch.arange(N, device=device)[None, :].expand(B, N)

    preds_last   = predictions[batch_idx, player_idx, last_idx, :]    # (B, N, F)
    targets_last = targets[batch_idx, player_idx, last_idx, :]

    # Euclidean displacement error
    disp_err = torch.norm(preds_last - targets_last, dim=-1)    # (B, N)
    # ensure only valid output timesteps > 0
    valid_mask = valid_counts > 0
    disp_err = disp_err * valid_mask

    denom = valid_mask.sum()
    if denom == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return disp_err.sum() / denom


def collate_default(batch):
    Xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    player_masks = torch.stack([b[2] for b in batch], dim=0)
    target_masks = torch.stack([b[3] for b in batch], dim=0)
    y_masks = torch.stack([b[4] for b in batch], dim=0)
    return Xs, ys, player_masks, target_masks, y_masks


class Trainer:
    def __init__(self, 
            model: GNNTransformer, 
            loss_fn: callable, 
            train_loader: DataLoader, 
            val_loader: DataLoader, 
            cfg: Config, 
            log_dir: str = 'runs', 
            logger=None
        ):
        """
        Train a GNN Transformer with padded players, targets and loss masking.
        Added: Early Stopping, warm up, adaptive LR scheduling on plateau

        Args:
            model (nn.Module)
            loss_fn (callable)
            train_loader (DataLoader)
            val_loader (DataLoader)
            cfg (Config): config with optimizer, scheduler, training attributes
            log_dir (str): base directory for TensorBoard logs
            logger (optional): wandb / tensorboard / custom logger
        """
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)
        self.model.to(self.device)

        self.optimizer = self._init_optimizer()
        self.warmup_scheduler, self.plateau_scheduler = self._init_schedulers()

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.start_time = time.time()
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}

        self.writer = logger or SummaryWriter(log_dir=os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S")))
        self.log_dir = self.writer.log_dir
        print(f'TensorBoard logs: {self.log_dir}')

    def _init_optimizer(self):
        opt_cfg = self.cfg.optimizer
        return torch.optim.Adam(
            self.model.parameters(),
            lr=opt_cfg.lr,
            betas=opt_cfg.betas,
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
        )
    
    def _init_schedulers(self):
        """ create warmup and plateau schedulers """
        sch_cfg = self.cfg.scheduler

        # Linear warmup scheduler
        def lr_lambda(epoch):
            if epoch < sch_cfg.warmup_epochs:
                return float(epoch + 1) / sch_cfg.warmup_epochs
            return 1.0
        
        warmup = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=sch_cfg.lr_factor,
            patience=sch_cfg.patience,
            min_lr=sch_cfg.min_lr,
        )

        return warmup, plateau
    
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss, total_valid = 0.0, 0

        for batch in self.train_loader:
            X_batch, y_batch, player_mask, target_mask, y_mask = [
                t.to(self.device) for t in batch
            ]

            self.optimizer.zero_grad(set_to_none=True)

            # construct decoder inputs: shift right with BOS = last observed frame
            B, N, T_out, F_out = y_batch.shape
            bos = X_batch[:, :, -1:, 0:F_out].clone()
            # may have introduced NaN values
            if torch.isnan(bos).any():
                bos = torch.nan_to_num(bos, nan=0.0)
            y_inputs = torch.cat([bos, y_batch[:, :, :-1, :]], dim=2)

            # forward
            pred_deltas = self.model(
                X_batch,
                tgt_inputs=y_inputs,
                player_mask=player_mask,
                target_mask=target_mask,
                y_mask=y_mask
            )

            # compute loss
            loss = self.loss_fn(pred_deltas[..., 0:2], y_batch[..., 0:2], y_mask)

            # backward
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.training.grad_clip_norm)
            self.optimizer.step()

            total_loss += loss.item() * y_mask.sum().item()
            total_valid += y_mask.sum().item()

        avg_loss = total_loss / max(1, total_valid)
        self.history['train_loss'].append(avg_loss)
        self.writer.add_scalar('Loss/Train', avg_loss, epoch)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss, total_valid = 0.0, 0

        for batch in self.val_loader:
            X_batch, y_batch, player_mask, target_mask, y_mask = [
                t.to(self.device) for t in batch
            ]

            # decoder inputs
            B, N, T_out, F_out = y_batch.shape
            bos = X_batch[:, :, -1:, 0:F_out].clone()
            # may have introduced NaN values
            if torch.isnan(bos).any():
                bos = torch.nan_to_num(bos, nan=0.0)
            y_inputs = torch.cat([bos, y_batch[:, :, :-1, :]], dim=2)

            # forward using predict
            pred_deltas = self.model.predict(
                src=X_batch,
                future_len=T_out,
                player_mask=player_mask,
                target_mask=target_mask
            )
            
            # loss and backward
            val_loss = self.loss_fn(pred_deltas[..., 0:2], y_batch[..., 0:2], y_mask)
            total_loss += val_loss.item() * y_mask.sum().item()
            total_valid += y_mask.sum().item()

        avg_loss = total_loss / max(1, total_valid)
        self.history['val_loss'].append(avg_loss)
        self.writer.add_scalar('Loss/Validation', avg_loss, epoch)
        return avg_loss
    
    def fit(self):
        cfg = self.cfg.training
        print(f'[Training]: Config:\n{asdict(self.cfg)}')

        for epoch in range(cfg.epochs):
            epoch_start = time.time()
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            # Warmup + Plateau scheduling
            if epoch < self.cfg.scheduler.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.plateau_scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            self.writer.add_scalar('LR', current_lr, epoch)

            # Early Stopping Bookkeeping
            improved = val_loss < self.best_val_loss - cfg.min_delta
            if improved:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.best_state = deepcopy(self.model.state_dict())
            else:
                self.epochs_no_improve += 1

            # Logging
            print(
                f'[Epoch {epoch+1}/{cfg.epochs}] '
                f'Train={train_loss:.4e} Val={val_loss:.4e} '
                f'LR={current_lr:.2e} Time={time.time()-epoch_start:.1f}s'
            )

            # Early stopping halt condition
            if self.epochs_no_improve >= cfg.early_stopping_patience:
                print(f'Early stopping after {epoch+1} epochs.')
                break
        
        # load best model
        self.model.load_state_dict(self.best_state)
        self.writer.close()

        print(f'Best Val Loss: {self.best_val_loss:.4e}')
        print(f'Total Time: {time.time() - self.start_time:.1f}s')
        print(f'TensorBoard logs saved at: {self.log_dir}')
        return self.model, self.history

