import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from copy import deepcopy
from dataclasses import asdict
import os
from sklearn.model_selection import train_test_split
from utils import Config
import utils
from Transformers import GNNTransformer
from feature_engineering import engineer_features
from data_sequencing import generate_sequences_4D
from FeatureScaler import FeatureScaler
from Transformers import GNNTransformer

def build_model(cfg: Config) -> torch.nn.Module:
    """
    Generates an untrained model based on cfg

    Parameters:
        cfg (Config): sampled configuration

    Returns:
        (torch.nn.Module): untrained model
    """
    
    return GNNTransformer(
        cfg.dataset.input_size,
        cfg.dataset.output_size,
        cfg=cfg.transformer
    )

def get_dataloaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load data into DataLoaders

    Parameters:
        cfg (Config): sampled configuration

    Returns:
        (tuple[DataLoader, DataLoader, DataLoader]): train, val, test DataLoaders
    """

    # get sequence data
    save_path = os.path.join(cfg.dataset.data_dir, 'Saves', 'GNNtransformer_data.npz')
    required_keys = ['X', 'y', 'player_mask', 'target_mask', 'y_mask', 'ids']
    data = utils.load_saved_data(save_path, required_keys)

    def load_data():
        # load data, and engineer features
        df_input, df_output, _, _ = utils.load_prediction_data(cfg.dataset)
        df_input['player_height'] = df_input['player_height'].apply(utils.height_to_inches)
        df_input = engineer_features(utils.invert_direction(df_input), cfg.dataset)
        df_output = utils.invert_direction(utils.map_play_direction(df_input, df_output))

        print('[INFO] Generating new data sequences...')
        X, y, player_mask, target_mask, y_mask, ids = generate_sequences_4D(df_input, cfg.dataset, df_output=df_output)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, X=X, y=y, player_mask=player_mask, target_mask=target_mask, y_mask=y_mask, ids=ids)
        print('[INFO] Data saved successfully to {save_path}')

        return X, y, player_mask, target_mask, y_mask, ids

    # If loading failed, generate and save new data
    if data is None:
        X, y, player_mask, target_mask, y_mask, ids = load_data()
    else:
        X, y, player_mask, target_mask, y_mask, ids = data['X'], data['y'], data['player_mask'], data['target_mask'], data['y_mask'], data['ids']
    
    B, N, T, F = X.shape
    if F != cfg.dataset.feature_len:
        print('[INFO] Outdated Dataset...')
        X, y, player_mask, target_mask, y_mask, ids = load_data()
    
    print('[INFO] Data ready for use.')

    # Make Train-Test split
    X_train, X_temp, y_train, y_temp, player_mask_train, player_mask_temp, target_mask_train, target_mask_temp, y_mask_train, y_mask_temp = train_test_split(
        X, y, player_mask, target_mask, y_mask, test_size=0.3, random_state=cfg.training.seed
    )

    # Make Validation-Test split
    X_val, X_test, y_val, y_test, player_mask_val, player_mask_test, target_mask_val, target_mask_test, y_mask_val, y_mask_test = train_test_split(
        X_temp, y_temp, player_mask_temp, target_mask_temp, y_mask_temp, test_size=0.3, random_state=cfg.training.seed
    )
    # Scale Features
    scaler = FeatureScaler(feature_names=cfg.dataset.features, method='standard', angle_features=cfg.dataset.angle_features)
    X_train_scaled = scaler.fit_transform(X_train, cfg.dataset.scaled_features)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    if not cfg.dataset.target_features:
        y_train_scaled = scaler.transform(y_train, ['x', 'y'])
        y_val_scaled = scaler.transform(y_val, ['x', 'y'])
        y_test_scaled = scaler.transform(y_test, ['x', 'y'])
    else:
        y_train_scaled = scaler.transform(y_train)
        y_val_scaled = scaler.transform(y_val)
        y_test_scaled = scaler.transform(y_test)

    # Obtain last positions (B, N, 2)
    last_train_scaled = X_train_scaled[:, :, -1, :2]
    last_val_scaled = X_val_scaled[:, :, -1, :2]
    last_test_scaled = X_test_scaled[:, :, -1, :2]

    # convert y to deltas from last known position
    y_train_deltas = prepare_targets_as_deltas(y_train_scaled, last_train_scaled, player_mask_train)
    y_val_deltas = prepare_targets_as_deltas(y_val_scaled, last_val_scaled, player_mask_val)
    y_test_deltas = prepare_targets_as_deltas(y_test_scaled, last_test_scaled, player_mask_test)

    # wrap data
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_deltas, dtype=torch.float32),
        torch.tensor(player_mask_train, dtype=torch.bool),
        torch.tensor(target_mask_train, dtype=torch.bool),
        torch.tensor(y_mask_train, dtype=torch.bool)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val_deltas, dtype=torch.float32),
        torch.tensor(player_mask_val, dtype=torch.bool),
        torch.tensor(target_mask_val, dtype=torch.bool),
        torch.tensor(y_mask_val, dtype=torch.bool)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test_deltas, dtype=torch.float32),
        torch.tensor(player_mask_test, dtype=torch.bool),
        torch.tensor(target_mask_test, dtype=torch.bool),
        torch.tensor(y_mask_test, dtype=torch.bool)
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True,  collate_fn=collate_default)
    val_loader   = DataLoader(  val_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_default)
    test_loader  = DataLoader( test_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_default)

    return train_loader, val_loader, test_loader, scaler

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
        predictions (torch.Tensor): Predicted values (B, N, T_out, F)
        targets (torch.Tensor): Target values (B, N, T_out, F)
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

class maskedCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.huber = maskedHuberLoss(cfg.training.delta, cfg.training.time_decay)
        self.smooth = smoothnessLoss(cfg.training.lambda_vel, cfg.training.lambda_acc)

    def forward(self, pred, target, last_obs, mask, player_mask):
        huber_loss = self.huber(pred, target, mask)
        if last_obs is None:
            return huber_loss, None
        smooth_output = self.smooth(pred, last_obs, mask, player_mask)
        return huber_loss, smooth_output

class maskedHuberLoss(nn.Module):
    """
    Compute masked Huber loss with time decay over output timesteps

    Parameters:
        predictions (torch.Tensor): Predicted values (B, N, T_out, F) or (S, T_out, F)
        targets (torch.Tensor): Target values (B, N, T_out, F) or (S, T_out, F)
        mask (torch.Tensor): (B, N, T_out) -> True where valid output

    Returns:
        loss (torch.Tensor): Scalar loss value
    """
    def __init__(self, delta=0.5, time_decay=0.03):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay

    def forward(self, pred, target, mask):
        err = pred - target
        abs_err = torch.abs(err)
        # huber loss
        # if abs err <= delta threshold squared loss
        # if abs err > delta  L1 style loss
        huber = torch.where(abs_err <= self.delta, 0.5 * torch.pow(err, 2),
                            self.delta * (abs_err - 0.5 * self.delta))
        
        if self.time_decay > 0:
            T = pred.size(2)
            t = torch.arange(T, device=pred.device).float()
            weight = torch.exp(-self.time_decay * t)
            weight = weight.view(1, 1, T, 1)
            huber = huber * weight
            mask = mask.unsqueeze(-1) * weight

        denom = mask.sum()
        if denom <= 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        loss = (huber * mask).sum() / denom
        return loss

class smoothnessLoss(nn.Module):
    """
    Penalize large changes in x/y velocities and accelerations across predicted
    outputs, using the same reconstruction logic as reconstruct_absolute_from_deltas,
    but keeping gradient flow (no .detach() or numpy conversion).

    Parameters:
        predictions (torch.Tensor): (B, N, T_out, 2) predicted deltas (scaled)
        last_obs (torch.Tensor): (B, N, 1, 2) last absolute positions from inputs
        mask (torch.Tensor): (B, N, T_out) True where valid prediction
        player_mask (torch.Tensor): (B, N) True where player is valid (not padded)
        lambda_vel (float): weighting for velocity smoothness loss
        lambda_acc (float): weighting for acceleration smoothness loss
    Returns:
        total_loss (torch.Tensor), loss_dict (dict)
    """
    def __init__(self, lambda_vel=1.0, lambda_acc=1.0):
        super().__init__()
        self.lambda_vel = lambda_vel
        self.lambda_acc = lambda_acc
    
    def forward(self, pred, last_obs, mask, player_mask):
        device = pred.device
        B, N, T_out, _ = pred.shape
        if torch.isnan(last_obs).any():
            last_obs = torch.nan_to_num(last_obs, nan=0.0)

        # Reconstruct absolute positions
        abs_pos = torch.zeros_like(pred)
        abs_pos[:, :, 0, :] = last_obs[:, :, 0, :] + pred[:, :, 0, :]
        for t in range(1, T_out):
            abs_pos[:, :, t, :] = abs_pos[:, :, t-1, :] + pred[:, :, t, :]

        # mask invalid players
        valid_mask = player_mask.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
        abs_pos = abs_pos * valid_mask

        # Extend positions and mask with last_obs
        pos = torch.cat([last_obs, abs_pos], dim=2)  # (B, N, T_out+1, 2)
        extended_mask = torch.cat(
            [torch.ones_like(mask[..., :1], dtype=torch.bool, device=device), mask], dim=2
        )

        # compute velocities
        vel = pos[..., 1:, :] - pos[..., :-1, :]
        mask_vel = extended_mask[..., 1:] & extended_mask[..., :-1] & player_mask.unsqueeze(-1)

        # compute accelerations
        acc = vel[..., 1:, :] - vel[..., :-1, :]
        mask_acc = mask_vel[..., 1:] & mask_vel[..., :-1]

        # penalize large velocities and accelerations 
        vel_diff_sq = (vel.pow(2).sum(dim=-1)) * mask_vel
        acc_diff_sq = (acc.pow(2).sum(dim=-1)) * mask_acc

        denom_vel = mask_vel.sum()
        denom_acc = mask_acc.sum()

        vel_loss = vel_diff_sq.sum() / denom_vel if denom_vel > 0 else torch.tensor(0.0, device=device)
        acc_loss = acc_diff_sq.sum() / denom_acc if denom_acc > 0 else torch.tensor(0.0, device=device)

        total = self.lambda_vel * vel_loss + self.lambda_acc * acc_loss
        return total, {'vel': vel_loss.item(), 'acc': acc_loss.item()}

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
            train_loader (DataLoader)
            val_loader (DataLoader)
            cfg (Config): config with optimizer, scheduler, training attributes
            log_dir (str): base directory for TensorBoard logs
            logger (optional): wandb / tensorboard / custom logger
        """
        self.model = model
        self.loss_fn = maskedCriterion(cfg)
        self.test_loss_fn = masked_FDE_loss
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
        self.history = {'train_loss': [], 'val_loss': [], 'lr': [], 'teacher_forcing': []}

        self.writer = logger or SummaryWriter(log_dir=os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S")))
        self.log_dir = self.writer.log_dir
        print(f'TensorBoard logs: {self.log_dir}')

        self.ss_start_p = cfg.training.start_p
        self.ss_lowest_p = cfg.training.lowest_p
        self.ss_decay = cfg.training.decay_epochs

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
    
    def _get_teacher_forcing_ratio(self, epoch):
        # no decay
        if self.ss_decay <= 0:
            return self.ss_start_p
        
        # warm up phase (no decay yet)
        delay = self.cfg.scheduler.warmup_epochs
        if epoch < delay:
            return self.ss_start_p
        
        # compute normalized progress over decay window [0, 1]
        t = (epoch - delay) / max(1.0, self.ss_decay)
        t = min(max(t, 0.0), 1.0)

        # inverse sigmoid decay
        k = 7.5     # shape factor
        inv_sig = 1.0 / (1.0 + np.exp(k * (t - 0.5)))
        ratio = self.ss_lowest_p + (self.ss_start_p - self.ss_lowest_p) * inv_sig
        return ratio
    
    def train_one_epoch(self, epoch):
        self.model.train()
        total_valid = 0
        total_loss, total_pred_loss, total_vel_loss, total_acc_loss = 0.0, 0.0, 0.0, 0.0

        tfr = self._get_teacher_forcing_ratio(epoch)
        self.history['teacher_forcing'].append(tfr)
        self.writer.add_scalar('Hyperparams/Teacher_Forcing_Ratio', tfr, epoch)

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
            
            # if teacher forcing ratio near 1, skip sampling
            if tfr > 0.99:
                y_inputs = torch.cat([bos, y_batch[:, :, :-1, :]], dim=2)

                # forward
                pred_deltas = self.model(
                    X_batch,
                    tgt_inputs=y_inputs,
                    player_mask=player_mask,
                    target_mask=target_mask,
                    y_mask=y_mask
                )
            else:
                y_inputs_teacher = torch.cat([bos, y_batch[:, :, :-1, :]], dim=2)

                # initial forward with full teacher forcing
                pred_teacher = self.model(
                    X_batch,
                    tgt_inputs=y_inputs_teacher,
                    player_mask=player_mask,
                    target_mask=target_mask,
                    y_mask=y_mask
                )

                # soft interpolation between teacher forcing and predictions
                pred_shifted = torch.cat([bos, pred_teacher[:, :, :-1, :]], dim=2)

                # blend
                y_inputs_soft = tfr * y_inputs_teacher + (1 - tfr) * pred_shifted

                # forward again with blended inputs
                pred_deltas = self.model(
                    X_batch,
                    tgt_inputs=y_inputs_soft,
                    player_mask=player_mask,
                    target_mask=target_mask,
                    y_mask=y_mask
                )

            # compute huber and smooth loss
            huber_loss, (smooth_loss, smooth_log) = self.loss_fn(
                pred_deltas[..., 0:2], 
                y_batch[..., 0:2], 
                X_batch[:, :, -1:, :2],
                y_mask,
                player_mask
            )

            # combine for total loss
            loss = huber_loss + smooth_loss

            # backward
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.training.grad_clip_norm)
            self.optimizer.step()

            total_loss += loss.item() * y_mask.sum().item()
            total_pred_loss += huber_loss.item() * y_mask.sum().item()
            total_vel_loss += smooth_log['vel'] * y_mask.sum().item()
            total_acc_loss += smooth_log['acc'] * y_mask.sum().item()
            total_valid += y_mask.sum().item()

        avg_loss = total_loss / max(1, total_valid)
        avg_pred_loss = total_pred_loss / max(1, total_valid)
        avg_vel_loss = total_vel_loss / max(1, total_valid)
        avg_acc_loss = total_acc_loss / max(1, total_valid)
        self.history['train_loss'].append(avg_loss)
        self.writer.add_scalar('Loss/Train', avg_loss, epoch)
        self.writer.add_scalars(
            "Loss/Smoothness",
            {
                "Velocity": avg_vel_loss,
                "Acceleration": avg_acc_loss
            },
            global_step=epoch
        )
        self.writer.add_scalars(
            "Loss/Components",
            {
                "Predictions": avg_pred_loss,
                "Smoothness": avg_vel_loss + avg_acc_loss
            },
            global_step=epoch
        )
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
            
            last_pos = X_batch[:, :, -1, 0:2]   # (B, N, 2)

            pred_abs = utils.reconstruct_absolute_from_deltas(last_pos, pred_deltas, player_mask)
            pred_abs = torch.tensor(pred_abs, dtype=pred_deltas.dtype, device=self.device)
            true_abs = utils.reconstruct_absolute_from_deltas(last_pos, y_batch, player_mask)
            true_abs = torch.tensor(true_abs, dtype=y_batch.dtype, device=self.device)

            # loss - Final Displacement Error
            val_loss = self.test_loss_fn(pred_abs[..., 0:2], true_abs[..., 0:2], y_mask)
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
            self.writer.add_scalar('Hyperparams/LR', current_lr, epoch)

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

