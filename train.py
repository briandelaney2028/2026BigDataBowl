import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time
import copy
from utils import Config


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

def train_gnn_transformer(
        model: nn.Module,
        train_dataset: TensorDataset,
        val_dataset: TensorDataset = None,
        batch_size: int = Config.BATCH_SIZE,
        epochs: int = Config.EPOCHS,
        lr: float = Config.ETA,
        device: str = Config.DEVICE,
        grad_clip_norm: float = Config.GRAD_CLIP_NORM,
        verbose: bool = True
):
    """
    Train a GNN Transformer with padded players, targets and loss masking

    Parameters:
        model (nn.Module): GNNTransformer: model(X_batch, y_inputs) during training (with y_inputs = shifted right)
        train_dataset (TensorDataset): PyTorch dataset with X, y, and mask tensors
        val_dataset (TensorDataset, optional): PyTorch dataset with X, y, and mask tensors for validation
        batch_size (int): Batch size
        epochs (int): Training epochs
        lr (float): Learning Rate
        device (str): 'cuda' or 'cpu'
        grad_clip_norm (float): Gradient clipping norm
        verbose (bool): Whether to print training progress
    
    Returns:
        model (nn.Module): Trained model
        history (dict): Training and validation loss history
    """

    device = torch.device(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_default)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_default)
    
    history = {'train_loss': [], 'val_loss': []}
    start_time = time.time()
    
    # train
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        model.train()
        total_se = 0.0
        total_valid = 0

        for X_batch, y_batch, player_mask, target_mask, y_mask in train_loader:
            # move to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            player_mask = player_mask.to(device)
            target_mask = target_mask.to(device)
            y_mask = y_mask.to(device)

            optimizer.zero_grad()

            # construct decoder inputs: shift right with BOS = last observed frame
            B, N, T_out, F_out = y_batch.shape
            bos = X_batch[:, :, -1:, 0:F_out].clone()
            # may have introduced NaN values
            if torch.isnan(bos).any():
                bos = torch.nan_to_num(bos, nan=0.0)
            y_inputs = torch.cat([bos, y_batch[:, :, :-1, :]], dim=2)
            
            # forward
            pred_deltas = model(
                src=X_batch,
                tgt_inputs=y_inputs,
                player_mask=player_mask,
                target_mask=target_mask,
                y_mask=y_mask
            )
            
            # compute masked loss
            loss = masked_mse_loss(pred_deltas[..., 0:2], y_batch[..., 0:2], y_mask)
            
            # backprop
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            
            batch_valid = y_mask.sum().item()
            total_se += loss.item() * batch_valid
            total_valid += batch_valid

        train_epoch_loss = total_se / max(1, total_valid)
        history['train_loss'].append(train_epoch_loss)

        # validation
        val_epoch_loss = None
        if val_loader is not None:
            model.eval()
            val_total_se = 0.0
            val_total_valid = 0

            with torch.no_grad():
                for X_batch, y_batch, player_mask, target_mask, y_mask in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    player_mask = player_mask.to(device)
                    target_mask = target_mask.to(device)
                    y_mask = y_mask.to(device)

                    # decoder inputs
                    B, N, T_out, F_out = y_batch.shape
                    bos = X_batch[:, :, -1:, 0:F_out].clone()
                    # may have introduced NaN values
                    if torch.isnan(bos).any():
                        bos = torch.nan_to_num(bos, nan=0.0)
                    y_inputs = torch.cat([bos, y_batch[:, :, :-1, :]], dim=2)

                    # forward using predict
                    pred_deltas = model.predict(
                        src=X_batch,
                        future_len=T_out,
                        player_mask=player_mask,
                        target_mask=target_mask
                    )
                    
                    val_loss = masked_mse_loss(pred_deltas[..., 0:2], y_batch[..., 0:2], y_mask)
                    val_valid = y_mask.sum().item()
                    val_total_se += val_loss.item() * val_valid
                    val_total_valid += val_valid

            val_epoch_loss = val_total_se / max(1.0, val_total_valid)
            history['val_loss'].append(val_epoch_loss)

        # log
        epoch_time = time.time() - epoch_start_time
        if verbose:
            if val_epoch_loss is None:
                print(f'[Epoch {epoch}/{epochs}] Train Loss={train_epoch_loss:.4e}; Time={epoch_time:.1f}s')
            else:
                print(f'[Epoch {epoch}/{epochs}] Train Loss={train_epoch_loss:.4e}; Val Loss={val_epoch_loss:.4e}; Time={epoch_time:.1f}s')

    total_time = time.time() - start_time
    if verbose:
        print(f'Total Training Time: {total_time:.1f}s')
    
    return model, history

def enhanced_gnn_train(
    model: nn.Module,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset = None,
    batch_size: int = Config.BATCH_SIZE,
    epochs: int = Config.EPOCHS,
    lr: float = Config.ETA,
    device: str = Config.DEVICE,
    grad_clip_norm: float = Config.GRAD_CLIP_NORM,
    verbose: bool = True,
    early_stopping: bool = Config.EARLY_STOPPING,
    patience_es: int = Config.PATIENCE,
    min_delta: float = Config.MIN_DELTA,
    warmup_epochs: int = Config.WARMUP_EPOCHS,
    lr_factor: float = Config.ETA_FACTOR,
    patience_lr: int = Config.ETA_PATIENCE,
    min_lr: float = Config.ETA_MIN
):
    """
    Train a GNN Transformer with padded players, targets and loss masking.
    Added: Early Stopping, warm up, adaptive LR scheduling on plateau

    Parameters:
        model (nn.Module): GNNTransformer: model(X_batch, y_inputs) during training (with y_inputs = shifted right)
        train_dataset (TensorDataset): PyTorch dataset with X, y, and mask tensors
        val_dataset (TensorDataset, optional): PyTorch dataset with X, y, and mask tensors for validation
        batch_size (int): Batch size
        epochs (int): Training epochs
        lr (float): Learning Rate
        device (str): 'cuda' or 'cpu'
        grad_clip_norm (float): Gradient clipping norm
        verbose (bool): Whether to print training progress
        early_stopping (bool): Whether to implement early stopping
        patience_es (int): Number of epochs without improvement before stopping
        min_delta (float):  minimum improvement necessary for a new best score
        warmup_epochs (int): number of warmup epochs
        lr_factor (float): factor by which the learning rate will be reduce
        patience_lr (int): number of allowed epochs with no improvement after which lr is reduced
        min_lr (float): minimum allowed lr

    
    Returns:
        model (nn.Module): Trained model
        history (dict): Training and validation loss history
    """

    device = torch.device(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_factor,
        patience=patience_lr,
        verbose=verbose,
        min_lr=min_lr
    )

    # load data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_default)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_default)
    
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    # early stopping variables
    best_val_loss = float('inf')
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_se = 0.0
        total_valid = 0

        # Warm Up
        if epoch <= warmup_epochs:
            warmup_lr = lr * (epoch / warmup_epochs)
            for g in optimizer.param_groups:
                g['lr'] = warmup_lr
        
        # Training
        for X_batch, y_batch, player_mask, target_mask, y_mask in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            player_mask = player_mask.to(device)
            target_mask = target_mask.to(device)
            y_mask = y_mask.to(device)

            optimizer.zero_grad()

            # construct decoder inputs: shift right with BOS = last observed frame
            B, N, T_out, F_out = y_batch.shape
            bos = X_batch[:, :, -1:, 0:F_out].clone()
            # may have introduced NaN values
            if torch.isnan(bos).any():
                bos = torch.nan_to_num(bos, nan=0.0)
            y_inputs = torch.cat([bos, y_batch[:, :, :-1, :]], dim=2)

            # forward
            pred_deltas = model(
                X_batch,
                tgt_inputs=y_inputs,
                player_massk=player_mask,
                target_mask=target_mask,
                y_mask=y_mask
            )

            # compute loss
            loss = masked_mse_loss(pred_deltas[..., 0:2], y_batch[..., 0:2], y_mask)

            # backward
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            total_se += loss.itme() * y_mask.sum().item()
            total_valid += y_mask.sum().item()

        train_epoch_loss = total_se / max(1, total_valid)
        history['train_loss'].append(train_epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        # Validation
        val_epoch_loss = None
        if val_loader is not None:
            model.eval()
            val_total_se = 0.0
            val_total_valid = 0

            with torch.no_grad():
                for X_batch, y_batch, player_mask, target_mask, y_mask in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    player_mask = player_mask.to(device)
                    target_mask = target_mask.to(device)
                    y_mask = y_mask.to(device)

                    # decoder inputs
                    B, N, T_out, F_out = y_batch.shape
                    bos = X_batch[:, :, -1:, 0:F_out].clone()
                    # may have introduced NaN values
                    if torch.isnan(bos).any():
                        bos = torch.nan_to_num(bos, nan=0.0)
                    y_inputs = torch.cat([bos, y_batch[:, :, :-1, :]], dim=2)

                    # forward using predict
                    pred_deltas = model.predict(
                        src=X_batch,
                        future_len=T_out,
                        player_mask=player_mask,
                        target_mask=target_mask
                    )
                    
                    # loss and backward
                    val_loss = masked_mse_loss(pred_deltas[..., 0:2], y_batch[..., 0:2], y_mask)
                    val_total_se += val_loss.item() * y_mask.sum().item()
                    val_total_valid += y_mask.sum().item()

            val_epoch_loss = val_total_se / max(1, val_total_valid)
            history['val_loss'].append(val_epoch_loss)

            # scheduler update (after warmup)
            if epoch > warmup_epochs:
                scheduler.step(val_epoch_loss)

        # Early Stopping
        if val_loader is not None and early_stopping and val_epoch_loss is not None:
            # if new best
            if val_epoch_loss + min_delta < best_val_loss:
                best_val_loss = val_epoch_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_no_improve = 0
            else:   # no improvement
                epochs_no_improve += 1
                if epochs_no_improve >= patience_es:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience_es} epochs).")
                        print(f"Best validation loss: {best_val_loss:.4e} at epoch {best_epoch}")
                    model.load_state_dict(best_state_dict)
                    break
        
        # Logging
        epoch_time = time.time() - epoch_start_time
        if verbose:
            if val_epoch_loss is None:
                print(f'[Epoch {epoch}/{epochs}] Train Loss={train_epoch_loss:.4e}; Time={epoch_time:.1f}s')
            else:
                print(f'[Epoch {epoch}/{epochs}] Train Loss={train_epoch_loss:.4e}; Val Loss={val_epoch_loss:.4e}; Time={epoch_time:.1f}s')
        
    total_time = time.time() - start_time
    if verbose:
        print(f'Total Training Time: {total_time:.1f}s')
    
    return model, history