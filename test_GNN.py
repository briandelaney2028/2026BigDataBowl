import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import utils
from utils import Config
from feature_engineering import engineer_features
from data_sequencing import generate_sequences_4D
from FeatureScaler import FeatureScaler
from train import Trainer, prepare_targets_as_deltas, masked_mse_loss, collate_default, masked_FDE_loss
from Transformers import GNNTransformer
from typing import Union, List, Dict
import os

def evaluate_model(
        model: torch.nn.Module,
        test_loader: DataLoader,
        loss_fns: Union[torch.nn.Module, List[torch.nn.Module]],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, float]:
    """
    Evaluate a fully trained GNNTransformer on a held-out test set.

    Parameters:
        model (nn.Module): trained model
        test_loader (DataLoader): held out test data
        loss_fns (Union[torch.nn.Module, List[torch.nn.Module]]): one or more loss fns
        device (str): device

    Returns:
        Dict[str, float]: averaged loss for each proved loss function
    """
    model.eval()
    model.to(device)

    if not isinstance(loss_fns, list):
        loss_fns = [loss_fns]

    total_losses = [0.0 for _ in loss_fns]
    total_valid = 0

    for batch in test_loader:
        X_batch, y_batch, player_mask, target_mask, y_mask = [
            t.to(device) for t in batch
        ]

        # decoder autoregressive inputs
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

        # compute each loss
        for i, loss_fn in enumerate(loss_fns):
            loss = loss_fn(pred_deltas[..., 0:2], y_batch[..., 0:2], y_mask)
            if getattr(loss_fn, '__name__') == 'masked_FDE_loss':
                valid_counts = y_mask.sum(dim=-1)
                valid_mask = valid_counts > 0
                total_losses[i] = loss.item() * valid_mask.sum().item()
            else:
                total_losses[i] = loss.item() * y_mask.sum().item()

        total_valid += y_mask.sum().item()

    avg_losses = [total_loss / max(1, total_valid) for total_loss in total_losses]

    results = {
        getattr(loss_fn, '__name__', f'loss_{i+1}'): avg_losses[i]
        for i, loss_fn in enumerate(loss_fns)
    }

    print('\nFinal Test Evaluation')
    for name, value in results.items():
        print(f'{name}: {value:.4e}')
    return results


cfg = Config()

utils.set_seed(cfg.training.seed)
    
# df_input, df_output, _, _ = utils.load_prediction_data(cfg.dataset)
# df_input['player_height'] = df_input['player_height'].apply(utils.height_to_inches)
# df_input = engineer_features(utils.invert_direction(df_input), cfg.dataset)
# df_output = utils.invert_direction(utils.map_play_direction(df_input, df_output))

# # get X, y
# # player_mask shows where padded players are
# # target_mask shows who has y data
# # y_mask shows where padded timesteps are in y_data
# X, y, player_mask, target_mask, y_mask, ids = generate_sequences_4D(df_input, cfg.dataset, df_output=df_output,
#                       sequence_length=10, data_fraction=1.0)
save_path = os.path.join(cfg.dataset.data_dir, 'Saves/GNNtransformer_data.npz')
# np.savez(save_path, X=X, y=y, player_mask=player_mask, target_mask=target_mask, y_mask=y_mask, ids=ids)

# Load data
data = np.load(save_path, allow_pickle=True)
X, y, player_mask, target_mask, y_mask, ids = data['X'], data['y'], data['player_mask'], data['target_mask'], data['y_mask'], data['ids']

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
y_test_deltas = prepare_targets_as_deltas(y_test, last_test_scaled, player_mask_test)

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

gnn_transformer = GNNTransformer(
    in_feats=cfg.dataset.input_size,
    output_size=cfg.dataset.output_size,
    cfg=cfg.transformer
)
print("Model:", gnn_transformer)

train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True,  collate_fn=collate_default)
val_loader   = DataLoader(  val_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_default)
test_loader  = DataLoader( test_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_default)

trainer = Trainer(
    model=gnn_transformer,
    loss_fn = masked_mse_loss,
    train_loader=train_loader,
    val_loader=val_loader,
    cfg=cfg,

)
trained_gnn_transformer, transformer_history = trainer.fit()

model_folder = os.path.join(cfg.dataset.saves_dir, 'Models/')
model_path = os.path.join(model_folder, 'test_gnn_enhncd_train.pth')
scaler_folder = os.path.join(cfg.dataset.saves_dir, 'Scalers/')
scaler_path = os.path.join(scaler_folder, 'test_gnn_scaler_enhncd_train.pkl')

torch.save(trained_gnn_transformer, model_path)
scaler.save(scaler_path)

loss_fns = [masked_mse_loss, masked_FDE_loss]
evaluate_model(
    trained_gnn_transformer,
    test_loader,
    loss_fns=loss_fns
)