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

cfg = Config()

utils.set_seed(cfg.training.seed)
    
df_input, df_output, _, _ = utils.load_prediction_data(cfg.dataset)
df_input['player_height'] = df_input['player_height'].apply(utils.height_to_inches)
df_input = engineer_features(utils.invert_direction(df_input), cfg.dataset)
df_output = utils.invert_direction(utils.map_play_direction(df_input, df_output))

# get X, y
# player_mask shows where padded players are
# target_mask shows who has y data
# y_mask shows where padded timesteps are in y_data
save_path = os.path.join(cfg.dataset.data_dir, 'Saves', 'GNNtransformer_data.npz')
required_keys = ['X', 'y', 'player_mask', 'target_mask', 'y_mask', 'ids']
data = utils.load_saved_data(save_path, required_keys)

# If loading failed, generate and save new data
if data is None:
    print('[INFO] Generating new data sequences...')
    X, y, player_mask, target_mask, y_mask, ids = generate_sequences_4D(df_input, cfg.dataset, df_output=df_output)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, X=X, y=y, player_mask=player_mask, target_mask=target_mask, y_mask=y_mask, ids=ids)
    print('[INFO] Data saved successfully to {save_path}')
else:
    X, y, player_mask, target_mask, y_mask, ids = data['X'], data['y'], data['player_mask'], data['target_mask'], data['y_mask'], data['ids']
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
utils.evaluate_model(
    trained_gnn_transformer,
    test_loader,
    loss_fns=loss_fns
)