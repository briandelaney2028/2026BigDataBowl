import torch
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import utils
from utils import Config
from feature_engineering import engineer_features
from FeatureScaler import FeatureScaler
from train import prepare_targets_as_deltas, train_gnn_transformer
from Transformers import GNNTransformer
from plot_play import plot_play




utils.set_seed(Config.SEED)
    
# df_input, df_output, _, _ = utils.load_prediction_data()
# df_input['player_height'] = df_input['player_height'].apply(utils.height_to_inches)
# df_input = engineer_features(utils.invert_direction(df_input))
# df_output = utils.invert_direction(utils.map_play_direction(df_input, df_output))

# # get X, y
# # player_mask shows where padded players are
# # target_mask shows who has y data
# # y_mask shows where padded timesteps are in y_data
# X, y, player_mask, target_mask, y_mask, ids = generate_sequences_4D(df_input, df_output=df_output,
#                       sequence_length=10, data_fraction=1.0, target_features=TARGET_FEATURES)
# np.savez('GNNtransformer_data.npz', X=X, y=y, player_mask=player_mask, target_mask=target_mask, y_mask=y_mask, ids=ids)

# Load data
data = np.load('GNNtransformer_data.npz', allow_pickle=True)
X, y, player_mask, target_mask, y_mask, ids = data['X'], data['y'], data['player_mask'], data['target_mask'], data['y_mask'], data['ids']

# Make Train-Test split
X_train, X_temp, y_train, y_temp, player_mask_train, player_mask_temp, target_mask_train, target_mask_temp, y_mask_train, y_mask_temp = train_test_split(
    X, y, player_mask, target_mask, y_mask, test_size=0.3, random_state=42
)

# Make Validation-Test split
X_val, X_test, y_val, y_test, player_mask_val, player_mask_test, target_mask_val, target_mask_test, y_mask_val, y_mask_test = train_test_split(
    X_temp, y_temp, player_mask_temp, target_mask_temp, y_mask_temp, test_size=0.3, random_state=42
)
# Scale Features
scaler = FeatureScaler(feature_names=Config.FEATURES, method='standard', angle_features=Config.ANGLE_FEATURES)
X_train_scaled = scaler.fit_transform(X_train, Config.SCALED_FEATURES)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
if not Config.TARGET_FEATURES:
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

input_size = X_train_scaled.shape[-1]
output_size = y_train_deltas.shape[-1]
gnn_transformer = GNNTransformer(
    in_feats=input_size,
    output_size=output_size
)
print("Model:", gnn_transformer)
trained_gnn_transformer, transformer_history = train_gnn_transformer(
    gnn_transformer, 
    train_dataset, 
    val_dataset=val_dataset,
    batch_size=32, 
    epochs=10
)

torch.save(trained_gnn_transformer, 'test_gnn.pth')
scaler.save('test_gnn_scaler.pkl')