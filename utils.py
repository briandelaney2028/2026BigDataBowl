import os
import pandas as pd
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import seaborn as sns
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataclasses import dataclass, field
from typing import Union

@dataclass
class OptimizerConfig:
    lr: float = 3e-4            # learning rate
    betas: tuple = (0.9, 0.98)  # Adam beta values
    eps: float = 1e-9           # Adam stability term
    weight_decay: float = 1e-4  # Adam L2 penalty

@dataclass
class SchedulerConfig:
    warmup_epochs: int = 5      # num warmup epochs
    lr_factor: float = 0.5      # how much LR is reduced ReduceLROnPlateau
    patience: int = 8           # how long to wait before reducing LR
    min_lr: float = 1e-6        # minimum value for LR

@dataclass
class TrainingConfig:
    epochs: int = 200                   # number epochs if no early stopping
    batch_size: int = 32                # batch size
    grad_clip_norm: float = 1.0         # gradient clipping norm value
    early_stopping: bool = True         # whether early stopping
    early_stopping_patience: int = 20   # how long to wait w/o improvement before stopping
    min_delta: float = 1e-4             # minimum improvement necessary 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42      # random seed
    start_p: float = 1.0            # init teacher forcing prob
    lowest_p: float = 0.2           # lowest decayed teacher forcing prob
    decay_epochs: float = 15        # period over which decaying occurs
    delta: float = 0.5              # delta for Huber loss
    time_decay: float = 0.03        # time decay factor for Huber loss
    lambda_vel: float = 0.1        # weight on velocity smoothness
    lambda_acc: float = 0.05       # weight on acceleration smoothness

@dataclass
class TransformerConfig:
    d_model: int = 128                  # model internal dimension
    nhead: int = 8                      # number of transformer attention heads
    num_encoder_layers: int = 3         # num encoder layers
    num_decoder_layers: int = 3         # num decoder layers
    dim_feedforward: int = 256          # FFN dimension
    dropout: float = 0.1                # dropout fraction
    max_len: int = 500                  # max length of encodings
    pad_embedding_scale: float = 0.1    # scale for leanred pad embedding init
    gnn_nhead: int = 4                  # number of multi-agent GNN attention heads
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class DatasetConfig:
    data_dir: str = 'Data/'         # data directory
    saves_dir: str = 'Saves/'        # model and scalers directory
    data_fraction: float = 1.0      # fraction of data to sequence
    min_players: int = 7            # min number of valid players to sequence
    history_window: int = 5         # window for average, sliding-window metrics
    stride: int = 2                 # stride for sliding-window metrics
    sequence_length: int = 10       # input sequence length T_in
    target_features: bool = False   # whether to have GNNTransformer decode synthetic features
    dt: float = 0.1                 # time between frames
    n_route_clusters = 7            # num means for route clustering
    # ID columns
    id_cols: list = field(default_factory=lambda: [
        "game_id", "play_id", "nfl_id"
    ])
    # Features fed into GNNTransformer
    features: list = field(default_factory=lambda: [
        "x", "y", "absolute_yardline_number", "player_height", "num_frames_output",
        "player_weight", "s", "a", "dir", "o", "ball_land_x",
        "ball_land_y", "is_offense", "is_defense", "is_defensive_coverage",
        "is_other_route_runner", "is_passer", "is_targeted_receiver",
        "player_bmi", "x_velocity", "y_velocity", "angle_diff", "jerk",
        "angular_velocity", "rolling_x_velocity_std",
        "rolling_y_velocity_std", "rolling_a_std", "dist_to_ball_land",
        "dist_from_los", "bearing_to_ball_land", "bearing_diff_o",
        "bearing_diff_dir", "nearest_opp_dist", "closing_speed", "num_nearby_opp_3", 
        "num_nearby_opp_5", "mirror_wr_vx", "mirror_wr_vy", "mirror_offset_x",
        "mirror_offset_y", "mirror_wr_dist", "traj_straightness", "traj_max_turn",
        "traj_mean_turn", "traj_depth", "traj_width", "speed_mean", "speed_change",
        "route_pattern", "frame_id"
    ])
    # Features with angular dimensions
    angle_features: list = field(default_factory=lambda: [
        "dir", "o", "angle_diff", "bearing_to_ball_land",
        "bearing_diff_o", "bearing_diff_dir", "traj_max_turn", "traj_mean_turn"
    ])
    # Features needing scaling
    scaled_features: list = field(default_factory=lambda: [
    "x", "y", "absolute_yardline_number",
    "player_height", "player_weight", "player_bmi",
    "s", "a", "x_velocity", "y_velocity", "angular_velocity",
    "jerk", "rolling_x_velocity_std", "rolling_y_velocity_std",
    "rolling_a_std", "dist_to_ball_land", "dist_from_los",
    "ball_land_x", "ball_land_y", "closing_speed",
    "mirror_wr_vx", "mirror_wr_vy", "mirror_offset_x", "mirror_offset_y",
    "mirror_wr_dist", "traj_straightness", "traj_depth", "traj_width",
    "speed_mean", "speed_change", "num_nearby_opp_3", "num_nearby_opp_5",
    "frame_id", "num_frames_output"
])

    # F_in
    input_size = len(features.default_factory()) + len(angle_features.default_factory())
    output_size = 2 if not target_features else input_size

@dataclass
class LoggingConfig:
    log_dir: str = "runs/"
    project_name: str = "GNN_Transformer_Training"
    save_every: int = 10  # epochs
    verbose: bool = True

@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determnistic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def load_prediction_data(cfg: DatasetConfig):
    """
    Loads all training input and output CSV files and test data from data/train
    into DataFrames

    Parameters:
        cfg (DatasetConfig): dataset configuration

    Returns:
        train_input  (pd.DataFrame): Training input  data DataFrame
        train_output (pd.DataFrame): Training output data DataFrame
        test_input  (pd.DataFrame): Test data input  DataFrame
        test_template (pd.DataFrame): Template for submission DataFrame
    Raises:
        FileNotFoundError: If data/train dir does not exist
    """
    input_path = os.path.join(cfg.data_dir, 'train/')
    # collect all training csv files
    input_files  = glob.glob(os.path.join(input_path,  'input_2023_w*.csv'))
    output_files = glob.glob(os.path.join(input_path, 'output_2023_w*.csv'))

    if not input_files:
        raise FileNotFoundError('No input files found in data/train dir')

    # read into dfs
    train_input  = pd.concat([pd.read_csv(f) for f in input_files])
    train_output = pd.concat([pd.read_csv(f) for f in output_files])

    # read test data into dfs
    test_input  = pd.read_csv(os.path.join(cfg.data_dir, 'test_input.csv'))
    test_template = pd.read_csv(os.path.join(cfg.data_dir, 'test.csv'))

    return train_input, train_output, test_input, test_template

def load_supplemental_data(cfg: DatasetConfig):
    """
    Loads all supplemental CSV files from
    supplemental/114239_nfl_competition_files_published_analytics_final

    Parameters:
        cfg (DatasetConfig): dataset configuration

    Returns:
        df_supp (pd.DataFrame): Supplemental data DataFrame
        df_input (pd.DataFrame): Training input  data DataFrame
        df_output (pd.DataFrame): Training output data DataFrame
    Raises:
        FileNotFoundError: If dir does not exist
    """

    # load training input and output data
    df_input, df_output, _, _ = load_prediction_data()

    # load supplemental data
    supp_path = os.path.join(cfg.data_dir, 'supplementary_data.csv')

    if os.path.isfile(supp_path):
        df_supp = pd.read_csv(supp_path)
    else:
        raise FileNotFoundError('No supplementary data found in supplemental dir')

    return df_supp, df_input, df_output

def height_to_inches(height_str: str) -> int:
    """
    Converts a height string in the format 'X-Y' to inches

    Parameters:
        height_str (str): Height string in the format 'X-Y'
    Returns:
        int: Height in inches
    """
    try:
        feet, inches = map(int, height_str.split('-'))
        return feet * 12 + inches
    except Exception as e:
        print(f"Error converting height {height_str}: {e}")
        return np.nan

def invert_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the 'play_direction' and adjusts 'x_input', 'y_input', 'o', 'dir' accordingly
    EXAMPLE:
    A play moving left at the absolute coordinates (x=30, y=30) with orientation 270 and direction 180
    is transformed to a play moving right at (x=90, y=23.3) with orientation 90 and direction 0

    Parameters:
        df (pd.DataFrame): Input DataFrame
    Returns:
        df (pd.DataFrame): Transformed DataFrame
    """

    df = df.copy()
    left_data = df['play_direction'].str.lower().eq('left')
    
    # adjust player x and y coordinates
    df.loc[left_data, 'x'] = 120 - df.loc[left_data, 'x']
    df.loc[left_data, 'y'] = 53.3 - df.loc[left_data, 'y']

    # check if not an input df
    if 'absolute_yardline_number' not in df.columns:
        return df

    # adjust line of scrimmage
    df.loc[left_data, 'absolute_yardline_number'] = 120 - df.loc[left_data, 'absolute_yardline_number']
    # adjust orientation and direction (already non-negative)
    df.loc[left_data, 'o'] = 360 - df.loc[left_data, 'o']
    df.loc[left_data, 'dir'] = 360 - df.loc[left_data, 'dir']
    # adjust ball_land coordinates
    df.loc[left_data, 'ball_land_x'] = 120 - df.loc[left_data, 'ball_land_x']
    df.loc[left_data, 'ball_land_y'] = 53.3 - df.loc[left_data, 'ball_land_y']

    return df

def map_play_direction(df_input, df_output):
    """
    Maps the 'play_direction' column from df_input to df_output

    Parameters:
        df_input (pd.DataFrame): Input DataFrame
        df_output (pd.DataFrame): Output DataFrame
    Returns:
        df_output (pd.DataFrame): Transformed DataFrame
    """

    # obtain information from df_input
    direction_map = df_input[['game_id', 'play_id', 'play_direction']].drop_duplicates()

    # merge with df_output
    merged = df_output.merge(direction_map, on=['game_id', 'play_id'], how='left')

    return merged

def load_saved_data(save_path: str, required_keys: tuple[str]):
    """
    Attempt to load existing data

    Parameters:
        save_path (str): path to saved data
        required_keys (list[str]): necessary keys in saved data

    Returns:
        (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) or None:
            will return data if it checks out else None
    """
    if not os.path.exists(save_path):
        print(f'[INFO] Save file not found at {save_path}. Generating new data...')
        return None
    try:
        data = np.load(save_path, allow_pickle=True)
        if not all(k in data.keys() for k in required_keys):
            print('[WARNING] Save file missing required keys. Regenerating data...')
            return None
        print(f'[INFO] Loaded data successfully from {save_path}')
        return data
    except Exception as e:
        print(f'[ERROR] Failed to load save file ({e}). Regenerating data...')
        return None

def evaluate_model(
        model: torch.nn.Module,
        test_loader: DataLoader,
        loss_fns: Union[torch.nn.Module, list[torch.nn.Module]],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> dict[str, float]:
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

if __name__=='__main__':
    df_train = load_prediction_data(method='left', temporal=True)
    print(df_train[df_train['play_direction'] == 'left'].loc[:1000:50, ['absolute_yardline_number', 'player_name', 'x_input', 'o', 'ball_land_x', 'y_target']])

    df_train = invert_direction(df_train)
    print(df_train[df_train['play_direction'] == 'left'].loc[:1000:50, ['absolute_yardline_number', 'player_name', 'x_input', 'o', 'ball_land_x', 'y_target']])
