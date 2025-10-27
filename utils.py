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


class Config:
    DATA_DIR = 'Data/'

    SEED = 42

    HISTORY_WINDOW = 5
    
    SEQUENCE_LENGTH = 10
    DATA_FRACTION = 1.0
    MIN_PLAYERS = 7
    TARGET_FEATURES = False

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Transformer Hyperparameters
    D_MODEL = 128
    NHEAD = 8
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DIM_FEEDFORWARD = 256
    DROPOUT = 0.1
    MAX_LEN = 500
    PAD_EMBEDDING_SCALE = 0.1

    # FrameGNN Hyperparameters
    GNN_NHEAD = 4

    # Training Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 10
    ETA = 1e-4
    GRAD_CLIP_NORM = 1.0

    # Datset Info

    FEATURES = ['x', 'y', 'absolute_yardline_number', 'player_height', 'num_frames_output',
                'player_weight', 's', 'a', 'dir', 'o', 'ball_land_x',
                'ball_land_y', 'is_offense', 'is_defense', 'is_defensive_coverage',
                'is_other_route_runner', 'is_passer', 'is_targeted_receiver',
                'player_bmi', 'x_velocity', 'y_velocity', 'angle_diff', 'jerk',
                'angular_velocity', 'rolling_x_velocity_std',
                'rolling_y_velocity_std', 'rolling_a_std', 'dist_to_ball_land',
                'dist_from_los', 'bearing_to_ball_land', 'bearing_diff_o',
                'bearing_diff_dir', 'frame_id']
    ID_COLS = ['game_id', 'play_id', 'nfl_id']
    dt = 0.1
    ANGLE_FEATURES = ['dir', 'o', 'angle_diff', 'bearing_to_ball_land', 
                    'bearing_diff_o', 'bearing_diff_dir']
    SCALED_FEATURES = ['absolute_yardline_number', 'player_height', 'player_weight', 
                    'x', 'y', 's', 'a', 'ball_land_x', 'ball_land_y', 'player_bmi',
                    'x_velocity', 'y_velocity', 'jerk', 'angular_velocity', 
                    'rolling_x_velocity_std', 'rolling_y_velocity_std', 
                    'rolling_a_std', 'dist_to_ball_land', 'dist_from_los', 'frame_id',
                    'num_frames_output']



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

def load_prediction_data():
    """
    Loads all training input and output CSV files and test data from data/train
    into DataFrames

    Returns:
        train_input  (pd.DataFrame): Training input  data DataFrame
        train_output (pd.DataFrame): Training output data DataFrame
        test_input  (pd.DataFrame): Test data input  DataFrame
        test_template (pd.DataFrame): Template for submission DataFrame
    Raises:
        FileNotFoundError: If data/train dir does not exist
    """
    input_path = os.path.join(Config.DATA_DIR, 'train/')
    # collect all training csv files
    input_files  = glob.glob(os.path.join(input_path,  'input_2023_w*.csv'))
    output_files = glob.glob(os.path.join(input_path, 'output_2023_w*.csv'))

    if not input_files:
        raise FileNotFoundError('No input files found in data/train dir')

    # read into dfs
    train_input  = pd.concat([pd.read_csv(f) for f in input_files])
    train_output = pd.concat([pd.read_csv(f) for f in output_files])

    # read test data into dfs
    test_input  = pd.read_csv(os.path.join(Config.DATA_DIR, 'test_input.csv'))
    test_template = pd.read_csv(os.path.join(Config.DATA_DIR, 'test.csv'))

    return train_input, train_output, test_input, test_template

def load_supplemental_data():
    """
    Loads all supplemental CSV files from
    supplemental/114239_nfl_competition_files_published_analytics_final

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
    supp_path = os.path.join(Config.DATA_DIR, 'supplementary_data.csv')

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

if __name__=='__main__':
    df_train = load_prediction_data(method='left', temporal=True)
    print(df_train[df_train['play_direction'] == 'left'].loc[:1000:50, ['absolute_yardline_number', 'player_name', 'x_input', 'o', 'ball_land_x', 'y_target']])

    df_train = invert_direction(df_train)
    print(df_train[df_train['play_direction'] == 'left'].loc[:1000:50, ['absolute_yardline_number', 'player_name', 'x_input', 'o', 'ball_land_x', 'y_target']])
