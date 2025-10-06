import pandas as pd
import numpy as np
import os
import glob


DATA_DIR = 'Data/'

def load_training_df(method:str='inner', temporal:bool=False)->pd.DataFrame:
    """
    Loads and merges all training input and output CSV files from the Data/train directory into a single DataFrame.

    Parameters:
        method (str): Merge method for pandas.merge (e.g., 'inner', 'left').
            'inner' retains only matched keys, 'left' retains all rows from input.
        temporal (bool):
            If True, merges input and output data on ['game_id', 'play_id', 'nfl_id', 'frame_id'] (frame-level, temporal join).
            If False, aggregates input data to the last frame per ['game_id', 'play_id', 'nfl_id'] before merging on those keys (non-temporal, play-level join).

    Returns:
        pd.DataFrame: Merged DataFrame containing input features and true output values, either at the frame or play level depending on 'temporal'.

    Raises:
        FileNotFoundError: If no input files are found in the expected directory.
    """
    input_path = os.path.join(DATA_DIR, 'train/')
    # collect all csvs
    input_files  = glob.glob(os.path.join(input_path,  'input_2023_w*.csv'))
    output_files = glob.glob(os.path.join(input_path, 'output_2023_w*.csv'))

    if not input_files:
        raise FileNotFoundError
    
    # read into a single df
    df_input  = pd.concat([pd.read_csv(f) for f in input_files])
    df_output = pd.concat([pd.read_csv(f) for f in output_files])

    if temporal:
        df_merged = pd.merge(
            df_input, df_output,
            on=['game_id', 'play_id', 'nfl_id', 'frame_id'], 
            how=method,
            suffixes=('_input', '_target')
        )
    else:
        input_grouped = df_input.groupby(['game_id', 'play_id', 'nfl_id']).last().reset_index()
        input_grouped = input_grouped.drop('frame_id', axis=1)
        df_merged = pd.merge(
            input_grouped, df_output,
            on=['game_id', 'play_id', 'nfl_id'], 
            how=method,
            suffixes=('_input', '_target')
        )
    return df_merged

def load_supplemental_df(method:str='inner')->pd.DataFrame:
    """
    Loads the supplementary data CSV and merges it with the training data DataFrame.

    Parameters:
        method (str): Merge method for pandas.merge (e.g., 'inner', 'left').
            'inner' retains only matched keys, 'left' retains all rows from training data.

    Returns:
        pd.DataFrame: Merged DataFrame containing training data and supplementary information.
    """
    supp_path = os.path.join(DATA_DIR, 'supplementary_data.csv')
    df_supp = pd.read_csv(supp_path)
    df_merged = load_training_df()
    df_all = pd.merge(
        df_merged, df_supp,
        on=['game_id', 'play_id'], how=method
    )
    return df_all

def height_to_inches(height_str: str) -> int:
    """
    Converts height from feet-inches format (e.g., "6-2") to total inches.

    Parameters:
        height_str (str): Height in feet-inches format.

    Returns:
        int: Height in total inches.
    """
    try:
        feet, inches = map(int, height_str.split('-'))
        return feet * 12 + inches
    except Exception as e:
        return np.nan

if __name__=='__main__':
    df_train = load_training_df(temporal=False)
    print(df_train.loc[:20, ['frame_id', 'x_input', 'x_target']])

    # df_supp = load_supplemental_df()
    # print(df_supp.head())