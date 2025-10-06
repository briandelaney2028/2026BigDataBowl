import pandas as pd
import os
import glob


DATA_DIR = 'Data/'

def load_training_df(method='inner'):
    
    input_path = os.path.join(DATA_DIR, 'train/')
    # collect all csvs
    input_files  = glob.glob(os.path.join(input_path,  'input_2023_w*.csv'))
    output_files = glob.glob(os.path.join(input_path, 'output_2023_w*.csv'))

    if not input_files:
        raise FileNotFoundError
    
    # read into a single df
    df_input  = pd.concat([pd.read_csv(f) for f in input_files])
    df_output = pd.concat([pd.read_csv(f) for f in output_files])

    df_output = df_output.rename(columns={'x':'x_true', 'y':'y_true'})
    ### how = 'inner' -> inner vector product only retains matched keys
    ### how = 'left'  -> will retain all rows in df_input
    df_merged = pd.merge(
        df_input, df_output,
        on=['game_id', 'play_id', 'nfl_id', 'frame_id'], how=method
    )
    return df_merged

def load_supplemental_df(method='inner'):
    supp_path = os.path.join(DATA_DIR, 'supplementary_data.csv')
    df_supp = pd.read_csv(supp_path)
       
    df_merged = load_training_df()
    
    df_all = pd.merge(
        df_merged, df_supp,
        on=['game_id', 'play_id'], how=method
    )
    return df_all


if __name__=='__main__':
    df_train = load_training_df()
    print(df_train.head())

    df_supp = load_supplemental_df()
    print(df_supp.head())