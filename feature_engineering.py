import pandas as pd
import numpy as np
from utils import height_to_inches, Config


def engineer_features(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Engineers revised set of new features for the given DataFrame.
    NOTE: Coordinate system 0 deg is along y-axis and increases clockwise

    Parameters:
        df (pd.DataFrame): Input DataFrame
        verbose (bool): Whether to print progress messages
    Returns:
        df (pd.DataFrame): Transformed DataFrame
    """
    df = df.copy()
    df.sort_values(by=['game_id', 'play_id', 'nfl_id', 'frame_id'], ascending=True, inplace=True)

    #########################
    ### One-Hot Encodings ###
    #########################
    if verbose:
        print('---Generating One-Hot Encodings---')

    # vectorized .map() to set encodings
    side_map = {'Offense': 1, 'Defense': 0}

    ### generate offense or defense switches
    df['is_offense'] = df['player_side'].map(side_map).fillna(0).astype(int)
    df['is_defense'] = 1 - df['is_offense']

    ### generate player_role switches
    role_cols = ['is_defensive_coverage', 'is_other_route_runner', 'is_passer', 'is_targeted_receiver']
    role_targets = ['Defensive Coverage', 'Other Route Runner', 'Passer', 'Targeted Receiver']
    for col, role in zip(role_cols, role_targets):
        df[col] = (df['player_role'] == role).astype(np.int8)
    
    ######################
    ### Player Physics ###
    ######################
    if verbose:
        print('---Generating Player Physics---')

    ### generate player_bmi [kg/m^2]
    df['player_bmi'] = 703 * df['player_weight'] / (df['player_height'] ** 2)

    ### generate x_velo and y_velo features [yd/s]
    speed = df['s'].to_numpy()
    direction_rad = np.deg2rad(df['dir'].to_numpy())   # deg -> rad
    df['x_velocity'] = speed * np.sin(direction_rad)
    df['y_velocity'] = speed * np.cos(direction_rad)

    ### generate angle diff between orientation and direction [deg] [-180, 180)
    df['angle_diff'] = ((df['o'] - df['dir'] + 180) % 360) - 180

    ### generate player jerk [yd/s^3] and angular velo [deg/s]
    # NOTE: Data grouped by game_id, play_id, nfl_id to ensure jerk
    #       is calculated per player per play per game
    group_keys = ['game_id', 'play_id', 'nfl_id']
    # temporary columns
    df['a_diff'] = df.groupby(group_keys)['a'].transform('diff')
    df['o_diff'] = df.groupby(group_keys)['o'].transform('diff')
    
    # delta t is 0.1 for all timesteps
    df['jerk'] = df['a_diff'] / 0.1
    df['angular_velocity'] = df['o_diff'] / 0.1

    # replace NaNs with next value (assume constant)
    df[['jerk', 'angular_velocity']] = df[['jerk', 'angular_velocity']].bfill()
    # drop temp cols
    df.drop(['a_diff', 'o_diff'], axis=1, inplace=True)

    ##### Rolling Metrics #####
    if verbose:
        print('---Generating Rolling Metrics---')

    ### generate rolling std for velocity
    grouped = df.groupby(group_keys, group_keys=False)
    df['rolling_x_velocity_std'] = grouped['x_velocity'].rolling(window=Config.HISTORY_WINDOW, min_periods=1).std().reset_index(level=[0,1,2], drop=True)
    df['rolling_y_velocity_std'] = grouped['y_velocity'].rolling(window=Config.HISTORY_WINDOW, min_periods=1).std().reset_index(level=[0,1,2], drop=True)

    ### generate rolling mean/std for acceleration
    df['rolling_a_std'] = grouped['a'].rolling(window=Config.HISTORY_WINDOW, min_periods=1).std().reset_index(level=[0,1,2], drop=True)

    # fill NaNs with zeros
    for col in ['rolling_x_velocity_std', 'rolling_y_velocity_std', 'rolling_a_std']:
        df.fillna({col: 0.0}, inplace=True)

    #############################
    ### Spatial Relationships ###
    #############################
    if verbose:
        print('---Generating Spatial Relationships---')

    ### generate euclidean distance to ball_land [yd]
    dx = df['ball_land_x'] - df['x']
    dy = df['ball_land_y'] - df['y']
    df['dist_to_ball_land'] = np.hypot(dx, dy)

    ### generate distance from line of scrimmage [yd]
    # NOTE: with standardized play direction, positive values indicate distance downfield
    df['dist_from_los'] = df['x'] - df['absolute_yardline_number']

    ### generate bearing to ball_land [deg]
    bearing_deg = np.degrees(np.arctan2(dx, dy))       # bearing in deg
    df['bearing_to_ball_land'] = (bearing_deg + 360) % 360 # normalize to [0, 360) 0° along +y, 90° along +x

    ### generate bearing diff between player orientation and bearing to ball_land [deg]
    df['bearing_diff_o'] = ((df['o'] - bearing_deg + 180) % 360) - 180    # normalize to [-180, 180]

    ### generate bearing diff between player direction and bearing to ball_land [deg]
    df['bearing_diff_dir'] = ((df['dir'] - bearing_deg + 180) % 360) - 180

    if verbose:
        print('---Feature Engineering Complete---')

    return df

