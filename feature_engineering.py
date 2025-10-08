import pandas as pd
import numpy as np
from utils import height_to_inches


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers new features for the given DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing raw features.

    Returns:
        pd.DataFrame: DataFrame augmented with engineered features.
    """
    
    ######################
    ### Player Physics ###
    ######################

    ### generate player_bmi [kg/m^2]
    df['player_bmi'] = 703 * df['player_weight'] / (df['player_height'] ** 2)

    ### generate x_velocity and y_velocity features [yd/s]
    speed = df['s']
    direction_rad = df['dir'] * (np.pi / 180)  # Convert degrees to radians
    df['x_velocity'] = speed * np.cos(direction_rad)
    df['y_velocity'] = speed * np.sin(direction_rad)

    ### generate angle diff between orientation and direction [deg]
    orientation = df['o']
    direction = df['dir']
    angle_diff_raw = orientation - direction
    # normalize to [-180, 180]
    df['angle_diff'] = ((angle_diff_raw + 180) % 360) - 180

    ### generate momentum [slug*yd/s]
    df['player_momentum'] = 0.03108 * df['player_weight'] * df['s']

    ### generate player jerk [yd/s^3]
    # NOTE: Data grouped by game_id, play_id, nfl_id to ensure jerk is calculated per player per play
    # calculate jerk for all rows naively
    df['jerk'] = df['a'].diff() / 0.1
    # identify rows with frame_ids > 1 for each player in each play in each game
    same_sequence = (
        (df['game_id'] == df['game_id'].shift()) &
        (df['play_id'] == df['play_id'].shift()) &
        (df['nfl_id'] == df['nfl_id'].shift()) 
    )
    # set jerk to NA for rows with frame_id == 1 (i.e., break in sequence)
    df.loc[~same_sequence, 'jerk'] = pd.NA
    # fill NA values with next frame's jerk value (i.e., assume jerk is constant over the first 0.1s interval)
    df['jerk'] = df['jerk'].bfill()

    ### generate angular velocity [deg/s]
    # NOTE: Data grouped by game_id, play_id, nfl_id to ensure angular velocity is calculated per player per play
    # calculate angular velocity for all rows naively
    df['angular_velocity'] = df['o'].diff() / 0.1
    # identify rows with frame_ids > 1 for each player in each play in each game with same_sequence above
    df['angular_velocity'].loc[~same_sequence] = pd.NA
    # fill NA values with next frame's angular velocity value (i.e., assume angular velocity is constant over the first 0.1s interval)
    df['angular_velocity'] = df['angular_velocity'].bfill()

    ### generate path curvature [1/yd]
    ang_velo_rad = df['angular_velocity'] * (np.pi / 180)  # Convert degrees to radians
    speed_nonzero = df['s'].replace(0, np.nan)  # Replace 0 speed with NaN to avoid division by zero
    df['path_curvature'] = ang_velo_rad / speed_nonzero
    # Fill NaN values (from 0 speed) with 0 curvature (i.e., assume straight path when stationary)
    df['path_curvature'] = df['path_curvature'].fillna(0)

    #############################
    ### Spatial Relationships ###
    #############################

    ### generate euclidean distance to ball_land [yd]
    df['euclidean_dist_to_ball_land'] = np.sqrt(
        (df['x_input'] - df['ball_land_x'])**2 + (df['y_input'] - df['ball_land_y'])**2
    )
    
    ### generate distance from line of scrimmage [yd]
    # NOTE: with standardized play direction, positive values indicate distance downfield
    df['dist_from_los'] = df['x_input'] - df['absolute_yardline_number']

    ### generate bearing to ball_land [deg]
    delta_x = df['ball_land_x'] - df['x_input']
    delta_y = df['ball_land_y'] - df['y_input']
    bearing_rad = np.arctan2(delta_y, delta_x)  # Bearing in radians
    bearing_deg = np.degrees(bearing_rad)  # Convert to degrees
    df['bearing_to_ball_land'] = (bearing_deg + 360) % 360  # Normalize to [0, 360)

    ### generate bearing diff between player orientation and bearing to ball_land [deg]
    bearing_diff_raw = df['o'] - df['bearing_to_ball_land']
    # normalize to [-180, 180]
    df['bearing_diff'] = ((bearing_diff_raw + 180) % 360) - 180
    
    ### generate bearing diff between player direction and bearing to ball_land [deg]
    bearing_diff_dir_raw = df['dir'] - df['bearing_to_ball_land']
    # normalize to [-180, 180]
    df['bearing_diff_dir'] = ((bearing_diff_dir_raw + 180) % 360) - 180

