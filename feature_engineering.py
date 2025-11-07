import pandas as pd
import numpy as np
from utils import DatasetConfig
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import pickle
from tqdm.auto import tqdm


def interaction_features(df: pd.DataFrame):
    """
    This function constructs per-player, per-play interaction features that describe how each player 
        (especially defenders) relates spatially and kinematically to both opponents and 
        (when applicable) their mirrored wide receiver (WR) counterpart.
    """
    out_rows = []

    for (gid, pid), play_df in tqdm(df.groupby(['game_id', 'play_id']), desc="Computing Interactions...", leave=False):
        frames = play_df['frame_id'].unique()
        players = play_df['nfl_id'].unique()
        T = len(frames)
        N = len(players)
        
        if N < 2:
            # too few players
            continue

        # location and velo data arrays (T, N)
        def pivot_array(col):
            tab = play_df.pivot(index='frame_id', columns='nfl_id', values=col)
            tab = tab.reindex(index=frames, columns=players)
            return tab.values  # shape (T, N)
        
        x = pivot_array('x')
        y = pivot_array('y')
        vx = pivot_array('x_velocity')
        vy = pivot_array('y_velocity')

        # should all be valid but just in case
        valid_mask = ~(np.isnan(x) | np.isnan(y))   # shape (T, N)

        # player metadata
        per_player = play_df.drop_duplicates('nfl_id').set_index('nfl_id').reindex(players)
        sides = per_player['player_side'].values  # shape (N,)
        roles = per_player['player_role'].values  # shape (N,)
        # receiver mask (per player)
        receiver_mask = np.isin(roles, ['Targeted Receiver', 'Other Route Runner'])

        # Build pairwise spatial relationships: shapes -> (T, N, N)
        # dx[t,i,j] = x[t,i] - x[t,j]
        dx = x[:, :, None] - x[:, None, :]
        dy = y[:, :, None] - y[:, None, :]
        # pairwise distances (T,N,N)
        dists = np.sqrt(dx**2 + dy**2)

        # opponent mask (N,N) : True when j is an opponent of i
        opp_mask = (sides[:, None] != sides[None, :])  # shape (N,N)

        # Valid (i,j) pairs in each frame
        valid_opp = valid_mask[:, :, None] & valid_mask[:, None, :] & opp_mask[None, :, :]  # (T,N,N)

        # mask out invalid opponents with inf
        dists_opp = np.where(valid_opp, dists, np.inf)  # (T,N,N)

        # nearest opponent distance and counts
        nearest_idx = np.argmin(dists_opp, axis=2)      # (T, N) indices along axis j
        nearest_dist = np.take_along_axis(dists_opp, nearest_idx[..., None], axis=2).squeeze(2)  # (T,N)
        num_nearby_opp_3 = np.sum(dists_opp < 3.0, axis=2)  # (T,N)
        num_nearby_opp_5 = np.sum(dists_opp < 5.0, axis=2)  # (T,N)

        # compute pairwise relative velocities
        rvx = vx[:, :, None] - vx[:, None, :]  # (T,N,N)
        rvy = vy[:, :, None] - vy[:, None, :]  # (T,N,N)
        # unit vector from opponent j to player i: (dx,dy) / dist
        denom = dists + 1e-5
        ux, uy = dx / denom, dy / denom
        closing_pair = -(rvx * ux + rvy * uy)  # (T,N,N)
        # For pairs that are not valid opponents, set to +inf so they won't be chosen by min ops
        closing_pair = np.where(valid_opp, closing_pair, np.inf)  # (T,N,N)
        closing_speed = np.take_along_axis(closing_pair, nearest_idx[..., None], axis=2).squeeze(2)

        # replace invalid entries with default values
        nearest_dist_filled = np.where(np.isfinite(nearest_dist), nearest_dist, 50.0)
        closing_speed_filled = np.where(np.isfinite(nearest_dist), closing_speed, 0)

        # Mirror WR features (for defenders only)
        if receiver_mask.any():
            rec_idx = np.where(receiver_mask)[0]  # indices of receivers
            rec_dx = dx[:, :, rec_idx]  # (T, N, N_rec) dx from player i to rec j
            rec_dy = dy[:, :, rec_idx]  # (T, N, N_rec)
            rec_dists = np.sqrt(rec_dx**2 + rec_dy**2)  # (T, N, N_rec)

            # Create mask where both i and rec j are present
            rec_valid_mask = valid_mask[:, :, None] & valid_mask[:, rec_idx][:, None, :]  # (T, N, N_rec)
            rec_dists_masked = np.where(rec_valid_mask, rec_dists, np.inf)  # (T,N,N_rec)

            # nearest receiver index per (T,N) among receivers
            closest_rec_rel_idx = np.argmin(rec_dists_masked, axis=2)  # (T,N) index into rec_idx
            closest_rec_dist = np.take_along_axis(rec_dists_masked, closest_rec_rel_idx[..., None], axis=2).squeeze(2)  # (T,N)

            # Gather receiver velocities for chosen receiver
            rec_vx, rec_vy = vx[:, rec_idx], vy[:, rec_idx]  # (T, N_rec)
            rec_x,  rec_y  =  x[:, rec_idx],  y[:, rec_idx]  # (T, N_rec)

            # Gather nearest receiver information
            rec_vx_chosen = np.take_along_axis(rec_vx, closest_rec_rel_idx, axis=1)  # (T,N)
            rec_vy_chosen = np.take_along_axis(rec_vy, closest_rec_rel_idx, axis=1)  # (T,N)
            rec_x_chosen = np.take_along_axis(rec_x, closest_rec_rel_idx, axis=1)  # (T,N)
            rec_y_chosen = np.take_along_axis(rec_y, closest_rec_rel_idx, axis=1)

            # mirror offsets
            mirror_offset_x = x - rec_x_chosen  # (T,N)
            mirror_offset_y = y - rec_y_chosen  # (T,N)
            mirror_wr_dist = np.where(np.isfinite(closest_rec_dist), closest_rec_dist, 50.0)
            mirror_wr_vx = np.where(np.isfinite(closest_rec_dist), rec_vx_chosen, 0.0)
            mirror_wr_vy = np.where(np.isfinite(closest_rec_dist), rec_vy_chosen, 0.0)
        else:
            # no receivers in play -> zeros/defaults
            mirror_offset_x = np.zeros_like(x)
            mirror_offset_y = np.zeros_like(x)
            mirror_wr_vx = np.zeros_like(x)
            mirror_wr_vy = np.zeros_like(x)
            mirror_wr_dist = np.full_like(x, 50.0)

        # collect valid (frame, player) rows
        t_idx, p_idx = np.where(valid_mask)  # arrays of equal length L
        L = len(t_idx)
        if L == 0:
            continue

        # extract feature values for those (t,p) pairs
        def gather(arr):
            return arr[t_idx, p_idx]

        rows = pd.DataFrame({
            'game_id': gid,
            'play_id': pid,
            'nfl_id': players[p_idx],
            'frame_id': frames[t_idx],
            'nearest_opp_dist': gather(nearest_dist_filled),
            'closing_speed': gather(closing_speed_filled),
            'num_nearby_opp_3': gather(num_nearby_opp_3).astype(int),
            'num_nearby_opp_5': gather(num_nearby_opp_5).astype(int),
            'mirror_wr_vx': gather(mirror_wr_vx),
            'mirror_wr_vy': gather(mirror_wr_vy),
            'mirror_offset_x': gather(mirror_offset_x),
            'mirror_offset_y': gather(mirror_offset_y),
            'mirror_wr_dist': gather(mirror_wr_dist)
        })

        # zero-out mirror WR features for non-defenders
        not_defender = roles[p_idx] != 'Defensive Coverage'
        rows.loc[not_defender, ['mirror_wr_vx','mirror_wr_vy',
                                'mirror_offset_x','mirror_offset_y','mirror_wr_dist']] = [0.0, 0.0, 0.0, 0.0, 50.0]
        
        out_rows.append(rows)

    # concat all plays
    if len(out_rows) == 0:
        # return empty DataFrame with columns
        cols = ['game_id','play_id','nfl_id','frame_id','nearest_opp_dist','closing_speed',
                'num_nearby_opp_3','num_nearby_opp_5','mirror_wr_vx','mirror_wr_vy',
                'mirror_offset_x','mirror_offset_y','mirror_wr_dist']
        return pd.DataFrame(columns=cols)

    result = pd.concat(out_rows, ignore_index=True)
    result = result.astype({
        'nearest_opp_dist': float,
        'closing_speed': float,
        'num_nearby_opp_3': int,
        'num_nearby_opp_5': int,
        'mirror_wr_dist': float,
    })
    return result

def route_features(df, cfg, fit=True):
    """
    Extract short-window route dynamics at a frame level
    Each player-play trajectory is divided into overlapping windows of length
    'cfg.dataset.history_window' spaced by 'cfg.dataset.stride'
    This includes overlapping window averaging
    """

    temp_records = []   # with overlaps
    window_size = cfg.history_window
    stride = cfg.stride

    for (gid, pid, nflid), group in tqdm(df.groupby(['game_id', 'play_id', 'nfl_id']), desc='Computing Route...', leave=False):
        traj = group.reset_index(drop=True)
        n = len(traj)
        if n < window_size:
            continue

        locations = traj[['x', 'y']].to_numpy()
        speeds = traj['s'].to_numpy()
        frame_ids = traj['frame_id'].to_numpy()

        # sliding window feature extraction
        for start in range(0, n, stride):
            end = min(start + window_size, n)   # safe slicing
            loc_seg = locations[start:end]
            spd_seg = speeds[start:end]
            fid_seg = frame_ids[start:end]  # assign to all frames in this slice

            if len(loc_seg) < 2:
                continue

            # geometric measures
            diffs = np.diff(loc_seg, axis=0)
            step_dists = np.sqrt((diffs ** 2).sum(axis=1))
            total_dist = np.sum(step_dists)
            dx, dy = loc_seg[-1] - loc_seg[0]
            displacement = np.sqrt(dx ** 2 + dy ** 2)
            straightness = displacement / (total_dist + 1e-5)

            # turning measures
            if len(diffs) > 1:
                angles = np.degrees(np.arctan2(diffs[:, 1], diffs[:, 0])) % 360     # [0, 360)
                angle_changes = np.abs(np.diff(angles))
                angle_changes = np.minimum(angle_changes, 360 - angle_changes)    # normalize to [0, 180]
                max_turn = np.max(angle_changes)  
                mean_turn = np.mean(angle_changes)
            else:
                max_turn = mean_turn = 0.0

            # speed measures
            speed_mean = np.mean(spd_seg)
            speed_change = spd_seg[-1] - spd_seg[0]

            # assign same features to all frames in window
            for fid in fid_seg:
                temp_records.append({
                    'game_id': gid, 'play_id': pid, 'nfl_id': nflid,
                    'frame_id': fid,
                    'traj_straightness': straightness,
                    'traj_max_turn': max_turn,
                    'traj_mean_turn': mean_turn,
                    'traj_depth': abs(dx),
                    'traj_width': abs(dy),
                    'speed_mean': speed_mean,
                    'speed_change': speed_change,
                })
    
    print('***Averaging overlapping windows***')
    # aggregate overlapping windows by averaging
    route_df = pd.DataFrame(temp_records)
    if route_df.empty:
        return route_df, kmeans, scaler     # nothing to fit/predict
    
    route_df = (
        route_df.groupby(['game_id', 'play_id', 'nfl_id', 'frame_id'], as_index=False).mean()
    )
    
    # clustering
    print('***Clustering***')
    feat_cols = [
        'traj_straightness', 'traj_max_turn', 'traj_mean_turn',
        'traj_depth', 'traj_width', 'speed_mean', 'speed_change'
    ]
    X = route_df[feat_cols].fillna(0)
    scaler_save_path = os.path.join(cfg.data_dir, 'Saves', 'route_scaler.pkl')
    kmeans_save_path = os.path.join(cfg.data_dir, 'Saves', 'route_kmeans.pkl')
    
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(
            n_clusters=cfg.n_route_clusters,
            random_state = 42,
            n_init=10
        )
        route_df['route_pattern'] = kmeans.fit_predict(X_scaled)
        
        # save scaler and kmeans
        with open(scaler_save_path, 'wb') as f:
            pickle.dump(scaler, f)
        with open(kmeans_save_path, 'wb') as f:
            pickle.dump(kmeans, f)

        return route_df
    else:
        with open(scaler_save_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(kmeans_save_path, 'rb') as f:
            kmeans = pickle.load(f)
        X_scaled = scaler.transform(X)
        route_df['route_pattern'] = kmeans.predict(X_scaled)
        return route_df

def engineer_features(df: pd.DataFrame, cfg: DatasetConfig, verbose=True, training=True) -> pd.DataFrame:
    """
    Engineers revised set of new features for the given DataFrame.
    NOTE: Coordinate system 0 deg is along y-axis and increases clockwise

    Parameters:
        df (pd.DataFrame): Input DataFrame
        cfg (DatasetConfig): dataset configuration
        verbose (bool): Whether to print progress messages
        training (bool): Whether creating training df or not
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
    df['rolling_x_velocity_std'] = grouped['x_velocity'].rolling(window=cfg.history_window, min_periods=1).std().reset_index(level=[0,1,2], drop=True)
    df['rolling_y_velocity_std'] = grouped['y_velocity'].rolling(window=cfg.history_window, min_periods=1).std().reset_index(level=[0,1,2], drop=True)

    ### generate rolling mean/std for acceleration
    df['rolling_a_std'] = grouped['a'].rolling(window=cfg.history_window, min_periods=1).std().reset_index(level=[0,1,2], drop=True)

    # fill NaNs with zeros
    for col in ['rolling_x_velocity_std', 'rolling_y_velocity_std', 'rolling_a_std']:
        df.fillna({col: 0.0}, inplace=True)

    ######################################
    ### Spatial-Temporal Relationships ###
    ######################################
    if verbose:
        print('---Generating Spatial-Temporal Relationships---')

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
        print('---Generating Nearby Player Information---')

    df_interaction = interaction_features(df)
    df = df.merge(df_interaction, on=['game_id', 'play_id', 'nfl_id', 'frame_id'], how='left')

    ### generate ball air time
    dt = cfg.dt
    df['air_time'] = df['num_frames_output'] * dt

    ### generate time until play ends
    num_frames_input = df.groupby(['game_id', 'play_id'])['frame_id'].transform('max')
    df['time_to_end'] = (num_frames_input + df['num_frames_output'] - df['frame_id']) * dt

    if verbose:
        print('---Generating Projected Endpoints---')
    # project initially off player momentum
    df['projected_x'] = df['x'] + df['x_velocity'] * df['time_to_end']
    df['projected_y'] = df['y'] + df['y_velocity'] * df['time_to_end']
    
    # assume targeted receivers will end up at the ball
    targeted_mask = df['player_role'] == 'Targeted Receiver'
    df.loc[targeted_mask, 'projected_x'] = df.loc[targeted_mask, 'ball_land_x']
    df.loc[targeted_mask, 'projected_y'] = df.loc[targeted_mask, 'ball_land_y']

    # assume defenders mirror receivers by maintaining offset
    defender_mask = df['player_role'] == 'Defensive Coverage'
    has_mirror = df.get('mirror_offset_x', 0.0).notna() & (df.get('mirror_wr_dist', 50) < 15)
    coverage_mask = defender_mask & has_mirror

    df.loc[coverage_mask, 'projected_x'] = (
        df.loc[coverage_mask, 'ball_land_x'] + 
        df.loc[coverage_mask, 'mirror_offset_x'].fillna(0.0)
    )
    df.loc[coverage_mask, 'projected_y'] = (
        df.loc[coverage_mask, 'ball_land_y'] + 
        df.loc[coverage_mask, 'mirror_offset_y'].fillna(0.0)
    )

    # Projections may push players out of bounds
    df['projected_x'] = df['projected_x'].clip(0, 120)
    df['projected_y'] = df['projected_y'].clip(0, 53.3)

    ### compute features from projected path
    # movement vectors
    df['projected_vector_x'] = df['projected_x'] - df['x']
    df['projected_vector_y'] = df['projected_y'] - df['y']
    df['projected_dist'] = np.sqrt(df['projected_vector_x'] ** 2 + df['projected_vector_y'] ** 2)

    # infer required velocity
    df['projected_x_velocity'] = df['projected_vector_x'] / df['time_to_end']
    df['projected_y_velocity'] = df['projected_vector_y'] / df['time_to_end']

    # compare against velocity data
    df['projected_x_velocity_err'] = df['projected_x_velocity'] - df['x_velocity']
    df['projected_y_velocity_err'] = df['projected_y_velocity'] - df['y_velocity']
    df['projected_velocity_err'] = np.sqrt(df['projected_x_velocity_err'] ** 2 + df['projected_y_velocity_err'] ** 2)

    # infer acceleration
    df['projected_x_acceleration'] = 2 * df['projected_vector_x'] / (df['time_to_end'] * df['time_to_end']).clip(-10, 10)
    df['projected_y_acceleration'] = 2 * df['projected_vector_y'] / (df['time_to_end'] * df['time_to_end']).clip(-10, 10)

    # projected path alignment
    projected_unit_x = df['projected_vector_x'] / (df['projected_dist'] + 0.1)
    projected_unit_y = df['projected_vector_y'] / (df['projected_dist'] + 0.1)
    df['projected_alignment'] = (df['x_velocity'] * projected_unit_x +
                                 df['y_velocity'] * projected_unit_y) / (df['s'] + 0.1)
    
    # role-specific projection data
    df['projected_receiver_speed'] = df['is_targeted_receiver'] * df['projected_dist'] / (df['time_to_end'] + 0.1)
    df['projected_defender_importance'] = df['is_defensive_coverage'] * (1.0 / (df.get('mirror_wr_dist', 50.0) + 1.0))

    if verbose:
        print('---Route Analysis---')

    route_df = route_features(df, cfg, fit=training)
        
    df = df.merge(route_df, on=['game_id', 'play_id', 'nfl_id', 'frame_id'], how='left')
    
    if verbose:
        print('---Feature Engineering Complete---')

    return df


if __name__ == '__main__':
    import utils
    cfg = utils.Config()
    df_input, df_output, _, _ = utils.load_prediction_data(cfg.dataset)
    df_input['player_height'] = df_input['player_height'].apply(utils.height_to_inches)
    df_input = engineer_features(utils.invert_direction(df_input), cfg.dataset)
    df_output = utils.invert_direction(utils.map_play_direction(df_input, df_output))