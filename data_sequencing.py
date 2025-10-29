import numpy as np
import pandas as pd
from utils import DatasetConfig


def estimate_angle_diff(angle_hist: np.ndarray) -> float:
    """
    Estimate a representative angle (degrees) from a history of circular angle values.

    This uses the circular mean formula to account for angle wrap-around
    (e.g. values near -180/180 degrees).

    Parameters
    ----------
    angle_hist : np.ndarray
        1-D array of historical angle values in degrees.

    Returns
    -------
    float
        Estimated mean angle in degrees, normalized to the range [-180, 180].
    """
    radians = np.deg2rad(angle_hist)
    mean_angle = np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
    mean_deg = np.rad2deg(mean_angle)
    mean_deg = (mean_deg + 180) % 360 - 180
    return mean_deg

def generate_sequences_4D(
        df_input:pd.DataFrame, cfg: DatasetConfig, df_output:pd.DataFrame=None,
        test_template:pd.DataFrame=None, is_training=True,
        sequence_length:int=5, data_fraction:float=1.0,
        min_players: int=7
                       )->tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare all features for sequential model training or testing

    Parameters:
        df_input (pd.DataFrame): Input DataFrame
        cfg (DatasetConfig): dataset configuration
        df_output (pd.DataFrame, optional): Output DataFrame. Defaults to None.
        test_template (pd.DataFrame, optional): Prediction template. Defaults to None.
        is_training (bool): switch for training purposes. Defaults to True.
        sequence_length (int): number of time frames to use for training. Defaults
            to 5 (smallest sequence in dataset).
        data_fraction (float): fraction of data to use for training. Defaults to 1.0.
        min_players (int): minimum number of valid players required per play to include
        
    Returns:
        X (np.ndarray): (B, N, T_in, F)
        y (np.ndarray or None): Target sequences (B, N, T_out, out_features),
            or (B, N, 0, 0) if is_training=False
        player_mask: (B, N) True = valid player, False = padded
        target_mask: (B, N) True = player has ground truth targets
        y_mask: (B, N, T_out) mask where True = valid y value, False = padded
        ids: (B, N) nfl_id of players, null where not a player
    """
    
    # set target df
    df_target = df_output if is_training else test_template

    # ensure either df_output or test_template is provided
    assert not (df_output is None and test_template is None), "No Target DataFrame provided"

    # sorting
    df_input.sort_values(by=cfg.id_cols + ['frame_id'], inplace=True)
    df_target.sort_values(by=cfg.id_cols + ['frame_id'], inplace=True)

    # grouping
    input_groups = dict(tuple(df_input.groupby(['game_id', 'play_id'], sort=False)))
    if is_training:
        output_groups = dict(tuple(df_output.groupby(cfg.id_cols, sort=False)))
    total_plays = len(input_groups.keys())
    print(f"Begin sequencing {total_plays} plays in input data")

    # Determine global max number players in dataset
    N_max = df_input.groupby(['game_id', 'play_id'])['nfl_id'].nunique().max()
    print(f"Max players per play: {N_max}")
    skipped = 0
    global_max_ylen = 0

    X_list, y_list, player_mask_list, target_mask_list, y_mask_list, id_list = [], [], [], [], [], []
    for idx, ((gid, pid), play_df) in enumerate(input_groups.items()):
        player_groups = dict(tuple(play_df.groupby('nfl_id', sort=False)))
        players = list(player_groups.keys())
        N = len(players)
        
        # check against minimum
        if N < min_players:
            skipped += 1
            continue

        # init containers
        X_play = np.full((N_max, sequence_length, len(cfg.features)), np.nan, dtype=np.float32)
        player_mask = np.zeros(N_max, dtype=bool)
        target_mask = np.zeros(N_max, dtype=bool)
        ids_play = np.full((N_max), np.nan, dtype=np.float32)

        # fill per-player sequences
        for i, nflid in enumerate(players):
            player_seq = player_groups[nflid].tail(sequence_length)
            seq = player_seq[cfg.features].to_numpy(dtype=np.float32)
            pad_len = sequence_length - len(seq)
            if pad_len > 0:
                seq = np.vstack([np.full((pad_len, len(cfg.features)), np.nan, dtype=np.float32), seq])
            X_play[i, :, :] = seq
            player_mask[i] = True
            target_mask[i] = bool(player_seq.iloc[-1]['player_to_predict'])
            ids_play[i] = nflid

        # append inputs/masks
        X_list.append(X_play)
        player_mask_list.append(player_mask)
        target_mask_list.append(target_mask)
        id_list.append(ids_play)

        # prepare target sequence if training
        if is_training:
            # just target x, y data
            if not cfg.target_features:
                y_dict = {nflid: output_groups[(gid,pid,nflid)][['x','y']].to_numpy(dtype=np.float32)
                          for nflid in players if (gid,pid,nflid) in output_groups}
                max_ylen = max((len(v) for v in y_dict.values()), default=0)
                global_max_ylen = max(global_max_ylen, max_ylen)

                y_play = np.zeros((N_max, max_ylen, 2), dtype=np.float32)
                y_mask_play = np.zeros((N_max, max_ylen), dtype=bool)
                for i, nflid in enumerate(players):
                    if nflid in y_dict:
                        yy = y_dict[nflid]
                        y_play[i, :len(yy)] = yy
                        y_mask_play[i, :len(yy)] = True
                y_list.append(y_play)
                y_mask_list.append(y_mask_play)
            
            # else:
            #     ### compute future features based on x, y position
            #     # absolute positions
            #     xs, ys = group['x'].to_numpy(), group['y'].to_numpy()

            #     # estimate velocity and accleration
            #     vxs = np.gradient(xs, dt)
            #     vys = np.gradient(ys, dt)
            #     axs = np.gradient(vxs, dt)
            #     ays = np.gradient(vys, dt)

            #     s = np.sqrt(vxs**2 + vys**2)
            #     a = np.sqrt(axs**2 + ays**2)

            #     # estimate direction
            #     dxs = np.diff(np.r_[df_sequence['x'].iloc[-1], xs])
            #     dys = np.diff(np.r_[df_sequence['y'].iloc[-1], ys])
            #     dir = (np.degrees(np.arctan2(dxs, dys))) % 360     # 0° along +y, 90° along +x

            #     # estimate orientation using 5 most recent time steps
            #     angle_hist = df_sequence['angle_diff'].iloc[-5:].to_numpy()
            #     mean_angle_diff = estimate_angle_diff(angle_hist)
            #     # calculate noise based on existing variance in angle_hist
            #     angle_std = np.std(angle_hist)
            #     noise = np.random.normal(0, 0.1*angle_std, len(dir))
            #     # add estimated angle diff and noise to dir vector
            #     o = (dir + mean_angle_diff + noise + 360) % 360      # [0, 360)

            #     # Engineer target features
            #     y_engineered = engineer_features(pd.DataFrame({
            #         'game_id': gid, 'play_id':pid, 'nfl_id':nflid,
            #         'frame_id':group['frame_id'], 'num_frames_output':data.iloc[0]['num_frames_output'],
            #         'x':xs, 'y':ys, 's':s, 'a':a, 'dir':dir, 'o':o,
            #         'absolute_yardline_number':data.iloc[0]['absolute_yardline_number'],
            #         'player_height':data.iloc[0]['player_height'],
            #         'player_weight':data.iloc[0]['player_weight'],
            #         'player_side':data.iloc[0]['player_side'],
            #         'player_role':data.iloc[0]['player_role'],
            #         'ball_land_x':data.iloc[0]['ball_land_x'],
            #         'ball_land_y':data.iloc[0]['ball_land_y']
            #     }), verbose=False)
                
            #     # compute displacement for each output frame
            #     y_engineered = y_engineered.assign(x=y_engineered['x'] - df_sequence['x'].iloc[-1],
            #                                        y=y_engineered['y'] - df_sequence['y'].iloc[-1])
            #     y_list.append(y_engineered[cfg.features].to_numpy())

        # report out
        if (idx + 1) % max(1, total_plays // 10) == 0:
            print(f"Processed {(idx + 1) / total_plays * 100:.0f}% of plays")

        # check against data fraction
        if data_fraction < 1.0 and (idx+1) / total_plays >= data_fraction:
            break

    # convert to ndarrys
    X = np.stack(X_list)
    player_mask = np.stack(player_mask_list)
    target_mask = np.stack(target_mask_list)
    ids = np.stack(id_list)
    y = None

    # pad y data to maximum output frames
    if is_training:
        y_padded = np.zeros((len(y_list), N_max, global_max_ylen, 2), dtype=np.float32)
        y_mask_padded = np.zeros((len(y_list), N_max, global_max_ylen), dtype=bool)
        for i, (yy, yy_mask) in enumerate(zip(y_list, y_mask_list)):
            T = yy.shape[1]
            y_padded[i, :, :T, :] = yy
            y_mask_padded[i, :, :T] = yy_mask
        y, y_mask = y_padded, y_mask_padded
    else:
        y, y_mask = None, None
        
    print(f"Generated {len(X)} plays (skipped {skipped}) | max target length {global_max_ylen}")
    return X, y, player_mask, target_mask, y_mask, ids