import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
from feature_engineering import engineer_features
from data_sequencing import generate_sequences_4D
from train import reconstruct_absolute_from_deltas

def plot_play(df_input, df_output, gid, pid, model, scaler):
    """
    Visualize a single play and overlay GNN-Transformer predicted future positions.

    Parameters
    ----------
    df_input : pd.DataFrame
        Input DataFrame containing past frames for the play.
    df_output : pd.DataFrame
        Output DataFrame containing true future frames for the play.
    gid : int
        Game id of the play to plot.
    pid : int
        Play id of the play to plot.
    model : nn.Module
        Trained GNNTransformer model used to predict futures.
    scaler : FeatureScaler
        Feature scaler used to scale/ inverse-scale model inputs and outputs.

    Returns
    -------
    None
        Displays a matplotlib figure with input, true future and predicted trajectories.
    """
    # check if gid and pid are in both df_input and df_output
    assert (gid in df_input['game_id'].values) and (pid in df_input['play_id'].values)
    assert (gid in df_output['game_id'].values) and (pid in df_output['play_id'].values)

    play = df_input[(df_input['game_id'] == gid) & (df_input['play_id'] == pid)]
    play_output = df_output[(df_output['game_id'] == gid) & (df_output['play_id'] == pid)].copy()
    
    engineered_play = engineer_features(play)
    
    (
        X,         # (1, N, T_in, F)
        y_sequence,   # (1, N, T_out, 2)
        player_mask,  # (1, N)
        target_mask,  # (1, N)
        y_mask,       # (1, N, T_out)
        id_map        # (1, N)
    ) = generate_sequences_4D(
        engineered_play,
        df_output=play_output,
        sequence_length=10,
    )

    num_frames_output = int(play.iloc[0]['num_frames_output'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # scale input
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    player_mask_t = torch.tensor(player_mask, dtype=torch.bool, device=device)
    target_mask_t = torch.tensor(target_mask, dtype=torch.bool, device=device)

    # predict displacements
    model.eval()
    with torch.no_grad():
        delta_scaled = model.predict(
            src=X_tensor,
            future_len=num_frames_output,
            target_mask=target_mask_t,
            player_mask=player_mask_t
        ).cpu().numpy()  # (1, N, T_out, 2)
        
    # get last known player positions
    last_pos = X[:, :, -1, 0:2]     # (1, N, 2)
    last_pos= np.where(player_mask[..., None], last_pos, np.nan)  # invalidate padded players

    # scale last positions
    last_pos = last_pos[..., None, :]   # (1, N, 1, 2)
    last_pos_scaled = scaler.transform(last_pos, feature_cols=['x', 'y'])[:, :, 0, :]   # (1, N, 2)

    # reconstruct absolute trajectories
    abs_pred_scaled = reconstruct_absolute_from_deltas(
        last_pos_scaled, delta_scaled, player_mask
    )

    # inverse scale dx, dy
    _, N, T_out, _ = abs_pred_scaled.shape
    abs_pred_scaled = abs_pred_scaled.reshape(1, N, T_out, -1)
    abs_pred = scaler.inverse_transform(abs_pred_scaled, feature_cols=['x', 'y'])   # (1, N, T_out, 2)

    # flatten shapes
    B, N, T_out, _ = abs_pred.shape
    player_ids = np.asarray(id_map).reshape(B, N)[0]    # (N,)
    valid_indices = np.where(target_mask[0])[0]        # indices for X and player_ids

    # check
    play_nfls = set(play_output['nfl_id'].unique())
    missing = [pid for pid in player_ids if (pid != -1 and pid not in play_nfls)]
    if missing:
        print("Warning: some player_ids not in play_output:", missing)

    # initialize prediction columns as NaN
    play_output = play_output.copy()
    play_output['transformer_x'] = np.nan
    play_output['transformer_y'] = np.nan

    # scatter back to full df
    for idx in valid_indices:
        nflid = int(player_ids[idx])
        if nflid == -1:
            # padded player, skip
            continue
        mask = (play_output['nfl_id'] == nflid)
        play_output.loc[mask, 'transformer_x'] = abs_pred[0, idx, :, 0]
        play_output.loc[mask, 'transformer_y'] = abs_pred[0, idx, : ,1]

    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # plot field
    field_width, field_length = 53.5, 120
    # field boundary
    field_rect = patches.Rectangle((0, 0), field_length, field_width,
                                   linewidth=2, edgecolor='green', facecolor='none', alpha=0.5)
    ax.add_patch(field_rect)

    # end zones
    ax.add_patch(patches.Rectangle((0, 0), 10, field_width,
                                   facecolor='lightgray', alpha=0.3))
    ax.add_patch(patches.Rectangle((100, 0), 10, field_width,
                                   facecolor='lightgray', alpha=0.3))
    # vertical yard lines
    for x in range(10, 120, 10):
        ax.axvline(x=x, color='black' if x not in (10, 110) else 'green', linestyle='--', linewidth=1.0, alpha=0.5)

    # color mapping
    color_map = {}
    color_cycle = plt.cm.tab10.colors
    color_map = {pos: color_cycle[i % len(color_cycle)] for i, pos in enumerate(play['player_position'].unique())}

    ax.scatter(play.iloc[0]['ball_land_x'], play.iloc[0]['ball_land_y'], s=30, marker='x', color='black')

    for player_id, player_df in play.groupby('nfl_id'):
        # pre throw
        pos = player_df.iloc[0]['player_position']
        color = color_map.get(pos, 'gray')

        # input trajectory
        ax.scatter(player_df['x'], player_df['y'], s=35, alpha=0.8, label=pos, color=color)

        # check for post throw
        if player_id in play_output['nfl_id'].values:
            true_future = play_output[play_output['nfl_id'] == player_id]
            ax.scatter(true_future['x'], true_future['y'], s=35, color=color, edgecolors='black', label=None)
            ax.scatter(true_future['transformer_x'], true_future['transformer_y'], s=55, linewidths=1.8, marker='D', facecolor='none', edgecolor=color, label=None)

    # --- First legend: player positions (colors) ---
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_positions = ax.legend(by_label.values(), by_label.keys(),
                                title="Player Position", loc='upper left', frameon=True)

    # --- Second legend: data sources (marker shapes) ---
    # Create custom legend handles for data type
    ball_handle = mlines.Line2D([], [], color='gray', marker='x',
                             linestyle='none', markersize=7, label='Ball Land')

    input_handle = mlines.Line2D([], [], color='gray', marker='o',
                                linestyle='None', markersize=7,
                                label='Input (Past)', markerfacecolor='gray')

    true_handle = mlines.Line2D([], [], color='gray', marker='o',
                                linestyle='None', markersize=7,
                                label='True Future', markerfacecolor='gray', markeredgecolor='black')

    pred_handle = mlines.Line2D([], [], color='gray', marker='D',
                                linestyle='None', markersize=9,
                                label='Predicted Future', markerfacecolor='none',
                                markeredgewidth=1.8)

    legend_sources = ax.legend(handles=[ball_handle, input_handle, true_handle, pred_handle],
                            title="Data Source", loc='upper right', frameon=True)

    # Add both legends to the same axes
    ax.add_artist(legend_positions)

    ax.set_title('Player Routes with Transformer Predictions')
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-1, field_length+1])
    ax.set_ylim([-2, field_width+2])
    plt.tight_layout()
    plt.show()