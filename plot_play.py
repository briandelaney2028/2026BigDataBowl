import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import os
import math
from feature_engineering import engineer_features
from data_sequencing import generate_sequences_4D
from train import reconstruct_absolute_from_deltas
from utils import Config
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

def plot_play(df_input, df_output, model, scaler, cfg):
    """
    Visualize a single play and overlay GNN-Transformer predicted future positions.

    Parameters
    ----------
    df_input (pd.DataFrame):
        Input DataFrame containing past frames for the play.
    df_output (pd.DataFrame):
        Output DataFrame containing true future frames for the play.
    model (nn.Module):
        Trained GNNTransformer model used to predict futures.
    scaler (FeatureScaler):
        Feature scaler used to scale/ inverse-scale model inputs and outputs.

    Returns
    -------
    None
        Displays a matplotlib figure with input, true future and predicted trajectories.
    """
    
    plays = df_input[['game_id', 'play_id']].drop_duplicates().values
    n_plays = len(plays)

    # grid layout
    ncols = int(math.ceil(math.sqrt(n_plays)))
    nrows = int(math.ceil(n_plays / ncols))
        
    # plotting
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(ncols * 6, nrows * 6), squeeze=False)
    axes = axes.flatten()

    # colormapping
    all_positions = df_input['player_position'].unique()
    color_cycle = plt.cm.tab10.colors
    color_map = {pos: color_cycle[i % len(color_cycle)] for i, pos in enumerate(all_positions)}
    
    for idx, (gid, pid) in enumerate(plays):
        ax = axes[idx]
        ax.set_title(f'Game {gid}, Play {pid}')

        play_input = df_input[(df_input['game_id'] == gid) & (df_input['play_id'] == pid)]
        play_output = df_output[(df_output['game_id'] == gid) & (df_output['play_id'] == pid)].copy()
        
        (
            X,         # (1, N, T_in, F)
            y_sequence,   # (1, N, T_out, 2)
            player_mask,  # (1, N)
            target_mask,  # (1, N)
            y_mask,       # (1, N, T_out)
            id_map        # (1, N)
        ) = generate_sequences_4D(
            play_input,
            cfg.dataset,
            df_output=play_output
        )

        num_frames_output = int(play_input.iloc[0]['num_frames_output'])
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

        # initialize prediction columns as NaN
        play_output = play_output.copy()
        play_output['transformer_x'] = np.nan
        play_output['transformer_y'] = np.nan

        # scatter back to full df
        for i in valid_indices:
            nflid = int(player_ids[i])
            if nflid == -1:
                # padded player, skip
                continue
            mask = (play_output['nfl_id'] == nflid)
            play_output.loc[mask, 'transformer_x'] = abs_pred[0, i, :, 0]
            play_output.loc[mask, 'transformer_y'] = abs_pred[0, i, : ,1]

    
        # plot field
        field_width, field_length = 53.5, 120
        # field boundary
        ax.add_patch(patches.Rectangle((0, 0), field_length, field_width,
                                    linewidth=2, edgecolor='green', facecolor='none', alpha=0.5))
        # end zones
        ax.add_patch(patches.Rectangle((0, 0), 10, field_width,
                                    facecolor='lightgray', alpha=0.3))
        ax.add_patch(patches.Rectangle((100, 0), 10, field_width,
                                    facecolor='lightgray', alpha=0.3))
        # vertical yard lines
        for x in range(10, 120, 10):
            ax.axvline(x=x, color='black' if x not in (10, 110) else 'green', linestyle='--', linewidth=1.0, alpha=0.5)

        for player_id, player_df in play_input.groupby('nfl_id'):
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

        # plot ball land
        ax.scatter(play_input.iloc[0]['ball_land_x'], play_input.iloc[0]['ball_land_y'], s=120, marker='x', color='black')

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-1, field_length+1])
        ax.set_ylim([-2, field_width+2])
        
    # hide any unsued subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    # --- First legend: player positions (colors) ---
    for pos, color in color_map.items():
        axes[0].scatter([], [], color=color, label=pos, s=40)
    legend_positions = axes[0].legend(title='Position', loc='upper left')
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

    legend_sources = axes[0].legend(handles=[ball_handle, input_handle, true_handle, pred_handle],
                            title="Data Source", loc='upper right', frameon=True)
    axes[0].add_artist(legend_positions)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    from utils import load_prediction_data, height_to_inches, invert_direction, map_play_direction
    from FeatureScaler import FeatureScaler

    cfg = Config()

    df_input, df_output, _, _ = load_prediction_data(cfg.dataset)
    df_input['player_height'] = df_input['player_height'].apply(height_to_inches)

    plays = [
        (2023112300, 55),
        (2023090700, 1711),
        (2023090700, 101),
        (2023113000, 87)
    ]

    # plot_play(df_input, df_output, plays, model, scaler, cfg)

    # get 9 random plays
    unique_pairs = df_input[['game_id', 'play_id']].drop_duplicates()
    sampled_pairs = unique_pairs.sample(n=9, replace=False).values
    plays = [(id[0], id[1]) for id in sampled_pairs]    

    df_plays = pd.DataFrame(plays, columns=['game_id', 'play_id'])
    df_filtered = df_input.merge(df_plays, on=['game_id', 'play_id'], how='inner')
    
    df_filtered = engineer_features(invert_direction(df_filtered), cfg.dataset, training=False)
    df_output = invert_direction(map_play_direction(df_filtered, df_output))
    
    model_path = os.path.join('Saves/', 'Models/', 'gnn_expanded_features.pth')
    scaler_path = os.path.join('Saves/', 'Scalers/', 'gnn_expanded_features.pkl')

    model = torch.load(model_path)
    scaler = FeatureScaler.load(scaler_path)

    plot_play(df_filtered, df_output, model, scaler, cfg)