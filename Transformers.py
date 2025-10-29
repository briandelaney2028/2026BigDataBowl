import torch
import torch.nn as nn
import math
from utils import TransformerConfig


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding module for Transformer inputs.

    This produces fixed sinusoidal position encodings as described in
    "Attention is All You Need" and registers them as a buffer.

    Parameters
    ----------
    d_model : int
        Dimensionality of model embeddings (must be even for sine/cos pairs).
    max_len : int, optional
        Maximum sequence length for which to precompute encodings (default 5000).
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Add positional encodings to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Tensor with positional encodings added, same shape as input.
        """
        return x + self.pe[:, :x.size(1)]
    

class Transformer(nn.Module):
    """
    Transformer encoder-decoder for sequence-to-sequence prediction.

    Parameters
    ----------
    input_size : int
        Number of input features per timestep.
    output_size : int
        Number of output features predicted per timestep (e.g., 2 for dx,dy).
    d_model : int, optional
        Dimension of model embeddings (default 128).
    nhead : int, optional
        Number of attention heads (default 8).
    num_encoder_layers : int, optional
        Number of encoder layers (default 3).
    num_decoder_layers : int, optional
        Number of decoder layers (default 3).
    dim_feedforward : int, optional
        Feedforward network dimension inside transformer layers (default 256).
    dropout : float, optional
        Dropout rate (default 0.1).
    max_len : int, optional
        Maximum sequence length for positional encodings (default 500).
    pad_embedding_scale : float, optional
        Scale for the learned pad embedding initialization.

    Notes
    -----
    Input tensors are expected with batch-first ordering: (batch, seq_len, features).
    """

    def __init__(self, input_size: int, output_size: int, 
                 d_model: int, 
                 nhead: int,
                 num_encoder_layers: int, 
                 num_decoder_layers: int,
                 dim_feedforward: int, 
                 dropout: float, 
                 max_len: int,
                 pad_embedding_scale: float
        ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model

        # Input linear projection
        self.input_proj = nn.Linear(input_size, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(d_model)

        # projection for tgt tensors
        self.y_input_proj = nn.Linear(output_size, d_model)
        self.y_input_dropout = nn.Dropout(dropout)
        self.y_input_norm = nn.LayerNorm(d_model)
        
        # learned pad embedding (d_model)
        self.pad_embedding = nn.Parameter(torch.randn(1, 1, d_model, dtype=torch.float32) * pad_embedding_scale)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.pos_decoder = PositionalEncoding(d_model, max_len)
        self.pos_encoder.pe = self.pos_encoder.pe.to(torch.float32)
        self.pos_decoder.pe = self.pos_decoder.pe.to(torch.float32)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, output_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # linear layers
        for linear in [self.input_proj, self.y_input_proj, self.output_proj]:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

        # learned pad embedding
        nn.init.normal_(self.pad_embedding, mean=0.0, std=0.01)

        # transformer layers
        for name, param in self.transformer.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    # Linear layers inside transformer (q, k, v, output, feedforward)
                    nn.init.xavier_uniform_(param)
                else:
                    # Biases
                    nn.init.zeros_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # layerNorm init
        for module in self.transformer.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _replace_padded_embeddings(self, embeddings, src_key_padding_mask):
        """
        Replace embeddings at padded positions with learned pad embedding

        Parameters:
            embeddings (torch.Tensor): Input embeddings (batch, seq_len, d_model)
            src_key_padding_mask (torch.Tensor): Padding mask (batch, seq_len)

        Returns:
            embeddings (torch.Tensor): Embeddings with padded positions replaced
        """
        if src_key_padding_mask is None:
            return embeddings
        pad_vec = self.pad_embedding.to(embeddings.device, dtype=embeddings.dtype)      # (1, 1, d_model)
        mask = src_key_padding_mask.unsqueeze(-1)       # (batch, seq_len, 1)
        # where mask is True, replace with pad_vec
        return torch.where(
            mask, 
            pad_vec.expand_as(embeddings),
            embeddings
        )
    
    def _add_positional(self, emb, which='encoder'):
        """
        Add positional encoding slice for embedding sequence emb
        Parameters
            emb: (batch, T, d_model)
            which: 'encoder' or 'decoder' -> chooses buffer
        """
        if which == 'encoder':
            pe = self.pos_encoder.pe    # (1, max_len, d_model)
        else:
            pe = self.pos_decoder.pe
        return emb + pe[:, :emb.size(1), :].to(emb.device)

    def forward(self, src, tgt_inputs, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Parameters
            src (torch.Tensor): Source sequence (batch, sequence_length, num_features)
            tgt_inputs (torch.Tensor): Target sequence (batch, max_output_len, num_features) -- shifted right by 1
            src_key_padding_mask (torch.Tensor, optional): Source padding mask (batch, sequence_length)
            tgt_key_padding_mask (torch.Tensor, optional): Target padding mask (batch, max_output_len)

        Returns
            outputs (torch.Tensor): Predicted future sequence [batch, future_len, input_size]
        """

        # Fill NaNs with zeros
        if torch.isnan(src).any():
            src = torch.nan_to_num(src, nan=0.0)

        # encoder embedding
        src_emb = self.input_proj(src)
        src_emb = self.input_norm(src_emb)
        src_emb = self.input_dropout(src_emb)

        # replace padded src embeddings with learned PAD vector
        if src_key_padding_mask is not None:
            src_emb = self._replace_padded_embeddings(src_emb, src_key_padding_mask)

        # add positional
        src_emb = self._add_positional(src_emb, which='encoder')

        # decoder embeddings
        # NOTE: DO NOT apply positional encoding here yet
        tgt_emb = self.y_input_proj(tgt_inputs)
        tgt_emb = self.y_input_norm(tgt_emb)
        tgt_emb = self.y_input_dropout(tgt_emb)
        # replace padded tgt embeddings
        if tgt_key_padding_mask is not None:
            tgt_emb = self._replace_padded_embeddings(tgt_emb, tgt_key_padding_mask)
        
        # Create a causal mask to prevent peeking at future target positions
        tgt_mask = self._generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt_emb.device)

        # add positional encodings to decoder now
        tgt_with_pos = self._add_positional(tgt_emb, which='decoder')
        
        # Forward through transformer
        out = self.transformer(src_emb, tgt_with_pos, tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask
                               )

        # predict dx, dy
        deltas = self.output_proj(out)  # (batch, seq_len, 2)
        return deltas

    def predict(self, src, future_len, src_key_padding_mask=None):
        """
        Autoregressive inference loop for prediction

        Parameters:
            src (torch.Tensor): Input sequence (batch, sequence_length, num_features)
            future_len (int): Number of future time steps to predict
            src_key_padding_mask (torch.Tensor, optional): Source padding mask (batch, sequence_length)

        Returns:
            outputs (torch.Tensor): Predicted future sequence [batch, future_len, input_size]
        """
        self.eval()
        with torch.no_grad():
            if torch.isnan(src).any():
                src = torch.nan_to_num(src, nan=0.0)

            # encoder
            src_emb = self.input_proj(src)
            src_emb = self.input_norm(src_emb)
            src_emb = self.input_dropout(src_emb)
            if src_key_padding_mask is not None:
                src_emb = self._replace_padded_embeddings(src_emb, src_key_padding_mask)
            src_emb = self._add_positional(src_emb, which='encoder')
            
            memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

            # initialize decoder input with last observed frame
            last_delta = src[:, -1:, 0:self.output_size].clone()  # (batch, 1, num_features)
            tgt_emb = self.y_input_proj(last_delta)
            tgt_emb = self.y_input_norm(tgt_emb)
            tgt_emb = self.y_input_dropout(tgt_emb)
            
            preds = []
            for _ in range(future_len):
                tgt_with_pos = self._add_positional(tgt_emb, which='decoder')
                tgt_mask = self._generate_square_subsequent_mask(tgt_with_pos.size(1)).to(tgt_with_pos.device)
                out = self.transformer.decoder(tgt_with_pos, memory,
                                               tgt_mask=tgt_mask,
                                               memory_key_padding_mask=src_key_padding_mask)
                next_delta = self.output_proj(out[:, -1:, :])  # (batch, 1, 2)
                preds.append(next_delta)

                # embed predicted dleta and append to tgt_emb
                next_emb = self.y_input_proj(next_delta)
                next_emb = self.y_input_norm(next_emb)
                next_emb = self.y_input_dropout(next_emb)
                tgt_emb = torch.cat([tgt_emb, next_emb], dim=1)

            return torch.cat(preds, dim=1)  # (batch, future_len, 2)

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        """Generates a causal mask for decoder"""
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)

class  FrameGNN(nn.Module):
    """
    Frame-wise graph attention module.

    For each time frame, player nodes attend to other players in the same frame
    using a multi-head attention mechanism. This module projects input features
    to queries/keys/values and returns per-player embeddings.

    Parameters
    ----------
    in_feats : int
        Number of input features per node.
    d_model : int
        Output embedding dimension.
    n_heads : int, optional
        Number of attention heads (default 4).
    attn_dropout : float, optional
        Dropout applied to attention weights.
    residual : bool, optional
        Whether to use a residual connection (default True).

    Input
    -----
    x : torch.Tensor
        Tensor of shape (B, N, T, F).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, N, T, d_model).
    """
    def __init__(self, in_feats, d_model, n_heads=4, attn_dropout=0.0, residual=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.in_feats = in_feats
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.residual = residual

        # projections from node features to multihead attention
        self.q_lin = nn.Linear(in_feats, d_model)
        self.k_lin = nn.Linear(in_feats, d_model)
        self.v_lin = nn.Linear(in_feats, d_model)

        # output projection back to d_model
        self.out_lin = nn.Linear(d_model, d_model)

        # if in_feats != d_model, residual projection
        self.residual_proj = nn.Linear(in_feats, d_model) if in_feats != d_model else None
        # LayerNorm after residual
        self.norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for lin in [self.q_lin, self.k_lin, self.v_lin, self.out_lin]:
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        if self.residual_proj is not None:
            nn.init.xavier_uniform_(self.residual_proj.weight)
            if self.residual_proj.bias is not None:
                nn.init.zeros_(self.residual_proj.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, x, player_mask=None):
        """
        Forward pass for frame-wise attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (B, N, T, F).
        player_mask : torch.Tensor or None
            Boolean mask (B, N) marking valid (True) players; used to mask
            attention to padded players.

        Returns
        -------
        torch.Tensor
            Output embeddings with shape (B, N, T, d_model).
        """
        B, N, T, F = x.shape

        # collapse batch and time so attnetion can be computed per-frame in batch: (B*T, N, F)
        xt = x.permute(0, 2, 1, 3).reshape(B*T, N, F)   # (B*T, N, F)
        if torch.isnan(xt).any():
            xt = torch.nan_to_num(xt, nan=0.0)

        Q = self.q_lin(xt).view(B*T, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_lin(xt).view(B*T, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_lin(xt).view(B*T, N, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention per head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (BT, heads, N, N)

        # apply node_make if provided to prevent attention to paddings
        if player_mask is not None:
            mask_expanded = player_mask.unsqueeze(1).repeat(1, T, 1)    # (B, N) -> (B, T, N)
            mask_bt = mask_expanded.reshape(B*T, 1, 1, N)
            mask_bt_q = mask_bt.transpose(-1, -2)  
            scores = scores.masked_fill(~mask_bt  , float('-inf'))      # mask invalid keys
            scores = scores.masked_fill(~mask_bt_q, float('-inf'))      # mask invalid queries

        # clean scores obtained from padded players
        all_inf = torch.all(scores == float('-inf'), dim=-1, keepdim=True)
        scores = torch.where(all_inf, torch.zeros_like(scores), scores)

        # compute safe attention
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # weighted sum of values
        out = torch.matmul(attn, V)     # (BT, heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B*T, N, self.d_model)
        out = self.out_lin(out)     # (BT, N, d_model)

        # residual connection: project input to d_model if desired
        res = self.residual_proj(xt) if self.residual_proj is not None else xt
        if torch.isnan(res).any():
            res = torch.nan_to_num(res, nan=0.0)
        out = self.norm(out + res)

        # reshape back to (B, N, T, d_model)
        out = out.view(B, T, N, self.d_model).permute(0, 2, 1, 3)
        return out 

class GNNTransformer(nn.Module):
    """
    Hybrid model combining a frame-level GNN with a per-player Transformer.

    Workflow
    --------
    - FrameGNN embeds player-player relations for each frame producing per-player
      embeddings across time.
    - A Transformer models temporal dynamics of each player's embedding and
      autoregressively predicts future displacements.

    Parameters
    ----------
    in_feats : int
        Number of input features per player.
    output_size : int
        Dimension of model output per timestep (e.g. 2 for dx,dy).
    d_model : int, optional
        Hidden embedding dimension passed to the Transformer (default 128).
    """
    def __init__(self, in_feats: int, output_size: int,
                 cfg: TransformerConfig):
        """
        in_feats: number of input features per player
        output_size: dimension of model output per timestep (e.g., 2 for dx,dy)
        d_model: hidden dim
        """
        super().__init__()
        self.device = cfg.device
        self.in_feats = in_feats
        self.output_size = output_size
        self.d_model = cfg.d_model

        # Frame GNN
        self.frame_gnn = FrameGNN(
            in_feats=in_feats, 
            d_model=cfg.d_model,
            n_heads=cfg.gnn_nhead, 
            attn_dropout=cfg.dropout
        )

        # instantiate the base Transformer class
        self.transformer = Transformer(
            input_size=cfg.d_model,
            output_size=output_size,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
            pad_embedding_scale=cfg.pad_embedding_scale
        )

    @staticmethod
    def _select_targets_flatten(gnn_out, target_mask):
        """
        Select embeddings for targeted players and flatten them into a single batch.

        Parameters
        ----------
        gnn_out : torch.Tensor
            Tensor of shape (B, N, T, d_model) containing per-player embeddings.
        target_mask : torch.Tensor
            Boolean mask of shape (B, N) indicating which players are targets.

        Returns
        -------
        tuple
            (flat_src, index_info) where flat_src has shape (S, T, d_model) and
            index_info is a list of index tensors used to scatter results back.
        """
        B, N, T, d = gnn_out.shape
        index_info, blocks = [], []

        for b in range(B):
            idxs = torch.nonzero(target_mask[b].to(torch.bool), as_tuple=False).squeeze(1)
            index_info.append(idxs)
            if idxs.numel() > 0:
                blocks.append(gnn_out[b, idxs, :, :])   # (k, T, d)
        
        if len(blocks) == 0:
            # no targes in batch
            return torch.zeros((0, T, d), device=gnn_out.device, dtype=gnn_out.dtype), index_info
        else:
            return torch.cat(blocks, dim=0), index_info     # (S, T, d)
    
    @staticmethod
    def _select_y_inputs_flatten(y_inputs_all, target_mask):
        """
        Flatten y inputs for targeted players.

        Parameters
        ----------
        y_inputs_all : torch.Tensor or None
            Tensor of shape (B, N, T_out, output_size) containing decoder inputs.
        target_mask : torch.Tensor
            Boolean mask (B, N) indicating which players are targeted.

        Returns
        -------
        torch.Tensor or None
            Concatenated y inputs for all targeted players with shape (S, T_out, output_size)
            or an empty tensor with zero rows if none are targeted.
        """
        if y_inputs_all is None:
            return None
        
        B, N, T_out, outdim = y_inputs_all.shape
        blocks = []
        for b in range(B):
            idxs = torch.nonzero(target_mask[b], as_tuple=False).squeeze(1)
            if idxs.numel() > 0:
                blocks.append(y_inputs_all[b, idxs, :, :])
        
        if len(blocks) == 0:
            return torch.zeros((0, T_out, outdim), device=y_inputs_all.device, dtype=y_inputs_all.dtype)
        return torch.cat(blocks, dim=0) # (S, T_out, output_size)
    
    def _select_y_mask_flatten(self, y_mask, target_mask):
        """
        Flatten y_mask (B, N, T_out) -> (S, T_out) for targeted players
        Parameters:
            y_mask: (B, N, T_out)
            target_mask: (B, N)
        """
        B, N, T_out = y_mask.shape
        blocks = []
        for b in range(B):
            idxs = torch.nonzero(target_mask[b], as_tuple=False).squeeze(1)
            if idxs.numel() > 0:
                blocks.append(y_mask[b, idxs, :])
        if len(blocks) == 0:
            return torch.zeros((0, T_out), dtype=torch.bool, device=y_mask.device)
        return torch.cat(blocks, dim=0)
    
    @staticmethod
    def _ensure_bool(mask):
        """
        Ensure that the provided mask is a boolean tensor.

        Parameters
        ----------
        mask : torch.Tensor or None
            Input mask to coerce.

        Returns
        -------
        torch.Tensor or None
            Boolean mask or None if input was None.
        """
        if mask is None:
            return None
        return mask.bool() if mask.dtype != torch.bool else mask

    def forward(self, src, tgt_inputs=None, target_mask=None, player_mask=None, y_mask=None):
        """
        src: (B, N, T_in, F_in)
        tgt_inputs: (B, N, T_out, output_size) or None (for teacher forcing)
        target_mask: (B, N)  mask for which players will get x,y training data
        player_mask: (B, N) True for valid player, False for padded player
        y_mask: (B, N, T_out) True valid y timesteps

        Returns:
            outputs: (B, N_out, T_out, output_size)
        """
        B, N, T_in, F = src.shape
        device = src.device

        # construct src validity mask based on real players and non-padded inputs frames
        # check for NaNs (i.e., padded input frames)
        src_valid_mask = ~torch.isnan(src).all(dim=-1)  # (B, N, T_in)
        # combine with player_mask to invalidate padded players
        if player_mask is not None:
            pm_expand = player_mask.unsqueeze(-1).expand(-1, -1, T_in)  # (B, N, T_in)
            src_valid_mask = src_valid_mask & pm_expand

        # Frame GNN over ALL players with masking
        gnn_out = self.frame_gnn(src, player_mask=player_mask)

        # select only target player embeddings and flatten into shape (S, T_in, d_model)
        if target_mask is None:
            # predict for all players
            target_mask = torch.ones((B, N), dtype=torch.bool, device=device)
        flat_src, index_info = self._select_targets_flatten(gnn_out, target_mask)
        if flat_src.size(0) == 0:   # nothing to predict
            T_out = tgt_inputs.shape[2] if tgt_inputs is not None else 0
            return torch.zeros((B, N, T_out, self.output_size), device=device)

        # build src_key_padding_mask for flattened rows
        masks = []
        for b in range(B):
            idxs = index_info[b]
            k = int(idxs.numel())
            if k == 0:
                continue
            # gather valid flags for these players: (k, T_in)
            valid_rows = src_valid_mask[b, idxs, :].to(device)
            pad_rows = ~valid_rows  # true where padded
            masks.append(pad_rows)
        # shape (S, T_in)
        src_key_padding_mask_flat = torch.cat(masks, dim=0) if len(masks) > 0 else None
        if src_key_padding_mask_flat is not None:
            src_key_padding_mask_flat = self._ensure_bool(src_key_padding_mask_flat)
        
        # flatten tgt_inputs and y_mask for targeted players
        flat_y_inputs = self._select_y_inputs_flatten(tgt_inputs, target_mask) if tgt_inputs is not None else None
        flat_y_mask   = self._select_y_mask_flatten(y_mask, target_mask) if y_mask is not None else None
        tgt_key_padding_mask = (~flat_y_mask) if flat_y_mask is not None else None

        # run through transformer   shape (S, T_out, output_size)
        preds_flat = self.transformer(flat_src, flat_y_inputs,
                                      src_key_padding_mask=src_key_padding_mask_flat,
                                      tgt_key_padding_mask=tgt_key_padding_mask)
        
        # scatter preds_flat back into (B, N, T_tgt, out)
        S, T_out, outdim = preds_flat.shape
        preds_all = torch.zeros((B, N, T_out, outdim), device=device, dtype=preds_flat.dtype)

        ptr = 0
        for b in range(B):
            idxs = index_info[b]
            k = int(idxs.numel())
            if k > 0:
                preds_all[b, idxs, :, :] = preds_flat[ptr:ptr + k]
                ptr += k

        return preds_all
    
    def predict(self, src, future_len, target_mask=None, player_mask=None):
        """
        Autoregressive inference
        Parameters:
            src (B, N, T_in, F)
            future_len: T_tgt
            target_mask: (B, N)
            player_mask: (B, N) True for valid players
        
        Returns:
            preds_all: (B, N, future_len, output_size) with zero paddings
        """

        B, N, T_in, F = src.shape
        device = src.device

        # construct src validity mask
        src_valid_mask = ~torch.isnan(src).all(dim=-1)  # (B, N, T_in)
        if player_mask is not None:
            pm_expand = player_mask.unsqueeze(-1).expand(-1, -1, T_in)
            src_valid_mask = src_valid_mask & pm_expand

        # Frame GNN over ALL players
        gnn_out = self.frame_gnn(src, player_mask=player_mask)

        # flatten targets
        if target_mask is None:
            target_mask = torch.ones((B, N), dtype=torch.bool, device=device)
        flat_src, index_info = self._select_targets_flatten(gnn_out, target_mask)

        if flat_src.size(0) == 0:
            return torch.zeros((B, N, future_len, self.output_size), device=device)
        
        # build src_key_padding_mask (S, T_in)
        masks = []
        for b in range(B):
            idxs = index_info[b]
            k = int(idxs.numel())
            if k == 0:
                continue
            valid_rows = src_valid_mask[b, idxs, :].to(device)
            pad_rows = ~valid_rows
            masks.append(pad_rows)
        src_key_padding_mask_flat = torch.cat(masks, dim=0) if len(masks) > 0 else None
        if src_key_padding_mask_flat is not None:
            src_key_padding_mask_flat = self._ensure_bool(src_key_padding_mask_flat)
        
        preds_flat = self.transformer.predict(
            flat_src,
            future_len,
            src_key_padding_mask=src_key_padding_mask_flat,
        )

        # scatter back to (B, N, future_len, output_size)
        S, T_out, outdim = preds_flat.shape
        preds_all = torch.zeros((B, N, T_out, outdim), device=device, dtype=preds_flat.dtype)

        ptr = 0
        for b in range(B):
            idxs = index_info[b]
            k = int(idxs.numel())
            if k > 0:
                preds_all[b, idxs, :, :] = preds_flat[ptr:ptr + k]
                ptr += k
        
        return preds_all