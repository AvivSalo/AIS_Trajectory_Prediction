"""
GAT-LSTM: Ship trajectory prediction using Graph Attention Network + LSTM.

Reference:
  Zhao et al. (2023) "A ship trajectory prediction method based on GAT and LSTM"
  Ocean Engineering 289, 116159.
  https://doi.org/10.1016/j.oceaneng.2023.116159

Architecture (Section 3.3):
  1. For each time step t, a multi-head GAT (Eqs. 5-8) processes all agents as
     graph nodes, extracting spatially-aware representations that encode
     inter-agent dependencies.
  2. An LSTM (Section 3.3.3) reads the ego agent's sequence of GAT outputs,
     capturing the temporal evolution of the spatially-enriched features.
  3. A two-layer output head maps the final LSTM hidden state to all future
     positions simultaneously.

UniTraj adaptation notes:
  - Each agent in the batch is a graph node; edges are fully-connected and
    the attention mechanism learns which interactions matter.
  - Agent validity masking (obj_trajs_mask) prevents aggregation from
    padded / absent agents — equivalent to the paper's adjacency matrix A.
    Self-loops are implicit: valid node i always attends to itself.
  - Only the ego agent's LSTM output drives prediction (marginal, ego-centric),
    consistent with the thesis framing and the other baselines.
  - Input features: channels 0-3 of obj_trajs → (x, y, vx, vy).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from unitraj.models.base_model.base_model import BaseModel


# ---------------------------------------------------------------------------
# Graph Attention Layer
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """
    Multi-head graph attention layer (Eqs. 5-8, Zhao et al. 2023).

    Uses the additive-attention decomposition from Velickovic et al. (2018)
    for memory efficiency — mathematically equivalent to the paper's formulation:

        e_ij = LeakyReLU( a^T [W h_i || W h_j] )
             = LeakyReLU( a_L^T W h_i  +  a_R^T W h_j )

        α_ij = softmax_j( e_ij )

        h_i' = ||_{k=1}^{K}  ELU( Σ_j α_ij^k  W^k h_j )   (multi-head, Eq. 8)

    Topology: fully-connected graph; `valid` mask excludes absent/padded nodes.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim  # per head

        # Shared linear transform W (one projection block, split into K heads)
        self.W = nn.Linear(in_dim, num_heads * out_dim, bias=False)

        # Additive attention parameters: split attention vector into left/right halves
        self.a_l = nn.Parameter(torch.empty(1, num_heads, out_dim))
        self.a_r = nn.Parameter(torch.empty(1, num_heads, out_dim))
        nn.init.xavier_uniform_(self.a_l)
        nn.init.xavier_uniform_(self.a_r)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.attn_drop  = nn.Dropout(dropout)
        self.feat_drop  = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        h     : (B, N, in_dim)   node feature matrix
        valid : (B, N)  bool     True for nodes that should be included
        returns (B, N, num_heads * out_dim)
        """
        B, N, _ = h.shape
        h = self.feat_drop(h)

        # Linear projection → (B, N, K, out_dim)
        Wh = self.W(h).view(B, N, self.num_heads, self.out_dim)

        # Attention scores via additive decomposition (memory-efficient)
        # el[b,i,k] = (Wh[b,i,k,:] * a_l[k,:]).sum()
        el = (Wh * self.a_l).sum(-1)   # (B, N, K)
        er = (Wh * self.a_r).sum(-1)   # (B, N, K)

        # e[b,i,j,k] = el[b,i,k] + er[b,j,k]  — broadcast
        e = el.unsqueeze(2) + er.unsqueeze(1)  # (B, N, N, K)
        e = self.leaky_relu(e)

        # Mask invalid source nodes j (exclude from attention)
        if valid is not None:
            invalid_j = ~valid.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
            e = e.masked_fill(invalid_j, float('-inf'))

        # Normalize over neighbours j; nan_to_num handles all-masked rows
        alpha = F.softmax(e, dim=2)                        # (B, N, N, K)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.attn_drop(alpha)

        # Aggregate: out[b,i,k,d] = Σ_j alpha[b,i,j,k] * Wh[b,j,k,d]
        out = torch.einsum('bijk,bjkd->bikd', alpha, Wh)   # (B, N, K, out_dim)

        # Concatenate K heads → (B, N, K * out_dim)
        out = out.reshape(B, N, self.num_heads * self.out_dim)
        return F.elu(out)


# ---------------------------------------------------------------------------
# Core network
# ---------------------------------------------------------------------------

class GATLSTMNet(nn.Module):
    """
    Core GAT-LSTM backbone (Fig. 4, Zhao et al. 2023).

    GAT applied per time step → ego agent sequence → LSTM → output head.
    """

    def __init__(
        self,
        in_features: int,
        gat_hidden: int,
        num_heads: int,
        num_gat_layers: int,
        lstm_hidden: int,
        lstm_layers: int,
        future_len: int,
        dropout: float,
    ):
        super().__init__()
        self.future_len = future_len

        # Stack of GAT layers; each outputs gat_hidden * num_heads dims
        gat = []
        d_in = in_features
        for _ in range(num_gat_layers):
            gat.append(GATLayer(d_in, gat_hidden, num_heads, dropout))
            d_in = gat_hidden * num_heads
        self.gat_layers = nn.ModuleList(gat)
        self.gat_out_dim = d_in

        # LSTM for temporal modelling of the ego agent's GAT feature sequence
        self.lstm = nn.LSTM(
            input_size  = self.gat_out_dim,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )

        # Output head: final LSTM hidden → all future (x, y) at once
        self.output_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, future_len * 2),
        )

    def forward(
        self,
        x: torch.Tensor,        # (B, N, T, in_features)
        valid: torch.Tensor,     # (B, N, T)   bool
        ego_idx: torch.Tensor,   # (B,)         index of ego agent
    ) -> torch.Tensor:
        """Returns predicted future positions: (B, future_len, 2)."""
        B, N, T, _ = x.shape
        D = self.gat_out_dim

        # --- GAT at every time step ------------------------------------- #
        # Flatten (B, T) → single batch dim: (B*T, N, F)
        x_bt     = x.permute(0, 2, 1, 3).reshape(B * T, N, -1)
        valid_bt = valid.permute(0, 2, 1).reshape(B * T, N)

        h = x_bt
        for layer in self.gat_layers:
            h = layer(h, valid_bt)          # (B*T, N, gat_out_dim)

        # Restore time axis: (B*T, N, D) → (B, T, N, D)
        gat_out = h.view(B, T, N, D)

        # --- Extract ego agent sequence --------------------------------- #
        # gather along N dimension for each (B, T)
        ego_exp = ego_idx.view(B, 1, 1, 1).expand(-1, T, 1, D)  # (B, T, 1, D)
        ego_seq = gat_out.gather(2, ego_exp).squeeze(2)           # (B, T, D)

        # --- Temporal LSTM ---------------------------------------------- #
        lstm_out, _ = self.lstm(ego_seq)    # (B, T, lstm_hidden)
        last        = lstm_out[:, -1, :]    # (B, lstm_hidden)

        # --- Predict future positions ----------------------------------- #
        pred = self.output_head(last)               # (B, future_len * 2)
        return pred.view(B, self.future_len, 2)     # (B, future_len, 2)


# ---------------------------------------------------------------------------
# UniTraj wrapper
# ---------------------------------------------------------------------------

class GATLSTM(BaseModel):
    """
    GAT-LSTM wrapped for the UniTraj framework.

    Input batch keys used:
      input_dict['obj_trajs']              (B, max_agents, past_len, 39)
      input_dict['obj_trajs_mask']         (B, max_agents, past_len)  bool
      input_dict['track_index_to_predict'] (B,)
      input_dict['center_gt_trajs']        (B, future_len, 2)
      input_dict['center_gt_trajs_mask']   (B, future_len)

    Feature channels used from the 39-dim obj_trajs vector:
      0, 1 → x, y   (normalised ego-relative position)
      2, 3 → vx, vy (normalised velocity)
    """

    _FEAT = slice(0, 4)   # channels used as GAT node features

    def __init__(self, config):
        super().__init__(config)
        self.past_len   = config.get('past_len',   300)
        self.future_len = config.get('future_len', 300)
        self.num_modes  = config.get('num_modes',  1)

        self.model = GATLSTMNet(
            in_features    = config.get('in_features',    4),
            gat_hidden     = config.get('gat_hidden',     32),
            num_heads      = config.get('num_heads',      4),
            num_gat_layers = config.get('num_gat_layers', 2),
            lstm_hidden    = config.get('lstm_hidden',    128),
            lstm_layers    = config.get('lstm_layers',    2),
            future_len     = self.future_len,
            dropout        = config.get('dropout',        0.3),
        )

    # ---------------------------------------------------------------------- #

    def forward(self, batch):
        inputs    = batch['input_dict']
        B         = batch['batch_size']
        device    = inputs['obj_trajs'].device

        obj_trajs = inputs['obj_trajs']                                  # (B, A, T, 39)
        obj_mask  = inputs['obj_trajs_mask']                             # (B, A, T)
        track_idx = inputs['track_index_to_predict'].long().squeeze(-1)  # (B,)
        gt_trajs  = inputs['center_gt_trajs']                            # (B, future_len, 2)
        gt_mask   = inputs['center_gt_trajs_mask']                       # (B, future_len)

        # Extract 4 node features and zero-out invalid timesteps
        x = obj_trajs[..., self._FEAT]                          # (B, A, T, 4)
        x = x * obj_mask.unsqueeze(-1).float()

        # Run GAT-LSTM
        pred_xy = self.model(x, obj_mask.bool(), track_idx)     # (B, future_len, 2)

        # --- Build UniTraj prediction dict ---
        zeros     = torch.zeros(B, self.future_len, 3, device=device)
        pred_full = torch.cat([pred_xy, zeros], dim=-1)          # (B, future_len, 5)
        pred_traj = pred_full.unsqueeze(1).expand(
            -1, self.num_modes, -1, -1
        ).contiguous()                                            # (B, modes, future_len, 5)

        pred_prob = torch.zeros(B, self.num_modes, device=device)
        pred_prob[:, 0] = 1.0

        prediction = {
            'predicted_probability': pred_prob,
            'predicted_trajectory':  pred_traj,
        }

        # --- Loss (MSE in normalised ego-relative space) ---
        valid = gt_mask.bool()
        if valid.any():
            loss = F.mse_loss(pred_xy[valid], gt_trajs[valid])
        else:
            loss = pred_xy.sum() * 0.0

        return prediction, loss

    # ---------------------------------------------------------------------- #

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.config.get('learning_rate', 1e-3),
            weight_decay=self.config.get('weight_decay', 1e-4),
        )
        lr_sched = self.config.get('learning_rate_sched', None)
        if lr_sched is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=lr_sched, gamma=0.1
            )
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optimizer
