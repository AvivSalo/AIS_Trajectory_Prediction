"""
AIS-ACNet: Vessel trajectory prediction via dual-encoder dilated causal CNN
with auxiliary tasks and feature fusion.

Ported from https://github.com/yuyolshin/AIS-ACNet into the UniTraj framework.

Reference:
  Shin et al. (2024) "Deep learning framework for vessel trajectory prediction
  using auxiliary tasks and convolutional networks."
  Engineering Applications of Artificial Intelligence 132, 107936.
  https://doi.org/10.1016/j.engappai.2024.107936

Architecture changes from the original:
  - Dilation factors are configurable (default: exponential) to cover the
    5-minute / 300-step input sequences used in this project instead of the
    original 15-step sequences.
  - Processes a single ego vessel (N=1) rather than a crowd of vessels.
  - Works in UniTraj's normalized ego-relative coordinate space; no haversine
    de-normalization is needed.
  - The auxiliary tasks predict future Δx / Δy (normalised-position finite
    differences) instead of raw SOG / COG, because the UniTraj GT only
    contains future positions, not future speed/heading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from unitraj.models.base_model.base_model import BaseModel


# ---------------------------------------------------------------------------
# Core GWNet model — faithful port of gwnet from the original repo
# ---------------------------------------------------------------------------

class GWNet(nn.Module):
    """
    Dual-encoder dilated causal CNN backbone (GWNet) from AIS-ACNet.

    Input  : (B, 4, N, T)
               channels 0-1 → position features  → main-net
               channels 2-3 → velocity features  → aux-net
    Output : (x_pos, x_s, x_h)
               x_pos : (B, future_len, N, 2)  — predicted Δx and Δy positions
               x_s   : (B, future_len, N, 1)  — predicted aux channel 1 (vx)
               x_h   : (B, future_len, N, 1)  — predicted aux channel 2 (vy)
    """

    def __init__(
        self,
        future_len: int,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        dilation_factors=None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if dilation_factors is None:
            # Exponential dilations: receptive field = 1 + sum(d) = 1 + 1023 = 1024,
            # covering any input sequence up to 1023 steps (well beyond our T=300).
            dilation_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        self.dropout = dropout
        self.dilation_factors = dilation_factors
        self.num_layers = len(dilation_factors)

        # ------------------------------------------------------------------ #
        # Main-net (position encoder)
        # ------------------------------------------------------------------ #
        self.start_conv = nn.Conv2d(2, residual_channels, (1, 1))

        self.filter_convs   = nn.ModuleList()
        self.gate_convs     = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs     = nn.ModuleList()
        self.fusion_convs   = nn.ModuleList()
        self.fusion_out1    = nn.ModuleList()
        self.bn             = nn.ModuleList()

        # ------------------------------------------------------------------ #
        # Aux-net (velocity encoder)
        # ------------------------------------------------------------------ #
        self.start_conv_a = nn.Conv2d(2, residual_channels, (1, 1))

        self.filter_convs_a   = nn.ModuleList()
        self.gate_convs_a     = nn.ModuleList()
        self.residual_convs_a = nn.ModuleList()
        self.skip_convs_a     = nn.ModuleList()
        self.fusion_convs_a   = nn.ModuleList()
        self.bn_a             = nn.ModuleList()

        for d in dilation_factors:
            # Main-net per-layer modules
            self.filter_convs.append(
                nn.Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=d))
            self.gate_convs.append(
                nn.Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=d))
            self.residual_convs.append(
                nn.Conv2d(dilation_channels, residual_channels, (1, 1)))
            self.fusion_convs.append(
                nn.Conv2d(residual_channels, residual_channels, (1, 1)))
            self.fusion_out1.append(
                nn.Conv2d(residual_channels, residual_channels, (1, 1)))
            self.skip_convs.append(
                nn.Conv2d(dilation_channels, skip_channels, (1, 1)))
            self.bn.append(nn.BatchNorm2d(residual_channels))

            # Aux-net per-layer modules
            self.filter_convs_a.append(
                nn.Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=d))
            self.gate_convs_a.append(
                nn.Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=d))
            self.residual_convs_a.append(
                nn.Conv2d(dilation_channels, residual_channels, (1, 1)))
            self.fusion_convs_a.append(
                nn.Conv2d(residual_channels, residual_channels, (1, 1)))
            self.skip_convs_a.append(
                nn.Conv2d(dilation_channels, skip_channels, (1, 1)))
            self.bn_a.append(nn.BatchNorm2d(residual_channels))

        # ------------------------------------------------------------------ #
        # Output heads
        # ------------------------------------------------------------------ #
        # Main head: predict future_len values for each of x and y
        self.end_conv_1  = nn.Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2x = nn.Conv2d(end_channels,  future_len,   (1, 1), bias=True)
        self.end_conv_2y = nn.Conv2d(end_channels,  future_len,   (1, 1), bias=True)

        # Aux head: predict future_len values for each of vx and vy
        self.end_conv_a_1 = nn.Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_a_s = nn.Conv2d(end_channels,  future_len,   (1, 1), bias=True)
        self.end_conv_a_h = nn.Conv2d(end_channels,  future_len,   (1, 1), bias=True)

        # Receptive field: the model will be left-padded so the final temporal
        # dimension collapses to exactly 1 step before the output heads.
        # RF = 1 + sum(d * (kernel_size - 1)) = 1 + sum(dilations) for k=2.
        self.receptive_field = 1 + sum(d * (kernel_size - 1) for d in dilation_factors)

    # ---------------------------------------------------------------------- #

    def forward(self, inp: torch.Tensor):
        """
        inp : (B, 4, N, T)
        """
        T = inp.size(3)
        if T < self.receptive_field:
            inp = F.pad(inp, (self.receptive_field - T, 0, 0, 0))

        x   = self.start_conv(inp[:, :2])    # (B, residual_channels, N, T')
        x_a = self.start_conv_a(inp[:, 2:])

        skip: torch.Tensor | None = None
        skip_a: torch.Tensor | None = None

        for i in range(self.num_layers):

            # ---- Main-net WaveNet layer (gated activation) ----
            residual = x
            filter_  = torch.tanh(self.filter_convs[i](residual))
            gate     = torch.sigmoid(self.gate_convs[i](residual))
            x = filter_ * gate                                          # gated output

            # ---- Aux-net WaveNet layer ----
            residual_a = x_a
            filter_a   = torch.tanh(self.filter_convs_a[i](residual_a))
            gate_a     = torch.sigmoid(self.gate_convs_a[i](residual_a))
            x_a = filter_a * gate_a

            # ---- Skip accumulation (trim to current s size, then sum) ----
            s   = self.skip_convs[i](x)
            s_a = self.skip_convs_a[i](x_a)
            if skip is None:
                skip   = s
                skip_a = s_a
            else:
                skip   = s   + skip[:, :, :, -s.size(3):]
                skip_a = s_a + skip_a[:, :, :, -s_a.size(3):]

            # ---- Residual projection ----
            x   = self.residual_convs[i](x)
            x_a = self.residual_convs_a[i](x_a)

            # ---- Feature fusion: auxiliary information → main-net ----
            x_fuse   = self.fusion_convs[i](x)
            x_a_fuse = self.fusion_convs_a[i](x_a)
            z = torch.sigmoid(x_fuse + x_a_fuse)
            x = self.fusion_out1[i](z * x_fuse + (1 - z) * x_a_fuse)

            # ---- Add residual shortcuts ----
            x   = x   + residual[:, :, :, -x.size(3):]
            x_a = x_a + residual_a[:, :, :, -x_a.size(3):]

            x   = self.bn[i](x)
            x_a = self.bn_a[i](x_a)

        # ---- Main output head ----
        # skip: (B, skip_channels, N, 1)  after full temporal collapse
        out   = F.relu(self.end_conv_1(F.relu(skip)))       # (B, end_channels, N, 1)
        x_lat = self.end_conv_2x(out)                        # (B, future_len,   N, 1)
        x_lon = self.end_conv_2y(out)                        # (B, future_len,   N, 1)
        x_pos = torch.cat([x_lat, x_lon], dim=-1)            # (B, future_len,   N, 2)

        # ---- Aux output head ----
        out_a = F.relu(self.end_conv_a_1(F.relu(skip_a)))
        x_s   = self.end_conv_a_s(out_a)                    # (B, future_len,   N, 1)
        x_h   = self.end_conv_a_h(out_a)                    # (B, future_len,   N, 1)

        return x_pos, x_s, x_h


# ---------------------------------------------------------------------------
# UniTraj wrapper
# ---------------------------------------------------------------------------

class AISACNet(BaseModel):
    """
    AIS-ACNet wrapped for the UniTraj framework.

    Input batch keys used:
      input_dict['obj_trajs']              (B, max_agents, past_len, 39)
      input_dict['obj_trajs_mask']         (B, max_agents, past_len)
      input_dict['track_index_to_predict'] (B,)
      input_dict['center_gt_trajs']        (B, future_len, 2)
      input_dict['center_gt_trajs_mask']   (B, future_len)

    Feature mapping from the 39-channel obj_trajs:
      channels 0-1 → x, y  (normalised ego-relative position)  → main-net
      channels 2-3 → vx, vy (normalised velocity)               → aux-net
    """

    # Feature slices in the 39-dim per-timestep feature vector
    _POS = slice(0, 2)   # x, y
    _VEL = slice(2, 4)   # vx, vy

    def __init__(self, config):
        super().__init__(config)
        self.past_len   = config.get('past_len',   300)
        self.future_len = config.get('future_len', 300)
        self.num_modes  = config.get('num_modes',  1)

        nhid        = config.get('nhid',        32)
        skip_mult   = config.get('skip_channels_multiplier',  8)
        end_mult    = config.get('end_channels_multiplier',  16)
        kernel_size = config.get('kernel_size', 2)
        dilations   = config.get('dilation_factors',
                                 [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
        dropout     = config.get('dropout', 0.3)

        self.alpha = config.get('aux_sog_weight', 0.2)   # weight for vx aux loss
        self.beta  = config.get('aux_cog_weight', 0.2)   # weight for vy aux loss

        self.model = GWNet(
            future_len        = self.future_len,
            residual_channels = nhid,
            dilation_channels = nhid,
            skip_channels     = nhid * skip_mult,
            end_channels      = nhid * end_mult,
            kernel_size       = kernel_size,
            dilation_factors  = dilations,
            dropout           = dropout,
        )

    # ---------------------------------------------------------------------- #

    def _extract_ego(self, batch):
        """Extract and mask ego-vessel features from the batch."""
        inputs       = batch['input_dict']
        obj_trajs    = inputs['obj_trajs']           # (B, max_agents, T, 39)
        obj_mask     = inputs['obj_trajs_mask']      # (B, max_agents, T)
        track_idx    = inputs['track_index_to_predict'].long().squeeze(-1)  # (B,)

        B      = obj_trajs.shape[0]
        device = obj_trajs.device

        b_idx = torch.arange(B, device=device)
        ego   = obj_trajs[b_idx, track_idx]          # (B, T, 39)
        mask  = obj_mask[b_idx, track_idx].float()   # (B, T)

        pos = ego[:, :, self._POS] * mask.unsqueeze(-1)   # (B, T, 2)
        vel = ego[:, :, self._VEL] * mask.unsqueeze(-1)   # (B, T, 2)

        # GWNet expects (B, 4, N, T) — N=1 (single ego vessel)
        inp = torch.cat([pos, vel], dim=-1)               # (B, T, 4)
        inp = inp.permute(0, 2, 1).unsqueeze(2)           # (B, 4, 1, T)
        return inp

    # ---------------------------------------------------------------------- #

    def forward(self, batch):
        inputs = batch['input_dict']
        B      = batch['batch_size']
        device = inputs['obj_trajs'].device

        # ---- Encode ----
        inp               = self._extract_ego(batch)        # (B, 4, 1, T)
        x_pos, x_s, x_h  = self.model(inp)
        # x_pos : (B, future_len, 1, 2)
        # x_s   : (B, future_len, 1, 1)
        # x_h   : (B, future_len, 1, 1)

        pred_xy = x_pos.squeeze(2)                          # (B, future_len, 2)

        # ---- Build UniTraj prediction dict ----
        zeros = torch.zeros(B, self.future_len, 3, device=device)
        pred_full = torch.cat([pred_xy, zeros], dim=-1)     # (B, future_len, 5)

        predicted_trajectory = pred_full.unsqueeze(1).expand(
            -1, self.num_modes, -1, -1
        ).contiguous()                                       # (B, num_modes, future_len, 5)

        predicted_probability = torch.zeros(B, self.num_modes, device=device)
        predicted_probability[:, 0] = 1.0

        prediction = {
            'predicted_probability': predicted_probability,
            'predicted_trajectory':  predicted_trajectory,
        }

        # ---- Loss ----
        gt_trajs = inputs['center_gt_trajs']        # (B, future_len, 2)
        gt_mask  = inputs['center_gt_trajs_mask']   # (B, future_len)
        valid    = gt_mask.bool()

        # Main position loss (MSE in normalised ego-relative space)
        if valid.any():
            loss_d = F.mse_loss(pred_xy[valid], gt_trajs[valid])
        else:
            loss_d = pred_xy.sum() * 0.0

        # Auxiliary losses on normalised velocity (Δpos per timestep)
        # Targets derived from finite differences of GT positions.
        # Only valid for t ≥ 1 (first diff is undefined).
        loss = loss_d
        if (self.alpha > 0.0 or self.beta > 0.0) and valid.any():
            pred_vx = x_s[:, :, 0, 0]   # (B, future_len)
            pred_vy = x_h[:, :, 0, 0]   # (B, future_len)

            # GT velocity: forward difference in normalised position space
            gt_dx = torch.zeros_like(gt_trajs[:, :, 0])   # (B, future_len)
            gt_dy = torch.zeros_like(gt_trajs[:, :, 1])
            gt_dx[:, 1:] = gt_trajs[:, 1:, 0] - gt_trajs[:, :-1, 0]
            gt_dy[:, 1:] = gt_trajs[:, 1:, 1] - gt_trajs[:, :-1, 1]

            # Auxiliary mask: require both t and t-1 to be valid
            valid_aux = valid.clone()
            valid_aux[:, 0] = False

            if valid_aux.any():
                if self.alpha > 0.0:
                    loss = loss + self.alpha * F.mse_loss(
                        pred_vx[valid_aux], gt_dx[valid_aux]
                    )
                if self.beta > 0.0:
                    loss = loss + self.beta * F.mse_loss(
                        pred_vy[valid_aux], gt_dy[valid_aux]
                    )

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