"""
TrAISformer baseline wrapped as a UniTraj BaseModel.

==============================================================================
DATA UNDERSTANDING (what the loader gives us)
==============================================================================

The UniTraj AIS loader (``unitraj/datasets/ais_dataset.py``) produces, per batch:

    batch['input_dict']:
      obj_trajs           (B, max_agents, past_len, F)
            F = 8 useful channels + padding to 39 for Wayformer compatibility.
            Channels:
              0: x         (ego-relative, normalized by position_scale=100m)
              1: y         (ego-relative, normalized by position_scale=100m)
              2: vx        (normalized by velocity_scale=20 m/s)
              3: vy        (normalized by velocity_scale=20 m/s)
              4: sin(heading_rel_to_ego_initial)
              5: cos(heading_rel_to_ego_initial)
              6: speed     (normalized by velocity_scale=20 m/s)
              7..: zero-padding
      obj_trajs_mask      (B, max_agents, past_len)  bool validity
      center_gt_trajs     (B, future_len, 2)         ego future (x, y) normalized
      center_gt_trajs_mask(B, future_len)
      center_gt_final_valid_idx (B,)
      ...

Frame is **ego-relative** (rotated so ego's initial heading aligns to +y) and
**normalized** (positions / 100m, velocities / 20 m/s).

Empirical ranges from EC2 training data (1000 scenarios via
``scripts/inspect_ais_ranges.py``):

    x (lateral)  p99.9: ±26 hm   ≈ ±2.6 km
    y (forward)  p99.9: ±25 hm   ≈ ±2.5 km
    speed         max : 0.54     ≈ 21 knots
    cog (rel.)        : full ±π circle, std ~0.61 rad (~35°)

==============================================================================
PATH B ADAPTATION (5-min / 1 Hz instead of the paper's 10-min / 3h-15h)
==============================================================================

  - **Single-agent only**: we feed `obj_trajs[:, 0]` (ego). Other agents are
    ignored because TrAISformer is marginal by design. This matches Aviv's
    thesis ego-centric stance.
  - **Coordinate frame**: ego-relative x, y, NOT geographic lat/lon. cog
    here is the heading relative to ego's initial heading, NOT absolute COG.
    This is consistent with how the loader encodes everything else (rotated
    frame) and avoids a coordinate-mismatch between past (from loader, has
    sin/cos heading channels) and future (we derive cog from finite
    differences on `gt_trajs`).
  - **Token length**: past_len + future_len = 600 (vs. 108 in the paper).
    Attention is O(N²), so a 600-token context is ~31× more attention work
    per layer than the paper. The transformer hyperparams in
    ``traisformer.yaml`` are sized accordingly.
  - **Multi-modal output**: K independent autoregressive rollouts with
    temperature sampling. K is configurable; default K=6 matches Wayformer
    for an apples-to-apples ``minADE6 / minFDE6`` comparison. K=1 selects
    greedy single-mode for ablations.
  - **Loss**: 4 per-channel cross-entropies, masked so only valid FUTURE
    positions contribute (past positions are context, not targets).

==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import optim

from unitraj.models.base_model.base_model import BaseModel
from unitraj.models.traisformer.tokenizer import FourHotTokenizer
from unitraj.models.traisformer.traisformer_net import TrAISformerNet


# Names of the config keys the wrapper needs to forward into the net + tokenizer.
_FOURHOT_KEYS = (
    "x_min", "x_max", "y_min", "y_max",
    "sog_max", "cog_min", "cog_max",
    "x_size", "y_size", "sog_size", "cog_size",
)
_NET_KEYS = ("n_embd", "n_head", "n_layer", "dropout")


class TrAISformer(BaseModel):
    """TrAISformer wrapped as a UniTraj BaseModel."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Horizons mirror the parent (config_5min_future*.yaml).
        self.past_len = int(config["past_len"])
        self.future_len = int(config["future_len"])

        # Multi-modal output knobs.
        self.num_modes = int(config.get("num_modes", 6))
        self.sample_temperature = float(config.get("sample_temperature", 1.0))

        # Build a tiny attribute namespace for the tokenizer + net.
        cfg = SimpleNamespace()
        for k in _FOURHOT_KEYS + _NET_KEYS:
            setattr(cfg, k, config[k])
        cfg.max_seqlen = self.past_len + self.future_len

        self.tokenizer = FourHotTokenizer(cfg)
        self.net = TrAISformerNet(cfg)

    # ==================================================================
    # Channel extraction: UniTraj features → (x, y, sog, cog)
    # ==================================================================

    def _past_channels(self, obj_trajs: torch.Tensor) -> torch.Tensor:
        """Pull (x, y, speed, cog) for the ego past trajectory.

        ``cog`` is computed from the loader's sin/cos heading channels (4, 5),
        which already encode the ego-relative heading (see loader docs above).
        """
        ego = obj_trajs[:, 0]  # (B, past_len, F)
        x = ego[..., 0]
        y = ego[..., 1]
        sin_h = ego[..., 4]
        cos_h = ego[..., 5]
        speed = ego[..., 6]
        cog = torch.atan2(sin_h, cos_h)  # ego-relative heading, radians
        return torch.stack([x, y, speed, cog], dim=-1)

    def _future_channels(self, gt_trajs: torch.Tensor) -> torch.Tensor:
        """Derive (x, y, sog, cog) for the future from gt (x, y) only.

        The loader only exposes future positions — not future sog/cog — so we
        compute them by finite differences.

        Unit chain (so the derived sog matches the loader's normalized sog):
            position_scale = 100 m, velocity_scale = 20 m/s, dt = 1 s.
            dxy_m = dxy_norm * 100           (real-world meters per step)
            speed_m_per_s = dxy_m / 1        (because dt=1s)
            speed_norm = speed_m_per_s / 20 = dxy_norm * 5
        """
        x = gt_trajs[..., 0]
        y = gt_trajs[..., 1]

        dx = torch.zeros_like(x)
        dy = torch.zeros_like(y)
        dx[:, 1:] = x[:, 1:] - x[:, :-1]
        dy[:, 1:] = y[:, 1:] - y[:, :-1]
        # First step: replicate second step so we don't get a spurious 0.
        if x.size(1) > 1:
            dx[:, 0] = dx[:, 1]
            dy[:, 0] = dy[:, 1]

        speed_norm = torch.sqrt(dx * dx + dy * dy) * 5.0    # to UniTraj-normalized m/s
        cog = torch.atan2(dx, dy)                            # match loader's atan2(vx, vy) convention
        return torch.stack([x, y, speed_norm, cog], dim=-1)

    # ==================================================================
    # Forward (training + validation)
    # ==================================================================

    def forward(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        inputs = batch["input_dict"]
        obj_trajs = inputs["obj_trajs"]                                   # (B, A, T_past, F)
        center_gt = inputs["center_gt_trajs"]                             # (B, T_fut, 2)
        center_gt_mask = inputs["center_gt_trajs_mask"]                   # (B, T_fut)

        B = obj_trajs.shape[0]
        device = obj_trajs.device

        past_ch = self._past_channels(obj_trajs)                          # (B, T_past, 4)
        future_ch = self._future_channels(center_gt)                      # (B, T_fut, 4)

        seq = torch.cat([past_ch, future_ch], dim=1)                      # (B, T, 4)
        tokens = self.tokenizer.encode(seq)                               # (B, T, 4)

        # Standard next-token prediction: feed positions [0..T-2], target [1..T-1].
        in_tokens = tokens[:, :-1]
        tgt_tokens = tokens[:, 1:]

        # Loss mask: only count positions whose target lies in the FUTURE.
        # Target index i corresponds to original sequence position i+1, so the
        # first future-targeting index is past_len - 1.
        T_minus_1 = tgt_tokens.size(1)
        valid = torch.zeros(B, T_minus_1, dtype=torch.bool, device=device)
        valid[:, self.past_len - 1:] = center_gt_mask.bool()[:, : T_minus_1 - (self.past_len - 1)]

        logits = self.net(in_tokens)
        loss = TrAISformerNet.loss(logits, tgt_tokens, valid)

        # Prediction dict — shape required by BaseModel.log_info:
        #   predicted_probability: (B, num_modes)
        #   predicted_trajectory:  (B, num_modes, future_len, 5)  [x, y, log_std_x, log_std_y, rho]
        if self.training:
            # Cheap: argmax of teacher-forced logits at future positions.
            pred_xy = self._teacher_forced_xy(logits)                     # (B, T_fut, 2)
            xy = pred_xy.unsqueeze(1).expand(B, self.num_modes, -1, -1).contiguous()
        else:
            # Expensive but real: K autoregressive rollouts with temperature.
            xy = self._sample_modes(past_ch, K=self.num_modes)            # (B, K, T_fut, 2)

        prediction = {
            "predicted_probability": torch.full((B, self.num_modes), 1.0 / self.num_modes, device=device),
            "predicted_trajectory": self._pack_gmm(xy),
        }
        return prediction, loss

    # ==================================================================
    # Teacher-forced decoding (for training metrics)
    # ==================================================================

    def _teacher_forced_xy(self, logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Argmax-decode x, y at the future positions of the teacher-forced output.

        ``logits[ch]`` has shape (B, T-1, ch_size). The first index whose target
        is a future position is ``past_len - 1``.
        """
        start = self.past_len - 1
        x_idx = logits["x"][:, start:, :].argmax(dim=-1)                  # (B, T_fut)
        y_idx = logits["y"][:, start:, :].argmax(dim=-1)
        return self.tokenizer.decode_xy(x_idx, y_idx)                     # (B, T_fut, 2)

    # ==================================================================
    # Autoregressive multi-modal sampling
    # ==================================================================

    @torch.no_grad()
    def _sample_modes(self, past_ch: torch.Tensor, K: int) -> torch.Tensor:
        """Run K independent autoregressive rollouts using KV caching.

        Without caching, each future timestep re-runs full attention over the
        entire (growing) sequence — O(T_fut * (T_past + T_fut)^2 * n_layer).
        With a KV cache we pay O((T_past + T_fut) * n_layer * n_embd^2) once
        plus O(T_fut * (T_past + T_fut) * n_layer * n_embd) for the per-step
        attentions — roughly a 50-100x speedup at our shapes.

        past_ch : (B, T_past, 4)
        returns : (B, K, T_fut, 2)
        """
        B = past_ch.shape[0]
        T_past = self.past_len
        T_fut = self.future_len

        # Encode past once; tile K-fold along batch dim.
        past_tok = self.tokenizer.encode(past_ch)                         # (B, T_past, 4)
        tokens_bk = past_tok.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, T_past, 4)

        # 1) Initial pass: warm the KV cache with the entire past in one shot.
        logits, caches = self.net.forward_step(tokens_bk, past_caches=None, position_offset=0)
        last_logits = {ch: logits[ch][:, -1, :] for ch in logits}

        # 2) Autoregressive loop: each iteration appends exactly one new token,
        #    and only that token's Q is computed against the cached K, V.
        generated = []
        for step in range(T_fut):
            next_tok = self._sample_next(last_logits)                     # (B*K, 4)
            generated.append(next_tok)
            next_tok_seq = next_tok.unsqueeze(1)                          # (B*K, 1, 4)
            logits, caches = self.net.forward_step(
                next_tok_seq, past_caches=caches, position_offset=T_past + step,
            )
            last_logits = {ch: logits[ch][:, -1, :] for ch in logits}

        fut_tokens = torch.stack(generated, dim=1)                        # (B*K, T_fut, 4)
        xy = self.tokenizer.decode_xy(fut_tokens[..., 0], fut_tokens[..., 1])
        return xy.view(B, K, T_fut, 2)

    def _sample_next(self, last_logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Temperature-sample one token per channel from the final-position logits."""
        out = []
        for ch in ("x", "y", "sog", "cog"):
            scaled = last_logits[ch] / self.sample_temperature
            probs = F.softmax(scaled, dim=-1)
            idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            out.append(idx)
        return torch.stack(out, dim=-1)                                   # (B*K, 4)

    # ==================================================================
    # GMM packing
    # ==================================================================

    @staticmethod
    def _pack_gmm(xy: torch.Tensor) -> torch.Tensor:
        """xy: (B, K, T_fut, 2) → (B, K, T_fut, 5) with log_std=0, rho=0.

        BaseModel.log_info uses only the first 2 channels for ADE/FDE, but the
        full 5-channel shape is required for compatibility with other models in
        the pipeline (e.g. Wayformer outputs a real GMM).
        """
        B, K, T, _ = xy.shape
        log_std = torch.zeros(B, K, T, 2, device=xy.device, dtype=xy.dtype)
        rho = torch.zeros(B, K, T, 1, device=xy.device, dtype=xy.dtype)
        return torch.cat([xy, log_std, rho], dim=-1)

    # ==================================================================
    # Optimizer
    # ==================================================================

    def configure_optimizers(self):
        lr = float(self.config.get("learning_rate", 6e-4))
        wd = float(self.config.get("weight_decay", 0.0))
        opt = optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        sched = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(self.config.get("max_epochs", 50))
        )
        return {"optimizer": opt, "lr_scheduler": sched}