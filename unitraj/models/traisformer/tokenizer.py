"""
Fourhot tokenizer for TrAISformer.

WHAT THIS DOES
--------------
Quantize a continuous (x, y, sog, cog) 4-tuple per timestep into 4 categorical
bin indices — one per channel — that the GPT-style transformer can predict
with cross-entropy. This is the "fourhot" representation from the TrAISformer
paper (CIA-Oceanix/TrAISformer).

UNITS / FRAME
-------------
Inputs are expected in the UniTraj-normalized, ego-relative space produced by
``unitraj/datasets/ais_dataset.py``:

  - x, y   : ego-relative meters / ``position_scale`` (= 100m by default)
             i.e. integers below mean "hectometers from ego start, rotated to
             ego initial heading". Range observed on EC2 training data ~ ±26 hm.
  - sog    : speed in m/s / ``velocity_scale`` (= 20 m/s by default).
             Range observed ~ 0..0.54 (0..21 kn).
  - cog    : heading **relative to ego's initial heading**, in radians, in
             ``[-π, π]``. NOTE: this is *not* absolute COG — using ego-relative
             heading is consistent with how the rest of the loader encodes
             features (rotated frame).

The grid bounds (x_min..cog_max) and per-channel bin counts are read from the
model config (see ``unitraj/configs/method/traisformer.yaml``). They were
derived empirically from the real EC2 training set via
``scripts/inspect_ais_ranges.py`` (p0.1 / p99.9 with 10% headroom).
"""

from __future__ import annotations

import torch


class FourHotTokenizer:
    """Bin (x, y, sog, cog) continuous values to/from integer token indices."""

    def __init__(self, cfg):
        self.x_min, self.x_max = float(cfg.x_min), float(cfg.x_max)
        self.y_min, self.y_max = float(cfg.y_min), float(cfg.y_max)
        # sog is always non-negative; we treat 0..sog_max as the bin range.
        self.sog_max = float(cfg.sog_max)
        self.cog_min, self.cog_max = float(cfg.cog_min), float(cfg.cog_max)

        self.x_size = int(cfg.x_size)
        self.y_size = int(cfg.y_size)
        self.sog_size = int(cfg.sog_size)
        self.cog_size = int(cfg.cog_size)

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """``features``: (..., 4) continuous (x, y, sog, cog). Returns (..., 4) int64."""
        x_idx = self._bin(features[..., 0], self.x_min, self.x_max, self.x_size)
        y_idx = self._bin(features[..., 1], self.y_min, self.y_max, self.y_size)
        sog_idx = self._bin(features[..., 2], 0.0, self.sog_max, self.sog_size)
        cog_idx = self._bin(features[..., 3], self.cog_min, self.cog_max, self.cog_size)
        return torch.stack([x_idx, y_idx, sog_idx, cog_idx], dim=-1)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens``: (..., 4) int. Returns (..., 4) float bin-center values."""
        x = self._unbin(tokens[..., 0], self.x_min, self.x_max, self.x_size)
        y = self._unbin(tokens[..., 1], self.y_min, self.y_max, self.y_size)
        sog = self._unbin(tokens[..., 2], 0.0, self.sog_max, self.sog_size)
        cog = self._unbin(tokens[..., 3], self.cog_min, self.cog_max, self.cog_size)
        return torch.stack([x, y, sog, cog], dim=-1)

    # Convenience: just (x, y) decode — used by the wrapper to convert predicted
    # tokens back to UniTraj-normalized positions for ADE/FDE.
    def decode_xy(self, x_idx: torch.Tensor, y_idx: torch.Tensor) -> torch.Tensor:
        x = self._unbin(x_idx, self.x_min, self.x_max, self.x_size)
        y = self._unbin(y_idx, self.y_min, self.y_max, self.y_size)
        return torch.stack([x, y], dim=-1)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _bin(values: torch.Tensor, lo: float, hi: float, n: int) -> torch.Tensor:
        u = (values - lo) / (hi - lo)
        u = u.clamp(0.0, 1.0 - 1e-6)
        return (u * n).long()

    @staticmethod
    def _unbin(idx: torch.Tensor, lo: float, hi: float, n: int) -> torch.Tensor:
        # bin centres
        u = (idx.float() + 0.5) / n
        return u * (hi - lo) + lo
