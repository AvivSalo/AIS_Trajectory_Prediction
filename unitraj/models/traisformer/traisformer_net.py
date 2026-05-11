"""
TrAISformer transformer backbone.

Architecture is the GPT-style autoregressive transformer from the TrAISformer
paper (Nguyen & Fablet, 2024 — CIA-Oceanix/TrAISformer). We re-implement it in
clean PyTorch so it slots into UniTraj's BaseModel/Lightning pipeline cleanly.

Key design points (kept faithful to the paper):
  - 4 separate token embeddings (one per channel: x, y, sog, cog), summed
    together to form the input vector at each timestep. This is the "fourhot"
    additive embedding.
  - Causal self-attention with a triangular mask (next-token prediction).
  - 4 separate output heads — one per channel — each predicting a categorical
    distribution over its channel's bins. Total loss is the mean of the 4
    cross-entropies.

Adaptation notes for Aviv's UniTraj data (Path B, 5-min/1Hz):
  - max_seqlen here is past_len + future_len = 600 (vs. 108 in the paper).
    Attention is O(N²) so a 600-token context is ~31× more attention work per
    layer than the paper. Default 8 layers × 8 heads × n_embd=256 stays
    tractable on a single GPU. Drop n_layer to 4 for faster iteration.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Transformer building blocks (minGPT-style)
# ----------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, max_seqlen: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        head = C // self.n_head
        q = q.view(B, T, self.n_head, head).transpose(1, 2)
        k = k.view(B, T, self.n_head, head).transpose(1, 2)
        v = v.view(B, T, self.n_head, head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, max_seqlen: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_seqlen, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ----------------------------------------------------------------------
# TrAISformer net
# ----------------------------------------------------------------------


class TrAISformerNet(nn.Module):
    """GPT-style transformer with fourhot input + 4 categorical output heads."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Per-channel token embeddings. Each maps a bin index → n_embd vector.
        # The 4 are SUMMED at each timestep (fourhot additive embedding).
        self.x_emb = nn.Embedding(cfg.x_size, cfg.n_embd)
        self.y_emb = nn.Embedding(cfg.y_size, cfg.n_embd)
        self.sog_emb = nn.Embedding(cfg.sog_size, cfg.n_embd)
        self.cog_emb = nn.Embedding(cfg.cog_size, cfg.n_embd)

        # Learned positional embeddings (size = max past + future = 600 here).
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_seqlen, cfg.n_embd))
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.n_embd, cfg.n_head, cfg.max_seqlen, cfg.dropout)
            for _ in range(cfg.n_layer)
        ])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # Per-channel output heads. Each is a linear projection from hidden to
        # vocab size of that channel. Predicted logits then go through
        # cross-entropy against the next-step token.
        self.x_head = nn.Linear(cfg.n_embd, cfg.x_size)
        self.y_head = nn.Linear(cfg.n_embd, cfg.y_size)
        self.sog_head = nn.Linear(cfg.n_embd, cfg.sog_size)
        self.cog_head = nn.Linear(cfg.n_embd, cfg.cog_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        tokens : (B, T, 4) int64 — per-channel bin indices.
        returns: dict[channel] -> (B, T, channel_size) logits
        """
        B, T, _ = tokens.shape
        h = (self.x_emb(tokens[..., 0])
             + self.y_emb(tokens[..., 1])
             + self.sog_emb(tokens[..., 2])
             + self.cog_emb(tokens[..., 3]))
        h = self.drop(h + self.pos_emb[:, :T])

        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)

        return {
            "x":   self.x_head(h),
            "y":   self.y_head(h),
            "sog": self.sog_head(h),
            "cog": self.cog_head(h),
        }

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def loss(
        logits: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean of 4 per-channel cross-entropies, restricted to valid positions.

        logits     : dict of (B, T, channel_size)
        targets    : (B, T, 4) int64
        valid_mask : (B, T) bool — only loss-bearing positions (e.g. future)
        """
        total = 0.0
        denom = valid_mask.float().sum().clamp(min=1.0)
        for i, name in enumerate(["x", "y", "sog", "cog"]):
            l = logits[name]
            t = targets[..., i]
            ce = F.cross_entropy(
                l.reshape(-1, l.size(-1)),
                t.reshape(-1),
                reduction="none",
            ).view_as(t)
            total = total + (ce * valid_mask.float()).sum() / denom
        return total / 4.0