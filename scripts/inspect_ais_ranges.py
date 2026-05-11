"""
Inspect the value ranges of UniTraj-processed AIS scenarios so we can set
fourhot quantization bounds for the TrAISformer baseline.

Outputs min / max / percentile stats for the ego trajectory's
(x, y, speed, cog) channels across all timesteps (past + future combined),
and prints a ready-to-paste YAML snippet for
``unitraj/configs/method/traisformer.yaml``.

The script reuses ``AISDataset`` so the numbers reflect *exactly* what the
model will see at train time (same normalization, same ego-relative frame).

Usage
-----
Local sample:
    python scripts/inspect_ais_ranges.py \\
        --data data/val_9_scenes_from_ec2 \\
        --max-scenarios 0

EC2 full training set:
    python scripts/inspect_ais_ranges.py \\
        --data /home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/data/processed_ais_4hours_optimized/train \\
        --max-scenarios 5000
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unitraj.datasets.ais_dataset import AISDataset  # noqa: E402


def build_loader_config(data_path: str, max_files: int | None) -> dict:
    """Minimal config to instantiate AISDataset for inspection.

    ``max_files`` caps how many pickle files the loader reads upfront — the
    AIS loader greedily slurps every file in ``data_path`` into memory before
    we get to iterate, so for big training sets we MUST cap this or we wait
    minutes and burn tens of GB just to read a few thousand scenarios.
    """
    return {
        "val_data_path": [data_path],
        "train_data_path": [data_path],
        "past_len": 300,
        "future_len": 300,
        "stride": 300,
        "trajectory_sample_interval": 1,
        "max_num_agents": 32,
        "num_agent_feature": 7,
        "normalize_data": True,
        "position_scale": 100.0,
        "velocity_scale": 20.0,
        "max_data_num": [max_files] if max_files else [None],
        "starting_frame": [0],
        "object_type": ["VESSEL"],
        "use_cache": False,
        "overwrite_cache": False,
        "store_data_in_memory": False,
    }


def collect_ego_channels(dataset: AISDataset, max_scenarios: int) -> dict[str, np.ndarray]:
    """Walk scenarios, pull ego (x, y, speed, cog), return arrays of valid samples."""
    xs, ys, sogs, cogs = [], [], [], []
    n = len(dataset) if max_scenarios in (0, None) else min(max_scenarios, len(dataset))

    print(f"Iterating {n} scenarios out of {len(dataset)}...")
    for i in range(n):
        item = dataset[i]
        ego_past = item["obj_trajs"][0]            # (past_len, F)
        ego_past_mask = item["obj_trajs_mask"][0]  # (past_len,)

        # Past
        valid = ego_past_mask.astype(bool)
        x = ego_past[valid, 0]
        y = ego_past[valid, 1]
        sin_h = ego_past[valid, 4]
        cos_h = ego_past[valid, 5]
        speed = ego_past[valid, 6]
        cog = np.arctan2(sin_h, cos_h)

        xs.append(x); ys.append(y); sogs.append(speed); cogs.append(cog)

        # Future (single-agent gt)
        gt = item["center_gt_trajs"]               # (future_len, 2) normalized
        gt_mask = item["center_gt_trajs_mask"].astype(bool)
        xs.append(gt[gt_mask, 0]); ys.append(gt[gt_mask, 1])
        # Future has no speed/heading channels in center_gt_trajs — skip.

        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{n}")

    return {
        "x": np.concatenate(xs),
        "y": np.concatenate(ys),
        "speed": np.concatenate(sogs),
        "cog": np.concatenate(cogs),
    }


def summarise(name: str, arr: np.ndarray) -> dict[str, float]:
    pcts = np.percentile(arr, [0.1, 1, 5, 50, 95, 99, 99.9])
    stats = {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p0.1": float(pcts[0]),
        "p1": float(pcts[1]),
        "p5": float(pcts[2]),
        "p50": float(pcts[3]),
        "p95": float(pcts[4]),
        "p99": float(pcts[5]),
        "p99.9": float(pcts[6]),
    }
    print(f"\n[{name}]  n={stats['count']:,}")
    print(f"  min/max     : {stats['min']:+.4f} / {stats['max']:+.4f}")
    print(f"  mean / std  : {stats['mean']:+.4f} / {stats['std']:.4f}")
    print(f"  p0.1 / p99.9: {stats['p0.1']:+.4f} / {stats['p99.9']:+.4f}")
    print(f"  p1   / p99  : {stats['p1']:+.4f} / {stats['p99']:+.4f}")
    print(f"  p5   / p95  : {stats['p5']:+.4f} / {stats['p95']:+.4f}")
    return stats


def emit_yaml_snippet(stats: dict[str, dict[str, float]], pad: float = 0.10) -> str:
    """Produce ready-to-paste YAML for traisformer.yaml.

    Bounds use p0.1 / p99.9 with ``pad`` extra fraction on each side to leave
    headroom for tail samples without wasting too much fourhot resolution.
    """
    def widen(lo: float, hi: float) -> tuple[float, float]:
        span = hi - lo
        return lo - pad * span, hi + pad * span

    x_lo, x_hi = widen(stats["x"]["p0.1"], stats["x"]["p99.9"])
    y_lo, y_hi = widen(stats["y"]["p0.1"], stats["y"]["p99.9"])
    sog_hi = max(stats["speed"]["p99.9"] * (1 + pad), 1.0)

    return (
        "\n# === Paste into unitraj/configs/method/traisformer.yaml ===\n"
        "# Bounds derived from empirical p0.1/p99.9 of ego (x, y, speed)\n"
        "# in UniTraj-normalized space (position_scale=100m, velocity_scale=20).\n"
        "fourhot_bounds:\n"
        f"  x_min: {x_lo:.4f}\n"
        f"  x_max: {x_hi:.4f}\n"
        f"  y_min: {y_lo:.4f}\n"
        f"  y_max: {y_hi:.4f}\n"
        f"  sog_max: {sog_hi:.4f}   # speed clamped to [0, sog_max]\n"
        "  cog_min: -3.1416         # radians, full circle\n"
        "  cog_max:  3.1416\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Path to UniTraj-processed AIS directory.")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Cap on pickle files loaded upfront. ~26 scenarios per file. "
                         "Recommend 200 for a 5000-scenario sample.")
    ap.add_argument("--max-scenarios", type=int, default=0,
                    help="Cap on scenarios to iterate (0 = all).")
    ap.add_argument("--pad", type=float, default=0.10,
                    help="Fraction of range to add as headroom on each side of p0.1/p99.9.")
    args = ap.parse_args()

    cfg = build_loader_config(args.data, args.max_files)
    dataset = AISDataset(config=cfg, is_validation=True)

    if len(dataset) == 0:
        print(f"ERROR: no scenarios found at {args.data}", file=sys.stderr)
        return 1

    channels = collect_ego_channels(dataset, args.max_scenarios)

    stats = {name: summarise(name, arr) for name, arr in channels.items()}

    print(emit_yaml_snippet(stats, pad=args.pad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
