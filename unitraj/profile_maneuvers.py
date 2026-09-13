"""
Model-free maneuver profiler for AIS datasets.

Scans a dataset (train or val) and classifies every window's GROUND-TRUTH future
maneuver, then reports the distribution. This quantifies the straight-line bias that
makes Wayformer collapse to "predict straight ahead".

Usage (run from the `unitraj/` directory, like train.py / evaluation.py):

    # profile the validation set referenced by the active config
    python profile_maneuvers.py profile_split=val

    # profile the training set, capped to 500 batches, with explicit data path
    python profile_maneuvers.py profile_split=train profile_max_batches=500 \
        train_data_path='[/home/aviv/Projects/UniTraj/data/processed_ais_data]'

    # profile the high-risk eval set
    python profile_maneuvers.py profile_split=val \
        val_data_path='[/home/aviv/Projects/UniTraj/data/high_risk_22_eval]'

Outputs (under claudedocs/):
    maneuver_profile_<split>_<exp_name>.md   — human-readable summary
    maneuver_profile_<split>_<exp_name>.csv  — per-sample metrics
"""

import os
import csv
from collections import Counter

import numpy as np
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets import build_dataset
from datasets.maneuver_utils import classify_batch, COARSE_ORDER, TRAJ_TYPE_NAMES
from utils.utils import set_seed

CLAUDEDOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "claudedocs")


def _hist(values, edges):
    values = np.asarray([v for v in values if np.isfinite(v)])
    counts = [int(((values >= edges[k]) & (values < edges[k + 1])).sum())
              for k in range(len(edges) - 1)]
    return counts


def _pct(n, total):
    return f"{100.0 * n / total:5.1f}%" if total else "  0.0%"


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg['eval'] = True  # disable any train-time augmentation/sampling tricks

    split = str(getattr(cfg, 'profile_split', 'val'))
    max_batches = getattr(cfg, 'profile_max_batches', None)
    position_scale = float(getattr(cfg, 'position_scale', 100.0))
    past_len = int(getattr(cfg, 'past_len', 300))
    batch_size = int(getattr(cfg, 'eval_batch_size', 96))

    dataset = build_dataset(cfg, val=(split == 'val'))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=cfg.load_num_workers,
                        shuffle=False, drop_last=False, collate_fn=dataset.collate_fn)

    print(f"\nProfiling maneuvers — split={split}  past_len={past_len}  position_scale={position_scale}")
    print(f"Dataset size: {len(dataset)}  batch_size={batch_size}")

    rows = []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= int(max_batches):
            print(f"Stopping early at {max_batches} batches (profile_max_batches).")
            break
        per_sample = classify_batch(batch['input_dict'], position_scale=position_scale,
                                    past_len=past_len, dt=1.0)
        rows.extend(per_sample)
        if bi % 20 == 0:
            print(f"  batch {bi}: {len(rows)} samples profiled")

    total = len(rows)
    if total == 0:
        print("No samples profiled — check data paths.")
        return

    coarse_counts = Counter(r['coarse'] for r in rows)
    type_counts = Counter(r['traj_type_name'] for r in rows)
    headings = [r['heading_change_deg'] for r in rows]
    cv_fdes = [r['cv_fde_m'] for r in rows]
    lat_devs = [r['max_lateral_dev_m'] for r in rows]

    heading_edges = [0, 5, 10, 20, 30, 45, 60, 90, 180]
    cvfde_edges = [0, 5, 10, 25, 50, 100, 250, 500, 1e9]
    heading_hist = _hist(headings, heading_edges)
    cvfde_hist = _hist(cv_fdes, cvfde_edges)

    # ---- Write report ----
    os.makedirs(CLAUDEDOCS, exist_ok=True)
    exp_name = str(getattr(cfg, 'exp_name', 'run'))
    base = f"maneuver_profile_{split}_{exp_name}".replace('/', '_')
    md_path = os.path.join(CLAUDEDOCS, base + ".md")
    csv_path = os.path.join(CLAUDEDOCS, base + ".csv")

    lines = []
    lines.append(f"# Maneuver Profile — split=`{split}`  exp=`{exp_name}`\n")
    lines.append(f"- Total windows profiled: **{total:,}**")
    lines.append(f"- past_len={past_len}, position_scale={position_scale}, dt=1.0s (1 Hz)")
    val_paths = getattr(cfg, 'val_data_path' if split == 'val' else 'train_data_path', '')
    lines.append(f"- data_path: `{val_paths}`\n")

    lines.append("## Coarse maneuver distribution\n")
    lines.append("| Bucket | Count | Share |")
    lines.append("|--------|------:|------:|")
    for b in COARSE_ORDER:
        c = coarse_counts.get(b, 0)
        lines.append(f"| {b} | {c:,} | {_pct(c, total)} |")
    turn = coarse_counts.get('mild_turn', 0) + coarse_counts.get('sharp_turn', 0)
    lines.append(f"\n**Any turn (mild+sharp): {turn:,} ({_pct(turn, total)})**  —  "
                 f"**straight+stationary: {coarse_counts.get('straight',0)+coarse_counts.get('stationary',0):,}**\n")

    lines.append("## 8-class (WOD) distribution\n")
    lines.append("| Type | Count | Share |")
    lines.append("|------|------:|------:|")
    for name in sorted(type_counts, key=lambda n: -type_counts[n]):
        lines.append(f"| {name} | {type_counts[name]:,} | {_pct(type_counts[name], total)} |")

    lines.append("\n## Heading-change distribution (deg)\n")
    lines.append("| Range | Count | Share |")
    lines.append("|-------|------:|------:|")
    for k in range(len(heading_edges) - 1):
        lab = f"{heading_edges[k]}–{heading_edges[k+1]}°"
        lines.append(f"| {lab} | {heading_hist[k]:,} | {_pct(heading_hist[k], total)} |")

    lines.append("\n## Constant-velocity FDE distribution (m)\n")
    lines.append("_How far a straight-line (CV) prediction misses the true endpoint. "
                 "Large values = strong maneuvers a straight prediction cannot capture._\n")
    lines.append("| Range | Count | Share |")
    lines.append("|-------|------:|------:|")
    cvfde_labels = ['0–5', '5–10', '10–25', '25–50', '50–100', '100–250', '250–500', '500+']
    for k in range(len(cvfde_labels)):
        lines.append(f"| {cvfde_labels[k]} m | {cvfde_hist[k]:,} | {_pct(cvfde_hist[k], total)} |")

    finite_cvfde = np.array([v for v in cv_fdes if np.isfinite(v)])
    finite_head = np.array([v for v in headings if np.isfinite(v)])
    finite_lat = np.array([v for v in lat_devs if np.isfinite(v)])
    lines.append("\n## Summary statistics\n")
    lines.append(f"- heading_change_deg: mean={finite_head.mean():.1f}  median={np.median(finite_head):.1f}  p90={np.percentile(finite_head,90):.1f}")
    lines.append(f"- cv_fde_m:           mean={finite_cvfde.mean():.1f}  median={np.median(finite_cvfde):.1f}  p90={np.percentile(finite_cvfde,90):.1f}")
    lines.append(f"- max_lateral_dev_m:  mean={finite_lat.mean():.1f}  median={np.median(finite_lat):.1f}  p90={np.percentile(finite_lat,90):.1f}")

    report = "\n".join(lines) + "\n"
    with open(md_path, 'w') as f:
        f.write(report)

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + report)
    print(f"Wrote: {md_path}")
    print(f"Wrote: {csv_path}")


if __name__ == '__main__':
    main()