"""
Lightweight, standalone maneuver profiler for AIS pickle scenes.

Unlike profile_maneuvers.py (which builds the full AISDataset + DataLoader + torch
stack — heavy enough to peg the machine on 4-hour scenarios), this reads the processed
pickles DIRECTLY, one file at a time, with pure numpy. It mirrors the windowing in
AISDataset._process_pickle_scene (ego = first track, sliding window of past_len+future_len
at the given stride, 500 m/step corruption filter) so the maneuver distribution matches
what the model actually trains on.

Pickle format:
    {'scenario_id': str,
     'tracks': {track_id: {'timestamps':[...],
                           'state': {'position': [T,2] meters, 'velocity': [T,2] m/s}}}}

Usage (run from unitraj/, conda env unitraj):
    python -u profile_maneuvers_fast.py \
        --data /home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/data/processed_ais_4hours_optimized/val \
        --past 300 --future 300 --stride 150 --max-files 0 --tag val

Output: claudedocs/maneuver_profile_fast_<tag>.{md,csv}
"""

import os
import csv
import glob
import pickle
import argparse
from collections import Counter

import numpy as np

from unitraj.datasets.maneuver_utils import classify_maneuver, COARSE_ORDER

CLAUDEDOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "claudedocs")
META = {"dataset_mapping.pkl", "dataset_summary.pkl", "file_list.pkl"}


def _hist(values, edges):
    v = np.asarray([x for x in values if np.isfinite(x)])
    return [int(((v >= edges[k]) & (v < edges[k + 1])).sum()) for k in range(len(edges) - 1)]


def _pct(n, total):
    return f"{100.0 * n / total:5.1f}%" if total else "  0.0%"


def iter_windows(positions, past, future, stride):
    """Yield (past_xy, future_xy) windows mirroring AISDataset._process_pickle_scene."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        return
    positions = positions[:, :2]
    win = past + future
    n = positions.shape[0]
    for s in range(0, n - win + 1, stride):
        seg = positions[s:s + win]
        step = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        if step.size and step.max() > 500.0:   # corruption filter, matches dataset
            continue
        yield seg[:past], seg[past:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory containing ais_*/*.pkl")
    ap.add_argument("--past", type=int, default=300)
    ap.add_argument("--future", type=int, default=300)
    ap.add_argument("--stride", type=int, default=150)
    ap.add_argument("--max-files", type=int, default=0, help="0 = all")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "ais_*", "*.pkl")))
    files += [f for f in sorted(glob.glob(os.path.join(args.data, "*.pkl")))
              if os.path.basename(f) not in META]
    if args.max_files > 0:
        files = files[:args.max_files]
    print(f"Found {len(files)} pickle scenes in {args.data}")
    print(f"Windowing: past={args.past} future={args.future} stride={args.stride}")

    rows = []
    n_scenes = 0
    for fi, fpath in enumerate(files):
        try:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
            tracks = data.get("tracks", {})
            if not tracks:
                continue
            ego_id, ego = next(iter(tracks.items()))
            positions = ego["state"]["position"]
        except Exception as e:
            print(f"  skip {os.path.basename(fpath)}: {e}")
            continue
        n_scenes += 1
        for past_xy, fut_xy in iter_windows(positions, args.past, args.future, args.stride):
            rows.append(classify_maneuver(past_xy, fut_xy, dt=1.0))
        if fi % 10 == 0:
            print(f"  [{fi+1}/{len(files)}] scenes={n_scenes} windows={len(rows)}", flush=True)

    total = len(rows)
    if total == 0:
        print("No windows produced — check --data path / window size vs scene length.")
        return

    coarse = Counter(r["coarse"] for r in rows)
    types = Counter(r["traj_type_name"] for r in rows)
    headings = [r["heading_change_deg"] for r in rows]
    cv_fdes = [r["cv_fde_m"] for r in rows]
    lat = [r["max_lateral_dev_m"] for r in rows]

    h_edges = [0, 5, 10, 20, 30, 45, 60, 90, 180]
    cv_edges = [0, 5, 10, 25, 50, 100, 250, 500, 1e9]
    cv_labels = ['0–5', '5–10', '10–25', '25–50', '50–100', '100–250', '250–500', '500+']
    h_hist = _hist(headings, h_edges)
    cv_hist = _hist(cv_fdes, cv_edges)

    L = []
    L.append(f"# Maneuver Profile (fast) — tag=`{args.tag}`\n")
    L.append(f"- data: `{args.data}`")
    L.append(f"- scenes={n_scenes}  windows={total:,}  past={args.past} future={args.future} stride={args.stride} dt=1s\n")

    L.append("## Coarse maneuver distribution\n")
    L.append("| Bucket | Count | Share |")
    L.append("|--------|------:|------:|")
    for b in COARSE_ORDER:
        L.append(f"| {b} | {coarse.get(b,0):,} | {_pct(coarse.get(b,0), total)} |")
    turn = coarse.get('mild_turn', 0) + coarse.get('sharp_turn', 0)
    straight = coarse.get('straight', 0) + coarse.get('stationary', 0)
    L.append(f"\n**Turns (mild+sharp): {turn:,} ({_pct(turn,total)})  |  straight+stationary: {straight:,} ({_pct(straight,total)})**\n")

    L.append("## 8-class (WOD) distribution\n")
    L.append("| Type | Count | Share |")
    L.append("|------|------:|------:|")
    for name in sorted(types, key=lambda n: -types[n]):
        L.append(f"| {name} | {types[name]:,} | {_pct(types[name], total)} |")

    L.append("\n## Heading-change distribution (deg over the future)\n")
    L.append("| Range | Count | Share |")
    L.append("|-------|------:|------:|")
    for k in range(len(h_edges) - 1):
        L.append(f"| {h_edges[k]}–{h_edges[k+1]}° | {h_hist[k]:,} | {_pct(h_hist[k], total)} |")

    L.append("\n## Constant-velocity FDE distribution (m)\n")
    L.append("_How far a straight-line prediction misses the true endpoint. Large = strong maneuver._\n")
    L.append("| Range | Count | Share |")
    L.append("|-------|------:|------:|")
    for k in range(len(cv_labels)):
        L.append(f"| {cv_labels[k]} m | {cv_hist[k]:,} | {_pct(cv_hist[k], total)} |")

    fh = np.array([x for x in headings if np.isfinite(x)])
    fc = np.array([x for x in cv_fdes if np.isfinite(x)])
    fl = np.array([x for x in lat if np.isfinite(x)])
    L.append("\n## Summary statistics\n")
    L.append(f"- heading_change_deg: mean={fh.mean():.1f} median={np.median(fh):.1f} p90={np.percentile(fh,90):.1f} p99={np.percentile(fh,99):.1f}")
    L.append(f"- cv_fde_m:           mean={fc.mean():.1f} median={np.median(fc):.1f} p90={np.percentile(fc,90):.1f} p99={np.percentile(fc,99):.1f}")
    L.append(f"- max_lateral_dev_m:  mean={fl.mean():.1f} median={np.median(fl):.1f} p90={np.percentile(fl,90):.1f}")

    report = "\n".join(L) + "\n"
    os.makedirs(CLAUDEDOCS, exist_ok=True)
    base = f"maneuver_profile_fast_{args.tag}".replace("/", "_")
    md = os.path.join(CLAUDEDOCS, base + ".md")
    cf = os.path.join(CLAUDEDOCS, base + ".csv")
    with open(md, "w") as f:
        f.write(report)
    with open(cf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + report)
    print(f"Wrote: {md}")
    print(f"Wrote: {cf}")


if __name__ == "__main__":
    main()