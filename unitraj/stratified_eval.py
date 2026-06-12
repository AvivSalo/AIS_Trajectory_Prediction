"""
Maneuver-stratified evaluation for the AIS Wayformer (and baselines).

Runs a checkpoint over the validation set and reports the standard geometric metrics
(minADE6 / minFDE6 / Miss@{2,10,20}m) BUT broken down by the ground-truth maneuver
bucket (stationary / straight / mild_turn / sharp_turn). It also:

  1. Compares the model against the OLS linear baseline *within each bucket* — if the
     model ≈ baseline on straight windows and both miss badly on turns, the model has
     learned the constant-velocity shortcut.
  2. Classifies the model's OWN top-probability output trajectory and compares the
     predicted maneuver distribution to the GT distribution — direct mode-collapse
     evidence (e.g. "GT turns 30% of the time, model predicts a turn 2% of the time").

NOTE on Miss@2m: over a 5-min/1Hz horizon a flat 2 m endpoint threshold is near-
unachievable, so it saturates to ~100%. The 10 m / 20 m variants and minFDE (meters)
are the informative numbers for ships.

Usage (run from the `unitraj/` directory):

    python stratified_eval.py \
        ckpt_path=/path/to/best.ckpt \
        val_data_path='[/home/aviv/Projects/UniTraj/data/high_risk_22_eval]'

Output: claudedocs/stratified_eval_<exp_name>.{md,csv}
"""

import os
import csv
from collections import Counter, defaultdict

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from models import build_model
from datasets import build_dataset
from datasets.maneuver_utils import classify_maneuver, classify_batch, signed_turn_deg, COARSE_ORDER
from utils.utils import set_seed

CLAUDEDOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "claudedocs")
BUCKETS = ["stationary", "straight", "mild_turn", "sharp_turn"]


def _to_device(input_dict, device):
    out = {}
    for k, v in input_dict.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def _load_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [load] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [load] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")


def _pct(n, total):
    return 100.0 * n / total if total else 0.0


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg["eval"] = True

    position_scale = float(getattr(cfg, "position_scale", 100.0))
    past_len = int(getattr(cfg, "past_len", 300))
    batch_size = int(getattr(cfg, "eval_batch_size", 96))
    model_name = str(getattr(cfg, "model_name", ""))
    max_batches = getattr(cfg, "eval_max_batches", None)

    device = torch.device("cuda" if (torch.cuda.is_available() and not cfg.debug) else "cpu")

    model = build_model(cfg)
    if getattr(cfg, "ckpt_path", None):
        print(f"Loading checkpoint: {cfg.ckpt_path}")
        _load_ckpt(model, cfg.ckpt_path)
    model = model.to(device).eval()

    val_set = build_dataset(cfg, val=True)
    loader = DataLoader(val_set, batch_size=batch_size, num_workers=cfg.load_num_workers,
                        shuffle=False, drop_last=False, collate_fn=val_set.collate_fn)
    print(f"Val dataset size: {len(val_set)}  batch_size={batch_size}  device={device}")

    # Lazy baseline (skip if the model itself is the baseline)
    baseline = None
    if model_name != "baseline_linear":
        from unitraj.models.baseline_linear.baseline_linear import BaselineLinear
        baseline = BaselineLinear(config=cfg).to(device).eval()

    rows = []  # per-sample dicts

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= int(max_batches):
                print(f"Stopping early at {max_batches} batches (eval_max_batches).")
                break
            batch["input_dict"] = _to_device(batch["input_dict"], device)
            inp = batch["input_dict"]

            prediction, _ = model(batch)
            pred = prediction["predicted_trajectory"][:, :, :, :2]      # [B, K, T, 2] (normalized)
            prob = prediction.get("predicted_probability")              # [B, K] (softmaxed in forward)
            gt = inp["center_gt_trajs"][:, :, :2]                       # [B, T, 2]
            gt_e = gt.unsqueeze(1)                                       # [B, 1, T, 2]

            ade_modes = torch.norm(pred - gt_e, dim=-1).mean(-1)        # [B, K]
            fde_modes = torch.norm(pred[:, :, -1] - gt_e[:, :, -1], dim=-1)  # [B, K]
            min_ade = (ade_modes.min(1).values * position_scale).cpu().numpy()
            min_fde = (fde_modes.min(1).values * position_scale).cpu().numpy()

            top_mode = (prob.argmax(1) if prob is not None
                        else torch.zeros(pred.shape[0], dtype=torch.long, device=device))

            # Baseline
            bl_min_fde = None
            if baseline is not None:
                bl_pred, _ = baseline(batch)
                bl_xy = bl_pred["predicted_trajectory"][:, :, :, :2]
                bl_fde = torch.norm(bl_xy[:, :, -1] - gt_e[:, :, -1], dim=-1)
                bl_ade = torch.norm(bl_xy - gt_e, dim=-1).mean(-1)
                bl_min_fde = (bl_fde.min(1).values * position_scale).cpu().numpy()
                bl_min_ade = (bl_ade.min(1).values * position_scale).cpu().numpy()

            # GT maneuver labels (shared classifier)
            gt_labels = classify_batch(inp, position_scale=position_scale, past_len=past_len, dt=1.0)

            # Predicted top-mode maneuver labels — reuse the ego past, swap in predicted future
            obj_trajs = inp["obj_trajs"].cpu().numpy()
            obj_mask = inp["obj_trajs_mask"].cpu().numpy()
            tip = inp["track_index_to_predict"].cpu().numpy().reshape(-1)
            pred_np = pred.cpu().numpy()
            gt_np = gt.cpu().numpy()
            B = pred_np.shape[0]
            for i in range(B):
                a = int(tip[i])
                m = obj_mask[i, a, :past_len].astype(bool)
                past_xy = obj_trajs[i, a, :past_len, :2][m] * position_scale
                pred_top_xy = pred_np[i, int(top_mode[i].item())] * position_scale
                gt_fut_xy = gt_np[i] * position_scale
                pred_lab = classify_maneuver(past_xy, pred_top_xy, dt=1.0)

                # Signed turns (deg): + = left/CCW, - = right/CW. Tests "does the model
                # continue the past turn or straighten from the last past point?"
                past_turn = signed_turn_deg(past_xy)
                gt_fut_turn = signed_turn_deg(gt_fut_xy)
                pred_fut_turn = signed_turn_deg(pred_top_xy)

                rows.append({
                    "gt_coarse": gt_labels[i]["coarse"],
                    "gt_heading_deg": gt_labels[i]["heading_change_deg"],
                    "gt_cv_fde_m": gt_labels[i]["cv_fde_m"],
                    "pred_coarse": pred_lab["coarse"],
                    "pred_heading_deg": pred_lab["heading_change_deg"],
                    "past_turn_deg": past_turn,
                    "gt_fut_turn_deg": gt_fut_turn,
                    "pred_fut_turn_deg": pred_fut_turn,
                    "min_ade_m": float(min_ade[i]),
                    "min_fde_m": float(min_fde[i]),
                    "bl_min_fde_m": float(bl_min_fde[i]) if bl_min_fde is not None else float("nan"),
                    "bl_min_ade_m": float(bl_min_ade[i]) if bl_min_fde is not None else float("nan"),
                })

            if bi % 10 == 0:
                print(f"  batch {bi}: {len(rows)} samples")

    total = len(rows)
    if total == 0:
        print("No samples evaluated — check data paths / ckpt.")
        return

    ade = np.array([r["min_ade_m"] for r in rows])
    fde = np.array([r["min_fde_m"] for r in rows])
    bl_fde = np.array([r["bl_min_fde_m"] for r in rows])

    # ---- Build report ----
    lines = []
    exp_name = str(getattr(cfg, "exp_name", "run"))
    lines.append(f"# Stratified Evaluation — exp=`{exp_name}`  model=`{model_name}`\n")
    lines.append(f"- ckpt: `{getattr(cfg, 'ckpt_path', 'none')}`")
    lines.append(f"- val_data_path: `{getattr(cfg, 'val_data_path', '')}`")
    lines.append(f"- samples: **{total:,}**  past_len={past_len}  future_len={getattr(cfg,'future_len','?')}\n")

    lines.append("## Overall\n")
    lines.append(f"- minADE6: **{ade.mean():.2f} m**   minFDE6: **{fde.mean():.2f} m**")
    lines.append(f"- Miss@2m: {_pct((fde>2).sum(),total):.1f}%   Miss@10m: {_pct((fde>10).sum(),total):.1f}%   Miss@20m: {_pct((fde>20).sum(),total):.1f}%")
    if baseline is not None:
        lines.append(f"- Baseline (OLS) minFDE6: {np.nanmean(bl_fde):.2f} m   → model−baseline minFDE: **{fde.mean()-np.nanmean(bl_fde):+.2f} m**")
    lines.append("")

    # ---- By GT maneuver bucket ----
    lines.append("## Metrics by GROUND-TRUTH maneuver bucket\n")
    lines.append("_If model minFDE ≈ baseline minFDE on `straight` but both blow up on turns, the model is riding the constant-velocity shortcut._\n")
    lines.append("| GT bucket | Count | Share | minADE | minFDE | Miss@10m | Miss@20m | BL minFDE | model−BL |")
    lines.append("|-----------|------:|------:|-------:|-------:|---------:|---------:|----------:|---------:|")
    for b in BUCKETS:
        idx = np.array([r["gt_coarse"] == b for r in rows])
        n = int(idx.sum())
        if n == 0:
            lines.append(f"| {b} | 0 | 0.0% | – | – | – | – | – | – |")
            continue
        f_b = fde[idx]
        a_b = ade[idx]
        bl_b = bl_fde[idx]
        gap = (f_b.mean() - np.nanmean(bl_b)) if baseline is not None else float("nan")
        lines.append(f"| {b} | {n:,} | {_pct(n,total):.1f}% | {a_b.mean():.2f} | {f_b.mean():.2f} | "
                     f"{_pct((f_b>10).sum(),n):.1f}% | {_pct((f_b>20).sum(),n):.1f}% | "
                     f"{np.nanmean(bl_b):.2f} | {gap:+.2f} |")
    lines.append("")

    # ---- Mode-collapse: predicted vs GT maneuver distribution ----
    gt_dist = Counter(r["gt_coarse"] for r in rows)
    pred_dist = Counter(r["pred_coarse"] for r in rows)
    lines.append("## Mode collapse — predicted vs GT maneuver distribution\n")
    lines.append("| Bucket | GT share | Predicted (top-mode) share |")
    lines.append("|--------|---------:|---------------------------:|")
    for b in COARSE_ORDER:
        lines.append(f"| {b} | {_pct(gt_dist.get(b,0),total):.1f}% | {_pct(pred_dist.get(b,0),total):.1f}% |")

    # Of the GT turns, how many did the model predict as straight/stationary?
    turn_idx = [r for r in rows if r["gt_coarse"] in ("mild_turn", "sharp_turn")]
    if turn_idx:
        flattened = sum(1 for r in turn_idx if r["pred_coarse"] in ("straight", "stationary"))
        lines.append(f"\n**Of {len(turn_idx):,} GT-turn windows, the model's top mode was straight/stationary "
                     f"{flattened:,} times ({_pct(flattened, len(turn_idx)):.1f}%).**")
    gt_head = np.array([r["gt_heading_deg"] for r in rows if np.isfinite(r["gt_heading_deg"])])
    pr_head = np.array([r["pred_heading_deg"] for r in rows if np.isfinite(r["pred_heading_deg"])])
    lines.append(f"\n- Mean GT heading change: {gt_head.mean():.1f}°   |   "
                 f"Mean predicted top-mode heading change: {pr_head.mean():.1f}°")

    # ---- Turn-continuation: does the model continue a mid-turn past, or straighten? ----
    # Tests the observation: "when the past 300 steps are curved (mid-turn), the model
    # draws a straight line from the last point instead of continuing the turn."
    PAST_TURN_THR = 15.0   # deg over the past window to call the ship "mid-turn"
    STRAIGHT_THR = 10.0    # deg over the future below which a prediction is "straight"
    curved = [r for r in rows if abs(r["past_turn_deg"]) >= PAST_TURN_THR]
    lines.append("\n## Turn continuation — when the PAST is curved (ship mid-turn)\n")
    lines.append(f"_Past considered curved if |past signed turn| ≥ {PAST_TURN_THR}°. "
                 f"Prediction 'straight' if |future signed turn| < {STRAIGHT_THR}°. "
                 f"'Continues' = predicted turn same direction as the past turn._\n")
    if not curved:
        lines.append(f"No curved-past windows found (of {total:,}).")
    else:
        nc = len(curved)
        # What GT does after a curved past:
        gt_cont = sum(1 for r in curved if np.sign(r["gt_fut_turn_deg"]) == np.sign(r["past_turn_deg"])
                      and abs(r["gt_fut_turn_deg"]) >= STRAIGHT_THR)
        gt_straight = sum(1 for r in curved if abs(r["gt_fut_turn_deg"]) < STRAIGHT_THR)
        # What the MODEL (most-likely mode) does after a curved past:
        pr_cont = sum(1 for r in curved if np.sign(r["pred_fut_turn_deg"]) == np.sign(r["past_turn_deg"])
                      and abs(r["pred_fut_turn_deg"]) >= STRAIGHT_THR)
        pr_straight = sum(1 for r in curved if abs(r["pred_fut_turn_deg"]) < STRAIGHT_THR)
        lines.append(f"- Curved-past windows: **{nc:,}** ({_pct(nc,total):.1f}% of all)")
        lines.append(f"- Ground truth: continues the turn {gt_cont:,} ({_pct(gt_cont,nc):.1f}%) | "
                     f"goes straight {gt_straight:,} ({_pct(gt_straight,nc):.1f}%)")
        lines.append(f"- Model top-mode: continues the turn {pr_cont:,} ({_pct(pr_cont,nc):.1f}%) | "
                     f"goes straight {pr_straight:,} ({_pct(pr_straight,nc):.1f}%)")
        # The key test: among curved-past windows where GT KEEPS turning, what does the model do?
        gt_keeps = [r for r in curved if np.sign(r["gt_fut_turn_deg"]) == np.sign(r["past_turn_deg"])
                    and abs(r["gt_fut_turn_deg"]) >= STRAIGHT_THR]
        if gt_keeps:
            model_straight = sum(1 for r in gt_keeps if abs(r["pred_fut_turn_deg"]) < STRAIGHT_THR)
            lines.append(f"\n**Of {len(gt_keeps):,} windows where the past is curved AND the ship keeps "
                         f"turning, the model's top mode goes straight {model_straight:,} times "
                         f"({_pct(model_straight,len(gt_keeps)):.1f}%).**")
            pt = np.array([r["past_turn_deg"] for r in gt_keeps])
            gtf = np.array([r["gt_fut_turn_deg"] for r in gt_keeps])
            prf = np.array([r["pred_fut_turn_deg"] for r in gt_keeps])
            lines.append(f"- Mean signed turn (deg): past={pt.mean():+.1f}  GT-future={gtf.mean():+.1f}  "
                         f"model-future={prf.mean():+.1f}  (same sign as past = continuing)")

    report = "\n".join(lines) + "\n"

    os.makedirs(CLAUDEDOCS, exist_ok=True)
    base = f"stratified_eval_{exp_name}".replace("/", "_")
    md_path = os.path.join(CLAUDEDOCS, base + ".md")
    csv_path = os.path.join(CLAUDEDOCS, base + ".csv")
    with open(md_path, "w") as f:
        f.write(report)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + report)
    print(f"Wrote: {md_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
