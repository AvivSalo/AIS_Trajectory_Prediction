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

# ── Scene-context stratification ─────────────────────────────────────────────
# Three axes orthogonal to the GT-maneuver buckets above, all derived from the
# same window (ego past + neighbour pasts), so no chart or external data needed.
KN = 0.514444                      # m/s per knot
ENCOUNTER_RANGE_M = 5556.0         # 3 nautical miles: beyond this, no encounter
HEADING_SAMPLE = 30                # steps used to estimate a stable heading

SPEED_BUCKETS = ["anchored", "drifting", "manoeuvring",
                 "reduced speed", "service speed", "high speed"]
DENSITY_BUCKETS = ["0", "1-2", "3-5", "6+"]
ENCOUNTER_BUCKETS = ["none", "overtaking", "crossing", "head-on"]
PROXIMITY_BUCKETS = ["<0.1 nm", "0.1-0.5 nm", "0.5-1 nm", "1-3 nm", ">3 nm", "no neighbour"]


def _speed_bucket(speed_kn):
    # Boundaries follow the observed distribution of this fleet, which is bimodal:
    # ~30% of windows sit below 1 kn and ~60% at 10-16 kn, with a near-empty band
    # between 4 and 10 kn. Even splits would leave buckets unpopulated. Names use
    # the conventional maritime regimes rather than arbitrary labels.
    if not np.isfinite(speed_kn) or speed_kn < 0.5:
        return "anchored"          # alongside, moored or at anchor
    if speed_kn < 1.0:
        return "drifting"          # set by wind/current, no effective way on
    if speed_kn < 4.0:
        return "manoeuvring"       # harbour/pilotage speed
    if speed_kn < 10.0:
        return "reduced speed"     # below service speed
    if speed_kn < 14.0:
        return "service speed"     # normal sea speed for this fleet
    return "high speed"


def _proximity_bucket(nearest_m):
    if not np.isfinite(nearest_m):
        return "no neighbour"
    nm = nearest_m / 1852.0
    if nm < 0.1:
        return "<0.1 nm"
    if nm < 0.5:
        return "0.1-0.5 nm"
    if nm < 1.0:
        return "0.5-1 nm"
    if nm < 3.0:
        return "1-3 nm"
    return ">3 nm"


def _density_bucket(n):
    if n <= 0:
        return "0"
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    return "6+"


def _heading_deg(xy):
    """Course over the last HEADING_SAMPLE steps, in degrees. NaN if too short."""
    if xy.shape[0] < 2:
        return float("nan")
    k = min(HEADING_SAMPLE, xy.shape[0] - 1)
    v = xy[-1] - xy[-1 - k]
    if np.linalg.norm(v) < 1e-6:
        return float("nan")
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _ang_diff(a, b):
    """Absolute difference between two courses, folded into [0, 180]."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


def _scene_context(obj_xy, obj_ok, ego_idx, ego_past_xy, past_path_m, past_len, dt=1.0):
    """Traffic density, nearest-neighbour range and encounter geometry for one window.

    obj_xy   : [A, T, 2] all agents' positions in the shared ego-relative frame (metres)
    obj_ok   : [A, T]    per-step validity mask
    Encounter class follows the COLREG *geometry* only (relative course), not
    right-of-way: it is a descriptive bucket, not a rule determination.
    """
    speed_kn = (past_path_m / (past_len * dt)) / KN if past_len else float("nan")

    n_near = 0
    best_d = float("inf")
    best_heading = float("nan")
    for a in range(obj_xy.shape[0]):
        if a == ego_idx:
            continue
        m = obj_ok[a, :past_len].astype(bool)
        if m.sum() < 2:
            continue
        n_near += 1
        both = m & obj_ok[ego_idx, :past_len].astype(bool)
        if both.sum() == 0:
            continue
        d = float(np.linalg.norm(obj_xy[a, :past_len][both] - obj_xy[ego_idx, :past_len][both],
                                 axis=1).min())
        if d < best_d:
            best_d = d
            best_heading = _heading_deg(obj_xy[a, :past_len][m])

    if not np.isfinite(best_d) or best_d > ENCOUNTER_RANGE_M:
        encounter = "none"
    else:
        delta = _ang_diff(_heading_deg(ego_past_xy), best_heading)
        if not np.isfinite(delta):
            encounter = "none"
        elif delta >= 135.0:
            encounter = "head-on"
        elif delta >= 45.0:
            encounter = "crossing"
        else:
            encounter = "overtaking"

    return {
        "speed_kn": speed_kn,
        "speed_regime": _speed_bucket(speed_kn),
        "n_neighbours": n_near,
        "density_bucket": _density_bucket(n_near),
        "nearest_m": best_d if np.isfinite(best_d) else float("nan"),
        "proximity_bucket": _proximity_bucket(best_d),
        "encounter_geom": encounter,
    }


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


def _cross_table(lines, rows, row_field, row_order, col_field, col_order,
                 row_label, values, fde, bl_fde, has_baseline, min_n=20):
    """2-D markdown table over two stratification axes.

    values="margin" prints model−baseline minFDE per cell (the interaction test:
    does the margin widen with density *at a fixed speed*?); values="model"
    prints the model's own minFDE (how hard the cell is). Cell counts are shown
    in parentheses; cells below min_n are suppressed as unreliable.
    """
    lines.append(f"| {row_label} | " + " | ".join(col_order) + " |")
    lines.append("|---" + "|---:" * len(col_order) + "|")
    for rb in row_order:
        cells = []
        for cb in col_order:
            idx = np.array([(r.get(row_field) == rb and r.get(col_field) == cb)
                            for r in rows])
            n = int(idx.sum())
            if n < min_n:
                cells.append(f"– ({n})" if n else "–")
                continue
            if values == "margin":
                if not has_baseline:
                    cells.append("–")
                    continue
                v = fde[idx].mean() - np.nanmean(bl_fde[idx])
                cells.append(f"{v:+.1f} ({n})")
            else:
                cells.append(f"{fde[idx].mean():.1f} ({n})")
        lines.append(f"| {rb} | " + " | ".join(cells) + " |")
    lines.append("")


def _bucket_table(lines, rows, field, order, label, ade, fde, bl_fde, has_baseline):
    """Emit one markdown metrics table, stratified by rows[field]."""
    total = len(rows)
    lines.append(f"| {label} | Count | Share | minADE | minFDE | Miss@10m | Miss@20m "
                 f"| BL minFDE | model−BL |")
    lines.append("|---|------:|------:|-------:|-------:|---------:|---------:|----------:|---------:|")
    seen = set()
    for b in order:
        idx = np.array([r.get(field) == b for r in rows])
        n = int(idx.sum())
        seen.add(b)
        if n == 0:
            lines.append(f"| {b} | 0 | 0.0% | – | – | – | – | – | – |")
            continue
        f_b, a_b, bl_b = fde[idx], ade[idx], bl_fde[idx]
        gap = (f_b.mean() - np.nanmean(bl_b)) if has_baseline else float("nan")
        lines.append(f"| {b} | {n:,} | {_pct(n,total):.1f}% | {a_b.mean():.2f} | {f_b.mean():.2f} | "
                     f"{_pct((f_b>10).sum(),n):.1f}% | {_pct((f_b>20).sum(),n):.1f}% | "
                     f"{np.nanmean(bl_b):.2f} | {gap:+.2f} |")
    # Surface any value the declared order missed, rather than silently dropping it.
    extra = sorted({str(r.get(field)) for r in rows} - seen)
    for b in extra:
        idx = np.array([str(r.get(field)) == b for r in rows])
        n = int(idx.sum())
        f_b, a_b, bl_b = fde[idx], ade[idx], bl_fde[idx]
        gap = (f_b.mean() - np.nanmean(bl_b)) if has_baseline else float("nan")
        lines.append(f"| {b} (unlisted) | {n:,} | {_pct(n,total):.1f}% | {a_b.mean():.2f} | "
                     f"{f_b.mean():.2f} | {_pct((f_b>10).sum(),n):.1f}% | "
                     f"{_pct((f_b>20).sum(),n):.1f}% | {np.nanmean(bl_b):.2f} | {gap:+.2f} |")
    lines.append("")


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

            # Error of the TOP-RANKED mode alone. minADE_K / minFDE_K take the best
            # of K after comparing against ground truth, which no deployed system can
            # do; an operational predictor must commit to the mode it ranks highest.
            # For single-mode models these collapse onto minADE / minFDE.
            _ar = torch.arange(pred.shape[0], device=device)
            top_ade = (ade_modes[_ar, top_mode] * position_scale).cpu().numpy()
            top_fde = (fde_modes[_ar, top_mode] * position_scale).cpu().numpy()

            # Brier-FDE, matching base_model.compute_official_evaluation():
            #   brier_fde = minFDE + (1 - p[best-FDE mode])^2
            # A model that emits no probabilities is treated as fully confident (p=1),
            # which collapses Brier-FDE onto minFDE — the behaviour the deterministic
            # baselines already show in their logged val numbers.
            best_fde_idx = fde_modes.argmin(1)
            if prob is not None:
                p_best = prob[torch.arange(pred.shape[0], device=device), best_fde_idx]
            else:
                p_best = torch.ones(pred.shape[0], device=device)
            brier_fde = (min_fde + ((1.0 - p_best) ** 2).cpu().numpy())

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

                # Motion filter: signed-turn is meaningless at near-zero speed (anchored
                # vessels yield huge spurious turns from AIS noise). Flag genuinely moving.
                past_path = float(np.linalg.norm(np.diff(past_xy, axis=0), axis=1).sum()) if past_xy.shape[0] >= 2 else 0.0
                past_disp = float(np.linalg.norm(past_xy[-1] - past_xy[0])) if past_xy.shape[0] >= 2 else 0.0
                moving = bool(past_path >= 150.0 and past_disp >= 80.0)

                # Scene context: density / encounter geometry / speed regime.
                ctx = _scene_context(
                    obj_xy=obj_trajs[i, :, :, :2] * position_scale,
                    obj_ok=obj_mask[i],
                    ego_idx=a,
                    ego_past_xy=past_xy,
                    past_path_m=past_path,
                    past_len=past_len,
                )

                rows.append({
                    "moving": moving,
                    "past_path_m": past_path,
                    "speed_kn": ctx["speed_kn"],
                    "speed_regime": ctx["speed_regime"],
                    "n_neighbours": ctx["n_neighbours"],
                    "density_bucket": ctx["density_bucket"],
                    "nearest_m": ctx["nearest_m"],
                    "proximity_bucket": ctx["proximity_bucket"],
                    "encounter_geom": ctx["encounter_geom"],
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
                    "top_ade_m": float(top_ade[i]),
                    "top_fde_m": float(top_fde[i]),
                    "brier_fde_m": float(brier_fde[i]),
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
    brier = np.array([r["brier_fde_m"] for r in rows])
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
    lines.append(f"- Brier-FDE: **{brier.mean():.2f}**")
    lines.append(f"- Miss@2m: {_pct((fde>2).sum(),total):.1f}%   Miss@10m: {_pct((fde>10).sum(),total):.1f}%   Miss@20m: {_pct((fde>20).sum(),total):.1f}%")
    t_ade = np.array([r["top_ade_m"] for r in rows])
    t_fde = np.array([r["top_fde_m"] for r in rows])
    lines.append(f"- TOP-MODE only (no oracle): ADE **{t_ade.mean():.2f} m**   "
                 f"FDE **{t_fde.mean():.2f} m**   Miss@10m: {_pct((t_fde>10).sum(),total):.1f}%")
    lines.append(f"  _minADE6/minFDE6 above select the best of K modes after seeing the "
                 f"ground truth; the line above is the single highest-probability mode, "
                 f"i.e. what a deployed system would act on. Oracle gain: "
                 f"{t_ade.mean()-ade.mean():+.2f} m ADE, {t_fde.mean()-fde.mean():+.2f} m FDE._")
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

    # ---- Scene context: density / encounter geometry / speed regime ----
    has_bl = baseline is not None

    lines.append("## Metrics by TRAFFIC DENSITY\n")
    lines.append("_Neighbouring vessels with at least two valid observations in the past "
                 "window. If the model−BL margin widens with density, multi-agent context "
                 "is doing the work. NOTE: the count is censored at 9 — the scene "
                 "representation admits 10 agents including the ego, so `6+` conflates "
                 "6 neighbours with any larger number._\n")
    _bucket_table(lines, rows, "density_bucket", DENSITY_BUCKETS,
                  "Neighbours", ade, fde, bl_fde, has_bl)

    lines.append("## Metrics by PROXIMITY to the nearest vessel\n")
    lines.append("_Minimum ego-to-neighbour range over the past window, bucketed at the "
                 "thresholds used in the dataset overview (Figure 1). `no neighbour` = no "
                 "other vessel observed in the window._\n")
    _bucket_table(lines, rows, "proximity_bucket", PROXIMITY_BUCKETS,
                  "Nearest vessel", ade, fde, bl_fde, has_bl)

    lines.append("## Metrics by ENCOUNTER GEOMETRY\n")
    lines.append(f"_Relative course to the closest-approaching vessel within "
                 f"{ENCOUNTER_RANGE_M/1852.0:.0f} nm: head-on ≥135°, crossing 45–135°, "
                 f"overtaking <45°, `none` = no vessel within range. Descriptive "
                 f"geometry only — not a COLREG right-of-way determination._\n")
    _bucket_table(lines, rows, "encounter_geom", ENCOUNTER_BUCKETS,
                  "Encounter", ade, fde, bl_fde, has_bl)

    lines.append("## Metrics by SPEED REGIME\n")
    lines.append("_Mean speed over the past window: anchored <0.5 kn, drifting 0.5–1 kn, "
                 "manoeuvring 1–4 kn, reduced speed 4–10 kn, service speed 10–14 kn, "
                 "high speed ≥14 kn. Boundaries follow this fleet's observed bimodal "
                 "distribution (~30% below 1 kn, ~60% at 10–16 kn)._\n")
    _bucket_table(lines, rows, "speed_regime", SPEED_BUCKETS,
                  "Speed", ade, fde, bl_fde, has_bl)

    # ---- Density x speed: does interaction help once difficulty is controlled? ----
    lines.append("## Traffic density CONTROLLED FOR speed regime\n")
    lines.append("_Traffic density and speed are confounded: dense water means ports and "
                 "anchorages, where vessels are slow and easy to predict, so a density "
                 "breakdown on its own cannot separate 'more neighbours to attend to' "
                 "from 'easier windows'. Reading **across a row** holds speed roughly "
                 "fixed and varies only the number of neighbours._\n")
    lines.append("_Cells are model−baseline minFDE (negative = model better) with the "
                 "window count in parentheses; cells with fewer than 20 windows are "
                 "suppressed. If multi-agent context is the source of the model's "
                 "advantage, the margin should become more negative left-to-right "
                 "within a row._\n")
    _cross_table(lines, rows, "speed_regime", SPEED_BUCKETS,
                 "density_bucket", DENSITY_BUCKETS,
                 "Speed \\ neighbours", "margin", fde, bl_fde, has_bl)

    lines.append("_The same cells, showing the model's own minFDE — i.e. how hard each "
                 "combination actually is._\n")
    _cross_table(lines, rows, "speed_regime", SPEED_BUCKETS,
                 "density_bucket", DENSITY_BUCKETS,
                 "Speed \\ neighbours", "model", fde, bl_fde, has_bl)

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
    n_moving = sum(1 for r in rows if r["moving"])
    # MOVING-ONLY: exclude anchored/near-stationary windows where signed-turn is AIS noise.
    curved = [r for r in rows if r["moving"] and abs(r["past_turn_deg"]) >= PAST_TURN_THR]
    lines.append("\n## Turn continuation — when the PAST is curved (ship mid-turn)\n")
    lines.append(f"_MOVING windows only (past path ≥ 150 m AND displacement ≥ 80 m): "
                 f"{n_moving:,} of {total:,} windows are moving; the rest are anchored/"
                 f"stopping and excluded (their signed-turn is AIS heading noise)._\n")
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

    # ---- W&B ----
    # Every eval run logs the settings that actually shaped the number (data path,
    # stride, max_data_num, horizons, num_modes, ckpt) alongside the metrics, so two
    # runs can never again be compared without knowing whether they were measured
    # the same way. Set wandb_log=False to skip.
    if bool(getattr(cfg, "wandb_log", True)):
        try:
            import wandb
            eval_cfg = {
                "ckpt_path": str(getattr(cfg, "ckpt_path", "")),
                "val_data_path": [str(x) for x in (getattr(cfg, "val_data_path", []) or [])],
                "max_data_num": getattr(cfg, "max_data_num", None),
                "stride": getattr(cfg, "stride", None),
                "past_len": past_len,
                "future_len": getattr(cfg, "future_len", None),
                "num_modes": getattr(cfg, "num_modes", None),
                "eval_batch_size": batch_size,
                "eval_max_batches": max_batches,
                "position_scale": position_scale,
                "model_name": model_name,
                "n_samples": total,
            }
            run = wandb.init(
                project=str(getattr(cfg, "wandb_project", "unitraj")),
                entity=getattr(cfg, "wandb_entity", None),
                name=exp_name,
                job_type="eval",
                config=eval_cfg,
                settings=wandb.Settings(init_timeout=180),
            )
            run.log({
                "test/minADE6": float(ade.mean()),
                "test/minFDE6": float(fde.mean()),
                "test/brier_fde": float(brier.mean()),
                "test/miss_rate_2m": float((fde > 2).mean()),
                "test/miss_rate_10m": float((fde > 10).mean()),
                "test/miss_rate_20m": float((fde > 20).mean()),
                "test/n_samples": total,
            })
            run.finish()
            print(f"Logged to W&B: {exp_name}")
        except Exception as exc:
            print(f"[wandb] logging skipped: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
