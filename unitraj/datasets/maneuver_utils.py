"""
Maneuver classification utilities for AIS trajectory diagnosis (and, later, rebalancing).

Why this module exists
----------------------
The AIS dataloader (`AISDataset`) does NOT inherit `BaseDataset`, so it bypasses
the Waymo-style postprocess that computes `trajectory_type` / `kalman_difficulty`
(`base_dataset.py:453-456`). Instead it hardcodes `trajectory_type = 0` and
`kalman_difficulty = 45` (`ais_dataset.py:907,911`). As a result we have no per-sample
notion of "is this window a straight line or a maneuver?" — which is exactly what we
need to (a) quantify the straight-line bias and (b) drive maneuver-aware resampling.

This module provides a *frame-invariant* maneuver classifier that works directly on
ego-relative xy positions (meters), independent of whether positions were rotated to
the ego heading. It reuses the existing 8-class `classify_track` (WOD scheme) for
consistency with `base_model.py`/`data_analysis.py`, and adds coarse buckets plus the
continuous metrics that matter for the constant-velocity-shortcut analysis:

  - heading_change_deg : net course change over the future (deg)
  - max_lateral_dev_m  : max perpendicular deviation of GT from the constant-velocity
                         line extrapolated from the last observed velocity (meters)
  - cv_fde_m           : final displacement error of the constant-velocity prediction
                         (== the "Kalman difficulty" idea: how much a straight-line
                         baseline misses by). Large cv_fde_m == strong maneuver.

All functions are pure-numpy and model-free.
"""

import numpy as np

from unitraj.datasets.common_utils import classify_track, TrajectoryType

# Coarse bucket names keyed by the 8-class WOD trajectory type.
TRAJ_TYPE_NAMES = {
    TrajectoryType.STATIONARY: "stationary",
    TrajectoryType.STRAIGHT: "straight",
    TrajectoryType.STRAIGHT_RIGHT: "straight_right",
    TrajectoryType.STRAIGHT_LEFT: "straight_left",
    TrajectoryType.RIGHT_U_TURN: "right_u_turn",
    TrajectoryType.RIGHT_TURN: "right_turn",
    TrajectoryType.LEFT_U_TURN: "left_u_turn",
    TrajectoryType.LEFT_TURN: "left_turn",
    -1: "unknown",
}

COARSE_ORDER = ["stationary", "straight", "mild_turn", "sharp_turn", "unknown"]


def _heading(vec):
    """Heading (rad) of a 2D vector via atan2(y, x). Returns nan for ~zero vectors."""
    if np.linalg.norm(vec) < 1e-6:
        return np.nan
    return float(np.arctan2(vec[1], vec[0]))


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def coarse_bucket(heading_change_deg, max_lateral_dev_m, speed0_mps,
                  straight_heading_deg=10.0, sharp_heading_deg=30.0,
                  lateral_straight_m=15.0, min_speed_mps=1.0,
                  net_displacement_m=10.0, end_displacement_m=None):
    """
    Map continuous metrics to a coarse bucket: stationary / straight / mild_turn / sharp_turn.

    Thresholds are deliberately conservative for the 5-min/1Hz AIS regime, where a
    "straight" cargo run still wanders a few meters laterally. Tune via args.
    """
    if speed0_mps < min_speed_mps and (end_displacement_m is not None and
                                       end_displacement_m < net_displacement_m):
        return "stationary"
    if heading_change_deg < straight_heading_deg and max_lateral_dev_m < lateral_straight_m:
        return "straight"
    if heading_change_deg < sharp_heading_deg:
        return "mild_turn"
    return "sharp_turn"


def classify_maneuver(past_xy, future_xy, dt=1.0,
                      straight_heading_deg=10.0, sharp_heading_deg=30.0,
                      lateral_straight_m=15.0):
    """
    Classify a single trajectory window.

    Args:
        past_xy:   [Tp, 2] ego-relative past positions in METERS (any consistent frame).
        future_xy: [Tf, 2] ego-relative GT future positions in METERS (same frame).
        dt:        seconds per step (1.0 at 1 Hz).

    Returns:
        dict with keys: traj_type (int 0-7 or -1), traj_type_name, coarse,
        heading_change_deg, max_lateral_dev_m, cv_fde_m, speed0_mps.
        Returns coarse="unknown" / traj_type=-1 when the window is too short/degenerate.
    """
    past_xy = np.asarray(past_xy, dtype=np.float64)
    future_xy = np.asarray(future_xy, dtype=np.float64)

    result = {
        "traj_type": -1,
        "traj_type_name": "unknown",
        "coarse": "unknown",
        "heading_change_deg": float("nan"),
        "max_lateral_dev_m": float("nan"),
        "cv_fde_m": float("nan"),
        "speed0_mps": float("nan"),
    }

    if past_xy.shape[0] < 2 or future_xy.shape[0] < 1:
        return result

    p0 = past_xy[-1]                          # last observed position
    v0 = (past_xy[-1] - past_xy[-2]) / dt     # last observed velocity (CV reference)
    speed0 = float(np.linalg.norm(v0))
    result["speed0_mps"] = speed0

    # --- Constant-velocity (straight-line) reference path ---
    T = future_xy.shape[0]
    t = np.arange(1, T + 1) * dt
    cv = p0[None, :] + v0[None, :] * t[:, None]         # [T, 2]
    result["cv_fde_m"] = float(np.linalg.norm(future_xy[-1] - cv[-1]))

    # --- Max lateral deviation of GT from the CV line ---
    if speed0 > 1e-6:
        dirn = v0 / speed0
        normal = np.array([-dirn[1], dirn[0]])
        lateral = np.abs((future_xy - p0) @ normal)
        result["max_lateral_dev_m"] = float(lateral.max())
    else:
        result["max_lateral_dev_m"] = float(np.linalg.norm(future_xy - cv, axis=1).max())

    # --- Net heading change (initial course vs final course) ---
    h0 = _heading(v0) if speed0 > 1e-6 else _heading(future_xy[0] - p0)
    if future_xy.shape[0] >= 2:
        hf = _heading(future_xy[-1] - future_xy[-2])
    else:
        hf = _heading(future_xy[-1] - p0)
    if np.isnan(h0) or np.isnan(hf):
        dheading = 0.0
    else:
        dheading = float(np.degrees(np.abs(_wrap_to_pi(hf - h0))))
    result["heading_change_deg"] = dheading

    end_disp = float(np.linalg.norm(future_xy[-1] - p0))

    # --- 8-class WOD label (consistent with base_model.py per-type metrics) ---
    # Pass the *actual* headings so classify_track is frame-invariant.
    try:
        start_v = v0
        end_v = (future_xy[-1] - future_xy[-2]) if future_xy.shape[0] >= 2 else v0
        traj_type = classify_track(
            start_point=p0, end_point=future_xy[-1],
            start_velocity=start_v, end_velocity=end_v,
            start_heading=h0 if not np.isnan(h0) else 0.0,
            end_heading=hf if not np.isnan(hf) else 0.0,
        )
    except Exception:
        traj_type = -1
    result["traj_type"] = int(traj_type)
    result["traj_type_name"] = TRAJ_TYPE_NAMES.get(traj_type, "unknown")

    result["coarse"] = coarse_bucket(
        dheading, result["max_lateral_dev_m"], speed0,
        straight_heading_deg=straight_heading_deg, sharp_heading_deg=sharp_heading_deg,
        lateral_straight_m=lateral_straight_m,
        end_displacement_m=end_disp,
    )
    return result


def _valid_xy(traj, mask, max_len=None):
    """Return the valid xy rows (mask>0) of a [T, >=2] trajectory in order."""
    traj = np.asarray(traj)
    xy = traj[:, :2]
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        if max_len is not None:
            xy, m = xy[:max_len], m[:max_len]
        xy = xy[m]
    elif max_len is not None:
        xy = xy[:max_len]
    return xy


def classify_batch(input_dict, position_scale=100.0, past_len=300, dt=1.0,
                   to_numpy=True, **thresholds):
    """
    Classify every sample in a batch's `input_dict` (the dict produced by AISDataset.collate_fn).

    Uses the EGO past (`obj_trajs[track_index_to_predict]`) and the ego GT future
    (`center_gt_trajs`), both converted from normalized units back to meters via
    `position_scale`. Frame is consistent between the two (same ego-centric normalized frame).

    Returns a list (len == batch_size) of dicts from `classify_maneuver`.
    """
    def _np(x):
        if to_numpy and hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    obj_trajs = _np(input_dict["obj_trajs"])                    # [B, A, T_past, F]
    obj_mask = _np(input_dict["obj_trajs_mask"])                # [B, A, T_past]
    gt = _np(input_dict["center_gt_trajs"])                     # [B, T_fut, >=2]
    gt_mask = _np(input_dict["center_gt_trajs_mask"])           # [B, T_fut]
    tip = _np(input_dict["track_index_to_predict"]).reshape(-1)  # [B]

    B = obj_trajs.shape[0]
    out = []
    for i in range(B):
        a = int(tip[i])
        past_xy = _valid_xy(obj_trajs[i, a, :, :2], obj_mask[i, a], max_len=past_len) * position_scale
        fut_xy = _valid_xy(gt[i, :, :2], gt_mask[i]) * position_scale
        out.append(classify_maneuver(past_xy, fut_xy, dt=dt, **thresholds))
    return out