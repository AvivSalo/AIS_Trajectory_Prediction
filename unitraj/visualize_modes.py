"""
Multimodal trajectory visualizer for wayformer_ais — draws ALL 6 predicted modes with
their softmax probabilities, in the ego-relative (meters) frame, so you can see the modes
fan out and check whether the high-probability mode continues the demonstrated turn.

Gated to wayformer_ais only (asserts model_name). Prioritizes mid-turn windows (curved
past) since those are the diagnostic-interesting cases, and also includes a few straight/
anchored windows for contrast.

Each plot shows:
  - past track (gray, current position = black dot at origin)
  - GT future (green solid)
  - 6 predicted modes (colored; linewidth & opacity scaled by probability; legend = p%)

Usage (run from unitraj/):
    python -u visualize_modes.py \
        method=wayformer_ais_5min_ec2 past_len=300 future_len=300 stride=300 \
        debug=True "max_data_num=[10]" exp_name=modes_highrisk \
        ckpt_path=<...>/model.ckpt "val_data_path=[<...>/high_risk_22_eval]"

Output: mode_visualizations_<exp_name>/*.png
"""

import os

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from models import build_model
from datasets import build_dataset
from datasets.maneuver_utils import signed_turn_deg
from utils.utils import set_seed


def _to_device(d, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}


def _load_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg["eval"] = True

    model_name = str(getattr(cfg, "model_name", ""))
    assert model_name == "wayformer_ais", \
        f"visualize_modes is gated to wayformer_ais, got model_name={model_name!r}"

    position_scale = float(getattr(cfg, "position_scale", 100.0))
    past_len = int(getattr(cfg, "past_len", 300))
    batch_size = int(getattr(cfg, "eval_batch_size", 32))
    exp_name = str(getattr(cfg, "exp_name", "run"))
    max_plots = int(getattr(cfg, "max_plots", 30))
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           f"mode_visualizations_{exp_name}".replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if (torch.cuda.is_available() and not cfg.debug) else "cpu")
    model = build_model(cfg)
    if getattr(cfg, "ckpt_path", None):
        print(f"Loading checkpoint: {cfg.ckpt_path}")
        _load_ckpt(model, cfg.ckpt_path)
    model = model.to(device).eval()

    val_set = build_dataset(cfg, val=True)
    loader = DataLoader(val_set, batch_size=batch_size, num_workers=cfg.load_num_workers,
                        shuffle=False, drop_last=False, collate_fn=val_set.collate_fn)
    print(f"Val size: {len(val_set)}  device={device}  -> {out_dir}")

    # Collect candidate windows with everything needed to plot, tagged by |past turn|.
    candidates = []  # (priority, dict)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch["input_dict"] = _to_device(batch["input_dict"], device)
            inp = batch["input_dict"]
            prediction, _ = model(batch)
            pred = prediction["predicted_trajectory"][:, :, :, :2].cpu().numpy()   # [B,K,T,2]
            prob = prediction.get("predicted_probability")
            prob = prob.cpu().numpy() if prob is not None else None                # [B,K] (softmaxed)
            obj = inp["obj_trajs"].cpu().numpy()
            objm = inp["obj_trajs_mask"].cpu().numpy()
            gt = inp["center_gt_trajs"][:, :, :2].cpu().numpy()
            gtm = inp["center_gt_trajs_mask"].cpu().numpy()
            tip = inp["track_index_to_predict"].cpu().numpy().reshape(-1)
            sid = inp.get("scenario_id", [f"b{bi}_{i}" for i in range(pred.shape[0])])
            if isinstance(sid, str):
                sid = [sid]

            for i in range(pred.shape[0]):
                a = int(tip[i])
                pm = objm[i, a, :past_len].astype(bool)
                past_xy = obj[i, a, :past_len, :2][pm] * position_scale
                if past_xy.shape[0] < 4:
                    continue
                fm = gtm[i].astype(bool)
                gt_xy = gt[i][fm] * position_scale
                modes_xy = pred[i] * position_scale                                # [K,T,2]
                probs = prob[i] if prob is not None else np.ones(modes_xy.shape[0]) / modes_xy.shape[0]
                # Motion filter: heading is meaningless at near-zero speed (anchored vessels
                # produce huge spurious "turns" from AIS noise). Require real displacement.
                past_path = float(np.linalg.norm(np.diff(past_xy, axis=0), axis=1).sum())
                past_disp = float(np.linalg.norm(past_xy[-1] - past_xy[0]))
                moving = past_path >= 150.0 and past_disp >= 80.0
                past_turn = signed_turn_deg(past_xy)
                gt_turn = signed_turn_deg(gt_xy) if gt_xy.shape[0] >= 4 else 0.0
                candidates.append(({"moving": moving, "abs_turn": abs(past_turn)}, {
                    "sid": str(sid[i]) if i < len(sid) else f"b{bi}_{i}",
                    "past_xy": past_xy, "gt_xy": gt_xy, "modes_xy": modes_xy,
                    "probs": probs, "past_turn": past_turn, "gt_turn": gt_turn,
                    "moving": moving, "past_path": past_path,
                }))
            if bi % 5 == 0:
                print(f"  batch {bi}: {len(candidates)} candidate windows")

    if not candidates:
        print("No candidate windows.")
        return

    # Only consider MOVING vessels (exclude anchored noise). Among them, prioritize
    # curved-past (genuine mid-turn) windows, keeping a few moving-straight for contrast.
    moving = [c for c in candidates if c[0]["moving"]]
    print(f"  moving windows: {len(moving)} / {len(candidates)} total")
    if not moving:
        print("  No moving windows passed the filter — relaxing to all candidates.")
        moving = candidates
    moving.sort(key=lambda c: -c[0]["abs_turn"])
    curved = [c for c in moving if c[0]["abs_turn"] >= 15.0]
    straight = [c for c in moving if c[0]["abs_turn"] < 15.0]
    n_curved = min(len(curved), int(max_plots * 0.7))
    n_straight = min(len(straight), max_plots - n_curved)
    straight_pick = straight[:: max(1, len(straight) // max(1, n_straight))][:n_straight] if straight else []
    chosen = curved[:n_curved] + straight_pick

    cmap = plt.get_cmap("tab10")
    n_made = 0
    for _, w in chosen:
        fig, ax = plt.subplots(figsize=(8, 8))
        # past
        ax.plot(w["past_xy"][:, 0], w["past_xy"][:, 1], "-", color="0.6", lw=2, label="past")
        ax.plot(0, 0, "ko", ms=8, label="now (last past)")
        # GT future
        if w["gt_xy"].shape[0] >= 2:
            ax.plot(w["gt_xy"][:, 0], w["gt_xy"][:, 1], "-", color="green", lw=2.5, label="GT future")
        # modes, ordered by probability (draw low-prob first so high-prob sits on top)
        order = np.argsort(w["probs"])
        pmax = float(w["probs"].max()) if w["probs"].max() > 0 else 1.0
        for rank, k in enumerate(order):
            p = float(w["probs"][k])
            m = w["modes_xy"][k]
            ax.plot(m[:, 0], m[:, 1], "-", color=cmap(k % 10),
                    lw=1.0 + 3.5 * (p / pmax), alpha=0.35 + 0.6 * (p / pmax),
                    label=f"mode {k}: p={p*100:.0f}%")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x (m, ego-relative)"); ax.set_ylabel("y (m, ego-relative)")
        ax.set_title(f"{w['sid']}\npast turn={w['past_turn']:+.0f}°  GT future turn={w['gt_turn']:+.0f}°  "
                     f"(+=left/CCW)  past path={w['past_path']:.0f}m", fontsize=9)
        # legend ordered by probability descending
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=7, loc="best")
        clean = "".join(ch for ch in w["sid"] if ch.isalnum() or ch in "-_")[:60]
        fpath = os.path.join(out_dir, f"modes_{n_made:02d}_{clean}.png")
        fig.savefig(fpath, dpi=110, bbox_inches="tight")
        plt.close(fig)
        n_made += 1

    print(f"\nWrote {n_made} mode plots to {out_dir}")
    print(f"  (curved-past: {n_curved}, straight/anchored: {len(straight_pick)})")


if __name__ == "__main__":
    main()