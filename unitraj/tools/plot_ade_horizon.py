"""
Generate figures/ade_vs_horizon.png

Every point is measured. stratified_eval.py truncates the prediction horizon at
each of the 300 timesteps in the window and recomputes minADE_6 / minFDE_6 there,
writing claudedocs/stratified_eval_test2025_<model>_horizon.csv. This script plots
those files directly — nothing here is shaped, interpolated or extrapolated, and
the t=300 s value of each curve reproduces that model's row in Table 8.

The window itself is 300 s, so the x-axis stops at 5 minutes: no model in the
comparison emits a longer forecast (every output head is exactly 300 timesteps).
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CLAUDEDOCS = "/home/aviv/Projects/UniTraj/claudedocs"
OUT = ("/home/aviv/Projects/UniTraj/Paper_Multi_Agent_Social_Interaction_"
       "AIS_Trajectory_Prediction_Comparing_Methods/figures/ade_vs_horizon.png")

# label -> (csv stem, colour, marker)
MODELS = [
    ("Baseline Linear", "baseline",    "#555555", "s"),
    ("TrAISformer",     "traisformer", "#E07B00", "^"),
    ("AIS-ACNet",       "ais_acnet",   "#2B6CB0", "D"),
    ("Wayformer-AIS",   "wayformer",   "#2D9A4E", "o"),
    ("GAT-LSTM",        "gat_lstm",    "#C0392B", "v"),
]
MARK_MIN = (1, 3, 5)          # minutes at which a marker is drawn


def load(stem):
    path = os.path.join(CLAUDEDOCS, f"stratified_eval_test2025_{stem}_horizon.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["min_ade_m"]) for r in rows])


ade = {label: load(stem) for label, stem, _, _ in MODELS}
t = np.arange(1, len(next(iter(ade.values()))) + 1) / 60.0   # seconds -> minutes

fig, ax = plt.subplots(figsize=(8, 5.5))

for label, _, colour, marker in MODELS:
    lead = label == "Wayformer-AIS"
    ax.plot(t, ade[label], color=colour, linewidth=2.4 if lead else 1.7,
            zorder=4 if lead else 3, label=label)
    for m in MARK_MIN:
        ax.plot(m, ade[label][m * 60 - 1], color=colour, marker=marker,
                markersize=8, markeredgecolor="white", markeredgewidth=0.8,
                zorder=5 if lead else 4)

ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_formatter(mticker.NullFormatter())

ax.set_xlim(0, 5.05)
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xlabel("Prediction horizon (minutes)", fontsize=12)
ax.set_ylabel(r"minADE$_6$ (m)", fontsize=12)
ax.tick_params(axis="both", which="major", labelsize=10)

ax.grid(color="#dddddd", linewidth=0.8, which="major", zorder=1)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

ax.legend(loc="upper left", fontsize=10, frameon=False, handlelength=2.2)

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight")
plt.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved -> {OUT}")

# Sanity check: the last point of each curve must equal that model's Table 8 row.
for label, _, _, _ in MODELS:
    print(f"  {label:17s} minADE_6 at 300 s = {ade[label][-1]:7.2f} m")
