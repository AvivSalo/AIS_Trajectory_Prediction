# Stratified Evaluation — exp=`val2024_traisformer`  model=`traisformer`

- ckpt: `/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/unitraj_ckpt/traisformer_5min_v3/epoch=32-val/brier_fde=133.97.ckpt`
- val_data_path: `['/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/data/processed_ais_4hours_optimized/val']`
- samples: **10,336**  past_len=300  future_len=300

## Overall

- minADE6: **101.19 m**   minFDE6: **188.97 m**
- Brier-FDE: **189.66**
- Miss@2m: 99.6%   Miss@10m: 85.0%   Miss@20m: 77.6%
- Baseline (OLS) minFDE6: 65.85 m   → model−baseline minFDE: **+123.11 m**

## Metrics by GROUND-TRUTH maneuver bucket

_If model minFDE ≈ baseline minFDE on `straight` but both blow up on turns, the model is riding the constant-velocity shortcut._

| GT bucket | Count | Share | minADE | minFDE | Miss@10m | Miss@20m | BL minFDE | model−BL |
|-----------|------:|------:|-------:|-------:|---------:|---------:|----------:|---------:|
| stationary | 1,572 | 15.2% | 7.92 | 8.15 | 17.5% | 0.2% | 6.07 | +2.08 |
| straight | 1,410 | 13.6% | 96.99 | 181.36 | 94.2% | 85.0% | 50.84 | +130.52 |
| mild_turn | 6,723 | 65.0% | 121.85 | 229.15 | 98.1% | 93.9% | 77.29 | +151.86 |
| sharp_turn | 631 | 6.1% | 122.79 | 228.31 | 92.9% | 79.4% | 126.50 | +101.82 |

## Mode collapse — predicted vs GT maneuver distribution

| Bucket | GT share | Predicted (top-mode) share |
|--------|---------:|---------------------------:|
| stationary | 15.2% | 17.3% |
| straight | 13.6% | 1.6% |
| mild_turn | 65.0% | 63.0% |
| sharp_turn | 6.1% | 18.1% |
| unknown | 0.0% | 0.0% |

**Of 7,354 GT-turn windows, the model's top mode was straight/stationary 313 times (4.3%).**

- Mean GT heading change: 11.8°   |   Mean predicted top-mode heading change: 14.7°

## Turn continuation — when the PAST is curved (ship mid-turn)

_MOVING windows only (past path ≥ 150 m AND displacement ≥ 80 m): 8,153 of 10,336 windows are moving; the rest are anchored/stopping and excluded (their signed-turn is AIS heading noise)._

_Past considered curved if |past signed turn| ≥ 15.0°. Prediction 'straight' if |future signed turn| < 10.0°. 'Continues' = predicted turn same direction as the past turn._

- Curved-past windows: **152** (1.5% of all)
- Ground truth: continues the turn 56 (36.8%) | goes straight 68 (44.7%)
- Model top-mode: continues the turn 30 (19.7%) | goes straight 87 (57.2%)

**Of 56 windows where the past is curved AND the ship keeps turning, the model's top mode goes straight 25 times (44.6%).**
- Mean signed turn (deg): past=+5.2  GT-future=+16.2  model-future=+2.9  (same sign as past = continuing)
