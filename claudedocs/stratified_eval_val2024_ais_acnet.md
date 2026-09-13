# Stratified Evaluation — exp=`val2024_ais_acnet`  model=`ais_acnet`

- ckpt: `/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/unitraj_ckpt/ais_acnet_5min_v1/epoch=99-val/brier_fde=66.84.ckpt`
- val_data_path: `['/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/data/processed_ais_4hours_optimized/val']`
- samples: **10,336**  past_len=300  future_len=300

## Overall

- minADE6: **33.13 m**   minFDE6: **69.90 m**
- Brier-FDE: **69.90**
- Miss@2m: 96.2%   Miss@10m: 81.1%   Miss@20m: 67.6%
- Baseline (OLS) minFDE6: 65.85 m   → model−baseline minFDE: **+4.04 m**

## Metrics by GROUND-TRUTH maneuver bucket

_If model minFDE ≈ baseline minFDE on `straight` but both blow up on turns, the model is riding the constant-velocity shortcut._

| GT bucket | Count | Share | minADE | minFDE | Miss@10m | Miss@20m | BL minFDE | model−BL |
|-----------|------:|------:|-------:|-------:|---------:|---------:|----------:|---------:|
| stationary | 1,572 | 15.2% | 3.34 | 5.98 | 15.9% | 3.2% | 6.07 | -0.10 |
| straight | 1,410 | 13.6% | 29.14 | 58.76 | 87.0% | 69.4% | 50.84 | +7.91 |
| mild_turn | 6,723 | 65.0% | 39.90 | 83.49 | 94.4% | 81.8% | 77.29 | +6.20 |
| sharp_turn | 631 | 6.1% | 44.15 | 109.19 | 88.7% | 73.1% | 126.50 | -17.31 |

## Mode collapse — predicted vs GT maneuver distribution

| Bucket | GT share | Predicted (top-mode) share |
|--------|---------:|---------------------------:|
| stationary | 15.2% | 14.7% |
| straight | 13.6% | 8.6% |
| mild_turn | 65.0% | 63.3% |
| sharp_turn | 6.1% | 13.5% |
| unknown | 0.0% | 0.0% |

**Of 7,354 GT-turn windows, the model's top mode was straight/stationary 605 times (8.2%).**

- Mean GT heading change: 11.8°   |   Mean predicted top-mode heading change: 34.7°

## Turn continuation — when the PAST is curved (ship mid-turn)

_MOVING windows only (past path ≥ 150 m AND displacement ≥ 80 m): 8,153 of 10,336 windows are moving; the rest are anchored/stopping and excluded (their signed-turn is AIS heading noise)._

_Past considered curved if |past signed turn| ≥ 15.0°. Prediction 'straight' if |future signed turn| < 10.0°. 'Continues' = predicted turn same direction as the past turn._

- Curved-past windows: **152** (1.5% of all)
- Ground truth: continues the turn 56 (36.8%) | goes straight 68 (44.7%)
- Model top-mode: continues the turn 0 (0.0%) | goes straight 151 (99.3%)

**Of 56 windows where the past is curved AND the ship keeps turning, the model's top mode goes straight 56 times (100.0%).**
- Mean signed turn (deg): past=+5.2  GT-future=+16.2  model-future=+0.5  (same sign as past = continuing)
