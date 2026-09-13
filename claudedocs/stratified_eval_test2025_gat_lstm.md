# Stratified Evaluation — exp=`test2025_gat_lstm`  model=`gat_lstm`

- ckpt: `/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/unitraj_ckpt/gat_lstm_5min_v1/epoch=20-val/brier_fde=659.19.ckpt`
- val_data_path: `['/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/data/processed_ais_4hours_optimized/test']`
- samples: **11,891**  past_len=300  future_len=300

## Overall

- minADE6: **409.42 m**   minFDE6: **819.12 m**
- Brier-FDE: **819.12**
- Miss@2m: 100.0%   Miss@10m: 100.0%   Miss@20m: 99.7%
- Baseline (OLS) minFDE6: 51.43 m   → model−baseline minFDE: **+767.69 m**

## Metrics by GROUND-TRUTH maneuver bucket

_If model minFDE ≈ baseline minFDE on `straight` but both blow up on turns, the model is riding the constant-velocity shortcut._

| GT bucket | Count | Share | minADE | minFDE | Miss@10m | Miss@20m | BL minFDE | model−BL |
|-----------|------:|------:|-------:|-------:|---------:|---------:|----------:|---------:|
| stationary | 2,203 | 18.5% | 323.88 | 648.57 | 100.0% | 99.7% | 6.05 | +642.52 |
| straight | 1,621 | 13.6% | 430.34 | 859.06 | 99.9% | 99.8% | 35.26 | +823.80 |
| mild_turn | 7,241 | 60.9% | 433.07 | 866.01 | 100.0% | 99.8% | 61.31 | +804.71 |
| sharp_turn | 826 | 6.9% | 389.25 | 784.59 | 99.9% | 98.9% | 117.71 | +666.88 |

## Mode collapse — predicted vs GT maneuver distribution

| Bucket | GT share | Predicted (top-mode) share |
|--------|---------:|---------------------------:|
| stationary | 18.5% | 0.0% |
| straight | 13.6% | 0.2% |
| mild_turn | 60.9% | 50.0% |
| sharp_turn | 6.9% | 49.8% |
| unknown | 0.0% | 0.0% |

**Of 8,067 GT-turn windows, the model's top mode was straight/stationary 19 times (0.2%).**

- Mean GT heading change: 12.7°   |   Mean predicted top-mode heading change: 47.8°

## Turn continuation — when the PAST is curved (ship mid-turn)

_MOVING windows only (past path ≥ 150 m AND displacement ≥ 80 m): 8,705 of 11,891 windows are moving; the rest are anchored/stopping and excluded (their signed-turn is AIS heading noise)._

_Past considered curved if |past signed turn| ≥ 15.0°. Prediction 'straight' if |future signed turn| < 10.0°. 'Continues' = predicted turn same direction as the past turn._

- Curved-past windows: **156** (1.3% of all)
- Ground truth: continues the turn 72 (46.2%) | goes straight 73 (46.8%)
- Model top-mode: continues the turn 10 (6.4%) | goes straight 136 (87.2%)

**Of 72 windows where the past is curved AND the ship keeps turning, the model's top mode goes straight 62 times (86.1%).**
- Mean signed turn (deg): past=-3.3  GT-future=-2.3  model-future=+4.8  (same sign as past = continuing)
