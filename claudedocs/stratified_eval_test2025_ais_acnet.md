# Stratified Evaluation — exp=`test2025_ais_acnet`  model=`ais_acnet`

- ckpt: `/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/unitraj_ckpt/ais_acnet_5min_v1/epoch=99-val/brier_fde=66.84.ckpt`
- val_data_path: `['/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/data/processed_ais_4hours_optimized/test']`
- samples: **11,891**  past_len=300  future_len=300

## Overall

- minADE6: **27.03 m**   minFDE6: **56.76 m**
- Brier-FDE: **56.76**
- Miss@2m: 96.4%   Miss@10m: 78.8%   Miss@20m: 65.7%
- Baseline (OLS) minFDE6: 51.43 m   → model−baseline minFDE: **+5.33 m**

## Metrics by GROUND-TRUTH maneuver bucket

_If model minFDE ≈ baseline minFDE on `straight` but both blow up on turns, the model is riding the constant-velocity shortcut._

| GT bucket | Count | Share | minADE | minFDE | Miss@10m | Miss@20m | BL minFDE | model−BL |
|-----------|------:|------:|-------:|-------:|---------:|---------:|----------:|---------:|
| stationary | 2,203 | 18.5% | 3.44 | 6.18 | 18.0% | 2.3% | 6.05 | +0.13 |
| straight | 1,621 | 13.6% | 24.31 | 48.52 | 84.6% | 67.7% | 35.26 | +13.26 |
| mild_turn | 7,241 | 60.9% | 33.76 | 70.37 | 95.3% | 84.4% | 61.31 | +9.07 |
| sharp_turn | 826 | 6.9% | 36.26 | 88.53 | 85.2% | 66.8% | 117.71 | -29.18 |

## Mode collapse — predicted vs GT maneuver distribution

| Bucket | GT share | Predicted (top-mode) share |
|--------|---------:|---------------------------:|
| stationary | 18.5% | 17.6% |
| straight | 13.6% | 7.7% |
| mild_turn | 60.9% | 59.6% |
| sharp_turn | 6.9% | 15.1% |
| unknown | 0.0% | 0.0% |

**Of 8,067 GT-turn windows, the model's top mode was straight/stationary 669 times (8.3%).**

- Mean GT heading change: 12.7°   |   Mean predicted top-mode heading change: 39.8°

## Turn continuation — when the PAST is curved (ship mid-turn)

_MOVING windows only (past path ≥ 150 m AND displacement ≥ 80 m): 8,705 of 11,891 windows are moving; the rest are anchored/stopping and excluded (their signed-turn is AIS heading noise)._

_Past considered curved if |past signed turn| ≥ 15.0°. Prediction 'straight' if |future signed turn| < 10.0°. 'Continues' = predicted turn same direction as the past turn._

- Curved-past windows: **156** (1.3% of all)
- Ground truth: continues the turn 72 (46.2%) | goes straight 73 (46.8%)
- Model top-mode: continues the turn 0 (0.0%) | goes straight 156 (100.0%)

**Of 72 windows where the past is curved AND the ship keeps turning, the model's top mode goes straight 72 times (100.0%).**
- Mean signed turn (deg): past=-3.3  GT-future=-2.3  model-future=+0.9  (same sign as past = continuing)
