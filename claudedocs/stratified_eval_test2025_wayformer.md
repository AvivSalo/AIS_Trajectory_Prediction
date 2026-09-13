# Stratified Evaluation — exp=`test2025_wayformer`  model=`wayformer_ais`

- ckpt: `/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj_ckpt/wayformer_5min_s150_200_v2/epoch=96-val/brier_fde=36.97.ckpt`
- val_data_path: `['/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/data/processed_ais_4hours_optimized/test']`
- samples: **11,891**  past_len=300  future_len=300

## Overall

- minADE6: **15.57 m**   minFDE6: **31.71 m**
- Brier-FDE: **32.31**
- Miss@2m: 91.6%   Miss@10m: 67.3%   Miss@20m: 41.6%
- Baseline (OLS) minFDE6: 51.43 m   → model−baseline minFDE: **-19.73 m**

## Metrics by GROUND-TRUTH maneuver bucket

_If model minFDE ≈ baseline minFDE on `straight` but both blow up on turns, the model is riding the constant-velocity shortcut._

| GT bucket | Count | Share | minADE | minFDE | Miss@10m | Miss@20m | BL minFDE | model−BL |
|-----------|------:|------:|-------:|-------:|---------:|---------:|----------:|---------:|
| stationary | 2,203 | 18.5% | 2.12 | 2.84 | 1.2% | 0.4% | 6.05 | -3.20 |
| straight | 1,621 | 13.6% | 11.91 | 22.25 | 72.2% | 34.5% | 35.26 | -13.00 |
| mild_turn | 7,241 | 60.9% | 19.11 | 38.44 | 86.0% | 55.2% | 61.31 | -22.87 |
| sharp_turn | 826 | 6.9% | 27.51 | 68.25 | 70.7% | 46.4% | 117.71 | -49.46 |

## Mode collapse — predicted vs GT maneuver distribution

| Bucket | GT share | Predicted (top-mode) share |
|--------|---------:|---------------------------:|
| stationary | 18.5% | 22.4% |
| straight | 13.6% | 8.4% |
| mild_turn | 60.9% | 59.1% |
| sharp_turn | 6.9% | 10.1% |
| unknown | 0.0% | 0.0% |

**Of 8,067 GT-turn windows, the model's top mode was straight/stationary 885 times (11.0%).**

- Mean GT heading change: 12.7°   |   Mean predicted top-mode heading change: 33.7°

## Turn continuation — when the PAST is curved (ship mid-turn)

_MOVING windows only (past path ≥ 150 m AND displacement ≥ 80 m): 8,705 of 11,891 windows are moving; the rest are anchored/stopping and excluded (their signed-turn is AIS heading noise)._

_Past considered curved if |past signed turn| ≥ 15.0°. Prediction 'straight' if |future signed turn| < 10.0°. 'Continues' = predicted turn same direction as the past turn._

- Curved-past windows: **156** (1.3% of all)
- Ground truth: continues the turn 72 (46.2%) | goes straight 73 (46.8%)
- Model top-mode: continues the turn 1 (0.6%) | goes straight 153 (98.1%)

**Of 72 windows where the past is curved AND the ship keeps turning, the model's top mode goes straight 71 times (98.6%).**
- Mean signed turn (deg): past=-3.3  GT-future=-2.3  model-future=+1.0  (same sign as past = continuing)
