# Stratified Evaluation — exp=`val2024_wayformer`  model=`wayformer_ais`

- ckpt: `/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj_ckpt/wayformer_5min_s150_200_v2/epoch=96-val/brier_fde=36.97.ckpt`
- val_data_path: `['/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/data/processed_ais_4hours_optimized/val']`
- samples: **10,336**  past_len=300  future_len=300

## Overall

- minADE6: **18.07 m**   minFDE6: **36.94 m**
- Brier-FDE: **37.56**
- Miss@2m: 92.7%   Miss@10m: 71.3%   Miss@20m: 44.2%
- Baseline (OLS) minFDE6: 65.85 m   → model−baseline minFDE: **-28.91 m**

## Metrics by GROUND-TRUTH maneuver bucket

_If model minFDE ≈ baseline minFDE on `straight` but both blow up on turns, the model is riding the constant-velocity shortcut._

| GT bucket | Count | Share | minADE | minFDE | Miss@10m | Miss@20m | BL minFDE | model−BL |
|-----------|------:|------:|-------:|-------:|---------:|---------:|----------:|---------:|
| stationary | 1,572 | 15.2% | 2.09 | 2.82 | 1.5% | 0.6% | 6.07 | -3.25 |
| straight | 1,410 | 13.6% | 14.09 | 27.06 | 76.0% | 41.3% | 50.84 | -23.78 |
| mild_turn | 6,723 | 65.0% | 21.05 | 42.35 | 86.3% | 54.6% | 77.29 | -34.94 |
| sharp_turn | 631 | 6.1% | 34.96 | 86.41 | 75.1% | 48.8% | 126.50 | -40.09 |

## Mode collapse — predicted vs GT maneuver distribution

| Bucket | GT share | Predicted (top-mode) share |
|--------|---------:|---------------------------:|
| stationary | 15.2% | 18.4% |
| straight | 13.6% | 8.5% |
| mild_turn | 65.0% | 63.1% |
| sharp_turn | 6.1% | 10.0% |
| unknown | 0.0% | 0.0% |

**Of 7,354 GT-turn windows, the model's top mode was straight/stationary 768 times (10.4%).**

- Mean GT heading change: 11.8°   |   Mean predicted top-mode heading change: 30.7°

## Turn continuation — when the PAST is curved (ship mid-turn)

_MOVING windows only (past path ≥ 150 m AND displacement ≥ 80 m): 8,153 of 10,336 windows are moving; the rest are anchored/stopping and excluded (their signed-turn is AIS heading noise)._

_Past considered curved if |past signed turn| ≥ 15.0°. Prediction 'straight' if |future signed turn| < 10.0°. 'Continues' = predicted turn same direction as the past turn._

- Curved-past windows: **152** (1.5% of all)
- Ground truth: continues the turn 56 (36.8%) | goes straight 68 (44.7%)
- Model top-mode: continues the turn 4 (2.6%) | goes straight 143 (94.1%)

**Of 56 windows where the past is curved AND the ship keeps turning, the model's top mode goes straight 56 times (100.0%).**
- Mean signed turn (deg): past=+5.2  GT-future=+16.2  model-future=+2.5  (same sign as past = continuing)
