# Stratified Evaluation — exp=`test2025_traisformer`  model=`traisformer`

- ckpt: `/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/unitraj_ckpt/traisformer_5min_v3/epoch=32-val/brier_fde=133.97.ckpt`
- val_data_path: `['/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/data/processed_ais_4hours_optimized/test']`
- samples: **11,891**  past_len=300  future_len=300

## Overall

- minADE6: **100.43 m**   minFDE6: **186.49 m**
- Brier-FDE: **187.19**
- Miss@2m: 99.8%   Miss@10m: 81.9%   Miss@20m: 73.5%
- Baseline (OLS) minFDE6: 51.43 m   → model−baseline minFDE: **+135.06 m**

## Metrics by GROUND-TRUTH maneuver bucket

_If model minFDE ≈ baseline minFDE on `straight` but both blow up on turns, the model is riding the constant-velocity shortcut._

| GT bucket | Count | Share | minADE | minFDE | Miss@10m | Miss@20m | BL minFDE | model−BL |
|-----------|------:|------:|-------:|-------:|---------:|---------:|----------:|---------:|
| stationary | 2,203 | 18.5% | 7.87 | 8.12 | 19.6% | 0.2% | 6.05 | +2.08 |
| straight | 1,621 | 13.6% | 99.38 | 181.13 | 90.9% | 79.3% | 35.26 | +145.88 |
| mild_turn | 7,241 | 60.9% | 126.99 | 237.20 | 98.0% | 94.1% | 61.31 | +175.89 |
| sharp_turn | 826 | 6.9% | 116.50 | 228.27 | 88.7% | 76.6% | 117.71 | +110.56 |

## Mode collapse — predicted vs GT maneuver distribution

| Bucket | GT share | Predicted (top-mode) share |
|--------|---------:|---------------------------:|
| stationary | 18.5% | 21.2% |
| straight | 13.6% | 2.4% |
| mild_turn | 60.9% | 59.1% |
| sharp_turn | 6.9% | 17.3% |
| unknown | 0.0% | 0.0% |

**Of 8,067 GT-turn windows, the model's top mode was straight/stationary 468 times (5.8%).**

- Mean GT heading change: 12.7°   |   Mean predicted top-mode heading change: 14.4°

## Turn continuation — when the PAST is curved (ship mid-turn)

_MOVING windows only (past path ≥ 150 m AND displacement ≥ 80 m): 8,705 of 11,891 windows are moving; the rest are anchored/stopping and excluded (their signed-turn is AIS heading noise)._

_Past considered curved if |past signed turn| ≥ 15.0°. Prediction 'straight' if |future signed turn| < 10.0°. 'Continues' = predicted turn same direction as the past turn._

- Curved-past windows: **156** (1.3% of all)
- Ground truth: continues the turn 72 (46.2%) | goes straight 73 (46.8%)
- Model top-mode: continues the turn 26 (16.7%) | goes straight 97 (62.2%)

**Of 72 windows where the past is curved AND the ship keeps turning, the model's top mode goes straight 41 times (56.9%).**
- Mean signed turn (deg): past=-3.3  GT-future=-2.3  model-future=-5.3  (same sign as past = continuing)
