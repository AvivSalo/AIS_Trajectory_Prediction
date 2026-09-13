# Stratified Evaluation — exp=`test2025_baseline`  model=`baseline_linear`

- ckpt: `/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/unitraj_ckpt/eval_baseline_linear_5min_v1/epoch=0-val/brier_fde=65.85.ckpt`
- val_data_path: `['/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/data/processed_ais_4hours_optimized/test']`
- samples: **11,891**  past_len=300  future_len=300

## Overall

- minADE6: **26.70 m**   minFDE6: **51.43 m**
- Brier-FDE: **51.43**
- Miss@2m: 95.3%   Miss@10m: 68.1%   Miss@20m: 45.8%

## Metrics by GROUND-TRUTH maneuver bucket

_If model minFDE ≈ baseline minFDE on `straight` but both blow up on turns, the model is riding the constant-velocity shortcut._

| GT bucket | Count | Share | minADE | minFDE | Miss@10m | Miss@20m | BL minFDE | model−BL |
|-----------|------:|------:|-------:|-------:|---------:|---------:|----------:|---------:|
| stationary | 2,203 | 18.5% | 3.73 | 6.05 | 15.1% | 2.7% | nan | +nan |
| straight | 1,621 | 13.6% | 19.62 | 35.26 | 70.3% | 41.0% | nan | +nan |
| mild_turn | 7,241 | 60.9% | 32.05 | 61.31 | 83.3% | 59.5% | nan | +nan |
| sharp_turn | 826 | 6.9% | 55.04 | 117.71 | 71.2% | 49.5% | nan | +nan |

## Mode collapse — predicted vs GT maneuver distribution

| Bucket | GT share | Predicted (top-mode) share |
|--------|---------:|---------------------------:|
| stationary | 18.5% | 17.9% |
| straight | 13.6% | 13.4% |
| mild_turn | 60.9% | 60.5% |
| sharp_turn | 6.9% | 8.2% |
| unknown | 0.0% | 0.0% |

**Of 8,067 GT-turn windows, the model's top mode was straight/stationary 846 times (10.5%).**

- Mean GT heading change: 12.7°   |   Mean predicted top-mode heading change: 24.4°

## Turn continuation — when the PAST is curved (ship mid-turn)

_MOVING windows only (past path ≥ 150 m AND displacement ≥ 80 m): 8,705 of 11,891 windows are moving; the rest are anchored/stopping and excluded (their signed-turn is AIS heading noise)._

_Past considered curved if |past signed turn| ≥ 15.0°. Prediction 'straight' if |future signed turn| < 10.0°. 'Continues' = predicted turn same direction as the past turn._

- Curved-past windows: **156** (1.3% of all)
- Ground truth: continues the turn 72 (46.2%) | goes straight 73 (46.8%)
- Model top-mode: continues the turn 0 (0.0%) | goes straight 156 (100.0%)

**Of 72 windows where the past is curved AND the ship keeps turning, the model's top mode goes straight 72 times (100.0%).**
- Mean signed turn (deg): past=-3.3  GT-future=-2.3  model-future=-0.0  (same sign as past = continuing)
