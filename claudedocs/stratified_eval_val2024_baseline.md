# Stratified Evaluation — exp=`val2024_baseline`  model=`baseline_linear`

- ckpt: `/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/unitraj_ckpt/eval_baseline_linear_5min_v1/epoch=0-val/brier_fde=65.85.ckpt`
- val_data_path: `['/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/data/processed_ais_4hours_optimized/val']`
- samples: **10,336**  past_len=300  future_len=300

## Overall

- minADE6: **34.19 m**   minFDE6: **65.85 m**
- Brier-FDE: **65.85**
- Miss@2m: 95.7%   Miss@10m: 71.3%   Miss@20m: 49.9%

## Metrics by GROUND-TRUTH maneuver bucket

_If model minFDE ≈ baseline minFDE on `straight` but both blow up on turns, the model is riding the constant-velocity shortcut._

| GT bucket | Count | Share | minADE | minFDE | Miss@10m | Miss@20m | BL minFDE | model−BL |
|-----------|------:|------:|-------:|-------:|---------:|---------:|----------:|---------:|
| stationary | 1,572 | 15.2% | 3.75 | 6.07 | 14.8% | 2.9% | nan | +nan |
| straight | 1,410 | 13.6% | 27.85 | 50.84 | 75.3% | 48.1% | nan | +nan |
| mild_turn | 6,723 | 65.0% | 40.18 | 77.29 | 83.5% | 61.3% | nan | +nan |
| sharp_turn | 631 | 6.1% | 60.42 | 126.50 | 73.2% | 50.1% | nan | +nan |

## Mode collapse — predicted vs GT maneuver distribution

| Bucket | GT share | Predicted (top-mode) share |
|--------|---------:|---------------------------:|
| stationary | 15.2% | 14.7% |
| straight | 13.6% | 14.0% |
| mild_turn | 65.0% | 63.9% |
| sharp_turn | 6.1% | 7.4% |
| unknown | 0.0% | 0.0% |

**Of 7,354 GT-turn windows, the model's top mode was straight/stationary 775 times (10.5%).**

- Mean GT heading change: 11.8°   |   Mean predicted top-mode heading change: 20.8°

## Turn continuation — when the PAST is curved (ship mid-turn)

_MOVING windows only (past path ≥ 150 m AND displacement ≥ 80 m): 8,153 of 10,336 windows are moving; the rest are anchored/stopping and excluded (their signed-turn is AIS heading noise)._

_Past considered curved if |past signed turn| ≥ 15.0°. Prediction 'straight' if |future signed turn| < 10.0°. 'Continues' = predicted turn same direction as the past turn._

- Curved-past windows: **152** (1.5% of all)
- Ground truth: continues the turn 56 (36.8%) | goes straight 68 (44.7%)
- Model top-mode: continues the turn 0 (0.0%) | goes straight 152 (100.0%)

**Of 56 windows where the past is curved AND the ship keeps turning, the model's top mode goes straight 56 times (100.0%).**
- Mean signed turn (deg): past=+5.2  GT-future=+16.2  model-future=-0.0  (same sign as past = continuing)
