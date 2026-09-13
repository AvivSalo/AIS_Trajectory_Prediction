# Stratified Evaluation — exp=`val2024_gat_lstm`  model=`gat_lstm`

- ckpt: `/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/unitraj_ckpt/gat_lstm_5min_v1/epoch=20-val/brier_fde=659.19.ckpt`
- val_data_path: `['/home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/data/processed_ais_4hours_optimized/val']`
- samples: **10,336**  past_len=300  future_len=300

## Overall

- minADE6: **379.23 m**   minFDE6: **756.05 m**
- Brier-FDE: **756.05**
- Miss@2m: 100.0%   Miss@10m: 100.0%   Miss@20m: 99.5%
- Baseline (OLS) minFDE6: 65.85 m   → model−baseline minFDE: **+690.19 m**

## Metrics by GROUND-TRUTH maneuver bucket

_If model minFDE ≈ baseline minFDE on `straight` but both blow up on turns, the model is riding the constant-velocity shortcut._

| GT bucket | Count | Share | minADE | minFDE | Miss@10m | Miss@20m | BL minFDE | model−BL |
|-----------|------:|------:|-------:|-------:|---------:|---------:|----------:|---------:|
| stationary | 1,572 | 15.2% | 369.96 | 739.64 | 100.0% | 98.7% | 6.07 | +733.56 |
| straight | 1,410 | 13.6% | 391.40 | 778.47 | 100.0% | 99.7% | 50.84 | +727.63 |
| mild_turn | 6,723 | 65.0% | 381.91 | 761.06 | 100.0% | 99.8% | 77.29 | +683.78 |
| sharp_turn | 631 | 6.1% | 346.47 | 693.36 | 99.7% | 97.8% | 126.50 | +566.86 |

## Mode collapse — predicted vs GT maneuver distribution

| Bucket | GT share | Predicted (top-mode) share |
|--------|---------:|---------------------------:|
| stationary | 15.2% | 0.0% |
| straight | 13.6% | 0.1% |
| mild_turn | 65.0% | 53.6% |
| sharp_turn | 6.1% | 46.3% |
| unknown | 0.0% | 0.0% |

**Of 7,354 GT-turn windows, the model's top mode was straight/stationary 10 times (0.1%).**

- Mean GT heading change: 11.8°   |   Mean predicted top-mode heading change: 44.4°

## Turn continuation — when the PAST is curved (ship mid-turn)

_MOVING windows only (past path ≥ 150 m AND displacement ≥ 80 m): 8,153 of 10,336 windows are moving; the rest are anchored/stopping and excluded (their signed-turn is AIS heading noise)._

_Past considered curved if |past signed turn| ≥ 15.0°. Prediction 'straight' if |future signed turn| < 10.0°. 'Continues' = predicted turn same direction as the past turn._

- Curved-past windows: **152** (1.5% of all)
- Ground truth: continues the turn 56 (36.8%) | goes straight 68 (44.7%)
- Model top-mode: continues the turn 10 (6.6%) | goes straight 133 (87.5%)

**Of 56 windows where the past is curved AND the ship keeps turning, the model's top mode goes straight 46 times (82.1%).**
- Mean signed turn (deg): past=+5.2  GT-future=+16.2  model-future=+4.1  (same sign as past = continuing)
