# Training Wayformer with AIS Ship Trajectory Data

This guide explains how to prepare and use AIS (Automatic Identification System) ship trajectory data with the UniTraj Wayformer model.

## Setup

1. Make sure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

2. Set up Weights & Biases for training monitoring:
```bash
wandb login YOUR_API_KEY
```

## Data Preparation Pipeline

### 1. Data Format Requirements
Your AIS data should be in CSV format with the following columns:
- `time`: Timestamp
- `own_latitude`, `own_longitude`: Ship position
- `own_sog`: Speed Over Ground
- `own_cog`: Course Over Ground
- `host_name`: Vessel identifier

For multi-agent scenarios (when there are nearby ships), target ship data should have columns:
- `target_latitude`, `target_longitude`
- `target_sog`, `target_cog`
- `target_target_id`: Target vessel identifier

### 2. Data Preprocessing

1. Place your AIS CSV files in `data/ais_data_from_influx_csv/`

2. Run the preprocessor:
```bash
python unitraj/tools/ais_data_preprocessor.py
```

This script will:
- Convert AIS data into Wayformer's expected format
- Split data into train (80%) and validation (20%) sets
- Create required dataset summaries
- Save processed data in:
  - `data/processed_ais_data/train/`
  - `data/processed_ais_data/val/`

The preprocessor performs these steps:
1. Converts timestamps to seconds from start
2. Extracts features for both own ship and target ships
3. Creates scene-based data files in pickle format
4. Generates dataset summaries required by the dataloader
5. Automatically splits data into training and validation sets

## Model Configuration

The Wayformer configuration for AIS data is in `unitraj/configs/method/wayformer_ais.yaml`:

Key settings:
- `num_agent_feature: 5` (timestamp, lat, lon, speed, heading)
- `use_map_lanes: False` (no map features for maritime data)
- `max_num_agents: 16` (adjustable based on your scenarios)
- `num_map_feature: 0` (no map features for maritime domain)

## Training

Start training with:

```bash
python unitraj/train.py method=wayformer_ais
```

The training process will:
- Load preprocessed AIS data
- Initialize the Wayformer model
- Log training progress to Weights & Biases
- Save checkpoints in `unitraj_ckpt/ais_wayformer/`

Monitor training progress:
1. Through Weights & Biases web interface
2. Through local TensorBoard:
```bash
tensorboard --logdir lightning_logs/
```

## Model Output

The Wayformer model predicts:
- Future trajectories for all vessels in the scene
- Multiple prediction modes for each vessel (default: 6 modes)
- Position (latitude, longitude) and velocity (speed, heading)

## Tips and Troubleshooting

1. Data Quality:
   - Ensure consistent sampling rate in AIS data
   - Check for missing values in CSV files
   - Verify target ship data format if multi-agent scenarios are present

2. Training:
   - Start with a small subset to verify the pipeline
   - Monitor the validation metrics for overfitting
   - Adjust batch size based on your GPU memory
   - Use `wandb` dashboard to monitor training progress

3. Common Issues:
   - If data loader shows 0 samples, verify dataset_summary.pkl was created
   - If out of memory, reduce batch_size in configuration
   - If no target ships detected, verify CSV column names match expected format

## Directory Structure After Processing
```
UniTraj/
├── data/
│   ├── ais_data_from_influx_csv/    # Raw AIS CSV files
│   └── processed_ais_data/
│       ├── train/                    # Training data + dataset_summary.pkl
│       └── val/                      # Validation data + dataset_summary.pkl
├── unitraj/
│   ├── configs/
│   │   └── method/
│   │       └── wayformer_ais.yaml   # AIS-specific configuration
│   └── tools/
│       └── ais_data_preprocessor.py  # Data preprocessing script
└── unitraj_ckpt/                     # Trained model checkpoints
    └── ais_wayformer/
```

## Evaluation

After training, evaluate the model with proper Hydra syntax:

### Basic Evaluation
```bash
conda activate unitraj
python unitraj/evaluation.py ckpt_path="unitraj_ckpt/ais_wayformer/epoch\=147-val/brier_fde\=637.07.ckpt"
```

### Advanced Evaluation with Custom Data
```bash
conda activate unitraj
HYDRA_FULL_ERROR=1 python unitraj/evaluation.py \
  ckpt_path="unitraj_ckpt/ais_wayformer/epoch\=147-val/brier_fde\=637.07.ckpt" \
  val_data_path='["/home/aviv/Projects/UniTraj/data/ais_test_sample"]'
```

### Debug Mode (CPU only, faster for testing)
```bash
conda activate unitraj
python unitraj/evaluation.py \
  ckpt_path="unitraj_ckpt/ais_wayformer/epoch\=147-val/brier_fde\=637.07.ckpt" \
  debug=True
```

**Important Notes:**
- **Escape equals signs**: Use `\=` in checkpoint paths (e.g., `epoch\=147`)
- **List syntax**: Use `'["/path"]'` for data paths, not just `"/path"`
- **CSV data**: The AIS dataset expects CSV files, not PKL files
- **HYDRA_FULL_ERROR=1**: Enables detailed error messages for debugging

### Evaluation Output
The evaluation will generate:
- Performance metrics (FDE, ADE, etc.) printed to console
- Interactive HTML visualizations in `evaluation_visualizations/` directory
- WandB logging (unless debug=True)

Replace the checkpoint path with your actual best checkpoint from training.
