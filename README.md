<div style="text-align: center;">
  <img src="https://github.com/vita-epfl/UniTraj/blob/main/docs/assets/unitraj.gif" alt="Demo" width="300">
</div>

# UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unitraj-a-unified-framework-for-scalable/trajectory-prediction-on-nuscenes)](https://paperswithcode.com/sota/trajectory-prediction-on-nuscenes?p=unitraj-a-unified-framework-for-scalable)

[**Website**](https://vita-epfl.github.io/UniTraj/) |
[**Paper**](https://arxiv.org/pdf/2403.15098.pdf)

## 🌟 Overview

**UniTraj** is a comprehensive, unified framework for scalable trajectory prediction that supports both **terrestrial vehicles** and **marine vessels**. The framework enables researchers and practitioners to:

- 🚗 **Train and evaluate** trajectory prediction models on real-world datasets (Waymo, nuPlan, nuScenes, Argoverse2)
- 🚢 **Marine Trajectory Prediction** with AIS (Automatic Identification System) data for vessel trajectory forecasting
- 🔧 **Unified Pipeline** for different domains with consistent APIs and configurations
- 📊 **Advanced Evaluation** with comprehensive metrics and visualization tools

![system](docs/assets/framework.png)

### Key Features

💡 **Multi-Domain Support**: From autonomous vehicles to marine vessels, UniTraj handles diverse trajectory prediction tasks

🔥 **Powered by Modern Tools**: [Hydra](https://hydra.cc/docs/intro/), [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), and [WandB](https://wandb.ai/site) for easy configuration, training, and logging

🎯 **State-of-the-Art Models**: Includes Wayformer, AutoBot, MTR, and other leading trajectory prediction architectures

🌊 **Marine Innovation**: First unified framework to support both land and sea trajectory prediction with AIS data integration

![system](docs/assets/support.png)

## 🚢 Marine Trajectory Prediction (NEW!)

UniTraj now supports **maritime vessel trajectory prediction** using AIS data! This groundbreaking feature enables:

- **Real-world vessel tracking** and prediction in marine environments
- **Multi-vessel interaction modeling** for collision avoidance and navigation
- **Wayformer adaptation** for marine-specific trajectory patterns
- **CSV-based AIS data processing** with automatic preprocessing pipeline

### Quick Start for Marine Prediction

```bash
# 1. Prepare your AIS CSV data
mkdir -p data/ais_data_from_influx_csv
# Place your AIS CSV files here

# 2. Train on marine data
cd unitraj
python train.py method=wayformer_ais

# 3. Evaluate marine model
python evaluation.py ckpt_path=path/to/checkpoint.ckpt
```

📖 **Detailed Guide**: See [README_AIS_TRAJ.md](README_AIS_TRAJ.md) for complete AIS integration documentation.

## 📰 News & Updates

### Jan. 2025
- 🚢 **NEW**: Marine vessel trajectory prediction with AIS data integration!
- 🔧 **Enhanced**: Unified pipeline supporting both terrestrial and marine domains

### Dec. 2024
- 🔥 UniTraj now supports data selection with [TAROT](https://github.com/vita-epfl/TAROT)! Try to use less data for improved performance.

### Nov. 2024
- Adding AV2 evaluation tools.
- Using h5 format data cache for faster loader.

### Sep. 2024
- New website is live! Check it out [here](https://vita-epfl.github.io/UniTraj/).

### Jul. 2024
- 🚀 Accepted to ECCV 2024! 

### Mar. 2024
- 🌐 Launched the official [UniTraj Website](https://www.epfl.ch/labs/vita/research/prediction/vehicle-trajectory-prediction/unitraj-a-unified-framework-for-scalable-vehicle-trajectory-prediction/).
- 📝 Published UniTraj in [Arxiv](https://arxiv.org/pdf/2403.15098.pdf).

## 🛠 Installation & Quick Start

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended)
- Conda or virtualenv

### Step 1: Environment Setup

```bash
# Create and activate conda environment
conda create -n unitraj python=3.9
conda activate unitraj
```

### Step 2: Install Dependencies

```bash
# Clone the repository
git clone https://github.com/vita-epfl/UniTraj.git
cd UniTraj

# Install ScenarioNet (for terrestrial vehicle data)
# Follow: https://scenarionet.readthedocs.io/en/latest/install.html

# Install UniTraj
pip install -r requirements.txt
python setup.py develop
```

### Step 3: Verify Installation

```bash
cd unitraj

# Test with sample data (terrestrial vehicles)
python train.py method=autobot

# Test with marine data (if you have AIS CSV files)
python train.py method=wayformer_ais debug=True
```

## 🏗 Code Structure

UniTraj follows a modular architecture with three main components:

```
unitraj/
├── configs/                    # Configuration files
│   ├── config.yaml            # Main configuration
│   └── method/                # Model-specific configs
│       ├── autobot.yaml       # AutoBot configuration
│       ├── MTR.yaml           # MTR configuration
│       ├── wayformer.yaml     # Wayformer configuration
│       └── wayformer_ais.yaml # Marine Wayformer config
├── datasets/                   # Dataset implementations
│   ├── base_dataset.py        # Base dataset class
│   ├── autobot_dataset.py     # AutoBot dataset
│   ├── wayformer_dataset.py   # Wayformer dataset
│   ├── MTR_dataset.py         # MTR dataset
│   └── ais_dataset.py         # AIS marine dataset
├── models/                     # Model implementations
│   ├── autobot/               # AutoBot model
│   ├── mtr/                   # MTR model
│   ├── wayformer/             # Wayformer model
│   └── base_model/            # Base model classes
├── tools/                      # Utility scripts
│   ├── ais_data_preprocessor.py # AIS data preprocessing
│   └── convert_ais_to_scenarios.py # AIS format conversion
└── utils/                      # Helper utilities
```

### Architecture Principles

- **Inheritance-based Design**: Each model inherits from base classes for consistency
- **Hydra Configuration**: Hierarchical configuration management
- **Domain Agnostic**: Same pipeline works for different domains (land/sea)
- **Extensible**: Easy to add new models, datasets, and domains

## 🚀 Usage Guide

### 1. Data Preparation

#### For Terrestrial Vehicles
UniTraj uses [ScenarioNet](https://github.com/metadriverse/scenarionet) format. Process your data with ScenarioNet first:

```bash
# Follow ScenarioNet documentation for your specific dataset
# (Waymo, nuScenes, Argoverse2, nuPlan)
```

#### For Marine Vessels
Prepare AIS data in CSV format:

```bash
# Place AIS CSV files in data/ais_data_from_influx_csv/
# Required columns: time, own_latitude, own_longitude, own_sog, own_cog, host_name
# Optional: target_latitude, target_longitude, target_sog, target_cog, target_target_id

# Preprocess AIS data
cd unitraj
python tools/ais_data_preprocessor.py
```

### 2. Configuration

UniTraj uses [Hydra](https://hydra.cc/docs/intro/) for configuration management:

- **Main Config**: `unitraj/configs/config.yaml` - Universal settings
- **Method Configs**: `unitraj/configs/method/*.yaml` - Model-specific settings

#### Example: Configure for Marine Training
```yaml
# In config.yaml
train_data_path: ["data/ais_data_from_influx_csv"]
method: wayformer_ais

# In method/wayformer_ais.yaml
use_map_lanes: false  # No road maps in marine environment
use_map_image: false  # No road images
max_num_agents: 16    # Max vessels to track
```

### 3. Training

#### Terrestrial Vehicle Training
```bash
cd unitraj

# Train AutoBot on nuScenes/Waymo
python train.py method=autobot

# Train Wayformer with specific config
python train.py method=wayformer train_data_path=["path/to/data"]

# Train with debugging (small dataset)
python train.py method=autobot debug=True
```

#### Marine Vessel Training
```bash
cd unitraj

# Train on AIS data
python train.py method=wayformer_ais

# Quick test with small dataset
python train.py method=wayformer_ais debug=True method.max_epochs=5

# Full training with custom settings
python train.py method=wayformer_ais method.max_epochs=150 method.train_batch_size=32
```

#### Training Monitoring
- **WandB Dashboard**: Automatic logging to Weights & Biases
- **Local Logs**: Check `lightning_logs/` directory
- **Checkpoints**: Saved in `unitraj_ckpt/[experiment_name]/`

### 4. Evaluation

#### Basic Evaluation
```bash
cd unitraj

# Evaluate with checkpoint path
python evaluation.py ckpt_path=path/to/checkpoint.ckpt

# Evaluate marine model
python evaluation.py ckpt_path=unitraj_ckpt/ais_wayformer/best.ckpt
```

#### Advanced Evaluation Options
```bash
# Evaluate with official metrics (Waymo/nuScenes)
python evaluation.py eval_waymo=True eval_nuscenes=True

# Evaluate on specific validation set
python evaluation.py val_data_path=["path/to/val/data"]

# Debug evaluation (CPU only)
python evaluation.py debug=True
```

### 5. Data Analysis

```bash
cd unitraj

# Analyze dataset statistics
python data_analysis.py

# Analyze specific dataset
python data_analysis.py data_path=["path/to/data"]
```

## 📊 Supported Models & Datasets

### Models
- **AutoBot**: Autoregressive trajectory prediction
- **Wayformer**: Transformer-based multi-modal prediction  
- **MTR**: Motion Transformer for trajectory forecasting
- **Wayformer-AIS**: Marine-adapted Wayformer for vessels

### Datasets

#### Terrestrial Vehicle Datasets
- **Waymo Open Dataset**: Large-scale autonomous driving data
- **nuScenes**: Multi-modal autonomous driving dataset
- **Argoverse2**: High-definition map trajectory data
- **nuPlan**: Planning-focused autonomous driving dataset

#### Marine Vessel Datasets
- **AIS Data**: Automatic Identification System vessel tracking data
- **CSV Format**: Direct support for maritime CSV data files
- **Real-time Streams**: Compatible with live AIS data feeds

## 🔧 Advanced Features

### Custom Model Implementation

1. **Create Model Config**:
```yaml
# configs/method/my_model.yaml
model_name: my_model
# ... model-specific parameters
```

2. **Implement Dataset** (optional):
```python
# datasets/my_dataset.py
from .base_dataset import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, config, is_validation=False):
        super().__init__(config, is_validation)
        # Custom initialization
```

3. **Implement Model**:
```python
# models/my_model/my_model.py
import pytorch_lightning as pl

class MyModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Model implementation
```

4. **Register Components**:
```python
# Update datasets/__init__.py and models/__init__.py
__all__ = {
    # ... existing models
    'my_model': MyModel,
}
```

### Multi-Domain Training

Train models that work across different domains:

```bash
# Train on both terrestrial and marine data
python train.py method=wayformer train_data_path=["terrestrial/data", "marine/data"]
```

### Hyperparameter Tuning

```bash
# Use Hydra's multirun for hyperparameter search
python train.py -m method=wayformer method.learning_rate=0.001,0.0001 method.batch_size=32,64
```

## 🤝 Contributing to UniTraj

We welcome contributions! Here's how to get started:

### Adding New Models
1. Follow the structure in `models/` directory
2. Inherit from base classes for consistency  
3. Add configuration files in `configs/method/`
4. Update registry in `__init__.py` files

### Adding New Domains
1. Create domain-specific dataset in `datasets/`
2. Adapt existing models or create new ones
3. Add preprocessing tools in `tools/`
4. Document usage patterns

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to all public methods
- Include unit tests for new features
- Update documentation and examples

## 📈 Performance & Benchmarks

### Terrestrial Vehicle Results
- **nuScenes**: State-of-the-art performance on trajectory prediction
- **Waymo**: Competitive results on large-scale dataset
- **Argoverse2**: Strong performance on HD map scenarios

### Marine Vessel Results
- **AIS Validation**: Successful trajectory prediction on real vessel data
- **Multi-vessel Scenarios**: Effective handling of vessel interactions
- **Real-time Capability**: Suitable for live navigation systems

## 🔍 Dataset Structure Reference

### Standard UniTraj Format

#### Core Trajectory Data
- **obj_trajs**: Historical trajectories `[N_agents, T_hist, Features]`
  - `[0:3]` position (x, y, z)  
  - `[3:6]` size (l, w, h)
  - `[6:11]` type_onehot
  - `[11:33]` time_onehot
  - `[33:35]` heading_encoding
  - `[35:37]` velocity (vx, vy)
  - `[37:39]` acceleration (ax, ay)
- **obj_trajs_mask**: Valid mask for trajectories
- **track_index_to_predict**: Training sample indices

#### Map Information (Terrestrial Only)
- **map_polylines**: Road network polylines
- **map_polylines_mask**: Valid mask for map data
- **map_center**: Map coordinate center

#### Ground Truth & Metadata
- **center_gt_trajs**: Ground truth future trajectories
- **center_gt_trajs_mask**: Valid mask for ground truth
- **dataset_name**: Dataset identifier ("waymo", "nuscenes", "ais")
- **kalman_difficulty**: Prediction difficulty level
- **trajectory_type**: Trajectory classification

### AIS-Specific Format

#### Marine Vessel Data
- **CSV Columns**: `time, own_latitude, own_longitude, host_name, own_sog, own_cog`
- **Multi-vessel**: `target_latitude, target_longitude, target_sog, target_cog, target_target_id`
- **Coordinate System**: Relative positioning from ego vessel origin
- **Features**: Position, velocity, heading for marine navigation

## 🐛 Troubleshooting

### Common Issues

#### Installation Problems
```bash
# CUDA compatibility issues
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ScenarioNet installation
pip install scenarionet-toolkit
```

#### Training Issues
```bash
# Out of memory
python train.py method.train_batch_size=16  # Reduce batch size

# Data loading errors
python train.py debug=True  # Enable debug mode

# Checkpoint loading
python evaluation.py ckpt_path="path/with/spaces/checkpoint.ckpt"  # Use quotes
```

#### AIS Data Issues
```bash
# No scenarios found
ls data/ais_data_from_influx_csv/*.csv  # Verify CSV files exist

# Preprocessing errors
python tools/ais_data_preprocessor.py --debug  # Debug preprocessing

# Column format issues
# Ensure CSV has required columns: time, own_latitude, own_longitude, own_sog, own_cog, host_name
```

### Getting Help
- 📖 **Documentation**: Check method-specific config files
- 🐛 **Issues**: Open GitHub issues for bugs
- 💬 **Discussions**: Use GitHub Discussions for questions
- 📧 **Contact**: Reach out to maintainers for collaboration

## 📄 License & Citation

### Citation
If you use UniTraj in your research, please cite:

```bibtex
@article{feng2024unitraj,
  title={UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction},
  author={Feng, Lan and Bahari, Mohammadhossein and Amor, Kaouther Messaoud Ben and Zablocki, {\'E}loi and Cord, Matthieu and Alahi, Alexandre},
  journal={arXiv preprint arXiv:2403.15098},
  year={2024}
}
```

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🌟 Star this repo if UniTraj helps your research!**

**🤝 Contributions welcome - let's advance trajectory prediction together!**

