#!/usr/bin/env python
"""
Create a single-scene overfitting dataset for debugging.
Copies one pickle file from x-pressfeeders-aquarius to both train and val directories.
"""

import pickle
import shutil
from pathlib import Path
import numpy as np


def create_overfit_dataset(source_pickle, output_dir, num_samples=1):
    """
    Create overfitting dataset with a single scene repeated.

    Args:
        source_pickle: Path to source pickle file
        output_dir: Output directory for overfit dataset
        num_samples: How many samples to extract from the pickle (default: 1)
    """

    print("="*80)
    print("CREATING OVERFITTING DATASET")
    print("="*80)

    # Load source pickle
    print(f"\nLoading source pickle: {source_pickle}")
    with open(source_pickle, 'rb') as f:
        source_data = pickle.load(f)

    print(f"Source pickle contains {len(source_data)} scenarios")

    # Extract first N scenarios
    overfit_data = source_data[:num_samples]

    print(f"\nExtracted {len(overfit_data)} scenario(s) for overfitting:")
    for i, scenario in enumerate(overfit_data):
        print(f"  Scenario {i}:")
        print(f"    ID: {scenario.get('scenario_id', 'unknown')}")
        print(f"    Num agents: {len(scenario.get('agents', []))} + 1 ego")
        if 'ego' in scenario:
            past_len = len(scenario['ego']['past_trajectory'])
            future_len = len(scenario['ego']['future_trajectory'])
            print(f"    Past timesteps: {past_len}")
            print(f"    Future timesteps: {future_len}")

    # Create output directories
    output_path = Path(output_dir)
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'

    # Remove existing directories
    if output_path.exists():
        print(f"\nRemoving existing directory: {output_path}")
        shutil.rmtree(output_path)

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Get original filename
    source_name = Path(source_pickle).stem

    # Save to both train and val (same data for perfect overfitting)
    train_file = train_dir / f'{source_name}_overfit.pkl'
    val_file = val_dir / f'{source_name}_overfit.pkl'

    print(f"\nSaving overfitting dataset:")
    print(f"  Train: {train_file}")
    with open(train_file, 'wb') as f:
        pickle.dump(overfit_data, f)

    print(f"  Val: {val_file}")
    with open(val_file, 'wb') as f:
        pickle.dump(overfit_data, f)

    print("\n" + "="*80)
    print("OVERFITTING DATASET CREATED SUCCESSFULLY")
    print("="*80)
    print(f"\nDataset location: {output_path}")
    print(f"  Train samples: {len(overfit_data)}")
    print(f"  Val samples: {len(overfit_data)} (same as train for perfect overfitting)")
    print(f"\nUsage in config.yaml:")
    print(f"  train_data_path: [ \"{train_dir}\" ]")
    print(f"  val_data_path: [ \"{val_dir}\" ]")


if __name__ == '__main__':
    # Source pickle from processed data
    source_pickle = 'data/processed_ais_data/train/ais_x-pressfeeders-aquarius_20250315_060000/ais_x-pressfeeders-aquarius_20250315_060000.pkl'

    # Output directory for overfitting
    output_dir = 'data/debug_overfit_xpressfeeders'

    # Create with 1 sample
    create_overfit_dataset(source_pickle, output_dir, num_samples=1)
