#!/usr/bin/env python3
"""Extract single scene for overfitting test."""
import pickle
import numpy as np
import os

# Load the scene
scene_path = 'data/processed_ais_data/train/ais_x-pressfeeders-aquarius_20250315_060000/ais_x-pressfeeders-aquarius_20250315_060000.pkl'

print(f"📂 Loading scene from {scene_path}")
with open(scene_path, 'rb') as f:
    scene_data = pickle.load(f)

print(f"✅ Loaded scene: {scene_data['scenario_id']}")
print(f"   Keys: {list(scene_data.keys())}")
print(f"   Tracks: {list(scene_data['tracks'].keys())[:5]}...")

# Get ego vessel (first track)
track_ids = list(scene_data['tracks'].keys())
ego_id = track_ids[0]

print(f"\n🎯 Extracting ego vessel: {ego_id}")
print(f"   Position shape: {scene_data['tracks'][ego_id]['state']['position'].shape}")
print(f"   Velocity shape: {scene_data['tracks'][ego_id]['state']['velocity'].shape}")

# Create single-vessel scene with only ego
single_scene = {
    'scenario_id': scene_data['scenario_id'],
    'tracks': {ego_id: scene_data['tracks'][ego_id]},
    'timestamps': scene_data['timestamps'],
    'scenario_features': scene_data['scenario_features']
}

# Create output directory
os.makedirs('data/ais_single_scene', exist_ok=True)
os.makedirs('data/ais_single_scene/ais_single_ego', exist_ok=True)

# Save scene
output_path = 'data/ais_single_scene/ais_single_ego/ais_single_ego.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(single_scene, f)

print(f"\n💾 Saved single-vessel scene to: {output_path}")

# Create dataset files
from collections import defaultdict

# Create both train and val pointing to same scene (for overfitting)
summary = {
    "meta_info": {
        "dataset_name": "ais_single_scene",
        "total_frames": 1,
        "raw_data_format": "ais"
    },
    "scenarios": {
        "ais_single_ego": {
            "scenario_id": "ais_single_ego",
            "dataset_name": "ais_single_scene",
            "num_frames": len(single_scene['timestamps']),
            "num_agents": 1,
            "file_path": output_path
        }
    }
}

mapping = {"ais_single_ego": "ais_single_ego.pkl"}
file_list = {"ais_single_ego": output_path}

# Save dataset files
for filename in ['dataset_summary.pkl', 'dataset_mapping.pkl', 'file_list.pkl']:
    data = summary if 'summary' in filename else (mapping if 'mapping' in filename else file_list)
    with open(f'data/ais_single_scene/{filename}', 'wb') as f:
        pickle.dump(data, f)

print(f"✅ Created dataset files in data/ais_single_scene/")
print(f"   Ready for training with past_len=60")
print(f"   Same scene for train & val (overfitting test)")
