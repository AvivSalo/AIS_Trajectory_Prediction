import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
import pickle
import random
import shutil

def extract_agent_data(df, prefix='own'):
    """Extract agent data for either own ship or target ship."""
    return pd.DataFrame({
        'agent_id': df['host_name'] if prefix == 'own' else df[f'{prefix}_target_id'],
        'latitude': df[f'{prefix}_latitude'],
        'longitude': df[f'{prefix}_longitude'],
        'sog': df[f'{prefix}_sog'],
        'cog': df[f'{prefix}_cog']
    })

def process_ais_file(file_path, output_dir):
    """Process a single AIS CSV file and convert it to the format required by Wayformer."""
    df = pd.read_csv(file_path)

    # Convert timestamp to seconds from start
    df['time'] = pd.to_datetime(df['time'])
    start_time = df['time'].min()
    df['timestamp'] = (df['time'] - start_time).dt.total_seconds()

    # Initialize list to store all agent trajectories
    all_agents_data = []

    # Always process own ship
    own_data = extract_agent_data(df, prefix='own')
    own_data['timestamp'] = df['timestamp']
    all_agents_data.append(own_data)

    # Check if target data exists and process it
    target_columns = [col for col in df.columns if col.startswith('target_')]
    if target_columns:
        target_data = extract_agent_data(df, prefix='target')
        target_data['timestamp'] = df['timestamp']
        all_agents_data.append(target_data)

    # Create scene ID based on the own ship and timestamp
    scene_id = f"ais_{df['host_name'].iloc[0]}_{start_time.strftime('%Y%m%d_%H%M%S')}"

    # Create scenario directory
    scenario_dir = os.path.join(output_dir, scene_id)
    os.makedirs(scenario_dir, exist_ok=True)

    # Get reference position from first agent's first position for relative coordinates
    reference_lat = None
    reference_lon = None
    for agent_data in all_agents_data:
        if len(agent_data) > 0:
            first_row = agent_data.iloc[0]
            if not (np.isnan(first_row['latitude']) or np.isnan(first_row['longitude'])):
                reference_lat = first_row['latitude']
                reference_lon = first_row['longitude']
                break

    if reference_lat is None or reference_lon is None:
        print(f"Warning: No valid reference position found in {file_path}, skipping...")
        return None, None

    # Process each agent's trajectory
    trajectories = []
    agent_ids = []
    for agent_data in all_agents_data:
        curr_agent_ids = agent_data['agent_id'].unique()
        for agent_id in curr_agent_ids:
            agent_traj = agent_data[agent_data['agent_id'] == agent_id]
            agent_ids.append(agent_id)

            # Extract raw data
            timestamps = agent_traj['timestamp'].values
            latitudes = agent_traj['latitude'].values
            longitudes = agent_traj['longitude'].values
            sogs = agent_traj['sog'].values
            cogs = agent_traj['cog'].values

            # Convert lat/lon to relative x/y in meters
            lat_diffs = latitudes - reference_lat
            lon_diffs = longitudes - reference_lon

            # Convert to meters
            x_meters = lon_diffs * 111320 * np.cos(np.radians(reference_lat))
            y_meters = lat_diffs * 110540

            # Convert SOG (knots) to m/s
            speeds = sogs * 0.514444

            # Convert COG and speed to vx, vy
            heading_rads = np.radians(cogs)
            vx = speeds * np.sin(heading_rads)
            vy = speeds * np.cos(heading_rads)

            # Create trajectory array [timestamp, x_meters, y_meters, vx, vy]
            trajectory = np.column_stack([
                timestamps,
                x_meters,
                y_meters,
                vx,
                vy
            ])

            # Filter out rows with NaN values
            valid_mask = ~np.isnan(trajectory).any(axis=1)
            trajectory = trajectory[valid_mask]

            if len(trajectory) > 0:  # Only add if we have data
                trajectories.append(trajectory.astype(np.float32))

    # Create data dictionary in Wayformer format
    scene_data = {
        'scenario_id': scene_id,
        'tracks': {},
        'timestamps': df['timestamp'].values,
        'scenario_features': np.array([])  # Empty array for scenario-level features
    }

    for idx, trajectory in enumerate(trajectories):
        agent_id = str(agent_ids[idx])
        scene_data['tracks'][agent_id] = {
            'object_type': 'VESSEL',  # Adding vessel type
            'object_id': agent_id,
            'timestamps': trajectory[:, 0],
            'state': {
                'position': trajectory[:, 1:3],  # x_meters, y_meters (relative)
                'velocity': trajectory[:, 3:5],  # vx, vy (m/s)
            }
        }

    # Save scenario data
    scenario_file = os.path.join(scenario_dir, f"{scene_id}.pkl")
    with open(scenario_file, 'wb') as f:
        pickle.dump(scene_data, f)

    return scene_id, scenario_file

def create_dataset_files(data_dir, dataset_name):
    """Create all required dataset files for the Wayformer dataloader."""
    scenario_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('ais_')]

    # Create dataset summary
    summary = {
        "meta_info": {
            "dataset_name": dataset_name,
            "total_frames": len(scenario_dirs),
            "raw_data_format": "ais"
        },
        "scenarios": {}
    }

    # Create mapping and file_list
    mapping = {}
    file_list = {}

    for scenario_id in scenario_dirs:
        scenario_dir = os.path.join(data_dir, scenario_id)
        scenario_file = os.path.join(scenario_dir, f"{scenario_id}.pkl")

        with open(scenario_file, 'rb') as f:
            scene_data = pickle.load(f)

        # Update mapping
        mapping[scenario_id] = os.path.basename(scenario_file)

        # Update file_list
        file_list[scenario_id] = scenario_file

        # Update summary
        summary["scenarios"][scenario_id] = {
            "scenario_id": scenario_id,
            "dataset_name": dataset_name,
            "num_frames": len(scene_data['timestamps']),
            "num_agents": len(scene_data['tracks']),
            "file_path": scenario_file
        }

    # Save all files at dataset root level
    with open(os.path.join(data_dir, 'dataset_summary.pkl'), 'wb') as f:
        pickle.dump(summary, f)
    with open(os.path.join(data_dir, 'dataset_mapping.pkl'), 'wb') as f:
        pickle.dump(mapping, f)
    with open(os.path.join(data_dir, 'file_list.pkl'), 'wb') as f:
        pickle.dump(file_list, f)

    return summary, mapping, file_list

def split_data(output_dir):
    """Split the processed data into train and validation sets with proper directory structure."""
    # Create dataset directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get all scenario directories (directories starting with 'ais_')
    scenario_dirs = [d for d in os.listdir(output_dir)
                     if os.path.isdir(os.path.join(output_dir, d))
                     and d.startswith('ais_')
                     and d not in ['train', 'val']]
    random.shuffle(scenario_dirs)

    # FOR OVERFITTING: Copy ALL scenarios to both train and val
    split_idx = len(scenario_dirs)  # All to train AND val
    train_dirs = scenario_dirs  # All scenarios
    val_dirs = scenario_dirs  # Same scenarios for overfitting

    # Move directories to respective splits
    for d in train_dirs:
        src = os.path.join(output_dir, d)
        dst = os.path.join(train_dir, d)
        shutil.copytree(src, dst)  # COPY instead of move for overfitting
    for d in val_dirs:
        src = os.path.join(output_dir, d)
        dst = os.path.join(val_dir, d)
        shutil.copytree(src, dst)  # COPY instead of move for overfitting

    # Create summaries and required files for both splits
    print("\nGenerating dataset files...")
    train_summary, _, _ = create_dataset_files(train_dir, "ais_dataset_train")
    val_summary, _, _ = create_dataset_files(val_dir, "ais_dataset_val")

    print(f"\nSplit {len(scenario_dirs)} scenarios into:")
    print(f"- Training: {len(train_dirs)} scenarios in {train_dir}")
    print(f"- Validation: {len(val_dirs)} scenarios in {val_dir}")

    return train_summary, val_summary

def main():
    # Use absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_dir = os.path.join(base_dir, "data/ais_data_from_influx_csv")
    output_dir = os.path.join(base_dir, "data/processed_ais_data")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process all CSV files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in {input_dir}")
    for csv_file in tqdm(csv_files):
        try:
            scene_id, output_file = process_ais_file(csv_file, output_dir)
            if output_file is not None:
                print(f"Processed {csv_file} -> {output_file}")
            else:
                print(f"Skipped {csv_file} (no valid data)")
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

    # Split data and create all required dataset files
    print("\nSplitting data and creating dataset files...")
    train_summary, val_summary = split_data(output_dir)

    print("\nProcessing complete!")
    print(f"Train set: {train_summary['meta_info']['total_frames']} scenes")
    print(f"Validation set: {val_summary['meta_info']['total_frames']} scenes")

if __name__ == "__main__":
    main()
