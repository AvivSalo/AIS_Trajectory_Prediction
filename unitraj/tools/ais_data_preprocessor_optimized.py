import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
import pickle
import random
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial

def extract_agent_data(df, prefix='own'):
    """Extract agent data for either own ship or target ship."""
    return pd.DataFrame({
        'agent_id': df['host_name'] if prefix == 'own' else df[f'{prefix}_target_id'],
        'latitude': df[f'{prefix}_latitude'],
        'longitude': df[f'{prefix}_longitude'],
        'sog': df[f'{prefix}_sog'],
        'cog': df[f'{prefix}_cog']
    })

def interpolate_agent_trajectory(agent_data, target_interval=1.0, max_gap=400.0):
    """
    Interpolate target vessel trajectory to consistent 1-second intervals.

    NOTE: Using ORIGINAL implementation for correctness.
    Multiprocessing provides the main speedup (80-100x).
    max_gap=400.0 matches original to handle large AIS target gaps (20-380s).
    """
    if len(agent_data) < 2:
        return agent_data  # Can't interpolate single point

    # Check if interpolation needed
    time_diffs = np.diff(agent_data['timestamp'].values)
    if np.max(time_diffs) <= target_interval * 1.5:
        return agent_data  # Already consistent

    # Sort by timestamp to ensure correct ordering
    agent_data = agent_data.sort_values('timestamp').reset_index(drop=True)

    # Build interpolated segments, respecting max_gap constraint
    interpolated_segments = []

    for i in range(len(agent_data) - 1):
        # Always include the current observation
        current_obs = agent_data.iloc[i:i+1].copy()
        interpolated_segments.append(current_obs)

        # Check gap to next observation
        t_current = agent_data.iloc[i]['timestamp']
        t_next = agent_data.iloc[i+1]['timestamp']
        gap = t_next - t_current

        # Only interpolate if gap is significant but not too large
        if gap > target_interval * 1.5 and gap <= max_gap:
            # Create interpolation timestamps (excluding endpoints)
            t_interp = np.arange(t_current + target_interval, t_next, target_interval)

            if len(t_interp) > 0:
                # Interpolate ONLY position (lat/lon)
                lat_interp = np.interp(t_interp,
                                      [t_current, t_next],
                                      [agent_data.iloc[i]['latitude'], agent_data.iloc[i+1]['latitude']])
                lon_interp = np.interp(t_interp,
                                      [t_current, t_next],
                                      [agent_data.iloc[i]['longitude'], agent_data.iloc[i+1]['longitude']])

                # Create DataFrame for interpolated points (SOG and COG will be calculated later)
                interp_segment = pd.DataFrame({
                    'timestamp': t_interp,
                    'latitude': lat_interp,
                    'longitude': lon_interp,
                    'sog': np.nan,  # Will be calculated from position differences
                    'cog': np.nan,  # Will be calculated from position differences
                    'agent_id': agent_data['agent_id'].iloc[0]
                })
                interpolated_segments.append(interp_segment)

    # Add the last observation
    interpolated_segments.append(agent_data.iloc[-1:].copy())

    # Combine all segments
    result = pd.concat(interpolated_segments, ignore_index=True).sort_values('timestamp').reset_index(drop=True)

    # Calculate SOG and COG from interpolated positions
    for i in range(len(result)):
        if pd.isna(result.loc[i, 'sog']) or pd.isna(result.loc[i, 'cog']):
            # Calculate from position differences
            if i < len(result) - 1:
                # Use forward difference
                lat1, lon1 = result.loc[i, 'latitude'], result.loc[i, 'longitude']
                lat2, lon2 = result.loc[i+1, 'latitude'], result.loc[i+1, 'longitude']
                dt = result.loc[i+1, 'timestamp'] - result.loc[i, 'timestamp']
            elif i > 0:
                # Use backward difference for last point
                lat1, lon1 = result.loc[i-1, 'latitude'], result.loc[i-1, 'longitude']
                lat2, lon2 = result.loc[i, 'latitude'], result.loc[i, 'longitude']
                dt = result.loc[i, 'timestamp'] - result.loc[i-1, 'timestamp']
            else:
                continue  # Single point, keep original values

            # Calculate distance in meters using Haversine approximation
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            avg_lat = (lat1 + lat2) / 2

            # Convert to meters
            dy = dlat * 110540  # meters per degree latitude
            dx = dlon * 111320 * np.cos(np.radians(avg_lat))  # meters per degree longitude

            # Calculate speed (SOG) in knots
            distance_m = np.sqrt(dx**2 + dy**2)
            speed_ms = distance_m / dt if dt > 0 else 0
            sog_knots = speed_ms / 0.514444  # Convert m/s to knots

            # Calculate course (COG) in degrees
            cog_deg = (np.degrees(np.arctan2(dx, dy)) + 360) % 360

            result.loc[i, 'sog'] = sog_knots
            result.loc[i, 'cog'] = cog_deg

    return result

def process_ais_file(file_path, output_dir):
    """Process a single AIS CSV file and convert it to the format required by Wayformer."""
    # OPTIMIZATION 3: Use PyArrow engine for 2-3x faster CSV reading
    try:
        df = pd.read_csv(file_path, engine='pyarrow')
    except Exception:
        # Fallback to default engine if pyarrow not available
        df = pd.read_csv(file_path)

    # Check if target data exists
    target_columns = [col for col in df.columns if col.startswith('target_')]
    has_targets = len(target_columns) > 0 and 'target_target_id' in df.columns

    if has_targets:
        rows_before = len(df)
        df = df.drop_duplicates(subset=['time', 'own_latitude', 'own_longitude', 'target_target_id'], keep='first')
        rows_after = len(df)
    else:
        rows_before = len(df)
        df = df.drop_duplicates(subset=['time', 'own_latitude', 'own_longitude'], keep='first')
        rows_after = len(df)

    # Convert timestamp to seconds from start
    df['time'] = pd.to_datetime(df['time'])
    start_time = df['time'].min()
    df['timestamp'] = (df['time'] - start_time).dt.total_seconds()

    # Initialize list to store all agent trajectories
    all_agents_data = []

    # Always process own ship
    own_data = extract_agent_data(df, prefix='own')
    own_data['timestamp'] = df['timestamp']

    if has_targets:
        own_rows_before = len(own_data)
        own_data = own_data.drop_duplicates(subset=['timestamp', 'latitude', 'longitude'], keep='first')
        own_rows_after = len(own_data)

    all_agents_data.append(own_data)


    # Check if target data exists and process it
    if has_targets:
        target_data = extract_agent_data(df, prefix='target')
        target_data['timestamp'] = df['timestamp']

        # Interpolate target trajectories (using original correct implementation)
        target_ids = target_data['agent_id'].unique()
        interpolated_targets = []

        for target_id in target_ids:
            target_traj = target_data[target_data['agent_id'] == target_id].copy()

            if len(target_traj) >= 2:
                target_interp = interpolate_agent_trajectory(target_traj, target_interval=1.0, max_gap=400.0)
                interpolated_targets.append(target_interp)
            else:
                interpolated_targets.append(target_traj)

        if interpolated_targets:
            target_data = pd.concat(interpolated_targets, ignore_index=True)

        all_agents_data.append(target_data)

    # Create scene ID
    scene_id = f"ais_{df['host_name'].iloc[0]}_{start_time.strftime('%Y%m%d_%H%M%S')}"

    # Create scenario directory
    scenario_dir = os.path.join(output_dir, scene_id)
    os.makedirs(scenario_dir, exist_ok=True)

    # Get reference position
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
        return None, None

    # Process each agent's trajectory
    trajectories = []
    agent_ids = []
    for agent_data in all_agents_data:
        curr_agent_ids = agent_data['agent_id'].unique()
        for agent_id in curr_agent_ids:
            agent_traj = agent_data[agent_data['agent_id'] == agent_id]
            agent_ids.append(agent_id)

            # Extract raw data as numpy arrays (faster than pandas)
            timestamps = agent_traj['timestamp'].values
            latitudes = agent_traj['latitude'].values
            longitudes = agent_traj['longitude'].values
            sogs = agent_traj['sog'].values
            cogs = agent_traj['cog'].values

            # Vectorized coordinate conversion
            lat_diffs = latitudes - reference_lat
            lon_diffs = longitudes - reference_lon
            x_meters = lon_diffs * 111320 * np.cos(np.radians(reference_lat))
            y_meters = lat_diffs * 110540

            # Vectorized velocity conversion
            speeds = sogs * 0.514444
            heading_rads = np.radians(cogs)
            vx = speeds * np.sin(heading_rads)
            vy = speeds * np.cos(heading_rads)

            # Create trajectory array
            trajectory = np.column_stack([timestamps, x_meters, y_meters, vx, vy])

            # Filter NaN values
            valid_mask = ~np.isnan(trajectory).any(axis=1)
            trajectory = trajectory[valid_mask]

            if len(trajectory) > 0:
                trajectories.append(trajectory.astype(np.float32))

    # Create data dictionary
    scene_data = {
        'scenario_id': scene_id,
        'tracks': {},
        'timestamps': df['timestamp'].values,
        'scenario_features': np.array([])
    }

    for idx, trajectory in enumerate(trajectories):
        agent_id = str(agent_ids[idx])
        scene_data['tracks'][agent_id] = {
            'object_type': 'VESSEL',
            'object_id': agent_id,
            'timestamps': trajectory[:, 0],
            'state': {
                'position': trajectory[:, 1:3],
                'velocity': trajectory[:, 3:5],
            }
        }

    # Save scenario data
    scenario_file = os.path.join(scenario_dir, f"{scene_id}.pkl")
    with open(scenario_file, 'wb') as f:
        pickle.dump(scene_data, f)

    return scene_id, scenario_file

def process_file_wrapper(args):
    """Wrapper function for multiprocessing that handles errors gracefully."""
    file_path, output_dir = args
    try:
        scene_id, output_file = process_ais_file(file_path, output_dir)
        if output_file is not None:
            return ('success', file_path, output_file)
        else:
            return ('skipped', file_path, 'no valid data')
    except Exception as e:
        return ('error', file_path, str(e))

def create_dataset_files(data_dir, dataset_name):
    """Create all required dataset files for the Wayformer dataloader."""
    scenario_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('ais_')]

    summary = {
        "meta_info": {
            "dataset_name": dataset_name,
            "total_frames": len(scenario_dirs),
            "raw_data_format": "ais"
        },
        "scenarios": {}
    }

    mapping = {}
    file_list = {}

    for scenario_id in scenario_dirs:
        scenario_dir = os.path.join(data_dir, scenario_id)
        scenario_file = os.path.join(scenario_dir, f"{scenario_id}.pkl")

        with open(scenario_file, 'rb') as f:
            scene_data = pickle.load(f)

        mapping[scenario_id] = os.path.basename(scenario_file)
        file_list[scenario_id] = scenario_file

        summary["scenarios"][scenario_id] = {
            "scenario_id": scenario_id,
            "dataset_name": dataset_name,
            "num_frames": len(scene_data['timestamps']),
            "num_agents": len(scene_data['tracks']),
            "file_path": scenario_file
        }

    with open(os.path.join(data_dir, 'dataset_summary.pkl'), 'wb') as f:
        pickle.dump(summary, f)
    with open(os.path.join(data_dir, 'dataset_mapping.pkl'), 'wb') as f:
        pickle.dump(mapping, f)
    with open(os.path.join(data_dir, 'file_list.pkl'), 'wb') as f:
        pickle.dump(file_list, f)

    return summary, mapping, file_list

def split_data(output_dir):
    """Split the processed data into train and validation sets."""
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    scenario_dirs = [d for d in os.listdir(output_dir)
                     if os.path.isdir(os.path.join(output_dir, d))
                     and d.startswith('ais_')
                     and d not in ['train', 'val']]
    random.shuffle(scenario_dirs)

    split_idx = int(len(scenario_dirs) * 0.8)
    train_dirs = scenario_dirs[:split_idx]
    val_dirs = scenario_dirs[split_idx:]

    for d in train_dirs:
        src = os.path.join(output_dir, d)
        dst = os.path.join(train_dir, d)
        shutil.move(src, dst)
    for d in val_dirs:
        src = os.path.join(output_dir, d)
        dst = os.path.join(val_dir, d)
        shutil.move(src, dst)

    print("\nGenerating dataset files...")
    train_summary, _, _ = create_dataset_files(train_dir, "ais_dataset_train")
    val_summary, _, _ = create_dataset_files(val_dir, "ais_dataset_val")

    print(f"\nSplit {len(scenario_dirs)} scenarios into:")
    print(f"- Training: {len(train_dirs)} scenarios in {train_dir}")
    print(f"- Validation: {len(val_dirs)} scenarios in {val_dir}")

    return train_summary, val_summary

def main():
    """OPTIMIZED MAIN: Uses multiprocessing for parallel file processing."""
    # Use absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_dir = os.path.join(base_dir, "data/ais_data_from_influx_csv")
    output_dir = os.path.join(base_dir, "data/processed_ais_optimized_test")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get CSV files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in {input_dir}")

    # OPTIMIZATION 1: Multiprocessing - Process files in parallel
    num_workers = cpu_count()  # Use ALL available CPU cores
    print(f"Using {num_workers} parallel workers (all available cores) for processing")

    # Prepare arguments for multiprocessing
    args_list = [(csv_file, output_dir) for csv_file in csv_files]

    # Process files in parallel with progress bar
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_file_wrapper, args_list),
            total=len(csv_files),
            desc="Processing CSV files"
        ))

    # Report results
    success_count = sum(1 for r in results if r[0] == 'success')
    skipped_count = sum(1 for r in results if r[0] == 'skipped')
    error_count = sum(1 for r in results if r[0] == 'error')

    print(f"\nProcessing complete:")
    print(f"  ✓ Success: {success_count}")
    print(f"  - Skipped: {skipped_count}")
    print(f"  ✗ Errors: {error_count}")

    if error_count > 0:
        print("\nErrors encountered:")
        for status, file_path, error_msg in results:
            if status == 'error':
                print(f"  {file_path}: {error_msg}")

    # Split data and create dataset files
    print("\nSplitting data and creating dataset files...")
    train_summary, val_summary = split_data(output_dir)

    print("\nProcessing complete!")
    print(f"Train set: {train_summary['meta_info']['total_frames']} scenes")
    print(f"Validation set: {val_summary['meta_info']['total_frames']} scenes")

if __name__ == "__main__":
    main()