#!/usr/bin/env python3

# This V2 file is for handel the Danish AIS data which is different from the Orca AI InfluxDB format.

"""
Unified AIS Data Preprocessor

Handles both InfluxDB and Danish AIS formats with automatic detection.
Converts raw CSV data to Wayformer-compatible pickle scenarios with train/val split.

Supports:
- InfluxDB 16-column format (own + target vessels)
- InfluxDB 7-column format (own vessel only)
- Danish AIS 25-column format (auto-converts to 16-column)
"""

import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
import pickle
import random
import shutil
import argparse
import logging
from pathlib import Path

from ais_conversion_utils import (
    haversine_distance,
    calculate_bearing,
    calculate_cpa_tcpa,
    parse_danish_timestamp,
    parse_influxdb_timestamp,
    latlon_to_meters,
    knots_to_ms,
    course_speed_to_velocity
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_format(csv_file):
    """
    Detect AIS CSV format type by examining columns.

    Returns:
        str: 'danish' | 'influxdb_16col' | 'influxdb_7col' | 'unknown'
    """
    try:
        # Read first row to check columns
        df_sample = pd.read_csv(csv_file, nrows=1, skipinitialspace=True)
        df_sample.columns = df_sample.columns.str.strip().str.lstrip('#').str.strip()
        cols = set(df_sample.columns)

        # Danish format detection
        danish_indicators = {'Timestamp', 'MMSI', 'Type of mobile', 'SOG', 'COG'}
        if danish_indicators.issubset(cols):
            logger.info(f"Detected Danish AIS format: {csv_file}")
            return 'danish'

        # InfluxDB 16-column format (with targets)
        influx_16col_indicators = {'time', 'own_latitude', 'own_longitude', 'host_name',
                                    'target_latitude', 'target_longitude', 'target_target_id'}
        if influx_16col_indicators.issubset(cols):
            logger.info(f"Detected InfluxDB 16-column format: {csv_file}")
            return 'influxdb_16col'

        # InfluxDB 7-column format (own vessel only)
        influx_7col_indicators = {'time', 'own_latitude', 'own_longitude', 'host_name',
                                   'own_sog', 'own_cog'}
        if influx_7col_indicators.issubset(cols):
            logger.info(f"Detected InfluxDB 7-column format: {csv_file}")
            return 'influxdb_7col'

        logger.warning(f"Unknown format: {csv_file}, columns: {list(cols)[:5]}...")
        return 'unknown'

    except Exception as e:
        logger.error(f"Error detecting format for {csv_file}: {e}")
        return 'unknown'


def _convert_danish_chunked(input_path, output_path, max_distance=10000,
                            max_targets_per_vessel=20, keep_base_stations=False,
                            chunk_size=1000000, max_rows=None):
    """
    Convert large Danish AIS files using chunked processing for memory efficiency.

    OPTIMIZED: Only computes distance for filtering. Skips CPA/TCPA/bearing since
    they are not used by the dataloader (which only needs position and velocity).

    Strategy:
    1. Process file in chunks to avoid loading all 17M records
    2. Group records by timestamp within each chunk
    3. Write output incrementally

    Args:
        max_rows: Maximum number of rows to process (None = all rows)
    """
    logger.info(f"Starting chunked conversion (optimized): {chunk_size:,} records per chunk")
    if max_rows:
        logger.info(f"Row limit: {max_rows:,} rows (for development/testing)")

    # Initialize output file (write header implicitly disabled in final output)
    first_chunk = True
    total_rows_written = 0
    total_rows_read = 0
    chunk_num = 0

    # Read file in chunks
    chunk_iterator = pd.read_csv(input_path, skipinitialspace=True, chunksize=chunk_size, nrows=max_rows)

    for chunk_df in chunk_iterator:
        chunk_num += 1
        logger.info(f"Processing chunk {chunk_num} ({len(chunk_df):,} records)")

        # Clean column names
        chunk_df.columns = chunk_df.columns.str.strip().str.lstrip('#').str.strip()

        # Filter base stations
        if not keep_base_stations and 'Type of mobile' in chunk_df.columns:
            chunk_df = chunk_df[chunk_df['Type of mobile'] != 'Base Station']

        # Parse timestamps
        chunk_df['parsed_time'] = chunk_df['Timestamp'].apply(parse_danish_timestamp)
        chunk_df = chunk_df.dropna(subset=['parsed_time'])

        # Filter valid data
        chunk_df = chunk_df.dropna(subset=['MMSI', 'Latitude', 'Longitude'])
        chunk_df = chunk_df[chunk_df['Latitude'].between(-90, 90) & chunk_df['Longitude'].between(-180, 180)]

        if len(chunk_df) == 0:
            logger.info(f"Chunk {chunk_num}: No valid records after filtering")
            continue

        logger.info(f"Chunk {chunk_num}: Processing {len(chunk_df)} valid records")

        # Process this chunk's timestamps
        output_rows = []
        timestamps = chunk_df['parsed_time'].unique()

        for timestamp in timestamps:
            time_group = chunk_df[chunk_df['parsed_time'] == timestamp].copy()

            if len(time_group) < 2:
                # Single vessel - add with no target (unused columns set to 0 or NaN)
                for _, own in time_group.iterrows():
                    output_rows.append({
                        'time': timestamp,
                        'own_latitude': own['Latitude'],
                        'own_longitude': own['Longitude'],
                        'host_name': str(int(own['MMSI'])),
                        'own_sog': own['SOG'] if pd.notna(own['SOG']) else 0.0,
                        'own_cog': own['COG'] if pd.notna(own['COG']) else (own['Heading'] if pd.notna(own['Heading']) else 0.0),
                        'own_rot': 0.0,  # Unused by dataloader
                        'target_latitude': np.nan,
                        'target_longitude': np.nan,
                        'target_distance': np.nan,
                        'target_sog': np.nan,
                        'target_cog': np.nan,
                        'target_cpa': 0.0,  # Unused by dataloader
                        'target_tcpa': 0.0,  # Unused by dataloader
                        'target_bearing': 0.0,  # Unused by dataloader
                        'target_target_id': np.nan
                    })
                continue

            # Process vessel-to-vessel relationships
            for _, own in time_group.iterrows():
                own_mmsi = own['MMSI']
                own_lat = own['Latitude']
                own_lon = own['Longitude']
                own_sog = own['SOG'] if pd.notna(own['SOG']) else 0.0
                own_cog = own['COG'] if pd.notna(own['COG']) else (own['Heading'] if pd.notna(own['Heading']) else 0.0)

                # Find nearby targets (ONLY compute distance, skip CPA/TCPA/bearing)
                targets = []
                for _, target in time_group.iterrows():
                    if target['MMSI'] == own_mmsi:
                        continue

                    target_lat = target['Latitude']
                    target_lon = target['Longitude']
                    target_sog = target['SOG'] if pd.notna(target['SOG']) else 0.0
                    target_cog = target['COG'] if pd.notna(target['COG']) else (target['Heading'] if pd.notna(target['Heading']) else 0.0)

                    # Only compute distance for filtering
                    distance = haversine_distance(own_lat, own_lon, target_lat, target_lon)

                    if distance <= max_distance:
                        targets.append({
                            'distance': distance,
                            'target_latitude': target_lat,
                            'target_longitude': target_lon,
                            'target_sog': target_sog,
                            'target_cog': target_cog,
                            'target_target_id': int(target['MMSI'])
                        })

                # Sort by distance, take closest
                targets.sort(key=lambda x: x['distance'])
                targets = targets[:max_targets_per_vessel]

                if not targets:
                    output_rows.append({
                        'time': timestamp,
                        'own_latitude': own_lat,
                        'own_longitude': own_lon,
                        'host_name': str(int(own_mmsi)),
                        'own_sog': own_sog,
                        'own_cog': own_cog,
                        'own_rot': 0.0,
                        'target_latitude': np.nan,
                        'target_longitude': np.nan,
                        'target_distance': np.nan,
                        'target_sog': np.nan,
                        'target_cog': np.nan,
                        'target_cpa': 0.0,
                        'target_tcpa': 0.0,
                        'target_bearing': 0.0,
                        'target_target_id': np.nan
                    })
                else:
                    for target_info in targets:
                        output_rows.append({
                            'time': timestamp,
                            'own_latitude': own_lat,
                            'own_longitude': own_lon,
                            'host_name': str(int(own_mmsi)),
                            'own_sog': own_sog,
                            'own_cog': own_cog,
                            'own_rot': 0.0,  # Unused
                            'target_latitude': target_info['target_latitude'],
                            'target_longitude': target_info['target_longitude'],
                            'target_distance': target_info['distance'],
                            'target_sog': target_info['target_sog'],
                            'target_cog': target_info['target_cog'],
                            'target_cpa': 0.0,  # Unused - set to 0 for compatibility
                            'target_tcpa': 0.0,  # Unused - set to 0 for compatibility
                            'target_bearing': 0.0,  # Unused - set to 0 for compatibility
                            'target_target_id': f"({target_info['target_target_id']})"
                        })

        # Write chunk results incrementally
        if output_rows:
            chunk_output_df = pd.DataFrame(output_rows)

            # Write with or without header depending on whether this is first chunk
            if first_chunk:
                chunk_output_df.to_csv(output_path, index=False, header=False, mode='w')
                first_chunk = False
            else:
                chunk_output_df.to_csv(output_path, index=False, header=False, mode='a')

            total_rows_written += len(chunk_output_df)
            logger.info(f"Chunk {chunk_num}: Wrote {len(chunk_output_df):,} rows (total: {total_rows_written:,})")

    logger.info(f"Chunked conversion complete: {total_rows_written:,} total rows written")
    logger.info(f"Output: {output_path}")
    return output_path


def convert_danish_to_influxdb(input_path, output_path, max_distance=10000,
                                 max_targets_per_vessel=20, keep_base_stations=False,
                                 chunk_size=1000000, max_rows=None):
    """
    Convert Danish AIS CSV to InfluxDB 16-column format with target vessel relationships.

    Automatically uses chunked processing for files >1M records for memory efficiency.

    Args:
        input_path: Danish AIS CSV file
        output_path: Output InfluxDB format CSV
        max_distance: Maximum distance (meters) to consider vessels as targets
        max_targets_per_vessel: Max targets to include per own vessel per timestamp
        keep_base_stations: Keep base station records
        chunk_size: Records per chunk for large files (default: 1M)
        max_rows: Maximum number of rows to process (None = all rows)
    """
    logger.info(f"Converting Danish AIS: {input_path} -> {output_path}")

    # Check file size to decide on batch processing
    import os
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    use_chunks = file_size_mb > 100  # Use chunks for files >100MB

    if use_chunks:
        logger.info(f"Large file detected ({file_size_mb:.1f} MB), using chunked processing")
        return _convert_danish_chunked(input_path, output_path, max_distance,
                                       max_targets_per_vessel, keep_base_stations, chunk_size, max_rows)

    # Small file - process normally
    logger.info(f"Processing file ({file_size_mb:.1f} MB) in memory")

    # Read Danish CSV
    df = pd.read_csv(input_path, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lstrip('#').str.strip()

    initial_count = len(df)

    # Filter base stations
    if not keep_base_stations and 'Type of mobile' in df.columns:
        df = df[df['Type of mobile'] != 'Base Station']
        logger.info(f"Filtered {initial_count - len(df)} base stations")

    # Parse timestamps
    df['parsed_time'] = df['Timestamp'].apply(parse_danish_timestamp)
    df = df.dropna(subset=['parsed_time'])

    # Filter valid data
    df = df.dropna(subset=['MMSI', 'Latitude', 'Longitude'])
    df = df[df['Latitude'].between(-90, 90) & df['Longitude'].between(-180, 180)]

    logger.info(f"Processing {len(df)} valid records")

    # Group by timestamp to find vessels at same time
    output_rows = []
    timestamps = df['parsed_time'].unique()

    for timestamp in tqdm(timestamps, desc="Processing timestamps"):
        time_group = df[df['parsed_time'] == timestamp].copy()

        if len(time_group) < 2:
            # Single vessel - add with no target
            for _, own in time_group.iterrows():
                output_rows.append({
                    'time': timestamp,
                    'own_latitude': own['Latitude'],
                    'own_longitude': own['Longitude'],
                    'host_name': str(int(own['MMSI'])),
                    'own_sog': own['SOG'] if pd.notna(own['SOG']) else 0.0,
                    'own_cog': own['COG'] if pd.notna(own['COG']) else (own['Heading'] if pd.notna(own['Heading']) else 0.0),
                    'own_rot': own['ROT'] if pd.notna(own['ROT']) else 0.0,
                    'target_latitude': np.nan,
                    'target_longitude': np.nan,
                    'target_distance': np.nan,
                    'target_sog': np.nan,
                    'target_cog': np.nan,
                    'target_cpa': np.nan,
                    'target_tcpa': np.nan,
                    'target_bearing': np.nan,
                    'target_target_id': np.nan
                })
            continue

        # Process each vessel as "own" vessel
        for _, own in time_group.iterrows():
            own_mmsi = own['MMSI']
            own_lat = own['Latitude']
            own_lon = own['Longitude']
            own_sog = own['SOG'] if pd.notna(own['SOG']) else 0.0
            own_cog = own['COG'] if pd.notna(own['COG']) else (own['Heading'] if pd.notna(own['Heading']) else 0.0)
            own_rot = own['ROT'] if pd.notna(own['ROT']) else 0.0

            # Find nearby vessels as targets
            targets = []
            for _, target in time_group.iterrows():
                if target['MMSI'] == own_mmsi:
                    continue

                target_lat = target['Latitude']
                target_lon = target['Longitude']
                target_sog = target['SOG'] if pd.notna(target['SOG']) else 0.0
                target_cog = target['COG'] if pd.notna(target['COG']) else (target['Heading'] if pd.notna(target['Heading']) else 0.0)

                distance = haversine_distance(own_lat, own_lon, target_lat, target_lon)

                if distance <= max_distance:
                    bearing = calculate_bearing(own_lat, own_lon, target_lat, target_lon)
                    cpa, tcpa = calculate_cpa_tcpa(own_lat, own_lon, own_sog, own_cog,
                                                    target_lat, target_lon, target_sog, target_cog)

                    targets.append({
                        'distance': distance,
                        'target_latitude': target_lat,
                        'target_longitude': target_lon,
                        'target_sog': target_sog,
                        'target_cog': target_cog,
                        'target_cpa': cpa,
                        'target_tcpa': tcpa,
                        'target_bearing': bearing,
                        'target_target_id': int(target['MMSI'])
                    })

            # Sort by distance and take closest
            targets.sort(key=lambda x: x['distance'])
            targets = targets[:max_targets_per_vessel]

            if not targets:
                output_rows.append({
                    'time': timestamp,
                    'own_latitude': own_lat,
                    'own_longitude': own_lon,
                    'host_name': str(int(own_mmsi)),
                    'own_sog': own_sog,
                    'own_cog': own_cog,
                    'own_rot': own_rot,
                    'target_latitude': np.nan,
                    'target_longitude': np.nan,
                    'target_distance': np.nan,
                    'target_sog': np.nan,
                    'target_cog': np.nan,
                    'target_cpa': np.nan,
                    'target_tcpa': np.nan,
                    'target_bearing': np.nan,
                    'target_target_id': np.nan
                })
            else:
                for target_info in targets:
                    output_rows.append({
                        'time': timestamp,
                        'own_latitude': own_lat,
                        'own_longitude': own_lon,
                        'host_name': str(int(own_mmsi)),
                        'own_sog': own_sog,
                        'own_cog': own_cog,
                        'own_rot': own_rot,
                        'target_latitude': target_info['target_latitude'],
                        'target_longitude': target_info['target_longitude'],
                        'target_distance': target_info['distance'],
                        'target_sog': target_info['target_sog'],
                        'target_cog': target_info['target_cog'],
                        'target_cpa': target_info['target_cpa'],
                        'target_tcpa': target_info['target_tcpa'],
                        'target_bearing': target_info['target_bearing'],
                        'target_target_id': f"({target_info['target_target_id']})"
                    })

    # Create output dataframe
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_path, index=False, header=False)

    logger.info(f"Converted: {len(output_df)} rows generated")
    return output_path


def extract_agent_data(df, prefix='own'):
    """Extract agent data for either own ship or target ship."""
    return pd.DataFrame({
        'agent_id': df['host_name'] if prefix == 'own' else df[f'{prefix}_target_id'],
        'latitude': df[f'{prefix}_latitude'],
        'longitude': df[f'{prefix}_longitude'],
        'sog': df[f'{prefix}_sog'],
        'cog': df[f'{prefix}_cog']
    })


def process_ais_file(file_path, output_dir, dataset_name='ais_dataset', max_rows=None):
    """
    Process a single AIS CSV file and convert it to Wayformer format.

    Auto-detects format and converts if needed.

    Args:
        max_rows: Maximum number of rows to process from CSV (None = all rows)
    """
    # Detect format
    file_format = detect_format(file_path)

    # Convert Danish to InfluxDB if needed
    if file_format == 'danish':
        converted_path = file_path.replace('.csv', '_converted.csv')
        convert_danish_to_influxdb(file_path, converted_path, max_rows=max_rows)
        file_path = converted_path
        file_format = 'influxdb_16col'

    # Load CSV (now guaranteed to be InfluxDB format)
    df = pd.read_csv(file_path, header=None, skipinitialspace=True)

    # Assign column names based on format
    if file_format == 'influxdb_16col':
        df.columns = ['time', 'own_latitude', 'own_longitude', 'host_name', 'own_sog',
                       'own_cog', 'own_rot', 'target_latitude', 'target_longitude',
                       'target_distance', 'target_sog', 'target_cog', 'target_cpa',
                       'target_tcpa', 'target_bearing', 'target_target_id']
    elif file_format == 'influxdb_7col':
        df.columns = ['time', 'own_latitude', 'own_longitude', 'host_name',
                       'own_sog', 'own_cog', 'own_rot']

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
    if file_format == 'influxdb_16col':
        target_columns = [col for col in df.columns if col.startswith('target_')]
        if target_columns:
            target_data = extract_agent_data(df, prefix='target')
            target_data['timestamp'] = df['timestamp']
            all_agents_data.append(target_data)

    # Create scene ID
    scene_id = f"ais_{df['host_name'].iloc[0]}_{start_time.strftime('%Y%m%d_%H%M%S')}"

    # Create scenario directory
    scenario_dir = os.path.join(output_dir, scene_id)
    os.makedirs(scenario_dir, exist_ok=True)

    # Get reference position for relative coordinates
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
        logger.warning(f"No valid reference position in {file_path}, skipping")
        return None, None

    # Process each agent's trajectory
    trajectories = []
    agent_ids = []

    for agent_data in all_agents_data:
        curr_agent_ids = agent_data['agent_id'].unique()
        for agent_id in curr_agent_ids:
            agent_traj = agent_data[agent_data['agent_id'] == agent_id]
            agent_ids.append(agent_id)

            timestamps = agent_traj['timestamp'].values
            latitudes = agent_traj['latitude'].values
            longitudes = agent_traj['longitude'].values
            sogs = agent_traj['sog'].values
            cogs = agent_traj['cog'].values

            # Convert to relative meters
            x_meters, y_meters = latlon_to_meters(latitudes, longitudes, reference_lat, reference_lon)

            # Convert SOG to m/s and get velocity components
            speeds = knots_to_ms(sogs)
            vx, vy = course_speed_to_velocity(cogs, speeds)

            # Create trajectory array
            trajectory = np.column_stack([timestamps, x_meters, y_meters, vx, vy])

            # Filter out NaN values
            valid_mask = ~np.isnan(trajectory).any(axis=1)
            trajectory = trajectory[valid_mask]

            if len(trajectory) > 0:
                trajectories.append(trajectory.astype(np.float32))

    # Create data dictionary in Wayformer format
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


def create_dataset_files(data_dir, dataset_name):
    """Create all required dataset files for the Wayformer dataloader."""
    scenario_dirs = [d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('ais_')]

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

    # Save all files
    with open(os.path.join(data_dir, 'dataset_summary.pkl'), 'wb') as f:
        pickle.dump(summary, f)
    with open(os.path.join(data_dir, 'dataset_mapping.pkl'), 'wb') as f:
        pickle.dump(mapping, f)
    with open(os.path.join(data_dir, 'file_list.pkl'), 'wb') as f:
        pickle.dump(file_list, f)

    return summary, mapping, file_list


def split_data(output_dir, train_ratio=0.8):
    """Split processed data into train and validation sets."""
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    scenario_dirs = [d for d in os.listdir(output_dir)
                     if os.path.isdir(os.path.join(output_dir, d))
                     and d.startswith('ais_')
                     and d not in ['train', 'val']]
    random.shuffle(scenario_dirs)

    split_idx = int(len(scenario_dirs) * train_ratio)
    train_dirs = scenario_dirs[:split_idx]
    val_dirs = scenario_dirs[split_idx:]

    # Move directories
    for d in train_dirs:
        shutil.move(os.path.join(output_dir, d), os.path.join(train_dir, d))
    for d in val_dirs:
        shutil.move(os.path.join(output_dir, d), os.path.join(val_dir, d))

    logger.info("Generating dataset files...")
    train_summary, _, _ = create_dataset_files(train_dir, "ais_dataset_train")
    val_summary, _, _ = create_dataset_files(val_dir, "ais_dataset_val")

    logger.info(f"Split {len(scenario_dirs)} scenarios:")
    logger.info(f"  Training: {len(train_dirs)} in {train_dir}")
    logger.info(f"  Validation: {len(val_dirs)} in {val_dir}")

    return train_summary, val_summary


def main():
    parser = argparse.ArgumentParser(
        description='Unified AIS Data Preprocessor - handles InfluxDB and Danish formats'
    )
    parser.add_argument('--input-dir', type=str,
                        help='Input directory containing AIS CSV files')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for processed scenarios')
    parser.add_argument('--dataset-name', type=str, default='ais_dataset',
                        help='Dataset name for metadata')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Train/val split ratio (default: 0.8)')
    parser.add_argument('--max-distance', type=float, default=10000,
                        help='Max distance for target vessels when converting Danish (meters)')
    parser.add_argument('--max-targets', type=int, default=20,
                        help='Max targets per vessel when converting Danish')
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Maximum number of rows to process from CSV (None = all rows, useful for testing)')

    args = parser.parse_args()

    # HARDCODED FOR DANISH AIS DATA
    if not args.input_dir or not args.output_dir:
        # Use Danish data paths
        input_dir = "/home/aviv/Projects/UniTraj/data/ais_data_from_danish_kindon/csv"
        output_dir = "/home/aviv/Projects/UniTraj/data/processed_danish_ais"
        args.dataset_name = "danish_ais_dataset"
        args.max_distance = 10000 # 10 kilometers
        args.max_targets = 20 # max 20 targets per own vessel
        args.train_ratio = 0.8 # 80% train, 20% val
        args.max_rows = 50000 # TEMPORARY: Process only 50000 rows for testing
        logger.info("Using hardcoded Danish AIS paths:")
        logger.info(f"  Input: {input_dir}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Max distance: {args.max_distance}m, Max targets: {args.max_targets}")
        logger.info(f"  Max rows: {args.max_rows if args.max_rows else 'ALL'} (change to None for full processing)")
    else:
        input_dir = args.input_dir
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Process all CSV files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {input_dir}")

    for csv_file in tqdm(csv_files, desc="Processing files"):
        try:
            scene_id, output_file = process_ais_file(csv_file, output_dir, args.dataset_name, max_rows=args.max_rows)
            if output_file:
                logger.info(f"Processed {os.path.basename(csv_file)} -> {scene_id}")
            else:
                logger.warning(f"Skipped {csv_file} (no valid data)")
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {str(e)}")

    # Split data and create dataset files
    logger.info("\nSplitting data and creating dataset files...")
    train_summary, val_summary = split_data(output_dir, args.train_ratio)

    logger.info("\nProcessing complete!")
    logger.info(f"Train set: {train_summary['meta_info']['total_frames']} scenes")
    logger.info(f"Validation set: {val_summary['meta_info']['total_frames']} scenes")


if __name__ == "__main__":
    main()

# This V2 file is for handel the Danish AIS data which is different from the Orca AI InfluxDB format.