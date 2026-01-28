#!/usr/bin/env python3
"""
Convert Danish AIS CSV to full InfluxDB format with target vessel relationships.

This creates the 16-column format:
    time, own_latitude, own_longitude, host_name, own_sog, own_cog, own_rot,
    target_latitude, target_longitude, target_distance, target_sog, target_cog,
    target_cpa, target_tcpa, target_bearing, target_target_id

Each row represents one vessel (own) observing another nearby vessel (target).
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters using Haversine formula."""
    R = 6371000  # Earth radius in meters

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point 1 to point 2 in degrees (0-360)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearing = np.arctan2(x, y)
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


def calculate_cpa_tcpa(own_lat, own_lon, own_sog, own_cog,
                        target_lat, target_lon, target_sog, target_cog):
    """
    Calculate Closest Point of Approach (CPA) and Time to CPA (TCPA).

    Returns:
        cpa: Distance at closest approach (meters)
        tcpa: Time to closest approach (seconds)
    """
    # Convert speeds from knots to m/s
    own_v = own_sog * 0.514444
    target_v = target_sog * 0.514444

    # Convert headings to velocity components
    own_vx = own_v * np.sin(np.radians(own_cog))
    own_vy = own_v * np.cos(np.radians(own_cog))
    target_vx = target_v * np.sin(np.radians(target_cog))
    target_vy = target_v * np.cos(np.radians(target_cog))

    # Relative velocity
    rel_vx = target_vx - own_vx
    rel_vy = target_vy - own_vy
    rel_v_squared = rel_vx**2 + rel_vy**2

    # If relative velocity is near zero, vessels are moving together
    if rel_v_squared < 1e-6:
        distance = haversine_distance(own_lat, own_lon, target_lat, target_lon)
        return distance, 0.0

    # Convert lat/lon to relative position (approximate for small distances)
    lat_to_m = 111320  # meters per degree latitude
    lon_to_m = 111320 * np.cos(np.radians((own_lat + target_lat) / 2))

    rel_x = (target_lon - own_lon) * lon_to_m
    rel_y = (target_lat - own_lat) * lat_to_m

    # Time to CPA: when relative position is perpendicular to relative velocity
    tcpa = -(rel_x * rel_vx + rel_y * rel_vy) / rel_v_squared

    # If TCPA is negative, CPA is in the past (vessels diverging)
    if tcpa < 0:
        tcpa = 0
        cpa = haversine_distance(own_lat, own_lon, target_lat, target_lon)
    else:
        # Calculate position at TCPA
        cpa_x = rel_x + rel_vx * tcpa
        cpa_y = rel_y + rel_vy * tcpa
        cpa = np.sqrt(cpa_x**2 + cpa_y**2)

    return cpa, tcpa


def parse_danish_timestamp(timestamp_str):
    """Convert Danish timestamp to ISO format."""
    try:
        dt = datetime.strptime(timestamp_str, "%d/%m/%Y %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return None


def convert_danish_to_influx_with_targets(input_path, output_path,
                                           max_distance=10000,
                                           max_targets_per_vessel=20,
                                           keep_base_stations=False):
    """
    Convert Danish AIS to InfluxDB format with target vessel relationships.

    Args:
        input_path: Danish AIS CSV file
        output_path: Output InfluxDB format CSV
        max_distance: Maximum distance (meters) to consider vessels as targets
        max_targets_per_vessel: Max targets to include per own vessel per timestamp
        keep_base_stations: Keep base station records
    """
    logger.info(f"Reading Danish AIS data from: {input_path}")

    # Read Danish CSV
    df = pd.read_csv(input_path, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lstrip('#').str.strip()

    initial_count = len(df)
    logger.info(f"Loaded {initial_count} records")

    # Filter base stations
    if not keep_base_stations and 'Type of mobile' in df.columns:
        df = df[df['Type of mobile'] != 'Base Station']
        logger.info(f"Filtered base stations: {initial_count - len(df)} removed")

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

    logger.info(f"Processing {len(timestamps)} unique timestamps")

    for timestamp in timestamps:
        time_group = df[df['parsed_time'] == timestamp].copy()

        if len(time_group) < 2:
            # Single vessel at this time - add with no target
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
                    continue  # Skip self

                target_lat = target['Latitude']
                target_lon = target['Longitude']
                target_sog = target['SOG'] if pd.notna(target['SOG']) else 0.0
                target_cog = target['COG'] if pd.notna(target['COG']) else (target['Heading'] if pd.notna(target['Heading']) else 0.0)

                # Calculate distance
                distance = haversine_distance(own_lat, own_lon, target_lat, target_lon)

                if distance <= max_distance:
                    # Calculate bearing
                    bearing = calculate_bearing(own_lat, own_lon, target_lat, target_lon)

                    # Calculate CPA and TCPA
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

            # Sort targets by distance and take closest ones
            targets.sort(key=lambda x: x['distance'])
            targets = targets[:max_targets_per_vessel]

            if not targets:
                # No nearby targets - add row with NaN targets
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
                # Add one row per target
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

    logger.info(f"Generated {len(output_df)} rows with target relationships")
    logger.info(f"  Own-target pairs: {output_df['target_target_id'].notna().sum()}")
    logger.info(f"  Single vessels: {output_df['target_target_id'].isna().sum()}")

    # Write without header (matching existing format)
    output_df.to_csv(output_path, index=False, header=False)
    logger.info(f"Written to: {output_path}")

    # Statistics
    if len(output_df) > 0:
        logger.info("Conversion Statistics:")
        logger.info(f"  Unique own vessels: {output_df['host_name'].nunique()}")
        logger.info(f"  Unique target vessels: {output_df[output_df['target_target_id'].notna()]['target_target_id'].nunique()}")
        logger.info(f"  Avg targets per vessel: {output_df.groupby('host_name').size().mean():.2f}")
        logger.info(f"  Time range: {output_df['time'].min()} to {output_df['time'].max()}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Danish AIS to InfluxDB format with target vessels'
    )
    parser.add_argument('--input', type=str, help='Input Danish AIS CSV')
    parser.add_argument('--output', type=str, help='Output InfluxDB CSV')
    parser.add_argument('--max-distance', type=float, default=10000,
                        help='Max distance (m) for target vessels (default: 10km)')
    parser.add_argument('--max-targets', type=int, default=20,
                        help='Max targets per vessel (default: 20)')
    parser.add_argument('--keep-base-stations', action='store_true',
                        help='Keep base station records')

    args = parser.parse_args()

    if not args.input or not args.output:
        parser.print_help()
        logger.error("\nError: Must provide --input and --output paths")
        return

    convert_danish_to_influx_with_targets(
        args.input,
        args.output,
        max_distance=args.max_distance,
        max_targets_per_vessel=args.max_targets,
        keep_base_stations=args.keep_base_stations
    )


if __name__ == '__main__':
    main()