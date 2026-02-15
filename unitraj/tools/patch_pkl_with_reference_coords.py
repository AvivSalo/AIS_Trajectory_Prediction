#!/usr/bin/env python3
"""
Patch existing pickle files to add reference_lat and reference_lon.
Extracts coordinates from original CSV files (local or EC2).

Naming convention:
- PKL: ais_{vessel-name}_{YYYYMMDD}_{HHMMSS}
- CSV: kepler_{vessel-name}_{YYYY-MM-DD}_{HHMM}_{YYYY-MM-DD}_{HHMM}_part-N.csv

The PKL timestamp comes from df['time'].min() in the CSV.
"""

import pickle
import sys
from pathlib import Path
import argparse
import pandas as pd
import subprocess
import tempfile


def parse_scenario_id(scenario_id):
    """Parse scenario_id to extract vessel name, date, and time.

    Example: ais_shell-macoma_20240324_200000
    Returns: ('shell-macoma', '2024-03-24', '2000')
    """
    parts = scenario_id.replace('ais_', '').split('_')
    if len(parts) < 3:
        return None

    vessel_name = '_'.join(parts[:-2])
    date_str = parts[-2]  # YYYYMMDD
    time_str = parts[-1]  # HHMMSS

    # Convert date format: YYYYMMDD → YYYY-MM-DD
    if len(date_str) == 8 and date_str.isdigit():
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    else:
        return None

    # Convert time format: HHMMSS → HHMM (for CSV filename matching)
    if len(time_str) >= 4:
        formatted_time = time_str[:4]
    else:
        return None

    return vessel_name, formatted_date, formatted_time


def find_matching_csv_ec2(scenario_id, csv_dir, ssh_key, ec2_host):
    """Find matching CSV file on EC2 for the given scenario_id."""
    parsed = parse_scenario_id(scenario_id)
    if not parsed:
        print(f"  ⚠️  Could not parse scenario_id: {scenario_id}")
        return None

    vessel_name, date, time = parsed

    # Build search pattern: kepler_{vessel}_{date}_{time}*.csv
    pattern = f"kepler_{vessel_name}_{date}_{time}*.csv"
    remote_search_cmd = f"ls {csv_dir}/{pattern} 2>/dev/null"

    try:
        cmd = f"ssh -i {ssh_key} ubuntu@{ec2_host} \"{remote_search_cmd}\""
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and result.stdout.strip():
            csv_files = result.stdout.strip().split('\n')
            if csv_files:
                csv_path = csv_files[0]
                print(f"  Found CSV: {Path(csv_path).name}")
                return csv_path

        print(f"  ⚠️  No CSV found for pattern: {pattern}")
        return None

    except Exception as e:
        print(f"  ⚠️  Error searching CSV: {e}")
        return None


def get_reference_from_csv_ec2(csv_path, ssh_key, ec2_host):
    """Extract reference coordinates from CSV file on EC2."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        tmp_path = tmp.name

    try:
        # Copy CSV from EC2
        cmd = f"scp -i {ssh_key} ubuntu@{ec2_host}:{csv_path} {tmp_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=30)

        if result.returncode != 0:
            print(f"  ⚠️  Failed to copy CSV from EC2")
            Path(tmp_path).unlink(missing_ok=True)
            return None

        # Read CSV and get first valid coordinates
        df = pd.read_csv(tmp_path)

        if 'own_latitude' in df.columns and 'own_longitude' in df.columns:
            for _, row in df.iterrows():
                lat = row['own_latitude']
                lon = row['own_longitude']
                if not (pd.isna(lat) or pd.isna(lon)):
                    print(f"  ✅ Extracted: lat={lat:.6f}, lon={lon:.6f}")
                    Path(tmp_path).unlink(missing_ok=True)
                    return float(lat), float(lon)

        print(f"  ⚠️  No valid coordinates in CSV")
        Path(tmp_path).unlink(missing_ok=True)
        return None

    except Exception as e:
        print(f"  ⚠️  Error reading CSV: {e}")
        Path(tmp_path).unlink(missing_ok=True)
        return None


def get_reference_from_csv_local(csv_path):
    """Extract reference coordinates from local CSV file."""
    try:
        df = pd.read_csv(csv_path)

        if 'own_latitude' in df.columns and 'own_longitude' in df.columns:
            for _, row in df.iterrows():
                lat = row['own_latitude']
                lon = row['own_longitude']
                if not (pd.isna(lat) or pd.isna(lon)):
                    print(f"  ✅ Extracted: lat={lat:.6f}, lon={lon:.6f}")
                    return float(lat), float(lon)

        print(f"  ⚠️  No valid coordinates in CSV")
        return None

    except Exception as e:
        print(f"  ⚠️  Error reading CSV: {e}")
        return None


def patch_pickle_file(pkl_path, csv_dir=None, ssh_key=None, ec2_host=None, use_ec2=False):
    """Add reference coordinates to a pickle file."""

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Check if already has reference coordinates
    if 'reference_lat' in data and 'reference_lon' in data:
        print(f"✓ Already has coords: {pkl_path.name}")
        return True, "already_patched"

    scenario_id = data['scenario_id']
    print(f"\n📍 Patching: {scenario_id}")

    ref_coords = None

    # Try to get from CSV
    if use_ec2 and ssh_key and ec2_host and csv_dir:
        # EC2 mode: find and fetch CSV from EC2
        csv_path = find_matching_csv_ec2(scenario_id, csv_dir, ssh_key, ec2_host)
        if csv_path:
            ref_coords = get_reference_from_csv_ec2(csv_path, ssh_key, ec2_host)
    elif csv_dir:
        # Local mode: find CSV locally
        parsed = parse_scenario_id(scenario_id)
        if parsed:
            vessel_name, date, time = parsed
            pattern = f"kepler_{vessel_name}_{date}_{time}*.csv"
            csv_files = list(Path(csv_dir).glob(pattern))
            if csv_files:
                print(f"  Found CSV: {csv_files[0].name}")
                ref_coords = get_reference_from_csv_local(csv_files[0])

    # Use default if CSV extraction failed
    if ref_coords is None:
        print(f"  ⚠️  Using default Mediterranean coordinates")
        ref_coords = (31.833351, 34.618101)
        status = "default_coords"
    else:
        status = "success"

    # Add reference coordinates to pickle
    data['reference_lat'] = float(ref_coords[0])
    data['reference_lon'] = float(ref_coords[1])

    # Save back
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"  ✅ Patched: lat={ref_coords[0]:.6f}, lon={ref_coords[1]:.6f}")
    return True, status


def main():
    parser = argparse.ArgumentParser(
        description='Patch pickle files with reference coordinates from CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Patch from EC2 CSVs
  python patch_pkl_with_reference_coords.py \\
    /path/to/val_10_scenes_from_ec2 \\
    --csv-dir /home/ubuntu/projects/AIS_Trajectory_Prediction/unitraj/data/ais_4hours_sample \\
    --ssh-key /path/to/key.pem \\
    --ec2-host ec2-x-x-x-x.compute-1.amazonaws.com \\
    --use-ec2

  # Patch from local CSVs
  python patch_pkl_with_reference_coords.py \\
    /path/to/pickles \\
    --csv-dir /path/to/local/csvs
        """
    )
    parser.add_argument('data_dir', type=str, help='Directory containing scenario subdirectories')
    parser.add_argument('--csv-dir', type=str, help='Directory containing CSV files (local or EC2 path)')
    parser.add_argument('--ssh-key', type=str, help='SSH key for EC2 access')
    parser.add_argument('--ec2-host', type=str, help='EC2 host address')
    parser.add_argument('--use-ec2', action='store_true', help='Fetch CSVs from EC2')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return 1

    # Find all pickle files
    pkl_files = []
    for scenario_dir in data_dir.iterdir():
        if scenario_dir.is_dir() and scenario_dir.name.startswith('ais_'):
            pkl_path = scenario_dir / f"{scenario_dir.name}.pkl"
            if pkl_path.exists():
                pkl_files.append(pkl_path)

    print(f"Found {len(pkl_files)} pickle files to patch")
    print(f"CSV source: {'EC2' if args.use_ec2 else 'Local'} ({args.csv_dir})")
    print("="*70)

    stats = {'success': 0, 'already_patched': 0, 'default_coords': 0, 'failed': 0}

    for pkl_path in pkl_files:
        try:
            success, status = patch_pickle_file(
                pkl_path,
                csv_dir=args.csv_dir,
                ssh_key=args.ssh_key,
                ec2_host=args.ec2_host,
                use_ec2=args.use_ec2
            )
            if success:
                stats[status] += 1
            else:
                stats['failed'] += 1
        except Exception as e:
            print(f"❌ Error patching {pkl_path.name}: {e}")
            stats['failed'] += 1

    print("="*70)
    print(f"✅ Patched with real coordinates: {stats['success']}")
    print(f"⚠️  Patched with default coordinates: {stats['default_coords']}")
    print(f"✓  Already had coordinates: {stats['already_patched']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"Total: {len(pkl_files)}")

    if stats['success'] > 0:
        print("\n🎯 Next steps:")
        print("   1. Run evaluation to verify correct visualization")
        print("   2. Check that vessels appear at proper geographic locations")

    return 0


if __name__ == '__main__':
    sys.exit(main())
