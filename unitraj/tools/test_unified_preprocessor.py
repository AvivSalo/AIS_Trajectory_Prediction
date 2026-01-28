#!/usr/bin/env python3
"""
Test script for unified AIS data preprocessor.

Validates:
- Format detection (Danish, InfluxDB 16-col, InfluxDB 7-col)
- Danish to InfluxDB conversion
- Scenario generation from all formats
- Train/val split
- Metadata file generation
"""

import os
import sys
import tempfile
import shutil
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ais_data_preprocessor_v2 import (
    detect_format,
    convert_danish_to_influxdb,
    process_ais_file,
    split_data
)


def create_test_danish_csv(output_path):
    """Create a small test Danish AIS CSV."""
    data = """# Timestamp,Type of mobile,MMSI,Latitude,Longitude,Navigational status,ROT,SOG,COG,Heading,IMO,Callsign,Name,Ship type,Cargo type,Width,Length,Type of position fixing device,Draught,Destination,ETA,Data source type,A,B,C,D
27/02/2025 00:00:00,Class A,266220000,57.061732,9.967687,Under way using engine,,0.6,265.8,229,,,,,,,,,,,AIS,,,,
27/02/2025 00:00:01,Class A,266220000,57.061733,9.967690,Under way using engine,,0.6,265.8,229,,,,,,,,,,,AIS,,,,
27/02/2025 00:00:02,Class A,266220000,57.061734,9.967693,Under way using engine,,0.6,265.8,229,,,,,,,,,,,AIS,,,,
27/02/2025 00:00:00,Class A,219021266,57.062000,9.968000,Under way using engine,,1.2,180.0,180,,,,,,,,,,,AIS,,,,
27/02/2025 00:00:01,Class A,219021266,57.062001,9.968000,Under way using engine,,1.2,180.0,180,,,,,,,,,,,AIS,,,,
27/02/2025 00:00:02,Class A,219021266,57.062002,9.968000,Under way using engine,,1.2,180.0,180,,,,,,,,,,,AIS,,,,
"""
    with open(output_path, 'w') as f:
        f.write(data)
    print(f"✅ Created test Danish CSV: {output_path}")


def create_test_influxdb_16col_csv(output_path):
    """Create a small test InfluxDB 16-column CSV."""
    data = """2025-03-15 00:00:00,29.2994,-136.1729,arc-commitment,17.5,102.0,-0.6,29.3010,-136.1710,200.5,15.2,98.3,195.2,10.5,85.2,(vessel-2)
2025-03-15 00:00:01,29.2995,-136.1728,arc-commitment,17.5,102.0,-0.6,29.3011,-136.1709,199.8,15.2,98.3,194.5,9.5,85.1,(vessel-2)
2025-03-15 00:00:02,29.2996,-136.1727,arc-commitment,17.5,102.0,-0.6,29.3012,-136.1708,199.1,15.2,98.3,193.8,8.5,85.0,(vessel-2)
"""
    with open(output_path, 'w') as f:
        f.write(data)
    print(f"✅ Created test InfluxDB 16-col CSV: {output_path}")


def create_test_influxdb_7col_csv(output_path):
    """Create a small test InfluxDB 7-column CSV."""
    data = """2025-03-15 00:00:00,29.2994,-136.1729,arc-commitment,17.5,102.0,-0.6
2025-03-15 00:00:01,29.2995,-136.1728,arc-commitment,17.5,102.0,-0.6
2025-03-15 00:00:02,29.2996,-136.1727,arc-commitment,17.5,102.0,-0.6
"""
    with open(output_path, 'w') as f:
        f.write(data)
    print(f"✅ Created test InfluxDB 7-col CSV: {output_path}")


def test_format_detection():
    """Test automatic format detection."""
    print("\n" + "="*80)
    print("TEST 1: Format Detection")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test Danish format
        danish_file = os.path.join(tmpdir, "danish_test.csv")
        create_test_danish_csv(danish_file)
        format_type = detect_format(danish_file)
        assert format_type == 'danish', f"Expected 'danish', got '{format_type}'"
        print(f"✅ Danish format detected correctly")

        # Test InfluxDB 16-col format
        influx16_file = os.path.join(tmpdir, "influx16_test.csv")
        create_test_influxdb_16col_csv(influx16_file)
        format_type = detect_format(influx16_file)
        assert format_type == 'influxdb_16col', f"Expected 'influxdb_16col', got '{format_type}'"
        print(f"✅ InfluxDB 16-column format detected correctly")

        # Test InfluxDB 7-col format
        influx7_file = os.path.join(tmpdir, "influx7_test.csv")
        create_test_influxdb_7col_csv(influx7_file)
        format_type = detect_format(influx7_file)
        assert format_type == 'influxdb_7col', f"Expected 'influxdb_7col', got '{format_type}'"
        print(f"✅ InfluxDB 7-column format detected correctly")

    print("✅ All format detection tests passed!")


def test_danish_conversion():
    """Test Danish to InfluxDB conversion."""
    print("\n" + "="*80)
    print("TEST 2: Danish to InfluxDB Conversion")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        danish_file = os.path.join(tmpdir, "danish_test.csv")
        output_file = os.path.join(tmpdir, "converted.csv")

        create_test_danish_csv(danish_file)
        convert_danish_to_influxdb(danish_file, output_file)

        # Validate output
        assert os.path.exists(output_file), "Output file not created"

        df = pd.read_csv(output_file, header=None)
        assert len(df.columns) == 16, f"Expected 16 columns, got {len(df.columns)}"
        assert len(df) > 0, "No data in output file"

        print(f"✅ Converted {len(df)} rows")
        print(f"✅ Output has correct 16-column format")

        # Check for target relationships
        df.columns = ['time', 'own_lat', 'own_lon', 'host_name', 'own_sog', 'own_cog', 'own_rot',
                       'target_lat', 'target_lon', 'target_dist', 'target_sog', 'target_cog',
                       'target_cpa', 'target_tcpa', 'target_bearing', 'target_id']

        with_targets = df['target_id'].notna().sum()
        print(f"✅ Generated {with_targets} rows with target relationships")

    print("✅ Danish conversion test passed!")


def test_scenario_generation():
    """Test scenario generation from InfluxDB format."""
    print("\n" + "="*80)
    print("TEST 3: Scenario Generation")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        influx_file = os.path.join(tmpdir, "influx_test.csv")
        output_dir = os.path.join(tmpdir, "scenarios")

        create_test_influxdb_16col_csv(influx_file)
        os.makedirs(output_dir, exist_ok=True)

        scene_id, scenario_file = process_ais_file(influx_file, output_dir)

        assert scene_id is not None, "No scenario ID returned"
        assert os.path.exists(scenario_file), f"Scenario file not created: {scenario_file}"

        # Load and validate pickle
        with open(scenario_file, 'rb') as f:
            data = pickle.load(f)

        assert 'scenario_id' in data, "Missing scenario_id"
        assert 'tracks' in data, "Missing tracks"
        assert 'timestamps' in data, "Missing timestamps"

        num_agents = len(data['tracks'])
        print(f"✅ Generated scenario: {scene_id}")
        print(f"✅ Number of agents: {num_agents}")
        print(f"✅ Number of timestamps: {len(data['timestamps'])}")

        # Validate track structure
        for agent_id, track in data['tracks'].items():
            assert 'object_type' in track, f"Missing object_type for {agent_id}"
            assert 'state' in track, f"Missing state for {agent_id}"
            assert 'position' in track['state'], f"Missing position for {agent_id}"
            assert 'velocity' in track['state'], f"Missing velocity for {agent_id}"
            print(f"✅ Agent {agent_id}: {len(track['timestamps'])} frames")

    print("✅ Scenario generation test passed!")


def test_train_val_split():
    """Test train/validation split."""
    print("\n" + "="*80)
    print("TEST 4: Train/Val Split")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "scenarios")
        os.makedirs(output_dir, exist_ok=True)

        # Generate multiple scenarios
        for i in range(10):
            influx_file = os.path.join(tmpdir, f"influx_test_{i}.csv")
            create_test_influxdb_16col_csv(influx_file)
            process_ais_file(influx_file, output_dir)

        # Perform split
        train_summary, val_summary = split_data(output_dir, train_ratio=0.8)

        train_count = train_summary['meta_info']['total_frames']
        val_count = val_summary['meta_info']['total_frames']

        print(f"✅ Train scenarios: {train_count}")
        print(f"✅ Val scenarios: {val_count}")
        print(f"✅ Split ratio: {train_count / (train_count + val_count):.2f}")

        # Validate metadata files
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')

        assert os.path.exists(os.path.join(train_dir, 'dataset_summary.pkl')), "Missing train summary"
        assert os.path.exists(os.path.join(train_dir, 'dataset_mapping.pkl')), "Missing train mapping"
        assert os.path.exists(os.path.join(train_dir, 'file_list.pkl')), "Missing train file list"

        assert os.path.exists(os.path.join(val_dir, 'dataset_summary.pkl')), "Missing val summary"
        assert os.path.exists(os.path.join(val_dir, 'dataset_mapping.pkl')), "Missing val mapping"
        assert os.path.exists(os.path.join(val_dir, 'file_list.pkl')), "Missing val file list"

        print(f"✅ All metadata files generated")

    print("✅ Train/val split test passed!")


def test_end_to_end():
    """Test complete end-to-end pipeline."""
    print("\n" + "="*80)
    print("TEST 5: End-to-End Pipeline")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(input_dir, exist_ok=True)

        # Create mixed format test data
        create_test_danish_csv(os.path.join(input_dir, "danish_1.csv"))
        create_test_danish_csv(os.path.join(input_dir, "danish_2.csv"))
        create_test_influxdb_16col_csv(os.path.join(input_dir, "influx16_1.csv"))
        create_test_influxdb_7col_csv(os.path.join(input_dir, "influx7_1.csv"))

        # Import and run main pipeline
        from ais_data_preprocessor_v2 import main as preprocess_main
        import sys

        # Mock command line args
        old_argv = sys.argv
        sys.argv = [
            'ais_data_preprocessor_v2.py',
            '--input-dir', input_dir,
            '--output-dir', output_dir,
            '--dataset-name', 'test_dataset',
            '--train-ratio', '0.8'
        ]

        try:
            preprocess_main()
        finally:
            sys.argv = old_argv

        # Validate output structure
        assert os.path.exists(os.path.join(output_dir, 'train')), "Missing train directory"
        assert os.path.exists(os.path.join(output_dir, 'val')), "Missing val directory"

        print(f"✅ End-to-end pipeline completed successfully")

    print("✅ End-to-end test passed!")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("UNIFIED AIS PREPROCESSOR TEST SUITE")
    print("="*80)

    try:
        test_format_detection()
        test_danish_conversion()
        test_scenario_generation()
        test_train_val_split()
        test_end_to_end()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
