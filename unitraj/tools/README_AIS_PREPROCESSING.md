# AIS Data Preprocessing - Unified Pipeline

Complete pipeline for converting raw AIS data (Danish or InfluxDB formats) to Wayformer-compatible training scenarios.

## Overview

The unified preprocessing pipeline supports:
- **Danish AIS Format** (25 columns) - Auto-converts to InfluxDB format
- **InfluxDB 16-column Format** (own + target vessels)
- **InfluxDB 7-column Format** (own vessel only)
- **Automatic Format Detection**
- **Train/Val Split**
- **Wayformer Scenario Generation**

## File Organization

```
unitraj/tools/
├── ais_conversion_utils.py              # Shared utility functions
├── ais_data_preprocessor_v2.py          # ✨ NEW: Unified preprocessor (RECOMMENDED)
├── ais_data_preprocessor.py             # Original preprocessor (InfluxDB only)
├── convert_danish_ais_with_targets.py   # Standalone Danish converter (16-col)
├── convert_danish_ais_to_influx.py      # Standalone Danish converter (7-col)
├── test_danish_conversion.py            # Testing script
└── README_AIS_PREPROCESSING.md          # This file
```

## Quick Start

### Recommended: Use Unified Preprocessor

```bash
cd /home/aviv/Projects/UniTraj/unitraj/tools

# Process all files (auto-detects format)
python ais_data_preprocessor_v2.py \
  --input-dir /path/to/ais_csv_files \
  --output-dir /path/to/processed_scenarios \
  --dataset-name my_ais_dataset \
  --train-ratio 0.8

# For Danish data with custom settings
python ais_data_preprocessor_v2.py \
  --input-dir /path/to/danish_ais \
  --output-dir /path/to/output \
  --max-distance 10000 \
  --max-targets 20 \
  --train-ratio 0.8
```

### Output Structure

```
processed_scenarios/
├── train/
│   ├── ais_vessel1_20250101_120000/
│   │   └── ais_vessel1_20250101_120000.pkl
│   ├── ais_vessel2_20250101_120100/
│   │   └── ais_vessel2_20250101_120100.pkl
│   ├── dataset_summary.pkl
│   ├── dataset_mapping.pkl
│   └── file_list.pkl
└── val/
    ├── ais_vessel3_20250101_120200/
    │   └── ais_vessel3_20250101_120200.pkl
    ├── dataset_summary.pkl
    ├── dataset_mapping.pkl
    └── file_list.pkl
```

## Data Format Details

### Danish AIS Format (25 columns)

**Input**:
```csv
# Timestamp,Type of mobile,MMSI,Latitude,Longitude,Navigational status,ROT,SOG,COG,Heading,IMO,Callsign,Name,Ship type,Cargo type,Width,Length,Type of position fixing device,Draught,Destination,ETA,Data source type,A,B,C,D
27/02/2025 00:00:00,Class A,266220000,57.061732,9.967687,Under way using engine,,0.6,265.8,229,Unknown,Unknown,,Undefined,,,,Undefined,,Unknown,,AIS,,,,
```

**Auto-Conversion**: Converts to InfluxDB 16-column format with:
- Vessel-to-vessel relationships computed
- CPA/TCPA collision avoidance metrics
- Distance and bearing calculations
- Up to 20 nearest targets per vessel

### InfluxDB 16-Column Format (with targets)

**Format**:
```
time,own_latitude,own_longitude,host_name,own_sog,own_cog,own_rot,target_latitude,target_longitude,target_distance,target_sog,target_cog,target_cpa,target_tcpa,target_bearing,target_target_id
```

**Example**:
```
2025-02-27 00:00:00,55.965918,11.844717,219017815,0.0,64.7,0.0,55.96402,11.845398,215.26,0.0,165.7,215.26,0.0,168.64,(219021266)
```

### InfluxDB 7-Column Format (own vessel only)

**Format**:
```
time,own_latitude,own_longitude,host_name,own_sog,own_cog,own_rot
```

**Example**:
```
2025-03-15 06:00:00,29.29945,-136.17298,arc-commitment,17.5,102.0,-0.6
```

## Utility Functions (ais_conversion_utils.py)

### Geographic Calculations

```python
from ais_conversion_utils import (
    haversine_distance,
    calculate_bearing,
    calculate_cpa_tcpa,
    latlon_to_meters
)

# Distance between two points (meters)
dist = haversine_distance(lat1, lon1, lat2, lon2)

# Bearing from point1 to point2 (0-360 degrees)
bearing = calculate_bearing(lat1, lon1, lat2, lon2)

# Collision avoidance metrics
cpa, tcpa = calculate_cpa_tcpa(
    own_lat, own_lon, own_sog, own_cog,
    target_lat, target_lon, target_sog, target_cog
)

# Convert to relative meters from reference point
x_meters, y_meters = latlon_to_meters(lat, lon, ref_lat, ref_lon)
```

### Timestamp Parsing

```python
from ais_conversion_utils import parse_danish_timestamp, parse_influxdb_timestamp

# Danish format: DD/MM/YYYY HH:MM:SS -> YYYY-MM-DD HH:MM:SS
iso_time = parse_danish_timestamp("27/02/2025 00:00:00")

# InfluxDB format: Already ISO, just validates
iso_time = parse_influxdb_timestamp("2025-02-27 00:00:00")
```

### Velocity Conversions

```python
from ais_conversion_utils import knots_to_ms, course_speed_to_velocity

# Convert knots to m/s
speed_ms = knots_to_ms(17.5)  # 17.5 knots -> 9.0 m/s

# Get velocity components from course and speed
vx, vy = course_speed_to_velocity(course_degrees=102.0, speed_ms=9.0)
```

## Wayformer Scenario Format

Each scenario pickle file contains:

```python
{
    'scenario_id': 'ais_266220000_20250227_000000',
    'timestamps': np.array([0.0, 1.0, 2.0, ...]),  # Seconds from start
    'tracks': {
        '266220000': {  # Own vessel
            'object_type': 'VESSEL',
            'object_id': '266220000',
            'timestamps': np.array([...]),
            'state': {
                'position': np.array([[x1, y1], [x2, y2], ...]),  # Meters, relative
                'velocity': np.array([[vx1, vy1], [vx2, vy2], ...])  # m/s
            }
        },
        '219021266': {  # Target vessel
            'object_type': 'VESSEL',
            'object_id': '219021266',
            'timestamps': np.array([...]),
            'state': {
                'position': np.array([[...], [...]]),
                'velocity': np.array([[...], [...]])
            }
        }
    },
    'scenario_features': np.array([])  # Empty for AIS data
}
```

## Parameters

### ais_data_preprocessor_v2.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-dir` | `data/ais_data_from_influx_csv` | Directory with CSV files |
| `--output-dir` | `data/processed_ais_data` | Output directory |
| `--dataset-name` | `ais_dataset` | Dataset name for metadata |
| `--train-ratio` | `0.8` | Train/val split ratio |
| `--max-distance` | `10000` | Max distance for targets (meters) |
| `--max-targets` | `20` | Max targets per vessel |

### convert_danish_ais_with_targets.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | Required | Input Danish CSV file |
| `--output` | Required | Output InfluxDB CSV file |
| `--max-distance` | `10000` | Max distance for targets (meters) |
| `--max-targets` | `20` | Max targets per vessel |
| `--keep-base-stations` | `False` | Keep base station records |

## Processing Pipeline Flow

```
┌─────────────────────────┐
│  Raw AIS CSV Files      │
│  (Danish or InfluxDB)   │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Format Detection       │
│  (auto-detect type)     │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Danish Conversion      │◄── Only if Danish format
│  (to InfluxDB 16-col)   │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Scenario Generation    │
│  - Parse timestamps     │
│  - Extract trajectories │
│  - Convert to relative  │
│  - Compute velocities   │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Train/Val Split        │
│  (80/20 default)        │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Metadata Generation    │
│  - dataset_summary.pkl  │
│  - dataset_mapping.pkl  │
│  - file_list.pkl        │
└─────────────────────────┘
```

## Coordinate Systems

### Input: WGS84 Lat/Lon (Absolute)
- Latitude: -90° to +90° (North positive)
- Longitude: -180° to +180° (East positive)

### Scenario: Relative X/Y (Meters)
- Origin: First valid position of first agent
- X-axis: East (positive) / West (negative)
- Y-axis: North (positive) / South (negative)
- Conversion: ~111,320 meters per degree latitude
- Conversion: ~111,320 * cos(lat) meters per degree longitude

### Velocity Components
- Speed Over Ground (SOG): Knots → m/s (multiply by 0.514444)
- Course Over Ground (COG): 0° = North, 90° = East, 180° = South, 270° = West
- vx = speed * sin(course)  # East component
- vy = speed * cos(course)  # North component

## Maritime Metrics

### CPA (Closest Point of Approach)
Distance at which two vessels will be closest if they maintain current courses and speeds.

**Formula**:
```python
# Relative velocity
rel_vx = target_vx - own_vx
rel_vy = target_vy - own_vy

# Relative position
rel_x = (target_lon - own_lon) * lon_to_m
rel_y = (target_lat - own_lat) * lat_to_m

# Time to CPA
tcpa = -(rel_x * rel_vx + rel_y * rel_vy) / (rel_vx^2 + rel_vy^2)

# Distance at CPA
cpa = sqrt((rel_x + rel_vx*tcpa)^2 + (rel_y + rel_vy*tcpa)^2)
```

### TCPA (Time to CPA)
Time (in seconds) until the CPA occurs.

- TCPA > 0: Vessels approaching
- TCPA = 0: At CPA now
- TCPA < 0: Vessels diverging (set to 0)

## Integration with Training

### Update Config

Edit `unitraj/configs/config.yaml`:

```yaml
data:
  train_data:
    - ais_dataset_train
  val_data:
    - ais_dataset_val

  ais_dataset_train:
    type: 'ais_dataset'
    data_path: 'data/processed_ais_data/train'

  ais_dataset_val:
    type: 'ais_dataset'
    data_path: 'data/processed_ais_data/val'
```

### Train Model

```bash
cd /home/aviv/Projects/UniTraj/unitraj
python train.py method=wayformer_ais
```

## Performance Expectations

### Danish Conversion
- **Input**: 17M records (daily file)
- **Processing Time**: ~60 seconds
- **Throughput**: ~280K records/second
- **Output**: 16-column InfluxDB format with relationships

### Scenario Generation
- **Input**: 1000 CSV rows
- **Processing Time**: ~1-2 seconds
- **Output**: 1 scenario pickle file
- **Memory**: Processes row-by-row (memory efficient)

### Full Pipeline (100 CSV files, 1M records total)
- **Conversion**: ~5 minutes
- **Scenario Generation**: ~10 minutes
- **Train/Val Split**: ~1 minute
- **Total**: ~16 minutes

## Troubleshooting

### Issue: "Unknown format" detected
**Cause**: CSV columns don't match expected formats
**Solution**: Check column names with `head -1 file.csv` and verify format

### Issue: "No valid reference position"
**Cause**: All lat/lon values are NaN
**Solution**: Check data quality, filter out invalid records

### Issue: Memory usage high
**Cause**: Processing very large files
**Solution**: Use chunked processing (future enhancement) or split files

### Issue: Few target relationships generated
**Cause**: `max_distance` too small or vessels far apart
**Solution**: Increase `--max-distance` (default 10km = 5.4 nautical miles)

### Issue: Too many target relationships
**Cause**: Dense vessel traffic, `max_targets` too high
**Solution**: Reduce `--max-targets` or increase quality threshold

## Advanced Usage

### Batch Processing Multiple Directories

```bash
#!/bin/bash
# Process multiple data sources

for dir in danish_feb danish_mar danish_apr; do
    python ais_data_preprocessor_v2.py \
        --input-dir data/${dir}/csv \
        --output-dir data/processed/${dir} \
        --dataset-name ${dir}_dataset
done
```

### Custom Format Detection

```python
from ais_data_preprocessor_v2 import detect_format

# Check format before processing
format_type = detect_format('path/to/file.csv')
print(f"Detected format: {format_type}")
```

### Standalone Danish Conversion

```bash
# Convert single file
python convert_danish_ais_with_targets.py \
    --input data/danish/aisdk-2025-02-27.csv \
    --output data/converted/aisdk-2025-02-27_converted.csv \
    --max-distance 15000 \
    --max-targets 30
```

## Data Quality Checks

### Pre-Processing Validation

```bash
# Check file format
head -1 your_file.csv

# Count records
wc -l your_file.csv

# Check for missing values
grep -c ",,," your_file.csv

# Sample data quality
head -100 your_file.csv | tail -10
```

### Post-Processing Validation

```python
import pickle

# Check scenario file
with open('scenario.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Scenario ID: {data['scenario_id']}")
print(f"Num agents: {len(data['tracks'])}")
print(f"Num frames: {len(data['timestamps'])}")

# Check trajectory quality
for agent_id, track in data['tracks'].items():
    print(f"{agent_id}: {len(track['timestamps'])} frames")
    print(f"  Position range: {track['state']['position'].min()} to {track['state']['position'].max()}")
    print(f"  Velocity range: {track['state']['velocity'].min()} to {track['state']['velocity'].max()}")
```

## Next Steps

1. **Process Danish data**: Convert large Danish dataset to InfluxDB format
2. **Generate scenarios**: Run unified preprocessor on all CSV files
3. **Validate output**: Check pickle files and metadata
4. **Update config**: Point training to new dataset
5. **Train model**: Run Wayformer training with expanded data

## References

- **Haversine Formula**: https://en.wikipedia.org/wiki/Haversine_formula
- **CPA/TCPA Calculations**: Maritime collision avoidance regulations
- **AIS Standard**: ITU-R M.1371
- **WGS84 Coordinate System**: https://en.wikipedia.org/wiki/World_Geodetic_System
