#!/usr/bin/env python3
"""
Create timeline visualizations showing complete vessel trajectories with overlapping predictions.

This script loads original pickle files to get complete ground truth trajectories,
then overlays predictions from different time windows as semi-transparent "fans"
to show how prediction accuracy evolves over time.
"""

import os
import pickle
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import logging
import math

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


def load_pickle_file(pickle_path):
    """Load a pickle file containing scenario data."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def group_html_files_by_scenario(viz_dir):
    """Group HTML visualization files by base scenario (vessel and date)."""
    viz_path = Path(viz_dir)
    html_files = list(viz_path.glob("ais_*.html"))

    # Pattern: ais_{vessel}_{date}_t{time_offset}.html
    import re
    pattern = r'ais_(.+)_(\d{8}_\d{6})_t(\d+)\.html'

    scenarios = defaultdict(list)

    for html_file in html_files:
        match = re.match(pattern, html_file.name)
        if match:
            vessel, date, time_offset = match.groups()
            scenario_key = f"{vessel}_{date}"
            scenarios[scenario_key].append({
                'time_offset': int(time_offset),
                'file': html_file
            })

    # Sort each scenario's time windows by time_offset
    for scenario_key in scenarios:
        scenarios[scenario_key].sort(key=lambda x: x['time_offset'])

    return scenarios


def parse_prediction_from_html(html_path):
    """Extract prediction coordinates from an HTML visualization file."""
    import re

    with open(html_path, 'r') as f:
        content = f.read()

    # Extract prediction coordinates for all vessels
    pred_pattern = r'var vessel(\\d+)_pred_coords = (\\[\\[.+?\\]\\]);'
    pred_matches = re.findall(pred_pattern, content)

    predictions = {}
    for vessel_idx, coords_str in pred_matches:
        try:
            coords = json.loads(coords_str)
            if coords:  # Only add if not empty
                predictions[int(vessel_idx)] = coords
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse prediction coords for vessel {vessel_idx}: {e}")

    return predictions


def extract_reference_point_from_html(html_path):
    """
    Extract the reference lat/lon point from HTML file.

    The first ground truth coordinate in the HTML represents the starting point
    which corresponds to (0,0) in the pickle relative coordinates.
    """
    import re

    with open(html_path, 'r') as f:
        content = f.read()

    # Extract first ground truth coordinate (vessel 0, first point)
    gt_pattern = r'var vessel0_gt_coords = (\[\[.+?\]\]);'
    match = re.search(gt_pattern, content)

    if match:
        try:
            coords = json.loads(match.group(1))
            if coords and len(coords) > 0:
                # First coordinate is the reference point
                return coords[0][0], coords[0][1]  # (lat, lon)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse GT coords: {e}")

    return None


def find_matching_pickle_file(scenario_key, data_dirs):
    """
    Find the pickle file that matches the scenario key.

    scenario_key format: vessel_YYYYMMDD_HHMMSS (may have scenario_ais_ prefix to strip)
    pickle file format: data/processed_ais_data/val/ais_{scenario}/ais_{scenario}.pkl
    """
    # Strip "scenario_ais_" prefix if present (some HTML files have this)
    clean_key = scenario_key.replace('scenario_ais_', '')

    # Extract vessel and datetime from scenario_key
    parts = clean_key.split('_')
    if len(parts) < 3:
        return None

    vessel_name = '_'.join(parts[:-2])  # Everything before date and time
    date_str = parts[-2]  # YYYYMMDD
    time_str = parts[-1]  # HHMMSS

    # Try different pickle file patterns
    for data_dir in data_dirs:
        data_path = Path(data_dir)

        # Pattern 1: ais_{vessel}_{date}_{time}/ais_{vessel}_{date}_{time}.pkl
        scenario_name = f"ais_{clean_key}"
        pickle_path = data_path / scenario_name / f"{scenario_name}.pkl"
        if pickle_path.exists():
            logger.info(f"Found pickle file: {pickle_path}")
            return pickle_path

        # Pattern 2: Direct pickle file in root
        pickle_path = data_path / f"{scenario_name}.pkl"
        if pickle_path.exists():
            logger.info(f"Found pickle file: {pickle_path}")
            return pickle_path

        # Pattern 3: Search recursively for any matching pickle
        matching_files = list(data_path.glob(f"**/*{vessel_name}*{date_str}*{time_str}*.pkl"))
        if matching_files:
            # Filter out metadata files
            matching_files = [f for f in matching_files if f.name not in ["dataset_mapping.pkl", "dataset_summary.pkl", "file_list.pkl"]]
            if matching_files:
                logger.info(f"Found pickle file: {matching_files[0]}")
                return matching_files[0]

    logger.warning(f"Could not find pickle file for scenario: {scenario_key} (cleaned: {clean_key})")
    return None


def extract_complete_trajectories(pickle_data):
    """
    Extract complete ground truth trajectories from pickle data.

    Returns:
        tuple: (trajectories_dict, reference_point)
            - trajectories_dict: {vessel_idx: [(lat, lon), ...]} with complete trajectories
            - reference_point: (ref_lat, ref_lon) - the starting location to use for conversion
    """
    trajectories = {}
    reference_point = None

    # The pickle data structure:
    # pickle_data['tracks'] = {vessel_id: {'state': {'position': [[x, y], ...]}}}
    # Position is in relative meters, need to convert to lat/lon

    if 'tracks' in pickle_data:
        tracks = pickle_data['tracks']
        vessel_idx = 0

        for vessel_id, track_data in tracks.items():
            # Skip invalid vessel IDs
            if vessel_id == 'nan' or not isinstance(track_data, dict):
                continue

            if 'state' in track_data and 'position' in track_data['state']:
                positions = track_data['state']['position']  # [[x, y], ...]

                # Use the first valid position as the reference point (origin)
                # This represents where the trajectory starts
                if reference_point is None and len(positions) > 0:
                    # Find first non-zero position or use (0,0) location
                    # The (0,0) point in relative coordinates IS the reference location
                    # We need to extract the actual lat/lon reference from somewhere
                    # For now, we'll work backwards from the HTML visualizations
                    # which have the correct absolute coordinates
                    pass

                # For now, just store the relative positions
                # We'll convert them using the reference from HTML
                trajectories[vessel_idx] = positions.tolist()
                vessel_idx += 1

    return trajectories, reference_point


def meters_to_latlon(x_meters, y_meters, ref_lat, ref_lon):
    """Convert relative meters to lat/lon using reference point."""
    # Conversion factors
    meters_per_deg_lat = 110540.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(ref_lat))

    lat_diff = y_meters / meters_per_deg_lat
    lon_diff = x_meters / meters_per_deg_lon

    lat = ref_lat + lat_diff
    lon = ref_lon + lon_diff

    return lat, lon


def create_timeline_visualization(scenario_key, time_windows, pickle_trajectories, output_dir):
    """
    Create a combined timeline visualization for a scenario.

    Args:
        scenario_key: Scenario identifier
        time_windows: List of dicts with 'time_offset' and 'file' keys
        pickle_trajectories: Dict of {vessel_idx: [[x_meters, y_meters], ...]} relative positions
        output_dir: Output directory for HTML file
    """
    logger.info(f"Creating timeline visualization for {scenario_key} with {len(time_windows)} time windows")

    # Extract reference point from first HTML file
    reference_point = extract_reference_point_from_html(time_windows[0]['file'])
    if reference_point is None:
        logger.warning(f"Could not extract reference point for {scenario_key}")
        return

    ref_lat, ref_lon = reference_point
    logger.info(f"Using reference point: lat={ref_lat:.6f}, lon={ref_lon:.6f}")

    # Convert pickle trajectories from relative meters to lat/lon
    complete_gt_trajectories = {}
    for vessel_idx, rel_positions in pickle_trajectories.items():
        latlon_coords = []
        for x, y in rel_positions:
            lat, lon = meters_to_latlon(x, y, ref_lat, ref_lon)
            latlon_coords.append([lat, lon])
        complete_gt_trajectories[vessel_idx] = latlon_coords

    # Parse predictions from all time windows
    all_pred_coords = defaultdict(list)  # vessel_idx -> list of (time_offset, coordinates)

    for window in time_windows:
        predictions = parse_prediction_from_html(window['file'])

        # Collect predictions with time offset
        for vessel_idx, coords in predictions.items():
            all_pred_coords[vessel_idx].append({
                'time_offset': window['time_offset'],
                'coords': coords
            })

    # Calculate center point for map from complete ground truth
    all_coords_flat = []
    for coords in complete_gt_trajectories.values():
        all_coords_flat.extend(coords)

    if not all_coords_flat:
        logger.warning(f"No coordinates found for {scenario_key}")
        return

    center_lat = sum(c[0] for c in all_coords_flat) / len(all_coords_flat)
    center_lon = sum(c[1] for c in all_coords_flat) / len(all_coords_flat)

    # Generate HTML
    html = generate_timeline_html(
        scenario_key=scenario_key,
        center_lat=center_lat,
        center_lon=center_lon,
        ground_truth=complete_gt_trajectories,
        predictions=all_pred_coords,
        time_windows=time_windows
    )

    # Write output file
    output_path = Path(output_dir) / f"{scenario_key}_timeline.html"
    with open(output_path, 'w') as f:
        f.write(html)

    logger.info(f"Created timeline visualization: {output_path}")


def generate_timeline_html(scenario_key, center_lat, center_lon, ground_truth, predictions, time_windows):
    """Generate HTML for timeline visualization."""

    vessel_colors = ['#228B22', '#4169E1', '#DC143C', '#FF8C00', '#9932CC', '#FF1493', '#00CED1', '#32CD32']

    # Generate ground truth polylines (complete trajectories)
    gt_js = ""
    for vessel_idx, coords in sorted(ground_truth.items()):
        color = vessel_colors[vessel_idx % len(vessel_colors)]
        coords_json = json.dumps(coords)

        gt_js += f"""
        // Vessel {vessel_idx + 1} Complete Ground Truth
        var vessel{vessel_idx}_complete_gt = {coords_json};
        if (vessel{vessel_idx}_complete_gt.length > 0) {{
            var vessel{vessel_idx}_gt_line = L.polyline(vessel{vessel_idx}_complete_gt, {{
                color: '{color}',
                weight: 3,
                opacity: 0.9
            }}).addTo(map);
            vessel{vessel_idx}_gt_line.bindPopup('<b>Vessel {vessel_idx + 1} - Complete Ground Truth</b><br>Full trajectory from original data');
            allLayers.push(vessel{vessel_idx}_gt_line);

            // Add direction arrows
            var vessel{vessel_idx}_gt_arrows = L.polylineDecorator(vessel{vessel_idx}_gt_line, {{
                patterns: [{{
                    offset: '10%',
                    repeat: '15%',
                    symbol: L.Symbol.arrowHead({{
                        pixelSize: 12,
                        polygon: false,
                        pathOptions: {{
                            stroke: true,
                            weight: 2,
                            color: '{color}',
                            opacity: 0.9
                        }}
                    }})
                }}]
            }}).addTo(map);

            vessel{vessel_idx}_complete_gt.forEach(function(coord) {{ bounds.extend(coord); }});
        }}
"""

    # Generate prediction polylines with time offset colors (semi-transparent fans)
    pred_js = ""
    for vessel_idx, pred_list in sorted(predictions.items()):
        color = vessel_colors[vessel_idx % len(vessel_colors)]

        for i, pred_data in enumerate(pred_list):
            time_offset = pred_data['time_offset']
            coords = pred_data['coords']
            coords_json = json.dumps(coords)

            # Vary opacity based on time offset (earlier predictions more transparent)
            opacity = 0.3 + (i / max(len(pred_list), 1)) * 0.4  # Range: 0.3 to 0.7

            pred_js += f"""
        // Vessel {vessel_idx + 1} Prediction from t={time_offset}s
        var vessel{vessel_idx}_pred_t{time_offset} = {coords_json};
        if (vessel{vessel_idx}_pred_t{time_offset}.length > 0) {{
            var vessel{vessel_idx}_pred_t{time_offset}_line = L.polyline(vessel{vessel_idx}_pred_t{time_offset}, {{
                color: '{color}',
                weight: 2,
                opacity: {opacity:.2f},
                dashArray: '10, 5'
            }}).addTo(map);
            vessel{vessel_idx}_pred_t{time_offset}_line.bindPopup('<b>Vessel {vessel_idx + 1} - Prediction from t={time_offset}s</b><br>Predicted path from time offset {time_offset}s');
            allLayers.push(vessel{vessel_idx}_pred_t{time_offset}_line);

            // Add direction arrows
            var vessel{vessel_idx}_pred_t{time_offset}_arrows = L.polylineDecorator(vessel{vessel_idx}_pred_t{time_offset}_line, {{
                patterns: [{{
                    offset: '10%',
                    repeat: '15%',
                    symbol: L.Symbol.arrowHead({{
                        pixelSize: 10,
                        polygon: false,
                        pathOptions: {{
                            stroke: true,
                            weight: 2,
                            color: '{color}',
                            opacity: {opacity:.2f}
                        }}
                    }})
                }}]
            }}).addTo(map);

            vessel{vessel_idx}_pred_t{time_offset}.forEach(function(coord) {{ bounds.extend(coord); }});
        }}
"""

    # Build time window info
    time_info = ", ".join([f"t={w['time_offset']}s" for w in time_windows[:5]])  # Show first 5
    if len(time_windows) > 5:
        time_info += f" ... ({len(time_windows)} total)"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Timeline: {scenario_key}</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet-polylinedecorator@1.6.0/dist/leaflet.polylineDecorator.css" />
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; }}
        #map {{ height: 100vh; }}
        .legend {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            font-size: 14px;
            max-width: 300px;
        }}
        .legend h4 {{
            margin-top: 0;
            font-size: 16px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>

    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet-polylinedecorator@1.6.0/dist/leaflet.polylineDecorator.js"></script>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 10);

        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);

        var allLayers = [];
        var bounds = L.latLngBounds();

        {gt_js}

        {pred_js}

        // Fit map to show all trajectories
        if (bounds.isValid()) {{
            map.fitBounds(bounds, {{padding: [20, 20]}});
        }}

        // Add legend
        var legend = L.control({{position: 'topright'}});
        legend.onAdd = function (map) {{
            var div = L.DomUtil.create('div', 'legend');
            div.innerHTML = '<h4>Timeline: {scenario_key}</h4>' +
                           '<p><strong>Vessels:</strong> {len(ground_truth)}</p>' +
                           '<p><strong>Time Windows:</strong> {len(time_windows)}</p>' +
                           '<p style="font-size: 12px;">{time_info}</p>' +
                           '<p><span style="color: #228B22;">━━━</span> Ground Truth (complete)</p>' +
                           '<p><span style="color: #228B22;">┅┅┅</span> Predictions (lighter = earlier)</p>' +
                           '<p style="font-size: 11px; color: #666;">Click trajectories for details</p>';
            return div;
        }};
        legend.addTo(map);

    </script>
</body>
</html>"""

    return html


def main():
    """Main entry point."""
    # Paths
    viz_dir = Path(__file__).parent.parent / "evaluation_visualizations_debug_1_step"
    output_dir = viz_dir / "timelines"
    output_dir.mkdir(exist_ok=True)

    # Data directories to search for pickle files
    # The data directory is at ../data relative to the unitraj directory
    unitraj_dir = Path(__file__).parent.parent
    data_dirs = [
        unitraj_dir.parent / "data" / "processed_ais_data" / "val",
        unitraj_dir.parent / "data" / "processed_ais_data" / "train"
    ]

    logger.info(f"Looking for pickle files in: {data_dirs[0]}")

    logger.info(f"Scanning for visualization files in: {viz_dir}")

    # Group HTML files by scenario
    scenarios = group_html_files_by_scenario(viz_dir)

    logger.info(f"Found {len(scenarios)} unique scenarios")

    # Create timeline visualization for each scenario
    for scenario_key, time_windows in scenarios.items():
        try:
            # Find matching pickle file
            pickle_file = find_matching_pickle_file(scenario_key, data_dirs)

            if pickle_file is None:
                logger.warning(f"Skipping {scenario_key}: no pickle file found")
                continue

            # Load complete ground truth from pickle
            pickle_data = load_pickle_file(pickle_file)
            pickle_trajectories, _ = extract_complete_trajectories(pickle_data)

            if not pickle_trajectories:
                logger.warning(f"Skipping {scenario_key}: no ground truth trajectories in pickle")
                continue

            # Create timeline visualization
            create_timeline_visualization(scenario_key, time_windows, pickle_trajectories, output_dir)

        except Exception as e:
            logger.error(f"Failed to create timeline for {scenario_key}: {e}", exc_info=True)

    logger.info(f"Timeline visualizations created in: {output_dir}")


if __name__ == "__main__":
    main()