"""
Evaluation utilities for trajectory prediction.

This module contains shared utilities used across evaluation:
- CoordinateConverter: Convert between meters and lat/lon
- ReferenceLoader: Load reference coordinates from pickle/CSV
- PickleDataLoader: Load and parse AIS pickle files
- MetricsCalculator: Calculate per-scene evaluation metrics
"""

import numpy as np
import pickle
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class CoordinateConverter:
    """
    Coordinate transformation utilities for AIS maritime data.
    Converts between meters (model space) and lat/lon (geographic space).
    """

    # Conversion constants
    METERS_PER_DEG_LAT = 110540.0
    METERS_PER_DEG_LON_BASE = 111320.0

    @staticmethod
    def meters_to_latlon(x_meters: float, y_meters: float,
                        ref_lat: float, ref_lon: float) -> Tuple[float, float]:
        """
        Convert relative meters to lat/lon using reference point.

        ⚠️  WARNING: This function expects ABSOLUTE coordinates in METERS!
        If you're passing model outputs, make sure to:
        1. Denormalize: multiply by position_scale (e.g., 100.0)
        2. Add absolute offset: add ego_last_abs_x/y from pickle data
        3. Then pass to this function

        Args:
            x_meters: X coordinate in meters
            y_meters: Y coordinate in meters
            ref_lat: Reference latitude
            ref_lon: Reference longitude

        Returns:
            Tuple of (latitude, longitude)
        """
        # Validation: detect if normalized values are accidentally passed
        if abs(x_meters) < 1.0 and abs(y_meters) < 1.0:
            logger.warning(
                f"⚠️  COORDINATE WARNING: Values look like NORMALIZED coordinates!\n"
                f"   Received: x={x_meters:.6f}, y={y_meters:.6f}\n"
                f"   Expected: absolute coordinates in meters (typically 10-1000+ meters)"
            )

        # Conversion factors
        meters_per_deg_lat = CoordinateConverter.METERS_PER_DEG_LAT
        meters_per_deg_lon = CoordinateConverter.METERS_PER_DEG_LON_BASE * math.cos(math.radians(ref_lat))

        lat_diff = y_meters / meters_per_deg_lat
        lon_diff = x_meters / meters_per_deg_lon

        lat = ref_lat + lat_diff
        lon = ref_lon + lon_diff

        return lat, lon

    @staticmethod
    def xy_batch_to_latlon(xy_coords: np.ndarray, ref_lat: float, ref_lon: float) -> np.ndarray:
        """
        Convert batch of x/y coordinates (meters) to lat/lon.

        Args:
            xy_coords: Array of shape [batch_size, time_steps, 2] with x/y in meters
            ref_lat: Reference latitude
            ref_lon: Reference longitude

        Returns:
            Array of shape [batch_size, time_steps, 2] with lat/lon coordinates
        """
        latlon_coords = np.zeros_like(xy_coords, dtype=np.float64)

        for batch_idx in range(xy_coords.shape[0]):
            for time_idx in range(xy_coords.shape[1]):
                x = float(xy_coords[batch_idx, time_idx, 0])
                y = float(xy_coords[batch_idx, time_idx, 1])

                lat, lon = CoordinateConverter.meters_to_latlon(x, y, ref_lat, ref_lon)

                latlon_coords[batch_idx, time_idx, 0] = lat
                latlon_coords[batch_idx, time_idx, 1] = lon

        return latlon_coords


class ReferenceLoader:
    """Load reference coordinates from pickle files or CSV files."""

    def __init__(self, config=None):
        """
        Initialize reference loader.

        Args:
            config: Configuration dict with data paths
        """
        self.config = config

    def get_reference_coords(self, scenario_id: str) -> Tuple[float, float]:
        """
        Get reference coordinates for a scenario.

        Tries multiple sources in order:
        1. Pickle file (preferred)
        2. CSV file
        3. Fallback coordinates

        Args:
            scenario_id: Scenario identifier

        Returns:
            Tuple of (latitude, longitude)
        """
        # Try pickle first
        coords = self._from_pickle(scenario_id)
        if coords is not None:
            return coords

        # Try CSV
        coords = self._from_csv(scenario_id)
        if coords is not None:
            return coords

        # Fallback
        logger.warning(f"Could not find reference coordinates for {scenario_id}, using fallback")
        return (-34.755450, 22.990367)  # Default fallback

    def _from_pickle(self, scenario_id: str) -> Optional[Tuple[float, float]]:
        """Load reference coordinates from pickle file."""
        try:
            pickle_data = PickleDataLoader.load_scenario(scenario_id, self.config)
            if pickle_data and 'reference_coordinates' in pickle_data:
                ref_coords = pickle_data['reference_coordinates']
                return (float(ref_coords['latitude']), float(ref_coords['longitude']))
        except Exception as e:
            logger.debug(f"Failed to load reference from pickle for {scenario_id}: {e}")

        return None

    def _from_csv(self, scenario_id: str) -> Optional[Tuple[float, float]]:
        """Load reference coordinates from CSV file."""
        try:
            # Parse scenario ID to get vessel name and date
            parts = scenario_id.split('_')
            if len(parts) < 4:
                return None

            vessel_name = '_'.join(parts[1:-2])  # Everything between 'ais_' and date
            date = parts[-2]  # Date part

            # Look for CSV file
            csv_pattern = f"data/ais_data_from_influx_csv/kepler_{vessel_name}_{date}_{date}_part-1.csv"
            csv_path = Path(csv_pattern)

            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if not df.empty and 'latitude' in df.columns and 'longitude' in df.columns:
                    ref_lat = df['latitude'].iloc[0]
                    ref_lon = df['longitude'].iloc[0]
                    logger.info(f"✅ Loaded reference from CSV: ({ref_lat}, {ref_lon})")
                    return (float(ref_lat), float(ref_lon))
        except Exception as e:
            logger.debug(f"Failed to load reference from CSV for {scenario_id}: {e}")

        return None


class PickleDataLoader:
    """Load and parse AIS pickle files."""

    @staticmethod
    def load_scenario(scenario_id: str, config=None) -> Optional[Dict]:
        """
        Load pickle file for a scenario.

        Args:
            scenario_id: Scenario identifier
            config: Configuration dict with data paths

        Returns:
            Pickle data dict or None if not found
        """
        if config is None:
            return None

        # Remove time offset suffix if present (e.g., _t0, _t300)
        scenario_id_base = scenario_id
        if '_t' in scenario_id:
            import re
            match = re.match(r'(.*?)_t\d+$', scenario_id)
            if match:
                scenario_id_base = match.group(1)

        # Search for pickle file in val data paths
        val_paths = config.get('val_data_path', [])
        if isinstance(val_paths, str):
            val_paths = [val_paths]

        for data_path in val_paths:
            data_path_obj = Path(data_path)

            # Try standard structure: data_path/scenario_id/scenario_id.pkl
            pickle_path = data_path_obj / scenario_id_base / f"{scenario_id_base}.pkl"
            if pickle_path.exists():
                try:
                    with open(pickle_path, 'rb') as f:
                        pickle_data = pickle.load(f)
                        logger.info(f"✅ Found pickle file: {pickle_path}")
                        return pickle_data
                except Exception as e:
                    logger.warning(f"Failed to load pickle file {pickle_path}: {e}")

            # Search all subdirectories for .pkl files
            if data_path_obj.exists():
                for pkl_file in data_path_obj.rglob("*.pkl"):
                    # Skip summary files
                    if pkl_file.name in ['dataset_summary.pkl', 'dataset_mapping.pkl', 'file_list.pkl']:
                        continue

                    try:
                        with open(pkl_file, 'rb') as f:
                            pickle_data = pickle.load(f)
                            # Check if this matches our scenario
                            if pickle_data.get('scenario_id', '').startswith(scenario_id_base):
                                logger.info(f"✅ Found pickle file: {pkl_file}")
                                return pickle_data
                    except Exception:
                        continue

        logger.warning(f"Could not find pickle file for {scenario_id}")
        return None

    @staticmethod
    def extract_timeline_data(pickle_data: Dict, ref_coords: Tuple[float, float]) -> List[Dict]:
        """
        Extract full timeline data from pickle for interactive visualization.

        Args:
            pickle_data: Loaded pickle data
            ref_coords: Reference (lat, lon) for coordinate conversion

        Returns:
            List of vessel timeline dicts with positions and speeds
        """
        tracks = pickle_data['tracks']
        track_ids = list(tracks.keys())
        ref_lat, ref_lon = ref_coords

        timeline_data = []

        for track_id in track_ids:
            track = tracks[track_id]
            positions = track['state']['position']  # [timesteps, 2] in meters
            velocities = track['state']['velocity']  # [timesteps, 2] in m/s

            # Convert all positions to lat/lon
            all_positions_latlon = []
            all_speeds_knots = []

            for t in range(len(positions)):
                x, y = positions[t]
                lat, lon = CoordinateConverter.meters_to_latlon(
                    float(x), float(y), ref_lat, ref_lon
                )
                all_positions_latlon.append([float(lat), float(lon)])

                # Calculate speed in knots
                vx, vy = velocities[t]
                speed_ms = np.sqrt(vx**2 + vy**2)
                speed_knots = speed_ms * 1.94384  # Convert m/s to knots
                all_speeds_knots.append(float(speed_knots))

            timeline_data.append({
                'positions': all_positions_latlon,
                'speeds': all_speeds_knots
            })

        return timeline_data


class MetricsCalculator:
    """Calculate evaluation metrics for trajectory predictions."""

    @staticmethod
    def calculate_scene_metrics(predictions: np.ndarray, ground_truth: np.ndarray,
                               miss_threshold: float = 2.0) -> Dict[str, float]:
        """
        Calculate ADE, FDE, and miss rate for a single scene.

        Args:
            predictions: Predicted trajectory [time_steps, 2]
            ground_truth: Ground truth trajectory [time_steps, 2]
            miss_threshold: Distance threshold for miss rate (meters)

        Returns:
            Dict with 'ade', 'fde', 'miss_rate' metrics
        """
        # Average Displacement Error
        displacements = np.linalg.norm(predictions - ground_truth, axis=-1)
        ade = float(np.mean(displacements))

        # Final Displacement Error
        fde = float(np.linalg.norm(predictions[-1] - ground_truth[-1]))

        # Miss Rate (1 if FDE > threshold, 0 otherwise)
        miss_rate = 1.0 if fde > miss_threshold else 0.0

        return {
            'ade': ade,
            'fde': fde,
            'miss_rate': miss_rate
        }