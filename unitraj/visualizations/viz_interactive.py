"""
Interactive timeline visualizer for trajectory prediction.

This module creates frame-by-frame interactive HTML visualizations with:
- Timeline controls (Previous/Next/Play/Pause buttons)
- Configurable time steps (1s to 10min)
- Global toggle buttons for History/GT/Predictions
- Leaflet.js maps with vessel trajectories
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .viz_core import BaseVisualizer
from ..evaluation_utils import (
    CoordinateConverter,
    ReferenceLoader,
    PickleDataLoader,
    MetricsCalculator
)

logger = logging.getLogger(__name__)


class InteractiveTimelineVisualizer(BaseVisualizer):
    """
    Creates interactive timeline visualizations with frame-by-frame navigation.

    Features:
    - Timeline controls with Previous/Next/Play buttons
    - Configurable time steps (1s, 5s, 10s, 30s, 1min, 5min, 10min)
    - Global toggle buttons for History/GT/Predictions
    - Keyboard shortcuts (Arrow keys, Space)
    - Automatic map bounds adjustment
    """

    def __init__(self, config=None):
        """
        Initialize interactive timeline visualizer.

        Args:
            config: Configuration dict with data paths and settings
        """
        super().__init__(config)
        self.coord_converter = CoordinateConverter()
        self.ref_loader = ReferenceLoader(config=config)
        self.pickle_loader = PickleDataLoader()

    def create_visualization(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        scenario_id: str,
        output_dir: str,
        output_filename: str,
        metrics: Optional[Dict[str, float]] = None,
        scene_context: Optional[Dict] = None
    ) -> str:
        """
        Create interactive timeline visualization.

        Args:
            predictions: Model predictions [1, future_len, 2] in XY meters
            ground_truth: Ground truth [1, future_len, 2] in XY meters
            scenario_id: Scenario identifier
            output_dir: Output directory
            output_filename: Output HTML filename
            metrics: Optional evaluation metrics
            scene_context: Multi-agent scene data with timeline

        Returns:
            Path to generated HTML file
        """
        # Get reference coordinates
        ref_lat, ref_lon = self.ref_loader.get_reference_coordinates(scenario_id)

        # Process predictions and ground truth into lat/lon
        pred_coords = self._process_predictions(
            predictions, ground_truth, ref_lat, ref_lon, scene_context
        )

        # Extract window start index from scenario_id
        window_start_idx = self._extract_window_start_idx(scenario_id)

        # Process multi-agent scene data
        if scene_context is not None:
            past_len = scene_context['obj_trajs'].shape[1]
            future_len = scene_context.get('future_gt', np.zeros((1, 60, 2))).shape[1]

            # Load pickle data for absolute positions
            pickle_data = PickleDataLoader.load_scenario(scenario_id, self.config)

            if pickle_data:
                result = self._process_multi_agent_scene(
                    scene_context, pickle_data, ref_lat, ref_lon,
                    past_len, future_len, window_start_idx
                )
                all_vessel_past_coords = result['past_coords']
                all_vessel_future_coords = result['future_coords']
                all_vessel_ids = result['vessel_ids']
                all_vessel_speeds = result['speeds']
                all_vessel_timeline_data = result['timeline_data']
                total_timesteps = result['total_timesteps']
                predicted_idx = scene_context.get('track_idx', 0)
            else:
                logger.warning("Could not load pickle data, using minimal scene context")
                all_vessel_past_coords = []
                all_vessel_future_coords = []
                all_vessel_ids = []
                all_vessel_speeds = []
                all_vessel_timeline_data = []
                total_timesteps = 0
                predicted_idx = 0
        else:
            # Single-agent mode
            past_len = 21
            future_len = ground_truth.shape[1] if len(ground_truth.shape) > 1 else 60
            total_timesteps = 0
            all_vessel_past_coords = []
            all_vessel_future_coords = []
            all_vessel_ids = []
            all_vessel_speeds = []
            all_vessel_timeline_data = []
            predicted_idx = 0
            window_start_idx = 0

        # Generate vessel colors
        vessel_colors = [self.color_palette.get_color(i) for i in range(len(all_vessel_ids))]
        predicted_vessel_color = vessel_colors[0] if vessel_colors else self.color_palette.DEFAULT_COLORS[0]

        # Calculate map center (average of all positions)
        center_lat, center_lon = self._calculate_map_center(
            all_vessel_past_coords, all_vessel_future_coords, ref_lat, ref_lon
        )

        # Format metrics
        metrics_html = self._format_metrics_html(metrics)

        # Load HTML template
        template_path = Path(__file__).parent / 'templates' / 'interactive_timeline.html'
        with open(template_path, 'r') as f:
            html_template = f.read()

        # Replace template variables
        html_content = html_template.format(
            scenario_id=scenario_id,
            center_lat=center_lat,
            center_lon=center_lon,
            vessel_count=len(all_vessel_ids),
            metrics_html=metrics_html,
            pred_coords=json.dumps(pred_coords),
            all_vessel_past_coords=json.dumps(all_vessel_past_coords),
            all_vessel_future_coords=json.dumps(all_vessel_future_coords),
            all_vessel_speeds=json.dumps(all_vessel_speeds),
            timeline_data=json.dumps(all_vessel_timeline_data),
            total_timesteps=total_timesteps,
            past_len=past_len,
            future_len=future_len,
            window_start_idx=window_start_idx,
            vessel_ids=json.dumps(all_vessel_ids),
            vessel_colors=json.dumps(vessel_colors),
            predicted_vessel_color=predicted_vessel_color,
            predicted_idx=predicted_idx,
            max_current_time=max(0, total_timesteps - past_len - future_len) if total_timesteps > 0 else 100,
            has_timeline='true' if total_timesteps > 0 else 'false',
            timeline_display='block' if total_timesteps > 0 else 'none',
            map_height='calc(100vh - 140px)' if total_timesteps > 0 else '100vh'
        )

        # Write HTML file
        output_path = Path(output_dir) / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)

        logger.info(f"✅ Interactive visualization saved: {output_path}")
        return str(output_path)

    def _process_predictions(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        ref_lat: float,
        ref_lon: float,
        scene_context: Optional[Dict] = None
    ) -> List[List[float]]:
        """
        Convert predictions from meters to lat/lon coordinates.

        Args:
            predictions: Model predictions [1, future_len, 2]
            ground_truth: Ground truth [1, future_len, 2]
            ref_lat: Reference latitude
            ref_lon: Reference longitude
            scene_context: Optional scene context with reference position

        Returns:
            List of [lat, lon] prediction coordinates
        """
        if predictions is None or len(predictions) == 0:
            return []

        pred_coords = []
        reference_position = scene_context.get('reference_position', np.zeros(2)) if scene_context else np.zeros(2)

        for t in range(predictions.shape[1]):
            x = float(predictions[0, t, 0] + reference_position[0])
            y = float(predictions[0, t, 1] + reference_position[1])
            lat, lon = self.coord_converter.meters_to_latlon(x, y, ref_lat, ref_lon)
            pred_coords.append([lat, lon])

        return pred_coords

    def _process_multi_agent_scene(
        self,
        scene_context: Dict,
        pickle_data: Dict,
        ref_lat: float,
        ref_lon: float,
        past_len: int,
        future_len: int,
        window_start_idx: int
    ) -> Dict:
        """
        Process multi-agent scene data from pickle file.

        Args:
            scene_context: Scene context dict
            pickle_data: Loaded pickle data
            ref_lat: Reference latitude
            ref_lon: Reference longitude
            past_len: Number of past timesteps
            future_len: Number of future timesteps
            window_start_idx: Window start index

        Returns:
            Dict with processed vessel data
        """
        tracks = pickle_data['tracks']
        track_ids = list(tracks.keys())
        num_agents = scene_context['obj_trajs'].shape[0]

        all_vessel_past_coords = []
        all_vessel_future_coords = []
        all_vessel_ids = []
        all_vessel_speeds = []
        all_vessel_timeline_data = []

        # Get total timeline length
        total_timesteps = 0
        if len(track_ids) > 0:
            first_track = tracks[track_ids[0]]
            total_timesteps = len(first_track['state']['position'])

        for agent_idx in range(min(num_agents, len(track_ids))):
            track_id = track_ids[agent_idx]
            track_data = tracks[track_id]
            positions = track_data['state']['position']
            velocities = track_data['state']['velocity']

            # Convert ALL positions to lat/lon for timeline
            timeline_data = PickleDataLoader.extract_timeline_data(
                {'tracks': {track_id: track_data}},
                (ref_lat, ref_lon)
            )
            all_vessel_timeline_data.append(timeline_data[0])

            # Extract window-specific past/future coordinates
            past_start = window_start_idx
            past_end = window_start_idx + past_len
            future_end = past_end + future_len

            # PAST coordinates
            past_coords = []
            past_speeds = []
            for t in range(past_start, min(past_end, len(positions))):
                x, y = positions[t]
                lat, lon = self.coord_converter.meters_to_latlon(float(x), float(y), ref_lat, ref_lon)
                past_coords.append([lat, lon])

                vx, vy = velocities[t]
                speed_ms = np.sqrt(vx**2 + vy**2)
                speed_knots = speed_ms * 1.94384
                past_speeds.append(float(speed_knots))

            # FUTURE coordinates
            future_coords = []
            future_speeds = []
            for t in range(past_end, min(future_end, len(positions))):
                x, y = positions[t]
                lat, lon = self.coord_converter.meters_to_latlon(float(x), float(y), ref_lat, ref_lon)
                future_coords.append([lat, lon])

                vx, vy = velocities[t]
                speed_ms = np.sqrt(vx**2 + vy**2)
                speed_knots = speed_ms * 1.94384
                future_speeds.append(float(speed_knots))

            if past_coords or future_coords:
                all_vessel_past_coords.append(past_coords)
                all_vessel_future_coords.append(future_coords)
                all_vessel_ids.append(agent_idx)

                avg_past_speed = np.mean(past_speeds) if past_speeds else 0.0
                avg_future_speed = np.mean(future_speeds) if future_speeds else 0.0
                all_vessel_speeds.append({
                    'past_speeds': past_speeds,
                    'future_speeds': future_speeds,
                    'avg_past': float(avg_past_speed),
                    'avg_future': float(avg_future_speed)
                })

        return {
            'past_coords': all_vessel_past_coords,
            'future_coords': all_vessel_future_coords,
            'vessel_ids': all_vessel_ids,
            'speeds': all_vessel_speeds,
            'timeline_data': all_vessel_timeline_data,
            'total_timesteps': total_timesteps
        }

    def _calculate_map_center(
        self,
        all_vessel_past_coords: List,
        all_vessel_future_coords: List,
        ref_lat: float,
        ref_lon: float
    ) -> Tuple[float, float]:
        """
        Calculate map center from vessel coordinates.

        Args:
            all_vessel_past_coords: Past coordinates for all vessels
            all_vessel_future_coords: Future coordinates for all vessels
            ref_lat: Reference latitude (fallback)
            ref_lon: Reference longitude (fallback)

        Returns:
            Tuple of (center_lat, center_lon)
        """
        all_coords = []
        for coords in all_vessel_past_coords + all_vessel_future_coords:
            all_coords.extend(coords)

        if all_coords:
            lats = [c[0] for c in all_coords]
            lons = [c[1] for c in all_coords]
            return np.mean(lats), np.mean(lons)
        else:
            return ref_lat, ref_lon