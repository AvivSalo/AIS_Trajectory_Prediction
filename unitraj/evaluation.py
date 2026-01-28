import pytorch_lightning as pl
import torch
import numpy as np
import os
import json
from typing import Dict, List, Any, Optional
import logging

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed
import hydra
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class EvaluationCallback(pl.Callback):
    """Custom callback to collect predictions and ground truth for visualization"""

    def __init__(self, config=None):
        super().__init__()
        self.predictions = []
        self.ground_truths = []
        self.scenario_ids = []
        self.scene_context = []  # Store multi-agent scene data
        self.metrics = {}
        self.output_dir = "evaluation_visualizations"
        self.config = config  # Store config to access data paths
        os.makedirs(self.output_dir, exist_ok=True)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect predictions and ground truth from each batch"""
        try:
            # Get prediction from the model's forward pass
            input_dict = batch['input_dict']
            prediction, loss = pl_module.forward(batch)

            # Extract predictions - shape: [batch_size, num_modes, future_len, 2]
            pred_trajs = prediction['predicted_trajectory']  # [B, num_modes, T, 2]
            # Take the best mode (mode 0) for visualization
            pred_trajs_best = pred_trajs[:, 0, :, :2]  # [B, T, 2]

            # Extract ground truth - shape: [batch_size, future_len, 2]
            if 'center_gt_trajs' in input_dict and input_dict['center_gt_trajs'] is not None:
                gt_trajs = input_dict['center_gt_trajs'][:, :, :2]  # [B, T, 2]
            else:
                # Fallback: extract future trajectory from obj_trajs if center_gt_trajs is missing
                track_idx = input_dict.get('track_index_to_predict', torch.tensor([0]))[0]
                if isinstance(track_idx, torch.Tensor):
                    track_idx = track_idx.item()
                obj_trajs = input_dict['obj_trajs']  # [B, num_agents, timesteps, features]

                # Get config values for past and future lengths
                past_len = getattr(pl_module.config, 'past_len', 21)
                future_len = getattr(pl_module.config, 'future_len', 60)

                # Extract future part of the ego agent trajectory
                ego_traj = obj_trajs[0, track_idx, :, :]  # [timesteps, features]
                if ego_traj.shape[0] > past_len:
                    # Get future trajectory: [future_len, 2] for x,y coordinates
                    future_end_idx = min(past_len + future_len, ego_traj.shape[0])
                    future_traj = ego_traj[past_len:future_end_idx, 1:3]  # Skip time (index 0), get x,y (indices 1,2)

                    # Pad if necessary to match expected future_len
                    if future_traj.shape[0] < future_len:
                        padding = torch.zeros(future_len - future_traj.shape[0], 2)
                        future_traj = torch.cat([future_traj, padding], dim=0)

                    gt_trajs = future_traj.unsqueeze(0)  # Add batch dimension: [1, future_len, 2]
                else:
                    # Create dummy ground truth if not enough data
                    gt_trajs = torch.zeros(1, future_len, 2)

            # Extract PAST trajectories for visualization (history)
            past_len = getattr(pl_module.config, 'past_len', 21)
            batch_size = pred_trajs_best.shape[0]
            past_trajs_list = []

            for scene_idx in range(batch_size):
                track_idx = input_dict['track_index_to_predict'][scene_idx].item() if isinstance(input_dict['track_index_to_predict'], torch.Tensor) else input_dict['track_index_to_predict'][scene_idx]
                # Extract past trajectory (first past_len timesteps)
                past_traj = input_dict['obj_trajs'][scene_idx, track_idx, :past_len, 0:2]  # [past_len, 2]
                past_trajs_list.append(past_traj.detach().cpu().numpy())

            past_trajs = np.array(past_trajs_list)  # [B, past_len, 2]
            
            # CRITICAL COORDINATE TRANSFORMATION FIX:
            # The dataset (ais_dataset.py:285-287) transforms ALL coordinates to EGO-RELATIVE:
            #   reference_position = positions[last_past_idx]  # Last past position (t=59)
            #   ego_positions_centered = ego_positions - reference_position  # Center on last past
            #
            # This means:
            # - Model was trained on ego-relative targets (relative to last observed position)
            # - Model outputs ego-relative predictions (relative to last observed position)
            # - GT in input_dict is ego-relative (relative to last observed position)
            #
            # For visualization, we need SCENARIO-RELATIVE coordinates (relative to first position in pickle)
            # Therefore: scenario_coords = ego_relative_coords + ego_last_pos

            logger.info("AIS data: Converting from ego-relative to scenario-relative coordinates")

            # Get configuration for trajectory lengths
            past_len = getattr(pl_module.config, 'past_len', 21)

            # Initialize arrays for scenario-relative coordinates
            pred_trajs_scenario = pred_trajs_best.detach().cpu().numpy().copy()
            gt_trajs_scenario = gt_trajs.detach().cpu().numpy().copy()

            # Transform predictions and GT from ego-relative to scenario-relative
            for scene_idx in range(pred_trajs_best.shape[0]):
                # Get the ego agent's last observed position (the "current time" reference point)
                track_idx = input_dict['track_index_to_predict'][scene_idx].item()
                ego_last_pos = input_dict['obj_trajs'][scene_idx, track_idx, past_len-1, 0:2].cpu().numpy()

                logger.info(f"[DEBUG_COORDS] Scene {scene_idx}:")
                logger.info(f"[DEBUG_COORDS]   ego_last_pos (NORMALIZED): x={ego_last_pos[0]:.6f}, y={ego_last_pos[1]:.6f}")
                logger.info(f"[DEBUG_COORDS]   Pred BEFORE transform (EGO-REL, NORMALIZED): first 3 points")
                for i in range(min(3, pred_trajs_scenario.shape[1])):
                    logger.info(f"[DEBUG_COORDS]     t={i}: x={pred_trajs_scenario[scene_idx, i, 0]:.6f}, y={pred_trajs_scenario[scene_idx, i, 1]:.6f}")

                logger.info(f"[DEBUG_COORDS]   GT BEFORE transform (EGO-REL, NORMALIZED): first 3 points")
                for i in range(min(3, gt_trajs_scenario.shape[1])):
                    logger.info(f"[DEBUG_COORDS]     t={i}: x={gt_trajs_scenario[scene_idx, i, 0]:.6f}, y={gt_trajs_scenario[scene_idx, i, 1]:.6f}")

                # Transform: scenario_coords = ego_relative_coords + ego_last_pos
                pred_trajs_scenario[scene_idx] = pred_trajs_scenario[scene_idx] + ego_last_pos
                gt_trajs_scenario[scene_idx] = gt_trajs_scenario[scene_idx] + ego_last_pos

                logger.info(f"[DEBUG_COORDS]   Pred AFTER transform (SCENARIO-REL, NORMALIZED): first 3 points")
                for i in range(min(3, pred_trajs_scenario.shape[1])):
                    logger.info(f"[DEBUG_COORDS]     t={i}: x={pred_trajs_scenario[scene_idx, i, 0]:.6f}, y={pred_trajs_scenario[scene_idx, i, 1]:.6f}")

                logger.info(f"[DEBUG_COORDS]   GT AFTER transform (SCENARIO-REL, NORMALIZED): first 3 points")
                for i in range(min(3, gt_trajs_scenario.shape[1])):
                    logger.info(f"[DEBUG_COORDS]     t={i}: x={gt_trajs_scenario[scene_idx, i, 0]:.6f}, y={gt_trajs_scenario[scene_idx, i, 1]:.6f}")

            pred_trajs_latlon = pred_trajs_scenario
            gt_trajs_latlon = gt_trajs_scenario
            
            # Process each scene individually instead of batching them
            batch_size = pred_trajs_best.shape[0]
            scenario_ids = input_dict.get('scenario_id', [f"batch_{batch_idx}_scenario_{i}" for i in range(batch_size)])

            if isinstance(scenario_ids, torch.Tensor):
                scenario_ids = scenario_ids.cpu().numpy().tolist()
            elif isinstance(scenario_ids, str):
                scenario_ids = [scenario_ids]
            elif isinstance(scenario_ids, np.ndarray):
                scenario_ids = scenario_ids.tolist()
            elif not isinstance(scenario_ids, (list, tuple)):
                scenario_ids = [str(scenario_ids)]

            # Ensure scenario_ids is a list of strings
            scenario_ids = [str(sid) for sid in scenario_ids]

            # Process each maritime scene individually
            for scene_idx in range(batch_size):
                scenario_id = str(scenario_ids[scene_idx])

                # Get multi-agent data for this specific scene
                scene_obj_trajs = input_dict['obj_trajs'][scene_idx]  # [num_agents, timesteps, features]
                scene_obj_mask = input_dict['obj_trajs_mask'][scene_idx]  # [num_agents, timesteps]

                # NOTE: Maritime scene visualization disabled - using _create_leaflet_visualization instead
                # which creates the combined vessel trajectory HTML files in on_validation_epoch_end
                # The Leaflet visualizations show predictions vs ground truth on interactive maps
                # self._create_maritime_scene_visualization(
                #     scenario_id,
                #     scene_obj_trajs,
                #     scene_obj_mask,
                #     pred_trajs_latlon[scene_idx:scene_idx+1],
                #     gt_trajs_latlon[scene_idx:scene_idx+1],
                #     pl_module
                # )

            # Keep for compatibility (but won't be used for final visualization)
            self.predictions.append(pred_trajs_latlon)
            self.ground_truths.append(gt_trajs_latlon)
            self.scenario_ids.extend([str(sid) for sid in scenario_ids])

            # Store multi-agent scene context for visualization INCLUDING PAST TRAJECTORIES
            for scene_idx in range(batch_size):
                # Get reference position if available (for de-centering ego-relative coordinates)
                ref_pos = input_dict.get('reference_position')
                if ref_pos is not None:
                    ref_pos = ref_pos[scene_idx].detach().cpu().numpy()  # [2]
                else:
                    ref_pos = np.zeros(2)

                # Get future GT trajectories for all agents
                all_agents_gt = input_dict.get('all_agents_gt_trajs')  # [B, num_agents, future_len, 2]
                all_agents_gt_masks = input_dict.get('all_agents_gt_masks')  # [B, num_agents, future_len]

                if all_agents_gt is not None:
                    future_gt = all_agents_gt[scene_idx].detach().cpu().numpy()  # [num_agents, future_len, 2]
                    future_gt_mask = all_agents_gt_masks[scene_idx].detach().cpu().numpy()  # [num_agents, future_len]
                else:
                    future_gt = None
                    future_gt_mask = None

                self.scene_context.append({
                    'obj_trajs': input_dict['obj_trajs'][scene_idx].detach().cpu().numpy(),  # [num_agents, past_len, features]
                    'obj_mask': input_dict['obj_trajs_mask'][scene_idx].detach().cpu().numpy(),  # [num_agents, past_len]
                    'track_idx': input_dict['track_index_to_predict'][scene_idx].item() if isinstance(input_dict['track_index_to_predict'], torch.Tensor) else 0,
                    'past_traj': past_trajs[scene_idx],  # [past_len, 2] - ego vessel history
                    'reference_position': ref_pos,  # [2] - centering offset for de-normalization
                    'future_gt': future_gt,  # [num_agents, future_len, 2] - future GT for all agents
                    'future_gt_mask': future_gt_mask  # [num_agents, future_len] - validity mask
                })
            
        except Exception as e:
            logger.warning(f"Failed to collect batch data for visualization: {str(e)}")

    def _get_reference_coordinates_from_csv(self, scenario_id):
        """
        Extract reference coordinates from original CSV file.
        The reference point is the first valid own_latitude/own_longitude in the CSV.

        Args:
            scenario_id: Scenario ID like "ais_arc-integrity_20250315_060000"

        Returns:
            Tuple of (reference_lat, reference_lon) or None if not found
        """
        try:
            import pandas as pd
            from pathlib import Path

            # Parse scenario_id to extract vessel name and timestamp
            # Format: ais_{vessel_name}_{YYYYMMDD}_{HHMMSS}_t{offset} or ais_{vessel_name}_{YYYYMMDD}_{HHMMSS}
            if not scenario_id.startswith('ais_'):
                logger.warning(f"Unexpected scenario_id format: {scenario_id}")
                return None

            # Remove time offset suffix if present (e.g., _t0, _t300, _t3900)
            scenario_id_base = scenario_id
            if '_t' in scenario_id:
                # Find the last occurrence of _t followed by digits
                import re
                match = re.match(r'(.*?)_t\d+$', scenario_id)
                if match:
                    scenario_id_base = match.group(1)

            parts = scenario_id_base.replace('ais_', '').split('_')
            if len(parts) < 3:
                logger.warning(f"Cannot parse scenario_id: {scenario_id}")
                return None

            # Reconstruct vessel name (everything except last 2 parts which are date and time)
            vessel_name = '_'.join(parts[:-2])
            date_str = parts[-2]  # YYYYMMDD

            # Convert date format: 20250315 -> 2025-03-15
            if len(date_str) == 8 and date_str.isdigit():
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            else:
                logger.warning(f"Unexpected date format in scenario_id: {date_str}")
                return None

            # Construct CSV filename: kepler_{vessel_name}_{YYYY-MM-DD}_{YYYY-MM-DD}_part-1.csv
            csv_filename = f"kepler_{vessel_name}_{formatted_date}_{formatted_date}_part-1.csv"

            # Look for CSV in data directory
            csv_dir = Path(__file__).parent.parent / "data" / "ais_data_from_influx_csv"
            csv_path = csv_dir / csv_filename

            if not csv_path.exists():
                logger.warning(f"CSV file not found: {csv_path}")
                return None

            # Read CSV and extract first valid row
            df = pd.read_csv(csv_path)

            # Get first row with valid coordinates
            for _, row in df.iterrows():
                if 'own_latitude' in row and 'own_longitude' in row:
                    lat = row['own_latitude']
                    lon = row['own_longitude']
                    if not (pd.isna(lat) or pd.isna(lon)):
                        logger.info(f"Found reference coordinates from {csv_filename}: lat={lat:.6f}, lon={lon:.6f}")
                        return float(lat), float(lon)

            logger.warning(f"No valid coordinates found in {csv_filename}")
            return None

        except Exception as e:
            logger.error(f"Failed to extract reference coordinates from CSV for {scenario_id}: {str(e)}")
            return None

    def _create_maritime_scene_visualization(self, scenario_id, scene_obj_trajs, scene_obj_mask,
                                           pred_trajs_latlon, gt_trajs_latlon, pl_module):
        """
        Create individual maritime scene visualization with ALL vessels from one pickle file
        Each scene represents a different geographic location with multiple vessels

        Now uses the enhanced visualization utilities to separate past, GT future, and predicted trajectories.
        """
        try:
            # Import visualization utilities
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
            from ais_visualization_utils.trajectory_viz import (
                split_trajectory,
                convert_xy_to_latlon,
                extract_trajectory_coordinates,
                create_html_visualization
            )
            import math

            # Get vessel trajectories from obj_trajs for this scene
            valid_agents = scene_obj_mask.sum(dim=1) > 0  # agents with at least one valid timestep
            valid_scene_trajs = scene_obj_trajs[valid_agents]  # [num_valid_agents, timesteps, features]

            if len(valid_scene_trajs) == 0:
                logger.warning(f"No valid agents found in scenario {scenario_id}")
                return

            # Get correct reference coordinates from original CSV file
            ref_coords = self._get_reference_coordinates_from_csv(scenario_id)
            if ref_coords is not None:
                reference_lat, reference_lon = ref_coords
                logger.info(f"Using reference coordinates for {scenario_id}: lat={reference_lat:.6f}, lon={reference_lon:.6f}")
            else:
                # Fallback to default (Mediterranean center) if CSV not found
                reference_lat, reference_lon = 31.833351, 34.618101
                logger.warning(f"Using fallback reference coordinates for {scenario_id}")

            # Get past_len from config (default 21)
            past_len = 21
            if hasattr(pl_module, 'config') and 'past_len' in pl_module.config:
                past_len = pl_module.config['past_len']

            # Extract past and future ground truth coordinates using utility function
            past_coords_list = []
            future_gt_coords_list = []
            valid_agent_indices = torch.where(scene_obj_mask.sum(dim=1) > 0)[0].cpu().numpy()

            for i, agent_idx in enumerate(valid_agent_indices):
                agent_traj = scene_obj_trajs[agent_idx]  # [timesteps, features]

                # Split into past and future using utility function
                past_traj, future_traj = split_trajectory(agent_traj, past_len)

                # Extract XY coordinates for past
                past_xy = past_traj[:, 0:2].cpu().numpy()  # [past_len, 2]
                # Filter out invalid past coordinates
                valid_past_mask = ~np.isnan(past_xy).any(axis=1) & (past_xy != 0).any(axis=1)
                if not valid_past_mask.any():
                    continue
                past_xy = past_xy[valid_past_mask]

                # Convert past trajectory to lat/lon
                past_coords = []
                for x, y in past_xy:
                    lat, lon = convert_xy_to_latlon(np.array([x, y]), reference_lat, reference_lon)
                    past_coords.append([lat, lon])
                past_coords_list.append(past_coords)

                # Extract XY coordinates for future GT
                future_xy = future_traj[:, 0:2].cpu().numpy()  # [future_len, 2]
                # Filter out invalid future coordinates
                valid_future_mask = ~np.isnan(future_xy).any(axis=1) & (future_xy != 0).any(axis=1)
                if not valid_future_mask.any():
                    future_gt_coords_list.append([])
                    continue
                future_xy = future_xy[valid_future_mask]

                # Convert future GT trajectory to lat/lon
                future_gt_coords = []
                for x, y in future_xy:
                    lat, lon = convert_xy_to_latlon(np.array([x, y]), reference_lat, reference_lon)
                    future_gt_coords.append([lat, lon])
                future_gt_coords_list.append(future_gt_coords)

            # Get predictions for ego agent and convert to lat/lon
            scene_pred_coords = []
            if len(pred_trajs_latlon) > 0 and len(pred_trajs_latlon[0]) > 0:
                pred_batch = pred_trajs_latlon[0]  # [timesteps, 2] - single agent prediction (relative meters)
                pred_coords = pred_batch if isinstance(pred_batch, np.ndarray) else pred_batch.cpu().numpy()

                # Convert predictions from relative meters to lat/lon
                pred_latlon = []
                for x, y in pred_coords:
                    lat, lon = convert_xy_to_latlon(np.array([x, y]), reference_lat, reference_lon)
                    pred_latlon.append([lat, lon])
                pred_coords = np.array(pred_latlon)

                # Filter out invalid predictions
                valid_pred_mask = ~np.isnan(pred_coords).any(axis=1) & (pred_coords != 0).any(axis=1)
                if valid_pred_mask.any():
                    # Add prediction for ego vessel (agent 0)
                    scene_pred_coords.append(pred_coords[valid_pred_mask].tolist())
                    # Add empty predictions for other agents until we implement multi-agent prediction
                    for _ in range(1, len(valid_agent_indices)):
                        scene_pred_coords.append([])
                else:
                    # Add empty predictions for all agents
                    for _ in range(len(valid_agent_indices)):
                        scene_pred_coords.append([])
            else:
                # Add empty predictions for all agents
                for _ in range(len(valid_agent_indices)):
                    scene_pred_coords.append([])

            # Calculate map center from all valid coordinates
            all_coords = []
            for coords_list in past_coords_list:
                all_coords.extend(coords_list)
            for coords_list in future_gt_coords_list:
                all_coords.extend(coords_list)
            for coords_list in scene_pred_coords:
                all_coords.extend(coords_list)

            if not all_coords:
                logger.warning(f"No valid coordinates found for scenario {scenario_id}")
                return

            center_lat = np.mean([coord[0] for coord in all_coords])
            center_lon = np.mean([coord[1] for coord in all_coords])

            # Create HTML visualization using utility function
            html_path = os.path.join(self.output_dir, f"{scenario_id}.html")
            create_html_visualization(
                scenario_id=scenario_id,
                past_coords_list=past_coords_list,
                future_gt_coords_list=future_gt_coords_list,
                pred_coords_list=scene_pred_coords,
                center_lat=center_lat,
                center_lon=center_lon,
                output_path=html_path
            )

        except Exception as e:
            logger.error(f"Failed to create maritime scene visualization for {scenario_id}: {str(e)}")
            import traceback
            traceback.print_exc()

    def _create_individual_html_visualization(self, scenario_id, gt_coords_list, pred_coords_list, center_lat, center_lon):
        """Create HTML file for individual maritime scene with multiple vessels"""
        try:
            html_filename = f"{scenario_id}.html"
            html_path = os.path.join(self.output_dir, html_filename)

            # Generate vessel colors
            colors = ['#228B22', '#4169E1', '#DC143C', '#FF8C00', '#9932CC', '#FF1493', '#00CED1', '#32CD32']

            html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Maritime Scene: {scenario_id}</title>
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

        // Add vessels
        var vesselColors = {colors};
        var vesselCount = 0;
'''

            # Add ground truth trajectories for all vessels
            for vessel_idx, gt_coords in enumerate(gt_coords_list):
                if len(gt_coords) > 0:
                    color = colors[vessel_idx % len(colors)]
                    # Convert numpy types to native Python types for JSON serialization
                    gt_coords_clean = [[float(coord[0]), float(coord[1])] for coord in gt_coords]
                    gt_coords_json = json.dumps(gt_coords_clean)

                    html_content += f'''
        // Vessel {vessel_idx + 1} Ground Truth
        var vessel{vessel_idx}_gt_coords = {gt_coords_json};
        if (vessel{vessel_idx}_gt_coords.length > 0) {{
            var vessel{vessel_idx}_gt_line = L.polyline(vessel{vessel_idx}_gt_coords, {{
                color: '{color}',
                weight: 3,
                opacity: 0.8
            }}).addTo(map);
            vessel{vessel_idx}_gt_line.bindPopup('<b>Vessel {vessel_idx + 1} - Ground Truth</b><br>Scene: {scenario_id}<br>Actual path from AIS data');
            allLayers.push(vessel{vessel_idx}_gt_line);

            // Add direction arrows to ground truth
            var vessel{vessel_idx}_gt_arrows = L.polylineDecorator(vessel{vessel_idx}_gt_line, {{
                patterns: [
                    {{
                        offset: '10%',
                        repeat: '15%',
                        symbol: L.Symbol.arrowHead({{
                            pixelSize: 12,
                            polygon: false,
                            pathOptions: {{
                                stroke: true,
                                weight: 2,
                                color: '{color}',
                                opacity: 0.8
                            }}
                        }})
                    }}
                ]
            }}).addTo(map);

            // Add start/end markers
            var startMarker = L.circleMarker(vessel{vessel_idx}_gt_coords[0], {{
                radius: 6,
                fillColor: '{color}',
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.9
            }}).addTo(map)
            .bindPopup('<b>Vessel {vessel_idx + 1} Start</b><br>Lat: ' + vessel{vessel_idx}_gt_coords[0][0].toFixed(6) + '<br>Lon: ' + vessel{vessel_idx}_gt_coords[0][1].toFixed(6));

            var endMarker = L.circleMarker(vessel{vessel_idx}_gt_coords[vessel{vessel_idx}_gt_coords.length-1], {{
                radius: 6,
                fillColor: '{color}',
                color: '#000',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.9
            }}).addTo(map)
            .bindPopup('<b>Vessel {vessel_idx + 1} End</b><br>Lat: ' + vessel{vessel_idx}_gt_coords[vessel{vessel_idx}_gt_coords.length-1][0].toFixed(6) + '<br>Lon: ' + vessel{vessel_idx}_gt_coords[vessel{vessel_idx}_gt_coords.length-1][1].toFixed(6));

            vessel{vessel_idx}_gt_coords.forEach(function(coord) {{ bounds.extend(coord); }});
        }}
'''

            # Add prediction trajectories
            for vessel_idx, pred_coords in enumerate(pred_coords_list):
                if len(pred_coords) > 0:
                    color = colors[vessel_idx % len(colors)]
                    # Convert numpy types to native Python types for JSON serialization
                    pred_coords_clean = [[float(coord[0]), float(coord[1])] for coord in pred_coords]
                    pred_coords_json = json.dumps(pred_coords_clean)

                    html_content += f'''
        // Vessel {vessel_idx + 1} Predictions
        var vessel{vessel_idx}_pred_coords = {pred_coords_json};
        if (vessel{vessel_idx}_pred_coords.length > 0) {{
            var vessel{vessel_idx}_pred_line = L.polyline(vessel{vessel_idx}_pred_coords, {{
                color: '{color}',
                weight: 3,
                opacity: 0.6,
                dashArray: '10, 5'
            }}).addTo(map);
            vessel{vessel_idx}_pred_line.bindPopup('<b>Vessel {vessel_idx + 1} - Prediction</b><br>Scene: {scenario_id}<br>Wayformer predicted path');
            allLayers.push(vessel{vessel_idx}_pred_line);

            // Add direction arrows to predictions
            var vessel{vessel_idx}_pred_arrows = L.polylineDecorator(vessel{vessel_idx}_pred_line, {{
                patterns: [
                    {{
                        offset: '10%',
                        repeat: '15%',
                        symbol: L.Symbol.arrowHead({{
                            pixelSize: 10,
                            polygon: false,
                            pathOptions: {{
                                stroke: true,
                                weight: 2,
                                color: '{color}',
                                opacity: 0.6
                            }}
                        }})
                    }}
                ]
            }}).addTo(map);

            vessel{vessel_idx}_pred_coords.forEach(function(coord) {{ bounds.extend(coord); }});
        }}
'''

            html_content += f'''
        // Fit map to show all trajectories
        if (bounds.isValid()) {{
            map.fitBounds(bounds, {{padding: [20, 20]}});
        }}

        // Add legend
        var legend = L.control({{position: 'topright'}});
        legend.onAdd = function (map) {{
            var div = L.DomUtil.create('div', 'legend');
            div.innerHTML = '<h4>Maritime Scene: {scenario_id}</h4>' +
                           '<p><strong>Vessels:</strong> {len(gt_coords_list)}</p>' +
                           '<p><span style="color: #228B22;">━━━</span> Ground Truth</p>' +
                           '<p><span style="color: #228B22;">┅┅┅</span> Predictions</p>' +
                           '<p><strong>● Start</strong> | <strong>● End</strong></p>';
            return div;
        }};
        legend.addTo(map);

    </script>
</body>
</html>'''

            with open(html_path, 'w') as f:
                f.write(html_content)

            logger.info(f"Created maritime scene visualization: {html_path}")

        except Exception as e:
            logger.error(f"Failed to create HTML visualization for {scenario_id}: {str(e)}")

    def _convert_xy_to_latlon(self, xy_coords: np.ndarray, reference_lat: float, reference_lon: float) -> np.ndarray:
        """
        Convert relative x/y coordinates (meters) back to lat/lon
        
        Reverses the conversion from AIS dataset:
        x = lon_diff * 111320 * cos(reference_lat) 
        y = lat_diff * 110540
        
        Args:
            xy_coords: Array of shape [batch_size, time_steps, 2] with x/y coordinates in meters
            reference_lat: Reference latitude for conversion
            reference_lon: Reference longitude for conversion
            
        Returns:
            Array of shape [batch_size, time_steps, 2] with lat/lon coordinates
        """
        import math
        
        latlon_coords = np.zeros_like(xy_coords, dtype=np.float64)
        
        for batch_idx in range(xy_coords.shape[0]):
            for time_idx in range(xy_coords.shape[1]):
                x = float(xy_coords[batch_idx, time_idx, 0])  # meters
                y = float(xy_coords[batch_idx, time_idx, 1])  # meters
                
                # Reverse the conversion with proper precision
                lat_diff = y / 110540.0  # Convert y back to lat difference
                lon_diff = x / (111320.0 * math.cos(math.radians(reference_lat)))  # Convert x back to lon difference
                
                lat = reference_lat + lat_diff
                lon = reference_lon + lon_diff
                
                latlon_coords[batch_idx, time_idx, 0] = lat
                latlon_coords[batch_idx, time_idx, 1] = lon
                
        return latlon_coords
    
    def _meters_to_latlon(self, x_meters: float, y_meters: float, ref_lat: float, ref_lon: float) -> tuple:
        """Convert relative meters to lat/lon using reference point.

        ⚠️  WARNING: This function expects ABSOLUTE coordinates in METERS, not normalized values!
        If you're passing model outputs, make sure to:
        1. Denormalize: multiply by position_scale (e.g., 100.0)
        2. Add absolute offset: add ego_last_abs_x/y from pickle data
        3. Then pass to this function
        """
        import math

        # VALIDATION: Detect if normalized values are accidentally passed
        # Maritime trajectories typically span 10-1000+ meters
        # If values are < 1.0, they're likely normalized values that need transformation
        if abs(x_meters) < 1.0 and abs(y_meters) < 1.0:
            logger.warning(
                f"⚠️  COORDINATE WARNING: Values look like NORMALIZED coordinates, not meters!\n"
                f"   Received: x={x_meters:.6f}, y={y_meters:.6f}\n"
                f"   Expected: absolute coordinates in meters (typically 10-1000+ meters)\n"
                f"   If these are model outputs, you must:\n"
                f"     1. Denormalize: multiply by position_scale (e.g., 100.0)\n"
                f"     2. Add offset: add ego_last_abs_x/y from pickle data\n"
                f"     3. Then convert to lat/lon\n"
                f"   This will cause spatial offset in visualization!"
            )

        # Conversion factors
        meters_per_deg_lat = 110540.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(ref_lat))

        lat_diff = y_meters / meters_per_deg_lat
        lon_diff = x_meters / meters_per_deg_lon

        lat = ref_lat + lat_diff
        lon = ref_lon + lon_diff

        return lat, lon

    def _load_original_pickle_data(self, scenario_id: str) -> Optional[Dict]:
        """Load original pickle file to get absolute positions of all vessels."""
        import pickle
        from pathlib import Path

        if self.config is None:
            return None

        # Remove time offset suffix if present (e.g., _t0, _t300)
        scenario_id_base = scenario_id
        if '_t' in scenario_id:
            import re
            match = re.match(r'(.*?)_t\d+$', scenario_id)
            if match:
                scenario_id_base = match.group(1)

        # Search for pickle file in val data paths
        val_paths = self.config.get('val_data_path', [])
        if isinstance(val_paths, str):
            val_paths = [val_paths]

        for data_path in val_paths:
            data_path_obj = Path(data_path)

            # Try multiple search patterns:
            # 1. Standard structure: data_path/scenario_id/scenario_id.pkl
            pickle_path = data_path_obj / scenario_id_base / f"{scenario_id_base}.pkl"
            if pickle_path.exists():
                try:
                    with open(pickle_path, 'rb') as f:
                        pickle_data = pickle.load(f)
                        # Verify this pickle contains our scenario
                        if pickle_data.get('scenario_id') == scenario_id_base or pickle_data.get('scenario_id', '').startswith(scenario_id_base):
                            logger.info(f"✅ Found pickle file: {pickle_path}")
                            return pickle_data
                except Exception as e:
                    logger.warning(f"Failed to load pickle file {pickle_path}: {e}")

            # 2. Search all subdirectories for any .pkl files
            if data_path_obj.exists():
                for pkl_file in data_path_obj.rglob("*.pkl"):
                    # Skip summary files
                    if pkl_file.name in ['dataset_summary.pkl', 'dataset_mapping.pkl', 'file_list.pkl']:
                        continue
                    try:
                        with open(pkl_file, 'rb') as f:
                            pickle_data = pickle.load(f)
                            # Check if this pickle contains our scenario
                            if isinstance(pickle_data, dict) and 'scenario_id' in pickle_data:
                                pkl_scenario_id = pickle_data['scenario_id']
                                if pkl_scenario_id == scenario_id_base or pkl_scenario_id.startswith(scenario_id_base):
                                    logger.info(f"✅ Found matching pickle file: {pkl_file}")
                                    return pickle_data
                    except Exception as e:
                        continue

        logger.warning(f"Could not find original pickle file for scenario: {scenario_id}")
        return None

    def _create_leaflet_visualization(
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
        Create Leaflet-based HTML visualization for a single scenario

        Args:
            predictions: Model predictions array [1, future_len, 2] in XY meters
            ground_truth: Ground truth array [1, future_len, 2] in XY meters
            scenario_id: Scenario identifier
            output_dir: Output directory
            output_filename: Output HTML filename
            metrics: Optional evaluation metrics

        Returns:
            Path to generated HTML file
        """
        # Get reference coordinates for this scenario
        ref_coords = self._get_reference_coordinates_from_csv(scenario_id)

        if ref_coords is None:
            # Fallback to default location if CSV lookup fails
            ref_lat, ref_lon = -34.755450, 22.990367
            print(f"Warning: Could not find reference coordinates for {scenario_id}, using fallback")
        else:
            ref_lat, ref_lon = ref_coords

        # Define color palette for multiple vessels
        vessel_colors = [
            '#228B22',  # Forest green
            '#FF4500',  # Orange red
            '#4169E1',  # Royal blue
            '#FFD700',  # Gold
            '#8B008B',  # Dark magenta
            '#00CED1',  # Dark turquoise
            '#FF1493',  # Deep pink
            '#32CD32',  # Lime green
            '#FF6347',  # Tomato
            '#4682B4',  # Steel blue
            '#DA70D6',  # Orchid
            '#00FA9A',  # Medium spring green
        ]

        # Process multi-agent scene or single agent
        predicted_idx = 0  # Default value

        if scene_context is not None:
            # Multi-agent mode: extract all vessels
            obj_trajs = scene_context['obj_trajs']  # [num_agents, past_len, features]
            obj_mask = scene_context['obj_mask']    # [num_agents, past_len]
            predicted_idx = scene_context['track_idx']  # Which agent is being predicted
            past_traj = scene_context.get('past_traj')  # [past_len, 2] - ego vessel past trajectory
            reference_position = scene_context.get('reference_position', np.zeros(2))  # [2] - centering offset
            future_gt = scene_context.get('future_gt')  # [num_agents, future_len, 2]
            future_gt_mask = scene_context.get('future_gt_mask')  # [num_agents, future_len]

            num_agents = obj_trajs.shape[0]
            past_len = obj_trajs.shape[1]  # Timesteps in past (should be 21)

            logger.info(f"🔍 Loading original pickle data for absolute vessel positions")

            # Load original pickle data to get absolute positions
            pickle_data = self._load_original_pickle_data(scenario_id)

            # Store all vessel trajectories SPLIT INTO PAST AND FUTURE
            all_vessel_past_coords = []  # Past (observed) trajectories
            all_vessel_future_coords = []  # Future GT trajectories
            all_vessel_ids = []
            all_vessel_speeds = []  # Speed statistics for each vessel: [past_speed, future_speed]
            predicted_vessel_color = vessel_colors[0]  # Default

            # Extract window start index from scenario_id (e.g., _t0, _t300)
            window_start_idx = 0
            if '_t' in scenario_id:
                import re
                match = re.search(r'_t(\d+)$', scenario_id)
                if match:
                    window_start_idx = int(match.group(1))

            if pickle_data is not None:
                logger.info(f"✅ Using original pickle data for absolute positions (window start: t={window_start_idx})")
                tracks = pickle_data['tracks']
                track_ids = list(tracks.keys())

                for agent_idx in range(min(num_agents, len(track_ids))):
                    track_id = track_ids[agent_idx]
                    track_data = tracks[track_id]
                    positions = track_data['state']['position']  # Absolute scenario-relative positions
                    velocities = track_data['state']['velocity']  # [vx, vy] in m/s

                    # Extract positions for this window
                    past_start = window_start_idx
                    past_end = window_start_idx + past_len
                    future_end = past_end + (future_gt.shape[1] if future_gt is not None else 5)

                    # PAST trajectories - use absolute positions from pickle
                    past_coords = []
                    past_speeds = []  # speeds for each past timestep
                    for t in range(past_start, min(past_end, len(positions))):
                        x, y = positions[t]
                        lat, lon = self._meters_to_latlon(float(x), float(y), ref_lat, ref_lon)
                        past_coords.append([lat, lon])

                        # Calculate speed from velocity components
                        vx, vy = velocities[t]
                        speed_ms = np.sqrt(vx**2 + vy**2)
                        speed_knots = speed_ms * 1.94384  # Convert m/s to knots
                        past_speeds.append(float(speed_knots))

                    # FUTURE trajectories - use absolute positions from pickle
                    future_coords = []
                    future_speeds = []  # speeds for each future timestep
                    for t in range(past_end, min(future_end, len(positions))):
                        x, y = positions[t]
                        lat, lon = self._meters_to_latlon(float(x), float(y), ref_lat, ref_lon)
                        future_coords.append([lat, lon])

                        # Calculate speed from velocity components
                        vx, vy = velocities[t]
                        speed_ms = np.sqrt(vx**2 + vy**2)
                        speed_knots = speed_ms * 1.94384  # Convert m/s to knots
                        future_speeds.append(float(speed_knots))

                    if past_coords or future_coords:
                        vessel_color = vessel_colors[agent_idx % len(vessel_colors)]
                        all_vessel_past_coords.append(past_coords)
                        all_vessel_future_coords.append(future_coords)
                        all_vessel_ids.append(agent_idx)

                        # Store speed statistics: [avg_past_speed, avg_future_speed]
                        avg_past_speed = np.mean(past_speeds) if past_speeds else 0.0
                        avg_future_speed = np.mean(future_speeds) if future_speeds else 0.0
                        all_vessel_speeds.append({
                            'past_speeds': past_speeds,
                            'future_speeds': future_speeds,
                            'avg_past': float(avg_past_speed),
                            'avg_future': float(avg_future_speed)
                        })

                        if agent_idx == predicted_idx:
                            predicted_vessel_color = vessel_color
                            logger.info(f"🎯 Predicted vessel (agent {agent_idx}): {len(past_coords)} past points, {len(future_coords)} future points")
                            logger.info(f"   Speed: history avg={avg_past_speed:.2f} kts, GT future avg={avg_future_speed:.2f} kts")
            else:
                logger.warning("⚠️  Could not load original pickle data, using ego-centric coordinates (vessels will appear stacked)")
                # Fallback to old method (will show stacked vessels)
                for agent_idx in range(num_agents):
                    valid_mask = obj_mask[agent_idx] > 0
                    if not np.any(valid_mask):
                        continue

                    agent_traj = obj_trajs[agent_idx]
                    past_coords = []
                    for t in range(past_len):
                        if valid_mask[t]:
                            x, y = agent_traj[t, 0], agent_traj[t, 1]
                            if not (np.isnan(x) or np.isnan(y)):
                                x_scenario = float(x + reference_position[0])
                                y_scenario = float(y + reference_position[1])
                                lat, lon = self._meters_to_latlon(x_scenario, y_scenario, ref_lat, ref_lon)
                                past_coords.append([lat, lon])

                    future_coords = []
                    if future_gt is not None and agent_idx < future_gt.shape[0]:
                        agent_future_gt = future_gt[agent_idx]
                        agent_future_mask = future_gt_mask[agent_idx]
                        for t in range(len(agent_future_gt)):
                            if agent_future_mask[t]:
                                x, y = agent_future_gt[t, 0], agent_future_gt[t, 1]
                                if not (np.isnan(x) or np.isnan(y)):
                                    x_scenario = float(x + reference_position[0])
                                    y_scenario = float(y + reference_position[1])
                                    lat, lon = self._meters_to_latlon(x_scenario, y_scenario, ref_lat, ref_lon)
                                    future_coords.append([lat, lon])

                    if past_coords or future_coords:
                        vessel_color = vessel_colors[agent_idx % len(vessel_colors)]
                        all_vessel_past_coords.append(past_coords)
                        all_vessel_future_coords.append(future_coords)
                        all_vessel_ids.append(agent_idx)
                        if agent_idx == predicted_idx:
                            predicted_vessel_color = vessel_color

            # Convert prediction to lat/lon
            # CRITICAL: Predictions are ego-relative (relative to last observed position at t=20)
            # We need to add the ego vessel's ABSOLUTE last observed position from pickle data
            pred_coords = []
            pred_speeds = []  # Speeds for predicted trajectory

            if pickle_data is not None:
                # Get ego vessel's absolute last observed position from pickle
                tracks = pickle_data['tracks']
                track_ids = list(tracks.keys())
                if predicted_idx < len(track_ids):
                    ego_track_id = track_ids[predicted_idx]
                    ego_positions = tracks[ego_track_id]['state']['position']

                    # Last observed position is at: window_start_idx + past_len - 1
                    last_observed_idx = window_start_idx + past_len - 1
                    if last_observed_idx < len(ego_positions):
                        ego_last_abs_x, ego_last_abs_y = ego_positions[last_observed_idx]
                        logger.info(f"🎯 Ego vessel last observed position (absolute): ({ego_last_abs_x:.2f}, {ego_last_abs_y:.2f})")

                        # Store absolute predicted positions for speed calculation
                        pred_positions_abs = []
                        # Get position_scale from config (predictions are normalized)
                        position_scale = getattr(self.config, 'position_scale', 100.0)

                        # Transform predictions: absolute = (ego_relative_normalized * scale) + ego_last_absolute
                        for x, y in predictions[0]:
                            if not (np.isnan(x) or np.isnan(y)):
                                # Denormalize first: multiply by position_scale to convert to meters
                                x_meters = float(x) * position_scale
                                y_meters = float(y) * position_scale
                                # Then add absolute offset to get absolute coordinates
                                abs_x = x_meters + float(ego_last_abs_x)
                                abs_y = y_meters + float(ego_last_abs_y)
                                pred_positions_abs.append([abs_x, abs_y])
                                lat, lon = self._meters_to_latlon(abs_x, abs_y, ref_lat, ref_lon)
                                pred_coords.append([lat, lon])

                        # Calculate prediction speeds from position deltas (1 second intervals)
                        for i in range(len(pred_positions_abs)):
                            if i == 0:
                                # Speed from last observed to first predicted
                                dx = pred_positions_abs[0][0] - ego_last_abs_x
                                dy = pred_positions_abs[0][1] - ego_last_abs_y
                            else:
                                # Speed between consecutive predictions
                                dx = pred_positions_abs[i][0] - pred_positions_abs[i-1][0]
                                dy = pred_positions_abs[i][1] - pred_positions_abs[i-1][1]

                            speed_ms = np.sqrt(dx**2 + dy**2)  # m/s (1 second timesteps)
                            speed_knots = speed_ms * 1.94384
                            pred_speeds.append(float(speed_knots))

                        # Log prediction speed stats
                        if pred_speeds:
                            avg_pred_speed = np.mean(pred_speeds)
                            logger.info(f"   Prediction speed: avg={avg_pred_speed:.2f} kts, speeds={[f'{s:.2f}' for s in pred_speeds]}")

                            # Update the speed data for the predicted vessel to include predictions
                            if predicted_idx < len(all_vessel_speeds):
                                all_vessel_speeds[predicted_idx]['pred_speeds'] = pred_speeds
                                all_vessel_speeds[predicted_idx]['avg_pred'] = float(avg_pred_speed)
                    else:
                        logger.warning(f"⚠️  Last observed index {last_observed_idx} out of range for ego vessel")
                        # Fallback to old method
                        for x, y in predictions[0]:
                            if not (np.isnan(x) or np.isnan(y)):
                                lat, lon = self._meters_to_latlon(float(x), float(y), ref_lat, ref_lon)
                                pred_coords.append([lat, lon])
                else:
                    logger.warning(f"⚠️  Predicted index {predicted_idx} out of range")
                    # Fallback to old method
                    for x, y in predictions[0]:
                        if not (np.isnan(x) or np.isnan(y)):
                            lat, lon = self._meters_to_latlon(float(x), float(y), ref_lat, ref_lon)
                            pred_coords.append([lat, lon])
            else:
                logger.warning("⚠️  No pickle data, using ref_lat/ref_lon directly (predictions will be wrong)")
                # Fallback to old method (will be wrong for ego-relative predictions)
                for x, y in predictions[0]:
                    if not (np.isnan(x) or np.isnan(y)):
                        lat, lon = self._meters_to_latlon(float(x), float(y), ref_lat, ref_lon)
                        pred_coords.append([lat, lon])

            # Convert GT future to lat/lon
            gt_future_coords = []
            gt_positions_abs = []  # Store absolute GT positions

            logger.info(f"[DEBUG_GT_VIZ] Converting GT to lat/lon:")
            logger.info(f"[DEBUG_GT_VIZ]   GT input (first 3, NORMALIZED): {[(float(x), float(y)) for x, y in ground_truth[0][:3]]}")

            if pickle_data is not None:
                # GT also needs denormalization and absolute offset (same as predictions!)
                position_scale = getattr(self.config, 'position_scale', 100.0)
                logger.info(f"[DEBUG_GT_VIZ]   position_scale={position_scale}")
                logger.info(f"[DEBUG_GT_VIZ]   ego_last_abs position: ({ego_last_abs_x:.2f}, {ego_last_abs_y:.2f})")

                for i, (x, y) in enumerate(ground_truth[0]):
                    if not (np.isnan(x) or np.isnan(y)):
                        # Denormalize first: multiply by position_scale
                        x_meters = float(x) * position_scale
                        y_meters = float(y) * position_scale
                        # Then add absolute offset
                        abs_x = x_meters + float(ego_last_abs_x)
                        abs_y = y_meters + float(ego_last_abs_y)
                        gt_positions_abs.append([abs_x, abs_y])
                        lat, lon = self._meters_to_latlon(abs_x, abs_y, ref_lat, ref_lon)
                        gt_future_coords.append([lat, lon])

                        if i < 3:
                            logger.info(f"[DEBUG_GT_VIZ]   t={i}: normalized=({x:.6f},{y:.6f}) → meters=({x_meters:.2f},{y_meters:.2f}) → abs=({abs_x:.2f},{abs_y:.2f}) → latlon=({lat:.6f},{lon:.6f})")
            else:
                # Fallback (should not happen)
                logger.warning("[DEBUG_GT_VIZ] No pickle_data, using ref_lat/ref_lon directly")
                for x, y in ground_truth[0]:
                    if not (np.isnan(x) or np.isnan(y)):
                        lat, lon = self._meters_to_latlon(float(x), float(y), ref_lat, ref_lon)
                        gt_future_coords.append([lat, lon])

            # Calculate center from all vessels (past + future)
            all_lats = []
            all_lons = []
            for coords_list in all_vessel_past_coords:
                all_lats.extend([c[0] for c in coords_list])
                all_lons.extend([c[1] for c in coords_list])
            for coords_list in all_vessel_future_coords:
                all_lats.extend([c[0] for c in coords_list])
                all_lons.extend([c[1] for c in coords_list])

            if all_lats and all_lons:
                center_lat = np.mean(all_lats)
                center_lon = np.mean(all_lons)
            else:
                center_lat, center_lon = ref_lat, ref_lon

        else:
            # Single-agent mode (backward compatibility)
            pred_coords = []
            for x, y in predictions[0]:
                if not (np.isnan(x) or np.isnan(y)):
                    lat, lon = self._meters_to_latlon(float(x), float(y), ref_lat, ref_lon)
                    pred_coords.append([lat, lon])

            gt_coords = []
            for x, y in ground_truth[0]:
                if not (np.isnan(x) or np.isnan(y)):
                    lat, lon = self._meters_to_latlon(float(x), float(y), ref_lat, ref_lon)
                    gt_coords.append([lat, lon])

            # Calculate center point from converted coordinates
            center_lat, center_lon = ref_lat, ref_lon  # Default to reference point

            # Try to get center from ground truth first
            if gt_coords:
                center_lat = np.mean([coord[0] for coord in gt_coords])
                center_lon = np.mean([coord[1] for coord in gt_coords])
            # If no valid ground truth, try predictions
            elif pred_coords:
                center_lat = np.mean([coord[0] for coord in pred_coords])
                center_lon = np.mean([coord[1] for coord in pred_coords])

            # Wrap in structure for template
            all_vessel_coords = [gt_coords] if gt_coords else []
            all_vessel_ids = [0]
            predicted_vessel_color = vessel_colors[0]
        
        # Format metrics for display
        metrics_html = ""
        if metrics:
            metrics_html = "<div class='metrics'><h4>📊 Evaluation Metrics</h4>"
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_html += f"<p><strong>{key}:</strong> {value:.4f}</p>"
                else:
                    metrics_html += f"<p><strong>{key}:</strong> {value}</p>"
            metrics_html += "</div>"
        
        html_template = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AIS Scenario: {scenario_id}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    
    <!-- Leaflet CSS and JS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
            integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        #map {{ height: 100vh; width: 100%; }}
        .info-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255, 255, 255, 0.95);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 320px;
            max-height: 80vh;
            overflow-y: auto;
        }}
        .info-panel h3 {{ margin-top: 0; color: #2c5aa0; }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 4px;
            margin-right: 10px;
            border-radius: 2px;
        }}
        .ground-truth {{ background-color: #228B22; }}
        .prediction {{ background-color: #FF4500; }}
        .vessel-point {{ background-color: #1E90FF; }}
        .status {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px;
            border-radius: 5px;
            z-index: 1000;
        }}
        .error {{
            background: #ffebee;
            border: 1px solid #f44336;
            color: #d32f2f;
            padding: 20px;
            margin: 20px;
            border-radius: 5px;
        }}
        .metrics {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #ddd;
        }}
        .metrics h4 {{ margin-top: 0; color: #2c5aa0; }}
        .metrics p {{ margin: 5px 0; font-size: 14px; }}
        .vessel-controls {{
            margin-top: 10px;
            padding: 8px;
            background: #f5f5f5;
            border-radius: 5px;
            font-size: 12px;
        }}
        .vessel-controls-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 5px;
        }}
        .toggle-btn {{
            padding: 3px 8px;
            margin: 2px;
            border: 1px solid #ccc;
            background: white;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.2s;
        }}
        .toggle-btn.active {{
            background: #4CAF50;
            color: white;
            border-color: #4CAF50;
        }}
        .toggle-btn:hover {{
            background: #e0e0e0;
        }}
        .toggle-btn.active:hover {{
            background: #45a049;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="info-panel">
        <h3>🚢 AIS Scenario</h3>
        <p><strong>Scenario:</strong> {scenario_id}</p>
        <p><strong>Model:</strong> Wayformer-AIS</p>
        <p><strong>Vessels:</strong> <span id="vessel-count">{len(all_vessel_ids)}</span></p>
        <div class="legend" id="legend-container">
            <h4>Vessels & Trajectories:</h4>
            <!-- Vessel legend will be dynamically generated -->
        </div>
        {metrics_html}
        <p><small>💡 Click trajectories for details</small></p>
        <p><small>🔍 Zoom and pan to explore</small></p>
        <p><small>🎯 Predicted vessel shown with thicker line</small></p>
    </div>
    
    <div class="status" id="status">
        📡 Loading scenario map...
    </div>

    <script>
        console.log('🚢 Loading AIS Scenario: {scenario_id}');

        // Trajectory data - multi-agent with PAST, FUTURE GT, and PREDICTIONS separated
        const predictionCoords = {json.dumps(pred_coords)};
        const allVesselsPast = {json.dumps(all_vessel_past_coords if scene_context else [])};  // Past (observed) trajectories
        const allVesselsFutureGT = {json.dumps(all_vessel_future_coords if scene_context else [])};  // Future GT trajectories
        const vesselIds = {json.dumps(all_vessel_ids)};  // Array of vessel indices
        const vesselColors = {json.dumps(vessel_colors[:len(all_vessel_ids)])};
        const predictedVesselColor = '{predicted_vessel_color}';
        const predictedVesselIdx = {predicted_idx if scene_context else 0};
        const allVesselSpeeds = {json.dumps(all_vessel_speeds if scene_context else [])};  // Speed data for each vessel

        console.log('📊 Prediction points:', predictionCoords.length);
        console.log('📊 Past trajectories:', allVesselsPast.length);
        console.log('📊 Future GT trajectories:', allVesselsFutureGT.length);
        console.log('📊 Vessel speeds:', allVesselSpeeds.length);

        // Check if Leaflet loaded
        if (typeof L === 'undefined') {{
            document.getElementById('map').innerHTML =
                '<div class="error">Error: Could not load Leaflet mapping library.</div>';
        }} else {{
            console.log('✅ Leaflet loaded successfully');

            try {{
                // Initialize map - will auto-zoom to fit bounds tightly
                const map = L.map('map').setView([{center_lat}, {center_lon}], 18);

                // Add OpenStreetMap tiles with MUCH higher zoom for detailed trajectory analysis
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '© OpenStreetMap contributors',
                    maxZoom: 25,  // Allow extreme zoom for meter-level precision
                    maxNativeZoom: 19  // OSM native max (will upscale beyond this)
                }}).addTo(map);

                const allLayers = [];
                // Store layer groups per vessel for toggle control
                const vesselLayerGroups = {{}};

                // Initialize layer groups for each vessel
                vesselIds.forEach(function(vesselId) {{
                    vesselLayerGroups[vesselId] = {{
                        past: [],
                        futureGT: [],
                        prediction: []
                    }};
                }});

                // 1. PAST TRAJECTORIES (HISTORY) - SOLID LINES
                allVesselsPast.forEach(function(vesselPast, idx) {{
                    if (vesselPast.length > 0) {{
                        const vesselId = vesselIds[idx];
                        const vesselColor = vesselColors[idx];
                        const isPredicted = (vesselId === predictedVesselIdx);

                        const pastLine = L.polyline(vesselPast, {{
                            color: vesselColor,
                            weight: isPredicted ? 8 : 5,
                            opacity: 1.0,
                            dashArray: ''  // SOLID for observed history
                        }}).addTo(map);

                        // Add speed information to history popup
                        let popupText = isPredicted
                            ? '<b>📊 Predicted Vessel ' + vesselId + ' - History (Observed)</b><br>Scenario: {scenario_id}<br>Past trajectory from AIS data'
                            : '<b>Vessel ' + vesselId + ' - History (Observed)</b><br>Scenario: {scenario_id}<br>Context vessel past';

                        // Add speed data if available
                        if (allVesselSpeeds[idx] && allVesselSpeeds[idx].avg_past) {{
                            popupText += '<br><strong>Average Speed:</strong> ' + allVesselSpeeds[idx].avg_past.toFixed(2) + ' knots';
                        }}

                        pastLine.bindPopup(popupText);
                        allLayers.push(pastLine);
                        vesselLayerGroups[vesselId].past.push(pastLine);

                        // Add position markers (every point for maximum detail)
                        vesselPast.forEach(function(coord, i) {{
                            // Build popup with speed if available
                            let popupContent = '<b>Vessel ' + vesselId + ' History t=' + (i+1) + '</b><br>Lat: ' + coord[0].toFixed(6) + '<br>Lon: ' + coord[1].toFixed(6);
                            if (allVesselSpeeds[idx] && allVesselSpeeds[idx].past_speeds && allVesselSpeeds[idx].past_speeds[i] !== undefined) {{
                                popupContent += '<br><strong>Speed:</strong> ' + allVesselSpeeds[idx].past_speeds[i].toFixed(2) + ' knots';
                            }}

                            const marker = L.circleMarker(coord, {{
                                radius: isPredicted ? 6 : 4,
                                fillColor: vesselColor,
                                color: '#fff',
                                weight: 2,
                                opacity: 1,
                                fillOpacity: 0.9
                            }}).addTo(map)
                            .bindPopup(popupContent);
                            vesselLayerGroups[vesselId].past.push(marker);
                        }});
                    }}
                }});

                // 2. FUTURE GT TRAJECTORIES - DASHED LINES
                allVesselsFutureGT.forEach(function(vesselFuture, idx) {{
                    if (vesselFuture.length > 0) {{
                        const vesselId = vesselIds[idx];
                        const vesselColor = vesselColors[idx];
                        const isPredicted = (vesselId === predictedVesselIdx);

                        const futureLine = L.polyline(vesselFuture, {{
                            color: vesselColor,
                            weight: isPredicted ? 8 : 5,
                            opacity: 0.8,
                            dashArray: '10, 5'  // DASHED for ground truth future
                        }}).addTo(map);

                        // Add speed information to GT future popup
                        let futurePopupText = isPredicted
                            ? '<b>🎯 Predicted Vessel ' + vesselId + ' - Ground Truth Future</b><br>Scenario: {scenario_id}<br>Actual future path from AIS data'
                            : '<b>Vessel ' + vesselId + ' - Ground Truth Future</b><br>Scenario: {scenario_id}<br>Context vessel future';

                        // Add speed data if available
                        if (allVesselSpeeds[idx] && allVesselSpeeds[idx].avg_future) {{
                            futurePopupText += '<br><strong>Average Speed:</strong> ' + allVesselSpeeds[idx].avg_future.toFixed(2) + ' knots';
                        }}

                        futureLine.bindPopup(futurePopupText);
                        allLayers.push(futureLine);
                        vesselLayerGroups[vesselId].futureGT.push(futureLine);

                        // Add position markers for ALL GT future points
                        vesselFuture.forEach(function(coord, i) {{
                            // Build popup with speed if available
                            let futurePopup = '<b>Vessel ' + vesselId + ' GT Future t=' + (i+1) + '</b><br>Lat: ' + coord[0].toFixed(6) + '<br>Lon: ' + coord[1].toFixed(6);
                            if (allVesselSpeeds[idx] && allVesselSpeeds[idx].future_speeds && allVesselSpeeds[idx].future_speeds[i] !== undefined) {{
                                futurePopup += '<br><strong>Speed:</strong> ' + allVesselSpeeds[idx].future_speeds[i].toFixed(2) + ' knots';
                            }}

                            const marker = L.circleMarker(coord, {{
                                radius: isPredicted ? 5 : 3,
                                fillColor: vesselColor,
                                color: '#fff',
                                weight: 1,
                                opacity: 1,
                                fillOpacity: 0.8
                            }}).addTo(map)
                            .bindPopup(futurePopup);
                            vesselLayerGroups[vesselId].futureGT.push(marker);
                        }});
                    }}
                }});

                // 3. PREDICTION TRAJECTORY - DOTTED LINE
                if (predictionCoords.length > 0) {{
                    const predLine = L.polyline(predictionCoords, {{
                        color: predictedVesselColor,
                        weight: 8,
                        opacity: 0.9,
                        dashArray: '2, 5'  // DOTTED for predictions
                    }}).addTo(map);
                    // Add speed information to prediction popup
                    let predPopupText = '<b>🤖 Wayformer Prediction</b><br>Scenario: {scenario_id}<br>Predicted trajectory for vessel ' + predictedVesselIdx;

                    // Add speed data if available
                    if (allVesselSpeeds[predictedVesselIdx] && allVesselSpeeds[predictedVesselIdx].avg_pred) {{
                        predPopupText += '<br><strong>Average Speed:</strong> ' + allVesselSpeeds[predictedVesselIdx].avg_pred.toFixed(2) + ' knots';
                    }}

                    predLine.bindPopup(predPopupText);
                    allLayers.push(predLine);
                    vesselLayerGroups[predictedVesselIdx].prediction.push(predLine);

                    // Add prediction position markers (every point for prediction visibility)
                    predictionCoords.forEach(function(coord, i) {{
                        // Build popup with speed if available
                        let predMarkerPopup = '<b>🎯 Prediction t=' + (i+1) + '</b><br>Lat: ' + coord[0].toFixed(6) + '<br>Lon: ' + coord[1].toFixed(6);
                        if (allVesselSpeeds[predictedVesselIdx] && allVesselSpeeds[predictedVesselIdx].pred_speeds && allVesselSpeeds[predictedVesselIdx].pred_speeds[i] !== undefined) {{
                            predMarkerPopup += '<br><strong>Speed:</strong> ' + allVesselSpeeds[predictedVesselIdx].pred_speeds[i].toFixed(2) + ' knots';
                        }}

                        const marker = L.circleMarker(coord, {{
                            radius: 5,
                            fillColor: predictedVesselColor,
                            color: '#000',
                            weight: 2,
                            opacity: 1,
                            fillOpacity: 0.9
                        }}).addTo(map)
                        .bindPopup(predMarkerPopup);
                        vesselLayerGroups[predictedVesselIdx].prediction.push(marker);
                    }});
                }}

                // Fit map to show all trajectories
                if (allLayers.length > 0) {{
                    const group = new L.featureGroup(allLayers);
                    map.fitBounds(group.getBounds().pad(0.1));
                }}

                // Generate dynamic legend with toggle controls
                const legendContainer = document.getElementById('legend-container');
                let legendHTML = '<h4>Vessels & Trajectories:</h4>';

                vesselIds.forEach(function(vesselId, idx) {{
                    const vesselColor = vesselColors[idx];
                    const isPredicted = (vesselId === predictedVesselIdx);
                    const label = isPredicted ? '🎯 Vessel ' + vesselId + ' (Predicted)' : 'Vessel ' + vesselId;
                    const weight = isPredicted ? 'font-weight: bold;' : '';

                    legendHTML += '<div class="legend-item" style="' + weight + '">' +
                                  '<div class="legend-color" style="background-color: ' + vesselColor + ';"></div>' +
                                  '<span>' + label + '</span></div>';

                    // Add toggle controls for this vessel
                    legendHTML += '<div class="vessel-controls">' +
                                  '<div class="vessel-controls-header">' +
                                  '<span style="font-size: 10px;">Show:</span>' +
                                  '</div>' +
                                  '<button class="toggle-btn active" id="toggle-history-' + vesselId + '" onclick="toggleVesselHistory(' + vesselId + ')">History</button>' +
                                  '<button class="toggle-btn active" id="toggle-gt-' + vesselId + '" onclick="toggleVesselGT(' + vesselId + ')">GT</button>';

                    // Only show prediction toggle for the predicted vessel
                    if (isPredicted) {{
                        legendHTML += '<button class="toggle-btn active" id="toggle-pred-' + vesselId + '" onclick="toggleVesselPrediction(' + vesselId + ')">Prediction</button>';
                    }}

                    legendHTML += '</div>';
                }});

                legendHTML += '<div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd;">' +
                              '<div class="legend-item"><span style="border-bottom: 3px solid #333;">━━━</span> History (Observed)</div>' +
                              '<div class="legend-item"><span style="border-bottom: 3px dashed #333;">┅┅┅</span> Future GT (Actual)</div>' +
                              '<div class="legend-item"><span style="border-bottom: 3px dotted #333;">····</span> Prediction (Model)</div>' +
                              '</div>';

                legendContainer.innerHTML = legendHTML;

                // Toggle functions for History, GT, and Predictions per vessel
                window.toggleVesselHistory = function(vesselId) {{
                    const layers = vesselLayerGroups[vesselId].past;
                    const btn = document.getElementById('toggle-history-' + vesselId);
                    const isActive = btn.classList.contains('active');

                    layers.forEach(function(layer) {{
                        if (isActive) {{
                            map.removeLayer(layer);
                        }} else {{
                            map.addLayer(layer);
                        }}
                    }});

                    btn.classList.toggle('active');
                }};

                window.toggleVesselGT = function(vesselId) {{
                    const layers = vesselLayerGroups[vesselId].futureGT;
                    const btn = document.getElementById('toggle-gt-' + vesselId);
                    const isActive = btn.classList.contains('active');

                    layers.forEach(function(layer) {{
                        if (isActive) {{
                            map.removeLayer(layer);
                        }} else {{
                            map.addLayer(layer);
                        }}
                    }});

                    btn.classList.toggle('active');
                }};

                window.toggleVesselPrediction = function(vesselId) {{
                    const layers = vesselLayerGroups[vesselId].prediction;
                    const btn = document.getElementById('toggle-pred-' + vesselId);
                    const isActive = btn.classList.contains('active');

                    layers.forEach(function(layer) {{
                        if (isActive) {{
                            map.removeLayer(layer);
                        }} else {{
                            map.addLayer(layer);
                        }}
                    }});

                    btn.classList.toggle('active');
                }};

                // Update status
                document.getElementById('status').innerHTML = '✅ Scenario loaded: ' + vesselIds.length + ' vessels';
                setTimeout(function() {{
                    document.getElementById('status').style.display = 'none';
                }}, 3000);

                console.log('✅ Scenario map initialized with ' + vesselIds.length + ' vessels');

            }} catch (error) {{
                console.error('❌ Error initializing map:', error);
                document.getElementById('map').innerHTML =
                    '<div class="error">Error initializing map: ' + error.message + '</div>';
            }}
        }}

        window.addEventListener('error', function(e) {{
            console.error('❌ JavaScript error:', e.error);
        }});

        window.addEventListener('load', function() {{
            console.log('🌍 Scenario visualization loaded');
        }});
    </script>
</body>
</html>'''
        
        # Write HTML file
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        return output_path

    def _create_timeline_visualizations(self, output_dir):
        """
        Create timeline visualizations that combine all time window predictions for each base scenario.
        Groups scenarios by base ID (vessel + date + time) and creates combined views showing
        the complete trajectory with all predictions from different time windows.
        """
        import glob
        import re
        from collections import defaultdict

        try:
            # Find all HTML files in output directory
            html_files = glob.glob(os.path.join(output_dir, "ais_*.html"))

            # Group files by base scenario (without time offset)
            scenario_groups = defaultdict(list)
            for html_file in html_files:
                filename = os.path.basename(html_file)
                # Extract base scenario ID (remove _t{number} suffix)
                match = re.match(r'(ais_.*?)(?:_t\d+)?\.html$', filename)
                if match:
                    base_id = match.group(1)
                    scenario_groups[base_id].append(html_file)

            logger.info(f"Found {len(scenario_groups)} base scenarios for timeline creation")

            timeline_files = []
            for base_id, scenario_files in scenario_groups.items():
                if len(scenario_files) <= 1:
                    # Skip if only one time window (no timeline needed)
                    continue

                logger.info(f"Creating timeline for {base_id} with {len(scenario_files)} time windows")

                # Read and parse all HTML files for this scenario
                all_gt_coords = {}  # vessel_idx -> list of coords
                all_pred_coords = {}  # vessel_idx -> list of coord lists (one per time window)
                center_lat = None
                center_lon = None
                vessel_colors = ['#228B22', '#4169E1', '#DC143C', '#FF8C00', '#9932CC', '#FF1493', '#00CED1', '#32CD32']

                for scenario_file in sorted(scenario_files):
                    with open(scenario_file, 'r') as f:
                        html_content = f.read()

                    # Extract center coordinates from first file
                    if center_lat is None:
                        center_match = re.search(r'var map = L\.map\(\'map\'\)\.setView\(\[([0-9.\-]+), ([0-9.\-]+)\]', html_content)
                        if center_match:
                            center_lat = float(center_match.group(1))
                            center_lon = float(center_match.group(2))

                    # Extract ground truth coordinates (only from first file, as they're the same)
                    if not all_gt_coords:
                        gt_pattern = r'var vessel(\d+)_gt_coords = (\[.*?\]);'
                        for match in re.finditer(gt_pattern, html_content, re.DOTALL):
                            vessel_idx = int(match.group(1))
                            coords_str = match.group(2)
                            try:
                                coords = json.loads(coords_str)
                                all_gt_coords[vessel_idx] = coords
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse GT coords for vessel {vessel_idx}")

                    # Extract prediction coordinates from all time windows
                    pred_pattern = r'var vessel(\d+)_pred_coords = (\[.*?\]);'
                    for match in re.finditer(pred_pattern, html_content, re.DOTALL):
                        vessel_idx = int(match.group(1))
                        coords_str = match.group(2)
                        try:
                            coords = json.loads(coords_str)
                            if len(coords) > 0:  # Only add non-empty predictions
                                if vessel_idx not in all_pred_coords:
                                    all_pred_coords[vessel_idx] = []
                                all_pred_coords[vessel_idx].append(coords)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse prediction coords for vessel {vessel_idx}")

                if not all_gt_coords:
                    logger.warning(f"No ground truth data found for {base_id}")
                    continue

                # Create combined timeline visualization
                timeline_filename = f"{base_id}_timeline.html"
                timeline_path = os.path.join(output_dir, timeline_filename)

                html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Timeline: {base_id}</title>
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
'''

                # Add ground truth for each vessel
                for vessel_idx in sorted(all_gt_coords.keys()):
                    gt_coords = all_gt_coords[vessel_idx]
                    color = vessel_colors[vessel_idx % len(vessel_colors)]
                    gt_coords_json = json.dumps(gt_coords)

                    html_content += f'''
        // Vessel {vessel_idx + 1} - Complete Ground Truth
        var vessel{vessel_idx}_gt_coords = {gt_coords_json};
        if (vessel{vessel_idx}_gt_coords.length > 0) {{
            var vessel{vessel_idx}_gt_line = L.polyline(vessel{vessel_idx}_gt_coords, {{
                color: '{color}',
                weight: 4,
                opacity: 0.9
            }}).addTo(map);
            vessel{vessel_idx}_gt_line.bindPopup('<b>Vessel {vessel_idx + 1} - Complete Ground Truth</b><br>Full trajectory from AIS data');
            allLayers.push(vessel{vessel_idx}_gt_line);

            // Add direction arrows
            var vessel{vessel_idx}_gt_arrows = L.polylineDecorator(vessel{vessel_idx}_gt_line, {{
                patterns: [{{
                    offset: '10%',
                    repeat: '15%',
                    symbol: L.Symbol.arrowHead({{
                        pixelSize: 12,
                        polygon: false,
                        pathOptions: {{ stroke: true, weight: 2, color: '{color}', opacity: 0.9 }}
                    }})
                }}]
            }}).addTo(map);

            // Start/End markers
            L.circleMarker(vessel{vessel_idx}_gt_coords[0], {{
                radius: 8,
                fillColor: '{color}',
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 1
            }}).addTo(map).bindPopup('<b>Vessel {vessel_idx + 1} Start</b>');

            L.circleMarker(vessel{vessel_idx}_gt_coords[vessel{vessel_idx}_gt_coords.length-1], {{
                radius: 8,
                fillColor: '{color}',
                color: '#000',
                weight: 2,
                opacity: 1,
                fillOpacity: 1
            }}).addTo(map).bindPopup('<b>Vessel {vessel_idx + 1} End</b>');

            vessel{vessel_idx}_gt_coords.forEach(function(coord) {{ bounds.extend(coord); }});
        }}
'''

                # Add all predictions for each vessel (prediction fan)
                for vessel_idx in sorted(all_pred_coords.keys()):
                    pred_coord_lists = all_pred_coords[vessel_idx]
                    color = vessel_colors[vessel_idx % len(vessel_colors)]

                    for pred_idx, pred_coords in enumerate(pred_coord_lists):
                        pred_coords_json = json.dumps(pred_coords)
                        opacity = 0.3 + (0.3 * pred_idx / max(len(pred_coord_lists) - 1, 1))  # Fade from 0.3 to 0.6

                        html_content += f'''
        // Vessel {vessel_idx + 1} - Prediction Window {pred_idx + 1}
        var vessel{vessel_idx}_pred{pred_idx}_coords = {pred_coords_json};
        if (vessel{vessel_idx}_pred{pred_idx}_coords.length > 0) {{
            var vessel{vessel_idx}_pred{pred_idx}_line = L.polyline(vessel{vessel_idx}_pred{pred_idx}_coords, {{
                color: '{color}',
                weight: 2,
                opacity: {opacity:.2f},
                dashArray: '5, 5'
            }}).addTo(map);
            vessel{vessel_idx}_pred{pred_idx}_line.bindPopup('<b>Vessel {vessel_idx + 1} - Prediction {pred_idx + 1}</b><br>Time window prediction');
            allLayers.push(vessel{vessel_idx}_pred{pred_idx}_line);

            // Add arrows to predictions
            var vessel{vessel_idx}_pred{pred_idx}_arrows = L.polylineDecorator(vessel{vessel_idx}_pred{pred_idx}_line, {{
                patterns: [{{
                    offset: '10%',
                    repeat: '20%',
                    symbol: L.Symbol.arrowHead({{
                        pixelSize: 8,
                        polygon: false,
                        pathOptions: {{ stroke: true, weight: 1, color: '{color}', opacity: {opacity:.2f} }}
                    }})
                }}]
            }}).addTo(map);

            vessel{vessel_idx}_pred{pred_idx}_coords.forEach(function(coord) {{ bounds.extend(coord); }});
        }}
'''

                html_content += f'''
        // Fit map to show all trajectories
        if (bounds.isValid()) {{
            map.fitBounds(bounds, {{padding: [20, 20]}});
        }}

        // Add legend
        var legend = L.control({{position: 'topright'}});
        legend.onAdd = function (map) {{
            var div = L.DomUtil.create('div', 'legend');
            div.innerHTML = '<h4>Timeline: {base_id}</h4>' +
                           '<p><strong>Vessels:</strong> {len(all_gt_coords)}</p>' +
                           '<p><strong>Time Windows:</strong> {len(scenario_files)}</p>' +
                           '<p><span style="color: #228B22;">━━━</span> Ground Truth</p>' +
                           '<p><span style="color: #228B22;">┅┅┅</span> Predictions (fan)</p>' +
                           '<p><small>Each dashed line is a prediction from different time window</small></p>';
            return div;
        }};
        legend.addTo(map);

    </script>
</body>
</html>'''

                with open(timeline_path, 'w') as f:
                    f.write(html_content)

                timeline_files.append(timeline_path)
                logger.info(f"Created timeline visualization: {timeline_filename}")

            return timeline_files

        except Exception as e:
            logger.error(f"Failed to create timeline visualizations: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def on_validation_epoch_end(self, trainer, pl_module):
        """Process collected data and create visualization"""
        if not self.predictions or not self.ground_truths:
            logger.warning("No prediction data collected for visualization")
            return
        
        try:
            # Concatenate all batches
            all_predictions = np.concatenate(self.predictions, axis=0)
            all_ground_truths = np.concatenate(self.ground_truths, axis=0)
            
            # Ensure we have matching scenario IDs
            num_scenarios = all_predictions.shape[0]
            if len(self.scenario_ids) < num_scenarios:
                # Pad with generic IDs if needed
                for i in range(len(self.scenario_ids), num_scenarios):
                    self.scenario_ids.append(f"scenario_{i}")
            elif len(self.scenario_ids) > num_scenarios:
                # Trim if too many
                self.scenario_ids = self.scenario_ids[:num_scenarios]
            
            # Get metrics from trainer logs
            if hasattr(trainer, 'logged_metrics'):
                self.metrics = {k: float(v) for k, v in trainer.logged_metrics.items() 
                              if isinstance(v, (int, float, torch.Tensor))}
            
            logger.info(f"Creating visualizations for {all_predictions.shape[0]} scenarios")
            logger.info(f"Prediction shape: {all_predictions.shape}")
            logger.info(f"Ground truth shape: {all_ground_truths.shape}")
            
            # Group scenarios by vessel (remove time suffix _tXXXXX)
            vessel_groups = {}
            for i in range(num_scenarios):
                scenario_id = self.scenario_ids[i]
                # Extract vessel name (everything before the last _tXXXXX)
                # Format: ais_{vessel_name}_{date}_{time}_t{offset}
                if '_t' in scenario_id:
                    vessel_id = scenario_id.rsplit('_t', 1)[0]  # Remove _tXXXXX suffix
                else:
                    vessel_id = scenario_id

                if vessel_id not in vessel_groups:
                    vessel_groups[vessel_id] = {
                        'indices': [],
                        'predictions': [],
                        'ground_truths': []
                    }

                vessel_groups[vessel_id]['indices'].append(i)
                vessel_groups[vessel_id]['predictions'].append(all_predictions[i])
                vessel_groups[vessel_id]['ground_truths'].append(all_ground_truths[i])

            logger.info(f"Grouped {num_scenarios} scenarios into {len(vessel_groups)} unique vessels")

            # Create combined visualization for each vessel
            output_dir = "evaluation_visualizations"
            os.makedirs(output_dir, exist_ok=True)

            created_files = []

            for vessel_id, data in vessel_groups.items():
                # OPTION 1: Show only FIRST prediction for clean visualization
                # This avoids the "cloud of waypoints" issue from concatenating overlapping predictions
                # Note: Timeline visualization (*_timeline.html) still shows all predictions as a fan

                # Use only the first time window prediction
                if len(data['predictions']) > 0:
                    pred_first = data['predictions'][0]  # [future_len, 2] - first prediction only
                    gt_first = data['ground_truths'][0]    # [future_len, 2] - corresponding GT

                    # Reshape to [1, future_len, 2] for visualization function
                    combined_predictions = pred_first.reshape(1, -1, 2)
                    combined_ground_truths = gt_first.reshape(1, -1, 2)
                else:
                    # Fallback: empty predictions
                    logger.warning(f"No predictions found for vessel {vessel_id}")
                    continue

                # Clean vessel ID for filename
                clean_vessel_id = "".join(c for c in vessel_id if c.isalnum() or c in ('-', '_')).rstrip()
                if not clean_vessel_id:
                    clean_vessel_id = f"vessel_{len(created_files)}"

                # Limit filename length to avoid filesystem issues
                max_id_length = 50
                if len(clean_vessel_id) > max_id_length:
                    clean_vessel_id = f"{clean_vessel_id[:max_id_length-10]}_{hash(vessel_id) % 10000:04d}"

                output_filename = f"ais_vessel_{clean_vessel_id}.html"

                # Get first scene context for this vessel (all segments share same scene structure)
                scene_ctx = self.scene_context[data['indices'][0]] if data['indices'] and data['indices'][0] < len(self.scene_context) else None

                output_path = self._create_leaflet_visualization(
                    predictions=combined_predictions,
                    ground_truth=combined_ground_truths,
                    scenario_id=vessel_id,
                    output_dir=output_dir,
                    output_filename=output_filename,
                    metrics=self.metrics,
                    scene_context=scene_ctx
                )
                
                created_files.append(output_path)
                logger.info(f"✅ Created combined visualization for vessel {vessel_id}: {output_filename}")
            
            logger.info(f"🎉 Created {len(created_files)} scenario visualizations")
            print(f"\n🚢 AIS Trajectory Visualizations Generated!")
            print(f"📁 Location: {os.path.abspath(output_dir)}")
            print(f"📄 Files created:")
            for file_path in created_files:
                print(f"   • {os.path.basename(file_path)}")
            print(f"🌐 Open any file in browser to view interactive map")

            # Create timeline visualizations (combine predictions from all time windows)
            logger.info("Creating timeline visualizations...")
            timeline_files = self._create_timeline_visualizations(output_dir)
            if timeline_files:
                print(f"\n🕒 Timeline Visualizations Generated!")
                print(f"📄 Combined timeline files:")
                for file_path in timeline_files:
                    print(f"   • {os.path.basename(file_path)}")

        except Exception as e:
            logger.error(f"Failed to create visualization: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            print(f"❌ Visualization creation failed: {str(e)}")
        finally:
            # Clean up collected data
            self.predictions = []
            self.ground_truths = []
            self.scenario_ids = []
        self.scene_context = []  # Store multi-agent scene data


@hydra.main(version_base=None, config_path="configs", config_name="config")
def evaluation(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg['eval'] = True

    model = build_model(cfg)

    val_set = build_dataset(cfg, val=True)

    eval_batch_size = cfg.method['eval_batch_size']

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=val_set.collate_fn)

    # Create visualization callback with config
    viz_callback = EvaluationCallback(config=cfg)

    trainer = pl.Trainer(
        inference_mode=True,
        logger=None if cfg.debug else WandbLogger(project="unitraj", name=cfg.exp_name),
        devices=1,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        callbacks=[viz_callback]  # Add visualization callback
    )

    # Run evaluation
    results = trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)
    
    # Print results summary
    if results:
        print(f"\n📊 Evaluation Results:")
        for result_dict in results:
            for key, value in result_dict.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
    
    return results


if __name__ == '__main__':
    evaluation()
