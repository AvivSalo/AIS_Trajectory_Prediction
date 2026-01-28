import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
import pickle

class AISDataset(Dataset):
    """
    AIS Marine Traffic Dataset for Wayformer
    
    Processes AIS CSV files containing marine vessel trajectory data.
    Each CSV represents one scenario with multiple interacting vessels.
    """
    
    def __init__(self, config=None, is_validation=False):
        self.config = config
        self.is_validation = is_validation

        # Get data paths
        if is_validation:
            self.data_paths = config['val_data_path']
        else:
            self.data_paths = config['train_data_path']

        # AIS-specific configuration
        self.past_len = config.get('past_len', 21)  # 2.1 seconds
        self.future_len = config.get('future_len', 60)  # 6.0 seconds
        self.max_num_agents = config.get('max_num_agents', 32)
        self.trajectory_sample_interval = config.get('trajectory_sample_interval', 1)
        # Sliding window stride - how many timesteps to skip between consecutive windows
        # Smaller stride = more overlapping windows = more training data
        # Larger stride = less overlap = faster training but less data
        self.stride = config.get('stride', 10)  # Default: 10 timesteps for 1-second data

        # Disable map features for marine environment
        self.use_map_lanes = False
        self.use_map_image = False

        # Feature dimensions
        self.agent_feature_dim = config.get('num_agent_feature', 39)

        # Normalization parameters (scale data to [-1, 1] range)
        self.normalize_data = config.get('normalize_data', False)

        if self.normalize_data:
            # Use explicit position_scale for ego-relative coordinates
            # Default: 100m (typical vessel movement range)
            # NOT map_range/2 which is for scenario-relative coordinates
            self.position_scale = config.get('position_scale', 100.0)
            self.velocity_scale = config.get('velocity_scale', 20.0)
        else:
            # If not normalizing, scale doesn't matter
            self.position_scale = 1.0
            self.velocity_scale = 1.0
        
        # Load and process all CSV files
        self.scenarios = []
        self.load_data()
        
    def load_data(self):
        """Load and preprocess AIS data files (CSV or pickle)"""
        print(f"Loading {'validation' if self.is_validation else 'training'} AIS data...")

        for data_path in self.data_paths:
            # Check for pickle files in subdirectories (scenario directories)
            pickle_files = list(Path(data_path).glob("ais_*/*.pkl"))
            # Also check root level for backward compatibility
            pickle_files.extend(list(Path(data_path).glob("*.pkl")))
            # Filter out metadata files
            pickle_files = [f for f in pickle_files if f.name not in ["dataset_mapping.pkl", "dataset_summary.pkl", "file_list.pkl"]]
            csv_files = list(Path(data_path).glob("*.csv"))

            print(f"Found {len(pickle_files)} pickle files and {len(csv_files)} CSV files in {data_path}")

            # Process pickle files if available
            if pickle_files:
                for pickle_file in pickle_files:
                    try:
                        scenarios = self._process_pickle_file(pickle_file)
                        self.scenarios.extend(scenarios)
                    except Exception as e:
                        warnings.warn(f"Failed to process {pickle_file}: {e}")
                        continue

            # Process CSV files
            for csv_file in csv_files:
                try:
                    scenarios = self._process_csv_file(csv_file)
                    self.scenarios.extend(scenarios)
                except Exception as e:
                    warnings.warn(f"Failed to process {csv_file}: {e}")
                    continue

        print(f"Loaded {len(self.scenarios)} scenarios from AIS data")
        
    def _process_csv_file(self, csv_file: Path) -> List[Dict]:
        """Process a single CSV file and extract multiple overlapping scenarios"""
        
        # Read CSV data
        df = pd.read_csv(csv_file)
        
        # Clean and filter relevant columns
        relevant_cols = [
            'time', 'own_latitude', 'own_longitude', 'own_sog', 'own_cog', 'host_name',
            'target_latitude', 'target_longitude', 'target_sog', 'target_cog', 'target_target_id'
        ]
        
        # Keep only rows with required columns
        df = df[relevant_cols].dropna(subset=['time', 'own_latitude', 'own_longitude'])
        
        if len(df) == 0:
            return []
            
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        
        # Group data by agents
        ego_agent = df['host_name'].iloc[0]  # First host_name as ego
        
        # Extract ego trajectory
        ego_data = df[['time', 'own_latitude', 'own_longitude', 'own_sog', 'own_cog']].copy()
        ego_data = ego_data.rename(columns={
            'own_latitude': 'lat', 'own_longitude': 'lon', 
            'own_sog': 'sog', 'own_cog': 'cog'
        })
        ego_data['agent_id'] = ego_agent
        
        # Extract target agents
        target_data_list = []
        for _, row in df.iterrows():
            if pd.notna(row['target_target_id']):
                target_data_list.append({
                    'time': row['time'],
                    'lat': row['target_latitude'],
                    'lon': row['target_longitude'], 
                    'sog': row['target_sog'],
                    'cog': row['target_cog'],
                    'agent_id': row['target_target_id']
                })
                
        # Combine all agent data
        all_agents_data = [ego_data]
        if target_data_list:
            target_df = pd.DataFrame(target_data_list)
            # Group by agent_id and add to agent list
            for agent_id, agent_df in target_df.groupby('agent_id'):
                agent_df = agent_df.reset_index(drop=True)
                all_agents_data.append(agent_df)
        
        # Create overlapping temporal windows
        scenarios = self._create_temporal_windows(all_agents_data, csv_file.stem)
        
        return scenarios
        
    def _create_temporal_windows(self, all_agents_data: List[pd.DataFrame], scenario_id: str) -> List[Dict]:
        """Create overlapping temporal windows from agent trajectories"""
        
        scenarios = []
        
        # Find common time range across all agents
        all_times = set()
        for agent_df in all_agents_data:
            all_times.update(agent_df['time'].values)
        
        sorted_times = sorted(list(all_times))
        
        if len(sorted_times) < (self.past_len + self.future_len):
            return []  # Not enough data
            
        # Create overlapping windows (stride = past_len//2 for 50% overlap)
        stride = max(1, self.past_len // 2)
        
        for start_idx in range(0, len(sorted_times) - (self.past_len + self.future_len) + 1, stride):
            window_times = sorted_times[start_idx:start_idx + self.past_len + self.future_len]
            past_times = window_times[:self.past_len]
            future_times = window_times[self.past_len:]
            
            # Extract agent trajectories for this window
            scenario_data = self._extract_window_data(all_agents_data, past_times, future_times)
            
            if scenario_data is not None:
                scenario_data['scenario_id'] = f"{scenario_id}_{start_idx}"
                scenarios.append(scenario_data)
                
        return scenarios
        
    def _extract_window_data(self, all_agents_data: List[pd.DataFrame], past_times: List, future_times: List) -> Optional[Dict]:
        """Extract agent data for a specific temporal window"""
        
        ego_data = None
        agents_data = []
        
        # Process each agent
        for i, agent_df in enumerate(all_agents_data):
            
            # Extract past trajectory
            past_traj = self._interpolate_trajectory(agent_df, past_times)
            future_traj = self._interpolate_trajectory(agent_df, future_times)
            
            if past_traj is None or len(past_traj) < self.past_len // 2:
                continue  # Skip agents with insufficient data
                
            agent_data = {
                'agent_id': agent_df['agent_id'].iloc[0],
                'past_trajectory': past_traj,
                'future_trajectory': future_traj
            }
            
            if i == 0:  # First agent is ego
                ego_data = agent_data
            else:
                agents_data.append(agent_data)
                
        if ego_data is None:
            return None
            
        # Limit number of agents
        agents_data = agents_data[:self.max_num_agents-1]
        
        return {
            'ego': ego_data,
            'agents': agents_data,
            'past_times': past_times,
            'future_times': future_times
        }

    def _process_pickle_file(self, pickle_file: Path) -> List[Dict]:
        """Process a pickle file directly as a complete maritime scene"""
        try:
            # Load pickle data
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)

            # Process directly as a complete scene instead of converting to CSV
            return self._process_pickle_scene(data)

        except Exception as e:
            warnings.warn(f"Failed to process pickle file {pickle_file}: {e}")
            return []

    def _process_pickle_scene(self, pickle_data: Dict) -> List[Dict]:
        """Process pickle data directly as a complete maritime scene with all vessels"""
        scenario_id = pickle_data['scenario_id']
        tracks = pickle_data['tracks']  # Dict of track_id -> track_data

        # Convert tracks dict to list for processing
        track_list = [(track_id, track_data) for track_id, track_data in tracks.items()]

        # First track is ego (own ship)
        if len(track_list) == 0:
            return []

        # Extract all agents with sufficient data
        scenarios = []

        # Use first track as ego
        ego_track_id, ego_track_data = track_list[0]

        # Get trajectory data
        timestamps = ego_track_data['timestamps']
        positions = ego_track_data['state']['position']  # [num_timesteps, 2]
        velocities = ego_track_data['state']['velocity']  # [num_timesteps, 2]

        # Create overlapping windows
        total_timesteps = len(timestamps)
        window_size = self.past_len + self.future_len

        for start_idx in range(0, total_timesteps - window_size + 1, self.stride):
            end_idx = start_idx + window_size

            # Extract ego trajectory for this window
            ego_positions = positions[start_idx:end_idx]
            ego_velocities = velocities[start_idx:end_idx]

            # CRITICAL FIX: Recenter coordinates relative to ego position at LAST PAST TIMESTEP
            # This matches Waymo/nuScenes convention: current_time_index is the reference (present moment)
            # Past and future are both relative to this "current" position
            last_past_idx = start_idx + self.past_len - 1  # Index of last observed position (current_time_index)
            reference_position = positions[last_past_idx].copy()  # Ego position at "current time"
            ego_positions_centered = ego_positions - reference_position  # Subtract reference from all positions

            # Build ego trajectory with recentered coordinates
            ego_trajectory = []
            for t in range(window_size):
                ego_trajectory.append({
                    'x': float(ego_positions_centered[t, 0]),
                    'y': float(ego_positions_centered[t, 1]),
                    'vx': float(ego_velocities[t, 0]),
                    'vy': float(ego_velocities[t, 1]),
                    'valid': True
                })

            ego_data = {
                'agent_id': ego_track_id,
                'past_trajectory': ego_trajectory[:self.past_len],
                'future_trajectory': ego_trajectory[self.past_len:]
            }

            # Process other tracks as agents (also recenter relative to ego's window start position)
            agents_data = []
            for track_id, track_data in track_list[1:]:
                track_timestamps = track_data['timestamps']
                track_positions = track_data['state']['position']
                track_velocities = track_data['state']['velocity']

                # Find overlapping time indices and recenter relative to ego reference
                agent_trajectory = []
                for t in range(start_idx, end_idx):
                    if t < len(track_timestamps):
                        # CRITICAL FIX: Subtract same reference position (ego at window start)
                        centered_x = float(track_positions[t, 0] - reference_position[0])
                        centered_y = float(track_positions[t, 1] - reference_position[1])
                        agent_trajectory.append({
                            'x': centered_x,
                            'y': centered_y,
                            'vx': float(track_velocities[t, 0]),
                            'vy': float(track_velocities[t, 1]),
                            'valid': True
                        })
                    else:
                        agent_trajectory.append({
                            'x': 0.0, 'y': 0.0, 'vx': 0.0, 'vy': 0.0, 'valid': False
                        })

                agent_data = {
                    'agent_id': track_id,
                    'past_trajectory': agent_trajectory[:self.past_len],
                    'future_trajectory': agent_trajectory[self.past_len:]
                }
                agents_data.append(agent_data)

            # Create scenario for this window
            scene_data = {
                'scenario_id': f"{scenario_id}_t{start_idx}",
                'ego': ego_data,
                'agents': agents_data[:self.max_num_agents-1],
                'reference_position': reference_position.copy()  # Store centering offset for visualization
            }
            scenarios.append(scene_data)

        return scenarios

    def _convert_pickle_to_csv(self, pickle_data: Dict) -> pd.DataFrame:
        """Convert pickle data format to CSV format expected by existing pipeline"""

        # Extract data from pickle format
        scenario_id = pickle_data['scenario_id']
        obj_trajs = pickle_data['obj_trajs']  # Shape: [num_agents, time_steps, features]
        obj_trajs_mask = pickle_data['obj_trajs_mask']  # Shape: [num_agents, time_steps]
        track_to_predict = pickle_data['track_index_to_predict']

        rows = []

        # Get the ego agent (the one to predict)
        ego_agent_id = f"ego_agent_{track_to_predict}"

        # Process each agent
        for agent_idx in range(obj_trajs.shape[0]):
            agent_traj = obj_trajs[agent_idx]  # [time_steps, features]
            agent_mask = obj_trajs_mask[agent_idx]  # [time_steps]

            # Skip if agent has no valid data
            if not agent_mask.any():
                continue

            # Generate agent ID
            if agent_idx == track_to_predict:
                agent_id = ego_agent_id
                is_ego = True
            else:
                agent_id = f"agent_{agent_idx}"
                is_ego = False

            # Process each timestep
            for time_idx in range(agent_traj.shape[0]):
                if not agent_mask[time_idx]:
                    continue

                # Extract features: [time, lat, lon, velocity, heading]
                # Pickle files from ais_data_preprocessor already contain absolute lat/lon
                features = agent_traj[time_idx]
                time_val = features[0]
                lat = features[1]  # Already latitude, not relative x
                lon = features[2]  # Already longitude, not relative y
                velocity = features[3] if len(features) > 3 else 0.0
                heading = features[4] if len(features) > 4 else 0.0

                # Create row for ego agent
                if is_ego:
                    row = {
                        'time': pd.Timestamp.now() + pd.Timedelta(seconds=time_val),
                        'host_name': agent_id,
                        'own_latitude': lat,
                        'own_longitude': lon,
                        'own_sog': velocity,
                        'own_cog': heading,
                        'target_latitude': np.nan,
                        'target_longitude': np.nan,
                        'target_sog': np.nan,
                        'target_cog': np.nan,
                        'target_target_id': np.nan
                    }
                else:
                    # Create row as target for ego agent
                    row = {
                        'time': pd.Timestamp.now() + pd.Timedelta(seconds=time_val),
                        'host_name': ego_agent_id,
                        'own_latitude': np.nan,  # Will be filled by ego data
                        'own_longitude': np.nan,
                        'own_sog': np.nan,
                        'own_cog': np.nan,
                        'target_latitude': lat,
                        'target_longitude': lon,
                        'target_sog': velocity,
                        'target_cog': heading,
                        'target_target_id': agent_id
                    }

                rows.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        # Fill ego data for target rows
        ego_rows = df[df['host_name'] == ego_agent_id].copy()
        if len(ego_rows) > 0:
            ego_time_data = {}
            for _, row in ego_rows.iterrows():
                if not pd.isna(row['own_latitude']):
                    ego_time_data[row['time']] = {
                        'lat': row['own_latitude'],
                        'lon': row['own_longitude'],
                        'sog': row['own_sog'],
                        'cog': row['own_cog']
                    }

            # Fill missing ego data in target rows
            for idx, row in df.iterrows():
                if pd.isna(row['own_latitude']) and row['time'] in ego_time_data:
                    df.at[idx, 'own_latitude'] = ego_time_data[row['time']]['lat']
                    df.at[idx, 'own_longitude'] = ego_time_data[row['time']]['lon']
                    df.at[idx, 'own_sog'] = ego_time_data[row['time']]['sog']
                    df.at[idx, 'own_cog'] = ego_time_data[row['time']]['cog']

        return df.dropna(subset=['time'])

    def _relative_xy_to_latlon(self, x: float, y: float, reference_lat: float = -34.755450, reference_lon: float = 22.990367) -> Tuple[float, float]:
        """Convert relative x/y coordinates back to lat/lon"""
        import math

        # Reverse the conversion from AIS preprocessing
        lat_diff = y / 110540.0  # Convert y back to lat difference
        lon_diff = x / (111320.0 * math.cos(math.radians(reference_lat)))  # Convert x back to lon difference

        lat = reference_lat + lat_diff
        lon = reference_lon + lon_diff

        return lat, lon

    def _process_csv_dataframe(self, df: pd.DataFrame, scenario_name: str) -> List[Dict]:
        """Process CSV DataFrame using existing logic (extracted from _process_csv_file)"""

        # Clean and filter relevant columns
        relevant_cols = [
            'time', 'own_latitude', 'own_longitude', 'own_sog', 'own_cog', 'host_name',
            'target_latitude', 'target_longitude', 'target_sog', 'target_cog', 'target_target_id'
        ]

        # Keep only rows with required columns
        df = df[relevant_cols].dropna(subset=['time', 'own_latitude', 'own_longitude'])

        if len(df) == 0:
            return []

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        # Use existing CSV processing logic with the converted dataframe
        # Extract ego trajectory
        ego_agent = df['host_name'].iloc[0]  # First host_name as ego

        ego_data = df[['time', 'own_latitude', 'own_longitude', 'own_sog', 'own_cog']].copy()
        ego_data = ego_data.rename(columns={
            'own_latitude': 'lat', 'own_longitude': 'lon',
            'own_sog': 'sog', 'own_cog': 'cog'
        })
        ego_data['agent_id'] = ego_agent
        ego_data = ego_data.dropna(subset=['lat', 'lon']).drop_duplicates(subset=['time'])

        # Extract other agents from target data
        target_data = df.dropna(subset=['target_latitude', 'target_longitude'])
        other_agents = {}

        for _, row in target_data.iterrows():
            agent_id = row['target_target_id']
            if pd.isna(agent_id) or agent_id == ego_agent:
                continue

            if agent_id not in other_agents:
                other_agents[agent_id] = []

            other_agents[agent_id].append({
                'time': row['time'],
                'lat': row['target_latitude'],
                'lon': row['target_longitude'],
                'sog': row['target_sog'],
                'cog': row['target_cog'],
                'agent_id': agent_id
            })

        # Convert to DataFrames
        for agent_id in other_agents:
            other_agents[agent_id] = pd.DataFrame(other_agents[agent_id]).drop_duplicates(subset=['time'])

        # Generate scenarios using existing logic
        all_agents_data = [ego_data]
        for agent_id, agent_df in other_agents.items():
            all_agents_data.append(agent_df)

        # Use existing temporal window creation method
        scenarios = self._create_temporal_windows(all_agents_data, scenario_name)

        return scenarios

    def _interpolate_trajectory(self, agent_df: pd.DataFrame, target_times: List) -> Optional[List[Dict]]:
        """Interpolate agent trajectory at target timestamps"""
        
        if len(agent_df) < 2:
            return None
            
        trajectory = []
        
        for target_time in target_times:
            # Find closest timestamps
            time_diffs = np.abs((agent_df['time'] - target_time).dt.total_seconds())
            closest_idx = time_diffs.argmin()
            
            if time_diffs.iloc[closest_idx] <= 2.0:  # Within 2 seconds
                # Use actual data point
                row = agent_df.iloc[closest_idx]
                trajectory.append({
                    'lat': row['lat'],
                    'lon': row['lon'], 
                    'sog': row['sog'],
                    'cog': row['cog'],
                    'valid': True
                })
            else:
                # Interpolate between two closest points
                interpolated = self._interpolate_point(agent_df, target_time)
                if interpolated is not None:
                    trajectory.append(interpolated)
                else:
                    trajectory.append({
                        'lat': 0.0, 'lon': 0.0, 'sog': 0.0, 'cog': 0.0, 'valid': False
                    })
                    
        return trajectory
        
    def _interpolate_point(self, agent_df: pd.DataFrame, target_time) -> Optional[Dict]:
        """Interpolate a single point using linear interpolation"""
        
        # Find two closest points before and after target time
        before_mask = agent_df['time'] <= target_time
        after_mask = agent_df['time'] >= target_time
        
        if not before_mask.any() or not after_mask.any():
            return None
            
        before_df = agent_df[before_mask]
        after_df = agent_df[after_mask]
        
        before_point = before_df.iloc[-1]  # Last point before
        after_point = after_df.iloc[0]     # First point after
        
        # Time-based linear interpolation
        dt_total = (after_point['time'] - before_point['time']).total_seconds()
        if dt_total <= 0:
            return None
            
        dt_target = (target_time - before_point['time']).total_seconds()
        alpha = dt_target / dt_total
        
        return {
            'lat': before_point['lat'] + alpha * (after_point['lat'] - before_point['lat']),
            'lon': before_point['lon'] + alpha * (after_point['lon'] - before_point['lon']),
            'sog': before_point['sog'] + alpha * (after_point['sog'] - before_point['sog']),
            'cog': before_point['cog'] + alpha * (after_point['cog'] - before_point['cog']),
            'valid': True
        }
        
    def _latlon_to_relative_xy(self, trajectories: List[Dict], reference_lat: float, reference_lon: float) -> List[Dict]:
        """Pass-through method - coordinates are already in relative meters from preprocessor"""

        # Coordinates are already in meters from the preprocessor, just return them
        return trajectories
        
    def _create_agent_features(self, trajectory: List[Dict]) -> np.ndarray:
        """Create feature vector for agent trajectory with ego-centric heading"""

        features = np.zeros((len(trajectory), self.agent_feature_dim + 1))  # +1 for validity mask

        # Get reference heading from first valid point for ego-centric transformation
        reference_heading = None
        for point in trajectory:
            if point['valid']:
                # Calculate absolute heading from first valid velocity
                reference_heading = math.atan2(point['vx'], point['vy'])
                break

        # If no valid points, return empty features
        if reference_heading is None:
            return features

        for t, traj_point in enumerate(trajectory):
            if traj_point['valid']:
                # Core features: [x, y, vx, vy]
                # Apply normalization if enabled
                if self.normalize_data:
                    features[t, 0] = traj_point['x'] / self.position_scale
                    features[t, 1] = traj_point['y'] / self.position_scale
                    features[t, 2] = traj_point['vx'] / self.velocity_scale
                    features[t, 3] = traj_point['vy'] / self.velocity_scale
                else:
                    features[t, 0] = traj_point['x']
                    features[t, 1] = traj_point['y']
                    features[t, 2] = traj_point['vx']
                    features[t, 3] = traj_point['vy']

                # Calculate speed (absolute, unchanged)
                speed = math.sqrt(traj_point['vx']**2 + traj_point['vy']**2)
                if self.normalize_data:
                    speed = speed / self.velocity_scale

                # Calculate heading relative to ego vessel (ego-centric)
                # Absolute heading from velocity
                heading_absolute = math.atan2(traj_point['vx'], traj_point['vy'])

                # Make heading relative to reference (ego vessel's initial heading)
                heading_relative = heading_absolute - reference_heading

                # Normalize to [-π, π]
                heading_relative = (heading_relative + math.pi) % (2 * math.pi) - math.pi

                features[t, 4] = math.sin(heading_relative)  # heading_sin (ego-centric)
                features[t, 5] = math.cos(heading_relative)  # heading_cos (ego-centric)
                features[t, 6] = speed  # speed (absolute, unchanged)

                # Pad remaining features with zeros
                # In a full implementation, you might add:
                # - acceleration, angular velocity, vessel dimensions, etc.

                features[t, -1] = 1.0  # Valid mask
            else:
                features[t, -1] = 0.0  # Invalid mask

        return features
        
    def __len__(self):
        return len(self.scenarios)
        
    def __getitem__(self, idx):
        """Get a single scenario for training/evaluation"""
        
        scenario = self.scenarios[idx]
        
        # Get reference point (first timestep of ego agent)
        ego_past = scenario['ego']['past_trajectory']

        # Check if trajectories already have x/y (from pickle) or lat/lon (from CSV)
        has_xy = 'x' in ego_past[0] if ego_past else False

        if has_xy:
            # Data already in relative meters from pickle, no conversion needed
            ego_past_xy = ego_past
            ego_future_xy = scenario['ego']['future_trajectory']
            reference_lat, reference_lon = 0.0, 0.0  # Not used for pickle data
        else:
            # CSV data with lat/lon, need conversion
            if not ego_past or not ego_past[0]['valid']:
                # Fallback to first valid point
                reference_lat = reference_lon = 0.0
                for point in ego_past:
                    if point['valid']:
                        reference_lat, reference_lon = point['lat'], point['lon']
                        break
            else:
                reference_lat, reference_lon = ego_past[0]['lat'], ego_past[0]['lon']

            # Convert ego trajectory to relative coordinates
            ego_past_xy = self._latlon_to_relative_xy(ego_past, reference_lat, reference_lon)
            ego_future_xy = self._latlon_to_relative_xy(scenario['ego']['future_trajectory'], reference_lat, reference_lon)
        
        # Create ego features
        ego_features = self._create_agent_features(ego_past_xy)
        
        # Process other agents
        agents_features_list = []
        agents_future_list = []
        
        for agent_data in scenario['agents']:
            agent_past_xy = self._latlon_to_relative_xy(agent_data['past_trajectory'], reference_lat, reference_lon)
            agent_future_xy = self._latlon_to_relative_xy(agent_data['future_trajectory'], reference_lat, reference_lon)
            
            agent_features = self._create_agent_features(agent_past_xy)
            agents_features_list.append(agent_features)
            agents_future_list.append(agent_future_xy)
            
        # Pad agents to max_num_agents
        while len(agents_features_list) < (self.max_num_agents - 1):
            dummy_features = np.zeros((self.past_len, self.agent_feature_dim + 1))
            agents_features_list.append(dummy_features)
            
        # Stack agents
        agents_tensor = np.stack(agents_features_list[:self.max_num_agents-1])
        
        # Create ground truth future trajectories for ALL agents (multi-agent prediction)
        all_agents_future_gt = np.zeros((self.max_num_agents, self.future_len, 3))  # [agents, timesteps, x/y/valid]

        # Ego agent future trajectory
        for t, point in enumerate(ego_future_xy[:self.future_len]):
            if self.normalize_data:
                all_agents_future_gt[0, t, 0] = point['x'] / self.position_scale
                all_agents_future_gt[0, t, 1] = point['y'] / self.position_scale
            else:
                all_agents_future_gt[0, t, 0] = point['x']
                all_agents_future_gt[0, t, 1] = point['y']
            all_agents_future_gt[0, t, 2] = 1.0 if point['valid'] else 0.0

        # Other agents future trajectories
        for agent_idx, agent_future in enumerate(agents_future_list[:self.max_num_agents-1]):
            for t, point in enumerate(agent_future[:self.future_len]):
                if self.normalize_data:
                    all_agents_future_gt[agent_idx+1, t, 0] = point['x'] / self.position_scale
                    all_agents_future_gt[agent_idx+1, t, 1] = point['y'] / self.position_scale
                else:
                    all_agents_future_gt[agent_idx+1, t, 0] = point['x']
                    all_agents_future_gt[agent_idx+1, t, 1] = point['y']
                all_agents_future_gt[agent_idx+1, t, 2] = 1.0 if point['valid'] else 0.0
            
        # Create dummy map features (not used for marine data)
        dummy_map = np.zeros((1, 20, 7))  # [num_polylines, points_per_polyline, features]
        dummy_map_mask = np.zeros((1, 20), dtype=bool)  # [num_polylines, points_per_polyline]
        
        # Combine ego and agents into single obj_trajs tensor
        # obj_trajs: [num_agents, timesteps, features]
        all_agents_features = np.zeros((self.max_num_agents, self.past_len, self.agent_feature_dim + 1))
        all_agents_features[0] = ego_features  # Ego is first agent
        
        for i, agent_features in enumerate(agents_features_list[:self.max_num_agents-1]):
            all_agents_features[i+1] = agent_features
            
        # Create masks
        obj_trajs_mask = all_agents_features[..., -1].astype(bool)  # Use validity mask
        
        # Multi-agent prediction: predict ALL agents with valid data
        agents_with_data = []
        agents_gt_trajs_list = []
        agents_gt_masks_list = []
        agents_final_valid_idx_list = []

        for agent_idx in range(self.max_num_agents):
            # Check if agent has sufficient data
            agent_gt = all_agents_future_gt[agent_idx]
            valid_timesteps = agent_gt[:, 2].sum()

            if valid_timesteps > self.future_len // 4:  # Require at least 25% valid future data
                agents_with_data.append(agent_idx)
                agents_gt_trajs_list.append(agent_gt[:, :2])  # x, y coordinates
                agents_gt_masks_list.append(agent_gt[:, 2].astype(bool))  # validity mask
                agents_final_valid_idx_list.append(int(agent_gt[:, 2].sum()) - 1)

        # For compatibility with single-agent models, keep track_index_to_predict as ego (0)
        track_index_to_predict = np.array([0])

        # Multi-agent ground truth data
        if len(agents_with_data) > 1:
            # Multi-agent scenario
            all_agents_gt_trajs = np.stack(agents_gt_trajs_list)  # [num_valid_agents, timesteps, 2]
            all_agents_gt_masks = np.stack(agents_gt_masks_list)  # [num_valid_agents, timesteps]
            all_agents_final_valid_idx = np.array(agents_final_valid_idx_list)
            agents_to_predict = np.array(agents_with_data)
        else:
            # Fallback to single-agent (ego only)
            all_agents_gt_trajs = all_agents_future_gt[0:1, :, :2]  # [1, timesteps, 2]
            all_agents_gt_masks = all_agents_future_gt[0:1, :, 2].astype(bool)  # [1, timesteps]
            all_agents_final_valid_idx = np.array([int(all_agents_future_gt[0, :, 2].sum()) - 1])
            agents_to_predict = np.array([0])

        # Legacy compatibility fields for single-agent models
        center_objects_type = np.array([0])  # Marine vessel type
        center_gt_trajs = all_agents_future_gt[0, :, :2]  # Ego agent for compatibility
        center_gt_trajs_mask = all_agents_future_gt[0, :, 2].astype(bool)
        center_gt_final_valid_idx = np.array([int(all_agents_future_gt[0, :, 2].sum()) - 1])
        
        # Get reference position for de-centering in visualization
        reference_position = scenario.get('reference_position', np.zeros(2))

        return {
            'scenario_id': np.array([scenario['scenario_id']]),
            'obj_trajs': all_agents_features[..., :-1].astype(np.float32),  # Remove validity mask
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict,
            'map_polylines': dummy_map.astype(np.float32),
            'map_polylines_mask': dummy_map_mask,
            'center_objects_type': center_objects_type,
            'center_gt_trajs': center_gt_trajs.astype(np.float32),
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            # Multi-agent prediction fields
            'all_agents_gt_trajs': all_agents_gt_trajs.astype(np.float32),
            'all_agents_gt_masks': all_agents_gt_masks,
            'all_agents_final_valid_idx': all_agents_final_valid_idx,
            'agents_to_predict': agents_to_predict,
            'reference_position': reference_position.astype(np.float32),  # Centering offset for visualization
        }
        
    def collate_fn(self, batch_list):
        """Collate function for DataLoader"""
        
        batch_size = len(batch_list)

        # Stack all tensors
        obj_trajs = np.stack([item['obj_trajs'] for item in batch_list])
        obj_trajs_mask = np.stack([item['obj_trajs_mask'] for item in batch_list])
        track_index_to_predict = np.stack([item['track_index_to_predict'] for item in batch_list])
        map_polylines = np.stack([item['map_polylines'] for item in batch_list])
        map_polylines_mask = np.stack([item['map_polylines_mask'] for item in batch_list])
        center_objects_type = np.stack([item['center_objects_type'] for item in batch_list])
        center_gt_trajs = np.stack([item['center_gt_trajs'] for item in batch_list])
        center_gt_trajs_mask = np.stack([item['center_gt_trajs_mask'] for item in batch_list])
        center_gt_final_valid_idx = np.stack([item['center_gt_final_valid_idx'] for item in batch_list])
        scenario_id = np.concatenate([item['scenario_id'] for item in batch_list])
        reference_position = np.stack([item['reference_position'] for item in batch_list])  # [B, 2]

        # Multi-agent prediction fields - handle variable sizes
        max_agents_to_predict = max(len(item['agents_to_predict']) for item in batch_list)
        all_agents_gt_trajs_padded = []
        all_agents_gt_masks_padded = []
        all_agents_final_valid_idx_padded = []
        agents_to_predict_padded = []

        for item in batch_list:
            # Pad multi-agent data to consistent size
            agents_count = len(item['agents_to_predict'])
            if agents_count < max_agents_to_predict:
                # Pad with dummy data
                padding_needed = max_agents_to_predict - agents_count

                padded_gt = np.concatenate([
                    item['all_agents_gt_trajs'],
                    np.zeros((padding_needed, item['all_agents_gt_trajs'].shape[1], 2))
                ])
                padded_mask = np.concatenate([
                    item['all_agents_gt_masks'],
                    np.zeros((padding_needed, item['all_agents_gt_masks'].shape[1]), dtype=bool)
                ])
                padded_final_idx = np.concatenate([
                    item['all_agents_final_valid_idx'],
                    np.zeros(padding_needed, dtype=int)
                ])
                padded_agents = np.concatenate([
                    item['agents_to_predict'],
                    np.full(padding_needed, -1)  # -1 indicates dummy agent
                ])
            else:
                padded_gt = item['all_agents_gt_trajs']
                padded_mask = item['all_agents_gt_masks']
                padded_final_idx = item['all_agents_final_valid_idx']
                padded_agents = item['agents_to_predict']

            all_agents_gt_trajs_padded.append(padded_gt)
            all_agents_gt_masks_padded.append(padded_mask)
            all_agents_final_valid_idx_padded.append(padded_final_idx)
            agents_to_predict_padded.append(padded_agents)

        # Stack padded multi-agent data
        all_agents_gt_trajs = np.stack(all_agents_gt_trajs_padded)
        all_agents_gt_masks = np.stack(all_agents_gt_masks_padded)
        all_agents_final_valid_idx = np.stack(all_agents_final_valid_idx_padded)
        agents_to_predict = np.stack(agents_to_predict_padded)
        
        # Create trajectory type array (all marine vessels = type 0)
        trajectory_type = np.zeros(batch_size, dtype=np.int64)
        
        # Create kalman difficulty array (set to medium difficulty = 45) 
        # Shape: [batch_size, 3] for difficulties at 2s, 4s, 6s
        kalman_difficulty = np.full((batch_size, 3), 45, dtype=np.int64)
        
        # Convert to tensors
        input_dict = {
            'obj_trajs': torch.from_numpy(obj_trajs),
            'obj_trajs_mask': torch.from_numpy(obj_trajs_mask),
            'track_index_to_predict': torch.from_numpy(track_index_to_predict),
            'map_polylines': torch.from_numpy(map_polylines),
            'map_polylines_mask': torch.from_numpy(map_polylines_mask),
            'center_objects_type': center_objects_type,
            'center_gt_trajs': torch.from_numpy(center_gt_trajs),
            'center_gt_trajs_mask': torch.from_numpy(center_gt_trajs_mask),
            'center_gt_final_valid_idx': torch.from_numpy(center_gt_final_valid_idx),
            'trajectory_type': torch.from_numpy(trajectory_type),
            'kalman_difficulty': torch.from_numpy(kalman_difficulty),
            'scenario_id': scenario_id,
            'dataset_name': ['ais'] * batch_size,  # Dataset identifier
            # Multi-agent prediction fields
            'all_agents_gt_trajs': torch.from_numpy(all_agents_gt_trajs),
            'all_agents_gt_masks': torch.from_numpy(all_agents_gt_masks),
            'all_agents_final_valid_idx': torch.from_numpy(all_agents_final_valid_idx),
            'agents_to_predict': torch.from_numpy(agents_to_predict),
            'reference_position': torch.from_numpy(reference_position),  # [B, 2] - centering offset
        }
        
        return {
            'batch_size': batch_size,
            'input_dict': input_dict,
            'batch_sample_count': batch_size
        } 