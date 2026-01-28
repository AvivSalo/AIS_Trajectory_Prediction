#!/usr/bin/env python3
"""
AIS Trajectory Visualization Tool using Kepler.gl

This module creates interactive HTML visualizations of AIS trajectory predictions
vs ground truth using Kepler.gl. It processes model outputs and generates
beautiful, interactive maps for marine trajectory analysis.

Author: UniTraj AIS Integration
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import keplergl
import pandas as pd
import numpy as np
import json
import os
from jinja2 import Environment, FileSystemLoader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AISTrajectoryVisualizer:
    """
    Visualize AIS trajectory predictions vs ground truth using Kepler.gl
    
    This class creates interactive HTML visualizations that allow users to:
    - Compare predicted vs actual vessel trajectories
    - Animate trajectory playback over time
    - Analyze prediction accuracy spatially
    - Export results for presentations and reports
    """
    
    def __init__(self, output_dir: str = "visualization_outputs"):
        """
        Initialize the AIS trajectory visualizer
        
        Args:
            output_dir: Directory to save HTML visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Kepler.gl configuration for marine trajectory visualization
        self.kepler_config = self._create_kepler_config()
        
    def _create_kepler_config(self) -> Dict[str, Any]:
        """
        Create optimized Kepler.gl configuration for marine trajectory visualization
        
        Returns:
            Dictionary containing Kepler.gl configuration
        """
        return {
            "version": "v1",
            "config": {
                "visState": {
                    "filters": [
                        {
                            "dataId": ["predictions", "ground_truth"],
                            "id": "time_filter",
                            "name": ["timestamp"],
                            "type": "timeRange",
                            "value": [0, 1000000000],
                            "enlarged": True,
                            "plotType": "histogram",
                            "animationWindow": "free",
                            "yAxis": None,
                            "speed": 1
                        }
                    ],
                    "layers": [
                        {
                            "id": "ground_truth_trajectory",
                            "type": "trip",
                            "config": {
                                "dataId": "ground_truth",
                                "label": "Ground Truth Trajectory",
                                "color": [34, 139, 34],  # Forest Green
                                "highlightColor": [252, 242, 26, 255],
                                "columns": {
                                    "geojson": "trajectory"
                                },
                                "isVisible": True,
                                "visConfig": {
                                    "opacity": 0.8,
                                    "thickness": 3,
                                    "trailLength": 120,
                                    "currentTime": None
                                },
                                "hidden": False,
                                "textLabel": [
                                    {
                                        "field": {
                                            "name": "vessel_id",
                                            "type": "string"
                                        },
                                        "color": [255, 255, 255],
                                        "size": 18,
                                        "offset": [0, 0],
                                        "anchor": "start",
                                        "alignment": "center"
                                    }
                                ]
                            },
                            "visualChannels": {
                                "colorField": {
                                    "name": "vessel_type",
                                    "type": "string"
                                },
                                "colorScale": "quantile",
                                "sizeField": None,
                                "sizeScale": "linear"
                            }
                        },
                        {
                            "id": "predicted_trajectory",
                            "type": "trip",
                            "config": {
                                "dataId": "predictions",
                                "label": "Predicted Trajectory",
                                "color": [255, 69, 0],  # Red Orange
                                "highlightColor": [252, 242, 26, 255],
                                "columns": {
                                    "geojson": "trajectory"
                                },
                                "isVisible": True,
                                "visConfig": {
                                    "opacity": 0.8,
                                    "thickness": 2,
                                    "trailLength": 120,
                                    "currentTime": None
                                },
                                "hidden": False,
                                "textLabel": [
                                    {
                                        "field": {
                                            "name": "vessel_id",
                                            "type": "string"
                                        },
                                        "color": [255, 255, 255],
                                        "size": 18,
                                        "offset": [0, 0],
                                        "anchor": "start",
                                        "alignment": "center"
                                    }
                                ]
                            },
                            "visualChannels": {
                                "colorField": {
                                    "name": "prediction_mode",
                                    "type": "integer"
                                },
                                "colorScale": "quantile",
                                "sizeField": None,
                                "sizeScale": "linear"
                            }
                        },
                        {
                            "id": "vessel_positions",
                            "type": "point",
                            "config": {
                                "dataId": "positions",
                                "label": "Vessel Positions",
                                "color": [30, 144, 255],  # Dodger Blue
                                "highlightColor": [252, 242, 26, 255],
                                "columns": {
                                    "lat": "latitude",
                                    "lng": "longitude",
                                    "altitude": None
                                },
                                "isVisible": True,
                                "visConfig": {
                                    "radius": 8,
                                    "fixedRadius": False,
                                    "opacity": 0.8,
                                    "outline": True,
                                    "thickness": 2,
                                    "strokeColor": [255, 255, 255],
                                    "colorRange": {
                                        "name": "Global Warming",
                                        "type": "sequential",
                                        "category": "Uber",
                                        "colors": [
                                            "#5A1846",
                                            "#900C3F",
                                            "#C70039",
                                            "#E3611C",
                                            "#F1920E",
                                            "#FFC300"
                                        ]
                                    },
                                    "strokeColorRange": {
                                        "name": "Global Warming",
                                        "type": "sequential",
                                        "category": "Uber",
                                        "colors": [
                                            "#5A1846",
                                            "#900C3F",
                                            "#C70039",
                                            "#E3611C",
                                            "#F1920E",
                                            "#FFC300"
                                        ]
                                    },
                                    "radiusRange": [0, 50],
                                    "filled": True
                                },
                                "hidden": False,
                                "textLabel": [
                                    {
                                        "field": {
                                            "name": "vessel_id",
                                            "type": "string"
                                        },
                                        "color": [255, 255, 255],
                                        "size": 18,
                                        "offset": [0, 0],
                                        "anchor": "start",
                                        "alignment": "center"
                                    }
                                ]
                            },
                            "visualChannels": {
                                "colorField": {
                                    "name": "data_type",
                                    "type": "string"
                                },
                                "colorScale": "ordinal",
                                "strokeColorField": None,
                                "strokeColorScale": "quantile",
                                "sizeField": {
                                    "name": "speed",
                                    "type": "real"
                                },
                                "sizeScale": "linear"
                            }
                        }
                    ],
                    "interactionConfig": {
                        "tooltip": {
                            "fieldsToShow": {
                                "predictions": [
                                    {"name": "vessel_id", "format": None},
                                    {"name": "timestamp", "format": None},
                                    {"name": "prediction_mode", "format": None},
                                    {"name": "confidence", "format": None}
                                ],
                                "ground_truth": [
                                    {"name": "vessel_id", "format": None},
                                    {"name": "timestamp", "format": None},
                                    {"name": "speed", "format": None},
                                    {"name": "heading", "format": None}
                                ],
                                "positions": [
                                    {"name": "vessel_id", "format": None},
                                    {"name": "data_type", "format": None},
                                    {"name": "latitude", "format": None},
                                    {"name": "longitude", "format": None},
                                    {"name": "speed", "format": None},
                                    {"name": "heading", "format": None}
                                ]
                            },
                            "compareMode": False,
                            "compareType": "absolute",
                            "enabled": True
                        },
                        "brush": {"size": 0.5, "enabled": False},
                        "geocoder": {"enabled": False},
                        "coordinate": {"enabled": False}
                    },
                    "layerBlending": "normal",
                    "splitMaps": [],
                    "animationConfig": {
                        "currentTime": None,
                        "speed": 1
                    }
                },
                "mapState": {
                    "bearing": 0,
                    "dragRotate": False,
                    "latitude": 0,
                    "longitude": 0,
                    "pitch": 0,
                    "zoom": 8,
                    "isSplit": False
                },
                "mapStyle": {
                    "styleType": "dark",
                    "topLayerGroups": {},
                    "visibleLayerGroups": {
                        "label": True,
                        "road": True,
                        "border": False,
                        "building": True,
                        "water": True,
                        "land": True,
                        "3d building": False
                    },
                    "threeDBuildingColor": [
                        9.665468314072013,
                        17.18305478057247,
                        31.1442867897876
                    ],
                    "mapStyles": {}
                }
            }
        }
    
    def process_model_predictions(
        self, 
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        scenario_ids: List[str],
        timestamps: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Process model predictions and ground truth for visualization
        
        Args:
            predictions: Model predictions [batch_size, future_len, 2] (lat, lon)
            ground_truth: Ground truth trajectories [batch_size, future_len, 2]
            scenario_ids: List of scenario identifiers
            timestamps: Optional timestamps for each prediction point
            
        Returns:
            Tuple of (predictions_df, ground_truth_df, positions_df)
        """
        logger.info("Processing model predictions for visualization...")
        
        batch_size, future_len, _ = predictions.shape
        
        # Generate timestamps if not provided
        if timestamps is None:
            timestamps = self._generate_timestamps(batch_size, future_len)
        
        # Process predictions
        predictions_data = []
        ground_truth_data = []
        positions_data = []
        
        for i in range(batch_size):
            scenario_id = scenario_ids[i] if i < len(scenario_ids) else f"scenario_{i}"
            vessel_id = f"vessel_{scenario_id}"
            
            # Extract valid prediction points (non-zero)
            pred_mask = ~((predictions[i] == 0).all(axis=1))
            gt_mask = ~((ground_truth[i] == 0).all(axis=1))
            
            if pred_mask.sum() > 1:  # Need at least 2 points for trajectory
                pred_coords = predictions[i][pred_mask]
                pred_times = timestamps[i][pred_mask] if timestamps is not None else np.arange(pred_mask.sum())
                
                # Create trajectory GeoJSON for predictions
                pred_trajectory = self._create_trajectory_geojson(
                    pred_coords, pred_times, vessel_id
                )
                
                predictions_data.append({
                    'vessel_id': vessel_id,
                    'scenario_id': scenario_id,
                    'trajectory': json.dumps(pred_trajectory),
                    'prediction_mode': 1,  # Best prediction mode
                    'confidence': 0.8,  # Placeholder confidence
                    'start_time': pred_times[0],
                    'end_time': pred_times[-1]
                })
                
                # Add individual position points for predictions
                for j, (coord, time) in enumerate(zip(pred_coords, pred_times)):
                    positions_data.append({
                        'vessel_id': vessel_id,
                        'latitude': float(coord[0]),
                        'longitude': float(coord[1]),
                        'timestamp': time,
                        'data_type': 'prediction',
                        'speed': 10.0,  # Placeholder
                        'heading': 0.0,  # Placeholder
                        'point_index': j
                    })
            
            if gt_mask.sum() > 1:  # Need at least 2 points for trajectory
                gt_coords = ground_truth[i][gt_mask]
                gt_times = timestamps[i][gt_mask] if timestamps is not None else np.arange(gt_mask.sum())
                
                # Create trajectory GeoJSON for ground truth
                gt_trajectory = self._create_trajectory_geojson(
                    gt_coords, gt_times, vessel_id
                )
                
                ground_truth_data.append({
                    'vessel_id': vessel_id,
                    'scenario_id': scenario_id,
                    'trajectory': json.dumps(gt_trajectory),
                    'vessel_type': 'cargo',  # Placeholder
                    'start_time': gt_times[0],
                    'end_time': gt_times[-1]
                })
                
                # Add individual position points for ground truth
                for j, (coord, time) in enumerate(zip(gt_coords, gt_times)):
                    positions_data.append({
                        'vessel_id': vessel_id,
                        'latitude': float(coord[0]),
                        'longitude': float(coord[1]),
                        'timestamp': time,
                        'data_type': 'ground_truth',
                        'speed': 12.0,  # Placeholder
                        'heading': 0.0,  # Placeholder
                        'point_index': j
                    })
        
        # Create DataFrames
        predictions_df = pd.DataFrame(predictions_data)
        ground_truth_df = pd.DataFrame(ground_truth_data)
        positions_df = pd.DataFrame(positions_data)
        
        logger.info(f"Processed {len(predictions_df)} prediction trajectories")
        logger.info(f"Processed {len(ground_truth_df)} ground truth trajectories")
        logger.info(f"Processed {len(positions_df)} position points")
        
        return predictions_df, ground_truth_df, positions_df
    
    def _generate_timestamps(self, batch_size: int, future_len: int) -> np.ndarray:
        """
        Generate timestamps for trajectory points
        
        Args:
            batch_size: Number of scenarios
            future_len: Number of future timesteps
            
        Returns:
            Timestamp array [batch_size, future_len]
        """
        base_time = datetime.now().timestamp()
        timestamps = np.zeros((batch_size, future_len))
        
        for i in range(batch_size):
            start_time = base_time + i * 3600  # 1 hour apart
            for j in range(future_len):
                timestamps[i, j] = start_time + j * 60  # 1 minute intervals
        
        return timestamps
    
    def _create_trajectory_geojson(
        self, 
        coordinates: np.ndarray, 
        timestamps: np.ndarray, 
        vessel_id: str
    ) -> Dict[str, Any]:
        """
        Create GeoJSON trajectory for Kepler.gl trip layer
        
        Args:
            coordinates: Array of [lat, lon] coordinates
            timestamps: Array of timestamps
            vessel_id: Vessel identifier
            
        Returns:
            GeoJSON trajectory object
        """
        # Convert coordinates to [lon, lat] for GeoJSON standard
        coords_lonlat = [[float(coord[1]), float(coord[0]), float(ts)] 
                        for coord, ts in zip(coordinates, timestamps)]
        
        return {
            "type": "Feature",
            "properties": {
                "vendor": vessel_id,
                "timestamps": timestamps.tolist()
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords_lonlat
            }
        }
    
    def _calculate_map_bounds(
        self, 
        predictions_df: pd.DataFrame, 
        ground_truth_df: pd.DataFrame,
        positions_df: pd.DataFrame
    ) -> Tuple[float, float, float, float]:
        """
        Calculate optimal map bounds for the visualization
        
        Args:
            predictions_df: Predictions DataFrame
            ground_truth_df: Ground truth DataFrame
            positions_df: Positions DataFrame
            
        Returns:
            Tuple of (min_lat, max_lat, min_lon, max_lon)
        """
        all_lats = []
        all_lons = []
        
        # Extract coordinates from positions
        if not positions_df.empty:
            all_lats.extend(positions_df['latitude'].tolist())
            all_lons.extend(positions_df['longitude'].tolist())
        
        if all_lats and all_lons:
            min_lat, max_lat = min(all_lats), max(all_lats)
            min_lon, max_lon = min(all_lons), max(all_lons)
            
            # Add padding (10% of range)
            lat_padding = (max_lat - min_lat) * 0.1
            lon_padding = (max_lon - min_lon) * 0.1
            
            return (
                min_lat - lat_padding,
                max_lat + lat_padding,
                min_lon - lon_padding,
                max_lon + lon_padding
            )
        else:
            # Default to global view
            return (-90, 90, -180, 180)
    
    def create_visualization(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        scenario_ids: List[str],
        output_filename: str = "ais_trajectory_visualization.html",
        timestamps: Optional[np.ndarray] = None,
        title: str = "AIS Trajectory Prediction Visualization"
    ) -> str:
        """
        Create interactive Kepler.gl visualization HTML file
        
        Args:
            predictions: Model predictions array
            ground_truth: Ground truth array
            scenario_ids: List of scenario identifiers
            output_filename: Output HTML filename
            timestamps: Optional timestamp array
            title: Visualization title
            
        Returns:
            Path to generated HTML file
        """
        logger.info(f"Creating Kepler.gl visualization: {title}")
        
        # Process data
        predictions_df, ground_truth_df, positions_df = self.process_model_predictions(
            predictions, ground_truth, scenario_ids, timestamps
        )
        
        # Calculate map bounds
        min_lat, max_lat, min_lon, max_lon = self._calculate_map_bounds(
            predictions_df, ground_truth_df, positions_df
        )
        
        # Update map state with calculated bounds
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        
        # Calculate appropriate zoom level
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        max_range = max(lat_range, lon_range)
        
        # Rough zoom calculation (adjust as needed)
        if max_range > 10:
            zoom = 6
        elif max_range > 5:
            zoom = 8
        elif max_range > 1:
            zoom = 10
        else:
            zoom = 12
        
        # Update config with calculated values
        config = self.kepler_config.copy()
        config["config"]["mapState"]["latitude"] = center_lat
        config["config"]["mapState"]["longitude"] = center_lon
        config["config"]["mapState"]["zoom"] = zoom
        
        # Prepare datasets for Kepler.gl
        datasets = []
        
        if not predictions_df.empty:
            datasets.append({
                "info": {"id": "predictions", "label": "Predictions"},
                "data": predictions_df.to_dict('records')
            })
        
        if not ground_truth_df.empty:
            datasets.append({
                "info": {"id": "ground_truth", "label": "Ground Truth"},
                "data": ground_truth_df.to_dict('records')
            })
        
        if not positions_df.empty:
            datasets.append({
                "info": {"id": "positions", "label": "Vessel Positions"},
                "data": positions_df.to_dict('records')
            })
        
        # Generate HTML
        html_content = self._generate_html(datasets, config, title)
        
        # Save HTML file
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Visualization saved to: {output_path}")
        return output_path
    
    def _generate_html(
        self, 
        datasets: List[Dict], 
        config: Dict, 
        title: str
    ) -> str:
        """
        Generate complete HTML file with embedded Kepler.gl
        
        Args:
            datasets: List of datasets for visualization
            config: Kepler.gl configuration
            title: Page title
            
        Returns:
            Complete HTML content
        """
        # Convert datasets and config to JSON strings
        datasets_json = json.dumps(datasets, indent=2)
        config_json = json.dumps(config, indent=2)
        
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; font-family: Arial, sans-serif; }}
        #map {{ position: absolute; width: 100%; height: 100%; }}
        .loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 18px;
            color: #333;
            text-align: center;
        }}
        .error {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #ffebee;
            border: 1px solid #f44336;
            border-radius: 5px;
            padding: 20px;
            max-width: 500px;
            color: #d32f2f;
        }}
        .info-panel {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px;
            border-radius: 5px;
            font-family: Arial, sans-serif;
            z-index: 1000;
            max-width: 300px;
        }}
        .info-panel h3 {{ margin-top: 0; color: #4CAF50; }}
        .legend {{
            margin-top: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 3px;
            margin-right: 10px;
            border-radius: 2px;
        }}
        .ground-truth {{ background-color: #228B22; }}
        .prediction {{ background-color: #FF4500; }}
        .position {{ background-color: #1E90FF; }}
    </style>
</head>
<body>
    <div id="map">
        <div class="loading">
            🚢 Loading AIS Trajectory Visualization...<br>
            <small>Please wait while Kepler.gl initializes</small>
        </div>
    </div>
    <div class="info-panel" id="info-panel" style="display: none;">
        <h3>🚢 AIS Trajectory Visualization</h3>
        <p><strong>Model:</strong> Wayformer-AIS</p>
        <p><strong>Scenarios:</strong> {len([d for d in datasets if d["info"]["id"] == "predictions"])}</p>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color ground-truth"></div>
                <span>Ground Truth</span>
            </div>
            <div class="legend-item">
                <div class="legend-color prediction"></div>
                <span>Predictions</span>
            </div>
            <div class="legend-item">
                <div class="legend-color position"></div>
                <span>Vessel Positions</span>
            </div>
        </div>
        <p><small>💡 Click play button to animate trajectories</small></p>
        <p><small>🎯 Use filters to focus on specific vessels</small></p>
    </div>
    
    <!-- Load Kepler.gl -->
    <script src="https://unpkg.com/kepler.gl@3.0.0/umd/keplergl.min.js"></script>
    
    <script>
        console.log('🚢 Starting AIS Trajectory Visualization...');
        
        // Data and configuration
        const datasets = {datasets_json};
        const config = {config_json};
        
        console.log('📊 Datasets:', datasets);
        console.log('⚙️ Configuration:', config);
        
        // Check if Kepler.gl loaded
        if (typeof KeplerGl === 'undefined') {{
            console.error('❌ Kepler.gl failed to load from CDN');
            document.getElementById('map').innerHTML = 
                '<div class="error">Error: Could not load Kepler.gl from CDN.<br>Please check your internet connection.</div>';
        }} else {{
            console.log('✅ Kepler.gl loaded successfully');
            
            try {{
                // Initialize Kepler.gl
                const keplerGl = new KeplerGl({{
                    mapboxApiAccessToken: 'pk.eyJ1IjoidWJlcmRhdGEiLCJhIjoiY2pwY3owbGFxMDVwNTNxbzlkdWc3dXh5eCJ9.VxOxqFkKxRJfvj9e2Oj8Rg',
                    width: window.innerWidth,
                    height: window.innerHeight,
                    appName: 'AIS Trajectory Visualization',
                    version: 'v1.0'
                }});
                
                // Mount Kepler.gl to the DOM
                const app = document.getElementById('map');
                app.innerHTML = ''; // Clear loading message
                app.appendChild(keplerGl.getElement());
                
                // Show info panel
                document.getElementById('info-panel').style.display = 'block';
                
                // Add data to map
                keplerGl.addDataToMap({{
                    datasets: datasets,
                    config: config.config,
                    options: {{
                        centerMap: true,
                        readOnly: false
                    }}
                }});
                
                console.log('✅ Map initialized with', datasets.length, 'datasets');
                
                // Handle window resize
                window.addEventListener('resize', function() {{
                    keplerGl.updateSize({{
                        width: window.innerWidth,
                        height: window.innerHeight
                    }});
                }});
                
            }} catch (error) {{
                console.error('❌ Error initializing Kepler.gl:', error);
                document.getElementById('map').innerHTML = 
                    '<div class="error">Error initializing visualization:<br>' + error.message + 
                    '<br><br>Please check the browser console for more details.</div>';
            }}
        }}
        
        // Additional error handling
        window.addEventListener('error', function(e) {{
            console.error('❌ JavaScript error:', e.error);
        }});
        
        window.addEventListener('load', function() {{
            console.log('🌍 Page fully loaded');
        }});
    </script>
</body>
</html>
"""
        return html_template
    
    def create_comparison_report(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        scenario_ids: List[str],
        metrics: Dict[str, float],
        output_filename: str = "ais_prediction_report.html"
    ) -> str:
        """
        Create a comprehensive comparison report with metrics and visualization
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth trajectories
            scenario_ids: Scenario identifiers
            metrics: Dictionary of evaluation metrics
            output_filename: Output filename
            
        Returns:
            Path to generated report
        """
        # Create main visualization
        viz_path = self.create_visualization(
            predictions, ground_truth, scenario_ids,
            output_filename.replace('.html', '_map.html'),
            title="AIS Prediction Analysis"
        )
        
        # Calculate additional statistics
        batch_size = predictions.shape[0]
        avg_trajectory_length = np.mean([
            np.sum(~((predictions[i] == 0).all(axis=1))) for i in range(batch_size)
        ])
        
        # Generate report HTML
        report_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AIS Trajectory Prediction Report</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5; 
        }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .header p {{ margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9; }}
        .content {{ padding: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: #f8f9fa; border-left: 4px solid #4CAF50; padding: 20px; border-radius: 5px; }}
        .metric-card h3 {{ margin: 0 0 10px 0; color: #333; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #4CAF50; }}
        .metric-unit {{ font-size: 0.8em; color: #666; margin-left: 5px; }}
        .section {{ margin: 30px 0; }}
        .section h2 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        .viz-embed {{ width: 100%; height: 600px; border: none; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stats-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        .stats-table th, .stats-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .stats-table th {{ background-color: #f8f9fa; font-weight: 600; }}
        .footer {{ text-align: center; padding: 20px; background-color: #f8f9fa; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚢 AIS Trajectory Prediction Report</h1>
            <p>Wayformer-AIS Model Performance Analysis</p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="content">
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>📊 Average Displacement Error</h3>
                    <div class="metric-value">{metrics.get('minADE6', 0):.2f}<span class="metric-unit">m</span></div>
                </div>
                <div class="metric-card">
                    <h3>🎯 Final Displacement Error</h3>
                    <div class="metric-value">{metrics.get('minFDE6', 0):.2f}<span class="metric-unit">m</span></div>
                </div>
                <div class="metric-card">
                    <h3>📈 Miss Rate</h3>
                    <div class="metric-value">{metrics.get('miss_rate', 0):.1%}</div>
                </div>
                <div class="metric-card">
                    <h3>🚢 Scenarios Analyzed</h3>
                    <div class="metric-value">{batch_size}</div>
                </div>
            </div>
            
            <div class="section">
                <h2>📍 Interactive Trajectory Visualization</h2>
                <p>The map below shows predicted trajectories (red) vs ground truth (green) for all analyzed scenarios. 
                Use the time filter to animate trajectories and compare prediction accuracy over time.</p>
                <iframe class="viz-embed" src="{os.path.basename(viz_path)}"></iframe>
            </div>
            
            <div class="section">
                <h2>📈 Performance Statistics</h2>
                <table class="stats-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>minADE6</strong></td>
                            <td>{metrics.get('minADE6', 0):.2f} m</td>
                            <td>Minimum Average Displacement Error over 6 prediction modes</td>
                        </tr>
                        <tr>
                            <td><strong>minFDE6</strong></td>
                            <td>{metrics.get('minFDE6', 0):.2f} m</td>
                            <td>Minimum Final Displacement Error over 6 prediction modes</td>
                        </tr>
                        <tr>
                            <td><strong>Miss Rate</strong></td>
                            <td>{metrics.get('miss_rate', 0):.1%}</td>
                            <td>Percentage of predictions with FDE > 2m threshold</td>
                        </tr>
                        <tr>
                            <td><strong>Avg Trajectory Length</strong></td>
                            <td>{avg_trajectory_length:.1f} points</td>
                            <td>Average number of predicted trajectory points</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>🔍 Analysis Insights</h2>
                <ul>
                    <li><strong>Prediction Accuracy:</strong> {'Excellent' if metrics.get('minFDE6', 1000) < 50 else 'Good' if metrics.get('minFDE6', 1000) < 200 else 'Needs Improvement'} - Final displacement error of {metrics.get('minFDE6', 0):.1f}m indicates {'high' if metrics.get('minFDE6', 1000) < 50 else 'moderate' if metrics.get('minFDE6', 1000) < 200 else 'low'} prediction accuracy</li>
                    <li><strong>Model Performance:</strong> Miss rate of {metrics.get('miss_rate', 0):.1%} shows {'strong' if metrics.get('miss_rate', 1) < 0.2 else 'acceptable' if metrics.get('miss_rate', 1) < 0.5 else 'concerning'} overall performance</li>
                    <li><strong>Trajectory Quality:</strong> Average trajectory length of {avg_trajectory_length:.1f} points provides {'detailed' if avg_trajectory_length > 30 else 'adequate' if avg_trajectory_length > 15 else 'limited'} prediction resolution</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by UniTraj AIS Trajectory Visualizer | Powered by Kepler.gl</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, output_filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Comprehensive report saved to: {report_path}")
        return report_path


def create_ais_visualization(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    scenario_ids: List[str],
    output_dir: str = "visualization_outputs",
    output_filename: str = "ais_trajectory_visualization.html",
    metrics: Optional[Dict[str, float]] = None
) -> str:
    """
    Convenience function to create AIS trajectory visualization
    
    Args:
        predictions: Model predictions array [batch_size, future_len, 2]
        ground_truth: Ground truth array [batch_size, future_len, 2]
        scenario_ids: List of scenario identifiers
        output_dir: Output directory
        output_filename: Output filename
        metrics: Optional evaluation metrics
        
    Returns:
        Path to generated HTML file
    """
    visualizer = AISTrajectoryVisualizer(output_dir)
    
    if metrics:
        return visualizer.create_comparison_report(
            predictions, ground_truth, scenario_ids, metrics, output_filename
        )
    else:
        return visualizer.create_visualization(
            predictions, ground_truth, scenario_ids, output_filename
        )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Create AIS trajectory visualization')
    parser.add_argument('--predictions', required=True, help='Path to predictions numpy file')
    parser.add_argument('--ground_truth', required=True, help='Path to ground truth numpy file')
    parser.add_argument('--scenario_ids', help='Path to scenario IDs file')
    parser.add_argument('--output_dir', default='visualization_outputs', help='Output directory')
    parser.add_argument('--output_file', default='ais_visualization.html', help='Output filename')
    
    args = parser.parse_args()
    
    # Load data
    predictions = np.load(args.predictions)
    ground_truth = np.load(args.ground_truth)
    
    if args.scenario_ids:
        with open(args.scenario_ids, 'r') as f:
            scenario_ids = [line.strip() for line in f]
    else:
        scenario_ids = [f"scenario_{i}" for i in range(predictions.shape[0])]
    
    # Create visualization
    output_path = create_ais_visualization(
        predictions, ground_truth, scenario_ids,
        args.output_dir, args.output_file
    )
    
    print(f"✅ Visualization created: {output_path}") 