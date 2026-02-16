"""
Core visualization components for trajectory prediction.

This module contains base classes and utilities for creating trajectory visualizations:
- VesselColorPalette: Color management for multi-vessel visualization
- BaseVisualizer: Abstract base class for all visualizers
- VisualizationFactory: Factory pattern for creating visualizers
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VesselColorPalette:
    """Color palette manager for multiple vessels in maritime visualization."""

    # Maritime-themed color palette - 12 distinct colors
    DEFAULT_COLORS = [
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

    def __init__(self, colors: Optional[List[str]] = None):
        """
        Initialize color palette.

        Args:
            colors: Optional custom color list (hex colors)
        """
        self.colors = colors if colors is not None else self.DEFAULT_COLORS

    def get_color(self, index: int) -> str:
        """
        Get color for vessel by index (cycles through palette).

        Args:
            index: Vessel index

        Returns:
            Hex color string
        """
        return self.colors[index % len(self.colors)]

    def get_predicted_vessel_color(self, predicted_idx: int, num_agents: int) -> str:
        """
        Get color for the predicted vessel (highlighted).

        Args:
            predicted_idx: Index of predicted vessel
            num_agents: Total number of agents

        Returns:
            Hex color string
        """
        return self.get_color(predicted_idx)


class BaseVisualizer(ABC):
    """
    Abstract base class for all trajectory visualizers.

    Provides common interface and utilities for creating trajectory visualizations.
    """

    def __init__(self, config=None):
        """
        Initialize visualizer.

        Args:
            config: Configuration dict with visualization settings
        """
        self.config = config
        self.color_palette = VesselColorPalette()

    @abstractmethod
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
        Create visualization for a scenario.

        Args:
            predictions: Model predictions [1, future_len, 2] in XY meters
            ground_truth: Ground truth [1, future_len, 2] in XY meters
            scenario_id: Scenario identifier
            output_dir: Output directory
            output_filename: Output HTML filename
            metrics: Optional evaluation metrics
            scene_context: Optional multi-agent scene data

        Returns:
            Path to generated visualization file
        """
        pass

    def _format_metrics_html(self, metrics: Optional[Dict[str, float]]) -> str:
        """
        Format metrics dictionary as HTML.

        Args:
            metrics: Evaluation metrics dict

        Returns:
            HTML string with formatted metrics
        """
        if not metrics:
            return ""

        html = "<div class='metrics'><h4>📊 Evaluation Metrics</h4>"
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                html += f"<p><strong>{key}:</strong> {value:.4f}</p>"
            else:
                html += f"<p><strong>{key}:</strong> {value}</p>"
        html += "</div>"

        return html

    def _extract_window_start_idx(self, scenario_id: str) -> int:
        """
        Extract window start index from scenario_id (e.g., _t0, _t300).

        Args:
            scenario_id: Scenario identifier

        Returns:
            Window start index (0 if not found)
        """
        import re
        if '_t' in scenario_id:
            match = re.search(r'_t(\d+)$', scenario_id)
            if match:
                return int(match.group(1))
        return 0


class VisualizationFactory:
    """
    Factory for creating visualization instances based on type.

    Provides a centralized way to create different visualizer types.
    """

    _visualizer_types = {}

    @classmethod
    def register_visualizer(cls, vis_type: str, visualizer_class):
        """
        Register a new visualizer type.

        Args:
            vis_type: Type identifier (e.g., 'interactive', 'static')
            visualizer_class: Visualizer class (must inherit from BaseVisualizer)
        """
        if not issubclass(visualizer_class, BaseVisualizer):
            raise TypeError(f"{visualizer_class} must inherit from BaseVisualizer")

        cls._visualizer_types[vis_type] = visualizer_class
        logger.info(f"Registered visualizer type: {vis_type}")

    @classmethod
    def create_visualizer(cls, vis_type: str, config=None) -> BaseVisualizer:
        """
        Create a visualizer instance.

        Args:
            vis_type: Type identifier (e.g., 'interactive', 'static')
            config: Configuration dict

        Returns:
            Visualizer instance

        Raises:
            ValueError: If visualizer type not registered
        """
        if vis_type not in cls._visualizer_types:
            available_types = ', '.join(cls._visualizer_types.keys())
            raise ValueError(
                f"Unknown visualizer type: {vis_type}. "
                f"Available types: {available_types}"
            )

        visualizer_class = cls._visualizer_types[vis_type]
        return visualizer_class(config=config)

    @classmethod
    def list_visualizers(cls) -> List[str]:
        """
        List all registered visualizer types.

        Returns:
            List of visualizer type identifiers
        """
        return list(cls._visualizer_types.keys())
