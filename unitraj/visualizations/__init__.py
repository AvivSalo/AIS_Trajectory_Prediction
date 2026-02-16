"""
Visualization module for trajectory prediction.

This module provides visualization tools for maritime trajectory prediction:
- Interactive timeline visualizations with Leaflet maps
- Static snapshot visualizations
- Factory pattern for easy visualizer creation
"""

from .viz_core import BaseVisualizer, VesselColorPalette, VisualizationFactory
from .viz_interactive import InteractiveTimelineVisualizer

# Auto-register visualizer types
VisualizationFactory.register_visualizer('interactive', InteractiveTimelineVisualizer)

__all__ = [
    'BaseVisualizer',
    'VesselColorPalette',
    'VisualizationFactory',
    'InteractiveTimelineVisualizer',
]
