"""
Visualization module for trajectory prediction.

This module provides visualization tools for maritime trajectory prediction:
- Interactive timeline visualizations with Leaflet maps
- Static snapshot visualizations
- Factory pattern for easy visualizer creation
"""

from .viz_core import BaseVisualizer, VesselColorPalette, VisualizationFactory
from .viz_leaflet import LeafletVisualizer

# viz_interactive uses relative imports that require running as part of the unitraj package.
# Wrap in try/except so that importing visualizations.viz_leaflet still works when
# evaluation.py is run directly (python evaluation.py from unitraj/).
try:
    from .viz_interactive import InteractiveTimelineVisualizer
    VisualizationFactory.register_visualizer('interactive', InteractiveTimelineVisualizer)
    _interactive_available = True
except ImportError:
    InteractiveTimelineVisualizer = None
    _interactive_available = False

__all__ = [
    'BaseVisualizer',
    'VesselColorPalette',
    'VisualizationFactory',
    'LeafletVisualizer',
]
if _interactive_available:
    __all__.append('InteractiveTimelineVisualizer')
