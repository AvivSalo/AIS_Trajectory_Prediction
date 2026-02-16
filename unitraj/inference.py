"""
Production inference service for real-time trajectory prediction.

This module provides a production-ready inference API for the Wayformer-AIS model:
- Load trained model checkpoint
- Process incoming AIS data
- Generate trajectory predictions
- Return results in standardized format
- Optional visualization generation

Usage:
    predictor = TrajectoryPredictor(checkpoint_path='best_model.ckpt')
    prediction = predictor.predict(ais_data)

    # With visualization
    predictor.predict_and_visualize(ais_data, output_dir='results/')
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from omegaconf import DictConfig
import pytorch_lightning as pl

from .models import build_model
from .datasets import build_dataset
from .evaluation_utils import CoordinateConverter, MetricsCalculator
from .visualizations import VisualizationFactory

logger = logging.getLogger(__name__)


class TrajectoryPredictor:
    """
    Production inference service for trajectory prediction.

    Features:
    - Load trained model from checkpoint
    - Process raw AIS data into model format
    - Generate multi-mode predictions
    - Calculate confidence scores
    - Optional visualization generation
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[DictConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize trajectory predictor.

        Args:
            checkpoint_path: Path to trained model checkpoint (.ckpt)
            config: Optional configuration dict
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config = config
        self.device = device

        # Load model
        self.model = self._load_model()
        self.model.eval()
        self.model.to(device)

        # Initialize utilities
        self.coord_converter = CoordinateConverter()
        self.metrics_calculator = MetricsCalculator()

        logger.info(f" TrajectoryPredictor initialized with checkpoint: {checkpoint_path}")
        logger.info(f"   Device: {device}")

    def _load_model(self) -> pl.LightningModule:
        """
        Load trained model from checkpoint.

        Returns:
            Loaded PyTorch Lightning model

        Raises:
            FileNotFoundError: If checkpoint file not found
            RuntimeError: If model loading fails
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            # Build model from config
            if self.config is None:
                # Try to extract config from checkpoint
                if 'hyper_parameters' in checkpoint:
                    self.config = checkpoint['hyper_parameters'].get('config')
                else:
                    raise RuntimeError("No config provided and cannot extract from checkpoint")

            model = build_model(self.config)

            # Load weights
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            logger.info(f" Model loaded: {self.config.method}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def predict(
        self,
        input_data: Union[Dict, torch.Tensor],
        return_confidence: bool = True,
        return_all_modes: bool = False
    ) -> Dict:
        """
        Generate trajectory prediction from input data.

        Args:
            input_data: Input batch dict or preprocessed tensor
            return_confidence: Whether to return confidence scores
            return_all_modes: Whether to return all modes or just best mode

        Returns:
            Dict with keys:
                - 'trajectory': Predicted trajectory [time_steps, 2] or [num_modes, time_steps, 2]
                - 'confidence': Confidence scores (if return_confidence=True)
                - 'metadata': Additional prediction metadata
        """
        with torch.no_grad():
            # Prepare input
            if isinstance(input_data, dict):
                batch = {'input_dict': input_data}
            else:
                batch = input_data

            # Move to device
            batch = self._to_device(batch, self.device)

            # Forward pass
            prediction, _ = self.model.forward(batch)

            # Extract trajectories [batch_size, num_modes, time_steps, 2]
            pred_trajs = prediction['predicted_trajectory'].cpu().numpy()

            # Process output
            result = {
                'trajectory': pred_trajs[0, 0] if not return_all_modes else pred_trajs[0],
                'metadata': {
                    'num_modes': pred_trajs.shape[1],
                    'time_steps': pred_trajs.shape[2],
                    'device': self.device
                }
            }

            # Add confidence scores
            if return_confidence and 'predicted_probability' in prediction:
                confidences = prediction['predicted_probability'].cpu().numpy()
                result['confidence'] = confidences[0]
                result['metadata']['best_mode_confidence'] = float(confidences[0, 0])

            return result

    def predict_and_visualize(
        self,
        input_data: Dict,
        output_dir: str,
        scenario_id: str,
        ground_truth: Optional[np.ndarray] = None,
        reference_coords: Optional[Tuple[float, float]] = None,
        visualizer_type: str = 'interactive'
    ) -> Tuple[Dict, str]:
        """
        Generate prediction and create visualization.

        Args:
            input_data: Input batch dict
            output_dir: Output directory for visualization
            scenario_id: Scenario identifier
            ground_truth: Optional ground truth trajectory for comparison
            reference_coords: Optional (lat, lon) reference coordinates
            visualizer_type: Type of visualizer ('interactive' or 'static')

        Returns:
            Tuple of (prediction_dict, visualization_path)
        """
        # Generate prediction
        prediction_result = self.predict(input_data, return_confidence=True)

        # Create visualization
        visualizer = VisualizationFactory.create_visualizer(
            visualizer_type,
            config=self.config
        )

        # Prepare data for visualization
        predictions = np.expand_dims(prediction_result['trajectory'], axis=0)  # [1, T, 2]
        gt = ground_truth if ground_truth is not None else predictions  # Fallback

        # Generate visualization
        viz_path = visualizer.create_visualization(
            predictions=predictions,
            ground_truth=gt,
            scenario_id=scenario_id,
            output_dir=output_dir,
            output_filename=f"{scenario_id}_prediction.html",
            metrics=self._calculate_metrics(predictions[0], gt[0]) if ground_truth is not None else None,
            scene_context=input_data.get('scene_context')
        )

        logger.info(f" Prediction and visualization complete: {viz_path}")

        return prediction_result, viz_path

    def batch_predict(
        self,
        input_batch: List[Dict],
        progress_callback: Optional[callable] = None
    ) -> List[Dict]:
        """
        Generate predictions for multiple inputs.

        Args:
            input_batch: List of input dicts
            progress_callback: Optional callback function(current, total)

        Returns:
            List of prediction dicts
        """
        results = []

        for i, input_data in enumerate(input_batch):
            result = self.predict(input_data)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(input_batch))

        logger.info(f" Batch prediction complete: {len(results)} predictions")
        return results

    def _calculate_metrics(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics for a prediction.

        Args:
            prediction: Predicted trajectory [time_steps, 2]
            ground_truth: Ground truth trajectory [time_steps, 2]

        Returns:
            Dict with ADE, FDE, miss_rate metrics
        """
        return self.metrics_calculator.calculate_scene_metrics(
            predictions=prediction,
            ground_truth=ground_truth,
            miss_threshold=2.0
        )

    def _to_device(self, batch: Dict, device: str) -> Dict:
        """
        Recursively move batch tensors to device.

        Args:
            batch: Batch dict
            device: Target device

        Returns:
            Batch with tensors moved to device
        """
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(device)
            elif isinstance(value, dict):
                result[key] = self._to_device(value, device)
            else:
                result[key] = value
        return result

    def get_model_info(self) -> Dict:
        """
        Get model information and configuration.

        Returns:
            Dict with model metadata
        """
        return {
            'checkpoint': str(self.checkpoint_path),
            'method': self.config.method if self.config else 'unknown',
            'device': self.device,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


# Convenience function for quick inference
def predict_from_checkpoint(
    checkpoint_path: str,
    input_data: Dict,
    config: Optional[DictConfig] = None,
    device: str = 'auto'
) -> Dict:
    """
    Quick prediction from checkpoint without creating predictor object.

    Args:
        checkpoint_path: Path to model checkpoint
        input_data: Input data dict
        config: Optional configuration
        device: Device ('auto', 'cuda', or 'cpu')

    Returns:
        Prediction dict
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    predictor = TrajectoryPredictor(checkpoint_path, config=config, device=device)
    return predictor.predict(input_data)


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trajectory prediction inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to input data file")
    parser.add_argument("--output", type=str, default="predictions/", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")

    args = parser.parse_args()

    # Initialize predictor
    predictor = TrajectoryPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # Load input data
    # TODO: Implement data loading from args.input

    print(" Inference service initialized")
    print(f"   Model: {predictor.get_model_info()}")