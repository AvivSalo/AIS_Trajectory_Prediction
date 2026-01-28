#!/usr/bin/env python3
"""
Analyze Velocity Direction Accuracy for Wayformer AIS Predictions

This script analyzes:
1. Velocity magnitude error (speed loss)
2. Velocity direction error (heading loss)
3. Velocity vector comparison (predicted vs GT)
"""

import numpy as np
import pickle
import torch
import math
from pathlib import Path
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_velocity_from_positions(positions, dt=1.0):
    """
    Calculate velocities from position sequences using finite differences

    Args:
        positions: array of shape [N, 2] with (x, y) positions
        dt: time interval between positions (default 1.0 second)

    Returns:
        velocities: array of shape [N-1, 2] with (vx, vy) velocities
    """
    if len(positions) < 2:
        return np.array([])

    velocities = np.diff(positions, axis=0) / dt
    return velocities


def calculate_speed_error(pred_velocities, gt_velocities):
    """Calculate speed magnitude error"""
    pred_speeds = np.linalg.norm(pred_velocities, axis=-1)
    gt_speeds = np.linalg.norm(gt_velocities, axis=-1)

    speed_error = np.abs(pred_speeds - gt_speeds)
    speed_error_percent = (speed_error / (gt_speeds + 1e-6)) * 100

    return {
        'mae': np.mean(speed_error),
        'rmse': np.sqrt(np.mean(speed_error**2)),
        'mean_percent_error': np.mean(speed_error_percent),
        'pred_speeds': pred_speeds,
        'gt_speeds': gt_speeds
    }


def calculate_heading_error(pred_velocities, gt_velocities):
    """Calculate heading direction error in degrees"""
    # Calculate headings from velocity vectors
    pred_headings = np.arctan2(pred_velocities[:, 1], pred_velocities[:, 0])
    gt_headings = np.arctan2(gt_velocities[:, 1], gt_velocities[:, 0])

    # Calculate angular difference (wrap to [-pi, pi])
    heading_diff = pred_headings - gt_headings
    heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
    heading_diff_deg = np.degrees(heading_diff)

    return {
        'mae_deg': np.mean(np.abs(heading_diff_deg)),
        'rmse_deg': np.sqrt(np.mean(heading_diff_deg**2)),
        'pred_headings_deg': np.degrees(pred_headings),
        'gt_headings_deg': np.degrees(gt_headings),
        'errors_deg': heading_diff_deg
    }


def calculate_velocity_similarity(pred_velocities, gt_velocities):
    """Calculate cosine similarity between velocity vectors"""
    # Normalize vectors
    pred_norm = pred_velocities / (np.linalg.norm(pred_velocities, axis=-1, keepdims=True) + 1e-6)
    gt_norm = gt_velocities / (np.linalg.norm(gt_velocities, axis=-1, keepdims=True) + 1e-6)

    # Cosine similarity
    cosine_sim = np.sum(pred_norm * gt_norm, axis=-1)

    # Angular difference from cosine similarity
    angular_diff_deg = np.degrees(np.arccos(np.clip(cosine_sim, -1, 1)))

    return {
        'mean_cosine_similarity': np.mean(cosine_sim),
        'mean_angular_diff_deg': np.mean(angular_diff_deg),
        'cosine_similarities': cosine_sim
    }


def analyze_prediction_file(prediction_pkl_path, original_pkl_path):
    """
    Analyze velocity accuracy from prediction and original data files

    Args:
        prediction_pkl_path: Path to prediction results pickle
        original_pkl_path: Path to original scenario pickle with GT velocities
    """
    logger.info("="*80)
    logger.info(f"Analyzing: {Path(prediction_pkl_path).name}")
    logger.info("="*80)

    # Load prediction data
    with open(prediction_pkl_path, 'rb') as f:
        pred_data = pickle.load(f)

    # Load original data with GT velocities
    with open(original_pkl_path, 'rb') as f:
        orig_data = pickle.load(f)

    # Extract prediction info
    scenario_id = pred_data.get('scenario_id', 'unknown')
    predicted_trajectory = pred_data.get('predicted_trajectory', None)  # [modes, timesteps, 2]
    predicted_probability = pred_data.get('predicted_probability', None)  # [modes]

    if predicted_trajectory is None:
        logger.error("No predicted trajectory found in file")
        return None

    # Get best mode prediction
    if predicted_probability is not None:
        best_mode = np.argmax(predicted_probability)
        pred_positions = predicted_trajectory[best_mode]
    else:
        pred_positions = predicted_trajectory[0]

    logger.info(f"Predicted positions shape: {pred_positions.shape}")
    logger.info(f"Best mode: {best_mode if predicted_probability is not None else 0}")

    # Calculate predicted velocities from position deltas
    pred_velocities = calculate_velocity_from_positions(pred_positions, dt=1.0)
    logger.info(f"Predicted velocities shape: {pred_velocities.shape}")

    # Extract GT velocities from original data
    tracks = orig_data['tracks']
    track_ids = list(tracks.keys())

    # Find the predicted vessel (usually track index 0 = ego)
    track_index_to_predict = pred_data.get('track_index_to_predict', 0)
    predicted_track_id = track_ids[track_index_to_predict]

    gt_velocities_full = tracks[predicted_track_id]['state']['velocity']
    gt_positions_full = tracks[predicted_track_id]['state']['position']

    # Extract GT velocities for future timesteps
    window_start_idx = pred_data.get('window_start_idx', 0)
    past_len = pred_data.get('past_len', 21)
    future_len = pred_positions.shape[0]

    future_start_idx = window_start_idx + past_len
    future_end_idx = future_start_idx + future_len

    gt_velocities = gt_velocities_full[future_start_idx:future_end_idx]
    gt_positions = gt_positions_full[future_start_idx:future_end_idx]

    logger.info(f"GT velocities shape: {gt_velocities.shape}")
    logger.info(f"GT positions shape: {gt_positions.shape}")

    # Ensure matching lengths
    min_len = min(len(pred_velocities), len(gt_velocities))
    pred_velocities = pred_velocities[:min_len]
    gt_velocities = gt_velocities[:min_len]

    # Calculate metrics
    speed_metrics = calculate_speed_error(pred_velocities, gt_velocities)
    heading_metrics = calculate_heading_error(pred_velocities, gt_velocities)
    similarity_metrics = calculate_velocity_similarity(pred_velocities, gt_velocities)

    # Print results
    logger.info("\n" + "="*80)
    logger.info("VELOCITY ANALYSIS RESULTS")
    logger.info("="*80)

    logger.info("\n📏 SPEED (Magnitude) Errors:")
    logger.info(f"  MAE:  {speed_metrics['mae']:.4f} m/s  ({speed_metrics['mae']*1.94384:.2f} knots)")
    logger.info(f"  RMSE: {speed_metrics['rmse']:.4f} m/s  ({speed_metrics['rmse']*1.94384:.2f} knots)")
    logger.info(f"  Mean Percent Error: {speed_metrics['mean_percent_error']:.2f}%")
    logger.info(f"  Predicted Speed: {np.mean(speed_metrics['pred_speeds']):.4f} m/s ({np.mean(speed_metrics['pred_speeds'])*1.94384:.2f} knots)")
    logger.info(f"  GT Speed:        {np.mean(speed_metrics['gt_speeds']):.4f} m/s ({np.mean(speed_metrics['gt_speeds'])*1.94384:.2f} knots)")

    logger.info("\n🧭 HEADING (Direction) Errors:")
    logger.info(f"  MAE:  {heading_metrics['mae_deg']:.2f}°")
    logger.info(f"  RMSE: {heading_metrics['rmse_deg']:.2f}°")
    logger.info(f"  Mean Predicted Heading: {np.mean(heading_metrics['pred_headings_deg']):.2f}°")
    logger.info(f"  Mean GT Heading:        {np.mean(heading_metrics['gt_headings_deg']):.2f}°")

    logger.info("\n📐 VELOCITY VECTOR Similarity:")
    logger.info(f"  Mean Cosine Similarity: {similarity_metrics['mean_cosine_similarity']:.4f}")
    logger.info(f"  Mean Angular Difference: {similarity_metrics['mean_angular_diff_deg']:.2f}°")

    # Per-timestep breakdown
    logger.info("\n⏱️ PER-TIMESTEP BREAKDOWN:")
    for t in range(min_len):
        pred_speed_kts = np.linalg.norm(pred_velocities[t]) * 1.94384
        gt_speed_kts = np.linalg.norm(gt_velocities[t]) * 1.94384
        heading_error = heading_metrics['errors_deg'][t]

        logger.info(f"  t={t+1}: Pred={pred_speed_kts:5.2f} kts, GT={gt_speed_kts:5.2f} kts, "
                   f"Speed_Err={abs(pred_speed_kts-gt_speed_kts):5.2f} kts, "
                   f"Heading_Err={heading_error:6.2f}°")

    logger.info("="*80)

    return {
        'speed': speed_metrics,
        'heading': heading_metrics,
        'similarity': similarity_metrics,
        'pred_velocities': pred_velocities,
        'gt_velocities': gt_velocities,
        'pred_positions': pred_positions,
        'gt_positions': gt_positions
    }


def create_velocity_plots(results, output_path='velocity_analysis.png'):
    """Create visualization plots for velocity analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Velocity Direction Analysis', fontsize=16, fontweight='bold')

    pred_velocities = results['pred_velocities']
    gt_velocities = results['gt_velocities']
    speed_metrics = results['speed']
    heading_metrics = results['heading']

    timesteps = np.arange(len(pred_velocities)) + 1

    # Plot 1: Speed comparison
    ax = axes[0, 0]
    ax.plot(timesteps, speed_metrics['pred_speeds']*1.94384, 'o-', label='Predicted', color='red', linewidth=2)
    ax.plot(timesteps, speed_metrics['gt_speeds']*1.94384, 's-', label='Ground Truth', color='blue', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Speed (knots)')
    ax.set_title('Speed Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Speed error
    ax = axes[0, 1]
    speed_errors = np.abs(speed_metrics['pred_speeds'] - speed_metrics['gt_speeds']) * 1.94384
    ax.bar(timesteps, speed_errors, color='orange', alpha=0.7)
    ax.axhline(np.mean(speed_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(speed_errors):.2f} kts')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Speed Error (knots)')
    ax.set_title('Speed Magnitude Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Heading comparison
    ax = axes[0, 2]
    ax.plot(timesteps, heading_metrics['pred_headings_deg'], 'o-', label='Predicted', color='red', linewidth=2)
    ax.plot(timesteps, heading_metrics['gt_headings_deg'], 's-', label='Ground Truth', color='blue', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Heading (degrees)')
    ax.set_title('Heading Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Heading error
    ax = axes[1, 0]
    ax.bar(timesteps, np.abs(heading_metrics['errors_deg']), color='purple', alpha=0.7)
    ax.axhline(heading_metrics['mae_deg'], color='red', linestyle='--', linewidth=2,
               label=f'MAE: {heading_metrics["mae_deg"]:.2f}°')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Heading Error (degrees)')
    ax.set_title('Heading Direction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Velocity vectors (quiver plot)
    ax = axes[1, 1]
    # Plot GT velocities
    ax.quiver(np.zeros(len(gt_velocities)), timesteps, gt_velocities[:, 0], gt_velocities[:, 1],
             color='blue', alpha=0.7, scale=30, width=0.005, label='GT')
    # Plot predicted velocities
    ax.quiver(np.ones(len(pred_velocities)), timesteps, pred_velocities[:, 0], pred_velocities[:, 1],
             color='red', alpha=0.7, scale=30, width=0.005, label='Predicted')
    ax.set_xlabel('Velocity Type')
    ax.set_ylabel('Timestep')
    ax.set_title('Velocity Vectors (vx, vy)')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['GT', 'Predicted'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Velocity component comparison
    ax = axes[1, 2]
    width = 0.35
    x = timesteps
    ax.bar(x - width/2, pred_velocities[:, 0], width, label='Pred vx', color='red', alpha=0.7)
    ax.bar(x + width/2, gt_velocities[:, 0], width, label='GT vx', color='blue', alpha=0.7)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('vx (m/s)')
    ax.set_title('Velocity X Component')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"📊 Saved velocity analysis plot to: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Example usage for the x-pressfeeders-aquarius scenario
    base_path = Path("/")

    # Path to original data with GT velocities
    original_pkl = base_path / "data/debug_overfit_xpressfeeders/val/ais_x-pressfeeders-aquarius_20250315_060000/ais_x-pressfeeders-aquarius_20250315_060000.pkl"

    # Path to predictions (you'll need to run evaluation first to generate this)
    # For now, we'll demonstrate with the analysis logic

    logger.info("🚀 Starting Velocity Direction Analysis")
    logger.info(f"Original data: {original_pkl}")

    if not original_pkl.exists():
        logger.error(f"Original pickle file not found: {original_pkl}")
        exit(1)

    # Load the original data to demonstrate velocity extraction
    with open(original_pkl, 'rb') as f:
        data = pickle.load(f)

    logger.info("\n📦 Original Data Structure:")
    logger.info(f"  Scenario ID: {data.get('scenario_id', 'N/A')}")
    logger.info(f"  Tracks: {list(data['tracks'].keys())[:5]}... ({len(data['tracks'])} total)")

    # Analyze first vessel
    first_track_id = list(data['tracks'].keys())[0]
    track_data = data['tracks'][first_track_id]

    velocities = track_data['state']['velocity']
    positions = track_data['state']['position']

    logger.info(f"\n🎯 Vessel '{first_track_id}':")
    logger.info(f"  Positions shape: {positions.shape}")
    logger.info(f"  Velocities shape: {velocities.shape}")

    # Calculate speeds from velocities
    speeds_ms = np.linalg.norm(velocities, axis=1)
    speeds_kts = speeds_ms * 1.94384

    logger.info(f"\n📊 Speed Statistics:")
    logger.info(f"  Mean: {np.mean(speeds_kts):.2f} knots")
    logger.info(f"  Min:  {np.min(speeds_kts):.2f} knots")
    logger.info(f"  Max:  {np.max(speeds_kts):.2f} knots")
    logger.info(f"  Std:  {np.std(speeds_kts):.2f} knots")

    # Show sample velocities
    logger.info(f"\n🔍 Sample Velocities (first 5 timesteps):")
    for i in range(min(5, len(velocities))):
        vx, vy = velocities[i]
        speed_kts = np.linalg.norm([vx, vy]) * 1.94384
        heading_deg = np.degrees(np.arctan2(vy, vx))
        logger.info(f"  t={i}: vx={vx:7.3f} m/s, vy={vy:7.3f} m/s, speed={speed_kts:6.2f} kts, heading={heading_deg:6.1f}°")

    logger.info("\n" + "="*80)
    logger.info("✅ Analysis Complete")
    logger.info("\nTo analyze predictions:")
    logger.info("  1. Run: cd /home/aviv/Projects/UniTraj/unitraj && python3 evaluation.py")
    logger.info("  2. Run: python3 analyze_velocity_direction.py [prediction_pkl_path] [original_pkl_path]")
    logger.info("="*80)
