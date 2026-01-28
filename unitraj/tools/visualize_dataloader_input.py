#!/usr/bin/env python
"""
Simple visualization script to verify dataloader output.
Plots vessel trajectories from the exact same input the Wayformer model receives.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unitraj.datasets.ais_dataset import AISDataset


def visualize_scene(data_dict, scene_idx=0, save_path=None):
    """
    Visualize one scene showing all vessels at each timestamp.

    Args:
        data_dict: Output from dataloader collate_fn (input_dict)
        scene_idx: Which scene in the batch to visualize (0 to batch_size-1)
        save_path: Optional path to save animation
    """
    # Extract data for this scene
    obj_trajs = data_dict['obj_trajs'][scene_idx].cpu().numpy()  # [max_agents, past_len, feature_dim]
    track_index_to_predict = data_dict['track_index_to_predict'][scene_idx].item()

    # For multi-agent GT, use all_agents_gt_trajs (only contains agents with valid GT)
    all_agents_gt_trajs = data_dict['all_agents_gt_trajs'][scene_idx].cpu().numpy()  # [num_valid_agents, future_len, 2]
    all_agents_gt_masks = data_dict['all_agents_gt_masks'][scene_idx].cpu().numpy()  # [num_valid_agents, future_len]
    agents_to_predict = data_dict['agents_to_predict'][scene_idx].cpu().numpy()  # [num_valid_agents] - indices into obj_trajs

    max_agents, past_len, feature_dim = obj_trajs.shape
    future_len = all_agents_gt_trajs.shape[1]

    # Build complete trajectory for each agent that has GT
    all_positions = []
    valid_agent_indices = []

    for gt_idx, agent_idx in enumerate(agents_to_predict):
        if agent_idx < 0:  # Skip padding agents
            continue

        agent_idx = int(agent_idx)

        # Get past trajectory for this agent
        past_pos = obj_trajs[agent_idx, :, 0:2]  # [past_len, 2]

        # Get future GT for this agent
        future_pos = all_agents_gt_trajs[gt_idx, :, :]  # [future_len, 2]

        # Combine past and future
        agent_full_traj = np.concatenate([past_pos, future_pos], axis=0)  # [past_len+future_len, 2]

        all_positions.append(agent_full_traj)
        valid_agent_indices.append(agent_idx)

    if len(all_positions) == 0:
        print(f"Scene {scene_idx}: No valid agents with GT found")
        return

    all_positions = np.stack(all_positions)  # [num_valid_agents, past_len+future_len, 2]
    num_valid_agents = len(valid_agent_indices)
    total_timesteps = past_len + future_len

    print(f"\nScene {scene_idx}:")
    print(f"  Total agents in scene: {num_valid_agents} (ego at index {track_index_to_predict})")
    print(f"  Past timesteps: {past_len}")
    print(f"  Future timesteps: {future_len}")
    print(f"  Total timesteps: {total_timesteps}")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Calculate plot bounds
    all_flat_positions = all_positions.reshape(-1, 2)
    x_min, x_max = all_flat_positions[:, 0].min(), all_flat_positions[:, 0].max()
    y_min, y_max = all_flat_positions[:, 1].min(), all_flat_positions[:, 1].max()

    # Add padding
    x_padding = (x_max - x_min) * 0.1 if x_max > x_min else 100
    y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 100
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f'Scene {scene_idx} - Vessel Trajectories (Ego-Relative Coordinates)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Colors: ego in red, others in blue
    colors = ['red' if agent_idx == track_index_to_predict else 'blue' for agent_idx in valid_agent_indices]

    # Plot complete trajectories as faint lines
    for traj_idx, (agent_idx, color) in enumerate(zip(valid_agent_indices, colors)):
        trajectory = all_positions[traj_idx]
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                color=color, alpha=0.2, linewidth=1, linestyle='--')

    # Create scatter plots for current positions (will be updated in animation)
    scatter_plots = []
    for agent_idx, color in zip(valid_agent_indices, colors):
        scatter = ax.scatter([], [], c=color, s=100, alpha=0.8,
                           label='Ego' if agent_idx == track_index_to_predict else None,
                           edgecolors='black', linewidths=1.5)
        scatter_plots.append(scatter)

    # Add legend (only once for ego)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:1], labels[:1], loc='upper right', fontsize=10)

    # Vertical line to separate past/future
    vline = ax.axvline(x=0, color='green', linestyle=':', linewidth=2, alpha=0.5, label='Present')

    # Text for timestep counter
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def init():
        """Initialize animation"""
        for scatter in scatter_plots:
            scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return scatter_plots + [time_text]

    def update(frame):
        """Update animation for each frame"""
        # Update positions for all agents at this timestep
        for traj_idx, scatter in enumerate(scatter_plots):
            pos = all_positions[traj_idx, frame]
            scatter.set_offsets([pos])

        # Update timestep text
        phase = "PAST" if frame < past_len else "FUTURE"
        relative_time = frame if frame < past_len else (frame - past_len)
        time_text.set_text(f'Timestep: {frame}/{total_timesteps-1} ({phase} t={relative_time})')

        return scatter_plots + [time_text]

    # Create animation
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                  frames=total_timesteps, interval=200,
                                  blit=True, repeat=True)

    if save_path:
        print(f"  Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=5)
        print(f"  Saved!")

    plt.tight_layout()
    return fig, anim


def main():
    """Visualize 5 sample scenes from the dataloader"""

    print("="*80)
    print("DATALOADER INPUT VISUALIZATION")
    print("="*80)
    print("\nLoading dataset...")

    # Create config matching the dataset requirements
    config = {
        'train_data_path': ['data/processed_ais_data/train'],
        'val_data_path': ['data/processed_ais_data/val'],
        'past_len': 20,
        'future_len': 5,
        'max_num_agents': 10,
        'stride': 10,
        'trajectory_sample_interval': 1,
        'num_agent_feature': 39,
        'normalize_data': False,
        'map_range': 22000.0
    }

    # Load dataset with config
    dataset = AISDataset(config=config, is_validation=False)

    print(f"Dataset loaded: {len(dataset)} scenarios")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=5,  # Get 5 scenes at once
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )

    # Get one batch (5 scenes)
    print("\nGetting batch of 5 scenes...")
    batch_data = next(iter(dataloader))

    # Extract input_dict from batch structure
    data_dict = batch_data['input_dict']

    print(f"\nBatch shape:")
    print(f"  obj_trajs: {data_dict['obj_trajs'].shape}")  # [5, num_agents, past_len, feature_dim]
    print(f"  center_gt_trajs: {data_dict['center_gt_trajs'].shape}")  # [5, future_len, 2]
    print(f"  track_index_to_predict: {data_dict['track_index_to_predict'].shape}")  # [5]

    # Create output directory
    output_dir = 'unitraj/dataloader_visualizations'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nVisualizing 5 scenes and saving to {output_dir}/...")
    print("="*80)

    # Visualize each scene
    figs = []
    for scene_idx in range(5):
        save_path = f'{output_dir}/scene_{scene_idx}.gif'
        result = visualize_scene(data_dict, scene_idx=scene_idx, save_path=save_path)
        if result is not None:
            figs.append(result)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAnimations saved to: {output_dir}/")
    print("Files created:")
    for i in range(5):
        print(f"  - scene_{i}.gif")
    print("\nEach animation shows:")
    print("  - Red dots: Ego vessel (the one we're predicting)")
    print("  - Blue dots: Other vessels in the scene")
    print("  - Faint dashed lines: Complete trajectories")
    print("  - Timestep counter: Shows past vs future phases")
    print("  - Coordinates: Ego-relative (same as model input)")
    print("\nPress Ctrl+C to close windows...")

    # Show plots (will display all 5 animations)
    plt.show()


if __name__ == '__main__':
    main()
