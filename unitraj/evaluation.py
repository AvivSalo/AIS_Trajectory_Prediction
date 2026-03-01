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
from visualizations.viz_leaflet import LeafletVisualizer

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

    def on_validation_epoch_end(self, trainer, pl_module):
        """Process collected data and create visualization"""
        if not self.predictions or not self.ground_truths:
            logger.warning("No prediction data collected for visualization")
            return
        
        try:
            leaflet_viz = LeafletVisualizer(config=self.config)

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

                output_path = leaflet_viz.create_visualization(
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
            timeline_files = leaflet_viz.create_timeline_visualizations(output_dir)
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
