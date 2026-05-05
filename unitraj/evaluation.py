import pytorch_lightning as pl
import torch
import numpy as np
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

torch.set_float32_matmul_precision('medium')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed
import hydra
from omegaconf import OmegaConf
from visualizations.viz_leaflet import LeafletVisualizer
from benchmark.report import create_report

logger = logging.getLogger(__name__)


class EvaluationCallback(pl.Callback):
    """Callback to collect per-window predictions and compute dense evaluation metrics."""

    def __init__(self, config=None):
        super().__init__()
        self.vessel_pred_windows = {}    # {vessel_id: [(time_offset, pred_norm[T,2])]}
        self.all_sample_metrics = []     # [{vessel_id, time_offset, min_ade_m, min_fde_m, miss}]
        self.vessel_first_gt = {}        # {vessel_id: gt_norm [T,2]} — GT for first window only
        self.vessel_first_scene_ctx = {} # {vessel_id: scene_context dict} — first window only
        exp_name = str(getattr(config, 'exp_name', '')) if config is not None else ''
        self.output_dir = f"evaluation_visualizations_{exp_name}" if exp_name else "evaluation_visualizations"
        self.config = config
        self._eval_start_time = None
        self._batch_count = 0
        os.makedirs(self.output_dir, exist_ok=True)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._eval_start_time = time.time()
        self._batch_count = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect lean GPU metrics and per-window predictions."""
        self._batch_count += 1
        try:
            input_dict = batch['input_dict']
            prediction, loss = pl_module.forward(batch)

            pred_trajs = prediction['predicted_trajectory']  # [B, C, T, 2]
            position_scale = getattr(pl_module.config, 'position_scale', 100.0)
            past_len = getattr(pl_module.config, 'past_len', 300)

            # Require center_gt_trajs for metrics
            if 'center_gt_trajs' not in input_dict or input_dict['center_gt_trajs'] is None:
                return
            gt_trajs = input_dict['center_gt_trajs'][:, :, :2]  # [B, T, 2]

            # Per-sample min-ADE / min-FDE on GPU
            pred_xy = pred_trajs[:, :, :, :2]              # [B, C, T, 2]
            gt_xy = gt_trajs.unsqueeze(1)                   # [B, 1, T, 2]
            ade_per_mode = torch.norm(pred_xy - gt_xy, dim=-1).mean(dim=-1)        # [B, C]
            fde_per_mode = torch.norm(pred_xy[:, :, -1, :] - gt_xy[:, :, -1, :], dim=-1)  # [B, C]
            min_ade_m = (ade_per_mode.min(1).values * position_scale).cpu().numpy()  # [B]
            min_fde_m = (fde_per_mode.min(1).values * position_scale).cpu().numpy()  # [B]
            miss = (min_fde_m > 2.0)  # [B] bool

            # Parse scenario IDs
            batch_size = pred_trajs.shape[0]
            scenario_ids = input_dict.get('scenario_id', [f"batch_{batch_idx}_{i}" for i in range(batch_size)])
            if isinstance(scenario_ids, torch.Tensor):
                scenario_ids = scenario_ids.cpu().numpy().tolist()
            elif isinstance(scenario_ids, str):
                scenario_ids = [scenario_ids]
            elif isinstance(scenario_ids, np.ndarray):
                scenario_ids = scenario_ids.tolist()
            elif not isinstance(scenario_ids, (list, tuple)):
                scenario_ids = [str(scenario_ids)]
            scenario_ids = [str(sid) for sid in scenario_ids]

            for i in range(batch_size):
                scenario_id = scenario_ids[i]

                # Parse vessel_id and time_offset from scenario_id (format: …_tXXXXX)
                if '_t' in scenario_id:
                    vessel_id = scenario_id.rsplit('_t', 1)[0]
                    try:
                        time_offset = int(scenario_id.rsplit('_t', 1)[1])
                    except ValueError:
                        time_offset = 0
                else:
                    vessel_id = scenario_id
                    time_offset = 0

                # Store normalized ego-relative prediction (mode 0, best mode)
                pred_norm = pred_trajs[i, 0, :, :2].detach().cpu().numpy()  # [T, 2]
                gt_norm = gt_trajs[i].detach().cpu().numpy()                 # [T, 2]

                self.vessel_pred_windows.setdefault(vessel_id, []).append((time_offset, pred_norm))

                self.all_sample_metrics.append({
                    'vessel_id': vessel_id,
                    'time_offset': time_offset,
                    'min_ade_m': float(min_ade_m[i]),
                    'min_fde_m': float(min_fde_m[i]),
                    'miss': bool(miss[i]),
                    'miss_5m':  bool(min_fde_m[i] > 5.0),
                    'miss_10m': bool(min_fde_m[i] > 10.0),
                    'miss_20m': bool(min_fde_m[i] > 20.0),
                })

                # Store first window's data for visualization (GT + scene context)
                if vessel_id not in self.vessel_first_gt:
                    self.vessel_first_gt[vessel_id] = gt_norm

                    track_idx = (input_dict['track_index_to_predict'][i].item()
                                 if isinstance(input_dict['track_index_to_predict'], torch.Tensor)
                                 else int(input_dict['track_index_to_predict'][i]))

                    ref_pos_raw = input_dict.get('reference_position')
                    ref_pos = (ref_pos_raw[i].detach().cpu().numpy()
                               if ref_pos_raw is not None else np.zeros(2))

                    all_agents_gt = input_dict.get('all_agents_gt_trajs')
                    all_agents_gt_masks = input_dict.get('all_agents_gt_masks')

                    self.vessel_first_scene_ctx[vessel_id] = {
                        'obj_trajs': input_dict['obj_trajs'][i].detach().cpu().numpy(),
                        'obj_mask': input_dict['obj_trajs_mask'][i].detach().cpu().numpy(),
                        'track_idx': track_idx,
                        'past_traj': input_dict['obj_trajs'][i, track_idx, :past_len, 0:2].detach().cpu().numpy(),
                        'reference_position': ref_pos,
                        'future_gt': (all_agents_gt[i].detach().cpu().numpy()
                                      if all_agents_gt is not None else None),
                        'future_gt_mask': (all_agents_gt_masks[i].detach().cpu().numpy()
                                           if all_agents_gt_masks is not None else None),
                    }

        except Exception as e:
            logger.warning(f"Failed to collect batch data: {str(e)}")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute aggregate + per-timestamp metrics, then create visualizations and report."""
        if not self.all_sample_metrics:
            logger.warning("No metrics collected for visualization")
            return

        eval_duration = time.time() - self._eval_start_time if self._eval_start_time else 0

        try:
            all_ade  = np.array([s['min_ade_m'] for s in self.all_sample_metrics])
            all_fde  = np.array([s['min_fde_m'] for s in self.all_sample_metrics])
            all_miss = np.array([s['miss'] for s in self.all_sample_metrics], dtype=float)
            n_samples = len(self.all_sample_metrics)
            n_vessels = len(self.vessel_pred_windows)
            n_batches = self._batch_count

            # --- Per-timestamp bins ---
            bins       = [0, 60, 120, 180, 240, 300, 400, 500, 600, 9999]
            bin_labels = ['0–60s', '60–120s', '120–180s', '180–240s',
                          '240–300s', '300–400s', '400–500s', '500–600s', '600+s']
            per_bin = []
            for j, (lo, hi) in enumerate(zip(bins, bins[1:])):
                mask = np.array([lo <= s['time_offset'] < hi for s in self.all_sample_metrics])
                if mask.sum() > 0:
                    per_bin.append({
                        'label':   bin_labels[j],
                        'count':   int(mask.sum()),
                        'ade':     float(all_ade[mask].mean()),
                        'fde':     float(all_fde[mask].mean()),
                        'miss':    float(all_miss[mask].mean()) * 100,
                        'miss_5m':  float((all_fde[mask] > 5.0).mean()) * 100,
                        'miss_10m': float((all_fde[mask] > 10.0).mean()) * 100,
                        'miss_20m': float((all_fde[mask] > 20.0).mean()) * 100,
                    })

            # --- Per-vessel metrics ---
            vessel_metrics_map = {}
            for m in self.all_sample_metrics:
                vid = m['vessel_id']
                vessel_metrics_map.setdefault(vid, {'ade': [], 'fde': [], 'miss': [], 'miss_5m': [], 'miss_10m': [], 'miss_20m': []})
                vessel_metrics_map[vid]['ade'].append(m['min_ade_m'])
                vessel_metrics_map[vid]['fde'].append(m['min_fde_m'])
                vessel_metrics_map[vid]['miss'].append(float(m['miss']))
                vessel_metrics_map[vid]['miss_5m'].append(float(m['miss_5m']))
                vessel_metrics_map[vid]['miss_10m'].append(float(m['miss_10m']))
                vessel_metrics_map[vid]['miss_20m'].append(float(m['miss_20m']))

            # --- Print summary ---
            print(f"\n📊 Dense Evaluation — Val Scenes (stride=1)")
            print(f"   Samples: {n_samples:>8,}  |  Vessels: {n_vessels}  |  Batches: {n_batches:,}")
            print(f"   Inference: {eval_duration:.1f}s total  |  {eval_duration/n_samples*1000:.1f}ms/sample  |  {n_samples/eval_duration:.0f} samples/s")
            print(f"\n   ── Overall ──────────────────────────────────────────────────")
            print(f"   minADE6: {all_ade.mean():.2f} m   |  minFDE6: {all_fde.mean():.2f} m  |  Miss@2m: {all_miss.mean()*100:.1f}%")
            print(f"\n   ── By Time Offset (prediction start) ────────────────────────")
            print(f"   {'Phase':<14}  {'Samples':>8}  {'minADE6':>8}  {'minFDE6':>8}  {'Miss%':>7}")
            print(f"   {'-'*58}")
            for s in per_bin:
                print(f"   t={s['label']:<12}  {s['count']:>8,}  {s['ade']:>8.2f}  {s['fde']:>8.2f}  {s['miss']:>7.1f}%")

            # --- Visualizations ---
            # VIZ_STRIDE=1: show every eval window (stride=60 already limits count).
            # Higher values skip windows → big gaps between prediction updates in viz.
            VIZ_STRIDE = 1
            leaflet_viz = LeafletVisualizer(config=self.config)

            created_files = []
            per_vessel_list = []

            for vessel_id, windows in self.vessel_pred_windows.items():
                sorted_windows = sorted(windows, key=lambda x: x[0])
                display_windows = sorted_windows[::VIZ_STRIDE]
                if not display_windows:
                    continue

                first_time_offset, first_pred_norm = display_windows[0]
                first_gt_norm = self.vessel_first_gt.get(vessel_id)
                if first_gt_norm is None:
                    continue

                # Per-vessel metrics for individual viz sidebar
                vm = vessel_metrics_map.get(vessel_id, {})
                scene_metrics = {}
                if vm:
                    scene_metrics = {
                        'val/minADE6':   float(np.mean(vm['ade'])),
                        'val/minFDE6':   float(np.mean(vm['fde'])),
                        'val/miss_rate': float(np.mean(vm['miss'])),
                        'val/miss_5m':   float(np.mean(vm['miss_5m'])),
                        'val/miss_10m':  float(np.mean(vm['miss_10m'])),
                        'val/miss_20m':  float(np.mean(vm['miss_20m'])),
                    }

                scenario_id_with_t = f"{vessel_id}_t{first_time_offset}"
                scene_ctx = self.vessel_first_scene_ctx.get(vessel_id)

                clean_id = "".join(c for c in vessel_id if c.isalnum() or c in ('-', '_')).rstrip()
                if not clean_id:
                    clean_id = f"vessel_{len(created_files)}"
                if len(clean_id) > 50:
                    clean_id = f"{clean_id[:40]}_{hash(vessel_id) % 10000:04d}"

                output_filename = f"ais_vessel_{clean_id}.html"
                output_path = leaflet_viz.create_visualization(
                    predictions=first_pred_norm.reshape(1, -1, 2),
                    ground_truth=first_gt_norm.reshape(1, -1, 2),
                    scenario_id=scenario_id_with_t,
                    output_dir=self.output_dir,
                    output_filename=output_filename,
                    metrics=scene_metrics,
                    scene_context=scene_ctx,
                    prediction_windows=display_windows,
                    viz_stride=VIZ_STRIDE,
                )
                created_files.append(output_path)
                logger.info(f"✅ Created viz for vessel {vessel_id}")

                per_vessel_list.append({
                    'vessel_id':   vessel_id,
                    'samples':     len(vm.get('ade', [])),
                    'ade':         float(np.mean(vm['ade'])) if vm else 0.0,
                    'fde':         float(np.mean(vm['fde'])) if vm else 0.0,
                    'miss':        float(np.mean(vm['miss'])) * 100 if vm else 0.0,
                    'p90_fde':     float(np.percentile(vm['fde'], 90)) if vm else 0.0,
                    'html_file':   output_filename,
                })

            # --- FDE CDF (150 sample points) ---
            sorted_fde = np.sort(all_fde)
            p99_fde = float(np.percentile(all_fde, 99))
            cdf_x_arr = np.linspace(0, p99_fde, 150)
            cdf_x = [round(float(v), 3) for v in cdf_x_arr]
            cdf_y = [round(float((sorted_fde <= v).mean() * 100), 2) for v in cdf_x_arr]

            # --- ADE histogram ---
            hist_edges = [0, 0.5, 1, 2, 3, 5, 8, 12, 20, 9999]
            hist_labels_list = ['0–0.5', '0.5–1', '1–2', '2–3', '3–5', '5–8', '8–12', '12–20', '20+']
            hist_counts_list = [
                int(((all_ade >= hist_edges[k]) & (all_ade < hist_edges[k + 1])).sum())
                for k in range(len(hist_labels_list))
            ]

            # --- Generate aggregate report ---
            cfg = self.config
            report_data = {
                'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'exp_name':         str(getattr(cfg, 'exp_name', 'unknown')),
                'ckpt_path':        str(getattr(cfg, 'ckpt_path', 'unknown')),
                'stride':           int(getattr(cfg, 'stride', 1)),
                'past_len':         int(getattr(cfg, 'past_len', 300)),
                'future_len':       int(getattr(cfg, 'future_len', 60)),
                'eval_batch_size':  int(getattr(cfg, 'eval_batch_size', 96)),
                'load_num_workers': int(getattr(cfg, 'load_num_workers', 8)),
                'val_data_path':    str(getattr(cfg, 'val_data_path', '')),
                'total_samples':    n_samples,
                'n_vessels':        n_vessels,
                'n_batches':        n_batches,
                'eval_duration_s':  round(eval_duration, 1),
                'ms_per_sample':    round(eval_duration / n_samples * 1000, 2) if n_samples else 0,
                'ms_per_batch':     round(eval_duration / n_batches * 1000, 1) if n_batches else 0,
                'samples_per_sec':  round(n_samples / eval_duration, 1) if eval_duration else 0,
                # Overall metrics
                'mean_ade':   round(float(all_ade.mean()), 3),
                'std_ade':    round(float(all_ade.std()), 3),
                'p50_ade':    round(float(np.percentile(all_ade, 50)), 3),
                'p90_ade':    round(float(np.percentile(all_ade, 90)), 3),
                'p95_ade':    round(float(np.percentile(all_ade, 95)), 3),
                'mean_fde':   round(float(all_fde.mean()), 3),
                'std_fde':    round(float(all_fde.std()), 3),
                'p50_fde':    round(float(np.percentile(all_fde, 50)), 3),
                'p90_fde':    round(float(np.percentile(all_fde, 90)), 3),
                'p95_fde':    round(float(np.percentile(all_fde, 95)), 3),
                'miss_2m':    round(float((all_fde > 2.0).mean()) * 100, 1),
                'miss_5m':    round(float((all_fde > 5.0).mean()) * 100, 1),
                'miss_10m':   round(float((all_fde > 10.0).mean()) * 100, 1),
                'miss_20m':   round(float((all_fde > 20.0).mean()) * 100, 1),
                'per_bin':      per_bin,
                'per_vessel':   per_vessel_list,
                # Chart data
                'cdf_x':        cdf_x,
                'cdf_y':        cdf_y,
                'hist_labels':  hist_labels_list,
                'hist_counts':  hist_counts_list,
            }

            report_path = create_report(
                report_data=report_data,
                output_dir=self.output_dir,
            )

            print(f"\n   ── HTML Visualizations ──────────────────────────────────────")
            print(f"   {len(created_files)} vessel files in {os.path.abspath(self.output_dir)}/")
            print(f"   📋 Report: {os.path.abspath(report_path)}")
            print(f"🌐 Open report.html in browser to view full evaluation summary")

        except Exception as e:
            logger.error(f"Failed to create visualizations: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self.vessel_pred_windows = {}
            self.all_sample_metrics = []
            self.vessel_first_gt = {}
            self.vessel_first_scene_ctx = {}


@hydra.main(version_base=None, config_path="configs", config_name="config")
def evaluation(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg['eval'] = True

    model = build_model(cfg)
    val_set = build_dataset(cfg, val=True)
    eval_batch_size = cfg.method['eval_batch_size']

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers,
        shuffle=False, drop_last=False, collate_fn=val_set.collate_fn)

    viz_callback = EvaluationCallback(config=cfg)

    trainer = pl.Trainer(
        inference_mode=True,
        logger=None if cfg.debug else WandbLogger(project="unitraj", name=cfg.exp_name),
        devices=1,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        callbacks=[viz_callback],
    )

    results = trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)

    if results:
        print(f"\n📊 Evaluation Results:")
        for result_dict in results:
            for key, value in result_dict.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")

    return results


if __name__ == '__main__':
    evaluation()
