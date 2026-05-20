import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
import wandb
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, find_latest_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import os


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)

    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg, val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices),  1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices), 1)

    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        monitor='val/brier_fde',  # Replace with your validation metric
        filename='{epoch}-{val/brier_fde:.2f}',
        save_top_k=1,
        mode='min',  # 'min' for loss/error, 'max' for accuracy
        dirpath=f'./unitraj_ckpt/{cfg.exp_name}'
    )

    call_backs.append(checkpoint_callback)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=train_set.collate_fn)

    # Optional val-batch cap: useful for models with expensive validation
    # (e.g. autoregressive TrAISformer at K=6 modes — full val epoch ~hours).
    # Set `limit_val_batches: N` in the parent config to cap each val pass.
    limit_val_batches = cfg.get('limit_val_batches', None)

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=None if cfg.debug else WandbLogger(project="unitraj", name=cfg.exp_name, id=cfg.exp_name,
                                                   config=OmegaConf.to_container(cfg, resolve=True),
                                                   settings=wandb.Settings(init_timeout=300, _service_wait=300)),
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="auto" if cfg.debug else "ddp",
        callbacks=call_backs,
        limit_val_batches=limit_val_batches if limit_val_batches is not None else 1.0,
    )

    # automatically resume training
    if cfg.ckpt_path is None and not cfg.debug:
        # Pattern to match all .ckpt files in the base_path recursively
        search_pattern = os.path.join('./unitraj', cfg.exp_name, '**', '*.ckpt')
        cfg.ckpt_path = find_latest_checkpoint(search_pattern)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)

    if not cfg.debug and checkpoint_callback.best_model_path:
        config_path = os.path.join(checkpoint_callback.dirpath, "config.yaml")
        OmegaConf.save(cfg, config_path)

        artifact = wandb.Artifact(name=cfg.exp_name, type="model")
        artifact.add_file(checkpoint_callback.best_model_path, name="best_model.ckpt")
        artifact.add_file(config_path, name="config.yaml")
        wandb.log_artifact(artifact)


if __name__ == '__main__':
    train()
