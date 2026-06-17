#!/usr/bin/env python3

import os
import random
import logging

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from omegaconf import OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import wandb

from gaide.common.training_utils import custom_collate_fn
from gaide.utils.training.ur5e_pcd_sampler import UR5ePCDSampler

log = logging.getLogger(__name__)


class TrainAgent:
    def __init__(self, cfg):
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.use_wandb = cfg.wandb is not None
        if cfg.wandb is not None:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        self.policy = hydra.utils.instantiate(cfg.policy)
        self.policy.to(cfg.device)
        log.info(f"Parameters: {sum(p.numel() for p in self.policy.parameters()):,}")

        self.checkpoint_dir = os.path.join(cfg.logdir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.dataset = hydra.utils.instantiate(cfg.dataset)
        train_indices, val_indices = self.dataset.train_val_split(train_ratio=cfg.data_split_ratio)
        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)

        dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        dataloader_cfg["collate_fn"] = custom_collate_fn
        val_dataloader_cfg = OmegaConf.to_container(cfg.val_dataloader, resolve=True)
        val_dataloader_cfg["collate_fn"] = custom_collate_fn
        self.dataloader_train = DataLoader(self.train_dataset, **dataloader_cfg)
        self.dataloader_val = DataLoader(self.val_dataset, **val_dataloader_cfg)

        self.optimizer = torch.optim.AdamW(
            params=self.policy.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=cfg.optimizer.betas,
        )

        self.cfg = cfg
        self.robot_pcd_sampler = UR5ePCDSampler(device=cfg.device, num_points=cfg.num_robot_pcd_points)

    def run(self):
        raise NotImplementedError

    def save_model(self):
        data = {"epoch": self.steps, "model": self.policy.state_dict()}
        save_path = os.path.join(self.checkpoint_dir, f"state_{self.steps}.pt")
        torch.save(data, save_path)
        log.info(f"Saved model to {save_path}")

    @property
    def output_dir(self):
        return HydraConfig.get().runtime.output_dir
