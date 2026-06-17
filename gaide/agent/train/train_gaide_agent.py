#!/usr/bin/env python3

import copy
import logging

import numpy as np
import torch
import tqdm
import wandb

from gaide.agent.train.train_agent import TrainAgent
from gaide.common.checkpoint_util import TopKCheckpointManager
from gaide.common.training_utils import to_device
from gaide.common.timer import Timer

log = logging.getLogger(__name__)

JOINT_LIMIT = 2 * np.pi


class TrainGAIDEAgent(TrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        timer = Timer()

        TopKCheckpointManager(
            save_dir=self.checkpoint_dir,
            **cfg.checkpoint.topk,
        )

        self.steps = 1
        loss_train_epoch = []

        while self.steps <= cfg.train.gradient_steps:
            with tqdm.tqdm(
                self.dataloader_train,
                desc="Training",
                leave=False,
                mininterval=cfg.train.tqdm_interval_sec,
            ) as tepoch:
                for batch in tepoch:
                    batch = to_device(batch, cfg.device)
                    features, actions = batch
                    features = self.prepare_features(features)

                    self.policy.train()
                    loss = self.policy.compute_loss(features=features, actions=actions)
                    loss.backward()
                    loss_train_epoch.append(loss.item())
                    tepoch.set_postfix(loss=loss.item(), refresh=False)

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.steps += 1

                    loss_val_epoch = []
                    if self.steps % cfg.train.val_freq == 0:
                        self.policy.eval()
                        with torch.no_grad():
                            for val_batch in self.dataloader_val:
                                val_batch = to_device(val_batch, cfg.device)
                                val_features, val_actions = val_batch
                                val_features = self.prepare_features(val_features)
                                loss_val = self.policy.compute_loss(features=val_features, actions=val_actions)
                                loss_val_epoch.append(loss_val.item())
                        self.policy.train()
                    loss_val = np.mean(loss_val_epoch) if loss_val_epoch else None

                    if self.steps % cfg.train.save_model_freq == 0 or self.steps == cfg.train.gradient_steps:
                        self.save_model()

                    if self.steps % cfg.train.log_freq == 0:
                        log.info(
                            f"step {self.steps}: train loss {np.mean(loss_train_epoch):.4f} | {timer():.2f}s"
                        )
                        if self.use_wandb:
                            if loss_val is not None:
                                wandb.log({"loss/val": loss_val}, step=self.steps)
                            wandb.log({"loss/train": np.mean(loss_train_epoch)}, step=self.steps)
                        loss_train_epoch = []

    def prepare_features(self, features):
        batch_size, context_len, num_joints = features["current_cfgs"].shape
        features["current_cfgs"] = features["current_cfgs"].reshape(batch_size * context_len, -1)
        features["goal_cfgs"] = features["goal_cfgs"].reshape(batch_size * context_len, -1)
        features["scene_pcds"] = features["scene_pcds"].reshape(-1, self.cfg.num_scene_pcd_points, 3)
        features["robot_origins"] = features["robot_origins"].reshape(batch_size * context_len, 2, 3)

        features["current_pcds"] = self.robot_pcd_sampler.sample(
            robot_origins=features["robot_origins"],
            robot_configs=features["current_cfgs"],
        )

        features["current_cfgs"] = features["current_cfgs"] / JOINT_LIMIT
        features["goal_cfgs"] = features["goal_cfgs"] / JOINT_LIMIT

        return features
