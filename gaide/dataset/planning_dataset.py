#!/usr/bin/env python3

import os
import torch

from gaide.dataset.base_dataset import BaseDataset


class PlanningDataset(BaseDataset):
    def __init__(self, path: str, normal_info: bool = True):
        self.planning_data = torch.load(
            os.path.join(path, "planning_data.pth"),
            weights_only=False,
            map_location="cpu"
        )

    @torch.no_grad()
    def compute_delta_norm(self, chunk=4096, use_mask=True):
        delta = self.planning_data["delta_cfgs"]
        N, P, D = delta.shape
        acc_sum = torch.zeros(D, dtype=torch.float64)
        acc_sq = torch.zeros(D, dtype=torch.float64)
        acc_cnt = 0.0

        if use_mask:
            masks = self.planning_data["future_masks"]

        for i in range(0, N, chunk):
            x = delta[i:i + chunk].float()
            if use_mask:
                m = masks[i:i + chunk].to(torch.float32).unsqueeze(-1)
                acc_cnt += m.sum().item()
                acc_sum += (x * m).sum(dim=(0, 1)).to(torch.float64)
                acc_sq += ((x ** 2) * m).sum(dim=(0, 1)).to(torch.float64)
            else:
                c = x.shape[0] * x.shape[1]
                acc_cnt += c
                acc_sum += x.sum(dim=(0, 1)).to(torch.float64)
                acc_sq += (x.square()).sum(dim=(0, 1)).to(torch.float64)
            del x

        mean = (acc_sum / acc_cnt).view(1, 1, -1).float()
        var = (acc_sq / acc_cnt).view(1, 1, -1).float() - mean ** 2
        std = torch.sqrt(torch.clamp(var, 1e-6))
        return mean.squeeze().squeeze(), std.squeeze().squeeze()

    def __len__(self):
        return self.planning_data['current_cfgs'].shape[0]

    def __getitem__(self, idx):
        current_cfgs = self.planning_data["current_cfgs"][idx].float()
        context_len = current_cfgs.shape[0]

        goal_cfgs = self.planning_data["goal_cfgs"][idx].float()
        goal_cfgs = goal_cfgs.unsqueeze(0).expand(context_len, -1)

        ws_ids = int(self.planning_data["ws_ids"][idx])
        if ws_ids > 17:
            ws_ids = ws_ids - 1
        scene_pcds = self.planning_data["scene_pcds"][ws_ids].float()
        scene_pcds = scene_pcds.unsqueeze(0).expand(context_len, scene_pcds.size(0), scene_pcds.size(1))

        robot_origins = self.planning_data["robot_origins"][ws_ids].float()
        robot_origins = robot_origins.unsqueeze(0).expand(context_len, -1, -1)

        future_masks = self.planning_data["future_masks"][idx].to(torch.float32)
        next_cfgs = self.planning_data["next_cfgs"][idx].float()

        return current_cfgs, goal_cfgs, scene_pcds, robot_origins, future_masks, next_cfgs
