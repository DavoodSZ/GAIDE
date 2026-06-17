#!/usr/bin/env python3

import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule


class RobotPCD(nn.Module):
    def __init__(self):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.1,
                nsample=64,
                mlp=[0, 64, 64, 64, 128],
                use_xyz=True,
                bn=False,
            )
        )

    def forward(self, pointcloud: torch.Tensor):
        features = None
        xyz = pointcloud
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        return features.transpose(1, 2), xyz


class ScenePCD(nn.Module):
    def __init__(self):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.1,
                nsample=64,
                mlp=[0, 64, 64, 64, 128],
                use_xyz=True,
                bn=False,
            )
        )

    def forward(self, pointcloud: torch.Tensor):
        features = None
        xyz = pointcloud
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        return features.transpose(1, 2), xyz
