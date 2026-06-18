#!/usr/bin/env python3

import torch.nn as nn

from gaide.model.common import (
    ConfigEmbedding,
    RobotPCD,
    ScenePCD,
    GraphTransformerEncoder,
    TransformerDecoder,
)


class GAIDE(nn.Module):
    """GAIDE: Graph-based Attention masking for spatIal- and embodiment-aware motion planning DEcoder.

    Proposed method. Uses mixed graph attention masking: layers alternate between
    applying the kinematic-chain adjacency mask and attending to all tokens.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        heads: int = 8,
        feedforward_dim: int = 1024,
        num_layers: int = 6,
        dof: int = 6,
        chunk: int = 1,
    ):
        super().__init__()

        self.nets = nn.ModuleDict()

        self.nets["joint_embed"] = ConfigEmbedding(dof=dof, hidden_dim=hidden_dim)
        self.nets["goal_embed"] = ConfigEmbedding(dof=dof, hidden_dim=hidden_dim)

        self.nets["robot_pcd"] = RobotPCD(hidden_dim=hidden_dim)
        self.nets["scene_pcd"] = ScenePCD(hidden_dim=hidden_dim)

        self.nets["trans_encoder"] = GraphTransformerEncoder(
            hidden_dim=hidden_dim,
            heads=heads,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            mask_mode="mixed",
        )
        self.nets["trans_decoder"] = TransformerDecoder(
            hidden_dim=hidden_dim,
            heads=heads,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            dof=dof,
            chunk=chunk,
        )

    def forward(self, features, context_len=None):
        joint_features = self.nets["joint_embed"](features["current_cfgs"])
        goal_features = self.nets["goal_embed"](features["goal_cfgs"])
        batch_size, _ = joint_features.shape

        robot_features, _ = self.nets["robot_pcd"](pointcloud=features["current_pcds"])
        scene_features, _ = self.nets["scene_pcd"](pointcloud=features["scene_pcds"])

        memory = self.nets["trans_encoder"](
            joint_features=joint_features,
            goal_features=goal_features,
            robot_features=robot_features,
            scene_features=scene_features,
        )
        return self.nets["trans_decoder"](memory=memory, batch_size=batch_size)
