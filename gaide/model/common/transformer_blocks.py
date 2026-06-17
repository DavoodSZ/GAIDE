#!/usr/bin/env python3

import torch
import torch.nn as nn

from .pe import PositionalEncoding
from .masked_encoder import MaskedTransformerEncoder


class GraphTransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        heads: int = 8,
        feedforward_dim: int = 1024,
        num_layers: int = 6,
        mask_mode: str = "mixed",
    ):
        super().__init__()

        self.nets = nn.ModuleDict()
        self.params = nn.ParameterDict()

        self.params["scene_type"] = nn.Parameter(torch.zeros(hidden_dim))
        self.params["robot_type"] = nn.Parameter(torch.zeros(hidden_dim))
        self.params["joint_type"] = nn.Parameter(torch.zeros(hidden_dim))
        self.params["goal_type"] = nn.Parameter(torch.zeros(hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            batch_first=True,
            dropout=0.0,
        )
        self.nets["encoder"] = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            mask_mode=mask_mode,
        )

    def forward(self, joint_features, goal_features, robot_features, scene_features):
        adj_matrix = self._build_adj_matrix(robot_features, scene_features)

        joint_features = joint_features + self.params["joint_type"]
        goal_features = goal_features + self.params["goal_type"]
        robot_features = robot_features + self.params["robot_type"]
        scene_features = scene_features + self.params["scene_type"]

        robot_features = joint_features.unsqueeze(1) + goal_features.unsqueeze(1) + robot_features
        tokens = torch.cat([robot_features, scene_features], dim=1)

        memory = self.nets["encoder"](src=tokens, mask=~adj_matrix)
        return memory

    def _build_adj_matrix(self, robot_features: torch.Tensor, scene_features: torch.Tensor):
        _, robot_nodes, _ = robot_features.shape
        _, scene_nodes, _ = scene_features.shape
        num_nodes = robot_nodes + scene_nodes
        device = robot_features.device

        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)

        # Robot kinematic chain: upper-triangular + diagonal (each joint attends
        # to itself and all downstream joints, reflecting a chain topology)
        robot_adj = torch.zeros(robot_nodes, robot_nodes, dtype=torch.bool, device=device)
        robot_adj.fill_diagonal_(True)
        for j in range(robot_nodes):
            if j > 0:
                robot_adj[j, j - 1] = True
            if j + 1 < robot_nodes:
                robot_adj[j, j + 1:] = True
        adj[:robot_nodes, :robot_nodes] = robot_adj

        # Every scene point attends to all robot joints
        adj[robot_nodes:, :robot_nodes] = True

        # Scene points only attend to themselves (no scene-scene attention)
        adj[robot_nodes:, robot_nodes:] = torch.eye(scene_nodes, dtype=torch.bool, device=device)
        return adj


class TransformerDecoder(nn.Module):
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
        self.params = nn.ParameterDict()

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            batch_first=True,
        )
        self.nets["pose_enc"] = PositionalEncoding(embed_dim=hidden_dim)
        self.nets["decoder"] = nn.TransformerDecoder(
            decoder_layer=dec_layer,
            num_layers=num_layers,
        )
        self.nets["head"] = nn.Linear(hidden_dim, dof)
        self.params["action_queries"] = nn.Parameter(torch.randn(chunk, hidden_dim))

    def forward(self, memory, batch_size):
        tgt = self.params["action_queries"].unsqueeze(0).expand(batch_size, -1, -1)
        tgt = tgt + self._embed_timesteps(tgt)
        tgt = self.nets["decoder"](tgt=tgt, memory=memory)
        return self.nets["head"](tgt)

    def _embed_timesteps(self, embeddings):
        timesteps = (
            torch.arange(
                0,
                embeddings.shape[1],
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            .unsqueeze(0)
            .repeat(embeddings.shape[0], 1)
        )
        return self.nets["pose_enc"](timesteps)
