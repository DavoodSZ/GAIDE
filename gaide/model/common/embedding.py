#!/usr/bin/env python3

import torch.nn as nn


class ConfigEmbedding(nn.Module):
    def __init__(self, dof=6, hidden_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dof, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim))

    def forward(self, features):
        return self.mlp(features)
