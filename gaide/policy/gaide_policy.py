#!/usr/bin/env python3

import torch
import torch.nn.functional as F

from gaide.policy.base_policy import BasePolicy


class GAIDEPolicy(BasePolicy):
    def __init__(self, network, dof: int = 6, device: str = "cuda:0"):
        super().__init__()
        self.network = network

    def compute_loss(self, features, actions):
        predicted = self.network(features=features)
        return F.mse_loss(predicted, actions, reduction="mean")

    @torch.no_grad()
    def forward(self, features):
        return self.network(features=features)
