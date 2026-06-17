#!/usr/bin/env python3

import random

import numpy as np
import torch
import hydra


class EvalAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.policy = hydra.utils.instantiate(cfg.policy).to(cfg.device)
        self.planner = hydra.utils.instantiate(cfg.planner)

    def run(self):
        pass
