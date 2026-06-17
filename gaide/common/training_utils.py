#!/usr/bin/env python3

import numpy as np
import torch


def custom_collate_fn(batch):
    current_cfgs, goal_cfgs, scene_pcds, robot_origins, future_masks, next_cfgs = zip(*batch)
    features = {
        "current_cfgs": torch.stack(current_cfgs),
        "goal_cfgs": torch.stack(goal_cfgs),
        "scene_pcds": torch.stack(scene_pcds),
        "robot_origins": torch.stack(robot_origins),
    }
    labels = torch.stack(next_cfgs)
    return features, labels


def to_device(features, device):
    if torch.is_tensor(features):
        return features.to(device=device)
    if isinstance(features, np.ndarray):
        return torch.as_tensor(features, dtype=torch.float32, device=device)
    if isinstance(features, dict):
        return {k: to_device(v, device) for k, v in features.items()}
    if isinstance(features, list):
        return [to_device(x, device) for x in features]
    if isinstance(features, tuple):
        return tuple(to_device(x, device) for x in features)
