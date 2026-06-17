#!/usr/bin/env python3

import torch 
from dataclasses import dataclass

from curobo.types.base import TensorDeviceType 
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig


@dataclass
class CollisionResult:
    in_collision: bool
    world_collision: bool
    self_collision: bool
    max_world_distance: float
    max_self_distance: float
    world_penetration_distance: float
    self_penetration_distance: float
    world_penetration_distances: list
    self_penetration_distances: list
    world_penetration_mean: float
    world_penetration_std: float
    self_penetration_mean: float
    self_penetration_std: float
    num_world_penetrations: int
    num_self_penetrations: int

def collision_checking(
    configs,
    robot_file,
    world_config,
    collision_activation_distance=0.0,
    tolerance: float=0.01
):
    '''
    Collision checking the traje
    '''
    tensor_args = TensorDeviceType()
    config = RobotWorldConfig.load_from_config(
        robot_config=robot_file,
        world_model=world_config,
        collision_activation_distance=collision_activation_distance
    )

    curobo_fn = RobotWorld(config)

    d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(configs)
    in_collision = (torch.cat([d_world, d_self], dim=-1) > tolerance).any().item()
    return in_collision


def collision_checking_detailed(
    configs,
    robot_file,
    world_config,
    collision_activation_distance=0.0,
    tolerance: float=0.01
):
    '''
    cuRobo returns positive distance when a collision is inside the activation
    region. With activation distance 0, positive values are penetration.
    '''
    tensor_args = TensorDeviceType()
    config = RobotWorldConfig.load_from_config(
        robot_config=robot_file,
        world_model=world_config,
        collision_activation_distance=collision_activation_distance
    )

    curobo_fn = RobotWorld(config)

    d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(configs)
    max_world_distance = d_world.max().item() if d_world.numel() else float("-inf")
    max_self_distance = d_self.max().item() if d_self.numel() else float("-inf")
    world_penetrations = d_world[d_world > 0].detach().cpu().flatten().tolist()
    self_penetrations = d_self[d_self > 0].detach().cpu().flatten().tolist()
    world_penetrations_tensor = torch.as_tensor(world_penetrations, dtype=torch.float32)
    self_penetrations_tensor = torch.as_tensor(self_penetrations, dtype=torch.float32)
    world_penetration_mean = world_penetrations_tensor.mean().item() if world_penetrations else 0.0
    world_penetration_std = world_penetrations_tensor.std(unbiased=False).item() if world_penetrations else 0.0
    self_penetration_mean = self_penetrations_tensor.mean().item() if self_penetrations else 0.0
    self_penetration_std = self_penetrations_tensor.std(unbiased=False).item() if self_penetrations else 0.0
    world_collision = bool(max_world_distance > tolerance)
    self_collision = bool(max_self_distance > tolerance)
    return CollisionResult(
        in_collision=world_collision or self_collision,
        world_collision=world_collision,
        self_collision=self_collision,
        max_world_distance=max_world_distance,
        max_self_distance=max_self_distance,
        world_penetration_distance=max(0.0, max_world_distance),
        self_penetration_distance=max(0.0, max_self_distance),
        world_penetration_distances=world_penetrations,
        self_penetration_distances=self_penetrations,
        world_penetration_mean=world_penetration_mean,
        world_penetration_std=world_penetration_std,
        self_penetration_mean=self_penetration_mean,
        self_penetration_std=self_penetration_std,
        num_world_penetrations=len(world_penetrations),
        num_self_penetrations=len(self_penetrations),
    )

    
