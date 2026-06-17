#!/usr/bin/env python3

import torch

from curobo.types.base import TensorDeviceType
from curobo.util_file import get_robot_path, join_path, load_yaml
from curobo.types.robot import RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig


class UR5eFK:
    def __init__(self, device: str = None):
        tensor_args = TensorDeviceType()
        self.robot_file = "ur5e_c.yml"
        self.device = device if device and torch.cuda.is_available() else "cpu"

        config_file = load_yaml(join_path(get_robot_path(), self.robot_file))
        urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
        base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
        ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
        robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)

    def fk(self, configs: torch.Tensor):
        pose = self.kin_model.get_state(configs)
        return pose


def collision_checking(
    configs,
    robot_file,
    world_config,
    collision_activation_distance: float = 0.0,
    tolerance: float = 0.01,
) -> bool:
    tensor_args = TensorDeviceType()
    config = RobotWorldConfig.load_from_config(
        robot_config=robot_file,
        world_model=world_config,
        collision_activation_distance=collision_activation_distance,
    )
    curobo_fn = RobotWorld(config)
    d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(configs)
    in_collision = (torch.cat([d_world, d_self], dim=-1) > tolerance).any().item()
    return in_collision
