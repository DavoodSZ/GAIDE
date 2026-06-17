#!/usr/bin/env python3

import torch
from typing import Literal

from .torch_ur5e import UR5eSampler
from gaide.utils.training.utils import visualize_multiple_pcds


class UR5ePCDSampler:
    def __init__(self, device, num_points: int):
        self.device = device
        self.num_points = num_points

        self.gpu_fk_sampler = UR5eSampler(device=device, use_cache=False)

    def sample(self, robot_origins: torch.Tensor, robot_configs: torch.Tensor, debug=False):
        '''
        Sample point-cloud on the surface of the robot at the given configuration.
        '''
        # points within robot base frame
        points1 = self.gpu_fk_sampler.sample(robot_configs, self.num_points)
        # Moves points to the world frame
        points = self.transform_pcd(points=points1, origin=robot_origins)
        if debug:
            visualize_multiple_pcds([points1, points], colors=[[1,0,0], [0,1,0]])
        return points

    def transform_pcd(self, points: torch.Tensor, origin: torch.Tensor):
        '''
        Transforms the generated pointcloud to the world-frame.
        '''
        translation = origin[:, 0]
        rotation = matrix_from_euler(origin[:, 1], convention="XYZ")

        points_world = torch.bmm(points, rotation.transpose(1, 2)) + translation.unsqueeze(1)
        return points_world


def matrix_from_euler(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    borrowed from IsaacLab: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/utils/math.py
    Convert rotations given as Euler angles (intrinsic) in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians. Shape is (..., 3).
        convention: Convention string of three uppercase letters from {"X", "Y", and "Z"}.
            For example, "XYZ" means that the rotations should be applied first about x,
            then y, then z.

    Returns:
        Rotation matrices. Shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L194-L220
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [_axis_angle_rotation(c, e) for c, e in zip(convention, torch.unbind(euler_angles, -1))]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def _axis_angle_rotation(axis: Literal["X", "Y", "Z"], angle: torch.Tensor) -> torch.Tensor:
    """
    borrowed from IsaacLab: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/utils/math.py
    Return the rotation matrices for one of the rotations about an axis of which Euler angles describe,
    for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: Euler angles in radians of any shape.

    Returns:
        Rotation matrices. Shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L164-L191
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))
