#!/usr/bin/env python3

import os
import numpy as np
import pybullet as pyb
from scipy.spatial.transform import Rotation as R
import torch

from gaide.utils.pybullet.bullet_named_utils import getJointInfo
from gaide.eval_task.collision_checking.transformation_utils import quaternion_from_rpy, pose_inverse, pose_multiply


class Manipulator:
    def __init__(self, simulator, device, table_info):
        self.simulator = simulator
        self.device = device

        self.origin = self.get_origin(table_info=table_info)
        if self.simulator is not None:
            self.robot = self.load_robot()
            (
                self.link_index_map,
                self.joint_index_map,
                self.moveable_joint_indices,
                self.joint_limits,
            ) = self.get_robot_info()
            self.simulator.marionette(self.robot, self.home_config, self.moveable_joint_indices)

    def get_origin(self, table_info: dict):
        if table_info is not None:
            table_height = table_info["height_range"] + table_info["height"] / 2
            pedestal_height = self.pedestal_dimension[2] * self.pedestal_scale
            origin_z = pedestal_height - table_height
        else:
            origin_z = 0.0
        return [[0.0, 0.0, origin_z], [0.0, 0.0, 0.0]]

    def load_robot(self):
        return self.simulator.add_primitive(self.urdf_path, self.origin)

    def get_robot_info(self):
        link_index_map = {}
        joint_index_map = {}
        moveable_joint_indices = []

        n = pyb.getNumJoints(self.robot, physicsClientId=0)
        for i in range(n):
            info = getJointInfo(self.robot, i, decode='utf-8', physicsClientId=0)
            joint_index_map[info.jointName] = i
            link_index_map[info.linkName] = i
            if info.jointType != pyb.JOINT_FIXED:
                moveable_joint_indices.append(i)

        assert len(moveable_joint_indices) >= self.DOF
        moveable_joint_indices = moveable_joint_indices[:self.DOF]

        joint_limits = np.zeros((len(moveable_joint_indices), 2))
        for idx, i in enumerate(moveable_joint_indices):
            info = getJointInfo(self.robot, i, decode="utf-8", physicsClientId=0)
            joint_limits[idx, :] = np.array([float(info[8]), float(info[9])])

        return link_index_map, joint_index_map, moveable_joint_indices, joint_limits

    def config_in_base(self, points, table_info: dict = None):
        origin = self.get_origin(table_info=table_info)
        translation = torch.as_tensor(origin[0], dtype=torch.float32, device=self.device)
        rotation = torch.as_tensor(
            R.from_euler('xyz', origin[1]).as_matrix(), dtype=torch.float32, device=self.device
        )
        return (points - translation) @ rotation


class UR5e(Manipulator):
    def __init__(self, simulator, device, table_info=None):
        super().__init__(simulator=simulator, device=device, table_info=table_info)

    @property
    def urdf_path(self):
        from gaide import DIRECTORY
        return os.path.join(DIRECTORY, "assets/robots/ur5e/ur5e.urdf")

    @property
    def home_config(self):
        return [np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]

    @property
    def DOF(self):
        return 6

    @property
    def pedestal_scale(self):
        return 0.55

    @property
    def pedestal_dimension(self):
        return [0.9143719673156738, 0.822801411151886, 0.8666549921035767]
