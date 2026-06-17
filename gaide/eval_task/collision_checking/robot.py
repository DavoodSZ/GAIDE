#!/usr/bin/env python3

'''
Everything should be in batch here to work effectively with CuRobo
'''

import torch
from scipy.spatial.transform import Rotation as R
import pybullet as pyb

from .transformation_utils import *
from gaide.utils.pybullet import (
    getJointInfo,
    getJointStates,
)

class HemisphereWorkspace:
    def __init__(self, radius, origin, num_points: int=100):
        self.radius = radius
        self.origin = np.array(origin)
        self.num_points = num_points

    def points_in_workspace(self):
        '''
        Generates random points in the robotic manipulator workspace
        '''
        points = []
        for i in range(self.num_points):
            workspace_point = self.point_in_workspace()
            points.append(workspace_point)

        return points

    def point_in_workspace(self, i=None, j=None, k=None):
        '''
        Generates a random point within the workspace of the robotic manipulator (upper hemisphere)
        '''
        if i is None or j is None or k is None:
            i = np.random.uniform(0, 1)
            j = np.random.uniform(0, 1)
            k = np.random.uniform(0, 1)
        j = j ** 0.5
        output = np.array([
            self.radius * j * np.cos(i * np.pi * 2) * np.cos(k * np.pi / 2),
            self.radius * j * np.sin(i * np.pi * 2) * np.cos(k * np.pi /2),
            self.radius * j * np.sin(k * np.pi / 2),
        ])
        output = output + self.origin
        assert np.linalg.norm(np.array(output) - self.origin) < self.radius
        return output.tolist()


class Manipulator:
    def __init__(self, simulator, ws_info: dict, device, gen_mode: str="neuralmp"):
        self.simulator = simulator
        self.origin = self.get_origin(ws_info=ws_info, gen_mode=gen_mode)
        self.device = device

        if self.simulator is not None:
            self.robot = self.load_robot()

            # Get robot info
            (
                self.link_index_map,
                self.joint_index_map,
                self.moveable_joint_indices,
                self.joint_limits
            ) = self.robot_info()

            # robot workspace
            self.workspace = HemisphereWorkspace(radius=self.reachability, origin=self.origin[0], num_points=100)
            self.workspace_points = self.workspace.points_in_workspace()

    def get_origin(self, ws_info: dict, gen_mode: str="neuralmp"):
        '''
        Gets the base origin of robotic manipulator
        '''
        if gen_mode == "neuralmp":
            table_height = ws_info["front_table"]["dim"]["height"] + ws_info["front_table"]["thickness"] / 2
        elif gen_mode == "scene_synthesizer":
            table_height = ws_info["front_table"]["dim"]["height"]
        else:
            raise ValueError(f"Generation mode, {gen_mode} is not valid! Valid modes are: ['NeuralMP', 'SceneSynthesizer']")

        # Pedestal height
        pedestal_height = self.pedestal_dimension[2] * self.pedestal_scale

        # manipulator base origin
        origin_x = self.base_offset[0]
        origin_y = self.base_offset[1]
        origin_z = pedestal_height - table_height

        origin_xyz = [origin_x, origin_y, origin_z]
        origin_xyzw = [0.0, 0.0, 0.0]

        return [origin_xyz, origin_xyzw]

    def load_robot(self):
        # Spwan the robot
        robot_bullet = self.simulator.add_primitive(self.urdf_path, self.origin)
        return robot_bullet

    def robot_info(self):
        '''
        Gets the robot info from the URDF file.
        '''
        link_index_map = {}
        joint_index_map = {}
        moveable_joint_indices = []

        n = pyb.getNumJoints(self.robot, physicsClientId=0)
        for i in range(n):
            info = getJointInfo(
                self.robot, i, decode='utf-8', physicsClientId=0
            )
            joint_index_map[info.jointName] = i
            link_index_map[info.linkName] = i

            if info.jointType != pyb.JOINT_FIXED:
                moveable_joint_indices.append(i)

        assert len(moveable_joint_indices) >= self.DOF, "Moveable joints should at least be equal to manipulator DOF"
        moveable_joint_indices = moveable_joint_indices[:self.DOF]

        joint_limits = np.zeros((len(moveable_joint_indices), 2))
        idx = 0
        for i in moveable_joint_indices:
            info = getJointInfo(self.robot, i, decode="utf-8", physicsClientId=0)
            lower = float(info[8])
            upper = float(info[9])
            joint_limits[idx, :] = np.array([lower, upper])
            idx += 1

        return link_index_map, joint_index_map, moveable_joint_indices, joint_limits

    def config_in_base(self, points, ws_info: dict, gen_mode: str="neuralmp"):
        '''
        Transforms the config from the world frame to robotic manipulator base frame in batch
        '''
        origin = self.get_origin(ws_info=ws_info, gen_mode=gen_mode)

        # Bring to robot frame
        translation = torch.as_tensor(origin[0], dtype=torch.float32, device=self.device)
        rotation = torch.as_tensor(R.from_euler('xyz', origin[1]).as_matrix(), dtype=torch.float32, device=self.device)

        # Transform the global frame
        points_base = (points - translation) @ rotation
        return points_base

    def world_to_base(self, pose: list):
        '''
        Transforms the query point from world frame to the base frame of the robotic manipulator.
        '''
        world_pose = np.array(pose[0])
        world_quat = pose[1]

        base_pose = np.array([0.0, 0.0, 0.0])
        base_quat = quaternion_from_rpy(*self.base_orientation)

        # Inverse base pose
        base_inv_pose, base_inv_quat = pose_inverse(base_pose, base_quat)

        # Multiply
        pose_base, pose_quat = pose_multiply(base_inv_pose, base_inv_quat, world_pose, world_quat)

        return pose_base, pose_quat

    def grasp_to_ee(self, pose):
        '''
        Moves the grasp to the EE for effective inverse kinematics.
        '''
        pos, quat = self.world_to_base(pose=pose)
        pos_t, quat_t = self.open_grasp_T
        position, quat = pose_multiply(pos, quat, pos_t, quat_t)
        return [position, quat]

    @property
    def open_grasp_T(self):
        '''
        Transformation from EE to the Grasp when the gripper is open.
        '''
        pos1, quat1 = self.grasp_pose

        pos1, quat1 = inverse(pos1, quat1)
        return pos1, quat1

class UR5e(Manipulator):
    def __init__(self, simulator, ws_info: dict, device, gen_mode: str="neuralmp", name="ur5e"):
        super().__init__(simulator=simulator, ws_info=ws_info, device=device, gen_mode=gen_mode)

    @property
    def urdf_path(self):
        from gaide import DIRECTORY
        return os.path.join(DIRECTORY, "assets/robots/ur5e/ur5e.urdf")

    @property
    def home_config(self):
        return [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]

    @property
    def name(self):
        return "ur5e"

    @property
    def DOF(self):
        return 6

    @property
    def pedestal_scale(self):
        return 0.55  # Determined via trial-and-error in the URDF file.

    @property
    def pedestal_offset(self):
        return 0.05            # Determined via trial-and-error in the URDF file.

    @property
    def pedestal_dimension(self):
        return [0.9143719673156738, 0.822801411151886, 0.8666549921035767]   # This the original dimension, without any scale

    @property
    def base_offset(self):
        return [0.0, 0.0, 0.0]

    @property
    def base_orientation(self):
        return [0.0, 0.0, 1.57] # from the URDF - to be compatible with our workspaces

    @property
    def grasp_pose(self):
        pos = np.array([0.177292655, 0.0, 0.0])
        rpy = [0.0, 0.0, 0.0]
        quat = quaternion_from_rpy(rpy[0], rpy[1], rpy[2])
        return pos, quat

    @property
    def reachability(self):
        return 0.85

    @property
    def vel_limit(self):
        return [3.15, 3.15, 3.15, 3.15, 3.15, 3.15]

    @property
    def acc_limit(self):
        return [12.0, 12.0, 12.0, 12.0, 12.0, 12.0]

    @property
    def jerk_limit(self):
        return [500.0, 500.0, 500.0, 500.0, 500.0, 500.0]

class SphericalUR5e(Manipulator):
    def __init__(self, simulator, ws_info: dict, device, gen_mode: str="neuralmp", name="ur5e"):
        super().__init__(simulator=simulator, ws_info=ws_info, device=device, gen_mode=gen_mode)

    @property
    def urdf_path(self):
        from gaide import DIRECTORY
        return os.path.join(DIRECTORY, "assets/robots/ur5e/ur5e_spherized_visual.urdf")

    @property
    def home_config(self):
        return [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]

    @property
    def name(self):
        return "ur5e"

    @property
    def DOF(self):
        return 6

    @property
    def pedestal_scale(self):
        return 0.55  # Determined via trial-and-error in the URDF file.

    @property
    def pedestal_offset(self):
        return 0.05            # Determined via trial-and-error in the URDF file.

    @property
    def pedestal_dimension(self):
        return [0.9143719673156738, 0.822801411151886, 0.8666549921035767]   # This the original dimension, without any scale

    @property
    def base_offset(self):
        return [0.0, 0.0, 0.0]

    @property
    def base_orientation(self):
        return [0.0, 0.0, 1.57] # from the URDF - to be compatible with our workspaces

    @property
    def grasp_pose(self):
        pos = np.array([0.177292655, 0.0, 0.0])
        rpy = [0.0, 0.0, 0.0]
        quat = quaternion_from_rpy(rpy[0], rpy[1], rpy[2])
        return pos, quat

    @property
    def reachability(self):
        return 0.85

class Franka(Manipulator):
    def __init__(self, simulator, ws_info: dict, gen_mode: str="neuralmp", name="panda"):
        super().__init__(simulator=simulator, ws_info=ws_info, gen_mode=gen_mode)

    @property
    def urdf_path(self):
        from gaide import DIRECTORY
        return os.path.join(DIRECTORY, "assets/robots/panda/panda.urdf")

    @property
    def home_config(self):
        return [0.0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi / 4]

    @property
    def name(self):
        return "panda"

    @property
    def DOF(self):
        return 7

    @property
    def pedestal_scale(self):
        return 0.55  # Determined via trial-and-error in the URDF file.

    @property
    def pedestal_offset(self):
        return 0.05            # Determined via trial-and-error in the URDF file.

    @property
    def pedestal_dimension(self):
        return [0.9143719673156738, 0.822801411151886, 0.8666549921035767]   # This the original dimension, without any scale

    @property
    def base_offset(self):
        return [0.0, 0.0, 0.0]

    @property
    def base_orientation(self):
        return [0.0, 0.0, 0.0] # Franka doesn't have any base_orientaion within the URDF

    @property
    def grasp_pose(self):
        '''
        TODO: get these values from URDF
        '''
        pos = np.array([0.0, 0.0, 0.1])
        rpy = [0.0, 0.0, 0.0]
        quat = quaternion_from_rpy(rpy[0], rpy[1], rpy[2])
        return pos, quat

    @property
    def reachability(self):
        return 0.855
