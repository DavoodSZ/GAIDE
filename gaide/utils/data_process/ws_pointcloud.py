#!/usr/bin/env python3

# TODO: Assign different number of points for each primitive.

import trimesh
import os
import json
import pickle
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from gaide.utils.training.torch_ur5e import UR5eSampler

class PointCloudSampler:
    '''
    Samples pointcloud from the surface of all the meshes within the workspace.
    '''
    def __init__(self,
        gen_mode: str="neuralmp",
        num_points: int=1024,
        robot_name: str="ur5e",
    ):
        self.gen_mode = gen_mode
        self.num_points = num_points
        self.robot_name = robot_name

    def sample(self, ws_log_dir: str):
        '''
        Samples pointcloud on the body of the primitives as a sensing modality.
        '''
        # load workspace information
        self.primitive_nums, self.primitive_locations, self.primitive_dimensions = self.load_ws_info(ws_log_dir=ws_log_dir)

        # sample on the body and transform.
        for key in self.primitive_locations.keys():
            if key == "front_table":
                log_dir1 = os.path.join(ws_log_dir, 'table_1')

                # Sampling point cloud on table.
                origin = self.primitive_locations[key]["location"]
                filepath = os.path.join(log_dir1, 'table/table_0_mesh.obj')

                pcd = self.mesh_sample(filepath=filepath, origin=origin, num_points=self.num_points, name="table")

                for key1, value1 in self.primitive_nums[key].items():
                    for i in range(value1):
                        primitive_origin = self.primitive_locations[key]["primitives"][f"{key1}_{i}"]
                        filepath = os.path.join(log_dir1, f"{key1}/{key1}_{i}_mesh.obj")

                        pcd_primitive = self.mesh_sample(filepath=filepath, origin=primitive_origin, num_points=self.num_points, name=key1)
                        if pcd_primitive is not None:
                            pcd = np.concatenate((pcd, pcd_primitive), axis=0)

            elif key == "left_table":
                log_dir1 = os.path.join(ws_log_dir, 'table_2')

                # Sampling point cloud on table.
                origin = self.primitive_locations[key]["location"]
                filepath = os.path.join(log_dir1, 'table/table_0_mesh.obj')

                pcd_primitive = self.mesh_sample(filepath=filepath, origin=origin, num_points=self.num_points, name="table")
                if pcd_primitive is not None:
                    pcd = np.concatenate((pcd, pcd_primitive), axis=0)

                for key1, value1 in self.primitive_nums[key].items():
                    for i in range(value1):
                        primitive_origin = self.primitive_locations[key]["primitives"][f"{key1}_{i}"]
                        filepath = os.path.join(log_dir1, f"{key1}/{key1}_{i}_mesh.obj")

                        pcd_primitive = self.mesh_sample(filepath=filepath, origin=primitive_origin, num_points=self.num_points, name=key1)
                        if pcd_primitive is not None:
                            pcd = np.concatenate((pcd, pcd_primitive), axis=0)

            elif key == "right_table":
                if "left_table" in self.primitive_locations:
                    log_dir1 = os.path.join(ws_log_dir, 'table_3')
                else:
                    log_dir1 = os.path.join(ws_log_dir, 'table_2')

                # Sampling point cloud on table.
                origin = self.primitive_locations[key]["location"]
                filepath = os.path.join(log_dir1, 'table/table_0_mesh.obj')

                pcd_primitive = self.mesh_sample(filepath=filepath, origin=origin, num_points=self.num_points, name="table")
                if pcd_primitive is not None:
                    pcd = np.concatenate((pcd, pcd_primitive), axis=0)

                for key1, value1 in self.primitive_nums[key].items():
                    for i in range(value1):
                        primitive_origin = self.primitive_locations[key]["primitives"][f"{key1}_{i}"]
                        filepath = os.path.join(log_dir1, f"{key1}/{key1}_{i}_mesh.obj")

                        pcd_primitive = self.mesh_sample(filepath=filepath, origin=primitive_origin, num_points=self.num_points, name=key1)
                        if pcd_primitive is not None:
                            pcd = np.concatenate((pcd, pcd_primitive), axis=0)

            elif key == "back_table":
                log_dir1 = os.path.join(ws_log_dir, 'table_4')

                # Sampling point cloud on table.
                origin = self.primitive_locations[key]["location"]
                filepath = os.path.join(log_dir1, 'table/table_0_mesh.obj')

                pcd_primitive = self.mesh_sample(filepath=filepath, origin=origin, num_points=self.num_points, name="table")
                if pcd_primitive is not None:
                    pcd = np.concatenate((pcd, pcd_primitive), axis=0)

                for key1, value1 in self.primitive_nums[key].items():
                    for i in range(value1):
                        primitive_origin = self.primitive_locations[key]["primitives"][f"{key1}_{i}"]
                        filepath = os.path.join(log_dir1, f"{key1}/{key1}_{i}_mesh.obj")

                        pcd_primitive = self.mesh_sample(filepath=filepath, origin=primitive_origin, num_points=self.num_points, name=key1)
                        if pcd_primitive is not None:
                            pcd = np.concatenate((pcd, pcd_primitive), axis=0)

            elif key == "on_ground":
                log_dir1 = os.path.join(ws_log_dir, "on_ground")

                # Adding on_ground primitives
                for key1, value1 in self.primitive_locations[key].items():
                    primitive_origin = self.primitive_locations[key][key1]
                    filepath = os.path.join(log_dir1, f"{key1}/{key1}_0_mesh.obj")

                    pcd_primitive = self.mesh_sample(filepath=filepath, origin=primitive_origin, num_points=self.num_points, name=key1)
                    if pcd_primitive is not None:
                        pcd = np.concatenate((pcd, pcd_primitive), axis=0)

        return pcd

    def robot_sampler(self, ws_log_dir: str, robot_name: str="ur5e", robot_config: np.ndarray= np.array([np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])):
        '''
        Samples points on the body of the robot.
        '''
        primitive_nums, primitive_locations, primitive_dimensions = self.load_ws_info(ws_log_dir=ws_log_dir)

        device = "cuda"
        if self.robot_name == "ur5e":
            gpu_fk_sampler = UR5eSampler(device, use_cache=False)
            robot_config = torch.tensor(robot_config).to(device)
            # sample
            points = gpu_fk_sampler.sample(robot_config, self.num_points).cpu().squeeze().numpy()
            # robot origin
            robot_origin = self.get_robot_origin(ws_info=primitive_dimensions, gen_mode=self.gen_mode)
            # Move points to the world coordinate
            points = self.transform_pcd(points=points, origin=robot_origin)
            return points

        elif self.robot_name == "panda":
            pass

        else:
            raise ValueError("Robot is not supported at the moment...")

    def load_ws_info(self, ws_log_dir: str):
        '''
        Loads workspace info for pointcloud sampling
        '''
        # number of primitives
        with open(os.path.join(ws_log_dir, 'primitive_nums.json'), 'r') as f:
            primitive_nums = json.load(f)
        with open(os.path.join(ws_log_dir, 'primitive_locations.json'), 'r') as f:
            primitive_locations = json.load(f)
        with open(os.path.join(ws_log_dir, "primitive_dimension.json"), 'r') as f:
            primitive_dimensions = json.load(f)

        return primitive_nums, primitive_locations, primitive_dimensions

    def mesh_sample(self, filepath: str, origin: list, name: str, num_points: int=1024):
        '''
        Sample pointclouds on the surface of the mesh and trasnforms them to the global coordinates.
        '''
        # sample
        mesh = trimesh.load(filepath, force="mesh")

        points = np.array(trimesh.sample.sample_surface(mesh, num_points)[0])
        # transform point cloud
        points = self.transform_pcd(points=points, origin=origin)

        if not np.isfinite(points).all():
            return None
        return points

    def transform_pcd(self, points: np.ndarray, origin: list):
        '''
        Transforms the generated pointcloud to the world coordinate.
        '''
        rpy = origin[1]
        translation = np.array(origin[0])

        # Rotation matrix
        Rmat = R.from_euler('xyz', rpy).as_matrix()

        # Transorm the pointcloud to the world coordinate
        points_world = points @ Rmat.T + translation
        return points_world

    def save_pcd(self, pcd: np.ndarray, save_dir: str):
        '''
        Saves the workspace pointcloud
        '''
        with open(os.path.join(save_dir, "pcd.pkl"), 'wb') as f:
            pickle.dump(pcd, f)

    def _preprocess_dir(self, log_dir: str):
        '''
        Clear the directory from the previously generated content and creates a newly empty one
        '''
        # Delete previous directory
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        # Create new empty one
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def get_robot_origin(self, ws_info, gen_mode: str="neuralmp"):
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
        origin_x = 0.0
        origin_y = 0.0
        origin_z = pedestal_height - table_height

        origin_xyz = [origin_x, origin_y, origin_z]
        origin_xyzw = [0.0, 0.0, 0.0]

        return [origin_xyz, origin_xyzw]

    @property
    def pedestal_scale(self):
        return 0.55

    @property
    def pedestal_dimension(self):
        return [0.9143719673156738, 0.822801411151886, 0.8666549921035767]
