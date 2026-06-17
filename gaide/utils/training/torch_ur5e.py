#!/usr/bin/env python3

from pathlib import Path
import logging
import torch
import numpy as np
import trimesh

from .torch_urdf import TorchURDF
from .ur5e import UR5eRobot

def transform_pointcloud(pc, transformation_matrix, in_place=True):
    """

    Parameters
    ----------
    pc: A pytorch tensor pointcloud, maybe with some addition dimensions.
        This should have shape N x [3 + M] where N is the number of points
        M could be some additional mask dimensions or whatever, but the
        3 are x-y-z
    transformation_matrix: A 4x4 homography

    Returns
    -------
    Mutates the pointcloud in place and transforms x, y, z according the homography

    """
    assert isinstance(pc, torch.Tensor)
    assert type(pc) == type(transformation_matrix)
    assert pc.ndim == transformation_matrix.ndim
    if pc.ndim == 3:
        N, M = 1, 2
    elif pc.ndim == 2:
        N, M = 0, 1
    else:
        raise Exception("Pointcloud must have dimension Nx3 or BxNx3")
    xyz = pc[..., :3]
    ones_dim = list(xyz.shape)
    ones_dim[-1] = 1
    ones_dim = tuple(ones_dim)
    homogeneous_xyz = torch.cat((xyz, torch.ones(ones_dim, device=xyz.device)), dim=M)
    transformed_xyz = torch.matmul(
        transformation_matrix, homogeneous_xyz.transpose(N, M)
    )
    if in_place:
        pc[..., :3] = transformed_xyz[..., :3, :].transpose(N, M)
        return pc
    return torch.cat((transformed_xyz[..., :3, :].transpose(N, M), pc[..., 3:]), dim=M)

class UR5eSampler:
    '''
    This class allows for fast pointcloud sampling from the surface of a robot.
    At initialization, it loads a URDF and samples points from the mesh of each link.
    The points per link are based on the (very approximate) surface area of the link.

    Then, after instantiation, the sample method takes in a batch of configurations
    and produces pointclouds for each configuration by running FK on a subsample
    of the per-link pointclouds that are established at initialization.
    '''

    def __init__(self, device, num_fixed_points=None, use_cache=False, default_prismatic_value=0.025, with_base_link=True):
        logging.getLogger("trimesh").setLevel("ERROR")
        self.num_fixed_points = num_fixed_points
        self.default_prismatic_value = default_prismatic_value
        self.with_base_link = with_base_link
        self._init_internal_(device, use_cache)

    def _init_internal_(self, device, use_cache):
        self.robot = TorchURDF.load(UR5eRobot.urdf, lazy_load_meshes=True, device=device)
        self.links = [l for l in self.robot.links if len(l.visuals)]
        if use_cache and self._init_from_cache_(device):
            return
        meshes = [trimesh.load(Path(UR5eRobot.urdf).parent / l.visuals[0].geometry.mesh.filename, force="mesh") for l in self.links]
        areas = [mesh.bounding_box_oriented.area for mesh in meshes]
        if self.num_fixed_points is not None:
            num_points = np.round(self.num_fixed_points * np.array(areas) / np.sum(areas))
            num_points[0] += self.num_fixed_points - np.sum(num_points)
            assert np.sum(num_points) == self.num_fixed_points
        else:
            num_points = np.round(4096 * np.array(areas) / np.sum(areas))
        self.points = {}
        for ii in range(len(meshes)):
            pc = trimesh.sample.sample_surface(meshes[ii], int(num_points[ii]))[0]
            self.points[self.links[ii].name] = torch.as_tensor(pc, device=device).unsqueeze(0)
        # If we made it all the way here with the use_cache flag set,
        # Then we should be creating new cache files locally.
        if use_cache:
            points_to_save = {k: tensor.squeeze(0).cpu().numpy() for k, tensor in self.points.items()}
            file_name = self._get_cache_file_name_()
            print(f'Saving new file to cache: {file_name}')
            np.save(file_name, points_to_save)

    def _get_cache_file_name_(self):
        if self.num_fixed_points is not None:
            return (UR5eRobot.pointcloud_cache / f'fixed_point_cloud_{self.num_fixed_points}.npy')
        else:
            return UR5eRobot.pointcloud_cache / "full_point_cloud.npy"

    def _init_from_cache_(self, device, frame=None):
        raise NotImplementedError

    def end_effector_pose(self, config):
        raise NotImplementedError

    def sample_end_effector(self, poses, num_points, gripper_width=None, frame=None):
        raise NotImplementedError

    def sample(self, config, num_points=None):
        '''
        Samples points from the surface of the robot by calling FK.

        Parameters
        ----------
        config: Tensor of length (, M) or (N, M) where M is the number of
            actuated joints.
        num_points: Number of points desired.

        Returns
        -------
        N x num points x 3 pointcloud of robot points
        '''
        # One should be None.
        assert bool(self.num_fixed_points is None) ^ bool(num_points is None)
        if config.ndim == 1:
            config = config.unsqueeze(0)
        # Gripper joints are all fixed for the UR5 Robot. So don't need to repeat what is done for Franka.
        cfg = config
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())
        assert len(self.links) == len(values),'Should have the same length.'
        fk_transforms = {}
        fk_points = []
        for idx, l in enumerate(self.links):
            if l.name == "base_link" and not self.with_base_link:
                continue
            fk_transforms[l.name] = values[idx]
            pc = transform_pointcloud(self.points[l.name].float().repeat((fk_transforms[l.name].shape[0], 1, 1)),
                                      fk_transforms[l.name], in_place=True)
            fk_points.append(pc)
        pc = torch.cat(fk_points, dim=1)
        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]
