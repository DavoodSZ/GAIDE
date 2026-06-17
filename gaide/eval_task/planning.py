#!/usr/bin/env python3

import os
import json
import pickle
import numpy as np
import torch
import time

from gaide.utils.pybullet.pybullet_utils import LaunchSim
from gaide.eval_task.curobo_fk import UR5eFK, collision_checking
from gaide.utils.training.ur5e_pcd_sampler import UR5ePCDSampler

JOINT_LIMIT = 2 * np.pi


class Planning:
    def __init__(self):
        if self.sanity_check:
            self.simulator = LaunchSim(connection_type="gui")
        else:
            self.simulator = None

        self.fk = UR5eFK(device=self.device)
        self.robot_pcd_sampler = UR5ePCDSampler(device=self.device, num_points=self.num_robot_points)
        self.robot_file = "ur5e_c.yml"

    @torch.no_grad()
    def plan_network(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        model,
        use_optimization: bool = True,
        optim_sample_trajectories: int = 100,
    ):
        delta_scale = load_normalization_info(path=self.normalization_info_path)
        if self.model_path:
            model = load_model(model=model, path=self.model_path)
        model.eval()

        goal_pose = self.fk.fk(configs=goal)

        features = {}
        features["robot_origins"] = torch.tensor(
            self.env_info["robot_origin"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        features["goal_cfgs"] = goal / JOINT_LIMIT
        scene_pcd = self.ws_pointcloud_sampler.sample(asset_dir=self.ws_dir, env_info=self.env_info)
        features["scene_pcds"] = torch.as_tensor(scene_pcd, dtype=torch.float32, device=self.device).unsqueeze(0)

        if use_optimization:
            features["robot_origins"] = features["robot_origins"].repeat(optim_sample_trajectories, 1, 1).contiguous()
            features["goal_cfgs"] = features["goal_cfgs"].repeat(optim_sample_trajectories, 1).contiguous()
            features["scene_pcds"] = features["scene_pcds"].repeat(optim_sample_trajectories, 1, 1).contiguous()
            start = start.repeat(optim_sample_trajectories, 1)
            goal = goal.repeat(optim_sample_trajectories, 1)

        reached = False
        collision_free = False
        trajectory = [start]
        qt = start
        t_start = time.time()
        for i in range(self.max_rollout_len):
            features["current_pcds"] = self.robot_pcd_sampler.sample(
                robot_origins=features["robot_origins"], robot_configs=qt
            ).contiguous()
            features["current_cfgs"] = qt / JOINT_LIMIT
            delta_chunk = model(features=features)
            delta_chunk = unnormalize_output(configs=delta_chunk, delta_scale=delta_scale)
            for j in range(delta_chunk.shape[1]):
                delta_q = delta_chunk[:, j, :]
                qt = qt + delta_q
                trajectory.append(qt)

        t_end = time.time()
        print(f"policy rollout time: {t_end - t_start:.3f}s")

        output_traj = torch.stack(trajectory, dim=1)
        num_traj, traj_length, action_dim = output_traj.shape
        flat_trajs = output_traj.reshape(num_traj * traj_length, action_dim)
        traj_poses = self.fk.fk(configs=flat_trajs)

        ee_pos = traj_poses.ee_position.reshape(num_traj, traj_length, -1)
        ee_quat = traj_poses.ee_quaternion.reshape(num_traj, traj_length, -1)

        goal_pos = goal_pose.ee_position.squeeze(0)
        goal_quat = goal_pose.ee_quaternion.squeeze(0)

        pose_err = torch.norm(ee_pos - goal_pos[None, None, :], dim=-1)
        goal_q = goal_quat[None, None, :].expand(num_traj, traj_length, 4).contiguous()
        ang_err = torch.rad2deg(quaternion_to_radians(quat_mul(ee_quat, quat_conjugate(goal_q))))

        valid = (pose_err < self.dist_tol) & (ang_err < self.ang_tol)
        success_mask = valid.any(dim=1)
        success_ids = torch.nonzero(success_mask, as_tuple=False).squeeze(-1)

        if success_ids.numel() == 0:
            return reached, collision_free, None

        reached = True
        first_idx = valid.float().argmax(dim=1)
        first_idx_success = first_idx[success_ids]

        for k in range(success_ids.numel()):
            tid = int(success_ids[k].item())
            t_end_idx = int(first_idx_success[k].item())
            traj_k = output_traj[tid, :t_end_idx + 1, :]
            in_collision = self.check_collision(traj=traj_k)
            if not in_collision:
                collision_free = True
                return reached, collision_free, traj_k

        return reached, collision_free, None

    def check_collision(self, traj: torch.Tensor) -> bool:
        return collision_checking(
            configs=traj,
            robot_file=self.robot_file,
            world_config=self.world_config,
        )

    def batch_clean_planning_problems(self):
        pass

    def load_planning_problems(self, ik_planning_problem_dir: str):
        with open(os.path.join(ik_planning_problem_dir, "config_planning_problems.pkl"), "rb") as f:
            planning_configs = pickle.load(f)
        return planning_configs

    def load_env_info(self, log_dir: str):
        with open(os.path.join(log_dir, "info.json"), "r") as f:
            env_info = json.load(f)
        return env_info


# ========== Planning Utils ==========

def load_normalization_info(path: str):
    return torch.load(path, weights_only=False, map_location="cpu")


def load_model(model, path: str):
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return model


def unnormalize_output(configs: torch.Tensor, delta_scale: dict) -> torch.Tensor:
    return configs * delta_scale["std"].to(configs.device) + delta_scale["mean"].to(configs.device)


@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


@torch.jit.script
def quaternion_to_radians(q: torch.Tensor) -> torch.Tensor:
    w = q[..., 0]
    xyz = q[..., 1:]
    theta = 2.0 * torch.atan2(torch.linalg.norm(xyz, dim=-1), w)
    wrapped_angle = (theta + torch.pi) % (2.0 * torch.pi)
    return torch.where(torch.abs(wrapped_angle) < 1e-12, torch.pi, wrapped_angle - torch.pi)
