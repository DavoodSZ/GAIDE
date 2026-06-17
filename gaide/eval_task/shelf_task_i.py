#!/usr/bin/env python3

import os
import json
import numpy as np
import torch
import time
import trimesh
from scipy.spatial.transform import Rotation as R

from gaide.eval_task.planning import Planning
from gaide.eval_task.robot import UR5e

THRESHOLD = 0.05


class ShelfTaskI(Planning):
    """Shelf Task I — single-shelf evaluation."""

    def __init__(
        self,
        device: str,
        num_robot_points: int,
        problem_dir: str,
        ws_dir: str,
        env_dir: str,
        save_dir: str,
        normalization_info_path: str,
        model_path: str = None,
        max_rollout_len: int = 100,
        dist_tol: float = 0.05,
        ang_tol: float = 15.0,
        sanity_check: bool = False,
        debug: bool = False,
    ):
        self.device = device
        self.num_robot_points = num_robot_points
        self.problem_dir = problem_dir
        self.ws_dir = ws_dir
        self.env_dir = env_dir
        self.save_dir = save_dir
        self.normalization_info_path = normalization_info_path
        self.model_path = model_path
        self.max_rollout_len = max_rollout_len
        self.dist_tol = dist_tol
        self.ang_tol = ang_tol
        self.sanity_check = sanity_check
        self.debug = debug

        self.ws_pointcloud_sampler = PointCloudSampler(num_points=num_robot_points, robot_name="ur5e")
        super().__init__()

    def plan(
        self,
        planning_difficulty: str = "free2shelf",
        model=None,
        optim_sample_trajectories: int = 100,
        denoising_steps: int = 1,
    ):
        env_idx = 0
        self.env_info = self.load_env_info(log_dir=os.path.join(self.env_dir, f"env_{env_idx}"))
        os.makedirs(self.save_dir, exist_ok=True)

        planning_configs = self.load_planning_problems(
            ik_planning_problem_dir=os.path.join(self.problem_dir, f"env_{env_idx}")
        )
        start_configs, end_configs = self.batch_clean_planning_problems(
            planning_configs=planning_configs, planning_difficulty=planning_difficulty
        )

        self.table_info = self.load_table_info()
        self.robot = UR5e(simulator=self.simulator, device=self.device, table_info=self.table_info)
        self.world_config = make_world_config(
            ws_dir=self.ws_dir, env_info=self.env_info, table_info=self.table_info, robot=self.robot
        )

        successful_trajs = []
        planning_times = []
        problem_idxs = 0
        success_idxs = 0
        reached_idxs = 0

        for i in range(start_configs.shape[0]):
            start, goal = start_configs[i], end_configs[i]
            start_pos = self.fk.fk(configs=start.unsqueeze(0)).ee_position.clone().detach()
            goal_pos = self.fk.fk(configs=goal.unsqueeze(0)).ee_position.clone().detach()
            if torch.norm(start_pos - goal_pos) < THRESHOLD:
                continue

            problem_idxs += 1
            t_start = time.time()
            reached, collision_free, trajectory = self.plan_network(
                start=start.unsqueeze(0),
                goal=goal.unsqueeze(0),
                model=model,
                optim_sample_trajectories=optim_sample_trajectories,
            )
            if reached:
                reached_idxs += 1
                if collision_free:
                    success_idxs += 1
                    successful_trajs.append(trajectory.detach().cpu())
                    planning_times.append(time.time() - t_start)

            if problem_idxs >= 100:
                break

        print(f"{problem_idxs} problems tried: {success_idxs} successful, {reached_idxs} reached goal.")
        planning_info = {
            "success_rate": success_idxs / max(problem_idxs, 1),
            "reached_goal": reached_idxs / max(problem_idxs, 1),
            "mean_time": float(np.mean(planning_times)) if planning_times else 0.0,
            "std_time": float(np.std(planning_times)) if planning_times else 0.0,
        }
        with open(
            os.path.join(self.save_dir, f"{planning_difficulty}_{optim_sample_trajectories}_trajs.json"), "w"
        ) as f:
            json.dump(planning_info, f, indent=2)

    def batch_clean_planning_problems(self, planning_configs, planning_difficulty: str = "free2shelf"):
        pairs = {
            "free2free": [("table", "table")],
            "free2shelf": [("table", "shelf")],
            "shelf2free": [("shelf", "table")],
            "shelf2shelf": [("shelf", "shelf")],
        }
        if planning_difficulty not in pairs:
            raise ValueError(f"Unknown planning_difficulty: {planning_difficulty}")

        start_configs = None
        end_configs = None
        for src_key, dst_key in pairs[planning_difficulty]:
            for config in planning_configs[src_key]:
                batch = torch.tensor(config, device=self.device, dtype=torch.float32).unsqueeze(0)
                batch = batch.repeat_interleave(planning_configs[dst_key].shape[0], dim=0)
                start_configs = batch if start_configs is None else torch.cat([start_configs, batch], dim=0)
                end = torch.tensor(planning_configs[dst_key], device=self.device, dtype=torch.float32)
                end_configs = end if end_configs is None else torch.cat([end_configs, end], dim=0)

        return start_configs, end_configs

    def load_table_info(self):
        with open(os.path.join(self.ws_dir, "table/dim_info.json"), "r") as f:
            return json.load(f)


class PointCloudSampler:
    def __init__(self, num_points: int = 1024, robot_name: str = "ur5e"):
        self.num_points = num_points

    def sample(self, asset_dir: str, env_info: dict):
        pcd = self._mesh_sample(
            filepath=os.path.join(asset_dir, "table/table_mesh.obj"),
            origin=env_info["table_origin"],
        )
        for obj_key in [k for k in env_info if k not in ("table_origin", "robot_origin")]:
            fp = os.path.join(asset_dir, f"{obj_key}/{obj_key}.obj")
            if os.path.exists(fp):
                pcd = np.concatenate([pcd, self._mesh_sample(fp, env_info[obj_key])], axis=0)
        return pcd

    def _mesh_sample(self, filepath: str, origin: list):
        mesh = trimesh.load(filepath, force="mesh")
        points = np.array(trimesh.sample.sample_surface(mesh, self.num_points)[0])
        Rmat = R.from_euler('xyz', origin[1]).as_matrix()
        return points @ Rmat.T + np.array(origin[0])


def make_world_config(ws_dir: str, env_info: dict, table_info: dict, robot):
    world_config = {"mesh": {}}
    for obj_key, origin in env_info.items():
        if obj_key == "robot_origin":
            continue
        pos = robot.config_in_base(
            points=torch.as_tensor(origin[0], device=robot.device, dtype=torch.float32),
            table_info=table_info,
        ).tolist()
        q = list(R.from_euler('xyz', origin[1]).as_quat()[[3, 0, 1, 2]].astype(float))
        name = "table" if obj_key == "table_origin" else obj_key
        fp = os.path.join(ws_dir, "table/table_mesh.obj" if obj_key == "table_origin" else f"{obj_key}/{obj_key}.obj")
        world_config["mesh"][name] = {"pose": pos + q, "file_path": fp}
    return world_config
