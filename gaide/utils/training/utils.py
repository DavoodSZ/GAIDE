#!/usr/bin/env python3

import numpy as np
import torch
import open3d as o3d


def visualize_pcd(pcd, colors=None, idx=0):
    '''
    Visualizes PCD for debugging
    '''
    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()
    if pcd.ndim == 3:
        pcd = pcd[idx]
    assert pcd.shape[1] == 3, "PCD must be Nx3"

    geo = o3d.geometry.PointCloud()
    geo.points = o3d.utility.Vector3dVector(pcd.astype(np.float64))
    if colors is not None:
        geo.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))

    o3d.visualization.draw_geometries([geo])

def to_o3d_pcd(pcd, color=None, idx=0):
    '''
    Torch.Tensor --> Open3D PointCloud
    '''
    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()
    if pcd.ndim == 3:
        pcd = pcd[idx]
    assert pcd.shape[1] == 3, "PCD must be Nx3"

    geo = o3d.geometry.PointCloud()
    geo.points = o3d.utility.Vector3dVector(pcd.astype(np.float64))
    if color is not None:
        if isinstance(color, torch.Tensor):
            color = color.detach().cpu().numpy()
        color = np.array(color, dtype=np.float64)
        if color.ndim == 1:
            color = np.tile(color, (pcd.shape[0], 1))
        geo.colors = o3d.utility.Vector3dVector(np.clip(color, 0, 1))

    return geo

def visualize_multiple_pcds(pointclouds, colors=None):
    if colors is None:
        colors = [None] * len(pointclouds)
    geoms = [to_o3d_pcd(pcd, c) for pcd, c in zip(pointclouds, colors)]
    o3d.visualization.draw_geometries(geoms)

def save_pcd_visualization(pointclouds, colors=None, filename="pcd.png"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # offscreen
    geoms = []
    if colors is None:
        colors = [None] * len(pointclouds)
    for pcd, c in zip(pointclouds, colors):
        geoms.append(to_o3d_pcd(pcd, c))
        vis.add_geometry(geoms[-1])

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename)  # saves to file
    vis.destroy_window()
    print(f"Saved visualization to {filename}")
