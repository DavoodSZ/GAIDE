#!/usr/bin/env python3

#!/usr/bin/env python3

import os
import torch
import numba as nb 
import numpy as np 

def make_world_config(ws_dir: str, primitive_locations: dict, primitive_nums: dict, primitive_dimensions: dict, robot, gen_mode: str="neuralmp"):
    world_config = {"mesh": {}}

    for key in primitive_locations.keys():
        if key == "front_table":
            dim_key = "table_1"
            source_dir = os.path.join(ws_dir, dim_key)
            table_pos = primitive_locations[key]["location"][0]
            table_pos = robot.config_in_base(points=torch.as_tensor(
                table_pos, 
                device=robot.device, 
                dtype=torch.float32), 
                ws_info=primitive_dimensions, 
                gen_mode=gen_mode).tolist()
            table_quat = [float(x) for x in quaternion_from_rpy(*primitive_locations[key]["location"][0])]
            table_origin = table_pos + table_quat
            table_dir = os.path.join(source_dir, "table")
            table_obj = os.path.join(table_dir, 'table_0_mesh.obj')
            world_config["mesh"][key] = {
                "pose": table_origin,
                "file_path": table_obj
            }
            for key1, value1 in primitive_nums[key].items():
                for i in range(value1):
                    primitive_pos = primitive_locations[key]["primitives"][f"{key1}_{i}"][0]
                    primitive_pos = robot.config_in_base(
                        points=torch.as_tensor(primitive_pos, device=robot.device, dtype=torch.float32),
                        ws_info=primitive_dimensions,
                        gen_mode=gen_mode).tolist()
                    primitive_quat = [float(x) for x in quaternion_from_rpy(*primitive_locations[key]["primitives"][f"{key1}_{i}"][1])]
                    origin = primitive_pos + primitive_quat
                    primitive_dir = os.path.join(source_dir, f"{key1}")
                    primitive_obj = os.path.join(primitive_dir, f"{key1}_{i}_mesh.obj")
                    world_config["mesh"][f"{key}_{key1}_{i}"] = {
                        "pose": origin,
                        "file_path": primitive_obj
                    }
        elif key == "left_table":
            dim_key = "table_2"
            source_dir = os.path.join(ws_dir, dim_key)
            table_pos = primitive_locations[key]["location"][0]
            table_pos = robot.config_in_base(points=torch.as_tensor(
                table_pos, 
                device=robot.device, 
                dtype=torch.float32), 
                ws_info=primitive_dimensions, 
                gen_mode=gen_mode).tolist()
            table_quat = [float(x) for x in quaternion_from_rpy(*primitive_locations[key]["location"][0])]
            table_origin = table_pos + table_quat
            table_dir = os.path.join(source_dir, "table")
            table_obj = os.path.join(table_dir, 'table_0_mesh.obj')
            world_config["mesh"][key] = {
                "pose": table_origin,
                "file_path": table_obj
            }
            for key1, value1 in primitive_nums[key].items():
                for i in range(value1):
                    primitive_pos = primitive_locations[key]["primitives"][f"{key1}_{i}"][0]
                    primitive_pos = robot.config_in_base(
                        points=torch.as_tensor(primitive_pos, device=robot.device, dtype=torch.float32),
                        ws_info=primitive_dimensions,
                        gen_mode=gen_mode).tolist()
                    primitive_quat = [float(x) for x in quaternion_from_rpy(*primitive_locations[key]["primitives"][f"{key1}_{i}"][1])]
                    origin = primitive_pos + primitive_quat
                    primitive_dir = os.path.join(source_dir, f"{key1}")
                    primitive_obj = os.path.join(primitive_dir, f"{key1}_{i}_mesh.obj")
                    world_config["mesh"][f"{key}_{key1}_{i}"] = {
                        "pose": origin,
                        "file_path": primitive_obj
                    }
        elif key == "right_table":
            if "left_table" in primitive_locations:
                dim_key = "table_3"
            else:
                dim_key = "table_2"
            source_dir = os.path.join(ws_dir, dim_key)
            table_pos = primitive_locations[key]["location"][0]
            table_pos = robot.config_in_base(points=torch.as_tensor(
                table_pos, 
                device=robot.device, 
                dtype=torch.float32), 
                ws_info=primitive_dimensions, 
                gen_mode=gen_mode).tolist()
            table_quat = [float(x) for x in quaternion_from_rpy(*primitive_locations[key]["location"][0])]
            table_origin = table_pos + table_quat
            table_dir = os.path.join(source_dir, "table")
            table_obj = os.path.join(table_dir, 'table_0_mesh.obj')
            world_config["mesh"][key] = {
                "pose": table_origin,
                "file_path": table_obj
            }
            for key1, value1 in primitive_nums[key].items():
                for i in range(value1):
                    primitive_pos = primitive_locations[key]["primitives"][f"{key1}_{i}"][0]
                    primitive_pos = robot.config_in_base(
                        points=torch.as_tensor(primitive_pos, device=robot.device, dtype=torch.float32),
                        ws_info=primitive_dimensions,
                        gen_mode=gen_mode).tolist()
                    primitive_quat = [float(x) for x in quaternion_from_rpy(*primitive_locations[key]["primitives"][f"{key1}_{i}"][1])]
                    origin = primitive_pos + primitive_quat
                    primitive_dir = os.path.join(source_dir, f"{key1}")
                    primitive_obj = os.path.join(primitive_dir, f"{key1}_{i}_mesh.obj")
                    world_config["mesh"][f"{key}_{key1}_{i}"] = {
                        "pose": origin,
                        "file_path": primitive_obj
                    }
        elif key == "back_table":
            dim_key = "table_4"
            source_dir = os.path.join(ws_dir, dim_key)
            table_pos = primitive_locations[key]["location"][0]
            table_pos = robot.config_in_base(points=torch.as_tensor(
                table_pos, 
                device=robot.device, 
                dtype=torch.float32), 
                ws_info=primitive_dimensions, 
                gen_mode=gen_mode).tolist()
            table_quat = [float(x) for x in quaternion_from_rpy(*primitive_locations[key]["location"][0])]
            table_origin = table_pos + table_quat
            table_dir = os.path.join(source_dir, "table")
            table_obj = os.path.join(table_dir, 'table_0_mesh.obj')
            world_config["mesh"][key] = {
                "pose": table_origin,
                "file_path": table_obj
            }
            for key1, value1 in primitive_nums[key].items():
                for i in range(value1):
                    primitive_pos = primitive_locations[key]["primitives"][f"{key1}_{i}"][0]
                    primitive_pos = robot.config_in_base(
                        points=torch.as_tensor(primitive_pos, device=robot.device, dtype=torch.float32),
                        ws_info=primitive_dimensions,
                        gen_mode=gen_mode).tolist()
                    primitive_quat = [float(x) for x in quaternion_from_rpy(*primitive_locations[key]["primitives"][f"{key1}_{i}"][1])]
                    origin = primitive_pos + primitive_quat
                    primitive_dir = os.path.join(source_dir, f"{key1}")
                    primitive_obj = os.path.join(primitive_dir, f"{key1}_{i}_mesh.obj")
                    world_config["mesh"][f"{key}_{key1}_{i}"] = {
                        "pose": origin,
                        "file_path": primitive_obj
                    }
        elif key == "on_ground":
            source_dir = os.path.join(ws_dir, key)
            for key1, value1 in primitive_locations[key].items():
                pos = primitive_locations[key][key1][0]
                pos = robot.config_in_base(
                    points=torch.as_tensor(pos, device=robot.device, dtype=torch.float32),
                    ws_info=primitive_dimensions,
                    gen_mode=gen_mode).tolist()
                quat = [float(x) for x in quaternion_from_rpy(*primitive_locations[key][key1][1])]
                origin = pos + quat
                primitive_dir = os.path.join(source_dir, f"{key1}")
                primitive_obj = os.path.join(primitive_dir, f"{key1}_0_mesh.obj")
                world_config["mesh"][key] = {
                    "pose": origin,
                    "file_path": primitive_obj
                }
    
    return world_config

@nb.jit(nopython=True, cache=True)
def quaternion_trace_method(matrix, rtol=1e-7, atol=1e-7):
    '''
    This code uses a modification of the algorithm described in:
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    which is itself based on the method described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    Altered to work with the column vector convention instead of row vectors
    '''
    assert matrix.shape == (3, 3)
    if not np.allclose(np.dot(matrix, matrix.conj().transpose()),
                       np.eye(3), rtol=rtol, atol=atol, equal_nan=False):
        raise ValueError("Matrix must be orthogonal, i.e., its transpose should be its inverse")

    # Re-implemented `np.isclose` for Numba
    if np.abs(np.linalg.det(matrix) - 1.0) > atol + rtol:
        raise ValueError(
            "Matrix must be special orthogonal i.e. its determinant must be +1.0"
        )
    m = (
        matrix.conj().transpose()
    )  # This method assumes row-vector and postmultiplication of that vector
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = [m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = [m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = [m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = [t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]]

    q = np.array(q).astype("float64")
    q *= 0.5 / np.sqrt(t)
    return q

@nb.jit(nopython=True, cache=True)
def quaternion_from_rpy(r, p, y):
    c3, c2, c1 = np.cos(np.array([r, p, y]))
    s3, s2, s1 = np.sin(np.array([r, p, y]))

    matrix = np.array(
        [
            [c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2)],
            [c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3)],
            [-s2, c2 * s3, c2 * c3],
        ],
        dtype=np.float64,
    )
    q = quaternion_trace_method(matrix)
    q = np.array([q[0], q[1], q[2], q[3]])
    return q