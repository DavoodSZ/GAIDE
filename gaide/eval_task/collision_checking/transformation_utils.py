#!/usr/bin/env python3

'''
LINK: https://github.com/fishbotics/geometrout/blob/main/geometrout/transform.py
'''

import numba as nb 
import numpy as np 

@nb.jit(nopython=True, cache=True)
def quaternion_to_matrix(q):
    '''
    NOTE: PyBullet quaternion convenctions is: X-Y-Z-W
    '''
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ],
    )

@nb.jit(nopython=True, cache=True)
def quaternion_to_rpy(q):
    matrix = quaternion_to_matrix(q)
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    pitch = np.arctan2(-matrix[2, 0], np.sqrt(matrix[2, 1] ** 2 + matrix[2, 2] ** 2))
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    return roll, pitch, yaw

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

@nb.jit(nopython=True, cache=True)
def normalize(v):
    return v / np.linalg.norm(v)

@nb.jit(nopython=True, cache=True)
def quaternion_inverse(q):
    qinv = np.copy(q)
    qinv[1:] *= -1
    return qinv

@nb.jit(nopython=True, cache=True)
def pose_inverse(pos, q):
    qinv = quaternion_inverse(q)
    posinv = np.dot(-quaternion_to_matrix(qinv), pos)
    return posinv, qinv

@nb.jit(nopython=True, cache=True)
def inverse(pos, q):
    q = normalize(q)
    pos, q = pose_inverse(pos, q)
    return pos, q

@nb.jit(nopython=True, cache=True)
def pose_multiply(pos1, q1, pos2, q2):
    q = rotation_multiply(q1, q2)
    pos = np.dot(quaternion_to_matrix(q1), pos2) + pos1
    return pos, q

@nb.jit(nopython=True, cache=True)
def rotation_multiply(q1, q2):
    '''
    RoboCu Convention
    '''
    x0, y0, z0, w0 = q1
    x1, y1, z1, w1 = q2
    w = w0 * w1 + -x0 * x1 + -y0 * y1 + -z0 * z1
    x = x0 * w1 + w0 * x1 + -z0 * y1 + y0 * z1
    y = y0 * w1 + z0 * x1 + w0 * y1 + -x0 * z1
    z = z0 * w1 + -y0 * x1 + x0 * y1 + w0 * z1
    return np.array([w, x, y, z])

