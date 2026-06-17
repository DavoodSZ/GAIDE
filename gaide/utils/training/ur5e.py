#!/usr/bin/env python3
'''
Link: https://github.com/mihdalal/robofin/blob/192c500a801ae938586d49ccde37cbc6c29ae305/robofin/robots.py
'''

from pathlib import Path
from .transform_pointcloud import SE3, SO3
from ikfast_ur5e import get_fk, get_ik
import numpy as np
import os

from gaide import DIRECTORY

class UR5eRobot:
    # TODO: remove this after making this more general.
    JOINT_LIMITS = np.array(
        [
            (-2*np.pi, 2*np.pi),
            (-2*np.pi, 2*np.pi),
            (-2*np.pi, 2*np.pi),
            (-2*np.pi, 2*np.pi),
            (-2*np.pi, 2*np.pi),
            (-2*np.pi, 2*np.pi),
        ]
    )
    VELOCITY_LIMIT = np.array([])     # TBD
    ACCELERATION_LIMIT = np.array([]) # TBD
    DOF = 6
    EFF_LIST = set()                  # TBD
    EFF_T_LIST = {}                   # TBD
    urdf = os.path.join(DIRECTORY, "assets/robots/ur5e/ur5e.urdf")
