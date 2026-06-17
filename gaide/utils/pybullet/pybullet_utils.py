#!/usr/bin/env python3

import numpy as np
import pybullet as pyb
import pybullet_utils.bullet_client as bc
import pybullet_data
import time

from .bullet_named_utils import (
    getJointInfo,
    getJointStates,
    getLinkState
)

class LaunchSim:
    '''
    The class for launching the PyBullet simulation environment.
    '''
    def __init__(self, connection_type : str = 'gui') -> None:
        self.connection_type = connection_type

        # Create the server
        self. client_obj = self._get_pybullet_server()

        # Set the config of simulation space
        self._set_simulation_env()

    def _get_pybullet_server(self)-> bc.BulletClient:
        '''
        Creates the server.

        Returns:
            A bc.BulletClient
        '''
        if self.connection_type == 'direct':
            p = bc.BulletClient(pyb.DIRECT)
        elif self.connection_type == 'gui':
            p = bc.BulletClient(pyb.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0') # While background.
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, lightPosition=[0, 0, 0.1]) # Disables GUI components and specifies the light position.
            p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=50, cameraPitch=-30, cameraTargetPosition=[0, 0, 0]) # Camera for how the scene is shown.
        else:
            raise TypeError(f'Cannot provide a server based on the provided server type, {self.connection_type}')
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        return p

    def _set_simulation_env(self)->None:
        '''
        Sets the environment for the current client object.
        '''
        self.client_obj.resetSimulation()
        self.client_obj.setGravity(0, 0, -9.81)

    def load_plane(self, origin, plane_collision_distance=0.01)->None:
        '''
        Loads the base plane in the simulation environment.

        NOTE: This is just for appearances. At the end, blender is better for visualization.
        '''
        origin[2] = -origin[2] - plane_collision_distance
        plane = self.client_obj.loadURDF("plane.urdf", origin)
        return plane

    def add_primitive(self, urdf_dir: str, origin: list, useFixedBase=True)->None:
        '''
        Adds the promitive to the scene

        Args:
            urdf_dir (str): The path to the primitive's URDF for loading the primitive from.
            origin (list): The origin of the primitive with respect to the world-frame (position, orientation (rpy)).
        '''
        primitive = self.client_obj.loadURDF(
            urdf_dir,
            basePosition=origin[0],
            baseOrientation=pyb.getQuaternionFromEuler(origin[1]),
            useFixedBase=useFixedBase,
            flags=pyb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
        )
        return primitive

    def get_bounding_box(self, body_id, link_id=-1):
        '''
        Retrieves the Axis-Aligned Bounding Box (AABB) for a specified object or link.

        Args:
            bullet_id: the ID of the object within the PyBullet Simulator.
            link_id: The specific link ID if we have a multi-linked object.

        Returns:
            tuple (minimum corner, maximum corner): The smallest and largest Z, Y, and Z values of the bounding box.
        '''
        min_aabb, max_aabb = self.client_obj.getAABB(bodyUniqueId=body_id, linkIndex=link_id)
        return np.array(min_aabb), np.array(max_aabb)


    def collision_checking(
        self, scene: list, primitive: int, primitive_position: list=None, primitive_name:str=None, table_aabb=None, primitive_info=None, push_distance=0.2, max_attempts=100):
        '''
        Adds the newly generated primitive to the exisitng scene.

        Args:
            scene (list): A list containing the IDs of the object on the table scene
                NOTE: Table and Plane and potentially the robot are excluded. TODO: Robot will be added later.
            primitive (int): The ID of the newly sampled primitive to be added to the scene.
            table_aabb tuple (minimum corner, maximum corner): The smallest and largest Z, Y, and Z values of the bounding box.
            primitive_info: To infer the size of the base for proper placing on the table.
            max_attempts (int): Number of iterations for moving the object out of collision.
        '''
        # Get the primitive offset for effective placement
        if primitive_info is not None:
            primitive_offset = np.min([primitive_info[primitive_name]["base_size"][0], primitive_info[primitive_name]["base_size"][1]])
        else:
            primitive_offset = 0

        # First primitive is always collision free.
        if len(scene) == 0:
            pos, ori = self.client_obj.getBasePositionAndOrientation(primitive)
            primitive_position = np.array(pos)
            if table_aabb is not None:
                min_aabb, max_aabb = table_aabb
                primitive_position[:2] = np.clip(primitive_position[:2], min_aabb[:2] + primitive_offset, max_aabb[:2] - primitive_offset)
            primitive_position = list(primitive_position)
            self.client_obj.resetBasePositionAndOrientation(primitive, primitive_position, ori)
            self.client_obj.stepSimulation()
            return True, primitive_position

        # Add collision-free in the scene.
        for _ in range(max_attempts):
            contacts = []
            closest_points = []

            # Bring object within the bounds
            pos, ori = self.client_obj.getBasePositionAndOrientation(primitive)
            pos = np.array(pos)
            if table_aabb is not None:
                min_aabb, max_aabb = table_aabb
                pos[:2] = np.clip(pos[:2], min_aabb[:2] + primitive_offset, max_aabb[:2] - primitive_offset)
            pos = list(pos)
            self.client_obj.resetBasePositionAndOrientation(primitive, pos, ori)
            self.client_obj.stepSimulation()


            for obj in scene:
                contact_points = self.client_obj.getClosestPoints(bodyA=primitive, bodyB=obj, distance=0.01)
                if len(contact_points) > 0:
                    contact_point = contact_points[0]
                    normal = contact_point[7]
                    normal = np.array([normal[0], normal[1], normal[2]])
                    contacts.append(normal)

            if len(contacts) == 0:
                print(f'Object {primitive} placed successfully, collision-free')
                return True, pos                                                   # No collision detected, success

            summed_normal = np.array([0.0, 0.0, 0.0])

            for normal in contacts:
                summed_normal += normal

            # TODO: Check how the algorithm works with/without this normalization.
            summed_normal = summed_normal / np.linalg.norm(summed_normal)

            # Get current position
            pos, ori = self.client_obj.getBasePositionAndOrientation(primitive)

            # Compute new position by moving along the summed normal (TODO: clip to be on the table)
            new_pos = np.array(pos) + push_distance * summed_normal
            new_pos[2] = pos[2]

            # Bound to the table limits
            if table_aabb is not None:
                min_aabb, max_aabb = table_aabb
                new_pos[:2] = np.clip(new_pos[:2], min_aabb[:2] + primitive_offset, max_aabb[:2] - primitive_offset)
            primitive_position = list(new_pos)

            # Apply new position
            self.client_obj.resetBasePositionAndOrientation(primitive, primitive_position, ori)
            self.client_obj.stepSimulation()

            for obj in scene:
                contact_points = self.client_obj.getClosestPoints(bodyA=primitive, bodyB=obj, distance=0.01)
                if len(contact_points) > 0:
                    contact_point = contact_points[0]
                    closest_points.append(contact_point[8])

            if len(closest_points) == 0:
                return True, primitive_position
            else:
                min_dist = np.min(np.array(closest_points))
                if np.isclose(min_dist, 0.0):
                    return True, primitive_position

        # Remove the object from the scene
        self.remove_object(primitive)
        print(f'Could not resolve collision for object {primitive} after max attempts')

        return False, None

    @property
    def scene_num_bodies(self):
        '''
        Returns the number of bodies within the workspace.
        '''
        num_bodies = self.client_obj.getNumBodies()
        print(f'Number of objects in the scene: {num_bodies}')

    def remove_object(self, idx: int):
        '''
        Removes the object from the scene.

        Args:
            idx (int): The object ID to be removed from the scene.
        '''
        self.client_obj.removeBody(idx)

    def control_joints(
        self, uid, joints, positions, velocity=0.1, acceleration=0.2
    ):
        '''
        Position control of robot joints.
        '''
        return pyb.setJointMotorControlArray(
            uid, joints, pyb.POSITION_CONTROL, targetPositions=positions, positionGains=[velocity]*len(joints)
        )

    def step(self):
        '''
        Runs the simulation
        '''
        pyb.stepSimulation()

    def stop_simulation(self):
        '''
        Stops the simulation.
        '''
        time.sleep(2)
        pyb.disconnect()

    def get_base(self, uid):
        '''
        Returns the base position and orientation of the given Bullet object.
        '''
        # get position and orientation
        position, orientation = self.client_obj.getBasePositionAndOrientation(uid)

        # Convert quaternion to euler
        orientation = self.client_obj.getEulerFromQuaternion(orientation)

        origin = [position, orientation]
        return origin

    def reset_simulation(self):
        '''
        Resets the PyBullet Simulation Environment.
        '''
        self.client_obj.resetSimulation()
        self.client_obj.setGravity(0, 0, -9.81)

    def point_collision_checking(self, primitive_bullet, primitive_points, radius=0.01):
        '''
        utility function for checking whether sampled points are is collision or no.
        '''
        points = []
        for point in primitive_points:
            # Create fake bodies based on sampled points
            collision_shape = pyb.createCollisionShape(pyb.GEOM_SPHERE, radius=radius)
            ghost_body = pyb.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=collision_shape, basePosition=point
            )
            # check collision
            contacts = pyb.getClosestPoints(bodyA=ghost_body, bodyB=primitive_bullet, distance=0)

            if len(contacts) > 0:
                continue
            else:
                points.append(list(point))

        return np.array(points)

    def get_link_pose(self, body, link):
        '''
        Gets the position and orientation of the link within the workspace
        '''
        states = getLinkState(bodyUniqueId=body, linkIndex=link, computeForwardKinematics=True)
        return states

    def marionette(self, body, state, joints, dof: int=6, velocities=None):
        '''
        Borrowed from Robofin
        '''
        if velocities is None:
            velocities = [0.0 for _ in state]
        assert len(state) == len(velocities)

        for i in range(dof):
            self.client_obj.resetJointState(
                body, joints[i], state[i], targetVelocity=velocities[i])

    def in_collision(self, robot_id, obstacles, name: str="ur5e", check_self=True):
        '''
        Collision checks the robot configuration
        '''
        if name == "ur5e":
            # self collision
            collision_mat = np.array([link[8] for link in self.client_obj.getClosestPoints(
                bodyA=robot_id, bodyB=robot_id, distance=2)]).reshape(18, 18)                        # 8x8 without gripper, 18x18 with gripper

            self_contact = np.diag(collision_mat)
            adj_contact_up = np.diag(collision_mat, 1)
            adj_contact_down = np.diag(collision_mat, -1)
            link_offset = np.diag(self_contact) + np.diag(adj_contact_up, k=1) + np.diag(adj_contact_down, k=-1)
            collMat = collision_mat - link_offset

            # Gripper should avoid collision with the body of the robot
            collMat[6:18, 6:18] = 0
            minDist = np.min(collMat)

            if minDist < 0 and not np.isclose(minDist, 1e-6):
                return True

            # Collision with other components of the workspace
            assert isinstance(obstacles, list), 'Obstalces have to be a list.'
            dists = [cp[8] for obs in obstacles for cp in self.client_obj.getClosestPoints(bodyA=obs, bodyB=robot_id, distance=2)]
            minDist = min(dists) if dists else float("inf")
            if minDist < 0 and not np.isclose(minDist, 0.0):
                return True

            return False
        elif name == "panda":
            collision_mat = np.array([link[8] for link in self.client_obj.getClosestPoints(
                bodyA=robot_id, bodyB=robot_id, distance=2)]).reshape(11, 11)

            self_contact = np.diag(collision_mat)
            adj_contact_up = np.diag(collision_mat, 1)
            adj_contact_down = np.diag(collision_mat, 1)
            link_offset = np.diag(self_contact) + np.diag(adj_contact_up, k=1) + np.diag(adj_contact_down, k=-1)
            collMat = collision_mat - link_offset

            # Gripper should avoid collision with the body of the robot
            collMat[6:11, 6:11] = 0
            minDist = np.min(collMat)

            if minDist < 0 and not np.isclose(minDist, 0.0):
                print("robot is in self collision")
                return True

            # Collision with other components of the workspace
            assert isinstance(obstacles, list), 'Obstalces have to be a list.'
            minDist = min((min(link[8] for link in self.client_obj.getClosestPoints(bodyA=obs, bodyB=robot_id, distance=100)) for obs in obstacles))
            if minDist < 0 and not np.isclose(minDist, 0.0):
                return True

            return False
        else:
            raise ValueError(f"The name {name} is not valid. Acceptable names are: ['ur5e', 'panda']")
