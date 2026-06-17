#!/usr/bin/env python3

'''
Wrapper for pybullet methods that return large tuples.
This sub-module returns the same informat5ion in a named tuple, so the fields can 
be easily indentified.
'''

from collections import namedtuple
import pybullet as pyb 

# Fields name are camel case to match PyBullet convention
DynamicsInfo = namedtuple(
    "DynamicsInfo",
    [
        "mass",
        "lateralFriction",
        "localInertiaDiagonal",
        "localInerialPos",  # TODO fix this name in a future major release
        "localInertialOrn",
        "restitution",
        "rollingFriction",
        "spinningFriction",
        "contactDamping",
        "contactStiffness",
        "bodyType",
        "collisionMargin",
    ],
)

ContactPoint = namedtuple(
    "ContactPoint",
    [
        "contactFlag",
        "bodyUniqueIdA",
        "bodyUniqueIdB",
        "linkIndexA",
        "linkIndexB",
        "positionOnA",
        "positionOnB",
        "contactNormalOnB",
        "contactDistance",
        "normalForce",
        "lateralFriction1",
        "lateralFrictionDir1",
        "lateralFriction2",
        "lateralFrictionDir2",
    ],
)

ConstraintInfo = namedtuple(
    "ConstraintInfo",
    [
        "parentBodyUniqueId",
        "parentJointIndex",
        "childBodyUniqueId",
        "childLinkIndex",
        "constraintType",
        "jointAxis",
        "jointPivotInParent",
        "jointPivotInChild",
        "jointFrameOrientationParent",
        "jointFrameOrientationChild",
        "maxAppliedForce",
        "gearRatio",
        "gearAuxLink",
        "relativePositionTarget",
        "erp",
    ],
)

JointInfo = namedtuple(
    "JointInfo",
    [
        "jointIndex",
        "jointName",
        "jointType",
        "qIndex",
        "uIndex",
        "flags",
        "jointDamping",
        "jointFriction",
        "jointLowerLimit",
        "jointUpperLimit",
        "jointMaxForce",
        "jointMaxVelocity",
        "linkName",
        "jointAxis",
        "parentFramePos",
        "parentFrameOrn",
        "parentIndex",
    ],
)

JointState = namedtuple(
    "JointState",
    [
        "jointPosition",
        "jointVelocity",
        "jointReactionForces",
        "appliedJointMotorTorque",
    ],
)

LinkState = namedtuple(
    "LinkState",
    [
        "linkWorldPosition",
        "linkWorldOrientation",
        "localInertialFramePosition",
        "localInertialFrameOrientation",
        "worldLinkFramePosition",
        "worldLinkFrameOrientation",
        "worldLinkLinearVelocity",
        "worldLinkAngularVelocity",
    ],
)

def getDynamicsInfo(bodyUniqueId, linkIndex, physicsClientId=0):
    '''
    Get information about the dynamic properties of a link

    Args:
        bodyUniqueID: The UID of the body.
        linkIndex: The index of the link on the body.
        physicsClientId: The UID of the physics server to use.

    Returns:
        DynamicsInfo
    '''
    return DynamicsInfo(*pyb.getDynamicsInfo(bodyUniqueId=bodyUniqueId, linkIndex=linkIndex, physicsClientId=physicsClientId))

def getContactPoints(bodyA=-1, bodyB=-1, linkIndexA=-2, linkIndexB=-2, physicsClientId=0):
    '''
    Get the contact points between two bodies or specific links.

    Args:
        bodyA: The UID of the first body.
        bodyB: The UID of the second body.
        linkIndexA: The index of the link on the first body (optional). If not provided, checks all links on the body.
        linkIndexB: The index of the link on the second body (optional). If not provided, checks all links on the body.
        physicsClientID: The UID of the physics server to use.

    Returns:
        A list of ContactPoint
    '''

    points_raw = pyb.getContactPoints(bodyA=bodyA, bodyB=bodyB, linkIndexA=linkIndexA, linkIndexB=linkIndexB, physicsClientId=physicsClientId)
    return [ContactPoint(*point) for point in points_raw]

def getClosestPoints(bodyA, bodyB, distance, linkIndexA=-2, linkIndexB=-2, physicsClientId=0):
    '''
    Get the closest points between two bodies or specific links.

    Args:
        bodyA: The UID of the first body.
        bodyB: The UID of the second body.
        linkIndexA: The index of the link on the first body (optional). If not provided, checks all links on the body.
        linkIndexB: The index of the link on the second body (optional). If not provided, checks all links on the body.
        physicsClientId: The UID of the physics server to use.

    Returns:
        A list of ContactPoint representing the closest points. Normal force is always zero.
    '''

    points_raw = pyb.getClosestPoints(bodyA=bodyA, bodyB=bodyB, linkIndexA=linkIndexA, linkIndexB=linkIndexB, physicsClientId=physicsClientId)
    return [ContactPoint(*point) for point in points_raw]

def getConstraintInfo(constraintUniqueId, physicsClientID=0):
    '''
    Get information about a constraint.

    args: 
        constraintUniqueId: UID of the constraint
        physicsClientId: UID of the physics server to use.

    Returns:
        ConstraintInfo
    '''

    return ConstraintInfo(*pyb.getConstraintInfo(constraintUniqueId=constraintUniqueId, physicsClientId=physicsClientId))

def getJointInfo(bodyUniqueId, jointIndex, physicsClientId=0, decode=None):
    '''
    Get information about a joint.
    The one difference from the PyBullet API is the addition of optional decode argument.
    If decode is not None, then it is used to decode the strings of bytes returned by the PyBullet API for the jointName and linkName fields.

    Args:
        bodyUniqueId: UID of the body with the joint.
        jointIndex: Index of the joint on the body.
        physicsClientId: The UID of the physics server to use.
        decode: The encoding to use to convert the bytes fields to strings.

    Returns:
    JointInfo
    '''

    info = JointInfo(*pyb.getJointInfo(bodyUniqueId=bodyUniqueId, jointIndex=jointIndex, physicsClientId=physicsClientId))
    if decode is not None:
        info = info._replace(jointName=info.jointName.decode(decode), linkName=info.linkName.decode(decode))
    return info

def getJointState(bodyUniqueId, jointIndex, physicsClientId=0):
    '''
    Get the state of a joint

    Args:
        bodyUniqueId: UID of the body with the joint.
        jointIndex: Index of the joint on the body.
        physicsClientId: The UID of the physics server to use.

    Returns:
        JointState
    '''

    return JointState(*pyb.getJointState(bodyUniqueId=bodyUniqueId, jointIndex=jointIndex, physicsClientId=physicsClientId))

def getJointStates(bodyUniqueId, jointIndices, physicsClientId=0):
    '''
    Get the state of multiple joints at once.

    Args:
        bodyUniqueid: UID of the body with the joint.
        jointIndices: List of indices of joints on the body.
        physicsClientId: The UID of the physics server to use.

    Returns:
        A list of jointState
    '''

    states_raw = pyb.getJointStates(bodyUniqueId=bodyUniqueId, jointIndices=jointIndices, physicsClientId=physicsClientId)
    return [JointState(*state) for state in states_raw]

def getLinkState(bodyUniqueId, linkIndex, computeLinkVelocity=False, computeForwardKinematics=False, physicsClientId=0):
    '''
    Get the state of a link.

    One difference from the pyBullet API is that if ``computeLinkVelocity=False``, we still return the ``worldLinkLinearVelocity`` and 
    ``worldLinkAngularVelocity`` fields, but they are set to ``None``. PyBullet just doesn't return the fields, such that the tuple has
    6 fields rather than 8.

    Args: 
        bodyUniqueId: UID of the body with the link.
        linkIndex: Index of the link on the body. Note that the link index corresponds to the index of its parent joint.
        computeLinkVelocity (bool): If ``True``, compute the velocity of the link's CoM in the world frame. Otherwise, these fields are set to None.
        computeForwardKinematics (bool): If True, update the position and orientation of the link in the worldframe. otherwise, the position and
        orientation will not be up to date if the simulation has been stepped.
        physicsClientId: The UID of the physics.
    
    Returns:
        LinkState
    '''

    state_raw = pyb.getLinkState(bodyUniqueId=bodyUniqueId, linkIndex=linkIndex, computeLinkVelocity=computeLinkVelocity, computeForwardKinematics=computeForwardKinematics,
                                physicsClientId=physicsClientId)
    # pybullet just returns a smaller tuple when link velocity is not computed;
    # we instead keep it the same size but set velocity fields to None
    if not computeLinkVelocity:
        state_raw = state_raw + (None, None)
    return LinkState(*state_raw)

def getLinkStates(bodyUniqueId, linkIndices, computeLinkVelocity=False, computeForwardKinematics=False, physicsClientId=0):
    '''
    Get the state of multiple links at once.

    One difference from the pyBullet API is that if computeLinkVelocity=False, we still return the worldLinkLinearVelocity and 
    worldLinkAngularVelocity fields, but they are set to None. PyBullet just doesn't return the fields, such that the tuple has 
    6 fields rather than 8.

    Args:
        bodyUniqueId: UID of the body with the link.
        linkIndices: List of indices of the link on the body. Note that the link index corresponds to the index of its parent joint.
        computeLinkVelocity: If True, compute the velocity of the link's CoM in the world frame. otherwise, these fields are set to None.
        computeForwardKinematics: If True, update the position and orientation of the link in the world frame. Otherwise, the position and 
        orientation will not be up to date if the simulation has been stepped.
        physicsClientId: The UID of the physics server to use.

    Returns:
        List of LinkState
    '''

    states_raw = pyb.getLinkStates(bodyUniqueId=bodyUniqueId, linkIndices=linkIndices, computeLinkVelocity=computeLinkVelocity, computeForwardKinematics=computeForwardKinematics,
                                   physicsClientId=physicsClientId)
    if not computeLinkVelocity:
        states_raw = [s+(None, None) for s in states_raw]
    return [LinkState(*state) for state in states_raw]

