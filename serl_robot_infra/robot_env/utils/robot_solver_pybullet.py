# ★ 更新：所有 FK / IK / 末端速度均改为使用 **URDF link frame（原始 link 坐标系）** 而非质心（CoM）获取位姿！请注意 PyBullet 中：getLinkState 返回值 index 4 和 5 才是 link 原始 frame 的 world pose。

"""
pybullet_kinematics.py

Utilities to perform FK, IK and end-effector velocity (Jacobian) calculations in PyBullet.

Features:
- load a URDF and query base vs link-frame vs link CoM poses
- forward kinematics (pose & homogeneous matrix) for any link
- inverse kinematics wrapper that maps results to movable joints
- compute end-effector spatial velocity from joint velocities (Jacobian)
- handy quaternion and transform helpers

Usage: run this file directly or import the `PyBulletKinematics` class in your code.

Note: this code expects PyBullet installed and accessible. It uses numpy heavily.
"""

import time
from typing import List, Tuple, Optional

import numpy as np
import pybullet as p
import pybullet_data
import threading

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x,y,z,w] to 3x3 rotation matrix."""
    x, y, z, w = q
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Zero-norm quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q = q1 * q2 (both [x,y,z,w])."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([x, y, z, w])


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse of unit quaternion [x,y,z,w] is [-x,-y,-z,w]."""
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def homogeneous_from_pos_quat(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Return 4x4 homogeneous matrix from position and quaternion."""
    R = quat_to_rot(quat)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


class RobotSolver:
    def __init__(self, urdf_path: str, base_position=(0, 0, 0), use_gui: bool = False):
        self.urdf_path = urdf_path
        self.base_position = base_position
        self.use_gui = use_gui
        if self.use_gui:
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # load URDF
        self.robot_id = p.loadURDF(self.urdf_path, base_position, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_info = [p.getJointInfo(self.robot_id, i) for i in range(self.num_joints)]
        # pick only movable joints
        self.movable_joints = [i for i in range(self.num_joints) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
        print(f"{self.movable_joints=}")
        self._lock = threading.Lock()

    def disconnect(self):
        p.disconnect(self.cid)

    # ----------------------------- pose helpers -----------------------------
    def get_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return base (root) pose in world frame: (pos, quat).
        Use p.getBasePositionAndOrientation for the base.
        """
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        return np.array(pos), np.array(orn)

    def get_link_com_pose(self, link_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return link's center-of-mass pose in world frame (linkWorldPosition, linkWorldOrientation).

        getLinkState(...) returns a tuple where:
          0: linkWorldPosition (CoM【center of Mass】 position in world)
          1: linkWorldOrientation (CoM orientation in world)
          2: localInertialPosition
          3: localInertialOrientation
          4: worldLinkFramePosition (the URDF 'link frame' origin in world)
          5: worldLinkFrameOrientation
          6: worldLinkLinearVelocity
          7: worldLinkAngularVelocity
        """
        ls = p.getLinkState(self.robot_id, link_index)
        pos = np.array(ls[0])
        quat = np.array(ls[1])
        return pos, quat

    def get_link_frame_pose(self, link_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the URDF 'link frame' pose in world (worldLinkFramePosition/orientation)."""
        ls = p.getLinkState(self.robot_id, link_index)
        pos = np.array(ls[4])
        quat = np.array(ls[5])
        return pos, quat

    # ----------------------------- forward kinematics -----------------------------
    def forward_kinematics(self, joint_positions: List[float], link_index: int=18) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Set the robot to `joint_positions` (for movable joints only) and return
        the link pose in world and the 4x4 homogeneous transform.

        Returns: (pos_world, quat_world, T_world)
        """
        if len(joint_positions) != len(self.movable_joints):
            raise ValueError("joint_positions must have length equal to the number of movable joints")
        # set joint states (use resetJointState to instantly set configuration)
        with self._lock:
            for jid, q in zip(self.movable_joints, joint_positions):
                p.resetJointState(self.robot_id, jid, q)
            # step simulation so internal kinematics update (not strictly necessary for resetJointState but safe)
            # p.stepSimulation()

            pos, quat = self.get_link_frame_pose(link_index)
            T = homogeneous_from_pos_quat(pos, quat)
            return pos, quat, T

    # ----------------------------- inverse kinematics -----------------------------
    def inverse_kinematics(self, target_pos: List[float], target_quat: Optional[List[float]], link_index: int=18,
                           rest_poses: Optional[List[float]] = None, max_iters: int = 1) -> List[float]:
        """Compute IK using PyBullet's calculateInverseKinematics and map results to movable joints.

        - `target_quat` should be [x,y,z,w] or None (if None, only position IK)
        - returns list of joint angles for movable_joints
        """
        # run the IK solver
        with self._lock:
            if rest_poses is None:
                if target_quat is None:
                    sol = p.calculateInverseKinematics(self.robot_id, link_index, target_pos)
                else:
                    sol = p.calculateInverseKinematics(self.robot_id, link_index, target_pos, target_quat)
            else:
                if target_quat is None:
                    sol = p.calculateInverseKinematics(self.robot_id, link_index, target_pos, restPoses=rest_poses)
                else:
                    sol = p.calculateInverseKinematics(self.robot_id, link_index, target_pos, target_quat, restPoses=rest_poses)
            return sol
        # `sol` is a list for all joints (including fixed); we extract values for movable_joints
        # sol_list = list(sol)
        # result = [sol_list[j-self.movable_joints[0]] for j in self.movable_joints]

        # # optionally apply and iterate a few times (useful if you want to settle with PD motors)
        # for it in range(max_iters):
        #     for jid, q in zip(self.movable_joints, result):
        #         p.resetJointState(self.robot_id, jid, q)
        #     p.stepSimulation()
        # return sol

    # ----------------------------- Jacobian / velocity -----------------------------
    def ee_velocity_from_joint_vel(self, joint_positions: List[float], joint_velocities: List[float],
                                   link_index: int, local_point=[0, 0, 0], in_frame: str = 'world') -> np.ndarray:
        """Compute end-effector spatial velocity given joint positions and velocities.

        - joint_positions, joint_velocities: lists aligned with movable_joints
        - link_index: the link to evaluate Jacobian at (EE link)
        - local_point: 3-vector in link-frame where velocity is evaluated (default link origin)
        - in_frame: 'world' (default) or 'base' or 'ee' : which frame the returned twist is expressed in

        Returns 6x1 vector [v; omega] stacked (linear; angular) in the chosen frame.
        """
        if len(joint_positions) != len(self.movable_joints) or len(joint_velocities) != len(self.movable_joints):
            raise ValueError("joint_positions and joint_velocities must match movable_joints length")

        # set joint positions
        for jid, q in zip(self.movable_joints, joint_positions):
            p.resetJointState(self.robot_id, jid, q)
        p.stepSimulation()

        # zero_acc = [0.0] * len(self.movable_joints)
        # zero_vel = [0.0] * len(self.movable_joints)

        # calculateJacobian takes full lists (len = num_joints) for positions/vels/accs; we pass lists aligned to movable joints
        # but PyBullet expects full vectors of size num_joints where fixed joints are ignored; our approach: build full arrays
        full_pos = [0.0] * len(self.movable_joints)
        full_vel = [0.0] * len(self.movable_joints)
        full_acc = [0.0] * len(self.movable_joints)
        for idx, jid in enumerate(self.movable_joints):
            full_pos[idx] = joint_positions[idx]
            full_vel[idx] = joint_velocities[idx]
            full_acc[idx] = 0.0

        # PyBullet returns (linear, angular) jacobians as lists-of-lists shape (3 x n), so we convert to numpy
        J_lin, J_ang = p.calculateJacobian(self.robot_id, link_index, local_point, full_pos, full_vel, full_acc)
        J_lin = np.array(J_lin)  # shape (3, N)
        J_ang = np.array(J_ang)  # shape (3, N)
        J = np.vstack((J_lin, J_ang))  # shape (6, N)

        # we only need columns for movable_joints -> extract
        J_cols = J[:, :len(self.movable_joints)]  # shape (6, m)
        qdot = np.array(joint_velocities).reshape(-1, 1)  # (m,1)
        twist_world = J_cols.dot(qdot).reshape(6)  # [v_world; omega_world]

        if in_frame == 'world':
            return twist_world
        elif in_frame == 'base':
            base_pos, base_quat = self.get_base_pose()
            Rb = quat_to_rot(base_quat)
            # linear part rotate to base frame; angular part rotate as well
            v_world = twist_world[:3]
            w_world = twist_world[3:]
            v_base = Rb.T.dot(v_world)
            w_base = Rb.T.dot(w_world)
            return np.hstack((v_base, w_base))
        elif in_frame == 'ee':
            # transform world twist into EE frame using EE rotation
            ee_pos, ee_quat = self.get_link_frame_pose(link_index)
            Ree = quat_to_rot(ee_quat)
            v_ee = Ree.T.dot(twist_world[:3])
            w_ee = Ree.T.dot(twist_world[3:])
            return np.hstack((v_ee, w_ee))
        else:
            raise ValueError("in_frame must be one of 'world','base','ee'")


# ----------------------------- example usage -----------------------------
if __name__ == '__main__':
    # path to URDF: adapt to your project structure
    urdf = '/home/cowa/hil-serl/serl_robot_infra/robot_env/cowarm/urdf/cowa_4rad_w_arm_6dof.urdf'
    kb = RobotSolver(urdf, base_position=(0, 0, 0), use_gui=False)

    print('robot_id', kb.robot_id)
    print('num_joints', kb.num_joints)
    print('movable_joints', kb.movable_joints)

    # example: compute FK for given joint angles (length must match movable_joints)
    joint_angles = [0.0] * len(kb.movable_joints)
    ee_link = 18 # change to your end-effector link index

    pos_world, quat_world, T = kb.forward_kinematics(joint_angles, ee_link)
    print('EE pose in world:', pos_world, quat_world)

    # example IK: target at same pose -> should return near-zero moves
    ik_result = kb.inverse_kinematics(pos_world.tolist(), quat_world.tolist(), ee_link)
    print('IK result for movable joints:', ik_result)

    # example velocity: small non-zero joint speeds for testing
    qdot = [0.1] * len(kb.movable_joints)
    twist_world = kb.ee_velocity_from_joint_vel(joint_angles, qdot, ee_link, local_point=[0, 0, 0], in_frame='world')
    print('Twist (world) [vx,vy,vz, wx,wy,wz]:', twist_world)

    kb.disconnect()
