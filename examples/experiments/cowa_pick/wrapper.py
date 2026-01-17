from typing import OrderedDict
from serl_robot_infra.robot_env.utils.rotations import euler_2_quat
import numpy as np
import requests
import copy
import gymnasium as gym
import time
from msgs.crmw_pb2 import Service
from serl_robot_infra.robot_env.envs.cowa_arm_env import cowa_env

class PickEnv(cowa_env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_regrasp = False
    
    def exec_cmd(self,s: Service):
        if len(s.arg) > 0:
            if s.arg[0] == b'terminate':
                self.terminate = True
                s.ret = b"terminate"
                print("terminate")
            elif s.arg[0] == b'success':
                self.success_key[0] = True
                s.ret = b"success"
                print("success")
            elif s.arg[0] == b'fail':
                self.success_key[0] = False
                s.ret = b"fail"
                print("fail")
            elif s.arg[0] == b'regrasp':
                self.should_regrasp = True
                s.ret = b"regrasp"
                print("regrasp")
        return s

    def reset(self, **kwargs):

        # Move above the target pose
        self._update_currpos()
        reset_pose = copy.deepcopy(self.config.TARGET_POSE)
        reset_pose[1] += 0.04
        self.interpolate_move(reset_pose, timeout=0.5)

        obs, info = super().reset(**kwargs)
        time.sleep(1)
        self.success = False
        self._update_currpos()
        obs = self._get_obs()
        return obs, info
    
    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        self._send_command(goal, 95)
        time.sleep(timeout)
        self._update_currpos()
    
    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """

        # Perform joint reset if needed
        # if joint_reset:
        #     print("JOINT RESET")
        #     requests.post(self.url + "jointreset")
        #     time.sleep(0.5)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1)

        # Change to compliance mode



class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 50) or (
            action[-1] > 0.5 and self.last_gripper_pos < 50
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info
