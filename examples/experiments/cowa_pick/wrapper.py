from typing import OrderedDict
from serl_robot_infra.robot_env.utils.rotations import euler_2_quat
import numpy as np
import requests
import copy
import gymnasium as gym
import time
import pycrmw
from msgs.crmw_pb2 import Service
from msgs import wheel_pb2
from serl_robot_infra.robot_env.envs.cowa_arm_env import cowa_env

class PickEnv(cowa_env):
    def __init__(self, **kwargs):
        self.record_classifier_data = kwargs.pop('record_classifier_data', False)
        super().__init__(**kwargs)
        self.should_regrasp = False
        self.go_next_round = False
        self.remote_flag = False
    def exec_cmd(self,s: Service):
        if len(s.arg) > 0:
            if s.arg[0] == b'terminate':
                self.terminate = True
                s.ret = b"terminate"
                print("terminate")
            elif s.arg[0] == b'success':
                if self.record_classifier_data:
                    self.success_key[0] = True
                    s.ret = b"success"
                    print("success")
                else:
                    self.success = True
            elif s.arg[0] == b'fail':
                if self.record_classifier_data:
                    self.success_key[0] = False
                    s.ret = b"fail"
                    print("fail")
                else:
                    self.terminate = True
            elif s.arg[0] == b'regrasp':
                self.should_regrasp = True
                s.ret = b"regrasp"
                print("regrasp")
            elif s.arg[0] == b'next':
                self.go_next_round = True
                s.ret = b"next"
                print('next round')
            elif s.arg[0] == b'remote':
                self.remote_flag = True
                s.ret = b"remote"
                print("go expert")
            elif s.arg[0] == b'exit_remote':
                self.remote_flag = False
                s.ret = b"exit_remote"
                print("go policy")
        return s
    # rpc
    def RpcRequest(self,cmd):
        # cmd: b'support' or b'release_support' 
        rpc = pycrmw.ServiceFind("WheelArmRpc", Service, Service)
        # rpc = pycrmw.ServiceFind("WheelLegRpc", crmw_pb2.Service, crmw_pb2.Service)
        arg = Service()
        arg.arg.append(cmd)
        try:
            r = rpc(arg)
            if b'exit_remote' in r.ret:
                print("reset")
            return True
        except:
            return False
    def reset(self, **kwargs):

        # Move above the target pose
        self._update_currpos()
        self.remote_flag = False
        # reset_pose = copy.deepcopy(self.config.TARGET_POSE)
        # reset_pose[2] += 0.2
        # self.interpolate_move(reset_pose, timeout=0.5)
        self.success_key[0] = False
        obs, info = super().reset(**kwargs)
        # 等待用户输入
        self.go_next_round = False
        print("wait cmd to next round")
        while self.go_next_round == False:
            time.sleep(0.01)
        obs = self._get_obs()
        self.success = False
        self.terminate = False 

        return obs, info
    
    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        self._send_command(goal, 5, self.q)
        time.sleep(timeout)
        self._update_currpos()
    
    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # step1: close gripper
        self._update_currpos()
        self._send_command(self.currpos, 5, self.q)
        time.sleep(0.5)

        # step2: reset pose
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
            self.interpolate_move(reset_pose, timeout=2)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1.0)
        # step3: open gripper
        self._update_currpos()
        self._send_command(self.currpos, 95, self.q)

class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 6]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.5) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.5
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, 6]
        return observation, reward, terminated, truncated, info
