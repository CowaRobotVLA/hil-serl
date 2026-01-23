"""Gym Interface for Franka"""
import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict

# from franka_env.camera.video_capture import VideoCapture
# from franka_env.camera.rs_capture import RSCapture
from serl_robot_infra.robot_env.utils.rotations import euler_2_quat, quat_2_euler
import pycrmw
import sys,os
from serl_robot_infra.robot_env.utils.cr_node_util import ThreadSafeStack, RawImageDecoder
from serl_robot_infra.robot_env.envs.wrappers import HilserlArmControllerWrapper
from msgs.crmw_pb2 import Service

class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k], axis=1
            )

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    """Default configuration for FrankaEnv. Fill in the values below."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    IMAGE_CROP: dict[str, callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((6,))
    GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    RESET_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    LOAD_PARAM: Dict[str, float] = {
        "mass": 0.0,
        "F_x_center_load": [0.0, 0.0, 0.0],
        "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0


##############################################################################


class cowa_env(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        set_load=False,
    ):
        pycrmw.Init(sys.argv)
        if not pycrmw.IsOK():
            os._exit(0)
        self.node = pycrmw.Node("test")
        self.arm_controller = HilserlArmControllerWrapper(self.node, 7)
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )
        self._update_currpos()
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD  # reset the robot joint every 200 cycles

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(0, 100, shape=(1,)),
                        "q": gym.spaces.Box(
                            -np.inf, np.inf, shape=(6,)
                        ),  # xyz + quat
                        "dq": gym.spaces.Box(
                            -np.inf, np.inf, shape=(6,)
                        ),  # xyz + quat
                        # "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        # "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {cam_name: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                                for cam_name in config.IMAGE_CROP.keys()}
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return

        self.cap = None
        # cameras = ["panorama/1", "panorama/2", "panorama/3", "left/1", "right/1"]
        self.init_cameras()
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.url)
            self.displayer.start()

        if set_load:
            input("Put arm into programing mode and press enter.")
            requests.post(self.url + "set_load", json=self.config.LOAD_PARAM)
            input("Put arm into execution mode and press enter.")
            for _ in range(2):
                self._recover()
                time.sleep(1)
        pycrmw.ServiceRegister("hil_serl_arg", self.exec_cmd, Service)
        # if not fake_env:
        #     from pynput import keyboard
        #     self.terminate = False
        #     def on_press(key):
        #         if key == keyboard.Key.esc:
        #             self.terminate = True
        #     self.listener = keyboard.Listener(on_press=on_press)
        #     self.listener.start()
        self.terminate = False

        self.success_key = [False]
        print("Initialized Franka")

    def exec_cmd(self,s: Service):
        if len(s.arg) > 0:
            if s.arg[0] == b'terminate':
                self.terminate = True
                s.ret = b"terminate"
                print("terminate")
            elif s.arg[0] == b'success':
                self.success_key = True
                s.ret = b"success"
                print("success")
            elif s.arg[0] == b'fail':
                self.success_key = False
                s.ret = b"fail"
                print("fail")
        return s

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]
        self._update_currpos()
        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_rotvec(action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()

        gripper_action = (action[-1] + 1)* self.action_scale[2]
        self._send_command(self.nextpos, gripper_action, self.q)
        # self._send_command([0.4, 0, -0.1, 0, 1, 0, 0], gripper_action, self.q)

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        return ob, int(reward), done, False, {"succeed": reward}

    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]["tcp_pose"]
        # convert from quat to euler first
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T  @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        # print(f"Delta: {delta}")
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {self._REWARD_THRESHOLD}')
            return False

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the camera stacks (adapted for ThreadSafeStack)."""
        images = {}
        display_images = {}
        full_res_images = {}
        
        # 遍历配置中的每个相机，这与 init_cameras 的逻辑保持一致
        for cam_name, crop_func in self.config.IMAGE_CROP.items():
            # 1. 获取对应的 stack key (去除首尾斜杠，和你 init 中一致)
            key_name = cam_name.strip("/")
            
            # 确保 stack 存在
            if key_name not in self.camera_stacks:
                continue
                
            stack = self.camera_stacks[key_name]
            
            # 2. 从 Stack 获取图片
            # 逻辑：pop 返回 (flag, image)，如果 flag 为 False 则循环等待
            flag, rgb = stack.pop()
            while not flag:
                time.sleep(0.001) # 短暂休眠防止 CPU 空转 (Busy Waiting)
                flag, rgb = stack.pop()
                
            # 3. 图片处理流程 (保留原版逻辑)
            try:
                # 裁剪 (使用配置中的 crop_func)
                cropped_rgb = crop_func(rgb) if crop_func else rgb
                
                # 调整大小 (Resize)
                # 注意：这里假设 observation_space 的 key 与 cam_name (原始名) 或 key_name (处理名) 对应
                # 为了稳健，优先尝试用 key_name，如果原版 obs space 有斜杠，可能需要调整这里
                target_shape = self.observation_space["images"][key_name].shape[:2][::-1]
                resized = cv2.resize(cropped_rgb, target_shape)
                
                # 格式转换与存储
                images[key_name] = resized[..., ::-1] # BGR to RGB
                display_images[key_name] = resized
                display_images[key_name + "_full"] = cropped_rgb
                full_res_images[key_name] = copy.deepcopy(cropped_rgb)
                # cv2.imwrite("test.png", resized)
            except Exception as e:
                print(f"[Error] Processing image for {key_name}: {e}")
                # 如果处理出错，可以选择返回旧数据或者抛出异常
                # 这里简单演示跳过，或者你可以根据需要添加重试逻辑
                continue

        # 4. 保存视频帧 (保留原版逻辑)
        if self.save_video:
            self.recording_frames.append(full_res_images)

        # 5. 显示图片 (保留原版逻辑)
        if self.display_image:
            self.img_queue.put(display_images)

        return images

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        steps = int(timeout * self.hz)
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self.nextpos = p
        self._update_currpos()

    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Change to precision mode for reset        # Use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

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
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.joint_reset_cycle!=0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self.go_to_reset(joint_reset=joint_reset)
        self.curr_path_length = 0

        time.sleep(1)
        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    if self.url == "http://127.0.0.1:5000/":
                        video_path = f'./videos/left_{camera_key}_{timestamp}.mp4'
                    else:
                        video_path = f'./videos/right_{camera_key}_{timestamp}.mp4'
                    
                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
                
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self):
        """Init both wrist cameras."""
        # 假设这是你的 camera 名字列表
        # 1. 初始化容器字典
        # 使用字典存储，方便后续通过名字访问，例如 self.camera_stacks['panorama/3']
        self.camera_stacks = {}
        self.camera_decoders = {}
        self.camera_readers = {}

        for cam_name, _ in self.config.IMAGE_CROP.items():
            # 清理名字，作为字典的 Key (去除首尾斜杠，防止 key 混乱)
            # 例如 "/panorama/3" -> "panorama/3"
            key_name = cam_name.strip("/")

            # --- 1. 创建独立的 Stack ---
            stack = ThreadSafeStack(10)
            self.camera_stacks[key_name] = stack

            # --- 2. 创建独立的 Decoder (绑定到上面的 stack) ---
            decoder = RawImageDecoder(stack, freq=10)
            self.camera_decoders[key_name] = decoder

            # --- 3. 拼接 Topic 路径 ---
            # 你的范例路径是: /camera/panorama/3/image_raw
            # 逻辑是: /camera/ + {名字} + /image_raw
            topic_path = f"/camera/{key_name}/image_raw"

            # --- 4. 创建 Reader ---
            reader = self.node.CreateReader(topic_path, decoder)
            self.camera_readers[key_name] = reader

            print(f"[Info] Initialized Camera: {key_name} | Topic: {topic_path}")

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _recover(self):
        """Internal function to recover the robot from error state."""
        requests.post(self.url + "clearerr")

    def _send_command(self, eepos, grip_pos, q):
        q = self.arm_controller.get_q_by_ee_pos(eepos[:3], eepos[3:], grip_pos, q)
        self.arm_controller.set_target(q) 

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """

        self.currpos = self.arm_controller.get_eepos_state()
        self.q, self.dq= self.arm_controller.get_arm_state()
        # self.currforce = np.array(ps["force"])
        # self.currtorque = np.array(ps["torque"])
        # self.currjacobian = np.reshape(np.array(ps["jacobian"]), (6, 7))
        self.curr_gripper_pos = self.q[0] / 100.0 # 0~100 -> 0~1
        self.q = self.q[1:]
        self.dq = self.dq[1:]
        # self.currtorque = self.currtorque[1:]

    def update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        ps = requests.post(self.url + "getstate").json()
        self.currpos = np.array(ps["pose"])
        self.currvel = np.array(ps["vel"])

        self.currforce = np.array(ps["force"])
        self.currtorque = np.array(ps["torque"])
        self.currjacobian = np.reshape(np.array(ps["jacobian"]), (6, 7))

        self.q = np.array(ps["q"])
        self.dq = np.array(ps["dq"])

        self.curr_gripper_pos = np.array(ps["gripper_pos"])

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "q": self.q,
            "dq": self.dq,
            "gripper_pose": self.curr_gripper_pos,
            # "tcp_force": self.currforce,
            # "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self):
        # if hasattr(self, 'listener'):
        #     self.listener.stop()
        # self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()
