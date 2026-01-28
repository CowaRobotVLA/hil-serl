import os
import torch
import numpy as np
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# 计算 'examples' 目录的路径 (假设是当前文件的上两级: cowa_pick -> experiments -> examples)
examples_dir = os.path.join(current_dir, "..", "..")
# 将其加入系统路径
sys.path.append(examples_dir)

from serl_robot_infra.robot_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from serl_robot_infra.robot_env.envs.relative_env import RelativeFrame
from serl_robot_infra.robot_env.envs.cowa_arm_env import DefaultEnvConfig
from serl_launcher_torch.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher_torch.wrappers.chunking import ChunkingWrapper
from serl_launcher_torch.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.cowa_pick.wrapper import PickEnv, GripperPenaltyWrapper


class EnvConfig(DefaultEnvConfig):
    SERVER_URL: str = "http://127.0.0.2:5000/"
    IMAGE_CROP = {"panorama/3": lambda img: img,
                  "surround/front": lambda img: img}
    TARGET_POSE = np.array([0.5,0.0,-0.1, 0, np.pi, 0])
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.2, 0, 0, 0])
    ACTION_SCALE = np.array([0.1, 0.1, 50])
    RANDOM_RESET = False
    DISPLAY_IMAGE = False
    RANDOM_XY_RANGE = 0.01
    RANDOM_RZ_RANGE = 0.1
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.1, 0.1, 0.3, 0.2, 0.2, 0.3])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.1, 0.05, 0.2, 0.2, 0.2, 0.3])
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.006,
        "translational_clip_y": 0.0059,
        "translational_clip_z": 0.0035,
        "translational_clip_neg_x": 0.005,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.0035,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.015,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.015,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0.0,
    }
    MAX_EPISODE_LENGTH = 30000


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["panorama/3", "surround/front"]
    classifier_keys = ["panorama/3", "surround/front"]
    proprio_keys = ["tcp_pose", "q", "dq", "gripper_pose"]
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet18-pretrained"
    setup_mode = "single-arm-learned-gripper"
    # "single-arm-learned-gripper"  # "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False, record_classifier_data = False, device = "cuda"):
        keyboard_classifier = True
        env = PickEnv(
            fake_env=fake_env, save_video=save_video, record_classifier_data=record_classifier_data, config=EnvConfig()
        )
        if not fake_env:
            env = SpacemouseIntervention(env)
        # env = RelativeFrame(env)
        # env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            if not keyboard_classifier:
                classifier = load_classifier_func(
                    sample=env.observation_space.sample(),
                    image_keys=self.image_keys,
                    checkpoint_path=os.path.abspath("classifier_ckpt/classifier.pth"),
                    device=device,
                )

            def reward_func(obs):
                if keyboard_classifier:
                    if env.success == True:
                        return 1
                    else:
                        return 0
                else:
                    sigmoid = lambda x: 1 / (1 + torch.exp(-x))
                    return int(sigmoid(classifier(obs)) > 0.7 and obs["state"][0, 6] < 0.6)
            # def reward_func(obs):
            #     return int(classifier(obs) > 0.5)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env