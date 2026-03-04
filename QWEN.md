# HIL-SERL 项目上下文文档 (修改版)

## 项目概述

**HIL-SERL** (Human-in-the-Loop Soft Actor-Critic Reinforcement Learning) 是一个用于机器人操作训练的强化学习软件套件。本项目基于 [UC Berkeley 的 HIL-SERL](https://hil-serl.github.io/) 开源代码进行了修改，主要改动包括：

1. **环境定义修改**: 使用自定义的 `cowa_env` 替代原始 Franka 环境
2. **PyTorch 迁移**: 将原始 JAX/Flax 实现迁移到 PyTorch
3. **VLA + RL 融合**: 集成 Flow Matching 的 VLA (Vision-Language-Action) 模型与 SAC 算法

**原项目主页**: https://hil-serl.github.io/  
**原论文**: [Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning](https://arxiv.org/abs/2410.21845)

## 技术栈

- **主要语言**: Python 3.11.14
- **深度学习框架**: PyTorch 2.7.1
- **机器人框架**: ROS2 (pycrmw), 自定义机械臂控制器
- **视觉**: OpenCV, 自定义相机接口
- **环境**: gymnasium 0.29.1
- **VLA 模型**: OpenPI (PyTorch 版本)

## 项目结构 (修改后)

```
hil-serl/
├── serl_launcher/                    # 核心 RL 训练代码 (PyTorch 版本)
│   ├── serl_launcher_torch/          # PyTorch 实现
│   │   ├── agents/                   # Agent 策略
│   │   │   ├── continuous/           # 连续控制 agent
│   │   │   │   ├── sac.py            # Soft Actor-Critic (原始)
│   │   │   │   ├── pi05.py           # [修改] Flow Matching VLA+RL
│   │   │   │   ├── pi05_self.py      # [修改] 自定义 PI05 实现
│   │   │   │   └── bc.py             # Behavior Cloning
│   │   │   └── action_model/         # [修改] 动作模型
│   │   │       └── openpi_action_model.py  # [修改] OpenPI PyTorch 实现
│   │   ├── wrappers/                 # Gym 环境封装器
│   │   ├── data/                     # Replay buffer 和数据存储
│   │   ├── vision/                   # 视觉模型和工具
│   │   ├── networks/                 # 网络架构 (Actor-Critic, MLP)
│   │   └── utils/                    # 训练工具函数
│   ├── tests/                        # 单元测试
│   ├── setup.py
│   └── requirements.txt
├── serl_robot_infra/                 # [修改] 机器人基础设施
│   ├── robot_env/                    # [修改] 自定义环境
│   │   └── envs/
│   │       └── cowa_arm_env.py       # [修改] 自定义机械臂环境
│   ├── robot_servers/                # Flask 服务器
│   └── franka_env/                   # 原始 Franka 环境 (保留)
├── examples/                         # 示例脚本
│   ├── experiments/                  # 实验配置
│   ├── record_demos.py               # 演示数据采集
│   ├── record_success_fail.py        # 奖励分类器数据采集
│   └── train_rlpd.py                 # RLPD 训练脚本
├── docs/                             # 文档
│   └── franka_walkthrough.md         # Franka 使用指南
├── environment.yml                   # Conda 环境配置
└── setup.py
```

## 核心修改说明

### 1. 环境定义 (`cowa_arm_env.py`)

**位置**: `serl_robot_infra/robot_env/envs/cowa_arm_env.py`

**主要修改**:
- 使用 `pycrmw` (ROS2 中间件) 替代原始 HTTP 服务器通信
- 使用 `HilserlArmControllerWrapper` 直接控制机械臂
- 相机系统使用 `ThreadSafeStack` 和 `RawImageDecoder` 实现线程安全的图像采集
- 支持多相机配置 (`IMAGE_CROP` 字典)
- 添加 `Service` RPC 接口用于外部控制 (terminate/success/fail)

**环境配置类**:
```python
class DefaultEnvConfig:
    SERVER_URL: str = "http://127.0.0.1:5000/"
    IMAGE_CROP: dict[str, callable] = {}  # 相机裁剪函数
    TARGET_POSE: np.ndarray = np.zeros((6,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    LOAD_PARAM: Dict[str, float] = {...}
    MAX_EPISODE_LENGTH: int = 100
```

**观察空间**:
```python
self.observation_space = gym.spaces.Dict({
    "state": gym.spaces.Dict({
        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),  # xyz + quat
        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
        "gripper_pose": gym.spaces.Box(0, 100, shape=(1,)),
        "q": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
        "dq": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
    }),
    "images": gym.spaces.Dict({
        cam_name: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
        for cam_name in config.IMAGE_CROP.keys()
    }),
})
```

### 2. VLA 模型 (`openpi_action_model.py`)

**位置**: `serl_launcher/serl_launcher_torch/agents/action_model/openpi_action_model.py`

**主要修改**:
- 基于 OpenPI PyTorch 实现
- 集成 Flow Matching 用于动作生成
- 支持多种噪声方法：`flow_sde`, `flow_noise`, `flow_cps`
- 添加 Value Head 用于 PI05 模式 (Critic 估计)

**配置类 (`OpenPi0Config`)**:
```python
@dataclass(frozen=True)
class OpenPi0Config(Pi0Config):
    config_name: str = "pi0_libero"  # pi0_libero, pi05_libero, pi0_maniskill...
    num_images_in_input: int = 2
    noise_method: str = "flow_sde"  # flow_sde, flow_noise, flow_cps
    noise_level: float = 0.5
    action_chunk: int = 5  # 动作块大小
    action_env_dim: int = 7  # 环境动作维度
    num_steps: int = 10  # 去噪步数
    add_value_head: bool = False  # PI05 模式
    value_after_vlm: bool = False  # Value after VLM
    train_expert_only: bool = False
    joint_logprob: bool = False
    # ... 更多配置
```

**核心方法**:
- `default_forward()`: 训练时前向传播，计算 log probs 和 values
- `predict_action_batch()`: 推理时动作采样
- `sample_actions()`: Flow Matching 去噪过程
- `sample_mean_var_val()`: 采样均值、方差和值

### 3. PI05 Agent (`pi05.py`)

**位置**: `serl_launcher/serl_launcher_torch/agents/continuous/pi05.py`

**主要修改**:
- 将 Flow Matching VLA 模型包装为 RL Agent
- 支持 PI0 (纯 Flow Matching) 和 PI05 (带 Value Head) 两种模式
- 实现与 SACAgent 兼容的接口 (`update`, `sample_actions`, `state_dict`)

**核心接口**:
```python
class PI05Agent:
    def update(self, batch, networks_to_update=frozenset({"model"})) -> Dict
    def sample_actions(self, observations, argmax=False, mode="eval") -> torch.Tensor
    def state_dict(self) -> dict
    def load_state_dict(self, state_dict: dict, strict: bool = True)
    
    @classmethod
    def create(cls, sample_obs, sample_action, config=None, **kwargs) -> "PI05Agent"
```

**训练流程**:
```python
# 创建 Agent
agent = PI05Agent.create(
    sample_obs=obs,
    sample_action=action,
    action_chunk=5,
    num_steps=10,
    noise_method="flow_sde",
    add_value_head=True,  # PI05 模式
    model_lr=3e-4,
)

# 训练更新
info = agent.update(batch)

# 推理采样
actions = agent.sample_actions(observations, mode="eval")
```

## 构建和运行

### 训练流程

#### 1. 训练奖励分类器
```bash
# 采集成功/失败数据
python record_success_fail.py --exp_name <task_name> --successes_needed 200

# 训练分类器
cd experiments/<task_name>
python ../../train_reward_classifier.py --exp_name <task_name>
```

#### 2. 采集演示数据
```bash
python ../../record_demos.py --exp_name <task_name> --successes_needed 20
```

#### 3. 策略训练
```bash
# 启动 Actor 节点 (环境交互)
bash run_actor.sh

# 启动 Learner 节点 (策略训练)
bash run_learner.sh
```

### 测试

```bash
cd serl_launcher
python -m pytest tests/ -v
```

## Agent 架构对比

| Agent | 框架 | 算法 | 适用场景 |
|-------|------|------|---------|
| **SACAgent** | PyTorch | Soft Actor-Critic | 标准连续控制任务 |
| **PI05Agent** | PyTorch | Flow Matching + VLA+RL | VLA 模型微调，精密操作 |
| **BC** | PyTorch | Behavior Cloning | 模仿学习/预训练 |


## 开发约定

### 代码风格
- 遵循 black(ruff) 规范
- 使用 type hints 进行类型注解

### 测试实践
- 单元测试使用 `pytest` 框架
- 测试文件位于 `tests/` 目录
- 测试命名：`test_<functionality>.py`

### 配置管理
- 实验配置在 `examples/experiments/<task_name>/config.py`
- 使用数据类或配置类管理超参数
- 支持通过命令行参数覆盖配置

## 已知问题和注意事项

### PI05Agent 实现注意事项

1. **Value Loss 计算**: 当前实现使用当前状态的 values 计算 TD target，应该使用 next state 的 values
2. **load_state_dict**: 不会恢复 config 中的可调用对象（如 `augmentation_function`）
3. **global_step 更新**: 需要在训练循环中手动更新 `config["global_step"]` 以支持噪声退火
4. **梯度裁剪**: Flow Matching 训练通常需要梯度裁剪来稳定训练
5. **推理速度**: Flow Matching 需要多步去噪，推理速度较慢

