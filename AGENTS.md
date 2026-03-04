# AGENTS.md - HIL-SERL Agentic Coding Guidelines

This file provides context for agentic coding agents operating in this repository.

## Project Overview

**HIL-SERL** (Human-in-the-Loop Soft Actor-Critic Reinforcement Learning) is a PyTorch-based RL framework for robotic manipulation. Modified from UC Berkeley's HIL-SERL with:
- Custom `cowa_env` robot environment (ROS2/pcrmw)
- Flow Matching VLA + RL integration (PI05 agent)
- Behavior Cloning and SAC agents

**Tech Stack**: Python 3.10+, PyTorch 2.4+, gymnasium 0.29.1, ROS2

---

## Build, Lint, and Test Commands

### Running Tests

```bash
# Run all tests
cd serl_launcher
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_pi05_agent.py -v

# Run single test function
python -m pytest tests/test_pi05_agent.py::TestPI05AgentCreation::test_create_agent_default_config -v

# Run tests with coverage
python -m pytest tests/ -v --cov=serl_launcher_torch --cov-report=term-missing
```

### Installation

```bash
# Create conda environment
conda create -n hilserl python=3.10
conda activate hilserl

# Install main package
cd serl_launcher
pip install -e .
pip install -r requirements.txt

# Install robot infrastructure (follow serl_robot_infra/README.md)
```

### Linting & Formatting

```bash
# Install pre-commit hooks (if configured)
pre-commit install
pre-commit run --all-files

# Format with ruff (if installed)
ruff check --fix .
ruff format .

# Type checking (mypy)
mypy serl_launcher_torch --ignore-missing-imports
```

---

## Code Style Guidelines

### Imports

Order imports by category, separated by blank lines:
1. Standard library (`os`, `sys`, `typing`, `collections`, etc.)
2. Third-party packages (`numpy`, `torch`, `gymnasium`, `wandb`, etc.)
3. Local packages (`serl_launcher_torch`, `serl_robot_infra`, etc.)

```python
# Standard library
import os
import sys
from typing import Dict, Optional, Tuple
from collections import defaultdict
from functools import partial

# Third-party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import wandb
from tqdm import tqdm

# Local
from serl_launcher_torch.agents.continuous.sac import SACAgent
from serl_launcher_torch.networks.actor_critic_nets import Critic, Policy
from serl_launcher_torch.utils.train_utils import _unpack
```

### Type Hints

- Use type hints for all function signatures
- Use `Dict`, `List`, `Tuple` from `typing` (or Python 3.9+ native syntax)
- Prefer explicit return types

```python
def create_sample_obs(
    batch_size: int = 4,
    image_size: Tuple[int, int, int] = (3, 256, 256),
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Create sample observation dictionary for testing."""
    return {
        "main_images": torch.randn(batch_size, *image_size, device=device),
        "states": torch.randn(batch_size, 7, device=device),
    }
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `SACAgent`, `PI05Agent`, `DefaultEnvConfig` |
| Functions/methods | snake_case | `sample_actions`, `update`, `create_sample_obs` |
| Constants | UPPER_SNAKE_CASE | `MAX_EPISODE_LENGTH`, `BATCH_SIZE` |
| Variables | snake_case | `action_chunk`, `policy_loss` |
| Private methods | `_snake_case` | `_prepare_batch_for_model` |

### Docstrings

Use Google-style docstrings for classes and functions:

```python
class PI05Agent:
    """
    PyTorch implementation of PI05 agent with Flow Matching VLA.

    Supports both PI0 (pure Flow Matching) and PI05 (with Value Head) modes.
    Implements SACAgent-compatible interface for RL training.

    Args:
        actor: Policy network
        critic: Value network
        ...
    """

    def update(self, batch: Dict, networks_to_update: FrozenSet = frozenset({"model"})) -> Dict:
        """
        Perform one training update step.

        Args:
            batch: Dictionary containing observations, actions, rewards, etc.
            networks_to_update: Which networks to update (default: {"model"})

        Returns:
            Dictionary of loss values and metrics
        """
```

### Error Handling

- Use explicit exceptions with informative messages
- Never use bare `except:` or `except Exception:`
- Validate inputs at function boundaries

```python
# Good
def concat_batches(offline_batch, online_batch, axis=1):
    if not isinstance(offline_batch, dict) or not isinstance(online_batch, dict):
        raise TypeError(f"Expected dict, got {type(offline_batch)} and {type(online_batch)}")

# Bad
try:
    batch[k] = torch.cat((v, online_batch[k]), dim=axis)
except:
    continue
```

### Configuration Management

- Use dataclasses or configuration classes for experiment configs
- Support command-line overrides
- Store configs in `examples/experiments/<task_name>/config.py`

```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 3e-4
    discount: float = 0.99
    device: str = "cuda"
```

---

## Project Structure

```
hil-serl/
├── serl_launcher/                 # Core RL training (PyTorch)
│   ├── serl_launcher_torch/
│   │   ├── agents/               # Agent implementations (SAC, PI05, BC)
│   │   ├── networks/             # Neural network architectures
│   │   ├── wrappers/             # Gym environment wrappers
│   │   ├── data/                 # Replay buffer, dataset
│   │   ├── vision/               # Vision models, augmentations
│   │   └── utils/                # Training utilities
│   └── tests/                    # Unit tests
├── serl_robot_infra/             # Robot infrastructure
│   └── robot_env/envs/           # Custom environments (cowa_arm_env)
├── examples/
│   ├── experiments/              # Task-specific configs
│   ├── train_rlpd.py             # RL training scripts
│   └── record_demos.py           # Data collection
└── pyproject.toml                # Project configuration
```

---

## Agent Architecture Reference

| Agent | Algorithm | Use Case |
|-------|-----------|----------|
| `SACAgent` | Soft Actor-Critic | Standard continuous control |
| `PI05Agent` | Flow Matching + VLA + RL | VLA fine-tuning, precision tasks |
| `BC` | Behavior Cloning | Imitation learning / pretraining |

---

## Common Development Patterns

### Testing

- Test files: `tests/test_*.py` or `serl_launcher/tests/test_*.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<functionality>_<specific_case>`
- Use fixtures for common setup

```python
class TestPI05AgentCreation:
    def test_create_agent_default_config(self):
        agent = PI05Agent.create(sample_obs=obs, sample_action=action)
        assert agent is not None
```

### Device Management

- Always move tensors to the correct device
- Use `agent.device` or `next(model.parameters()).device`
- Wrap inference in `torch.no_grad()`

```python
def sample_actions(self, observations, mode="eval"):
    self.eval()
    with torch.no_grad():
        actions = self.model(observations)
    return actions
```

### Batch Processing

- Use `defaultdict` for flexible batch handling
- Check tensor shapes explicitly
- Handle nested observation dicts

---

## Known Issues & Caveats

1. **PI05Agent Value Loss**: Uses current state's values for TD target (should use next state)
2. **load_state_dict**: Does not restore callable config objects (e.g., `augmentation_function`)
3. **global_step**: Must manually update `config["global_step"]` for noise annealing
4. **Flow Matching Inference**: Multi-step denoising is computationally expensive

---

## Key Files

- Agent implementations: `serl_launcher/serl_launcher_torch/agents/continuous/`
- VLA model: `serl_launcher/serl_launcher_torch/networks/RLinf_plug/openpi/openpi_action_model.py`
- Robot environment: `serl_robot_infra/robot_env/envs/cowa_arm_env.py`
- Test example: `serl_launcher/tests/test_pi05_agent.py`
