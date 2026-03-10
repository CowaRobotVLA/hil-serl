from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from serl_launcher_torch.networks.RLinf_plug.openpi import get_model, get_sac_model
from serl_launcher_torch.networks.RLinf_plug.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from serl_launcher_torch.networks.RLinf_plug.openpi.openpi_action_sac_model import (
    OpenPi0ForSACActionPrediction,
)
from torch.amp import GradScaler, autocast


def _expand_to_target_dim(
    tensor: torch.Tensor | None,
    target_shape: torch.Size,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    tensor_out = tensor
    if tensor_out.shape != target_shape:
        while len(tensor_out.shape) < len(target_shape):
            tensor_out = tensor_out.unsqueeze(-1)
    return tensor_out


def _preprocess_loss_inputs(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    logprob_type: str | None = None,
    single_action_dim: int | None = None,
    loss_mask: torch.Tensor | None = None,
    loss_mask_sum: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    prev_values: torch.Tensor | None = None,
    returns: torch.Tensor | None = None,
    reward_type: str | None = None,
    versions: torch.Tensor | None = None,
    **kwargs,
) -> dict:
    if reward_type == "chunk_level":
        advantages = advantages.flatten()
        if loss_mask is not None:
            loss_mask = loss_mask.flatten()
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.flatten()
        if values is not None:
            values = values.flatten()
        if prev_values is not None:
            prev_values = prev_values.flatten()
        if returns is not None:
            returns = returns.flatten()

    bsz = logprobs.shape[0]
    proximal_logprobs = kwargs.get("proximal_logprobs", None)
    if logprob_type == "token_level":
        if single_action_dim is None:
            raise ValueError("single_action_dim is required for token_level logprob_type")
        logprobs = logprobs.reshape(bsz, -1, single_action_dim)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim)
        if proximal_logprobs is not None:
            proximal_logprobs = proximal_logprobs.reshape(bsz, -1, single_action_dim)
        if versions is not None:
            versions = versions.reshape(bsz, -1, single_action_dim)
        advantages = advantages.unsqueeze(-1)
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(-1)
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.unsqueeze(-1)
    elif logprob_type == "action_level":
        if single_action_dim is None:
            raise ValueError("single_action_dim is required for action_level logprob_type")
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        if proximal_logprobs is not None:
            proximal_logprobs = proximal_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        if versions is not None:
            versions = versions.reshape(bsz, -1, single_action_dim)[..., 0]
    elif logprob_type == "chunk_level":
        if single_action_dim is None:
            raise ValueError("single_action_dim is required for chunk_level logprob_type")
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
        if proximal_logprobs is not None:
            proximal_logprobs = proximal_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
        if versions is not None:
            versions = versions.reshape(bsz, -1, single_action_dim)[:, 0, 0]

    target_shape = logprobs.shape
    advantages = _expand_to_target_dim(advantages, target_shape)
    loss_mask = _expand_to_target_dim(loss_mask, target_shape)
    loss_mask_sum = _expand_to_target_dim(loss_mask_sum, target_shape)
    values = _expand_to_target_dim(values, target_shape)
    prev_values = _expand_to_target_dim(prev_values, target_shape)
    returns = _expand_to_target_dim(returns, target_shape)
    versions = _expand_to_target_dim(versions, target_shape)

    kwargs.update(
        {
            "logprobs": logprobs,
            "old_logprobs": old_logprobs,
            "proximal_logprobs": proximal_logprobs,
            "versions": versions,
            "advantages": advantages,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "values": values,
            "prev_values": prev_values,
            "returns": returns,
        }
    )
    return kwargs


def _postprocess_loss_metric(metrics_data: dict) -> dict:
    for key, value in metrics_data.items():
        if isinstance(value, torch.Tensor):
            metrics_data[key] = value.detach().item()
        elif isinstance(value, (float, int)):
            metrics_data[key] = value
    return metrics_data


class PI05Agent:
    """
    PyTorch agent for PI05 (Pretrained Image-Action-Output) model.
    This agent wraps the OpenPi0ForRLActionPrediction model for reinforcement learning
    with action chunking and flow-matching based action generation.

    Supports several different algorithms depending on configuration:
     - PI0 (default flow-matching with SDE)
     - PI05 (with value head for critic estimation)
    """

    def __init__(
        self,
        model: OpenPi0ForRLActionPrediction | OpenPi0ForSACActionPrediction,
        model_optimizer: torch.optim.Optimizer,
        config: dict,
    ):
        self.model = model
        self.model_optimizer = model_optimizer
        self.config = config
        self.device = next(model.parameters()).device
        self._training = True

        self.scaler = GradScaler()

    def state_dict(self) -> dict:
        """Return serializable state dictionary"""
        serializable_config = {k: v for k, v in self.config.items() if not callable(v)}

        return {
            "model": self.model.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "config": serializable_config,
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Load state dictionary"""
        self.model.load_state_dict(state_dict["model"], strict=strict)

        if "model_optimizer" in state_dict:
            self.model_optimizer.load_state_dict(state_dict["model_optimizer"])
        if "config" in state_dict:
            self.config.update(state_dict["config"])

    def to(self, device: torch.device) -> "PI05Agent":
        """Move agent to device"""
        device = torch.device(device) if isinstance(device, str) else device
        self.model = self.model.to(device)
        self.device = device
        return self

    def train(self, mode: bool = True) -> "PI05Agent":
        """Set training mode"""
        self._training = mode
        self.model.train(mode)
        return self

    def eval(self) -> "PI05Agent":
        """Set evaluation mode"""
        return self.train(False)

    def _move_batch_to_device(self, batch: dict) -> dict:
        """Recursively move batch tensors to device"""
        result = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                result[k] = self._move_batch_to_device(v)
            elif isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            elif isinstance(v, np.ndarray):
                result[k] = torch.from_numpy(v).to(self.device)
            else:
                result[k] = v
        return result

    def _prepare_batch_for_model(self, batch: dict[str, torch.Tensor]) -> dict:
        return self.model.prepare_batch_for_training(batch=batch, config=self.config)

    def policy_loss(
        self,
        batch: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        if kwargs:
            kwargs = _preprocess_loss_inputs(**kwargs)
            if "batch" in kwargs and isinstance(kwargs["batch"], dict):
                batch = kwargs["batch"]

        if batch is None:
            raise ValueError("policy_loss requires either `batch` or keyword loss inputs.")

        loss, metrics_data = self.model_loss_fn(batch=batch)
        metrics_data = _postprocess_loss_metric(metrics_data)
        return loss, metrics_data

    def model_loss_fn(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        return self.model.compute_training_loss(
            batch=batch,
            config=self.config,
            device=self.device,
        )

    def update(
        self,
        batch: dict[str, torch.Tensor],
        networks_to_update: frozenset[str] = frozenset({"model"}),
    ) -> dict:
        """
        Update agent parameters using gradient descent.

        Args:
            batch: Dictionary containing batch of transitions
            networks_to_update: Set of network components to update

        Returns:
            Dictionary of training statistics
        """
        batch = self._move_batch_to_device(batch)

        # Apply data augmentation if configured
        if self.config.get("augmentation_function") is not None:
            aug_seed = torch.randint(0, 2**31, (1,)).item()
            batch = self.config["augmentation_function"](batch, aug_seed)

        info = {}

        # Update model
        if "model" in networks_to_update:
            self.model_optimizer.zero_grad()

            with autocast(self.device):
                loss, loss_info = self.policy_loss(batch=batch)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.model_optimizer)
            self.scaler.update()

            info.update(loss_info)

        return info

    @torch.no_grad()
    def sample_actions(
        self,
        observations: dict[str, torch.Tensor],
        argmax: bool = False,
        mode: str = "eval",
    ) -> torch.Tensor:
        """
        Sample actions from policy.

        Args:
            observations: Dictionary of observations
            argmax: If True, use argmax for discrete actions (not used for continuous)
            mode: "train" or "eval" mode for sampling

        Returns:
            Actions tensor [batch, action_dim] or [batch, action_chunk, action_dim]
        """
        observations = self._move_batch_to_device(observations)

        # Prepare observations for PI05 model
        obs_dict = {}

        # Copy image and state observations
        for key in observations:
            if "image" in key.lower() or "state" in key.lower():
                obs_dict[key] = observations[key]

        # Add task descriptions
        if "task_descriptions" in observations:
            obs_dict["task_descriptions"] = observations["task_descriptions"]
        elif "prompt" in observations:
            obs_dict["task_descriptions"] = observations["prompt"]

        # Use model's predict_action_batch method
        self.model.eval()

        # Create proper input format for PI05
        env_obs = obs_dict

        # Sample actions using model's inference method
        # mode: "train" or "eval"
        actions, result = self.model.predict_action_batch(
            env_obs=env_obs,
            mode=mode,  # type: ignore
            compute_values=False,
            return_obs=True,
        )

        # Convert to torch tensor
        actions = torch.from_numpy(actions).to(self.device)

        # Return first action in chunk if single action needed
        if len(actions.shape) == 3:
            actions = actions[:, 0, :]  # [batch, action_dim]

        return actions

    @classmethod
    def create(
        cls,
        action_chunk: int = 5,
        action_env_dim: int = 7,
        num_steps: int = 10,
        noise_method: str = "flow_sde",
        noise_level: float = 0.5,
        add_value_head: bool = True,
        train_expert_only: bool = False,
        augmentation_function: Callable | None = None,
        model_lr: float = 3e-4,
        discount: float = 0.97,
        use_sac_model: bool = False,
        device: str = "cuda",
        seed: int = 42,
        **kwargs,
    ) -> "PI05Agent":
        """
        Create PI05 agent from configuration.

        Args:
            model_lr: Learning rate for model optimizer
            discount: Discount factor for RL
            action_chunk: Number of actions to predict in chunk
            action_env_dim: Environment action dimension
            num_steps: Number of denoising steps
            noise_method: Noise method for flow matching
            noise_level: Noise level for SDE
            add_value_head: Whether to add value head for critic
            train_expert_only: Whether to train only expert model
            augmentation_function: Data augmentation function
            device: Device to create agent on
            seed: Random seed

        Returns:
            PI05Agent instance
        """
        torch.manual_seed(seed)
        # 先写死一份config
        config_path = Path(__file__).resolve().parent / "model_configs" / "pi0_5.yaml"
        cfg: DictConfig = OmegaConf.load(str(config_path))
        cfg.model_path = ""
        cfg.add_value_head = add_value_head
        cfg.openpi.value_after_vlm = True
        cfg.openpi.action_chunk = action_chunk
        cfg.openpi.action_env_dim = action_env_dim
        cfg.openpi.num_steps = num_steps
        cfg.openpi.noise_method = noise_method
        cfg.openpi.noise_level = noise_level
        cfg.openpi.add_value_head = add_value_head
        cfg.openpi.train_expert_only = train_expert_only
        cfg.openpi.joint_logprob = kwargs.get("joint_logprob", False)
        cfg.openpi.ignore_last = kwargs.get("ignore_last", False)

        if use_sac_model:
            model = get_sac_model(cfg)
        else:
            model = get_model(cfg)
        # model = OpenPi0ForRLActionPrediction(config)

        # Setup transforms if available
        # if kwargs.get("transforms") is not None:
        #     model.setup_wrappers(
        #         transforms=kwargs["transforms"],
        #         output_transforms=kwargs.get("output_transforms", []),
        #     )

        # Create optimizer
        model_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_lr,
        )

        # Build config dict for agent
        agent_config = {
            "discount": discount,
            "action_chunk": action_chunk,
            "action_env_dim": action_env_dim,
            "num_steps": num_steps,
            "noise_method": noise_method,
            "noise_level": noise_level,
            "add_value_head": add_value_head,
            "train_expert_only": train_expert_only,
            "use_sac_model": use_sac_model,
            "augmentation_function": augmentation_function,
            "value_coef": kwargs.get("value_coef", 1.0),
            "joint_logprob": kwargs.get("joint_logprob", False),
            "ignore_last": kwargs.get("ignore_last", False),
            "global_step": 0,
        }

        # Create agent
        agent = cls(
            model=model,
            model_optimizer=model_optimizer,
            config=agent_config,
        )

        # Move to device
        agent = agent.to(device)

        return agent
