from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from serl_launcher_torch.networks.RLinf_plug.openpi import get_model
from serl_launcher_torch.networks.RLinf_plug.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from serl_launcher_torch.networks.actor_critic_nets import Critic
from torch.amp import GradScaler, autocast


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
        model: OpenPi0ForRLActionPrediction,
        model_optimizer: torch.optim.Optimizer,
        q1: nn.Module,
        q2: nn.Module,
        q1_target: nn.Module,
        q2_target: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        config: dict,
    ):
        self.model = model
        self.model_optimizer = model_optimizer
        self.config = config
        self.device = next(model.parameters()).device
        self._training = True

        # Twin Q-networks for SAC
        self.q1 = q1
        self.q2 = q2
        self.q1_target = q1_target
        self.q2_target = q2_target
        self.q_optimizer = q_optimizer

        # Entropy temperature (alpha)
        self.log_alpha = torch.zeros(1, device=self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -config.get("action_env_dim", 7)  # -|A| as target

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
        """
        Prepare batch data for PI05 model input.
        Converts standard RL batch format to PI05 observation format.
        """
        # Extract observations and actions
        obs_keys = batch["observations"]
        next_obs_keys = batch["next_observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        masks = batch["masks"]

        # Build observation dict for PI05
        # PI05 expects observations with images, states, and task descriptions
        obs_dict = {}

        # Copy image observations
        for key in obs_keys:
            if "image" in key.lower() or "state" in key.lower():
                obs_dict[key] = obs_keys[key]

        # Add task descriptions if available
        if "task_descriptions" in obs_keys:
            obs_dict["task_descriptions"] = obs_keys["task_descriptions"]
        elif "prompt" in obs_keys:
            obs_dict["task_descriptions"] = obs_keys["prompt"]

        # Ensure proper shape for actions: [batch, action_chunk, action_dim]
        action_shape = actions.shape
        if len(action_shape) == 2:
            # If actions are [batch, action_dim], need to add action_chunk dimension
            # This assumes single-step actions, will need to be expanded
            batch_size = action_shape[0]
            action_dim = action_shape[1]
            action_chunk = self.config.get("action_chunk", 5)
            # Repeat action for action_chunk steps
            actions = actions.unsqueeze(1).repeat(1, action_chunk, 1)

        # Build model input dict
        model_input = {
            "observation": obs_dict,
            "actions": actions,
            "rewards": rewards,
            "masks": masks,
        }

        # Add denoise_inds for training (required by PI05)
        batch_size = actions.shape[0]
        num_steps = self.config.get("num_steps", 10)

        if self.config.get("joint_logprob", False):
            # For joint logprob, use all denoise steps
            denoise_inds = torch.arange(num_steps).unsqueeze(0).repeat(batch_size, 1)
        else:
            # For single step, sample random denoise index
            if self.config.get("ignore_last", False):
                denoise_inds = torch.randint(0, num_steps - 1, (batch_size, 1))
            else:
                denoise_inds = torch.randint(0, num_steps, (batch_size, 1))

        model_input["denoise_inds"] = denoise_inds
        model_input["chains"] = actions  # Initial action chain

        return model_input

    def _update_target_networks(self, tau: float = 0.005):
        """Soft update target Q-networks."""
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _get_q_values(self, observations: dict, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values from twin Q-networks."""
        # Flatten observations for critic
        obs_features = []
        for k, v in observations.items():
            if "image" in k.lower():
                # For image observations, use global average pooling
                if len(v.shape) == 5:  # [B, T, C, H, W]
                    v = v.mean(dim=1)  # [B, C, H, W]
                obs_features.append(v.flatten(1))
            elif "state" in k.lower():
                obs_features.append(v)

        if obs_features:
            obs_flat = torch.cat(obs_features, dim=1)
        else:
            # Fallback: just use any available tensor
            obs_flat = list(observations.values())[0]
            if len(obs_flat.shape) > 2:
                obs_flat = obs_flat.flatten(1)

        # Ensure actions match batch size
        if len(actions.shape) == 3:  # [B, action_chunk, dim]
            actions = actions[:, 0, :]  # [B, dim]

        q1_vals = self.q1(obs_flat, actions)
        q2_vals = self.q2(obs_flat, actions)
        return q1_vals, q2_vals

    def model_loss_fn(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """
        Compute PI05 model loss using standard SAC algorithm.

        SAC Loss Components:
        1. Q-loss: MSE(Q(s,a), r + γ * Q_target(s',a'))
        2. Policy loss: α * log π(a|s) - min(Q1(s,a), Q2(s,a))
        3. Alpha loss: -α * (log π + target_entropy)
        """
        # Move batch to device
        batch = self._move_batch_to_device(batch)

        observations = batch["observations"]
        next_observations = batch["next_observations"]
        actions = batch["actions"]  # [B, action_dim] or [B, action_chunk, action_dim]
        rewards = batch["rewards"]  # [B]
        masks = batch["masks"]  # [B]
        discount = self.config.get("discount", 0.97)

        # === Q-FUNCTION LOSS (Critic) ===
        # Get Q-values for current (s, a)
        q1_vals, q2_vals = self._get_q_values(observations, actions)

        # Get target Q-values for next state
        # Sample actions from policy for next state
        with torch.no_grad():
            # Get actions from model for next observation
            next_obs_dict = {}
            for k, v in next_observations.items():
                if "image" in k.lower() or "state" in k.lower():
                    next_obs_dict[k] = v

            # Add task descriptions if available
            if "task_descriptions" in next_observations:
                next_obs_dict["task_descriptions"] = next_observations["task_descriptions"]
            elif "prompt" in next_observations:
                next_obs_dict["task_descriptions"] = next_observations["prompt"]

            # Sample actions from model for next state
            next_actions, _ = self.model.predict_action_batch(
                env_obs=next_obs_dict,
                mode="train",
                compute_values=False,
                return_obs=False,
            )
            next_actions = torch.from_numpy(next_actions).to(self.device)

            # Handle action shape
            if len(next_actions.shape) == 3:
                next_actions = next_actions[:, 0, :]  # [B, action_dim]

            # Get target Q-values
            q1_target, q2_target = self._get_q_values(next_observations, next_actions)
            q_target = torch.min(q1_target, q2_target)

            # Compute TD target: r + γ * (1-done) * Q(s', a')
            td_target = rewards + discount * masks * q_target.squeeze(-1)

        # Q-loss: MSE(Q(s,a), td_target)
        q1_loss = F.mse_loss(q1_vals.squeeze(-1), td_target)
        q2_loss = F.mse_loss(q2_vals.squeeze(-1), td_target)
        q_loss = q1_loss + q2_loss

        # === POLICY LOSS (Actor) ===
        # Sample new actions from current policy
        obs_dict = {}
        for k, v in observations.items():
            if "image" in k.lower() or "state" in k.lower():
                obs_dict[k] = v

        if "task_descriptions" in observations:
            obs_dict["task_descriptions"] = observations["task_descriptions"]
        elif "prompt" in observations:
            obs_dict["task_descriptions"] = observations["prompt"]

        # Get actions from model
        policy_actions, _ = self.model.predict_action_batch(
            env_obs=obs_dict,
            mode="train",
            compute_values=False,
            return_obs=False,
        )
        policy_actions = torch.from_numpy(policy_actions).to(self.device)

        if len(policy_actions.shape) == 3:
            policy_actions = policy_actions[:, 0, :]  # [B, action_dim]

        # Get Q-values for policy actions
        q1_policy, q2_policy = self._get_q_values(observations, policy_actions)
        q_policy = torch.min(q1_policy, q2_policy)

        # Policy loss: -Q(s, a_policy) (standard SAC policy loss)
        policy_loss = -q_policy.mean()

        # === ALPHA LOSS (Entropy Temperature) ===
        # Use entropy from policy distribution - approximate with action variance
        # Since we don't have direct log_probs here, use a simpler alpha loss
        alpha_loss = -self.log_alpha.exp() * (policy_loss.detach() + self.target_entropy)
        alpha_loss = alpha_loss.mean()

        # === TOTAL LOSS ===
        # For training, we only update model (policy) parameters
        # Q-networks are updated separately in update()
        total_loss = policy_loss + alpha_loss

        # === INFO DICT ===
        alpha = self.log_alpha.exp().detach()
        info = {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha.item(),
            "q1_mean": q1_vals.mean().item(),
            "q2_mean": q2_vals.mean().item(),
            "q_target_mean": q_target.mean().item(),
            "td_target_mean": td_target.mean().item(),
        }

        return total_loss, info

    def update(
        self,
        batch: dict[str, torch.Tensor],
        networks_to_update: frozenset[str] = frozenset({"model", "q", "alpha"}),
    ) -> dict:
        """
        Update agent parameters using gradient descent.

        Args:
            batch: Dictionary containing batch of transitions
            networks_to_update: Set of network components to update
                - "model": Flow matching policy (actor)
                - "q": Twin Q-networks (critic)
                - "alpha": Entropy temperature

        Returns:
            Dictionary of training statistics
        """
        batch = self._move_batch_to_device(batch)

        # Apply data augmentation if configured
        if self.config.get("augmentation_function") is not None:
            aug_seed = torch.randint(0, 2**31, (1,)).item()
            batch = self.config["augmentation_function"](batch, aug_seed)

        info = {}

        # Update Q-networks (Critic)
        if "q" in networks_to_update:
            self.q_optimizer.zero_grad()

            observations = batch["observations"]
            next_observations = batch["next_observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            masks = batch["masks"]
            discount = self.config.get("discount", 0.97)

            # Get current Q-values
            q1_vals, q2_vals = self._get_q_values(observations, actions)

            # Get target Q-values for next state
            with torch.no_grad():
                # Sample actions from model for next state
                next_obs_dict = {}
                for k, v in next_observations.items():
                    if "image" in k.lower() or "state" in k.lower():
                        next_obs_dict[k] = v

                if "task_descriptions" in next_observations:
                    next_obs_dict["task_descriptions"] = next_observations["task_descriptions"]
                elif "prompt" in next_observations:
                    next_obs_dict["task_descriptions"] = next_observations["prompt"]

                next_actions, _ = self.model.predict_action_batch(
                    env_obs=next_obs_dict,
                    mode="train",
                    compute_values=False,
                    return_obs=False,
                )
                next_actions = torch.from_numpy(next_actions).to(self.device)

                if len(next_actions.shape) == 3:
                    next_actions = next_actions[:, 0, :]

                q1_target, q2_target = self._get_q_values(next_observations, next_actions)
                q_target = torch.min(q1_target, q2_target)
                td_target = rewards + discount * masks * q_target.squeeze(-1)

            # Q-loss
            q1_loss = F.mse_loss(q1_vals.squeeze(-1), td_target)
            q2_loss = F.mse_loss(q2_vals.squeeze(-1), td_target)
            q_loss = q1_loss + q2_loss

            q_loss.backward()
            self.q_optimizer.step()

            # Update target networks
            self._update_target_networks()

            info["q1_loss"] = q1_loss.item()
            info["q2_loss"] = q2_loss.item()
            info["q_loss"] = q_loss.item()

        # Update model (Policy)
        if "model" in networks_to_update:
            self.model_optimizer.zero_grad()

            with autocast(self.device):
                loss, loss_info = self.model_loss_fn(batch)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.model_optimizer)
            self.scaler.update()

            info.update(loss_info)

        # Update alpha (Temperature)
        if "alpha" in networks_to_update:
            # Alpha is already updated in model_loss_fn via the combined loss
            # But we should also do a separate update for the log_alpha parameter
            alpha = self.log_alpha.exp().detach()
            info["alpha"] = alpha.item()

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
        cfg: DictConfig = OmegaConf.load("serl_launcher/serl_launcher_torch/agents/continuous/model_configs/pi0_5.yaml")
        cfg.model_path = ""
        cfg.add_value_head = True
        cfg.openpi.value_after_vlm = True

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

        # === Create Twin Q-networks for SAC ===
        # Get observation and action dimensions from config
        obs_dim = cfg.openpi.vision_encoder.image_size[0] * cfg.openpi.vision_encoder.image_size[1] * 3  # Simplified
        # Use action_env_dim from parameter
        action_dim = action_env_dim

        # Q-network takes concatenated obs + action as input
        q_input_dim = obs_dim + action_dim  # This will be adjusted based on actual obs

        # Create Q-networks with simple MLP
        q1 = nn.Sequential(
            nn.Linear(q_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)

        q2 = nn.Sequential(
            nn.Linear(q_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)

        # Target Q-networks (copy weights)
        q1_target = nn.Sequential(
            nn.Linear(q_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)

        q2_target = nn.Sequential(
            nn.Linear(q_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)

        # Copy weights to target networks
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())

        # Q-network optimizer
        q_lr = kwargs.get("q_lr", 3e-4)
        q_optimizer = torch.optim.Adam(
            list(q1.parameters()) + list(q2.parameters()),
            lr=q_lr,
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
            q1=q1,
            q2=q2,
            q1_target=q1_target,
            q2_target=q2_target,
            q_optimizer=q_optimizer,
            config=agent_config,
        )

        # Move to device
        agent = agent.to(device)

        return agent
