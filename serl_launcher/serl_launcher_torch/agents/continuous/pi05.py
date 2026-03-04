from functools import partial
from typing import Iterable, Optional, Tuple, FrozenSet, Dict, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from serl_launcher_torch.networks.RLinf_plug.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
    OpenPi0Config,
)


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
        serializable_config = {k: v for k, v in self.config.items()
                              if not callable(v)}
        
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
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
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
    
    def _prepare_batch_for_model(self, batch: Dict[str, torch.Tensor]) -> Dict:
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
    
    def model_loss_fn(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PI05 model loss.
        Uses flow-matching objective with optional value head.
        """
        # Prepare batch for model
        model_input = self._prepare_batch_for_model(batch)
        
        # Move to device
        model_input = self._move_batch_to_device(model_input)
        
        # Set global step for noise annealing
        self.model.set_global_step(self.config.get("global_step", 0))
        
        # Forward pass through model
        with autocast('cuda'):
            outputs = self.model.default_forward(
                data=model_input,
                compute_values=self.config.get("add_value_head", False),
            )
        
        # Extract outputs
        log_probs = outputs["logprobs"]  # [batch, action_chunk, action_dim]
        values = outputs["values"]  # [batch, 1]
        entropy = outputs.get("entropy", None)  # [batch, 1]
        
        # Compute policy loss (negative log likelihood)
        # Average over action chunk and action dimensions
        policy_loss = -log_probs.mean()
        
        # Compute value loss if value head is enabled
        value_loss = torch.tensor(0.0, device=self.device)
        if self.config.get("add_value_head", False):
            # TD-style value loss
            rewards = batch["rewards"]
            masks = batch["masks"]
            discount = self.config.get("discount", 0.97)
            
            # Compute TD target
            with torch.no_grad():
                # Use bootstrapped value from next state if available
                target_values = rewards + discount * masks * values
            
            # MSE loss for value prediction
            value_loss = F.mse_loss(values.squeeze(), target_values.squeeze())
        
        # Total loss
        total_loss = policy_loss
        
        if self.config.get("add_value_head", False):
            value_coef = self.config.get("value_coef", 1.0)
            total_loss = total_loss + value_coef * value_loss
        
        # Compute info dict
        info = {
            "policy_loss": policy_loss.item(),
            "log_probs_mean": log_probs.mean().item(),
            "log_probs_std": log_probs.std().item(),
        }
        
        if self.config.get("add_value_head", False):
            info["value_loss"] = value_loss.item()
            info["values_mean"] = values.mean().item()
        
        if entropy is not None:
            info["entropy"] = entropy.mean().item()
        
        return total_loss, info
    
    def update(
        self,
        batch: Dict[str, torch.Tensor],
        networks_to_update: FrozenSet[str] = frozenset({"model"})
    ) -> Dict:
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
            
            with autocast('cuda'):
                loss, loss_info = self.model_loss_fn(batch)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.model_optimizer)
            self.scaler.update()
            
            info.update(loss_info)
        
        return info
    
    @torch.no_grad()
    def sample_actions(
        self,
        observations: Dict[str, torch.Tensor],
        argmax: bool = False,
        mode: str = "eval"
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
        actions, result = self.model.predict_action_batch(
            env_obs=env_obs,
            mode=mode,
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
        sample_obs: Dict[str, torch.Tensor],
        sample_action: torch.Tensor,
        config: OpenPi0Config = None,
        model_lr: float = 3e-4,
        discount: float = 0.97,
        action_chunk: int = 5,
        action_env_dim: int = 7,
        num_steps: int = 10,
        noise_method: str = "flow_sde",
        noise_level: float = 0.5,
        add_value_head: bool = False,
        train_expert_only: bool = False,
        augmentation_function: Optional[Callable] = None,
        device: str = "cuda",
        **kwargs,
    ) -> "PI05Agent":
        """
        Create PI05 agent from configuration.
        
        Args:
            sample_obs: Sample observation from environment
            sample_action: Sample action from environment
            config: OpenPi0Config object (optional, will create default if not provided)
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
            
        Returns:
            PI05Agent instance
        """
        torch.manual_seed(kwargs.get("seed", 42))
        
        # Create default config if not provided
        if config is None:
            config = OpenPi0Config(
                config_name=kwargs.get("config_name", "pi05_libero"),
                num_images_in_input=kwargs.get("num_images_in_input", 2),
                noise_method=noise_method,
                noise_level=noise_level,
                action_chunk=action_chunk,
                action_env_dim=action_env_dim,
                num_steps=num_steps,
                train_expert_only=train_expert_only,
                add_value_head=add_value_head,
                **kwargs,
            )
        
        # Determine action dimension from sample action
        if len(sample_action.shape) == 1:
            action_dim = sample_action.shape[0]
        else:
            action_dim = sample_action.shape[-1]
        
        # Override config action_dim with environment action dimension
        config.action_dim = action_dim
        
        # Create model
        model = OpenPi0ForRLActionPrediction(config)
        
        # Initialize model with dummy input to set up weights
        dummy_obs = {}
        for k, v in sample_obs.items():
            if isinstance(v, torch.Tensor):
                dummy_obs[k] = torch.zeros(1, *v.shape[1:], device='cpu', dtype=v.dtype)
            elif isinstance(v, np.ndarray):
                dummy_obs[k] = torch.zeros(1, *v.shape[1:], device='cpu', dtype=torch.float32)
            else:
                dummy_obs[k] = v
        
        # Setup transforms if available
        if kwargs.get("transforms") is not None:
            model.setup_wrappers(
                transforms=kwargs["transforms"],
                output_transforms=kwargs.get("output_transforms", []),
            )
        
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
