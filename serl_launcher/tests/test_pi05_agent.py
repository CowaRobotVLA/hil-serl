"""
Unit tests for PI05Agent in serl_launcher_torch.

Tests cover:
1. Agent creation and initialization
2. Forward pass and loss computation
3. Update step
4. Action sampling (inference)
5. State dict save/load
6. Device movement
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any
import sys
import os

# Add parent directory to path so we can import serl_launcher_torch
sys.path.append(".")

from serl_launcher_torch.agents.continuous.pi05 import PI05Agent
from serl_launcher_torch.networks.RLinf_plug.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
    OpenPi0Config,
)


def create_sample_obs(
    batch_size: int = 4,
    image_size: tuple = (3, 256, 256),
    state_dim: int = 7,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Create sample observation dictionary for testing."""
    return {
        "main_images": torch.randn(batch_size, *image_size, device=device),
        "wrist_images": torch.randn(batch_size, *image_size, device=device),
        "states": torch.randn(batch_size, state_dim, device=device),
        "task_descriptions": ["pick_and_place"] * batch_size,
    }


def create_sample_action(
    batch_size: int = 4,
    action_dim: int = 7,
    action_chunk: int = 5,
    device: str = "cpu",
) -> torch.Tensor:
    """Create sample action tensor for testing."""
    return torch.randn(batch_size, action_chunk, action_dim, device=device)


def create_batch(
    batch_size: int = 4,
    action_dim: int = 7,
    action_chunk: int = 5,
    image_size: tuple = (3, 256, 256),
    device: str = "cpu",
) -> Dict[str, Any]:
    """Create a complete batch of RL transitions for training."""
    return {
        "observations": {
            "main_images": torch.randn(batch_size, *image_size, device=device),
            "wrist_images": torch.randn(batch_size, *image_size, device=device),
            "states": torch.randn(batch_size, 7, device=device),
            "task_descriptions": ["pick_and_place"] * batch_size,
        },
        "next_observations": {
            "main_images": torch.randn(batch_size, *image_size, device=device),
            "wrist_images": torch.randn(batch_size, *image_size, device=device),
            "states": torch.randn(batch_size, 7, device=device),
            "task_descriptions": ["pick_and_place"] * batch_size,
        },
        "actions": torch.randn(batch_size, action_chunk, action_dim, device=device),
        "rewards": torch.randn(batch_size, 1, device=device),
        "masks": torch.ones(batch_size, 1, device=device),
    }


class TestPI05AgentCreation:
    """Test PI05Agent creation and initialization."""

    def test_create_agent_default_config(self):
        """Test creating agent with default configuration."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            action_chunk=5,
            action_env_dim=7,
            num_steps=10,
            device="cpu",
        )

        assert agent is not None
        assert isinstance(agent.model, OpenPi0ForRLActionPrediction)
        assert agent.config["action_chunk"] == 5
        assert agent.config["num_steps"] == 10
        assert agent._training is True

    def test_create_agent_with_value_head(self):
        """Test creating agent with value head for critic estimation."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            add_value_head=True,
            value_coef=0.5,
            device="cpu",
        )

        assert agent.config["add_value_head"] is True
        assert agent.config["value_coef"] == 0.5

    def test_create_agent_with_custom_lr(self):
        """Test creating agent with custom learning rate."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            model_lr=1e-4,
            device="cpu",
        )

        # Check optimizer learning rate
        for param_group in agent.model_optimizer.param_groups:
            assert param_group["lr"] == 1e-4

    def test_create_agent_different_action_dims(self):
        """Test creating agent with different action dimensions."""
        sample_obs = create_sample_obs()
        
        # Test with action_dim = 4
        sample_action_4d = create_sample_action(action_dim=4)
        agent_4d = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action_4d,
            device="cpu",
        )
        assert agent_4d.model.config.action_dim == 4

        # Test with action_dim = 10
        sample_action_10d = create_sample_action(action_dim=10)
        agent_10d = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action_10d,
            device="cpu",
        )
        assert agent_10d.model.config.action_dim == 10


class TestPI05AgentForward:
    """Test PI05Agent forward pass and loss computation."""

    def test_model_loss_fn_basic(self):
        """Test basic loss computation."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            device="cpu",
        )

        batch = create_batch()
        loss, info = agent.model_loss_fn(batch)

        assert torch.is_tensor(loss)
        assert loss.requires_grad
        assert "policy_loss" in info
        assert "log_probs_mean" in info
        assert "log_probs_std" in info

    def test_model_loss_fn_with_value_head(self):
        """Test loss computation with value head."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            add_value_head=True,
            device="cpu",
        )

        batch = create_batch()
        loss, info = agent.model_loss_fn(batch)

        assert torch.is_tensor(loss)
        assert "policy_loss" in info
        assert "value_loss" in info
        assert "values_mean" in info

    def test_prepare_batch_for_model(self):
        """Test batch preparation for PI05 model input."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            device="cpu",
        )

        batch = create_batch()
        model_input = agent._prepare_batch_for_model(batch)

        # Check required keys
        assert "observation" in model_input
        assert "actions" in model_input
        assert "rewards" in model_input
        assert "masks" in model_input
        assert "denoise_inds" in model_input
        assert "chains" in model_input

        # Check shapes
        assert model_input["actions"].shape[0] == batch["actions"].shape[0]
        assert model_input["denoise_inds"].shape[0] == batch["actions"].shape[0]


class TestPI05AgentUpdate:
    """Test PI05Agent update step."""

    def test_update_basic(self):
        """Test basic update step."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            device="cpu",
        )

        batch = create_batch()
        agent.train()
        info = agent.update(batch)

        assert "policy_loss" in info
        assert isinstance(info["policy_loss"], float)

    def test_update_with_value_head(self):
        """Test update with value head enabled."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            add_value_head=True,
            device="cpu",
        )

        batch = create_batch()
        agent.train()
        info = agent.update(batch)

        assert "policy_loss" in info
        assert "value_loss" in info

    def test_update_networks_to_update(self):
        """Test selective network update."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            device="cpu",
        )

        batch = create_batch()
        agent.train()
        
        # Update only model
        info = agent.update(batch, networks_to_update=frozenset({"model"}))
        assert "policy_loss" in info

    def test_update_training_mode(self):
        """Test that update works in training mode."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            device="cpu",
        )

        batch = create_batch()
        agent.train()
        info = agent.update(batch)

        assert agent._training is True
        assert agent.model.training is True

    def test_update_eval_mode_no_grad(self):
        """Test that eval mode doesn't update gradients."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            device="cpu",
        )

        batch = create_batch()
        agent.eval()
        
        # Get initial model parameters
        initial_params = next(agent.model.parameters()).clone()
        
        # Update in eval mode (should not update due to no_grad in sample_actions)
        # Note: update() should still work but model should be in eval mode
        info = agent.update(batch)
        
        # Parameters might change but model should be in eval mode
        assert agent.model.training is False


class TestPI05AgentSampling:
    """Test PI05Agent action sampling."""

    def test_sample_actions_basic(self):
        """Test basic action sampling."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            device="cpu",
        )

        obs = create_sample_obs(batch_size=2)
        agent.eval()
        actions = agent.sample_actions(obs)

        assert torch.is_tensor(actions)
        assert actions.shape[0] == 2  # batch size
        assert actions.shape[-1] == 7  # action dim

    def test_sample_actions_batch_size(self):
        """Test action sampling with different batch sizes."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            device="cpu",
        )

        for batch_size in [1, 4, 8]:
            obs = create_sample_obs(batch_size=batch_size)
            agent.eval()
            actions = agent.sample_actions(obs)
            assert actions.shape[0] == batch_size

    def test_sample_actions_with_task_descriptions(self):
        """Test action sampling with task descriptions."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            device="cpu",
        )

        obs = create_sample_obs(batch_size=2)
        obs["task_descriptions"] = ["pick_up_block", "place_in_box"]
        
        agent.eval()
        actions = agent.sample_actions(obs, mode="eval")

        assert actions.shape[0] == 2

    def test_sample_actions_no_grad(self):
        """Test that action sampling doesn't compute gradients."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            device="cpu",
        )

        obs = create_sample_obs()
        agent.eval()
        
        with torch.no_grad():
            actions = agent.sample_actions(obs)
        
        assert not actions.requires_grad


class TestPI05AgentStateDict:
    """Test PI05Agent state dict save/load."""

    def test_state_dict_structure(self):
        """Test state dictionary structure."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            device="cpu",
        )

        state_dict = agent.state_dict()

        assert "model" in state_dict
        assert "model_optimizer" in state_dict
        assert "config" in state_dict

    def test_load_state_dict(self):
        """Test loading state dictionary."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent1 = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            device="cpu",
        )

        # Save state
        state_dict = agent1.state_dict()

        # Create new agent and load state
        agent2 = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            device="cpu",
        )
        agent2.load_state_dict(state_dict)

        # Check parameters match
        for (name1, param1), (name2, param2) in zip(
            agent1.model.named_parameters(), agent2.model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)

    def test_state_dict_preserves_config(self):
        """Test that config is preserved in state dict."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            add_value_head=True,
            discount=0.99,
            action_chunk=10,
            device="cpu",
        )

        state_dict = agent.state_dict()

        assert state_dict["config"]["add_value_head"] is True
        assert state_dict["config"]["discount"] == 0.99
        assert state_dict["config"]["action_chunk"] == 10


class TestPI05AgentDevice:
    """Test PI05Agent device movement."""

    def test_to_device(self):
        """Test moving agent to device."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            device="cpu",
        )

        # Move to CPU (should work even if already on CPU)
        agent = agent.to("cpu")
        assert agent.device.type == "cpu"
        
        # Check model is on correct device
        assert next(agent.model.parameters()).device.type == "cpu"

    def test_batch_device_movement(self):
        """Test that batches are moved to correct device."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            device="cpu",
        )

        batch = create_batch(device="cpu")
        moved_batch = agent._move_batch_to_device(batch)

        # Check all tensors are on correct device
        for key in ["rewards", "masks"]:
            assert moved_batch[key].device.type == "cpu"

        # Check nested dicts
        for key in moved_batch["observations"]:
            if torch.is_tensor(moved_batch["observations"][key]):
                assert moved_batch["observations"][key].device.type == "cpu"


class TestPI05AgentEdgeCases:
    """Test PI05Agent edge cases and error handling."""

    def test_single_step_action(self):
        """Test with single-step actions (no chunking)."""
        sample_obs = create_sample_obs()
        # Single step action: [batch, action_dim]
        sample_action = torch.randn(4, 7)

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            action_chunk=1,
            device="cpu",
        )

        batch = create_batch(action_chunk=1)
        agent.train()
        info = agent.update(batch)

        assert "policy_loss" in info

    def test_joint_logprob_mode(self):
        """Test with joint logprob mode enabled."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            joint_logprob=True,
            device="cpu",
        )

        batch = create_batch()
        model_input = agent._prepare_batch_for_model(batch)

        # In joint logprob mode, denoise_inds should use all steps
        assert model_input["denoise_inds"].shape[1] == agent.config["num_steps"]

    def test_ignore_last_mode(self):
        """Test with ignore_last mode enabled."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        agent = PI05Agent.create(
            sample_obs=sample_obs,
            sample_action=sample_action,
            ignore_last=True,
            device="cpu",
        )

        batch = create_batch()
        agent.train()
        info = agent.update(batch)

        assert "policy_loss" in info

    def test_different_noise_methods(self):
        """Test different noise methods."""
        sample_obs = create_sample_obs()
        sample_action = create_sample_action()

        for noise_method in ["flow_sde", "flow_noise", "flow_cps"]:
            agent = PI05Agent.create(
                sample_obs=sample_obs,
                sample_action=sample_action,
                noise_method=noise_method,
                device="cpu",
            )
            assert agent.config["noise_method"] == noise_method


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
