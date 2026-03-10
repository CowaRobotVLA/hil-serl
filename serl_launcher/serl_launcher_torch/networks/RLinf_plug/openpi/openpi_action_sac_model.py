from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F
from openpi.models import model as _model

from serl_launcher_torch.networks.RLinf_plug.base_policy import ForwardType
from serl_launcher_torch.networks.RLinf_plug.modules.q_head import QHead
from serl_launcher_torch.networks.RLinf_plug.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)


CriticFeatureMode = Literal["suffix_mean", "suffix_chunk_mean", "vlm_mean"]
CriticActionMode = Literal["flatten_chunk", "first_step"]


@dataclass(frozen=True)
class OpenPi0SACConfig(OpenPi0Config):
    critic_hidden_sizes: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_feature_mode: CriticFeatureMode = "suffix_chunk_mean"
    critic_action_mode: CriticActionMode = "flatten_chunk"
    critic_detach_features: bool = False
    critic_train_action_encoder: bool = False


class OpenPi0ForSACActionPrediction(OpenPi0ForRLActionPrediction):
    def __init__(self, config: OpenPi0SACConfig):
        super().__init__(config)
        self._validate_sac_config()

        critic_hidden_size = self._get_critic_state_feature_dim()
        critic_action_dim = self._get_critic_action_feature_dim()
        critic_hidden_sizes = list(self.config.critic_hidden_sizes)

        self.q1 = QHead(
            hidden_size=critic_hidden_size,
            action_feature_dim=critic_action_dim,
            hidden_dims=critic_hidden_sizes,
            output_dim=1,
            train_action_encoder=self.config.critic_train_action_encoder,
        )
        self.q2 = QHead(
            hidden_size=critic_hidden_size,
            action_feature_dim=critic_action_dim,
            hidden_dims=critic_hidden_sizes,
            output_dim=1,
            train_action_encoder=self.config.critic_train_action_encoder,
        )
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        self._freeze_target_critics()

    def _validate_sac_config(self) -> None:
        if self.config.critic_feature_mode not in {
            "suffix_mean",
            "suffix_chunk_mean",
            "vlm_mean",
        }:
            raise ValueError("critic_feature_mode must be one of {'suffix_mean', 'suffix_chunk_mean', 'vlm_mean'}")
        if self.config.critic_action_mode not in {"flatten_chunk", "first_step"}:
            raise ValueError("critic_action_mode must be one of {'flatten_chunk', 'first_step'}")

    def _freeze_target_critics(self) -> None:
        self.q1_target.requires_grad_(False)
        self.q2_target.requires_grad_(False)
        self.q1_target.eval()
        self.q2_target.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.q1_target.train(False)
        self.q2_target.train(False)
        return self

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        return super().forward(forward_type=forward_type, **kwargs)

    def hard_sync_target_critics(self) -> None:
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self._freeze_target_critics()

    @torch.no_grad()
    def soft_sync_target_critics(self, tau: float = 0.005) -> None:
        for target_param, source_param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.mul_(1 - tau)
            target_param.data.add_(tau * source_param.data)
        for target_param, source_param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.mul_(1 - tau)
            target_param.data.add_(tau * source_param.data)

    def _get_backbone_hidden_size(self) -> int:
        if "pi05_" in self.config.config_name:
            return 2048
        return 1024

    def _get_critic_state_feature_dim(self) -> int:
        return self._get_backbone_hidden_size()

    def _get_critic_action_feature_dim(self) -> int:
        if self.config.critic_action_mode == "first_step":
            return self.config.action_env_dim
        return self.config.action_chunk * self.config.action_env_dim

    def _get_model_device(self) -> torch.device:
        return next(self.parameters()).device

    def _prepare_observation_context(self, data: dict[str, Any]) -> tuple[Any, ...]:
        observation = self.input_transform(data, transpose=False)
        observation = _model.Observation.from_dict(observation)
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        device = self._get_model_device()
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = self._prepare_attention_masks_4d(
            self._build_prefix_attention_mask(prefix_pad_masks, prefix_att_masks)
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            prefix_pad_masks,
            prefix_output,
            past_key_values,
        )

    def _build_prefix_attention_mask(
        self, prefix_pad_masks: torch.Tensor, prefix_att_masks: torch.Tensor
    ) -> torch.Tensor:
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        return make_att_2d_masks(prefix_pad_masks, prefix_att_masks)

    def _pool_prefix_features(self, prefix_output: torch.Tensor) -> torch.Tensor:
        if "pi05_" in self.config.config_name:
            lang_token_len = 200
            all_token_length = 968
        else:
            lang_token_len = 48
            all_token_length = 816

        if self.config.value_vlm_mode == "mean_token":
            prefix_mask = (
                [True] * 256 * self.config.num_images_in_input
                + [False] * 256 * (3 - self.config.num_images_in_input)
                + [True] * lang_token_len
            )
        elif self.config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (all_token_length - 1) + [True]
        elif self.config.value_vlm_mode == "first_token":
            prefix_mask = [True] + [False] * (all_token_length - 1)
        else:
            raise ValueError("value_vlm_mode must be one of {'mean_token', 'last_token', 'first_token'}")

        pooled = prefix_output[:, prefix_mask, :].mean(dim=1, keepdim=False)
        return pooled.to(dtype=torch.float32)

    def _prepare_critic_action_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 2:
            actions = actions[:, None, :]
        if actions.ndim != 3:
            raise ValueError("Critic actions must have shape [batch, action_dim] or [batch, action_chunk, action_dim].")

        batch_size = actions.shape[0]
        device = actions.device
        dtype = actions.dtype
        critic_actions = torch.zeros(
            batch_size,
            self.config.action_chunk,
            self.config.action_env_dim,
            device=device,
            dtype=dtype,
        )
        steps = min(actions.shape[1], self.config.action_chunk)
        dims = min(actions.shape[2], self.config.action_env_dim)
        critic_actions[:, :steps, :dims] = actions[:, :steps, :dims]
        return critic_actions.contiguous()

    def _prepare_model_action_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        critic_actions = self._prepare_critic_action_tensor(actions)
        batch_size = critic_actions.shape[0]
        model_actions = torch.zeros(
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
            device=critic_actions.device,
            dtype=critic_actions.dtype,
        )
        steps = min(self.config.action_chunk, self.config.action_horizon)
        dims = min(self.config.action_env_dim, self.config.action_dim)
        model_actions[:, :steps, :dims] = critic_actions[:, :steps, :dims]
        return model_actions.contiguous()

    def _flatten_critic_actions(self, critic_actions: torch.Tensor) -> torch.Tensor:
        if self.config.critic_action_mode == "first_step":
            return critic_actions[:, 0, :]
        return critic_actions.reshape(critic_actions.shape[0], -1)

    def _extract_critic_state_features(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_output: torch.Tensor,
        past_key_values: Any,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.critic_feature_mode == "vlm_mean":
            state_features = self._pool_prefix_features(prefix_output)
        else:
            model_actions = self._prepare_model_action_tensor(actions)
            timestep = torch.zeros(model_actions.shape[0], device=model_actions.device)
            suffix_out = self.get_suffix_out(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=model_actions,
                timestep=timestep,
            )
            if self.config.critic_feature_mode == "suffix_chunk_mean":
                suffix_out = suffix_out[:, : self.config.action_chunk, :]
            state_features = suffix_out.mean(dim=1, keepdim=False)

        if self.config.critic_detach_features:
            state_features = state_features.detach()
        return state_features.to(dtype=torch.float32)

    def _select_critic_pair(self, use_target: bool) -> tuple[QHead, QHead]:
        if use_target:
            return self.q1_target, self.q2_target
        return self.q1, self.q2

    def sac_q_forward(
        self,
        data: dict[str, Any],
        actions: torch.Tensor,
        use_target: bool = False,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        _, _, _, _, state, prefix_pad_masks, prefix_output, past_key_values = self._prepare_observation_context(data)
        critic_actions = self._prepare_critic_action_tensor(actions).to(state.device)
        state_features = self._extract_critic_state_features(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            prefix_output=prefix_output,
            past_key_values=past_key_values,
            actions=critic_actions,
        )
        action_features = self._flatten_critic_actions(critic_actions).to(dtype=torch.float32)

        q1_module, q2_module = self._select_critic_pair(use_target=use_target)
        q1 = q1_module(state_features, action_features).squeeze(-1)
        q2 = q2_module(state_features, action_features).squeeze(-1)

        outputs: dict[str, torch.Tensor] = {
            "q1": q1,
            "q2": q2,
            "q_min": torch.minimum(q1, q2),
            "actions": critic_actions,
        }
        if return_features:
            outputs["state_features"] = state_features
            outputs["action_features"] = action_features
        return outputs

    def sac_forward(
        self,
        data: dict[str, Any],
        actions: torch.Tensor | None = None,
        use_target: bool = False,
        compute_values: bool = False,
    ) -> dict[str, Any]:
        outputs = super().default_forward(data=data, compute_values=compute_values)
        critic_actions = actions
        if critic_actions is None:
            if "critic_actions" in data:
                critic_actions = data["critic_actions"]
            elif "actions" in data:
                critic_actions = data["actions"]
            elif "chains" in data:
                critic_actions = data["chains"][:, -1]
            else:
                raise ValueError("SAC forward requires `actions`, `critic_actions`, or `chains`.")

        critic_outputs = self.sac_q_forward(
            data=data,
            actions=critic_actions,
            use_target=use_target,
            return_features=False,
        )
        outputs.update(critic_outputs)
        return outputs

    def _prepare_sampling_inputs(
        self, env_obs: dict[str, Any]
    ) -> tuple[_model.Observation, dict[str, Any], dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        observation = _model.Observation.from_dict(processed_obs)

        forward_inputs = {
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
        }
        forward_inputs.update(to_process_obs)
        forward_inputs.pop("prompt", None)
        return observation, processed_obs, forward_inputs

    def _transition_obs_to_env_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        env_obs: dict[str, Any] = {
            "main_images": obs.get("main_images"),
            "states": obs.get("states"),
            "task_descriptions": obs.get("task_descriptions", obs.get("prompt")),
            "wrist_images": obs.get("wrist_images"),
        }
        if env_obs["wrist_images"] is None and "wrist_image" in obs:
            env_obs["wrist_images"] = obs.get("wrist_image")
        return env_obs

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values: bool = False,
        return_obs: bool = True,
        compute_q_values: bool = False,
        use_target_critic: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        observation, processed_obs, forward_inputs = self._prepare_sampling_inputs(env_obs)
        outputs = self.sample_actions(observation, mode=mode, compute_values=compute_values)
        actions = self.output_transform({"actions": outputs["actions"], "state": observation.state})["actions"].numpy()

        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }
        if return_obs:
            result["processed_obs"] = processed_obs
        if compute_q_values:
            action_tensor = torch.from_numpy(actions).to(self._get_model_device())
            q_outputs = self.sac_q_forward(
                data=forward_inputs,
                actions=action_tensor,
                use_target=use_target_critic,
            )
            result.update(q_outputs)
        return actions, result

    def evaluate_q_batch(
        self,
        env_obs: dict[str, Any],
        actions: torch.Tensor,
        use_target: bool = False,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        _, _, forward_inputs = self._prepare_sampling_inputs(env_obs)
        return self.sac_q_forward(
            data=forward_inputs,
            actions=actions,
            use_target=use_target,
            return_features=return_features,
        )

    def compute_training_loss(
        self,
        batch: dict[str, Any],
        config: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        self.set_global_step(config.get("global_step", 0))

        rewards = batch["rewards"].to(device).reshape(-1)
        masks = batch["masks"].to(device).reshape(-1)
        discount = config.get("discount", 0.97)
        alpha = config.get("sac_alpha", 0.0)

        current_env_obs = self._transition_obs_to_env_obs(batch["observations"])
        next_env_obs = self._transition_obs_to_env_obs(batch["next_observations"])

        replay_actions = batch["actions"]
        if torch.is_tensor(replay_actions):
            replay_actions = replay_actions.to(device)

        current_q = self.evaluate_q_batch(
            env_obs=current_env_obs,
            actions=replay_actions,
            use_target=False,
            return_features=False,
        )

        policy_actions, policy_result = self.predict_action_batch(
            env_obs=current_env_obs,
            mode="train",
            compute_values=False,
            return_obs=False,
            compute_q_values=False,
            use_target_critic=False,
        )
        policy_actions_t = torch.from_numpy(policy_actions).to(device)
        policy_q = self.evaluate_q_batch(
            env_obs=current_env_obs,
            actions=policy_actions_t,
            use_target=False,
            return_features=False,
        )

        next_actions, next_result = self.predict_action_batch(
            env_obs=next_env_obs,
            mode="train",
            compute_values=False,
            return_obs=False,
            compute_q_values=False,
            use_target_critic=False,
        )
        next_actions_t = torch.from_numpy(next_actions).to(device)
        target_next_q = self.evaluate_q_batch(
            env_obs=next_env_obs,
            actions=next_actions_t,
            use_target=True,
            return_features=False,
        )

        next_log_probs = next_result["prev_logprobs"].to(device)
        if next_log_probs.ndim >= 3:
            next_log_probs_reduced = next_log_probs.mean(dim=(1, 2))
        else:
            next_log_probs_reduced = next_log_probs.reshape(next_log_probs.shape[0], -1).mean(dim=1)

        td_target = rewards + discount * masks * (target_next_q["q_min"] - alpha * next_log_probs_reduced)

        critic_loss = F.mse_loss(current_q["q1"], td_target) + F.mse_loss(current_q["q2"], td_target)

        policy_log_probs = policy_result["prev_logprobs"].to(device)
        if policy_log_probs.ndim >= 3:
            policy_log_probs_reduced = policy_log_probs.mean(dim=(1, 2))
        else:
            policy_log_probs_reduced = policy_log_probs.reshape(policy_log_probs.shape[0], -1).mean(dim=1)

        actor_loss = (alpha * policy_log_probs_reduced - policy_q["q_min"]).mean()
        total_loss = actor_loss + critic_loss

        info: dict[str, float] = {
            "policy_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "log_probs_mean": policy_log_probs.mean().item(),
            "log_probs_std": policy_log_probs.std().item(),
            "q_min_mean": policy_q["q_min"].mean().item(),
            "q1_mean": current_q["q1"].mean().item(),
            "q2_mean": current_q["q2"].mean().item(),
            "target_q_mean": td_target.mean().item(),
        }
        return total_loss, info
