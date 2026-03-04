from serl_launcher.serl_launcher_torch.agents.action_model import OpenPi0Config, OpenPi0ForRLActionPrediction


class PI05Agent:
    def __init__(
        self,
        config: OpenPi0Config,
    ):
        self.model = OpenPi0ForRLActionPrediction(config)    
    
    def _compute_next_actions(
        self, 
        obs_enc: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_action_distribution = self.actor(obs_enc)
        next_actions, next_actions_log_probs = next_action_distribution.sample_and_log_prob()
        
        assert next_actions_log_probs.shape == (batch["actions"].shape[0],)
        
        return next_actions, next_actions_log_probs
    
    def critic_loss_fn(
        self, 
        obs_enc: torch.Tensor,
        next_obs_enc: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        batch_size = batch["rewards"].shape[0]
        
        with torch.no_grad():
            next_actions, next_actions_log_probs = self._compute_next_actions(next_obs_enc, batch)
            
            target_qs = self.critic_target(next_obs_enc, next_actions)
            
            if self.config["critic_subsample_size"] is not None:
                indices = torch.randperm(self.config["critic_ensemble_size"])
                indices = indices[:self.config["critic_subsample_size"]]
                target_qs = target_qs[indices]
            
            target_q = target_qs.min(dim=0)[0]
            assert target_q.shape == (batch_size,)
            
            # Compute backup
            target = (
                batch["rewards"] + 
                self.config["discount"] * batch["masks"] * target_q
            )
            
            if self.config["backup_entropy"]:
                temperature = self.temp()
                target = target - temperature * next_actions_log_probs

        # Extract continuous actions from batch
        actions_continuous = batch["actions"][..., :-1]
        
        current_qs = self.critic(obs_enc, actions_continuous)
        assert current_qs.shape == (self.config["critic_ensemble_size"], batch_size)
        
        critic_loss = F.mse_loss(
            current_qs, 
            target.unsqueeze(0).expand(self.config["critic_ensemble_size"], -1)
        )

        info = {
            "critic_loss": critic_loss.item(),
            "q_values": current_qs.mean().item(),
            "target_q": target.mean().item(),
        }
        
        return critic_loss, info
    
    def grasp_critic_loss_fn(
        self, 
        obs_enc: torch.Tensor,
        next_obs_enc: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        batch_size = batch["rewards"].shape[0]
        
        # Convert gripper action from [-1, 1] to {0, 1, 2}
        grasp_action = torch.round(batch["actions"][..., -1]).long() + 1
        
        with torch.no_grad():
            # Get next grasp Q-values from target network
            target_next_grasp_qs = self.grasp_critic_target(next_obs_enc)
            assert target_next_grasp_qs.shape == (batch_size, 3)
            
            # Get next grasp Q-values from online network
            next_grasp_qs = self.grasp_critic(next_obs_enc)
            
            # Select best next grasp action using online network
            best_next_grasp_action = next_grasp_qs.argmax(dim=-1)
            assert best_next_grasp_action.shape == (batch_size,)
            
            # Evaluate best action with target network
            target_next_grasp_q = target_next_grasp_qs[torch.arange(batch_size), best_next_grasp_action]
            assert target_next_grasp_q.shape == (batch_size,)
            
            # Compute target Q-values
            grasp_rewards = batch["rewards"] + batch["grasp_penalty"]
            target_grasp_q = (
                grasp_rewards + 
                self.config["discount"] * batch["masks"] * target_next_grasp_q
            )
            assert target_grasp_q.shape == (batch_size,)
        
        # Get predicted Q-values from online network
        predicted_grasp_qs = self.grasp_critic(obs_enc)
        assert predicted_grasp_qs.shape == (batch_size, 3)
        
        # Select predicted Q-value for taken grasp action
        predicted_grasp_q = predicted_grasp_qs[torch.arange(batch_size), grasp_action]
        assert predicted_grasp_q.shape == (batch_size,)
        
        # Compute MSE loss
        grasp_critic_loss = F.mse_loss(predicted_grasp_q, target_grasp_q)
        
        info = {
            "grasp_critic_loss": grasp_critic_loss.item(),
            "predicted_grasp_qs": predicted_grasp_qs.mean().item(),
            "target_grasp_qs": target_grasp_q.mean().item(),
            "grasp_rewards": grasp_rewards.mean().item(),
        }
        
        return grasp_critic_loss, info
    
    def actor_loss_fn(
        self, 
        obs_enc: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        temperature = self.temp().detach()
        dist = self.actor(obs_enc)
        actions, log_probs = dist.sample_and_log_prob()
        
        # Extract continuous actions (excluding gripper action)
        # actions_continuous = actions[..., :-1]
        
        # Get Q-values for the sampled continuous actions
        q_values = self.critic(obs_enc, actions)
        q_values = q_values.mean(dim=0)  # Average across ensemble
        
        actor_loss = (temperature * log_probs - q_values).mean()
        
        info = {
            "actor_loss": actor_loss.item(),
            "entropy": -log_probs.mean().item(),
            "temperature": temperature.item(),
        }
        
        return actor_loss, info

    def temperature_loss_fn(
        self, 
        obs_enc: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute temperature loss and info dict"""
    
        _, next_actions_log_probs = self._compute_next_actions(obs_enc, batch)
        entropy = -next_actions_log_probs.mean()
            
        temperature_loss = self.temp(
            lhs=entropy.detach(),
            rhs=self.config["target_entropy"]
        )
        
        info = {"temperature_loss": temperature_loss.item()}
        return temperature_loss, info
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Recursively move batch tensors to device."""
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

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        networks_to_update: FrozenSet[str] = frozenset({"actor", "critic", "grasp_critic", "temperature"})
    ) -> Dict:
        batch = self._move_batch_to_device(batch)
        
        # Apply data augmentation
        if self.config.get("augmentation_function") is not None:
            aug_seed = torch.randint(0, 2**31, (1,)).item()
            batch = self.config["augmentation_function"](batch, aug_seed)

        reward_bias = self.config.get("reward_bias", 0.0)
        if reward_bias != 0.0:
            batch = {**batch, "rewards": batch["rewards"] + reward_bias}
        
        info = {}

        obs_enc = self.encoder(batch["observations"])
        next_obs_enc = self.encoder(batch["next_observations"])
        
        # Update critic
        if "critic" in networks_to_update:
            self.critic_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()

            with autocast('cuda'):
                critic_loss, critic_info = self.critic_loss_fn(obs_enc, next_obs_enc.detach(), batch)

            self.scaler.scale(critic_loss).backward()
            self.scaler.step(self.critic_optimizer)
            self.scaler.step(self.encoder_optimizer)
            self.scaler.update()
            info.update(critic_info)
            
            with torch.no_grad():
                tau = self.config["soft_target_update_rate"]
                for target, source in zip(
                    self.critic_target.parameters(), 
                    self.critic.parameters()
                ):
                    target.data.mul_(1 - tau)
                    target.data.add_(tau * source.data)
        
        # Update grasp critic
        if "grasp_critic" in networks_to_update:
            self.grasp_critic_optimizer.zero_grad()

            with autocast('cuda'):
                grasp_critic_loss, grasp_critic_info = self.grasp_critic_loss_fn(
                    obs_enc.detach(), next_obs_enc.detach(), batch
                )

            self.scaler.scale(grasp_critic_loss).backward()
            self.scaler.step(self.grasp_critic_optimizer)
            self.scaler.update()
            info.update(grasp_critic_info)
            
            with torch.no_grad():
                tau = self.config["soft_target_update_rate"]
                for target, source in zip(
                    self.grasp_critic_target.parameters(), 
                    self.grasp_critic.parameters()
                ):
                    target.data.mul_(1 - tau)
                    target.data.add_(tau * source.data)
        
        # Update actor
        if "actor" in networks_to_update:
            self.actor_optimizer.zero_grad()

            with autocast('cuda'):
                actor_loss, actor_info = self.actor_loss_fn(obs_enc.detach())

            self.scaler.scale(actor_loss).backward()
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
            info.update(actor_info)
        
        # Update temperature
        if "temperature" in networks_to_update:
            self.temp_optimizer.zero_grad()

            with autocast('cuda'):
                temp_loss, temp_info = self.temperature_loss_fn(next_obs_enc.detach(), batch)

            self.scaler.scale(temp_loss).backward()
            self.scaler.step(self.temp_optimizer)
            self.scaler.update()
            info.update(temp_info)
            
        return info

    @torch.no_grad()
    def sample_actions(
        self,
        observations: Dict[str, torch.Tensor],
        argmax: bool = False
    ) -> torch.Tensor:
        """Sample actions from policy"""
        observations = self._move_batch_to_device(observations)
        obs_enc = self.encoder(observations)
        
        # Sample continuous actions
        dist = self.actor(obs_enc)
        if argmax:
            continuous_actions = dist.mode()
        else:
            continuous_actions = dist.sample()
        
        # Get grasp Q-values and select best grasp action
        grasp_q_values = self.grasp_critic(obs_enc)
        grasp_action = grasp_q_values.argmax(dim=-1)
        grasp_action = grasp_action - 1  # Map back to {-1, 0, 1}
        
        # Combine continuous and grasp actions
        actions = torch.cat([continuous_actions, grasp_action.unsqueeze(-1)], dim=-1)
        return actions

    @classmethod
    def create_pixels(
        cls,
        config: OpenPi0Config
    ) -> PI05Agent:

        model = cls(
            config=config,
        )
        
        # Create critics
        critic_hidden_dims = [encoder_output_dim + action_dim] + critic_network_kwargs.get("hidden_dims", [256, 256])
        critics = []
        for _ in range(critic_ensemble_size):
            critic_network = MLP(
                hidden_dims=critic_hidden_dims,
                activate_final=True,
                use_layer_norm=critic_network_kwargs.get("use_layer_norm", False),
                activations=critic_network_kwargs.get("activation", nn.Tanh()),
            )
            
            critics.append(Critic(network=critic_network))
        
        critic = CriticEnsemble(critics)
        critic_target = deepcopy(critic)
        
        # Create grasp critic
        grasp_critic_hidden_dims = [encoder_output_dim] + grasp_critic_network_kwargs.get("hidden_dims", [256, 256])
        grasp_critic_network = MLP(
            hidden_dims=grasp_critic_hidden_dims,
            activate_final=True,
            use_layer_norm=grasp_critic_network_kwargs.get("use_layer_norm", False),
            activations=grasp_critic_network_kwargs.get("activation", nn.Tanh()),
        )
        grasp_critic = GraspCritic(
            network=grasp_critic_network,
            output_dim=3,  # 3 discrete gripper actions
        )
        grasp_critic_target = deepcopy(grasp_critic)
        
        # Create temperature (Lagrange multiplier)
        temp = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
        )
        # Set target entropy
        target_entropy = kwargs.get("target_entropy")
        if target_entropy is None:
            target_entropy = -action_dim / 2
        
        # Build config with pixel-specific fields
        config_kwargs = {
            "discount": kwargs.get("discount", 0.97),
            "soft_target_update_rate": kwargs.get("soft_target_update_rate", 0.005),
            "target_entropy": target_entropy,
            "backup_entropy": kwargs.get("backup_entropy", False),
            "critic_ensemble_size": critic_ensemble_size,
            "critic_subsample_size": critic_subsample_size,
            "image_keys": image_keys,
            "augmentation_function": augmentation_function,
            "reward_bias": reward_bias,
        }
    
        # Create optimizers
        temp_optimizer = torch.optim.Adam(temp.parameters(), lr=3e-4)
        encoder_optimizer = torch.optim.Adam(encoder_def.parameters(), lr=3e-4)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
        grasp_critic_optimizer = torch.optim.Adam(grasp_critic.parameters(), lr=3e-4)
        
        agent = cls(
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            grasp_critic=grasp_critic,
            grasp_critic_target=grasp_critic_target,
            temp=temp,
            encoder=encoder_def,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            grasp_critic_optimizer=grasp_critic_optimizer,
            temp_optimizer=temp_optimizer,
            encoder_optimizer=encoder_optimizer,
            config=config_kwargs,
        )
        
        return agent
