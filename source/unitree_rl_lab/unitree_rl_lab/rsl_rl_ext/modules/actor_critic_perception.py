import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent


class ActorCriticPerception(ActorCriticRecurrent):
    """
    Perception-enhanced Actor-Critic network with terrain encoder.

    Extends ActorCriticRecurrent with terrain encoding capabilities:
    1) Terrain encoder: MLP that encodes 187D height map to terrain features
    2) Shared terrain features for both actor and critic
    3) Proper integration with RNN memory modules for sequence processing

    The network follows this pipeline:
    187D height map → terrain encoder → concatenate with other observations → RNN memory → MLP heads
    """

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        lstm_hidden_size=256,
        terrain_encoder_dims=[128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        """
        Args:
            obs: Observation specification
            obs_groups: Dictionary mapping observation groups to their names
            num_actions: Number of action dimensions
            actor_obs_normalization: Whether to normalize actor observations
            critic_obs_normalization: Whether to normalize critic observations
            actor_hidden_dims: MLP hidden dimensions for actor head (after RNN)
            critic_hidden_dims: MLP hidden dimensions for critic head (after RNN)
            lstm_hidden_size: Hidden size for LSTM memory (maps to rnn_hidden_dim)
            terrain_encoder_dims: MLP dimensions for terrain encoder
            activation: Activation function for MLP layers
            init_noise_std: Initial standard deviation for action noise
            noise_std_type: Type of noise parameterization ("scalar" or "log")
            **kwargs: Additional arguments passed to parent class
        """
        # Map our parameters to parent class parameters
        kwargs['rnn_hidden_dim'] = lstm_hidden_size
        kwargs['rnn_type'] = 'lstm'  # Explicitly use LSTM

        # Call parent class initialization (creates memory_a, memory_c, actor, critic)
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        # Store terrain encoder dimensions
        self.terrain_encoder_dims = terrain_encoder_dims
        self.terrain_input_dim = 187  # Fixed: 17x11 height map

        # Calculate observation dimensions without terrain
        self.actor_obs_dim_no_terrain = self._calculate_obs_dim_no_terrain(
            obs, obs_groups["policy"], terrain_dim=self.terrain_input_dim
        )
        self.critic_obs_dim_no_terrain = self._calculate_obs_dim_no_terrain(
            obs, obs_groups["critic"], terrain_dim=self.terrain_input_dim
        )

        # Build terrain encoder (shared between actor and critic)
        self.terrain_encoder = self._build_terrain_encoder(terrain_encoder_dims, activation)
        self.terrain_feat_dim = terrain_encoder_dims[-1]

        # Verify dimensions
        total_actor_obs = self.actor_obs_dim_no_terrain + self.terrain_feat_dim
        total_critic_obs = self.critic_obs_dim_no_terrain + self.terrain_feat_dim

        print(f"[ActorCriticPerception] Actor obs dim (no terrain): {self.actor_obs_dim_no_terrain}")
        print(f"[ActorCriticPerception] Critic obs dim (no terrain): {self.critic_obs_dim_no_terrain}")
        print(f"[ActorCriticPerception] Terrain feature dim: {self.terrain_feat_dim}")
        print(f"[ActorCriticPerception] Total actor obs dim: {total_actor_obs}")
        print(f"[ActorCriticPerception] Total critic obs dim: {total_critic_obs}")
        print(f"[ActorCriticPerception] Terrain encoder: {self.terrain_encoder}")

    def _calculate_obs_dim_no_terrain(self, obs, obs_group_names, terrain_dim=187):
        """Calculate total observation dimension excluding terrain height map."""
        total_dim = 0
        for obs_group in obs_group_names:
            total_dim += obs[obs_group].shape[-1]

        # Verify that we have at least terrain_dim dimensions
        if total_dim < terrain_dim:
            raise ValueError(
                f"Total observation dimension ({total_dim}) is less than terrain input dimension ({terrain_dim})"
            )

        return total_dim - terrain_dim

    def _build_terrain_encoder(self, encoder_dims, activation="elu"):
        """Build MLP terrain encoder."""
        layers = []
        input_dim = self.terrain_input_dim

        for i, hidden_dim in enumerate(encoder_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "elu":
                layers.append(nn.ELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            input_dim = hidden_dim

        return nn.Sequential(*layers)

    def get_actor_obs(self, obs):
        """
        Get actor observations with terrain encoding applied.

        Extracts height map (last 187 dimensions), encodes it, and concatenates
        with other observations.
        """
        # Get raw concatenated observations from parent class
        raw_obs = super().get_actor_obs(obs)

        # Extract height map (assumed to be the last 187 dimensions)
        height_scan = raw_obs[..., -self.terrain_input_dim:]  # (batch, 187)
        other_obs = raw_obs[..., :-self.terrain_input_dim]    # (batch, other_dim)

        # Encode terrain
        terrain_feat = self.terrain_encoder(height_scan)  # (batch, terrain_feat_dim)

        # Concatenate terrain features with other observations
        encoded_obs = torch.cat([terrain_feat, other_obs], dim=-1)  # (batch, total_dim)

        return encoded_obs

    def get_critic_obs(self, obs):
        """
        Get critic observations with terrain encoding applied.

        Same logic as get_actor_obs but for critic observation group.
        """
        # Get raw concatenated observations from parent class
        raw_obs = super().get_critic_obs(obs)

        # Extract height map (assumed to be the last 187 dimensions)
        height_scan = raw_obs[..., -self.terrain_input_dim:]  # (batch, 187)
        other_obs = raw_obs[..., :-self.terrain_input_dim]    # (batch, other_dim)

        # Encode terrain
        terrain_feat = self.terrain_encoder(height_scan)  # (batch, terrain_feat_dim)

        # Concatenate terrain features with other observations
        encoded_obs = torch.cat([terrain_feat, other_obs], dim=-1)  # (batch, total_dim)

        return encoded_obs

    # Note: All other methods (act, evaluate, update_distribution, etc.) are inherited
    # from ActorCriticRecurrent and will automatically use our overridden
    # get_actor_obs and get_critic_obs methods.