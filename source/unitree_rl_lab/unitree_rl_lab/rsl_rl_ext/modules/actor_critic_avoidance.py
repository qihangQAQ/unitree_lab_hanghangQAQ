import torch
import torch.nn as nn

from rsl_rl.networks import EmpiricalNormalization, Memory
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent
from torch.distributions import Normal


class ActorCriticAvoidance(ActorCriticRecurrent):
    """Depth-aware recurrent actor-critic for avoidance."""

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
        depth_obs_group_name="perception",
        depth_image_shape=(90, 160),
        depth_encoder_channels=[16, 32, 64, 64],
        depth_encoder_hidden_dims=[256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        kwargs["rnn_hidden_dim"] = lstm_hidden_size
        kwargs["rnn_type"] = "lstm"

        self.depth_obs_group_name = depth_obs_group_name
        self.depth_image_shape = tuple(depth_image_shape)
        self.depth_input_dim = self.depth_image_shape[0] * self.depth_image_shape[1]
        self.depth_encoder_channels = depth_encoder_channels
        self.depth_encoder_hidden_dims = depth_encoder_hidden_dims
        self.depth_feat_dim = depth_encoder_hidden_dims[-1]

        self.actor_vector_obs_dim = self._calculate_vector_obs_dim(obs, obs_groups["policy"])
        self.critic_vector_obs_dim = self._calculate_vector_obs_dim(obs, obs_groups["critic"])

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

        self.depth_encoder = self._build_depth_encoder(activation)
        self.encoded_actor_obs_dim = self.actor_vector_obs_dim + self.depth_feat_dim
        self.encoded_critic_obs_dim = self.critic_vector_obs_dim + self.depth_feat_dim

        self._recreate_memory_modules()
        self._recreate_observation_normalizers()

        print(f"[ActorCriticAvoidance] Actor vector dim: {self.actor_vector_obs_dim}")
        print(f"[ActorCriticAvoidance] Critic vector dim: {self.critic_vector_obs_dim}")
        print(f"[ActorCriticAvoidance] Depth feature dim: {self.depth_feat_dim}")
        print(f"[ActorCriticAvoidance] Encoded actor dim: {self.encoded_actor_obs_dim}")
        print(f"[ActorCriticAvoidance] Encoded critic dim: {self.encoded_critic_obs_dim}")

    def _calculate_vector_obs_dim(self, obs, group_names):
        total_dim = 0
        for group_name in group_names:
            assert len(obs[group_name].shape) == 2, "The avoidance policy only supports 1D observation groups."
            if group_name == self.depth_obs_group_name:
                if obs[group_name].shape[-1] != self.depth_input_dim:
                    raise ValueError(
                        f"Depth observation dim mismatch: got {obs[group_name].shape[-1]},"
                        f" expected {self.depth_input_dim}."
                    )
                continue
            total_dim += obs[group_name].shape[-1]
        return total_dim

    def _make_activation(self, activation: str):
        activation = activation.lower()
        if activation == "elu":
            return nn.ELU()
        if activation == "relu":
            return nn.ReLU()
        if activation == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported activation: {activation}")

    def _build_depth_encoder(self, activation: str):
        layers = []
        in_channels = 1
        act = self._make_activation(activation)
        for out_channels in self.depth_encoder_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2))
            layers.append(self._make_activation(activation))
            in_channels = out_channels

        conv_encoder = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *self.depth_image_shape)
            conv_out_dim = conv_encoder(dummy).flatten(start_dim=1).shape[-1]

        mlp_layers = [nn.Flatten(start_dim=1)]
        input_dim = conv_out_dim
        for hidden_dim in self.depth_encoder_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(self._make_activation(activation))
            input_dim = hidden_dim

        return nn.Sequential(conv_encoder, *mlp_layers)

    def _encode_group_obs(self, obs, group_names):
        vector_terms = []
        depth_flat = None
        for group_name in group_names:
            group_obs = obs[group_name]
            if group_name == self.depth_obs_group_name:
                depth_flat = group_obs
            else:
                vector_terms.append(group_obs)

        if depth_flat is None:
            raise KeyError(f"Missing depth observation group '{self.depth_obs_group_name}'.")

        vector_obs = torch.cat(vector_terms, dim=-1) if vector_terms else None
        leading_shape = depth_flat.shape[:-1]
        depth_image = depth_flat.reshape(-1, 1, *self.depth_image_shape)
        depth_features = self.depth_encoder(depth_image).reshape(*leading_shape, self.depth_feat_dim)

        if vector_obs is None:
            return depth_features
        return torch.cat([vector_obs, depth_features], dim=-1)

    def update_distribution(self, *args, **kwargs):
        super().update_distribution(*args, **kwargs)

        mean = self.distribution.mean
        if self.noise_std_type == "scalar":
            std = torch.clamp(self.std.expand_as(mean), min=1e-6)
        elif self.noise_std_type == "log":
            log_std = torch.clamp(self.log_std, min=-5.0, max=2.0)
            std = torch.clamp(torch.exp(log_std).expand_as(mean), min=1e-6, max=10.0)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
        self.distribution = Normal(mean, std)

    def _recreate_memory_modules(self):
        rnn = self.memory_a.rnn
        if isinstance(rnn, nn.LSTM):
            rnn_type = "lstm"
        elif isinstance(rnn, nn.GRU):
            rnn_type = "gru"
        else:
            rnn_type = "lstm"

        rnn_num_layers = rnn.num_layers
        rnn_hidden_dim = rnn.hidden_size

        self.memory_a = Memory(
            input_size=self.encoded_actor_obs_dim,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )
        self.memory_c = Memory(
            input_size=self.encoded_critic_obs_dim,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        device = next(self.parameters()).device
        self.memory_a.to(device)
        self.memory_c.to(device)

    def _recreate_observation_normalizers(self):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.encoded_actor_obs_dim)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(self.encoded_critic_obs_dim)

    def get_actor_obs(self, obs):
        return self._encode_group_obs(obs, self.obs_groups["policy"])

    def get_critic_obs(self, obs):
        return self._encode_group_obs(obs, self.obs_groups["critic"])
