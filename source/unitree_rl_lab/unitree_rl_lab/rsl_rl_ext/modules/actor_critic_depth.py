import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


class Conv2dDepthEncoder(nn.Module):
    """Minimal CNN encoder for stacked depth images.

    Input: (N, C, H, W) where C = num_stacked_frames
    Output: (N, output_size) flat feature vector
    """

    def __init__(
        self,
        in_channels: int,
        in_height: int,
        in_width: int,
        channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        paddings: list[int],
        hidden_sizes: list[int],
        output_size: int,
        nonlinearity: str = "ReLU",
        use_maxpool: bool = False,
    ):
        super().__init__()
        if isinstance(nonlinearity, str):
            nonlinearity = getattr(nn, nonlinearity)
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)

        in_c = [in_channels] + channels[:-1]
        conv_layers = []
        for ic, oc, k, s, p in zip(in_c, channels, kernel_sizes, strides, paddings):
            conv_layers.append(nn.Conv2d(ic, oc, k, s, p))
            conv_layers.append(nonlinearity())
            if use_maxpool and s > 1:
                conv_layers.append(nn.MaxPool2d(s))
        self.conv = nn.Sequential(*conv_layers)

        # Compute conv output spatial size: (H + 2p - k) / s + 1
        h, w = in_height, in_width
        for k, s, p in zip(kernel_sizes, strides, paddings):
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1
        self._conv_flat_dim = channels[-1] * h * w

        # MLP head
        mlp_layers = []
        in_dim = self._conv_flat_dim
        for h in hidden_sizes:
            mlp_layers.append(nn.Linear(in_dim, h))
            mlp_layers.append(nonlinearity())
            in_dim = h
        if output_size > 0:
            mlp_layers.append(nn.Linear(in_dim, output_size))
        self.head = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        self._output_size = output_size if output_size > 0 else self._conv_flat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        y = self.conv(x)
        y = y.flatten(1)
        return self.head(y)

    @property
    def output_size(self) -> int:
        return self._output_size


class ActorCriticDepth(nn.Module):
    """Actor-Critic with CNN depth encoder (no LSTM).

    Architecture:
        depth_image (N, stacked_frames, H, W) → CNN → depth_feat (N, output_size)
        proprioception (N, D_prop) ──────────────────────────┘
        Concatenate → MLP actor / MLP critic
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        # --- depth encoder params ---
        depth_encoder_channels=[4],
        depth_encoder_kernel_sizes=[3],
        depth_encoder_strides=[1],
        depth_encoder_paddings=[1],
        depth_encoder_hidden_sizes=[256, 256],
        depth_encoder_output_size=128,
        depth_encoder_nonlinearity="ReLU",
        depth_encoder_use_maxpool=True,
        depth_obs_key="depth_image",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticDepth.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.depth_obs_key = depth_obs_key
        self.obs_groups = obs_groups

        # --- Identify depth image shape (N, stacked_frames, H, W) ---
        depth_shape = obs[self.depth_obs_key].shape
        assert len(depth_shape) == 4, f"Depth image must be 4D (N, C, H, W), got {depth_shape}"
        _, depth_c, depth_h, depth_w = depth_shape

        # --- Build depth CNN encoder ---
        self.depth_encoder = Conv2dDepthEncoder(
            in_channels=depth_c,
            in_height=depth_h,
            in_width=depth_w,
            channels=depth_encoder_channels,
            kernel_sizes=depth_encoder_kernel_sizes,
            strides=depth_encoder_strides,
            paddings=depth_encoder_paddings,
            hidden_sizes=depth_encoder_hidden_sizes,
            output_size=depth_encoder_output_size,
            nonlinearity=depth_encoder_nonlinearity,
            use_maxpool=depth_encoder_use_maxpool,
        )
        depth_feat_dim = self.depth_encoder.output_size
        print(f"Depth CNN encoder: input (C={depth_c}, H={depth_h}, W={depth_w}) → output {depth_feat_dim}")

        # --- Compute proprioception dimensions (exclude depth_image) ---
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            if obs_group == self.depth_obs_key:
                continue
            assert len(obs[obs_group].shape) == 2, (
                f"Observation group '{obs_group}' must be 2D (N, D), got {obs[obs_group].shape}"
            )
            num_actor_obs += obs[obs_group].shape[-1]
        num_actor_obs += depth_feat_dim  # add encoded depth features

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            if obs_group == self.depth_obs_key:
                continue
            assert len(obs[obs_group].shape) == 2, (
                f"Critic observation group '{obs_group}' must be 2D (N, D), got {obs[obs_group].shape}"
            )
            num_critic_obs += obs[obs_group].shape[-1]
        num_critic_obs += depth_feat_dim

        print(f"Actor encoded obs dim: {num_actor_obs}")
        print(f"Critic encoded obs dim: {num_critic_obs}")

        # --- Actor ---
        self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()
        print(f"Actor MLP: {self.actor}")

        # --- Critic ---
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()
        print(f"Critic MLP: {self.critic}")

        # --- Action noise ---
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _encode_depth(self, depth_img: torch.Tensor) -> torch.Tensor:
        """Encode stacked depth image through CNN. Input: (N, C, H, W)."""
        return self.depth_encoder(depth_img)

    def _build_obs(self, obs, group_names):
        """Encode depth image, concatenate with proprioception."""
        depth_img = obs[self.depth_obs_key]  # (N, C, H, W) or (T, N, C, H, W) in recurrent case
        depth_feat = self._encode_depth(depth_img)  # (N, feat_dim) or (T, N, feat_dim)

        other_list = []
        for obs_group in group_names:
            if obs_group == self.depth_obs_key:
                continue
            other_list.append(obs[obs_group])
        other = torch.cat(other_list, dim=-1)
        return torch.cat([other, depth_feat], dim=-1)

    def get_actor_obs(self, obs):
        return self._build_obs(obs, self.obs_groups["policy"])

    def get_critic_obs(self, obs):
        return self._build_obs(obs, self.obs_groups["critic"])

    def update_distribution(self, obs):
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        std = torch.clamp(std, min=1e-6, max=10.0)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self.actor(obs)

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
