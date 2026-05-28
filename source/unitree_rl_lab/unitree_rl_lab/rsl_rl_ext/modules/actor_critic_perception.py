import torch
import torch.nn as nn

from rsl_rl.networks import EmpiricalNormalization, Memory
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent
from torch.distributions import Normal


class ActorCriticPerception(ActorCriticRecurrent):
    """
    Perception-enhanced Actor-Critic network with terrain encoder + LSTM.

    Network pipeline:
    Raw obs → split into [proprioception | height_map(187D)]
                → terrain encoder (MLP) → terrain features
                → concat [proprioception | terrain_features]
                → LSTM memory → MLP heads (actor/critic)
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
        terrain_encoder_dims=[128],  # Terrain encoder MLP hidden dims, final dim = terrain feature dim
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
            terrain_encoder_dims: MLP dimensions for terrain encoder (e.g. [256, 128])
            activation: Activation function for MLP layers
            init_noise_std: Initial standard deviation for action noise
            noise_std_type: Type of noise parameterization ("scalar" or "log")
            **kwargs: Additional arguments passed to parent class
        """
        # Map our parameters to parent class parameters
        kwargs['rnn_hidden_dim'] = lstm_hidden_size
        kwargs['rnn_type'] = 'lstm'  # Explicitly use LSTM

        # Store terrain encoder parameters
        self.terrain_encoder_dims = terrain_encoder_dims
        self.terrain_input_dim = 187  # Fixed: 17x11 height map (187 scan dots)

        # DEBUG: inspect obs_groups structure and obs shapes
        print(f"[DEBUG] obs keys: {list(obs.keys())}")
        print(f"[DEBUG] obs_groups['policy']: {obs_groups['policy']}")
        print(f"[DEBUG] obs_groups['critic']: {obs_groups['critic']}")
        for g in obs_groups["policy"]:
            print(f"[DEBUG] obs['{g}'].shape: {obs[g].shape}")
        for g in obs_groups["critic"]:
            print(f"[DEBUG] obs['{g}'].shape: {obs[g].shape}")

        # Calculate observation dimensions WITHOUT terrain (proprioception only)
        self.actor_obs_dim_no_terrain = self._calculate_obs_dim_no_terrain(
            obs, obs_groups["policy"], terrain_dim=187
        )
        self.critic_obs_dim_no_terrain = self._calculate_obs_dim_no_terrain(
            obs, obs_groups["critic"], terrain_dim=187
        )

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

        # Build terrain encoder (shared between actor and critic)
        self.terrain_encoder = self._build_terrain_encoder(terrain_encoder_dims, activation)
        self.terrain_feat_dim = terrain_encoder_dims[-1]

        # Calculate encoded observation dimensions (proprioception + terrain features)
        self.encoded_actor_obs_dim = self.actor_obs_dim_no_terrain + self.terrain_feat_dim
        self.encoded_critic_obs_dim = self.critic_obs_dim_no_terrain + self.terrain_feat_dim

        print(f"[ActorCriticPerception] Original actor obs dim: {self._get_original_actor_obs_dim(obs, obs_groups)}")
        print(f"[ActorCriticPerception] Original critic obs dim: {self._get_original_critic_obs_dim(obs, obs_groups)}")
        print(f"[ActorCriticPerception] Actor obs dim (no terrain): {self.actor_obs_dim_no_terrain}")
        print(f"[ActorCriticPerception] Critic obs dim (no terrain): {self.critic_obs_dim_no_terrain}")
        print(f"[ActorCriticPerception] Terrain feat dim: {self.terrain_feat_dim}")
        print(f"[ActorCriticPerception] Encoded actor obs dim (→LSTM): {self.encoded_actor_obs_dim}")
        print(f"[ActorCriticPerception] Encoded critic obs dim (→LSTM): {self.encoded_critic_obs_dim}")

        # Recreate memory modules with encoded observation dimensions
        self._recreate_memory_modules()

        # Recreate observation normalizers if needed
        self._recreate_observation_normalizers()

        print(f"[ActorCriticPerception] Terrain encoder: {self.terrain_encoder}")
        print(f"[ActorCriticPerception] Memory_a input size (→LSTM): {self.encoded_actor_obs_dim}")
        print(f"[ActorCriticPerception] Memory_c input size (→LSTM): {self.encoded_critic_obs_dim}")

    def _calculate_obs_dim_no_terrain(self, obs, obs_group_names, terrain_dim=187):
        """Calculate total observation dimension (terrain_dim=0 when encoder disabled)."""
        total_dim = 0
        for obs_group in obs_group_names:
            # Follow the same logic as parent class
            assert len(obs[obs_group].shape) == 2, "The ActorCriticRecurrent module only supports 1D observations."
            total_dim += obs[obs_group].shape[-1]

        # Verify that we have at least terrain_dim dimensions
        if total_dim < terrain_dim:
            raise ValueError(
                f"Total observation dimension ({total_dim}) is less than terrain input dimension ({terrain_dim})"
            )

        return total_dim - terrain_dim

    def _get_original_actor_obs_dim(self, obs, obs_groups):
        """Get original actor observation dimension (before encoding)."""
        return self._calculate_obs_dim_no_terrain(obs, obs_groups["policy"], terrain_dim=0)

    def _get_original_critic_obs_dim(self, obs, obs_groups):
        """Get original critic observation dimension (before encoding)."""
        return self._calculate_obs_dim_no_terrain(obs, obs_groups["critic"], terrain_dim=0)

    # 编码器设定（已禁用）
    def _build_terrain_encoder(self, encoder_dims, activation="elu"):
        """Build MLP terrain encoder (not used - encoder functionality disabled)."""
        layers = []
        input_dim = self.terrain_input_dim

        for _, hidden_dim in enumerate(encoder_dims):
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

    def update_distribution(self, *args, **kwargs):
        """
        重写 update_distribution 方法：
        保留父类 ActorCriticRecurrent 的所有前向传播（包括 LSTM 的 hidden state 处理），
        仅在最后一步对标准差 std 进行 clamp 限制，以防止训练后期数值爆炸或 NaN。
        """
        # 1. 调用父类方法，完成前向传播并生成基础的 self.distribution
        super().update_distribution(*args, **kwargs)

        # 2. 从父类计算好的分布中提取当前的 mean
        mean = self.distribution.mean

        # 3. 加入你的 clamp 限制逻辑来计算 std
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
            std = torch.clamp(std, min=1e-6)  # scalar 也加个保险
        elif self.noise_std_type == "log":
            #  保险1：限制 log_std 防止指数爆炸
            log_std_clamped = torch.clamp(self.log_std, min=-5.0, max=2.0)
            
            # 还原为标准差 (也就是建议里的 scale)
            std = torch.exp(log_std_clamped).expand_as(mean)
            
            #  保险2：硬性限制最终的 std (scale) 的绝对物理范围
            std = torch.clamp(std, min=1e-6, max=10.0)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # 4. 用同样的 mean 和限制过的 std 重新构建分布，覆盖掉父类生成的分布
        self.distribution = Normal(mean, std)


    def _recreate_memory_modules(self):
        """Recreate memory modules with original observation dimensions (encoder disabled)."""
        print(f"[ActorCriticPerception._recreate_memory_modules] Starting (encoder disabled)...")
        print(f"[ActorCriticPerception._recreate_memory_modules] Original memory_a type: {type(self.memory_a)}")

        # Get RNN parameters from existing memory modules
        # Determine rnn type by checking the class of self.memory_a.rnn
        if hasattr(self.memory_a, 'rnn'):
            rnn = self.memory_a.rnn
            print(f"[ActorCriticPerception._recreate_memory_modules] Original rnn type: {type(rnn)}")
            if isinstance(rnn, nn.LSTM):
                rnn_type = "lstm"
            elif isinstance(rnn, nn.GRU):
                rnn_type = "gru"
            else:
                rnn_type = "lstm"  # default

            rnn_num_layers = rnn.num_layers
            rnn_hidden_dim = rnn.hidden_size
            print(f"[ActorCriticPerception._recreate_memory_modules] Original rnn params: num_layers={rnn_num_layers}, hidden_size={rnn_hidden_dim}")
        else:
            # Fallback to defaults (from parent class initialization)
            rnn_type = "lstm"
            rnn_num_layers = 1
            rnn_hidden_dim = 256  # default from ActorCriticRecurrent
            print(f"[ActorCriticPerception._recreate_memory_modules] Using defaults: num_layers={rnn_num_layers}, hidden_size={rnn_hidden_dim}")

        print(f"[ActorCriticPerception._recreate_memory_modules] Creating new memory_a with input_size={self.encoded_actor_obs_dim} (original obs, encoder disabled)")
        print(f"[ActorCriticPerception._recreate_memory_modules] Creating new memory_c with input_size={self.encoded_critic_obs_dim} (original obs, encoder disabled)")

        # Recreate actor memory with encoded observation dimension
        self.memory_a = Memory(
            input_size=self.encoded_actor_obs_dim,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim
        )

        # Recreate critic memory with encoded observation dimension
        self.memory_c = Memory(
            input_size=self.encoded_critic_obs_dim,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim
        )

        # Copy device from existing parameters
        if hasattr(self, '_parameters') and len(self._parameters) > 0:
            device = next(self.parameters()).device
            self.memory_a.to(device)
            self.memory_c.to(device)
            print(f"[ActorCriticPerception._recreate_memory_modules] Memory modules moved to device: {device}")

        print(f"[ActorCriticPerception._recreate_memory_modules] New memory_a type: {type(self.memory_a)}")
        print(f"[ActorCriticPerception._recreate_memory_modules] New memory_a.rnn type: {type(self.memory_a.rnn)}")
        print(f"[ActorCriticPerception._recreate_memory_modules] New memory_a.rnn.input_size: {self.memory_a.rnn.input_size}")
        print(f"[ActorCriticPerception._recreate_memory_modules] New memory_c.rnn.input_size: {self.memory_c.rnn.input_size}")

    def _recreate_observation_normalizers(self):
        """Recreate observation normalizers with original observation dimensions (encoder disabled)."""
        if self.actor_obs_normalization:
            # Replace the identity normalizer with empirical normalizer
            self.actor_obs_normalizer = EmpiricalNormalization(self.encoded_actor_obs_dim)

        if self.critic_obs_normalization:
            # Replace the identity normalizer with empirical normalizer
            self.critic_obs_normalizer = EmpiricalNormalization(self.encoded_critic_obs_dim)

    def get_actor_obs(self, obs):
        """
        Split observations into proprioception + terrain, encode terrain, then concatenate.

        Returns: [proprioception | terrain_features] → fed to LSTM memory
        """
        raw_obs = super().get_actor_obs(obs)
        proprio_obs = raw_obs[..., :self.actor_obs_dim_no_terrain]
        terrain_obs = raw_obs[..., self.actor_obs_dim_no_terrain:]
        terrain_features = self.terrain_encoder(terrain_obs)
        return torch.cat([proprio_obs, terrain_features], dim=-1)

    def get_critic_obs(self, obs):
        """
        Split observations into proprioception + terrain, encode terrain, then concatenate.

        Returns: [proprioception | terrain_features] → fed to LSTM memory
        """
        raw_obs = super().get_critic_obs(obs)
        proprio_obs = raw_obs[..., :self.critic_obs_dim_no_terrain]
        terrain_obs = raw_obs[..., self.critic_obs_dim_no_terrain:]
        terrain_features = self.terrain_encoder(terrain_obs)
        return torch.cat([proprio_obs, terrain_features], dim=-1)

    # Note: All other methods (act, evaluate, update_distribution, etc.) are inherited
    # from ActorCriticRecurrent and will automatically use our overridden
    # get_actor_obs and get_critic_obs methods.