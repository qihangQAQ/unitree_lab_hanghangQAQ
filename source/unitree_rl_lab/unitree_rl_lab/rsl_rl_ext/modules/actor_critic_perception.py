import torch
import torch.nn as nn

from rsl_rl.networks import EmpiricalNormalization, Memory
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent
from torch.distributions import Normal


class ActorCriticPerception(ActorCriticRecurrent):
    """
    Perception-enhanced Actor-Critic network with terrain encoder.

    This network extends ActorCriticRecurrent with terrain encoding capabilities:
    1) Terrain encoder: MLP that encodes 651D height map (31x21) to terrain features (128D)
    2) Shared terrain features for both actor and critic
    3) Proper integration with RNN memory modules for sequence processing

    Network pipeline:
    Raw observations → Split terrain/non-terrain → Terrain encoder (651→128) →
    Concatenate with non-terrain → RNN memory (LSTM) → MLP heads
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
        terrain_encoder_dims=[128],  # 编码器维度参数（输入651维，输出128维）
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
            terrain_encoder_dims: MLP dimensions for terrain encoder (input 651D, output is last dimension)
            activation: Activation function for MLP layers
            init_noise_std: Initial standard deviation for action noise
            noise_std_type: Type of noise parameterization ("scalar" or "log")
            **kwargs: Additional arguments passed to parent class
        """
        # Map our parameters to parent class parameters
        kwargs['rnn_hidden_dim'] = lstm_hidden_size
        kwargs['rnn_type'] = 'lstm'  # Explicitly use LSTM

        # Store terrain encoder dimensions before calling parent init
        # 编码器功能已注释掉，保留参数但不使用
        # self.terrain_encoder_dims = terrain_encoder_dims
        # self.terrain_input_dim = 187  # Fixed: 17x11 height map
        self.terrain_encoder_dims = terrain_encoder_dims  # 编码器维度参数
        self.terrain_input_dim = 651  # 31x21 height map

        # Calculate observation dimensions without terrain
        # 启用编码器后，需要减去原始地形维度
        self.actor_obs_dim_no_terrain = self._calculate_obs_dim_no_terrain(
            obs, obs_groups["policy"], terrain_dim=self.terrain_input_dim  # 651维地形输入
        )
        self.critic_obs_dim_no_terrain = self._calculate_obs_dim_no_terrain(
            obs, obs_groups["critic"], terrain_dim=self.terrain_input_dim  # 651维地形输入
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
        # 编码器输出设定 -- 按照默认配置 [256, 128]，输出维度为 128
        self.terrain_feat_dim = terrain_encoder_dims[-1]
        # self.terrain_encoder = None  # 编码器已禁用
        # self.terrain_feat_dim = 0    # 特征维度为0

        # Calculate encoded observation dimensions
        # 编码观测维度 = 非地形维度 + 地形特征维度
        self.encoded_actor_obs_dim = self.actor_obs_dim_no_terrain + self.terrain_feat_dim
        self.encoded_critic_obs_dim = self.critic_obs_dim_no_terrain + self.terrain_feat_dim

        print(f"[ActorCriticPerception] Original actor obs dim: {self._get_original_actor_obs_dim(obs, obs_groups)}")
        print(f"[ActorCriticPerception] Original critic obs dim: {self._get_original_critic_obs_dim(obs, obs_groups)}")
        print(f"[ActorCriticPerception] Actor obs dim (no terrain): {self.actor_obs_dim_no_terrain} (减去{self.terrain_input_dim}维地形输入)")
        print(f"[ActorCriticPerception] Critic obs dim (no terrain): {self.critic_obs_dim_no_terrain} (减去{self.terrain_input_dim}维地形输入)")
        print(f"[ActorCriticPerception] Terrain feature dim: {self.terrain_feat_dim} (编码器输出维度)")
        print(f"[ActorCriticPerception] Encoded actor obs dim: {self.encoded_actor_obs_dim} (非地形{self.actor_obs_dim_no_terrain} + 地形特征{self.terrain_feat_dim})")
        print(f"[ActorCriticPerception] Encoded critic obs dim: {self.encoded_critic_obs_dim} (非地形{self.critic_obs_dim_no_terrain} + 地形特征{self.terrain_feat_dim})")

        # Recreate memory modules with encoded observation dimensions
        self._recreate_memory_modules()

        # Recreate observation normalizers if needed
        self._recreate_observation_normalizers()

        print(f"[ActorCriticPerception] Terrain encoder: {self.terrain_encoder} (编码器已启用，输入{self.terrain_input_dim}维，输出{self.terrain_feat_dim}维)")
        print(f"[ActorCriticPerception] Memory_a input size updated to: {self.encoded_actor_obs_dim} (编码后观测输入LSTM)")
        print(f"[ActorCriticPerception] Memory_c input size updated to: {self.encoded_critic_obs_dim} (编码后观测输入LSTM)")

    def _calculate_obs_dim_no_terrain(self, obs, obs_group_names, terrain_dim=651):
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

    # 编码器设定
    def _build_terrain_encoder(self, encoder_dims, activation="elu"):
        """Build MLP terrain encoder (输入651维，输出128维)."""
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
        """Recreate memory modules with encoded observation dimensions (encoder enabled)."""
        print(f"[ActorCriticPerception._recreate_memory_modules] Starting (encoder enabled)...")
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

        print(f"[ActorCriticPerception._recreate_memory_modules] Creating new memory_a with input_size={self.encoded_actor_obs_dim} (encoded obs, encoder enabled)")
        print(f"[ActorCriticPerception._recreate_memory_modules] Creating new memory_c with input_size={self.encoded_critic_obs_dim} (encoded obs, encoder enabled)")

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
        """Recreate observation normalizers with encoded observation dimensions (encoder enabled)."""
        if self.actor_obs_normalization:
            # Replace the identity normalizer with empirical normalizer
            self.actor_obs_normalizer = EmpiricalNormalization(self.encoded_actor_obs_dim)

        if self.critic_obs_normalization:
            # Replace the identity normalizer with empirical normalizer
            self.critic_obs_normalizer = EmpiricalNormalization(self.encoded_critic_obs_dim)

    # 构建Actor观测 -- 地形编码后返回
    def get_actor_obs(self, obs):
        """
        Get actor observations with terrain encoding.

        Returns encoded observations (non-terrain + terrain features) to Actor network.
        """
        # Get raw concatenated observations from parent class
        raw_obs = super().get_actor_obs(obs)

        # 分割地形和非地形部分
        non_terrain_dim = self.actor_obs_dim_no_terrain
        terrain_input = raw_obs[..., -self.terrain_input_dim:]
        non_terrain = raw_obs[..., :non_terrain_dim]  # 前面的非地形观测

        # 编码地形
        terrain_features = self.terrain_encoder(terrain_input)

        # 拼接编码后观测
        encoded_obs = torch.cat([non_terrain, terrain_features], dim=-1)
        return encoded_obs

    # 构建Critic观测 -- 地形编码后返回
    def get_critic_obs(self, obs):
        """
        Get critic observations with terrain encoding.

        Returns encoded observations (non-terrain + terrain features) to Critic network.
        """
        # Get raw concatenated observations from parent class
        raw_obs = super().get_critic_obs(obs)

        # 分割地形和非地形部分
        non_terrain_dim = self.critic_obs_dim_no_terrain
        terrain_input = raw_obs[..., -self.terrain_input_dim:]  # 最后651维地形输入
        non_terrain = raw_obs[..., :non_terrain_dim]  # 前面的非地形观测

        # 编码地形
        terrain_features = self.terrain_encoder(terrain_input)

        # 拼接编码后观测
        encoded_obs = torch.cat([non_terrain, terrain_features], dim=-1)
        return encoded_obs

    # Note: All other methods (act, evaluate, update_distribution, etc.) are inherited
    # from ActorCriticRecurrent and will automatically use our overridden
    # get_actor_obs and get_critic_obs methods.