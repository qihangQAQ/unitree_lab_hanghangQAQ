"""AME (Attention-Based Map Encoding) training configuration."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlAMEActorCriticCfg(RslRlPpoActorCriticCfg):
    """AME Actor-Critic configuration with terrain encoder."""

    class_name: str = "ActorCriticEncoder"

    # Terrain encoder parameters
    map_scan_dim: list[int] = [33, 21, 3]  # L=33, W=21, 3D coordinates
    mha_dim: int = 64  # MHA feature dimension
    num_heads: int = 16  # Number of attention heads
    cnn_downsample: bool = True  # Use CNN stride-based downsampling
    attach_global: bool = False  # Add max-pooled global feature


@configclass
class AMEPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """AME PPO Runner configuration."""

    # ============== WandB Configuration ===========
    logger = "wandb"
    wandb_project = "Unitree_g1_AME"
    run_name = "AME_Run"
    experiment_name = "unitree_ame"
    # ==============================================

    class_name: str = "OnPolicyRunner"

    # Basic training parameters
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 100
    empirical_normalization = False

    # Policy configuration with AME encoder
    policy = RslRlAMEActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        noise_std_type="scalar",
    )

    # Algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
