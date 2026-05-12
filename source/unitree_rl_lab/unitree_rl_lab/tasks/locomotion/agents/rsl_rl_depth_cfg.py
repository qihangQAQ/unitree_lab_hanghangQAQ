from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlDepthActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-Critic config with depth CNN encoder (no LSTM)."""

    class_name: str = "ActorCriticDepth"

    # Depth encoder params (replicate InstinctLab parkour config)
    depth_encoder_channels: list[int] = [4]
    depth_encoder_kernel_sizes: list[int] = [3]
    depth_encoder_strides: list[int] = [1]
    depth_encoder_paddings: list[int] = [1]
    depth_encoder_hidden_sizes: list[int] = [256, 256]
    depth_encoder_output_size: int = 128
    depth_encoder_nonlinearity: str = "ReLU"
    depth_encoder_use_maxpool: bool = True
    depth_obs_key: str = "depth_image"


@configclass
class UnitreeDepthRunnerCfg(RslRlOnPolicyRunnerCfg):
    logger = "wandb"
    wandb_project = "Unitree_g1_Velocity_Depth"
    run_name = "Depth_Run"
    experiment_name = "unitree_depth"

    class_name: str = "OnPolicyRunner"

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    empirical_normalization = False

    policy = RslRlDepthActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
