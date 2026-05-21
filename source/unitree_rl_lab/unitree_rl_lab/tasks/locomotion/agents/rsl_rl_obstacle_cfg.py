from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


# ==============================================================================
# 1) Obstacle-specific configuration classes
# ==============================================================================

@configclass
class RslRlObstacleActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-Critic config with height scan + LSTM for obstacle avoidance.

    Extends PPO actor-critic configuration with:
    - LSTM hidden size
    - Terrain encoder dimensions (disabled at runtime)
    """

    class_name: str = "ActorCriticPerception"

    lstm_hidden_size: int = 256
    terrain_encoder_dims: list[int] = [128]


# ==============================================================================
# 2) Runner configuration for training
# ==============================================================================

@configclass
class UnitreeObstacleRunnerCfg(RslRlOnPolicyRunnerCfg):
    # ============== WandB Configuration ===========
    logger = "wandb"
    wandb_project = "Unitree_g1_Velocity_Obstacle"
    run_name = "Obstacle_Run"
    experiment_name = "unitree_obstacle"
    # ==============================================

    class_name: str = "OnPolicyRunner"

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    empirical_normalization = False

    # Policy configuration (MLP, no LSTM)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )

    # LSTM 备选方案（解封下面 + 注释上面即可切换为 LSTM 网络）
    # policy = RslRlObstacleActorCriticCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[256, 128],
    #     critic_hidden_dims=[256, 128],
    #     activation="elu",
    #     noise_std_type="log",
    #     lstm_hidden_size=256,
    #     terrain_encoder_dims=[128],
    # )

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
