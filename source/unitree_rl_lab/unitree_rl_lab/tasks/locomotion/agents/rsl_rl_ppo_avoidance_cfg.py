from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlAvoidanceActorCriticCfg(RslRlPpoActorCriticRecurrentCfg):
    class_name: str = "ActorCriticAvoidance"
    depth_obs_group_name: str = "perception"
    depth_image_shape: list[int] = [90, 160]
    depth_encoder_channels: list[int] = [16, 32, 64, 64]
    depth_encoder_hidden_dims: list[int] = [256, 128]
    lstm_hidden_size: int = 256


@configclass
class UnitreeAvoidancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    logger = "wandb"
    wandb_project = "Unitree_g1_Avoidance"
    run_name = "avoidance_lstm"
    experiment_name = "unitree_g1_29dof_velocity_avoidance"

    obs_groups = {
        "policy": ["policy", "proprio", "perception"],
        "critic": ["policy", "proprio", "perception", "critic"],
    }

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    empirical_normalization = False

    policy = RslRlAvoidanceActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        lstm_hidden_size=256,
        depth_obs_group_name="perception",
        depth_image_shape=[90, 160],
        depth_encoder_channels=[16, 32, 64, 64],
        depth_encoder_hidden_dims=[256, 128],
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
