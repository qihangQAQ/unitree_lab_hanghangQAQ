# rsl_rl_np3o_cfg.py

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from dataclasses import dataclass, field


# ==============================================================================
# 1. 定义 N-P3O 特有的配置类 (Blueprint)
#    这里定义了算法需要的新参数结构
# ==============================================================================

@configclass
class RslRlNp3oActorCriticCfg(RslRlPpoActorCriticCfg):
    """
    继承自 PPO 的策略配置，增加了 Cost Critic (成本判别器) 的网络结构定义。
    """
    class_name: str = "ActorCriticNp3o"  # 对应稍后你要写的算法类名

    # [新增] Cost Critic 的网络隐藏层结构
    # 通常和 Value Critic 保持一致，或者稍微小一点
    cost_critic_hidden_dims: list[int] = [512, 256, 128]


    # [新增] Cost Critic 是否使用 softplus 激活函数输出
    # 论文建议 Cost 预测值应非负，Softplus 可以保证这一点
    cost_critic_output_activation: str = "softplus"
    cost_critic_obs_normalization: bool = False      # 是否为 Cost Critic 开启独立归一化

@configclass
class RslRlNp3oAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """
    继承自 PPO 的算法配置，增加了 N-P3O 特有的超参数。
    """
    class_name: str = "NP3O"  # 对应稍后你要写的算法类名

    # [新增] 惩罚系数 Kappa (κ)
    # 这是 N-P3O 中最重要的参数，控制违反约束时的惩罚力度
    kappa: float = 1.0

    # [新增] Cost Critic 的 Loss 权重
    # 类似于 value_loss_coef，用于平衡 Task Value 和 Cost Value 的学习速度
    cost_value_loss_coef: float = 0.5

    # [新增] 是否对 Cost Advantage 进行归一化
    # 这是 "N"-P3O 的核心 (Normalized)，论文建议设为 True 以解耦 Reward 和 Cost 的量级
    normalize_cost_advantage: bool = True

    # [新增] Cost Surrogate 的裁剪参数
    # 类似于 PPO 的 clip_param，防止 Cost 策略更新步幅过大
    cost_clip_param: float = 0.2


# ==============================================================================
# 2. 实例化具体的运行配置 (Implementation)
#    这里填入具体的数值，就像你之前的 rsl_rl_ppo_cfg.py
# ==============================================================================

@configclass
class UnitreeNp3oRunnerCfg(RslRlOnPolicyRunnerCfg):
    # ============== WandB 配置 (保持不变) ===========
    logger = "wandb"
    wandb_project = "Unitree_g1_Velocity_Constraint"  # 修改项目名以区分
    run_name = "NP3O_Test_Run"
    experiment_name = "unitree_np3o"
    # ==============================================

    # 基础运行参数
    class_name: str = "NP3ORunner"
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    empirical_normalization = False

    # --------------------------------------------------------
    # 策略网络配置 (使用上面定义的 Np3o 类)
    # --------------------------------------------------------
    policy = RslRlNp3oActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",

        # [N-P3O 新增参数配置]
        cost_critic_hidden_dims=[512, 256, 128],  # 独立的 Cost 网络
        cost_critic_output_activation="softplus",
    )

    # --------------------------------------------------------
    # 算法超参数配置 (使用上面定义的 Np3o 类)
    # --------------------------------------------------------
    algorithm = RslRlNp3oAlgorithmCfg(
        # --- PPO 原有参数 ---
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

        # --- [N-P3O 新增参数配置] ---
        kappa=1.0,  # 初始惩罚权重，论文说 N-P3O 对此不敏感，1.0 是个好起点
        cost_value_loss_coef=0.5,  # Cost Critic 的学习权重
        normalize_cost_advantage=True,  # 开启归一化
        cost_clip_param=0.2,  # Cost 裁剪
    )