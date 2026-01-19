from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


# ==============================================================================
# 1) Blueprint：NP3O 特有的配置类（继承 PPO 的 cfg，补上 NP3O 新字段）
# ==============================================================================

@configclass
class RslRlNp3oActorCriticCfg(RslRlPpoActorCriticCfg):
    """
    NP3O policy 配置：
    - 继承 PPO policy cfg
    - 额外补充 cost critics 的网络结构
    """
    # 新增：policy 类名（必须与代码一致）
    class_name: str = "ActorCriticNP3O"  # 你的最新类名

    # 新增超参数：cost_critic_hidden_dims -- cost critic 隐层结构（需写入 cfg）
    # 说明：NP3OActorCritic 新增了 cost_critic_1/2，这里控制它们的 MLP 结构
    cost_critic_hidden_dims: list[int] = [512, 256, 128]


@configclass
class RslRlNp3oAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """
    NP3O algorithm 配置：
    - 继承 PPO algorithm cfg
    - 增加 cost / 约束相关超参数
    """
    # 新增：algorithm 类名（必须与代码一致）
    # class_name: str = "NP3O"

    # ==================== NP3O 新增超参数（需写入 cfg） ====================
    # 新增超参数：num_costs -- 约束数量（关节限位、碰撞=2）
    num_costs: int = 2

    # 新增超参数：cost_gamma -- cost 折扣因子（用于 cost returns/GAE）
    cost_gamma: float = 0.99
    # 新增超参数：cost_lam -- cost GAE lambda
    cost_lam: float = 0.95

    # 新增超参数：eps_cost -- 每个约束的阈值 epsilon_i（长度=num_costs）
    # 说明：顺序必须与 env extras["costs"] 的维度一致：0=关节限位，1=碰撞
    eps_cost: list[float] = [0.0, 0.0]

    # 新增超参数：kappa_cost -- 每个约束的权重 kappa_i（长度=num_costs）
    kappa_cost: list[float] = [1.0, 1.0]

    # 新增超参数：cost_value_loss_coef -- cost critic loss 系数
    cost_value_loss_coef: float = 1.0

    # 新增超参数：normalize_cost_advantage -- 是否对每个 cost advantage 单独归一化
    normalize_cost_advantage: bool = True

    # 新增超参数：use_clipped_cost_value_loss -- cost critic 是否使用 value clipping（复用 clip_param）
    use_clipped_cost_value_loss: bool = True
    # ======================================================================


# ==============================================================================
# 2) Implementation：具体运行配置（你实际训练时用的 RunnerCfg）
# ==============================================================================

@configclass
class UnitreeNp3oRunnerCfg(RslRlOnPolicyRunnerCfg):
    # ============== WandB 配置（沿用你原风格） ===========
    logger = "wandb"
    wandb_project = "Unitree_g1_Position_Constraint"
    run_name = "NP3O_Run"
    experiment_name = "unitree_np3o"
    # ===================================================

    # 新增：runner 类名（必须与代码一致）
    class_name: str = "NP3ORunner"

    # 基础运行参数（与你 PPO 版保持一致即可）
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    empirical_normalization = False

    # policy：换成 NP3O policy cfg
    policy = RslRlNp3oActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",

        # 新增超参数：cost_critic_hidden_dims（需写入 cfg）
        cost_critic_hidden_dims=[512, 256, 128],
    )

    # algorithm：换成 NP3O algorithm cfg
    algorithm = RslRlNp3oAlgorithmCfg(
        # --- PPO 原有参数（保留） ---
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,

        # --- NP3O 新增参数（需写入 cfg） ---
        num_costs=2,
        cost_gamma=0.99,
        cost_lam=0.95,
        eps_cost=[0.0, 0.0],
        kappa_cost=[1.0, 1.0],
        cost_value_loss_coef=1.0,
        normalize_cost_advantage=True,
        use_clipped_cost_value_loss=True,
    )
