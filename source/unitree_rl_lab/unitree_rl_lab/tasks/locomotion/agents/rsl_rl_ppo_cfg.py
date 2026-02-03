# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):

    # ============== 新增wandb ===========
    logger = "wandb"                            # ← 开启 wandb
    wandb_project = "Unitree_g1_Hand_tracking"       # ← wandb 项目名
    run_name = "hanghangQAQ"                           # ← 自定义 run 名
    # ===================================
    
    num_steps_per_env = 24              # 每个环境在一次更新前采集的步数。
    max_iterations = 10000              # 最大训练迭代次数。
    save_interval = 100                 #模型保存频率。
    experiment_name = ""                # 实验的大类名称
    empirical_normalization = False     # 是否使用经验归一化
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,            # 初始探索噪声的标准差 (值越大，初期动作越随机/狂野)
        actor_hidden_dims=[512, 256, 128],# Actor (策略) 网络的隐藏层结构
        critic_hidden_dims=[512, 256, 128],# Critic (价值) 网络的隐藏层结构
        activation="elu",# 神经网络层的激活函数 (这里使用 ELU)
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,# 价值损失 (Value Loss) 在总损失中的权重系数
        use_clipped_value_loss=True,# 是否对价值损失进行截断 (防止 Critic 更新过猛，增加稳定性)
        clip_param=0.2,# PPO 的核心截断参数 (Epsilon，限制策略更新幅度，防止偏离太远)
        entropy_coef=0.01,# 熵系数 (值越大越鼓励探索，防止过早收敛到局部最优)
        num_learning_epochs=5,# 每次采集完数据后，重复利用这批数据学习的轮数
        num_mini_batches=4,# 每次更新时将数据切分成多少个 Mini-batch
        learning_rate=1.0e-3,# 学习率 (控制网络权重更新的步长)
        schedule="adaptive",# 学习率调度器 (adaptive 表示根据 KL 散度自动调整 LR)
        gamma=0.99,# 折扣因子 (0~1之间，越接近1代表机器人越重视未来的长期奖励)
        lam=0.95,# GAE Lambda 参数 (用于平衡优势估计的偏差与方差)
        desired_kl=0.01,# 期望的 KL 散度 (配合 adaptive 使用，若实际 KL 超过此值则降低学习率)
        max_grad_norm=1.0,# 梯度裁剪阈值 (防止梯度数值爆炸导致训练崩溃)
    )
