import torch
import torch.nn as nn

from rsl_rl.networks import MLP
from rsl_rl.modules.actor_critic import ActorCritic


class ActorCriticNP3O(ActorCritic):
    """
    N-P3O 专用 Actor-Critic 网络。

    在 PPO 的 ActorCritic 基础上：
    1) 保留原有 actor（策略网络）
    2) 保留原有 reward critic（V_r）
    3) 新增多头成本价值函数：
        - cost_critic：输出所有成本的价值估计（默认2个：关节限位、碰撞）
    """

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        cost_critic_hidden_dims=[256, 256, 256],# 新增超参数：cost critic 隐层结构（需写入 cfg）
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        # ------------------------------------------------------------
        # 1. 先构建 PPO 原生的 Actor + Reward Critic
        #    包括：
        #    - actor 网络
        #    - critic (reward value V_r)
        #    - obs normalizer
        #    - action noise
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # 2. 计算 critic 输入维度（和 reward critic 完全一致）
        #    NP3O 中：
        #    - reward critic
        #    - cost critic
        #    使用同一份 critic observation
        # ------------------------------------------------------------
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            num_critic_obs += obs[obs_group].shape[-1]

        # ------------------------------------------------------------
        # 3. 新增：多头成本价值函数
        # ------------------------------------------------------------

        # 从 kwargs 中获取成本数量，默认为 2（保持向后兼容）
        num_costs = kwargs.get('num_costs', 2)
        self.num_costs = num_costs

        # 多头成本价值函数：共享隐藏层，输出所有成本的价值估计
        # 数学：V_c(s) = [V_c1(s), V_c2(s), ..., V_cC(s)]，其中 C = num_costs
        self.cost_critic = MLP(
            num_critic_obs,           # 输入维度
            num_costs,                # 输出维度 = 成本约束数量
            cost_critic_hidden_dims,  # 共享隐藏层结构
            activation,
        )

        print(f"[NP3O] 多头成本价值函数: 输入={num_critic_obs}, 输出={num_costs}, 隐藏层={cost_critic_hidden_dims}")
        print("[NP3O] Cost Critic:", self.cost_critic)

    # ======================================================================
    # 新增接口函数（供 NP3O 算法调用）
    # ======================================================================

    # 新增：reward value 评估函数（显式命名，语义更清晰）
    # 作用：估计奖励回报的状态价值 V_r(s)
    def evaluate_reward(self, obs):
        return super().evaluate(obs)

    # 新增：同时评估所有 cost value
    # 作用：
    #   给定 critic observation，返回所有成本的价值估计
    #   输出形状: (batch_size, num_costs)，其中 num_costs = 2（默认：关节限位、碰撞）
    def evaluate_costs(self, obs):

        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)

        # 单次前向传播，获取所有成本的价值估计
        # 形状: (batch_size, num_costs)
        all_costs = self.cost_critic(critic_obs)

        return all_costs

    # 新增：单独评估「关节限位」cost value（成本索引 0）
    # 作用：
    #   在算法中构造 cost-1 的 GAE / value loss / 日志时调用
    def evaluate_cost_1(self, obs):
        all_costs = self.evaluate_costs(obs)
        # 提取第一个成本，保持二维形状 (N, 1)
        return all_costs[:, 0:1]

    # 新增：单独评估「碰撞」cost value（成本索引 1）
    # 作用：
    #   在算法中构造 cost-2 的 GAE / value loss / 日志时调用
    def evaluate_cost_2(self, obs):
        all_costs = self.evaluate_costs(obs)
        # 提取第二个成本，保持二维形状 (N, 1)
        return all_costs[:, 1:2]

    # 新增：通用方法，获取第 i 个成本的价值估计
    # 作用：
    #   用于扩展更多成本约束时的通用接口
    def evaluate_cost_i(self, obs, i: int):
        """获取第 i 个成本的价值估计（0-based 索引）"""
        all_costs = self.evaluate_costs(obs)
        return all_costs[:, i:i+1]  # 保持二维形状 (N, 1)
