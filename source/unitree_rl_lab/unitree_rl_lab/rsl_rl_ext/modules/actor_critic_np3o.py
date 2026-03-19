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
    3) 新增两个 cost critic：
        - cost_critic_1：关节限位约束（joint limit cost）
        - cost_critic_2：碰撞约束（collision cost）
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
        # 3. 新增：Cost Critic 网络
        # ------------------------------------------------------------

        # 新增：关节限位 cost critic
        # 对应约束：joint position limits
        # 估计 V_c1(s) = E[∑ γ^t * cost_joint_limit_t]
        self.cost_critic_1 = MLP(
            num_critic_obs,
            1,
            cost_critic_hidden_dims,
            activation,
        )

        # 新增：碰撞 cost critic
        # 对应约束：robot-obstacle / robot-body collision
        # 估计 V_c2(s) = E[∑ γ^t * cost_collision_t]
        self.cost_critic_2 = MLP(
            num_critic_obs,
            1,
            cost_critic_hidden_dims,
            activation,
        )

        print("[NP3O] Cost Critic 1 (joint limit):", self.cost_critic_1)
        print("[NP3O] Cost Critic 2 (collision):", self.cost_critic_2)

    # ======================================================================
    # 新增接口函数（供 NP3O 算法调用）
    # ======================================================================

    # 新增：reward value 评估函数（显式命名，语义更清晰）
    # 作用：估计奖励回报的状态价值 V_r(s)
    def evaluate_reward(self, obs):
        return super().evaluate(obs)

    # 新增：同时评估两个 cost value
    # 作用：
    #   给定 critic observation，返回：
    #   - V_c1(s)：关节限位 cost 的 value
    #   - V_c2(s)：碰撞 cost 的 value
    def evaluate_costs(self, obs):

        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)

        vc1 = self.cost_critic_1(critic_obs)
        vc2 = self.cost_critic_2(critic_obs)

        return torch.cat([vc1, vc2], dim=-1)  # (N, 2)

    # 新增：单独评估「关节限位」cost value
    # 作用：
    #   在算法中构造 cost-1 的 GAE / value loss / 日志时调用
    def evaluate_cost_1(self, obs):
        vc1, _ = self.evaluate_costs(obs)
        return vc1

    # 新增：单独评估「碰撞」cost value
    # 作用：
    #   在算法中构造 cost-2 的 GAE / value loss / 日志时调用
    def evaluate_cost_2(self, obs):
        _, vc2 = self.evaluate_costs(obs)
        return vc2
