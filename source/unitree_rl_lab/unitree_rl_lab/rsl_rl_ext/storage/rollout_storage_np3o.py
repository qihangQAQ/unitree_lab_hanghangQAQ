from __future__ import annotations

import torch

from rsl_rl.storage.rollout_storage import RolloutStorage


class NP3ORolloutStorage(RolloutStorage):
    """
    NP3O 专用 RolloutStorage。

    在 PPO RolloutStorage 基础上新增：
      - costs（来自 env）
      - cost_values（来自 cost critics）
      - cost_returns / cost_advantages（用于 NP3O 的 actor/critic 更新）
    """

    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            # ==================== 新增：cost 存储 ====================
            # 新增：costs -- env 输出的每步 cost（shape: [num_envs, num_costs]）
            self.costs = None
            # 新增：cost_values -- cost critics 的 Vc(s)（shape: [num_envs, num_costs]）
            self.cost_values = None
            # =======================================================

        def clear(self):
            super().clear()
            self.costs = None
            self.cost_values = None

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs,
        actions_shape,
        device="cpu",

        # ==================== 新增：NP3O 超参数/配置 ====================
        # 新增超参数：num_costs -- cost/约束数量（你这里是 2：关节限位、碰撞；需要写入 cfg）
        num_costs: int = 2,
        # =============================================================
    ):
        super().__init__(
            training_type=training_type,
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs=obs,
            actions_shape=actions_shape,
            device=device,
        )

        # ==================== 新增：cost buffer 初始化 ====================
        self.num_costs = int(num_costs)

        # env step cost： (T, N, C)
        self.costs = torch.zeros(num_transitions_per_env, num_envs, self.num_costs, device=self.device)

        if training_type == "rl":
            # cost critics values： (T, N, C)
            self.cost_values = torch.zeros(num_transitions_per_env, num_envs, self.num_costs, device=self.device)
            # cost returns / advantages： (T, N, C)
            self.cost_returns = torch.zeros(num_transitions_per_env, num_envs, self.num_costs, device=self.device)
            self.cost_advantages = torch.zeros(num_transitions_per_env, num_envs, self.num_costs, device=self.device)
        # ================================================================

    def add_transitions(self, transition: Transition):
        # 先走 PPO 原有逻辑（obs/actions/reward/done/value/logp/mu/sigma/hidden_states）
        super().add_transitions(transition)

        # super() 已经把 step += 1 做掉了，所以这里要用 self.step-1 写入
        t = self.step - 1

        # ==================== 新增：写入 cost / cost_values ====================
        # 新增：存储 env 输出的 costs
        if transition.costs is None:
            raise ValueError("NP3O Transition.costs is None. Env must provide extras['costs'] and runner must pass it.")
        self.costs[t].copy_(transition.costs)

        # 新增：存储 cost critics 的 value 预测（仅 RL 训练需要）
        if self.training_type == "rl":
            if transition.cost_values is None:
                raise ValueError("NP3O Transition.cost_values is None. ActorCritic must output cost values Vc(s).")
            self.cost_values[t].copy_(transition.cost_values)
        # ====================================================================

    # ==================== 新增：计算 cost 的 returns/advantages ====================
    # 新增：compute_cost_returns -- 对每个约束 i 分别做 GAE，得到 cost_returns & cost_advantages
    def compute_cost_returns(
        self,
        last_cost_values: torch.Tensor,
        cost_gamma: float,
        cost_lam: float,
        normalize_cost_advantage: bool = True,
    ):
        """
        last_cost_values: shape (num_envs, num_costs)
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        advantage = torch.zeros(self.num_envs, self.num_costs, device=self.device)

        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_cost_values
            else:
                next_values = self.cost_values[step + 1]  # (N,C)

            next_is_not_terminal = 1.0 - self.dones[step].float()  # (N,1)

            # TD error per cost:
            # delta_c = c_t + gamma_c * Vc(s_{t+1}) - Vc(s_t)
            delta = self.costs[step] + next_is_not_terminal * cost_gamma * next_values - self.cost_values[step]

            # GAE per cost:
            advantage = delta + next_is_not_terminal * cost_gamma * cost_lam * advantage

            # return_c = advantage_c + Vc(s_t)
            self.cost_returns[step] = advantage + self.cost_values[step]

        # cost advantages
        self.cost_advantages = self.cost_returns - self.cost_values

        # 可选归一化（通常每个 cost 单独归一化，避免量纲不同）
        # if normalize_cost_advantage:
        #     mean = self.cost_advantages.mean(dim=(0, 1), keepdim=True)  # (1,1,C)
        #     std = self.cost_advantages.std(dim=(0, 1), keepdim=True) + 1e-8
        #     self.cost_advantages = (self.cost_advantages - mean) / std
        if normalize_cost_advantage:
            # 修改：将均值和标准差保存为类属性，供后续 update() 使用
            self.cost_advantages_mean = self.cost_advantages.mean(dim=(0, 1), keepdim=True)
            self.cost_advantages_std = self.cost_advantages.std(dim=(0, 1), keepdim=True) + 1e-8
            self.cost_advantages = (self.cost_advantages - self.cost_advantages_mean) / self.cost_advantages_std
        else:
            # 如果不归一化，设定默认均值为 0，方差为 1，保持数学公式的通用性
            self.cost_advantages_mean = torch.zeros((1, 1, self.num_costs), device=self.device)
            self.cost_advantages_std = torch.ones((1, 1, self.num_costs), device=self.device)
    # ==========================================================================

    # ==================== 新增：mini-batch 产出 cost batch ====================
    # 新增：mini_batch_generator -- 在 PPO yield 的基础上，多 yield costs/cost_values/cost_returns/cost_advantages
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # PPO core flatten
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # ==================== 新增：cost flatten ====================
        costs = self.costs.flatten(0, 1)                     # (B,C)
        cost_values = self.cost_values.flatten(0, 1)         # (B,C)
        cost_returns = self.cost_returns.flatten(0, 1)       # (B,C)
        cost_advantages = self.cost_advantages.flatten(0, 1) # (B,C)
        # ==========================================================

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]

                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # ==================== 新增：cost batch ====================
                costs_batch = costs[batch_idx]                         # (MB,C)
                target_cost_values_batch = cost_values[batch_idx]      # (MB,C)
                cost_returns_batch = cost_returns[batch_idx]           # (MB,C)
                cost_advantages_batch = cost_advantages[batch_idx]     # (MB,C)
                # ==========================================================

                yield (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (None, None),
                    None,
                    # ==================== 新增：额外返回 cost 信息 ====================
                    costs_batch,
                    target_cost_values_batch,
                    cost_advantages_batch,
                    cost_returns_batch,
                    # ==========================================================
                )
    # ======================================================================
