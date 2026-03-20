# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable

# ==================== 新增：导入 NP3O 专用 storage ====================
# 新增：NP3ORolloutStorage -- 存 costs / cost_values / cost_returns / cost_advantages
from ..storage.rollout_storage_np3o import NP3ORolloutStorage
# ================================================================


class NP3O:
    """N-P3O algorithm (PPO + explicit long-term constraints via costs)."""

    def __init__(
        self,
        policy,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=0.001,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,

        # ==================== 新增：NP3O 超参数 ====================
        # 新增超参数：num_costs -- 约束数量 C（你这里=2：关节限位、碰撞；需写入 cfg）
        # 数学公式：C = 约束数量，对应论文中的约束数量
        # 参数含义：C 表示有多少个不同的成本约束需要同时优化
        # 变量对应：self.num_costs = C
        num_costs: int = 2,

        # 新增超参数：cost_gamma -- 成本折扣因子 γ_c（可与 reward gamma 相同；需写入 cfg）
        # 数学公式：γ_c ∈ [0,1)，用于计算成本回报 R_c^i = ∑_{t=0}^{∞} γ_c^t c_t^i
        # 参数含义：折扣未来成本的权重，γ_c越小越重视近期成本
        # 变量对应：self.cost_gamma = γ_c
        cost_gamma: float = 0.99,
        # 新增超参数：cost_lam -- 成本 GAE 的 λ_c 参数（需写入 cfg）
        # 数学公式：λ_c ∈ [0,1]，用于计算成本优势 A_c^i = GAE(γ_c, λ_c)
        # 参数含义：权衡偏差与方差的参数，λ_c=1 为高偏差低方差（MC估计），λ_c=0 为低偏差高方差（TD估计）
        # 变量对应：self.cost_lam = λ_c
        cost_lam: float = 0.95,

        # 新增超参数：eps_cost -- 每个约束的阈值 ε_i（ε_i在论文中常设为0）
        # 数学公式：ε_i ∈ ℝ，约束 i 的可接受成本上限，论文公式(2)：J^C_i(π) ≤ ε_i
        # 参数含义：约束 i 的最大允许折扣累计成本，ε_i=0 表示零成本约束
        # 变量对应：self.eps_cost[i] = ε_i
        eps_cost=None,

        # 新增超参数：kappa_cost -- 每个约束的约束惩罚权重κ_i (控制每个约束的惩罚强度)
        kappa_cost=None,

        # 新增超参数：cost_value_loss_coef -- cost critic 回归损失系数（需写入 cfg）
        cost_value_loss_coef: float = 1.0,

        # 新增超参数：normalize_cost_advantage -- 是否对每个 cost advantage 单独归一化（需写入 cfg）
        normalize_cost_advantage: bool = True,

        # 新增超参数：use_clipped_cost_value_loss -- 成本critic是否使用clipped loss（需写入 cfg）
        use_clipped_cost_value_loss: bool = True,
        # ==========================================================

        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components（保持与 PPO 一致）
        if rnd_cfg is not None:
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components（保持与 PPO 一致）
        if symmetry_cfg is not None:
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # core components
        self.policy = policy
        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = NP3ORolloutStorage.Transition()

        # PPO parameters（沿用）
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # ==================== 新增：NP3O 参数缓存 ====================
        self.num_costs = int(num_costs)
        self.cost_gamma = float(cost_gamma)
        self.cost_lam = float(cost_lam)
        self.cost_value_loss_coef = float(cost_value_loss_coef)
        self.normalize_cost_advantage = bool(normalize_cost_advantage)
        self.use_clipped_cost_value_loss = bool(use_clipped_cost_value_loss)

        # 论文公式(2)：设置阈值ε_i (默认全部为0)
        if eps_cost is None:
            eps_cost = torch.zeros(self.num_costs, device=self.device)
        # 论文公式(5)(7)：设置惩罚权重κ_i (默认全部为1)
        if kappa_cost is None:
            kappa_cost = torch.ones(self.num_costs, device=self.device)

        self.eps_cost = torch.as_tensor(eps_cost, device=self.device, dtype=torch.float32)      # (C,)
        self.kappa_cost = torch.as_tensor(kappa_cost, device=self.device, dtype=torch.float32) # (C,)
        # ==========================================================

    # ==================== 新增：init_storage 使用 NP3ORolloutStorage ====================
    # 新增：init_storage -- 替换为 NP3O 版 storage（多存 costs / cost values / cost GAE）
    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        self.storage = NP3ORolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
            num_costs=self.num_costs,  # 新增超参数：num_costs（需写入 cfg）
        )
    # ============================================================================

    def act(self, obs):
        # 与 PPO act 相同框架 :contentReference[oaicite:8]{index=8}
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs

        # ==================== 新增：记录 cost values（来自 cost critics） ====================
        # 新增：cost_values -- rollout 阶段额外保存 V_c_i(s)，用于 cost-GAE / cost critic loss
        # 兼容两种返回：tuple(vc1,vc2) 或 (N,C) tensor
        cost_vals = self.policy.evaluate_costs(obs)
        if isinstance(cost_vals, tuple):
            cost_vals = torch.cat([cv.detach() for cv in cost_vals], dim=-1)  # (N,C)
        else:
            cost_vals = cost_vals.detach()
        self.transition.cost_values = cost_vals
        # ========================================================================

        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        # 与 PPO 相同：更新 normalizer + 存 rewards/dones :contentReference[oaicite:9]{index=9}
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        # ==================== 新增：记录 env 输出的 costs ====================
        # 新增：costs -- env 每 step 输出的 costs（shape: [num_envs, num_costs]）
        if "costs" not in extras:
            raise KeyError("NP3O requires extras['costs'] from env (shape: [num_envs, num_costs]).")
        self.transition.costs = extras["costs"].to(self.device)
        # =====================================================================

        # ==================== 修正：Reward 与 Cost 的同步 Bootstrap ====================
        if "time_outs" in extras:
            # 1. Reward Bootstrap (你原有的代码)
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

            # 2. Cost Bootstrap (修正新增)
            # 提取 timeout 掩码，shape: (num_envs, 1)
            time_outs_mask = extras["time_outs"].unsqueeze(1).to(self.device).float()

            # transition.cost_values 包含通过 Cost Critic 评估的 V_c(s)
            # shape 广播：(N, C) * (N, 1) -> (N, C)
            # 注意：cost bootstrap 在环境成本基础上添加
            self.transition.costs += self.cost_gamma * self.transition.cost_values * time_outs_mask
        # ===============================================================================

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    # ==================== 新增：同时计算 reward returns 和 cost returns ====================
    # 新增：compute_returns -- reward 用 PPO 的 compute_returns，cost 用 compute_cost_returns
    # 公式：对于每个成本i，计算成本回报（cost return）和成本优势（cost advantage）
    #  成本回报：R_c^i = ∑_{l=0}^{∞} (γ_c)^l c_{t+l}^i，其中γ_c = cost_gamma
    #  成本优势：A_c^i = GAE(γ_c, λ_c) 使用成本TD误差 δ_c^i = c_t^i + γ_c * V_c^i(s_{t+1}) - V_c^i(s_t)
    #  具体实现在 NP3ORolloutStorage.compute_cost_returns 中
    def compute_returns(self, obs):
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

        # ---- cost side ----
        last_cost_values = self.policy.evaluate_costs(obs)
        if isinstance(last_cost_values, tuple):
            last_cost_values = torch.cat([cv.detach() for cv in last_cost_values], dim=-1)  # (N,C)
        else:
            last_cost_values = last_cost_values.detach()

        self.storage.compute_cost_returns(
            last_cost_values,
            self.cost_gamma,                       # 新增超参数：cost_gamma（需写入 cfg）
            self.cost_lam,                         # 新增超参数：cost_lam（需写入 cfg）
            normalize_cost_advantage=self.normalize_cost_advantage,  # 新增超参数：normalize_cost_advantage（需写入 cfg）
        )
    # ===============================================================================

    def update(self):  # noqa: C901
        """
        NP3O 更新步骤，包括：
        1. 采样 mini-batch（包含 cost 相关信息）
        2. 计算 PPO surrogate loss（奖励优势）
        3. 计算 reward value loss（奖励值函数）
        4. 对每个成本约束 i：
            a. 估计约束违反 violation_i = max(J^C_i(π) - ε_i, 0)，其中 J^C_i(π) ≈ mean(cost_i) 在本 batch 中
            b. 计算成本替代损失 cost_surrogate_loss_i，使用成本优势 A^C_i（最小化成本）
            c. 计算成本惩罚 penalty_i = κ_i * violation_i * cost_surrogate_loss_i
            d. 计算成本值函数损失 cost_value_loss_i（拟合成本回报）
        5. 总损失 = surrogate_loss + value_loss_coef * value_loss
                 + cost_value_loss_coef * total_cost_value_loss
                 + total_cost_penalty
                 - entropy_coef * entropy
        6. 反向传播并更新策略参数
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0

        # ==================== 新增：NP3O 统计量 ====================
        mean_cost_value_loss = torch.zeros(self.num_costs, device=self.device)
        mean_cost_penalty = torch.zeros(self.num_costs, device=self.device)
        mean_cost_violation = torch.zeros(self.num_costs, device=self.device)
        # =========================================================

        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None

        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator：NP3ORolloutStorage 会额外 yield cost batch
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            # ==================== 新增：兼容 PPO vs NP3O batch 结构 ====================
            # 新增：如果 storage 是 NP3O，会多出来 4 个 cost batch
            if len(batch) == 10:
                (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    hid_states_batch,
                    masks_batch,
                ) = batch
                costs_batch = None
                target_cost_values_batch = None
                cost_advantages_batch = None
                cost_returns_batch = None
            else:
                (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    hid_states_batch,
                    masks_batch,
                    costs_batch,
                    target_cost_values_batch,
                    cost_advantages_batch,
                    cost_returns_batch,
                ) = batch
            # ======================================================================

            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # symmetry augmentation（沿用 PPO 框架） :contentReference[oaicite:10]{index=10}
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)

                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

                # ==================== 新增：同步 repeat cost batch ====================
                if costs_batch is not None:
                    costs_batch = costs_batch.repeat(num_aug, 1)
                    target_cost_values_batch = target_cost_values_batch.repeat(num_aug, 1)
                    cost_advantages_batch = cost_advantages_batch.repeat(num_aug, 1)
                    cost_returns_batch = cost_returns_batch.repeat(num_aug, 1)
                # ================================================================

            # forward（沿用 PPO：act -> logp -> value -> entropy） :contentReference[oaicite:11]{index=11}
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])

            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL adaptive LR（沿用 PPO） :contentReference[oaicite:12]{index=12}
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # PPO surrogate loss（原版） :contentReference[oaicite:13]{index=13}
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # reward value loss（原版） :contentReference[oaicite:14]{index=14}
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # ==================== 新增：NP3O cost surrogate penalty + cost value loss ====================
            # 新增：NP3O 在 actor loss 中加入约束项：
            #   violation_i = relu(J^Ci(π) - eps_i)，J^Ci 用本 batch 的 mean(cost_i) 做经验估计 :contentReference[oaicite:15]{index=15}
            #   penalty_i = kappa_i * violation_i * cost_surrogate_loss_i
            # 这里 cost_surrogate_loss_i 用和 PPO 一样的 clip 形式，但 advantage 换成 cost_advantage（最小化 cost）
            total_cost_penalty = 0.0
            total_cost_value_loss = 0.0

            if costs_batch is None:
                raise RuntimeError("NP3O update requires cost batches. Please use NP3ORolloutStorage.")
            # 评估当前 cost values
            cost_value_pred = self.policy.evaluate_costs(obs_batch)
            if isinstance(cost_value_pred, tuple):
                cost_value_pred = torch.cat([cv for cv in cost_value_pred], dim=-1)  # (B,C)

            for i in range(self.num_costs):
                """
                对每个成本约束 i 独立计算：
                1. 约束违反 violation_i = max(J^C_i(π) - ε_i, 0)
                   其中 J^C_i(π) 是成本 i 的期望折扣累计成本，用本 batch 的 mean(cost_i) 近似
                2. 成本替代损失 cost_surrogate_loss_i = max(cost_sur, cost_sur_clipped).mean()
                   其中 cost_sur = A^C_i * ratio, cost_sur_clipped = A^C_i * clip(ratio, 1-ε, 1+ε)
                   （与 PPO 类似，但优势 A^C_i 是成本优势，目标是最小化成本）
                3. 惩罚项 penalty_i = κ_i * violation_i * cost_surrogate_loss_i
                4. 成本值函数损失 cost_value_loss_i，拟合成本回报 R^C_i
                """
                # ---- violation estimate ----
                # Jc_i：step中第i个cost的违规次数
                Jc_i = cost_returns_batch[:, i].mean()
                # violation_i：max(0,Jc_i - self.eps_cost[i]) 
                # self.eps_cost[i]：设定的约束违规阈值ϵi
                violation_i = torch.relu(Jc_i - self.eps_cost[i])

                # ---- 2. 获取归一化补偿项 ----
                # 提取 storage 中保存的统计量
                mu_c_i = self.storage.cost_advantages_mean[0, 0, i]
                sigma_c_i = self.storage.cost_advantages_std[0, 0, i]

                # ---- cost surrogate loss (minimize cost) ----
                cost_adv_i = cost_advantages_batch[:, i]
                cost_sur = torch.squeeze(cost_adv_i) * ratio
                cost_sur_clipped = torch.squeeze(cost_adv_i) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                # cost_surrogate_loss_i = torch.max(cost_sur, cost_sur_clipped).mean()
                L_C_i_CLIP_N = torch.max(cost_sur, cost_sur_clipped).mean()

                # ---- 4. 构建加法偏移违规目标 L_C_i^VIOL_N ----
                # 对应公式中的常数偏移项：[(1 - gamma) * (J_C_i - eps_i) + mu_c_i] / sigma_c_i
                offset_i = ((1.0 - self.cost_gamma) * (Jc_i - self.eps_cost[i]) + mu_c_i) / sigma_c_i
                L_C_i_VIOL_N = L_C_i_CLIP_N + offset_i
                
                # ---- penalty ----
                # self.kappa_cost[i]：第 i 个约束的惩罚权重系数，来源于cfg中的kappa_cost
                # cost_surrogate_loss_i：基于 PPO 裁剪逻辑计算出的 Cost 代理损失
                penalty_i = self.kappa_cost[i] * torch.relu(L_C_i_VIOL_N)
                total_cost_penalty = total_cost_penalty + penalty_i

                # ---- cost value loss ----
                # 拟合成本值函数 V^C_i(s) 到成本回报 R^C_i，其中 R^C_i = A^C_i + V^C_i(s)
                # 使用 clipped value loss 防止值函数更新过大（与 PPO reward critic 类似）
                # target_cost_values_batch 是 rollout 时的旧 Vc(s)（用于 clipping）
                pred_i = cost_value_pred[:, i : i + 1]
                ret_i = cost_returns_batch[:, i : i + 1]
                old_i = target_cost_values_batch[:, i : i + 1]

                if self.use_clipped_cost_value_loss:
                    pred_clipped = old_i + (pred_i - old_i).clamp(-self.clip_param, self.clip_param)
                    vl = (pred_i - ret_i).pow(2)
                    vl_clip = (pred_clipped - ret_i).pow(2)
                    cost_value_loss_i = torch.max(vl, vl_clip).mean()
                else:
                    cost_value_loss_i = (pred_i - ret_i).pow(2).mean()

                total_cost_value_loss = total_cost_value_loss + cost_value_loss_i

                # stats
                mean_cost_violation[i] += violation_i.detach()
                mean_cost_penalty[i] += penalty_i.detach()
                mean_cost_value_loss[i] += cost_value_loss_i.detach()
            # =======================================================================

            # 总 loss：PPO loss + reward value loss + cost value loss + cost penalty - entropy bonus
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                + self.cost_value_loss_coef * total_cost_value_loss     # 新增超参数：cost_value_loss_coef（需写入 cfg）
                + total_cost_penalty
                - self.entropy_coef * entropy_batch.mean()
            )

            # symmetry loss（沿用 PPO） :contentReference[oaicite:16]{index=16}
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(obs=None, actions=action_mean_orig, env=self.symmetry["_env"])

                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss（沿用 PPO） :contentReference[oaicite:17]{index=17}
            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # backward & step（沿用 PPO） :contentReference[oaicite:18]{index=18}
            self.optimizer.zero_grad()
            loss.backward()

            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        # ==================== 新增：NP3O 平均统计量 ====================
        mean_cost_value_loss = (mean_cost_value_loss / num_updates).detach().cpu()
        mean_cost_penalty = (mean_cost_penalty / num_updates).detach().cpu()
        mean_cost_violation = (mean_cost_violation / num_updates).detach().cpu()
        # ============================================================

        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            # ==================== 新增：NP3O 日志 ====================
            "cost_value_loss": mean_cost_value_loss.tolist(),
            "cost_penalty": mean_cost_penalty.tolist(),
            "cost_violation": mean_cost_violation.tolist(),
            # =======================================================
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        
        clean = {}
        for k, v in loss_dict.items():
            if isinstance(v, (list, tuple)):
                for i, vi in enumerate(v):
                    clean[f"{k}_{i}"] = float(vi)
            else:
                # tensor -> float
                if hasattr(v, "item"):
                    clean[k] = float(v.item())
                else:
                    clean[k] = float(v)

        return clean

    # ===== multi-gpu helper（保持与 PPO 一致） =====
    def broadcast_parameters(self):
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
