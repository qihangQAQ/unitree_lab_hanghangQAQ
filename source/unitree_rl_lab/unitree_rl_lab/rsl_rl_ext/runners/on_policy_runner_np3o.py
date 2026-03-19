from __future__ import annotations

import warnings
import torch

from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.utils import resolve_obs_groups

from ..modules.actor_critic_np3o import ActorCriticNP3O
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

# ==================== 新增：导入 NP3O 算法类 ====================
# 新增：NP3O -- 约束版 on-policy 算法（需要 env extras['costs'] + cost critics）
from ..algorithms.np3o import NP3O
# ===============================================================


class NP3ORunner(OnPolicyRunner):
    """NP3O 专用 Runner：复用 OnPolicyRunner 的训练循环，仅替换算法装配与（可选）cost 日志。"""

    # ==================== 新增：构造 NP3O 算法 ====================
    # 新增：重写 _construct_algorithm -- 用 NP3O 替换 PPO，并确保 policy 是带 cost critics 的 ActorCritic
    def _construct_algorithm(self, obs) -> NP3O:
        # resolve RND config（沿用原 Runner）:contentReference[oaicite:4]{index=4}
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # resolve symmetry config（沿用原 Runner）:contentReference[oaicite:5]{index=5}
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # resolve deprecated normalization config（沿用原 Runner）:contentReference[oaicite:6]{index=6}
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # ==================== 新增：构造 NP3O ActorCritic ====================
        # 新增：policy class_name 需要指向你实现的 NP3OActorCritic（带 evaluate_costs）
        # 新增超参数：policy.class_name（需写入 cfg）= "NP3OActorCritic"（或你命名的 ActorCriticNp3o）
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        # ================================================================

        # ==================== 新增：构造 NP3O 算法实例 ====================
        # 新增：alg.class_name（需写入 cfg）= "NP3O"
        # 注意：这里直接用 NP3O，而不是 eval(class_name) 再手动判断，减少出错面
        # print("alg_cfg keys:", self.alg_cfg.keys())
        # print("alg_cfg:", self.alg_cfg)
        # 过滤字段
        self.alg_cfg.pop("class_name", None)
        alg = NP3O(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)
        # ===============================================================

        # 初始化 storage（NP3O.init_storage 内部会创建 NP3ORolloutStorage）
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return alg
    # =================================================================
