"""Batched MPPI optimizer — direct reimplementation from reference FDM.

Reference: ``BatchedMPPIOptimizer`` in
``fdm/exts/fdm/fdm/planner/sampling_planner/trajectory_optimizer_mbrl.py``
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn

from unitree_rl_lab.tasks.fdm.planner.mppi_config import ActionConfig, MPPIConfig


def truncated_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """In-place truncated normal (clamped at ±2σ).  Mirrors reference implementation."""
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2.0) & (tmp > -2.0)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
    return tensor


class BatchedMPPIOptimizer(nn.Module):
    """MPPI optimizer supporting batched environments.

    One instance handles ``batch_size`` environments.  ``optimize()`` returns
    the weighted action mean for the given ``env_ids``.
    """

    def __init__(self, mppi_cfg: MPPIConfig, action_cfg: ActionConfig, device: torch.device, batch_size: int = 1):
        super().__init__()
        self.num_iterations = mppi_cfg.num_iterations
        self.population_size = mppi_cfg.population_size
        self.gamma = mppi_cfg.gamma
        self.sigma = mppi_cfg.sigma
        self.beta = mppi_cfg.beta

        self.action_dim = action_cfg.action_dim
        self.planning_horizon = action_cfg.traj_dim

        # action bounds as tensors  (horizon, action_dim)
        _lb = torch.tensor(action_cfg.lower_bound_tensor, dtype=torch.float32, device=device)
        _ub = torch.tensor(action_cfg.upper_bound_tensor, dtype=torch.float32, device=device)
        self.register_buffer("lower_bound", _lb)
        self.register_buffer("upper_bound", _ub)

        # variance upper-bound: sigma²
        self.register_buffer("var", torch.full_like(_lb, self.sigma**2))

        # mean initialised at centre of action bounds  (batch_size, horizon, action_dim)
        init_mean = (_lb + _ub) / 2.0
        self.register_buffer("mean", init_mean.unsqueeze(0).expand(batch_size, -1, -1).clone())

        # cache the non-moving centre value for reset
        self.register_buffer("_centre", init_mean.clone())

        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int]) -> None:
        """Reset mean to centre of action bounds for the given environments."""
        for eid in env_ids:
            self.mean[eid] = self._centre.clone()

    @torch.no_grad()
    def optimize(
        self,
        obj_fun: Callable[[torch.Tensor, bool], torch.Tensor],
        env_ids: Sequence[int] | None = None,
        x0: torch.Tensor | None = None,
        x0_env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Run MPPI optimisation and return the weighted action mean.

        Args:
            obj_fun:  ``fn(population, only_rollout) -> values`` where
                ``population`` is ``(pop_size, num_envs, horizon, action_dim)``
                and ``values`` is ``(pop_size, num_envs)`` (higher = better).
            env_ids:  which environments to optimise (default: all).
            x0:       optional initial guess  ``(num_x0_envs, horizon, action_dim)``.
            x0_env_ids: which environments ``x0`` applies to.

        Returns:
            ``(len(env_ids), planning_horizon, action_dim)`` optimised action sequences.
        """
        if env_ids is None:
            env_ids = list(range(self._batch_size))
        BS = len(env_ids)

        # ---- optional x0 injection ----
        if x0 is not None and x0_env_ids is not None:
            for i, eid in enumerate(x0_env_ids):
                self.mean[eid] = x0[i].clone()

        # ---- mean shift (carry-over from previous plan) ----
        past_action = self.mean[env_ids, 0].clone()  # (BS, action_dim)
        self.mean[env_ids, :-1] = self.mean[env_ids, 1:].clone()

        for _k in range(self.num_iterations):
            # -- sample noise --
            noise = torch.empty(
                self.population_size, BS, self.planning_horizon, self.action_dim,
                device=self.mean.device,
            )
            truncated_normal_(noise)

            # -- constrained variance --
            lb_dist = self.mean[env_ids] - self.lower_bound  # (BS, H, A)
            ub_dist = self.upper_bound - self.mean[env_ids]  # (BS, H, A)
            mv = torch.minimum((lb_dist / 2.0) ** 2, (ub_dist / 2.0) ** 2)  # (BS, H, A)
            constrained_var = torch.minimum(mv, self.var)  # (BS, H, A)
            noise = noise * constrained_var.sqrt()  # (pop, BS, H, A)

            # -- temporal correlation (beta smoothing) --
            population = noise.clone()  # (pop, BS, H, A)
            population[:, :, 0, :] += self.beta * self.mean[env_ids, 0] + (1.0 - self.beta) * past_action
            for i in range(self.planning_horizon - 1):
                population[:, :, i + 1, :] += (
                    self.beta * self.mean[env_ids, i + 1] + (1.0 - self.beta) * population[:, :, i, :]
                )

            # -- clip to bounds --
            population = torch.clamp(population, self.lower_bound, self.upper_bound)

            # -- evaluate --
            values = obj_fun(population, False)  # (pop, BS)
            values = torch.nan_to_num(values, nan=-1e-10)

            # -- MPPI weights --
            max_vals = values.max(dim=0, keepdim=True)[0]  # (1, BS)
            weights = torch.exp(self.gamma * (values - max_vals))  # (pop, BS)
            norm = weights.sum(dim=0, keepdim=True) + 1e-10  # (1, BS)

            # -- weighted mean --
            self.mean[env_ids] = (population * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0) / norm.unsqueeze(-1)

        return self.mean[env_ids].clone()
