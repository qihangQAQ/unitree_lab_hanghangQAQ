"""FDM trajectory optimizer — cost function that evaluates action sequences through FDM.

Reference: ``SimpleSE2TrajectoryOptimizer`` in
``fdm/exts/fdm/fdm/planner/sampling_planner/trajectory_optimizer.py``
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial.distance import cdist

from unitree_rl_lab.tasks.fdm.model.fdm_model import FDMModel
from unitree_rl_lab.tasks.fdm.planner.mppi_config import PlannerConfig
from unitree_rl_lab.tasks.fdm.planner.mppi_optimizer import BatchedMPPIOptimizer
from unitree_rl_lab.tasks.fdm.planner.se2_utils import cosine_distance, smallest_angle
from unitree_rl_lab.tasks.fdm.planner.state_transformer import fdm_output_to_world_frame


class FDMTrajectoryOptimizer:
    """Cost function evaluated by the MPPI optimizer.

    Holds a frozen FDM model and current planning context (goal, state history,
    height scan, etc.).  ``_evaluate_population()`` is the callable passed to
    ``BatchedMPPIOptimizer.optimize()``.
    """

    def __init__(self, fdm_model: FDMModel, config: PlannerConfig, device: torch.device):
        self.fdm_model = fdm_model
        self.cfg = config
        self.device = device

        self.optimizer = BatchedMPPIOptimizer(config.mppi, config.action, device, batch_size=1)

        # ---- planning context (set before each optimise call) ----
        self._goal: torch.Tensor | None = None  # (BS, 3) world frame [x, y, yaw]
        self._state_history: torch.Tensor | None = None  # (BS, H, state_dim) transformed
        self._proprio_history: torch.Tensor | None = None  # (BS, H, proprio_dim)
        self._height_scan: torch.Tensor | None = None  # (BS, H_img, W_img)
        self._world_pose: torch.Tensor | None = None  # (BS, 7) at planning time

        # ---- debug ----
        self.latest_states: torch.Tensor | None = None  # (BS, NR, horizon, 3) world
        self.latest_costs: torch.Tensor | None = None  # (BS, NR)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_context(
        self,
        goal: torch.Tensor,
        state_history: torch.Tensor,
        proprio_history: torch.Tensor,
        height_scan: torch.Tensor,
        world_pose: torch.Tensor,
    ) -> None:
        """Store the current planning context.  Called before each replan."""
        self._goal = goal
        self._state_history = state_history
        self._proprio_history = proprio_history
        self._height_scan = height_scan
        self._world_pose = world_pose

    @torch.no_grad()
    def plan(self, env_ids=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Run MPPI and return ``(best_actions, best_trajectory)``.

        Returns:
            best_actions: (1, traj_dim, 3) SE2 velocity commands in body frame.
            best_trajectory: (1, traj_dim, 3) world-frame [x, y, yaw].
        """
        best_actions = self.optimizer.optimize(
            obj_fun=self._evaluate_population, env_ids=env_ids
        )  # (BS, horizon, action_dim)

        # Roll out the best actions to get world-frame trajectory
        best_trajectory = self._evaluate_population(
            best_actions.unsqueeze(0), only_rollout=True
        )  # (BS, 1, horizon, 3)
        best_trajectory = best_trajectory.squeeze(1)  # (BS, horizon, 3)

        return best_actions, best_trajectory

    # ------------------------------------------------------------------
    # Callable passed to MPPI
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_population(
        self, population: torch.Tensor, only_rollout: bool = False
    ) -> torch.Tensor:
        """Evaluate a population of action sequences through FDM.

        Args:
            population: (pop_size, BS, horizon, action_dim).
            only_rollout:  if True, return world-frame trajectory states instead of costs
                (matching reference FDM's b_obj_func_N_step signature).

        Returns:
            When only_rollout=False: (pop_size, BS) negated costs.
            When only_rollout=True:  (BS, NR, horizon, 3) world-frame [x, y, yaw].
        """
        NR_TRAJ = population.shape[0]  # pop_size
        BS = population.shape[1]  # num envs (usually 1)

        # Permute: (BS, NR, horizon, action_dim)
        population_bs = population.permute(1, 0, 2, 3)

        # ---- expand context for each sampled trajectory ----
        total = BS * NR_TRAJ

        state_exp = self._repeat_interleave_bs(self._state_history, NR_TRAJ)  # (total, H, S)
        proprio_exp = self._repeat_interleave_bs(self._proprio_history, NR_TRAJ)  # (total, H, P)
        extero_exp = self._repeat_interleave_bs(self._height_scan, NR_TRAJ)  # (total, *img)
        actions_flat = population_bs.reshape(total, self.cfg.prediction_horizon, 3)
        dummy = torch.zeros(total, 1, device=self.device)  # unused add_extero

        # ---- FDM forward (minibatch to avoid OOM) ----
        fdm_batch = self.cfg.fdm_batch_size
        local_states_list = []
        coll_probs_list = []

        for start in range(0, total, fdm_batch):
            end = min(start + fdm_batch, total)
            model_in = (
                state_exp[start:end],
                proprio_exp[start:end],
                extero_exp[start:end],
                actions_flat[start:end],
                dummy[start:end],
            )
            out_states, coll_probs, _energy = self.fdm_model.forward(model_in)
            # out_states:  (chunk, horizon, 4)  — local frame
            # coll_probs: (chunk, horizon)
            local_states_list.append(out_states)
            coll_probs_list.append(coll_probs)

        local_states = torch.cat(local_states_list, dim=0)  # (total, horizon, 4)
        coll_probs = torch.cat(coll_probs_list, dim=0)  # (total, horizon)

        # Reshape: (BS, NR, horizon, ...)
        local_states = local_states.reshape(BS, NR_TRAJ, self.cfg.prediction_horizon, 4)
        coll_probs = coll_probs.reshape(BS, NR_TRAJ, self.cfg.prediction_horizon)

        # ---- transform to world frame ----
        world_states = fdm_output_to_world_frame(
            local_states.reshape(BS * NR_TRAJ, self.cfg.prediction_horizon, 4),
            self._world_pose.repeat_interleave(NR_TRAJ, dim=0),
        )  # (BS*NR, horizon, 3)
        world_states = world_states.reshape(BS, NR_TRAJ, self.cfg.prediction_horizon, 3)

        # ---- only rollout: return trajectory states (matching reference FDM) ----
        if only_rollout:
            return world_states  # (BS, NR, horizon, 3)

        # ---- costs ----
        running_cost = self._running_cost(world_states, population_bs)  # (BS, NR, horizon)
        running_cost = running_cost.mean(dim=2)  # (BS, NR)

        terminal_cost = self._terminal_cost(world_states[:, :, -1, :])  # (BS, NR)
        collision_cost = self._collision_cost(world_states, coll_probs)  # (BS, NR)

        total_cost = running_cost + terminal_cost + collision_cost  # (BS, NR)

        # Cache for debugging / visualisation
        self.latest_states = world_states
        self.latest_costs = total_cost
        self.latest_collision_probs = coll_probs

        # MPPI maximises → return negative cost  (pop_size, BS)
        return -total_cost.T

    # ------------------------------------------------------------------
    # Cost components
    # ------------------------------------------------------------------

    def _running_cost(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Per-timestep running cost.  (BS, NR, horizon)."""
        BS, NR, H = states.shape[0], states.shape[1], states.shape[2]
        cost = torch.zeros(BS, NR, H, device=self.device)

        # -- control effort (all zero by default) --
        if self.cfg.state_cost_w_action_trans_forward > 0:
            cost += torch.abs(actions[..., 0]) * self.cfg.state_cost_w_action_trans_forward
        if self.cfg.state_cost_w_action_trans_side > 0:
            cost += torch.abs(actions[..., 1]) * self.cfg.state_cost_w_action_trans_side
        if self.cfg.state_cost_w_action_rot > 0:
            cost += torch.abs(actions[..., 2]) * self.cfg.state_cost_w_action_rot

        # -- early goal reaching bonus --
        pos_offset = torch.norm(states[..., :2] - self._goal[:, None, None, :2], dim=-1)
        if self._goal.shape[-1] >= 3:
            heading_dist = cosine_distance(states[..., 2], self._goal[:, None, None, 2]) / 2.0
        else:
            heading_dist = torch.zeros_like(pos_offset)

        near_goal = (pos_offset < 0.3) & (heading_dist < 0.3)
        if self.cfg.state_cost_w_early_goal_reaching > 0:
            cost[near_goal] -= self.cfg.state_cost_w_early_goal_reaching
        cost[near_goal] = 0.0  # zero out cost when near goal

        # -- early stopping bonus (reward for stopping after reaching goal) --
        if self.cfg.state_cost_w_early_stopping > 0:
            action_mag = torch.abs(actions).sum(dim=-1)  # (BS, NR, H)
            non_zero = (action_mag > 0.05).float()
            # find last non-zero index per trajectory
            last_non_zero = torch.flip(non_zero, dims=[2]).float().argmax(dim=2)  # (BS, NR)
            stop_frac = (H - 1 - last_non_zero) / H  # fraction of trailing zeros
            cost -= stop_frac * self.cfg.state_cost_w_early_stopping

        # -- velocity tracking (zero by default) --
        if self.cfg.state_cost_velocity_tracking > 0:
            vel_xy = torch.norm(actions[..., :2], dim=-1)
            cost += (
                torch.abs(vel_xy - self.cfg.state_cost_desired_velocity)
                * self.cfg.state_cost_velocity_tracking
            )

        return cost

    def _terminal_cost(self, final_states: torch.Tensor) -> torch.Tensor:
        """Terminal cost: position + heading error to goal.  (BS, NR)."""
        BS, NR = final_states.shape[0], final_states.shape[1]
        goal = self._goal[:, None, :].expand(BS, NR, 3)  # (BS, NR, 3)

        pos_cost = (
            torch.norm(final_states[..., :2] - goal[..., :2], dim=-1)
            * self.cfg.terminal_cost_w_position
        )

        rot_cost = torch.zeros_like(pos_cost)
        if self.cfg.terminal_cost_w_rotation > 0:
            rot_cost = (
                smallest_angle(final_states[..., 2], goal[..., 2])
                * self.cfg.terminal_cost_w_rotation
            )

        total = pos_cost + rot_cost

        # close-reward: scale down cost when near goal
        if self.cfg.terminal_cost_distance_threshold > 0:
            close = pos_cost / max(self.cfg.terminal_cost_w_position, 1e-6) < self.cfg.terminal_cost_distance_threshold
            total[close] /= self.cfg.terminal_cost_close_reward

        return total

    def _collision_cost(
        self, states: torch.Tensor, collision_probs: torch.Tensor
    ) -> torch.Tensor:
        """Collision cost with continuous gradient + neighbour smoothing (aligned with FDM).

        Args:
            states:          (BS, NR, horizon, 3) world-frame trajectory waypoints.
            collision_probs: (BS, NR, horizon) per-step collision probabilities.

        Returns:
            (BS, NR) collision cost per trajectory.
        """
        threshold = self.cfg.collision_threshold - self.cfg.collision_cost_safety_factor
        BS, NR, H = collision_probs.shape

        # 1) continuous cost: sum of collision probs along trajectory
        cost = collision_probs.sum(dim=-1) * self.cfg.collision_cost_traj_factor  # (BS, NR)

        # 2) high-risk penalty on top
        high_risk = torch.any(collision_probs > threshold, dim=-1)  # (BS, NR)
        cost[high_risk] += self.cfg.collision_cost_high_risk_factor

        # 3) neighbour smoothing (eliminates isolated false positives, matching FDM)
        cost_pre = cost.clone()
        for env_id in range(BS):
            # flatten trajectory xy into (NR, horizon*2)
            flat = states[env_id, :, :, :2].reshape(NR, -1).cpu().numpy()
            dist = cdist(flat, flat, metric="euclidean")
            neighbours = np.argsort(dist, axis=1)[:, 1:self.cfg.num_neighbors + 1]  # (NR, K)
            dist_t = torch.tensor(dist, device=cost.device)
            cost[env_id] += (
                cost_pre[env_id][neighbours.flatten()]
                .reshape(NR, self.cfg.num_neighbors)
                / dist_t[torch.arange(NR, device=cost.device)[:, None].repeat(1, self.cfg.num_neighbors), neighbours]
            ).sum(dim=-1)

        return cost

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _repeat_interleave_bs(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
        """Repeat each batch element ``repeats`` times, keeping other dims.

        tensor: (BS, ...) → (BS * repeats, ...)
        e.g.  (1, 10, 8) with repeats=1024 → (1024, 10, 8)
        """
        return tensor.repeat_interleave(repeats, dim=0)
