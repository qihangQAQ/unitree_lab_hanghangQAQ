"""Main FDM-MPPI planner orchestrator.

Manages observation history buffers, triggers MPPI replanning, and executes
the first action of the planned sequence.

Reference: ``FDMPlanner`` in ``fdm/exts/fdm/fdm/planner/planner.py``
"""

from __future__ import annotations

import torch

from unitree_rl_lab.tasks.fdm.model.fdm_model import FDMConfig, FDMModel
from unitree_rl_lab.tasks.fdm.planner.mppi_config import PlannerConfig
from unitree_rl_lab.tasks.fdm.planner.state_transformer import transform_state_history_to_local
from unitree_rl_lab.tasks.fdm.planner.trajectory_optimizer import FDMTrajectoryOptimizer
from unitree_rl_lab.tasks.fdm.planner.visualizer import FDMVisualizer


class FDMMPPIPlanner:
    """Orchestrator that bridges Isaac Lab with MPPI-FDM planning.

    Each simulation step, ``step()`` updates history buffers and optionally
    triggers a replan.  The caller injects the returned SE2 velocity into the
    environment's command manager (exactly like data collection).
    """

    def __init__(self, config: PlannerConfig, env, policy, device: torch.device):
        self.cfg = config
        self.env = env
        self.policy = policy
        self.device = torch.device(device)

        # unwrapped Isaac Lab env
        self.base_env = env.unwrapped

        # ---- load frozen FDM model ----
        self.fdm_model = self._load_fdm_model()

        # ---- trajectory optimizer ----
        self.traj_opt = FDMTrajectoryOptimizer(self.fdm_model, config, self.device)

        # ---- visualizer (GUI-mode only; no-op in headless) ----
        self._visualizer = FDMVisualizer()

        # ---- timing ----
        step_dt = self.base_env.step_dt  # e.g. 0.02 s
        self.planner_decimation = int((1.0 / config.planner_frequency) / step_dt)
        self.decimation_counter = 0
        self.cmd_timestep = config.action.dt  # 0.5 s
        self.steps_per_cmd = int(self.cmd_timestep / step_dt)  # 25
        self.history_collection_interval = self.steps_per_cmd / config.history_length  # 2.5

        # ---- observation history buffers ----
        # raw state dim: pos(3)+quat(4)+collision(1)+hard_contact(1)+friction(2) = 11
        raw_state_dim = self.base_env.observation_manager.group_obs_dim["fdm_state"][0]
        proprio_dim = self.base_env.observation_manager.group_obs_dim["fdm_obs_proprioceptive"][0]

        self._state_history = torch.zeros(
            1, config.history_length, raw_state_dim, device=self.device
        )
        self._proprio_history = torch.zeros(
            1, config.history_length, proprio_dim, device=self.device
        )
        self._obs_step_counter = 0

        # ---- feet contact gating ----
        contact_sensor = self.base_env.scene.sensors["contact_forces"]
        feet_ids, feet_names = contact_sensor.find_bodies(".*ankle_roll.*")
        if len(feet_ids) == 0:
            raise RuntimeError("No foot bodies matched '.*ankle_roll.*'")
        print(f"[INFO] FDM planner feet-contact bodies: {feet_names}")
        self._feet_body_ids = torch.tensor(feet_ids, dtype=torch.long, device=self.device)
        self._feet_contact_seen = torch.zeros(1, len(feet_ids), dtype=torch.bool, device=self.device)
        self._feet_all_contact = torch.zeros(1, dtype=torch.bool, device=self.device)

        # ---- action buffer ----
        self._current_actions = torch.zeros(1, config.action.traj_dim, 3, device=self.device)
        self._latest_trajectory = torch.zeros(1, config.action.traj_dim, 3)

        # ---- goal ----
        self._goal = torch.tensor(
            [[config.goal_position[0], config.goal_position[1], config.goal_heading]],
            device=self.device,
        )
        self._goal_reached = False

        # ---- info ----
        self.last_info: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, obs: dict | torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Called every simulation step.  Returns ``(se2_action, info)``.

        Args:
            obs: environment observation (for low-level policy later).

        Returns:
            se2_action: (1, 3) SE2 velocity command in body frame [vx, vy, vyaw].
            info: dict with optional planning info.
        """
        info: dict = {}

        # ---- update feet contact ----
        self._update_feet_contact()

        # ---- update history ----
        if self._feet_all_contact.item():
            if int(self._obs_step_counter % self.history_collection_interval) == 0:
                self._update_history()
            self._obs_step_counter += 1

        # ---- replan ----
        history_ready = self._obs_step_counter >= self.cfg.history_length
        decimation_ready = self.decimation_counter >= self.planner_decimation

        if decimation_ready and history_ready:
            self._replan()
            info["planned"] = True
            info["best_trajectory"] = self._latest_trajectory.detach().cpu().clone()
            self.decimation_counter = 0

        self.decimation_counter += 1
        self.last_info = info
        return self._current_actions[:, 0, :].clone(), info

    def reset(self) -> None:
        """Reset history buffers (call on episode reset)."""
        self._state_history.zero_()
        self._proprio_history.zero_()
        self._obs_step_counter = 0
        self.decimation_counter = 0
        self._feet_contact_seen.zero_()
        self._feet_all_contact.zero_()
        self._current_actions.zero_()
        self.traj_opt.optimizer.reset([0])

    def set_goal(self, x: float, y: float, yaw: float = 0.0) -> None:
        """Update the goal pose in world frame."""
        self._goal = torch.tensor([[x, y, yaw]], device=self.device)
        self._goal_reached = False
        self._visualizer.draw_goal_sphere(x=x, y=y, z=0.05, reached=False)

    def mark_goal_reached(self) -> None:
        """Turn the goal marker green."""
        self._goal_reached = True
        self._visualizer.draw_goal_sphere(
            x=float(self._goal[0, 0].item()),
            y=float(self._goal[0, 1].item()),
            z=0.05,
            reached=True,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_fdm_model(self) -> FDMModel:
        """Load frozen FDM model from checkpoint."""
        ckpt = torch.load(self.cfg.fdm_checkpoint, map_location=self.device, weights_only=False)
        fdm_cfg = FDMConfig(**ckpt["cfg"])
        if isinstance(fdm_cfg.height_scan_shape, list):
            fdm_cfg.height_scan_shape = tuple(fdm_cfg.height_scan_shape)
        model = FDMModel(fdm_cfg, device=self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"[INFO] Loaded FDM model from {self.cfg.fdm_checkpoint}")
        return model

    def _update_feet_contact(self) -> None:
        """Track whether each foot has touched ground since last reset."""
        contact_forces = self.base_env.scene.sensors["contact_forces"].data.net_forces_w
        contact_now = torch.norm(contact_forces[:, self._feet_body_ids], dim=-1) > 1.0
        self._feet_contact_seen[contact_now] = True
        self._feet_all_contact = torch.all(self._feet_contact_seen, dim=-1)

    def _update_history(self) -> None:
        """Roll history buffers and insert current observations."""
        state_raw = self._get_fdm_state()  # (1, raw_state_dim)
        proprio = self._get_fdm_proprioceptive()  # (1, proprio_dim)

        self._state_history = torch.roll(self._state_history, 1, dims=1)
        self._state_history[:, 0] = state_raw

        self._proprio_history = torch.roll(self._proprio_history, 1, dims=1)
        self._proprio_history[:, 0] = proprio

    def _replan(self) -> None:
        """Prepare context and run MPPI optimisation."""
        # ---- transform state history to local frame ----
        state_local = transform_state_history_to_local(self._state_history)
        # (1, H, raw_state_dim) → (1, H, state_dim)

        # ---- get current world pose ----
        robot = self.base_env.scene["robot"]
        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w[:, [1, 2, 3, 0]]  # wxyz → xyzw
        world_pose = torch.cat([root_pos, root_quat], dim=-1)  # (1, 7)

        # ---- get current height scan ----
        height_scan = self._get_fdm_height_scan()  # (1, H_img, W_img)

        # ---- set planner context ----
        self.traj_opt.set_context(
            goal=self._goal,
            state_history=state_local,
            proprio_history=self._proprio_history.clone(),
            height_scan=height_scan,
            world_pose=world_pose,
        )

        # ---- run MPPI ----
        best_actions, best_trajectory = self.traj_opt.plan(env_ids=[0])
        # best_actions: (1, horizon, 3)
        # best_trajectory: (1, horizon, 3)

        # Store the full plan.  The MPPI optimizer already performs mean-shift
        # internally, so best_actions[0] is the action for the current step.
        self._current_actions = best_actions.clone()
        self._latest_trajectory = best_trajectory.detach().cpu().clone()

        # ---- visualisation (simplified: best trajectory + goal marker only) ----
        robot_z = float(root_pos[0, 2].item())
        self._visualizer.clear()
        # best trajectory (green line)
        self._visualizer.draw_best_trajectory(
            world_states=best_trajectory,   # (1, horizon, 3)
            robot_base_z=robot_z,
        )
        # goal marker (red=not reached, green=reached)
        self._visualizer.draw_goal_sphere(
            x=float(self._goal[0, 0].item()),
            y=float(self._goal[0, 1].item()),
            z=0.05,
            reached=self._goal_reached,
        )

    # ------------------------------------------------------------------
    # Observation helpers (same pattern as data collection)
    # ------------------------------------------------------------------

    def _get_fdm_state(self) -> torch.Tensor:
        return self.base_env.observation_manager.compute_group("fdm_state")

    def _get_fdm_proprioceptive(self) -> torch.Tensor:
        from unitree_rl_lab.tasks.fdm.mdp.observations import _update_joint_history

        _update_joint_history(self.base_env)
        return self.base_env.observation_manager.compute_group("fdm_obs_proprioceptive")

    def _get_fdm_height_scan(self) -> torch.Tensor:
        raw = self.base_env.observation_manager.compute_group("fdm_obs_exteroceptive")
        if raw.dim() == 2:
            raw = raw.reshape(raw.shape[0], 60, 46)
        elif raw.dim() == 4:
            raw = raw.squeeze(1)
        return raw
