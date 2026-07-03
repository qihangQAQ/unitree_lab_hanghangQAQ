"""Configuration dataclasses for the MPPI-FDM planner."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActionConfig:
    """SE(2) velocity action bounds and trajectory parameters."""

    action_dim: int = 3  # (vx, vy, vyaw) in body frame
    traj_dim: int = 10  # prediction horizon steps
    dt: float = 0.5  # seconds per step
    lower_bound: tuple[float, float, float] = (-0.6, -0.5, -1.57)
    upper_bound: tuple[float, float, float] = (1.0, 0.5, 1.57)

    @property
    def lower_bound_tensor(self) -> list[list[float]]:
        return [[self.lower_bound[0], self.lower_bound[1], self.lower_bound[2]]] * self.traj_dim

    @property
    def upper_bound_tensor(self) -> list[list[float]]:
        return [[self.upper_bound[0], self.upper_bound[1], self.upper_bound[2]]] * self.traj_dim


@dataclass
class MPPIConfig:
    """MPPI optimizer hyperparameters (aligned with reference FDM)."""

    num_iterations: int = 1
    population_size: int = 1024
    gamma: float = 1.0  # reward scaling / inverse temperature
    sigma: float = 0.87  # noise std upper bound
    beta: float = 0.6  # temporal correlation (0=fully correlated, 1=uncorrelated)


@dataclass
class PlannerConfig:
    """Top-level planner configuration."""

    # ---- FDM model ----
    fdm_checkpoint: str = ""
    prediction_horizon: int = 10
    history_length: int = 10
    state_dim: int = 8  # transformed: pos(2) + yaw_sin_cos(2) + collision(1) + hard_contact(1) + friction(2)
    proprio_dim: int = 302  # G1-29dof
    height_scan_shape: tuple[int, int] = (60, 46)
    device: str = "cuda"

    # ---- sub-configs ----
    action: ActionConfig = field(default_factory=ActionConfig)
    mppi: MPPIConfig = field(default_factory=MPPIConfig)
    planner_frequency: float = 2.0  # Hz, how often to replan

    # ---- cost weights (aligned with reference planner_cfg.py defaults) ----
    state_cost_w_action_rot: float = 0.0
    state_cost_w_action_trans_forward: float = 0.0
    state_cost_w_action_trans_side: float = 0.0
    state_cost_w_early_goal_reaching: float = 0.0
    state_cost_w_early_stopping: float = 0.0
    state_cost_velocity_tracking: float = 0.0
    state_cost_desired_velocity: float = 0.0

    terminal_cost_w_position: float = 10.0
    terminal_cost_w_rotation: float = 0.0
    terminal_cost_distance_threshold: float = 0.3
    terminal_cost_close_reward: float = 10.0

    collision_cost_traj_factor: float = 0.5
    collision_cost_high_risk_factor: float = 1000.0
    collision_cost_safety_factor: float = 0.0
    collision_threshold: float = 0.5
    num_neighbors: int = 3

    # ---- goal ----
    goal_position: tuple[float, float] = (3.0, 0.0)
    goal_heading: float = 0.0

    # ---- FDM minibatch ----
    fdm_batch_size: int = 128  # max samples per FDM forward call
