"""FDM-specific observation functions."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


"""
Root state observations for FDM.
"""


def base_position(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Position of the asset's root in world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


def base_orientation_xyzw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Orientation of the asset's root in world frame.

    Note: converts the quaternion to (x, y, z, w) format."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_quat_w[:, [1, 2, 3, 0]]


def base_collision(
    env: ManagerBasedRLEnv, threshold: float = 1.0, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    """Check if the base has collision exceeding the force threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    normed_force = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1).flatten(start_dim=1)
    return (torch.max(normed_force, dim=1)[0] > threshold).unsqueeze(-1).float()


def energy_consumption(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), energy_scale_factor: float = 0.001
) -> torch.Tensor:
    """The energy consumption of the asset. Computed as the sum of the squared applied torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    return (asset.data.applied_torque**2).sum(dim=-1).unsqueeze(-1) * energy_scale_factor


class FrictionObservation:
    """Friction observation for FDM."""

    def __init__(self):
        pass

    def _setup_view(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        asset: Articulation = env.scene[asset_cfg.name]
        self._num_shapes_per_body_mapping = []
        for link_path in asset.root_physx_view.link_paths[0]:
            link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)
            self._num_shapes_per_body_mapping.append(link_physx_view.max_shapes)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """The friction coefficients of the asset."""
        asset: Articulation = env.scene[asset_cfg.name]

        # setup the view
        if not hasattr(self, "_num_shapes_per_body_mapping"):
            self._setup_view(env, asset_cfg)

        # get materials of the bodies
        materials = asset.root_physx_view.get_material_properties()
        static_friction = torch.zeros(
            (env.num_envs, len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies, 1),
            device=env.device,
        )

        # get material properties for the bodies
        for idx, body_id in enumerate(
            asset_cfg.body_ids if isinstance(asset_cfg.body_ids, list) else range(asset.num_bodies)
        ):
            # start index of shape
            start_idx = sum(self._num_shapes_per_body_mapping[:body_id])
            # end index of shape
            end_idx = start_idx + self._num_shapes_per_body_mapping[body_id]
            # get the static friction
            if end_idx - start_idx > 1:
                static_friction[:, idx, 0] = materials[:, start_idx:end_idx, 0].mean(dim=-1)
            else:
                static_friction[:, idx] = materials[:, start_idx:end_idx, 0]

        return static_friction.squeeze(-1)

    def __name__(self):
        return "FrictionObservation"


"""
Joint observations for FDM.
"""


def joint_torque(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint torques of the asset."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque


def joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions of the asset."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos


def joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities of the asset."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Most recent low-level action sent to the environment."""
    return env.action_manager.action


def second_last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Low-level action sent one environment step before the most recent action."""
    return env.action_manager.prev_action


"""
Joint history observations for FDM.

These replace the ActuatorNetMLP's internal history buffers which are not
available when using ImplicitActuator.
"""


class JointHistoryBuffer:
    """Manages joint position error and velocity history for FDM observations.

    This is a replacement for ActuatorNetMLP's internal history buffers,
    designed to work with ImplicitActuator.
    """

    _instances = {}

    def __init__(self, num_envs: int, num_joints: int, history_length: int = 5, device: str = "cpu"):
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.history_length = history_length
        self.device = device

        # History buffers
        self._joint_pos_error_history = torch.zeros(num_envs, history_length, num_joints, device=device)
        self._joint_vel_history = torch.zeros(num_envs, history_length, num_joints, device=device)

        # Track last commanded positions for computing error
        self._last_cmd_pos = torch.zeros(num_envs, num_joints, device=device)

    @classmethod
    def get_instance(cls, env) -> "JointHistoryBuffer":
        """Get or create a singleton instance for the environment."""
        key = id(env)
        if key not in cls._instances:
            num_envs = env.num_envs
            num_joints = env.scene["robot"].num_joints
            device = str(env.device)
            cls._instances[key] = cls(num_envs, num_joints, history_length=5, device=device)
        return cls._instances[key]

    def update(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, cmd_pos: torch.Tensor):
        """Update history buffers with current state.

        Args:
            joint_pos: Current joint positions (num_envs, num_joints)
            joint_vel: Current joint velocities (num_envs, num_joints)
            cmd_pos: Commanded joint positions (num_envs, num_joints)
        """
        # Roll history
        self._joint_pos_error_history = torch.roll(self._joint_pos_error_history, 1, dims=1)
        self._joint_vel_history = torch.roll(self._joint_vel_history, 1, dims=1)

        # Compute position error (commanded - actual)
        pos_error = cmd_pos - joint_pos

        # Store current values
        self._joint_pos_error_history[:, 0] = pos_error
        self._joint_vel_history[:, 0] = joint_vel

        # Update last commanded positions
        self._last_cmd_pos = cmd_pos.clone()

    def reset(self, env_ids: torch.Tensor):
        """Reset history for specified environments."""
        self._joint_pos_error_history[env_ids] = 0.0
        self._joint_vel_history[env_ids] = 0.0
        self._last_cmd_pos[env_ids] = 0.0

    def get_pos_error_history(self, history_idx: int = 0) -> torch.Tensor:
        """Get joint position error at specified history index."""
        return self._joint_pos_error_history[:, history_idx]

    def get_vel_history(self, history_idx: int = 0) -> torch.Tensor:
        """Get joint velocity at specified history index."""
        return self._joint_vel_history[:, history_idx]


def _update_joint_history(env: ManagerBasedRLEnv):
    """Update joint history buffers (call this before reading history)."""
    buffer = JointHistoryBuffer.get_instance(env)
    robot = env.scene["robot"]
    # Get commanded positions from action manager
    cmd_pos = env.action_manager.action  # This is the joint position command
    buffer.update(robot.data.joint_pos, robot.data.joint_vel, cmd_pos)


def joint_pos_error_history(
    env: ManagerBasedRLEnv, history_idx: int = 0
) -> torch.Tensor:
    """Joint position error at specified history index.

    Args:
        env: The environment instance.
        history_idx: History index (0 = current, 1 = one step ago, etc.)

    Returns:
        torch.Tensor: Joint position error (num_envs, num_joints)
    """
    buffer = JointHistoryBuffer.get_instance(env)
    return buffer.get_pos_error_history(history_idx)


def joint_velocity_history(
    env: ManagerBasedRLEnv, history_idx: int = 0
) -> torch.Tensor:
    """Joint velocity at specified history index.

    Args:
        env: The environment instance.
        history_idx: History index (0 = current, 1 = one step ago, etc.)

    Returns:
        torch.Tensor: Joint velocity (num_envs, num_joints)
    """
    buffer = JointHistoryBuffer.get_instance(env)
    return buffer.get_vel_history(history_idx)


"""
Velocity observations for FDM.
"""


def base_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def base_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def projected_gravity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Projected gravity in the asset's root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


"""
Height scan observations for FDM.
"""


def height_scan_square_fdm(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    shape: list[int] | None = None,
    offset: float = 0.5,
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame given in the square pattern of the sensor."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    height = sensor.data.ray_hits_w[..., 2] + offset - sensor.data.pos_w[:, 2].unsqueeze(1)
    # assign max distance to inf values
    height[torch.isinf(height)] = sensor.cfg.max_distance
    height[torch.isnan(height)] = sensor.cfg.max_distance

    shape = shape if shape is not None else [int(math.sqrt(height.shape[1])), int(math.sqrt(height.shape[1]))]
    # unflatten the height scan to make use of spatial information
    height_square = torch.unflatten(height, 1, (shape[0], shape[1]))
    # the height scan is mirrored as the pattern is created from neg to pos whereas in the robotics frame, the left of
    # the robot is positive and the right is negative
    height_square = torch.flip(height_square, dims=[1])
    # unsqueeze to make compatible with convolutional layers
    return height_square.unsqueeze(1)
