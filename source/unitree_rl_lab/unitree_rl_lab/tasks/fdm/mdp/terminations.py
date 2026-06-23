"""FDM-specific termination functions."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import carb

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def illegal_contact_delayed(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, delay: int = 1
) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold after a certain delay.

    This allows to record the observations when the robot is in contact and do not directly terminate.
    Useful in navigation tasks / forward dynamics model learning.
    The delay is multiplied by the decimation of the low-level policy to make sure that the obstacles
    are properly detected.

    Args:
        env: The environment instance.
        threshold: Force threshold in Newtons.
        sensor_cfg: Sensor configuration for contact detection.
        delay: Number of control steps to delay termination (default: 1).

    Returns:
        torch.Tensor: Boolean tensor (num_envs,) indicating which environments should terminate.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    # Make sure delay is not larger than history length
    delay_physics_timestep = delay * env.cfg.decimation
    if delay_physics_timestep >= net_contact_forces.shape[1]:
        carb.log_warn(
            f"Delay {delay} requires a history length of {delay_physics_timestep} but current length is only"
            f" {net_contact_forces.shape[1]}. Setting delay to {net_contact_forces.shape[1] - 1}"
        )
        delay_physics_timestep = net_contact_forces.shape[1] - 1

    # Check if any contact force exceeds the threshold
    # Newest contact force is at history idx 0, so delay removes the newest contact force
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, delay_physics_timestep:, sensor_cfg.body_ids], dim=-1), dim=1)[0]
        > threshold,
        dim=1,
    )
