"""Transform between world-frame observations and FDM local-frame input/output."""

from __future__ import annotations

import torch

from unitree_rl_lab.tasks.fdm.training.fdm_dataset import (
    quat_conjugate,
    quat_mul,
    quat_rotate,
    yaw_from_quat,
)
from unitree_rl_lab.tasks.fdm.planner.se2_utils import get_se2, get_x_y_yaw


def transform_state_history_to_local(state_history: torch.Tensor) -> torch.Tensor:
    """Transform world-frame state history into the initial-frame representation.

    This mirrors ``FDMTrajectoryDataset._transform_state_history()`` exactly.
    The FDM model was trained on states in this coordinate system, so inference
    must use the identical transform.

    Raw state layout (from FdmStateCfg):
        [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w,
         collision, hard_contact, friction_left, friction_right]

    Args:
        state_history: (batch, history_length, raw_state_dim).
            raw_state_dim >= 7 (pos + quat).  Extra dims passed through unchanged.

    Returns:
        (batch, history_length, state_dim) where:
            [local_x, local_y, sin(yaw), cos(yaw), collision, hard_contact, friction...]
    """
    initial = state_history[:, 0, :7]  # first frame world pose
    initial_pos = initial[:, :3]
    initial_quat = initial[:, 3:7]

    # positions in the frame of the initial pose
    positions_local = quat_rotate(
        quat_conjugate(initial_quat[:, None, :]),
        state_history[:, :, :3] - initial_pos[:, None, :],
    )

    # orientation relative to initial
    yaw = yaw_from_quat(
        quat_mul(quat_conjugate(initial_quat[:, None, :]), state_history[:, :, 3:7])
    )
    yaw_sin_cos = torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)

    # remaining state dims (collision, hard_contact, friction) pass through unchanged
    rest = state_history[:, :, 7:]

    return torch.cat([positions_local[:, :, :2], yaw_sin_cos, rest], dim=-1)


def fdm_output_to_world_frame(
    fdm_trajectory: torch.Tensor,
    world_pose: torch.Tensor,
) -> torch.Tensor:
    """Transform FDM trajectory output from initial-local frame to world frame.

    The FDM model outputs trajectories relative to the robot's pose at prediction
    time.  This function composes those local trajectories with the world pose
    to get world-frame waypoints.

    Args:
        fdm_trajectory: (batch, prediction_horizon, 4) from FDM.forward().
            [local_x, local_y, sin(local_yaw), cos(local_yaw)]
        world_pose: (batch, 7) robot world pose at prediction time.
            [x, y, z, qx, qy, qz, qw]  (matches base_orientation_xyzw)

    Returns:
        (batch, prediction_horizon, 3) world-frame states [x, y, yaw].
    """
    initial_pos = world_pose[:, :2]  # (batch, 2)
    initial_quat = world_pose[:, 3:7]  # (batch, 4)
    initial_yaw = yaw_from_quat(initial_quat)  # (batch,)

    horizon = fdm_trajectory.shape[1]

    # Build SE(2) for robot world pose at prediction time: T_world_initial
    initial_world_state = torch.stack(
        [initial_pos[:, 0], initial_pos[:, 1], initial_yaw], dim=-1
    )  # (batch, 3)
    se2_world_initial = get_se2(initial_world_state[:, None, :].expand(-1, horizon, -1))
    # (batch, horizon, 3, 3)

    # Build SE(2) for FDM local trajectory: T_initial_point
    local_yaw = torch.atan2(fdm_trajectory[:, :, 2], fdm_trajectory[:, :, 3])
    local_state = torch.stack(
        [fdm_trajectory[:, :, 0], fdm_trajectory[:, :, 1], local_yaw], dim=-1
    )  # (batch, horizon, 3)
    se2_local = get_se2(local_state)  # (batch, horizon, 3, 3)

    # Compose: T_world_point = T_world_initial @ T_initial_point
    se2_world = se2_world_initial @ se2_local  # (batch, horizon, 3, 3)

    return get_x_y_yaw(se2_world)  # (batch, horizon, 3)
