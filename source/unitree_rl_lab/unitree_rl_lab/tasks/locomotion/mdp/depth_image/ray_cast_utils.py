"""Utility functions for ray-cast sensors (adapted from Isaac Lab 2.3.2)."""

from __future__ import annotations

import torch

import omni.physics.tensors.impl.api as physx

from isaaclab.utils.math import convert_quat

# XformPrimView was removed from Isaac Lab 0.47.2. For our use case,
# this code path is never hit (static ground mesh, track_mesh_transforms=False).
try:
    from isaaclab.sim.views import XformPrimView
except ImportError:
    XformPrimView = None  # type: ignore


def obtain_world_pose_from_view(
    physx_view,
    env_ids: torch.Tensor,
    clone: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if XformPrimView is not None and isinstance(physx_view, XformPrimView):
        pos_w, quat_w = physx_view.get_world_poses(env_ids)
    elif isinstance(physx_view, physx.ArticulationView):
        pos_w, quat_w = physx_view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
        quat_w = convert_quat(quat_w, to="wxyz")
    elif isinstance(physx_view, physx.RigidBodyView):
        pos_w, quat_w = physx_view.get_transforms()[env_ids].split([3, 4], dim=-1)
        quat_w = convert_quat(quat_w, to="wxyz")
    else:
        raise NotImplementedError(f"Cannot get world poses for prim view of type '{type(physx_view)}'.")

    if clone:
        return pos_w.clone(), quat_w.clone()
    else:
        return pos_w, quat_w
