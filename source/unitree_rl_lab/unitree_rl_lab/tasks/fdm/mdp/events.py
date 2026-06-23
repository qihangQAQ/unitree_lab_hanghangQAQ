from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TerrainAnalysisRootResetLocal:
    """Local USD-safe root reset inspired by nav-suite's TerrainAnalysisRootReset.

    This avoids importing nav-suite into the IsaacLab environment while preserving the
    important safety behavior: sample over USD origins, query terrain height by raycast,
    and spawn above the local maximum height around the robot footprint.
    """

    def __init__(
        self,
        robot_dim: float = 0.7,
        grid_resolution: float = 0.1,
        safety_margin: float = 0.3,
        ray_start_height: float = 20.0,
        ray_distance: float = 50.0,
    ):
        self.robot_dim = robot_dim
        self.grid_resolution = grid_resolution
        self.safety_margin = safety_margin
        self.ray_start_height = ray_start_height
        self.ray_distance = ray_distance

    def _raycast_closest(self, starts: torch.Tensor, directions: torch.Tensor, max_dist: float) -> torch.Tensor | None:
        try:
            import carb
            from omni.physx import get_physx_scene_query_interface
        except Exception:
            return None

        hits = []
        query = get_physx_scene_query_interface()
        for start, direction in zip(starts.detach().cpu().numpy(), directions.detach().cpu().numpy()):
            hit = query.raycast_closest(carb.Float3(start), carb.Float3(direction), max_dist)
            if hit["hit"]:
                position = hit["position"]
                hits.append([position[0], position[1], position[2]])
            else:
                hits.append([float("nan"), float("nan"), float("nan")])
        return torch.tensor(hits, device=starts.device, dtype=starts.dtype)

    def _local_spawn_height(self, xy: torch.Tensor, fallback_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        radius = self.robot_dim * 0.5
        offsets = torch.tensor(
            [
                [0.0, 0.0],
                [radius, 0.0],
                [-radius, 0.0],
                [0.0, radius],
                [0.0, -radius],
                [radius, radius],
                [radius, -radius],
                [-radius, radius],
                [-radius, -radius],
            ],
            device=xy.device,
            dtype=xy.dtype,
        )
        sample_xy = xy[:, None, :] + offsets[None, :, :]
        starts = torch.zeros((sample_xy.numel() // 2, 3), device=xy.device, dtype=xy.dtype)
        starts[:, :2] = sample_xy.reshape(-1, 2)
        starts[:, 2] = self.ray_start_height
        directions = torch.zeros_like(starts)
        directions[:, 2] = -1.0

        hit_points = self._raycast_closest(starts, directions, self.ray_distance)
        if hit_points is None:
            return fallback_z, torch.ones(xy.shape[0], device=xy.device, dtype=torch.bool)

        hit_z = hit_points[:, 2].reshape(xy.shape[0], -1)
        valid = torch.isfinite(hit_z).any(dim=1)
        hit_z = torch.where(torch.isfinite(hit_z), hit_z, fallback_z[:, None])
        return torch.max(hit_z, dim=1).values, valid

    def _has_clearance(self, xy: torch.Tensor, ground_z: torch.Tensor) -> torch.Tensor:
        radius = self.robot_dim * 0.5
        directions_xy = torch.tensor(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [math.sqrt(0.5), math.sqrt(0.5)],
                [math.sqrt(0.5), -math.sqrt(0.5)],
                [-math.sqrt(0.5), math.sqrt(0.5)],
                [-math.sqrt(0.5), -math.sqrt(0.5)],
            ],
            device=xy.device,
            dtype=xy.dtype,
        )
        starts = torch.zeros((xy.shape[0] * directions_xy.shape[0], 3), device=xy.device, dtype=xy.dtype)
        starts[:, :2] = xy[:, None, :].expand(-1, directions_xy.shape[0], -1).reshape(-1, 2)
        starts[:, 2] = ground_z[:, None].expand(-1, directions_xy.shape[0]).reshape(-1) + 0.35
        directions = torch.zeros_like(starts)
        directions[:, :2] = directions_xy.repeat(xy.shape[0], 1)

        hit_points = self._raycast_closest(starts, directions, radius)
        if hit_points is None:
            return torch.ones(xy.shape[0], device=xy.device, dtype=torch.bool)
        blocked = torch.isfinite(hit_points[:, 2]).reshape(xy.shape[0], -1).any(dim=1)
        return ~blocked

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: torch.Tensor,
        yaw_range: tuple[float, float],
        velocity_range: dict[str, tuple[float, float]],
        xy_range: dict[str, tuple[float, float]] | None = None,
        max_attempts: int = 20,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        root_states = asset.data.default_root_state[env_ids].clone()

        if xy_range is None:
            xy_range = {"x": (-4.0, 4.0), "y": (-4.0, 4.0)}

        env_origins = env.scene.env_origins[env_ids]
        positions = root_states[:, :3] + env_origins
        accepted = torch.zeros(len(env_ids), device=asset.device, dtype=torch.bool)

        for _ in range(max_attempts):
            pending = ~accepted
            if not pending.any():
                break
            count = int(pending.sum().item())
            x = sample_uniform(xy_range["x"][0], xy_range["x"][1], (count, 1), device=asset.device).squeeze(1)
            y = sample_uniform(xy_range["y"][0], xy_range["y"][1], (count, 1), device=asset.device).squeeze(1)
            candidate_xy = env_origins[pending, :2] + torch.stack((x, y), dim=1)

            fallback_z = env_origins[pending, 2]
            ground_z, valid_height = self._local_spawn_height(candidate_xy, fallback_z)
            clear = self._has_clearance(candidate_xy, ground_z)
            good = valid_height & clear

            pending_indices = torch.nonzero(pending, as_tuple=False).squeeze(-1)
            good_indices = pending_indices[good]
            positions[good_indices, :2] = candidate_xy[good]
            positions[good_indices, 2] = (
                ground_z[good] + root_states[good_indices, 2] + max(self.grid_resolution * 2.0, self.safety_margin)
            )
            accepted[good_indices] = True

        if (~accepted).any():
            pending_indices = torch.nonzero(~accepted, as_tuple=False).squeeze(-1)
            fallback_xy = env_origins[~accepted, :2]
            fallback_z, _ = self._local_spawn_height(fallback_xy, env_origins[~accepted, 2])
            positions[pending_indices, :2] = fallback_xy
            positions[pending_indices, 2] = (
                fallback_z + root_states[pending_indices, 2] + max(self.grid_resolution * 2.0, self.safety_margin)
            )

        yaw_samples = sample_uniform(yaw_range[0], yaw_range[1], (len(env_ids), 1), device=asset.device)
        orientations = quat_from_euler_xyz(
            torch.zeros_like(yaw_samples), torch.zeros_like(yaw_samples), yaw_samples
        ).squeeze(1)

        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
        velocities = root_states[:, 7:13] + rand_samples

        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
        if isinstance(asset, Articulation):
            asset.write_joint_state_to_sim(
                asset.data.default_joint_pos[env_ids].clone(),
                asset.data.default_joint_vel[env_ids].clone(),
                env_ids=env_ids,
            )

    def __name__(self):
        return "TerrainAnalysisRootResetLocal"
