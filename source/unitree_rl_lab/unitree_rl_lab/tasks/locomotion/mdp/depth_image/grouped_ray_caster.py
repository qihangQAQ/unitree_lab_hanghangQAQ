from __future__ import annotations

import logging
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from .multi_mesh_ray_caster import MultiMeshRayCaster
from .ray_cast_utils import obtain_world_pose_from_view

from .utils.raycast import raycast_mesh_grouped

if TYPE_CHECKING:
    from .grouped_ray_caster_cfg import GroupedRayCasterCfg

logger = logging.getLogger(__name__)


class GroupedRayCaster(MultiMeshRayCaster):
    """Grouped Ray Caster sensor reads multiple isaacsim prim path and keep updating the mesh
    positions before casting rays.
    """

    cfg: GroupedRayCasterCfg

    def __init__(self, cfg: GroupedRayCasterCfg):
        super().__init__(cfg)

    def _initialize_warp_meshes(self):
        super()._initialize_warp_meshes()

        total_meshes_per_env = self._mesh_positions_w.shape[1]
        mesh_wp_ids_tensor = torch.zeros(
            (self._num_envs, total_meshes_per_env),
            dtype=torch.int64,
            device=self._device,
        )

        mesh_idx = 0
        for target_cfg in self._raycast_targets_cfg:
            prims = sim_utils.find_matching_prims(target_cfg.prim_expr)
            ids = []
            for prim in prims:
                prim_path = prim.GetPath().pathString
                prim_path_ = regex.sub(r"env_\d+", "env_0", prim_path)
                assert prim_path_ in self.meshes, (
                    f"Mesh at prim path {prim_path} (casted to {prim_path_}) not found in the mesh cache"
                    f" {self.meshes.keys()}"
                )
                ids.append(self.meshes[prim_path_].id)

            ids_tensor = torch.tensor(ids, device=self._device, dtype=torch.int64)
            count = self._num_meshes_per_env[target_cfg.prim_expr]

            if len(ids) == 1:
                mesh_wp_ids_tensor[:, mesh_idx] = ids_tensor[0]
            elif len(ids) == count:
                mesh_wp_ids_tensor[:, mesh_idx : mesh_idx + count] = ids_tensor.unsqueeze(0)
            elif len(ids) == self._num_envs * count:
                mesh_wp_ids_tensor[:, mesh_idx : mesh_idx + count] = ids_tensor.view(self._num_envs, count)
            else:
                logger.warning(f"Mismatch in mesh counts for {target_cfg.prim_expr}")

            mesh_idx += count

        self._mesh_wp_ids = mesh_wp_ids_tensor.flatten()

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        self._create_ray_collision_groups()

    def _create_ray_collision_groups(self):
        self._ray_collision_groups = (
            torch.arange(self._num_envs, dtype=torch.int32, device=self._device).unsqueeze(1).repeat(1, self.num_rays)
        )

        _mesh_idxs_for_group = torch.ones(
            (self._mesh_positions_w.shape[0], self._mesh_positions_w.shape[1]),
            dtype=torch.int32,
            device=self._device,
        ).fill_(-1)
        mesh_idx = 0
        total_meshes = self._mesh_positions_w.shape[1]
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            count = self._num_meshes_per_env[target_cfg.prim_expr]
            indices = (
                torch.arange(self._num_envs, device=self._device).unsqueeze(1) * total_meshes
                + torch.arange(count, device=self._device).unsqueeze(0)
                + mesh_idx
            )
            _mesh_idxs_for_group[:, mesh_idx : mesh_idx + count] = indices.int()
            mesh_idx += count
        self._mesh_idxs_for_group = _mesh_idxs_for_group.flatten(0, 1)

        _meah_idxs_slice_for_group = torch.arange(self._num_envs + 1, dtype=torch.int32, device=self._device)
        _meah_idxs_slice_for_group *= self._mesh_positions_w.shape[1]
        self._meah_idxs_slice_for_group = _meah_idxs_slice_for_group

    def _update_mesh_transforms(self, env_ids: torch.Tensor | None = None):
        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            pos_w, ori_w = obtain_world_pose_from_view(view, None)
            pos_w = pos_w.squeeze(0) if len(pos_w.shape) == 3 else pos_w
            ori_w = ori_w.squeeze(0) if len(ori_w.shape) == 3 else ori_w

            if target_cfg.prim_expr in MultiMeshRayCaster.mesh_offsets:
                pos_offset, ori_offset = MultiMeshRayCaster.mesh_offsets[target_cfg.prim_expr]
                pos_w -= pos_offset
                ori_w = math_utils.quat_mul(ori_offset.expand(ori_w.shape[0], -1), ori_w)

            count = view.count
            if count != 1:
                count = count // self._num_envs
                pos_w = pos_w.view(self._num_envs, count, 3)
                ori_w = ori_w.view(self._num_envs, count, 4)

            self._mesh_positions_w[:, mesh_idx : mesh_idx + count] = pos_w
            self._mesh_orientations_w[:, mesh_idx : mesh_idx + count] = ori_w
            mesh_idx += count

    def _get_mesh_transforms_and_inv_transforms(self):
        mesh_transforms = torch.concatenate(
            [self._mesh_positions_w, self._mesh_orientations_w],
            dim=-1,
        ).reshape(-1, 7)
        inv_q = math_utils.quat_inv(self._mesh_orientations_w)
        inv_p = math_utils.quat_apply(inv_q, -self._mesh_positions_w)
        mesh_inv_transforms = torch.concatenate(
            [inv_p, inv_q],
            dim=-1,
        ).reshape(-1, 7)
        return mesh_transforms, mesh_inv_transforms

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        self._update_ray_infos(env_ids)
        self._update_mesh_transforms(env_ids)

        mesh_transforms, mesh_inv_transforms = self._get_mesh_transforms_and_inv_transforms()

        mesh_wp = [i for i in self.meshes.values()][0]
        self._data.ray_hits_w[env_ids], _, _, _, _ = raycast_mesh_grouped(
            mesh_wp_device=mesh_wp.device,
            mesh_wp_ids=self._mesh_wp_ids,
            mesh_transforms=mesh_transforms,
            mesh_inv_transforms=mesh_inv_transforms,
            ray_group_ids=self._ray_collision_groups[env_ids],
            mesh_idxs_for_group=self._mesh_idxs_for_group,
            meah_idxs_slice_for_group=self._meah_idxs_slice_for_group,
            ray_starts=self._ray_starts_w[env_ids],
            ray_directions=self._ray_directions_w[env_ids],
            max_dist=self.cfg.max_distance,
            min_dist=self.cfg.min_distance,
        )
