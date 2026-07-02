"""Dataset utilities for training FDM from collected rollout pickle files."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def load_fdm_payload(path: str | Path) -> dict[str, Any]:
    """Load a rollout pickle produced by ``collect_fdm_rollouts.py``."""
    with open(Path(path).expanduser(), "rb") as f:
        payload = pickle.load(f)
    if "data" not in payload:
        raise ValueError(f"Expected payload with a 'data' key, got keys: {list(payload.keys())}")
    return payload


class OfflineFDMTrajectoryDataset(Dataset):
    """Slice collected FDM rollouts into fixed-horizon training examples."""

    def __init__(
        self,
        payload: dict[str, Any],
        prediction_horizon: int = 10,
        num_samples: int | None = None,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 0,
        device: str = "cpu",
        normalize_hard_contact: bool = True,
        collision_rate: float | None = None,
        include_collision_samples: bool = True,
        sample_filter_first_steps_coll: int = 0,
        small_motion_ratio: float | None = 0.1,
        small_motion_threshold: float = 1.0,
        height_threshold: float | None = None,
        outlier_threshold: float = 10.0,
    ):
        self.meta = payload.get("meta", {})
        data = payload["data"]
        self.actions = data["actions"].float().to(device)
        self.states_raw = data["states"].float().to(device)
        self.proprio = data["proprioceptive"].float().to(device)
        self.extero = data["exteroceptive"].float().to(device)
        self.prediction_horizon = prediction_horizon
        self.history_length = self.states_raw.shape[2]
        self.state_dim_raw = self.states_raw.shape[-1]
        self.proprio_dim = self.proprio.shape[-1]
        self.height_scan_shape = tuple(self.extero.shape[-2:])
        self.command_timestep = float(self.meta.get("command_timestep", 0.5))
        self.normalize_hard_contact = normalize_hard_contact
        self.collision_rate_target = collision_rate
        self.include_collision_samples = include_collision_samples
        self.sample_filter_first_steps_coll = sample_filter_first_steps_coll
        self.small_motion_ratio = small_motion_ratio
        self.small_motion_threshold = small_motion_threshold
        self.height_threshold = height_threshold
        self.outlier_threshold = outlier_threshold
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

        max_start = self.actions.shape[1] - prediction_horizon - 1
        if max_start <= 1:
            raise ValueError(
                f"Trajectory length {self.actions.shape[1]} is too short for horizon {prediction_horizon}."
            )

        all_regular_indices = torch.cartesian_prod(
            torch.arange(self.actions.shape[0], device=device),
            torch.arange(1, max_start, device=device),
        )
        perm = torch.randperm(all_regular_indices.shape[0], generator=self.generator, device=device)
        all_regular_indices = all_regular_indices[perm]
        split_at = int(all_regular_indices.shape[0] * (1.0 - val_ratio))
        if split == "train":
            regular_indices = all_regular_indices[:split_at]
        elif split in {"val", "validation"}:
            regular_indices = all_regular_indices[split_at:]
        else:
            raise ValueError(f"Unknown split '{split}'. Expected train or val.")

        collision_indices = torch.empty((0, 2), device=device, dtype=torch.long)
        if include_collision_samples:
            collision_indices = self._sample_collision_indices(max_start=max_start)
            if collision_indices.numel() > 0:
                perm = torch.randperm(collision_indices.shape[0], generator=self.generator, device=device)
                collision_indices = collision_indices[perm]
                coll_split_at = int(collision_indices.shape[0] * (1.0 - val_ratio))
                collision_indices = collision_indices[:coll_split_at] if split == "train" else collision_indices[coll_split_at:]

        indices = self._select_indices(regular_indices, collision_indices, num_samples)
        indices, self.filter_stats = self._filter_indices(indices)
        self.indices = indices.cpu()

        transformed0 = self._transform_state_history(self.states_raw[:1, :1].reshape(1, self.history_length, -1))
        self.state_dim = transformed0.shape[-1]
        self._normalize_limits = self._compute_hard_contact_limits() if normalize_hard_contact else None
        self._collision_rate = self._compute_collision_rate_for_indices(self.indices)

    @property
    def collision_rate(self) -> float:
        return self._collision_rate

    def __len__(self) -> int:
        return self.indices.shape[0]

    def __getitem__(self, index: int):
        env_idx, start_idx = self.indices[index].tolist()
        state_seq = self.states_raw[env_idx, start_idx]
        state_history = self._transform_state_history(state_seq.unsqueeze(0)).squeeze(0)
        if self._normalize_limits is not None and state_history.shape[-1] > 5:
            state_history[..., 5] = self._normalize_hard_contact(state_history[..., 5])

        future_raw = torch.stack(
            [self.states_raw[env_idx, start_idx + step + 1, 0] for step in range(self.prediction_horizon)], dim=0
        )
        target = self._transform_future(future_raw.unsqueeze(0), self.states_raw[env_idx, start_idx, 0].unsqueeze(0))
        target = target.squeeze(0)
        target = self._hold_after_collision(target)
        if self._normalize_limits is not None and target.shape[-1] > 5:
            target[..., 5] = self._normalize_hard_contact(target[..., 5])

        action_seq = torch.stack(
            [self.actions[env_idx, start_idx + step] for step in range(self.prediction_horizon)], dim=0
        )
        extero = self.extero[env_idx, start_idx]
        if extero.dim() == 2:
            extero = extero.unsqueeze(0)
        perfect_velocity = self._perfect_velocity_following(action_seq)
        add_extero = torch.zeros(1, dtype=state_history.dtype)
        return state_history, self.proprio[env_idx, start_idx], extero.float(), action_seq, add_extero, target, perfect_velocity

    def _select_indices(
        self, regular_indices: torch.Tensor, collision_indices: torch.Tensor, num_samples: int | None
    ) -> torch.Tensor:
        """Select regular and collision-near windows, matching official FDM sampling behavior."""
        if num_samples is None:
            if self.collision_rate_target is None:
                return torch.vstack([regular_indices, collision_indices]) if len(collision_indices) else regular_indices
            num_samples = regular_indices.shape[0]

        if self.collision_rate_target is None:
            regular = regular_indices[: min(num_samples, regular_indices.shape[0])]
            if len(collision_indices):
                return torch.vstack([regular, collision_indices])
            return regular

        regular_count = min(int(num_samples * (1.0 - self.collision_rate_target)), regular_indices.shape[0])
        collision_count = int(num_samples * self.collision_rate_target)
        regular = regular_indices[:regular_count]
        if collision_count == 0 or collision_indices.shape[0] == 0:
            return regular
        if collision_count <= collision_indices.shape[0]:
            collision = collision_indices[:collision_count]
        else:
            repeats = collision_count // collision_indices.shape[0] + 1
            collision = collision_indices.repeat(repeats, 1)[:collision_count]
        return torch.vstack([regular, collision])

    def _sample_collision_indices(self, max_start: int) -> torch.Tensor:
        """Sample windows that start shortly before a collision, as in official FDM."""
        if self.state_dim_raw <= 7:
            return torch.empty((0, 2), device=self.states_raw.device, dtype=torch.long)
        collision_samples = torch.where(self.states_raw[:, 1 : -self.prediction_horizon + 2, 0, 7] > 0.5)
        if collision_samples[0].shape[0] == 0:
            return torch.empty((0, 2), device=self.states_raw.device, dtype=torch.long)
        offsets = torch.randint(
            2,
            self.prediction_horizon,
            (collision_samples[0].shape[0],),
            generator=self.generator,
            device=self.states_raw.device,
        )
        collision_abs_idx = collision_samples[1] + 1
        start_idx = torch.clip(collision_abs_idx - offsets, min=1, max=max_start - 1)
        return torch.vstack([collision_samples[0], start_idx]).T

    def _filter_indices(self, indices: torch.Tensor) -> tuple[torch.Tensor, dict[str, int]]:
        if indices.shape[0] == 0:
            return indices, {"initial": 0, "kept": 0}

        state_history, target, future_local_pos = self._window_tensors_for_indices(indices)
        keep = torch.ones(indices.shape[0], dtype=torch.bool, device=indices.device)
        stats = {"initial": indices.shape[0]}

        if target.shape[-1] > 4:
            collision_env, collision_idx = torch.where(target[..., 4] > 0.5)
            early_collision = torch.unique(collision_env[collision_idx < self.sample_filter_first_steps_coll])
            keep[early_collision] = False
            stats["early_collision"] = int(early_collision.shape[0])

            initial_collision = torch.where(state_history[:, 0, 4] > 0.5)[0]
            keep[initial_collision] = False
            stats["initial_collision"] = int(initial_collision.shape[0])

        if self.small_motion_ratio is not None:
            small_motion = torch.where(torch.norm(torch.abs(target[:, -1, :2]), dim=1) < self.small_motion_threshold)[0]
            small_ratio = small_motion.shape[0] / max(target.shape[0], 1)
            if small_ratio > self.small_motion_ratio:
                num_remove = int(
                    (self.small_motion_ratio * target.shape[0] - small_motion.shape[0])
                    / (self.small_motion_ratio - 1)
                )
                if num_remove > 0:
                    perm = torch.randperm(small_motion.shape[0], generator=self.generator, device=indices.device)
                    remove = small_motion[perm[:num_remove]]
                    keep[remove] = False
                    stats["small_motion"] = int(remove.shape[0])

        if self.height_threshold is not None:
            height_diff = torch.max(torch.abs(future_local_pos[:, 1:, 2] - future_local_pos[:, :-1, 2]), dim=-1)[0]
            low_height = torch.where(height_diff < self.height_threshold)[0]
            keep[low_height] = False
            stats["height_threshold"] = int(low_height.shape[0])

        max_distance = torch.norm(target[:, -1, :2], dim=1)
        outlier = torch.where(
            torch.logical_or(
                torch.any(torch.any(torch.abs(target[..., :3]) > self.outlier_threshold, dim=-1), dim=-1),
                max_distance > self.outlier_threshold,
            )
        )[0]
        keep[outlier] = False
        stats["outlier"] = int(outlier.shape[0])
        stats["kept"] = int(keep.sum().item())
        return indices[keep], stats

    def _window_tensors_for_indices(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        env_idx = indices[:, 0]
        start_idx = indices[:, 1]
        horizon_offsets = torch.arange(self.prediction_horizon, device=indices.device)
        future_raw = self.states_raw[env_idx[:, None], start_idx[:, None] + horizon_offsets[None, :] + 1, 0]
        initial = self.states_raw[env_idx, start_idx, 0]
        state_history = self._transform_state_history(self.states_raw[env_idx, start_idx])
        target, future_local_pos = self._transform_future(
            future_raw,
            initial,
            return_local_positions=True,
        )
        return state_history, target, future_local_pos

    def _compute_collision_rate_for_indices(self, indices: torch.Tensor) -> float:
        if indices.shape[0] == 0:
            return 0.0
        _, target, _ = self._window_tensors_for_indices(indices.to(self.states_raw.device))
        if target.shape[-1] <= 4:
            return 0.0
        return float(torch.any(target[..., 4] > 0.5, dim=1).float().mean().cpu())

    def _compute_hard_contact_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.state_dim_raw <= 8:
            return torch.tensor(0.0), torch.tensor(1.0)
        values = self.states_raw[..., 8]
        return values.min(), values.max().clamp_min(values.min() + 1e-6)

    def _normalize_hard_contact(self, value: torch.Tensor) -> torch.Tensor:
        assert self._normalize_limits is not None
        min_value, max_value = self._normalize_limits
        return (value - min_value) / (max_value - min_value + 1e-6)

    def _transform_state_history(self, state_history: torch.Tensor) -> torch.Tensor:
        initial = state_history[:, 0, :7]
        initial_pos = initial[:, :3]
        initial_quat = initial[:, 3:7]
        positions_local = quat_rotate(quat_conjugate(initial_quat[:, None, :]), state_history[:, :, :3] - initial_pos[:, None, :])
        yaw = yaw_from_quat(quat_mul(quat_conjugate(initial_quat[:, None, :]), state_history[:, :, 3:7]))
        yaw_sin_cos = torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)
        rest = state_history[:, :, 7:]
        return torch.cat([positions_local[:, :, :2], yaw_sin_cos, rest], dim=-1)

    def _transform_future(
        self, future_states: torch.Tensor, initial_state: torch.Tensor, return_local_positions: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        initial_pos = initial_state[:, :3]
        initial_quat = initial_state[:, 3:7]
        positions_local = quat_rotate(quat_conjugate(initial_quat[:, None, :]), future_states[:, :, :3] - initial_pos[:, None, :])
        yaw = yaw_from_quat(quat_mul(quat_conjugate(initial_quat[:, None, :]), future_states[:, :, 3:7]))
        yaw_sin_cos = torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)
        rest = future_states[:, :, 7:]
        target = torch.cat([positions_local[:, :, :2], yaw_sin_cos, rest], dim=-1)
        if return_local_positions:
            return target, positions_local
        return target

    def _hold_after_collision(self, target: torch.Tensor) -> torch.Tensor:
        if target.shape[-1] <= 4:
            return target
        collision_idx = torch.where(target[:, 4] > 0.5)[0]
        if collision_idx.shape[0] == 0:
            return target
        first = int(collision_idx[0].item())
        target = target.clone()
        target[first:] = target[first]
        return target

    def _perfect_velocity_following(self, actions: torch.Tensor) -> torch.Tensor:
        distance = actions * self.command_timestep
        cumulative_yaw = distance[:, 2].cumsum(dim=0)
        rot = torch.stack(
            [
                torch.stack([torch.cos(cumulative_yaw), -torch.sin(cumulative_yaw)], dim=-1),
                torch.stack([torch.sin(cumulative_yaw), torch.cos(cumulative_yaw)], dim=-1),
            ],
            dim=-2,
        )
        rot = torch.roll(rot, shifts=1, dims=0)
        rot[0] = torch.eye(2, device=rot.device, dtype=rot.dtype)
        local_steps = torch.matmul(rot, distance[:, :2].unsqueeze(-1)).squeeze(-1)
        cumulative_pos = local_steps.cumsum(dim=0)
        return torch.cat([cumulative_pos, torch.sin(cumulative_yaw[:, None]), torch.cos(cumulative_yaw[:, None])], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(dim=-1)
    bx, by, bz, bw = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(v[..., :1])
    return quat_mul(quat_mul(q, torch.cat([v, zeros], dim=-1)), quat_conjugate(q))[..., :3]


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
