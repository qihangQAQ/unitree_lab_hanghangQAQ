"""Observation term for TiledCamera depth images with noise pipeline + history buffer."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg

from .utils.async_circular_buffer import AsyncCircularBuffer
from .utils.noise_cfg import CropAndResizeCfg, DepthNormalizationCfg, GaussianBlurNoiseCfg
from .utils.noise_model import crop_and_resize, depth_normalization, gaussian_blur_noise

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def apply_noise_pipeline(
    depth: torch.Tensor,
    noise_pipeline: list,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply noise pipeline to raw depth image. depth shape: (N_, H, W, 1)."""
    for noise_cfg in noise_pipeline:
        depth = noise_cfg.func(depth, noise_cfg, env_ids)
    return depth


class tiled_depth_with_noise_and_history(ManagerTermBase):
    """Observation term: read depth from TiledCamera, apply noise, buffer history.

    Returns stacked depth frames of shape (N, num_output_frames, H, W).
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.sensor_cfg: SceneEntityCfg = cfg.params.get("sensor_cfg", SceneEntityCfg("camera"))
        self.history_length: int = cfg.params.get("history_length", 37)
        self.history_skip_frames: int = cfg.params.get("history_skip_frames", 5)
        self.num_output_frames: int = cfg.params.get("num_output_frames", 8)

        # Build noise pipeline from config
        self.noise_pipeline: list = []
        for noise_cfg in cfg.params.get("noise_pipeline", []):
            noise_cfg.device = env.device
            self.noise_pipeline.append(noise_cfg)

        self._buffer = AsyncCircularBuffer(self.history_length, env.num_envs, str(env.device))
        self._step_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        # Precompute frame offset (reverse order: oldest first, newest last)
        self._frame_offset = torch.flip(
            torch.arange(
                0,
                self.num_output_frames * self.history_skip_frames,
                self.history_skip_frames,
                device=env.device,
            ),
            dims=(0,),
        )

        # Verify buffer capacity
        frames_needed = (self.num_output_frames - 1) * self.history_skip_frames + 1
        assert frames_needed <= self.history_length, (
            f"History buffer too short: need {frames_needed} frames but only {self.history_length} slots"
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._buffer.reset(env_ids)
        self._step_counter[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
        history_length: int = 37,
        history_skip_frames: int = 5,
        num_output_frames: int = 8,
        noise_pipeline: list | None = None,
    ) -> torch.Tensor:
        # 1. Read raw depth from TiledCamera
        raw_depth = env.scene.sensors[sensor_cfg.name].data.output["distance_to_image_plane"].clone()
        # raw_depth: (N, H, W, 1)  — squeeze later after noise

        # 2. Apply noise pipeline
        if self.noise_pipeline:
            raw_depth = apply_noise_pipeline(raw_depth, self.noise_pipeline, self._ALL_INDICES)

        # 3. Push to history buffer (store as (N, H, W) for simplicity)
        self._buffer.append(raw_depth.squeeze(-1), self._ALL_INDICES)

        # 4. Sample frames from buffer
        history_buf = self._buffer.buffer  # (N, history_length, H, W)

        frame_indices = self.history_length - self._frame_offset - 1
        frame_indices = frame_indices.clamp(min=0).to(torch.long)  # (num_output_frames,)

        batch_indices = torch.arange(history_buf.shape[0], device=history_buf.device)
        sampled = history_buf[batch_indices[:, None], frame_indices.unsqueeze(0)]  # (N, num_output_frames, H, W)

        return sampled
