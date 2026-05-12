from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import cv2

from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera

    from .noisy_grouped_raycaster_camera import NoisyGroupedRayCasterCamera
    from .grouped_ray_caster_camera import GroupedRayCasterCamera


def _debug_visualize_image(
    image: torch.Tensor,
    scale_up_vis: int = 5,
    window_name: str = "vis_image",
) -> None:
    img = (image * 255.0 / image.max()).cpu().numpy().astype("uint8")
    img = cv2.resize(img, (img.shape[1] * scale_up_vis, img.shape[0] * scale_up_vis), interpolation=cv2.INTER_AREA)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey(1)


def visualizable_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    data_type: str = "rgb",
    debug_vis: bool = False,
    scale_up_vis: int = 5,
    history_skip_frames: int = 0,
) -> torch.Tensor:
    sensor: TiledCamera | Camera | RayCasterCamera | GroupedRayCasterCamera | NoisyGroupedRayCasterCamera = (
        env.scene.sensors[sensor_cfg.name]
    )

    images = sensor.data.output[data_type].clone()
    if "history" in data_type:
        images = images.squeeze(-1)
        if history_skip_frames > 0:
            images = images[:, ::history_skip_frames, :, :]
    else:
        images = images.permute(0, 3, 1, 2)

    if debug_vis:
        _debug_visualize_image(
            images.permute(1, 2, 0, 3).flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2), scale_up_vis
        )

    return images


class delayed_visualizable_image(ManagerTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.sensor_cfg = cfg.params.get("sensor_cfg", SceneEntityCfg("camera"))
        self.data_type = cfg.params["data_type"]
        assert "history" in self.data_type, "data_type must have 'history' in it"
        self.sensor: NoisyGroupedRayCasterCamera = env.scene.sensors[self.sensor_cfg.name]
        self.delayed_frame_ranges = cfg.params.get("delayed_frame_ranges", (0, 0))
        self.delayed_frame_distribution: Literal["uniform", "log_uniform"] = cfg.params.get(
            "delayed_frame_distribution", "uniform"
        )
        self._num_delayed_frames = torch.zeros(env.num_envs, device=env.device)
        self.history_skip_frames = max(cfg.params.get("history_skip_frames", 1), 1)
        self.num_output_frames = max(cfg.params.get("num_output_frames", 0), 1)
        assert len(self.sensor.data.output[self.data_type].shape) >= 5, (
            f"sensor data of type {self.data_type} should have (N, history, H, W, C) shape, but got"
            f" {self.sensor.data.output[self.data_type].shape}"
        )
        self.sensor_history_length = self.sensor.data.output[self.data_type].shape[1]

        self.frame_offset = torch.flip(
            torch.arange(
                0,
                self.num_output_frames * self.history_skip_frames,
                self.history_skip_frames,
                device=env.device,
            ),
            dims=(0,),
        )

        self.check_delay_bounds()

    def check_delay_bounds(self) -> None:
        max_delayed_frames = self.delayed_frame_ranges[1]
        frames_needed_if_no_delay = (self.num_output_frames - 1) * self.history_skip_frames + 1
        if (frames_needed_if_no_delay + max_delayed_frames) > self.sensor_history_length:
            raise ValueError(
                "The delayed frame ranges are too large for the sensor history length. The maximum delayed frames is"
                f" {max_delayed_frames}, but the frames needed if no delay is {frames_needed_if_no_delay}, which is"
                f" {frames_needed_if_no_delay + max_delayed_frames}."
            )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._num_delayed_frames[env_ids] = _randomize_prop_by_op(
            self._num_delayed_frames[env_ids].unsqueeze(-1),
            self.delayed_frame_ranges,
            None,
            slice(None),
            operation="abs",
            distribution=self.delayed_frame_distribution,
        ).squeeze(-1)

    def __call__(
        self,
        env: ManagerBasedEnv,
        data_type: str,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
        history_skip_frames: int = 0,
        num_output_frames: int = 0,
        delayed_frame_ranges: tuple[int, int] = (0, 0),
        delayed_frame_distribution: Literal["uniform", "log_uniform"] = "uniform",
        debug_vis: bool = False,
        scale_up_vis: int = 5,
    ) -> torch.Tensor:
        images = self.sensor.data.output[self.data_type].clone()
        images = images.squeeze(-1)

        frame_indices = (
            self.sensor_history_length - self.frame_offset.unsqueeze(0) - self._num_delayed_frames.unsqueeze(1) - 1
        )
        frame_indices = frame_indices.to(torch.long)
        assert (frame_indices >= 0).all(), f"frame_indices should be non-negative, but got {frame_indices}"
        assert (
            frame_indices < self.sensor_history_length
        ).all(), f"frame_indices should be less than the sensor history length {self.sensor_history_length}"

        batch_indices = (
            torch.arange(images.shape[0], device=images.device)
            .unsqueeze(1)
            .expand(-1, frame_indices.shape[1])
            .to(torch.long)
        )
        delayed_frames = images[batch_indices, frame_indices]

        if debug_vis:
            _debug_visualize_image(
                delayed_frames.permute(1, 2, 0, 3).flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2),
                scale_up_vis,
            )
        return delayed_frames
