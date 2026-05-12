from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Sequence

from torchvision.transforms import GaussianBlur

if TYPE_CHECKING:
    from .noise_cfg import (
        CropAndResizeCfg,
        DepthNormalizationCfg,
        GaussianBlurNoiseCfg,
        ImageNoiseCfg,
    )


class ImageNoiseModel:
    def __init__(self, cfg: ImageNoiseCfg, num_envs: int = 1, device: str | torch.device = "cpu"):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

    def __call__(self, data: torch.Tensor, cfg: ImageNoiseCfg, env_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        return data

    def reset(self, env_ids: Sequence[int] | None = None):
        pass


def depth_normalization(
    data: torch.Tensor, cfg: DepthNormalizationCfg, env_ids: torch.Tensor | Sequence[int]
) -> torch.Tensor:
    if data.dim() == 4 and data.shape[-1] == 1:
        data = data.permute(0, 3, 1, 2)

    min_depth = cfg.depth_range[0]
    max_depth = cfg.depth_range[1]
    data = data.clip(min_depth, max_depth)

    if cfg.normalize:
        data = (data - min_depth) / (max_depth - min_depth)
        data = data * (cfg.output_range[1] - cfg.output_range[0]) + cfg.output_range[0]

    if len(data.shape) == 4 and data.shape[1] == 1:
        data = data.permute(0, 2, 3, 1)

    return data


def crop_and_resize(
    data: torch.Tensor,
    cfg: CropAndResizeCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    crop_region = cfg.crop_region
    start_up = crop_region[0]
    end_down = data.shape[1] - crop_region[1]
    start_left = crop_region[2]
    end_right = data.shape[2] - crop_region[3]
    cropped = data[:, start_up:end_down, start_left:end_right, :]
    if cfg.resize_shape is None:
        return cropped
    else:
        cropped = cropped.permute(0, 3, 1, 2)
        resized = F.interpolate(cropped, size=cfg.resize_shape, mode="bilinear", align_corners=False)
        resized = resized.permute(0, 2, 3, 1)
        return resized


def gaussian_blur_noise(
    data: torch.Tensor,
    cfg: GaussianBlurNoiseCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    data = data.permute(0, 3, 1, 2)
    blur_transform = GaussianBlur(kernel_size=cfg.kernel_size, sigma=cfg.sigma)
    blurred = blur_transform(data)
    blurred = blurred.permute(0, 2, 3, 1)
    return blurred
