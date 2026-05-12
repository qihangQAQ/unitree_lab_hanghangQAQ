import torch
from dataclasses import MISSING
from typing import Callable

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg

from .noise_model import (
    ImageNoiseModel,
    crop_and_resize,
    depth_normalization,
    gaussian_blur_noise,
)


@configclass
class ImageNoiseCfg(NoiseCfg):
    func: Callable[[torch.Tensor, NoiseCfg, torch.Tensor], torch.Tensor] | type[ImageNoiseModel] = ImageNoiseModel
    device: str | torch.device = "cpu"


@configclass
class CropAndResizeCfg(ImageNoiseCfg):
    crop_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    resize_shape: tuple[int, int] = None
    func = crop_and_resize


@configclass
class GaussianBlurNoiseCfg(ImageNoiseCfg):
    kernel_size: int = 3
    sigma: float = 1.0
    func = gaussian_blur_noise


@configclass
class DepthNormalizationCfg(ImageNoiseCfg):
    depth_range: tuple[float, float] = (0.0, 10.0)
    normalize: bool = True
    output_range: tuple[float, float] = (0.0, 1.0)
    func = depth_normalization
