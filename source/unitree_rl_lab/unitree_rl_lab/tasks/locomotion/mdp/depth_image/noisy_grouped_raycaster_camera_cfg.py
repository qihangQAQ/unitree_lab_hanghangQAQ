from dataclasses import MISSING

from isaaclab.utils import configclass

from .grouped_ray_caster_camera_cfg import GroupedRayCasterCameraCfg
from .noisy_camera_cfg import NoisyCameraCfgMixin
from .noisy_grouped_raycaster_camera import NoisyGroupedRayCasterCamera


@configclass
class NoisyGroupedRayCasterCameraCfg(NoisyCameraCfgMixin, GroupedRayCasterCameraCfg):
    class_type: type = NoisyGroupedRayCasterCamera
