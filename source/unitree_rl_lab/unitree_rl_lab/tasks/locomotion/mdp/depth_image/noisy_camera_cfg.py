from isaaclab.utils import configclass


@configclass
class NoisyCameraCfgMixin:
    noise_pipeline: dict[str, configclass] = {}
    data_histories: dict[str, int] = {}
