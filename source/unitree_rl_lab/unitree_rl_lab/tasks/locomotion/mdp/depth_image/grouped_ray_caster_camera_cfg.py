from dataclasses import MISSING
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .grouped_ray_caster_camera import GroupedRayCasterCamera
from .grouped_ray_caster_cfg import GroupedRayCasterCfg


@configclass
class GroupedRayCasterCameraCfg(GroupedRayCasterCfg):
    class_type: type = GroupedRayCasterCamera

    @configclass
    class OffsetCfg:
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        convention: Literal["opengl", "ros", "world"] = "ros"

    offset: OffsetCfg = OffsetCfg()

    data_types: list[str] = ["distance_to_image_plane"]

    depth_clipping_behavior: Literal["max", "zero", "none"] = "none"

    pattern_cfg: PinholeCameraPatternCfg = MISSING

    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/GroupedRayCasterCamera",
        markers={
            "hit": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            ),
        },
    )

    def __post_init__(self):
        self.attach_yaw_only = False
