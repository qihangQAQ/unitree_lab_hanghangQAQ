from dataclasses import MISSING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from .multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

from .grouped_ray_caster import GroupedRayCaster


@configclass
class GroupedRayCasterCfg(MultiMeshRayCasterCfg):
    class_type: type = GroupedRayCaster

    min_distance: float = 0.0


def get_link_prim_targets(
    links: list[str],
    prefix: str = "/World/envs/env_.*/Robot/",
    suffix: str = "/visuals",
    is_shared=True,
    **kwargs: dict,
) -> list[MultiMeshRayCasterCfg.RaycastTargetCfg]:
    return [
        MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=f"{prefix}{link}{suffix}", is_shared=is_shared, **kwargs)
        for link in links
    ]
