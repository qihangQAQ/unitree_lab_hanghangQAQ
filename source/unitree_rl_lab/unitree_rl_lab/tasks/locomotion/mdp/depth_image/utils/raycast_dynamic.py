"""Stub for raycast_dynamic_meshes from Isaac Lab 2.3.2.

This function is imported by multi_mesh_ray_caster* modules but never actually
called at runtime — all _update_buffers_impl() methods are overridden by
GroupedRayCaster/GroupedRayCasterCamera which use raycast_mesh_grouped instead.
"""


def raycast_dynamic_meshes(*args, **kwargs):
    raise NotImplementedError(
        "raycast_dynamic_meshes should not be called — _update_buffers_impl is "
        "overridden by GroupedRayCaster which uses raycast_mesh_grouped."
    )
