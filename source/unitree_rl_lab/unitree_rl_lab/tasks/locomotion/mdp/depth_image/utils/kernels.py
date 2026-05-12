import warp as wp


@wp.kernel(enable_backward=False)
def raycast_mesh_kernel_grouped_transformed(
    mesh_wp_ids: wp.array(dtype=wp.uint64),
    mesh_transforms: wp.array(dtype=wp.transform),
    mesh_inv_transforms: wp.array(dtype=wp.transform),
    ray_collision_groups: wp.array(dtype=wp.int32),
    mesh_idxs_for_group: wp.array(dtype=wp.int32),
    meah_idxs_slice_for_group: wp.array(dtype=wp.int32),
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    ray_hits: wp.array(dtype=wp.vec3),
    ray_distance: wp.array(dtype=wp.float32),
    ray_normal: wp.array(dtype=wp.vec3),
    ray_face_id: wp.array(dtype=wp.int32),
    ray_mesh_id: wp.array(dtype=wp.int16),
    max_dist: float = 1e6,
    min_dist: float = 0.0,
    return_distance: int = False,
    return_normal: int = False,
    return_face_id: int = False,
    return_mesh_id: int = False,
):
    tid = wp.tid()
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    ray_distance_buf = float(max_dist)
    ray_collision_group = int(ray_collision_groups[tid])
    start = ray_starts[tid]
    direction = ray_directions[tid]

    for idx in range(
        meah_idxs_slice_for_group[ray_collision_group], meah_idxs_slice_for_group[ray_collision_group + 1]
    ):
        mesh_idx = int(mesh_idxs_for_group[idx])
        mesh_wp_id = mesh_wp_ids[mesh_idx]

        mesh_transform = mesh_transforms[mesh_idx]
        mesh_inv_transform = mesh_inv_transforms[mesh_idx]
        start_local = wp.transform_point(mesh_inv_transform, start)
        direction_local = wp.transform_vector(mesh_inv_transform, direction)

        query_returns = wp.mesh_query_ray(mesh_wp_id, start_local, direction_local, max_dist)
        if query_returns.result and query_returns.t < ray_distance_buf and query_returns.t > min_dist:
            ray_hits[tid] = start + direction * query_returns.t
            ray_distance_buf = query_returns.t
            if return_distance == 1:
                ray_distance[tid] = query_returns.t
            if return_normal == 1:
                n = wp.transform_vector(mesh_transform, query_returns.normal)
                ray_normal[tid] = n
            if return_face_id == 1:
                ray_face_id[tid] = query_returns.face
            if return_mesh_id == 1:
                ray_mesh_id[tid] = wp.int16(mesh_idx)
