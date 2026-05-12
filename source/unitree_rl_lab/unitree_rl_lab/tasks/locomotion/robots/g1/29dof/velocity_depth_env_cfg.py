import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.terrains import TerrainImporterCfg, FlatPatchSamplingCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import G1_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp import depth_image as depth_image_mdp
from unitree_rl_lab.tasks.locomotion.mdp.depth_image.utils.noise_cfg import (
    CropAndResizeCfg,
    DepthNormalizationCfg,
    GaussianBlurNoiseCfg,
)
# from unitree_rl_lab.tasks.locomotion.mdp.depth_image import (
#     NoisyGroupedRayCasterCameraCfg,
# )
from unitree_rl_lab.tasks.fdm.mdp.terrains import (
    MeshPillarTerrainCfg,
    SingleObjectTerrainCfg,
    StairsRampEvalTerrainCfg,
)
from unitree_rl_lab.tasks.fdm.mdp.terrains.single_object import cross_object_pattern

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)

ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.23),
            step_width=0.3,
            platform_width=3.0,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            obstacle_height_range=(0.0, 0.2),
            obstacle_width_range=(1.0, 2.0),
            num_obstacles=40,
            obstacle_height_mode="fixed",
        ),
        "random_rough": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            amplitude_range=(0.0, 0.05),
            num_waves=3,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_uniform": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1,
            noise_range=(0.0, 0.06),
            noise_step=0.02,
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.15, 0.25),
            platform_width=3.0,
        ),
    },
)
for sub_terrain_name, sub_terrain_cfg in ROUGH_TERRAINS_CFG.sub_terrains.items():
    sub_terrain_cfg.flat_patch_sampling = {
        "init_pos": FlatPatchSamplingCfg(num_patches=2, patch_radius=[0.01, 0.1, 0.5, 1.0], max_height_diff=0.5)
    }


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    # [已注释] 射线深度相机 (MultiMeshRayCaster — 依赖isaaclab2.3.2，当前环境2.3.0不可用)
    # camera = NoisyGroupedRayCasterCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    #     mesh_prim_paths=["/World/ground"],
    #     ray_alignment="yaw",
    #     pattern_cfg=PinholeCameraPatternCfg(
    #         focal_length=1.0,
    #         horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),
    #         vertical_aperture=2 * math.tan(math.radians(58.29) / 2),
    #         width=64,
    #         height=36,
    #     ),
    #     debug_vis=False,
    #     data_types=["distance_to_image_plane"],
    #     update_period=0.02,
    #     depth_clipping_behavior="max",
    #     offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
    #         pos=(0.0487988662332928, 0.01, 0.4378029937970051),
    #         rot=(0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0),
    #         convention="world",
    #     ),
    #     min_distance=0.1,
    #     noise_pipeline={
    #         "crop_and_resize": CropAndResizeCfg(crop_region=(18, 0, 16, 16)),
    #         "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
    #         "depth_normalization": DepthNormalizationCfg(
    #             depth_range=(0.0, 2.5),
    #             normalize=True,
    #             output_range=(0.0, 1.0),
    #         ),
    #     },
    #     data_histories={"distance_to_image_plane_noised": 37},
    # )

    # [新增] USD 渲染深度相机 (TiledCamera, 2.3.0 自带, 支持多mesh自遮挡)
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/depth_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0487988662332928, 0.01, 0.4378029937970051),
            rot=(0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0),
            convention="world",
        ),
        spawn=PinholeCameraCfg(
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),
            vertical_aperture=2 * math.tan(math.radians(58.29) / 2),
            clipping_range=(0.1, 2.5),
        ),
        data_types=["distance_to_image_plane"],
        width=64,
        height=36,
        update_period=0.02,
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_from_terrain,
        mode="reset",
        params={
            "pose_range": {"yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.7), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 机身角速度 (3)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        # 重力向量 (3)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # 指令线速度 + 航向 (4)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 关节位置 (29)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        # 关节速度 (29)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        # 上一帧动作 (29)
        last_action = ObsTerm(func=mdp.last_action)

        # [已注释] 射线深度相机观测
        # depth_image = ObsTerm(
        #     func=depth_image_mdp.delayed_visualizable_image,
        #     params={
        #         "data_type": "distance_to_image_plane_noised_history",
        #         "sensor_cfg": SceneEntityCfg("camera"),
        #         "history_skip_frames": 5,
        #         "num_output_frames": 8,
        #         "delayed_frame_ranges": (0, 1),
        #         "debug_vis": False,
        #     },
        #     noise=None,
        #     history_length=0,
        # )

        # [新增] TiledCamera 深度图 (带噪声管线+历史缓冲)
        depth_image = ObsTerm(
            func=depth_image_mdp.tiled_depth_with_noise_and_history,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "history_length": 37,
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "noise_pipeline": [
                    CropAndResizeCfg(crop_region=(18, 0, 16, 16)),
                    GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
                    DepthNormalizationCfg(
                        depth_range=(0.0, 2.5),
                        normalize=True,
                        output_range=(0.0, 1.0),
                    ),
                ],
            },
            noise=None,
            history_length=0,  # 历史由tiled_depth_with_noise_and_history内部管理
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        # [已注释] 射线深度相机观测
        # depth_image = ObsTerm(
        #     func=depth_image_mdp.delayed_visualizable_image,
        #     params={
        #         "data_type": "distance_to_image_plane_noised_history",
        #         "sensor_cfg": SceneEntityCfg("camera"),
        #         "history_skip_frames": 5,
        #         "num_output_frames": 8,
        #         "delayed_frame_ranges": (0, 1),
        #         "debug_vis": False,
        #     },
        #     noise=None,
        #     history_length=0,
        # )

        # [新增] TiledCamera 深度图
        depth_image = ObsTerm(
            func=depth_image_mdp.tiled_depth_with_noise_and_history,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "history_length": 37,
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "noise_pipeline": [
                    CropAndResizeCfg(crop_region=(18, 0, 16, 16)),
                    GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
                    DepthNormalizationCfg(
                        depth_range=(0.0, 2.5),
                        normalize=True,
                        output_range=(0.0, 1.0),
                    ),
                ],
            },
            noise=None,
            history_length=0,
        )

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # ==========================================
    # 1. 任务与存活
    # ==========================================

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # ==========================================
    # 2. 基座与姿态
    # ==========================================

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)

    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )

    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )

    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # ==========================================
    # 3. 关节控制与平滑
    # ==========================================

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)

    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)

    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # ==========================================
    # 4. 足端与步态
    # ==========================================

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_biped,
        weight=0.50,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "command_name": "base_velocity",
            "threshold": 0.35,
            "max_air_time": 0.8,
        },
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    feet_too_near = RewTerm(
        func=mdp.feet_too_near,
        weight=-1.0,
        params={
            "threshold": 0.18,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "command_name": "base_velocity",
        },
    )

    feet_force = RewTerm(
        func=mdp.feet_contact_force_penalty,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 500.0,
            "max_excess": 400.0,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # ==========================================
    # 5. 安全与物理限制
    # ==========================================

    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso.*"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment with depth camera."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.camera.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16

        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None

        self.scene.terrain.terrain_generator.num_rows = 4
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.difficulty_range = (1, 1)
        self.scene.terrain.max_init_terrain_level = 9

        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
