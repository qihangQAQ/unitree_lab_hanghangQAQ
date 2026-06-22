"""AME (Attention-Based Map Encoding) velocity tracking environment configuration.

This configuration implements the AME method for learning generalized legged locomotion
with attention-based terrain encoding.
"""

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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import G1_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.terrain.loco_hf_terrains import (
    HfAlternateColumnStakesTerrainCfg,
    HfConcentricGapTerrainCfg,
    HfDoubleColumnStakesTerrainCfg,
    HfStonesBridgeTerrainCfg,
)

# ==============================================================================
# Terrain Configuration
# ==============================================================================

FINETUNE = True

ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=50.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, downsampled_scale=0.1, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_steppingstones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.2,
            stone_height_max=0.05,
            stone_width_range=(0.25, 0.5),
            stone_distance_range=(0.05, 0.25),
            platform_width=2.0,
            holes_depth=-2.0,
            border_width=0.25,
        ),
        "hf_gaps": HfConcentricGapTerrainCfg(
            proportion=0.2,
            gap_width_range=(0.1, 0.5),
            platform_width=2.0,
            border_width=0.25,
            gap_depth=-2.0,
            ground_width_range=(0.5, 0.5),
            ground_height_max=0.025,
        ),
    },
)

FINETUNE_ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=50.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stakes1": HfDoubleColumnStakesTerrainCfg(
            proportion=0.1,
            stake_height_max=0.03,
            stake_side_range=(0.20, 0.40),
            stake_gap_range=(0.1, 0.3),
            column_gap_range=(0.1, 0.1),
            column_jitter=0.0,
            holes_depth=-2.0,
            platform_width=2.0,
            border_width=0.25,
        ),
        "stakes2": HfAlternateColumnStakesTerrainCfg(
            proportion=0.2,
            stake_height_max=0.03,
            stake_side_range=(0.20, 0.40),
            stake_gap_range=(0.05, 0.15),
            column_gap_range=(0.0, 0.2),
            column_jitter=0.0,
            holes_depth=-2.0,
            platform_width=2.0,
            border_width=0.25,
        ),
        "stakes3": HfAlternateColumnStakesTerrainCfg(
            proportion=0.2,
            stake_height_max=0.03,
            stake_side_range=(0.20, 0.40),
            stake_gap_range=(0.05, 0.25),
            column_gap_range=(0.3, 0.2),
            column_jitter=0.0,
            holes_depth=-2.0,
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_gaps": HfConcentricGapTerrainCfg(
            proportion=0.1,
            gap_width_range=(0.2, 0.6),
            platform_width=2.0,
            border_width=0.25,
            gap_depth=-2.0,
            ground_width_range=(0.5, 0.5),
            ground_height_max=0.03,
        ),
        "stonebridge": HfStonesBridgeTerrainCfg(
            proportion=0.1,
            platform_width=2.0,
            border_width=0.25,
            holes_depth=-2.0,
            stone_height_max=0.03,
            stone_width_range=(0.25, 0.35),
            stone_distance_range=(0.3, 0.5),
            stone_length_range=(0.6, 1.0),
            stone_lateral_distance_range=(0.0, 0.0),
        ),
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.1,
            rail_height_range=(0.25, 0.05),
            rail_thickness_range=(0.1, 0.3),
            platform_width=2.0,
        ),
    },
)


# ==============================================================================
# Scene Configuration
# ==============================================================================


@configclass
class AMESceneCfg(InteractiveSceneCfg):
    """Configuration for the AME terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FINETUNE_ROUGH_TERRAINS_CFG if FINETUNE else ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
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
    # 高程图扫描: 33x21 网格, 0.05m 分辨率, 1.6m x 1.0m 范围
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
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


# ==============================================================================
# MDP Settings
# ==============================================================================


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-2.0, n_max=2.0))
        last_action = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": True},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

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
        height_scan = ObsTerm(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
        )

    # privileged observations
    critic: CriticCfg = CriticCfg()


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
            "restitution_range": (0.0, 0.1),
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

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
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
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # ==========================================
    # 1. Task & Survival
    # ==========================================

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    # ==========================================
    # 2. Base & Posture
    # ==========================================

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # ==========================================
    # 3. Joints & Regularization
    # ==========================================

    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)

    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-7)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

    dof_torques_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.05)

    # ==========================================
    # 4. Feet & Gait
    # ==========================================

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 0.6,
        },
    )

    feet_air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    feet_too_near = RewTerm(
        func=mdp.feet_too_near,
        weight=-1.0,
        params={
            "threshold": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    
    # ==========================================
    # 5. Safety & Limits
    # ==========================================

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

    # ==========================================
    # 6. Joint Deviation
    # ==========================================

    # -- coordination (cross-body coordination)
    joint_coordination = RewTerm(
        func=mdp.joint_coordination_rel,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "coord_joints": [
                ["left_hip_pitch_joint", "right_shoulder_pitch_joint"],
                ["right_hip_pitch_joint", "left_shoulder_pitch_joint"],
            ],
            "coord_signs": [
                [1.0, 1.0],
                [1.0, 1.0],
            ],
        },
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
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
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "torso_link",
                    ".*_shoulder_.*_link",
                    ".*_hip_.*_link",
                    ".*_knee_link",
                    ".*_elbow_link",
                    "waist_.*_link",
                    "pelvis",
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# ==============================================================================
# Environment Configuration
# ==============================================================================


@configclass
class AMEEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the AME locomotion velocity-tracking environment."""

    # Scene settings
    scene: AMESceneCfg = AMESceneCfg(num_envs=4096, env_spacing=2.5)
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
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        if FINETUNE:
            # Stage 2: Finetune with noise and randomization
            self.events.reset_base.params = {
                "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            }
            self.commands.base_velocity.ranges.heading = (0.0, 0.0)

            # Finetune reward weights
            self.rewards.termination_penalty.weight = -200.0
            self.rewards.track_lin_vel_xy_exp.weight = 2.0
            self.rewards.track_ang_vel_z_exp.weight = 3.0
            self.rewards.ang_vel_xy_l2.weight = -0.05
            self.rewards.undesired_contacts.weight = -1.0
            self.rewards.joint_vel_l2.weight = -0.001
            self.rewards.joint_acc_l2.weight = -1.25e-7
            self.rewards.dof_torques_l2.weight = -1.5e-7
            self.rewards.dof_pos_limits.weight = -1.0
            self.rewards.dof_torques_limits.weight = -0.05
            self.rewards.action_rate_l2.weight = -0.05
            self.rewards.flat_orientation_l2.weight = -5.0
            self.rewards.feet_air_time.weight = 0.5
            self.rewards.feet_air_time_variance.weight = -2.0
            self.rewards.feet_slide.weight = -0.3
            self.rewards.feet_stumble.weight = -5.0
            self.rewards.feet_too_near.weight = -5.0
            self.rewards.joint_coordination.weight = -0.5
            self.rewards.joint_deviation_hip.weight = -0.1
            self.rewards.joint_deviation_arms.weight = -0.3
            self.rewards.joint_deviation_waists.weight = -1.0
        else:
            # Stage 1: Basic training without noise
            # Disable domain randomization
            self.events.push_robot = None
            self.events.add_base_mass = None
            self.events.base_com = None
            # Disable observation noise
            self.observations.policy.base_ang_vel.noise = None
            self.observations.policy.projected_gravity.noise = None
            self.observations.policy.velocity_commands.noise = None
            self.observations.policy.joint_pos_rel.noise = None
            self.observations.policy.joint_vel_rel.noise = None
            self.observations.policy.last_action.noise = None
            self.observations.policy.height_scan.params["noise"] = False
            # Stage 1 reward weights
            self.rewards.termination_penalty.weight = -200
            self.rewards.track_lin_vel_xy_exp.weight = 2.0
            self.rewards.track_ang_vel_z_exp.weight = 3.0
            self.rewards.ang_vel_xy_l2.weight = -0.05
            self.rewards.undesired_contacts.weight = -1.0
            self.rewards.dof_torques_l2.weight = -1.5e-7
            self.rewards.joint_acc_l2.weight = -1.25e-7
            self.rewards.joint_vel_l2.weight = -0.001
            self.rewards.dof_pos_limits.weight = -1.0
            self.rewards.dof_torques_limits.weight = -0.01
            self.rewards.action_rate_l2.weight = -0.01
            self.rewards.flat_orientation_l2.weight = -2.0
            self.rewards.feet_air_time.weight = 0.25
            self.rewards.feet_air_time_variance.weight = -0.7
            self.rewards.feet_slide.weight = -0.1
            self.rewards.feet_stumble.weight = -2.0
            self.rewards.feet_too_near.weight = -1.0
            self.rewards.joint_coordination.weight = -0.2
            self.rewards.joint_deviation_hip.weight = -0.1
            self.rewards.joint_deviation_arms.weight = -0.3
            self.rewards.joint_deviation_waists.weight = -1.0


@configclass
class AMEPlayEnvCfg(AMEEnvCfg):
    """Configuration for AME play/evaluation."""

    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        # Use flat terrain to reduce Vulkan rendering pressure
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None

        # Disable terrain curriculum
        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None

        # Reset pose
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Fixed velocity for play
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Disable randomization for play
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"] = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
