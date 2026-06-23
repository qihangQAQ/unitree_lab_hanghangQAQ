"""FDM data collection environment configuration for G1-29dof.

This configuration extends the velocity_perception environment for FDM data collection.
It adds FDM-specific sensors and observations while keeping the original policy observations intact.
"""

import math

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import G1_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp as loco_mdp
from unitree_rl_lab.tasks.fdm import FDM_DATA_DIR
from unitree_rl_lab.tasks.fdm.mdp.events import TerrainAnalysisRootResetLocal
from unitree_rl_lab.tasks.fdm.mdp.nav_terrain_importer import FDMNavTerrainImporterCfg
from unitree_rl_lab.tasks.fdm.mdp import observations as fdm_mdp
from unitree_rl_lab.tasks.fdm.mdp import terminations as fdm_term


def add_hidden_ground_plane(physics_material: sim_utils.RigidBodyMaterialCfg = None):
    """Add a hidden ground plane for USD terrain.

    This replicates NavTerrainImporter's groundplane logic.
    The ground plane provides collision for the robot while the USD terrain is only visual.

    Args:
        physics_material: Physics material configuration. If None, uses default.
    """
    import isaacsim.core.utils.prims as prim_utils

    # Check if ground plane already exists
    if prim_utils.is_prim_path_valid("/World/GroundPlane"):
        print("[INFO] Ground plane already exists, skipping creation")
        return

    if physics_material is None:
        physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )

    # Add a ground plane (same as NavTerrainImporter)
    ground_plane_cfg = sim_utils.GroundPlaneCfg(physics_material=physics_material)
    ground_plane = ground_plane_cfg.func("/World/GroundPlane", ground_plane_cfg)
    ground_plane.visible = False  # Hidden - only provides collision

    print("[INFO] Added hidden ground plane for USD terrain collision")


# ============================================================
# Scene Configuration
# ============================================================


@configclass
class FDMCollectSceneCfg(InteractiveSceneCfg):
    """Scene configuration for FDM data collection with G1 robot.

    Uses USD terrain (maze) and includes both:
    - Policy height scanner (17x11 for velocity_perception policy)
    - FDM height scanner (60x46 for FDM model training)
    """

    # Ground terrain: USD maze terrain
    terrain = FDMNavTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{FDM_DATA_DIR}/Terrains/navigation_terrain_wall_usd_merge_large_single_object_maze.usd",
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
        env_spacing=10.0,
        usd_uniform_env_spacing=10.0,
        regular_spawning=False,
        groundplane=True,
        add_colliders=True,
    )

    # Robot
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Policy height scanner (for velocity_perception policy input: 17x11 = 187)
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # FDM height scanner (for FDM model training: 60x46)
    fdm_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 4.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(4.5, 5.9)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=10.0,
    )

    # Contact forces sensor
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=6,  # FDM uses 6 for history
        track_air_time=True,
    )

    # Lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ============================================================
# Commands Configuration
# ============================================================


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = loco_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1000000.0, 1000000.0),  # Never resample - we inject commands manually
        rel_standing_envs=0.0,  # No standing envs
        rel_heading_envs=0.0,  # No heading control
        heading_command=False,
        debug_vis=False,
        ranges=loco_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 1.5),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


# ============================================================
# Actions Configuration
# ============================================================


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = loco_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


# ============================================================
# Policy and Critic Observations (copied from velocity_perception)
# ============================================================


@configclass
class PolicyCfg(ObsGroup):
    """Policy observations for Actor - same as velocity_perception."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(func=loco_mdp.base_ang_vel, scale=1.0)
    projected_gravity = ObsTerm(func=loco_mdp.projected_gravity)
    velocity_commands = ObsTerm(func=loco_mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos_rel = ObsTerm(func=loco_mdp.joint_pos_rel)
    joint_vel_rel = ObsTerm(func=loco_mdp.joint_vel_rel, scale=1.0)
    last_action = ObsTerm(func=loco_mdp.last_action)
    height_scanner = ObsTerm(
        func=loco_mdp.height_scan_hpc,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "offset": 0.5,
        },
        scale=1.0,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )

    def __post_init__(self):
        self.history_length = 1
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class CriticCfg(ObsGroup):
    """Critic observations - same as velocity_perception."""

    base_lin_vel = ObsTerm(func=loco_mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=loco_mdp.base_ang_vel, scale=1.0)
    projected_gravity = ObsTerm(func=loco_mdp.projected_gravity)
    velocity_commands = ObsTerm(func=loco_mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos_rel = ObsTerm(func=loco_mdp.joint_pos_rel)
    joint_vel_rel = ObsTerm(func=loco_mdp.joint_vel_rel, scale=1.0)
    last_action = ObsTerm(func=loco_mdp.last_action)
    feet_contact = ObsTerm(
        func=loco_mdp.feet_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 0.5,
        },
    )
    height_scanner = ObsTerm(
        func=loco_mdp.height_scan_hpc,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "offset": 0.5,
        },
        scale=1.0,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )

    def __post_init__(self):
        self.history_length = 1


# ============================================================
# FDM State Observations
# ============================================================


@configclass
class FdmStateCfg(ObsGroup):
    """FDM state observations."""

    base_position = ObsTerm(func=fdm_mdp.base_position)
    base_orientation = ObsTerm(func=fdm_mdp.base_orientation_xyzw)
    base_collision = ObsTerm(
        func=fdm_mdp.base_collision,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso.*"), "threshold": 1.0},
    )
    hard_contact = ObsTerm(func=fdm_mdp.energy_consumption, params={"energy_scale_factor": 0.001})
    friction = ObsTerm(
        func=fdm_mdp.FrictionObservation(),
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])},
    )

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


# ============================================================
# FDM Proprioceptive Observations
# ============================================================


@configclass
class FdmProprioceptiveCfg(ObsGroup):
    """FDM proprioceptive observations."""

    velocity_commands = ObsTerm(func=loco_mdp.generated_commands, params={"command_name": "base_velocity"})
    projected_gravity = ObsTerm(func=fdm_mdp.projected_gravity)
    base_lin_vel = ObsTerm(func=fdm_mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=fdm_mdp.base_ang_vel)
    joint_torque = ObsTerm(func=fdm_mdp.joint_torque)
    joint_pos = ObsTerm(func=fdm_mdp.joint_pos)
    joint_vel = ObsTerm(func=fdm_mdp.joint_vel)
    joint_pos_error_idx0 = ObsTerm(func=fdm_mdp.joint_pos_error_history, params={"history_idx": 0})
    joint_pos_error_idx2 = ObsTerm(func=fdm_mdp.joint_pos_error_history, params={"history_idx": 2})
    joint_pos_error_idx4 = ObsTerm(func=fdm_mdp.joint_pos_error_history, params={"history_idx": 4})
    joint_vel_idx2 = ObsTerm(func=fdm_mdp.joint_velocity_history, params={"history_idx": 2})
    joint_vel_idx4 = ObsTerm(func=fdm_mdp.joint_velocity_history, params={"history_idx": 4})
    last_action = ObsTerm(func=fdm_mdp.last_action)
    second_last_action = ObsTerm(func=fdm_mdp.second_last_action)

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


# ============================================================
# FDM Exteroceptive Observations
# ============================================================


@configclass
class FdmExteroceptiveCfg(ObsGroup):
    """FDM exteroceptive observations (height scan)."""

    height_scan = ObsTerm(
        func=fdm_mdp.height_scan_square_fdm,
        params={
            "sensor_cfg": SceneEntityCfg("fdm_height_scanner"),
            "shape": (60, 46),
            "offset": 0.5,
        },
        clip=(-1.0, 1.5),
    )

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


# ============================================================
# Observations Configuration
# ============================================================


@configclass
class ObservationsCfg:
    """Observation specifications."""

    # Policy observations (for velocity_perception policy Actor)
    policy: PolicyCfg = PolicyCfg()

    # Critic observations (for velocity_perception policy Critic)
    critic: CriticCfg = CriticCfg()

    # FDM observations
    fdm_state: FdmStateCfg = FdmStateCfg()
    fdm_obs_proprioceptive: FdmProprioceptiveCfg = FdmProprioceptiveCfg()
    fdm_obs_exteroceptive: FdmExteroceptiveCfg = FdmExteroceptiveCfg()


# ============================================================
# Events Configuration
# ============================================================


@configclass
class EventCfg:
    """Event configuration for FDM data collection."""

    # startup
    physics_material = EventTerm(
        func=loco_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )

    # reset
    reset_base = EventTerm(
        func=TerrainAnalysisRootResetLocal(
            robot_dim=0.7,
            grid_resolution=0.1,
            safety_margin=0.3,
        ),
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "yaw_range": (-3.14, 3.14),
            "xy_range": {"x": (-4.0, 4.0), "y": (-4.0, 4.0)},
            "max_attempts": 20,
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0, 0),
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=loco_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


# ============================================================
# Terminations Configuration
# ============================================================


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # No time_out - matches FDM official behavior
    # Use delayed contact termination to capture collision data before reset
    base_contact = DoneTerm(
        func=fdm_term.illegal_contact_delayed,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso.*"), "threshold": 1.0, "delay": 1},
    )


# ============================================================
# Main Environment Configuration
# ============================================================


@configclass
class FDMCollectEnvCfg(ManagerBasedRLEnvCfg):
    """FDM data collection environment configuration for G1-29dof.

    This environment is designed for collecting FDM training data using the
    velocity_perception policy as the low-level controller.
    """

    # Scene settings
    scene: FDMCollectSceneCfg = FDMCollectSceneCfg(num_envs=256, env_spacing=10.0)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards = None  # No rewards for data collection
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None  # No curriculum for data collection

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 1000.0  # Long episodes for data collection
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.enable_scene_query_support = True
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = 0.1
        self.scene.fdm_height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class FDMCollectPlayEnvCfg(FDMCollectEnvCfg):
    """Play configuration for FDM data collection."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
