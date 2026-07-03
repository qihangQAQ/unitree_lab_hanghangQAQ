"""FDM-MPPI demo environment configuration for G1-29dof.

Based on the FDM collect env, but uses **mesh terrain** (generator-based)
instead of USD maze.  Compact, single-env, suitable for MPPI demonstrations.

Usage::

    python scripts/rsl_rl/mppi_play.py \
        --task Unitree-G1-29dof-FDM-Demo \
        --checkpoint <policy_ckpt> --fdm_checkpoint <fdm_ckpt>
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import G1_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp as loco_mdp
from unitree_rl_lab.tasks.fdm.mdp import observations as fdm_mdp
from unitree_rl_lab.tasks.fdm.mdp import terminations as fdm_term
from unitree_rl_lab.tasks.fdm.mdp.events import TerrainAnalysisRootResetLocal
from unitree_rl_lab.tasks.fdm.terrain_cfg import PLANNER_EVAL_CFG as DEMO_TERRAIN_CFG

# ============================================================
# Scene
# ============================================================


@configclass
class FDM_Demo_SceneCfg(InteractiveSceneCfg):
    """Scene with mesh terrain, robot, two height scanners, and contact sensor."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=DEMO_TERRAIN_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
            "TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    fdm_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 4.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(4.5, 5.9)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=10.0,
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=6,
        track_air_time=False,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ============================================================
# Observations (same structure as FDM collect env)
# ============================================================


@configclass
class PolicyCfg(ObsGroup):
    """Policy observations (velocity_perception Actor)."""

    base_ang_vel = ObsTerm(func=loco_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=loco_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(func=loco_mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos_rel = ObsTerm(func=loco_mdp.joint_pos_rel)
    joint_vel_rel = ObsTerm(func=loco_mdp.joint_vel_rel)
    last_action = ObsTerm(func=loco_mdp.last_action)
    height_scanner = ObsTerm(
        func=loco_mdp.height_scan_hpc,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "offset": 0.5,
        },
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False
        self.history_length = 1


@configclass
class FdmStateCfg(ObsGroup):
    """FDM state observations (world pose, collision, energy, friction)."""

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
        self.history_length = 1


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
        self.history_length = 1


@configclass
class FdmExteroceptiveCfg(ObsGroup):
    """FDM exteroceptive observations (60x46 height scan)."""

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
        self.history_length = 1


@configclass
class CriticCfg(ObsGroup):
    """Critic (privileged) observations — must match velocity_perception checkpoint dims."""

    base_lin_vel = ObsTerm(func=loco_mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=loco_mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=loco_mdp.projected_gravity)
    velocity_commands = ObsTerm(func=loco_mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos_rel = ObsTerm(func=loco_mdp.joint_pos_rel)
    joint_vel_rel = ObsTerm(func=loco_mdp.joint_vel_rel)
    last_action = ObsTerm(func=loco_mdp.last_action)
    feet_contact = ObsTerm(
        func=loco_mdp.feet_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "threshold": 0.5},
    )
    height_scanner = ObsTerm(
        func=loco_mdp.height_scan_hpc,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "offset": 0.5,
        },
    )

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False
        self.history_length = 1


@configclass
class ObservationsCfg:
    """Observation specifications."""

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    fdm_state: FdmStateCfg = FdmStateCfg()
    fdm_obs_proprioceptive: FdmProprioceptiveCfg = FdmProprioceptiveCfg()
    fdm_obs_exteroceptive: FdmExteroceptiveCfg = FdmExteroceptiveCfg()


# ============================================================
# Actions (same as FDM collect env)
# ============================================================


@configclass
class ActionsCfg:
    """Action specifications."""

    JointPositionAction = loco_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


# ============================================================
# Commands (same as FDM collect env — never auto-resample)
# ============================================================


@configclass
class CommandsCfg:
    """Command specifications.  MPPI injects commands manually."""

    base_velocity = loco_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e6, 1e6),  # never auto-resample
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=loco_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 1.5),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


# ============================================================
# Events (same as FDM collect env)
# ============================================================


@configclass
class EventCfg:
    """Event configuration."""

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
            "xy_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)},
            "max_attempts": 50,
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
# Terminations
# ============================================================


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=loco_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=fdm_term.illegal_contact_delayed,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso.*"), "threshold": 1.0, "delay": 1},
    )


# ============================================================
# Main Environment Configuration
# ============================================================


@configclass
class FDM_Demo_EnvCfg(ManagerBasedRLEnvCfg):
    """FDM-MPPI demo environment for G1-29dof."""

    scene: FDM_Demo_SceneCfg = FDM_Demo_SceneCfg(num_envs=1, env_spacing=10.0)

    episode_length_s = 1000.0
    decimation = 4

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    rewards = None
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.enable_scene_query_support = True
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = 0.1
        self.scene.fdm_height_scanner.update_period = self.decimation * self.sim.dt
