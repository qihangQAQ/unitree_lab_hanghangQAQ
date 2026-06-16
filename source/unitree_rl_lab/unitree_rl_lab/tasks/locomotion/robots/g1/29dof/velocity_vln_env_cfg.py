"""G1 VLN environment configuration for VLM-guided navigation testing.

This config is a stripped-down version of velocity_env_cfg.py, designed for P0 VLM integration:
- Same G1 29-dof robot and policy observation format (compatible with velocity_env_cfg checkpoints)
- RGB TiledCamera for VLM visual input (512x512)
- Flat terrain, single env, no termination/curriculum/domain randomization
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.unitree import G1_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# Simple flat terrain for VLN testing
FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Scene with G1 robot, flat terrain, and RGB camera."""

    # flat terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FLAT_TERRAIN_CFG,
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

    # robot
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # RGB camera for VLM input (mounted on torso, same pose as depth camera in velocity_depth_env_cfg)
    rgb_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/rgb_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0488, 0.01, 0.4378),
            rot=(0.9135, 0.00436, 0.4067, 0.0),
            convention="world",
        ),
        spawn=PinholeCameraCfg(
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(60) / 2),
            vertical_aperture=2 * math.tan(math.radians(45) / 2),
            clipping_range=(0.1, 20.0),
        ),
        data_types=["rgb"],
        width=512,
        height=512,
        update_period=0.02,
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    # lighting
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Minimal events — only clean reset, no randomization."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class CommandsCfg:
    """Velocity command config — compatible with velocity_env_cfg policy."""

    base_velocity = mdp.DiscreteAxisLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),  # never resample (we override manually)
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.DiscreteAxisLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
        limit_ranges=mdp.DiscreteAxisLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class ActionsCfg:
    """Joint position actions — same as velocity_env_cfg."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observations matching velocity_env_cfg format for policy compatibility."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations — must match velocity_env_cfg exactly."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False  # no noise during VLM testing
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Empty reward config for inference-only VLN play."""

    pass


@configclass
class TerminationsCfg:
    """Only time_out termination — no early termination during VLN testing."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RobotVlnEnvCfg(ManagerBasedRLEnvCfg):
    """Base env config for G1 VLN testing."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=128, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 600.0  # long episode for VLN testing
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Sensor update periods
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.rgb_camera.update_period = self.decimation * self.sim.dt

        # Disable terrain curriculum
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotVlnPlayEnvCfg(RobotVlnEnvCfg):
    """Play-mode config: single env, flat terrain, no noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
