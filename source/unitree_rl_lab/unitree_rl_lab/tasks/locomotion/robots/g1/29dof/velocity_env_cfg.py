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
        # # 1. 随机金字塔阶梯
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.0, 0.23), # 下界改为 0.0
        #     step_width=0.3,
        #     platform_width=3.0,
        # ),
        # 2. 离散障碍物
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            obstacle_height_range=(0.0, 0.08), # 下界改为 0.0
            obstacle_width_range=(1.0, 2.0),
            num_obstacles=40,
            obstacle_height_mode="fixed",
        ),
        # # 3. 标准阶梯
        # "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.0, 0.2), # 下界改为 0.0
        #     step_width=0.3,
        #     platform_width=3.0
        # ),
        # 4. 坑洼/波浪地面
        "random_rough": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            amplitude_range=(0.0, 0.08), # 下界改为 0.0
            num_waves=3,
        ),
        # 5. 平地保持不变
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        # 不规则地面地形（坑洼）
        "random_uniform": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1,
            noise_range=(0.0, 0.08),   # 最大 8cm 凹凸
            noise_step=0.02,
        ),
        # # 不规则高低地形（台阶）
        # "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.05,
        #     grid_width=0.49,
        #     grid_height_range=(0.0, 0.25),
        # ),

        # 坡度地形
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.15, 0.35),   # 约 8.6°‑14.3°
            platform_width=3.0,
        ),
        # "pit": terrain_gen.MeshPitTerrainCfg(
        #     proportion=0.05,
        #     pit_depth_range=(0.3, 0.5), # 坑深 30‑50cm
        # ),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=ROUGH_TERRAINS_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level= 0 ,
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
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
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
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-5.0, 5.0),
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
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP  (LeggedLab-aligned: heading-based angular velocity)."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.57, 1.57),
            heading=(-3.1416, 3.1416),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.57, 1.57),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True,
        clip={".*": (-1.0, 1.0)},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        # 高度扫描（187）
        # height_scanner = ObsTerm(
        #     func=mdp.height_scan_hpc,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("height_scanner"),
        #         "offset": 0.78,  # 👈 核心修改：与目标躯干高度对齐，让平地归零
        #     },
        #     scale = 1.0,
        #     clip=(-1.0, 1.0),    # 既然平地归零了，台阶和坑的起伏很少超过 1 米
        #     noise=Unoise(n_min=-0.1, n_max=0.1) # 根据 base_env_config.py 还原 0.1 的噪声
        # )
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})


        def __post_init__(self):
            self.history_length = 5
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
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # height_scanner = ObsTerm(
        #     func=mdp.height_scan_hpc,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("height_scanner"),
        #         "offset": 0.78,  # 👈 核心修改：与目标躯干高度对齐，让平地归零
        #     },
        #     scale = 1.0,
        #     clip=(-1.0, 1.0),    # 既然平地归零了，台阶和坑的起伏很少超过 1 米
        #     noise=Unoise(n_min=-0.1, n_max=0.1) # 根据 base_env_config.py 还原 0.1 的噪声
        # )

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP  (LeggedLab-aligned structure)."""

    # ==========================================
    # 1. 任务与存活 (Task & Survival)
    # ==========================================

    # 鼓励机器人跟踪目标水平面内的线速度指令（X、Y 方向）。
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # 鼓励机器人跟踪目标偏航角速度指令（Z 轴旋转），使用 world 坐标系。
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # 强烈惩罚回合终止，促使机器人尽可能长时间存活。
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # ==========================================
    # 2. 基座与姿态 (Base & Posture)
    # ==========================================

    # 惩罚基座在 Z 方向的线速度，防止上下跳动。
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    # 惩罚基座绕 X、Y 轴的角速度，抑制俯仰与滚转晃动。
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # 惩罚基座倾斜（重力投影），鼓励躯干在水平面上保持直立。
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # 惩罚躯干连杆自身倾斜，进一步稳定上半身姿态。
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")},
    )
    # 惩罚非脚踝部位与地面的接触（如躯干、手臂等）。
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"),
        },
    )
    # 惩罚双脚同时离地（腾空），避免跳跃。
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # ==========================================
    # 3. 关节控制与平滑 (Joints & Regularization)
    # ==========================================

    # 惩罚过大的关节加速度，抑制关节运动的突然抖动。
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # 惩罚相邻帧动作输出的变化率，促使控制信号平滑。
    action_rate = RewTerm(func=mdp.action_rate_l2_safe, weight=-0.01, params={"clip_value": 1.0})
    # 惩罚关节位置超出软限位，保护机械结构。
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    # 惩罚能量消耗（关节速度与力矩乘积的绝对值之和）。
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    # 惩罚髋、肩、肘关节偏离中立位置。
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_yaw.*",
                    ".*_hip_roll.*",
                    ".*_shoulder_pitch.*",
                    ".*_elbow.*",
                ],
            )
        },
    )
    # 惩罚手臂和腰部关节偏离中立位置。
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                    ".*_shoulder_roll.*",
                    ".*_shoulder_yaw.*",
                    ".*_wrist.*",
                ],
            )
        },
    )
    # 惩罚腿部关节（髋 pitch、膝、踝）偏离中立位置。
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle.*"],
            )
        },
    )

    # ==========================================
    # 4. 足端与步态 (Feet & Gait)
    # ==========================================

    # 鼓励单脚支撑时合理的步态切换节奏（LeggedLab 风格：不依赖 phase）。
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={
            "command_name": "base_velocity",
            "threshold": 0.4,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    # 惩罚脚在接触地面时的水平滑动，防止打滑。
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    # 惩罚脚部撞击竖直障碍物或台阶边缘，防止绊倒。
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
    )
    # 惩罚双脚距离过近，避免自碰撞。
    feet_too_near = RewTerm(
        func=mdp.feet_too_near,
        weight=-2.0,
        params={
            "threshold": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )
    # 惩罚足端着地时产生过大的冲击力，促使软着陆。
    feet_force = RewTerm(
        func=mdp.feet_contact_force_penalty,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 500.0,
            "max_excess": 400.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    # bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso.*"), "threshold": 1.0},
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

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
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
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
        self.scene.num_envs = 4
        
        # --- 核心地形难度控制 ---
        
        # 1. 确保 Play 时完全禁用地形课程，防止环境原点乱跑
        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None 

        # 2. 地形网格设计 (10行 x 20列 = 200块场地)
        # self.scene.terrain.terrain_generator.num_rows = 10
        # self.scene.terrain.terrain_generator.num_cols = 20

        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 4

        # 2. 锁定地形难度（最低难度0 - 最高难度1）
        # 在原本 10 个等级（0到9）中，Level 2 的难度大约是：2 / (10 - 1) ≈ 0.222
        # 我们将难度下界和上界都死锁在 0.222，这样生成出来的所有地形都是标准的 Level 2 难度
        self.scene.terrain.terrain_generator.difficulty_range = (1, 1)
        
        # 3.机器人出生点难度设置
        self.scene.terrain.max_init_terrain_level = 9

        # 4. 关闭地形自动升降级
        # 训练时需要课程，但 play 时如果它摔倒了就会被传回简单地形。
        # 关掉它，让机器人死磕当前脚下的复杂地形，方便你观察。
        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None
 

        # 命令范围已在 CommandsCfg 中一步到位，无需从 limit_ranges 覆盖
