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

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
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
        # 1. 随机金字塔阶梯
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.23), # 下界改为 0.0
            step_width=0.3,
            platform_width=3.0,
        ),
        # 2. 离散障碍物
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            obstacle_height_range=(0.0, 0.2), # 下界改为 0.0
            obstacle_width_range=(1.0, 2.0),
            num_obstacles=40,
        ),
        # 3. 标准阶梯
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.2), # 下界改为 0.0
            step_width=0.3,
            platform_width=3.0
        ),
        # 4. 坑洼/波浪地面
        "random_rough": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            amplitude_range=(0.0, 0.05), # 下界改为 0.0
            num_waves=3,
        ),
        # 5. 平地保持不变
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=ROUGH_TERRAINS_CFG,  # None, COBBLESTONE_ROAD_CFG
        max_init_terrain_level=0,  # 初始从中间难度开始，ROUGH_TERRAINS_CFG有10个等级
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
        debug_vis=True,
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
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
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

        # observation terms (order preserved)
        # 机身角速度（3）
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        # 重力向量（3）、
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # 指令根节点线速度（3）
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 关节位置（29）
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        # 关节速度（29）
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        # 上一帧动作（29）
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        # 新增观测
        # 局部机身线速度（3）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # 局部机身线速度（3维）
        # 基座 高度（1）
        base_height = ObsTerm(func=mdp.base_height)  # 基座高度（1维）
        # •双脚接触力（6）
        feet_contact_forces = ObsTerm(
            func=mdp.feet_contact_forces,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
        )
        # 高度扫描（187）
        height_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )

        def __post_init__(self):
            self.history_length = 1  # 取消历史帧
            self.enable_corruption = False  # 禁用观测干扰
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
        height_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )
        feet_contact_forces = ObsTerm(
            func=mdp.feet_contact_forces,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
        )
        # 基座高度对于critic可能也是有用的特权信息
        base_height = ObsTerm(func=mdp.base_height)

        def __post_init__(self):
            self.history_length = 1

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # 正向奖励，鼓励机器人保持存活（未触发终止条件）。
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    # 惩罚基座在Z方向的线速度，防止机器人上下跳动。
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # 惩罚基座绕X、Y轴的角速度，抑制俯仰与滚转方向的转动。
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # 惩罚关节速度的平方和，限制关节转动过快。
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    # 惩罚关节加速度的平方和，抑制关节运动的突变。
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # 惩罚动作的变化率，促使控制信号平滑。
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    # 惩罚关节位置超出软限位的程度，保护机械结构。
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    # 惩罚能量消耗（关节速度与力矩乘积的绝对值之和）。
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # 惩罚手臂关节偏离默认位置（L1偏差），鼓励回到中立姿态。
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
    # 惩罚腰部关节偏离默认位置，保持腰部姿态。
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
    # 惩罚特定腿部关节（如髋关节）偏离默认位置。
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    # 惩罚基座倾斜，通过重力投影在水平面的分量鼓励保持直立。
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    # 惩罚基座高度偏离目标值（0.78），控制机器人站立高度。
    base_height = RewTerm(
        func=mdp.base_height_l2, 
        weight=-2, 
        params={"target_height": 0.78, "sensor_cfg": SceneEntityCfg("height_scanner")}
    )

    # -- feet
    # 正向奖励，根据相位和接触状态鼓励脚部按步态周期正确着地
    # gait = RewTerm(
    #     func=mdp.feet_gait,
    #     weight=0.5,
    #     params={
    #         "period": 0.8,
    #         "offset": [0.0, 0.5],
    #         "threshold": 0.55,
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    #     },
    # )
    # 新增 -- 1. 奖励摆动脚有合理离地时间
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

    # 新增 -- 2.惩罚脚撞到竖直面/台阶边
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    # 新增 --3.双脚过近惩罚
    feet_too_near = RewTerm(
        func=mdp.feet_too_near,
        weight=-1.0,
        params={
            "threshold": 0.18,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # 新增 -- 4.静止命令时鼓励双脚都接触地面：可选，小权重
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "command_name": "base_velocity",
        },
    )

    # 新增 -- 5. 足端落地冲击惩罚
    feet_force = RewTerm(
        func=mdp.feet_contact_force_penalty,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 500.0,
            "max_excess": 400.0,
        },
    )
  
    # 惩罚脚在接触地面时的水平滑动，防止打滑。
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    # # 正向奖励，鼓励摆动脚在离地阶段达到目标离地高度。
    # feet_clearance = RewTerm(
    #     func=mdp.foot_clearance_reward,
    #     weight=1.0,
    #     params={
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "target_height": 0.1,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    #     },
    # )




    # 惩罚除脚踝外其他身体部位与地面的接触，避免意外碰撞。
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
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.9})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    terrain_levels = CurrTerm(func=mdp.terrain_levels_hpc_style)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


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
        self.scene.terrain.terrain_generator.num_rows = 10
        self.scene.terrain.terrain_generator.num_cols = 20

        # self.scene.terrain.terrain_generator.num_rows = 1
        # self.scene.terrain.terrain_generator.num_cols = 4

        # 2. 锁定地形难度（最低难度0 - 最高难度1）
        # 在原本 10 个等级（0到9）中，Level 2 的难度大约是：2 / (10 - 1) ≈ 0.222
        # 我们将难度下界和上界都死锁在 0.222，这样生成出来的所有地形都是标准的 Level 2 难度
        self.scene.terrain.terrain_generator.difficulty_range = (0, 1)
        
        # 3.机器人出生点难度设置
        self.scene.terrain.max_init_terrain_level = 9

        # 4. 关闭地形自动升降级
        # 训练时需要课程，但 play 时如果它摔倒了就会被传回简单地形。
        # 关掉它，让机器人死磕当前脚下的复杂地形，方便你观察。
        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None
 

        # 放开速度命令范围进行评估
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
