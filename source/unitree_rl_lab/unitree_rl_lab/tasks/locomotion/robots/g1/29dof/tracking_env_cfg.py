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


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
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
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(5.0, 5.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    hand_tracking = mdp.HandTrackingCommandCfg(
        asset_name="robot",
        ee_link_idx=29,  # 【注意】请确保 29 是你 G1 机器人右腕/末端真实的 link index
        resampling_time_range=(1.0e9, 1.0e9),  # 【关键补丁】：设为极大值，确保命令永不中途超时刷新
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
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 新增 -- 命令观测
        tracking_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "hand_tracking"}
        )

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))

        last_action = ObsTerm(func=mdp.last_action)

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

        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        tracking_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "hand_tracking"}
        )

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""


    # =================== 1. 手臂追踪核心任务 (你总结的 4个追踪 + 1个平滑) ===================
    # (1) 软位置追踪
    ee_pos_tracking_soft = RewTerm(
        func=mdp.ee_reach_pos_target_soft,
        weight=2.0,
        params={"command_name": "hand_tracking", "std": 0.10},
    )
    # (2) 硬位置追踪
    ee_pos_tracking_tight = RewTerm(
        func=mdp.ee_reach_pos_target_tight,
        weight=3.0,
        params={"command_name": "hand_tracking", "std": 0.02},
    )
    # (3) 姿态追踪
    ee_rot_tracking = RewTerm(
        func=mdp.reach_rot_target,
        weight=2.5,
        params={"command_name": "hand_tracking", "std": 0.15},
    )
    # (4) 速度追踪 (注意：这里对应你 rewards.py 里的函数名 ee_velocity_tracking)
    ee_tangential_speed_tracking = RewTerm(
        func=mdp.ee_velocity_tracking,
        weight=2.0,
        params={
            "command_name": "hand_tracking",
            "asset_cfg": SceneEntityCfg("robot"),
            "ee_body_name": "right_wrist_yaw_link",  # 确保这是真实的连杆名
            "std": 0.08,
        },
    )
    # (5) 动作平滑度惩罚
    action_smoothness = RewTerm(
        func=mdp.ee_action_smoothness_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # =================== 2. 底盘移动与特定步态 ===================
    # 特定侧滑步态 (你要求的步态)
    feet_gait_spray = RewTerm(
        func=mdp.feet_gait_spray,  # 请确保你已经把这个函数加到了 rewards.py 中
        weight=2.0,
        params={
            "period": 0.6,
            "offset": [0.0, 0.5],
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "command_name": "hand_tracking",
            "move_speed_thresh": 0.02,
        },
    )

    # =================== 3. 前期单纯的任务追踪  ===================
    base_xy_pos_tracking = RewTerm(func=mdp.base_xy_pos_tracking, weight=2.0,
                                   params={"command_name": "hand_tracking", "std": 0.15})
    base_xy_velocity_tracking = RewTerm(func=mdp.base_xy_velocity_tracking, weight=3.0,
                                        params={"command_name": "hand_tracking", "std": 0.1})
    base_heading_hold = RewTerm(func=mdp.base_heading_hold, weight=1.5, params={"command_name": "hand_tracking"})

    # =================== 4. 手臂锁  ===================
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,  # 【核心修改】极其微弱的惩罚。允许自然摆臂，只惩罚极其夸张的乱甩
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

    # =================== 基础生存与姿态惩罚 ===================
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_.*_joint",
    #                 ".*_elbow_joint",
    #                 ".*_wrist_.*",
    #             ],
    #         )
    #     },
    # )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
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
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})

    # -- feet
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
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
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
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    # 新增：路径走完就结束
    path_completed = DoneTerm(
        func=mdp.path_completed,
        params={"command_name": "hand_tracking"},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    hand_tracking_levels = CurrTerm(
        func=mdp.hand_tracking_levels,  # 指向刚才写的新函数
        params={"reward_term_name": "ee_pos_tracking_tight"}
    )

    # 【新增】动态开启手臂奖励的课程
    arm_reward_levels = CurrTerm(
        func=mdp.arm_tracking_reward_curriculum,
        params={
            "trigger_reward_name": "base_xy_pos_tracking", # 只有底盘走得好，才解锁手臂
            "step_size": 0.05 # 经过 20 次优异表现才完全解锁 (越平滑越不容易掉分)
        }
    )



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
        self.episode_length_s = 60.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15

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
    """测试（play）环境配置，继承自训练配置，并覆盖适合推理的设置。"""

    def __post_init__(self):
        super().__post_init__()

        # ========== 1. 环境数量与仿真 ==========
        self.scene.num_envs = 1                     # 单环境，便于观察
        self.episode_length_s = 60.0                 # 保持与训练一致

        # 地形生成器若存在则调整为小尺寸，加快加载
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5

        # ========== 2. 命令：直接使用最高难度（limit_ranges） ==========
        cmd = self.commands.hand_tracking
        # 将当前 ranges 替换为 limit_ranges 实例（包含速度、喷涂距离、路径长度等）
        cmd.ranges = cmd.limit_ranges
        # 开启可视化，便于观察路径和目标点（可选）
        cmd.debug_vis = True

        # ========== 3. 禁用课程学习 ==========
        self.curriculum.hand_tracking_levels = None
        self.curriculum.arm_reward_levels = None   # 如果有手臂奖励课程，也禁用
        # 如果还有其他课程项（如 terrain_levels），也设为 None
        # self.curriculum.terrain_levels = None

        # ========== 4. 固定随机化事件 ==========
        # 移除 startup 随机化（摩擦、质量）
        self.events.physics_material = None
        self.events.add_base_mass = None

        # 固定 reset 时的位姿（取消随机偏移）
        if self.events.reset_base is not None:
            self.events.reset_base.params["pose_range"] = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        if self.events.reset_robot_joints is not None:
            self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)  # 1.0 倍默认关节角度

        # ========== 5. （可选）关闭观测噪声 ==========
        # self.observations.policy.enable_corruption = False

        # ========== 6. 终止条件（与训练一致，保留 path_completed） ==========
        # terminations 中的 path_completed 已启用，无需修改