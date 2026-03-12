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

    # 这里注意：虽然有 reset_base 和 reset_robot_joints，
    # 但由于我们的 HandTrackingCommand 在 _resample_command 中强制瞬移了 Base 和 Joints，
    # 这里的随机重置会被 Command 的逻辑优雅地覆盖掉（或者你可以考虑关掉这里的坐标重置）。
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw":(0,0)},
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
    hand_tracking = mdp.HandTrackingCommandCfg(

        asset_name="robot",
        
        # 【新增】真实数据集路径
        dataset_dir="/tmp/spray_painting_paths_v1",

        # 设为极大值，强制仅在 episode reset 时才生成新曲线
        resampling_time_range=(1.0e9, 1.0e9),

        # G1右手末端 right_wrist_yaw_link
        ee_link_idx=29,

        step_dt=0.02,

        # 喷枪长度15cm
        gun_length=0.15,
        
        # 【新增】待机姿态下，右手相对于基座的本地坐标系偏移量 (用于基座开局对齐)
        default_ee_local_pos=(0.4, -0.2, 0.2),

        debug_vis=True,

        ranges=mdp.HandTrackingCommandCfg.Ranges(
            velocity=(0.10, 0.10),  # 初始阶段固定 10cm/s
            spray_distance=(0.05, 0.10),
            path_length=(1.0, 6.0),
        ),

        # 这个 limit_ranges 是为你以后的“课程设计”预留的满级目标
        limit_ranges=mdp.HandTrackingCommandCfg.Ranges(
            velocity=(0.10, 0.40),
            spray_distance=(0.05, 0.15),
            path_length=(3.0, 6.0),
        )
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

        # 命令观测
        tracking_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "hand_tracking"}
        )

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))

        last_action = ObsTerm(func=mdp.last_action)

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

        tracking_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "hand_tracking"}
        )

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # =================== 喷漆任务核心奖励 ===================
    # 1. 软位置追踪 (容忍度较大 10cm)，引导致使手臂靠近曲线
    ee_pos_tracking_soft = RewTerm(
        func=mdp.ee_reach_pos_target_soft,
        weight=2.0,
        params={"command_name": "hand_tracking", "std": 0.10},
    )

    # 2. 硬位置追踪 (容忍度极小 2cm)，逼迫高精度贴合
    ee_pos_tracking_tight = RewTerm(
        func=mdp.ee_reach_pos_target_tight,
        weight=3.0,
        params={"command_name": "hand_tracking", "std": 0.02},
    )

    # 3. 姿态追踪 (喷枪严格对准墙面)
    ee_rot_tracking = RewTerm(
        func=mdp.reach_rot_target,
        weight=2.5,
        params={"command_name": "hand_tracking", "std": 0.15},
    )

    # 4. 切向速度跟踪 (保持匀速移动)
    # 更改
    ee_tangential_speed_tracking = RewTerm(
        func=mdp.ee_tangential_speed_tracking,
        weight=2.0,
        params={
            "command_name": "hand_tracking",
            "asset_cfg": SceneEntityCfg("robot"),
            "ee_body_name": "right_wrist_yaw_link",
            "std": 0.08,
        },
    )

    # 5. 动作平滑度惩罚 (喷漆必须丝滑)
    action_smoothness = RewTerm(
        func=mdp.ee_action_smoothness_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 
    # 6. 保持 reset 时 base 朝向（新增，抑制“侧身跟随”）
    base_heading_hold = RewTerm(
        func=mdp.base_heading_hold,
        weight=1.0,
        params={
            "command_name": "hand_tracking",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 新增 -- 1. 引导底盘靠近投影点
    base_xy_pos_tracking = RewTerm(
        func=mdp.base_xy_pos_tracking,
        weight=2.0,
        params={"command_name": "hand_tracking", "std": 0.15},
    )
    # 新增 -- 2. 引导底盘速度（侧滑步，竖直立定）
    base_xy_velocity_tracking = RewTerm(
        func=mdp.base_xy_velocity_tracking,
        weight=3.0,
        params={"command_name": "hand_tracking", "std": 0.1},
    )

    # 新增 -- 3. 强制迈腿奖励（不迈腿就扣分，鼓励学习 G1 的经典步态）
    feet_gait_spray = RewTerm(
        func=mdp.feet_gait_spray,  # 指向我们刚刚在 rewards.py 里写好的新函数
        weight=2.0,                # 权重给足，让它明确知道不迈腿就会被严惩
        params={
            "period": 0.6,         # G1 的典型单次迈步周期 (0.6秒左右比较稳健)
            "offset": [0.0, 0.5],  # 双足机器人的灵魂：左腿相位 0，右腿落后半圈 (0.5)
            # 这里复用你之前配置好的 contact_forces 传感器和脚部 body 名字
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), 
            "command_name": "hand_tracking", 
            "move_speed_thresh": 0.02,  # 阈值：当 XY 期望速度大于 2cm/s 时才要求迈步
        },
    )



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
        weight=-0.5,
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
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    # base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})

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
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    
    # 【新增】当路径走完时提前终止本轮 Episode
    path_completed = DoneTerm(
        func=mdp.path_completed, 
        params={"command_name": "hand_tracking"}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    hand_tracking_levels = CurrTerm(
        func=mdp.hand_tracking_levels,
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
        # 【修改】Episode 时长缩短为 45s
        self.episode_length_s = 45.0
        
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15

        # update sensor update periods
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

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

        # 1. 减少环境数量：Play 模式不需要 4096 个环境
        self.scene.num_envs = 1

        # 2. 地形设置：稍微调整一下规模
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5

        # 3. 设置命令难度为“最终目标”
        self.commands.hand_tracking.ranges = self.commands.hand_tracking.limit_ranges

        # 4. 禁用课程学习 (Curriculum)
        self.curriculum.hand_tracking_levels = None