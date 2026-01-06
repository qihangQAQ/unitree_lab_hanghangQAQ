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
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns,CameraCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg  # 引入针孔相机配置

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# 地形生成配置（定义鹅卵石路面）
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
# 机器人场景配置
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    # 地面地形配置
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",    # 地形在场景中的路径
        terrain_type="generator",     # 地形类型：生成器
        terrain_generator=COBBLESTONE_ROAD_CFG,  # 使用鹅卵石路面生成器
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,           # 碰撞组
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,      # 静摩擦系数
            dynamic_friction=1.0,     # 动摩擦系数
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,        # 是否启用调试可视化
    )
    # robots
    # 机器人配置
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    # 传感器配置
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
    # 灯光配置
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # 障碍设置
    # ================= 1. 一个长方体 (Box) =================
    obstacle_box_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box_0",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.2, 0.5), # 比如这是一个横着的长条
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)), # 红
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1000.0, 0.0, -10.0)), # 初始先扔远点，等 reset 再拉回来
    )

    # ================= 2. 一个圆柱体 (Cylinder) [你的第7个物体] =================
    obstacle_cylinder_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_0",
        spawn=sim_utils.CylinderCfg(
            radius=0.25, height=0.8,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.1)), # 黄
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1000.0, 0.0, -10.0)),
    )
    # ================= 修改后的深度相机配置 =================
    depth_camera = CameraCfg(
        # 1. 修改挂载点为 torso_link
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/depth_cam",

        update_period=0.1,
        height=80,
        width=128,
        data_types=["distance_to_image_plane"],

        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 5.0),
        ),

        # 2. 调整偏移量 (模拟头部位置)
        offset=CameraCfg.OffsetCfg(
            # pos=(x, y, z)
            # x=0.2: 向前伸 20cm (避免被胸部挡住)
            # y=0.0: 居中
            # z=0.5: 向上抬 50cm (假设 torso 原点在腰部，这个值需要您根据实测微调)
            pos=(0.25, 0.0, 0.2),

            # 保持 ROS 坐标系
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )


@configclass
# 事件配置，用于环境中的随机化和重置事件
class EventCfg:
    """Configuration for events."""

    # startup
    # 启动时事件
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),  # 静摩擦系数范围
            "dynamic_friction_range": (0.3, 1.0), # 动摩擦系数范围
            "restitution_range": (0.0, 0.0),      # 恢复系数范围
            "num_buckets": 64,                    # 桶数量
        },
    )

    add_base_mass = EventTerm(    # 随机化基础质量
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),# 质量分布参数
            "operation": "add",# 操作类型：添加
        },
    )

    # reset
    # 重置时事件
    # 重置时施加外部力和扭矩
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),   # 力范围
            "torque_range": (-0.0, 0.0), # 扭矩范围
        },
    )
    # 重置基座状态
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {  # 速度范围
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    # 重置机器人关节
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    # 间隔事件
    # 推动机器人
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
# 命令配置
class CommandsCfg:
    """Command specifications for the MDP."""

    # 基础速度命令配置
    position = mdp.UniformPositionCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 12.0),  # 重采样时间范围
        rel_standing_envs=0,                # 相对站立环境
        heading_command=False,              # 是否使用航向命令
        debug_vis=False,                    # 是否启用调试可视化
        max_goal_time_s=10,                 # 每个目标的最大时间
        randomize_goal_time_minus_s=3.0,    # 随机减去多少（例如 [max - 2, max]）
        # 位置范围
        ranges=mdp.UniformPositionCommandCfg.Ranges(
            pos_1=(1.0, 7.0), pos_2=(-0.5, 0.5), heading=(-0.3, 0.3),use_polar=False
        ),

        limit_ranges = mdp.UniformPositionCommandCfg.Ranges(
            pos_1=(1.0, 12.0),
            pos_2=(-2.0, 2.0),
            heading=(-0.3, 0.3),
            use_polar=False,
        ),

    )


@configclass
# 动作配置
class ActionsCfg:
    """Action specifications for the MDP."""

    # 关节位置动作配置
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
# 观测配置
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    # 策略网络观测组配置
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 观测项（保持顺序）
        # 保留原来观测
        # base角速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))# 基础角速度
        # 重力项（类似于base姿态角）
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))# 投影重力
        
        # 速度项，删除
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})# 速度命令
        # 新增1.位置控制命令（改个名就行）
        position_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "position"})
        
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))# 关节相对位置
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))# 关节相对速度
        last_action = ObsTerm(func=mdp.last_action)# 上一次动作

        # 新增2.剩余时长
        time_left = ObsTerm(func=mdp.command_time_left,params={"command_name": "position"})

        # 新增：射线感知
        ray2d_scan = ObsTerm(
            func=mdp.ray2d_distances,
            params={"command_name": "position"},
            # 仿照 Go1 的 noise_scales.ray2d，添加 0.2 程度的噪声
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )

        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})# 步态相位（已注释）

        def __post_init__(self):
            self.history_length = 5         # 历史长度
            self.enable_corruption = True   # 启用数据损坏
            self.concatenate_terms = True   # 连接观测项

    # observation groups
    # 观测组
    policy: PolicyCfg = PolicyCfg()

    @configclass
    # critic网络观测组配置
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        #相比于policy的观测，只多了一个基座线速度
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel) # 基础线速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 新增1.位置控制命令（改个名就行）
        position_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "position"})

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        # 新增2.剩余时长
        time_left = ObsTerm(func=mdp.command_time_left,params={"command_name": "position"})
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        # 新增：射线感知
        ray2d_scan = ObsTerm(
            func=mdp.ray2d_distances,
            params={"command_name": "position"},
            # 仿照 Go1 的 noise_scales.ray2d，添加 0.2 程度的噪声
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )

        def __post_init__(self):
            self.history_length = 5# 历史长度

    # privileged observations
    # 特权观测
    critic: CriticCfg = CriticCfg()


@configclass
class PosRewardParamsCfg:
    """Position 任务奖励的超参数（不要放到 RewardsCfg 里）"""

    position_target_sigma_soft: float = 1.5   # or 2.0
    position_target_sigma_tight: float = 0.3  # or 0.5
    heading_target_sigma: float = 0.7         # or 1.0
    rew_duration: float = 3.0                 # or 2.0


@configclass
# 奖励配置
# 这里只能写RewTrem实例，不能自己设定变量
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    # track_lin_vel_xy = RewTerm(
    #     func=mdp.track_lin_vel_xy_yaw_frame_exp,
    #     weight=1.0,
    #     params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    # )
    # track_ang_vel_z = RewTerm(
    #     func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )

    # =================== 新增 --task --position ====================================
    # position_target_sigma_soft: float = 1.5
    # position_target_sigma_tight: float = 0.3
    # # 朝向容忍
    # heading_target_sigma: float = 0.7
    # # 时间窗（秒）
    # rew_duration: float = 1.5
    
    # 1. 软位置追踪（到目标附近就给奖励）
    reach_pos_target_soft = RewTerm(
        func=mdp.reach_pos_target_soft,
        weight=3.0,
        params={
            "command_name": "position",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 2. 硬位置追踪（更靠近目标的额外奖励）
    reach_pos_target_tight = RewTerm(
        func=mdp.reach_pos_target_tight,
        weight=2.5,
        params={
            "command_name": "position",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # 3. 靠近目标时，奖励机器人把身体方向对准目标
    reach_heading_target = RewTerm(
        func=mdp.reach_heading_target,
        weight=1.0,
        params={
            "command_name": "position",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 4.鼓励机器人向目标的方向移动，而不是原地抖腿或乱走
    velo_dir = RewTerm(
        func=mdp.velo_dir,
        weight=0.5,
        params={
            "command_name": "position",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 5.机器人到达目标后，要求它站稳、站姿好
    stand_still_pos = RewTerm(
        func=mdp.stand_still_pos,
        weight=-1.5,   # 函数返回偏离量，所以这里是负号
        params={
            "command_name": "position",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 6.惩罚机器人在“离目标还很远”时发呆不动，或者面朝相反方向
    nomove = RewTerm(
        func=mdp.nomove,
        weight=-0.3,
        params={
            "command_name": "position",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 7.越靠近目标越不能摔，摔了惩罚加倍！
    termination = RewTerm(
        func=mdp.termination,
        weight=-1.5,  # 函数返回 0/1/放大系数，这里用负号做惩罚
        params={
            "command_name": "position",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 新增reward(行进过程朝向奖励，防止机器人背对期望位置走过去)
    face_target_while_moving = RewTerm(
        func=mdp.face_target_while_moving,
        weight=1.0,  # 推荐 1.0~2.0，先用 1.5
        params={
            "command_name": "position",
            "move_thresh": 0.4,
            "stop_radius": 0.15,
            "sigma": 0.6,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ============================================================================
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    # 基础相关惩罚
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5)          # 基础线速度Z惩罚
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02)       # 基础角速度XY惩罚
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)                   # 关节速度惩罚
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)                  # 关节加速度惩罚
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)                # 动作变化率惩罚
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)            # 关节位置限制惩罚
    energy = RewTerm(func=mdp.energy, weight=-2e-5)                             # 能量消耗惩罚

    # -- 关节偏差惩罚
    # 手臂关节偏差惩罚
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
    # 腰部关节偏差惩罚
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
    # 腿部关节偏差惩罚
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    # -- 机器人姿态惩罚
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-15, params={"target_height": 0.78})

    # -- feet
    # -- 脚部相关奖励
    # 步态奖励
    gait = RewTerm(
        func=mdp.feet_gait_pos,
        weight=1,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "position",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    # 脚部滑动惩罚
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    # 脚部离地高度奖励
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward_position,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
    # -- 其他惩罚
    # 不期望接触惩罚
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

    #碰撞惩罚
    obstacle_collision = RewTerm(
        func=mdp.obstacle_collision,
        weight=-8.0,  # 初始建议给予较大的负权重
        params={
            "threshold": 1.0,
            # 排除脚部(ankle)，只检测躯干和大腿
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
            "command_name": "position",
        },
    )


@configclass
# 终止条件配置
class TerminationsCfg:
    """Termination terms for the MDP."""

    # time_out = DoneTerm(func=mdp.time_out, time_out=True)# episode超时终止
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})# 基础高度过低终止
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})# 不良朝向终止
    goal_time_out = DoneTerm(func=mdp.goal_time_out, time_out=True)

@configclass
# 课程学习配置
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_pos) # 地形难度级别
    pos_cmd_levels = CurrTerm(mdp.position_cmd_levels)  



@configclass
# 机器人环境配置
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    # 场景设置
    #等价于 scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    # 基础设置
    observations: ObservationsCfg = ObservationsCfg()   # 观测配置
    actions: ActionsCfg = ActionsCfg()                  # 动作配置
    commands: CommandsCfg = CommandsCfg()               # 命令配置
    # MDP settings
    # MDP设置
    rewards: RewardsCfg = RewardsCfg()                  # 奖励配置
    pos_reward_params: PosRewardParamsCfg = PosRewardParamsCfg()#奖励超参数配置

    terminations: TerminationsCfg = TerminationsCfg()   # 终止配置
    events: EventCfg = EventCfg()                       # 事件配置
    curriculum: CurriculumCfg = CurriculumCfg()         # 课程学习配置

    # 后初始化设置
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4             # 降采样率
        self.episode_length_s = 10.0    # 回合长度（秒）
        # simulation settings
        self.sim.dt = 0.005             # 仿真时间步长
        self.sim.render_interval = self.decimation  # 渲染间隔
        self.sim.physics_material = self.scene.terrain.physics_material# 物理材质
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15          # PhysX GPU设置

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt                  # 接触力传感器更新周期
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt# 高度扫描器更新周期

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        # 检查是否启用地形级别课程学习 - 如果是，则为地形生成器启用课程学习
        # 这会生成难度逐渐增加的地形，对训练有用
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True  # 启用课程学习
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False # 禁用课程学习


@configclass
# 用于演示和测试的简化环境配置
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1    # 减少环境数量
        self.scene.terrain.terrain_generator.num_rows = 2       # 减少地形行数
        self.scene.terrain.terrain_generator.num_cols = 10      # 减少地形列数
        # self.commands.position.ranges = self.commands.position.limit_ranges # 使用限制范围作为位置范围

        # 为播放模式设置特定的命令范围
        self.commands.position.ranges = mdp.UniformPositionCommandCfg.Ranges(
            pos_1=(3.0, 8.0),      # 播放模式使用更小的范围
            pos_2=(-0.2, 0.2),     # 更小的横向偏移
            heading=(-0.3, 0.3),   # 更小的朝向变化
            use_polar=False
        )

        self.scene.env_spacing = 0.0
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (-3.14, 3.14),  # 保留随机朝向
        }

