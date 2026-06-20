import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# 注册位置控制命令
gym.register(
    id="Unitree-G1-29dof-Position",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point=f"{__name__}.position_env:LeggedRobotPosEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.position_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.position_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# 注册np3o版控制命令
gym.register(
    id="Unitree-G1-29dof-Position_np3o",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point=f"{__name__}.obstacle_avoid_env:LeggedRobotPosNp3oEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.position_env_cfg:RobotNP3OEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.position_env_cfg:RobotNP3OPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_np3o_cfg:UnitreeNp3oRunnerCfg",
    },
)


# 手臂追踪控制命令
gym.register(
    id="Unitree-G1-29dof-Hand_tracking",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point=f"{__name__}.hand_tracking_env:HandTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# 速度命令感知控制（带地形编码器和LSTM）
gym.register(
    id="Unitree-G1-29dof-Velocity-Perception",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_perception_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_perception_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_perception_cfg:UnitreePerceptionRunnerCfg",
    },
)

# 速度命令避障控制（高程图感知 + 前进+heading转向）
gym.register(
    id="Unitree-G1-29dof-Velocity-Obstacle",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_obstacle_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_obstacle_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_obstacle_cfg:UnitreeObstacleRunnerCfg",
    },
)

# 速度命令深度感知控制（用深度图替代高程图）
gym.register(
    id="Unitree-G1-29dof-Velocity-Depth",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_depth_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_depth_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_depth_cfg:UnitreeDepthRunnerCfg",
    },
)

# 速度命令感知控制 — Student 蒸馏（从 Teacher 行为克隆）
gym.register(
    id="Unitree-G1-29dof-Velocity-Student",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_student_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_student_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_student_cfg:UnitreeStudentRunnerCfg",
    },
)

# VLN (Vision-Language Navigation) — RGB 相机 + VLM 高层规划
gym.register(
    id="Unitree-G1-29dof-Velocity-VLN",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_vln_env_cfg:RobotVlnEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_vln_env_cfg:RobotVlnPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# AME (Attention-Based Map Encoding) 速度命令控制
gym.register(
    id="Unitree-G1-29dof-AME",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_ame_env_cfg:AMEEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_ame_env_cfg:AMEPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ame_cfg:AMEPPORunnerCfg",
    },
)
