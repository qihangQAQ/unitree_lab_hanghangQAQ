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