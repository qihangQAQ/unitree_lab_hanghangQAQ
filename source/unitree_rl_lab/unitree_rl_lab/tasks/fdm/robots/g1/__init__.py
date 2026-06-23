"""FDM tasks for G1-29dof robot."""

import gymnasium as gym

# FDM data collection task using velocity_perception as low-level controller
gym.register(
    id="Unitree-G1-29dof-FDM-Collect",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fdm_collect_env_cfg:FDMCollectEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.fdm_collect_env_cfg:FDMCollectPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_perception_cfg:UnitreePerceptionRunnerCfg",
    },
)
