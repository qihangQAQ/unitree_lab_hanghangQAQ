from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm


def goal_time_out(env, command_name: str = "position") -> torch.Tensor:
    """
    基于 CommandTerm.time_left 的超时终止条件。

    默认用 position 命令：
      - 找到 env.command_manager 里名字为 command_name 的 command term
      - 当它的 time_left <= 0 的时候，对应 env 触发 done
    """

    # return env.time_left <= 0.0
    cmd_term = env.command_manager.get_term(command_name)
    return cmd_term.time_left <= 0.0