#!/usr/bin/env python3

"""MPPI-FDM interactive planner demo.

Robot stands still.  Press SPACE to drop a random goal within ~5 m.
MPPI + FDM navigates to it autonomously.  Press SPACE again for the next goal.

Usage:
    python scripts/rsl_rl/mppi_play.py \\
        --task Unitree-G1-29dof-FDM-Demo \\
        --checkpoint logs/rsl_rl/unitree_perception/<run>/model_19999.pt \\
        --fdm_checkpoint logs/fdm/unitree_g1_fdm_online/<run>/model_best.pth
"""

from __future__ import annotations

import argparse
import os
import random
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "source" / "unitree_rl_lab"))

# ============================================================
# Argument parsing
# ============================================================

parser = argparse.ArgumentParser(description="MPPI-FDM interactive planner demo.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-FDM-Demo")
parser.add_argument("--checkpoint", type=str, default=None, help="Low-level locomotion policy checkpoint.")
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--load_checkpoint", type=str, default=None)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--fdm_checkpoint", type=str, required=True, help="Trained FDM model checkpoint.")
parser.add_argument("--disable_fabric", action="store_true", default=False)

# Goal
parser.add_argument("--goal_range", type=float, default=2.5, help="Max distance of random goal from robot (m).")
parser.add_argument("--goal_tolerance", type=float, default=0.5, help="Distance threshold for goal reached (m).")

# Timing
parser.add_argument("--max_steps", type=int, default=20000, help="Maximum simulation steps.")
parser.add_argument("--planner_freq", type=float, default=10.0, help="Planner replan frequency (Hz).")

# MPPI
parser.add_argument("--population_size", type=int, default=1024)
parser.add_argument("--sigma", type=float, default=0.87)
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=0.6)

# FDM
parser.add_argument("--fdm_batch_size", type=int, default=128)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

try:
    import unitree_rl_lab.rsl_rl_ext  # noqa: F401
except ImportError as exc:
    print(f"[WARNING] Could not import unitree_rl_lab.rsl_rl_ext: {exc}")

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import cli_args
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.tasks.fdm.planner.mppi_config import ActionConfig, MPPIConfig, PlannerConfig
from unitree_rl_lab.tasks.fdm.planner.fdm_planner import FDMMPPIPlanner


# ============================================================
# Keyboard reader (reuses pattern from keyboard_play.py)
# ============================================================


class SpaceKeyWatcher:
    """Non-blocking watcher — sets a flag when SPACE is pressed (edge-triggered)."""

    def __init__(self):
        self._old_settings = None
        self._running = False
        self._thread = None
        self.pressed = False  # edge-triggered flag, cleared by consumer

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    def _read_loop(self):
        fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(fd)
        was_space = False
        try:
            tty.setraw(fd)
            while self._running:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    ch = sys.stdin.read(1)
                    if ch == '\x03':  # Ctrl-C
                        self._running = False
                        break
                    is_space = (ch == ' ')
                    # edge: rising edge (not pressed → pressed)
                    if is_space and not was_space:
                        self.pressed = True
                    was_space = is_space
                else:
                    was_space = False  # reset on no key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_settings)

    def restore(self):
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass


# ============================================================
# Helpers
# ============================================================


def _inject_command(base_env, se2_cmd: torch.Tensor):
    """Inject SE(2) velocity into command manager."""
    cmd_term = base_env.command_manager.get_term("base_velocity")
    cmd_term.vel_command_b[:, 0] = se2_cmd[:, 0]
    cmd_term.vel_command_b[:, 1] = se2_cmd[:, 1]
    cmd_term.vel_command_b[:, 2] = se2_cmd[:, 2]


def _disable_command_autoupdate(base_env):
    """Prevent command manager from overwriting our injected commands."""
    cmd_term = base_env.command_manager.get_term("base_velocity")
    cmd_term.is_standing_env[:] = False
    cmd_term.is_heading_env[:] = False
    cmd_term.time_left[:] = 1e9


# ============================================================
# Main
# ============================================================


def main():
    device = args_cli.device

    # ---- environment ----
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.load_checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.load_checkpoint

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO] Task:        {args_cli.task}")
    print(f"[INFO] Policy:      {resume_path}")
    print(f"[INFO] FDM model:   {args_cli.fdm_checkpoint}")
    print(f"[INFO] Goal range:  {args_cli.goal_range:.1f} m")
    print(f"[INFO] Planner:     {args_cli.planner_freq:.0f} Hz, pop={args_cli.population_size}")
    print(f"[INFO] Press SPACE to drop a random goal.  Ctrl+C to quit.")
    print("-" * 60)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    # ---- hidden ground plane ----
    if env_cfg.scene.terrain.terrain_type == "usd":
        from unitree_rl_lab.tasks.fdm.robots.g1.fdm_collect_env_cfg import add_hidden_ground_plane
        add_hidden_ground_plane(physics_material=env_cfg.scene.terrain.physics_material)

    # ---- low-level locomotion policy ----
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=base_env.device)
    is_recurrent = hasattr(policy, "reset")

    # ---- planner config (initial goal is a dummy — will be overwritten on first SPACE) ----
    planner_cfg = PlannerConfig(
        fdm_checkpoint=args_cli.fdm_checkpoint,
        device=device,
        prediction_horizon=10,
        history_length=10,
        planner_frequency=args_cli.planner_freq,
        goal_position=(0.0, 0.0),
        goal_heading=0.0,
        action=ActionConfig(
            traj_dim=10,
            dt=0.5,
            lower_bound=(-0.6, -0.5, -1.57),
            upper_bound=(1.0, 0.5, 1.57),
        ),
        mppi=MPPIConfig(
            population_size=args_cli.population_size,
            sigma=args_cli.sigma,
            gamma=args_cli.gamma,
            beta=args_cli.beta,
        ),
        fdm_batch_size=args_cli.fdm_batch_size,
    )

    # ---- planner ----
    planner = FDMMPPIPlanner(planner_cfg, env, policy, torch.device(device))
    _disable_command_autoupdate(base_env)

    # ---- keyboard watcher ----
    kw = SpaceKeyWatcher()
    kw.start()

    # ---- initial observation ----
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    # ---- state machine ----
    IDLE = 0       # standing still, waiting for SPACE
    PLANNING = 1   # navigating to goal
    REACHED = 2    # goal reached, standing still, waiting for next SPACE

    state = IDLE
    se2_cmd = torch.zeros(1, 3, device=base_env.device)
    goal_tensor = torch.zeros(2, device=base_env.device)
    goal_count = 0
    plan_count = 0
    t_start = time.time()
    start_pos = base_env.scene["robot"].data.root_pos_w[0, :2].clone()

    print(f"[INFO] Start position: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
    print("[INFO] Robot standing by.  Press SPACE to begin.")
    print("-" * 60)

    with torch.inference_mode():
        for step in range(args_cli.max_steps):

            # ---- keyboard: SPACE → generate random goal ----
            if kw.pressed:
                kw.pressed = False
                robot_xy = base_env.scene["robot"].data.root_pos_w[0, :2]
                goal_x = float(robot_xy[0]) + random.uniform(-args_cli.goal_range, args_cli.goal_range)
                goal_y = float(robot_xy[1]) + random.uniform(-args_cli.goal_range, args_cli.goal_range)
                goal_tensor = torch.tensor([goal_x, goal_y], device=base_env.device)
                planner.set_goal(goal_x, goal_y)
                planner.reset()
                goal_count += 1
                plan_count = 0
                state = PLANNING
                print(f"\n[GOAL #{goal_count}] → ({goal_x:.2f}, {goal_y:.2f})  "
                      f"robot at ({float(robot_xy[0]):.2f}, {float(robot_xy[1]):.2f})")

            # ---- planner + policy ----
            if state == PLANNING:
                action, info = planner.step(obs)
                se2_cmd[0] = action[0]
                if info.get("planned"):
                    plan_count += 1
                    if plan_count <= 3 or plan_count % 20 == 1:
                        robot_xy = base_env.scene["robot"].data.root_pos_w[0, :2]
                        dist = float(torch.norm(robot_xy - goal_tensor).item())
                        print(f"  [plan #{plan_count:3d}] step={step:5d}  "
                              f"dist_to_goal={dist:.2f}m  cmd=({se2_cmd[0,0]:.2f},{se2_cmd[0,1]:.2f},{se2_cmd[0,2]:.2f})")
            else:
                se2_cmd.zero_()

            _inject_command(base_env, se2_cmd)

            # ---- low-level policy → joint actions ----
            policy_action = policy(obs)
            if isinstance(policy_action, tuple):
                policy_action = policy_action[0]

            # ---- step environment ----
            obs, _rewards, dones, _infos = env.step(policy_action)
            if isinstance(obs, tuple):
                obs = obs[0]

            # ---- handle done (fall) ----
            if dones.any():
                planner.reset()
                if is_recurrent:
                    policy.reset(torch.where(dones)[0])
                state = IDLE
                print(f"\n[FALL] Robot fell at step {step}.  Press SPACE to reset and continue.")

            # ---- goal check ----
            if state == PLANNING:
                robot_xy = base_env.scene["robot"].data.root_pos_w[0, :2]
                dist_to_goal = float(torch.norm(robot_xy - goal_tensor).item())
                if dist_to_goal < args_cli.goal_tolerance:
                    planner.mark_goal_reached()
                    state = REACHED
                    elapsed = time.time() - t_start
                    print(f"\n[REACHED #{goal_count}] step={step}, "
                          f"dist={dist_to_goal:.2f}m, time={elapsed:.1f}s")
                    print("[INFO] Standing by.  Press SPACE for next goal.")

    # ---- cleanup ----
    kw.stop()
    kw.restore()
    elapsed = time.time() - t_start
    print("-" * 60)
    print(f"[INFO] Done.  Goals completed: {goal_count}  Steps: {step+1}  Time: {elapsed:.1f}s")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
