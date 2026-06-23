#!/usr/bin/env python3

"""Collect FDM training data using velocity_perception as low-level controller.

This script follows FDM's data collection logic:
- SE(2) velocity commands sampled every 0.5s (25 simulation steps)
- Data collected at 0.5s intervals with history buffers
- Proper handling of episode resets and done signals
- Output: FDM-compatible trajectory data

Usage:
    python collect_fdm_rollouts.py --task Unitree-G1-29dof-FDM-Collect \
        --checkpoint logs/rsl_rl/unitree_perception/.../model_19999.pt \
        --num_envs 1024 --trajectory_length 150
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dataest/FDM"

# ============================================================
# Argument parsing
# ============================================================

parser = argparse.ArgumentParser(description="Collect FDM training data.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-FDM-Collect",
                    help="Unitree task to roll out.")
parser.add_argument("--num_envs", type=int, default=1024,
                    help="Number of parallel environments.")
parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                    help="Directory to save collected data.")
parser.add_argument("--run_name", type=str, default=None,
                    help="Optional name included in the output subdirectory.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to a policy checkpoint.")
parser.add_argument("--load_run", type=str, default=None,
                    help="Run folder used when resolving the default checkpoint.")
parser.add_argument("--load_checkpoint", type=str, default=None,
                    help="Checkpoint name used when resolving default path.")
parser.add_argument("--experiment_name", type=str, default=None,
                    help="Override RSL-RL experiment name.")
parser.add_argument("--resume", action="store_true", default=False,
                    help="Compatibility flag for RSL-RL config parsing.")
parser.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"})
parser.add_argument("--log_project_name", type=str, default=None)

# FDM-specific parameters
parser.add_argument("--command_timestep", type=float, default=0.5,
                    help="Time interval between SE(2) command updates (seconds).")
parser.add_argument("--history_length", type=int, default=10,
                    help="Number of history frames for proprioceptive/state data.")
parser.add_argument("--trajectory_length", type=int, default=150,
                    help="Number of command steps per environment trajectory.")
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable fabric and use USD I/O.")

# Command range options
parser.add_argument("--use_fdm_range", action="store_true", default=False,
                    help="Use FDM's command range (-0.1,1.5) instead of perception's (-0.6,1.0).")

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


# ============================================================
# SE(2) Command Sampler (FDM-style time-correlated)
# ============================================================


class SE2CommandSampler:
    """Time-correlated SE(2) velocity command sampler.

    Follows FDM's TimeCorrelatedCommandTrajectoryAgent logic:
    - 60% linear time-correlated
    - 40% normal time-correlated
    """

    def __init__(self, num_envs: int, device: str,
                 lin_vel_x_range: tuple = (-0.1, 1.5),
                 lin_vel_y_range: tuple = (-0.4, 0.4),
                 ang_vel_z_range: tuple = (-1.0, 1.0),
                 linear_ratio: float = 0.6,
                 normal_ratio: float = 0.4,
                 max_beta: float = 0.3,
                 sigma_scale: float = 0.3):
        self.num_envs = num_envs
        self.device = device

        # Velocity ranges
        self._limits_min = torch.tensor(
            [lin_vel_x_range[0], lin_vel_y_range[0], ang_vel_z_range[0]],
            device=device
        )
        self._limits_max = torch.tensor(
            [lin_vel_x_range[1], lin_vel_y_range[1], ang_vel_z_range[1]],
            device=device
        )

        # Time correlation parameters
        self._max_beta = max_beta
        self._max_sigma = sigma_scale * (self._limits_max - self._limits_min)

        # Environment split for different sampling modes
        self._linear_n = int(num_envs * linear_ratio)
        self._normal_n = int(num_envs * normal_ratio)

        # Current commands
        self.commands = torch.zeros(num_envs, 3, device=device)
        self._first_sample = True

    def sample(self) -> torch.Tensor:
        """Sample new SE(2) commands for all environments."""
        if self._first_sample:
            self.commands = torch.rand(self.num_envs, 3, device=self.device)
            self.commands = self._limits_min + self.commands * (self._limits_max - self._limits_min)
            self._first_sample = False
        else:
            self._update_linear_correlated()
            self._update_normal_correlated()
        return self.commands.clone()

    def sample_for_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample new commands for specific environments (used after reset)."""
        new_cmds = torch.rand(len(env_ids), 3, device=self.device)
        new_cmds = self._limits_min + new_cmds * (self._limits_max - self._limits_min)
        self.commands[env_ids] = new_cmds
        return self.commands[env_ids].clone()

    def _update_linear_correlated(self):
        """Linear time-correlated command update."""
        if self._linear_n == 0:
            return
        beta = torch.rand(self._linear_n, 1, device=self.device) * self._max_beta
        rand_cmds = torch.rand(self._linear_n, 3, device=self.device)
        rand_cmds = self._limits_min + rand_cmds * (self._limits_max - self._limits_min)
        self.commands[:self._linear_n] = (
            self.commands[:self._linear_n] * (1 - beta) + rand_cmds * beta
        )

    def _update_normal_correlated(self):
        """Normal time-correlated command update."""
        if self._normal_n == 0:
            return
        sigma = torch.rand(self._normal_n, 3, device=self.device) * self._max_sigma
        noise = torch.randn(self._normal_n, 3, device=self.device) * sigma
        self.commands[self._linear_n:self._linear_n + self._normal_n] += noise
        self.commands[self._linear_n:self._linear_n + self._normal_n] = torch.clamp(
            self.commands[self._linear_n:self._linear_n + self._normal_n],
            self._limits_min, self._limits_max
        )


# ============================================================
# FDM Data Buffer
# ============================================================


class FDMDataBuffer:
    """FDM-compatible data buffer with history support.

    Stores data in FDM format:
    - actions: (num_envs, T, 3) - SE(2) commands
    - states: (num_envs, T, history_length, state_dim)
    - proprioceptive: (num_envs, T, history_length, proprio_dim)
    - exteroceptive: (num_envs, T, H, W) - height scan

    Handles episode resets properly by clearing history buffers on done.
    """

    def __init__(self, num_envs: int, trajectory_length: int, history_length: int,
                 state_dim: int, proprio_dim: int, height_scan_shape: tuple):
        self.num_envs = num_envs
        self.trajectory_length = trajectory_length
        self.history_length = history_length

        # Main trajectory buffers (CPU to save GPU memory)
        self.actions = torch.zeros(num_envs, trajectory_length, 3)
        self.states = torch.zeros(num_envs, trajectory_length, history_length, state_dim)
        self.proprioceptive = torch.zeros(num_envs, trajectory_length, history_length, proprio_dim)
        self.exteroceptive = torch.zeros(num_envs, trajectory_length, *height_scan_shape)

        # Local history buffers (GPU for fast update)
        self._local_state_history = torch.zeros(num_envs, history_length, state_dim)
        self._local_proprio_history = torch.zeros(num_envs, history_length, proprio_dim)

        # Fill index per environment
        self.fill_idx = torch.zeros(num_envs, dtype=torch.long)
        self.is_filled = torch.zeros(num_envs, dtype=torch.bool)

        # Track steps since last history update per env
        self._steps_since_history = torch.zeros(num_envs, dtype=torch.long)

    @property
    def fill_ratio(self) -> float:
        return (self.fill_idx.float().mean() / self.trajectory_length).item()

    def reset_envs(self, env_ids: torch.Tensor):
        """Reset history buffers for done environments.

        This is critical to avoid cross-episode contamination.
        """
        self._local_state_history[env_ids] = 0.0
        self._local_proprio_history[env_ids] = 0.0
        self._steps_since_history[env_ids] = 0

    def update_history(self, env_ids: torch.Tensor, state: torch.Tensor, proprio: torch.Tensor):
        """Update history buffers for specified environments.

        Args:
            env_ids: Which environments to update
            state: FDM state (num_envs, state_dim) - already on GPU
            proprio: Proprioceptive obs (num_envs, proprio_dim) - already on GPU
        """
        # Roll and insert new data
        self._local_state_history[env_ids] = torch.roll(self._local_state_history[env_ids], 1, dims=1)
        self._local_state_history[env_ids, 0] = state[env_ids].to(self._local_state_history.device)

        self._local_proprio_history[env_ids] = torch.roll(self._local_proprio_history[env_ids], 1, dims=1)
        self._local_proprio_history[env_ids, 0] = proprio[env_ids].to(self._local_proprio_history.device)

    def add_trajectory_point(self, env_ids: torch.Tensor, se2_cmd: torch.Tensor,
                             height_scan: torch.Tensor):
        """Add a trajectory point for specified environments.

        Args:
            env_ids: Which environments to add data for
            se2_cmd: SE(2) commands (num_envs, 3)
            height_scan: Height scan (num_envs, H, W)
        """
        # Filter to non-filled environments
        active_mask = ~self.is_filled[env_ids]
        active_env_ids = env_ids[active_mask]

        if len(active_env_ids) == 0:
            return

        idx = self.fill_idx[active_env_ids]

        # Move data to CPU and store
        self.actions[active_env_ids, idx] = se2_cmd[active_env_ids].cpu()
        self.states[active_env_ids, idx] = self._local_state_history[active_env_ids].cpu()
        self.proprioceptive[active_env_ids, idx] = self._local_proprio_history[active_env_ids].cpu()
        self.exteroceptive[active_env_ids, idx] = height_scan[active_env_ids].cpu()

        # Update fill index
        self.fill_idx[active_env_ids] += 1
        self.is_filled[self.fill_idx >= self.trajectory_length] = True

    def get_trajectories(self) -> dict:
        """Get collected trajectories in FDM format."""
        return {
            "actions": self.actions,
            "states": self.states,
            "proprioceptive": self.proprioceptive,
            "exteroceptive": self.exteroceptive.to(torch.float16),  # FDM uses float16
        }


# ============================================================
# Helper functions
# ============================================================


def _get_fdm_state(base_env) -> torch.Tensor:
    """Extract FDM state from environment observations.

    Returns:
        torch.Tensor: (num_envs, state_dim) - on GPU
    """
    return base_env.observation_manager.compute_group("fdm_state")


def _get_fdm_proprioceptive(base_env) -> torch.Tensor:
    """Extract FDM proprioceptive observations.

    Returns:
        torch.Tensor: (num_envs, proprio_dim) - on GPU
    """
    # Update joint history buffers before computing observations
    from unitree_rl_lab.tasks.fdm.mdp import _update_joint_history
    _update_joint_history(base_env)
    return base_env.observation_manager.compute_group("fdm_obs_proprioceptive")


def _get_fdm_height_scan(base_env) -> torch.Tensor:
    """Extract FDM height scan and reshape to 2D.

    Returns:
        torch.Tensor: (num_envs, H, W) - on GPU
    """
    raw = base_env.observation_manager.compute_group("fdm_obs_exteroceptive")
    # Handle different possible shapes
    if raw.dim() == 2:
        # Flattened (N, H*W) -> reshape to (N, H, W)
        # Get expected shape from config
        h, w = 60, 46  # FDM default
        raw = raw.reshape(raw.shape[0], h, w)
    elif raw.dim() == 4:
        # (N, C, H, W) -> (N, H, W)
        raw = raw.squeeze(1)
    return raw


def _inject_command(base_env, se2_cmd: torch.Tensor):
    """Inject SE(2) command into the command manager.

    Uses vel_command_b (the actual command seen by the policy) instead of _command.
    Also disables standing/heading override and prevents automatic resampling.

    Args:
        base_env: The base environment
        se2_cmd: SE(2) commands (num_envs, 3) - [lin_vel_x, lin_vel_y, ang_vel_z]
    """
    try:
        cmd_term = base_env.command_manager.get_term("base_velocity")
        # Use vel_command_b - this is what the policy actually reads
        cmd_term.vel_command_b[:, 0] = se2_cmd[:, 0]  # lin_vel_x
        cmd_term.vel_command_b[:, 1] = se2_cmd[:, 1]  # lin_vel_y
        cmd_term.vel_command_b[:, 2] = se2_cmd[:, 2]  # ang_vel_z
    except Exception as e:
        print(f"[WARNING] Failed to inject command: {e}")


def _disable_command_autoupdate(base_env):
    """Disable command manager's automatic resampling and standing/heading override.

    This ensures our manually injected SE(2) commands are not overwritten.
    """
    try:
        cmd_term = base_env.command_manager.get_term("base_velocity")
        cmd_term.is_standing_env[:] = False
        cmd_term.is_heading_env[:] = False
        cmd_term.time_left[:] = 1e9
    except Exception as e:
        print(f"[WARNING] Failed to disable command autoupdate: {e}")


def _make_runner(env, agent_cfg, resume_path: str):
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    return runner


# ============================================================
# Main collection loop
# ============================================================


def main():
    # Parse environment config
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    # Parse agent config
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.load_checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.load_checkpoint

    # Resolve checkpoint path
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Loading policy checkpoint: {resume_path}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    # Add hidden ground plane for USD terrain collision (replicates NavTerrainImporter logic)
    if env_cfg.scene.terrain.terrain_type == "usd":
        from unitree_rl_lab.tasks.fdm.robots.g1.fdm_collect_env_cfg import add_hidden_ground_plane
        add_hidden_ground_plane(physics_material=env_cfg.scene.terrain.physics_material)

    # Create runner and get policy
    runner = _make_runner(env, agent_cfg, resume_path)
    policy = runner.get_inference_policy(device=base_env.device)

    # Calculate timing parameters
    step_dt = base_env.step_dt  # 0.005 * 4 = 0.02s
    steps_per_cmd = int(args_cli.command_timestep / step_dt)  # 0.5 / 0.02 = 25
    history_interval = max(1, int(0.05 / step_dt))  # 0.05 / 0.02 ≈ 2-3 steps

    print(f"[INFO] step_dt={step_dt:.4f}s, steps_per_cmd={steps_per_cmd}, history_interval={history_interval}")

    # Get observation dimensions from environment
    # Test compute to get actual shapes
    test_state = _get_fdm_state(base_env)
    test_proprio = _get_fdm_proprioceptive(base_env)
    test_height = _get_fdm_height_scan(base_env)

    state_dim = test_state.shape[1]
    proprio_dim = test_proprio.shape[1]
    height_scan_shape = test_height.shape[1:]  # (H, W)

    print(f"[INFO] state_dim={state_dim}, proprio_dim={proprio_dim}, height_scan={height_scan_shape}")

    # Select command range based on user choice
    if args_cli.use_fdm_range:
        # FDM's original range (may cause OOD for perception policy)
        lin_vel_x_range = (-0.1, 1.5)
        lin_vel_y_range = (-0.4, 0.4)
        ang_vel_z_range = (-1.0, 1.0)
        print("[INFO] Using FDM command range: x(-0.1,1.5), y(-0.4,0.4), yaw(-1.0,1.0)")
    else:
        # Match velocity_perception training range (safer for policy)
        lin_vel_x_range = (-0.6, 1.0)
        lin_vel_y_range = (-0.5, 0.5)
        ang_vel_z_range = (-1.57, 1.57)
        print("[INFO] Using perception command range: x(-0.6,1.0), y(-0.5,0.5), yaw(-1.57,1.57)")

    # Initialize SE(2) command sampler
    se2_sampler = SE2CommandSampler(
        num_envs=args_cli.num_envs,
        device=str(base_env.device),
        lin_vel_x_range=lin_vel_x_range,
        lin_vel_y_range=lin_vel_y_range,
        ang_vel_z_range=ang_vel_z_range,
    )

    # Initialize FDM data buffer (CPU)
    data_buffer = FDMDataBuffer(
        num_envs=args_cli.num_envs,
        trajectory_length=args_cli.trajectory_length,
        history_length=args_cli.history_length,
        state_dim=state_dim,
        proprio_dim=proprio_dim,
        height_scan_shape=height_scan_shape,
    )

    # Output directory
    run_name = args_cli.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args_cli.output_dir).expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Meta information
    meta = {
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "command_timestep": args_cli.command_timestep,
        "history_length": args_cli.history_length,
        "trajectory_length": args_cli.trajectory_length,
        "checkpoint": resume_path,
        "step_dt": step_dt,
        "state_dim": state_dim,
        "proprio_dim": proprio_dim,
        "height_scan_shape": list(height_scan_shape),
    }

    # ============================================================
    # Collection loop
    # ============================================================

    print(f"[INFO] Starting FDM data collection...")
    print(f"[INFO] Target: {args_cli.trajectory_length} trajectories per env")

    # Get initial observations
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    # Disable command manager's automatic resampling and standing/heading override
    _disable_command_autoupdate(base_env)

    # All env IDs (CPU for indexing with CPU buffers)
    all_env_ids_cpu = torch.arange(args_cli.num_envs)

    # Sample initial SE(2) command and inject
    se2_cmd = se2_sampler.sample()
    _inject_command(base_env, se2_cmd)

    # Track total simulation steps
    total_sim_steps = 0
    cmd_step_counter = 0

    # Get policy type for hidden state handling
    is_recurrent = hasattr(policy, 'reset')

    with torch.inference_mode():
        while not data_buffer.is_filled.all():
            # ---- Collect data BEFORE executing the command ----
            # This ensures we store: (current_state, future_action_to_execute)

            # Get active (non-filled) environments on CPU
            active_mask_cpu = ~data_buffer.is_filled
            active_env_ids_cpu = all_env_ids_cpu[active_mask_cpu]
            if len(active_env_ids_cpu) == 0:
                break

            # Get current FDM observations (before executing se2_cmd)
            state = _get_fdm_state(base_env)
            proprio = _get_fdm_proprioceptive(base_env)
            height_scan = _get_fdm_height_scan(base_env)

            # Update history buffers with current state
            data_buffer.update_history(active_env_ids_cpu, state.cpu(), proprio.cpu())

            # Collect: current state + future action (se2_cmd)
            data_buffer.add_trajectory_point(active_env_ids_cpu, se2_cmd.cpu(), height_scan.cpu())

            cmd_step_counter += 1

            # ---- Now execute the command for steps_per_cmd steps ----
            for sub_step in range(steps_per_cmd):
                # Policy inference
                actions = policy(obs)
                if isinstance(actions, tuple):
                    actions = actions[0]

                # Step environment
                next_obs, rewards, dones, infos = env.step(actions)
                total_sim_steps += 1

                # ---- Handle episode resets ----
                done_env_ids = torch.where(dones)[0]
                if len(done_env_ids) > 0:
                    # Reset history buffers for done environments (CPU)
                    data_buffer.reset_envs(done_env_ids.cpu())

                    # Reset LSTM hidden state for done environments
                    if is_recurrent:
                        policy.reset(done_env_ids)

                    # Sample new commands for reset environments
                    se2_sampler.sample_for_envs(done_env_ids)
                    _inject_command(base_env, se2_sampler.commands)

                    # Update local se2_cmd to stay in sync
                    se2_cmd = se2_sampler.commands.clone()

                # ---- Update history buffers ----
                # Only update after enough steps for history to be meaningful
                # and at the specified interval
                if sub_step % history_interval == 0:
                    state = _get_fdm_state(base_env)
                    proprio = _get_fdm_proprioceptive(base_env)
                    # Get active (non-filled) environments on CPU
                    active_mask_cpu = ~data_buffer.is_filled
                    active_env_ids_cpu = all_env_ids_cpu[active_mask_cpu]
                    if len(active_env_ids_cpu) > 0:
                        data_buffer.update_history(active_env_ids_cpu, state.cpu(), proprio.cpu())

                obs = next_obs

            # ---- Sample new SE(2) command for next iteration ----
            se2_cmd = se2_sampler.sample()
            _inject_command(base_env, se2_cmd)

            # Progress reporting
            if cmd_step_counter % 10 == 0:
                print(f"[INFO] Sim steps: {total_sim_steps}, "
                      f"CMD steps: {cmd_step_counter}, "
                      f"Fill ratio: {data_buffer.fill_ratio:.2%}")

    # ============================================================
    # Save data
    # ============================================================

    print("[INFO] Collection complete. Saving data...")

    trajectories = data_buffer.get_trajectories()
    payload = {
        "meta": meta,
        "data": trajectories,
    }

    output_path = output_dir / "fdm_trajectories.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[INFO] Saved to {output_path}")
    print(f"[INFO] Trajectory shape: actions={trajectories['actions'].shape}, "
          f"states={trajectories['states'].shape}, "
          f"proprioceptive={trajectories['proprioceptive'].shape}, "
          f"exteroceptive={trajectories['exteroceptive'].shape}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
