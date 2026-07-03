#!/usr/bin/env python3

"""Train Unitree FDM with official-style collect/train rounds.

This keeps the Unitree control stack intact:
SE(2) command -> low-level RSL-RL locomotion policy -> joint action -> env.step().
The FDM model still learns from SE(2) commands and resulting state trajectories.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "source" / "unitree_rl_lab"))


parser = argparse.ArgumentParser(description="Online FDM training with collect/train rounds.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-FDM-Collect")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to the low-level locomotion policy checkpoint.")
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--load_checkpoint", type=str, default=None)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"})
parser.add_argument("--log_project_name", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)

parser.add_argument("--command_timestep", type=float, default=0.5)
parser.add_argument("--history_length", type=int, default=10)
parser.add_argument("--trajectory_length", type=int, default=150)
parser.add_argument("--val_trajectory_length", type=int, default=150)
parser.add_argument("--collection_rounds", type=int, default=20)
parser.add_argument("--use_fdm_range", action="store_true", default=False)
parser.add_argument("--tail_fill_ratio", type=float, default=0.90)
parser.add_argument("--tail_fill_patience", type=float, default=1.5)
parser.add_argument("--tail_fill_stall_cmd_steps", type=int, default=1000)
parser.add_argument("--max_cmd_steps_per_round", type=int, default=0)

parser.add_argument("--output_root", default="logs/fdm/unitree_g1_fdm_online")
parser.add_argument("--run_name", default=None)
parser.add_argument("--resume_fdm_checkpoint", default=None)
parser.add_argument("--start_collection_round", type=int, default=0)
parser.add_argument("--prediction_horizon", type=int, default=10)
parser.add_argument("--num_samples", type=int, default=80000)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=3e-3)
parser.add_argument("--min_learning_rate", type=float, default=1e-6)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--val_ratio", type=float, default=0.1)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_hard_contact_normalization", action="store_true", default=False)
parser.add_argument("--collision_rate", type=float, default=None)
parser.add_argument("--no_collision_oversampling", action="store_true", default=False)
parser.add_argument("--sample_filter_first_steps_coll", type=int, default=0)
parser.add_argument("--small_motion_ratio", type=float, default=0.1)
parser.add_argument("--small_motion_threshold", type=float, default=1.0)
parser.add_argument("--height_threshold", type=float, default=None)
parser.add_argument("--outlier_threshold", type=float, default=10.0)
parser.add_argument("--no_noise", action="store_true", default=False, help="Disable observation noise during training.")
parser.add_argument("--save_round_datasets", action="store_true", default=False)

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
from unitree_rl_lab.tasks.fdm.training import FDMTrainer, FDMTrainingConfig
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


class SE2CommandSampler:
    """FDM-style time-correlated SE(2) command sampler."""

    def __init__(
        self,
        num_envs: int,
        device: str,
        lin_vel_x_range: tuple[float, float],
        lin_vel_y_range: tuple[float, float],
        ang_vel_z_range: tuple[float, float],
        linear_ratio: float = 0.6,
        normal_ratio: float = 0.4,
        min_correlation: float = 0.3,
        sigma_scale: float = 0.3,
    ):
        self.num_envs = num_envs
        self.device = device
        self._limits_min = torch.tensor([lin_vel_x_range[0], lin_vel_y_range[0], ang_vel_z_range[0]], device=device)
        self._limits_max = torch.tensor([lin_vel_x_range[1], lin_vel_y_range[1], ang_vel_z_range[1]], device=device)
        self._min_correlation = min_correlation
        self._max_sigma = sigma_scale * (self._limits_max - self._limits_min)
        self._linear_n = int(num_envs * linear_ratio)
        self._normal_n = int(num_envs * normal_ratio)
        self.commands = torch.zeros(num_envs, 3, device=device)
        self._first_sample = True

    def sample(self) -> torch.Tensor:
        if self._first_sample:
            self.commands = self._random_commands(self.num_envs)
            self._first_sample = False
        else:
            self._update_linear_correlated()
            self._update_normal_correlated()
        return self.commands.clone()

    def sample_for_envs(self, env_ids: torch.Tensor):
        self.commands[env_ids] = self._random_commands(len(env_ids))

    def _random_commands(self, count: int) -> torch.Tensor:
        rand = torch.rand(count, 3, device=self.device)
        return self._limits_min + rand * (self._limits_max - self._limits_min)

    def _update_linear_correlated(self):
        if self._linear_n == 0:
            return
        opposite_beta = torch.rand(self._linear_n, 1, device=self.device) * (1.0 - self._min_correlation)
        rand_cmds = self._random_commands(self._linear_n)
        self.commands[: self._linear_n] = (
            self.commands[: self._linear_n] * (1 - opposite_beta) + rand_cmds * opposite_beta
        )

    def _update_normal_correlated(self):
        if self._normal_n == 0:
            return
        target = slice(self._linear_n, self._linear_n + self._normal_n)
        sigma = torch.rand(self._normal_n, 3, device=self.device) * self._max_sigma
        self.commands[target] += torch.randn(self._normal_n, 3, device=self.device) * sigma
        self.commands[target] = torch.clamp(self.commands[target], self._limits_min, self._limits_max)


class FDMDataBuffer:
    """CPU trajectory buffer used for one online collection round."""

    def __init__(
        self,
        num_envs: int,
        trajectory_length: int,
        history_length: int,
        state_dim: int,
        proprio_dim: int,
        height_scan_shape: tuple[int, ...],
    ):
        self.num_envs = num_envs
        self.trajectory_length = trajectory_length
        self.history_length = history_length
        self.actions = torch.zeros(num_envs, trajectory_length, 3)
        self.states = torch.zeros(num_envs, trajectory_length, history_length, state_dim)
        self.proprioceptive = torch.zeros(num_envs, trajectory_length, history_length, proprio_dim)
        self.exteroceptive = torch.zeros(num_envs, trajectory_length, *height_scan_shape)
        self._local_state_history = torch.zeros(num_envs, history_length, state_dim)
        self._local_proprio_history = torch.zeros(num_envs, history_length, proprio_dim)
        self.fill_idx = torch.zeros(num_envs, dtype=torch.long)
        self.is_filled = torch.zeros(num_envs, dtype=torch.bool)

    @property
    def fill_ratio(self) -> float:
        return (self.fill_idx.float().mean() / self.trajectory_length).item()

    def reset_envs(self, env_ids: torch.Tensor):
        self._local_state_history[env_ids] = 0.0
        self._local_proprio_history[env_ids] = 0.0

    def update_history(self, env_ids: torch.Tensor, state: torch.Tensor, proprio: torch.Tensor):
        self._local_state_history[env_ids] = torch.roll(self._local_state_history[env_ids], 1, dims=1)
        self._local_state_history[env_ids, 0] = state[env_ids].to(self._local_state_history.device)
        self._local_proprio_history[env_ids] = torch.roll(self._local_proprio_history[env_ids], 1, dims=1)
        self._local_proprio_history[env_ids, 0] = proprio[env_ids].to(self._local_proprio_history.device)

    def add_trajectory_point(self, env_ids: torch.Tensor, se2_cmd: torch.Tensor, height_scan: torch.Tensor):
        active_env_ids = env_ids[~self.is_filled[env_ids]]
        if len(active_env_ids) == 0:
            return
        idx = self.fill_idx[active_env_ids]
        self.actions[active_env_ids, idx] = se2_cmd[active_env_ids].cpu()
        self.states[active_env_ids, idx] = self._local_state_history[active_env_ids].cpu()
        self.proprioceptive[active_env_ids, idx] = self._local_proprio_history[active_env_ids].cpu()
        self.exteroceptive[active_env_ids, idx] = height_scan[active_env_ids].cpu()
        self.fill_idx[active_env_ids] += 1
        self.is_filled[self.fill_idx >= self.trajectory_length] = True

    def fill_leftover_envs(self):
        full_env_ids = torch.where(self.is_filled)[0]
        leftover_env_ids = torch.where(~self.is_filled)[0]
        if len(leftover_env_ids) == 0:
            return
        if len(full_env_ids) == 0:
            print("[WARNING] Cannot fill leftover envs because no complete source env exists.")
            return
        print(f"[WARNING] Filling {len(leftover_env_ids)} leftover envs from {len(full_env_ids)} complete envs.")
        for offset, target_env in enumerate(leftover_env_ids):
            start = int(self.fill_idx[target_env].item())
            source_env = full_env_ids[offset % len(full_env_ids)]
            remaining = self.trajectory_length - start
            self.actions[target_env, start:] = self.actions[source_env, :remaining]
            self.states[target_env, start:] = self.states[source_env, :remaining]
            self.proprioceptive[target_env, start:] = self.proprioceptive[source_env, :remaining]
            self.exteroceptive[target_env, start:] = self.exteroceptive[source_env, :remaining]
            self.fill_idx[target_env] = self.trajectory_length
            self.is_filled[target_env] = True

    def to_payload(self, meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "meta": meta,
            "data": {
                "actions": self.actions,
                "states": self.states,
                "proprioceptive": self.proprioceptive,
                "exteroceptive": self.exteroceptive.to(torch.float16),
            },
        }


def _get_fdm_state(base_env) -> torch.Tensor:
    return base_env.observation_manager.compute_group("fdm_state")


def _get_fdm_proprioceptive(base_env) -> torch.Tensor:
    from unitree_rl_lab.tasks.fdm.mdp import _update_joint_history

    _update_joint_history(base_env)
    return base_env.observation_manager.compute_group("fdm_obs_proprioceptive")


def _get_fdm_height_scan(base_env) -> torch.Tensor:
    raw = base_env.observation_manager.compute_group("fdm_obs_exteroceptive")
    if raw.dim() == 2:
        raw = raw.reshape(raw.shape[0], 60, 46)
    elif raw.dim() == 4:
        raw = raw.squeeze(1)
    return raw


def _find_feet_body_ids(base_env, body_regex: str = ".*ankle_roll.*") -> torch.Tensor:
    contact_sensor = base_env.scene.sensors["contact_forces"]
    body_ids, body_names = contact_sensor.find_bodies(body_regex)
    if len(body_ids) == 0:
        raise RuntimeError(f"No foot bodies matched regex '{body_regex}' in contact_forces sensor.")
    print(f"[INFO] FDM feet-contact gating bodies: {body_names}")
    return torch.tensor(body_ids, dtype=torch.long, device=base_env.device)


def _update_feet_contact_seen(base_env, feet_body_ids: torch.Tensor, feet_contact_seen: torch.Tensor, dones=None):
    if dones is not None and torch.any(dones):
        feet_contact_seen[dones] = False
    contact_sensor = base_env.scene.sensors["contact_forces"]
    contact_now = torch.norm(contact_sensor.data.net_forces_w[:, feet_body_ids], dim=-1) > 1.0
    feet_contact_seen[contact_now] = True
    return torch.all(feet_contact_seen, dim=-1)


def _inject_command(base_env, se2_cmd: torch.Tensor):
    cmd_term = base_env.command_manager.get_term("base_velocity")
    cmd_term.vel_command_b[:, 0] = se2_cmd[:, 0]
    cmd_term.vel_command_b[:, 1] = se2_cmd[:, 1]
    cmd_term.vel_command_b[:, 2] = se2_cmd[:, 2]


def _disable_command_autoupdate(base_env):
    cmd_term = base_env.command_manager.get_term("base_velocity")
    cmd_term.is_standing_env[:] = False
    cmd_term.is_heading_env[:] = False
    cmd_term.time_left[:] = 1e9


def _make_runner(env, agent_cfg, resume_path: str):
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    return runner


def _command_ranges():
    if args_cli.use_fdm_range:
        print("[INFO] Using FDM command range: x(-0.1,1.0), y(-0.4,0.4), yaw(-1.0,1.0)")
        return (-0.1, 1.0), (-0.4, 0.4), (-1.0, 1.0)
    print("[INFO] Using perception command range: x(-0.6,1.0), y(-0.5,0.5), yaw(-1.57,1.57)")
    return (-0.6, 1.0), (-0.5, 0.5), (-1.57, 1.57)


def collect_round(
    env,
    base_env,
    policy,
    *,
    trajectory_length: int,
    round_name: str,
    dims: tuple[int, int, tuple[int, ...]],
    timing: tuple[float, int, float],
) -> dict[str, Any]:
    state_dim, proprio_dim, height_scan_shape = dims
    step_dt, steps_per_cmd, history_collection_interval = timing
    buffer = FDMDataBuffer(
        num_envs=args_cli.num_envs,
        trajectory_length=trajectory_length,
        history_length=args_cli.history_length,
        state_dim=state_dim,
        proprio_dim=proprio_dim,
        height_scan_shape=height_scan_shape,
    )

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    _disable_command_autoupdate(base_env)
    if hasattr(policy, "reset"):
        policy.reset(torch.arange(args_cli.num_envs, device=base_env.device))

    all_env_ids_device = torch.arange(args_cli.num_envs, device=base_env.device)
    feet_body_ids = _find_feet_body_ids(base_env)
    feet_contact_seen = torch.zeros(args_cli.num_envs, len(feet_body_ids), dtype=torch.bool, device=base_env.device)
    feet_all_contact = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=base_env.device)
    env_step_counter = torch.zeros(args_cli.num_envs, dtype=torch.long, device=base_env.device)
    last_record_step = torch.zeros(args_cli.num_envs, dtype=torch.long, device=base_env.device)

    lin_x, lin_y, yaw = _command_ranges()
    se2_sampler = SE2CommandSampler(args_cli.num_envs, str(base_env.device), lin_x, lin_y, yaw)
    se2_cmd = se2_sampler.sample()
    _inject_command(base_env, se2_cmd)

    is_recurrent = hasattr(policy, "reset")
    total_sim_steps = 0
    cmd_step_counter = 0
    info_counter = 1
    segment_start = time.time()
    segment_durations: list[float] = []
    last_progress_ratio = buffer.fill_ratio
    last_progress_cmd_step = 0
    progress_epsilon = 0.5 / max(args_cli.num_envs * trajectory_length, 1)
    tail_fill_applied = False

    print(f"[INFO] Collecting {round_name}: target={trajectory_length} command steps/env")
    with torch.inference_mode():
        while not buffer.is_filled.all():
            record_mask = (
                (~buffer.is_filled.to(base_env.device))
                & feet_all_contact
                & ((env_step_counter - last_record_step) >= steps_per_cmd)
            )
            record_env_ids_device = all_env_ids_device[record_mask]
            if len(record_env_ids_device) > 0:
                height_scan = _get_fdm_height_scan(base_env)
                record_env_ids_cpu = record_env_ids_device.cpu()
                buffer.add_trajectory_point(record_env_ids_cpu, se2_cmd.cpu(), height_scan.cpu())
                last_record_step[record_env_ids_device] = env_step_counter[record_env_ids_device]

            for _ in range(steps_per_cmd):
                actions = policy(obs)
                if isinstance(actions, tuple):
                    actions = actions[0]
                next_obs, _, dones, _ = env.step(actions)
                total_sim_steps += 1

                done_env_ids = torch.where(dones)[0]
                if len(done_env_ids) > 0:
                    buffer.reset_envs(done_env_ids.cpu())
                    env_step_counter[done_env_ids] = 0
                    last_record_step[done_env_ids] = 0
                    if is_recurrent:
                        policy.reset(done_env_ids)
                    se2_sampler.sample_for_envs(done_env_ids)
                    se2_cmd = se2_sampler.commands.clone()
                    _inject_command(base_env, se2_cmd)

                feet_all_contact = _update_feet_contact_seen(base_env, feet_body_ids, feet_contact_seen, dones=dones)

                # ---- collision force-record (capture collision frames even
                #      when they fall between regular recording intervals) ----
                unfilled = ~buffer.is_filled.to(base_env.device)
                state_fetched = False
                if unfilled.any():
                    state = _get_fdm_state(base_env)
                    state_fetched = True
                    collision_now = state[:, 7] > 0.5
                    collision_mask = collision_now & unfilled & feet_all_contact
                    if collision_mask.any():
                        coll_env_ids = all_env_ids_device[collision_mask]
                        proprio = _get_fdm_proprioceptive(base_env)
                        buffer.update_history(coll_env_ids.cpu(), state.cpu(), proprio.cpu())
                        height_scan = _get_fdm_height_scan(base_env)
                        buffer.add_trajectory_point(coll_env_ids.cpu(), se2_cmd.cpu(), height_scan.cpu())
                        last_record_step[coll_env_ids] = env_step_counter[coll_env_ids]

                # ---- regular history update ----
                history_due = (env_step_counter % history_collection_interval).to(torch.int) == 0
                history_mask = unfilled & feet_all_contact & history_due
                if history_mask.any():
                    history_env_ids = all_env_ids_device[history_mask]
                    if not state_fetched:
                        state = _get_fdm_state(base_env)
                    proprio = _get_fdm_proprioceptive(base_env)
                    buffer.update_history(history_env_ids.cpu(), state.cpu(), proprio.cpu())

                env_step_counter[feet_all_contact] += 1
                obs = next_obs

            se2_cmd = se2_sampler.sample()
            _inject_command(base_env, se2_cmd)
            cmd_step_counter += 1

            if buffer.fill_ratio > last_progress_ratio + progress_epsilon:
                last_progress_ratio = buffer.fill_ratio
                last_progress_cmd_step = cmd_step_counter

            if buffer.fill_ratio > 0.1 * info_counter:
                elapsed = time.time() - segment_start
                segment_durations.append(elapsed)
                print(
                    f"[INFO] {round_name} fill={buffer.fill_ratio:.2%}, "
                    f"segment={elapsed:.1f}s, sim_steps={total_sim_steps}, feet_ready={feet_all_contact.float().mean():.1%}"
                )
                segment_start = time.time()
                info_counter += 1

            if cmd_step_counter % 100 == 0:
                unfilled = int((~buffer.is_filled).sum().item())
                ready_unfilled = int(
                    ((~buffer.is_filled.to(base_env.device)) & feet_all_contact).sum().item()
                )
                print(
                    f"[INFO] {round_name} cmd_steps={cmd_step_counter}, "
                    f"fill={buffer.fill_ratio:.2%}, feet_ready={feet_all_contact.float().mean():.1%}, "
                    f"unfilled={unfilled}, ready_unfilled={ready_unfilled}"
                )

            tail_time_slow = (
                buffer.fill_ratio > args_cli.tail_fill_ratio
                and len(segment_durations) > 0
                and time.time() - segment_start > args_cli.tail_fill_patience * (sum(segment_durations) / len(segment_durations))
            )
            tail_stalled = (
                buffer.fill_ratio > args_cli.tail_fill_ratio
                and cmd_step_counter - last_progress_cmd_step >= args_cli.tail_fill_stall_cmd_steps
            )
            max_steps_reached = (
                args_cli.max_cmd_steps_per_round > 0 and cmd_step_counter >= args_cli.max_cmd_steps_per_round
            )
            if tail_time_slow or tail_stalled or max_steps_reached:
                reason = (
                    "time_slow" if tail_time_slow else "stall" if tail_stalled else "max_cmd_steps"
                )
                print(f"[WARNING] Collection tail trigger ({reason}). Applying FDM-style leftover fill.")
                buffer.fill_leftover_envs()
                tail_fill_applied = bool(buffer.is_filled.all())
                if tail_fill_applied:
                    break
                print("[WARNING] Leftover fill could not complete because no source trajectory was full yet.")

    meta = {
        "task": args_cli.task,
        "round": round_name,
        "num_envs": args_cli.num_envs,
        "command_timestep": args_cli.command_timestep,
        "history_length": args_cli.history_length,
        "trajectory_length": trajectory_length,
        "step_dt": step_dt,
        "steps_per_cmd": steps_per_cmd,
        "history_collection_interval": history_collection_interval,
        "feet_contact_gated": True,
        "tail_fill_ratio": args_cli.tail_fill_ratio,
        "tail_fill_stall_cmd_steps": args_cli.tail_fill_stall_cmd_steps,
        "max_cmd_steps_per_round": args_cli.max_cmd_steps_per_round,
        "tail_filled": tail_fill_applied,
        "state_dim": state_dim,
        "proprio_dim": proprio_dim,
        "height_scan_shape": list(height_scan_shape),
    }
    print(f"[INFO] {round_name} collection complete: fill={buffer.fill_ratio:.2%}")
    return buffer.to_payload(meta)


def _training_cfg() -> FDMTrainingConfig:
    return FDMTrainingConfig(
        dataset=None,
        output_root=args_cli.output_root,
        run_name=args_cli.run_name,
        device=args_cli.device,
        prediction_horizon=args_cli.prediction_horizon,
        num_samples=args_cli.num_samples,
        batch_size=args_cli.batch_size,
        epochs=args_cli.epochs,
        learning_rate=args_cli.learning_rate,
        min_learning_rate=args_cli.min_learning_rate,
        weight_decay=args_cli.weight_decay,
        val_ratio=args_cli.val_ratio,
        num_workers=args_cli.num_workers,
        seed=args_cli.seed,
        no_hard_contact_normalization=args_cli.no_hard_contact_normalization,
        inspect_only=False,
        collision_rate=args_cli.collision_rate,
        no_collision_oversampling=args_cli.no_collision_oversampling,
        sample_filter_first_steps_coll=args_cli.sample_filter_first_steps_coll,
        small_motion_ratio=args_cli.small_motion_ratio,
        small_motion_threshold=args_cli.small_motion_threshold,
        height_threshold=args_cli.height_threshold,
        outlier_threshold=args_cli.outlier_threshold,
        apply_noise=not args_cli.no_noise,
    )


def _save_payload(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
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
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Loading low-level policy checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    if env_cfg.scene.terrain.terrain_type == "usd":
        from unitree_rl_lab.tasks.fdm.robots.g1.fdm_collect_env_cfg import add_hidden_ground_plane

        add_hidden_ground_plane(physics_material=env_cfg.scene.terrain.physics_material)

    runner = _make_runner(env, agent_cfg, resume_path)
    policy = runner.get_inference_policy(device=base_env.device)

    step_dt = base_env.step_dt
    steps_per_cmd = int(args_cli.command_timestep / step_dt)
    history_collection_interval = steps_per_cmd / args_cli.history_length
    timing = (step_dt, steps_per_cmd, history_collection_interval)
    print(
        f"[INFO] step_dt={step_dt:.4f}s, steps_per_cmd={steps_per_cmd}, "
        f"history_collection_interval={history_collection_interval}"
    )

    test_state = _get_fdm_state(base_env)
    test_proprio = _get_fdm_proprioceptive(base_env)
    test_height = _get_fdm_height_scan(base_env)
    dims = (test_state.shape[1], test_proprio.shape[1], tuple(test_height.shape[1:]))
    print(f"[INFO] state_dim={dims[0]}, proprio_dim={dims[1]}, height_scan={dims[2]}")

    val_payload = collect_round(
        env,
        base_env,
        policy,
        trajectory_length=args_cli.val_trajectory_length,
        round_name="validation",
        dims=dims,
        timing=timing,
    )

    cfg = _training_cfg()
    model = None
    optimizer = None
    scheduler = None
    log_dir = None
    history = []

    for collection_round in range(args_cli.start_collection_round, args_cli.collection_rounds):
        round_name = f"train_round_{collection_round:02d}"
        train_payload = collect_round(
            env,
            base_env,
            policy,
            trajectory_length=args_cli.trajectory_length,
            round_name=round_name,
            dims=dims,
            timing=timing,
        )
        trainer = FDMTrainer(
            cfg,
            payload=train_payload,
            val_payload=val_payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            log_dir=log_dir,
        )
        if model is None and args_cli.resume_fdm_checkpoint is not None:
            checkpoint = torch.load(Path(args_cli.resume_fdm_checkpoint).expanduser(), map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[INFO] Resumed FDM model from: {args_cli.resume_fdm_checkpoint}")
        log_dir = trainer.log_dir
        if args_cli.save_round_datasets:
            if collection_round == 0:
                _save_payload(log_dir / "validation_dataset_payload.pkl", val_payload)
            _save_payload(log_dir / f"train_dataset_payload_{collection_round:02d}.pkl", train_payload)

        print(
            f"[INFO] Round {collection_round:02d}: train_samples={len(trainer.train_dataset)}, "
            f"val_samples={len(trainer.val_dataset)}, collision_rate={trainer.train_dataset.collision_rate:.4f}"
        )
        result = trainer.train()
        trainer.model.save(
            log_dir / f"model_collection_round_{collection_round:02d}.pth",
            meta={"collection_round": collection_round, "best_val_loss": result["best_val_loss"]},
        )
        history.append({"collection_round": collection_round, "best_val_loss": result["best_val_loss"]})
        with open(log_dir / "online_losses.yaml", "w") as f:
            yaml.safe_dump({"history": history}, f)
        model = trainer.model
        optimizer = trainer.optimizer
        scheduler = trainer.scheduler

    if model is not None and log_dir is not None:
        model.save(log_dir / "model_final.pth", meta={"history": history})
        print(f"[INFO] Online FDM training complete. Log dir: {log_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
