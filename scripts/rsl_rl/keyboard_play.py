"""Keyboard-controlled play script for velocity-tracking tasks.

WASD keys control the robot's velocity command in real time.
"""

# =====================================================================
#  KEYBOARD CONTROL CONFIG  — 修改这里的数值来调整各方向的速度
#  lin_vel: 线速度 (m/s),  ang_vel: 角速度 (rad/s)
# =====================================================================
KEY_VEL = {
    #  按键      指令类型       速度值    说明
    "W":   {"cmd": "lin_vel_x",  "val":  1.0},  # 前进
    "S":   {"cmd": "lin_vel_x",  "val": -0.5},  # 后退
    "A":   {"cmd": "lin_vel_y",  "val":  0.3},  # 左移
    "D":   {"cmd": "lin_vel_y",  "val": -0.3},  # 右移
    "Q":   {"cmd": "ang_vel_z",  "val":  2.4},  # 左转
    "E":   {"cmd": "ang_vel_z",  "val": -2.4},  # 右转
}
SHIFT_SPEED_SCALE = 2.0   # 按住 Shift 时的速度倍率
# =====================================================================

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import select
import sys
import termios
import threading
import time
import tty

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint with keyboard velocity control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity", help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from importlib.metadata import version

try:
    import unitree_rl_lab.rsl_rl_ext
    print("[INFO] NP3O extension imported, monkey-patch injected")
except ImportError as e:
    print(f"[WARNING] Unable to import NP3O extension: {e}")

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion.mdp.terrain.test_terrain_cfg import TEST_TERRAINS_CFG
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


class KeyboardReader:
    """Non-blocking keyboard input reader for Linux.

    In raw terminal mode, Shift+key sends the uppercase version of that key.
    We preserve the original case so callers can detect Shift via isupper().
    """

    def __init__(self):
        self._old_settings = None
        self._keys = set()  # raw characters, preserving case
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

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
        try:
            tty.setraw(fd)
            while self._running:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    ch = sys.stdin.read(1)
                    with self._lock:
                        if ch == '\x03':  # Ctrl-C
                            self._running = False
                        elif ch == '\x1b':  # Escape sequence (arrow keys etc.)
                            r2, _, _ = select.select([sys.stdin], [], [], 0.01)
                            if r2:
                                sys.stdin.read(2)
                        elif ch.strip():  # ignore non-printable
                            self._keys.add(ch)
                else:
                    with self._lock:
                        self._keys.clear()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_settings)

    def get_active_keys(self):
        """Return the set of currently pressed keys (raw case)."""
        with self._lock:
            return set(self._keys)

    def restore(self):
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass


def configure_for_keyboard(env_cfg):
    """Modify env config for keyboard-controlled play.

    - Single environment, test terrain from test_terrain_cfg.py
    - Disable terminations (keep only time_out)
    - Disable curriculum, domain randomization, and disturbances
    """
    from isaaclab.managers import EventTermCfg as EventTerm
    from unitree_rl_lab.tasks.locomotion import mdp

    # -- Single env
    env_cfg.scene.num_envs = 1

    # -- Replace terrain with the test terrain config (all types, equal proportions)
    n_types = len(TEST_TERRAINS_CFG.sub_terrains)
    TEST_TERRAINS_CFG.num_rows = 2
    TEST_TERRAINS_CFG.num_cols = n_types
    TEST_TERRAINS_CFG.curriculum = False
    TEST_TERRAINS_CFG.border_width = 25.0  # extra flat border for spawn area
    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = TEST_TERRAINS_CFG
    env_cfg.scene.terrain.max_init_terrain_level = 9

    # Spawn robot in the flat border zone just outside the terrain grid.
    # Grid: 2 rows × 8m = 16m tall, centered at origin → grid y ∈ [-8, 8]
    # Placing robot at y = 10 puts it on the border (flat), 2m from nearest terrain.
    SPAWN_Y = 10.0

    # -- Disable all terminations except time_out
    keep_attrs = {"time_out"}
    for attr in list(env_cfg.terminations.__dict__.keys()):
        if not attr.startswith("_") and attr not in keep_attrs:
            delattr(env_cfg.terminations, attr)

    # -- Disable curriculum
    for attr in list(env_cfg.curriculum.__dict__.keys()):
        if not attr.startswith("_"):
            delattr(env_cfg.curriculum, attr)

    # -- Replace events with clean reset-only events (no randomization / disturbances)
    for attr in list(env_cfg.events.__dict__.keys()):
        if not attr.startswith("_"):
            delattr(env_cfg.events, attr)

    env_cfg.events.reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (SPAWN_Y, SPAWN_Y), "yaw": (-1.57, -1.57)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    env_cfg.events.reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )

    # -- Prevent command resampling and standing/heading override
    env_cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False

    # -- Long episode
    env_cfg.episode_length_s = 600.0

    return env_cfg


def main():
    """Play with keyboard-controlled velocity commands."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg = configure_for_keyboard(env_cfg)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "keyboard_play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "NP3ORunner":
        try:
            runner = NP3ORunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        except NameError:
            from unitree_rl_lab.rsl_rl_ext import NP3ORunner as NP3ORunnerExt
            runner = NP3ORunnerExt(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # -- Prevent command manager from overwriting keyboard values
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    command_term.is_standing_env[:] = False
    command_term.is_heading_env[:] = False
    command_term.time_left[:] = 1e9

    # -- Inject zero initial command + flush observation history
    # Without this, the policy sees random commands from env reset in its
    # initial observation (and in the history_length=5 buffer), causing
    # wild first actions that deviate the robot from the intended path.
    command_term.vel_command_b[:, 0] = 0.0
    command_term.vel_command_b[:, 1] = 0.0
    command_term.vel_command_b[:, 2] = 0.0
    for _ in range(env_cfg.observations.policy.history_length):
        obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs = obs[0] if isinstance(obs, tuple) else obs

    # -- Start keyboard reader
    kb = KeyboardReader()
    kb.start()

    print("\n" + "=" * 60)
    print("  KEYBOARD CONTROL MODE")
    print("  Robot spawns on flat border — walk toward the terrain tiles!")
    print("  W: +{w[val]:.1f} m/s  S: {s[val]:.1f} m/s  A: +{a[val]:.1f} m/s  D: {d[val]:.1f} m/s".format(
        w=KEY_VEL["W"], s=KEY_VEL["S"], a=KEY_VEL["A"], d=KEY_VEL["D"]))
    print("  Q: +{q[val]:.1f} rad/s  E: {e[val]:.1f} rad/s  Shift: x{scale}".format(
        q=KEY_VEL["Q"], e=KEY_VEL["E"], scale=SHIFT_SPEED_SCALE))
    print("  Release all keys to stop.  Ctrl-C to exit.")
    print("=" * 60 + "\n")

    timestep = 0

    try:
        while simulation_app.is_running():
            start_time = time.time()

            # --- decode keyboard input ---
            keys_raw = kb.get_active_keys()
            # In raw mode, Shift+key produces uppercase; detect shift from any uppercase letter
            shift_held = any(k.isupper() for k in keys_raw)
            keys = {k.lower() for k in keys_raw}

            lin_vel_x = 0.0
            lin_vel_y = 0.0
            ang_vel_z = 0.0

            for key_name, cfg in KEY_VEL.items():
                if key_name.lower() in keys:
                    v = cfg["val"] * (SHIFT_SPEED_SCALE if shift_held else 1.0)
                    if cfg["cmd"] == "lin_vel_x":
                        lin_vel_x += v
                    elif cfg["cmd"] == "lin_vel_y":
                        lin_vel_y += v
                    elif cfg["cmd"] == "ang_vel_z":
                        ang_vel_z += v

            # --- inject velocity command ---
            command_term.vel_command_b[:, 0] = lin_vel_x
            command_term.vel_command_b[:, 1] = lin_vel_y
            command_term.vel_command_b[:, 2] = ang_vel_z

            # --- policy inference + env step ---
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

            if timestep % 100 == 0:
                boost = "(boost)" if shift_held else ""
                print(f"[{timestep}] vx={lin_vel_x:+.2f} vy={lin_vel_y:+.2f} vyaw={ang_vel_z:+.2f} {boost}", flush=True)

            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break
            else:
                timestep += 1

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received, exiting...")
    finally:
        kb.stop()
        kb.restore()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
