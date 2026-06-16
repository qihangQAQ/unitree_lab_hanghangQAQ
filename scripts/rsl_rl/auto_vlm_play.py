"""Automatic VLM-guided play script for G1 velocity-tracking tasks.

The control loop is intentionally simple:
  WAITING_VLM: keep sending zero velocity while a background VLM request runs.
  EXECUTING: execute the parsed velocity command for its requested duration.
  STOPPED: keep zero velocity after the VLM says stop.
"""

import argparse
import base64
import io
import json
import math
import os
import shlex
import shutil
import socket
import subprocess
import threading
import time
from importlib.metadata import version

import numpy as np
from PIL import Image

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Automatically play a checkpoint with VLM velocity commands.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity-VLN", help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--vlm_host", type=str, default="127.0.0.1", help="VLM server host or IP.")
parser.add_argument("--vlm_port", type=int, default=54321, help="VLM server port.")
parser.add_argument(
    "--instruction",
    type=str,
    default="Move forward and explore the environment safely. Avoid obstacles.",
    help="Navigation instruction text sent to the VLM.",
)
parser.add_argument("--image_interval", type=float, default=0.5, help="Seconds between RGB frames saved for VLM.")
parser.add_argument("--max_image_history", type=int, default=200, help="Maximum number of cached RGB frames.")
parser.add_argument("--status_interval", type=int, default=100, help="Print status every N environment steps.")
parser.add_argument(
    "--timeline_log",
    type=str,
    default="logs/vlm_runtime/auto_vlm_timeline.log",
    help="Path to the high-level VLM decision timeline log.",
)
parser.add_argument("--append_timeline_log", action="store_true", help="Append to the timeline log instead of truncating it.")
parser.add_argument("--open_log_terminal", action="store_true", help="Open a terminal window tailing the timeline log.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

TIMELINE_START_TIME = time.time()
TIMELINE_LOG_PATH = os.path.abspath(args_cli.timeline_log)
TIMELINE_LOCK = threading.Lock()


def log_timeline(message, also_print=True):
    elapsed = time.time() - TIMELINE_START_TIME
    line = f"{elapsed:8.2f}s  {message}"
    with TIMELINE_LOCK:
        with open(TIMELINE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    if also_print:
        print(line, flush=True)


def open_timeline_terminal(log_path):
    quoted_path = shlex.quote(log_path)
    tail_cmd = f"tail -f {quoted_path}; exec bash"
    terminal_cmds = [
        ["gnome-terminal", "--", "bash", "-lc", tail_cmd],
        ["konsole", "-e", "bash", "-lc", tail_cmd],
        ["xterm", "-e", "bash", "-lc", tail_cmd],
    ]
    for cmd in terminal_cmds:
        if shutil.which(cmd[0]):
            try:
                subprocess.Popen(cmd)
                print(f"[INFO] Opened timeline terminal with {cmd[0]}: tail -f {log_path}")
                return True
            except Exception as e:
                print(f"[WARNING] Failed to open {cmd[0]} for timeline log: {e}")
    print(f"[INFO] Could not auto-open a terminal. Run this in another terminal:\n  tail -f {log_path}")
    return False


os.makedirs(os.path.dirname(TIMELINE_LOG_PATH), exist_ok=True)
if not args_cli.append_timeline_log:
    with open(TIMELINE_LOG_PATH, "w", encoding="utf-8"):
        pass
log_timeline(f"Timeline log started: {TIMELINE_LOG_PATH}")
if args_cli.open_log_terminal:
    open_timeline_terminal(TIMELINE_LOG_PATH)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

try:
    import unitree_rl_lab.rsl_rl_ext  # noqa: F401

    print("[INFO] NP3O extension imported, monkey-patch injected")
except ImportError as e:
    print(f"[WARNING] Unable to import NP3O extension: {e}")

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


class VlmRequestState:
    """Thread-safe holder for one in-flight VLM request."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.result = None
        self.error = None
        self.started_at = 0.0

    def start(self):
        with self.lock:
            self.running = True
            self.result = None
            self.error = None
            self.started_at = time.time()

    def finish(self, result=None, error=None):
        with self.lock:
            self.running = False
            self.result = result
            self.error = error

    def snapshot(self):
        with self.lock:
            return self.running, self.result, self.error, self.started_at

    def clear_result(self):
        with self.lock:
            self.result = None
            self.error = None


def sample_images_and_send_to_vlm(image_list, vlm_host, vlm_port, query):
    """Sample 8 frames from image history, send to VLM server, return response text."""
    if len(image_list) == 0:
        log_timeline("VLM request skipped: no images accumulated yet")
        return None

    if len(image_list) < 8:
        log_timeline(f"Only {len(image_list)} cached images; padding VLM input with black frames")
        padded = image_list.copy()
        for _ in range(8 - len(image_list)):
            padded.insert(0, Image.new("RGB", padded[-1].size, (0, 0, 0)))
        image_list = padded

    num_images = len(image_list)
    indices = [int(i * (num_images - 1) / 7) for i in range(7)]
    sampled_images = [image_list[i] for i in indices]
    sampled_images.append(image_list[-1])

    encoded_images = []
    for img in sampled_images:
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img = img.clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            pil_img = img
        else:
            pil_img = Image.fromarray(np.array(img, dtype=np.uint8))

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        encoded_images.append(base64.b64encode(buf.getvalue()).decode())

    request_data = {"images": encoded_images, "query": query}

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(30.0)
        s.connect((vlm_host, vlm_port))
        data_bytes = json.dumps(request_data).encode()
        s.sendall(len(data_bytes).to_bytes(8, "big"))
        s.sendall(data_bytes)

        size_data = s.recv(8)
        size = int.from_bytes(size_data, "big")

        response_data = b""
        while len(response_data) < size:
            packet = s.recv(4096)
            if not packet:
                break
            response_data += packet

    return json.loads(response_data.decode())


def query_vlm_worker(image_snapshot, instruction, vlm_state):
    try:
        response = sample_images_and_send_to_vlm(
            image_snapshot,
            args_cli.vlm_host,
            args_cli.vlm_port,
            instruction,
        )
        vlm_state.finish(result=response)
    except Exception as e:
        vlm_state.finish(error=e)


def parse_vel_command(text):
    """Parse VLM text output into ([vx, vy, omega], duration_seconds)."""
    if text is None:
        return [0.0, 0.0, 0.0], 0.0

    t = text.lower()
    if "turn left" in t:
        if "45" in t:
            return [0.0, 0.0, math.pi / 6.0], 1.5
        if "30" in t:
            return [0.0, 0.0, math.pi / 6.0], 1.0
        if "15" in t:
            return [0.0, 0.0, math.pi / 6.0], 0.5
        return [0.0, 0.0, math.pi / 6.0], 0.5
    if "turn right" in t:
        if "45" in t:
            return [0.0, 0.0, -math.pi / 6.0], 1.5
        if "30" in t:
            return [0.0, 0.0, -math.pi / 6.0], 1.0
        if "15" in t:
            return [0.0, 0.0, -math.pi / 6.0], 0.5
        return [0.0, 0.0, -math.pi / 6.0], 0.5
    if "move forward" in t or "move" in t:
        if "75" in t:
            return [0.5, 0.0, 0.0], 1.5
        if "50" in t:
            return [0.5, 0.0, 0.0], 1.0
        if "25" in t:
            return [0.5, 0.0, 0.0], 0.5
        return [0.5, 0.0, 0.0], 0.5
    if "stop" in t:
        return [0.0, 0.0, 0.0], 0.0
    return [0.5, 0.0, 0.0], 0.5


def capture_rgb_image(env):
    rgb_data = env.unwrapped.scene["rgb_camera"].data.output["rgb"]
    rgb_np = rgb_data[0].cpu().numpy()
    if rgb_np.max() <= 1.0:
        rgb_np = (rgb_np * 255.0).clip(0, 255).astype(np.uint8)
    else:
        rgb_np = rgb_np.clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb_np)


def make_runner(env, agent_cfg, resume_path):
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
    return runner


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
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
            "video_folder": os.path.join(log_dir, "videos", "auto_vlm_play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = make_runner(env, agent_cfg, resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    steps_per_image = max(1, int(args_cli.image_interval / dt))
    zero_cmd = torch.tensor([0.0, 0.0, 0.0], device=env.unwrapped.device)
    current_cmd = zero_cmd.clone()
    cmd_steps_remaining = 0
    state = "WAITING_VLM"

    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    command_term.is_standing_env[:] = False
    command_term.is_heading_env[:] = False
    command_term.time_left[:] = 1e9

    image_observations = []
    vlm_state = VlmRequestState()

    print("\n" + "=" * 72)
    print("  AUTO VLM CONTROL MODE")
    print(f"  Task: {args_cli.task}")
    print(f"  VLM Server: {args_cli.vlm_host}:{args_cli.vlm_port}")
    print(f"  Instruction: {args_cli.instruction}")
    print(f"  Timeline log: {TIMELINE_LOG_PATH}")
    print("  WAITING_VLM sends zero velocity until the next VLM command returns.")
    print("=" * 72 + "\n")
    log_timeline(f"Task={args_cli.task} VLM={args_cli.vlm_host}:{args_cli.vlm_port}")
    log_timeline(f"Instruction: {args_cli.instruction}")

    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    timestep = 0
    try:
        while simulation_app.is_running():
            start_time = time.time()

            if timestep % steps_per_image == 0:
                try:
                    image_observations.append(capture_rgb_image(env))
                    if len(image_observations) > args_cli.max_image_history:
                        image_observations = image_observations[-args_cli.max_image_history :]
                except Exception as e:
                    print(f"[WARN] Failed to capture RGB image: {e}")

            running, result, error, started_at = vlm_state.snapshot()

            if state == "WAITING_VLM":
                active_cmd = zero_cmd
                if error is not None:
                    log_timeline(f"VLM request failed: {error}")
                    vlm_state.clear_result()
                    running = False

                if result is not None:
                    latency = time.time() - started_at
                    log_timeline(f"NaVILA output: {result} | inference_time={latency:.2f}s")
                    vel_cmd, duration = parse_vel_command(result)
                    current_cmd = torch.tensor(vel_cmd, device=env.unwrapped.device)
                    cmd_steps_remaining = int(duration / dt)
                    vlm_state.clear_result()

                    if cmd_steps_remaining > 0:
                        state = "EXECUTING"
                        active_cmd = current_cmd
                        log_timeline(
                            f"Execute cmd=[{vel_cmd[0]:+.2f}, {vel_cmd[1]:+.2f}, {vel_cmd[2]:+.2f}] "
                            f"duration={duration:.1f}s steps={cmd_steps_remaining}"
                        )
                    else:
                        state = "STOPPED"
                        active_cmd = zero_cmd
                        log_timeline("Stop received. Holding zero velocity.")

                elif not running and len(image_observations) > 0:
                    log_timeline(f"Request NaVILA with {len(image_observations)} cached frames")
                    vlm_state.start()
                    image_snapshot = list(image_observations)
                    thread = threading.Thread(
                        target=query_vlm_worker,
                        args=(image_snapshot, args_cli.instruction, vlm_state),
                        daemon=True,
                    )
                    thread.start()

            elif state == "EXECUTING":
                active_cmd = current_cmd
                cmd_steps_remaining -= 1
                if cmd_steps_remaining <= 0:
                    state = "WAITING_VLM"
                    active_cmd = zero_cmd
                    current_cmd = zero_cmd.clone()
                    log_timeline("Command finished. Holding zero velocity and requesting next VLM command.")

            elif state == "STOPPED":
                active_cmd = zero_cmd

            else:
                raise RuntimeError(f"Unknown control state: {state}")

            command_term.vel_command_b[:, 0] = active_cmd[0]
            command_term.vel_command_b[:, 1] = active_cmd[1]
            command_term.vel_command_b[:, 2] = active_cmd[2]

            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

            if timestep % args_cli.status_interval == 0:
                vx = command_term.vel_command_b[0, 0].item()
                vy = command_term.vel_command_b[0, 1].item()
                vyaw = command_term.vel_command_b[0, 2].item()
                running, _, _, started_at = vlm_state.snapshot()
                wait_s = time.time() - started_at if running else 0.0
                print(
                    f"[{timestep}] state={state} vx={vx:+.2f} vy={vy:+.2f} "
                    f"vyaw={vyaw:+.2f} imgs={len(image_observations)} "
                    f"vlm_running={running} wait={wait_s:.2f}s",
                    flush=True,
                )

            timestep += 1
            if args_cli.video and timestep >= args_cli.video_length:
                break

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        log_timeline("Keyboard interrupt received, exiting")
    finally:
        log_timeline("Closing environment")
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
