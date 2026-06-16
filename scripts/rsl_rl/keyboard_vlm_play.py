"""Keyboard + VLM hybrid control script for G1 velocity-tracking tasks.

WASD keys control the robot's velocity in real time (fallback).
Press 'V' to query the VLM server for a navigation command.
Press 'C' to toggle continuous VLM mode (query every few seconds).

Architecture:
  Keyboard ──→ velocity cmd ──→ low-level policy ──→ robot
  RGB cam ──→ image buffer ──→ TCP ──→ VLM server (4090) ──→ vel cmd ──→ policy
"""

# =====================================================================
#  KEYBOARD CONTROL CONFIG
# =====================================================================
KEY_VEL = {
    "W":   {"cmd": "lin_vel_x",  "val":  0.7},
    "S":   {"cmd": "lin_vel_x",  "val": -0.5},
    "A":   {"cmd": "lin_vel_y",  "val":  0.3},
    "D":   {"cmd": "lin_vel_y",  "val": -0.3},
    "Q":   {"cmd": "ang_vel_z",  "val":  1.5},
    "E":   {"cmd": "ang_vel_z",  "val": -1.5},
}
SHIFT_SPEED_SCALE = 2.0
# =====================================================================

"""Launch Isaac Sim Simulator first."""

import argparse
import base64
import io
import json
import math
import os
import select
import socket
import sys
import termios
import threading
import time
import tty
from importlib.metadata import version

import numpy as np
from PIL import Image

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint with keyboard + VLM velocity control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity-VLN", help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# VLM settings
parser.add_argument("--vlm_host", type=str, default="192.168.1.100",
                    help="VLM server IP address (your 4090 desktop).")
parser.add_argument("--vlm_port", type=int, default=54321, help="VLM server port.")
parser.add_argument("--instruction", type=str,
                    default="Move forward and explore the environment safely. Avoid obstacles.",
                    help="Navigation instruction text sent to the VLM.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Always enable cameras: VLM needs RGB images
args_cli.enable_cameras = True
if args_cli.video:
    args_cli.enable_cameras = True  # already set, but keep for clarity

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

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
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


# ==============================================================================
#  Keyboard input reader
# ==============================================================================

class KeyboardReader:
    """Non-blocking keyboard input reader for Linux.

    In raw terminal mode, Shift+key sends the uppercase version of that key.
    We preserve the original case so callers can detect Shift via isupper().
    """

    def __init__(self):
        self._old_settings = None
        self._keys = set()
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
                        elif ch.strip():
                            self._keys.add(ch)
                else:
                    with self._lock:
                        self._keys.clear()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_settings)

    def get_active_keys(self):
        with self._lock:
            return set(self._keys)

    def restore(self):
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass


# ==============================================================================
#  VLM communication
# ==============================================================================

def sample_images_and_send_to_vlm(image_list, vlm_host, vlm_port, query):
    """Sample 8 frames from image history, send to VLM server, return response text.

    Args:
        image_list: list of PIL Image objects (accumulated over time).
        vlm_host: VLM server hostname or IP.
        vlm_port: VLM server port.
        query: natural language instruction text.

    Returns:
        VLM output text string, or None on failure.
    """
    if len(image_list) == 0:
        print("[VLM] No images accumulated yet.")
        return None

    # Pad with black images if fewer than 8
    if len(image_list) < 8:
        print(f"[VLM] Only {len(image_list)} images accumulated, padding with black frames.")
        padded = image_list.copy()
        for _ in range(8 - len(image_list)):
            padded.insert(0, Image.new('RGB', padded[-1].size, (0, 0, 0)))
        image_list = padded

    # Uniformly sample 7 historical frames + 1 current
    num_images = len(image_list)
    indices = [int(i * (num_images - 1) / 7) for i in range(7)]
    sampled_images = [image_list[i] for i in indices]
    sampled_images.append(image_list[-1])

    # Encode to base64 JPEG
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

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(30.0)
            s.connect((vlm_host, vlm_port))
            data_bytes = json.dumps(request_data).encode()
            s.sendall(len(data_bytes).to_bytes(8, 'big'))
            s.sendall(data_bytes)

            size_data = s.recv(8)
            size = int.from_bytes(size_data, 'big')

            response_data = b''
            while len(response_data) < size:
                packet = s.recv(4096)
                if not packet:
                    break
                response_data += packet

            response = json.loads(response_data.decode())
            return response
    except socket.timeout:
        print("[VLM] Connection timed out. Is the VLM server running?")
        return None
    except ConnectionRefusedError:
        print(f"[VLM] Connection refused at {vlm_host}:{vlm_port}. Is the VLM server running?")
        return None
    except Exception as e:
        print(f"[VLM] Communication error: {e}")
        return None


def parse_vel_command(text):
    """Parse VLM text output into (velocity_vector, duration_seconds).

    Matches NaVILA-Bench's get_vel_command() logic.
    Returns: ([vx, vy, omega], time_in_seconds)
    """
    if text is None:
        return [0.0, 0.0, 0.0], 0.0

    t = text.lower()
    if "turn left" in t:
        if "45" in t:
            return [0.0, 0.0, math.pi / 6.0], 1.5
        elif "30" in t:
            return [0.0, 0.0, math.pi / 6.0], 1.0
        elif "15" in t:
            return [0.0, 0.0, math.pi / 6.0], 0.5
        return [0.0, 0.0, math.pi / 6.0], 0.5
    elif "turn right" in t:
        if "45" in t:
            return [0.0, 0.0, -math.pi / 6.0], 1.5
        elif "30" in t:
            return [0.0, 0.0, -math.pi / 6.0], 1.0
        elif "15" in t:
            return [0.0, 0.0, -math.pi / 6.0], 0.5
        return [0.0, 0.0, -math.pi / 6.0], 0.5
    elif "move forward" in t or "move" in t:
        if "75" in t:
            return [0.5, 0.0, 0.0], 1.5
        elif "50" in t:
            return [0.5, 0.0, 0.0], 1.0
        elif "25" in t:
            return [0.5, 0.0, 0.0], 0.5
        return [0.5, 0.0, 0.0], 0.5
    elif "stop" in t:
        return [0.0, 0.0, 0.0], 0.0
    else:
        return [0.5, 0.0, 0.0], 0.5


# ==============================================================================
#  Main
# ==============================================================================

def main():
    """Keyboard + VLM hybrid control loop."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

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
            "video_folder": os.path.join(log_dir, "videos", "keyboard_vlm_play"),
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

    dt = env.unwrapped.step_dt

    # Prevent command manager from overwriting our velocity values
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    command_term.is_standing_env[:] = False
    command_term.is_heading_env[:] = False
    command_term.time_left[:] = 1e9

    # Start keyboard reader
    kb = KeyboardReader()
    kb.start()

    # VLM state
    image_observations = []          # accumulated PIL Images
    vlm_vel_command = torch.tensor([0.0, 0.0, 0.0], device=env.unwrapped.device)
    vlm_steps_remaining = 0          # remaining env steps for current VLM command
    vlm_continuous_mode = False      # toggle for continuous VLM querying
    vlm_last_query_time = -999.0     # simulation time of last VLM query
    vlm_query_interval = 5.0         # seconds between auto-queries in continuous mode
    prev_v_key = False               # for edge-triggered V key
    prev_c_key = False               # for edge-triggered C key

    # Timing for image capture
    image_capture_interval = 0.5     # capture every 0.5s (2 Hz)
    steps_per_image = max(1, int(image_capture_interval / dt))

    print("\n" + "=" * 60)
    print("  KEYBOARD + VLM CONTROL MODE")
    print(f"  Task: {args_cli.task}")
    print(f"  VLM Server: {args_cli.vlm_host}:{args_cli.vlm_port}")
    print(f"  Instruction: {args_cli.instruction}")
    print("  W: +{:.1f} m/s  S: {:.1f} m/s  A: +{:.1f} m/s  D: {:.1f} m/s".format(
        KEY_VEL["W"]["val"], KEY_VEL["S"]["val"], KEY_VEL["A"]["val"], KEY_VEL["D"]["val"]))
    print("  Q: +{:.1f} rad/s  E: {:.1f} rad/s  Shift: x{}".format(
        KEY_VEL["Q"]["val"], KEY_VEL["E"]["val"], SHIFT_SPEED_SCALE))
    print("  V: Query VLM once   C: Toggle continuous VLM mode")
    print("  Release all keys to stop.  Ctrl-C to exit.")
    print("=" * 60 + "\n")

    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    timestep = 0
    sim_time = 0.0

    try:
        while simulation_app.is_running():
            start_time = time.time()

            # --- decode keyboard input ---
            keys_raw = kb.get_active_keys()
            shift_held = any(k.isupper() for k in keys_raw)
            keys = {k.lower() for k in keys_raw}

            # Edge-triggered key detection
            v_pressed = 'v' in keys
            c_pressed = 'c' in keys

            if v_pressed and not prev_v_key:
                # Single VLM query
                print(f"\n[VLM] Querying VLM server at {args_cli.vlm_host}:{args_cli.vlm_port}...")
                vlm_response = sample_images_and_send_to_vlm(
                    image_observations, args_cli.vlm_host, args_cli.vlm_port, args_cli.instruction
                )
                if vlm_response is not None:
                    print(f"[VLM] Response: {vlm_response}")
                    vel_cmd, duration = parse_vel_command(vlm_response)
                    vlm_vel_command = torch.tensor(vel_cmd, device=env.unwrapped.device)
                    vlm_steps_remaining = int(duration / dt) if duration > 0 else 0
                    print(f"[VLM] Command: vx={vel_cmd[0]:+.2f} vy={vel_cmd[1]:+.2f} "
                          f"vyaw={vel_cmd[2]:+.2f}  duration={duration:.1f}s  steps={vlm_steps_remaining}")
                else:
                    vlm_steps_remaining = 0
            prev_v_key = v_pressed

            if c_pressed and not prev_c_key:
                vlm_continuous_mode = not vlm_continuous_mode
                print(f"[VLM] Continuous mode: {'ON' if vlm_continuous_mode else 'OFF'}")
            prev_c_key = c_pressed

            # --- keyboard velocity ---
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

            # --- determine active velocity command ---
            # Continuous VLM mode: auto-query at intervals
            if vlm_continuous_mode and (sim_time - vlm_last_query_time) >= vlm_query_interval:
                print(f"\n[VLM] Auto-query at t={sim_time:.1f}s...")
                vlm_response = sample_images_and_send_to_vlm(
                    image_observations, args_cli.vlm_host, args_cli.vlm_port, args_cli.instruction
                )
                vlm_last_query_time = sim_time
                if vlm_response is not None:
                    print(f"[VLM] Response: {vlm_response}")
                    vel_cmd, duration = parse_vel_command(vlm_response)
                    vlm_vel_command = torch.tensor(vel_cmd, device=env.unwrapped.device)
                    vlm_steps_remaining = int(duration / dt) if duration > 0 else 0
                    print(f"[VLM] Command: vx={vel_cmd[0]:+.2f} vy={vel_cmd[1]:+.2f} "
                          f"vyaw={vel_cmd[2]:+.2f}  duration={duration:.1f}s")

            # If VLM command is active and no keyboard input, use VLM command
            anyone_key_pressed = any(k in keys for k in ['w', 'a', 's', 'd', 'q', 'e'])
            if vlm_steps_remaining > 0 and not anyone_key_pressed:
                command_term.vel_command_b[:, 0] = vlm_vel_command[0]
                command_term.vel_command_b[:, 1] = vlm_vel_command[1]
                command_term.vel_command_b[:, 2] = vlm_vel_command[2]
                vlm_steps_remaining -= 1
                vlm_active = True
            else:
                # Keyboard takes priority (or VLM command finished)
                command_term.vel_command_b[:, 0] = lin_vel_x
                command_term.vel_command_b[:, 1] = lin_vel_y
                command_term.vel_command_b[:, 2] = ang_vel_z
                vlm_active = False
                # If keyboard is used while VLM was active, cancel VLM
                if anyone_key_pressed and vlm_steps_remaining > 0:
                    vlm_steps_remaining = 0

            # --- capture RGB image for VLM ---
            if timestep % steps_per_image == 0:
                try:
                    rgb_data = env.unwrapped.scene["rgb_camera"].data.output["rgb"]
                    # Shape: (num_envs, H, W, 3), values typically float in [0, 1]
                    rgb_np = rgb_data[0].cpu().numpy()
                    if rgb_np.max() <= 1.0:
                        rgb_np = (rgb_np * 255.0).astype(np.uint8)
                    else:
                        rgb_np = rgb_np.astype(np.uint8)
                    pil_img = Image.fromarray(rgb_np)
                    image_observations.append(pil_img)
                    # Keep max 200 frames (100 seconds at 2Hz) to avoid memory blowup
                    if len(image_observations) > 200:
                        image_observations = image_observations[-200:]
                except Exception as e:
                    print(f"[WARN] Failed to capture RGB image: {e}")

            # --- policy inference + env step ---
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

            # --- status display ---
            if timestep % 100 == 0:
                boost = "(boost)" if shift_held else ""
                vlm_flag = "[VLM]" if vlm_active else ""
                mode = "[Auto]" if vlm_continuous_mode else ""
                vx = command_term.vel_command_b[0, 0].item()
                vy = command_term.vel_command_b[0, 1].item()
                vyaw = command_term.vel_command_b[0, 2].item()
                imgs = len(image_observations)
                print(f"[{timestep}] vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f} "
                      f"{vlm_flag}{mode} {boost} (imgs:{imgs})", flush=True)

            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break
            else:
                timestep += 1

            sim_time += dt

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
