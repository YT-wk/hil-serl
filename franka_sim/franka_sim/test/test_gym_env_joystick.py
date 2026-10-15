import argparse
import os
from pathlib import Path
import sys

import numpy as np

import gymnasium as gym

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in ("serl_robot_infra", "franka_sim", "examples"):
    sys.path.insert(0, str(_REPO_ROOT / _path))

# import controller wrappers
from franka_env.envs.wrappers import JoystickIntervention, KeyboardIntervention
from franka_env.spacemouse.spacemouse_expert import ControllerType

from franka_sim.utils.viewer_utils import DualMujocoViewer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller",
        type=str,
        default="xbox",
        choices=("keyboard", "xbox", "ps5"),
        help="Controller type.",
    )
    parser.add_argument(
        "--keyboard-pos-speed",
        type=float,
        default=0.2,
        help="Normalized xyz action magnitude for keyboard control.",
    )
    parser.add_argument("--image-obs", action="store_true", help="Include camera images in observations")
    parser.add_argument("--num-viewers", type=int, default=1, help="Number of MuJoCo viewer windows")

    args = parser.parse_args()
    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "DISPLAY is not set. Run this viewer script from a desktop/VNC/X11 session; "
            "headless EGL can render rgb_array tests, but mujoco.viewer requires a GUI display."
        )
    if os.environ.get("MUJOCO_GL") in {"egl", "osmesa"}:
        raise RuntimeError(
            "MUJOCO_GL is set to a headless backend. Unset MUJOCO_GL before running mujoco.viewer."
        )
    controller_type = ControllerType(args.controller)

    env_id = "PandaPickCubeVision-v0" if args.image_obs else "PandaPickCube-v0"
    env = gym.make(env_id, render_mode="human", image_obs=args.image_obs)
    if controller_type == ControllerType.KEYBOARD:
        env = KeyboardIntervention(env, pos_speed=args.keyboard_pos_speed)
        print(
            "Keyboard listener started. Keep keyboard focus on the terminal while "
            "watching the viewer to avoid MuJoCo viewer shortcuts changing the "
            "rendering. Controls: A/D left/right, W/S forward/back, Q/R up, Z/F down, "
            "X toggle gripper, [/] switch camera."
        )
    else:
        env = JoystickIntervention(env, controller_type=controller_type)

    env.reset()

    # Create the dual viewer.
    dual_viewer = DualMujocoViewer(
        env.unwrapped.model,
        env.unwrapped.data,
        num_viewers=args.num_viewers,
    )
    if controller_type == ControllerType.KEYBOARD:
        env.start_listening(camera_callback=dual_viewer.cycle_camera)

    try:
        with dual_viewer as viewer:
            for i in range(100000):
                env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
                viewer.sync()
    finally:
        env.close()


if __name__ == "__main__":
    main()
