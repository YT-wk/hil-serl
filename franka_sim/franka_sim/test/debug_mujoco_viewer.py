"""Diagnose MuJoCo/GLFW viewer crashes in isolated subprocesses.

Run from the ``franka_sim`` package directory:

    python franka_sim/test/debug_mujoco_viewer.py

Each probe runs in a fresh Python process so a segfault in one stage does not
stop the full diagnosis.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import platform
import shutil
import signal
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Probe:
    name: str
    code: str
    timeout: float = 15.0
    env: dict[str, str | None] | None = None


def _clean_code(code: str) -> str:
    return textwrap.dedent(code).strip() + "\n"


def _merged_env(overrides: dict[str, str | None] | None) -> dict[str, str]:
    env = os.environ.copy()
    if overrides:
        for key, value in overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
    return env


def _returncode_label(returncode: int) -> str:
    if returncode < 0:
        signum = -returncode
        try:
            signame = signal.Signals(signum).name
        except ValueError:
            signame = f"SIG{signum}"
        return f"{returncode} ({signame})"
    if returncode >= 128:
        signum = returncode - 128
        try:
            signame = signal.Signals(signum).name
        except ValueError:
            signame = f"SIG{signum}"
        return f"{returncode} (shell signal {signame})"
    return str(returncode)


def _run_command(label: str, args: list[str], timeout: float = 8.0) -> None:
    print(f"\n$ {' '.join(args)}")
    try:
        proc = subprocess.run(
            args,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        print(f"{label}: command not found")
        return
    except subprocess.TimeoutExpired as exc:
        print(f"{label}: timed out after {timeout}s")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr)
        return

    print(f"{label}: exit {_returncode_label(proc.returncode)}")
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())


def _run_probe(probe: Probe) -> bool:
    print("\n" + "=" * 88)
    print(f"PROBE: {probe.name}")
    print("-" * 88)
    env = _merged_env(probe.env)
    shown_env = {
        key: env.get(key)
        for key in (
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "MUJOCO_GL",
            "PYOPENGL_PLATFORM",
            "LD_LIBRARY_PATH",
            "CONDA_PREFIX",
        )
    }
    print(f"env: {shown_env}")
    try:
        proc = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", _clean_code(probe.code)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=probe.timeout,
        )
    except subprocess.TimeoutExpired as exc:
        print(f"RESULT: TIMEOUT after {probe.timeout}s")
        if exc.stdout:
            print("\n[stdout]\n" + exc.stdout.rstrip())
        if exc.stderr:
            print("\n[stderr]\n" + exc.stderr.rstrip())
        return False

    print(f"RESULT: exit {_returncode_label(proc.returncode)}")
    if proc.stdout:
        print("\n[stdout]\n" + proc.stdout.rstrip())
    if proc.stderr:
        print("\n[stderr]\n" + proc.stderr.rstrip())
    return proc.returncode == 0


def _print_header() -> None:
    print("MuJoCo viewer crash diagnosis")
    print(f"time: {_dt.datetime.now().isoformat(timespec='seconds')}")
    print(f"script: {Path(__file__).resolve()}")
    print(f"cwd: {Path.cwd()}")
    print(f"root: {ROOT}")
    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.replace(os.linesep, ' ')}")
    print(f"platform: {platform.platform()}")
    print(f"uname: {platform.uname()}")
    for key in (
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "LD_LIBRARY_PATH",
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "XDG_SESSION_TYPE",
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        print(f"{key}: {os.environ.get(key)}")


def _print_system_checks() -> None:
    print("\n" + "=" * 88)
    print("SYSTEM CHECKS")
    print("-" * 88)
    checks = [
        ("nvidia-smi", ["nvidia-smi"]),
        ("xdpyinfo", ["xdpyinfo", "-display", os.environ.get("DISPLAY", "")]),
        ("glxinfo", ["glxinfo", "-B"]),
        ("ldconfig libGL", ["bash", "-lc", "ldconfig -p | grep -E 'libGL|libEGL|libglfw'"]),
    ]
    for label, args in checks:
        if args[0] != "bash" and shutil.which(args[0]) is None:
            print(f"\n{label}: command not found")
            continue
        if "" in args:
            print(f"\n{label}: skipped because DISPLAY is empty")
            continue
        _run_command(label, args)


def _probes() -> list[Probe]:
    base_imports = """
        import os
        import sys
        print('python', sys.executable)
        print('DISPLAY', os.environ.get('DISPLAY'))
        print('MUJOCO_GL', os.environ.get('MUJOCO_GL'))
    """
    import_probe = (
        base_imports
        + """
        import mujoco
        import mujoco.viewer
        import glfw
        print('mujoco', mujoco.__version__)
        print('glfw module', glfw.__file__)
        """
    )
    env_construct = """
        from franka_sim import envs
        env = envs.PandaPickCubeGymEnv(render_mode='human', action_scale=(0.1, 1))
        print('env constructed')
        print('viewer initially', env._viewer)
        obs, info = env.reset()
        print('reset ok', sorted(obs.keys()))
        env.close()
        print('closed ok')
    """
    render = """
        from franka_sim import envs
        env = envs.PandaPickCubeGymEnv(render_mode='human', action_scale=(0.1, 1))
        env.reset()
        frames = env.render()
        print('render shapes', [frame.shape for frame in frames])
        env.close()
        print('closed ok')
    """
    one_viewer = """
        import time
        import mujoco.viewer
        from franka_sim import envs
        env = envs.PandaPickCubeGymEnv(render_mode='human', action_scale=(0.1, 1))
        env.reset()
        print('launching one passive viewer')
        viewer = mujoco.viewer.launch_passive(
            env.model,
            env.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        print('is_running', viewer.is_running())
        for i in range(20):
            env.step(env.action_space.sample())
            viewer.sync()
            if i in (0, 1, 19):
                print('sync', i)
            time.sleep(0.01)
        viewer.close()
        env.close()
        print('closed ok')
    """
    one_viewer_no_close = """
        import os
        import time
        import mujoco.viewer
        from franka_sim import envs
        env = envs.PandaPickCubeGymEnv(render_mode='human', action_scale=(0.1, 1))
        env.reset()
        print('launching one passive viewer')
        viewer = mujoco.viewer.launch_passive(
            env.model,
            env.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        print('is_running', viewer.is_running())
        for i in range(20):
            env.step(env.action_space.sample())
            viewer.sync()
            if i in (0, 1, 19):
                print('sync', i)
            time.sleep(0.01)
        print('exiting without viewer.close via os._exit(0)')
        os._exit(0)
    """
    one_viewer_slow_close = one_viewer.replace(
        "viewer.close()",
        "time.sleep(2.0)\n        viewer.close()",
    )
    dual_viewer = """
        import time
        from franka_sim import envs
        from franka_sim.utils.viewer_utils import DualMujocoViewer
        env = envs.PandaPickCubeGymEnv(render_mode='human', action_scale=(0.1, 1))
        env.reset()
        print('launching dual passive viewer')
        with DualMujocoViewer(env.model, env.data) as viewer:
            print('is_running', viewer.is_running())
            for i in range(20):
                env.step(env.action_space.sample())
                viewer.sync()
                if i in (0, 1, 19):
                    print('sync', i)
                time.sleep(0.01)
        env.close()
        print('closed ok')
    """
    long_dual_viewer = dual_viewer.replace("range(20)", "range(400)").replace(
        "time.sleep(0.01)", "time.sleep(0.002)"
    ).replace(
        "if i in (0, 1, 19):", "if i in (0, 1, 399):"
    )
    original_like = """
        import time
        import numpy as np
        from franka_sim import envs
        from franka_sim.utils.viewer_utils import DualMujocoViewer
        env = envs.PandaPickCubeGymEnv(render_mode='human', action_scale=(0.1, 1))
        action_spec = env.action_space
        def sample():
            action = np.random.uniform(action_spec.low, action_spec.high, action_spec.shape)
            return action.astype(action_spec.dtype)
        env.reset()
        with DualMujocoViewer(env.unwrapped.model, env.unwrapped.data) as viewer:
            for i in range(600):
                if not viewer.is_running():
                    print('viewer stopped at', i)
                    break
                step_start = time.time()
                env.step(sample())
                viewer.sync()
                if i in (0, 1, 599):
                    print('loop', i)
                delay = env.control_dt - (time.time() - step_start)
                if delay > 0:
                    time.sleep(delay)
        env.close()
        print('closed ok')
    """

    return [
        Probe("import mujoco / glfw", import_probe),
        Probe("construct env only", env_construct),
        Probe("offscreen render with current GL backend", render),
        Probe("offscreen render with MUJOCO_GL=egl", render, env={"MUJOCO_GL": "egl"}),
        Probe("offscreen render with MUJOCO_GL=glfw", render, env={"MUJOCO_GL": "glfw"}),
        Probe("offscreen render with DISPLAY unset", render, env={"DISPLAY": None}),
        Probe("one passive viewer", one_viewer, timeout=20.0),
        Probe("one passive viewer without close", one_viewer_no_close, timeout=20.0),
        Probe("one passive viewer slow close", one_viewer_slow_close, timeout=25.0),
        Probe("one passive viewer with MUJOCO_GL=egl", one_viewer, timeout=20.0, env={"MUJOCO_GL": "egl"}),
        Probe("one passive viewer with MUJOCO_GL=glfw", one_viewer, timeout=20.0, env={"MUJOCO_GL": "glfw"}),
        Probe("dual passive viewer", dual_viewer, timeout=20.0),
        Probe("dual passive viewer longer stress", long_dual_viewer, timeout=30.0),
        Probe("original human loop bounded", original_like, timeout=35.0),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-system",
        action="store_true",
        help="Skip nvidia-smi, xdpyinfo, glxinfo, and ldconfig checks.",
    )
    parser.add_argument(
        "--only",
        help="Run only probes whose names contain this case-insensitive text.",
    )
    args = parser.parse_args()

    _print_header()
    if not args.skip_system:
        _print_system_checks()

    probes = _probes()
    if args.only:
        needle = args.only.lower()
        probes = [probe for probe in probes if needle in probe.name.lower()]

    failed: list[str] = []
    for probe in probes:
        if not _run_probe(probe):
            failed.append(probe.name)

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("-" * 88)
    if not failed:
        print("All probes completed without a subprocess crash or timeout.")
        return 0

    print("Failed probes:")
    for name in failed:
        print(f"- {name}")
    print("\nIf RESULT is exit -11 or 139, that probe hit SIGSEGV.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
