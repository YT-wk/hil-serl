import os
from pathlib import Path
import sys

from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in ("serl_robot_infra", "franka_sim", "examples"):
    sys.path.insert(0, str(_REPO_ROOT / _path))

from experiments.mappings import CONFIG_MAPPING
from franka_env.envs.wrappers import KeyboardIntervention
from franka_sim.utils.viewer_utils import DualMujocoViewer

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "pick_cube_sim", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_enum(
    "controller",
    "xbox",
    ["keyboard", "xbox", "ps5"],
    "Controller device for human intervention.",
)
flags.DEFINE_float(
    "keyboard_pos_speed",
    0.2,
    "Normalized xyz action magnitude for keyboard control.",
)
if hasattr(flags, "DEFINE_alias"):
    try:
        flags.DEFINE_alias("keyboard-pos-speed", "keyboard_pos_speed")
    except Exception:
        pass


def save_demo_data(exp_name, transitions, success_count):
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{exp_name}_{success_count}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(
            f"saved {success_count} demos "
            f"({len(transitions)} transitions) to {file_name}"
        )


def find_wrapper(env, wrapper_type):
    current = env
    while current is not None:
        if isinstance(current, wrapper_type):
            return current
        current = getattr(current, "env", None)
    return None


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
        controller_type=FLAGS.controller,
        keyboard_pos_speed=FLAGS.keyboard_pos_speed,
    )

    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    # Create the dual viewer
    keyboard_intervention = find_wrapper(env, KeyboardIntervention)
    dual_viewer = DualMujocoViewer(
        env.unwrapped.model,
        env.unwrapped.data,
        num_viewers=1,
    )
    if keyboard_intervention is not None:
        keyboard_intervention.start_listening(camera_callback=dual_viewer.cycle_camera)

    print(
        "Use the controller to record demos. Keyboard controls: A/D left/right, "
        "W/S forward/back, Q/R up, Z/F down, X toggle gripper, [/] switch camera. "
        "If using keyboard, keep keyboard focus on the terminal while watching "
        "the viewer to avoid MuJoCo viewer shortcuts changing the rendering. "
        "Close the viewer or press Ctrl+C to save completed successful demos."
    )
    try:
        with dual_viewer as viewer:
            while viewer.is_running():
                actions = np.zeros(env.action_space.sample().shape)
                next_obs, rew, done, truncated, info = env.step(actions)
                viewer.sync()
                returns += rew
                if "intervene_action" in info:
                    actions = info["intervene_action"]
                episode_done = done or truncated
                transition = copy.deepcopy(
                    dict(
                        observations=obs,
                        actions=actions,
                        next_observations=next_obs,
                        rewards=rew,
                        masks=1.0 - episode_done,
                        dones=episode_done,
                        infos=info,
                    )
                )
                trajectory.append(transition)

                pbar.set_description(f"Return: {returns}")

                obs = next_obs
                if episode_done:
                    if info["succeed"]:
                        transitions.extend(copy.deepcopy(trajectory))
                        success_count += 1
                        print(
                            f"Success count: {success_count}/{success_needed}; "
                            f"saved trajectory transitions: {len(trajectory)}"
                        )
                        pbar.update(1)
                    else:
                        print(
                            f"Episode ended without success "
                            f"(done={done}, truncated={truncated}); "
                            f"discarded trajectory transitions: {len(trajectory)}"
                        )
                    trajectory = []
                    returns = 0
                    obs, info = env.reset()
                if success_count >= success_needed:
                    break
    except KeyboardInterrupt:
        print("\nInterrupted; saving completed successful demos before exit.")
    finally:
        if trajectory:
            print(
                f"Discarding unfinished trajectory with {len(trajectory)} transitions; "
                "only completed successful demos are saved."
            )
        pbar.close()
        env.close()
        save_demo_data(FLAGS.exp_name, transitions, success_count)

if __name__ == "__main__":
    app.run(main)
