import copy
import os
from pathlib import Path
import sys

from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
import time
import threading
from absl import app, flags

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in ("serl_robot_infra", "franka_sim", "examples"):
    sys.path.insert(0, str(_REPO_ROOT / _path))

from experiments.mappings import CONFIG_MAPPING
from franka_env.envs.wrappers import KeyboardIntervention

from franka_sim.utils.viewer_utils import DualMujocoViewer

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")
flags.DEFINE_integer("log_every", 500, "Print collection status every N failure transitions.")
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


success_key_count = 0
success_key_last_time = 0.0
success_key_lock = threading.Lock()


def request_success(source):
    global success_key_count, success_key_last_time
    now = time.time()
    with success_key_lock:
        if now - success_key_last_time < 0.15:
            return
        success_key_count += 1
        success_key_last_time = now
        pending = success_key_count
    print(f"\n[input] success mark received from {source}; pending={pending}", flush=True)


def consume_success_request():
    global success_key_count
    with success_key_lock:
        if success_key_count <= 0:
            return False
        success_key_count -= 1
        return True


def start_terminal_success_listener():
    if not sys.stdin or not sys.stdin.isatty():
        return

    def listen_for_enter():
        while True:
            line = sys.stdin.readline()
            if line == "":
                break
            request_success("terminal")

    thread = threading.Thread(target=listen_for_enter, daemon=True)
    thread.start()


def save_classifier_data(exp_name, successes, failures):
    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    success_count = len(successes)
    success_file_name = (
        f"./classifier_data/{exp_name}_{success_count}_success_images_{uuid}.pkl"
    )
    with open(success_file_name, "wb") as f:
        pkl.dump(successes, f)
        print(f"saved {success_count} successful transitions to {success_file_name}")

    failure_file_name = f"./classifier_data/{exp_name}_failure_images_{uuid}.pkl"
    with open(failure_file_name, "wb") as f:
        pkl.dump(failures, f)
        print(f"saved {len(failures)} failure transitions to {failure_file_name}")


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

    obs, _ = env.reset()
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    total_steps = 0
    episode_idx = 1
    episode_steps = 0
    success_episode_count = 0
    failure_segment = []
    success_segment = []
    start_terminal_success_listener()
    
    # Create the dual viewer
    keyboard_intervention = find_wrapper(env, KeyboardIntervention)
    dual_viewer = DualMujocoViewer(
        env.unwrapped.model,
        env.unwrapped.data,
        num_viewers=1,
    )
    if keyboard_intervention is not None:
        keyboard_intervention.start_listening(
            success_callback=request_success,
            camera_callback=dual_viewer.cycle_camera,
        )

    print(
        "Press Enter after the success condition is reached to record the "
        "successful segment and reset to the next episode.\n"
        "Keyboard controls: A/D left/right, W/S forward/back, Q/R up, Z/F down, "
        "X toggle gripper, [/] switch camera. If using keyboard, keep "
        "keyboard focus on the terminal while watching the viewer to avoid "
        "MuJoCo viewer shortcuts changing the rendering."
    )
    try:
        with dual_viewer as viewer:
            while viewer.is_running():
                actions = np.zeros(env.action_space.sample().shape)
                next_obs, rew, done, truncated, info = env.step(actions)
                viewer.sync()
                if "intervene_action" in info:
                    actions = info["intervene_action"]

                total_steps += 1
                episode_steps += 1
                manual_success = consume_success_request()
                env_success = info["succeed"]
                terminal_done = truncated or (done and not env_success)
                confirmed_success = manual_success and env_success
                record_done = confirmed_success or terminal_done

                transition = copy.deepcopy(
                    dict(
                        observations=obs,
                        actions=actions,
                        next_observations=next_obs,
                        rewards=rew,
                        masks=1.0 - record_done,
                        dones=record_done,
                    )
                )
                obs = next_obs

                if env_success:
                    success_segment.append(transition)
                else:
                    failure_segment.append(transition)

                if manual_success and not env_success:
                    print(
                        f"[ignored] enter received but no transition has satisfied "
                        f"success condition at the confirmation step in "
                        f"episode={episode_idx}; episode_step={episode_steps} "
                        f"total_step={total_steps} pending_success_condition="
                        f"{len(success_segment)}",
                        flush=True,
                    )

                if confirmed_success:
                    success_episode_count += 1
                    remaining = max(0, success_needed - len(successes))
                    success_to_store = success_segment[-remaining:] if remaining else []
                    failures.extend(copy.deepcopy(failure_segment))
                    successes.extend(copy.deepcopy(success_to_store))
                    pbar.update(len(success_to_store))
                    print(
                        f"[success] episode_successes={success_episode_count} "
                        f"success_transitions={len(successes)}/{success_needed} "
                        f"episode={episode_idx} episode_step={episode_steps} "
                        f"total_step={total_steps} "
                        f"stored_success_transitions={len(success_to_store)} "
                        f"success_condition_transitions={len(success_segment)} "
                        f"stored_failure_prefix={len(failure_segment)} "
                        f"reward={rew:.3f} env_done={done} truncated={truncated}",
                        flush=True,
                    )

                if FLAGS.log_every > 0 and total_steps % FLAGS.log_every == 0:
                    print(
                        f"[status] success_transitions={len(successes)}/{success_needed} "
                        f"failures={len(failures)} pending_failures={len(failure_segment)} "
                        f"pending_success_condition={len(success_segment)} "
                        f"episode={episode_idx} episode_step={episode_steps} "
                        f"total_step={total_steps}",
                        flush=True,
                    )

                if confirmed_success:
                    print(
                        f"[reset] reason=manual_confirmed_success "
                        f"finished_episode={episode_idx} episode_steps={episode_steps} "
                        f"success_transitions={len(successes)} failures={len(failures)}",
                        flush=True,
                    )
                    failure_segment = []
                    success_segment = []
                    obs, _ = env.reset()
                    episode_idx += 1
                    episode_steps = 0
                elif terminal_done:
                    failures.extend(copy.deepcopy(failure_segment))
                    if success_segment:
                        print(
                            f"[warning] episode ended before enter; discarded "
                            f"{len(success_segment)} success-condition transitions",
                            flush=True,
                        )
                    print(
                        f"[failure] reason={'truncated' if truncated else 'done'} "
                        f"stored_failed_transitions={len(failure_segment)} "
                        f"total_failures={len(failures)} episode={episode_idx}",
                        flush=True,
                    )
                    failure_segment = []
                    success_segment = []
                    obs, _ = env.reset()
                    episode_idx += 1
                    episode_steps = 0
                if len(successes) >= success_needed:
                    break
    except KeyboardInterrupt:
        print("\nInterrupted; saving collected transitions before exit.")
    finally:
        pending_count = len(failure_segment) + len(success_segment)
        if pending_count:
            failures.extend(copy.deepcopy(failure_segment))
            print(
                f"[save] stored unfinished non-success transitions as failures: "
                f"{len(failure_segment)}; discarded unconfirmed success-condition "
                f"transitions: {len(success_segment)}; total_failures={len(failures)}",
                flush=True,
            )
        pbar.close()
        env.close()
        save_classifier_data(FLAGS.exp_name, successes, failures)
        
if __name__ == "__main__":
    app.run(main)
