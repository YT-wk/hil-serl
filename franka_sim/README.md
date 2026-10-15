# Franka simulation

This package provides MuJoCo environments for a Franka arm with a Robotiq gripper, including state and image observations for the cube pickup task. The commands below use Bash on Linux. Run them from a graphical session when a MuJoCo or OpenCV window is needed.

## Setup

Choose the repository location and an existing Conda environment. Set these variables in **each terminal** used below:

```bash
cd /path/to/hil-serl
REPO_ROOT="$PWD"
CONDA_ENV=hilserl
conda activate "$CONDA_ENV"
```

Install the simulation package and its dependencies:

```bash
cd "$REPO_ROOT/franka_sim"
pip install -e .
pip install -r requirements.txt
```

Return to the repository root before running commands under `examples/`:

```bash
cd "$REPO_ROOT"
```

## Test the environment and controller

The controller test opens one MuJoCo viewer by default. Use `keyboard`, `xbox`, or `ps5`:

```bash
cd "$REPO_ROOT/franka_sim"
unset MUJOCO_GL
python franka_sim/test/test_gym_env_joystick.py --controller=xbox
```

For keyboard control, use `A/D` for left/right, `W/S` for forward/back, `Q/R` for up, `Z/F` for down, `X` to toggle the gripper, and `[`/`]` to switch cameras. Keep keyboard focus on the terminal while watching the viewer. Adjust movement speed with `--keyboard-pos-speed=0.2`.

To try two viewer windows, add `--num-viewers=2`. One window is the safer default on systems where a second GLFW window is unstable. The viewer requires a working graphical display; headless EGL rendering does not provide a MuJoCo viewer window.

## Collect data

Run both commands from the repository root. Choose `--controller keyboard` or `--controller ps5` if you are not using an Xbox controller.

Collect success and failure image transitions for the reward classifier:

```bash
cd "$REPO_ROOT"
python examples/record_success_fail_sim.py \
  --exp_name pick_cube_sim \
  --successes_needed 2000 \
  --controller xbox
```

The command writes a matching pair of `*.pkl` files to `classifier_data/`: one for successful transitions and one for failures. `--successes_needed` counts successful transitions, not complete episodes. Keep the two files from the same collection run together.

Collect successful demonstrations for RLPD:

```bash
python examples/record_demos_sim.py \
  --exp_name pick_cube_sim \
  --successes_needed 30 \
  --controller xbox
```

This writes a demonstration `*.pkl` file to `demo_data/`.

## Train the reward classifier

Replace the filenames below with the matching files produced by data collection. The run directory can be any writable location; this example keeps it under the repository.

```bash
cd "$REPO_ROOT"
SUCCESS_PKL="$REPO_ROOT/classifier_data/pick_cube_sim_<count>_success_images_<timestamp>.pkl"
FAILURE_PKL="$REPO_ROOT/classifier_data/pick_cube_sim_failure_images_<timestamp>.pkl"
CLASSIFIER_RUN="$REPO_ROOT/classifier_runs/pick_cube"

bash examples/train_reward_classifier_from_files.sh \
  --success "$SUCCESS_PKL" \
  --failure "$FAILURE_PKL" \
  --epochs 150 \
  --batch-size 32 \
  --run-dir "$CLASSIFIER_RUN" \
  --env "$CONDA_ENV"
```

The trained checkpoint is saved in `$CLASSIFIER_RUN/classifier_ckpt/`. The script disables Weights & Biases logging by default; pass `--wandb` to enable it.

## Train RLPD

Use the same policy checkpoint directory for the learner and actor. The demonstration file comes from `demo_data/`, and the classifier checkpoint comes from the previous step. Set these variables in **both terminals**, after the setup commands above:

```bash
DEMO_PKL="$REPO_ROOT/demo_data/pick_cube_sim_<count>_demos_<timestamp>.pkl"
CLASSIFIER_CKPT="$REPO_ROOT/classifier_runs/pick_cube/classifier_ckpt"
POLICY_CKPT="$REPO_ROOT/checkpoints/pick_cube_rlpd"
cd "$REPO_ROOT/examples/experiments/pick_cube_sim"
```

Terminal 1, learner:

```bash
bash run_learner.sh \
  --checkpoint_path "$POLICY_CKPT" \
  --demo_path "$DEMO_PKL" \
  --classifier_ckpt "$CLASSIFIER_CKPT" \
  --conda-env "$CONDA_ENV" \
  --debug
```

Terminal 2, actor (run after the learner starts):

```bash
bash run_actor.sh \
  --checkpoint_path "$POLICY_CKPT" \
  --classifier_ckpt "$CLASSIFIER_CKPT" \
  --conda-env "$CONDA_ENV" \
  --ip localhost \
  --controller xbox \
  --debug \
  --opencv-viewer \
  --intervention-latch \
  --intervention-toggle
```

The OpenCV viewer shows the front and wrist camera images while MuJoCo renders offscreen with EGL. Use `--no-viewer` for an unattended run, or `--viewer --num_viewers 1` for the native MuJoCo GLFW viewer. The GLFW mode may be less stable on some display setups. With intervention toggle enabled, use the gamepad Start button or keyboard Space to switch modes. `--debug` disables Weights & Biases logging for these scripts.

## Evaluate a checkpoint

Run evaluation after the policy checkpoint directory contains `checkpoint_*` subdirectories. The example evaluates the latest checkpoint for 50 episodes without opening a viewer:

```bash
cd "$REPO_ROOT/examples/experiments/pick_cube_sim"
bash run_eval.sh \
  --checkpoint_path "$POLICY_CKPT" \
  --classifier_ckpt "$CLASSIFIER_CKPT" \
  --conda-env "$CONDA_ENV" \
  --eval_n_trajs 50 \
  --no-viewer \
  --timeout-sec 600
```

Use `--step <number>` to select a specific checkpoint, or `--viewer --num-viewers 1` to watch evaluation from a graphical session. `--dry-run` prints the resolved evaluation command without starting it.

## Credits

The original simulation was built by [Kevin Zakka](https://kzakka.com/). The Gymnasium environment was adapted with his permission.
