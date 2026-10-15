import os
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    JoystickIntervention,
    KeyboardIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
    ControllerType
)

from experiments.pick_cube_sim.wrapper import GripperPenaltyWrapper

from franka_env.envs.relative_env import RelativeFrame
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig
from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv


class TrainConfig(DefaultTrainingConfig):
    controller_type = ControllerType.XBOX
    image_keys = ["front", "wrist"]
    classifier_keys = ["front", "wrist"]
    proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose"]
    buffer_period = 2000
    replay_buffer_capacity = 50000
    batch_size = 256
    random_steps = 0
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    fake_env = False
    classifier = False

    def get_environment(
        self,
        fake_env=False,
        save_video=False,
        classifier=False,
        controller_type=None,
        keyboard_pos_speed=0.2,
        intervention_latch=False,
        intervention_toggle=False,
    ):
        env = PandaPickCubeGymEnv(render_mode="human", image_obs=True, reward_type="sparse", time_limit=100.0, control_dt=0.1)
        if not fake_env:
            controller_type = self._normalize_controller_type(
                controller_type or self.controller_type
            )
            if controller_type == ControllerType.KEYBOARD:
                env = KeyboardIntervention(
                    env=env,
                    pos_speed=keyboard_pos_speed,
                    latch_intervention=intervention_latch,
                    intervention_toggle=intervention_toggle,
                )
            else:
                env = JoystickIntervention(
                    env=env,
                    controller_type=controller_type,
                    latch_intervention=intervention_latch,
                    intervention_toggle=intervention_toggle,
                )
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            from serl_launcher.networks.reward_classifier import load_classifier_func

            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return int(sigmoid(classifier(obs))[0] > 0.95)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env

    @staticmethod
    def _normalize_controller_type(controller_type):
        if isinstance(controller_type, ControllerType):
            return controller_type
        return ControllerType(str(controller_type).lower())
