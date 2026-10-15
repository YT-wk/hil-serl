import time
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import copy
from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert, JoystickExpert, ControllerType
import requests
from scipy.spatial.transform import Rotation as R
from typing import List
import inputs
import threading
from dataclasses import dataclass
from enum import Enum

sigmoid = lambda x: 1 / (1 + np.exp(-x))

class HumanClassifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if done:
            while True:
                try:
                    rew = int(input("Success? (1/0)"))
                    assert rew == 0 or rew == 1
                    break
                except:
                    continue
        info['succeed'] = rew
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz = None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or rew
        info['succeed'] = bool(rew)
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info
    
    
class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)
    
    def compute_reward(self, obs):
        rewards = [0] * len(self.reward_classifier_func)
        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue

            logit = classifier_func(obs).item()
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1

        reward = sum(rewards)
        return reward

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = (done or all(self.received)) # either environment done or all rewards satisfied
        info['succeed'] = all(self.received)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info['succeed'] = False
        return obs, info


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation


class Quat2R2Wrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotation matrix
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(9,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
        assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )
        self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["left/tcp_pose"]
        observation["state"]["left/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        tcp_pose = observation["state"]["right/tcp_pose"]
        observation["state"]["right/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = SpaceMouseExpert()
        self.left, self.right = False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)
        intervened = False
        
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.gripper_enabled:
            if self.left:  # close gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:  # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)
            expert_a[:6] += np.random.uniform(-0.5, 0.5, size=6)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            return expert_a, True

        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

class DualSpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None, gripper_enabled=True):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled

        self.expert = SpaceMouseExpert()
        self.left1, self.left2, self.right1, self.right2 = False, False, False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        intervened = False
        expert_a, buttons = self.expert.get_action()
        self.left1, self.left2, self.right1, self.right2 = tuple(buttons)


        if self.gripper_enabled:
            if self.left1:  # close gripper
                left_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.left2:  # open gripper
                left_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                left_gripper_action = np.zeros((1,))

            if self.right1:  # close gripper
                right_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right2:  # open gripper
                right_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                right_gripper_action = np.zeros((1,))
            expert_a = np.concatenate(
                (expert_a[:6], left_gripper_action, expert_a[6:], right_gripper_action),
                axis=0,
            )

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left1"] = self.left1
        info["left2"] = self.left2
        info["right1"] = self.right1
        info["right2"] = self.right2
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < 0.95
        ):
            return reward - self.penalty, self.penalty
        else:
            return reward, 0.0

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward, penalty = self.reward(reward, action)
        info["grasp_penalty"] = penalty
        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info

class DualGripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_gripper_pos_left = 0 #TODO: this assume gripper starts opened
        self.last_gripper_pos_right = 0 #TODO: this assume gripper starts opened
    
    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos_left==0):
            reward -= self.penalty
            self.last_gripper_pos_left = 1
        elif (action[6] > 0.5 and self.last_gripper_pos_left==1):
            reward -= self.penalty
            self.last_gripper_pos_left = 0
        if (action[13] < -0.5 and self.last_gripper_pos_right==0):
            reward -= self.penalty
            self.last_gripper_pos_right = 1
        elif (action[13] > 0.5 and self.last_gripper_pos_right==1):
            reward -= self.penalty
            self.last_gripper_pos_right = 0
        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        return observation, reward, terminated, truncated, info

class JoystickIntervention(gym.ActionWrapper):
    def __init__(
        self,
        env,
        action_indices=None,
        controller_type=ControllerType.XBOX,
        latch_intervention=False,
        intervention_toggle=False,
    ):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.action_indices = action_indices
        self.latch_intervention = latch_intervention
        self.intervention_toggle = intervention_toggle
        self.intervention_latched = False
        self._toggle_pressed = False

        self.expert = JoystickExpert(controller_type=controller_type)
        self.left, self.right = False, False

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: joystick action if nonezero; else, policy action
        """
        deadzone = 0.03

        if self.intervention_toggle:
            toggle_pressed = self.expert.get_toggle()
            if toggle_pressed and not self._toggle_pressed:
                self.intervention_latched = not self.intervention_latched
                mode = "intervention" if self.intervention_latched else "autonomous"
                print(f"[joystick] switched to {mode} mode", flush=True)
            self._toggle_pressed = toggle_pressed

        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)
        intervened = False

        if np.linalg.norm(expert_a) > deadzone:
            intervened = True

        if self.gripper_enabled:
            if self.left: # close gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right: # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)
            # expert_a[:6] += np.random.uniform(-0.5, 0.5, size=6)

        if self.action_indices is not None:
            filtred_expert_a = np.zeros_like(expert_a)
            filtred_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtred_expert_a

        if intervened:
            if self.latch_intervention and not self.intervention_latched:
                self.intervention_latched = True
                print("[joystick] intervention latched until success", flush=True)
            return expert_a, True

        if self.intervention_latched:
            return np.zeros_like(action), True

        return action, False
    
    def step(self, action):
        
        new_action, replaced = self.action(action)
        
        obs, rew, done, truncated, info = self.env.step(new_action)
        if self.intervention_latched and info.get("succeed", False):
            self.intervention_latched = False
            print("[joystick] success; policy control resumed", flush=True)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        self.intervention_latched = False
        self._toggle_pressed = False
        return self.env.reset(**kwargs)
    
    def close(self):
        self.expert.close()
        super().close()


class KeyboardIntervention(gym.ActionWrapper):
    def __init__(
        self,
        env,
        action_indices=None,
        pos_speed=0.2,
        gripper_speed=1.0,
        key_timeout=0.25,
        latch_intervention=False,
        intervention_toggle=False,
    ):
        super().__init__(env)

        self.gripper_enabled = self.action_space.shape != (6,)
        self.action_indices = action_indices
        self.latch_intervention = latch_intervention
        self.intervention_toggle = intervention_toggle
        self.intervention_latched = False
        self.pos_speed = float(pos_speed)
        self.gripper_speed = float(gripper_speed)
        self.key_timeout = float(key_timeout)
        self._pressed_keys = set()
        self._active_until = {}
        self._lock = threading.Lock()
        self._listener = None
        self._success_callback = None
        self._camera_callback = None
        self._gripper_closed = False
        self._gripper_command = 0.0
        self._gripper_command_until = 0.0
        self._toggle_requested = False

    def start_listening(self, success_callback=None, camera_callback=None):
        if self._listener is not None:
            return self._listener

        try:
            from pynput import keyboard
        except ImportError as exc:
            raise ImportError(
                "Continuous keyboard control requires pynput and a working "
                "desktop/X11 session. Run this from the noVNC/desktop terminal "
                "with a valid DISPLAY, or use a gamepad controller."
            ) from exc

        self._success_callback = success_callback
        self._camera_callback = camera_callback
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        return self._listener

    def key_callback(self, keycode):
        key = self._normalize_key(keycode)
        if key is None:
            return
        if key in {"a", "d", "w", "s", "q", "z", "r", "f"}:
            with self._lock:
                self._active_until[key] = time.time() + self.key_timeout

    def action(self, action: np.ndarray) -> np.ndarray:
        if self.intervention_toggle:
            with self._lock:
                toggle_requested = self._toggle_requested
                self._toggle_requested = False
            if toggle_requested:
                self.intervention_latched = not self.intervention_latched
                mode = "intervention" if self.intervention_latched else "autonomous"
                print(f"[keyboard] switched to {mode} mode", flush=True)

        expert_a = np.zeros(self.action_space.shape, dtype=np.float32)
        keys = self._active_keys()

        if expert_a.shape[0] >= 3:
            expert_a[0] = self.pos_speed * self._axis(keys, positive="w", negative="s")
            expert_a[1] = self.pos_speed * self._axis(keys, positive="a", negative="d")
            expert_a[2] = self.pos_speed * (
                self._axis(keys, positive="q", negative="z")
                + self._axis(keys, positive="r", negative="f")
            )

        if self.gripper_enabled and expert_a.shape[0] >= 7:
            with self._lock:
                if time.time() <= self._gripper_command_until:
                    expert_a[6] = self.gripper_speed * self._gripper_command

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if np.linalg.norm(expert_a) > 0.001:
            if self.latch_intervention and not self.intervention_latched:
                self.intervention_latched = True
                print("[keyboard] intervention latched until success", flush=True)
            return expert_a, True

        if self.intervention_latched:
            return np.zeros_like(action), True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if self.intervention_latched and info.get("succeed", False):
            self.intervention_latched = False
            print("[keyboard] success; policy control resumed", flush=True)
        if replaced:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        self.intervention_latched = False
        with self._lock:
            self._toggle_requested = False
        return self.env.reset(**kwargs)

    def close(self):
        with self._lock:
            self._pressed_keys.clear()
            self._active_until.clear()
            self._gripper_command = 0.0
            self._gripper_command_until = 0.0
            self._toggle_requested = False
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        super().close()

    def _active_keys(self):
        now = time.time()
        with self._lock:
            expired = [key for key, deadline in self._active_until.items() if deadline < now]
            for key in expired:
                del self._active_until[key]
            return set(self._pressed_keys) | set(self._active_until)

    def _on_press(self, key):
        normalized = self._normalize_key(key)
        if normalized is None:
            return
        if normalized in {"a", "d", "w", "s", "q", "z", "r", "f"}:
            with self._lock:
                self._pressed_keys.add(normalized)
        elif normalized == "x":
            with self._lock:
                if normalized in self._pressed_keys:
                    return
                self._pressed_keys.add(normalized)
                self._gripper_closed = not self._gripper_closed
                self._gripper_command = 1.0 if self._gripper_closed else -1.0
                self._gripper_command_until = time.time() + self.key_timeout
                state = "closed" if self._gripper_closed else "open"
            print(f"[keyboard] gripper={state}", flush=True)
        elif normalized == "space" and self.intervention_toggle:
            with self._lock:
                if normalized in self._pressed_keys:
                    return
                self._pressed_keys.add(normalized)
                self._toggle_requested = True
        elif normalized == "enter" and self._success_callback is not None:
            self._success_callback("keyboard")
        elif normalized in {"[", "]"} and self._camera_callback is not None:
            self._camera_callback(-1 if normalized == "[" else 1)

    def _on_release(self, key):
        normalized = self._normalize_key(key)
        if normalized is None:
            return
        with self._lock:
            self._pressed_keys.discard(normalized)

    @staticmethod
    def _axis(keys, positive, negative):
        return float(positive in keys) - float(negative in keys)

    @staticmethod
    def _normalize_key(keycode):
        if isinstance(keycode, str):
            return keycode.lower()
        name = getattr(keycode, "name", None)
        if name is not None:
            return name.lower()
        char = getattr(keycode, "char", None)
        if char is not None:
            return char.lower()
        try:
            return chr(keycode).lower()
        except (TypeError, ValueError):
            return None
