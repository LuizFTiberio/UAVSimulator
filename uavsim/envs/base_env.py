"""Base Gymnasium environment for UAV simulation."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from uavsim.sim.mujoco_sim import MuJoCoSimulator
from uavsim.vehicles.base import VehicleModel


class BaseUAVEnv(gym.Env):
    """Base environment wrapping MuJoCoSimulator for RL.

    Subclasses define the task (reward, termination, observation shaping).

    Observation (default): [position(3), velocity(3), quaternion(4), angular_velocity(3)] = 13.
    Action: motor commands [0, 1]^n_motors.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        vehicle: VehicleModel,
        max_episode_steps: int = 5000,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.vehicle = vehicle
        self.sim = MuJoCoSimulator(vehicle)
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._step_count = 0

        n_motors = vehicle.params.motor_positions.shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_motors,), dtype=np.float32)

        self._viewer = None

    def _get_obs(self) -> np.ndarray:
        state = self.sim.get_state()
        return np.concatenate([
            np.asarray(state.position),
            np.asarray(state.velocity),
            np.asarray(state.quaternion),
            np.asarray(state.angular_velocity),
        ]).astype(np.float32)

    def _get_info(self) -> dict:
        state = self.sim.get_state()
        return {
            "time": state.time,
            "position": np.asarray(state.position),
            "velocity": np.asarray(state.velocity),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self._step_count = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.sim.step(action)
        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = self._step_count >= self.max_episode_steps
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self) -> float:
        return 0.0

    def _check_terminated(self) -> bool:
        return False

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
