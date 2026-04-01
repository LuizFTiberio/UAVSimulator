"""Tests for Gymnasium environments."""

import numpy as np
import pytest

from uavsim.vehicles.multirotor import quadcopter
from uavsim.envs.hover_env import HoverEnv


@pytest.fixture
def env():
    e = HoverEnv()
    yield e
    e.close()


def test_env_reset(env):
    """Reset should return valid observation and info."""
    obs, info = env.reset()
    assert obs.shape == (13,)
    assert obs.dtype == np.float32
    assert "time" in info


def test_env_step(env):
    """Step with zero action should return valid outputs."""
    env.reset()
    action = np.zeros(4, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (13,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_env_observation_space(env):
    """Observation should be within the observation space after reset."""
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)


def test_env_action_space(env):
    """Random actions should be within bounds."""
    action = env.action_space.sample()
    assert env.action_space.contains(action)
    env.reset()
    obs, _, _, _, _ = env.step(action)
    assert obs.shape == (13,)


def test_env_multiple_steps(env):
    """Environment should handle multiple steps without error."""
    env.reset()
    action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert obs.shape == (13,)


def test_env_tilt_terminates(env):
    """Asymmetric thrust should cause a flip and terminate via tilt check."""
    env.reset()
    # Full thrust on one motor only -> rapid flip
    action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    terminated = False
    for _ in range(3000):
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    assert terminated, "Should terminate from extreme tilt"
