"""Tests for controllers."""

import jax
import jax.numpy as jnp
import pytest

from uavsim.core.types import PIDGains, VehicleState
from uavsim.controllers.pid import pid_init, pid_step
from uavsim.controllers.mixer import mix_x4
from uavsim.controllers.hover import default_hover_gains, hover_init, hover_step
from uavsim.vehicles.multirotor import quadcopter_params


@pytest.fixture
def params():
    return quadcopter_params()


# ── PID tests ────────────────────────────────────────────────────────────────

def test_pid_zero_error():
    """PID output should be zero for zero error."""
    gains = PIDGains(kp=5.0, ki=1.0, kd=2.0, max_output=10.0, integral_limit=5.0)
    state = pid_init(3)
    error = jnp.zeros(3)
    rate = jnp.zeros(3)
    new_state, output = pid_step(state, error, rate, 0.001, gains)
    assert jnp.allclose(output, 0.0, atol=1e-10)


def test_pid_proportional_response():
    """Non-zero error should produce proportional output on first step."""
    gains = PIDGains(kp=5.0, ki=0.0, kd=0.0, max_output=100.0, integral_limit=5.0)
    state = pid_init(3)
    error = jnp.array([1.0, 0.0, 0.0])
    rate = jnp.zeros(3)
    _, output = pid_step(state, error, rate, 0.001, gains)
    # P term = 5.0 * 1.0 = 5.0 (ignoring tiny integral from dt=0.001)
    assert jnp.allclose(output[0], 5.0, atol=0.01)


def test_pid_integral_accumulates():
    """Integral should accumulate over steps."""
    gains = PIDGains(kp=0.0, ki=10.0, kd=0.0, max_output=100.0, integral_limit=100.0)
    state = pid_init(1)
    error = jnp.array([1.0])
    rate = jnp.zeros(1)
    dt = 0.01

    for _ in range(100):
        state, output = pid_step(state, error, rate, dt, gains)

    # After 100 steps * dt=0.01 = 1.0s, integral = 1.0
    # Output = Ki * integral = 10.0 * 1.0 = 10.0
    assert jnp.allclose(output[0], 10.0, atol=0.5)


def test_pid_output_clipping():
    """Output should be clipped to max_output."""
    gains = PIDGains(kp=100.0, ki=0.0, kd=0.0, max_output=5.0, integral_limit=5.0)
    state = pid_init(1)
    error = jnp.array([10.0])
    rate = jnp.zeros(1)
    _, output = pid_step(state, error, rate, 0.001, gains)
    assert jnp.allclose(jnp.abs(output[0]), 5.0, atol=1e-5)


def test_pid_jit_compatible():
    """PID should be JIT-compilable."""
    gains = PIDGains(kp=5.0, ki=1.0, kd=2.0, max_output=10.0, integral_limit=5.0)
    state = pid_init(3)
    jitted = jax.jit(pid_step, static_argnums=(3,))
    _, output = jitted(state, jnp.ones(3), jnp.zeros(3), 0.001, gains)
    assert output.shape == (3,)


# ── mixer tests ──────────────────────────────────────────────────────────────

def test_mixer_hover_equal_throttle():
    """Hover thrust with zero torque -> equal throttle on all motors."""
    thrust = 4.0 * 2.943  # 4 * hover thrust per motor
    throttle = mix_x4(thrust, 0.0, 0.0, 0.0, 0.17, 2.943 * 4)
    assert jnp.allclose(throttle, throttle[0], atol=1e-5), (
        f"Throttle not equal: {throttle}")


def test_mixer_output_range():
    """Throttle should always be in [0, 1]."""
    # Extreme inputs
    throttle = mix_x4(100.0, 10.0, 10.0, 10.0, 0.17, 12.0)
    assert jnp.all(throttle >= 0.0) and jnp.all(throttle <= 1.0)


# ── hover controller tests ──────────────────────────────────────────────────

def test_hover_output_shape(params):
    """Hover controller should output (4,) throttle."""
    gains = default_hover_gains()
    state = hover_init(gains)
    vehicle = VehicleState(
        position=jnp.array([0.0, 0.0, 1.0]),
        quaternion=jnp.array([1.0, 0.0, 0.0, 0.0]),
        velocity=jnp.zeros(3),
        angular_velocity=jnp.zeros(3),
        time=0.0,
    )
    _, throttle = hover_step(
        state, vehicle, jnp.array([0.0, 0.0, 1.0]),
        params, gains, 0.001)
    assert throttle.shape == (4,)
    assert jnp.all(throttle >= 0.0) and jnp.all(throttle <= 1.0)


def test_hover_at_setpoint_near_50pct(params):
    """When at the setpoint with zero velocity, throttle should be ~50%."""
    gains = default_hover_gains()
    state = hover_init(gains)
    vehicle = VehicleState(
        position=jnp.array([0.0, 0.0, 1.0]),
        quaternion=jnp.array([1.0, 0.0, 0.0, 0.0]),
        velocity=jnp.zeros(3),
        angular_velocity=jnp.zeros(3),
        time=0.0,
    )
    _, throttle = hover_step(
        state, vehicle, jnp.array([0.0, 0.0, 1.0]),
        params, gains, 0.001)
    mean_throttle = float(jnp.mean(throttle))
    assert 0.4 < mean_throttle < 0.6, (
        f"Expected ~50% hover throttle, got {mean_throttle:.3f}")
