"""Tests for propulsion dynamics."""

import jax
import jax.numpy as jnp
import pytest

from uavsim.core.types import MultirotorParams
from uavsim.dynamics.propulsion import compute_rotor_wrench, throttle_to_omega
from uavsim.vehicles.multirotor import quadcopter_params


@pytest.fixture
def params():
    return quadcopter_params()


def test_hover_thrust_equals_weight(params):
    """At 50% throttle, total thrust should equal vehicle weight."""
    omega = throttle_to_omega(jnp.array([0.5, 0.5, 0.5, 0.5]), params.max_omega)
    force, torque = compute_rotor_wrench(omega, params)

    expected_weight = params.mass * params.gravity
    assert jnp.allclose(force[2], expected_weight, atol=0.01), (
        f"Hover thrust {float(force[2]):.4f} != weight {expected_weight:.4f}")


def test_zero_throttle_zero_force(params):
    """Zero throttle should produce zero force and torque."""
    omega = throttle_to_omega(jnp.zeros(4), params.max_omega)
    force, torque = compute_rotor_wrench(omega, params)

    assert jnp.allclose(force, 0.0, atol=1e-10)
    assert jnp.allclose(torque, 0.0, atol=1e-10)


def test_symmetric_throttle_zero_torque(params):
    """Equal throttle on all motors should produce zero roll/pitch torque."""
    omega = throttle_to_omega(jnp.array([0.6, 0.6, 0.6, 0.6]), params.max_omega)
    force, torque = compute_rotor_wrench(omega, params)

    # Roll and pitch should be zero
    assert jnp.allclose(torque[0], 0.0, atol=1e-10), f"Roll torque: {torque[0]}"
    assert jnp.allclose(torque[1], 0.0, atol=1e-10), f"Pitch torque: {torque[1]}"
    # Yaw should also be zero (CCW and CW balanced)
    assert jnp.allclose(torque[2], 0.0, atol=1e-10), f"Yaw torque: {torque[2]}"


def test_roll_torque_direction(params):
    """More thrust on front motors should produce roll torque.

    Roll = tau_x = sum(y_i * F_i).  Front motors have +y, back have -y.
    So front-heavy throttle -> positive roll torque.
    """
    omega = throttle_to_omega(
        jnp.array([0.7, 0.7, 0.3, 0.3]),  # front high, back low
        params.max_omega)
    _, torque = compute_rotor_wrench(omega, params)

    assert float(torque[0]) > 0.0, (
        f"Expected positive roll torque, got {float(torque[0]):.6f}")


def test_throttle_clipping(params):
    """Throttle should be clipped to [0, 1]."""
    omega = throttle_to_omega(jnp.array([-0.5, 1.5, 0.5, 0.5]), params.max_omega)
    assert jnp.all(omega >= 0.0)
    assert jnp.all(omega <= params.max_omega)


def test_jit_compatible(params):
    """compute_rotor_wrench should be JIT-compilable."""
    omega = jnp.array([400.0, 400.0, 400.0, 400.0])
    jitted = jax.jit(compute_rotor_wrench)
    force, torque = jitted(omega, params)
    assert force.shape == (3,)
    assert torque.shape == (3,)


def test_differentiable(params):
    """Thrust should be differentiable w.r.t. motor speed."""
    def total_thrust(omega):
        force, _ = compute_rotor_wrench(omega, params)
        return force[2]

    omega = jnp.array([400.0, 400.0, 400.0, 400.0])
    grad = jax.grad(total_thrust)(omega)
    # Gradient should be positive (more omega -> more thrust)
    assert jnp.all(grad > 0), f"Gradient should be positive: {grad}"
