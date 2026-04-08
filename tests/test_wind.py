"""Tests for wind and disturbance models."""

import numpy as np
import jax.numpy as jnp
import pytest

from uavsim.core.types import DrydenParams, BodyDragParams, VehicleState
from uavsim.disturbances.wind import (
    ConstantWind,
    DrydenWind,
    light_turbulence,
    moderate_turbulence,
    severe_turbulence,
)
from uavsim.vehicles.multirotor import multirotor_wrench, quadcopter_params


@pytest.fixture
def params():
    return quadcopter_params()


# ── ConstantWind ─────────────────────────────────────────────────────────────

class TestConstantWind:

    def test_returns_fixed_velocity(self):
        wind = ConstantWind([3.0, -1.0, 0.5])
        v = wind.step(altitude=10.0, dt=0.001)
        assert np.allclose(v, [3.0, -1.0, 0.5])

    def test_does_not_change_over_time(self):
        wind = ConstantWind([2.0, 0.0, 0.0])
        v1 = wind.step(10.0, 0.001)
        v2 = wind.step(10.0, 0.001)
        assert np.allclose(v1, v2)

    def test_default_is_zero(self):
        wind = ConstantWind()
        v = wind.step(10.0, 0.001)
        assert np.allclose(v, 0.0)

    def test_current_velocity_property(self):
        wind = ConstantWind([1.0, 2.0, 3.0])
        assert np.allclose(wind.current_velocity, [1.0, 2.0, 3.0])

    def test_reset_does_nothing(self):
        wind = ConstantWind([1.0, 0.0, 0.0])
        wind.reset()
        assert np.allclose(wind.current_velocity, [1.0, 0.0, 0.0])


# ── DrydenWind ───────────────────────────────────────────────────────────────

class TestDrydenWind:

    def test_output_shape(self):
        wind = DrydenWind(seed=42)
        v = wind.step(altitude=10.0, dt=0.001)
        assert v.shape == (3,)

    def test_nonzero_after_steps(self):
        wind = DrydenWind(seed=42)
        for _ in range(100):
            v = wind.step(10.0, 0.001)
        assert np.linalg.norm(v) > 0.0

    def test_reproducible_with_same_seed(self):
        w1 = DrydenWind(seed=123)
        w2 = DrydenWind(seed=123)
        for _ in range(50):
            v1 = w1.step(10.0, 0.001)
            v2 = w2.step(10.0, 0.001)
        assert np.allclose(v1, v2)

    def test_different_seeds_differ(self):
        w1 = DrydenWind(seed=1)
        w2 = DrydenWind(seed=2)
        for _ in range(50):
            v1 = w1.step(10.0, 0.001)
            v2 = w2.step(10.0, 0.001)
        assert not np.allclose(v1, v2)

    def test_reset_clears_state(self):
        wind = DrydenWind(seed=42)
        for _ in range(100):
            wind.step(10.0, 0.001)
        wind.reset()
        assert np.allclose(wind.current_velocity, 0.0)

    def test_mean_wind_added(self):
        mean = np.array([5.0, 0.0, 0.0])
        wind = DrydenWind(
            params=DrydenParams(sigma_u=0.0, sigma_v=0.0, sigma_w=0.0),
            mean_wind=mean,
            seed=0,
        )
        v = wind.step(10.0, 0.001)
        # With zero turbulence, output should be exactly the mean wind
        assert np.allclose(v, mean, atol=1e-10)

    def test_statistics_bounded(self):
        """Turbulence magnitude should stay within reasonable bounds."""
        wind = DrydenWind(seed=42)
        magnitudes = []
        for _ in range(10000):
            v = wind.step(10.0, 0.01)
            magnitudes.append(np.linalg.norm(v))
        magnitudes = np.array(magnitudes)
        # Mean should be non-zero (turbulence present)
        assert np.mean(magnitudes) > 0.01
        # Should not blow up
        assert np.max(magnitudes) < 20.0


# ── factory functions ────────────────────────────────────────────────────────

class TestWindFactories:

    def test_light_turbulence(self):
        wind = light_turbulence(seed=0)
        for _ in range(100):
            v = wind.step(10.0, 0.001)
        assert np.linalg.norm(v) < 10.0

    def test_moderate_turbulence(self):
        wind = moderate_turbulence(seed=0)
        for _ in range(100):
            v = wind.step(10.0, 0.001)
        assert np.linalg.norm(v) < 20.0

    def test_severe_turbulence(self):
        wind = severe_turbulence(seed=0)
        for _ in range(100):
            v = wind.step(10.0, 0.001)
        assert np.linalg.norm(v) < 30.0

    def test_mean_wind_passthrough(self):
        wind = light_turbulence(mean_wind=[3.0, 0.0, 0.0], seed=0)
        velocities = []
        for _ in range(1000):
            v = wind.step(10.0, 0.001)
            velocities.append(v.copy())
        mean_v = np.mean(velocities, axis=0)
        # X component should be roughly 3 (plus turbulence noise)
        assert abs(mean_v[0] - 3.0) < 1.5


# ── wind effect on multirotor wrench ────────────────────────────────────────

class TestWindOnMultirotor:

    def test_zero_wind_unchanged(self, params):
        """Zero wind should produce same thrust as original (plus tiny drag at v=0)."""
        state = VehicleState(
            position=jnp.zeros(3),
            quaternion=jnp.array([1.0, 0.0, 0.0, 0.0]),
            velocity=jnp.zeros(3),
            angular_velocity=jnp.zeros(3),
            time=0.0,
        )
        cmds = jnp.array([0.5, 0.5, 0.5, 0.5])
        F, T = multirotor_wrench(state, cmds, params, wind_velocity=jnp.zeros(3))
        weight = params.mass * params.gravity
        assert jnp.allclose(F[2], weight, atol=0.01)

    def test_headwind_creates_drag(self, params):
        """Headwind (wind blowing into the vehicle) should create drag force."""
        state = VehicleState(
            position=jnp.zeros(3),
            quaternion=jnp.array([1.0, 0.0, 0.0, 0.0]),
            velocity=jnp.zeros(3),
            angular_velocity=jnp.zeros(3),
            time=0.0,
        )
        cmds = jnp.array([0.5, 0.5, 0.5, 0.5])
        # Strong headwind from +X direction
        wind = jnp.array([10.0, 0.0, 0.0])
        F_wind, _ = multirotor_wrench(state, cmds, params, wind_velocity=wind)
        F_no_wind, _ = multirotor_wrench(state, cmds, params, wind_velocity=jnp.zeros(3))
        # The wind creates drag in +X direction (pushes the quad along)
        # because v_air = v_vehicle - v_wind = 0 - 10 = -10 (body sees headwind)
        # F_drag = -½ρCdA * |v_air| * v_air → positive X (force in wind direction)
        assert float(F_wind[0]) > float(F_no_wind[0])

    def test_crosswind_creates_lateral_force(self, params):
        """Crosswind should create a lateral force."""
        state = VehicleState(
            position=jnp.zeros(3),
            quaternion=jnp.array([1.0, 0.0, 0.0, 0.0]),
            velocity=jnp.zeros(3),
            angular_velocity=jnp.zeros(3),
            time=0.0,
        )
        cmds = jnp.array([0.5, 0.5, 0.5, 0.5])
        wind = jnp.array([0.0, 5.0, 0.0])
        F, _ = multirotor_wrench(state, cmds, params, wind_velocity=wind)
        # Wind from +Y → v_air_y = -5 → drag in +Y direction
        assert abs(float(F[1])) > 0.01
