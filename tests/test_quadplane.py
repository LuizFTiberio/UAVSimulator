"""Tests for quadplane: wing aerodynamics + pusher + combined wrench."""

import jax
import jax.numpy as jnp
import pytest

from uavsim.core.types import VehicleState, WingParams
from uavsim.dynamics.aerodynamics import compute_wing_wrench
from uavsim.vehicles.quadplane import (
    default_wing_params,
    default_pusher_params,
    quadplane_params,
    quadplane_wrench,
    quadplane,
)


@pytest.fixture
def wing():
    return default_wing_params()


@pytest.fixture
def qp_params():
    return quadplane_params()


def _make_state(velocity, quaternion=None):
    """Helper to build a VehicleState with given world-frame velocity."""
    if quaternion is None:
        quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])  # identity
    return VehicleState(
        position=jnp.zeros(3),
        quaternion=quaternion,
        velocity=jnp.array(velocity, dtype=float),
        angular_velocity=jnp.zeros(3),
        time=0.0,
    )


# ── wing aero tests ──────────────────────────────────────────────────────────

class TestWingAero:

    def test_zero_velocity_no_force(self, wing):
        """No airspeed → no aerodynamic force."""
        state = _make_state([0.0, 0.0, 0.0])
        F, T = compute_wing_wrench(state, wing)
        assert jnp.allclose(F, 0.0, atol=1e-8)
        assert jnp.allclose(T, 0.0, atol=1e-8)

    def test_below_transition_negligible(self, wing):
        """At 3 m/s (well below 7 m/s transition), forces should be near zero."""
        state = _make_state([3.0, 0.0, 0.0])
        F, _ = compute_wing_wrench(state, wing)
        assert jnp.linalg.norm(F) < 0.1  # sigmoid suppresses

    def test_above_transition_produces_lift(self, wing):
        """At 15 m/s forward, wing should produce upward lift (positive world Z)."""
        state = _make_state([15.0, 0.0, 0.0])
        F, _ = compute_wing_wrench(state, wing)
        # CL0 = 0.5 at zero AoA → lift should be positive Z in world frame
        assert float(F[2]) > 0.0, f"Expected positive lift, got F_z={float(F[2]):.4f}"

    def test_above_transition_produces_drag(self, wing):
        """At 15 m/s forward, there should be negative X drag."""
        state = _make_state([15.0, 0.0, 0.0])
        F, _ = compute_wing_wrench(state, wing)
        assert float(F[0]) < 0.0, f"Expected negative drag, got F_x={float(F[0]):.4f}"

    def test_lift_scales_with_speed_squared(self, wing):
        """Doubling speed should roughly 4x the aero force (q ∝ V²)."""
        state_10 = _make_state([10.0, 0.0, 0.0])
        state_20 = _make_state([20.0, 0.0, 0.0])
        F10, _ = compute_wing_wrench(state_10, wing)
        F20, _ = compute_wing_wrench(state_20, wing)
        # Both well above transition, so sigma ≈ 1
        ratio = jnp.linalg.norm(F20) / jnp.maximum(jnp.linalg.norm(F10), 1e-8)
        assert 3.5 < float(ratio) < 4.5, f"Force ratio={float(ratio):.2f}, expected ~4"

    def test_aoa_clamp(self, wing):
        """Extreme AoA should be clamped; forces should stay finite."""
        # Pure downward velocity → alpha = 90° raw, clamped to 15°
        state = _make_state([0.0, 0.0, -15.0])
        F, _ = compute_wing_wrench(state, wing)
        assert jnp.all(jnp.isfinite(F))

    def test_jit_compatible(self, wing):
        """compute_wing_wrench should be JIT-compilable."""
        state = _make_state([12.0, 0.0, 0.0])
        jitted = jax.jit(compute_wing_wrench, static_argnums=())
        F, T = jitted(state, wing)
        assert F.shape == (3,)
        assert T.shape == (3,)

    def test_differentiable(self, wing):
        """Lift should be differentiable w.r.t. velocity."""
        def lift_z(vel):
            s = _make_state(vel)
            F, _ = compute_wing_wrench(s, wing)
            return F[2]

        vel = jnp.array([15.0, 0.0, 0.0])
        grad = jax.grad(lift_z)(vel)
        assert jnp.all(jnp.isfinite(grad)), f"Non-finite gradient: {grad}"


# ── quadplane wrench tests ───────────────────────────────────────────────────

class TestQuadplaneWrench:

    def test_hover_thrust_equals_weight(self, qp_params):
        """At 50 % throttle on quad, pusher off, rotor thrust should balance weight."""
        state = _make_state([0.0, 0.0, 0.0])
        cmds = jnp.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
        F, _ = quadplane_wrench(state, cmds, qp_params)
        weight = qp_params.rotor.mass * qp_params.rotor.gravity
        assert jnp.allclose(F[2], weight, atol=0.05), (
            f"F_z={float(F[2]):.3f} vs weight={weight:.3f}")

    def test_pusher_adds_forward_force(self, qp_params):
        """Pusher at full throttle should add positive X force in body frame."""
        state = _make_state([0.0, 0.0, 0.0])
        cmds_no_push = jnp.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
        cmds_push    = jnp.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.0])
        F_no, _ = quadplane_wrench(state, cmds_no_push, qp_params)
        F_yes, _ = quadplane_wrench(state, cmds_push, qp_params)
        delta_x = float(F_yes[0] - F_no[0])
        assert delta_x > 1.0, f"Pusher should add >1 N forward, got {delta_x:.3f}"

    def test_pusher_no_torque(self, qp_params):
        """Pusher at CG should not add torque."""
        state = _make_state([0.0, 0.0, 0.0])
        cmds_no_push = jnp.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
        cmds_push    = jnp.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.0])
        _, T_no  = quadplane_wrench(state, cmds_no_push, qp_params)
        _, T_yes = quadplane_wrench(state, cmds_push, qp_params)
        assert jnp.allclose(T_no, T_yes, atol=1e-8)

    def test_forward_flight_has_lift_and_drag(self, qp_params):
        """In forward flight, total wrench should include aero contribution."""
        state = _make_state([15.0, 0.0, 0.0])
        cmds = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
        F_total, _ = quadplane_wrench(state, cmds, qp_params)

        # Compare with pure rotor + pusher force (no aero)
        from uavsim.vehicles.multirotor import multirotor_wrench
        F_rotor, _ = multirotor_wrench(state, cmds[:4], qp_params.rotor)

        diff = jnp.linalg.norm(F_total - F_rotor)
        assert float(diff) > 0.5, "Wing + pusher should add measurable force"

    def test_jit_compatible(self, qp_params):
        """quadplane_wrench should be JIT-compilable."""
        state = _make_state([10.0, 0.0, 0.0])
        cmds = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
        jitted = jax.jit(quadplane_wrench)
        F, T = jitted(state, cmds, qp_params)
        assert F.shape == (3,)
        assert T.shape == (3,)

    def test_aileron_produces_roll_torque(self, qp_params):
        """Positive aileron at speed should produce roll torque."""
        state = _make_state([15.0, 0.0, 0.0])
        cmds_neutral = jnp.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
        cmds_right   = jnp.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 1.0])
        _, T_n = quadplane_wrench(state, cmds_neutral, qp_params)
        _, T_r = quadplane_wrench(state, cmds_right, qp_params)
        delta = float(jnp.linalg.norm(T_r - T_n))
        assert delta > 0.01, f"Aileron should add roll torque, got delta={delta:.4f}"

    def test_aileron_zero_at_rest(self, qp_params):
        """Aileron should have no effect at zero airspeed."""
        state = _make_state([0.0, 0.0, 0.0])
        cmds_neutral = jnp.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
        cmds_right   = jnp.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 1.0])
        _, T_n = quadplane_wrench(state, cmds_neutral, qp_params)
        _, T_r = quadplane_wrench(state, cmds_right, qp_params)
        assert jnp.allclose(T_n, T_r, atol=1e-8)

    def test_flap_increases_lift(self, qp_params):
        """Full flap at speed should increase lift compared to no flap."""
        state = _make_state([12.0, 0.0, 0.0])
        cmds_clean = jnp.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
        cmds_flap  = jnp.array([0.5, 0.5, 0.5, 0.5, 0.0, 1.0, 0.0])
        F_clean, _ = quadplane_wrench(state, cmds_clean, qp_params)
        F_flap, _  = quadplane_wrench(state, cmds_flap, qp_params)
        # Flap should increase vertical force (more lift)
        assert float(F_flap[2]) > float(F_clean[2]), (
            f"Flap should increase lift: clean={float(F_clean[2]):.3f} "
            f"flap={float(F_flap[2]):.3f}")

    def test_flap_zero_at_rest(self, qp_params):
        """Flap should have no effect at zero airspeed."""
        state = _make_state([0.0, 0.0, 0.0])
        cmds_clean = jnp.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
        cmds_flap  = jnp.array([0.5, 0.5, 0.5, 0.5, 0.0, 1.0, 0.0])
        F_clean, _ = quadplane_wrench(state, cmds_clean, qp_params)
        F_flap, _  = quadplane_wrench(state, cmds_flap, qp_params)
        assert jnp.allclose(F_clean, F_flap, atol=1e-8)


# ── factory tests ────────────────────────────────────────────────────────────

class TestFactory:

    def test_quadplane_creates_vehicle_model(self):
        """quadplane() should return a VehicleModel."""
        from uavsim.vehicles.base import VehicleModel
        v = quadplane()
        assert isinstance(v, VehicleModel)
        assert v.mjcf_path.exists()

    def test_default_wing_params_area(self):
        """S = b² / AR."""
        wp = default_wing_params()
        expected = wp.wingspan ** 2 / wp.aspect_ratio
        assert jnp.allclose(wp.wing_area, expected, atol=1e-6)

    def test_default_wing_ar8(self):
        """Default wing should be AR=8, 1.2 m span."""
        wp = default_wing_params()
        assert wp.aspect_ratio == 8.0
        assert wp.wingspan == 1.2

    def test_default_pusher_max_thrust(self):
        """Default pusher at full throttle should give ~5 N."""
        pp = default_pusher_params()
        thrust = pp.kt * pp.max_omega ** 2
        assert jnp.allclose(thrust, 5.0, atol=0.01)

    def test_seven_actuators(self):
        """quadplane() should have 7 actuator names."""
        v = quadplane()
        assert len(v.actuator_names) == 7
        assert "act_pusher" in v.actuator_names
        assert "act_flap" in v.actuator_names
        assert "act_aileron" in v.actuator_names


# ── controller tests ─────────────────────────────────────────────────────────

class TestQuadplaneController:

    def test_outputs_seven_channels(self):
        """Controller should output (7,) commands."""
        from uavsim.controllers.quadplane_ctrl import QuadplaneTrajectoryController
        params = quadplane_params()
        ctrl = QuadplaneTrajectoryController(params)
        ctrl.set_waypoints([[50.0, 0.0, 5.0], [0.0, 0.0, 5.0]])
        state = _make_state([0.0, 0.0, 0.0])
        state = state._replace(position=jnp.array([0.0, 0.0, 5.0]))
        cmd = ctrl.update(state, 0.001)
        assert cmd.shape == (7,)

    def test_pusher_active_when_far(self):
        """Pusher should be active when far from waypoint (in TRANSITION phase)."""
        from uavsim.controllers.quadplane_ctrl import QuadplaneTrajectoryController, Phase
        params = quadplane_params()
        ctrl = QuadplaneTrajectoryController(params, cruise_speed=12.0)
        ctrl.set_waypoints([[50.0, 0.0, 5.0], [0.0, 0.0, 5.0]])
        # Simulate already transitioned: set phase and ramp
        ctrl._phase = Phase.TRANSITION
        ctrl._transition_ramp = 1.0
        state = _make_state([0.0, 0.0, 0.0])
        state = state._replace(position=jnp.array([0.0, 0.0, 5.0]))
        cmd = ctrl.update(state, 0.001)
        assert float(cmd[4]) > 0.0, "Pusher should be active when far from waypoint"

    def test_pusher_zero_at_final_waypoint(self):
        """Pusher should be zero when sitting on the final waypoint."""
        from uavsim.controllers.quadplane_ctrl import QuadplaneTrajectoryController
        params = quadplane_params()
        ctrl = QuadplaneTrajectoryController(params, acceptance_radius=3.0)
        ctrl.set_waypoints([[0.0, 0.0, 5.0]])
        state = _make_state([0.0, 0.0, 0.0])
        state = state._replace(position=jnp.array([0.0, 0.0, 5.0]))
        cmd = ctrl.update(state, 0.001)
        assert float(cmd[4]) == 0.0, "Pusher should be zero at final waypoint"

    def test_flap_deployed_at_low_speed(self):
        """At zero forward speed, flap should be fully deployed."""
        from uavsim.controllers.quadplane_ctrl import QuadplaneTrajectoryController
        params = quadplane_params()
        ctrl = QuadplaneTrajectoryController(params)
        ctrl.set_waypoints([[50.0, 0.0, 5.0], [0.0, 0.0, 5.0]])
        state = _make_state([0.0, 0.0, 0.0])
        state = state._replace(position=jnp.array([0.0, 0.0, 5.0]))
        cmd = ctrl.update(state, 0.001)
        assert float(cmd[5]) >= 0.9, f"Flap should be ~1.0 at rest, got {float(cmd[5]):.2f}"
