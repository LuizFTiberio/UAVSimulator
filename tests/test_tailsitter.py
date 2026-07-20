"""Tests for the tailsitter vehicle: propulsion + phi-theory wrench, factory,
and a MuJoCo load/step smoke test."""

import jax
import jax.numpy as jnp
import pytest

from uavsim.core.types import VehicleState
from uavsim.vehicles.base import VehicleModel
from uavsim.vehicles.tailsitter import (
    tailsitter_params,
    tailsitter_wrench,
    tailsitter,
    default_phi_wing_params,
)


def _make_state(velocity, angular_velocity=(0.0, 0.0, 0.0), quaternion=None):
    if quaternion is None:
        quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])
    return VehicleState(
        position=jnp.zeros(3),
        quaternion=quaternion,
        velocity=jnp.array(velocity, dtype=float),
        angular_velocity=jnp.array(angular_velocity, dtype=float),
        time=0.0,
    )


@pytest.fixture
def ts_params():
    return tailsitter_params()


# ── wrench tests ──────────────────────────────────────────────────────────────

class TestTailsitterWrench:

    def test_hover_symmetric_thrust_balances_weight(self, ts_params):
        """At 50% throttle both sides, no elevon, thrust should ~balance weight."""
        state = _make_state([0.0, 0.0, 0.0])
        cmds = jnp.array([0.5, 0.5, 0.0, 0.0])
        F, _ = tailsitter_wrench(state, cmds, ts_params)
        weight = ts_params.mass * ts_params.gravity
        # thrust is along body +x; at identity attitude, world F[0] should
        # equal the weight (kf sized for that at 50% throttle, Eq. 93 sizing).
        assert float(F[0]) == pytest.approx(weight, rel=0.05)

    def test_symmetric_thrust_zero_roll_and_yaw_at_hover(self, ts_params):
        """Symmetric throttle, no elevon, zero airspeed -> zero ROLL/YAW
        (spanwise symmetry, propulsion + aero combined). PITCH is NOT
        asserted zero: real cambered-airfoil data (Cl0 != 0, now the
        default since tailsitter_params()'s coefficients come from real
        Cessna-class data) means propwash alone produces lift even at
        zero elevon, which couples into a real pitch moment through the
        elevon's own moment arm -- a genuine trim requirement, not a bug.
        See uavsim/aero/tests/test_phi_theory.py's matching test and
        chat history for the full derivation."""
        state = _make_state([0.0, 0.0, 0.0])
        cmds = jnp.array([0.5, 0.5, 0.0, 0.0])
        _, T = tailsitter_wrench(state, cmds, ts_params)
        assert float(jnp.abs(T[0])) < 1e-6, "roll should be zero (spanwise symmetry)"
        assert float(jnp.abs(T[2])) < 1e-6, "yaw should be zero (spanwise symmetry)"

    def test_differential_throttle_gives_roll_authority_at_hover(self, ts_params):
        """Differential thrust magnitude at zero airspeed must produce a
        nonzero moment -- this is the whole reason for the dual-engine
        (not single-prop) configuration: hover controllability with no
        vertical tail."""
        state = _make_state([0.0, 0.0, 0.0])
        cmds = jnp.array([0.4, 0.6, 0.0, 0.0])
        _, T = tailsitter_wrench(state, cmds, ts_params)
        assert float(jnp.max(jnp.abs(T))) > 1e-3

    def test_differential_elevon_gives_authority_at_hover(self, ts_params):
        """Differential elevon (in propwash) must also give hover authority,
        matching Sec. VI.A of the paper."""
        state = _make_state([0.0, 0.0, 0.0])
        cmds = jnp.array([0.5, 0.5, 0.3, -0.3])
        _, T = tailsitter_wrench(state, cmds, ts_params)
        assert float(jnp.max(jnp.abs(T))) > 1e-3

    def test_symmetric_elevon_gives_pitch_authority_at_hover(self, ts_params):
        """Regression test for a real bug found while designing the hover
        controller: sections with NO chordwise (x) offset structurally
        cannot produce a pitch moment from elevon deflection (M_y needs an
        x-arm), contradicting the paper's Sec. VI.A claim that symmetric
        elevon deflection produces a b2 (pitch) moment. Checks pitch
        authority exists AND is isolated from roll/yaw for this symmetric
        case (paper: symmetric elevon -> pure b2, no b1/b3)."""
        state = _make_state([0.0, 0.0, 0.0])
        cmds = jnp.array([0.5, 0.5, 0.2, 0.2])  # symmetric (same sign, same magnitude)
        _, T = tailsitter_wrench(state, cmds, ts_params)
        assert float(jnp.abs(T[1])) > 1e-3, "symmetric elevon should produce a pitch (M_y) moment"
        assert float(jnp.abs(T[0])) < 1e-6, "symmetric elevon should NOT produce roll"
        assert float(jnp.abs(T[2])) < 1e-6, "symmetric elevon should NOT produce yaw"

    def test_differential_elevon_gives_roll_at_hover(self, ts_params):
        """Differential elevon must still give a real roll (M_x) moment.
        Not asserting pitch is exactly zero here: with real cambered-
        airfoil data, elevon deflection modulates the SAME camber term
        that gives symmetric-thrust its nonzero pitch (see
        test_symmetric_thrust_zero_roll_and_yaw_at_hover) asymmetrically
        left/right, so some baseline pitch coupling persists regardless
        of symmetric/differential split -- real physics, not a bug."""
        state = _make_state([0.0, 0.0, 0.0])
        cmds = jnp.array([0.5, 0.5, 0.2, -0.2])  # differential
        _, T = tailsitter_wrench(state, cmds, ts_params)
        assert float(jnp.abs(T[0])) > 1e-3, "differential elevon should produce a roll (M_x) moment"
        # Not asserting yaw is zero either: differential elevon changes
        # each side's effective Phi_fv (drag/lift coupling) asymmetrically,
        # which with real (nonzero Cd0/Cla) data genuinely couples into
        # yaw too -- same category of real cross-coupling as the pitch
        # note above, not a separate bug.

    def test_zero_throttle_zero_elevon_effect_at_hover(self, ts_params):
        """No thrust -> no propwash -> elevon does nothing at hover."""
        state = _make_state([0.0, 0.0, 0.0])
        cmds = jnp.array([0.0, 0.0, 0.3, -0.3])
        F, T = tailsitter_wrench(state, cmds, ts_params)
        assert float(jnp.max(jnp.abs(F))) < 1e-8
        assert float(jnp.max(jnp.abs(T))) < 1e-8

    def test_forward_flight_finite_and_nonzero(self, ts_params):
        """In cruise-like forward flight (thrust axis = velocity axis),
        wrench should be finite and nonzero."""
        state = _make_state([15.0, 0.0, 0.0])
        cmds = jnp.array([0.5, 0.5, 0.0, 0.0])
        F, T = tailsitter_wrench(state, cmds, ts_params)
        assert bool(jnp.all(jnp.isfinite(F)))
        assert bool(jnp.all(jnp.isfinite(T)))
        assert float(jnp.linalg.norm(F)) > 0.0

    def test_no_nan_across_hover_to_cruise_sweep(self, ts_params):
        """Smoothness through the whole hover -> transition -> cruise range,
        no singularity anywhere (the point of phi-theory)."""
        cmds = jnp.array([0.5, 0.5, 0.1, -0.1])
        for speed in [0.0, 1e-6, 0.5, 2.0, 8.0, 20.0]:
            for angle_deg in [0.0, 45.0, 90.0, 135.0, 180.0]:
                angle = jnp.radians(angle_deg)
                v = jnp.array([speed * jnp.cos(angle), 0.0, speed * jnp.sin(angle)])
                state = _make_state(v, angular_velocity=(0.2, -0.1, 0.3))
                F, T = tailsitter_wrench(state, cmds, ts_params)
                assert bool(jnp.all(jnp.isfinite(F)))
                assert bool(jnp.all(jnp.isfinite(T)))

    def test_jit_compatible(self, ts_params):
        """Matches the repo's established pattern (see mujoco_sim.py): params
        is bound via closure/partial before jitting, not passed as a traced
        jit argument. PhiWingSection.prop_indices is static Python-level
        topology (which propellers feed which section) used for indexing,
        not a differentiable leaf -- jitting directly over the whole params
        pytree (as a naive `jax.jit(tailsitter_wrench)` call would) breaks
        that indexing, since jit would try to trace the int tuple too."""
        from functools import partial
        state = _make_state([5.0, 0.0, 1.0])
        cmds = jnp.array([0.5, 0.5, 0.1, -0.1])
        jitted = jax.jit(partial(tailsitter_wrench, params=ts_params))
        F, T = jitted(state, cmds)
        assert F.shape == (3,)
        assert T.shape == (3,)

    def test_differentiable_wrt_state(self, ts_params):
        """Wrench should be differentiable w.r.t. velocity (needed for
        gradient-based sysID / MPC / differentiable-sim RL later)."""
        cmds = jnp.array([0.5, 0.5, 0.1, -0.1])

        def fx(vel):
            state = _make_state(vel, angular_velocity=(0.1, 0.1, 0.1))
            F, _ = tailsitter_wrench(state, cmds, ts_params)
            return F[0]

        vel = jnp.array([5.0, 0.0, 1.0])
        grad = jax.grad(fx)(vel)
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_differentiable_wrt_attitude_at_pure_axial_flow(self, ts_params):
        """The specific bug found while cruise-trim solving: differentiating
        the wrench w.r.t. attitude (via a pitch angle that changes v_inf_b's
        direction) at exactly zero attitude gives PURE AXIAL inflow into the
        propeller (zero lateral component) -- which triggered a NaN gradient
        through propulsion_bem.py's V_lateral=norm(...) before the safe_norm
        fix (see chat history). Regression test using the BEM_TABLE path
        specifically, since that's where the bug actually lived (SIMPLE
        kf*omega^2 mode never calls _velocity_decomposition at all)."""
        from uavsim.vehicles.tailsitter import cyclone_hover_capable_params, build_cyclone_bem_table
        from uavsim.core.math import quat_to_rotation_matrix
        pytest.importorskip("bem.core", reason="needs ccblade-jax on PYTHONPATH to build a table")

        table = build_cyclone_bem_table(n_omega=5, n_vinf=4, n_alpha=4)  # small/fast for a test
        params = cyclone_hover_capable_params(bem_table=table)
        cmds = jnp.array([0.7, 0.7, 0.0, 0.0])

        def quat_pitch(theta):
            return jnp.concatenate([jnp.array([jnp.cos(theta / 2)]), jnp.array([0., jnp.sin(theta / 2), 0.])])

        def fx(theta):
            q = quat_pitch(theta)
            state = _make_state([5.0, 0.0, 0.0], quaternion=q)  # pure axial at theta=0
            F, _ = tailsitter_wrench(state, cmds, params)
            return F[0]

        grad = jax.grad(fx)(0.0)
        assert bool(jnp.isfinite(grad)), f"gradient at pure-axial trim should be finite, got {grad}"


        """Gradient must stay finite even exactly at hover -- this is the
        actual point of using phi-theory instead of the classical
        alpha/beta-based model for RL/gradient-based training."""
        cmds = jnp.array([0.5, 0.5, 0.1, -0.1])

        def fx(vel):
            state = _make_state(vel, angular_velocity=(0.1, 0.1, 0.1))
            F, _ = tailsitter_wrench(state, cmds, ts_params)
            return F[0]

        grad = jax.grad(fx)(jnp.zeros(3))
        assert bool(jnp.all(jnp.isfinite(grad)))


# ── factory tests ────────────────────────────────────────────────────────────

class TestFactory:

    def test_tailsitter_creates_vehicle_model(self):
        v = tailsitter()
        assert isinstance(v, VehicleModel)
        assert v.mjcf_path.exists()

    def test_four_actuators(self):
        v = tailsitter()
        assert len(v.actuator_names) == 4
        assert "act_left" in v.actuator_names
        assert "act_right" in v.actuator_names

    def test_default_phi_wing_two_sections_two_props(self):
        wing = default_phi_wing_params()
        assert len(wing.sections) == 2
        assert len(wing.propellers) == 2

    def test_kf_sizes_hover_at_half_throttle(self):
        params = tailsitter_params(mass=1.2, gravity=9.81, max_omega=1200.0)
        omega_half = 0.5 * params.propulsion.max_omega
        total_thrust = 2 * params.propulsion.kf * omega_half ** 2
        weight = params.mass * params.gravity
        assert float(total_thrust) == pytest.approx(float(weight), rel=1e-6)


# ── MuJoCo load/step smoke test ──────────────────────────────────────────────

class TestMuJoCoIntegration:

    def test_model_loads_and_steps(self):
        """The MJCF must load, and a few sim steps at hover throttle must
        produce finite state with the vehicle roughly holding altitude
        (thrust ~ weight, symmetric, no wind -- shouldn't tumble in a few
        steps)."""
        from uavsim.sim.mujoco_sim import MuJoCoSimulator

        vehicle = tailsitter()
        sim = MuJoCoSimulator(vehicle)
        state = sim.reset()
        assert bool(jnp.all(jnp.isfinite(state.position)))

        cmds = jnp.array([0.5, 0.5, 0.0, 0.0])
        for _ in range(200):
            state = sim.step(cmds)

        assert bool(jnp.all(jnp.isfinite(state.position)))
        assert bool(jnp.all(jnp.isfinite(state.quaternion)))
        assert bool(jnp.all(jnp.isfinite(state.velocity)))
        assert bool(jnp.all(jnp.isfinite(state.angular_velocity)))
        # Shouldn't have diverged/gone NaN. NOT a tight bound: with real
        # cambered-airfoil coefficients (Cl0 != 0, the current default),
        # symmetric thrust + zero elevon has a real, physically expected
        # uncompensated pitch moment (see
        # tests/test_tailsitter.py::TestTailsitterWrench::
        # test_symmetric_thrust_zero_roll_and_yaw_at_hover and chat
        # history) -- this open-loop test has no controller to trim it
        # out, so real pitch rate builds up over 200 steps. This is a
        # coarse smoke test (finite, doesn't blow up) not a trim check;
        # a real hover controller (see tests/test_tailsitter_hover.py)
        # is what actually holds attitude.
        assert float(jnp.linalg.norm(state.angular_velocity)) < 20.0
