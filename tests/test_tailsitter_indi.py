"""Tests for the tail-sitter INDI attitude+rate controller.

The closed-loop tests mirror tests/test_tailsitter_hover.py so INDI can be read
as a drop-in for the fixed-trim mixer -- same maneuvers, same success gates.
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from uavsim.core.types import VehicleState
from uavsim.core.math import quat_to_rotation_matrix
from uavsim.vehicles.tailsitter import tailsitter_params, tailsitter
from uavsim.controllers.tailsitter_indi import (
    TailsitterINDIController,
    composite_inertia_from_model,
    indi_effectiveness,
    default_tailsitter_indi_gains,
    desired_attitude_transition,
    estimate_sideslip,
    coordinated_turn_rate,
    V_SIDESLIP_MIN,
    guidance_thrust_axis,
    guidance_attitude,
    guidance_extract_bank_pitch,
    guidance_accel,
    guidance_effectiveness_analytic,
    paper_lift_slope,
)
from uavsim.controllers.tailsitter_guidance import MissionGuidance, MissionLeg


def _quat_from_axis_angle(axis, angle):
    axis = jnp.asarray(axis, dtype=float)
    axis = axis / jnp.linalg.norm(axis)
    return jnp.concatenate([jnp.array([jnp.cos(angle / 2)]), jnp.sin(angle / 2) * axis])


@pytest.fixture(scope="module")
def params():
    return tailsitter_params()


@pytest.fixture(scope="module")
def sim_and_inertia(params):
    from uavsim.sim.mujoco_sim import MuJoCoSimulator
    vehicle = tailsitter(params=params)
    sim = MuJoCoSimulator(vehicle)
    I = composite_inertia_from_model(sim.model)
    return sim, I


# ── composite inertia ────────────────────────────────────────────────────────

class TestCompositeInertia:

    def test_offset_props_add_to_roll_and_yaw(self, sim_and_inertia):
        """The two propeller bodies at y=+/-0.3 m must add ~m*d^2 to roll (Ixx)
        and yaw (Izz) but little to pitch (Iyy) -- the whole reason to read the
        composite inertia rather than just the fuselage diaginertia."""
        _, I = sim_and_inertia
        assert I.shape == (3, 3)
        # symmetric, positive definite
        assert np.allclose(I, I.T, atol=1e-9)
        assert np.all(np.linalg.eigvalsh(I) > 0)
        Ixx, Iyy, Izz = I[0, 0], I[1, 1], I[2, 2]
        # fuselage-only diaginertia is [0.018, 0.006, 0.020]; props push Ixx/Izz up
        assert Ixx > 0.021, f"roll inertia should include prop offset: {Ixx}"
        assert Izz > 0.023, f"yaw inertia should include prop offset: {Izz}"
        assert Iyy < 0.008, f"pitch inertia barely changes: {Iyy}"


# ── control effectiveness (the INDI G) ───────────────────────────────────────

class TestEffectiveness:

    def test_G_well_conditioned_at_hover_trim(self, params, sim_and_inertia):
        _, I = sim_and_inertia
        ctrl = TailsitterINDIController(params, inertia=I)
        state = VehicleState(position=jnp.zeros(3),
                             quaternion=jnp.array([1., 0., 0., 0.]),
                             velocity=jnp.zeros(3), angular_velocity=jnp.zeros(3),
                             time=0.0)
        G, _ = ctrl._eff_jit(state, jnp.asarray(ctrl._u_trim))
        G = np.asarray(G)
        assert G.shape == (4, 4)
        assert np.all(np.isfinite(G))
        assert np.linalg.cond(G) < 1000.0

    def test_G_recomputed_differs_across_states(self, params, sim_and_inertia):
        """The point of INDI: G is state-dependent. G at hover attitude with
        forward airspeed must differ from G at rest -- otherwise it's just a
        fixed mixer."""
        _, I = sim_and_inertia
        inertia_inv = jnp.asarray(np.linalg.inv(I))
        q_hover = _quat_from_axis_angle([0, 1, 0], -jnp.pi / 2)
        rest = VehicleState(jnp.zeros(3), q_hover, jnp.zeros(3), jnp.zeros(3), 0.0)
        moving = VehicleState(jnp.zeros(3), q_hover,
                              jnp.array([0., 0., 12.0]),  # climbing = forward airspeed in hover
                              jnp.zeros(3), 0.0)
        u = jnp.array([0.5, 0.5, 0.0, 0.0])
        G_rest, _ = indi_effectiveness(rest, u, params, inertia_inv)
        G_move, _ = indi_effectiveness(moving, u, params, inertia_inv)
        assert not np.allclose(np.asarray(G_rest), np.asarray(G_move), atol=1e-3)

    def test_effectiveness_smooth_and_conditioned_across_envelope(self, params, sim_and_inertia):
        """Sanity-sweep G over pitch angle (hover->cruise) and airspeed (roadmap
        step 5): it must stay finite, well-conditioned, and vary SMOOTHLY --
        this is what lets a single recomputed-every-step G span the envelope
        with no scheduling. Also checks the qualitative INDI expectation that
        elevon PITCH authority grows with airspeed (the freestream term the
        propwash-only hover case lacks)."""
        _, I = sim_and_inertia
        inertia_inv = jnp.asarray(np.linalg.inv(I))
        u = jnp.array([0.6, 0.6, 0.0, 0.0])

        thetas = np.linspace(0.0, np.pi / 2, 10)   # nose-up (hover) -> nose-fwd
        speeds = np.linspace(0.0, 15.0, 8)
        conds, pitch_elevon = [], []
        G_prev = None
        for th in thetas:
            for V in speeds:
                q = _quat_from_axis_angle([0, 1, 0], -(np.pi / 2 - th))
                # airspeed along world +x (forward): body sees it per attitude
                st = VehicleState(jnp.zeros(3), q, jnp.array([V, 0.0, 0.0]),
                                  jnp.zeros(3), 0.0)
                G, _ = indi_effectiveness(st, u, params, inertia_inv)
                G = np.asarray(G)
                assert np.all(np.isfinite(G))
                conds.append(np.linalg.cond(G))
                # pitch row (1) vs elevon columns (2,3)
                pitch_elevon.append(abs(G[1, 2]) + abs(G[1, 3]))
                if G_prev is not None:
                    # smoothness: adjacent-in-speed G entries shouldn't jump
                    step = np.max(np.abs(G - G_prev))
                    assert step < 500.0, f"G jumped by {step} (theta={th:.2f}, V={V:.1f})"
                G_prev = G
            G_prev = None  # don't compare across the theta seam

        assert max(conds) < 5000.0, f"G ill-conditioned somewhere: cond={max(conds):.0f}"
        # elevon pitch authority at high speed should exceed that at hover
        pe = np.array(pitch_elevon).reshape(len(thetas), len(speeds))
        assert pe[:, -1].mean() > pe[:, 0].mean(), \
            "elevon pitch authority should grow with airspeed"


class TestTransitionAttitude:

    def test_reduces_to_hover_construction(self):
        """desired_attitude_transition must match the hover-only TRIAD for the
        vertical thrust axis (so it's a safe drop-in), and give identity for a
        horizontal (cruise) thrust axis without degenerating."""
        from uavsim.controllers.tailsitter_hover import (
            desired_attitude_from_thrust_direction)
        b1_hover = jnp.array([0.0, 0.0, 1.0])
        R_new = np.asarray(desired_attitude_transition(b1_hover, 0.0))
        R_old = np.asarray(desired_attitude_from_thrust_direction(b1_hover, 0.0))
        assert np.allclose(R_new, R_old, atol=1e-6)
        assert np.allclose(R_new.T @ R_new, np.eye(3), atol=1e-6)

    def test_nondegenerate_at_horizontal_thrust_axis(self):
        """The failure mode of the hover construction: b1 horizontal (cruise).
        The span-axis construction must stay a proper rotation there."""
        b1_cruise = jnp.array([1.0, 0.0, 0.0])
        R = np.asarray(desired_attitude_transition(b1_cruise, 0.0))
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
        assert np.allclose(R[:, 0], [1.0, 0.0, 0.0], atol=1e-6)


# ── closed-loop (drop-in mirror of the hover controller tests) ───────────────

class TestClosedLoopHover:

    def test_recovers_from_identity_attitude_to_hover(self, params, sim_and_inertia):
        """Spawn nose-forward (identity quaternion); hover is nose-up -- a ~90deg
        maneuver handled by the SAME continuous G, no mode switch."""
        sim, I = sim_and_inertia
        ctrl = TailsitterINDIController(params, inertia=I, dt=sim.dt)
        state = sim.reset(position=np.array([0.0, 0.0, 2.0]))
        ctrl.reset()

        setpoint = jnp.array([0.0, 0.0, 2.0])
        for _ in range(4000):  # 4 s
            cmds = ctrl.update(state, setpoint)
            state = sim.step(cmds)
            assert bool(jnp.all(jnp.isfinite(state.position))), "diverged"
            assert bool(jnp.all(jnp.isfinite(state.quaternion))), "diverged"

        pos_err = float(jnp.linalg.norm(state.position - setpoint))
        ang_rate = float(jnp.linalg.norm(state.angular_velocity))
        assert pos_err < 2.0, f"pos_err={pos_err:.2f} m"
        assert ang_rate < 3.0, f"still tumbling: |omega|={ang_rate:.2f}"

        R = quat_to_rotation_matrix(state.quaternion)
        alignment = float(jnp.dot(R[:, 0], jnp.array([0.0, 0.0, 1.0])))
        assert alignment > 0.7, f"did not reach hover attitude: b1.z={alignment:.2f}"

    def test_holds_position_from_near_hover_start(self, params, sim_and_inertia):
        """Start near hover attitude with a position offset; converge and hold."""
        import mujoco
        sim, I = sim_and_inertia
        ctrl = TailsitterINDIController(params, inertia=I, dt=sim.dt)
        sim.reset()

        q_hover = _quat_from_axis_angle([0, 1, 0], -jnp.pi / 2)
        sim.data.qpos[3:7] = np.asarray(q_hover)
        sim.data.qpos[0:3] = np.array([0.5, 0.3, 2.0])
        mujoco.mj_forward(sim.model, sim.data)
        state = sim.get_state()
        ctrl.reset()

        setpoint = jnp.array([0.0, 0.0, 2.0])
        for _ in range(3000):  # 3 s
            cmds = ctrl.update(state, setpoint)
            state = sim.step(cmds)
            assert bool(jnp.all(jnp.isfinite(state.position)))

        pos_err = float(jnp.linalg.norm(state.position - setpoint))
        assert pos_err < 0.5, f"did not hold position: pos_err={pos_err:.2f} m"


class TestTransition:

    def test_hover_forward_flight_hover_no_mode_switch(self, params, sim_and_inertia):
        """Roadmap step 8, the Phase A payoff: the SAME controller (no mode
        switch, no gain schedule) flies hover -> accelerate to forward flight
        -> decelerate -> hover, just by tracking a moving velocity reference.
        The continuous per-step G is what makes this one controller.

        The vehicle pitches well over toward horizontal at speed and returns to
        nose-up hover at the end; success = it never diverges, actually builds
        forward speed, and recovers a stable hover."""
        import mujoco
        sim, I = sim_and_inertia
        ctrl = TailsitterINDIController(params, inertia=I, dt=sim.dt,
                                        mass=sim.total_mass)  # model-based outer loop
        sim.reset()
        q_hover = _quat_from_axis_angle([0, 1, 0], -jnp.pi / 2)
        sim.data.qpos[3:7] = np.asarray(q_hover)
        sim.data.qpos[0:3] = np.array([0.0, 0.0, 2.0])
        mujoco.mj_forward(sim.model, sim.data)
        state = sim.get_state()
        ctrl.reset()

        V, dt = 6.0, sim.dt

        def vel_ref(t):  # trapezoidal forward-speed profile
            if t < 1.0:   return 0.0
            if t < 4.0:   return V * (t - 1.0) / 3.0
            if t < 7.0:   return V
            if t < 10.0:  return V * (1.0 - (t - 7.0) / 3.0)
            return 0.0

        x_ref, v_prev, max_speed = 0.0, 0.0, 0.0
        max_pitch = 0.0
        for k in range(13000):  # 13 s
            t = k * dt
            vx = vel_ref(t)
            ax = (vx - v_prev) / dt
            v_prev = vx
            x_ref += vx * dt
            cmds = ctrl.update(state, jnp.array([x_ref, 0.0, 2.0]),
                               setpoint_velocity=np.array([vx, 0.0, 0.0]),
                               accel_feedforward=np.array([ax, 0.0, 0.0]))
            state = sim.step(cmds)
            assert bool(jnp.all(jnp.isfinite(state.position))), f"diverged at t={t:.2f}"
            max_speed = max(max_speed, float(state.velocity[0]))
            b1 = np.asarray(quat_to_rotation_matrix(state.quaternion)[:, 0])
            max_pitch = max(max_pitch, float(np.degrees(np.arctan2(b1[0], b1[2]))))

        # actually transitioned: built real forward speed and pitched well over
        assert max_speed > 4.0, f"did not build forward speed: {max_speed:.1f} m/s"
        assert max_pitch > 30.0, f"did not pitch toward cruise: {max_pitch:.1f} deg"

        # recovered a stable nose-up hover at the end
        b1_end = np.asarray(quat_to_rotation_matrix(state.quaternion)[:, 0])
        assert b1_end[2] > 0.9, f"did not return to hover attitude: b1.z={b1_end[2]:.2f}"
        assert float(state.velocity[0]) < 0.5, "did not decelerate back to hover"
        assert float(jnp.linalg.norm(state.angular_velocity)) < 1.0, "not settled"
        assert abs(float(state.position[2]) - 2.0) < 1.0, "lost too much altitude"


class TestAllocationBackends:

    def test_pinv_and_wls_both_stabilise_hover(self, params, sim_and_inertia):
        """Both allocation backends should hold a near-hover start (WLS reduces
        to the pinv solution when nothing saturates, so near trim they agree)."""
        import mujoco
        sim, I = sim_and_inertia
        for use_wls in (True, False):
            ctrl = TailsitterINDIController(params, inertia=I, dt=sim.dt, use_wls=use_wls)
            sim.reset()
            q_hover = _quat_from_axis_angle([0, 1, 0], -jnp.pi / 2)
            sim.data.qpos[3:7] = np.asarray(q_hover)
            sim.data.qpos[0:3] = np.array([0.2, 0.0, 2.0])
            mujoco.mj_forward(sim.model, sim.data)
            state = sim.get_state()
            ctrl.reset()
            setpoint = jnp.array([0.0, 0.0, 2.0])
            for _ in range(2000):
                state = sim.step(ctrl.update(state, setpoint))
            pos_err = float(jnp.linalg.norm(state.position - setpoint))
            assert pos_err < 0.6, f"use_wls={use_wls}: pos_err={pos_err:.2f} m"


class TestExperimentalINDIOuter:

    def test_measured_accel_outer_loop_stays_bounded(self, params, sim_and_inertia):
        """The measured-acceleration INDI outer loop (use_indi_outer=True) is
        experimental and holds less tightly than the model-based default, but
        with the m*g hover-thrust baseline it must at least stay BOUNDED (not
        diverge) -- a regression guard on that fix (using the modeled thrust as
        the baseline instead made |omega| run away)."""
        import mujoco
        sim, I = sim_and_inertia
        ctrl = TailsitterINDIController(params, inertia=I, dt=sim.dt,
                                        mass=sim.total_mass, use_indi_outer=True)
        sim.reset()
        q_hover = _quat_from_axis_angle([0, 1, 0], -jnp.pi / 2)
        sim.data.qpos[3:7] = np.asarray(q_hover)
        sim.data.qpos[0:3] = np.array([0.0, 0.0, 2.0])
        mujoco.mj_forward(sim.model, sim.data)
        state = sim.get_state()
        ctrl.reset()

        setpoint = jnp.array([0.0, 0.0, 2.0])
        for _ in range(3000):
            state = sim.step(ctrl.update(state, setpoint))
            assert bool(jnp.all(jnp.isfinite(state.position)))

        # bounded (not diverging): stays in the neighbourhood, rate not growing
        assert float(jnp.linalg.norm(state.position - setpoint)) < 1.5
        assert float(jnp.linalg.norm(state.angular_velocity)) < 1.5


# ── Phase B: sideslip control (paper Sec. 3) ─────────────────────────────────

class TestSideslipEstimate:

    def test_beta_from_body_lateral_velocity(self):
        """Sideslip is the airspeed component along the span axis (body y):
        at identity attitude a velocity [4, 1, 0] gives beta = atan2(1, 4)."""
        state = VehicleState(
            position=jnp.zeros(3), quaternion=jnp.array([1.0, 0.0, 0.0, 0.0]),
            velocity=jnp.array([4.0, 1.0, 0.0]), angular_velocity=jnp.zeros(3), time=0.0)
        beta, V, v_body = estimate_sideslip(state)
        assert np.isclose(beta, np.arctan2(1.0, 4.0))
        assert np.isclose(V, np.hypot(4.0, 1.0))
        assert np.allclose(v_body, [4.0, 1.0, 0.0])

    def test_wind_is_subtracted(self):
        """A pure crosswind with zero ground velocity is pure sideslip: the
        vehicle sits still but the air moves along -body-y, so beta != 0."""
        state = VehicleState(
            position=jnp.zeros(3), quaternion=jnp.array([1.0, 0.0, 0.0, 0.0]),
            velocity=jnp.array([4.0, 0.0, 0.0]), angular_velocity=jnp.zeros(3), time=0.0)
        beta_nowind, _, _ = estimate_sideslip(state)
        beta_wind, _, _ = estimate_sideslip(state, wind_velocity=np.array([0.0, 1.0, 0.0]))
        assert np.isclose(beta_nowind, 0.0)
        # airspeed = v - wind = [4, -1, 0] -> negative beta
        assert beta_wind < -0.1


class TestCoordinatedTurnRate:

    def test_zero_bank_zero_rate(self):
        assert coordinated_turn_rate(0.0, 8.0, 9.81) == 0.0

    def test_positive_bank_positive_rate_and_speed_scaling(self):
        """g*tan(phi)/V: a bank gives a turn rate that falls with airspeed."""
        r_slow = coordinated_turn_rate(0.3, 5.0, 9.81)
        r_fast = coordinated_turn_rate(0.3, 10.0, 9.81)
        assert r_slow > 0 and r_fast > 0
        assert r_slow > r_fast
        assert np.isclose(r_slow, 9.81 * np.tan(0.3) / 5.0)

    def test_low_speed_is_guarded(self):
        """The 1/V feedforward is floored at V_SIDESLIP_MIN so it can't blow up
        near hover."""
        r = coordinated_turn_rate(0.3, 0.01, 9.81)
        assert np.isclose(r, 9.81 * np.tan(0.3) / V_SIDESLIP_MIN)


def _run_crosswind(params, use_sideslip_control, cross=2.0):
    """Fly a hover->forward-flight ramp through a steady +y crosswind. Returns
    the mean |beta| (deg) over the steady cruise window (t > 5 s)."""
    import mujoco
    from uavsim.sim.mujoco_sim import MuJoCoSimulator
    from uavsim.disturbances import ConstantWind
    sim = MuJoCoSimulator(tailsitter(params=params),
                          wind_model=ConstantWind(np.array([0.0, cross, 0.0])))
    I = composite_inertia_from_model(sim.model)
    ctrl = TailsitterINDIController(params, inertia=I, dt=sim.dt, mass=sim.total_mass,
                                    use_sideslip_control=use_sideslip_control)
    sim.reset()
    sim.data.qpos[3:7] = np.asarray(_quat_from_axis_angle([0, 1, 0], -jnp.pi / 2))
    sim.data.qpos[0:3] = np.array([0.0, 0.0, 3.0])
    mujoco.mj_forward(sim.model, sim.data)
    state = sim.get_state()
    ctrl.reset()
    V, dt = 8.0, sim.dt
    vel_ref = lambda t: 0.0 if t < 1 else (V * (t - 1) / 3 if t < 4 else V)
    x_ref, v_prev, betas = 0.0, 0.0, []
    for k in range(int(9.0 / dt)):
        t = k * dt
        vx = vel_ref(t); ax = (vx - v_prev) / dt; v_prev = vx; x_ref += vx * dt
        wv = sim.wind_velocity
        state = sim.step(ctrl.update(
            state, jnp.array([x_ref, 0.0, 3.0]), setpoint_velocity=np.array([vx, 0.0, 0.0]),
            accel_feedforward=np.array([ax, 0.0, 0.0]), wind_velocity=wv))
        assert bool(jnp.all(jnp.isfinite(state.position))), f"diverged t={t:.2f}"
        if t > 5.0:
            betas.append(abs(np.degrees(estimate_sideslip(state, wv)[0])))
    return float(np.mean(betas))


class TestSideslipControl:

    def test_crosswind_sideslip_is_regulated(self, params):
        """Roadmap Phase B step 3: a steady crosswind gives sustained sideslip
        the position loop CANNOT remove (it holds ground position, but airspeed
        = ground velocity - wind still has a span-axis component). Enabling
        sideslip control must drive that steady sideslip substantially toward
        zero (~15 deg uncontrolled -> ~5 deg controlled)."""
        beta_off = _run_crosswind(params, use_sideslip_control=False)
        beta_on = _run_crosswind(params, use_sideslip_control=True)
        assert beta_off > 12.0, f"crosswind should induce real sideslip: {beta_off:.1f} deg"
        assert beta_on < 8.0, f"sideslip not regulated: {beta_on:.1f} deg"
        assert beta_on < 0.5 * beta_off, f"insufficient reduction: {beta_off:.1f}->{beta_on:.1f}"

    def test_coordinated_turn_holds_zero_sideslip(self, params):
        """Roadmap Phase B step 3 ('holds it through a turn'): flying a gently
        curving velocity reference with a fixed heading builds huge sideslip (the
        nose does not follow the turning velocity); enabling sideslip control
        with the coordinated-turn feedforward (yaw_rate_ff = commanded turn rate)
        keeps the nose on the velocity, holding beta near zero through the turn."""
        import mujoco
        from uavsim.sim.mujoco_sim import MuJoCoSimulator
        V, rate, dt_turn = 5.0, 0.15, None

        def run(use_ss, coord):
            sim = MuJoCoSimulator(tailsitter(params=params))
            I = composite_inertia_from_model(sim.model)
            ctrl = TailsitterINDIController(params, inertia=I, dt=sim.dt,
                                            mass=sim.total_mass, use_sideslip_control=use_ss)
            sim.reset()
            sim.data.qpos[3:7] = np.asarray(_quat_from_axis_angle([0, 1, 0], -jnp.pi / 2))
            sim.data.qpos[0:3] = np.array([0.0, 0.0, 4.0])
            mujoco.mj_forward(sim.model, sim.data)
            state = sim.get_state(); ctrl.reset()
            dt = sim.dt
            speed = lambda t: 0.0 if t < 1 else (V * (t - 1) / 3 if t < 4 else V)
            psi_v, pos, vprev, betas = 0.0, np.array([0.0, 0.0, 4.0]), np.zeros(3), []
            for k in range(int(13.0 / dt)):
                t = k * dt; sp = speed(t); turning = t >= 5.0
                if turning:
                    psi_v += rate * dt
                vref = np.array([sp * np.cos(psi_v), sp * np.sin(psi_v), 0.0])
                aref = (vref - vprev) / dt; vprev = vref; pos = pos + vref * dt
                yrf = rate if (turning and coord) else 0.0
                state = sim.step(ctrl.update(state, jnp.asarray(pos), setpoint_velocity=vref,
                                             accel_feedforward=aref, yaw_rate_ff=yrf))
                assert bool(jnp.all(jnp.isfinite(state.position))), f"diverged t={t:.2f}"
                if t > 6.0:
                    b, Vv, _ = estimate_sideslip(state)
                    if Vv > 2.0:
                        betas.append(abs(np.degrees(b)))
            return float(np.mean(betas))

        beta_uncoordinated = run(use_ss=False, coord=False)
        beta_coordinated = run(use_ss=True, coord=True)
        assert beta_uncoordinated > 25.0, \
            f"fixed-heading turn should build sideslip: {beta_uncoordinated:.1f} deg"
        assert beta_coordinated < 8.0, \
            f"coordinated turn should hold near-zero sideslip: {beta_coordinated:.1f} deg"


# ── Phase D: Guidance INDI outer loop (paper Sec. 4, Eq. 18-33) ──────────────

class TestGuidanceParameterization:

    def test_thrust_axis_hover_and_cruise(self):
        """theta=0 -> thrust axis straight up (hover); theta=pi/2 -> horizontal
        along heading (cruise)."""
        b1_hover = np.asarray(guidance_thrust_axis(0.0, 0.0))
        assert np.allclose(b1_hover, [0., 0., 1.], atol=1e-6)
        b1_cruise = np.asarray(guidance_thrust_axis(np.pi/2, 0.0))
        assert np.allclose(b1_cruise, [1., 0., 0.], atol=1e-6)
        # heading rotates the axis in the horizontal plane
        b1_h = np.asarray(guidance_thrust_axis(np.pi/2, np.pi/2))
        assert np.allclose(b1_h, [0., 1., 0.], atol=1e-6)

    def test_extract_bank_pitch_roundtrip(self):
        """guidance_extract_bank_pitch inverts guidance_attitude (reads the INDI
        baseline v_f = [phi, theta] off the current attitude), across the whole
        envelope INCLUDING cruise (where bank does not move the thrust axis)."""
        for theta in (0.05, 0.5, 1.0, np.pi/2 - 1e-3):
            for phi in (-0.4, 0.0, 0.3):
                for psi in (0.0, 0.7, -1.2):
                    R = guidance_attitude(phi, theta, psi)
                    ph, th = guidance_extract_bank_pitch(R, psi)
                    assert np.isclose(float(th), theta, atol=1e-5), (theta, phi, psi, float(th))
                    assert np.isclose(float(ph), phi, atol=1e-5), (theta, phi, psi, float(ph))

    def test_attitude_reduces_to_wings_level_at_zero_bank(self):
        """phi=0 must reduce exactly to the wings-level construction."""
        for theta in (0.0, 0.6, 1.2):
            for psi in (0.0, 0.8):
                R = np.asarray(guidance_attitude(0.0, theta, psi))
                R0 = np.asarray(desired_attitude_transition(
                    guidance_thrust_axis(theta, psi), psi))
                assert np.allclose(R, R0, atol=1e-6)
                assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)


class TestGuidanceEffectiveness:

    def test_jacobian_G_well_conditioned_across_envelope(self, params, sim_and_inertia):
        """The outer G = d(accel)/d[phi, theta, T] must stay finite and well-
        conditioned from hover to cruise -- what lets one recomputed-every-step
        G span the envelope (the outer-loop analog of the inner-G sweep)."""
        m = 1.26
        Gfun = jax.jit(jax.jacobian(
            lambda ph, th, T, V: guidance_accel(
                ph, th, T, 0.0, jnp.array([V, 0., 0.]), params, m),
            argnums=(0, 1, 2)))
        for thdeg, V in [(1., 0.), (15., 1.), (45., 6.), (74., 12.), (79., 16.)]:
            d = Gfun(0.0, np.radians(thdeg), m*9.81 if V < 8 else 0.2*m*9.81, V)
            G = np.stack([np.asarray(d[0]), np.asarray(d[1]), np.asarray(d[2])], axis=1)
            assert np.all(np.isfinite(G))
            assert np.linalg.cond(G) < 2000.0, f"ill-conditioned at {thdeg}deg/{V}: {np.linalg.cond(G):.0f}"

    def test_jacobian_captures_lift_derivative_growing_with_airspeed(self, params):
        """The whole point: d(a_z)/d(theta) is a real lift derivative that grows
        with airspeed (the wing makes more lift faster). Compare cruise-attitude
        effectiveness at low vs high V."""
        m = 1.26
        dacc = jax.jacobian(lambda th, V: guidance_accel(
            0.0, th, 0.2*m*9.81, 0.0, jnp.array([V, 0., 0.]), params, m), argnums=0)
        daz_slow = float(np.asarray(dacc(np.radians(74.), 6.0))[2])
        daz_fast = float(np.asarray(dacc(np.radians(74.), 14.0))[2])
        # more negative at higher speed (stronger pitch->accel coupling via lift)
        assert abs(daz_fast) > abs(daz_slow)

    def test_jacobian_bank_gives_lateral_authority_via_lift_in_cruise(self, params):
        """The turn fix: in cruise the bank column d(accel)/d(phi) must produce a
        large HORIZONTAL (lateral) acceleration -- that is the lift vector being
        rolled into the turn. It must dwarf the hover bank authority (no lift)."""
        m = 1.26
        dphi = jax.jacobian(lambda ph, V, th: guidance_accel(
            ph, th, 0.2*m*9.81, 0.0, jnp.array([V, 0., 0.]), params, m), argnums=0)
        # cruise: theta high, real airspeed -> lift present
        a_cruise = np.asarray(dphi(0.0, 14.0, np.radians(78.)))
        lat_cruise = np.hypot(a_cruise[0], a_cruise[1])
        # hover: theta ~ 0, no airspeed -> banking rolls ~no lift
        a_hover = np.asarray(dphi(0.0, 0.0, np.radians(1.0)))
        lat_hover = np.hypot(a_hover[0], a_hover[1])
        assert lat_cruise > 3.0, f"bank gives too little lateral accel in cruise: {lat_cruise:.1f}"
        assert lat_cruise > 3.0 * lat_hover, "bank authority should be lift-dominated in cruise"

    def test_analytic_G_shape_and_thrust_column(self, params):
        """Analytic backend returns a 3x3 whose thrust column is b1/m (thrust
        acts along the thrust axis)."""
        m = 1.26
        b1 = np.asarray(guidance_thrust_axis(np.radians(70.), 0.0))
        G = np.asarray(guidance_effectiveness_analytic(
            0.0, np.radians(70.), 3.0, 0.0, 12.0, m, 9.81,
            paper_lift_slope(12.0, m), 1.0))
        assert G.shape == (3, 3)
        assert np.allclose(G[:, 2], b1/m, atol=1e-9)


class TestGuidanceINDIClosedLoop:

    def _fly_to_cruise(self, params, sim_and_inertia, **ctrl_kw):
        import mujoco
        from uavsim.sim.mujoco_sim import MuJoCoSimulator
        sim = MuJoCoSimulator(tailsitter(params=params))
        I = composite_inertia_from_model(sim.model)
        ctrl = TailsitterINDIController(params, inertia=I, dt=sim.dt,
                                        mass=sim.total_mass, use_sideslip_control=True,
                                        **ctrl_kw)
        alt, V = 10.0, 12.0
        legs = [MissionLeg(1.5, 0., 0., "hover"),
                MissionLeg(V*0.7, V, 0., "accel"),
                MissionLeg(4.0, V, 0., "cruise")]
        guid = MissionGuidance(legs, altitude=alt)
        sim.reset()
        sim.data.qpos[3:7] = np.asarray(_quat_from_axis_angle([0, 1, 0], -jnp.pi/2))
        sim.data.qpos[0:3] = np.array([0., 0., alt])
        mujoco.mj_forward(sim.model, sim.data)
        state = sim.get_state(); ctrl.reset(); guid.reset()
        dt = sim.dt
        alt_max = alt_min = alt
        for k in range(int(guid.total_time/dt)):
            ref = guid.step(dt)
            cmds = ctrl.update(state, jnp.asarray(ref.position),
                               setpoint_velocity=ref.velocity,
                               accel_feedforward=ref.acceleration,
                               desired_yaw=ref.yaw, yaw_rate_ff=ref.yaw_rate_ff)
            state = sim.step(cmds)
            assert bool(jnp.all(jnp.isfinite(state.position))), f"diverged t={k*dt:.2f}"
            if k*dt > 2.0:
                a = float(state.position[2]); alt_max = max(alt_max, a); alt_min = min(alt_min, a)
        thr = float(np.mean(np.clip(np.asarray(cmds)[:2], 0, 1)))
        b1 = np.asarray(quat_to_rotation_matrix(state.quaternion)[:, 0])
        pitch = float(np.degrees(np.arctan2(np.hypot(b1[0], b1[1]), b1[2])))
        return dict(thr=thr, pitch=pitch, vx=float(state.velocity[0]),
                    alt_excursion=alt_max-alt_min, alt=alt)

    def test_guidance_unloads_onto_wing_and_holds_altitude(self, params, sim_and_inertia):
        """The Phase D payoff: at 12 m/s cruise the Guidance INDI loop (jacobian)
        pitches the nose over, drops thrust (wing carries the weight), and holds
        altitude -- unlike the model-based loop, which keeps thrust high and
        climbs. Concrete gates: high pitch, low thrust, tight altitude."""
        r = self._fly_to_cruise(params, sim_and_inertia, use_guidance_indi=True,
                                guidance_effectiveness="jacobian")
        assert r["vx"] > 9.0, f"did not reach cruise speed: {r['vx']:.1f}"
        assert r["pitch"] > 60.0, f"did not pitch into wing-borne cruise: {r['pitch']:.0f} deg"
        assert r["thr"] < 0.40, f"thrust not unloaded onto wing: {r['thr']:.2f}"
        assert r["alt_excursion"] < 1.0, f"altitude not held: {r['alt_excursion']:.2f} m"

    def test_guidance_holds_altitude_better_than_model_loop(self, params, sim_and_inertia):
        """Head-to-head: guidance INDI holds altitude in the transition markedly
        better than the aerodynamically-blind model-based outer loop."""
        r_guid = self._fly_to_cruise(params, sim_and_inertia, use_guidance_indi=True)
        r_model = self._fly_to_cruise(params, sim_and_inertia)  # model-based default
        assert r_guid["alt_excursion"] < r_model["alt_excursion"], \
            f"guidance {r_guid['alt_excursion']:.2f} m should beat model {r_model['alt_excursion']:.2f} m"
        assert r_guid["thr"] < r_model["thr"], \
            f"guidance should cruise at lower thrust: {r_guid['thr']:.2f} vs {r_model['thr']:.2f}"

    def test_full_turning_mission_flies_coordinated(self, params, sim_and_inertia):
        """The turn fix: the full roadmap mission -- hover -> transition ->
        wing-borne cruise -> 180 deg coordinated turn -> transition -> hover --
        must fly on the guidance loop WITHOUT the pre-bank speed/altitude runaway
        (banking rolls the lift into the turn). Gates: speed tracks the reference
        (no runaway), altitude held through the turn, coordinated turn keeps
        sideslip small, and it returns to a stable hover."""
        import mujoco
        from uavsim.sim.mujoco_sim import MuJoCoSimulator
        from uavsim.controllers.tailsitter_guidance import roadmap_mission
        sim = MuJoCoSimulator(tailsitter(params=params))
        I = composite_inertia_from_model(sim.model)
        ctrl = TailsitterINDIController(params, inertia=I, dt=sim.dt,
                                        mass=sim.total_mass, use_sideslip_control=True,
                                        use_guidance_indi=True)
        alt, V = 10.0, 12.0
        guid = MissionGuidance(roadmap_mission(cruise_speed=V, altitude=alt), altitude=alt)
        sim.reset()
        sim.data.qpos[3:7] = np.asarray(_quat_from_axis_angle([0, 1, 0], -jnp.pi/2))
        sim.data.qpos[0:3] = np.array([0., 0., alt])
        mujoco.mj_forward(sim.model, sim.data)
        state = sim.get_state(); ctrl.reset(); guid.reset()
        dt = sim.dt
        vmax, alt_min, alt_max, betas = 0.0, alt, alt, []
        for k in range(int(guid.total_time/dt)):
            ref = guid.step(dt)
            cmds = ctrl.update(state, jnp.asarray(ref.position),
                               setpoint_velocity=ref.velocity,
                               accel_feedforward=ref.acceleration,
                               desired_yaw=ref.yaw, yaw_rate_ff=ref.yaw_rate_ff)
            state = sim.step(cmds)
            assert bool(jnp.all(jnp.isfinite(state.position))), f"diverged t={k*dt:.2f}"
            vmax = max(vmax, float(np.linalg.norm(np.asarray(state.velocity)[:2])))
            if k*dt > 2.0:
                a = float(state.position[2]); alt_min = min(alt_min, a); alt_max = max(alt_max, a)
            if ref.leg_name.startswith("cruise"):
                b, Vv, _ = estimate_sideslip(state)
                if Vv > 3.0:
                    betas.append(abs(np.degrees(b)))
        assert vmax < 18.0, f"speed ran away (missing bank?): vmax={vmax:.1f} m/s"
        assert (alt_max - alt_min) < 3.0, f"altitude not held through turn: {alt_max-alt_min:.1f} m"
        assert float(np.mean(betas)) < 8.0, f"turn not coordinated: mean |beta|={np.mean(betas):.1f} deg"
        b1 = np.asarray(quat_to_rotation_matrix(state.quaternion)[:, 0])
        assert b1[2] > 0.9, f"did not return to hover: b1.z={b1[2]:.2f}"
