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
)


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
