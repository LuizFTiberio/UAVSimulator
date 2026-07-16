"""Tests for the hover-only tail-sitter attitude controller."""

import jax.numpy as jnp
import numpy as np
import pytest

from uavsim.core.types import VehicleState
from uavsim.core.math import quat_to_rotation_matrix
from uavsim.vehicles.tailsitter import tailsitter_params
from uavsim.controllers.tailsitter_hover import (
    compute_hover_mixer,
    mix_hover,
    vee,
    attitude_error,
    desired_attitude_from_thrust_direction,
    default_tailsitter_hover_gains,
    tailsitter_hover_step,
    TailsitterHoverController,
)


def _quat_from_axis_angle(axis, angle):
    axis = jnp.asarray(axis, dtype=float)
    axis = axis / jnp.linalg.norm(axis)
    return jnp.concatenate([jnp.array([jnp.cos(angle / 2)]), jnp.sin(angle / 2) * axis])


def _state(position=(0., 0., 0.), velocity=(0., 0., 0.),
           angular_velocity=(0., 0., 0.), quaternion=(1., 0., 0., 0.)):
    return VehicleState(
        position=jnp.array(position, dtype=float),
        quaternion=jnp.array(quaternion, dtype=float),
        velocity=jnp.array(velocity, dtype=float),
        angular_velocity=jnp.array(angular_velocity, dtype=float),
        time=0.0,
    )


@pytest.fixture
def params():
    return tailsitter_params()


@pytest.fixture
def mixer(params):
    return compute_hover_mixer(params)


# ── mixer ─────────────────────────────────────────────────────────────────────

class TestMixer:

    def test_trim_output_recovers_trim_command(self, mixer):
        """Requesting exactly the trim output should recover ~trim commands."""
        cmds = mix_hover(mixer.output0, mixer)
        assert jnp.allclose(cmds, mixer.trim_cmds, atol=1e-4)

    def test_matrix_well_conditioned(self, mixer):
        cond = jnp.linalg.cond(jnp.linalg.inv(mixer.B_inv))
        assert float(cond) < 1000.0, f"Effectiveness matrix poorly conditioned: {float(cond)}"

    def test_pure_thrust_request_only_moves_throttle(self, mixer):
        """Requesting more thrust with zero moment should mostly move
        throttle symmetrically, not elevons (thrust row has zero elevon
        entries per the Jacobian structure)."""
        desired = mixer.output0 + jnp.array([2.0, 0.0, 0.0, 0.0])
        cmds = mix_hover(desired, mixer)
        assert float(jnp.abs(cmds[0] - cmds[1])) < 1e-3, "should stay symmetric throttle"

    def test_recovers_desired_moments_within_linear_region(self, params, mixer):
        """For small moment requests near trim, the linearized mixer should
        actually produce close to the requested moments when fed back
        through the real (nonlinear) wrench."""
        from uavsim.vehicles.tailsitter import tailsitter_wrench
        trim_state = VehicleState(position=jnp.zeros(3), quaternion=jnp.array([1., 0., 0., 0.]),
                                   velocity=jnp.zeros(3), angular_velocity=jnp.zeros(3), time=0.0)
        desired = mixer.output0 + jnp.array([0.0, 0.05, 0.02, 0.05])
        cmds = mix_hover(desired, mixer)
        F, T = tailsitter_wrench(trim_state, cmds, params)
        actual = jnp.array([F[0], T[0], T[1], T[2]])
        assert float(jnp.max(jnp.abs(actual - desired))) < 0.01


# ── attitude error (the actual gimbal-lock fix) ──────────────────────────────

class TestAttitudeError:

    def test_zero_error_when_aligned(self):
        R = jnp.eye(3)
        e = attitude_error(R, R)
        assert float(jnp.max(jnp.abs(e))) < 1e-10

    def test_vee_inverts_skew(self):
        from uavsim.aero.phi_theory import skew
        v = jnp.array([0.3, -0.7, 1.1])
        assert float(jnp.max(jnp.abs(vee(skew(v)) - v))) < 1e-10

    def test_no_singularity_at_90_degree_pitch(self):
        """This is the actual bug the existing HoverController has: Euler
        angles gimbal-lock at pitch=90deg, exactly the hover attitude.
        Rotation-matrix error must stay well-defined and finite there,
        AND must remain sensitive to roll/yaw perturbations applied at
        that attitude (gimbal lock's symptom is roll/yaw becoming
        indistinguishable -- check they're NOT indistinguishable here)."""
        q_hover = _quat_from_axis_angle([0, 1, 0], -jnp.pi / 2)  # nose-up: -90 about +Y
        R_hover = quat_to_rotation_matrix(q_hover)
        R_des = R_hover  # aligned

        # Perturb by a small roll vs a small yaw, from the SAME hover attitude
        q_roll_pert = _quat_from_axis_angle([1, 0, 0], 0.05)
        q_yaw_pert = _quat_from_axis_angle([0, 0, 1], 0.05)
        # Compose: apply small body-frame perturbation on top of hover attitude
        R_roll = R_hover @ quat_to_rotation_matrix(q_roll_pert)
        R_yaw = R_hover @ quat_to_rotation_matrix(q_yaw_pert)

        e_roll = attitude_error(R_roll, R_des)
        e_yaw = attitude_error(R_yaw, R_des)

        assert bool(jnp.all(jnp.isfinite(e_roll)))
        assert bool(jnp.all(jnp.isfinite(e_yaw)))
        # The two perturbations must produce genuinely DIFFERENT error
        # vectors -- if they were nearly identical / degenerate, that
        # would be the gimbal-lock symptom (roll and yaw indistinguishable).
        diff = float(jnp.linalg.norm(e_roll - e_yaw))
        assert diff > 1e-3, f"roll and yaw perturbations indistinguishable at hover attitude: diff={diff}"

    def test_error_magnitude_scales_with_angle(self):
        """Sanity: larger misalignment -> larger error norm (monotonic-ish
        check, small vs large angle about the same axis)."""
        R = jnp.eye(3)
        q_small = _quat_from_axis_angle([0, 0, 1], 0.1)
        q_large = _quat_from_axis_angle([0, 0, 1], 0.8)
        e_small = attitude_error(quat_to_rotation_matrix(q_small), R)
        e_large = attitude_error(quat_to_rotation_matrix(q_large), R)
        assert float(jnp.linalg.norm(e_large)) > float(jnp.linalg.norm(e_small))


class TestDesiredAttitudeConstruction:

    def test_orthonormal_rotation_matrix(self):
        b1 = jnp.array([0.1, 0.0, 0.995])
        b1 = b1 / jnp.linalg.norm(b1)
        R_des = desired_attitude_from_thrust_direction(b1, desired_yaw=0.3)
        should_be_I = R_des.T @ R_des
        assert jnp.allclose(should_be_I, jnp.eye(3), atol=1e-5)
        assert jnp.allclose(jnp.linalg.det(R_des), 1.0, atol=1e-5)

    def test_first_column_is_thrust_direction(self):
        b1 = jnp.array([0.2, 0.1, 0.97])
        b1 = b1 / jnp.linalg.norm(b1)
        R_des = desired_attitude_from_thrust_direction(b1, desired_yaw=0.0)
        assert jnp.allclose(R_des[:, 0], b1, atol=1e-6)


# ── closed-loop hover hold in MuJoCo ──────────────────────────────────────────

class TestClosedLoopHover:

    def test_recovers_from_identity_attitude_to_hover(self, params, mixer):
        """The vehicle spawns nose-forward (identity quaternion, matching
        the MJCF); hover attitude is nose-UP. This is a large (~90 degree)
        attitude maneuver -- exactly what Euler-angle control cannot do
        cleanly. Check the controller drives it to (roughly) hover attitude
        and holds position, without diverging."""
        from uavsim.sim.mujoco_sim import MuJoCoSimulator
        from uavsim.vehicles.tailsitter import tailsitter

        gains = default_tailsitter_hover_gains()
        vehicle = tailsitter(params=params)
        sim = MuJoCoSimulator(vehicle)
        state = sim.reset(position=np.array([0.0, 0.0, 2.0]))

        setpoint = jnp.array([0.0, 0.0, 2.0])
        n_steps = 4000  # 4 s at dt=0.001

        for _ in range(n_steps):
            cmds = tailsitter_hover_step(state, setpoint, params, mixer, gains)
            state = sim.step(cmds)
            assert bool(jnp.all(jnp.isfinite(state.position))), "diverged"
            assert bool(jnp.all(jnp.isfinite(state.quaternion))), "diverged"

        # After 4s, should be within a couple meters of the setpoint and
        # not tumbling wildly (loose bounds -- this is a sanity gate on the
        # aero model + controller stack, not a tuned-controller precision
        # check).
        pos_err = float(jnp.linalg.norm(state.position - setpoint))
        ang_rate = float(jnp.linalg.norm(state.angular_velocity))
        assert pos_err < 2.0, f"Failed to converge near setpoint: pos_err={pos_err:.2f} m"
        assert ang_rate < 3.0, f"Still tumbling after 4s: |omega|={ang_rate:.2f} rad/s"

        # body +x should be close to world +z (hover attitude reached)
        R = quat_to_rotation_matrix(state.quaternion)
        b1 = R[:, 0]
        alignment = float(jnp.dot(b1, jnp.array([0.0, 0.0, 1.0])))
        assert alignment > 0.7, f"Did not reach hover attitude: b1.z={alignment:.2f}"

    def test_holds_position_from_near_hover_start(self, params, mixer):
        """Starting already near hover attitude with a small position
        offset, should converge and hold with small steady-state error.

        Note: -90deg about +Y (not +90) gives nose-UP (body +x -> world
        +z). Got this backwards on the first attempt -- easy sign mistake,
        worth the explicit check below so it can't silently regress."""
        from uavsim.sim.mujoco_sim import MuJoCoSimulator
        from uavsim.vehicles.tailsitter import tailsitter

        gains = default_tailsitter_hover_gains()
        vehicle = tailsitter(params=params)
        sim = MuJoCoSimulator(vehicle)

        q_hover = _quat_from_axis_angle([0, 1, 0], -jnp.pi / 2)
        R_check = quat_to_rotation_matrix(q_hover)
        assert float(R_check[:, 0][2]) > 0.9, "test setup bug: this quaternion is not nose-up"

        sim.data.qpos[3:7] = np.asarray(q_hover)
        sim.data.qpos[0:3] = np.array([0.5, 0.3, 2.0])
        import mujoco
        mujoco.mj_forward(sim.model, sim.data)
        state = sim.get_state()

        setpoint = jnp.array([0.0, 0.0, 2.0])
        for _ in range(3000):  # 3 s
            cmds = tailsitter_hover_step(state, setpoint, params, mixer, gains)
            state = sim.step(cmds)
            assert bool(jnp.all(jnp.isfinite(state.position)))

        pos_err = float(jnp.linalg.norm(state.position - setpoint))
        assert pos_err < 0.5, f"Did not hold position: pos_err={pos_err:.2f} m"


# ── factory / stateful wrapper ────────────────────────────────────────────────

class TestControllerWrapper:

    def test_wrapper_outputs_valid_commands(self, params):
        ctrl = TailsitterHoverController(params)
        state = _state(position=(0.1, 0.0, 1.9))
        cmds = ctrl.update(state, jnp.array([0.0, 0.0, 2.0]))
        assert cmds.shape == (4,)
        assert bool(jnp.all(jnp.isfinite(cmds)))
        assert float(jnp.min(cmds[0:2])) >= 0.0
        assert float(jnp.max(cmds[0:2])) <= 1.0
