"""Hover-only attitude controller for the tail-sitter.

Scope (deliberately limited): position-hold in the hover regime (nose-up,
near-zero airspeed), as a diagnostic/sanity gate on the phi-theory aero
model -- NOT a transition or cruise controller. See discussion in chat
history for why: the existing HoverController/QuadplaneTrajectoryController
(uavsim/controllers/hover.py, quadplane_ctrl.py) cannot be reused here.
They compute attitude error from Euler angles (euler_from_quaternion),
which hits gimbal lock at pitch = +/-90 degrees -- exactly the tail-sitter's
NOMINAL hover attitude, not an edge case. This controller instead uses a
standard rotation-matrix attitude error (Lee, Leok, McClamroch-style
geometric control on SO(3)), which has no such singularity for any
attitude short of a 180-degree error.

Control allocation (throttle_left, throttle_right, elevon_left, elevon_right
-> thrust + roll/pitch/yaw moment) is NOT hand-decomposed per channel.
Jacobian inspection (see chat history) shows one genuinely coupled axis:
roll receives contributions from BOTH differential throttle (reaction
torque) and differential elevon, while thrust/pitch/yaw are each driven by
exactly one channel-pair. Rather than hand-designing a decoupling that
gets the roll coupling wrong, this module computes the linearized
control-effectiveness matrix directly via jax.jacobian at a trim point and
inverts it -- exact within the linear region around trim, and automatically
correct for whatever the actual coupling structure is, including for
different geometry (e.g. if you change delta_r_fraction, prop_y_offset,
etc. in tailsitter_params).
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from uavsim.core.types import VehicleState
from uavsim.core.math import quat_to_rotation_matrix
from uavsim.vehicles.tailsitter import TailsitterParams, tailsitter_wrench


# ── control-allocation mixer (linearized at trim via autodiff) ──────────────

class HoverMixer(NamedTuple):
    """Linearized control-effectiveness data, computed once per vehicle."""
    trim_cmds: jnp.ndarray   # (4,) motor_commands at the linearization point
    output0: jnp.ndarray     # (4,) [thrust_x, Mx, My, Mz] (body frame) at trim
    B_inv: jnp.ndarray       # (4,4) inverse Jacobian d[cmds]/d[thrust,Mx,My,Mz]


def compute_hover_mixer(params: TailsitterParams) -> HoverMixer:
    """Compute the linearized [thrust, Mx, My, Mz] -> commands mixer.

    Evaluated at identity attitude / zero velocity so the wrench is
    expressed directly in body frame (R=I => F_world=F_body), and at the
    vehicle's ACTUAL hover-trim throttle (solved from mass/kf/gravity,
    symmetric, zero elevon) -- not a hardcoded 50%.

    Earlier version hardcoded 0.5 here, which only happened to be correct
    for tailsitter_params()'s default (kf back-derived specifically to
    make 50% = hover trim). Once kf comes from real data (e.g.
    cyclone_hover_capable_params(), fit from an actual BEM solve), hover
    trim can land anywhere -- for that vehicle it's ~75%. Thrust is convex
    in throttle (kf*omega^2), so linearizing at a point far from the
    actual operating throttle systematically UNDERESTIMATES the true
    thrust at any commanded throttle above the linearization point
    (tangent line is a global under-estimator of a convex function) --
    commanding a large throttle jump from a low linearization point
    overshoots the actual thrust delivered. This caused a real runaway
    climb in closed-loop testing before the fix (see chat history).
    """
    trim_state = VehicleState(
        position=jnp.zeros(3), quaternion=jnp.array([1.0, 0.0, 0.0, 0.0]),
        velocity=jnp.zeros(3), angular_velocity=jnp.zeros(3), time=0.0)

    hover_throttle = jnp.sqrt(
        (params.mass * params.gravity) / (2.0 * params.propulsion.kf)
    ) / params.propulsion.max_omega
    hover_throttle = jnp.clip(hover_throttle, 0.05, 0.95)
    trim_cmds = jnp.array([hover_throttle, hover_throttle, 0.0, 0.0])

    def outputs(cmds):
        F, T = tailsitter_wrench(trim_state, cmds, params)
        return jnp.array([F[0], T[0], T[1], T[2]])

    B = jax.jacobian(outputs)(trim_cmds)
    B_inv = jnp.linalg.inv(B)
    output0 = outputs(trim_cmds)
    return HoverMixer(trim_cmds=trim_cmds, output0=output0, B_inv=B_inv)


def mix_hover(desired_thrust_Mx_My_Mz: jnp.ndarray, mixer: HoverMixer) -> jnp.ndarray:
    """Map desired [thrust, Mx, My, Mz] (body frame) to motor_commands.

    Linear within the region around trim where B is a good approximation
    of the (actually bilinear -- thrust*elevon) true effectiveness; for
    large excursions this will be less accurate. Fine for hover-hold near
    trim; would need re-linearizing (or a nonlinear solve) for anything
    more aggressive.
    """
    delta_cmds = mixer.B_inv @ (desired_thrust_Mx_My_Mz - mixer.output0)
    cmds = mixer.trim_cmds + delta_cmds
    cmds = cmds.at[0:2].set(jnp.clip(cmds[0:2], 0.0, 1.0))
    cmds = cmds.at[2:4].set(jnp.clip(cmds[2:4], -1.0, 1.0))
    return cmds


# ── geometric (rotation-matrix) attitude error ───────────────────────────────

def vee(M: jnp.ndarray) -> jnp.ndarray:
    """Inverse of the skew-symmetric map (uavsim.aero.phi_theory.skew)."""
    return jnp.array([M[2, 1], M[0, 2], M[1, 0]])


def attitude_error(R: jnp.ndarray, R_des: jnp.ndarray) -> jnp.ndarray:
    """Rotation-matrix attitude error (Lee, Leok, McClamroch), body frame.

    e_R = 0.5 * vee(R_des^T @ R - R^T @ R_des)

    Non-singular for any attitude error short of exactly 180 degrees --
    unlike Euler-angle error, which is singular at pitch=+/-90 degrees
    (the tail-sitter's own hover attitude). e_R = 0 iff R = R_des.
    """
    A = R_des.T @ R
    return 0.5 * vee(A - A.T)


def desired_attitude_from_thrust_direction(
    b1_des: jnp.ndarray, desired_yaw: float,
) -> jnp.ndarray:
    """Build a full desired rotation matrix from a desired body +x (thrust)
    direction plus a heading reference, TRIAD-style.

    b1_des is the only translationally-meaningful direction (thrust is
    along body +x); the rotation about b1_des is otherwise free, so a
    horizontal heading reference resolves it (matches how VTOL hover-yaw
    is conventionally defined when the thrust axis itself is vertical).

    Degenerates if b1_des is itself horizontal (b1_des parallel to the
    heading reference) -- acceptable for hover-only scope (b1_des stays
    close to vertical near trim); would need a fallback reference for any
    large-tilt / transition use.
    """
    h = jnp.array([jnp.cos(desired_yaw), jnp.sin(desired_yaw), 0.0])
    b2_des = jnp.cross(b1_des, h)
    b2_des = b2_des / jnp.maximum(jnp.linalg.norm(b2_des), 1e-6)
    b3_des = jnp.cross(b1_des, b2_des)
    return jnp.stack([b1_des, b2_des, b3_des], axis=1)  # columns = body axes in world


# ── gains / state ─────────────────────────────────────────────────────────────

class TailsitterHoverGains(NamedTuple):
    kp_pos: jnp.ndarray   # (3,) position proportional gain
    kd_pos: jnp.ndarray   # (3,) velocity (derivative) gain
    kR: float             # attitude proportional gain
    kOmega: float         # angular-rate damping gain


def default_tailsitter_hover_gains() -> TailsitterHoverGains:
    return TailsitterHoverGains(
        kp_pos=jnp.array([2.0, 2.0, 4.0]),
        kd_pos=jnp.array([2.5, 2.5, 3.5]),
        kR=6.0,
        kOmega=1.2,
    )


# ── step function ────────────────────────────────────────────────────────────

def tailsitter_hover_step(
    state: VehicleState,
    setpoint_position: jnp.ndarray,
    params: TailsitterParams,
    mixer: HoverMixer,
    gains: TailsitterHoverGains,
    desired_yaw: float = 0.0,
) -> jnp.ndarray:
    """Compute (4,) motor_commands for one control step.

    Position PD -> desired acceleration -> desired thrust vector (world) ->
    desired attitude (thrust direction + heading) -> rotation-matrix
    attitude error -> PD torque command -> linearized mixer -> commands.
    """
    pos_err = setpoint_position - state.position
    accel_cmd = gains.kp_pos * pos_err + gains.kd_pos * (-state.velocity)

    g_vec = jnp.array([0.0, 0.0, -params.gravity])
    F_thrust_world = params.mass * (accel_cmd - g_vec)
    T_cmd = jnp.linalg.norm(F_thrust_world)
    b1_des = F_thrust_world / jnp.maximum(T_cmd, 1e-6)

    R = quat_to_rotation_matrix(state.quaternion)
    R_des = desired_attitude_from_thrust_direction(b1_des, desired_yaw)

    e_R = attitude_error(R, R_des)
    M_cmd = -gains.kR * e_R - gains.kOmega * state.angular_velocity

    desired_output = jnp.concatenate([T_cmd[None], M_cmd])
    return mix_hover(desired_output, mixer)


# ── stateful wrapper (mirrors HoverController's interface style) ────────────

class TailsitterHoverController:
    def __init__(
        self,
        params: TailsitterParams,
        gains: TailsitterHoverGains | None = None,
    ):
        self.params = params
        self.gains = gains if gains is not None else default_tailsitter_hover_gains()
        self.mixer = compute_hover_mixer(params)
        self._step_jit = jax.jit(
            partial(tailsitter_hover_step, params=params, mixer=self.mixer, gains=self.gains))

    def update(
        self,
        vehicle: VehicleState,
        setpoint_position: jnp.ndarray,
        desired_yaw: float = 0.0,
    ) -> jnp.ndarray:
        return self._step_jit(vehicle, setpoint_position, desired_yaw=desired_yaw)
