"""Cascaded hover controller (position -> attitude -> motors).

Provides both a functional API (hover_init / hover_step) for JAX
and a stateful wrapper (HoverController) for interactive use.
"""

from functools import partial

import jax
import jax.numpy as jnp

from uavsim.core.types import (
    HoverGains, HoverState, MultirotorParams, PIDGains, PIDState, VehicleState,
)
from uavsim.core.math import euler_from_quaternion, wrap_angle
from uavsim.controllers.pid import pid_init, pid_step
from uavsim.controllers.mixer import mix_x4


# ── defaults ─────────────────────────────────────────────────────────────────

def default_hover_gains() -> HoverGains:
    """Default gains tuned for the 1.2 kg X-config quadcopter."""
    return HoverGains(
        kp_pos=3.0,
        ki_pos=0.1,
        kd_pos=3.0,
        pos_integral_limit=jnp.array([1.0, 1.0, 0.5]),
        att_gains=PIDGains(kp=8.0, ki=0.2, kd=3.0,
                           max_output=1.5, integral_limit=0.3),
        max_tilt=jnp.float32(jnp.deg2rad(30.0)),
        min_alt=0.12,
        min_thrust_ratio=0.3,
    )


# ── functional API ───────────────────────────────────────────────────────────

def hover_init(gains: HoverGains | None = None) -> HoverState:
    """Create a fresh hover controller state."""
    return HoverState(
        pos_integral=jnp.zeros(3),
        att_pid=pid_init(3),
    )


def hover_step(
    state: HoverState,
    vehicle: VehicleState,
    setpoint: jnp.ndarray,
    params: MultirotorParams,
    gains: HoverGains,
    dt: float,
    desired_yaw: float = 0.0,
) -> tuple[HoverState, jnp.ndarray]:
    """Advance the hover controller by one timestep.

    Parameters
    ----------
    state : HoverState
    vehicle : VehicleState (current vehicle state)
    setpoint : (3,) desired [x, y, z]
    params : MultirotorParams
    gains : HoverGains
    dt : timestep [s]
    desired_yaw : heading setpoint [rad]

    Returns
    -------
    new_state : HoverState
    throttle : (4,) motor throttle [0, 1]
    """
    attitude = euler_from_quaternion(vehicle.quaternion)
    position = vehicle.position
    velocity = vehicle.velocity
    ang_vel = vehicle.angular_velocity

    # ── ground avoidance ─────────────────────────────────────────────────
    safe_setpoint = setpoint.at[2].set(jnp.maximum(setpoint[2], gains.min_alt))

    # ── position PID (outer loop) ────────────────────────────────────────
    pos_err = safe_setpoint - position

    # Freeze integrator near the ground to prevent windup
    on_ground = position[2] < gains.min_alt + 0.05
    raw_integral = state.pos_integral + pos_err * dt
    raw_integral = jnp.clip(raw_integral,
                            -gains.pos_integral_limit, gains.pos_integral_limit)
    pos_integral = jnp.where(on_ground, state.pos_integral, raw_integral)

    accel_cmd = (gains.kp_pos * pos_err
                 + gains.ki_pos * pos_integral
                 + gains.kd_pos * (-velocity))

    # ── desired attitude from commanded XY acceleration ──────────────────
    desired_roll = jnp.clip(-accel_cmd[1] / params.gravity,
                            -gains.max_tilt, gains.max_tilt)
    desired_pitch = jnp.clip(accel_cmd[0] / params.gravity,
                             -gains.max_tilt, gains.max_tilt)
    desired_att = jnp.array([desired_roll, desired_pitch, desired_yaw])

    # ── total thrust with tilt compensation ──────────────────────────────
    tilt = jnp.abs(attitude[0]) + jnp.abs(attitude[1])
    cos_t = jnp.maximum(jnp.cos(tilt), 0.5)
    alt_accel = accel_cmd[2] + params.gravity
    hover_thrust = params.mass * params.gravity
    max_thrust = 4.0 * params.kt * params.max_omega ** 2
    min_thrust = gains.min_thrust_ratio * hover_thrust
    thrust_n = jnp.clip(params.mass * alt_accel / cos_t, min_thrust, max_thrust)

    # ── attitude PID (inner loop) ────────────────────────────────────────
    att_error = desired_att - attitude
    att_error = att_error.at[2].set(wrap_angle(att_error[2]))
    new_att_pid, torques = pid_step(
        state.att_pid, att_error, ang_vel, dt, gains.att_gains)

    # ── mix to motors ────────────────────────────────────────────────────
    max_thrust_per_motor = params.kt * params.max_omega ** 2
    throttle = mix_x4(thrust_n, torques[0], torques[1], torques[2],
                      params.arm_length, max_thrust_per_motor)

    new_state = HoverState(pos_integral=pos_integral, att_pid=new_att_pid)
    return new_state, throttle


# ── stateful wrapper ─────────────────────────────────────────────────────────

class HoverController:
    """Stateful wrapper around hover_init / hover_step for interactive use."""

    def __init__(
        self,
        params: MultirotorParams,
        gains: HoverGains | None = None,
    ):
        self.params = params
        self.gains = gains if gains is not None else default_hover_gains()
        self._state = hover_init(self.gains)

        # JIT-compile with params and gains baked in
        self._step_jit = jax.jit(
            partial(hover_step, params=params, gains=self.gains))

    def reset(self) -> None:
        self._state = hover_init(self.gains)

    def update(
        self,
        vehicle: VehicleState,
        setpoint: jnp.ndarray,
        dt: float,
        desired_yaw: float = 0.0,
    ) -> jnp.ndarray:
        """Compute motor throttle for one timestep. Returns (4,) throttle."""
        self._state, throttle = self._step_jit(
            self._state, vehicle, setpoint, dt=dt, desired_yaw=desired_yaw)
        return throttle
