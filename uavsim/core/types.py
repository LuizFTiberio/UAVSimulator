"""JAX-compatible state and parameter containers.

All containers are NamedTuples so they work as JAX pytrees
out of the box (jit, grad, vmap, etc.).
"""

from typing import NamedTuple

import jax.numpy as jnp


# ── vehicle state ────────────────────────────────────────────────────────────

class VehicleState(NamedTuple):
    """Full rigid-body state of a vehicle."""
    position: jnp.ndarray          # (3,) world frame [m]
    quaternion: jnp.ndarray        # (4,) [w, x, y, z]
    velocity: jnp.ndarray          # (3,) world frame [m/s]
    angular_velocity: jnp.ndarray  # (3,) body frame [rad/s]
    time: float                    # simulation time [s]


# ── multirotor parameters ───────────────────────────────────────────────────

class MultirotorParams(NamedTuple):
    """Physical parameters for a multirotor vehicle."""
    mass: float                    # total mass [kg]
    gravity: float                 # gravitational acceleration [m/s²]
    max_omega: float               # maximum motor speed [rad/s]
    kt: float                      # thrust coefficient [N·s²/rad²]
    km: float                      # drag torque coefficient [N·m·s²/rad²]
    motor_positions: jnp.ndarray   # (n_motors, 3) body frame [m]
    rotor_yaw_sign: jnp.ndarray   # (n_motors,) +1 CCW, -1 CW
    arm_length: float              # motor arm length [m]


# ── PID controller ───────────────────────────────────────────────────────────

class PIDState(NamedTuple):
    """Mutable state for a PID controller."""
    integral: jnp.ndarray          # accumulated integral term
    step_count: jnp.ndarray        # scalar int, number of steps taken


class PIDGains(NamedTuple):
    """PID controller gains and limits."""
    kp: float
    ki: float
    kd: float
    max_output: float
    integral_limit: float


# ── hover controller ─────────────────────────────────────────────────────────

class HoverState(NamedTuple):
    """State for the cascaded hover controller."""
    pos_integral: jnp.ndarray      # (3,) position error integral
    att_pid: PIDState              # attitude PID state


class HoverGains(NamedTuple):
    """Gains for the cascaded hover controller."""
    kp_pos: float
    ki_pos: float
    kd_pos: float
    pos_integral_limit: jnp.ndarray  # (3,) per-axis limits
    att_gains: PIDGains
    max_tilt: float                  # maximum tilt angle [rad]
    min_alt: float                   # minimum altitude [m]
    min_thrust_ratio: float          # minimum thrust as fraction of hover [0-1]
