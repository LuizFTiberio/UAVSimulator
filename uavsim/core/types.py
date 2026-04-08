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


# ── wing / aerodynamic parameters ────────────────────────────────────────────

class WingParams(NamedTuple):
    """Aerodynamic parameters for a fixed wing."""
    wingspan: float                # wing span [m]
    aspect_ratio: float            # AR = b² / S
    wing_area: float               # S [m²]
    rho: float                     # air density [kg/m³]
    CL0: float                     # zero-AoA lift coefficient
    CLa: float                     # lift-curve slope [1/rad]
    CD0: float                     # zero-lift drag coefficient
    oswald: float                  # Oswald efficiency factor
    alpha_max: float               # stall clamp [rad]
    transition_speed: float        # wing-on airspeed [m/s]
    transition_sharpness: float    # sigmoid steepness [1/(m/s)]
    CL_delta_f: float              # lift increment per rad flap [1/rad]
    max_flap: float                # max flap deflection [rad]
    Cl_delta_a: float              # roll moment coeff per rad aileron [1/rad]
    max_aileron: float             # max aileron deflection [rad]


# ── pusher motor parameters ───────────────────────────────────────────────────

class PusherParams(NamedTuple):
    """Parameters for a forward-thrust pusher motor."""
    kt: float                      # thrust coefficient [N·s²/rad²]
    max_omega: float               # maximum motor speed [rad/s]


# ── quadplane parameters ─────────────────────────────────────────────────────

class QuadplaneParams(NamedTuple):
    """Parameters for a quadplane (multirotor + fixed wing + pusher)."""
    rotor: MultirotorParams        # multirotor subsystem
    wing: WingParams               # fixed-wing subsystem
    pusher: PusherParams           # forward-thrust pusher motor


# ── body drag parameters ─────────────────────────────────────────────────────

class BodyDragParams(NamedTuple):
    """Bluff-body aerodynamic drag parameters.

    Used by multirotors (and any vehicle) to model translational drag
    as a function of airspeed: F_drag = -½ ρ Cd A |v_air| v_air.
    """
    rho: float = 1.225             # air density [kg/m³]
    Cd: float = 1.0                # drag coefficient (approx. flat plate for quad frame)
    frontal_area: float = 0.04     # effective frontal area [m²]


# ── Dryden wind parameters ───────────────────────────────────────────────────

class DrydenParams(NamedTuple):
    """MIL-HDBK-1797 Dryden turbulence model parameters.

    Scale lengths and intensities are for a fixed altitude band.
    For altitude-varying profiles, update these per step.
    """
    sigma_u: float = 1.06         # longitudinal turbulence intensity [m/s]
    sigma_v: float = 1.06         # lateral turbulence intensity [m/s]
    sigma_w: float = 0.7          # vertical turbulence intensity [m/s]
    Lu: float = 200.0             # longitudinal scale length [m]
    Lv: float = 200.0             # lateral scale length [m]
    Lw: float = 50.0              # vertical scale length [m]
