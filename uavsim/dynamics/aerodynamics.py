"""Aerodynamic models (wing, body drag) as pure JAX functions."""

import jax.numpy as jnp

from uavsim.core.types import VehicleState, WingParams
from uavsim.core.math import quat_to_rotation_matrix


def compute_wing_wrench(
    state: VehicleState,
    wing: WingParams,
    flap: float = 0.0,
    aileron: float = 0.0,
    wind_velocity: jnp.ndarray = jnp.zeros(3),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute world-frame aerodynamic force from a fixed wing.

    Lift and drag are computed in the body longitudinal (xz) plane.
    Flap deflection increases CL (symmetric, both sides down).
    Aileron deflection produces a rolling moment about the body X axis.

    The wing activates smoothly via a sigmoid centred at
    ``wing.transition_speed``.

    Parameters
    ----------
    state : VehicleState
    wing : WingParams
    flap : float
        Normalised flap command [0, 1].  0 = retracted, 1 = full.
    aileron : float
        Normalised aileron command [-1, 1].  Positive = roll right.
    wind_velocity : (3,) world-frame wind velocity [m/s]

    Returns
    -------
    F_world : (3,) aerodynamic force in world frame [N]
    T_world : (3,) aerodynamic torque in world frame [N·m]
    """
    R = quat_to_rotation_matrix(state.quaternion)
    # Body-frame airspeed (relative to the air mass)
    v_air_world = state.velocity - wind_velocity
    v_body = R.T @ v_air_world                              # (3,)

    u = v_body[0]   # forward
    w = v_body[2]   # downward (body z)

    # Airspeed in the longitudinal plane
    V_lon = jnp.sqrt(u ** 2 + w ** 2)
    V_total = jnp.linalg.norm(v_air_world)

    # Angle of attack (clamped for linear regime)
    alpha_raw = jnp.arctan2(w, u)
    alpha = jnp.clip(alpha_raw, -wing.alpha_max, wing.alpha_max)

    # Aerodynamic coefficients
    CL = wing.CL0 + wing.CLa * alpha

    # Flap contribution: symmetric deflection increases CL
    delta_f = jnp.clip(flap, 0.0, 1.0) * wing.max_flap
    CL = CL + wing.CL_delta_f * delta_f

    CD = wing.CD0 + CL ** 2 / (jnp.pi * wing.aspect_ratio * wing.oswald)

    # Dynamic pressure
    q = 0.5 * wing.rho * V_lon ** 2

    # Body-frame lift (perpendicular to V in xz) and drag (opposing V in xz)
    # Lift direction in body frame: rotate velocity 90° in xz plane
    L_mag = q * wing.wing_area * CL
    D_mag = q * wing.wing_area * CD

    # Unit vectors in body xz plane
    # Drag opposes velocity, lift is perpendicular (upward-ish)
    drag_dir_body = jnp.array([-u, 0.0, -w]) / jnp.maximum(V_lon, 1e-8)
    lift_dir_body = jnp.array([-w, 0.0,  u]) / jnp.maximum(V_lon, 1e-8)

    F_body = L_mag * lift_dir_body + D_mag * drag_dir_body

    # Smooth activation sigmoid: σ(V) = 1 / (1 + exp(-k*(V - V0)))
    sigma = jnp.where(
        V_total < 1e-3,
        0.0,
        1.0 / (1.0 + jnp.exp(-wing.transition_sharpness
                               * (V_total - wing.transition_speed))),
    )

    F_body = sigma * F_body

    # ── aileron rolling moment (body X-axis) ─────────────────────────────
    # L_roll = q * S * b * Cl_delta_a * delta_a
    delta_a = jnp.clip(aileron, -1.0, 1.0) * wing.max_aileron
    L_roll = q * wing.wing_area * wing.wingspan * wing.Cl_delta_a * delta_a
    T_body = sigma * jnp.array([L_roll, 0.0, 0.0])

    # Rotate to world frame
    F_world = R @ F_body
    T_world = R @ T_body

    return F_world, T_world
