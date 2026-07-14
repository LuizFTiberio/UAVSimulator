"""Propulsion models (motor + propeller) as pure JAX functions.

Each function is JIT-compatible and differentiable via jax.grad.

PropulsionModel.SIMPLE    — algebraic kt/km model (default, no extra deps)
PropulsionModel.BEM_LIVE  — CCBlade JAX solve per step (requires bem on PYTHONPATH)
PropulsionModel.BEM_TABLE — trilinear lookup into a pre-built T/Q table (fast path)
"""

from enum import Enum

import jax.numpy as jnp

from uavsim.core.types import MultirotorParams


class PropulsionModel(Enum):
    SIMPLE    = "simple"
    BEM_LIVE  = "bem_live"
    BEM_TABLE = "bem_table"


def compute_rotor_wrench(
    omega: jnp.ndarray,
    params: MultirotorParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute net body-frame force and torque from rotor speeds.

    Parameters
    ----------
    omega : (n_motors,) motor speeds [rad/s]
    params : MultirotorParams

    Returns
    -------
    force_body : (3,) net force in body frame [N]
    torque_body : (3,) net torque in body frame [N·m]
    """
    thrust = params.kt * omega ** 2                          # (n_motors,)

    # Net upward force (body Z axis)
    force_body = jnp.array([0.0, 0.0, jnp.sum(thrust)])

    # Torques from thrust moments: cross(arm_i, [0, 0, thrust_i])
    thrust_vectors = jnp.zeros_like(params.motor_positions).at[:, 2].set(thrust)
    moments = jnp.cross(params.motor_positions, thrust_vectors)  # (n, 3)
    torque_body = jnp.sum(moments, axis=0)                       # (3,)

    # Yaw drag torque
    yaw_drag = jnp.sum(params.rotor_yaw_sign * params.km * omega ** 2)
    torque_body = torque_body.at[2].add(yaw_drag)

    return force_body, torque_body


def throttle_to_omega(
    throttle: jnp.ndarray,
    max_omega: float,
) -> jnp.ndarray:
    """Convert normalised throttle [0, 1] to motor speed [rad/s]."""
    return jnp.clip(throttle, 0.0, 1.0) * max_omega


def compute_rotor_wrench_dispatch(
    omega: jnp.ndarray,
    v_air_body: jnp.ndarray,
    rotor_yaw_sign: jnp.ndarray,
    params: MultirotorParams,
    model: PropulsionModel = PropulsionModel.SIMPLE,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Route to the correct propulsion implementation.

    Parameters
    ----------
    omega         : (n_motors,) motor speeds [rad/s]
    v_air_body    : (3,) body-frame airspeed [m/s]  (ignored for SIMPLE)
    rotor_yaw_sign: (n_motors,) ±1                  (ignored for SIMPLE)
    params        : MultirotorParams; must have params.bem_config set for BEM modes
    model         : which implementation to use

    Returns
    -------
    force_body  : (3,) [N]
    torque_body : (3,) [N·m]
    """
    if model is PropulsionModel.SIMPLE:
        return compute_rotor_wrench(omega, params)

    if model is PropulsionModel.BEM_LIVE:
        from uavsim.dynamics.propulsion_bem import compute_rotor_wrench_bem
        return compute_rotor_wrench_bem(
            omega, v_air_body, rotor_yaw_sign, params.bem_config, params.motor_positions)

    if model is PropulsionModel.BEM_TABLE:
        from uavsim.dynamics.propulsion_bem import compute_rotor_wrench_bem_table
        return compute_rotor_wrench_bem_table(
            omega, v_air_body, rotor_yaw_sign, params.bem_config.bem_table, params.motor_positions
        )

    raise ValueError(f"Unknown PropulsionModel: {model!r}")
