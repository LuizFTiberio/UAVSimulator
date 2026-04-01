"""Propulsion models (motor + propeller) as pure JAX functions.

Each function is JIT-compatible and differentiable via jax.grad.
"""

import jax.numpy as jnp

from uavsim.core.types import MultirotorParams


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
