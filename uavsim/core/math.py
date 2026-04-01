"""Quaternion and rotation utilities in JAX."""

import jax.numpy as jnp


def quat_to_rotation_matrix(q: jnp.ndarray) -> jnp.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix (body -> world)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def euler_from_quaternion(q: jnp.ndarray) -> jnp.ndarray:
    """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw]."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    roll = jnp.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = jnp.clip(2*(w*y - z*x), -1.0, 1.0)
    pitch = jnp.arcsin(sinp)
    yaw = jnp.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return jnp.array([roll, pitch, yaw])


def wrap_angle(a: jnp.ndarray) -> jnp.ndarray:
    """Wrap angle(s) to (-pi, pi]."""
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))
