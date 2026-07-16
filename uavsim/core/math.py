"""Quaternion and rotation utilities in JAX."""

import jax.numpy as jnp


def safe_norm(x: jnp.ndarray, eps: float = 1e-9) -> jnp.ndarray:
    """Euclidean norm with a well-defined (finite) gradient at x=0.

    jnp.linalg.norm(x) is correct in VALUE everywhere (norm(0)=0), but its
    GRADIENT is NaN at exactly x=0 -- d|x|/dx = x/|x| is a genuine 0/0 at
    the origin, a real mathematical cusp, not a JAX quirk (checked
    directly: jax.grad(jnp.linalg.norm)(jnp.zeros(3)) -> [nan, nan, nan]).

    This matters throughout this simulator specifically because the
    physically important operating points -- hover (v=0 exactly) and
    trimmed axial flight (a velocity component exactly zero, e.g. zero
    lateral inflow into a propeller in pure axial flow) -- are exactly
    the points where some norm's argument lands on zero. Anyone
    differentiating the wrench w.r.t. STATE (not just commands) around
    those points -- gradient-based trim solving, MPC, RL, sysID -- will
    hit this. Found via a cruise-trim Newton/LM solve diverging to NaN;
    root-caused to this, not the solver (see chat history).

    Returns norm(x) to within ~eps at x=0 (negligible physically -- eps
    is far below any meaningful velocity/force scale in this simulator)
    in exchange for a finite, correct-to-first-order gradient everywhere,
    including at x=0 (where the true subgradient is any unit vector; this
    returns the zero vector's worth of sensitivity, i.e. d(safe_norm)/dx
    -> 0 as x->0, a defensible and stable convention).
    """
    return jnp.sqrt(jnp.sum(x ** 2) + eps ** 2)


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
