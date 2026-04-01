"""PID controller — functional JAX style.

Core functions (pid_init, pid_step) are JIT-compatible and differentiable.
"""

import jax.numpy as jnp

from uavsim.core.types import PIDState, PIDGains


def pid_init(dim: int) -> PIDState:
    """Create a fresh PID state for *dim*-dimensional control."""
    return PIDState(
        integral=jnp.zeros(dim),
        step_count=jnp.array(0, dtype=jnp.int32),
    )


def pid_step(
    state: PIDState,
    error: jnp.ndarray,
    rate: jnp.ndarray,
    dt: float,
    gains: PIDGains,
) -> tuple[PIDState, jnp.ndarray]:
    """Advance the PID by one timestep.

    Parameters
    ----------
    state : PIDState
    error : (dim,) tracking error (desired - current)
    rate : (dim,) measurement rate (angular velocity or velocity).
           Used as negative derivative feedback (rate damping).
    dt : timestep [s]
    gains : PIDGains

    Returns
    -------
    new_state : PIDState
    output : (dim,) control output (e.g. torque)
    """
    # Integral with anti-windup
    integral = state.integral + error * dt
    integral = jnp.clip(integral, -gains.integral_limit, gains.integral_limit)

    # Derivative: use negative rate (rate damping).
    # On the very first step, zero the d-term to avoid a kick.
    d_term = jnp.where(state.step_count > 0, -rate, jnp.zeros_like(rate))

    output = gains.kp * error + gains.ki * integral + gains.kd * d_term
    output = jnp.clip(output, -gains.max_output, gains.max_output)

    new_state = PIDState(
        integral=integral,
        step_count=state.step_count + 1,
    )
    return new_state, output
