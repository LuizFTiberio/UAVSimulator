"""Aerodynamic models (wing, body drag) as pure JAX functions.

Stub — to be implemented with analytical models or surrogates.
"""

import jax.numpy as jnp

from uavsim.core.types import VehicleState


def compute_aero_wrench(
    state: VehicleState,
    aero_params: ...,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute body-frame aerodynamic force and torque.

    Parameters
    ----------
    state : VehicleState
    aero_params : aerodynamic parameter container (TBD)

    Returns
    -------
    force_body : (3,) aerodynamic force in body frame [N]
    torque_body : (3,) aerodynamic torque in body frame [N·m]
    """
    raise NotImplementedError(
        "Aerodynamic models not yet implemented. "
        "Planned: panel methods, VLM, surrogate neural nets."
    )
