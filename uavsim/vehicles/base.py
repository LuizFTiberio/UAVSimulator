"""Base vehicle model — ties together params, MJCF, and dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class VehicleModel:
    """A vehicle definition providing parameters, physics model, and dynamics.

    Parameters
    ----------
    params : NamedTuple
        JAX-compatible parameter container (e.g. MultirotorParams).
    mjcf_path : Path
        Path to the MJCF XML model file.
    compute_wrench : callable
        ``(state, motor_commands, params) -> (F_world, T_world)``
        Pure JAX function computing net world-frame force and torque.
    actuator_names : tuple of str
        MuJoCo actuator names for visual prop spin (optional).
    spin_signs : tuple of float
        Spin direction sign per actuator for visual prop spin (optional).
    """
    params: Any
    mjcf_path: Path
    compute_wrench: Callable
    actuator_names: tuple[str, ...] = ()
    spin_signs: tuple[float, ...] = ()
