"""Multirotor vehicle definitions."""

from pathlib import Path

import jax.numpy as jnp

from uavsim.core.types import BodyDragParams, MultirotorParams, VehicleState
from uavsim.core.math import quat_to_rotation_matrix
from uavsim.dynamics.propulsion import compute_rotor_wrench, throttle_to_omega
from uavsim.vehicles.base import VehicleModel

# Default body drag for a quadcopter frame
_DEFAULT_DRAG = BodyDragParams()


# ── dynamics function ────────────────────────────────────────────────────────

def multirotor_wrench(
    state: VehicleState,
    motor_commands: jnp.ndarray,
    params: MultirotorParams,
    wind_velocity: jnp.ndarray = jnp.zeros(3),
    drag: BodyDragParams = _DEFAULT_DRAG,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute world-frame force and torque for a multirotor.

    Parameters
    ----------
    state : VehicleState (only quaternion is used)
    motor_commands : (n_motors,) normalised throttle [0, 1]
    params : MultirotorParams
    wind_velocity : (3,) world-frame wind velocity [m/s]
    drag : BodyDragParams  body aerodynamic drag parameters

    Returns
    -------
    F_world : (3,) net force in world frame [N]
    T_world : (3,) net torque in world frame [N·m]
    """
    omega = throttle_to_omega(motor_commands, params.max_omega)
    force_body, torque_body = compute_rotor_wrench(omega, params)
    R = quat_to_rotation_matrix(state.quaternion)
    F_rotor = R @ force_body
    T_rotor = R @ torque_body

    # Body drag from relative airspeed
    v_air_world = state.velocity - wind_velocity
    v_air_body = R.T @ v_air_world
    speed = jnp.linalg.norm(v_air_body)
    F_drag_body = -0.5 * drag.rho * drag.Cd * drag.frontal_area * speed * v_air_body
    F_drag_world = R @ F_drag_body

    return F_rotor + F_drag_world, T_rotor


# ── parameter factories ──────────────────────────────────────────────────────

def quadcopter_params(
    mass: float = 1.2,
    gravity: float = 9.81,
    arm_length: float = 0.17,
    max_omega: float = 838.0,
    km_ratio: float = 0.016,
) -> MultirotorParams:
    """Create default X-configuration quadcopter parameters.

    Thrust coefficient is derived so hover sits at exactly 50 % throttle.
    """
    kt = (mass * gravity / 4.0) / (0.5 * max_omega) ** 2
    km = kt * km_ratio

    motor_positions = jnp.array([
        [-arm_length,  arm_length, 0.0],   # FL
        [ arm_length,  arm_length, 0.0],   # FR
        [-arm_length, -arm_length, 0.0],   # BL
        [ arm_length, -arm_length, 0.0],   # BR
    ])
    rotor_yaw_sign = jnp.array([1.0, -1.0, -1.0, 1.0])

    return MultirotorParams(
        mass=mass,
        gravity=gravity,
        max_omega=max_omega,
        kt=kt,
        km=km,
        motor_positions=motor_positions,
        rotor_yaw_sign=rotor_yaw_sign,
        arm_length=arm_length,
    )


def quadcopter(
    params: MultirotorParams | None = None,
    mjcf_path: Path | str | None = None,
) -> VehicleModel:
    """Create a complete quadcopter vehicle model."""
    if params is None:
        params = quadcopter_params()
    if mjcf_path is None:
        mjcf_path = Path(__file__).parent.parent / "models" / "quadcopter.xml"
    else:
        mjcf_path = Path(mjcf_path)

    return VehicleModel(
        params=params,
        mjcf_path=mjcf_path,
        compute_wrench=multirotor_wrench,
        actuator_names=("act_fl", "act_fr", "act_bl", "act_br"),
        spin_signs=(1.0, -1.0, -1.0, 1.0),
    )


# ── slung-load variant ───────────────────────────────────────────────────────

def slung_quadcopter_params(
    payload_mass: float = 0.15,
    cable_mass: float = 0.02,
    **kwargs,
) -> MultirotorParams:
    """Quadcopter params with mass adjusted for a slung-load payload.

    The controller needs the *total* system mass for correct gravity
    compensation.  MuJoCo's model carries the physical mass distribution;
    this factory just ensures the feed-forward thrust matches the real weight.
    """
    base_mass = kwargs.pop("mass", 1.2)
    total_mass = base_mass + payload_mass + cable_mass
    return quadcopter_params(mass=total_mass, **kwargs)


def slung_quadcopter(
    payload_mass: float = 0.15,
    cable_mass: float = 0.02,
    **kwargs,
) -> VehicleModel:
    """Create a quadcopter vehicle tuned for slung-load transport.

    The returned VehicleModel uses the standard quadcopter MJCF as a
    fallback path; callers should pass the slung-load MJCF via
    ``MuJoCoSimulator(vehicle, mjcf_override=xml)``.
    """
    params = slung_quadcopter_params(
        payload_mass=payload_mass, cable_mass=cable_mass, **kwargs)
    return quadcopter(params=params)
