"""Quadplane vehicle — quadcopter + fixed wing + pusher motor."""

from pathlib import Path

import jax.numpy as jnp

from uavsim.core.types import (
    MultirotorParams,
    PusherParams,
    QuadplaneParams,
    VehicleState,
    WingParams,
)
from uavsim.core.math import quat_to_rotation_matrix
from uavsim.dynamics.propulsion import compute_rotor_wrench, throttle_to_omega
from uavsim.dynamics.aerodynamics import compute_wing_wrench
from uavsim.vehicles.base import VehicleModel


# ── dynamics function ────────────────────────────────────────────────────────

def quadplane_wrench(
    state: VehicleState,
    motor_commands: jnp.ndarray,
    params: QuadplaneParams,
    wind_velocity: jnp.ndarray = jnp.zeros(3),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute world-frame force and torque for a quadplane.

    Sums rotor wrench, wing aerodynamics, and pusher thrust.

    Parameters
    ----------
    state : VehicleState
    motor_commands : (7,) control vector
        indices 0-3: quad rotors [0, 1]
        index    4 : pusher motor [0, 1]
        index    5 : flap [0, 1]  (0 = retracted, 1 = full)
        index    6 : aileron [-1, 1]  (positive = roll right)
    params : QuadplaneParams
    wind_velocity : (3,) world-frame wind velocity [m/s]

    Returns
    -------
    F_world : (3,) net force in world frame [N]
    T_world : (3,) net torque in world frame [N·m]
    """
    R = quat_to_rotation_matrix(state.quaternion)

    # ── quad rotors ──────────────────────────────────────────────────────
    quad_cmds = motor_commands[:4]
    omega_quad = throttle_to_omega(quad_cmds, params.rotor.max_omega)
    force_body, torque_body = compute_rotor_wrench(omega_quad, params.rotor)
    F_rotor = R @ force_body
    T_rotor = R @ torque_body

    # ── pusher motor (body +X) ───────────────────────────────────────────
    pusher_cmd = jnp.clip(motor_commands[4], 0.0, 1.0)
    omega_pusher = pusher_cmd * params.pusher.max_omega
    thrust_pusher = params.pusher.kt * omega_pusher ** 2
    F_pusher_body = jnp.array([thrust_pusher, 0.0, 0.0])
    F_pusher = R @ F_pusher_body
    # Pusher at CG ⇒ no torque contribution

    # ── wing + flap + aileron ────────────────────────────────────────────
    flap_cmd = motor_commands[5]     # [0, 1]
    aileron_cmd = motor_commands[6]  # [-1, 1]
    F_wing, T_wing = compute_wing_wrench(
        state, params.wing, flap=flap_cmd, aileron=aileron_cmd,
        wind_velocity=wind_velocity)

    return F_rotor + F_pusher + F_wing, T_rotor + T_wing


# ── parameter factories ──────────────────────────────────────────────────────

def default_wing_params(
    wingspan: float = 1.2,
    aspect_ratio: float = 8.0,
    rho: float = 1.225,
    CL0: float = 0.5,
    CLa: float = 2.0 * jnp.pi,
    CD0: float = 0.05,
    oswald: float = 0.9,
    alpha_max: float = jnp.radians(15.0),
    transition_speed: float = 7.0,
    transition_sharpness: float = 3.0,
    CL_delta_f: float = 1.0,
    max_flap: float = jnp.radians(30.0),
    Cl_delta_a: float = 0.4,
    max_aileron: float = jnp.radians(25.0),
) -> WingParams:
    """Create default wing parameters (1.2 m span, AR 8, rectangular)."""
    wing_area = wingspan ** 2 / aspect_ratio
    return WingParams(
        wingspan=wingspan,
        aspect_ratio=aspect_ratio,
        wing_area=wing_area,
        rho=rho,
        CL0=CL0,
        CLa=CLa,
        CD0=CD0,
        oswald=oswald,
        alpha_max=alpha_max,
        transition_speed=transition_speed,
        transition_sharpness=transition_sharpness,
        CL_delta_f=CL_delta_f,
        max_flap=max_flap,
        Cl_delta_a=Cl_delta_a,
        max_aileron=max_aileron,
    )


def default_pusher_params(
    max_thrust: float = 5.0,
    max_omega: float = 700.0,
) -> PusherParams:
    """Create default pusher motor parameters.

    Sized so full throttle gives ~5 N forward thrust.
    """
    kt = max_thrust / max_omega ** 2
    return PusherParams(kt=kt, max_omega=max_omega)


def quadplane_params(
    mass: float = 1.5,
    gravity: float = 9.81,
    arm_length: float = 0.12,
    max_omega: float = 1000.0,
    km_ratio: float = 0.016,
    wing: WingParams | None = None,
    pusher: PusherParams | None = None,
) -> QuadplaneParams:
    """Create default quadplane parameters.

    Smaller quad frame (arm_length=0.12 m) with higher-speed motors.
    Thrust coefficient sized so hover is ~50 % throttle.
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

    rotor = MultirotorParams(
        mass=mass,
        gravity=gravity,
        max_omega=max_omega,
        kt=kt,
        km=km,
        motor_positions=motor_positions,
        rotor_yaw_sign=rotor_yaw_sign,
        arm_length=arm_length,
    )

    if wing is None:
        wing = default_wing_params()
    if pusher is None:
        pusher = default_pusher_params()

    return QuadplaneParams(rotor=rotor, wing=wing, pusher=pusher)


def quadplane(
    params: QuadplaneParams | None = None,
    mjcf_path: Path | str | None = None,
) -> VehicleModel:
    """Create a complete quadplane vehicle model."""
    if params is None:
        params = quadplane_params()
    if mjcf_path is None:
        mjcf_path = Path(__file__).parent.parent / "models" / "quadplane.xml"
    else:
        mjcf_path = Path(mjcf_path)

    return VehicleModel(
        params=params,
        mjcf_path=mjcf_path,
        compute_wrench=quadplane_wrench,
        actuator_names=("act_fl", "act_fr", "act_bl", "act_br",
                        "act_pusher", "act_flap", "act_aileron"),
        spin_signs=(1.0, -1.0, -1.0, 1.0, 1.0, 0.0, 0.0),
    )
