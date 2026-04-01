"""Motor mixer — maps thrust + torques to per-motor throttle.

X-configuration allocation matrix (4 motors).
"""

import jax.numpy as jnp


# Allocation matrix for X-config quadcopter.
# Rows = motors [FL, FR, BL, BR], columns = [thrust, roll, pitch, yaw].
#
# Motor positions (X-config):
#   FL (-x, +y)    FR (+x, +y)
#   BL (-x, -y)    BR (+x, -y)
#
# Roll  (x-axis torque) = arm * (F_FL + F_FR - F_BL - F_BR)
#   → increase +y motors, decrease -y motors
# Pitch (y-axis torque) = arm * (F_FL - F_FR + F_BL - F_BR)
#   → increase -x motors, decrease +x motors
# Yaw   (z-axis drag)   ∝ (F_FL - F_FR - F_BL + F_BR) via rotor_yaw_sign
#
ALLOC_X4 = jnp.array([
    [1.0,  1.0,  1.0,  1.0],   # FL (-x, +y, CCW)
    [1.0,  1.0, -1.0, -1.0],   # FR (+x, +y, CW)
    [1.0, -1.0,  1.0, -1.0],   # BL (-x, -y, CW)
    [1.0, -1.0, -1.0,  1.0],   # BR (+x, -y, CCW)
])


def mix_x4(
    thrust_n: float,
    roll_nm: float,
    pitch_nm: float,
    yaw_nm: float,
    arm_length: float,
    max_thrust_per_motor: float,
) -> jnp.ndarray:
    """Convert desired thrust + torques to per-motor throttle [0, 1].

    The simulator maps throttle -> omega linearly, but thrust ~ omega².
    So throttle = sqrt(force / max_force).

    Parameters
    ----------
    thrust_n : total upward thrust [N]
    roll_nm, pitch_nm, yaw_nm : body-frame torques [N·m]
    arm_length : distance from CoM to motor [m]
    max_thrust_per_motor : KT * max_omega² [N]

    Returns
    -------
    throttle : (4,) in [0, 1], order [FL, FR, BL, BR]
    """
    scale = jnp.array([
        thrust_n / 4.0,
        roll_nm / (4.0 * arm_length),
        pitch_nm / (4.0 * arm_length),
        yaw_nm / 4.0,
    ])
    per_motor_force = ALLOC_X4 @ scale

    throttle = jnp.sqrt(jnp.clip(per_motor_force / max_thrust_per_motor, 0.0, 1.0))
    return throttle
