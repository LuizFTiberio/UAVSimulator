"""BEM-based propulsion models using CCBlade JAX.

Requires CCBlade JAX on PYTHONPATH:
    export PYTHONPATH=/path/to/CCBlade_jax:$PYTHONPATH

Two modes:
  BEM_LIVE  — calls solve_rotor_oblique per step; exact but slower (JIT compile on first call)
  BEM_TABLE — trilinear lookup into a pre-built (omega, V_inf, alpha) table; fast for optimisation
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from typing import NamedTuple, Any

try:
    from bem.core import solve_rotor_oblique
    _HAS_BEM = True
except ImportError:
    _HAS_BEM = False


# ── parameter containers ─────────────────────────────────────────────────────

class BEMConfig(NamedTuple):
    """CCBlade inputs shared by all rotors on a vehicle (same blade geometry)."""
    geom: Any            # BladeGeom(r, chord, twist) from bem.core
    af: Any              # AirfoilData(alpha, cl, cd)  from bem.core
    rotor_params: Any    # RotorParams(Rhub, Rtip, B)  from bem.core
    rho: float = 1.225   # air density [kg/m³]
    bem_table: Any = None  # BEMTableParams (set after build_bem_table; None = use live solve)


class BEMTableParams(NamedTuple):
    """Pre-computed T/Q lookup table on a regular (omega, V_inf, alpha_tilt) grid."""
    omega_grid: jax.Array   # (N_omega,)  [rad/s]
    vinf_grid: jax.Array    # (N_vinf,)   [m/s]
    alpha_grid: jax.Array   # (N_alpha,)  [rad]
    T_table: jax.Array      # (N_omega, N_vinf, N_alpha)  [N]
    Q_table: jax.Array      # (N_omega, N_vinf, N_alpha)  [N·m]


# ── velocity decomposition ───────────────────────────────────────────────────

def _velocity_decomposition(
    v_air_body: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Map body-frame airspeed to (V_inf, alpha_tilt) for solve_rotor_oblique.

    Convention (z-up body frame, rotors face up):
      Vx  = -v_air_body[2]  — axial inflow into disk (+Vx = air enters from above)
      V_lateral = ||v_air_body[:2]||  — lateral component
      alpha_tilt — angle between rotor axis and freestream [0, π/2]
    """
    Vx = -v_air_body[2]
    V_lateral = jnp.linalg.norm(v_air_body[:2])
    V_inf = jnp.linalg.norm(v_air_body)
    alpha_tilt = jnp.arctan2(V_lateral, jnp.maximum(jnp.abs(Vx), 1e-6))
    return V_inf, alpha_tilt


# ── trilinear interpolation on a regular 3-D grid ───────────────────────────

def _trilinear_interp(
    x: jax.Array, y: jax.Array, z: jax.Array,
    x_grid: jax.Array, y_grid: jax.Array, z_grid: jax.Array,
    table: jax.Array,
) -> jax.Array:
    """Trilinear interpolation; clamps to grid boundaries."""
    x = jnp.clip(x, x_grid[0], x_grid[-1])
    y = jnp.clip(y, y_grid[0], y_grid[-1])
    z = jnp.clip(z, z_grid[0], z_grid[-1])

    ix = jnp.clip(jnp.searchsorted(x_grid, x, side="right") - 1, 0, x_grid.shape[0] - 2)
    iy = jnp.clip(jnp.searchsorted(y_grid, y, side="right") - 1, 0, y_grid.shape[0] - 2)
    iz = jnp.clip(jnp.searchsorted(z_grid, z, side="right") - 1, 0, z_grid.shape[0] - 2)

    tx = (x - x_grid[ix]) / (x_grid[ix + 1] - x_grid[ix])
    ty = (y - y_grid[iy]) / (y_grid[iy + 1] - y_grid[iy])
    tz = (z - z_grid[iz]) / (z_grid[iz + 1] - z_grid[iz])

    c000 = table[ix,     iy,     iz    ]
    c100 = table[ix + 1, iy,     iz    ]
    c010 = table[ix,     iy + 1, iz    ]
    c110 = table[ix + 1, iy + 1, iz    ]
    c001 = table[ix,     iy,     iz + 1]
    c101 = table[ix + 1, iy,     iz + 1]
    c011 = table[ix,     iy + 1, iz + 1]
    c111 = table[ix + 1, iy + 1, iz + 1]

    c00 = c000 * (1.0 - tx) + c100 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0  = c00  * (1.0 - ty) + c10  * ty
    c1  = c01  * (1.0 - ty) + c11  * ty
    return c0 * (1.0 - tz) + c1 * tz


# ── BEM_LIVE ─────────────────────────────────────────────────────────────────

def compute_rotor_wrench_bem(
    omega: jax.Array,              # (n_motors,) [rad/s]
    v_air_body: jax.Array,         # (3,) body-frame airspeed [m/s]
    rotor_yaw_sign: jax.Array,     # (n_motors,) ±1
    bem_config: BEMConfig,
    motor_positions: jax.Array,    # (n_motors, 3) motor CoM positions [m]
) -> tuple[jax.Array, jax.Array]:
    """Compute rotor wrench via live BEM solve (vmapped over motors).

    Returns
    -------
    force_body  : (3,) [N]   — [0, 0, sum(T)]
    torque_body : (3,) [N·m] — pitch/roll moments from differential thrust + yaw drag
    """
    if not _HAS_BEM:
        raise ImportError(
            "CCBlade JAX not found. Add it to PYTHONPATH:\n"
            "  export PYTHONPATH=/path/to/CCBlade_jax:$PYTHONPATH"
        )

    V_inf, alpha_tilt = _velocity_decomposition(v_air_body)

    def _solve_one(omega_i: jax.Array) -> tuple[jax.Array, jax.Array]:
        result = solve_rotor_oblique(
            V_inf, alpha_tilt, omega_i,
            bem_config.geom, bem_config.af, bem_config.rotor_params,
            bem_config.rho,
        )
        return result.T, result.Q

    T_per_rotor, Q_per_rotor = jax.vmap(_solve_one)(omega)

    force_body = jnp.zeros(3).at[2].set(jnp.sum(T_per_rotor))

    # Pitch/roll moments: cross(arm_i, [0, 0, T_i]) — same as SIMPLE model
    thrust_vecs = jnp.zeros_like(motor_positions).at[:, 2].set(T_per_rotor)
    torque_body = jnp.sum(jnp.cross(motor_positions, thrust_vecs), axis=0)
    torque_body = torque_body.at[2].add(jnp.sum(rotor_yaw_sign * Q_per_rotor))
    return force_body, torque_body


# ── BEM_TABLE ────────────────────────────────────────────────────────────────

def compute_rotor_wrench_bem_table(
    omega: jax.Array,              # (n_motors,) [rad/s]
    v_air_body: jax.Array,         # (3,) body-frame airspeed [m/s]
    rotor_yaw_sign: jax.Array,     # (n_motors,) ±1
    table_params: BEMTableParams,
    motor_positions: jax.Array,    # (n_motors, 3) motor CoM positions [m]
) -> tuple[jax.Array, jax.Array]:
    """Compute rotor wrench via trilinear lookup table.

    Returns
    -------
    force_body  : (3,) [N]
    torque_body : (3,) [N·m] — pitch/roll moments from differential thrust + yaw drag
    """
    V_inf, alpha_tilt = _velocity_decomposition(v_air_body)

    def _lookup_one(omega_i: jax.Array) -> tuple[jax.Array, jax.Array]:
        T = _trilinear_interp(
            omega_i, V_inf, alpha_tilt,
            table_params.omega_grid, table_params.vinf_grid, table_params.alpha_grid,
            table_params.T_table,
        )
        Q = _trilinear_interp(
            omega_i, V_inf, alpha_tilt,
            table_params.omega_grid, table_params.vinf_grid, table_params.alpha_grid,
            table_params.Q_table,
        )
        return T, Q

    T_per_rotor, Q_per_rotor = jax.vmap(_lookup_one)(omega)

    force_body = jnp.zeros(3).at[2].set(jnp.sum(T_per_rotor))

    # Pitch/roll moments: cross(arm_i, [0, 0, T_i]) — same as SIMPLE model
    thrust_vecs = jnp.zeros_like(motor_positions).at[:, 2].set(T_per_rotor)
    torque_body = jnp.sum(jnp.cross(motor_positions, thrust_vecs), axis=0)
    torque_body = torque_body.at[2].add(jnp.sum(rotor_yaw_sign * Q_per_rotor))
    return force_body, torque_body


def build_bem_table(
    geom: Any,
    af: Any,
    rotor: Any,
    rho: float,
    omega_grid: jax.Array,   # (N_omega,)
    vinf_grid: jax.Array,    # (N_vinf,)
    alpha_grid: jax.Array,   # (N_alpha,)
) -> BEMTableParams:
    """Pre-compute T and Q over a 3-D grid; returns BEMTableParams.

    Grid axes:
      omega_grid  — rotor speed [rad/s]
      vinf_grid   — freestream magnitude [m/s]
      alpha_grid  — tilt angle between rotor axis and freestream [rad]

    The table is built with a single vmapped JIT call.  Expect a one-time
    compile cost (~seconds); subsequent lookups are near-free.
    """
    if not _HAS_BEM:
        raise ImportError(
            "CCBlade JAX not found. Add it to PYTHONPATH:\n"
            "  export PYTHONPATH=/path/to/CCBlade_jax:$PYTHONPATH"
        )

    def _solve_one(omega: jax.Array, vinf: jax.Array, alpha: jax.Array):
        result = solve_rotor_oblique(vinf, alpha, omega, geom, af, rotor, rho)
        return result.T, result.Q

    # flatten 3-D grid → 1-D for a single batched vmap call
    O, V, A = jnp.meshgrid(omega_grid, vinf_grid, alpha_grid, indexing="ij")
    flat_O, flat_V, flat_A = O.ravel(), V.ravel(), A.ravel()

    T_flat, Q_flat = jax.jit(jax.vmap(_solve_one))(flat_O, flat_V, flat_A)

    shape = (omega_grid.shape[0], vinf_grid.shape[0], alpha_grid.shape[0])
    return BEMTableParams(
        omega_grid=omega_grid,
        vinf_grid=vinf_grid,
        alpha_grid=alpha_grid,
        T_table=T_flat.reshape(shape),
        Q_table=Q_flat.reshape(shape),
    )
