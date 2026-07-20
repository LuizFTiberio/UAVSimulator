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

from uavsim.core.math import safe_norm

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
    rotor_axis: jax.Array = jnp.array([0.0, 0.0, 1.0]),
) -> tuple[jax.Array, jax.Array]:
    """Map body-frame airspeed to (V_inf, alpha_tilt) for solve_rotor_oblique.

    rotor_axis : (3,) unit vector, the rotor's spin/thrust axis in body
        frame. Default [0,0,1] preserves the original quad/quadplane
        convention (rotors face up) -- for a tail-sitter (thrust along
        body +x), pass rotor_axis=[1,0,0].

    Generalizes the original hardcoded-Z decomposition via projection onto
    an arbitrary axis instead of fixed component indexing:
      Vx        = dot(v_air_body, rotor_axis)      — axial inflow
      V_lateral = ||v_air_body - Vx*rotor_axis||    — perpendicular component
      alpha_tilt — angle between rotor axis and freestream [0, pi]
    Reduces EXACTLY to the original for rotor_axis=[0,0,1]: original had
    Vx=-v_air_body[2] (opposite sign to dot(v,[0,0,1])=v_air_body[2]), but
    alpha_tilt only ever uses |Vx|, and Vx itself is never returned or
    passed to solve_rotor_oblique (which only takes V_inf, alpha_tilt) --
    so the sign difference is inert; |Vx| and alpha_tilt are identical
    either way. Checked directly rather than assumed.
    """
    Vx = jnp.dot(v_air_body, rotor_axis)
    v_lateral_vec = v_air_body - Vx * rotor_axis
    V_lateral = safe_norm(v_lateral_vec)
    V_inf = safe_norm(v_air_body)
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
    rotor_axis: jax.Array = jnp.array([0.0, 0.0, 1.0]),
) -> tuple[jax.Array, jax.Array]:
    """Compute rotor wrench via live BEM solve (vmapped over motors).

    rotor_axis : (3,) unit vector, thrust/spin axis in body frame. Default
        [0,0,1] (quad/quadplane convention, unchanged). For a tail-sitter
        (thrust along body +x), pass rotor_axis=[1,0,0].

    Returns
    -------
    force_body  : (3,) [N]   — sum(T) * rotor_axis
    torque_body : (3,) [N·m] — pitch/roll moments from differential thrust + reaction torque about rotor_axis
    """
    if not _HAS_BEM:
        raise ImportError(
            "CCBlade JAX not found. Add it to PYTHONPATH:\n"
            "  export PYTHONPATH=/path/to/CCBlade_jax:$PYTHONPATH"
        )

    V_inf, alpha_tilt = _velocity_decomposition(v_air_body, rotor_axis)

    def _solve_one(omega_i: jax.Array) -> tuple[jax.Array, jax.Array]:
        result = solve_rotor_oblique(
            V_inf, alpha_tilt, omega_i,
            bem_config.geom, bem_config.af, bem_config.rotor_params,
            bem_config.rho,
        )
        return result.T, result.Q

    T_per_rotor, Q_per_rotor = jax.vmap(_solve_one)(omega)

    force_body = jnp.sum(T_per_rotor) * rotor_axis

    # Pitch/roll/yaw moments: cross(arm_i, T_i * rotor_axis) — same as
    # SIMPLE model, generalized from the fixed .at[:,2] indexing.
    thrust_vecs = T_per_rotor[:, None] * rotor_axis[None, :]
    torque_body = jnp.sum(jnp.cross(motor_positions, thrust_vecs), axis=0)
    torque_body = torque_body + jnp.sum(rotor_yaw_sign * Q_per_rotor) * rotor_axis
    return force_body, torque_body


# ── BEM_TABLE ────────────────────────────────────────────────────────────────

def lookup_bem_table_per_rotor(
    omega: jax.Array,              # (n_motors,) [rad/s]
    v_air_body: jax.Array,         # (3,) body-frame airspeed [m/s]
    table_params: BEMTableParams,
    rotor_axis: jax.Array = jnp.array([0.0, 0.0, 1.0]),
) -> tuple[jax.Array, jax.Array]:
    """Per-rotor (T, Q) from the table, WITHOUT assembling a wrench.

    Factored out of compute_rotor_wrench_bem_table so callers that need
    each rotor's own thrust separately -- e.g. to feed a propwash-coupled
    aero model like uavsim/aero/phi_theory.py's per-section propeller
    assignment -- don't have to duplicate the lookup. Added additively;
    compute_rotor_wrench_bem_table's own signature/behavior is unchanged,
    it just calls this internally now.

    Returns
    -------
    T_per_rotor, Q_per_rotor : (n_motors,) each
    """
    V_inf, alpha_tilt = _velocity_decomposition(v_air_body, rotor_axis)

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

    return jax.vmap(_lookup_one)(omega)


def compute_rotor_wrench_bem_table(
    omega: jax.Array,              # (n_motors,) [rad/s]
    v_air_body: jax.Array,         # (3,) body-frame airspeed [m/s]
    rotor_yaw_sign: jax.Array,     # (n_motors,) ±1
    table_params: BEMTableParams,
    motor_positions: jax.Array,    # (n_motors, 3) motor CoM positions [m]
    rotor_axis: jax.Array = jnp.array([0.0, 0.0, 1.0]),
) -> tuple[jax.Array, jax.Array]:
    """Compute rotor wrench via trilinear lookup table.

    rotor_axis : (3,) unit vector, thrust/spin axis in body frame. Default
        [0,0,1] (quad/quadplane convention, unchanged). For a tail-sitter
        (thrust along body +x), pass rotor_axis=[1,0,0]. The table itself
        (built by build_bem_table) is axis-agnostic -- alpha_tilt is
        purely rotor-relative -- so the SAME table works for any
        rotor_axis; only this consumption step needed generalizing.

    Returns
    -------
    force_body  : (3,) [N]
    torque_body : (3,) [N·m] — pitch/roll moments from differential thrust + reaction torque about rotor_axis
    """
    T_per_rotor, Q_per_rotor = lookup_bem_table_per_rotor(omega, v_air_body, table_params, rotor_axis)

    force_body = jnp.sum(T_per_rotor) * rotor_axis

    thrust_vecs = T_per_rotor[:, None] * rotor_axis[None, :]
    torque_body = jnp.sum(jnp.cross(motor_positions, thrust_vecs), axis=0)
    torque_body = torque_body + jnp.sum(rotor_yaw_sign * Q_per_rotor) * rotor_axis
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
