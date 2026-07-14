"""Smoke test: BEM_LIVE vs BEM_TABLE propulsion modes on a quadcopter.

Install CCBlade JAX first (one-time setup):
    pip install -e /path/to/CCBlade_jax
    # or from GitHub:
    pip install git+https://github.com/<user>/CCBlade_jax.git

Then run from the UAVSimulator root:
    python examples/bem_propulsion_demo.py

Expected output: T and Q from both modes printed side-by-side,
confirming < 1 % agreement for the chosen operating point.
"""

import os
import sys

# x64 must be enabled before any JAX import
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# ── CCBlade imports ───────────────────────────────────────────────────────────
try:
    from bem.core import BladeGeom, AirfoilData, RotorParams, load_airfoil
except ImportError as exc:
    sys.exit(
        f"Cannot import CCBlade JAX: {exc}\n"
        f"  Install it with:\n"
        f"    pip install -e /path/to/CCBlade_jax\n"
        f"  or:\n"
        f"    pip install git+https://github.com/<user>/CCBlade_jax.git"
    )

# naca4412.dat ships with CCBlade_jax; find it relative to the installed package
import bem as _bem_pkg
_ccblade_path = os.path.dirname(os.path.dirname(os.path.abspath(_bem_pkg.__file__)))

# ── UAVSimulator imports ──────────────────────────────────────────────────────
from uavsim.dynamics.propulsion_bem import BEMConfig, build_bem_table
from uavsim.dynamics.propulsion import PropulsionModel
from uavsim.vehicles.multirotor import quadcopter_params
from uavsim.core.types import MultirotorParams

# ── blade definition (APC 10×5-like, simplified) ─────────────────────────────
NACA4412 = os.path.join(_ccblade_path, "data", "naca4412.dat")
if not os.path.isfile(NACA4412):
    sys.exit(f"naca4412.dat not found at {NACA4412}")
af = load_airfoil(NACA4412)

Rhub = 0.0254   # 1-inch hub radius [m]
Rtip = 0.127    # 5-inch tip radius  [m]
N_stations = 10

r     = jnp.linspace(Rhub + 1e-3, Rtip - 1e-3, N_stations)
chord = jnp.linspace(0.025, 0.012, N_stations)                    # tapering chord [m]
twist = jnp.linspace(40.0, 15.0, N_stations) * jnp.pi / 180.0    # root→tip twist [rad]

geom   = BladeGeom(r=r, chord=chord, twist=twist)
rotor  = RotorParams(Rhub=Rhub, Rtip=Rtip, B=2)
bem_cfg = BEMConfig(geom=geom, af=af, rotor_params=rotor, rho=1.225)

# ── vehicle & operating point ─────────────────────────────────────────────────
# Asymmetric RPM so rotor yaw torques don't cancel to zero (which would make
# the Q error check trivial).  Pair 0/3 spins faster than pair 1/2 → net +yaw.
n_motors      = 4
omega         = jnp.array([520.0, 480.0, 480.0, 520.0])       # [rad/s]
v_air_body    = jnp.array([0.0, 0.0, -5.0])                   # 5 m/s axial inflow
rotor_yaw_sign = jnp.array([1.0, -1.0, -1.0, 1.0])

# Attach BEM config to a MultirotorParams (SIMPLE coefficients unused in BEM modes)
base_params = quadcopter_params()
params = MultirotorParams(
    **{f: getattr(base_params, f) for f in base_params._fields if f != "bem_config"},
    bem_config=bem_cfg,
)

# ── BEM_LIVE ──────────────────────────────────────────────────────────────────
print("=== BEM_LIVE: compiling & running (first call may take ~10 s) ===")
from uavsim.dynamics.propulsion import compute_rotor_wrench_dispatch

force_live, torque_live = jax.jit(
    compute_rotor_wrench_dispatch,
    static_argnames=["model"],
)(omega, v_air_body, rotor_yaw_sign, params, model=PropulsionModel.BEM_LIVE)

T_live = float(force_live[2])
Q_live = float(abs(torque_live[2]))
print(f"  T = {T_live:.4f} N   Q = {Q_live:.6f} N·m  (sum over {n_motors} rotors)")

# ── build lookup table ────────────────────────────────────────────────────────
print("\n=== Building BEM lookup table (20 × 15 × 15 grid) ===")
omega_grid = jnp.linspace(100.0, 800.0, 20)
vinf_grid  = jnp.linspace(0.0,   20.0,  15)
alpha_grid = jnp.linspace(0.0,   jnp.pi / 2.0, 15)

table_params = build_bem_table(geom, af, rotor, 1.225, omega_grid, vinf_grid, alpha_grid)
print(f"  Table shape: T {table_params.T_table.shape}, Q {table_params.Q_table.shape}")

# attach table inside the BEM config so the dispatch can reach it
bem_cfg_with_table = BEMConfig(
    geom=geom, af=af, rotor_params=rotor, rho=1.225, bem_table=table_params
)
params_table = MultirotorParams(
    **{f: getattr(base_params, f) for f in base_params._fields if f != "bem_config"},
    bem_config=bem_cfg_with_table,
)

# ── BEM_TABLE ─────────────────────────────────────────────────────────────────
print("\n=== BEM_TABLE ===")
force_table, torque_table = jax.jit(
    compute_rotor_wrench_dispatch,
    static_argnames=["model"],
)(omega, v_air_body, rotor_yaw_sign, params_table, model=PropulsionModel.BEM_TABLE)

T_table = float(force_table[2])
Q_table = float(abs(torque_table[2]))
print(f"  T = {T_table:.4f} N   Q = {Q_table:.6f} N·m  (sum over {n_motors} rotors)")

# ── comparison ────────────────────────────────────────────────────────────────
T_err = abs(T_live - T_table) / max(abs(T_live), 1e-9) * 100.0
Q_err = abs(Q_live - Q_table) / max(abs(Q_live), 1e-9) * 100.0
print(f"\n  |ΔT| = {T_err:.3f} %   |ΔQ| = {Q_err:.3f} %")

if T_err < 1.0 and Q_err < 1.0:
    print("\nSMOKE TEST PASSED — BEM_LIVE and BEM_TABLE agree within 1 %")
else:
    print("\nWARNING — agreement > 1 %: consider a finer grid (increase N in omega/vinf/alpha_grid)")
    sys.exit(1)
