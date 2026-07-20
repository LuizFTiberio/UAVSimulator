"""Tests for propulsion_bem.py's rotor_axis generalization.

Context: originally hardcoded to body-Z (quad/quadplane convention,
"rotors face up"). Generalized to an arbitrary rotor_axis via vector
projection so it can also serve a tail-sitter's body-+x thrust axis.
No pre-existing pytest suite covered this module (only example scripts),
so this is the first regression coverage for it.
"""

from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from uavsim.dynamics.propulsion_bem import (
    _velocity_decomposition,
    compute_rotor_wrench_bem,
    compute_rotor_wrench_bem_table,
    build_bem_table,
    BEMConfig,
)

bem = pytest.importorskip(
    "bem.core", reason="CCBlade JAX not on PYTHONPATH -- these tests need the real solver"
)
from bem.core import load_airfoil, BladeGeom, RotorParams  # noqa: E402

# Resolve the airfoil data relative to the installed ccblade-jax package, so it
# works on any machine rather than a hardcoded per-session path.
NACA4412_PATH = str(Path(bem.__file__).resolve().parent.parent / "data" / "naca4412.dat")
if not Path(NACA4412_PATH).exists():
    pytest.skip(f"naca4412.dat not found next to bem package ({NACA4412_PATH})",
                allow_module_level=True)


def _small_prop_bem_config():
    af = load_airfoil(NACA4412_PATH)
    R = 0.10
    r = jnp.linspace(0.15 * R, 0.98 * R, 10)
    twist = jnp.arctan(0.5 * (2 * R) / (2 * jnp.pi * r))
    chord = jnp.full_like(r, 0.10 * (2 * R))
    geom = BladeGeom(r=r, chord=chord, twist=twist)
    rotor = RotorParams(Rhub=0.15 * R, Rtip=R, B=2)
    return BEMConfig(geom=geom, af=af, rotor_params=rotor, rho=1.225)


# ── velocity decomposition ──────────────────────────────────────────────────

class TestVelocityDecomposition:

    def test_default_axis_matches_original_hardcoded_z_formula(self):
        """|Vx| and alpha_tilt must exactly match the original
        Vx=-v[2], V_lateral=||v[:2]|| formulation (sign of Vx is inert,
        see propulsion_bem.py docstring -- checked here, not assumed)."""
        for v in [jnp.array([3.0, -1.0, 2.0]), jnp.array([0.0, 0.0, -5.0]),
                  jnp.array([1.0, 1.0, 0.0]), jnp.zeros(3)]:
            V_inf_new, alpha_new = _velocity_decomposition(v)

            Vx_old = -v[2]
            V_lateral_old = jnp.linalg.norm(v[:2])
            V_inf_old = jnp.linalg.norm(v)
            alpha_old = jnp.arctan2(V_lateral_old, jnp.maximum(jnp.abs(Vx_old), 1e-6))

            assert float(jnp.abs(V_inf_new - V_inf_old)) < 1e-6
            # alpha_tilt tolerance loosened from 1e-10: _velocity_decomposition
            # now uses safe_norm (eps=1e-9 inside the sqrt) instead of plain
            # jnp.linalg.norm, specifically to fix a NaN-gradient bug at
            # exactly-zero vectors (see core/math.py's safe_norm docstring,
            # and chat history for how this was found). At the all-zero test
            # vector specifically, safe_norm's eps=1e-9 interacts with the
            # 1e-6 floor on |Vx| to give arctan2(1e-9,1e-6)=0.001 rad
            # (~0.057deg) instead of exactly 0 -- negligible physically
            # (this is an angle-of-attack-like quantity; 0.057deg is well
            # below any real aerodynamic significance), but mechanically
            # larger than a bit-exact tolerance would allow.
            assert float(jnp.abs(alpha_new - alpha_old)) < 2e-3

    def test_x_axis_pure_axial_gives_zero_tilt(self):
        """Airspeed purely along the rotor axis (x) -> alpha_tilt ~ 0."""
        v = jnp.array([12.0, 0.0, 0.0])
        V_inf, alpha = _velocity_decomposition(v, rotor_axis=jnp.array([1.0, 0.0, 0.0]))
        assert float(V_inf) == pytest.approx(12.0)
        assert float(alpha) < 1e-4

    def test_x_axis_pure_lateral_gives_90deg_tilt(self):
        """Airspeed purely perpendicular to rotor axis -> alpha_tilt ~ pi/2."""
        v = jnp.array([0.0, 8.0, 0.0])
        V_inf, alpha = _velocity_decomposition(v, rotor_axis=jnp.array([1.0, 0.0, 0.0]))
        assert float(V_inf) == pytest.approx(8.0)
        assert float(jnp.abs(alpha - jnp.pi / 2)) < 1e-4


# ── wrench functions: default axis backward compatibility ───────────────────

class TestBackwardCompatibility:

    def test_bem_live_default_axis_matches_pre_generalization_output(self):
        """Reimplements the OLD hardcoded-Z force/torque assembly directly
        and checks the new generalized (default axis) function matches."""
        cfg = _small_prop_bem_config()
        omega = jnp.array([600.0, 620.0])
        v_air = jnp.array([1.0, 0.5, -2.0])
        yaw_sign = jnp.array([1.0, -1.0])
        motor_pos = jnp.array([[0.0, -0.15, 0.0], [0.0, 0.15, 0.0]])

        F_new, T_new = compute_rotor_wrench_bem(omega, v_air, yaw_sign, cfg, motor_pos)

        # old hardcoded-Z path, reimplemented inline for comparison
        from bem.core import solve_rotor_oblique
        Vx_old = -v_air[2]
        V_lateral_old = jnp.linalg.norm(v_air[:2])
        V_inf_old = jnp.linalg.norm(v_air)
        alpha_old = jnp.arctan2(V_lateral_old, jnp.maximum(jnp.abs(Vx_old), 1e-6))

        Ts, Qs = [], []
        for om in omega:
            r = solve_rotor_oblique(V_inf_old, alpha_old, om, cfg.geom, cfg.af, cfg.rotor_params, cfg.rho)
            Ts.append(r.T); Qs.append(r.Q)
        Ts, Qs = jnp.array(Ts), jnp.array(Qs)

        F_old = jnp.zeros(3).at[2].set(jnp.sum(Ts))
        thrust_vecs = jnp.zeros_like(motor_pos).at[:, 2].set(Ts)
        T_old = jnp.sum(jnp.cross(motor_pos, thrust_vecs), axis=0)
        T_old = T_old.at[2].add(jnp.sum(yaw_sign * Qs))

        assert jnp.allclose(F_new, F_old, atol=1e-8)
        assert jnp.allclose(T_new, T_old, atol=1e-8)

    def test_bem_table_default_axis_matches_pre_generalization_output(self):
        cfg = _small_prop_bem_config()
        table = build_bem_table(
            cfg.geom, cfg.af, cfg.rotor_params, cfg.rho,
            omega_grid=jnp.linspace(300.0, 900.0, 5),
            vinf_grid=jnp.linspace(0.0, 10.0, 4),
            alpha_grid=jnp.linspace(0.0, jnp.pi, 4),
        )
        omega = jnp.array([650.0, 700.0])
        v_air = jnp.array([2.0, -1.0, 3.0])
        yaw_sign = jnp.array([1.0, -1.0])
        motor_pos = jnp.array([[0.0, -0.15, 0.0], [0.0, 0.15, 0.0]])

        F_new, T_new = compute_rotor_wrench_bem_table(omega, v_air, yaw_sign, table, motor_pos)

        from uavsim.dynamics.propulsion_bem import _trilinear_interp
        Vx_old = -v_air[2]
        V_lateral_old = jnp.linalg.norm(v_air[:2])
        V_inf_old = jnp.linalg.norm(v_air)
        alpha_old = jnp.arctan2(V_lateral_old, jnp.maximum(jnp.abs(Vx_old), 1e-6))

        Ts, Qs = [], []
        for om in omega:
            T = _trilinear_interp(om, V_inf_old, alpha_old, table.omega_grid, table.vinf_grid, table.alpha_grid, table.T_table)
            Q = _trilinear_interp(om, V_inf_old, alpha_old, table.omega_grid, table.vinf_grid, table.alpha_grid, table.Q_table)
            Ts.append(T); Qs.append(Q)
        Ts, Qs = jnp.array(Ts), jnp.array(Qs)

        F_old = jnp.zeros(3).at[2].set(jnp.sum(Ts))
        thrust_vecs = jnp.zeros_like(motor_pos).at[:, 2].set(Ts)
        T_old = jnp.sum(jnp.cross(motor_pos, thrust_vecs), axis=0)
        T_old = T_old.at[2].add(jnp.sum(yaw_sign * Qs))

        assert jnp.allclose(F_new, F_old, atol=1e-6)
        assert jnp.allclose(T_new, T_old, atol=1e-6)


# ── x-axis (tail-sitter) correctness ─────────────────────────────────────────

class TestTailsitterAxis:

    def test_force_along_x_not_z(self):
        cfg = _small_prop_bem_config()
        omega = jnp.array([700.0, 700.0])
        v_air = jnp.zeros(3)  # hover, axial
        yaw_sign = jnp.array([-1.0, 1.0])
        motor_pos = jnp.array([[0.0, -0.15, 0.0], [0.0, 0.15, 0.0]])

        F, T = compute_rotor_wrench_bem(omega, v_air, yaw_sign, cfg, motor_pos,
                                         rotor_axis=jnp.array([1.0, 0.0, 0.0]))
        assert float(F[0]) > 0.0, "thrust should be along body +x"
        assert float(jnp.abs(F[1])) < 1e-8
        assert float(jnp.abs(F[2])) < 1e-8

    def test_symmetric_thrust_zero_moment(self):
        cfg = _small_prop_bem_config()
        omega = jnp.array([700.0, 700.0])
        v_air = jnp.zeros(3)
        yaw_sign = jnp.array([-1.0, 1.0])  # counter-rotating pair
        motor_pos = jnp.array([[0.0, -0.15, 0.0], [0.0, 0.15, 0.0]])

        _, T = compute_rotor_wrench_bem(omega, v_air, yaw_sign, cfg, motor_pos,
                                         rotor_axis=jnp.array([1.0, 0.0, 0.0]))
        assert float(jnp.max(jnp.abs(T))) < 1e-6

    def test_differential_thrust_gives_yaw_not_roll_or_pitch(self):
        """Differential thrust at spanwise (y) offset, thrust axis x ->
        moment should land purely on z (yaw), same geometric structure
        confirmed for phi_theory's SIMPLE thrust model earlier."""
        cfg = _small_prop_bem_config()
        omega = jnp.array([650.0, 750.0])
        v_air = jnp.zeros(3)
        yaw_sign = jnp.array([-1.0, 1.0])
        motor_pos = jnp.array([[0.0, -0.15, 0.0], [0.0, 0.15, 0.0]])

        _, T = compute_rotor_wrench_bem(omega, v_air, yaw_sign, cfg, motor_pos,
                                         rotor_axis=jnp.array([1.0, 0.0, 0.0]))
        assert float(jnp.abs(T[2])) > 1e-4, "differential thrust should give yaw"
        assert float(jnp.abs(T[1])) < 1e-8, "should not give pitch"

    def test_table_lookup_matches_live_solve_for_x_axis(self):
        """Table-based path should agree closely with the live BEM solve
        for the tail-sitter axis too, not just the default Z axis."""
        cfg = _small_prop_bem_config()
        table = build_bem_table(
            cfg.geom, cfg.af, cfg.rotor_params, cfg.rho,
            omega_grid=jnp.linspace(300.0, 900.0, 7),
            vinf_grid=jnp.linspace(0.0, 15.0, 6),
            alpha_grid=jnp.linspace(0.0, jnp.pi, 6),
        )
        omega = jnp.array([680.0, 680.0])
        v_air = jnp.array([4.0, 0.0, 0.0])  # some axial airspeed (transition-ish)
        yaw_sign = jnp.array([-1.0, 1.0])
        motor_pos = jnp.array([[0.0, -0.15, 0.0], [0.0, 0.15, 0.0]])
        axis = jnp.array([1.0, 0.0, 0.0])

        F_live, T_live = compute_rotor_wrench_bem(omega, v_air, yaw_sign, cfg, motor_pos, rotor_axis=axis)
        F_tab, T_tab = compute_rotor_wrench_bem_table(omega, v_air, yaw_sign, table, motor_pos, rotor_axis=axis)

        assert jnp.allclose(F_live, F_tab, atol=0.05)
        assert jnp.allclose(T_live, T_tab, atol=0.01)
