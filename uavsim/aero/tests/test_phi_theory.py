"""Tests for the phi-theory aero module (revised architecture -- see
phi_theory.py's module docstring and chat history for the design change
from an earlier per-section-freestream approach to a whole-vehicle Eq.14
term, prompted by switching from placeholder to real fitted
stability-derivative data).

Validation strategy:
  1. Cross-check against the paper's clean closed-form thin-airfoil result
     (still applies -- routed through the whole-vehicle term now).
  2. Direct checks of the new building blocks: eq14_whole_vehicle_wrench,
     geometric_phi_mv (Eq. 67 identity), elevon_phi_mv_act.
  3. Smoothness through zero airspeed, hover controllability (propwash
     mechanism, unchanged in spirit even though the surrounding
     architecture changed).
  4. safe_norm regression coverage (the gradient-at-zero bug found via
     cruise-trim solving).

We still do NOT test against the paper's Eqs. (97)-(98) -- see prior
reasoning (possible transcription corruption in that dense block).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from uavsim.core.types import VehicleState
from uavsim.core.math import safe_norm
from uavsim.aero.phi_theory import (
    PhiCoefficients,
    PhiWingParams,
    PhiWingSection,
    PhiPropeller,
    default_thin_airfoil_phi_fv,
    cessna_style_phi_coefficients,
    geometric_phi_mv,
    elevon_phi_mv_act,
    eq14_whole_vehicle_wrench,
    phi_wing_wrench,
    skew,
)


def _state(velocity, angular_velocity=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
    return VehicleState(
        position=jnp.zeros(3),
        quaternion=jnp.array(quat),
        velocity=jnp.array(velocity),
        angular_velocity=jnp.array(angular_velocity),
        time=0.0,
    )


def _symmetric_placeholder_coeffs(Cd0=0.02, Cy0=0.1, wingspan=1.0, chord=1.0,
                                   wing_ac_offset=(0.0, 0.0, 0.0), zeta_f=(0.0, 0.0, 0.0),
                                   phi_param=1.0):
    """Coefficients matching the OLD symmetric thin-airfoil placeholder,
    with zero rate damping / zero elevon-freestream terms -- used for
    tests that specifically want the classical closed-form comparison
    (which assumed a symmetric, uncambered airfoil) or that isolate one
    mechanism at a time."""
    return PhiCoefficients(
        Phi_fv=default_thin_airfoil_phi_fv(Cd0, Cy0),
        Phi_fw=jnp.zeros((3, 3)),
        Phi_mw=jnp.zeros((3, 3)),
        wing_ac_offset=jnp.array(wing_ac_offset),
        Cmde=0.0, Clda=0.0,
        zeta_f=jnp.array(zeta_f),
        phi_param=phi_param,
        wingspan=wingspan, chord=chord,
    )


# -- 1. cross-check against the paper's clean closed-form thin-airfoil result --

class TestThinAirfoilClosedForm:
    """Same closed-form derivation as before (Eq. 54/65), now routed
    through the whole-vehicle Eq.14 term via phi_wing_wrench with a
    single dry (no propeller) section -- should reduce identically to
    the old per-section computation for this a_k=0, omega=0 case."""

    @pytest.fixture
    def single_section_params(self):
        Cd0, Cy0 = 0.02, 0.1
        coeffs = _symmetric_placeholder_coeffs(Cd0, Cy0)
        section = PhiWingSection(area=1.0, aero_center=jnp.zeros(3))
        return PhiWingParams(coeffs=coeffs, rho=1.0, sections=(section,)), Cd0

    @pytest.mark.parametrize("alpha_deg", [0.0, 5.0, 15.0, 30.0, 45.0, 60.0, 89.0,
                                            120.0, 150.0, 179.0, -30.0, -90.0])
    def test_drag_coefficient_matches_closed_form(self, single_section_params, alpha_deg):
        params, Cd0 = single_section_params
        alpha = jnp.radians(alpha_deg)
        V = 10.0
        v_inf_b = jnp.array([V * jnp.cos(alpha), 0.0, V * jnp.sin(alpha)])
        state = _state(velocity=v_inf_b)

        F_world, _ = phi_wing_wrench(state, params, thrust_per_prop=jnp.zeros(0))

        v_hat = jnp.array([jnp.cos(alpha), 0.0, jnp.sin(alpha)])
        D = -jnp.dot(F_world, v_hat)
        q = 0.5 * params.rho * 1.0 * V ** 2
        Cd = D / q

        Cd_expected = Cd0 + 2.0 * jnp.pi * jnp.sin(alpha) ** 2
        assert float(jnp.abs(Cd - Cd_expected)) < 1e-5

    @pytest.mark.parametrize("alpha_deg", [5.0, 15.0, 30.0, 45.0, 60.0])
    def test_lift_magnitude_matches_closed_form(self, single_section_params, alpha_deg):
        params, _ = single_section_params
        alpha = jnp.radians(alpha_deg)
        V = 10.0
        v_inf_b = jnp.array([V * jnp.cos(alpha), 0.0, V * jnp.sin(alpha)])
        state = _state(velocity=v_inf_b)

        F_world, _ = phi_wing_wrench(state, params, thrust_per_prop=jnp.zeros(0))

        n_hat = jnp.array([-jnp.sin(alpha), 0.0, jnp.cos(alpha)])
        L = jnp.dot(F_world, n_hat)
        q = 0.5 * params.rho * 1.0 * V ** 2
        Cl = L / q

        Cl_expected_mag = jnp.pi * jnp.abs(jnp.sin(2.0 * alpha))
        assert float(jnp.abs(jnp.abs(Cl) - Cl_expected_mag)) < 1e-5


# -- 2. new building blocks -----------------------------------------------------

class TestGeometricPhiMv:
    """Eq. 67: Phi_mv = B^-1 [Delta_r x] Phi_fv."""

    def test_zero_offset_gives_zero_phi_mv(self):
        Phi_fv = jnp.eye(3) * 3.0
        Phi_mv = geometric_phi_mv(jnp.zeros(3), Phi_fv, wingspan=1.0, chord=1.0)
        assert float(jnp.max(jnp.abs(Phi_mv))) < 1e-10

    def test_matches_hand_computation(self):
        Delta_r = jnp.array([0.1, 0.0, 0.0])
        Phi_fv = jnp.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [3.0, 0.0, 1.0]])
        wingspan, chord = 2.0, 0.5
        Phi_mv = geometric_phi_mv(Delta_r, Phi_fv, wingspan, chord)

        B_diag = jnp.array([wingspan, chord, wingspan])
        expected = (skew(Delta_r) @ Phi_fv) / B_diag[:, None]
        assert jnp.allclose(Phi_mv, expected)

    def test_scales_linearly_with_offset(self):
        Phi_fv = jnp.eye(3)
        Phi_mv_1 = geometric_phi_mv(jnp.array([0.1, 0.0, 0.0]), Phi_fv, 1.0, 1.0)
        Phi_mv_2 = geometric_phi_mv(jnp.array([0.2, 0.0, 0.0]), Phi_fv, 1.0, 1.0)
        assert jnp.allclose(2.0 * Phi_mv_1, Phi_mv_2, atol=1e-10)


class TestElevonPhiMvAct:

    def test_symmetric_deflection_gives_pitch_only(self):
        coeffs = _symmetric_placeholder_coeffs()
        coeffs = coeffs._replace(Cmde=-1.28, Clda=0.178)
        M_act = elevon_phi_mv_act(coeffs, jnp.array([0.1, 0.1]))  # symmetric
        assert float(jnp.abs(M_act[0, 0])) < 1e-10, "no roll contribution from symmetric elevon"
        assert float(jnp.abs(M_act[1, 0])) > 1e-6, "should have pitch contribution"

    def test_differential_deflection_gives_roll_only(self):
        coeffs = _symmetric_placeholder_coeffs()
        coeffs = coeffs._replace(Cmde=-1.28, Clda=0.178)
        M_act = elevon_phi_mv_act(coeffs, jnp.array([0.1, -0.1]))  # differential
        assert float(jnp.abs(M_act[1, 0])) < 1e-10, "no pitch contribution from differential elevon"
        assert float(jnp.abs(M_act[0, 0])) > 1e-6, "should have roll contribution"

    def test_zero_deflection_gives_zero(self):
        coeffs = _symmetric_placeholder_coeffs()
        coeffs = coeffs._replace(Cmde=-1.28, Clda=0.178)
        M_act = elevon_phi_mv_act(coeffs, jnp.zeros(2))
        assert float(jnp.max(jnp.abs(M_act))) < 1e-10


class TestEq14WholeVehicleWrench:

    def test_zero_state_gives_zero_wrench(self):
        coeffs = cessna_style_phi_coefficients(
            Cd0=0.031, Cda=0.13, Cla=4.6, Cl0=0.31, Cy0=0.1,
            Clp=-0.47, Cmq=-12.4, Cnr=-0.099, Cmde=-1.28, Clda=0.178,
            wingspan=0.88, chord=0.17,
        )
        F, M = eq14_whole_vehicle_wrench(jnp.zeros(3), jnp.zeros(3), jnp.zeros((3, 3)),
                                          coeffs, rho=1.225, S=0.15)
        assert float(jnp.max(jnp.abs(F))) < 1e-6
        assert float(jnp.max(jnp.abs(M))) < 1e-6

    def test_rate_damping_opposes_rotation(self):
        """Real Cmq is negative (damping) -- a positive pitch rate should
        produce a restoring (negative) pitch moment."""
        coeffs = cessna_style_phi_coefficients(
            Cd0=0.031, Cda=0.13, Cla=4.6, Cl0=0.31, Cy0=0.1,
            Clp=-0.47, Cmq=-12.4, Cnr=-0.099, Cmde=-1.28, Clda=0.178,
            wingspan=0.88, chord=0.17,
        )
        omega_b = jnp.array([0.0, 0.5, 0.0])  # pitch rate only
        _, M = eq14_whole_vehicle_wrench(jnp.zeros(3), omega_b, jnp.zeros((3, 3)),
                                          coeffs, rho=1.225, S=0.15)
        assert float(M[1]) < 0.0, "positive pitch rate should give a restoring (negative) pitch moment"

    def test_no_nan_across_sweep_including_zero(self):
        coeffs = cessna_style_phi_coefficients(
            Cd0=0.031, Cda=0.13, Cla=4.6, Cl0=0.31, Cy0=0.1,
            Clp=-0.47, Cmq=-12.4, Cnr=-0.099, Cmde=-1.28, Clda=0.178,
            wingspan=0.88, chord=0.17,
        )
        for V in [0.0, 1e-6, 5.0, 20.0]:
            for wdeg in [0.0, 45.0, 90.0]:
                v_b = jnp.array([V, 0.0, 0.0])
                omega_b = 0.3 * jnp.array([jnp.cos(jnp.radians(wdeg)), jnp.sin(jnp.radians(wdeg)), 0.0])
                F, M = eq14_whole_vehicle_wrench(v_b, omega_b, jnp.zeros((3, 3)),
                                                  coeffs, rho=1.225, S=0.15)
                assert bool(jnp.all(jnp.isfinite(F)))
                assert bool(jnp.all(jnp.isfinite(M)))


class TestCessnaStyleConstructor:

    def test_phi_fv_matches_hand_assembly(self):
        Cd0, Cda, Cla, Cl0, Cy0 = 0.031, 0.13, 4.6, 0.31, 0.1
        coeffs = cessna_style_phi_coefficients(
            Cd0=Cd0, Cda=Cda, Cla=Cla, Cl0=Cl0, Cy0=Cy0,
            Clp=-0.47, Cmq=-12.4, Cnr=-0.099, Cmde=-1.28, Clda=0.178,
            wingspan=0.88, chord=0.17,
        )
        expected = jnp.array([
            [Cd0, 0.0, Cda + Cl0],
            [0.0, Cy0, 0.0],
            [Cl0, 0.0, -Cd0 + Cla],
        ])
        assert jnp.allclose(coeffs.Phi_fv, expected)

    def test_not_symmetric_in_general(self):
        """Real cambered-airfoil data (nonzero Cl0) gives a non-symmetric
        Phi_fv -- this is correct, not a bug (see PhiCoefficients
        docstring: symmetry was only a consequence of the symmetric-
        thin-airfoil special case)."""
        coeffs = cessna_style_phi_coefficients(
            Cd0=0.031, Cda=0.13, Cla=4.6, Cl0=0.31, Cy0=0.1,
            Clp=-0.47, Cmq=-12.4, Cnr=-0.099, Cmde=-1.28, Clda=0.178,
            wingspan=0.88, chord=0.17,
        )
        assert not jnp.allclose(coeffs.Phi_fv, coeffs.Phi_fv.T)

    def test_phi_fv_eigenvalues_positive(self):
        coeffs = cessna_style_phi_coefficients(
            Cd0=0.031, Cda=0.13, Cla=4.6, Cl0=0.31, Cy0=0.1,
            Clp=-0.47, Cmq=-12.4, Cnr=-0.099, Cmde=-1.28, Clda=0.178,
            wingspan=0.88, chord=0.17,
        )
        eigvals = jnp.linalg.eigvals(coeffs.Phi_fv)
        assert bool(jnp.all(jnp.real(eigvals) > 0))


# -- 3. smoothness through zero airspeed ----------------------------------------

class TestSmoothnessAndNoSingularity:
    def _two_prop_params(self):
        coeffs = cessna_style_phi_coefficients(
            Cd0=0.031, Cda=0.13, Cla=4.6, Cl0=0.31, Cy0=0.1,
            Clp=-0.47, Cmq=-12.4, Cnr=-0.099, Cmde=-1.28, Clda=0.178,
            wingspan=1.2, chord=0.15,
            wing_ac_offset=jnp.zeros(3),
            zeta_f=jnp.array([0.0, 0.3, 0.0]),
        )
        propellers = (
            PhiPropeller(position=jnp.array([0.0, -0.3, 0.0]), disk_area=0.05),
            PhiPropeller(position=jnp.array([0.0, 0.3, 0.0]), disk_area=0.05),
        )
        sections = (
            PhiWingSection(area=0.09, aero_center=jnp.array([-0.015, -0.3, 0.0]),
                            prop_indices=(0,), elevon_index=0),
            PhiWingSection(area=0.09, aero_center=jnp.array([-0.015, 0.3, 0.0]),
                            prop_indices=(1,), elevon_index=1),
        )
        return PhiWingParams(coeffs=coeffs, rho=1.225, sections=sections,
                              propellers=propellers)

    def test_no_nan_across_velocity_and_alpha_sweep(self):
        params = self._two_prop_params()
        thrust = jnp.array([5.0, 5.0])
        elevons = jnp.array([0.0, 0.0])
        for V in [0.0, 1e-6, 0.1, 1.0, 10.0, 50.0]:
            for alpha_deg in np.linspace(-180, 180, 13):
                alpha = jnp.radians(alpha_deg)
                v = jnp.array([V * jnp.cos(alpha), 0.0, V * jnp.sin(alpha)])
                state = _state(velocity=v, angular_velocity=(0.3, -0.2, 0.1))
                F, T = phi_wing_wrench(state, params, thrust, elevons)
                assert bool(jnp.all(jnp.isfinite(F)))
                assert bool(jnp.all(jnp.isfinite(T)))

    def test_continuous_through_zero_airspeed(self):
        params = self._two_prop_params()
        thrust = jnp.array([5.0, 6.0])
        elevons = jnp.array([0.05, -0.05])
        h = 1e-5
        state_minus = _state(velocity=(-h, 0.0, 0.0), angular_velocity=(0.1, 0.1, 0.1))
        state_zero = _state(velocity=(0.0, 0.0, 0.0), angular_velocity=(0.1, 0.1, 0.1))
        state_plus = _state(velocity=(h, 0.0, 0.0), angular_velocity=(0.1, 0.1, 0.1))

        F_m, T_m = phi_wing_wrench(state_minus, params, thrust, elevons)
        F_0, T_0 = phi_wing_wrench(state_zero, params, thrust, elevons)
        F_p, T_p = phi_wing_wrench(state_plus, params, thrust, elevons)

        assert float(jnp.max(jnp.abs(F_m - F_0))) < 1e-2
        assert float(jnp.max(jnp.abs(F_p - F_0))) < 1e-2
        assert float(jnp.max(jnp.abs(T_m - T_0))) < 1e-2
        assert float(jnp.max(jnp.abs(T_p - T_0))) < 1e-2

    def test_differentiable_through_zero_state(self):
        """The safe_norm regression case: gradient of the whole wrench
        w.r.t. velocity must stay finite at exactly zero state."""
        params = self._two_prop_params()
        thrust = jnp.array([5.0, 5.0])
        elevons = jnp.array([0.0, 0.0])

        def fx(vel):
            state = _state(velocity=vel, angular_velocity=(0.0, 0.0, 0.0))
            F, _ = phi_wing_wrench(state, params, thrust, elevons)
            return F[0]

        grad = jax.grad(fx)(jnp.zeros(3))
        assert bool(jnp.all(jnp.isfinite(grad)))


# -- 4. hover controllability from differential thrust/elevon (propwash) -------

class TestHoverControllability:

    def _symmetric_two_prop_params(self):
        coeffs = cessna_style_phi_coefficients(
            Cd0=0.031, Cda=0.13, Cla=4.6, Cl0=0.31, Cy0=0.1,
            Clp=-0.47, Cmq=-12.4, Cnr=-0.099, Cmde=-1.28, Clda=0.178,
            wingspan=1.2, chord=0.15,
            zeta_f=jnp.array([0.0, 0.3, 0.0]),
        )
        propellers = (
            PhiPropeller(position=jnp.array([0.0, -0.3, 0.0]), disk_area=0.05),
            PhiPropeller(position=jnp.array([0.0, 0.3, 0.0]), disk_area=0.05),
        )
        sections = (
            PhiWingSection(area=0.09, aero_center=jnp.array([-0.015, -0.3, 0.0]),
                            prop_indices=(0,), elevon_index=0),
            PhiWingSection(area=0.09, aero_center=jnp.array([-0.015, 0.3, 0.0]),
                            prop_indices=(1,), elevon_index=1),
        )
        return PhiWingParams(coeffs=coeffs, rho=1.225, sections=sections,
                              propellers=propellers)

    def test_symmetric_thrust_gives_zero_roll_and_yaw_at_hover(self):
        """Symmetric thrust/elevon must still give zero ROLL and YAW
        (spanwise symmetry). PITCH is a different story with real
        cambered-airfoil data: nonzero Cl0 means propwash alone produces
        lift even at zero elevon (a cambered surface generates lift from
        flow over it regardless of deflection), and that lift acting
        through the elevon's own moment arm (the nonzero x-offset
        aero_center) gives a genuine, physically real pitch moment --
        exactly analogous to a cambered wing needing pitch trim in normal
        flight, just now showing up via propwash too. Not a bug -- caught
        by an earlier version of this test asserting zero on ALL axes,
        which was only valid for the old placeholder uncambered (Cl0=0)
        model. Verified by hand (see chat history) that the moment
        magnitude matches Phi_fv @ (T,0,0) having a nonzero z-component
        specifically because Cl0 != 0."""
        params = self._symmetric_two_prop_params()
        state = _state(velocity=(0.0, 0.0, 0.0))
        _, T = phi_wing_wrench(state, params, jnp.array([5.0, 5.0]), jnp.array([0.0, 0.0]))
        assert float(jnp.abs(T[0])) < 1e-8, "roll should still be zero (spanwise symmetry)"
        assert float(jnp.abs(T[2])) < 1e-8, "yaw should still be zero (spanwise symmetry)"
        # pitch (T[1]) is NOT asserted zero here -- see docstring

    def test_differential_thrust_gives_nonzero_moment_at_hover(self):
        params = self._symmetric_two_prop_params()
        state = _state(velocity=(0.0, 0.0, 0.0))
        _, T = phi_wing_wrench(state, params, jnp.array([5.0, 7.0]), jnp.array([0.0, 0.0]))
        assert float(jnp.max(jnp.abs(T))) > 1e-6

    def test_differential_elevon_gives_nonzero_moment_at_hover(self):
        params = self._symmetric_two_prop_params()
        state = _state(velocity=(0.0, 0.0, 0.0))
        _, T = phi_wing_wrench(state, params, jnp.array([5.0, 5.0]), jnp.array([0.1, -0.1]))
        assert float(jnp.max(jnp.abs(T))) > 1e-6

    def test_zero_thrust_zero_elevon_effect_at_hover(self):
        """With no thrust, propwash-elevon does nothing at hover, AND the
        whole-vehicle freestream/elevon term is also zero at v=0 -- total
        wrench should be exactly zero."""
        params = self._symmetric_two_prop_params()
        state = _state(velocity=(0.0, 0.0, 0.0))
        F, T = phi_wing_wrench(state, params, jnp.array([0.0, 0.0]), jnp.array([0.3, -0.3]))
        assert float(jnp.max(jnp.abs(F))) < 1e-8
        assert float(jnp.max(jnp.abs(T))) < 1e-8


# -- 5. safe_norm regression (gradient-at-zero bug) -----------------------------

def test_safe_norm_matches_regular_norm_away_from_zero():
    x = jnp.array([3.0, -4.0, 0.0])
    assert float(jnp.abs(safe_norm(x) - jnp.linalg.norm(x))) < 1e-8


def test_safe_norm_value_near_zero_at_origin():
    assert float(safe_norm(jnp.zeros(3))) < 1e-6


def test_safe_norm_gradient_finite_at_origin():
    grad = jax.grad(lambda x: safe_norm(x))(jnp.zeros(3))
    assert bool(jnp.all(jnp.isfinite(grad))), "safe_norm gradient should be finite at the origin"
    bad_grad = jax.grad(lambda x: jnp.linalg.norm(x))(jnp.zeros(3))
    assert bool(jnp.any(jnp.isnan(bad_grad))), "sanity check: plain norm should be NaN here"


# -- 6. basic helper sanity ------------------------------------------------------

def test_skew_matrix_properties():
    v = jnp.array([1.0, 2.0, 3.0])
    S = skew(v)
    assert float(jnp.max(jnp.abs(S + S.T))) < 1e-10  # antisymmetric
    w = jnp.array([4.0, -1.0, 0.5])
    assert float(jnp.max(jnp.abs(S @ w - jnp.cross(v, w)))) < 1e-10


def test_default_thin_airfoil_phi_fv_shape_and_diag():
    Phi = default_thin_airfoil_phi_fv(Cd0=0.03, Cy0=0.2)
    assert Phi.shape == (3, 3)
    off_diag = Phi - jnp.diag(jnp.diag(Phi))
    assert float(jnp.max(jnp.abs(off_diag))) < 1e-10
    assert float(Phi[0, 0]) == pytest.approx(0.03)
    assert float(Phi[2, 2]) == pytest.approx(2 * np.pi + 0.03)
