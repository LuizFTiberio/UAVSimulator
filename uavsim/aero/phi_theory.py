"""phi-theory aerodynamic model for tail-sitters, as pure JAX functions.

Implements the {alpha, beta}-free, singularity-free wing aerodynamic law from:

    Lustosa, L. R., Defay, F., and Moschetta, J.-M., "Global Singularity-Free
    Aerodynamic Model for Algorithmic Flight Control of Tail Sitters,"
    Journal of Guidance, Control, and Dynamics, 42(2), 2019, pp. 303-316.
    https://doi.org/10.2514/1.G003374

Equation numbers referenced in comments/docstrings below are from that paper.

Architecture (revised from an earlier per-section-freestream design once
real fitted stability-derivative data became available -- see chat history
for the full design discussion):

  * Eq. (14)/(21): tau = -eta_norm * C @ Phi @ (C @ eta_b)   [WHOLE-VEHICLE
    "dry" wrench: freestream force/moment + real rotational-rate damping,
    eq14_whole_vehicle_wrench]
  * Eq. (67):      Phi_mv = B^-1 [Delta_r x] Phi_fv   [GEOMETRIC baseline
    moment-arm relation, used for the wing's own static margin -- kept
    geometric rather than importing a real aircraft's directly-fit Clb/
    Cma/Cnb, which would bundle in tail/fuselage contributions this
    tail-sitter doesn't have]
  * Cmde/Clda : elevon-in-FREESTREAM moment derivatives (whole-vehicle,
    elevon_phi_mv_act) -- a SECOND, physically distinct elevon mechanism
    alongside propwash (below), both add.
  * Eq. (79):      F_b = F_inf_b + F_p_b   [PER-SECTION propwash term only
    -- the whole-vehicle term above already covers the freestream part
    that Eq. (97)'s worked example also keeps whole-vehicle; per-section
    covers just the propwash/thrust-linear correction, elevon-modulated
    (Eq. 95), superposed over N sections/propellers]

An earlier version of this module also computed freestream force PER
SECTION (using each section's own "local velocity" v_cg + omega x a_k, a
rigid-body point-velocity argument) as a proxy for rotational-rate
coupling, in the absence of real Cmq/Clp/Cnr data. That mechanism is
retired now that real rate-damping data is available (Phi_mw) -- keeping
both would double-count rotational-rate effects through two different,
overlapping derivations. See git history / tests for the retired
mechanism and its (still valid, just superseded) symbolic verification.

generalized to an arbitrary number of wing "sections" (each with its own
area, aerodynamic-center position, and the set of propellers whose thrust
washes it -- possibly zero, i.e. a dry section, which now contributes
nothing beyond the whole-vehicle term, no separate per-section freestream
calculation needed).

Known simplification (flagged, not modeled): tandem/chordwise propeller
wake interference (a downstream disk sitting in an upstream disk's
slipstream) is not modeled. Each propeller's momentum-theory contribution
assumes it sees the undisturbed freestream. Left to sim2real.

JIT/grad note: PhiWingSection.prop_indices and elevon_index are static
Python-level topology (which propellers/elevons feed which section), used
for plain Python indexing, not differentiable leaves. If PhiWingParams is
passed as a direct jax.jit (or jax.grad) argument, JAX will try to trace
those ints too and indexing will break (see tests/test_tailsitter.py's
TestTailsitterWrench.test_jit_compatible for the failure mode and fix).
Bind params via closure/functools.partial before jitting -- the pattern
already used throughout this repo, e.g. uavsim/sim/mujoco_sim.py's
`jax.jit(partial(vehicle.compute_wrench, params=vehicle.params))` -- rather
than passing params as a traced jit argument. For future gradient-based
coefficient fitting (Phase 1+), target specific numeric leaves (e.g.
Phi_fv, an aero_center array) with jax.grad rather than differentiating
through the whole params pytree.
"""

from typing import NamedTuple

import jax.numpy as jnp

from uavsim.core.types import VehicleState
from uavsim.core.math import quat_to_rotation_matrix, safe_norm


# ── parameter containers ─────────────────────────────────────────────────────

class PhiPropeller(NamedTuple):
    """A propeller contributing thrust-induced wash to one or more sections.

    Thrust is assumed axial along body +x (standard forward-mounted
    tail-sitter convention, matching the paper's T_i = kf * omega_i^2 * b1_hat,
    Eq. 93). Thrust *magnitude* per propeller is supplied by the caller each
    step (e.g. from propulsion.py or propulsion_bem.py); this module only
    consumes the scalar magnitude.
    """
    position: jnp.ndarray   # (3,) body frame, relative to CG [m]
    disk_area: float        # Sp_j [m^2]


class PhiWingSection(NamedTuple):
    """A patch of wing area with its own aerodynamic center and propwash.

    prop_indices selects which entries of `propellers` (in PhiWingParams)
    wash this section; empty tuple = dry section (freestream-only).
    elevon_index selects which entry of the `elevon_commands` array deflects
    this section; None = no elevon on this section.
    """
    area: float                        # S_k [m^2]
    aero_center: jnp.ndarray           # a_k, (3,) body frame relative to CG [m]
    prop_indices: tuple[int, ...] = ()
    elevon_index: int | None = None


class PhiCoefficients(NamedTuple):
    """phi-theory aerodynamic coefficients (Eq. 14/20's full block structure).

    Phi_fv : (3,3) velocity->force. NOT necessarily symmetric -- symmetry
        was only a consequence of Sec. IV.C's symmetric-thin-airfoil
        special case (uncambered, dCd/dalpha=0 at alpha=0). Real fitted
        data for a cambered/real aircraft (nonzero Cl0) has no reason to
        be symmetric, and the dissipativity result (Theorem III.1) only
        constrains Phi's SYMMETRIC PART anyway (eta^T@Phi@eta is blind to
        the antisymmetric part by construction).
    Phi_fw : (3,3) angular-rate->force (often zero/small; e.g. Cyp, Cyr in
        a standard stability-derivative dataset). Kept general even though
        it happens to be zero in Cyclone's initial dataset.
    Phi_mw : (3,3) angular-rate->moment, i.e. rate damping (Eq. 60): standard
        stability derivatives Cl_p/Cl_q/Cl_r, Cm_p/Cm_q/Cm_r, Cn_p/Cn_q/Cn_r.
    wing_ac_offset : (3,) Delta r for the GEOMETRIC baseline Phi_mv (Eq. 67:
        Phi_mv = B^-1 [Delta_r x] Phi_fv), used for the wing's own freestream
        pitch/roll/yaw stiffness. Deliberately geometric, not a directly-fit
        constant (e.g. a real aircraft's Cma/Clb/Cnb) -- those bundle in
        contributions from hardware (tail, fuselage) this tail-sitter
        doesn't have, so importing them wholesale would borrow stability
        from parts that aren't there. zeros(3) (CG at the wing's own
        aerodynamic center, e.g. quarter-chord for a thin airfoil) is the
        natural default absent a tail -- see chat history.
    Cmde, Clda : elevon-in-FREESTREAM moment derivatives (pitch, roll), for
        the SAME two elevons already used for propwash authority --
        symmetric deflection = elevator-equivalent (Cmde), differential =
        aileron-equivalent (Clda). Distinct physical mechanism from the
        propwash term below (this one needs real airspeed, dominates
        cruise; propwash dominates hover) -- both add, not compete.
    zeta_f : (3,) elevon effectiveness for the PROPWASH mechanism (Eq. 95),
        per-section, unchanged from before -- still the only source of
        elevon authority at hover (this freestream mechanism needs V>0).
    phi_param : tunable phi in Eq. (17)'s eta_norm = sqrt(v^2 + phi*omega^2).
        No strong guidance in the paper on choosing it beyond phi>0;
        default 1.0 (no a priori preference between translational and
        rotational contributions to the norm) pending a better-motivated
        choice.
    wingspan, chord : reference dimensions [m] (b, c) for the B=diag(b,c,b)
        scaling in the moment channels (Eq. 11, Eq. 19).
    """
    Phi_fv: jnp.ndarray        # (3,3)
    Phi_fw: jnp.ndarray        # (3,3)
    Phi_mw: jnp.ndarray        # (3,3)
    wing_ac_offset: jnp.ndarray  # (3,)
    Cmde: float
    Clda: float
    zeta_f: jnp.ndarray        # (3,)
    phi_param: float
    wingspan: float
    chord: float


class PhiWingParams(NamedTuple):
    coeffs: PhiCoefficients
    rho: float
    sections: tuple[PhiWingSection, ...]
    propellers: tuple[PhiPropeller, ...] = ()


# ── helpers ───────────────────────────────────────────────────────────────────

def skew(v: jnp.ndarray) -> jnp.ndarray:
    """Vector-to-skew-symmetric-matrix operator [v x] (Eq. 3)."""
    return jnp.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def default_thin_airfoil_phi_fv(Cd0: float, Cy0: float) -> jnp.ndarray:
    """Symmetric thin-airfoil Phi_fv baseline (Eq. 65).

    Encodes the classical thin-airfoil result dCl/dalpha = 2*pi (Eq. 62),
    zero dCd/dalpha at alpha=0 (Eq. 63/64), for a symmetric (uncambered)
    airfoil. Cy0 is the (typically small) sideforce-related term.

    Fallback for when no real fitted stability-derivative data is
    available -- prefer cessna_style_phi_coefficients() (or your own
    directly-fit Phi_fv) once you have real Cl0/Cla/Cd0/Cda data, since
    that captures camber (nonzero Cl0) and lift slope explicitly, which
    this symmetric placeholder cannot.
    """
    return jnp.array([
        [Cd0, 0.0, 0.0],
        [0.0, Cy0, 0.0],
        [0.0, 0.0, 2.0 * jnp.pi + Cd0],
    ])


def geometric_phi_mv(Delta_r: jnp.ndarray, Phi_fv: jnp.ndarray,
                      wingspan: float, chord: float) -> jnp.ndarray:
    """Geometric baseline Phi_mv (Eq. 67): Phi_mv = B^-1 [Delta_r x] Phi_fv.

    B=diag(wingspan,chord,wingspan) is diagonal, so B^-1 @ M is just each
    ROW of M divided by the corresponding B entry.
    """
    B_diag = jnp.array([wingspan, chord, wingspan])
    return (skew(Delta_r) @ Phi_fv) / B_diag[:, None]


def elevon_phi_mv_act(coeffs: "PhiCoefficients", elevon_commands: jnp.ndarray) -> jnp.ndarray:
    """Elevon-in-freestream moment contribution to Phi_mv.

    Symmetric deflection (both elevons same sign/magnitude) = elevator-
    equivalent -> pitch (Cmde). Differential deflection = aileron-
    equivalent -> roll (Clda). No rudder/yaw term (not modeled, per
    scope decision -- see chat history). Only the first column is
    populated (matches the convention that these derivatives multiply
    the forward-velocity component v_b[0] specifically, standard for
    fixed-wing stability-derivative datasets normalized on forward
    dynamic pressure).

    elevon_commands assumed [left, right], matching phi_wing_wrench's
    existing convention (same array used for the propwash mechanism).
    """
    de = 0.5 * (elevon_commands[0] + elevon_commands[1])   # symmetric -> elevator
    da = 0.5 * (elevon_commands[1] - elevon_commands[0])   # differential -> aileron
    return -jnp.array([
        [coeffs.Clda * da, 0.0, 0.0],
        [coeffs.Cmde * de, 0.0, 0.0],
        [0.0,              0.0, 0.0],
    ])


def cessna_style_phi_coefficients(
    Cd0: float, Cda: float, Cla: float, Cl0: float, Cy0: float,
    Clp: float, Cmq: float, Cnr: float,
    Cmde: float, Clda: float,
    wingspan: float, chord: float,
    wing_ac_offset: jnp.ndarray = jnp.zeros(3),
    zeta_f: jnp.ndarray = jnp.zeros(3),
    phi_param: float = 1.0,
    Cyp: float = 0.0, Cyr: float = 0.0,
    Clq: float = 0.0, Clr: float = 0.0,
    Cmp: float = 0.0, Cmr: float = 0.0,
    Cnp: float = 0.0, Cnq: float = 0.0,
) -> "PhiCoefficients":
    """Build PhiCoefficients from a standard stability-derivative dataset
    (the format aircraft aerodynamic data is normally supplied in --
    e.g. a textbook/reference Cessna-class dataset), rather than a
    from-scratch symmetric thin-airfoil placeholder.

    Phi_fv assembled per Eq. 45-48's structure (Cd0, Cda+Cl0, Cy0, Cl0,
    -Cd0+Cla in the (1,1),(1,3),(2,2),(3,1),(3,3) entries -- NOT
    symmetric in general, see PhiCoefficients docstring). Phi_fw from
    Cyp/Cyr (rate->force, usually small/zero). Phi_mw from the
    Cl_/Cm_/Cn_ {p,q,r} rate-damping set (Eq. 60). Directly-fit Clb/Cma/
    Cnb (velocity->moment, e.g. a real aircraft's tail-driven static
    margin) are DELIBERATELY NOT used here -- see wing_ac_offset's
    docstring for why; pass wing_ac_offset instead to get Phi_mv
    geometrically.

    Sign convention note: Phi_mw carries a NEGATIVE sign (-0.5*[...], not
    the paper's literal +0.5*Eq.(60)), matching the source MATLAB
    implementation's own convention (which also negates Phi_mv and
    Phi_mv_act -- kept here too, in elevon_phi_mv_act). Verified directly,
    not just copied: a positive pitch rate with real (negative, damping)
    Cmq must produce a restoring (negative) pitch moment once run through
    the full Eq.(14) formula's own outer minus sign and B-matrix scaling;
    only the negative-Phi_mw convention gives that (see
    tests/test_phi_theory.py::TestEq14WholeVehicleWrench::
    test_rate_damping_opposes_rotation and chat history for how a sign
    bug here was initially caught).
    """
    Phi_fv = jnp.array([
        [Cd0,  0.0,  Cda + Cl0],
        [0.0,  Cy0,  0.0],
        [Cl0,  0.0,  -Cd0 + Cla],
    ])
    Phi_fw = 0.5 * jnp.array([
        [0.0,  0.0,  0.0],
        [Cyp,  0.0,  Cyr],
        [0.0,  0.0,  0.0],
    ])
    Phi_mw = -0.5 * jnp.array([
        [Clp, Clq, Clr],
        [Cmp, Cmq, Cmr],
        [Cnp, Cnq, Cnr],
    ])
    return PhiCoefficients(
        Phi_fv=Phi_fv, Phi_fw=Phi_fw, Phi_mw=Phi_mw,
        wing_ac_offset=wing_ac_offset, Cmde=Cmde, Clda=Clda,
        zeta_f=zeta_f, phi_param=phi_param, wingspan=wingspan, chord=chord,
    )


def elevon_phi_fv(coeffs: "PhiCoefficients", delta: jnp.ndarray) -> jnp.ndarray:
    """Elevon-deflected Phi_fv for the PROPWASH mechanism (Eq. 95):
    Phi_fv @ (I - delta * [zeta_f x]). Distinct from Cmde/Clda (freestream-
    only, whole-vehicle) -- see PhiCoefficients docstring."""
    return coeffs.Phi_fv @ (jnp.eye(3) - delta * skew(coeffs.zeta_f))


def eq14_whole_vehicle_wrench(
    v_b: jnp.ndarray, omega_b: jnp.ndarray, Phi_mv_total: jnp.ndarray,
    coeffs: "PhiCoefficients", rho: float, S: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """General Eq. (14)/(21) whole-vehicle "dry" wrench:
        tau = -0.5*rho*S*eta_norm * C @ Phi @ (C @ eta_b)
    with eta_b=[v_b; omega_b], eta_norm=sqrt(v^2+phi*omega^2) (Eq. 17),
    C=blockdiag(I3, diag(wingspan,chord,wingspan)) (Eq. 19), and the full
    2x2 block Phi = [[Phi_fv, Phi_fw],[Phi_mv_total, Phi_mw]] (Eq. 20).

    Phi_mv_total is passed in rather than read from coeffs because it
    varies with elevon deflection (geometric baseline + elevon-freestream
    contribution) -- see phi_wing_wrench.

    Uses safe_norm for eta_norm (see core/math.py) -- this is exactly the
    term that produced a NaN gradient at hover/pure-axial trim before that
    fix (see chat history). Reusing safe_norm directly (rather than a
    separately hand-rolled epsilon) via a scaled concatenation:
    eta_norm = safe_norm([v_b; sqrt(phi)*omega_b]) = sqrt(v^2+phi*omega^2+eps^2),
    exactly Eq. (17)'s weighted norm.
    """
    B_diag = jnp.array([coeffs.wingspan, coeffs.chord, coeffs.wingspan])
    eta_norm = safe_norm(jnp.concatenate([v_b, jnp.sqrt(coeffs.phi_param) * omega_b]))

    C_eta_b = jnp.concatenate([v_b, B_diag * omega_b])
    Phi = jnp.block([[coeffs.Phi_fv, coeffs.Phi_fw],
                      [Phi_mv_total, coeffs.Phi_mw]])
    Phi_C_eta = Phi @ C_eta_b
    C_Phi_C_eta = jnp.concatenate([Phi_C_eta[:3], B_diag * Phi_C_eta[3:]])

    tau = -0.5 * rho * S * eta_norm * C_Phi_C_eta
    return tau[:3], tau[3:]


# ── core wrench ──────────────────────────────────────────────────────────────

def phi_wing_wrench(
    state: VehicleState,
    params: PhiWingParams,
    thrust_per_prop: jnp.ndarray,
    elevon_commands: jnp.ndarray = jnp.zeros(1),
    wind_velocity: jnp.ndarray = jnp.zeros(3),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute world-frame phi-theory aerodynamic wrench (force, moment).

    Parameters
    ----------
    state : VehicleState
    params : PhiWingParams
    thrust_per_prop : (n_propellers,) scalar thrust magnitude per propeller
        [N], axial along body +x. Computed by the caller (propulsion model),
        not by this module.
    elevon_commands : (n_elevons,) normalised elevon deflection [rad], one
        entry per distinct elevon_index referenced by params.sections, AND
        used directly (as [left,right]) for the whole-vehicle elevator/
        aileron-equivalent split -- see elevon_phi_mv_act.
    wind_velocity : (3,) world-frame wind velocity [m/s]

    Returns
    -------
    F_world : (3,) aerodynamic force in world frame [N]
    T_world : (3,) aerodynamic torque (about CG) in world frame [N*m]

    Structure (revised -- see chat history for the design discussion that
    led here, prompted by switching from placeholder to real fitted
    stability-derivative data):

      1. WHOLE-VEHICLE "dry" term (Eq. 14, eq14_whole_vehicle_wrench):
         freestream force/moment (Phi_fv, Phi_fw), a Phi_mv built from a
         GEOMETRIC baseline (Eq. 67, wing_ac_offset) plus an elevon-in-
         freestream contribution (Cmde/Clda, real if you have them), and
         REAL rate damping (Phi_mw, e.g. Cmq/Clp/Cnr) if supplied. This
         replaces an earlier per-section "local velocity coupling" +
         ad hoc whole-vehicle damping approximation that was a reasonable
         proxy when no real rate-damping data existed, but would double-
         count against real fitted Cmq/Clp/Cnr once you have them.
      2. PER-SECTION propwash term (Eq. 79's F_p_b, unchanged): superposed
         thrust from each section's assigned propellers, elevon-modulated
         (Eq. 95) -- still the ONLY source of elevon authority at hover,
         since the whole-vehicle Cmde/Clda term needs real airspeed.

    These two elevon mechanisms are physically distinct (propwash needs
    thrust flowing over the surface; freestream needs airspeed) and both
    legitimately add -- not a double-count. See chat history for the
    Eq.(70)/(79) linearity argument for why they combine additively
    without a cross term, and for how they still interact: thrust itself
    (via a velocity-dependent BEM lookup, if used) falls off with
    airspeed, so propwash authority fades as freestream rises -- that's
    the actual coupling channel, not amplification.

    No angle-of-attack or sideslip is computed anywhere in this function --
    that is the entire point of phi-theory (Sec. III.B of the paper): it
    stays well-defined and smooth at exactly zero airspeed (hover), which is
    where the classical {alpha, beta}-based approach used elsewhere in this
    repo (uavsim/dynamics/aerodynamics.py) becomes singular and has to be
    patched over with a sigmoid mode blend.
    """
    R = quat_to_rotation_matrix(state.quaternion)

    # Freestream (airspeed) vector in body frame -- same sign convention as
    # the existing uavsim/dynamics/aerodynamics.py: v_inf_b = R.T @ (v - wind).
    v_inf_world = state.velocity - wind_velocity
    v_inf_b = R.T @ v_inf_world
    omega_b = state.angular_velocity  # no-wind-rotation case: omega_inf,b = omega_b (Sec. III.B, below Eq. 18)

    coeffs = params.coeffs
    S_total = sum(section.area for section in params.sections)

    # ── 1. whole-vehicle dry term ────────────────────────────────────────
    Phi_mv_geo = geometric_phi_mv(coeffs.wing_ac_offset, coeffs.Phi_fv,
                                   coeffs.wingspan, coeffs.chord)
    Phi_mv_act = elevon_phi_mv_act(coeffs, elevon_commands)
    F_dry, M_dry = eq14_whole_vehicle_wrench(
        v_inf_b, omega_b, Phi_mv_geo + Phi_mv_act, coeffs, params.rho, S_total)

    # ── 2. per-section propwash term (Eq. 79) ────────────────────────────
    F_total = F_dry
    M_total = M_dry

    for section in params.sections:
        if len(section.prop_indices) == 0:
            continue  # dry section: no propwash, already covered by the whole-vehicle term
        delta_k = (elevon_commands[section.elevon_index]
                   if section.elevon_index is not None else 0.0)
        Phi_fv_k = elevon_phi_fv(coeffs, delta_k)

        Sp_k = sum(params.propellers[j].disk_area for j in section.prop_indices)
        T_k_mag = sum(thrust_per_prop[j] for j in section.prop_indices)
        T_k_vec = jnp.array([T_k_mag, 0.0, 0.0])
        F_p_k = -(section.area / (2.0 * Sp_k)) * (Phi_fv_k @ T_k_vec)

        # Geometric moment (Eq. 67-style: moment = arm x force, applied
        # per-section since prop/section positions differ).
        M_p_k = jnp.cross(section.aero_center, F_p_k)

        F_total = F_total + F_p_k
        M_total = M_total + M_p_k

    F_world = R @ F_total
    T_world = R @ M_total
    return F_world, T_world
