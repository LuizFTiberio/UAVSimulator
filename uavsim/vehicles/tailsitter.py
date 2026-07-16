"""Tail-sitter vehicle — dual-engine flying wing with phi-theory aerodynamics.

Physical configuration mirrors the dual-engine tail-sitting flying wing that
Lustosa, Defay, and Moschetta (2019) validate phi-theory against (see
uavsim/aero/phi_theory.py). Motor command convention follows the paper's own
u = (omega1, omega2, delta1, delta2) (Eq. 100):

    motor_commands = [throttle_left, throttle_right, elevon_left, elevon_right]
    throttle in [0, 1], elevon in [-1, 1] (scaled to +/- max_elevon radians)

Propulsion has two modes (see TailsitterPropulsionParams): a SIMPLE
velocity-independent kf*omega^2 model (Eq. 93/94, fine for hover, wrong for
cruise/transition -- confirmed missing entirely from Cyclone's own thrust
data, see chat history), and a BEM_TABLE mode that reuses
uavsim/dynamics/propulsion_bem.py's real BEM solve/lookup
(github.com/LuizFTiberio/ccblade-jax). propulsion_bem.py originally
hardcoded rotor thrust along body +Z (quad convention) -- generalized to an
arbitrary rotor_axis (this module uses body +X) so it could be reused here
rather than reimplemented; see propulsion_bem.py's own docstring/tests for
that generalization and its backward-compatibility checks.

Known simplification (flagged, not modeled): propeller gyroscopic precession
(the paper's Eq. 88 term) is neglected, matching Sec. VI.C's own stated
approximation for slow spin-axis reorientation.
"""

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from uavsim.core.types import VehicleState
from uavsim.core.math import quat_to_rotation_matrix
from uavsim.aero.phi_theory import (
    PhiCoefficients,
    PhiWingParams,
    PhiWingSection,
    PhiPropeller,
    cessna_style_phi_coefficients,
    phi_wing_wrench,
)
from uavsim.vehicles.base import VehicleModel
from uavsim.dynamics.propulsion_bem import (
    BEMTableParams, build_bem_table, compute_rotor_wrench_bem_table, lookup_bem_table_per_rotor)


# ── parameters ────────────────────────────────────────────────────────────────

class TailsitterPropulsionParams(NamedTuple):
    """Propeller thrust/reaction-torque model.

    Two modes, selected by whether bem_table is set:

    SIMPLE (bem_table=None): T_i=kf*omega_i^2, N_i=spin_sign_i*km*omega_i^2
    (Eq. 93/94) -- velocity-independent. Fine for hover (V~0, where the
    omitted effect vanishes) but WRONG for cruise/transition: real thrust
    falls off substantially with advance ratio (confirmed directly from
    Cyclone's own thrust-vs-RPM data at multiple wind-tunnel speeds -- see
    chat history), which this constant-kf model cannot represent at all.

    BEM_TABLE (bem_table set): real thrust/torque from a validated BEM
    solve (github.com/LuizFTiberio/ccblade-jax, itself validated against
    CCBlade.jl), looked up on a precomputed (omega, V_inf, alpha_tilt)
    grid via uavsim/dynamics/propulsion_bem.py -- fast enough for
    real-time control, properly velocity- and inflow-angle-dependent.
    Build via build_cyclone_bem_table() or your own build_bem_table() call.

    Thrust is always along body +x; spin_signs sets each propeller's
    rotation SENSE (independent of thrust direction/magnitude), which
    only affects reaction-torque sign -- standard counter-rotating
    propeller pair convention (paper: omega1<0, omega2>0, Sec. VI.D), so
    that reaction torques partially cancel at symmetric thrust rather than
    both adding in the same direction.
    """
    kf: float                  # thrust coefficient [N*s^2/rad^2] (SIMPLE mode only)
    km: float                  # reaction-torque coefficient [N*m*s^2/rad^2] (SIMPLE mode only)
    max_omega: float           # [rad/s]
    spin_signs: jnp.ndarray    # (2,) +/-1, rotation sense per propeller
    bem_table: object = None   # BEMTableParams from propulsion_bem.py, or None for SIMPLE mode


class TailsitterParams(NamedTuple):
    mass: float                        # [kg] (reference only -- MuJoCo's own
                                        # body inertial mass is authoritative
                                        # for the actual sim dynamics)
    gravity: float                     # [m/s^2] (reference only, see above)
    propulsion: TailsitterPropulsionParams
    propeller_positions: jnp.ndarray   # (2,3) body frame, [left, right]
    max_elevon: float                  # [rad]
    phi_wing: PhiWingParams


# ── dynamics function ────────────────────────────────────────────────────────

def tailsitter_wrench(
    state: VehicleState,
    motor_commands: jnp.ndarray,
    params: TailsitterParams,
    wind_velocity: jnp.ndarray = jnp.zeros(3),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute world-frame force and torque for a tail-sitter.

    Sums a propulsion wrench block (raw thrust + its moment arm + reaction
    torque, Newton's-third-law reaction of the propellers on the airframe)
    and a phi-theory wing wrench block (freestream + propwash-induced
    aerodynamic force/moment, handles both cleanly through zero airspeed).

    Parameters
    ----------
    state : VehicleState
    motor_commands : (4,) control vector
        indices 0-1: propeller throttle [0, 1]  (left, right)
        indices 2-3: elevon deflection [-1, 1]   (left, right)
    params : TailsitterParams
    wind_velocity : (3,) world-frame wind velocity [m/s]

    Returns
    -------
    F_world : (3,) net force in world frame [N]
    T_world : (3,) net torque in world frame [N*m]
    """
    R = quat_to_rotation_matrix(state.quaternion)

    # ── propulsion ────────────────────────────────────────────────────────
    throttle = jnp.clip(motor_commands[:2], 0.0, 1.0)
    omega = throttle * params.propulsion.max_omega
    x_axis = jnp.array([1.0, 0.0, 0.0])

    if params.propulsion.bem_table is not None:
        v_air_body = R.T @ (state.velocity - wind_velocity)
        thrust_per_prop, _ = lookup_bem_table_per_rotor(
            omega, v_air_body, params.propulsion.bem_table, rotor_axis=x_axis)
        F_prop_body, M_prop_body = compute_rotor_wrench_bem_table(
            omega, v_air_body, params.propulsion.spin_signs,
            params.propulsion.bem_table, params.propeller_positions, rotor_axis=x_axis)
    else:
        thrust_per_prop = params.propulsion.kf * omega ** 2                       # (2,), Eq. 93
        reaction_torque_per_prop = (
            params.propulsion.spin_signs * params.propulsion.km * omega ** 2)     # (2,), Eq. 94
        thrust_vecs_body = jnp.zeros((2, 3)).at[:, 0].set(thrust_per_prop)
        F_prop_body = jnp.sum(thrust_vecs_body, axis=0)
        M_prop_body = jnp.sum(
            jnp.cross(params.propeller_positions, thrust_vecs_body), axis=0)
        M_prop_body = M_prop_body.at[0].add(jnp.sum(reaction_torque_per_prop))

    F_prop = R @ F_prop_body
    T_prop = R @ M_prop_body

    # ── wing (phi-theory, handles freestream + propwash + geometric moment
    #    from off-CG sections + rate damping; see aero/phi_theory.py) ─────
    elevon_cmd = jnp.clip(motor_commands[2:4], -1.0, 1.0) * params.max_elevon
    F_wing, T_wing = phi_wing_wrench(
        state, params.phi_wing, thrust_per_prop, elevon_cmd, wind_velocity)

    return F_prop + F_wing, T_prop + T_wing


# ── parameter factories ──────────────────────────────────────────────────────

def default_phi_wing_params(
    wingspan: float = 0.88,
    chord: float = 0.17,
    rho: float = 1.225,
    prop_y_offset: float = 0.30,
    prop_disk_radius: float = 0.105,
    cg_fraction_chord: float = 0.25,
    elevon_fraction_chord: float = 0.10,
    # Cessna-class stability derivatives (real, fitted data -- via Leandro
    # Lustosa, open-source; see chat history). Used as a starting point
    # ("get coefficients out of some existing airfoil/aircraft" per your
    # framing) WITHOUT changing Cyclone's geometry. These are real numbers
    # for a DIFFERENT airframe (a Cessna-class GA aircraft, not Cyclone's
    # actual custom airfoil), so treat resulting force/moment MAGNITUDES
    # as illustrative pending real Cyclone-specific data, same caveat as
    # before -- but the DERIVATIVE STRUCTURE (how force/moment responds to
    # velocity/rate/elevon) is now real aerodynamics, not a placeholder
    # symmetric-diagonal guess.
    Cd0: float = 0.031,
    Cda: float = 0.13,
    Cla: float = 4.6,
    Cl0: float = 0.31,
    Cy0: float = 0.1,
    Clp: float = -0.47,
    Cmq: float = -12.4,
    Cnr: float = -0.099,
    Cmde: float = -1.28,
    Clda: float = 0.178,
    phi_param: float = 1.0,
) -> PhiWingParams:
    """Default phi-theory wing params: 2 sections (left/right), 2 propellers.

    Geometry defaults are Cyclone's real dimensions (wingspan 0.88 m, chord
    0.17 m -- 0.88*0.17=0.150 m^2, matches the stated wing area; prop
    diameter 0.21 m). prop_y_offset is still a photo-estimate, not
    measured -- confirm/replace, it directly sizes roll/yaw authority.

    TWO DISTINCT chordwise offsets, easy to conflate, kept separate on
    purpose (see phi_theory.py's PhiCoefficients docstring):
      - delta_r (per-section aero_center x-offset): the ELEVON's own
        position behind CG, for the PROPWASH mechanism -- unchanged from
        before, still the only elevon authority source at hover.
      - wing_ac_offset (whole-vehicle, geometric Phi_mv): the WING's own
        aerodynamic-center offset from CG, for the FREESTREAM static-
        margin term. Zero here (CG at 25% chord = classical thin-airfoil
        quarter-chord AC = wing_ac_offset would be zero) -- deliberately
        NOT borrowing a real aircraft's directly-fit Cma/Clb/Cnb (which
        bundle in tail/fuselage contributions this tail-sitter lacks).
    """
    section_area = (wingspan / 2.0) * chord
    elevon_center_fraction = 1.0 - elevon_fraction_chord / 2.0
    delta_r = -(elevon_center_fraction - cg_fraction_chord) * chord
    wing_ac_fraction = 0.25  # classical thin-airfoil quarter-chord
    wing_ac_offset = -(wing_ac_fraction - cg_fraction_chord) * chord  # 0.0 for the default 25% CG

    coeffs = cessna_style_phi_coefficients(
        Cd0=Cd0, Cda=Cda, Cla=Cla, Cl0=Cl0, Cy0=Cy0,
        Clp=Clp, Cmq=Cmq, Cnr=Cnr, Cmde=Cmde, Clda=Clda,
        wingspan=wingspan, chord=chord,
        wing_ac_offset=jnp.array([wing_ac_offset, 0.0, 0.0]),
        zeta_f=jnp.array([0.0, 0.3, 0.0]),  # propwash elevon effectiveness, still placeholder
        phi_param=phi_param,
    )
    propellers = (
        PhiPropeller(position=jnp.array([0.0, -prop_y_offset, 0.0]),
                     disk_area=jnp.pi * prop_disk_radius ** 2),
        PhiPropeller(position=jnp.array([0.0, prop_y_offset, 0.0]),
                     disk_area=jnp.pi * prop_disk_radius ** 2),
    )
    sections = (
        PhiWingSection(area=section_area, aero_center=jnp.array([delta_r, -prop_y_offset, 0.0]),
                        prop_indices=(0,), elevon_index=0),
        PhiWingSection(area=section_area, aero_center=jnp.array([delta_r, prop_y_offset, 0.0]),
                        prop_indices=(1,), elevon_index=1),
    )
    return PhiWingParams(coeffs=coeffs, rho=rho, sections=sections, propellers=propellers)


def tailsitter_params(
    mass: float = 1.2,
    gravity: float = 9.81,
    max_omega: float = 1200.0,
    kf: float | None = None,
    km_ratio: float = 0.02,
    prop_y_offset: float = 0.3,
    max_elevon: float = jnp.radians(25.0),
    phi_wing: PhiWingParams | None = None,
    bem_table: BEMTableParams | None = None,
) -> TailsitterParams:
    """Create tail-sitter parameters.

    kf : thrust coefficient [N*s^2/rad^2], SIMPLE-mode only (ignored if
        bem_table is set). If None (default), kf is sized so combined
        2-propeller thrust supports weight at ~50% throttle -- a
        placeholder convenience for generic/test use, NOT a physical
        constraint. Pass a real kf (fit from actual thrust-vs-RPM data)
        for an actual vehicle -- see cyclone_params() for a worked example.

    bem_table : if set, propulsion uses the real velocity-dependent BEM
        lookup (see TailsitterPropulsionParams docstring) instead of
        SIMPLE kf*omega^2 -- see build_cyclone_bem_table() /
        cyclone_hover_capable_params() for a worked example.
    """
    if kf is None:
        kf = (mass * gravity / 2.0) / (0.5 * max_omega) ** 2
    km = kf * km_ratio

    propulsion = TailsitterPropulsionParams(
        kf=kf, km=km, max_omega=max_omega,
        spin_signs=jnp.array([-1.0, 1.0]),   # counter-rotating pair (Sec. VI.D convention)
        bem_table=bem_table,
    )
    propeller_positions = jnp.array([
        [0.0, -prop_y_offset, 0.0],   # left
        [0.0,  prop_y_offset, 0.0],   # right
    ])

    if phi_wing is None:
        phi_wing = default_phi_wing_params(prop_y_offset=prop_y_offset)

    return TailsitterParams(
        mass=mass,
        gravity=gravity,
        propulsion=propulsion,
        propeller_positions=propeller_positions,
        max_elevon=max_elevon,
        phi_wing=phi_wing,
    )


def cyclone_params() -> TailsitterParams:
    """Parameters for 'Cyclone': wingspan 0.88 m, chord 0.17 m (S=0.150 m^2,
    matches stated wing area), prop diameter 0.21 m, MTOW 1.5 kg,
    AXI2212-26 motor + HQ-Prop 8x5.

    kf = 5.86e-6 N*s^2/rad^2, least-squares fit to the static (0 m/s)
    thrust-vs-RPM curve at RPM in [6500, 9200] (avoiding the more
    nonlinear low-RPM region). NOT high-precision -- read off a plot, not
    raw log data; refine with real thrust-test data or the differentiable
    ccblade model when available.

    STILL PLACEHOLDER, not from real data (flagged, needs your numbers):
      - km (reaction-torque coefficient): the given data has thrust and
        efficiency curves, not torque -- nothing to fit km from yet.
      - Cd0, Cy0 (Phi_fv0 lift/drag coefficients), zeta_f_pitch
        (elevon effectiveness), Phi_momega_gain (rate damping): aero
        coefficients, not geometry -- you mentioned looking these up
        separately.
      - prop_y_offset: estimated from the vehicle photo (~0.30 m), not
        measured. This directly sizes roll/yaw control authority --
        worth confirming/replacing with the real spanwise mount position.
    """
    mass = 1.5
    max_omega = 9500.0 * 2.0 * jnp.pi / 60.0   # ~995 rad/s, top of the given RPM data
    kf = 5.86e-6

    phi_wing = default_phi_wing_params(
        wingspan=0.88, chord=0.17, prop_disk_radius=0.21 / 2.0, prop_y_offset=0.30,
    )
    return tailsitter_params(
        mass=mass, max_omega=float(max_omega), kf=kf,
        prop_y_offset=0.30, phi_wing=phi_wing,
    )


def build_cyclone_bem_table(
    omega_max: float = 950.0,
    n_omega: int = 12,
    v_inf_max: float = 30.0,
    n_vinf: int = 10,
    n_alpha: int = 8,
) -> BEMTableParams:
    """Build a real BEM thrust/torque table for Cyclone's resized (11 in)
    hover-capable propeller, via github.com/LuizFTiberio/ccblade-jax
    (validated against CCBlade.jl).

    Same blade design used for the diameter-sizing exercise (see chat
    history): 2-blade NACA 4412, D=11in (0.2794m), constant geometric
    pitch = 0.6*D, linear chord taper 10%->4% of D root-to-tip. This is a
    representative starting design, NOT validated against a manufactured
    propeller -- see cyclone_hover_capable_params()'s docstring.

    v_inf_max=30 m/s matches Cyclone's own stated flight speed range
    (Table 3: 0-30 m/s), so the table covers hover through full cruise
    speed. alpha_tilt spans the full [0, pi] range to also cover
    transition/off-axis inflow, not just trimmed axial flight.

    Requires ccblade-jax on PYTHONPATH to BUILD (raises ImportError via
    build_bem_table otherwise) -- but the resulting table is plain jnp
    arrays, so simulating with it afterward does NOT need ccblade-jax
    available; only building/rebuilding the table does.
    """
    import bem
    from bem.core import load_airfoil, BladeGeom, RotorParams
    # Prefer the airfoil data shipped next to the installed ccblade-jax package
    # (robust across machines); fall back to a cwd-relative copy.
    candidates = [
        Path(bem.__file__).resolve().parent.parent / "data" / "naca4412.dat",
        Path("data/naca4412.dat"),
    ]
    naca_path = next((str(c) for c in candidates if c.exists()), None)
    if naca_path is None:
        raise FileNotFoundError(
            "Could not find naca4412.dat next to the ccblade-jax package "
            f"(looked in {[str(c) for c in candidates]}).")
    af = load_airfoil(naca_path)

    D = 11 * 0.0254
    R = D / 2.0
    Rhub = 0.15 * R
    r = jnp.linspace(Rhub + 0.02 * R, 0.98 * R, 12)
    pitch = 0.6 * D
    twist = jnp.arctan(pitch / (2 * jnp.pi * r))
    c_root = 0.10 * D
    c_tip = 0.4 * c_root
    chord = c_root + (c_tip - c_root) * (r - Rhub) / (R - Rhub)

    geom = BladeGeom(r=r, chord=chord, twist=twist)
    rotor = RotorParams(Rhub=Rhub, Rtip=R, B=2)

    return build_bem_table(
        geom, af, rotor, rho=1.225,
        omega_grid=jnp.linspace(50.0, omega_max, n_omega),
        vinf_grid=jnp.linspace(0.0, v_inf_max, n_vinf),
        alpha_grid=jnp.linspace(0.0, jnp.pi, n_alpha),
    )


def cyclone_hover_capable_params(bem_table: BEMTableParams | None = None) -> TailsitterParams:
    """Cyclone with a LARGER, BEM-sized propeller so it can actually hover.

    cyclone_params() (real, as-measured 8x5 prop) has thrust-to-weight ~0.79
    at MTOW -- structurally cannot hover, confirmed in chat history. This
    variant keeps everything else (wing, mass, elevon geometry) the same
    but swaps in a propeller sized via a real BEM solve
    (github.com/LuizFTiberio/ccblade-jax, validated against CCBlade.jl) to
    hit T/W ~1.5 at hover trim, ~2.0 at max throttle.

    This is NOT the physical vehicle's actual propeller -- it's a
    resized-for-hover variant, for testing/validating the hover controller
    against a vehicle that can actually execute the maneuver, pending
    either a real larger prop being fitted to the airframe or confirmation
    from the authors that hover was never the intended flight mode (see
    chat history).

    bem_table : if provided, used directly (e.g. a cached table -- building
        one takes a few seconds). If None (default), builds one via
        build_cyclone_bem_table() -- requires ccblade-jax on PYTHONPATH.
        Propulsion then uses the REAL velocity-dependent BEM lookup, not
        the velocity-blind kf*omega^2 SIMPLE model -- this is what makes
        cruise/transition behavior physically meaningful rather than
        silently wrong (SIMPLE mode has no advance-ratio effect at all;
        Cyclone's own thrust data shows thrust roughly halving from 0 to
        15-20 m/s at fixed RPM -- see chat history).

    Design point (same blade as the earlier sizing exercise): D=11in
    (0.279 m), hover trim ~7540 RPM, T/W~1.98 at max_omega=908 rad/s
    (15% headroom over hover trim). km is now real (from the BEM solve's
    own torque, not a km_ratio guess).
    """
    if bem_table is None:
        bem_table = build_cyclone_bem_table()

    D = 11 * 0.0254
    phi_wing = default_phi_wing_params(
        wingspan=0.88, chord=0.17, prop_disk_radius=D / 2.0, prop_y_offset=0.30,
    )
    return tailsitter_params(
        mass=1.5, max_omega=908.0, kf=1.77e-5, km_ratio=0.0161,
        prop_y_offset=0.30, phi_wing=phi_wing, bem_table=bem_table,
    )


def tailsitter(
    params: TailsitterParams | None = None,
    mjcf_path: Path | str | None = None,
) -> VehicleModel:
    """Create a complete tail-sitter vehicle model."""
    if params is None:
        params = tailsitter_params()
    if mjcf_path is None:
        mjcf_path = Path(__file__).parent.parent / "models" / "tailsitter.xml"
    else:
        mjcf_path = Path(mjcf_path)

    return VehicleModel(
        params=params,
        mjcf_path=mjcf_path,
        compute_wrench=tailsitter_wrench,
        actuator_names=("act_left", "act_right", "act_elevon_left", "act_elevon_right"),
        spin_signs=(-1.0, 1.0, 0.0, 0.0),
    )
