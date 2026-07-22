"""Incremental Nonlinear Dynamic Inversion (INDI) control for the tail-sitter.

Single continuous attitude+rate controller meant to span the whole envelope
(hover -> transition -> cruise) with no mode switch -- the core claim of

    Smeur, Bronz, de Croon, "Incremental Control and Guidance of Hybrid
    Aircraft Applied to a Tailsitter Unmanned Air Vehicle," JGCD 2019
    (arxiv.org/pdf/1802.00714) -- flight-validated on the real Cyclone.

Equation numbers below are from that paper. Reference implementations for the
algorithm pattern: enac-drones/dronesim (quad, numpy) and Paparazzi's
stabilization_indi.c / wls_alloc.c (the original flight code).

How this differs from tailsitter_hover.py's mixer
-------------------------------------------------
compute_hover_mixer computes the control-effectiveness Jacobian ONCE at a
fixed hover trim and inverts it -- a fixed linear mixer, correct only near
that trim. INDI recomputes G = d[Omega_dot; T]/d[u] fresh from the CURRENT
state every step (still via jax.jacobian on the real tailsitter_wrench), and
drives the *increment* from the measured angular acceleration:

    u_c = u_f + G^+ (nu - [Omega_dot_meas; T_meas])          (Eq. 2-3)

The model enters only through G; the baseline Omega_dot comes from a
finite-difference of the gyro (Sec. 2). That's what makes one controller work
across the envelope instead of one mixer per trim.

Key advantage over the original Cyclone work (TODO roadmap): they had to fit
G(theta, V) from flight data (Eq. 7-12) because they had no aero model. We
have phi_theory + BEM, so G is just jax.jacobian at the current state --
continuous and correct everywhere, no scheduling tables.

Filtering (Sec. 2, "synchronisation"): the measured angular acceleration and
the actuator command feeding G MUST pass through the same low-pass filter, or
the incremental law is inconsistent (the accel you measure must correspond to
the command you subtract). A single first-order filter with time constant
`filt_tau` is applied to both here. In this noise-free sim the filter mostly
just adds matched lag; it matters for stability of the algebraic loop and for
eventual sim2real (real gyro + real actuator lag).

Scope of this module (Phase A):
  * Inner attitude+rate INDI loop (steps 1-3) + WLS allocation (step 4) --
    the core, validated. G recomputed from the current state every step.
  * Outer position/velocity loop (step 6). Two variants:
      - Model-based (DEFAULT, use_indi_outer=False): desired acceleration ->
        thrust vector = m*(a_ref - g). Continuous, stable, and it flies the
        full hover<->cruise<->hover transition (the inner INDI loop is what
        makes that a single controller with no mode switch). This is the
        validated outer loop.
      - Measured-acceleration INDI (use_indi_outer=True, EXPERIMENTAL): the
        paper's Sec. 4 incremental form, thrust vector = m*g*b1 + m*(a_ref -
        a_meas), with the hover-thrust (m*g) baseline used by the Cyclone/
        dronesim/Paparazzi position loops. Stable but holds less tightly than
        the model-based path (steady-state tilt) -- the Sec. 4.1 non-minimum-
        phase / cascade-bandwidth problem. A focused follow-up, not shipped
        as default.

Desired attitude uses a span-axis construction (desired_attitude_transition)
that, unlike the hover-only thrust-direction TRIAD, stays well-defined when
the thrust axis goes horizontal in cruise.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from uavsim.core.types import VehicleState
from uavsim.core.math import quat_to_rotation_matrix
from uavsim.vehicles.tailsitter import TailsitterParams, tailsitter_wrench
from uavsim.aero.phi_theory import phi_wing_wrench
from uavsim.controllers.wls_alloc import wls_alloc
# Reuse the singularity-free attitude machinery -- no reason to reimplement
# quaternion-error feedback (roadmap step 3); this rotation-matrix error has
# no Euler singularity at the tail-sitter's own +/-90deg hover attitude.
from uavsim.controllers.tailsitter_hover import attitude_error

U_MIN = np.array([0.0, 0.0, -1.0, -1.0])   # [thr_L, thr_R, elevon_L, elevon_R]
U_MAX = np.array([1.0, 1.0, 1.0, 1.0])

# Idle throttle floor. With T = kf*omega^2 and omega = throttle*max_omega, the
# thrust derivative dT/dthrottle = 2*kf*max_omega^2*throttle is EXACTLY ZERO at
# zero throttle -- so a propeller commanded fully off contributes an identically
# zero column to G, making the 4x4 effectiveness exactly rank-deficient and the
# allocation's inverse meaningless. Measured on the wind mission: cond(G) reached
# 4.6e11 (null direction exactly the dead throttle's axis) over 241 steps, all
# with a throttle at 0; a floor takes that to a single step. The cliff is sharp,
# not gradual -- cond is already ~25 at throttle 0.02 and only diverges at
# exactly 0 -- so this is cheap insurance rather than a tuning knob. Also
# physically right: ESCs should not stop a prop mid-flight.
# NOTE this does NOT remove the actuator-command spikes; those are dominated by
# elevon saturation in the hover-recovery leg (a control-authority problem), and
# flooring the throttle moved them by only 2 of 39 events.
MIN_THROTTLE = 0.05

# Below this forward (body-x) airspeed the sideslip angle is meaningless
# (a lateral component over ~zero forward speed carries no information) and the
# g*tan(phi)/V coordinated-turn term blows up, so the sideslip loop holds its
# heading below it. 2 m/s is comfortably into forward flight for this vehicle.
V_SIDESLIP_MIN = 2.0


# ── transition-capable desired attitude ─────────────────────────────────────

def desired_attitude_transition(b1_des: jnp.ndarray, heading: float) -> jnp.ndarray:
    """Build R_des (columns = body axes in world) from the desired thrust axis
    b1_des and a horizontal heading, using the WING/SPAN axis as the secondary
    reference rather than the heading direction itself.

    tailsitter_hover.desired_attitude_from_thrust_direction resolves the free
    roll about b1 with b2 = b1 x h (h = heading direction). That degenerates in
    cruise: the thrust axis there is nearly horizontal and nearly parallel to
    h, so b1 x h -> 0. Here the secondary reference is the span direction
    span = [-sin(heading), cos(heading), 0] (horizontal, perpendicular to the
    plane of flight), which the thrust axis never aligns with across a normal
    hover<->cruise transition (thrust stays in the vertical plane of travel).

    Reduces to exactly the hover construction for heading = 0 (verified across
    the whole b1 range), so it is a safe drop-in that additionally works
    through transition and cruise.
    """
    span = jnp.array([-jnp.sin(heading), jnp.cos(heading), 0.0])
    b3 = jnp.cross(b1_des, span)
    b3 = b3 / jnp.maximum(jnp.linalg.norm(b3), 1e-6)
    b2 = jnp.cross(b3, b1_des)
    b2 = b2 / jnp.maximum(jnp.linalg.norm(b2), 1e-6)
    return jnp.stack([b1_des, b2, b3], axis=1)


# ── sideslip estimation + coordinated-turn law (Phase B, Sec. 3) ─────────────

def estimate_sideslip(
    state: VehicleState, wind_velocity: np.ndarray = np.zeros(3),
) -> tuple[float, float, np.ndarray]:
    """Return (beta, V, v_body) -- sideslip angle, airspeed, body-frame airspeed.

    The Cyclone has no rudder or vertical tail, so nothing passively prevents
    sideslip; left uncontrolled it degrades wing efficiency and lift (paper
    Sec. 3). Sideslip is a nonzero airspeed component along the SPAN axis
    (body y -- the propellers sit at y=+/-0.3), i.e. the airflow hitting the
    wing edge-on rather than head-on:

        beta = asin(v_body_y / V)

    Note this is the standard asin-over-TOTAL-airspeed definition, NOT
    atan2(v_body_y, v_body_x). The two agree exactly whenever v_body_z = 0 and
    v_body_x > 0 (so they are interchangeable in trimmed forward flight), but
    atan2 is discontinuous: it wraps through +/-180 deg the instant v_body_x
    goes negative, and at hover -- where body-x points UP, so v_body_x is just
    the vertical airspeed hovering around zero -- it flips between ~0 and ~+/-180
    every time that crosses. That produced exactly the 360 deg beta spikes seen
    when logging sideslip through a hover leg or in wind. asin is bounded to
    +/-90 deg and continuous through v_body_x = 0, so it never wraps.

    This uses ground-truth simulator state (roadmap Phase B step 1, approach a
    -- "cheating" relative to a real vehicle but exact and fine for sim/RL). A
    real Cyclone would instead reconstruct beta from the lateral specific force
    (accelerometer, paper Eq. 13-16) -- noted as refinement (b), only needed if
    sim2real transfer matters.

    beta is only meaningful in forward flight; the caller gates on
    v_body[0] > V_SIDESLIP_MIN (below that there is no meaningful "forward" for
    the flow to slip away from).
    """
    R = np.asarray(quat_to_rotation_matrix(state.quaternion))
    v_air_world = np.asarray(state.velocity, dtype=float) - np.asarray(wind_velocity, dtype=float)
    v_body = R.T @ v_air_world
    V = float(np.linalg.norm(v_body))
    beta = float(np.arcsin(np.clip(v_body[1] / max(V, 1e-6), -1.0, 1.0)))
    return beta, V, v_body


def coordinated_turn_rate(bank: float, V: float, gravity: float) -> float:
    """Coordinated-turn heading-rate feedforward (paper Eq. 17, feedforward part):

        psi_dot_ff = g * tan(bank) / V

    When the vehicle banks by `bank` the lift/thrust vector tilts, producing a
    horizontal (centripetal) force g*tan(bank) per unit mass, which curves the
    flight path at exactly g*tan(bank)/V -- so rotating the heading reference at
    that rate keeps the nose tracking the turning velocity, holding beta~0
    through the turn instead of picking up sideslip.

    This is a GUIDANCE-side helper: pass the result as `yaw_rate_ff` to
    TailsitterINDIController.update(). It is deliberately NOT computed from the
    vehicle's *measured* bank inside the loop, because in this architecture bank
    is emergent -- the outer loop banks to hold lateral POSITION (e.g. crabbing
    against a crosswind), not only to turn -- so feeding measured bank back as a
    turn command misfires and actually degrades sideslip regulation (verified:
    measured-bank feedforward pushed a crosswind's steady sideslip from ~5deg
    back up to ~20deg). The turn rate must come from the commanded path (Phase C
    guidance), which is what this helper expresses.
    """
    V_eff = max(V, V_SIDESLIP_MIN)   # guard the 1/V feedforward at low speed
    return gravity * np.tan(bank) / V_eff


# ── composite inertia from the MuJoCo model ─────────────────────────────────

def composite_inertia_from_model(model, body_name: str = "body") -> np.ndarray:
    """Composite 3x3 rotational inertia of the vehicle about its own CoM,
    expressed in the `body_name` body frame, read straight from the MuJoCo
    model so it matches the actual sim dynamics.

    INDI's G maps command -> angular acceleration = I^-1 @ dM/du, so it needs
    the inertia the sim actually integrates. For this tail-sitter that is NOT
    just the fuselage's diaginertia: the two propeller bodies sit at y=+/-0.3 m,
    and their mass * offset^2 adds ~30% to roll and yaw inertia (parallel
    axis). Getting this wrong only scales G, which INDI's measured-Omega_dot
    feedback partly corrects -- but there's no reason to be sloppy when the
    exact composite is one forward-kinematics call away.

    Sums, over the subtree rooted at `body_name`, each body's own inertia
    (rotated into world) plus its parallel-axis term about the composite CoM,
    then rotates the world-frame result into the root body frame.
    """
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if root < 0:
        raise ValueError(f"body {body_name!r} not found in model")

    def in_subtree(i: int) -> bool:
        while i > 0:
            if i == root:
                return True
            i = model.body_parentid[i]
        return root == 0

    ids = [i for i in range(model.nbody) if i != 0 and in_subtree(i)]
    masses = np.array([model.body_mass[i] for i in ids])
    coms = np.array([data.xipos[i] for i in ids])          # world-frame CoMs
    com = (masses[:, None] * coms).sum(0) / masses.sum()

    I = np.zeros((3, 3))
    for i, m, ci in zip(ids, masses, coms):
        Ri = data.ximat[i].reshape(3, 3)                   # inertial frame -> world
        Ii = Ri @ np.diag(model.body_inertia[i]) @ Ri.T
        d = ci - com
        Ii += m * (float(d @ d) * np.eye(3) - np.outer(d, d))
        I += Ii

    Rb = data.xmat[root].reshape(3, 3)                     # body -> world
    return Rb.T @ I @ Rb


# ── control-effectiveness (jitted jax.jacobian at the current state) ────────

def indi_effectiveness(
    state: VehicleState,
    u0: jnp.ndarray,
    params: TailsitterParams,
    inertia_inv: jnp.ndarray,
    wind_velocity: jnp.ndarray = jnp.zeros(3),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (G, y0) at the current state.

    y(u) = [Omega_dot_body(u); F_body_x(u)] -- 3 body angular accelerations
    (I^-1 @ moment) plus body-x force (thrust axis). G = dy/du is the 4x4
    control-effectiveness; y0 = y(u0) supplies the modeled thrust baseline
    (the angular-accel baseline comes from the gyro instead, see the class).

    The Omega x I Omega gyroscopic term is dropped: it's independent of u so
    it vanishes from the Jacobian anyway, and INDI's baseline for the angular
    channels is the *measured* Omega_dot, not this modeled y0.
    """
    R = quat_to_rotation_matrix(state.quaternion)

    def y(u):
        F_world, T_world = tailsitter_wrench(state, u, params, wind_velocity)
        M_body = R.T @ T_world
        F_body_x = (R.T @ F_world)[0]
        ang_acc = inertia_inv @ M_body
        return jnp.concatenate([ang_acc, F_body_x[None]])

    return jax.jacobian(y)(u0), y(u0)


# ── Guidance INDI outer loop (paper Sec. 4, Eq. 18-33) ───────────────────────
#
# The paper's outer loop is ALSO INDI: it controls linear acceleration with the
# control vector v = [phi, theta, T] (bank, pitch, thrust) by inverting a
# control-effectiveness matrix G = G_T + G_L that -- crucially -- includes the
# derivative of WING LIFT w.r.t. pitch (Eq. 27-29). That lift term is what lets
# the controller trade thrust for pitch: pitch the nose over, let the wing pick
# up the weight, and pull thrust back. The model-based outer loop (default
# above) has no lift term, so it can only cancel gravity with thrust -- which is
# why it never unloads onto the wing and the props keep carrying the weight in
# cruise.
#
#     v = v_f + m (G_T + G_L)^-1 (xi_ddot_ref - xi_ddot_meas)          (Eq. 30)
#
# Parameterisation note (differs from the paper's frame, same idea): our thrust
# is along body-x (b1) and hover is nose-up, so the paper's roll-tilts-thrust
# does not map 1:1 (a bank ABOUT b1 makes no lateral force at hover, where lift
# ~ 0). We instead parameterise the thrust-axis DIRECTION by two tilt angles --
# theta (fore-aft, the angle-of-attack / lift driver) and lam (lateral) -- plus
# the thrust magnitude T. This spans 3D acceleration at hover AND cruise
# (verified well-conditioned across the envelope), and theta still carries the
# lift derivative exactly as the paper's theta does. Heading psi stays owned by
# the Phase-B sideslip loop ("psi is not free to choose", paper Sec. 4).
#
# Two effectiveness backends:
#   * "jacobian" (DEFAULT): G = jax.jacobian of the REAL specific force
#     (phi_theory + propulsion) w.r.t. [theta, lam, T] at the current state.
#     Exact lift derivative, continuous across the envelope, no fitting -- the
#     "we have a model, the Cyclone work didn't" advantage, applied to the
#     outer loop (same trick the inner loop already uses for its G).
#   * "analytic": the paper's Eq. 28-33 form, assembled explicitly from a
#     simplified thrust+lift model with a FITTED lift slope (Eq. 33) and the
#     Sec. 4.1 pitch-effectiveness over-statement. No autodiff of the true
#     plant -- the path you need for a real vehicle, where you cannot jacobian
#     an unknown aircraft's aerodynamics. Less accurate here, kept for sim2real.

def _heading_axis(psi: float) -> jnp.ndarray:
    """Horizontal forward (heading) axis -- the bank/roll axis (see below)."""
    return jnp.array([jnp.cos(psi), jnp.sin(psi), 0.0])


def _rot_about_axis(axis: jnp.ndarray, angle: float) -> jnp.ndarray:
    """Rodrigues rotation matrix about a unit `axis` by `angle`."""
    a = axis
    K = jnp.array([[0.0, -a[2], a[1]],
                   [a[2], 0.0, -a[0]],
                   [-a[1], a[0], 0.0]])
    return (jnp.eye(3) + jnp.sin(angle) * K
            + (1.0 - jnp.cos(angle)) * (K @ K))


def guidance_thrust_axis(theta: float, psi: float) -> jnp.ndarray:
    """World-frame thrust axis b1 for pitch-from-vertical `theta`, heading `psi`
    (zero bank): b1 = [sin(theta) cos(psi), sin(theta) sin(psi), cos(theta)].
    theta=0 -> straight up (hover); theta=pi/2 -> horizontal along heading."""
    return jnp.array([jnp.sin(theta) * jnp.cos(psi),
                      jnp.sin(theta) * jnp.sin(psi),
                      jnp.cos(theta)])


def guidance_attitude(phi: float, theta: float, psi: float) -> jnp.ndarray:
    """Desired attitude R for the outer control vector [phi (bank), theta
    (pitch), psi (heading)].

        R = Rot(h, phi) @ desired_attitude_transition(b1(theta, psi), psi)

    where h = horizontal heading axis. Bank `phi` is a roll about h -- which
    does DOUBLE DUTY, exactly as the paper's single roll channel: at hover
    (b1 up) rolling about the horizontal h tilts the thrust axis sideways
    (lateral thrust authority); in cruise (b1 ~ h) it rolls the wing/lift
    vector into the turn (banked coordinated turn). phi=0 reduces to the
    wings-level construction, so it is a safe superset of the earlier attitude.
    """
    b1 = guidance_thrust_axis(theta, psi)
    R0 = desired_attitude_transition(b1, psi)          # wings level
    return _rot_about_axis(_heading_axis(psi), phi) @ R0


def guidance_extract_bank_pitch(R: jnp.ndarray, psi: float) -> tuple[float, float]:
    """Inverse of guidance_attitude: recover (phi, theta) from a world attitude
    R at heading psi -- the INDI baseline v_f (increment added to current state).

    theta comes from the bank-invariant projection b1.h = sin(theta) (rolling
    about h leaves it unchanged, so this is valid across the whole envelope,
    incl. cruise where bank does not move b1). phi is then the rotation about h
    that maps the wings-level reference R0(theta, psi) onto R.
    """
    h = _heading_axis(psi)
    b1 = R[:, 0]
    theta = jnp.arcsin(jnp.clip(jnp.dot(b1, h), -1.0, 1.0))
    R0 = desired_attitude_transition(guidance_thrust_axis(theta, psi), psi)
    M = R @ R0.T                                        # = Rot(h, phi)
    vee = jnp.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    sin_phi = 0.5 * jnp.dot(vee, h)
    cos_phi = 0.5 * (jnp.trace(M) - 1.0)
    phi = jnp.arctan2(sin_phi, cos_phi)
    return phi, theta


def _rot_to_quat(R: jnp.ndarray) -> jnp.ndarray:
    """Rotation matrix -> quaternion [w,x,y,z], differentiable for rotation
    angles < 180 deg (always true in normal flight); single-branch trace form."""
    w = 0.5 * jnp.sqrt(jnp.maximum(1.0 + R[0, 0] + R[1, 1] + R[2, 2], 1e-12))
    x = (R[2, 1] - R[1, 2]) / (4.0 * w)
    y = (R[0, 2] - R[2, 0]) / (4.0 * w)
    z = (R[1, 0] - R[0, 1]) / (4.0 * w)
    return jnp.array([w, x, y, z])


def guidance_accel(
    phi: float, theta: float, T: float, psi: float,
    velocity: jnp.ndarray, params: TailsitterParams, mass: float,
    wind_velocity: jnp.ndarray = jnp.zeros(3),
) -> jnp.ndarray:
    """World-frame specific force (acceleration) as a function of the outer-loop
    control vector [phi (bank), theta (pitch), T (thrust)], using the REAL
    phi_theory + propulsion model at the given airspeed. jax.jacobian of this
    gives the "jacobian" effectiveness G = d(accel)/d[phi, theta, T].

    Because the attitude is built with a genuine bank (guidance_attitude), the
    phi-column of G AUTOMATICALLY carries the lift-roll term -- banking rolls the
    wing, phi_theory returns the tilted lift -- so the coordinated turn falls out
    of the inversion (matching Paparazzi's Gmat[0][*] 'lift'/'liftd' entries), no
    feedforward. Elevons held at zero (inner-loop control); rate zero (quasi-
    static force balance).
    """
    R = guidance_attitude(phi, theta, psi)
    q = _rot_to_quat(R)
    st = VehicleState(position=jnp.zeros(3), quaternion=q,
                      velocity=velocity, angular_velocity=jnp.zeros(3), time=0.0)
    thrust_pp = jnp.array([T / 2.0, T / 2.0])
    F_aero, _ = phi_wing_wrench(st, params.phi_wing, thrust_pp,
                                jnp.zeros(2), wind_velocity)
    g_vec = jnp.array([0.0, 0.0, -params.gravity])
    return g_vec + (F_aero + T * R[:, 0]) / mass


def guidance_effectiveness_analytic(
    phi: float, theta: float, T: float, psi: float, V: float,
    mass: float, gravity: float, lift_slope: float, pitch_scaling: float,
) -> jnp.ndarray:
    """Paparazzi/paper Eq. 28-33 effectiveness assembled explicitly (no autodiff
    of the plant) -- the sim2real backend. Columns are d(accel)/d[phi, theta, T].

    Thrust vector T*b1 and a lift vector L*lift_dir (L ~ mg*sin(theta), rolled
    with the bank), both built with a genuine bank so the phi-column carries the
    LIFT (like Paparazzi's Gmat[0][0] = ... + cphi*spsi*lift):
      * d/dphi   = (T (h x b1) + L (h x lift_dir)) / m     -- bank rotates thrust
        (hover) and lift (cruise) about h -> lateral/turn authority.
      * d/dtheta = (T db1/dtheta - pitch_scaling*lift_slope * lift_dir) / m
        -- thrust rotation + the FITTED lift slope (Eq. 33, negative in this
        convention, see the jacobian backend), scaled by the Sec. 4.1 fudge.
      * d/dT     = b1 / m.
    """
    h = _heading_axis(psi)
    Rphi = _rot_about_axis(h, phi)
    b1_0 = guidance_thrust_axis(theta, psi)
    b1 = Rphi @ b1_0
    up = jnp.array([0.0, 0.0, 1.0])
    lift_dir = Rphi @ up                                # wings-level lift ~ up, rolled
    L = mass * gravity * jnp.sin(theta)                 # crude lift magnitude (Eq. 31)

    db1_0_dtheta = jnp.array([jnp.cos(theta) * jnp.cos(psi),
                              jnp.cos(theta) * jnp.sin(psi),
                              -jnp.sin(theta)])
    db1_dtheta = Rphi @ db1_0_dtheta

    col_phi = (T * jnp.cross(h, b1) + L * jnp.cross(h, lift_dir)) / mass
    col_theta = (T * db1_dtheta - pitch_scaling * lift_slope * lift_dir) / mass
    col_T = b1 / mass
    return jnp.stack([col_phi, col_theta, col_T], axis=1)


def paper_lift_slope(V: float, mass: float) -> float:
    """Paper Eq. 33: dL/dtheta scheduled by airspeed (their flight-test fit).
    Returned as a positive magnitude of d|Lift|/dtheta [N/rad]."""
    return float(np.where(V < 12.0, 24.0 * mass, (V - 8.5) * 6.88 * mass))


# ── gains ───────────────────────────────────────────────────────────────────

class TailsitterINDIGains(NamedTuple):
    kp_pos: jnp.ndarray   # (3,) outer position P gain
    kd_pos: jnp.ndarray   # (3,) outer velocity D gain
    k_att: jnp.ndarray    # (3,) attitude loop: e_R -> Omega_ref  (Eq. 5-6)
    k_rate: jnp.ndarray   # (3,) rate loop: rate error -> Omega_dot_ref (Eq. 4)
    filt_tau: float       # (s) low-pass time constant, gyro-diff AND command
    k_beta: float = 1.5   # sideslip feedback gain (Phase B, Eq. 17)


def default_tailsitter_indi_gains() -> TailsitterINDIGains:
    return TailsitterINDIGains(
        kp_pos=jnp.array([2.0, 2.0, 4.0]),
        kd_pos=jnp.array([2.5, 2.5, 3.5]),
        k_att=jnp.array([8.0, 8.0, 4.0]),
        k_rate=jnp.array([18.0, 18.0, 10.0]),
        filt_tau=0.02,
        k_beta=1.5,
    )


# WLS per-axis objective priority [roll, pitch, yaw, thrust] (paper Sec. 2.7:
# pitch highest -- most likely to saturate and matters most for the
# vehicle's natural pitch-down tendency).
DEFAULT_WV = np.array([100.0, 1000.0, 0.1, 10.0])


# ── controller ───────────────────────────────────────────────────────────────

class TailsitterINDIController:
    """Stateful INDI attitude+rate controller (mirrors TailsitterHoverController).

    Holds the between-step state INDI needs: the previous angular velocity
    (for the finite-difference Omega_dot), the filtered Omega_dot and filtered
    command, and the last applied command (the increment baseline u_f).
    """

    def __init__(
        self,
        params: TailsitterParams,
        inertia: np.ndarray,
        gains: TailsitterINDIGains | None = None,
        dt: float = 0.001,
        use_wls: bool = True,
        use_indi_outer: bool = False,
        use_guidance_indi: bool = False,
        guidance_effectiveness: str = "jacobian",
        guidance_gain: float = 0.5,
        pitch_scaling: float = 1.0,
        use_sideslip_control: bool = False,
        mass: float | None = None,
        Wv: np.ndarray | None = None,
        min_throttle: float = MIN_THROTTLE,
    ):
        self.params = params
        # Actuator bounds, with the idle-throttle floor applied (see
        # MIN_THROTTLE: a fully-off propeller zeroes its own column of G).
        self.u_min = U_MIN.copy()
        self.u_min[:2] = float(min_throttle)
        self.u_max = U_MAX.copy()
        self.dt = float(dt)
        self.use_wls = use_wls
        self.use_indi_outer = use_indi_outer
        # Guidance INDI outer loop (paper Sec. 4): commands [theta, lam, T] via
        # an effectiveness matrix that INCLUDES the wing-lift derivative, so it
        # unloads weight onto the wing in cruise (unlike the model-based and
        # measured-accel outer loops, which have no lift term). Backend:
        # "jacobian" (default, exact G from phi_theory) or "analytic" (paper
        # Eq. 28-33 fitted form, for sim2real). Takes precedence over
        # use_indi_outer when set.
        self.use_guidance_indi = use_guidance_indi
        if guidance_effectiveness not in ("jacobian", "analytic"):
            raise ValueError("guidance_effectiveness must be 'jacobian' or 'analytic'")
        self.guidance_effectiveness = guidance_effectiveness
        self.guidance_gain = float(guidance_gain)
        # Sec. 4.1 non-minimum-phase over-statement of pitch effectiveness (the
        # analytic backend's "scaling" factor; the paper uses 2.0). Only used by
        # the analytic backend.
        self.pitch_scaling = float(pitch_scaling)
        # Phase B: regulate sideslip by integrating the coordinated-turn +
        # sideslip-feedback heading-rate law (Eq. 17) into the heading reference
        # fed to desired_attitude_transition. When off, desired_yaw is used
        # directly (hover/manual heading) exactly as before.
        self.use_sideslip_control = use_sideslip_control
        # Actual dynamic mass (pass sim.total_mass so it matches MuJoCo, incl.
        # the prop bodies); only scales the outer-loop increment, which INDI
        # tolerates, but no reason to be off by the ~5% the props add.
        self.mass = float(params.mass if mass is None else mass)
        self.Wv = DEFAULT_WV if Wv is None else np.asarray(Wv, dtype=float)
        self.gains = gains if gains is not None else default_tailsitter_indi_gains()

        self.inertia = np.asarray(inertia, dtype=float)
        inertia_inv = jnp.asarray(np.linalg.inv(self.inertia))
        self._eff_jit = jax.jit(
            partial(indi_effectiveness, params=params, inertia_inv=inertia_inv))

        # Outer-loop (guidance) effectiveness G = d(accel)/d[phi, theta, T],
        # jax.jacobian of the real specific force, jitted once (jacobian backend).
        def _guid_accel(phi, theta, T, psi, velocity, wind):
            return guidance_accel(phi, theta, T, psi, velocity, params,
                                  self.mass, wind)
        self._guid_G_jit = jax.jit(
            jax.jacobian(_guid_accel, argnums=(0, 1, 2)))
        self._guid_accel_jit = jax.jit(_guid_accel)

        # Hover-trim command as the initial baseline so the first increment is
        # small (same trim throttle compute_hover_mixer uses).
        hover_throttle = float(jnp.clip(
            jnp.sqrt((params.mass * params.gravity)
                     / (2.0 * params.propulsion.kf)) / params.propulsion.max_omega,
            0.05, 0.95))
        self._u_trim = np.array([hover_throttle, hover_throttle, 0.0, 0.0])
        self.reset()

    def reset(self) -> None:
        self.u_applied = self._u_trim.copy()   # last command sent
        self.u_f = self._u_trim.copy()          # filtered command (baseline)
        self.omega_prev: np.ndarray | None = None
        self.omega_dot_f = np.zeros(3)          # filtered measured angular accel
        # Outer-loop INDI state: filtered measured acceleration (the increment
        # baseline; the thrust-vector baseline is the modeled thrust each step).
        self.vel_prev: np.ndarray | None = None
        self.a_meas_f = np.zeros(3)
        # Phase B: integrated heading reference (lazily seeded from the first
        # desired_yaw) and last sideslip estimate (exposed for logging).
        self.psi_ref: float | None = None
        self.beta = 0.0
        # Guidance INDI commanded outer state (exposed for logging/plots).
        self.outer_theta = 0.0
        self.outer_phi = 0.0
        self.outer_T = self.mass * self.params.gravity

    def update(
        self,
        state: VehicleState,
        setpoint_position: jnp.ndarray,
        setpoint_velocity: np.ndarray | None = None,
        accel_feedforward: np.ndarray | None = None,
        desired_yaw: float = 0.0,
        yaw_rate_ff: float = 0.0,
        wind_velocity: np.ndarray | None = None,
    ) -> jnp.ndarray:
        """One control step -> (4,) motor_commands.

        setpoint_velocity / accel_feedforward default to zero (pure
        position-hold, i.e. hover). Feed a moving velocity/accel reference to
        command forward flight or a transition -- the SAME controller, no mode
        switch (that is the whole point of INDI here).

        desired_yaw is the heading reference for desired_attitude_transition.
        With use_sideslip_control=True (Phase B) it is only the INITIAL heading:
        the controller then integrates the sideslip-feedback law (Eq. 17) each
        step -- psi_dot_ref = yaw_rate_ff + k_beta*beta -- and drives the heading
        itself to hold beta ~ 0 (needs a nonzero wind_velocity or forward flight
        for sideslip to exist). yaw_rate_ff is the coordinated-turn feedforward
        the guidance is commanding (rad/s; use coordinated_turn_rate() to get it
        from a desired bank); it is ignored when use_sideslip_control=False.
        """
        g = self.gains
        alpha = self.dt / (g.filt_tau + self.dt)   # first-order LPF coefficient

        Omega = np.asarray(state.angular_velocity, dtype=float)
        pos = np.asarray(state.position, dtype=float)
        vel = np.asarray(state.velocity, dtype=float)

        # ── measured Omega_dot: finite difference + matched low-pass ────────
        if self.omega_prev is None:
            omega_dot_raw = np.zeros(3)
        else:
            omega_dot_raw = (Omega - self.omega_prev) / self.dt
        self.omega_prev = Omega
        self.omega_dot_f += alpha * (omega_dot_raw - self.omega_dot_f)
        # Same filter on the command that feeds G (synchronisation, Sec. 2).
        self.u_f += alpha * (self.u_applied - self.u_f)

        # ── measured linear acceleration (INDI outer-loop baseline) ─────────
        if self.vel_prev is None:
            a_raw = np.zeros(3)
        else:
            a_raw = (vel - self.vel_prev) / self.dt
        self.vel_prev = vel
        self.a_meas_f += alpha * (a_raw - self.a_meas_f)

        # ── outer loop: desired acceleration -> desired thrust vector ───────
        vel_ref = (np.zeros(3) if setpoint_velocity is None
                   else np.asarray(setpoint_velocity, dtype=float))
        acc_ff = (np.zeros(3) if accel_feedforward is None
                  else np.asarray(accel_feedforward, dtype=float))
        pos_err = np.asarray(setpoint_position, dtype=float) - pos
        a_ref = (np.asarray(g.kp_pos) * pos_err
                 + np.asarray(g.kd_pos) * (vel_ref - vel) + acc_ff)

        R = quat_to_rotation_matrix(state.quaternion)

        # ── control effectiveness at the CURRENT state (needed by both loops) ─
        wind_jax = (jnp.zeros(3) if wind_velocity is None
                    else jnp.asarray(wind_velocity, dtype=float))
        G, y0 = self._eff_jit(state, jnp.asarray(self.u_f), wind_velocity=wind_jax)
        G = np.asarray(G, dtype=float)
        T_model = float(y0[3])   # modeled current body-x force (~actual thrust)

        # ── heading reference (needed BEFORE the outer loop: the guidance loop
        # resolves attitude in this heading frame). Sideslip control (Phase B,
        # Sec. 3) integrates the coordinated-turn + beta-feedback law into it;
        # otherwise desired_yaw is the heading directly. ─────────────────────
        heading = desired_yaw
        if self.use_sideslip_control:
            if self.psi_ref is None:
                self.psi_ref = float(desired_yaw)   # seed from the initial heading
            wind_np = (np.zeros(3) if wind_velocity is None
                       else np.asarray(wind_velocity, dtype=float))
            self.beta, V_air, v_body = estimate_sideslip(state, wind_np)
            # gate the beta feedback on forward (body-x) airspeed: below it
            # sideslip is undefined. The commanded turn feedforward still applies
            # (it comes from guidance, not from measured flow).
            beta_fb = g.k_beta * self.beta if v_body[0] > V_SIDESLIP_MIN else 0.0
            self.psi_ref += (yaw_rate_ff + beta_fb) * self.dt
            heading = self.psi_ref

        # ── outer loop: desired acceleration -> (thrust magnitude, attitude) ────
        if self.use_guidance_indi:
            # Guidance INDI (paper Sec. 4 / Paparazzi guidance_indi_hybrid): invert
            # the outer effectiveness G = d(accel)/d[phi, theta, T] against the
            # measured-acceleration error. G's theta-column carries the lift
            # derivative (unloads weight onto the wing), and its phi-column carries
            # the lift-ROLL (banking the wing into a turn), so BOTH pitch-to-lift
            # and the coordinated-turn bank fall out of the same inversion -- no
            # feedforward. Baseline v_f = [phi0, theta0, T0] is read off the CURRENT
            # attitude + thrust (cmd = current + increment, as in dronesim/Paparazzi).
            phi0, theta0 = guidance_extract_bank_pitch(jnp.asarray(R), heading)
            T0 = T_model
            vel_jax = jnp.asarray(vel)
            if self.guidance_effectiveness == "jacobian":
                d = self._guid_G_jit(phi0, theta0, T0, heading, vel_jax, wind_jax)
                Gp = np.stack([np.asarray(d[0]), np.asarray(d[1]),
                               np.asarray(d[2])], axis=1)
            else:
                V_air = float(np.linalg.norm(
                    vel - (np.zeros(3) if wind_velocity is None
                           else np.asarray(wind_velocity, dtype=float))))
                Gp = np.asarray(guidance_effectiveness_analytic(
                    phi0, theta0, T0, heading, V_air, self.mass,
                    self.params.gravity, paper_lift_slope(V_air, self.mass),
                    self.pitch_scaling))
            dv = self.guidance_gain * (np.linalg.pinv(Gp) @ (a_ref - self.a_meas_f))
            phi_c = float(np.clip(float(phi0) + dv[0], np.radians(-60.0), np.radians(60.0)))
            # theta must be free to go NEGATIVE (thrust axis tilted BACKWARDS,
            # against the heading). It was clipped to [0, 100deg], which confines
            # the commanded thrust axis to the half-space b1.h >= 0: the vehicle
            # could then only ever accelerate along +heading. That is invisible on
            # the nominal mission (guidance rotates the heading onto the path, so
            # every commanded accel is forward), but any disturbance pushing it
            # downrange -- notably a headwind at hover, where the heading is pinned
            # -- demands a backward tilt, gets clipped to theta=0, and the
            # unrepresentable demand leaks through the pinv into the thrust
            # channel instead: the vehicle drifts downwind AND climbs. Measured on
            # a 3 m/s hover headwind: 12.9 m position error and a 3.8 m unwanted
            # climb, vs 0.00 m once theta may go negative.
            # Bounds are +/-90deg to match guidance_extract_bank_pitch, whose
            # arcsin(b1.h) baseline cannot represent |theta| > 90deg anyway (the
            # old 100deg upper bound was already unreachable in the round trip).
            theta_c = float(np.clip(float(theta0) + dv[1], np.radians(-90.0), np.radians(90.0)))
            T_des = float(max(T0 + dv[2], 0.0))
            R_des = guidance_attitude(phi_c, theta_c, heading)
            self.outer_theta, self.outer_phi, self.outer_T = theta_c, phi_c, T_des
        else:
            if self.use_indi_outer:
                # Measured-acceleration INDI thrust-VECTOR loop (experimental,
                # kept for comparison). Thrust vector's effect on specific force
                # is (1/m)*I; gravity + aero are already in a_meas, so the
                # increment is m*(a_ref - a_meas) on the current thrust vector.
                # Baseline is the HOVER thrust m*g (the "T = 9.81" Cyclone/
                # dronesim/Paparazzi assumption): using the modeled T here made
                # it diverge. It has NO lift term, so it holds loosely in cruise
                # (steady-state tilt) -- the reason the guidance loop above
                # exists. Default off.
                b1 = np.asarray(R[:, 0])
                T_vec_0 = self.mass * self.params.gravity * b1
                T_vec_ref = T_vec_0 + self.mass * (a_ref - self.a_meas_f)
            else:
                # Model-based fallback (the original hover construction; no aero).
                g_vec = np.array([0.0, 0.0, -self.params.gravity])
                T_vec_ref = self.mass * (a_ref - g_vec)
            T_des = float(np.linalg.norm(T_vec_ref))
            b1_des = T_vec_ref / max(T_des, 1e-6)
            R_des = desired_attitude_transition(jnp.asarray(b1_des), heading)

        e_R = np.asarray(attitude_error(R, R_des))

        # ── attitude loop (Eq. 5-6) -> rate ref -> rate loop (Eq. 4) ────────
        Omega_ref = -np.asarray(g.k_att) * e_R
        nu_ang = np.asarray(g.k_rate) * (Omega_ref - Omega)   # desired Omega_dot

        # ── incremental objective (Eq. 2-3) ─────────────────────────────────
        nu = np.array([nu_ang[0], nu_ang[1], nu_ang[2], T_des])
        y_meas = np.array([self.omega_dot_f[0], self.omega_dot_f[1],
                           self.omega_dot_f[2], T_model])
        nu_inc = nu - y_meas

        # ── allocation: solve for the increment du, bounds about u_f ────────
        du_min = self.u_min - self.u_f
        du_max = self.u_max - self.u_f
        if self.use_wls:
            du, _ = wls_alloc(nu_inc, G, du_min, du_max, Wv=self.Wv,
                              u_guess=np.zeros(4))
        else:
            du = np.clip(np.linalg.pinv(G) @ nu_inc, du_min, du_max)

        u_c = np.clip(self.u_f + du, self.u_min, self.u_max)
        self.u_applied = u_c
        return jnp.asarray(u_c)
