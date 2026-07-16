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
from uavsim.controllers.wls_alloc import wls_alloc
# Reuse the singularity-free attitude machinery -- no reason to reimplement
# quaternion-error feedback (roadmap step 3); this rotation-matrix error has
# no Euler singularity at the tail-sitter's own +/-90deg hover attitude.
from uavsim.controllers.tailsitter_hover import attitude_error

U_MIN = np.array([0.0, 0.0, -1.0, -1.0])   # [thr_L, thr_R, elevon_L, elevon_R]
U_MAX = np.array([1.0, 1.0, 1.0, 1.0])


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


# ── gains ───────────────────────────────────────────────────────────────────

class TailsitterINDIGains(NamedTuple):
    kp_pos: jnp.ndarray   # (3,) outer position P gain
    kd_pos: jnp.ndarray   # (3,) outer velocity D gain
    k_att: jnp.ndarray    # (3,) attitude loop: e_R -> Omega_ref  (Eq. 5-6)
    k_rate: jnp.ndarray   # (3,) rate loop: rate error -> Omega_dot_ref (Eq. 4)
    filt_tau: float       # (s) low-pass time constant, gyro-diff AND command


def default_tailsitter_indi_gains() -> TailsitterINDIGains:
    return TailsitterINDIGains(
        kp_pos=jnp.array([2.0, 2.0, 4.0]),
        kd_pos=jnp.array([2.5, 2.5, 3.5]),
        k_att=jnp.array([8.0, 8.0, 4.0]),
        k_rate=jnp.array([18.0, 18.0, 10.0]),
        filt_tau=0.02,
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
        mass: float | None = None,
        Wv: np.ndarray | None = None,
    ):
        self.params = params
        self.dt = float(dt)
        self.use_wls = use_wls
        self.use_indi_outer = use_indi_outer
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

    def update(
        self,
        state: VehicleState,
        setpoint_position: jnp.ndarray,
        setpoint_velocity: np.ndarray | None = None,
        accel_feedforward: np.ndarray | None = None,
        desired_yaw: float = 0.0,
        wind_velocity: np.ndarray | None = None,
    ) -> jnp.ndarray:
        """One control step -> (4,) motor_commands.

        setpoint_velocity / accel_feedforward default to zero (pure
        position-hold, i.e. hover). Feed a moving velocity/accel reference to
        command forward flight or a transition -- the SAME controller, no mode
        switch (that is the whole point of INDI here).
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

        if self.use_indi_outer:
            # INDI translational loop (Sec. 4): the thrust vector's effect on
            # specific force is (1/m)*I, and gravity + aero cancel because they
            # are already in the measured acceleration -- so the increment is
            # just m*(a_ref - a_meas) on top of the current thrust vector.
            # The baseline thrust magnitude is taken as the HOVER thrust m*g
            # (the "T = 9.81" assumption in the Cyclone/dronesim/Paparazzi INDI
            # position loops), not the modeled/measured thrust: using the
            # modeled T here (biased + jittery from the filtered command) made
            # this loop diverge (|omega| ran away); the fixed m*g baseline is
            # stable. Growing wing lift through a transition still shows up in
            # a_meas, so it is rejected without an explicit aero model here.
            # NOTE (experimental): even stabilised, this measurement-based loop
            # holds less tightly than the model-based path below (some
            # steady-state tilt) -- the Sec. 4.1 non-minimum-phase / cascade-
            # bandwidth problem. Default is use_indi_outer=False.
            b1 = np.asarray(R[:, 0])
            T_vec_0 = self.mass * self.params.gravity * b1
            T_vec_ref = T_vec_0 + self.mass * (a_ref - self.a_meas_f)
        else:
            # Model-based fallback (the original hover construction).
            g_vec = np.array([0.0, 0.0, -self.params.gravity])
            T_vec_ref = self.mass * (a_ref - g_vec)

        T_des = float(np.linalg.norm(T_vec_ref))
        b1_des = T_vec_ref / max(T_des, 1e-6)

        R_des = desired_attitude_transition(jnp.asarray(b1_des), desired_yaw)
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
        du_min = U_MIN - self.u_f
        du_max = U_MAX - self.u_f
        if self.use_wls:
            du, _ = wls_alloc(nu_inc, G, du_min, du_max, Wv=self.Wv,
                              u_guess=np.zeros(4))
        else:
            du = np.clip(np.linalg.pinv(G) @ nu_inc, du_min, du_max)

        u_c = np.clip(self.u_f + du, U_MIN, U_MAX)
        self.u_applied = u_c
        return jnp.asarray(u_c)
