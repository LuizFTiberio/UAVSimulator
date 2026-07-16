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

Scope of this module (Phase A, steps 1-4): the inner attitude+rate INDI loop
and WLS allocation. The outer position/velocity loop is still the simple PD ->
desired-thrust-vector construction reused from tailsitter_hover.py (good
enough to command attitude for closed-loop hover/attitude validation); the
full INDI outer loop (roadmap step 6) is a follow-up.
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
from uavsim.controllers.tailsitter_hover import (
    attitude_error,
    desired_attitude_from_thrust_direction,
)

U_MIN = np.array([0.0, 0.0, -1.0, -1.0])   # [thr_L, thr_R, elevon_L, elevon_R]
U_MAX = np.array([1.0, 1.0, 1.0, 1.0])


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
        Wv: np.ndarray | None = None,
    ):
        self.params = params
        self.dt = float(dt)
        self.use_wls = use_wls
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

    def update(
        self,
        state: VehicleState,
        setpoint_position: jnp.ndarray,
        desired_yaw: float = 0.0,
        wind_velocity: np.ndarray | None = None,
    ) -> jnp.ndarray:
        g = self.gains
        alpha = self.dt / (g.filt_tau + self.dt)   # first-order LPF coefficient

        Omega = np.asarray(state.angular_velocity, dtype=float)

        # ── measured Omega_dot: finite difference + matched low-pass ────────
        if self.omega_prev is None:
            omega_dot_raw = np.zeros(3)
        else:
            omega_dot_raw = (Omega - self.omega_prev) / self.dt
        self.omega_prev = Omega
        self.omega_dot_f += alpha * (omega_dot_raw - self.omega_dot_f)
        # Same filter on the command that feeds G (synchronisation, Sec. 2).
        self.u_f += alpha * (self.u_applied - self.u_f)

        # ── outer loop (reused PD -> desired thrust vector & attitude) ──────
        pos = np.asarray(state.position, dtype=float)
        vel = np.asarray(state.velocity, dtype=float)
        pos_err = np.asarray(setpoint_position, dtype=float) - pos
        accel_cmd = np.asarray(g.kp_pos) * pos_err - np.asarray(g.kd_pos) * vel

        g_vec = np.array([0.0, 0.0, -self.params.gravity])
        F_thrust_world = self.params.mass * (accel_cmd - g_vec)
        T_des = float(np.linalg.norm(F_thrust_world))
        b1_des = F_thrust_world / max(T_des, 1e-6)

        R = quat_to_rotation_matrix(state.quaternion)
        R_des = desired_attitude_from_thrust_direction(
            jnp.asarray(b1_des), desired_yaw)
        e_R = np.asarray(attitude_error(R, R_des))

        # ── attitude loop (Eq. 5-6) -> rate ref -> rate loop (Eq. 4) ────────
        Omega_ref = -np.asarray(g.k_att) * e_R
        nu_ang = np.asarray(g.k_rate) * (Omega_ref - Omega)   # desired Omega_dot

        # ── control effectiveness at the CURRENT state ──────────────────────
        wind_jax = (jnp.zeros(3) if wind_velocity is None
                    else jnp.asarray(wind_velocity, dtype=float))
        G, y0 = self._eff_jit(state, jnp.asarray(self.u_f), wind_velocity=wind_jax)
        G = np.asarray(G, dtype=float)
        T_model = float(y0[3])

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
