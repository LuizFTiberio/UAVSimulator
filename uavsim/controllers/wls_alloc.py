"""Weighted least-squares control allocation via Härkegård's active-set method.

Faithful port of the active-set solver used in the Cyclone INDI flight code:

  * Paparazzi   : sw/airborne/math/wls/wls_alloc.c   (the canonical C version,
                  Smeur's original -- clearest logic, followed here for the
                  step-length / Lagrange-multiplier branches)
  * enac-drones : dronesim/control/wls_alloc.py       (the quad reference the
                  TODO roadmap points at -- same algorithm, numpy)

Solves the constrained weighted least-squares problem

    min_u   || Wu (u - ud) ||^2  +  gamma^2 || Wv (B u - v) ||^2
    s.t.    umin <= u <= umax

by the active-set method: variables on a bound form the "working set"; the
remaining "free" variables are solved by unconstrained least squares each
iteration; the working set is grown (a free var hits a bound) or shrunk (a
bound var's Lagrange multiplier has the wrong sign) until optimal.

gamma^2 (large) prioritises hitting the control objective `v` over staying
near the preferred command `ud`; Wv weights the objective axes against each
other -- this is where per-axis priority enters (pitch >> roll >> thrust >>
yaw for the tail-sitter, per the INDI paper's Sec. 2.7).

Not jitted (numpy, not jax.numpy): the active set has data-dependent control
flow -- a while loop of runtime-dependent length, a variable-length working
set grown/shrunk by list append/remove, and a break on a runtime KKT test.
None of that survives jax.jit tracing as-is (it needs lax.while_loop over a
fixed-size boolean mask instead of the free-list). The problem is tiny (n_u
actuators, n_v objectives -- 4x4 for the tail-sitter) and runs in the Python
control loop alongside the jitted G computation, so a plain numpy solve is
more than fast enough and keeps the algorithm a line-for-line match to the
reference (which matters for trusting it).

CAVEAT (relevant to the repo's RL / multi-vehicle goal): because this is
numpy, the *whole* controller cannot be jit/vmap/grad-ed end-to-end (parallel
envs, differentiable-through-controller). When that's needed, swap in a
jittable allocator -- jaxopt's BoxOSQP/BoxCDQP, or a fixed-iteration masked
active-set in lax -- and keep THIS one as the reference oracle to test it
against. For single-vehicle closed-loop validation (Phase A), numpy is fine.

Used incrementally by the INDI controller: solve for the command *increment*
du with bounds [umin - u_f, umax - u_f] and objective (nu - y_measured),
preferred increment ud = 0 -- see uavsim/controllers/tailsitter_indi.py.
"""

from __future__ import annotations

import numpy as np

FLT_EPSILON = 1.19209290e-7


def wls_alloc(
    v: np.ndarray,
    B: np.ndarray,
    umin: np.ndarray,
    umax: np.ndarray,
    Wv: np.ndarray | None = None,
    Wu: np.ndarray | None = None,
    ud: np.ndarray | None = None,
    u_guess: np.ndarray | None = None,
    W_init: np.ndarray | None = None,
    gamma_sq: float = 1.0e5,
    imax: int = 100,
) -> tuple[np.ndarray, int]:
    """Solve the constrained WLS allocation problem (see module docstring).

    Parameters
    ----------
    v : (n_v,) control objective (desired virtual control, e.g. angular
        accelerations + thrust).
    B : (n_v, n_u) control effectiveness matrix, d[v]/d[u].
    umin, umax : (n_u,) actuator bounds.
    Wv : (n_v,) objective axis weights (default all ones). Larger = higher
        priority when actuators saturate.
    Wu : (n_u,) control-effort weights (default all ones).
    ud : (n_u,) preferred command (default zeros).
    u_guess : (n_u,) warm-start command (default midpoint of the bounds).
    W_init : (n_u,) initial working set (default all-free). Entry i: 0 free,
        +1 at upper bound, -1 at lower bound.
    gamma_sq : objective-vs-preference weight (large favours the objective).
    imax : max active-set iterations.

    Returns
    -------
    u : (n_u,) allocated command.
    iters : iterations taken.
    """
    v = np.asarray(v, dtype=float)
    B = np.asarray(B, dtype=float)
    umin = np.asarray(umin, dtype=float)
    umax = np.asarray(umax, dtype=float)

    n_u = umin.shape[0]
    n_v = v.shape[0]

    Wv = np.ones(n_v) if Wv is None else np.asarray(Wv, dtype=float)
    Wu = np.ones(n_u) if Wu is None else np.asarray(Wu, dtype=float)
    ud = np.zeros(n_u) if ud is None else np.asarray(ud, dtype=float)
    gamma = np.sqrt(gamma_sq)

    if u_guess is None:
        u = 0.5 * (umin + umax)
    else:
        u = np.clip(np.asarray(u_guess, dtype=float), umin, umax)

    W = np.zeros(n_u) if W_init is None else np.asarray(W_init, dtype=float).copy()

    # Stacked weighted system  A u ~ b:
    #   top n_v rows  : gamma * Wv * B          (the objective)
    #   bottom n_u    : diag(Wu)                (the preference toward ud)
    A = np.zeros((n_v + n_u, n_u))
    b = np.zeros(n_v + n_u)
    A[:n_v, :] = gamma * (Wv[:, None] * B)
    b[:n_v] = gamma * Wv * v
    A[n_v:, :] = np.diag(Wu)
    b[n_v:] = Wu * ud

    d = b - A @ u  # residual at the current u

    free = [i for i in range(n_u) if W[i] == 0.0]

    iters = 0
    while iters < imax:
        iters += 1
        p = np.zeros(n_u)
        p_free = np.zeros(0)

        if free:
            A_free = A[:, free]
            p_free = np.linalg.lstsq(A_free, d, rcond=None)[0]
            p[free] = p_free

        u_opt = u + p
        infeasible = [
            i for i in free
            if u_opt[i] > umax[i] + FLT_EPSILON or u_opt[i] < umin[i] - FLT_EPSILON
        ]

        if not infeasible:
            # Feasible step: accept it, then test optimality via multipliers.
            u = u_opt
            if free:
                d = d - A_free @ p_free
            lam = (A.T @ d) * W  # nonzero only for working-set (bound) vars

            neg = [i for i in range(n_u) if lam[i] < -FLT_EPSILON]
            if not neg:
                return u, iters  # KKT satisfied -> optimal
            # Release the bound variables whose multiplier has the wrong sign.
            for i in neg:
                W[i] = 0.0
                if i not in free:
                    free.append(i)
        else:
            # Infeasible full step: walk to the nearest binding bound.
            alpha = 1.0
            id_alpha = free[0]
            for i in free:
                if abs(p[i]) > FLT_EPSILON:
                    if p[i] < 0.0:
                        a = (umin[i] - u[i]) / p[i]
                    else:
                        a = (umax[i] - u[i]) / p[i]
                    if 0.0 <= a < alpha:
                        alpha = a
                        id_alpha = i
            u = np.clip(u + alpha * p, umin, umax)
            d = d - alpha * (A_free @ p_free)
            W[id_alpha] = 1.0 if p[id_alpha] > 0.0 else -1.0
            free.remove(id_alpha)

    return u, iters
