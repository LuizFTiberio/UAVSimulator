"""Model Predictive Control (MPC) trajectory-tracking controller.

Two-level architecture::

    ┌─────────────────────┐            ┌───────────────────────┐
    │  MPC (outer loop)   │─ setpoint →│  Hover PID (inner)    │─ throttle →
    │  double-integrator  │            │  attitude + mixing    │
    └─────────────────────┘            └───────────────────────┘

The *outer* MPC plans a smooth trajectory in position / velocity space
using a double-integrator prediction model.  The *inner* HoverController
tracks each immediate setpoint, handling attitude control and motor mixing.

This avoids the mixer-saturation problem that arises when an MPC directly
controls thrust + torques: the hover PID automatically balances thrust
vs. torque demands within the motor limits.

All optimisation (rollout, cost, gradient, solver loop) is JIT-compiled
via ``jax.lax.scan``.  The reference-trajectory builder runs in NumPy.
"""

from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from uavsim.core.types import HoverGains, MultirotorParams, VehicleState
from uavsim.controllers.hover import HoverController, default_hover_gains


# ═════════════════════════════════════════════════════════════════════════════
#  Types
# ═════════════════════════════════════════════════════════════════════════════

class MPCConfig(NamedTuple):
    """Tuning knobs (baked into the JIT graph via *partial*)."""

    # ── solver ──
    horizon: int                    # prediction steps
    dt_mpc: float                   # prediction time-step [s]
    n_iters: int                    # gradient-descent iterations per call
    lr: float                       # optimiser step size

    # ── cost weights ──
    w_pos: jnp.ndarray              # (3,) per-axis position weight
    w_vel: float                    # terminal velocity-damping weight
    w_accel: float                  # control-effort weight
    w_jerk: float                   # control-smoothness weight (Δu)
    w_terminal: float               # terminal-cost multiplier
    w_progress: float               # reward for velocity toward reference
    w_speed: float                  # penalty for deviating from desired_speed
    desired_speed: float            # target cruise speed for w_speed [m/s]

    # ── dynamics bounds ──
    max_accel: float                # maximum acceleration [m/s²]
    max_vel: float                  # maximum velocity [m/s]

    # ── reference generation ──
    cruise_speed: float             # lookahead speed [m/s]
    setpoint_steps: int             # how far into the optimal trajectory to
                                    # sample the setpoint for the inner PID

    # ── control rate ──
    dt_ctrl: float                  # MPC re-solve interval [s]


class MPCState(NamedTuple):
    """Warm-start state carried across MPC calls."""
    u_plan: jnp.ndarray             # (horizon, 3) acceleration plan [ax, ay, az]
    step_count: jnp.int32


# ═════════════════════════════════════════════════════════════════════════════
#  Defaults
# ═════════════════════════════════════════════════════════════════════════════

def default_mpc_config(params: MultirotorParams) -> MPCConfig:
    """Sensible defaults for the 1.2 kg X-config quadcopter."""
    max_tilt_rad = float(jnp.deg2rad(35.0))
    max_accel = params.gravity * jnp.tan(max_tilt_rad)   # ≈ 6.9 m/s²

    return MPCConfig(
        horizon=20,
        dt_mpc=0.05,                # → 1 s lookahead
        n_iters=8,
        lr=0.12,
        w_pos=jnp.array([10.0, 10.0, 15.0]),
        w_vel=2.0,
        w_accel=0.5,
        w_jerk=1.0,
        w_terminal=4.0,
        w_progress=0.0,             # off by default (tracking mode)
        w_speed=0.0,                # off by default
        desired_speed=0.0,
        max_accel=float(max_accel),
        max_vel=5.0,
        cruise_speed=3.5,
        setpoint_steps=8,           # 8 × 0.05 s = 0.4 s → ~1.4 m ahead
        dt_ctrl=0.02,               # 50 Hz re-solve rate
    )


def racing_mpc_config(params: MultirotorParams) -> MPCConfig:
    """Aggressive config for gate racing — maximise speed, cut corners."""
    max_tilt_rad = float(jnp.deg2rad(50.0))           # steeper tilt
    max_accel = params.gravity * jnp.tan(max_tilt_rad) # ≈ 11.7 m/s²

    return MPCConfig(
        horizon=20,
        dt_mpc=0.05,
        n_iters=8,
        lr=0.15,
        w_pos=jnp.array([6.0, 6.0, 12.0]),           # lighter tracking
        w_vel=0.5,                                     # don't brake hard at horizon
        w_accel=0.1,                                   # allow aggressive manoeuvres
        w_jerk=0.2,                                    # allow snappy transitions
        w_terminal=3.0,
        w_progress=4.0,                                # reward speed toward ref
        w_speed=1.5,                                   # track desired cruise speed
        desired_speed=4.0,                             # 4 m/s cruise target
        max_accel=float(max_accel),
        max_vel=8.0,                                   # higher speed cap
        cruise_speed=5.0,                              # faster reference generation
        setpoint_steps=10,                             # further lookahead
        dt_ctrl=0.02,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Prediction dynamics — double integrator with bounds
# ═════════════════════════════════════════════════════════════════════════════

def _di_step(x, u, dt, max_vel):
    """Advance a 6-D double-integrator state by one step.

    x = [px, py, pz, vx, vy, vz],  u = [ax, ay, az].
    """
    vel = x[3:6] + u * dt
    vel = jnp.clip(vel, -max_vel, max_vel)
    pos = x[:3] + vel * dt
    return jnp.concatenate([pos, vel])


# ═════════════════════════════════════════════════════════════════════════════
#  Cost & optimisation (fully JIT-friendly)
# ═════════════════════════════════════════════════════════════════════════════

def _clamp_accel(u, max_accel):
    return jnp.clip(u, -max_accel, max_accel)


def _clamp_plan(u_plan, max_accel):
    return jnp.clip(u_plan, -max_accel, max_accel)


def _total_cost(u_plan, x0, ref_pos, config):
    """Total shooting cost over the prediction horizon."""

    def stage(carry, inputs):
        x, u_prev = carry
        u_raw, rp = inputs
        u = _clamp_accel(u_raw, config.max_accel)

        pos_err = x[:3] - rp
        du = u - u_prev
        vel = x[3:6]

        # ── tracking + regularisation ──
        c = (jnp.sum(config.w_pos * pos_err ** 2)
             + config.w_accel * jnp.sum(u ** 2)
             + config.w_jerk * jnp.sum(du ** 2))

        # ── progress reward: velocity aligned with direction to reference ──
        dir_to_ref = rp - x[:3]
        dist = jnp.linalg.norm(dir_to_ref) + 1e-6
        dir_hat = dir_to_ref / dist
        c = c - config.w_progress * jnp.dot(vel, dir_hat)

        # ── speed tracking: penalise deviation from desired cruise speed ──
        speed = jnp.linalg.norm(vel) + 1e-6
        c = c + config.w_speed * (speed - config.desired_speed) ** 2

        x_next = _di_step(x, u, config.dt_mpc, config.max_vel)
        return (x_next, u), c

    init_carry = (x0, jnp.zeros(3))
    (x_final, _), costs = jax.lax.scan(
        stage, init_carry, (u_plan, ref_pos))

    # ── terminal cost ──
    pos_err_T = x_final[:3] - ref_pos[-1]
    terminal = config.w_terminal * (
        jnp.sum(config.w_pos * pos_err_T ** 2)
        + config.w_vel * jnp.sum(x_final[3:6] ** 2)
    )
    return jnp.sum(costs) + terminal


def _solve(x0, mpc_state, ref_pos, config):
    """Projected gradient descent on the acceleration plan.

    Returns updated ``MPCState``, optimal first accel ``(3,)``,
    the optimal predicted setpoint ``(3,)``, and per-iter costs.
    """
    cost_and_grad = jax.value_and_grad(_total_cost)

    def step(u, _):
        cost_val, g = cost_and_grad(u, x0, ref_pos, config)
        # Global-norm clip
        g_norm = jnp.linalg.norm(g)
        g = jnp.where(g_norm > 50.0, g * 50.0 / (g_norm + 1e-8), g)
        u = u - config.lr * g
        u = _clamp_plan(u, config.max_accel)
        return u, cost_val

    u_opt, costs = jax.lax.scan(step, mpc_state.u_plan, None,
                                length=config.n_iters)

    # Setpoint: use the reference-path position at a fixed lookahead.
    # This gives the inner hover PID a target ~cruise_speed * lookahead_time
    # ahead on the path, which is a much stronger signal than the MPC's
    # predicted position (which starts near zero from rest).
    sp_idx = min(config.setpoint_steps, config.horizon) - 1
    setpoint = ref_pos[sp_idx]

    new_state = MPCState(u_plan=u_opt,
                         step_count=mpc_state.step_count + 1)
    return new_state, setpoint, costs


# ═════════════════════════════════════════════════════════════════════════════
#  Reference-trajectory builder (NumPy — outside JIT)
# ═════════════════════════════════════════════════════════════════════════════

def _build_reference(pos, waypoints, idx, horizon, dt, speed):
    """Interpolate a reference path through upcoming waypoints."""
    refs = np.empty((horizon, 3), dtype=np.float32)
    p = np.asarray(pos, dtype=np.float64)
    i = idx

    for k in range(horizon):
        if i >= len(waypoints):
            refs[k] = np.asarray(waypoints[-1][:3], dtype=np.float32)
            continue

        target = np.asarray(waypoints[i][:3], dtype=np.float64)
        d = np.linalg.norm(target - p)
        step_dist = speed * dt

        if d <= step_dist and i < len(waypoints) - 1:
            p = target.copy()
            i += 1
        elif d > 1e-6:
            p = p + (target - p) / d * min(step_dist, d)
        else:
            p = target.copy()
        refs[k] = p

    return jnp.array(refs)


# ═════════════════════════════════════════════════════════════════════════════
#  High-level controller (drop-in replacement for TrajectoryController)
# ═════════════════════════════════════════════════════════════════════════════

class MPCController:
    """Two-level MPC trajectory-tracking controller.

    Outer loop: MPC plans a smooth position trajectory (double integrator).
    Inner loop: ``HoverController`` tracks each immediate setpoint.

    Interface mirrors ``TrajectoryController``:

    * ``set_waypoints(waypoints)``
    * ``update(vehicle, dt) → (4,) throttle [0, 1]``
    * properties ``done``, ``current_waypoint_index``

    Parameters
    ----------
    params : MultirotorParams
    config : MPCConfig, optional
    hover_gains : HoverGains, optional
        Gains for the inner hover PID.  If *None*, uses faster defaults
        tuned for MPC tracking.
    acceptance_radius : float
        Distance to consider a waypoint reached [m].
    """

    def __init__(
        self,
        params: MultirotorParams,
        config: MPCConfig | None = None,
        hover_gains: HoverGains | None = None,
        acceptance_radius: float = 1.0,
    ):
        self.params = params
        self.config = config if config is not None else default_mpc_config(params)
        self.radius = acceptance_radius

        # Inner hover PID (attitude + mixing)
        if hover_gains is None:
            from uavsim.core.types import PIDGains
            hover_gains = HoverGains(
                kp_pos=5.0, ki_pos=0.15, kd_pos=3.5,
                pos_integral_limit=jnp.array([1.0, 1.0, 0.5]),
                att_gains=PIDGains(kp=8.0, ki=0.2, kd=3.0,
                                   max_output=1.5, integral_limit=0.3),
                max_tilt=jnp.float32(jnp.deg2rad(35.0)),
                min_alt=0.10,
                min_thrust_ratio=0.3,
            )
        self._hover = HoverController(params, gains=hover_gains)

        self._wpts: list[np.ndarray] = []
        self._idx: int = 0
        self._mpc_state = self._fresh_state()
        self._dt_accum = 0.0
        self._last_setpoint = jnp.zeros(3)
        self._last_yaw = 0.0
        self.last_info: dict = {}

        # JIT-compile solver with config baked in
        self._solve_jit = jax.jit(partial(_solve, config=self.config))

    # ── internal ──

    def _fresh_state(self) -> MPCState:
        return MPCState(
            u_plan=jnp.zeros((self.config.horizon, 3)),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    # ── waypoint interface ──

    def set_waypoints(self, waypoints) -> None:
        self._wpts = [np.asarray(w, dtype=np.float32) for w in waypoints]
        self._idx = 0
        self._mpc_state = self._fresh_state()
        self._hover.reset()
        self._dt_accum = 0.0
        if self._wpts:
            self._last_setpoint = jnp.array(self._wpts[0][:3])

    @property
    def current_waypoint_index(self) -> int:
        return self._idx

    @property
    def done(self) -> bool:
        if not self._wpts:
            return True
        return self._idx >= len(self._wpts) - 1

    def reset(self) -> None:
        self._mpc_state = self._fresh_state()
        self._hover.reset()
        self._dt_accum = 0.0

    # ── main entry point ──

    def update(self, vehicle: VehicleState, dt: float) -> jnp.ndarray:
        """Compute motor throttle.  Returns (4,) throttle [0, 1]."""
        if not self._wpts:
            return jnp.zeros(4)

        # ── advance waypoint ──
        pos = np.asarray(vehicle.position)
        wpt = self._wpts[self._idx]
        if np.linalg.norm(pos - wpt[:3]) < self.radius \
                and self._idx < len(self._wpts) - 1:
            self._idx += 1

        # ── re-solve MPC at dt_ctrl rate ──
        self._dt_accum += dt
        if self._dt_accum >= self.config.dt_ctrl:
            self._dt_accum = 0.0

            # 6-D state for the double-integrator
            x0 = jnp.concatenate([vehicle.position, vehicle.velocity])

            ref_pos = _build_reference(
                pos, self._wpts, self._idx,
                self.config.horizon, self.config.dt_mpc,
                self.config.cruise_speed,
            )

            self._mpc_state, setpoint, costs = self._solve_jit(
                x0, self._mpc_state, ref_pos)

            # Extract yaw from current waypoint
            cur_wpt = self._wpts[min(self._idx, len(self._wpts) - 1)]
            self._last_yaw = float(cur_wpt[3]) if len(cur_wpt) >= 4 else 0.0
            self._last_setpoint = setpoint

            self.last_info = {
                "cost": float(costs[-1]),
                "setpoint": np.asarray(setpoint),
            }

        # ── inner loop: hover PID tracks the MPC setpoint ──
        return self._hover.update(
            vehicle, self._last_setpoint, dt,
            desired_yaw=self._last_yaw)

    # ── debug ────────────────────────────────────────────────────────────

    def diagnose_step(self, vehicle: VehicleState) -> dict:
        """Run ONE MPC solve with Python for-loops and full diagnostics."""
        pos = np.asarray(vehicle.position)
        x0 = jnp.concatenate([vehicle.position, vehicle.velocity])

        ref_pos = _build_reference(
            pos, self._wpts, max(self._idx, 0),
            self.config.horizon, self.config.dt_mpc,
            self.config.cruise_speed,
        )

        cfg = self.config
        cost_fn = partial(_total_cost, config=cfg)
        cost_and_grad = jax.value_and_grad(cost_fn)
        u = self._mpc_state.u_plan.copy()

        print(f"  pos = {np.asarray(x0[:3])}")
        print(f"  vel = {np.asarray(x0[3:6])}")
        print(f"  ref[0]  = {np.asarray(ref_pos[0])}")
        print(f"  ref[-1] = {np.asarray(ref_pos[-1])}")
        print()

        for i in range(cfg.n_iters):
            cost_val, g = cost_and_grad(u, x0, ref_pos)
            grad_norm = float(jnp.linalg.norm(g))
            g = jnp.where(grad_norm > 50.0, g * 50.0 / (grad_norm + 1e-8), g)
            u = u - cfg.lr * g
            u = _clamp_plan(u, cfg.max_accel)

            u0 = np.asarray(u[0])
            print(f"  iter {i:2d}: cost={float(cost_val):8.1f}  "
                  f"|g|={grad_norm:8.1f}  "
                  f"a[0]=[{u0[0]:.2f}, {u0[1]:.2f}, {u0[2]:.2f}] m/s²")

        # Compute resulting setpoint
        u0 = _clamp_accel(u[0], cfg.max_accel)
        x1 = _di_step(x0, u0, cfg.dt_mpc, cfg.max_vel)
        print(f"\n  → setpoint = {np.asarray(x1[:3])}")
        return {"cost": float(cost_val), "u_plan": np.asarray(u),
                "ref_pos": np.asarray(ref_pos), "setpoint": np.asarray(x1[:3])}
