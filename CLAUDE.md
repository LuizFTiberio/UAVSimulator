# UAVSimulator — Claude context

JAX + MuJoCo quadrotor simulator. Pure JAX physics, MuJoCo for rigid-body integration.

## Repo layout

```
uavsim/
  core/types.py          — NamedTuple pytrees (VehicleState, MultirotorParams, …)
  core/math.py           — quat_to_rotation_matrix, etc.
  dynamics/propulsion.py — simple T=kt·ω², PropulsionModel enum, dispatch
  dynamics/propulsion_bem.py — BEM_LIVE and BEM_TABLE implementations (CCBlade JAX)
  vehicles/multirotor.py — multirotor_wrench, quadcopter_params, quadcopter()
  vehicles/base.py       — VehicleModel dataclass
  sim/mujoco_sim.py      — MuJoCoSimulator (JAX forces → xfrc_applied → mj_step)
  controllers/mpc.py     — MPCController
  viz/                   — SimulationVisualizer, plot_flight_data
  aero/phi_theory.py     — phi-theory tailsitter aero (Lustosa JGCD 2019)
  vehicles/tailsitter.py — tailsitter_wrench, tailsitter_params, cyclone_*
  controllers/tailsitter_indi.py     — INDI attitude+rate + Guidance INDI outer loop
  controllers/tailsitter_guidance.py — mission leg → reference generator
  controllers/wls_alloc.py           — active-set WLS allocation (numpy, not jitted)
  disturbances/wind.py   — ConstantWind, DrydenWind
examples/
  propulsion_comparison_demo.py  ← in-progress (see below)
  tailsitter_mission_demo.py     ← hover/transition/cruise+turn/hover, --wind
  targets_demo.py, trajectory_demo.py, …
```

## CCBlade JAX dependency

Companion repo at `/home/luiztiberio/Documents/CCBlade_jax`.
Installed as `pip install -e /path/to/CCBlade_jax` (package name `ccblade-jax`, import as `bem`).
Always `jax.config.update("jax_enable_x64", True)` before any import.

## Propulsion integration (done)

`MultirotorParams` has an optional `bem_config: Any = None` field (last, backward-compat).
`PropulsionModel` enum: `SIMPLE` / `BEM_LIVE` / `BEM_TABLE`.
`compute_rotor_wrench_dispatch(omega, v_air_body, rotor_yaw_sign, params, model)` routes to the right implementation.

To use BEM in `multirotor_wrench`, pass `propulsion_model=PropulsionModel.BEM_LIVE` (or TABLE).
Since `MuJoCoSimulator` binds the wrench function via `partial(vehicle.compute_wrench, params=…)`,
create a custom VehicleModel with `compute_wrench = partial(multirotor_wrench, propulsion_model=…)`.

## Propulsion comparison demo — COMPLETE

File: `examples/propulsion_comparison_demo.py`

Run: `python examples/propulsion_comparison_demo.py` → produces `propulsion_comparison.png`

**Results (2026-05-15):**

| Model     | sim [s] | wall [s] | realtime |
|-----------|---------|----------|----------|
| SIMPLE    | 2.4     | 1.9      | 1.3×     |
| BEM_LIVE  | 2.3     | 7.3      | 0.3×     |
| BEM_TABLE | 2.4     | 2.3      | 1.0×     |

All three reach both waypoints. BEM_LIVE ≈ BEM_TABLE trajectory (< 1% difference).

**Two bugs fixed in this session:**

1. *Mass mismatch*: blade generates 2.378 N/rotor at hover ω=419 rad/s; default 1.2 kg quad
   needed 2.943 N/rotor → drone sank. Fixed: `_bem_hover_mass()` helper computes
   `mass_eff = 4*T_hover/g = 0.9697 kg`; all three sims use this mass (params + MJCF override).

2. *Missing pitch/roll moments in BEM*: `compute_rotor_wrench_bem` / `_bem_table` were
   returning only `[0,0,ΣT]` force and `[0,0,yaw]` torque — no `cross(arm_i, T_i)` moments.
   Without differential-thrust moments, the drone could not tilt to fly horizontally.
   Fixed: both BEM functions now accept `motor_positions` and compute moments identically
   to the SIMPLE model.

**BEM_LIVE is ~0.3× realtime on CPU** (expected, JIT-compiled bisection solver).
BEM_TABLE is ~1.2× realtime (fast path for trajectory optimisation).

## Key conventions

- Body frame: z-up, rotors face upward; Vx = -v_air_body[2] for BEM axial inflow.
- CCBlade Q is always positive magnitude; multiply by rotor_yaw_sign for net yaw.
- `PropulsionModel` enum must be a static JIT argument: `jax.jit(f, static_argnames=["model"])`.
- MuJoCoSimulator binds `propulsion_model` via Python `partial` closure, not a traced arg — correct.
- Do NOT create `jaxopt.Bisection` inside a jitted function; the module-level `_BISECTION` handles it.

## Tailsitter + INDI control (branch `feat_indi_control`)

Cyclone-style dual-engine flying wing, phi-theory aero, one continuous INDI
controller across hover → transition → cruise (Smeur/Bronz/de Croon JGCD 2019,
arxiv 1802.00714). Roadmap phases A–D all complete.

**There is ONE G in the attitude loop**, recomputed every control step by
`jax.jacobian` of the real `tailsitter_wrench` at the current state
(`indi_effectiveness`). *Not* a hover G plus a cruise G interpolated through
transition — that is the deliberate advantage over the paper, which had to fit
and schedule G(θ,V) from flight data (Eq. 7-12) for lack of an aero model. The
only fitted+scheduled quantity is `paper_lift_slope()`, which belongs to the
**outer** loop's optional `"analytic"` sim2real backend. Two distinct G's exist:
inner 4×4 (actuators → angular accel + thrust) and outer 3×3 ([φ,θ,T] → linear
accel).

**Integrator.** MJCF `timestep` is the **control period**; `MuJoCoSimulator(
substeps=N)` subdivides it for physics, so `sim.dt` stays 1 kHz (the rate the
INDI loop is tuned at) and `sim.physics_dt = dt/substeps`. Forces are a
zero-order hold on `xfrc_applied`, so the ZOH — not the integrator — caps the
global order at ~1; that's why it's Euler, and why accuracy comes from
substeps (wrench recomputed each substep), not from a fancier integrator.
Do **not** "fix" this with `mjcb_control` true-RK4: it works (~9000× lower
error) but is a process-global Python callback, incompatible with mjx/batched
rollouts and unvmappable, which conflicts with near-term parallel-env RL.

**Placeholders, not Cyclone data** — don't tune against these as if they were
measurements: `zeta_f_pitch=0.3` (the *only* source of hover elevon authority;
`Cmde`/`Clda` need V>0), `max_elevon=25°`, `prop_y_offset=0.30` (photo
estimate), and the Cessna-class stability derivatives in
`default_phi_wing_params`.

## Open — check next session

1. **`hover-end` does not recover hover in wind.** With 5 m/s steady wind the
   mission climbs 10 → ~16 m and finishes tilted (b1·z ≈ 0.4–0.6) and still
   moving (~0.7–3.5 m/s) instead of b1·z → 1.0. Reproduces at every
   `zeta_f` tried (0.3/0.6/1.0), so it is **not** simply elevon authority.
   Cause not established. Next step: read the 6-panel
   `plot_tailsitter_mission()` output and identify which loop (guidance
   reference / outer / attitude) gives up first — don't guess.
2. **Turbulence has never actually been flown.** `DrydenWind` was inert until
   2026-07-21 (missing `1/√dt`, ~180× too weak), so every historical
   "turbulence" run was mean-wind-only. `--turbulence light|moderate|severe`
   is genuinely untested in closed loop and may be much harsher than steady
   wind.
3. **Actuator command spikes remain.** The WLS allocation still produces
   full-range elevon reversals in one 1 ms step (max |Δu| = 2.0, ~37 events per
   mission), concentrated in `hover-end`. Root cause unknown.
4. **Controller gets ground-truth wind** in the demo (`wind_velocity=
   sim.wind_velocity`) — a cheat, same shortcut `estimate_sideslip` takes. A
   real vehicle needs a wind estimator or beta from the lateral accelerometer
   (paper Eq. 13-16). Behaviour may differ when blind.

### Negative results — do not retry these
- **Actuator rate limiting** to suppress the command spikes: made flight
  materially *worse* (max |Δv| 0.039 → 1.088, altitude sagged to 0.47 m). It
  violates INDI's incremental assumption unless the limit is also reflected in
  the model/filter feeding G.
- **Throttle floor as a spike fix**: `min_throttle` (now 0.05) does remove a
  genuine exact rank-deficiency in G (cond > 1e6 steps: 241 → 1, since
  `dT/dthrottle` is exactly 0 at zero throttle), but it moved the spikes by only
  2 of 39 events. Keep it as robustness; it is not the spike fix. The
  conditioning benefit fully saturates by 0.01.
- **Raising `zeta_f`** to cure `hover-end`: improves the end state
  monotonically but never recovers hover; see item 1.

## MuJoCo XML body masses (quadcopter.xml)

| body | mass |
|------|------|
| body (frame) | 1.0 kg |
| motor_fl/fr/bl/br | 0.05 kg each |
| **total** | **1.2 kg** |

For the 0.970 kg variant: set body mass to 0.770 kg, keep motors at 0.05 kg.
