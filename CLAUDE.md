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
examples/
  propulsion_comparison_demo.py  ← in-progress (see below)
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

## MuJoCo XML body masses (quadcopter.xml)

| body | mass |
|------|------|
| body (frame) | 1.0 kg |
| motor_fl/fr/bl/br | 0.05 kg each |
| **total** | **1.2 kg** |

For the 0.970 kg variant: set body mass to 0.770 kg, keep motors at 0.05 kg.
