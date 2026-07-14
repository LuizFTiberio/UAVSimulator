# UAVSim — JAX-first UAV Simulator with MuJoCo Physics

A modular, JAX-accelerated simulator for multirotors and VTOL UAVs. JAX handles dynamics and controllers (JIT compilation, automatic differentiation, vectorization), while MuJoCo provides rigid-body integration, contact physics, and 3D rendering.

## Project Structure

```
uavsim/
├── core/
│   ├── types.py              # NamedTuple state containers (automatic JAX pytrees)
│   └── math.py               # Quaternion & rotation utilities
├── dynamics/
│   ├── propulsion.py         # Rotor thrust & torque (pure JAX, JIT-compatible)
│   ├── propulsion_bem.py     # Blade-element momentum rotor model (CCBlade JAX)
│   └── aerodynamics.py       # Wing lift/drag, flap & aileron aerodynamics
├── disturbances/
│   └── wind.py               # Dryden turbulence (MIL-HDBK-1797), constant wind
├── controllers/
│   ├── pid.py                # Functional PID (init / step)
│   ├── mixer.py              # X-quad motor allocation matrix
│   ├── hover.py              # Cascaded position → attitude → motor controller
│   ├── trajectory.py         # Waypoint-following controller
│   ├── quadplane_ctrl.py     # Quadplane state-machine controller (7-channel)
│   ├── mpc.py                # Two-level MPC (outer planner + inner hover)
│   └── indi.py               # INDI controller (stub)
├── vehicles/
│   ├── base.py               # VehicleModel dataclass
│   ├── multirotor.py         # Quadcopter params, model, and wrench function
│   └── quadplane.py          # Quadplane (quad + wing + pusher) vehicle
├── missions/
│   └── transport.py          # Slung-load transport mission logic
├── sim/
│   └── mujoco_sim.py         # Vehicle-agnostic MuJoCo stepping
├── envs/
│   ├── base_env.py           # BaseUAVEnv (Gymnasium)
│   └── hover_env.py          # HoverEnv — registered as "uavsim/Hover-v0"
├── viz/
│   ├── viewer.py             # Real-time MuJoCo passive viewer
│   └── plotting.py           # Post-flight Matplotlib plots
└── models/
    ├── quadcopter.xml        # MJCF quadcopter model
    └── quadplane.xml         # MJCF quadplane model (quad + wing + pusher)
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/<your-username>/UAVSimulator.git
cd UAVSimulator

# Install in editable mode (all optional deps)
pip install -e ".[all]"

# Run an example
python examples/hover_demo.py
```

## Examples

| Script | Description |
|--------|-------------|
| `examples/hover_demo.py` | Hover at 1 m, apply a velocity disturbance at t=2 s, watch recovery |
| `examples/trajectory_demo.py` | Track a square waypoint pattern with MPC |
| `examples/targets_demo.py` | Fly through a series of 3D square gates using MPC |
| `examples/quadplane_demo.py` | Quadplane VTOL: takeoff, transition, cruise 70 m, decel, land |
| `examples/transport_demo.py` | Slung-load transport: carry a payload from A to B with a quadcopter |
| `examples/wind_demo.py` | Wind disturbance: 3 quads hover under no-wind, steady crosswind, and Dryden turbulence |
| `examples/bem_propulsion_demo.py` | Hover a quadcopter using the blade-element momentum rotor model |
| `examples/propulsion_comparison_demo.py` | Fly the same waypoint mission under all three propulsion models and compare trajectories & runtime |

Each example launches a real-time MuJoCo viewer and saves analysis plots on exit.

### 3D Viewer Controls

- **Right-click + drag** — Rotate viewpoint
- **Scroll wheel** — Zoom in/out
- **Left-click** — Select/inspect objects
- **Close window** — End simulation

## Features

- **JAX-accelerated dynamics** — JIT-compiled propulsion and aerodynamic models; `jax.grad` through thrust computations
- **Selectable propulsion models** — Algebraic `kt·ω²`, live blade-element momentum (BEM), or a pre-tabulated BEM lookup (see below)
- **MuJoCo physics** — Rigid-body integration, contacts, and rendering
- **Quadplane VTOL** — Full transition flight: hover → transition → cruise → decel → land, with state-machine controller, wing lift compensation, and coordinated aileron/rotor control
- **Multiple controllers** — PID, Hover (cascaded), Trajectory, Quadplane (7-channel state machine), MPC, INDI (stub)
- **Slung-load transport** — Cable-suspended payload delivery with pendulum dynamics
- **Wind & turbulence** — Dryden (MIL-HDBK-1797) turbulence model with body drag; wind velocity flows into vehicle dynamics for correct airspeed computation
- **Gymnasium environment** — `gymnasium.make("uavsim/Hover-v0")` for RL research
- **Modular vehicle model** — Params + MJCF + wrench function bundled in a single `VehicleModel`
- **Real-time 3D viewer** — MuJoCo passive rendering
- **Post-flight analysis** — Position, velocity, and orientation plots via Matplotlib

## Propulsion Models

Rotor forces can come from one of three interchangeable models, selected with the
`PropulsionModel` enum:

| Model | How it works | Speed (quadcopter, CPU) |
|-------|--------------|-------------------------|
| `SIMPLE` | Algebraic `T = kt·ω²`, `Q = km·ω²`. Default; no extra dependencies. | ~1.3× realtime |
| `BEM_LIVE` | Solves blade-element momentum equations every step via CCBlade JAX, capturing inflow, blade geometry, and airfoil polars. | ~0.3× realtime |
| `BEM_TABLE` | Trilinear interpolation into a thrust/torque table pre-computed from the BEM solver. Matches `BEM_LIVE` to under 1% on a waypoint mission. | ~1.0× realtime |

The BEM models need [`ccblade-jax`](https://github.com/LuizFTiberio/ccblade-jax), which
is not on PyPI — install it from git:

```bash
pip install git+https://github.com/LuizFTiberio/ccblade-jax.git
```

Call `jax.config.update("jax_enable_x64", True)` before importing it. CCBlade JAX is a
JAX reimplementation of [CCBlade.jl](https://github.com/byuflowlab/ccblade.jl) (Andrew
Ning, BYU FLOW Lab), validated against the Julia package as ground truth; cite the
original method as Ning, S. A. (2014), *A simple solution method for the blade element
momentum equations with guaranteed convergence*, Wind Energy 17(9), 1327–1345.

Attach a `BEMConfig` to
`MultirotorParams.bem_config`, then bind the model into the wrench function:

```python
from functools import partial
from uavsim.dynamics.propulsion import PropulsionModel
from uavsim.vehicles.multirotor import multirotor_wrench

vehicle = dataclasses.replace(
    quadcopter(params),
    compute_wrench=partial(multirotor_wrench, propulsion_model=PropulsionModel.BEM_TABLE),
)
```

`examples/propulsion_comparison_demo.py` runs all three side by side and plots the result.

## Dependencies

| Package | Role |
|---------|------|
| `jax` / `jaxlib` ≥ 0.4.20 | Dynamics & controller computation |
| `mujoco` ≥ 3.0.0 | Physics simulation & rendering |
| `numpy` ≥ 1.24 | Array utilities |
| `matplotlib` ≥ 3.6 | Plotting (optional) |
| `gymnasium` ≥ 0.29 | RL environment (optional) |
| `ccblade-jax` | Blade-element momentum propulsion (optional, install from git) |
| `pytest` ≥ 7.0 | Testing (optional) |

## Tests

```bash
pytest tests/
```

## License

MIT
