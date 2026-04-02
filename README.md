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
│   └── aerodynamics.py       # Body/wing aero (stub for future work)
├── controllers/
│   ├── pid.py                # Functional PID (init / step)
│   ├── mixer.py              # X-quad motor allocation matrix
│   ├── hover.py              # Cascaded position → attitude → motor controller
│   ├── trajectory.py         # Waypoint-following controller
│   ├── mpc.py                # Two-level MPC (outer planner + inner hover)
│   └── indi.py               # INDI controller (stub)
├── vehicles/
│   ├── base.py               # VehicleModel dataclass
│   └── multirotor.py         # Quadcopter params, model, and wrench function
├── sim/
│   └── mujoco_sim.py         # Vehicle-agnostic MuJoCo stepping
├── envs/
│   ├── base_env.py           # BaseUAVEnv (Gymnasium)
│   └── hover_env.py          # HoverEnv — registered as "uavsim/Hover-v0"
├── viz/
│   ├── viewer.py             # Real-time MuJoCo passive viewer
│   └── plotting.py           # Post-flight Matplotlib plots
└── models/
    └── quadcopter.xml        # MJCF quadcopter model
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/LuizFTiberio/UAVSimulator.git
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

Each example launches a real-time MuJoCo viewer and saves analysis plots on exit.

### 3D Viewer Controls

- **Right-click + drag** — Rotate viewpoint
- **Scroll wheel** — Zoom in/out
- **Left-click** — Select/inspect objects
- **Close window** — End simulation

## Features

- **JAX-accelerated dynamics** — JIT-compiled propulsion model; `jax.grad` through thrust computations
- **MuJoCo physics** — Rigid-body integration, contacts, and rendering
- **Multiple controllers** — PID, Hover (cascaded), Trajectory, MPC, INDI (stub)
- **Gymnasium environment** — `gymnasium.make("uavsim/Hover-v0")` for RL research
- **Modular vehicle model** — Params + MJCF + wrench function bundled in a single `VehicleModel`
- **Real-time 3D viewer** — MuJoCo passive rendering
- **Post-flight analysis** — Position, velocity, and orientation plots via Matplotlib

## Dependencies

| Package | Role |
|---------|------|
| `jax` / `jaxlib` ≥ 0.4.20 | Dynamics & controller computation |
| `mujoco` ≥ 3.0.0 | Physics simulation & rendering |
| `numpy` ≥ 1.24 | Array utilities |
| `matplotlib` ≥ 3.6 | Plotting (optional) |
| `gymnasium` ≥ 0.29 | RL environment (optional) |
| `pytest` ≥ 7.0 | Testing (optional) |

## Tests

```bash
pytest tests/
```

## License

MIT
