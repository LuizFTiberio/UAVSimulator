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

Each example launches a real-time MuJoCo viewer and saves analysis plots on exit.

### 3D Viewer Controls

- **Right-click + drag** — Rotate viewpoint
- **Scroll wheel** — Zoom in/out
- **Left-click** — Select/inspect objects
- **Close window** — End simulation

## Features

- **JAX-accelerated dynamics** — JIT-compiled propulsion and aerodynamic models; `jax.grad` through thrust computations
- **MuJoCo physics** — Rigid-body integration, contacts, and rendering
- **Quadplane VTOL** — Full transition flight: hover → transition → cruise → decel → land, with state-machine controller, wing lift compensation, and coordinated aileron/rotor control
- **Multiple controllers** — PID, Hover (cascaded), Trajectory, Quadplane (7-channel state machine), MPC, INDI (stub)
- **Slung-load transport** — Cable-suspended payload delivery with pendulum dynamics
- **Wind & turbulence** — Dryden (MIL-HDBK-1797) turbulence model with body drag; wind velocity flows into vehicle dynamics for correct airspeed computation
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
