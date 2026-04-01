# UAV Simulator - Quadcopter Control & Simulation

A MuJoCo-based simulator for multirotors and VTOL UAVs. This project provides a physics-accurate quadcopter simulator with control algorithms ready for extension to more complex vehicles.

## Project Structure

```
UAVSimulator/
├── main.py                 # Main script: Stability test with visualization
├── simulator.py            # Core MuJoCo physics simulator
├── controllers.py          # Attitude, hover, and trajectory control algorithms
├── visualizer.py           # Real-time 3D visualization using MuJoCo viewer
├── requirements.txt        # Python dependencies
├── models/
│   └── quadcopter.xml      # MuJoCo URDF model of quadcopter
└── README.md               # This file
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the simulation:**
   ```bash
   python main.py
   ```

   This launches a real-time 3D visualization showing:
   - Quadcopter hovering at 1.0 m altitude
   - Velocity disturbance applied at t=2 seconds (+1.0 m/s in X direction)
   - Controller recovery and return to hover
   - Real-time 3D viewing with MuJoCo viewer
   - Post-flight analysis plots saved as `stability_test_results.png`

3. **3D Viewer Controls:**
   - **Right-click + drag:** Rotate viewpoint
   - **Scroll wheel:** Zoom in/out
   - **Left-click:** Select/inspect objects
   - **Close window:** End simulation
   - **Ctrl+C in terminal:** Emergency stop

## Features

### Current Implementation (Quadcopter)
- **Physics-Accurate Simulation**: MuJoCo-based rigid body dynamics
- **Thrust Model**: Force from motors: F = thrust_coefficient × ω²
- **Attitude Control**: PID stabilization for roll, pitch, yaw angles
- **Hover Control**: Position-hold with cascaded attitude control
- **Velocity Disturbance**: Demonstrates controller recovery from impulses
- **Real-time Visualization**: 3D viewer showing quadcopter dynamics
- **Data Logging**: Complete state history (position, velocity, orientation, angular velocity)
- **Post-flight Analysis**: Matplotlib plots for performance evaluation

### Quadcopter Specifications
- **Mass**: 1.2 kg
- **Motor Configuration**: X-configuration (4 motors at corners)
- **Max Motor Speed**: 1000 rad/s
- **Max Thrust per Motor**: 5.0 N (theoretical)
- **Thrust Coefficient (kt)**: 1e-5
- **Torque Coefficient (km)**: 1e-7

### Controller Gains (Tunable)
Edit `controllers.py` to adjust:
- **Attitude Control**: `Kp=5.0, Ki=0.5, Kd=2.0`
- **Position Control**: `Kp=1.0, Kd=2.0`
- **Max Tilt Angle**: 30 degrees

## How It Works

### Quadcopter Physics
The simulator models a quadcopter with:
1. **Thrust Generation**: Each motor produces upward force proportional to rotor speed squared
2. **Moment Control**: Motor positions relative to body create moments (torques) for attitude control
3. **Rotor Drag**: Counter-torque opposing yaw maneuvers
4. **Cascaded Control**: High-level position commands → attitude references → motor commands

### Disturbance Test
The example demonstrates:
1. **t=0-2s**: Controller stabilizes quadcopter in hover at 1.0 m altitude
2. **t=2s**: +1.0 m/s velocity impulse applied in X direction
3. **t=2-5s**: Controller counter-acts the disturbance, reducing lateral drift and returning to hover
4. **Output**: Graphs showing position, velocity, and trajectory recovery

## Example Output

```
======================================================================
QUADCOPTER STABILITY TEST - Velocity Disturbance Recovery
======================================================================

Scenario:
  - Quadcopter hovers at 1.0 m altitude
  - At t=2 seconds: Apply +1.0 m/s velocity impulse in X direction
  - Controller recovers and returns to hover
  - Total simulation time: 5 seconds

======================================================================

Launching MuJoCo 3D Viewer...
Running simulation for 5.0 seconds...

Time     X(m)       Y(m)       Z(m)       Vx(m/s)    Vy(m/s)    Vz(m/s)
----------------------------------------------------------------------
0.00     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000
...
2.00     0.0024     -0.0001    1.0000     0.0012     -0.0008    -0.0002
----------------------------------------------------------------------

*** DISTURBANCE APPLIED at t=2.00s ***
*** Velocity impulse: [1.0 0.0 0.0] m/s ***

----------------------------------------------------------------------
2.01     0.0034     -0.0001    0.9999     1.0011     -0.0009    -0.0004
...
5.00     0.1234     0.0056     0.9998     -0.0145    0.0023     0.0001

======================================================================
FINAL RESULTS
======================================================================
Final Position:      [0.1234 0.0056 0.9998]
Final Velocity:      [-0.0145 0.0023 0.0001]
Position Error:      0.1245 m
Simulation Duration: 5.00 s

Disturbance Analysis:
  Peak X-velocity: 1.0023 m/s
  Peak X-position: 0.4523 m
  ✓ Disturbance successfully applied!

Generating plots...
✓ Plot saved as 'stability_test_results.png'

======================================================================
Test complete!
======================================================================
```

## Code Usage

### Running Your Own Simulation

```python
from simulator import QuadcopterSimulator
from controllers import HoverController
import numpy as np

# Initialize
sim = QuadcopterSimulator()
controller = HoverController()
setpoint = np.array([0.0, 0.0, 1.0])

# Run 10 steps
for i in range(10):
    state = sim.get_state()
    motor_cmd = controller.update(
        setpoint,
        state['position'],
        state['velocity'],
        sim.euler_from_quaternion(state['quaternion']),
        state['angular_velocity'],
        sim.dt
    )
    sim.step(motor_cmd)
```

### Applying Disturbances

```python
# Apply a velocity impulse
sim.apply_velocity_impulse(np.array([1.0, 0.0, 0.0]))  # +1 m/s in X

# Or modify directly (before calling mj_forward):
sim.data.cvel[body_id, 3:6] += velocity_vector
import mujoco
mujoco.mj_forward(sim.model, sim.data)
```

## Extending to VTOL

To add VTOL capability in the future:
1. **Create a new model**: `models/vtol.xml` with tilting motor arms
2. **Extend controllers**: Add transition logic between hover and cruise modes
3. **Add aerodynamics**: Model wing lift and drag
4. **Mode management**: Automatic or manual switching between flight modes

## System Requirements

- **Python**: 3.8+
- **OS**: Linux, macOS, or Windows
- **Key Dependencies**:
  - `mujoco`: ≥3.0.0
  - `numpy`: ≥1.24.0
  - `matplotlib`: ≥3.6.0

## Troubleshooting

**Q: Viewer doesn't open**
- Ensure your system has display capabilities
- Try SSH with X11 forwarding if on remote machine

**Q: Quadcopter crashes or diverges**
- Reduce controller gains in `controllers.py` (especially Kp)
- Check that thrust is sufficient (hover requires ~50% throttle)

**Q: Disturbance not visible**
- Check that velocities actually change in the output
- Verify `apply_velocity_impulse()` is called after t≥2s

**Q: Poor performance or slow simulation**
- Close other applications
- Reduce visualization refresh rate if needed
- Simulation runs at physics timestep regardless of rendering

## References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Quadcopter Dynamics](https://arxiv.org/abs/1301.3516)
- [PID Control Tuning](https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller)

## Future Roadmap

- [ ] VTOL aircraft model (tiltrotor configuration)
- [ ] Quadplane model (fixed-wing + quadcopter)
- [ ] Sensor simulation (IMU, GPS, magnetometer, barometer)
- [ ] Wind disturbance models
- [ ] Ground effect modeling
- [ ] Battery dynamics simulation
- [ ] Motor speed response (first-order lag)
- [ ] Propeller aerodynamic effects
- [ ] Real-time trajectory optimization
- [ ] Integration with RL frameworks (Stable-Baselines3, etc.)

## License

This project is provided for research and educational purposes.

## See Also

Related MuJoCo quadcopter/UAV projects for reference:
- gym_multirotor
- akshitj1/uav-mujoco
- dg7s/AutonomousDroneMujoco
