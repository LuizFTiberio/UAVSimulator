"""Trajectory tracking demo — square waypoint pattern."""

import numpy as np
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")

from uavsim.controllers.mpc import MPCController
from uavsim.vehicles.multirotor import quadcopter
from uavsim.sim.mujoco_sim import MuJoCoSimulator
from uavsim.controllers.trajectory import TrajectoryController
from uavsim.viz.viewer import SimulationVisualizer
from uavsim.viz.plotting import plot_flight_data

# ── parameters ───────────────────────────────────────────────────────────────

ALTITUDE = 1.0
WAYPOINTS = [
    [0.0, 0.0, ALTITUDE],
    [2.0, 0.0, ALTITUDE],
    [2.0, 2.0, ALTITUDE],
    [0.0, 2.0, ALTITUDE],
    [0.0, 0.0, ALTITUDE],
]
DURATION = 20.0


def main():
    print("\n" + "=" * 65)
    print("TRAJECTORY TRACKING — SQUARE PATTERN")
    print("=" * 65)
    print(f"  Waypoints : {len(WAYPOINTS)}")
    print(f"  Altitude  : {ALTITUDE} m")
    print(f"  Duration  : {DURATION} s")
    print("=" * 65 + "\n")

    vehicle = quadcopter()
    sim = MuJoCoSimulator(vehicle)
    #ctrl = TrajectoryController(vehicle.params)
    ctrl = MPCController(vehicle.params)
    ctrl.set_waypoints(WAYPOINTS)
    vis = SimulationVisualizer(sim)

    print("Launching MuJoCo viewer ...")
    vis.launch()

    num_steps = int(DURATION / sim.dt)
    print_every = max(1, int(0.5 / sim.dt))

    try:
        for step in range(num_steps):
            state = sim.get_state()
            cmd = ctrl.update(state, sim.dt)
            sim.step(cmd)

            if vis.is_running:
                vis.sync(real_time_factor=1.0)

            if step % print_every == 0:
                p = np.asarray(state.position)
                wpt_idx = ctrl.current_waypoint_index
                print(f"t={sim.current_time:6.2f}s  "
                      f"pos=[{p[0]:6.3f}, {p[1]:6.3f}, {p[2]:6.3f}]  "
                      f"wpt={wpt_idx}/{len(WAYPOINTS)-1}")

            # Stop once the final waypoint is reached and settled
            if ctrl.done:
                p = np.asarray(state.position)
                wpt = np.asarray(WAYPOINTS[-1])
                if np.linalg.norm(p - wpt) < 0.05:
                    print(f"\nFinal waypoint reached at t={sim.current_time:.2f}s — stopping.")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("\nSimulation done — close viewer to generate plots.")
        while vis.is_running:
            try:
                vis.sync(real_time_factor=0.0)
            except Exception:
                break
        vis.close()

    # Use the first waypoint as the "setpoint" for plotting
    plot_flight_data(
        sim.state_history,
        sim.time_history,
        np.array(WAYPOINTS[0]),
        save_path="trajectory_test_results.png",
        show=True,
    )


if __name__ == "__main__":
    main()
