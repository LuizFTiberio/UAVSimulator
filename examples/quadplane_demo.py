"""Quadplane straight-line demo — takeoff, cruise 70 m, decelerate & hold.

Simple A→B test case to validate the cruise control law:
  A = [0, 0, 8]   (takeoff & climb)
  B = [70, 0, 8]   (cruise 70 m, decelerate, hover-hold)
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")

from uavsim.vehicles.quadplane import quadplane
from uavsim.sim.mujoco_sim import MuJoCoSimulator
from uavsim.controllers.quadplane_ctrl import QuadplaneTrajectoryController
from uavsim.controllers.hover import default_hover_gains
from uavsim.core.types import HoverGains, PIDGains
from uavsim.viz.viewer import SimulationVisualizer
from uavsim.viz.plotting import plot_flight_data

# ── parameters ───────────────────────────────────────────────────────────────

ALTITUDE = 8.0
WAYPOINTS = [
    [  0.0, 0.0, ALTITUDE],   # A — takeoff & climb
    [ 70.0, 0.0, ALTITUDE],   # B — cruise target
]
DURATION = 40.0
CRUISE_SPEED = 12.0        # m/s target forward speed


def _quadplane_gains() -> HoverGains:
    """Gains tuned for the 1.5 kg quadplane with pusher."""
    return HoverGains(
        kp_pos=2.5,
        ki_pos=0.08,
        kd_pos=3.5,
        pos_integral_limit=np.array([2.0, 2.0, 1.0]),
        att_gains=PIDGains(kp=8.0, ki=0.2, kd=3.0,
                           max_output=1.5, integral_limit=0.3),
        max_tilt=np.float32(np.deg2rad(20.0)),
        min_alt=0.15,
        min_thrust_ratio=0.25,
    )


def main():
    print("\n" + "=" * 65)
    print("QUADPLANE A → B  (70 m straight line)")
    print("=" * 65)
    print(f"  A            : [0, 0, {ALTITUDE}]")
    print(f"  B            : [70, 0, {ALTITUDE}]")
    print(f"  Cruise speed : {CRUISE_SPEED} m/s")
    print(f"  Duration     : {DURATION} s")
    print("=" * 65 + "\n")

    vehicle = quadplane()
    sim = MuJoCoSimulator(vehicle)

    gains = _quadplane_gains()
    ctrl = QuadplaneTrajectoryController(
        vehicle.params,
        gains=gains,
        cruise_speed=CRUISE_SPEED,
        acceptance_radius=3.0,
        decel_radius=15.0,
    )
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
                v = np.asarray(state.velocity)
                speed = np.linalg.norm(v)
                wpt_idx = ctrl.current_waypoint_index
                pusher = float(cmd[4])
                phase = ctrl.phase.value
                print(f"t={sim.current_time:6.2f}s  "
                      f"pos=[{p[0]:7.2f}, {p[1]:7.2f}, {p[2]:6.2f}]  "
                      f"V={speed:5.2f} m/s  "
                      f"pusher={pusher:.2f}  "
                      f"wpt={wpt_idx}/{len(WAYPOINTS)-1}  "
                      f"phase={phase}")

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

    plot_flight_data(
        sim.state_history,
        sim.time_history,
        np.array(WAYPOINTS[0]),
        save_path="quadplane_trajectory_results.png",
        show=True,
    )


if __name__ == "__main__":
    main()
