"""Hover demo — disturbance recovery test.

Reproduces the original main.py scenario using the new package structure:
  1. Hover at [0, 0, 1.0] m for 2 seconds.
  2. Apply a +1 m/s X-axis velocity impulse.
  3. Watch the controller recover.
"""

import numpy as np
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")

from uavsim.vehicles.multirotor import quadcopter
from uavsim.sim.mujoco_sim import MuJoCoSimulator
from uavsim.controllers.hover import HoverController
from uavsim.viz.viewer import SimulationVisualizer
from uavsim.viz.plotting import plot_flight_data

# ── parameters ───────────────────────────────────────────────────────────────

SETPOINT = jnp.array([0.0, 0.0, 1.0])
DURATION = 8.0
DISTURBANCE_TIME = 2.0
DISTURBANCE_VEL = np.array([1.0, 0.0, 0.0])


def main():
    print("\n" + "=" * 65)
    print("QUADCOPTER HOVER — DISTURBANCE RECOVERY")
    print("=" * 65)
    print(f"  Setpoint    : {SETPOINT}")
    print(f"  Disturbance : {DISTURBANCE_VEL} m/s at t = {DISTURBANCE_TIME} s")
    print(f"  Duration    : {DURATION} s")
    print("=" * 65 + "\n")

    vehicle = quadcopter()
    sim = MuJoCoSimulator(vehicle)
    ctrl = HoverController(vehicle.params)
    vis = SimulationVisualizer(sim)

    print("Launching MuJoCo viewer ...")
    vis.launch()

    num_steps = int(DURATION / sim.dt)
    disturbance_done = False
    print_every = max(1, int(0.25 / sim.dt))

    hdr = f"{'t(s)':<7} {'x':>8} {'y':>8} {'z':>8} {'vx':>8} {'vy':>8} {'vz':>8}"
    print(hdr)
    print("-" * len(hdr))

    try:
        for step in range(num_steps):
            # Disturbance
            if not disturbance_done and sim.current_time >= DISTURBANCE_TIME:
                sim.apply_velocity_impulse(DISTURBANCE_VEL)
                disturbance_done = True
                print(f"\n*** DISTURBANCE at t={sim.current_time:.3f}s "
                      f"-> dv = {DISTURBANCE_VEL} m/s ***\n")

            # Controller
            state = sim.get_state()
            cmd = ctrl.update(state, SETPOINT, sim.dt)

            # Step
            sim.step(cmd)

            # Visualise
            if vis.is_running:
                vis.sync(real_time_factor=1.0)

            # Print
            if step % print_every == 0:
                p = np.asarray(state.position)
                v = np.asarray(state.velocity)
                print(f"{sim.current_time:<7.3f} "
                      f"{p[0]:>8.4f} {p[1]:>8.4f} {p[2]:>8.4f} "
                      f"{v[0]:>8.4f} {v[1]:>8.4f} {v[2]:>8.4f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("\nSimulation done — viewer stays open for inspection.")
        print("Close the viewer window to generate plots.")
        while vis.is_running:
            try:
                vis.sync(real_time_factor=0.0)
            except Exception:
                break
        vis.close()

    # Summary
    if sim.state_history:
        final = sim.state_history[-1]
        pos = np.asarray(final.position)
        vel = np.asarray(final.velocity)
        err = float(np.linalg.norm(pos - np.asarray(SETPOINT)))
        print(f"\n  Final position : {pos}")
        print(f"  Final velocity : {vel}")
        print(f"  |error|        : {err:.4f} m\n")

    # Plot
    plot_flight_data(
        sim.state_history,
        sim.time_history,
        np.asarray(SETPOINT),
        disturbance_time=DISTURBANCE_TIME,
        save_path="stability_test_results.png",
        show=True,
    )


if __name__ == "__main__":
    main()
