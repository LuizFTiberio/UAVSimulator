#!/usr/bin/env python3
"""Slung-load transport mission — carry a payload from A to B.

A quadcopter lifts a payload attached by a rigid cable, flies it across,
and lowers it onto the delivery point.  MuJoCo handles pendulum dynamics
and ground contact; the controller uses gentle gains to minimise swing.

Run:
    python run_transport.py
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")

from uavsim.vehicles.multirotor import slung_quadcopter
from uavsim.sim.mujoco_sim import MuJoCoSimulator
from uavsim.missions.transport import (
    MissionPhase,
    TransportMission,
    generate_slung_load_mjcf,
)
from uavsim.viz.viewer import SimulationVisualizer
from uavsim.viz.plotting import plot_flight_data

# ── mission parameters ───────────────────────────────────────────────────────

POINT_A = (0.0, 0.0)         # pickup XY [m]
POINT_B = (3.0, 0.0)         # delivery XY [m]
CRUISE_ALT = 1.5             # cruising altitude [m]

CABLE_LENGTH = 0.4            # rigid cable length [m]
PAYLOAD_MASS = 0.15           # payload mass [kg]
PAYLOAD_RADIUS = 0.04         # payload sphere radius [m]
CABLE_MASS = 0.02             # cable mass [kg]

DURATION = 30.0               # max sim time [s]


def main():
    # ── mission briefing ─────────────────────────────────────────────────
    total_mass = 1.2 + PAYLOAD_MASS + CABLE_MASS
    dist = np.linalg.norm(np.array(POINT_B) - np.array(POINT_A))
    place_alt = 0.04 + CABLE_LENGTH + PAYLOAD_RADIUS

    print("\n" + "=" * 65)
    print("  SLUNG-LOAD TRANSPORT MISSION")
    print("=" * 65)
    print(f"  Pickup   (A) : [{POINT_A[0]:.2f}, {POINT_A[1]:.2f}]  (green marker)")
    print(f"  Delivery (B) : [{POINT_B[0]:.2f}, {POINT_B[1]:.2f}]  (red marker)")
    print(f"  Distance     : {dist:.2f} m")
    print(f"  Cruise alt   : {CRUISE_ALT} m")
    print(f"  Cable length : {CABLE_LENGTH} m")
    print(f"  Payload      : {PAYLOAD_MASS} kg  (sphere r={PAYLOAD_RADIUS} m)")
    print(f"  Cable        : {CABLE_MASS} kg")
    print(f"  Total mass   : {total_mass:.2f} kg  (airframe 1.20 kg)")
    print(f"  Place alt    : {place_alt:.2f} m  (quad hover height for touchdown)")
    print(f"  Max duration : {DURATION} s")
    print("=" * 65 + "\n")

    # ── build scene ──────────────────────────────────────────────────────
    vehicle = slung_quadcopter(
        payload_mass=PAYLOAD_MASS, cable_mass=CABLE_MASS)
    mjcf_xml = generate_slung_load_mjcf(
        cable_length=CABLE_LENGTH,
        payload_mass=PAYLOAD_MASS,
        payload_radius=PAYLOAD_RADIUS,
        cable_mass=CABLE_MASS,
        point_a=POINT_A,
        point_b=POINT_B,
    )
    sim = MuJoCoSimulator(vehicle, mjcf_override=mjcf_xml)

    # Reset quad at point A with correct altitude
    start_alt = 0.04 + CABLE_LENGTH + PAYLOAD_RADIUS + 0.02
    sim.reset(position=np.array([POINT_A[0], POINT_A[1], start_alt]))

    # ── mission + visualiser ─────────────────────────────────────────────
    mission = TransportMission(
        sim=sim,
        point_a=POINT_A,
        point_b=POINT_B,
        params=vehicle.params,
        cruise_altitude=CRUISE_ALT,
        cable_length=CABLE_LENGTH,
        payload_radius=PAYLOAD_RADIUS,
    )
    vis = SimulationVisualizer(sim, cam_distance=5.0)

    print("Launching MuJoCo viewer ...")
    vis.launch()

    # ── simulation loop ──────────────────────────────────────────────────
    num_steps = int(DURATION / sim.dt)
    print_every = max(1, int(0.5 / sim.dt))
    prev_phase = None

    try:
        for step in range(num_steps):
            state = sim.get_state()
            cmd = mission.update(state, sim.dt)
            sim.step(cmd)

            if vis.is_running:
                vis.sync(real_time_factor=1.0)

            # Print phase transitions
            if mission.phase != prev_phase:
                _print_phase_banner(mission.phase)
                prev_phase = mission.phase

            # Periodic status
            if step % print_every == 0:
                p = np.asarray(state.position)
                pay = mission.payload_position
                swing = mission.swing_angle_deg
                phase = mission.phase.value
                extra = f"  swing={swing:5.1f}°" if mission.phase == MissionPhase.TRANSIT else ""
                print(
                    f"  t={sim.current_time:6.2f}s  "
                    f"quad=[{p[0]:6.3f}, {p[1]:6.3f}, {p[2]:6.3f}]  "
                    f"payload=[{pay[0]:6.3f}, {pay[1]:6.3f}, {pay[2]:6.3f}]"
                    f"{extra}  phase={phase}"
                )

            if mission.done:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # ── mission summary ──────────────────────────────────────────────────
    pay_final = mission.payload_position
    transit_dt = mission.transit_end_time - mission.transit_start_time

    print("\n" + "=" * 65)
    if mission.done:
        print("  MISSION COMPLETE")
    else:
        print("  MISSION TIMED OUT")
    print("=" * 65)
    print(f"  Total time     : {sim.current_time:.2f} s")
    if transit_dt > 0:
        print(f"  Transit time   : {transit_dt:.2f} s")
        print(f"  Avg speed      : {dist / transit_dt:.2f} m/s")
    print(f"  Max cable swing: {mission.max_swing_deg:.1f}°")
    print(f"  Placement error: {mission.placement_error:.3f} m")
    print(f"  Final payload  : [{pay_final[0]:.3f}, {pay_final[1]:.3f}, {pay_final[2]:.3f}]")
    print("=" * 65 + "\n")

    # ── plots ────────────────────────────────────────────────────────────
    if sim.state_history:
        try:
            plot_flight_data(sim.state_history, sim.time_history, show=False,
                             save_path="transport_mission_flight.png")
            print("Flight data plot saved → transport_mission_flight.png")
        except Exception as e:
            print(f"Plotting skipped: {e}")

    vis.close()


def _print_phase_banner(phase: MissionPhase) -> None:
    """Print a banner for a phase transition."""
    msgs = {
        MissionPhase.TAKEOFF:   "Phase: TAKEOFF — ascending to cruise altitude ...",
        MissionPhase.TRANSIT:   "Phase: TRANSIT — flying to delivery point ...",
        MissionPhase.DESCEND:   "Phase: DESCEND — lowering payload to ground ...",
        MissionPhase.STABILIZE: "Phase: STABILIZE — waiting for payload to settle ...",
        MissionPhase.DONE:      "Phase: DONE — payload delivered!",
    }
    print(f"\n  >>> {msgs.get(phase, phase.value)}\n")


if __name__ == "__main__":
    main()
