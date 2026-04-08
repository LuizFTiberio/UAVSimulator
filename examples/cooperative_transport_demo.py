#!/usr/bin/env python3
"""Cooperative slung-load transport — 3 quadcopters sharing one payload.

Three quadcopters lift a shared payload from point A and deliver it to
point B.  Each vehicle is connected to the payload by a rigid cable
(MuJoCo equality-connect constraint).  The controllers are vmapped
through JAX so the computational cost is nearly the same as a single
vehicle.

Run:
    python cooperative_transport_demo.py
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")

from uavsim.vehicles.multirotor import quadcopter, quadcopter_params
from uavsim.sim.mujoco_sim import MuJoCoSimulator
from uavsim.missions.cooperative_transport import (
    CoopPhase,
    CooperativeTransportMission,
    MultiVehicleSimAdapter,
    cooperative_hover_gains,
    generate_cooperative_mjcf,
)
from uavsim.viz.viewer import SimulationVisualizer
from uavsim.viz.plotting import plot_flight_data

# ── mission parameters ───────────────────────────────────────────────────────

N_VEHICLES = 3

POINT_A = (0.0, 0.0)         # pickup XY [m]
POINT_B = (10.0, 0.0)        # delivery XY [m]
CRUISE_ALT = 2.5             # cruising altitude [m] (higher for 3 cables)

CABLE_LENGTH = 1.0            # cable length per vehicle [m]
PAYLOAD_MASS = 0.30           # shared payload mass [kg]
PAYLOAD_RADIUS = 0.06         # payload sphere radius [m]
CABLE_MASS = 0.02             # mass per cable [kg]
FORMATION_RADIUS = 0.80      # horizontal offset of each quad from payload [m]
ATTACH_RADIUS = 0.10         # radius of attachment circle on payload [m]

DURATION = 60.0               # max sim time [s]


def main():
    # ── mission briefing ─────────────────────────────────────────────────
    base_mass = 1.2  # per vehicle airframe
    per_vehicle_mass = base_mass + CABLE_MASS + PAYLOAD_MASS / N_VEHICLES
    total_mass = N_VEHICLES * (base_mass + CABLE_MASS) + PAYLOAD_MASS
    dist = np.linalg.norm(np.array(POINT_B) - np.array(POINT_A))

    print("\n" + "=" * 65)
    print("  COOPERATIVE SLUNG-LOAD TRANSPORT")
    print("=" * 65)
    print(f"  Vehicles     : {N_VEHICLES} quadcopters")
    print(f"  Pickup   (A) : [{POINT_A[0]:.2f}, {POINT_A[1]:.2f}]  (green marker)")
    print(f"  Delivery (B) : [{POINT_B[0]:.2f}, {POINT_B[1]:.2f}]  (red marker)")
    print(f"  Distance     : {dist:.2f} m")
    print(f"  Cruise alt   : {CRUISE_ALT} m")
    print(f"  Cable length : {CABLE_LENGTH} m  (× {N_VEHICLES})")
    print(f"  Payload      : {PAYLOAD_MASS} kg  (sphere r={PAYLOAD_RADIUS} m)")
    print(f"  Formation R  : {FORMATION_RADIUS} m  (equilateral triangle)")
    print(f"  Per-vehicle  : {per_vehicle_mass:.2f} kg  (airframe + cable + payload/{N_VEHICLES})")
    print(f"  Total mass   : {total_mass:.2f} kg")
    print(f"  Max duration : {DURATION} s")
    print("=" * 65)
    print("  Strategy: jax.vmap(hover_step) → single JIT kernel for all")
    print("            controllers.  One mj_step() for entire scene.")
    print("=" * 65 + "\n")

    # ── build scene ──────────────────────────────────────────────────────
    # Each vehicle's controller needs mass = airframe + cable + share of payload
    # so thrust feed-forward is correct.
    params = quadcopter_params(mass=per_vehicle_mass)
    vehicle = quadcopter(params=params)

    mjcf_xml = generate_cooperative_mjcf(
        n_vehicles=N_VEHICLES,
        cable_length=CABLE_LENGTH,
        payload_mass=PAYLOAD_MASS,
        payload_radius=PAYLOAD_RADIUS,
        cable_mass=CABLE_MASS,
        attach_radius=ATTACH_RADIUS,
        point_a=POINT_A,
        point_b=POINT_B,
        formation_radius=FORMATION_RADIUS,
    )

    sim = MuJoCoSimulator(vehicle, mjcf_override=mjcf_xml)

    # ── multi-vehicle adapter ────────────────────────────────────────────
    body_names = [f"quad_{i}" for i in range(N_VEHICLES)]
    act_groups = [
        [f"quad_{i}_act_fl", f"quad_{i}_act_fr",
         f"quad_{i}_act_bl", f"quad_{i}_act_br"]
        for i in range(N_VEHICLES)
    ]
    adapter = MultiVehicleSimAdapter(sim, body_names, act_groups)

    # Restore MJCF default positions — MuJoCoSimulator.reset() overwrites
    # qpos[0:3] to [0,0,0.9] (designed for a single quad), but in our
    # multi-body scene qpos[0:3] belongs to the payload.  We need the
    # MJCF defaults so the connect constraints start satisfied.
    import mujoco as mj
    mj.mj_resetData(sim.model, sim.data)
    mj.mj_forward(sim.model, sim.data)
    sim.state_history.clear()
    sim.time_history.clear()

    # ── mission + visualiser ─────────────────────────────────────────────
    mission = CooperativeTransportMission(
        sim=sim,
        adapter=adapter,
        n_vehicles=N_VEHICLES,
        point_a=POINT_A,
        point_b=POINT_B,
        params=params,
        cruise_altitude=CRUISE_ALT,
        cable_length=CABLE_LENGTH,
        payload_radius=PAYLOAD_RADIUS,
        formation_radius=FORMATION_RADIUS,
        attach_radius=ATTACH_RADIUS,
    )

    vis = SimulationVisualizer(sim, cam_distance=8.0, cam_elevation=-30.0)

    print("Launching MuJoCo viewer ...")
    vis.launch()

    # ── simulation loop ──────────────────────────────────────────────────
    num_steps = int(DURATION / sim.dt)
    print_every = max(1, int(0.5 / sim.dt))
    prev_phase = None

    try:
        for step in range(num_steps):
            cmds = mission.update(sim.dt)
            adapter.apply_wrenches_and_step(cmds)

            if vis.is_running:
                vis.sync(real_time_factor=1.0)

            # Phase transitions
            if mission.phase != prev_phase:
                _print_phase_banner(mission.phase)
                prev_phase = mission.phase

            # Periodic status
            if step % print_every == 0:
                pay = mission.payload_position
                swing = mission.swing_angle_deg
                phase = mission.phase.value

                # Centroid of quads
                positions = np.array([
                    np.asarray(adapter.get_state(i).position)
                    for i in range(N_VEHICLES)
                ])
                centroid = positions.mean(axis=0)

                extra = f"  swing={swing:5.1f}°" if mission.phase == CoopPhase.TRANSIT else ""
                print(
                    f"  t={sim.current_time:6.2f}s  "
                    f"centroid=[{centroid[0]:6.3f}, {centroid[1]:6.3f}, {centroid[2]:6.3f}]  "
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
    print(f"  Vehicles       : {N_VEHICLES}")
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
                             save_path="cooperative_transport_flight.png")
            print("Flight data plot saved → cooperative_transport_flight.png")
        except Exception as e:
            print(f"Plotting skipped: {e}")

    vis.close()


def _print_phase_banner(phase: CoopPhase) -> None:
    msgs = {
        CoopPhase.TAKEOFF:   "Phase: TAKEOFF — formation ascending to cruise altitude ...",
        CoopPhase.TRANSIT:   "Phase: TRANSIT — formation flying to delivery point ...",
        CoopPhase.DESCEND:   "Phase: DESCEND — formation lowering payload to ground ...",
        CoopPhase.STABILIZE: "Phase: STABILIZE — holding position, waiting for payload to settle ...",
        CoopPhase.DONE:      "Phase: DONE — payload delivered!",
    }
    print(f"\n  >>> {msgs.get(phase, phase.value)}\n")


if __name__ == "__main__":
    main()
