"""Fly-through-targets demo — navigate a quadcopter through square gates."""

import numpy as np

import matplotlib
matplotlib.use("Agg")

from uavsim.vehicles.multirotor import quadcopter
from uavsim.sim.mujoco_sim import MuJoCoSimulator
from uavsim.controllers.trajectory import TrajectoryController
from uavsim.controllers.mpc import MPCController, default_mpc_config, racing_mpc_config
from uavsim.core.types import HoverGains, PIDGains
from uavsim.disturbances.wind import DrydenWind, DrydenParams
from uavsim.viz.viewer import SimulationVisualizer
from uavsim.viz.plotting import plot_flight_data

# ── gate definitions ─────────────────────────────────────────────────────────
# Each gate is a 4 m × 4 m square frame.  'normal' points in the direction
# the drone should fly through.

GATE_HALF_SIZE = 1.0  # half-width → 4 m total
BEAM_RADIUS = 0.06    # radius of each beam (capsule / box half-size)

GATES = [
    {"center": [5.0, 0.0, 2.0], "normal": [1, 0, 0]},   # fly east
    {"center": [5.0, 5.0, 3.0], "normal": [0, 1, 0]},   # fly north
    {"center": [0.0, 5.0, 2.5], "normal": [-1, 0, 0]},  # fly west
    {"center": [0.0, 0.0, 3.5], "normal": [0, -1, 0]},  # fly south
]

DURATION = 45.0

# ── controller selection ─────────────────────────────────────────────────────
# Choose one of:
#   "pid"         — pure cascaded PID  (TrajectoryController)
#   "mpc"         — MPC + PID, default tracking objective
#   "mpc-racing"  — MPC + PID, racing objective (progress reward + speed tracking)
CONTROLLER = "mpc-racing"

# ── wind settings ────────────────────────────────────────────────────────────
ENABLE_WIND = True          # set to False for calm conditions
MEAN_WIND   = [4.0, 2.0, 0.0]   # steady component [m/s]  (≈ 4.5 m/s from NE)
DRYDEN_PARAMS = DrydenParams(    # gust / turbulence intensities
    sigma_u=1.5, sigma_v=1.5, sigma_w=0.7,
    Lu=200.0, Lv=200.0, Lw=50.0,
)
WIND_SEED = 42

# Gate colours (RGBA)
COLOR_UPCOMING = "0.6 0.6 0.6 0.35"
COLOR_ACTIVE   = "1.0 0.55 0.0 1.0"
COLOR_CLEARED  = "0.2 0.85 0.2 0.7"


# ── helpers ──────────────────────────────────────────────────────────────────

def _gate_axes(normal):
    """Return two unit vectors (u, v) spanning the plane perpendicular to *normal*."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref);  u /= np.linalg.norm(u)
    v = np.cross(n, u);    v /= np.linalg.norm(v)
    return u, v


def gate_corners(center, normal, half_size):
    """Compute the 4 corners of a square gate perpendicular to *normal*."""
    c = np.asarray(center, dtype=float)
    u, v = _gate_axes(normal)
    return [
        c + half_size * u + half_size * v,
        c - half_size * u + half_size * v,
        c - half_size * u - half_size * v,
        c + half_size * u - half_size * v,
    ]


def _inject_gates_into_xml(base_xml_path, gates, half_size, beam_radius):
    """Read the base MJCF, add static gate bodies, return XML string."""
    with open(base_xml_path, "r") as f:
        xml = f.read()

    # Build gate material + bodies XML snippet
    gate_material = (
        '  <material name="mat_gate" rgba="0.6 0.6 0.6 0.35" '
        'specular="0.3" shininess="0.5"/>\n'
    )
    # Inject material into <asset> block (before closing </asset>)
    xml = xml.replace("</asset>", gate_material + "  </asset>")

    # Build gate body elements
    gate_bodies = ""
    for gi, gate in enumerate(gates):
        corners = gate_corners(gate["center"], gate["normal"], half_size)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for ei, (i1, i2) in enumerate(edges):
            p1, p2 = np.asarray(corners[i1]), np.asarray(corners[i2])
            mid = 0.5 * (p1 + p2)
            local_p1 = p1 - mid
            local_p2 = p2 - mid
            gate_bodies += (
                f'    <body name="gate{gi}_beam{ei}" '
                f'pos="{mid[0]:.4f} {mid[1]:.4f} {mid[2]:.4f}">\n'
                f'      <geom name="gate{gi}_geom{ei}" type="capsule" '
                f'fromto="{local_p1[0]:.4f} {local_p1[1]:.4f} {local_p1[2]:.4f} '
                f'{local_p2[0]:.4f} {local_p2[1]:.4f} {local_p2[2]:.4f}" '
                f'size="{beam_radius:.4f}" material="mat_gate" '
                f'rgba="0.6 0.6 0.6 0.35" '
                f'contype="2" conaffinity="0"/>\n'
                f'    </body>\n'
            )

    # Inject gate bodies into <worldbody> (before closing </worldbody>)
    xml = xml.replace("</worldbody>", gate_bodies + "  </worldbody>")
    return xml


def _gate_geom_names(gate_index):
    """Return the 4 geom names for a given gate index."""
    return [f"gate{gate_index}_geom{ei}" for ei in range(4)]


def _update_gates(model, gates, passed_plane):
    """Update gate colours and collision flags.

    Parameters
    ----------
    model : MuJoCo model
    gates : list of gate dicts
    passed_plane : set of gate indices the drone has physically flown past
    """
    import mujoco as mj

    colours = {
        "upcoming": np.array([0.6, 0.6, 0.6, 0.35], dtype=np.float32),
        "active":   np.array([1.0, 0.55, 0.0, 1.0], dtype=np.float32),
        "cleared":  np.array([0.2, 0.85, 0.2, 0.7], dtype=np.float32),
    }

    for gi in range(len(gates)):
        if gi in passed_plane:
            rgba = colours["cleared"]
            con = 0          # no collision — physically past the gate
        else:
            rgba = colours["upcoming"]
            con = 1          # collision enabled on all uncleared gates
        for name in _gate_geom_names(gi):
            gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                model.geom_rgba[gid] = rgba
                model.geom_conaffinity[gid] = con


def _check_gate_passage(position, gate, margin=0.3):
    """Return True if the drone has passed through the gate's plane.

    The drone must be on the far side of the gate (in the *normal* direction)
    by at least *margin* metres relative to the gate centre.
    """
    c = np.asarray(gate["center"], dtype=float)
    n = np.asarray(gate["normal"], dtype=float)
    n = n / np.linalg.norm(n)
    return float(np.dot(np.asarray(position) - c, n)) > margin


# ── main ─────────────────────────────────────────────────────────────────────
def _build_controller(vehicle, mode=CONTROLLER):
    """Instantiate the chosen controller."""
    import jax.numpy as jnp

    if mode == "pid":
        gains = HoverGains(
            kp_pos=3.5, ki_pos=0.1, kd_pos=3.0,
            pos_integral_limit=jnp.array([1.0, 1.0, 0.5]),
            att_gains=PIDGains(kp=8.0, ki=0.2, kd=3.0,
                               max_output=1.5, integral_limit=0.3),
            max_tilt=jnp.float32(jnp.deg2rad(35.0)),
            min_alt=0.12, min_thrust_ratio=0.3,
        )
        return TrajectoryController(vehicle.params, gains=gains,
                                    acceptance_radius=1.5)

    elif mode == "mpc":
        return MPCController(vehicle.params,
                             config=default_mpc_config(vehicle.params),
                             acceptance_radius=1.5)

    elif mode == "mpc-racing":
        return MPCController(vehicle.params,
                             config=racing_mpc_config(vehicle.params),
                             acceptance_radius=1.5)

    else:
        raise ValueError(f"Unknown controller mode: {mode!r}  "
                         f"(choose 'pid', 'mpc', or 'mpc-racing')")

def main():
    print("\n" + "=" * 65)
    print("FLY-THROUGH-TARGETS — GATE NAVIGATION (PHYSICAL GATES)")
    print("=" * 65)
    print(f"  Gates      : {len(GATES)}")
    print(f"  Gate size  : {GATE_HALF_SIZE * 2:.0f} m × {GATE_HALF_SIZE * 2:.0f} m")
    print(f"  Duration   : {DURATION} s")
    print(f"  Controller : {CONTROLLER}")
    print(f"  Wind       : {'Dryden  mean=' + str(MEAN_WIND) if ENABLE_WIND else 'OFF'}")
    print("=" * 65 + "\n")

    vehicle = quadcopter()

    # Build MJCF with physical gate bodies baked in
    mjcf_xml = _inject_gates_into_xml(
        vehicle.mjcf_path, GATES, GATE_HALF_SIZE, BEAM_RADIUS,
    )
    wind_model = (
        DrydenWind(params=DRYDEN_PARAMS, mean_wind=MEAN_WIND, seed=WIND_SEED)
        if ENABLE_WIND else None
    )
    sim = MuJoCoSimulator(vehicle, mjcf_override=mjcf_xml, wind_model=wind_model)

    # Waypoints: take-off → through each gate → descend.
    # Place each gate waypoint 2 m past the gate plane (along the normal)
    # so the drone must physically fly through the gate to reach it.
    GATE_OFFSET = 2.0
    waypoints = [[0.0, 0.0, 2.0]]          # take-off altitude
    for g in GATES:
        c = np.asarray(g["center"], dtype=float)
        n = np.asarray(g["normal"], dtype=float)
        n = n / np.linalg.norm(n)
        waypoints.append((c + n * GATE_OFFSET).tolist())
    waypoints.append([0.0, 0.0, 1.0])      # descend

    ctrl = _build_controller(vehicle)
    ctrl.set_waypoints(waypoints)

    vis = SimulationVisualizer(sim, cam_distance=14.0, cam_elevation=-30.0)
    print("Launching MuJoCo viewer ...")
    vis.launch()

    num_steps = int(DURATION / sim.dt)
    print_every = max(1, int(0.5 / sim.dt))
    passed_plane: set[int] = set()    # gate indices physically passed

    # Set initial gate colours and collision flags
    _update_gates(sim.model, GATES, passed_plane)

    try:
        for step in range(num_steps):
            state = sim.get_state()
            cmd = ctrl.update(state, sim.dt)
            sim.step(cmd)

            # Check physical gate passage (plane crossing)
            pos = np.asarray(state.position)
            updated = False
            for gi, gate in enumerate(GATES):
                if gi not in passed_plane and _check_gate_passage(pos, gate):
                    passed_plane.add(gi)
                    print(f"  ✓ Gate {gi + 1} cleared!")
                    updated = True
            if updated:
                _update_gates(sim.model, GATES, passed_plane)

            if vis.is_running:
                vis.sync(real_time_factor=1.0)

            if step % print_every == 0:
                p = np.asarray(state.position)
                target = "DONE" if ctrl.done else f"Gate {len(passed_plane) + 1}"
                print(
                    f"t={sim.current_time:6.2f}s  "
                    f"pos=[{p[0]:6.3f}, {p[1]:6.3f}, {p[2]:6.3f}]  "
                    f"target={target}"
                )

            if ctrl.done:
                p = np.asarray(state.position)
                wpt = np.asarray(waypoints[-1])
                if np.linalg.norm(p - wpt) < 0.05:
                    print(
                        f"\nAll gates cleared! Final position reached "
                        f"at t={sim.current_time:.2f}s"
                    )
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

    plot_flight_data(
        sim.state_history,
        sim.time_history,
        np.array(waypoints[0]),
        save_path="targets_test_results.png",
        show=True,
    )


if __name__ == "__main__":
    main()
