"""Sensor noise demo — 3 quadcopters fly side-by-side from A to B.

Three identical quadcopters depart from point A and fly to point B,
5 m apart laterally so they are clearly visible in MuJoCo:

  1. **Clean** — ground-truth state fed to the controller.
  2. **Moderate noise** — default sensor config.
  3. **Aggressive noise** — sim-to-real-grade noise with domain randomization.

The physics are always ground-truth; sensor noise only corrupts what the
controller *sees*, leading to visible trajectory divergence.

Run:
    python sensor_noise_demo.py
"""

import textwrap

import numpy as np
import jax
import jax.numpy as jnp
import mujoco

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uavsim.vehicles.multirotor import quadcopter, quadcopter_params
from uavsim.sim.mujoco_sim import MuJoCoSimulator
from uavsim.controllers.hover import HoverController, hover_init, hover_step, default_hover_gains
from uavsim.core.types import VehicleState
from uavsim.viz.viewer import SimulationVisualizer
from uavsim.sensors import SensorConfig, SensorSuite, noisy_sensor_suite


# ── scene parameters ─────────────────────────────────────────────────────────

N_VEHICLES = 3
LATERAL_SPACING = 1.  # metres between vehicles (Y axis)

POINT_A = np.array([0.0, 0.0, 1.5])
POINT_B = np.array([10.0, 0.0, 1.5])

LABELS = ["Clean", "Moderate noise", "Aggressive noise"]
BODY_COLORS = [
    "0.10 0.75 0.20",   # green  – clean
    "0.20 0.40 0.90",   # blue   – moderate
    "0.90 0.15 0.15",   # red    – aggressive
]

DURATION = 10.0
CRUISE_SPEED = 6.0


# ── MJCF generation ──────────────────────────────────────────────────────────

def _quad_body_xml(name: str, pos: tuple, rgba_body: str) -> str:
    """Generate MJCF snippet for one free-floating quadcopter."""
    px, py, pz = pos
    return textwrap.dedent(f"""\
    <body name="{name}" pos="{px} {py} {pz}">
      <freejoint name="{name}_root"/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.0082 0.0082 0.0149"/>
      <geom name="{name}_body" type="box" size="0.08 0.08 0.04"
            rgba="{rgba_body} 1"/>
      <!-- arms -->
      <geom type="box" size="0.17 0.015 0.008" euler="0 0  0.7854" rgba="0.2 0.2 0.2 1"/>
      <geom type="box" size="0.17 0.015 0.008" euler="0 0 -0.7854" rgba="0.2 0.2 0.2 1"/>
      <!-- FL motor -->
      <body name="{name}_motor_fl" pos="-0.17 0.17 0">
        <inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>
        <geom type="cylinder" size="0.016 0.02" rgba="0.1 0.1 0.1 1"/>
        <joint name="{name}_rotor_fl" type="hinge" axis="0 0 1" limited="false" damping="1e-4"/>
        <geom type="cylinder" size="0.115 0.003" pos="0 0 0.022" rgba="0.85 0.15 0.15 0.75"/>
      </body>
      <!-- FR motor -->
      <body name="{name}_motor_fr" pos="0.17 0.17 0">
        <inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>
        <geom type="cylinder" size="0.016 0.02" rgba="0.1 0.1 0.1 1"/>
        <joint name="{name}_rotor_fr" type="hinge" axis="0 0 1" limited="false" damping="1e-4"/>
        <geom type="cylinder" size="0.115 0.003" pos="0 0 0.022" rgba="0.85 0.15 0.15 0.75"/>
      </body>
      <!-- BL motor -->
      <body name="{name}_motor_bl" pos="-0.17 -0.17 0">
        <inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>
        <geom type="cylinder" size="0.016 0.02" rgba="0.1 0.1 0.1 1"/>
        <joint name="{name}_rotor_bl" type="hinge" axis="0 0 1" limited="false" damping="1e-4"/>
        <geom type="cylinder" size="0.115 0.003" pos="0 0 0.022" rgba="0.85 0.15 0.15 0.75"/>
      </body>
      <!-- BR motor -->
      <body name="{name}_motor_br" pos="0.17 -0.17 0">
        <inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>
        <geom type="cylinder" size="0.016 0.02" rgba="0.1 0.1 0.1 1"/>
        <joint name="{name}_rotor_br" type="hinge" axis="0 0 1" limited="false" damping="1e-4"/>
        <geom type="cylinder" size="0.115 0.003" pos="0 0 0.022" rgba="0.85 0.15 0.15 0.75"/>
      </body>
    </body>
    """)


def generate_scene_mjcf() -> str:
    """Build a MuJoCo scene with N_VEHICLES quads spaced laterally."""
    bodies = ""
    actuators = ""
    for i in range(N_VEHICLES):
        name = f"quad_{i}"
        y_offset = (i - (N_VEHICLES - 1) / 2.0) * LATERAL_SPACING
        pos = (POINT_A[0], POINT_A[1] + y_offset, POINT_A[2])
        bodies += _quad_body_xml(name, pos, BODY_COLORS[i])
        for motor in ("fl", "fr", "bl", "br"):
            actuators += (
                f'    <motor name="{name}_act_{motor}" '
                f'joint="{name}_rotor_{motor}" gear="0.01" ctrlrange="-1 1"/>\n'
            )

    # Markers at A (green) and B (red)
    markers = (
        f'    <geom name="marker_A" type="cylinder" size="0.3 0.005" '
        f'pos="{POINT_A[0]} {POINT_A[1]} 0.005" rgba="0.2 0.8 0.2 0.6"/>\n'
        f'    <geom name="marker_B" type="cylinder" size="0.3 0.005" '
        f'pos="{POINT_B[0]} {POINT_B[1]} 0.005" rgba="0.8 0.2 0.2 0.6"/>\n'
    )

    return textwrap.dedent(f"""\
    <?xml version="1.0" ?>
    <mujoco model="sensor_noise_comparison">
      <compiler inertiafromgeom="auto" angle="radian"/>
      <option timestep="0.001" gravity="0 0 -9.81" integrator="RK4"/>
      <visual>
        <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
        <map shadowclip="1" shadowscale="0.6"/>
        <quality shadowsize="2048"/>
        <global offwidth="1280" offheight="720"/>
      </visual>
      <asset>
        <texture name="skybox" type="skybox" builtin="gradient"
                 rgb1="0.4 0.6 0.8" rgb2="0.8 0.9 1.0" width="512" height="512"/>
        <texture name="tex_ground" type="2d" builtin="checker"
                 rgb1="0.45 0.45 0.45" rgb2="0.55 0.55 0.55" width="512" height="512"/>
        <material name="mat_ground" texture="tex_ground" texrepeat="8 8"
                  specular="0.1" shininess="0.1" reflectance="0.0"/>
      </asset>
      <worldbody>
        <geom name="ground" type="plane" size="40 40 0.1" material="mat_ground"/>
    {markers}
    {bodies}
      </worldbody>
      <actuator>
    {actuators}
      </actuator>
    </mujoco>
    """)


# ── multi-vehicle adapter (simplified from cooperative_transport) ────────────

class ThreeQuadAdapter:
    """Manages 3 independent free-flying quads in one MuJoCo scene."""

    def __init__(self, sim: MuJoCoSimulator):
        self.sim = sim
        self.n = N_VEHICLES

        self._body_ids = []
        self._qpos_offsets = []
        self._qvel_offsets = []
        self._act_groups: list[list[int]] = []

        for i in range(N_VEHICLES):
            name = f"quad_{i}"
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self._body_ids.append(bid)

            jnt_adr = sim.model.body_jntadr[bid]
            self._qpos_offsets.append(int(sim.model.jnt_qposadr[jnt_adr]))
            self._qvel_offsets.append(int(sim.model.jnt_dofadr[jnt_adr]))

            acts = []
            for motor in ("fl", "fr", "bl", "br"):
                aid = mujoco.mj_name2id(
                    sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"{name}_act_{motor}")
                if aid >= 0:
                    acts.append(aid)
            self._act_groups.append(acts)

        self._spin_signs = np.array(sim.vehicle.spin_signs, dtype=float)

    def get_state(self, idx: int) -> VehicleState:
        qa = self._qpos_offsets[idx]
        va = self._qvel_offsets[idx]
        d = self.sim.data
        return VehicleState(
            position=jnp.asarray(d.qpos[qa:qa + 3]),
            quaternion=jnp.asarray(d.qpos[qa + 3:qa + 7]),
            velocity=jnp.asarray(d.qvel[va:va + 3]),
            angular_velocity=jnp.asarray(d.qvel[va + 3:va + 6]),
            time=float(d.time),
        )

    def apply_wrenches_and_step(
        self,
        commands: list[jnp.ndarray],
    ) -> None:
        """Apply per-vehicle motor commands and step physics once."""
        for i in range(self.n):
            state_i = self.get_state(i)
            cmds_i = jnp.asarray(commands[i])
            wind_jax = jnp.zeros(3)
            if self.sim.wind_model is not None:
                alt = float(state_i.position[2])
                v_wind = self.sim.wind_model.step(alt, self.sim.dt)
                wind_jax = jnp.asarray(v_wind, dtype=jnp.float32)
            F, T = self.sim._compute_wrench_jit(
                state_i, cmds_i, wind_velocity=wind_jax)
            bid = self._body_ids[i]
            self.sim.data.xfrc_applied[bid, 0:3] = np.asarray(F)
            self.sim.data.xfrc_applied[bid, 3:6] = np.asarray(T)

            # Visual prop spin
            cmds_np = np.asarray(cmds_i)
            for j, aid in enumerate(self._act_groups[i]):
                self.sim.data.ctrl[aid] = self._spin_signs[j] * cmds_np[j]

        mujoco.mj_step(self.sim.model, self.sim.data)

        # Record first vehicle for history
        s0 = self.get_state(0)
        self.sim.state_history.append(s0)
        self.sim.time_history.append(s0.time)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  SENSOR NOISE DEMO — 3 QUADS, A → B")
    print("=" * 70)
    print(f"  Point A : {POINT_A}")
    print(f"  Point B : {POINT_B}")
    print(f"  Spacing : {LATERAL_SPACING} m (Y axis)")
    for i, label in enumerate(LABELS):
        color_tag = ["GREEN", "BLUE", "RED"][i]
        print(f"  Quad {i}  : {label}  ({color_tag})")
    print(f"  Duration: {DURATION} s")
    print("=" * 70)
    print("  Sensor noise only corrupts what the controller sees.")
    print("  Physics always uses ground truth.")
    print("=" * 70 + "\n")

    # ── build scene ──────────────────────────────────────────────────────
    params = quadcopter_params()
    vehicle = quadcopter(params=params)
    mjcf_xml = generate_scene_mjcf()
    sim = MuJoCoSimulator(vehicle, mjcf_override=mjcf_xml)

    # Reset to MJCF defaults (MuJoCoSimulator.reset sets qpos for a single body)
    mujoco.mj_resetData(sim.model, sim.data)
    mujoco.mj_forward(sim.model, sim.data)
    sim.state_history.clear()
    sim.time_history.clear()

    adapter = ThreeQuadAdapter(sim)

    # ── controllers (one per vehicle) ────────────────────────────────────
    controllers = [HoverController(params) for _ in range(N_VEHICLES)]

    # ── sensor suites ────────────────────────────────────────────────────
    sensors: list[SensorSuite | None] = [
        None,                                          # 0: clean
        SensorSuite(SensorConfig(), seed=42),          # 1: moderate
        noisy_sensor_suite(seed=42),                   # 2: aggressive
    ]
    # Domain randomise the aggressive one
    sensors[2].randomize(factor=1.5)

    # ── visualiser ───────────────────────────────────────────────────────
    vis = SimulationVisualizer(sim, cam_distance=15.0, cam_elevation=-30.0,
                               cam_azimuth=180.0)
    print("Launching MuJoCo viewer ...")
    vis.launch()

    # ── simulation loop ──────────────────────────────────────────────────
    num_steps = int(DURATION / sim.dt)
    print_every = max(1, int(0.5 / sim.dt))

    # Track position histories for the plot
    time_log: list[float] = []
    pos_logs: list[list[np.ndarray]] = [[] for _ in range(N_VEHICLES)]

    hdr = f"{'t(s)':<7}" + "".join(f"   {'x':>6} {'y':>6} {'z':>6}  " for l in LABELS)
    print(hdr)
    print("-" * len(hdr))

    try:
        for step in range(num_steps):
            t = sim.current_time

            # Moving setpoint: A → B at cruise speed, then hold B
            progress = min(1.0, t * CRUISE_SPEED / np.linalg.norm(POINT_B - POINT_A))
            base_setpoint = POINT_A + progress * (POINT_B - POINT_A)

            commands = []
            for i in range(N_VEHICLES):
                true_state = adapter.get_state(i)

                # Per-vehicle setpoint (offset in Y to keep lanes)
                y_offset = (i - (N_VEHICLES - 1) / 2.0) * LATERAL_SPACING
                setpoint = jnp.array([
                    base_setpoint[0],
                    base_setpoint[1] + y_offset,
                    base_setpoint[2],
                ])

                # Apply sensor noise: corrupt what the controller sees
                if sensors[i] is not None:
                    meas = sensors[i].observe(true_state, dt=sim.dt)
                    noisy_state = _state_from_measurements(meas, true_state, sensors[i])
                else:
                    noisy_state = true_state

                cmd = controllers[i].update(noisy_state, setpoint, sim.dt)
                commands.append(cmd)

            adapter.apply_wrenches_and_step(commands)

            # Camera: follow formation centroid
            if vis.is_running:
                centroid = np.mean(
                    [np.asarray(adapter.get_state(i).position) for i in range(N_VEHICLES)],
                    axis=0)
                with vis.viewer.lock():
                    alpha = 0.1
                    vis.viewer.cam.lookat[:] = (
                        (1 - alpha) * vis.viewer.cam.lookat + alpha * centroid)
                vis.viewer.sync()
                # Wall-clock sync for real-time
                import time as _time
                sim_elapsed = sim.current_time
                wall_elapsed = _time.monotonic() - vis._wall_start
                ahead = sim_elapsed - wall_elapsed
                if ahead > 0.001:
                    _time.sleep(ahead)

            # Logging
            if step % print_every == 0:
                time_log.append(t)
                parts = [f"{t:<7.2f}"]
                for i in range(N_VEHICLES):
                    p = np.asarray(adapter.get_state(i).position)
                    pos_logs[i].append(p.copy())
                    parts.append(f"   {p[0]:6.2f} {p[1]:6.2f} {p[2]:6.2f}  ")
                print("".join(parts))

    except KeyboardInterrupt:
        print("\nInterrupted.")

    # ── plot ──────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    _plot_results(time_log, pos_logs)

    vis.close()
    print("Done!")


def _state_from_measurements(
    meas: dict,
    true_state: VehicleState,
    suite: SensorSuite,
) -> VehicleState:
    """Build a VehicleState from noisy sensor measurements.

    GPS → position/velocity, IMU gyro → angular velocity.
    Quaternion stays ground-truth (would need a state estimator to fuse).
    """
    flat = suite.to_flat_obs(meas, true_state)
    return VehicleState(
        position=jnp.asarray(flat[0:3]),
        quaternion=jnp.asarray(flat[6:10]),
        velocity=jnp.asarray(flat[3:6]),
        angular_velocity=jnp.asarray(flat[10:13]),
        time=true_state.time,
    )


def _plot_results(times: list[float], pos_logs: list[list[np.ndarray]]) -> None:
    """Plot X/Z position per vehicle and XY ground track."""
    if not times:
        return

    t = np.array(times)
    colors = ["green", "royalblue", "red"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # X position (forward progress)
    ax = axes[0, 0]
    for i in range(N_VEHICLES):
        xs = np.array([p[0] for p in pos_logs[i]])
        ax.plot(t, xs, color=colors[i], lw=1.2, label=LABELS[i])
    ax.set_ylabel("X [m]")
    ax.set_title("Forward Progress")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Z position (altitude hold)
    ax = axes[0, 1]
    for i in range(N_VEHICLES):
        zs = np.array([p[2] for p in pos_logs[i]])
        ax.plot(t, zs, color=colors[i], lw=1.2, label=LABELS[i])
    ax.axhline(POINT_A[2], color="gray", ls="--", lw=0.8, label="Target alt")
    ax.set_ylabel("Z [m]")
    ax.set_title("Altitude Hold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Y deviation (lateral wander from sensor noise)
    ax = axes[1, 0]
    for i in range(N_VEHICLES):
        y_target = (i - (N_VEHICLES - 1) / 2.0) * LATERAL_SPACING
        ys = np.array([p[1] - y_target for p in pos_logs[i]])
        ax.plot(t, ys, color=colors[i], lw=1.2, label=LABELS[i])
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("Y deviation [m]")
    ax.set_xlabel("Time [s]")
    ax.set_title("Lateral Wander (from sensor noise)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-down ground track (X vs Y)
    ax = axes[1, 1]
    for i in range(N_VEHICLES):
        xs = np.array([p[0] for p in pos_logs[i]])
        ys = np.array([p[1] for p in pos_logs[i]])
        ax.plot(xs, ys, color=colors[i], lw=1.0, alpha=0.8, label=LABELS[i])
        ax.scatter(xs[0], ys[0], color=colors[i], marker="o", s=40, zorder=5)
        ax.scatter(xs[-1], ys[-1], color=colors[i], marker="x", s=60, zorder=5)
    ax.scatter([POINT_A[0]], [POINT_A[1]], color="green", marker="s", s=80,
               zorder=10, label="A")
    ax.scatter([POINT_B[0]], [POINT_B[1]], color="red", marker="s", s=80,
               zorder=10, label="B")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Ground Track (top-down)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Sensor Noise Impact: Clean vs Moderate vs Aggressive",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("sensor_noise_comparison.png", dpi=150)
    print("Saved → sensor_noise_comparison.png")


if __name__ == "__main__":
    main()
