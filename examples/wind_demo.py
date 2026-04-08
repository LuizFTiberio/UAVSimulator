"""Wind disturbance demo — hover under Dryden turbulence.

Three quadcopters hover side-by-side under different wind conditions:
  1. **No wind**   (green) — baseline calm hover.
  2. **Steady wind** (blue) — constant 5 m/s crosswind.
  3. **Dryden turbulence** (red) — moderate turbulence + 4 m/s mean wind.

The physics use true airspeed (vehicle velocity minus wind) for body drag,
so wind genuinely pushes the vehicle.  The controller sees ground-truth
state and must compensate.

Run:
    python wind_demo.py
"""

import textwrap

import numpy as np
import jax.numpy as jnp
import mujoco

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uavsim.vehicles.multirotor import quadcopter, quadcopter_params
from uavsim.sim.mujoco_sim import MuJoCoSimulator
from uavsim.controllers.hover import HoverController
from uavsim.core.types import VehicleState
from uavsim.viz.viewer import SimulationVisualizer
from uavsim.disturbances.wind import (
    ConstantWind,
    DrydenWind,
    moderate_turbulence,
)

# ── scene parameters ─────────────────────────────────────────────────────────

N_VEHICLES = 3
LATERAL_SPACING = 1.0

SETPOINT_ALT = 1.5
SETPOINT = np.array([0.0, 0.0, SETPOINT_ALT])

LABELS = ["No wind", "Steady 5 m/s crosswind", "Dryden turbulence"]
BODY_COLORS = [
    "0.10 0.75 0.20",   # green — calm
    "0.20 0.40 0.90",   # blue  — steady wind
    "0.90 0.15 0.15",   # red   — turbulence
]

DURATION = 7.0


# ── MJCF generation ──────────────────────────────────────────────────────────

def _quad_body_xml(name: str, pos: tuple, rgba_body: str) -> str:
    px, py, pz = pos
    return textwrap.dedent(f"""\
    <body name="{name}" pos="{px} {py} {pz}">
      <freejoint name="{name}_root"/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.0082 0.0082 0.0149"/>
      <geom name="{name}_body" type="box" size="0.08 0.08 0.04"
            rgba="{rgba_body} 1"/>
      <geom type="box" size="0.17 0.015 0.008" euler="0 0  0.7854" rgba="0.2 0.2 0.2 1"/>
      <geom type="box" size="0.17 0.015 0.008" euler="0 0 -0.7854" rgba="0.2 0.2 0.2 1"/>
      <body name="{name}_motor_fl" pos="-0.17 0.17 0">
        <inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>
        <geom type="cylinder" size="0.016 0.02" rgba="0.1 0.1 0.1 1"/>
        <joint name="{name}_rotor_fl" type="hinge" axis="0 0 1" limited="false" damping="1e-4"/>
        <geom type="cylinder" size="0.115 0.003" pos="0 0 0.022" rgba="0.85 0.15 0.15 0.75"/>
      </body>
      <body name="{name}_motor_fr" pos="0.17 0.17 0">
        <inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>
        <geom type="cylinder" size="0.016 0.02" rgba="0.1 0.1 0.1 1"/>
        <joint name="{name}_rotor_fr" type="hinge" axis="0 0 1" limited="false" damping="1e-4"/>
        <geom type="cylinder" size="0.115 0.003" pos="0 0 0.022" rgba="0.85 0.15 0.15 0.75"/>
      </body>
      <body name="{name}_motor_bl" pos="-0.17 -0.17 0">
        <inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>
        <geom type="cylinder" size="0.016 0.02" rgba="0.1 0.1 0.1 1"/>
        <joint name="{name}_rotor_bl" type="hinge" axis="0 0 1" limited="false" damping="1e-4"/>
        <geom type="cylinder" size="0.115 0.003" pos="0 0 0.022" rgba="0.85 0.15 0.15 0.75"/>
      </body>
      <body name="{name}_motor_br" pos="0.17 -0.17 0">
        <inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>
        <geom type="cylinder" size="0.016 0.02" rgba="0.1 0.1 0.1 1"/>
        <joint name="{name}_rotor_br" type="hinge" axis="0 0 1" limited="false" damping="1e-4"/>
        <geom type="cylinder" size="0.115 0.003" pos="0 0 0.022" rgba="0.85 0.15 0.15 0.75"/>
      </body>
    </body>
    """)


def generate_scene_mjcf() -> str:
    bodies = ""
    actuators = ""
    for i in range(N_VEHICLES):
        name = f"quad_{i}"
        y_offset = (i - (N_VEHICLES - 1) / 2.0) * LATERAL_SPACING
        pos = (0.0, y_offset, SETPOINT_ALT)
        bodies += _quad_body_xml(name, pos, BODY_COLORS[i])
        for motor in ("fl", "fr", "bl", "br"):
            actuators += (
                f'    <motor name="{name}_act_{motor}" '
                f'joint="{name}_rotor_{motor}" gear="0.01" ctrlrange="-1 1"/>\n'
            )

    return textwrap.dedent(f"""\
    <?xml version="1.0" ?>
    <mujoco model="wind_comparison">
      <compiler inertiafromgeom="auto" angle="radian"/>
      <option timestep="0.001" gravity="0 0 -9.81" integrator="RK4"/>
      <visual>
        <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
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
    {bodies}
      </worldbody>
      <actuator>
    {actuators}
      </actuator>
    </mujoco>
    """)


# ── multi-vehicle adapter ────────────────────────────────────────────────────

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
        wind_velocities: list[jnp.ndarray] | None = None,
    ) -> None:
        for i in range(self.n):
            state_i = self.get_state(i)
            cmds_i = jnp.asarray(commands[i])
            w_i = wind_velocities[i] if wind_velocities is not None else jnp.zeros(3)
            F, T = self.sim._compute_wrench_jit(
                state_i, cmds_i, wind_velocity=w_i)
            bid = self._body_ids[i]
            self.sim.data.xfrc_applied[bid, 0:3] = np.asarray(F)
            self.sim.data.xfrc_applied[bid, 3:6] = np.asarray(T)
            cmds_np = np.asarray(cmds_i)
            for j, aid in enumerate(self._act_groups[i]):
                self.sim.data.ctrl[aid] = self._spin_signs[j] * cmds_np[j]

        mujoco.mj_step(self.sim.model, self.sim.data)
        s0 = self.get_state(0)
        self.sim.state_history.append(s0)
        self.sim.time_history.append(s0.time)


# ── wind models (per vehicle) ────────────────────────────────────────────────

def _build_wind_models():
    return [
        ConstantWind([0.0, 0.0, 0.0]),                       # no wind
        ConstantWind([0.0, 5.0, 0.0]),                        # steady crosswind
        moderate_turbulence(mean_wind=[4.0, 3.0, 0.0], seed=42),  # Dryden
    ]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  WIND DISTURBANCE DEMO — 3 QUADS HOVERING")
    print("=" * 70)
    print(f"  Setpoint : [0, 0, {SETPOINT_ALT}]")
    print(f"  Spacing  : {LATERAL_SPACING} m (Y axis)")
    for i, label in enumerate(LABELS):
        color_tag = ["GREEN", "BLUE", "RED"][i]
        print(f"  Quad {i}   : {label}  ({color_tag})")
    print(f"  Duration : {DURATION} s")
    print("=" * 70)
    print("  Wind affects body drag via true airspeed.")
    print("  Controller sees ground-truth state and must compensate.")
    print("=" * 70 + "\n")

    # ── build scene ──────────────────────────────────────────────────────
    params = quadcopter_params()
    vehicle = quadcopter(params=params)
    mjcf_xml = generate_scene_mjcf()
    sim = MuJoCoSimulator(vehicle, mjcf_override=mjcf_xml)

    mujoco.mj_resetData(sim.model, sim.data)
    mujoco.mj_forward(sim.model, sim.data)
    sim.state_history.clear()
    sim.time_history.clear()

    adapter = ThreeQuadAdapter(sim)
    controllers = [HoverController(params) for _ in range(N_VEHICLES)]
    wind_models = _build_wind_models()

    vis = SimulationVisualizer(sim, cam_distance=8.0, cam_elevation=-25.0,
                               cam_azimuth=180.0)
    print("Launching MuJoCo viewer ...")
    vis.launch()

    num_steps = int(DURATION / sim.dt)
    print_every = max(1, int(0.5 / sim.dt))

    time_log: list[float] = []
    pos_logs: list[list[np.ndarray]] = [[] for _ in range(N_VEHICLES)]
    wind_logs: list[list[np.ndarray]] = [[] for _ in range(N_VEHICLES)]

    hdr = f"{'t(s)':<7}" + "".join(f"   {'x':>6} {'y':>6} {'z':>6}  " for _ in LABELS)
    print(hdr)
    print("-" * len(hdr))

    try:
        for step in range(num_steps):
            t = sim.current_time

            commands = []
            wind_vecs = []
            for i in range(N_VEHICLES):
                true_state = adapter.get_state(i)

                # Per-vehicle setpoint (offset in Y)
                y_offset = (i - (N_VEHICLES - 1) / 2.0) * LATERAL_SPACING
                sp = jnp.array([0.0, y_offset, SETPOINT_ALT])

                cmd = controllers[i].update(true_state, sp, sim.dt)
                commands.append(cmd)

                # Query wind for this vehicle
                alt = float(true_state.position[2])
                v_wind = wind_models[i].step(alt, sim.dt)
                wind_vecs.append(jnp.asarray(v_wind, dtype=jnp.float32))

            adapter.apply_wrenches_and_step(commands, wind_velocities=wind_vecs)

            if vis.is_running:
                vis.sync(real_time_factor=1.0)

            if step % print_every == 0:
                time_log.append(t)
                parts = [f"{t:<7.2f}"]
                for i in range(N_VEHICLES):
                    p = np.asarray(adapter.get_state(i).position)
                    pos_logs[i].append(p.copy())
                    wind_logs[i].append(wind_models[i].current_velocity.copy())
                    parts.append(f"   {p[0]:6.2f} {p[1]:6.2f} {p[2]:6.2f}  ")
                print("".join(parts))

    except KeyboardInterrupt:
        print("\nInterrupted.")

    print("\nGenerating plots ...")
    _plot_results(time_log, pos_logs, wind_logs)
    vis.close()
    print("Done!")


def _plot_results(
    times: list[float],
    pos_logs: list[list[np.ndarray]],
    wind_logs: list[list[np.ndarray]],
) -> None:
    if not times:
        return

    t = np.array(times)
    colors = ["green", "royalblue", "red"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # X displacement (should stay ~0 for hover)
    ax = axes[0, 0]
    for i in range(N_VEHICLES):
        xs = np.array([p[0] for p in pos_logs[i]])
        ax.plot(t, xs, color=colors[i], lw=1.2, label=LABELS[i])
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("X [m]")
    ax.set_title("X Displacement (drift from wind)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Y deviation from lane centre
    ax = axes[0, 1]
    for i in range(N_VEHICLES):
        y_target = (i - (N_VEHICLES - 1) / 2.0) * LATERAL_SPACING
        ys = np.array([p[1] - y_target for p in pos_logs[i]])
        ax.plot(t, ys, color=colors[i], lw=1.2, label=LABELS[i])
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("Y deviation [m]")
    ax.set_title("Lateral Drift (crosswind effect)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Z altitude hold
    ax = axes[1, 0]
    for i in range(N_VEHICLES):
        zs = np.array([p[2] for p in pos_logs[i]])
        ax.plot(t, zs, color=colors[i], lw=1.2, label=LABELS[i])
    ax.axhline(SETPOINT_ALT, color="gray", ls="--", lw=0.8, label="Target")
    ax.set_ylabel("Z [m]")
    ax.set_xlabel("Time [s]")
    ax.set_title("Altitude Hold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Wind magnitude over time
    ax = axes[1, 1]
    for i in range(N_VEHICLES):
        wm = np.array([np.linalg.norm(w) for w in wind_logs[i]])
        ax.plot(t, wm, color=colors[i], lw=1.0, alpha=0.8, label=LABELS[i])
    ax.set_ylabel("Wind speed [m/s]")
    ax.set_xlabel("Time [s]")
    ax.set_title("Wind Magnitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Wind Disturbance: No Wind vs Steady vs Dryden Turbulence",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("wind_comparison.png", dpi=150)
    print("Saved → wind_comparison.png")


if __name__ == "__main__":
    main()
