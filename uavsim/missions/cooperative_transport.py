"""Cooperative slung-load transport — N quadcopters sharing one payload.

Three (or more) quadcopters each attach to a shared payload via rigid
cables connected through ball joints.  The formation geometry, MJCF
scene, and mission state machine live here.

Computational strategy
----------------------
* **Single MuJoCo scene** — all vehicles + cables + payload in one
  ``mj_step`` call.  MuJoCo handles all inter-body constraints and
  contacts; the cost scales sub-linearly with the number of bodies.
* **Vmapped controllers** — ``jax.vmap(hover_step)`` batches all N
  hover controllers into a single JIT-compiled kernel.  On GPU this
  is essentially constant-time regardless of fleet size; on CPU it
  still eliminates Python-loop overhead.
* **Shared parameters & gains** — all vehicles are identical, so
  ``params`` and ``gains`` are broadcast (not replicated) by vmap.
"""

from __future__ import annotations

from enum import Enum

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from uavsim.controllers.hover import hover_init, hover_step
from uavsim.core.types import (
    HoverGains,
    HoverState,
    MultirotorParams,
    PIDGains,
    PIDState,
    VehicleState,
)
from uavsim.sim.mujoco_sim import MuJoCoSimulator


# ═════════════════════════════════════════════════════════════════════════════
#  Formation geometry
# ═════════════════════════════════════════════════════════════════════════════

def equilateral_offsets(
    n: int = 3,
    radius: float = 0.20,
) -> np.ndarray:
    """XY offsets for *n* evenly-spaced attach points on a circle.

    Returns (n, 2) array of [dx, dy] offsets from the payload centre.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles),
                            radius * np.sin(angles)])


# ═════════════════════════════════════════════════════════════════════════════
#  Gains tuned for cooperative transport
# ═════════════════════════════════════════════════════════════════════════════

def cooperative_hover_gains() -> HoverGains:
    """Gentle hover gains for cooperative slung-load transport.

    Even softer than the single-vehicle transport gains:
      * lower kp_pos  → reduces formation oscillation
      * higher kd_pos → damps cable-coupled modes
      * 15° max tilt  → avoids large lateral forces that pull
        the payload asymmetrically
    """
    return HoverGains(
        kp_pos=1.8,
        ki_pos=0.10,
        kd_pos=3.5,
        pos_integral_limit=jnp.array([1.0, 1.0, 0.5]),
        att_gains=PIDGains(
            kp=8.0, ki=0.2, kd=3.0,
            max_output=1.5, integral_limit=0.3,
        ),
        max_tilt=jnp.float32(jnp.deg2rad(15.0)),
        min_alt=0.12,
        min_thrust_ratio=0.3,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Vmapped hover controller
# ═════════════════════════════════════════════════════════════════════════════

def _batched_hover_init(n: int, gains: HoverGains) -> HoverState:
    """Create a batch of *n* hover states stacked along axis 0."""
    single = hover_init(gains)
    return jax.tree.map(lambda x: jnp.stack([x] * n), single)


def _make_batched_hover_step(
    params: MultirotorParams,
    gains: HoverGains,
):
    """Return a JIT-compiled function that steps N controllers in parallel.

    Signature::

        (batched_hover_state, batched_vehicle_state, batched_setpoints, dt)
        -> (new_batched_hover_state, batched_throttles)

    ``batched_vehicle_state`` is a VehicleState where every field has an
    extra leading axis of size N (one per vehicle).  ``batched_setpoints``
    is (N, 3).  ``batched_throttles`` is (N, 4).
    """
    # Closure captures params and gains so they don't collide with
    # positional args when vmapped.
    def _single_step(hover_state, vehicle_state, setpoint, dt):
        return hover_step(hover_state, vehicle_state, setpoint,
                          params, gains, dt)

    vmapped = jax.vmap(_single_step, in_axes=(0, 0, 0, None))

    @jax.jit
    def step(hover_states, vehicle_states, setpoints, dt):
        return vmapped(hover_states, vehicle_states, setpoints, dt)

    return step


# ═════════════════════════════════════════════════════════════════════════════
#  Multi-body simulator adapter
# ═════════════════════════════════════════════════════════════════════════════

class MultiVehicleSimAdapter:
    """Thin wrapper around a single MuJoCoSimulator that exposes per-vehicle API.

    All vehicles share the same ``VehicleModel`` (params + wrench fn).
    Physics is stepped once per call via a single ``mj_step``.
    """

    def __init__(
        self,
        sim: MuJoCoSimulator,
        body_names: list[str],
        actuator_groups: list[list[str]],
    ):
        self.sim = sim
        self.n = len(body_names)

        # Resolve body IDs
        self._body_ids = [
            mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in body_names
        ]

        # Resolve actuator IDs per vehicle (for prop spin visuals)
        self._act_groups: list[list[int]] = []
        for group in actuator_groups:
            ids = []
            for name in group:
                aid = mujoco.mj_name2id(
                    sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if aid >= 0:
                    ids.append(aid)
            self._act_groups.append(ids)

        # Find qpos/qvel offsets for each body's freejoint
        self._qpos_offsets: list[int] = []
        self._qvel_offsets: list[int] = []
        for bid in self._body_ids:
            jnt_adr = sim.model.body_jntadr[bid]
            self._qpos_offsets.append(int(sim.model.jnt_qposadr[jnt_adr]))
            self._qvel_offsets.append(int(sim.model.jnt_dofadr[jnt_adr]))

        self._spin_signs = np.array(sim.vehicle.spin_signs, dtype=float)

    def get_state(self, idx: int) -> VehicleState:
        """Return VehicleState for vehicle *idx*."""
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

    def get_batched_state(self) -> VehicleState:
        """Return VehicleState with fields stacked as (N, ...)."""
        states = [self.get_state(i) for i in range(self.n)]
        return jax.tree.map(lambda *xs: jnp.stack(xs), *states)

    def apply_wrenches_and_step(
        self,
        motor_commands: np.ndarray | jnp.ndarray,
    ) -> None:
        """Apply per-vehicle forces and step physics once.

        Parameters
        ----------
        motor_commands : (N, 4) throttles [0, 1]
        """
        cmds = np.asarray(motor_commands)
        for i in range(self.n):
            state_i = self.get_state(i)
            F, T = self.sim._compute_wrench_jit(state_i, jnp.asarray(cmds[i]))
            bid = self._body_ids[i]
            self.sim.data.xfrc_applied[bid, 0:3] = np.asarray(F)
            self.sim.data.xfrc_applied[bid, 3:6] = np.asarray(T)

            # Visual prop spin
            for j, aid in enumerate(self._act_groups[i]):
                self.sim.data.ctrl[aid] = self._spin_signs[j] * cmds[i][j]

        mujoco.mj_step(self.sim.model, self.sim.data)

        # Record state of first vehicle for history/plotting
        state0 = self.get_state(0)
        self.sim.state_history.append(state0)
        self.sim.time_history.append(state0.time)


# ═════════════════════════════════════════════════════════════════════════════
#  MJCF generation — multi-vehicle scene
# ═════════════════════════════════════════════════════════════════════════════

_ATTACH_OFFSET = 0.04  # cable attaches below quad centre [m]


def _quad_body_mjcf(
    name: str,
    pos: tuple[float, float, float],
    cable_length: float,
    cable_radius: float,
    cable_mass: float,
    cable_damping: float,
    mat_body: str = "mat_body",
) -> str:
    """Generate MJCF for one quadcopter with a cable hanging below it.

    The cable is a rigid rod child body attached via a ball joint at the
    quad's underside, hanging straight down.  A massless tip body sits at
    the cable bottom; a ``connect`` equality constraint (set up externally)
    pins it to the payload.
    """
    prefix = name
    px, py, pz = pos
    cable_ixx = cable_mass * cable_length ** 2 / 12.0
    cable_izz = cable_mass * cable_radius ** 2 / 2.0
    cable_com_z = -cable_length / 2.0
    return f"""
    <!-- ── {prefix} ──────────────────────────────────────────── -->
    <body name="{prefix}" pos="{px:.4f} {py:.4f} {pz:.4f}">
      <freejoint name="{prefix}_root"/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.0082 0.0082 0.0149"/>
      <geom name="{prefix}_geom" type="box" size="0.08 0.08 0.04" material="{mat_body}"/>

      <!-- Arms -->
      <geom name="{prefix}_arm1" type="box" size="0.17 0.015 0.008"
            euler="0 0  0.7854" material="mat_arm" contype="0" conaffinity="0"/>
      <geom name="{prefix}_arm2" type="box" size="0.17 0.015 0.008"
            euler="0 0 -0.7854" material="mat_arm" contype="0" conaffinity="0"/>

      <!-- FL motor -->
      <body name="{prefix}_motor_fl" pos="-0.17 0.17 0.0">
        <inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>
        <geom type="cylinder" size="0.016 0.02" material="mat_motor" contype="0" conaffinity="0"/>
        <joint name="{prefix}_rotor_fl" type="hinge" axis="0 0 1" limited="false" damping="1e-4"/>
        <geom type="cylinder" size="0.115 0.003" pos="0 0 0.022" material="mat_rotor" contype="0" conaffinity="0"/>
      </body>
      <!-- FR motor -->
      <body name="{prefix}_motor_fr" pos="0.17 0.17 0.0">
        <inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>
        <geom type="cylinder" size="0.016 0.02" material="mat_motor" contype="0" conaffinity="0"/>
        <joint name="{prefix}_rotor_fr" type="hinge" axis="0 0 1" limited="false" damping="1e-4"/>
        <geom type="cylinder" size="0.115 0.003" pos="0 0 0.022" material="mat_rotor" contype="0" conaffinity="0"/>
      </body>
      <!-- BL motor -->
      <body name="{prefix}_motor_bl" pos="-0.17 -0.17 0.0">
        <inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>
        <geom type="cylinder" size="0.016 0.02" material="mat_motor" contype="0" conaffinity="0"/>
        <joint name="{prefix}_rotor_bl" type="hinge" axis="0 0 1" limited="false" damping="1e-4"/>
        <geom type="cylinder" size="0.115 0.003" pos="0 0 0.022" material="mat_rotor" contype="0" conaffinity="0"/>
      </body>
      <!-- BR motor -->
      <body name="{prefix}_motor_br" pos="0.17 -0.17 0.0">
        <inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>
        <geom type="cylinder" size="0.016 0.02" material="mat_motor" contype="0" conaffinity="0"/>
        <joint name="{prefix}_rotor_br" type="hinge" axis="0 0 1" limited="false" damping="1e-4"/>
        <geom type="cylinder" size="0.115 0.003" pos="0 0 0.022" material="mat_rotor" contype="0" conaffinity="0"/>
      </body>

      <!-- Slung cable: ball joint at fuselage bottom, hangs down -->
      <body name="{prefix}_cable" pos="0 0 -{_ATTACH_OFFSET}">
        <joint name="{prefix}_cable_joint" type="ball" damping="{cable_damping}"/>
        <inertial pos="0 0 {cable_com_z:.6f}" mass="{cable_mass}"
                  diaginertia="{cable_ixx:.6e} {cable_ixx:.6e} {cable_izz:.6e}"/>
        <geom name="{prefix}_cable_geom" type="capsule"
              fromto="0 0 0  0 0 -{cable_length}" size="{cable_radius}"
              material="mat_cable"/>
        <!-- Tip body: connect constraint pins this to the payload -->
        <body name="{prefix}_cable_tip" pos="0 0 -{cable_length}">
          <inertial pos="0 0 0" mass="0.001" diaginertia="1e-8 1e-8 1e-8"/>
          <geom name="{prefix}_cable_tip_geom" type="sphere" size="0.01"
                rgba="0.9 0.9 0.9 0.8" contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>"""


def generate_cooperative_mjcf(
    n_vehicles: int = 3,
    cable_length: float = 0.5,
    payload_mass: float = 0.30,
    payload_radius: float = 0.06,
    cable_radius: float = 0.005,
    cable_mass: float = 0.02,
    cable_damping: float = 0.005,
    attach_radius: float = 0.10,
    point_a: tuple[float, float] = (0.0, 0.0),
    point_b: tuple[float, float] = (3.0, 0.0),
    formation_radius: float = 0.40,
) -> str:
    """Generate MJCF XML for N quadcopters cooperatively carrying one payload.

    The payload sits at ground level at point A.  Each quadcopter is
    positioned in an equilateral formation above the payload, connected
    by a cable (rigid rod + ball joint) to an attachment lug on the
    payload.

    Parameters
    ----------
    n_vehicles : int
        Number of quadcopters (default 3).
    cable_length : float
        Length of each cable [m].
    payload_mass : float
        Mass of the shared payload [kg].
    payload_radius : float
        Radius of the payload sphere [m].
    attach_radius : float
        Radius of the circle of cable attach points on the payload [m].
    formation_radius : float
        Horizontal offset of each quad from the payload centre [m].

    Returns
    -------
    str : Complete MJCF XML.
    """
    ax, ay = point_a
    bx, by = point_b

    # Payload inertia (solid sphere)
    payload_i = 0.4 * payload_mass * payload_radius ** 2

    # Formation offsets (XY) for vehicles
    offsets = equilateral_offsets(n_vehicles, formation_radius)

    # Horizontal gap from quad to its payload attach point:
    h_gap = formation_radius - attach_radius
    # Vertical reach of cable when spanning that horizontal gap:
    v_reach = (cable_length**2 - h_gap**2)**0.5 if cable_length > h_gap else 0.1

    # Attach point offsets on the payload
    attach_offsets = equilateral_offsets(n_vehicles, attach_radius)

    # Colours for each vehicle body
    body_colours = [
        "0.15 0.20 0.80 1",  # blue
        "0.15 0.70 0.20 1",  # green
        "0.80 0.20 0.15 1",  # red
        "0.70 0.50 0.10 1",  # amber
        "0.50 0.15 0.70 1",  # purple
        "0.15 0.60 0.70 1",  # teal
    ]

    # ── build XML ────────────────────────────────────────────────────────
    parts: list[str] = []

    # Header
    parts.append(f"""\
<?xml version="1.0" ?>
<!--
  Cooperative slung-load transport ({n_vehicles} quadcopters)
  Auto-generated by uavsim.missions.cooperative_transport
-->
<mujoco model="cooperative_slung_load">

  <compiler inertiafromgeom="auto"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>

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
    <material name="mat_ground"  texture="tex_ground" texrepeat="8 8"
              specular="0.1" shininess="0.1" reflectance="0.0"/>
    <material name="mat_arm"     rgba="0.20 0.20 0.20 1"  specular="0.2"/>
    <material name="mat_motor"   rgba="0.10 0.10 0.10 1"  specular="0.3"/>
    <material name="mat_rotor"   rgba="0.85 0.15 0.15 0.75" specular="0.2"/>
    <material name="mat_cable"   rgba="0.30 0.30 0.30 1"  specular="0.1"/>
    <material name="mat_payload" rgba="0.90 0.60 0.10 1"  specular="0.4" shininess="0.6"/>""")

    # Per-vehicle body material
    for i in range(n_vehicles):
        c = body_colours[i % len(body_colours)]
        parts.append(f'    <material name="mat_body_{i}" rgba="{c}" specular="0.5" shininess="0.7"/>')

    parts.append("  </asset>\n")

    # ── worldbody ────────────────────────────────────────────────────────
    parts.append("  <worldbody>")
    parts.append('    <geom name="ground" type="plane" size="20 20 0.1" material="mat_ground"/>')

    # Markers
    parts.append(f"""
    <!-- Pickup marker (green) -->
    <geom name="marker_a" type="cylinder"
          pos="{ax:.4f} {ay:.4f} 0.001" size="0.30 0.002"
          rgba="0.2 0.8 0.2 0.5" contype="0" conaffinity="0"/>
    <!-- Delivery marker (red) -->
    <geom name="marker_b" type="cylinder"
          pos="{bx:.4f} {by:.4f} 0.001" size="0.30 0.002"
          rgba="0.8 0.2 0.2 0.5" contype="0" conaffinity="0"/>""")

    # ── Payload body (standalone, free-floating) ───────────────────────
    pay_start_z = payload_radius + 0.02  # just above ground

    parts.append(f"""
    <!-- ══ Shared payload ══════════════════════════════════════════ -->
    <body name="payload" pos="{ax:.4f} {ay:.4f} {pay_start_z:.4f}">
      <freejoint name="payload_root"/>
      <inertial pos="0 0 0" mass="{payload_mass}"
                diaginertia="{payload_i:.6e} {payload_i:.6e} {payload_i:.6e}"/>
      <geom name="payload_geom" type="sphere" size="{payload_radius}"
            material="mat_payload"/>""")

    # Attach point markers on payload (visual only)
    for i in range(n_vehicles):
        dx, dy = attach_offsets[i]
        parts.append(f'      <geom name="attach_{i}" type="sphere" size="0.012"'
                     f' pos="{dx:.4f} {dy:.4f} 0" rgba="1 1 1 0.6"'
                     f' contype="0" conaffinity="0"/>')

    parts.append("    </body>")

    # ── Quadcopter bodies (each with cable hanging below) ────────────
    # IMPORTANT: quads start directly above their payload attach point
    # so the cable tip coincides with the attach point at compile time.
    # This is required for MuJoCo's connect constraint to auto-compute
    # the correct body2 anchor.  The controller spreads the formation
    # out during the TAKEOFF phase.
    init_alt = pay_start_z + cable_length + _ATTACH_OFFSET
    for i in range(n_vehicles):
        dx, dy = attach_offsets[i]
        qx = ax + dx
        qy = ay + dy
        qz = init_alt
        parts.append(_quad_body_mjcf(
            f"quad_{i}", (qx, qy, qz),
            cable_length=cable_length,
            cable_radius=cable_radius,
            cable_mass=cable_mass,
            cable_damping=cable_damping,
            mat_body=f"mat_body_{i}"))

    parts.append("  </worldbody>\n")

    # ── Equality constraints: pin each cable tip to the payload ──────
    # MuJoCo connect: forces body2's origin to coincide with `anchor`
    # on body1.  Each cable tip is at the bottom of a cable hanging
    # from a quad; we pin it to the corresponding attach lug on the
    # payload.
    parts.append("  <!-- Cable tips pinned to payload attach points -->")
    parts.append("  <equality>")
    for i in range(n_vehicles):
        dx, dy = attach_offsets[i]
        parts.append(
            f'    <connect name="link_{i}" body1="payload" body2="quad_{i}_cable_tip"'
            f' anchor="{dx:.4f} {dy:.4f} 0"'
            f' solref="0.005 1" solimp="0.95 0.99 0.001"/>'
        )
    parts.append("  </equality>\n")

    # ── Contact exclusions ───────────────────────────────────────────────
    parts.append("  <contact>")
    for i in range(n_vehicles):
        parts.append(f'    <exclude body1="quad_{i}" body2="payload"/>')
        parts.append(f'    <exclude body1="quad_{i}_cable" body2="payload"/>')
        parts.append(f'    <exclude body1="quad_{i}_cable_tip" body2="payload"/>')
        for j in range(i + 1, n_vehicles):
            parts.append(f'    <exclude body1="quad_{i}" body2="quad_{j}"/>')
    parts.append("  </contact>\n")

    # ── Actuators (visual prop spin) ─────────────────────────────────────
    parts.append("  <actuator>")
    for i in range(n_vehicles):
        p = f"quad_{i}"
        parts.append(f'    <motor name="{p}_act_fl" joint="{p}_rotor_fl" gear="0.01" ctrlrange="-1 1"/>')
        parts.append(f'    <motor name="{p}_act_fr" joint="{p}_rotor_fr" gear="0.01" ctrlrange="-1 1"/>')
        parts.append(f'    <motor name="{p}_act_bl" joint="{p}_rotor_bl" gear="0.01" ctrlrange="-1 1"/>')
        parts.append(f'    <motor name="{p}_act_br" joint="{p}_rotor_br" gear="0.01" ctrlrange="-1 1"/>')
    parts.append("  </actuator>")

    parts.append("\n</mujoco>")

    return "\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
#  Mission state machine
# ═════════════════════════════════════════════════════════════════════════════

class CoopPhase(Enum):
    """Phases of a cooperative transport mission."""
    TAKEOFF   = "takeoff"
    TRANSIT   = "transit"
    DESCEND   = "descend"
    STABILIZE = "stabilize"
    DONE      = "done"


class CooperativeTransportMission:
    """State machine for N quadcopters cooperatively transporting a payload.

    Each vehicle tracks its own formation waypoint (offset from the
    desired payload position).  Controllers are vmapped for efficiency.

    Parameters
    ----------
    sim : MuJoCoSimulator
        Simulator loaded with the cooperative MJCF scene.
    adapter : MultiVehicleSimAdapter
        Multi-body adapter wrapping *sim*.
    n_vehicles : int
        Number of quadcopters.
    point_a, point_b : (2,) array-like
        Pickup and delivery XY coordinates.
    params : MultirotorParams
        Vehicle parameters (mass should include per-vehicle cable share).
    cruise_altitude : float
        Cruising altitude for the payload centre [m].
    cable_length : float
        Length of each cable [m].
    payload_radius : float
        Payload sphere radius [m].
    formation_radius : float
        Horizontal offset of each vehicle from the payload centre [m].
    gains : HoverGains, optional
        Controller gains.  Defaults to *cooperative_hover_gains()*.
    """

    def __init__(
        self,
        sim: MuJoCoSimulator,
        adapter: MultiVehicleSimAdapter,
        n_vehicles: int = 3,
        point_a: tuple[float, float] = (0.0, 0.0),
        point_b: tuple[float, float] = (3.0, 0.0),
        params: MultirotorParams | None = None,
        cruise_altitude: float = 1.5,
        cable_length: float = 0.5,
        payload_radius: float = 0.06,
        formation_radius: float = 0.40,
        attach_radius: float = 0.10,
        gains: HoverGains | None = None,
    ):
        self.sim = sim
        self.adapter = adapter
        self.n = n_vehicles
        self.point_a = np.asarray(point_a, dtype=float)
        self.point_b = np.asarray(point_b, dtype=float)
        self.cruise_alt = cruise_altitude
        self.cable_length = cable_length
        self.payload_radius = payload_radius
        self.formation_radius = formation_radius

        # Formation offsets (XY, relative to payload centre)
        self._offsets = equilateral_offsets(n_vehicles, formation_radius)  # (N, 2)

        # Payload body ID
        self._payload_id = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_BODY, "payload")

        # Vertical reach of cable given formation geometry
        h_gap = formation_radius - attach_radius
        self._v_reach = (cable_length**2 - h_gap**2)**0.5 if cable_length > h_gap else 0.1

        # Placement altitude: payload just above ground
        self._place_alt = payload_radius + self._v_reach + _ATTACH_OFFSET

        # Controller (vmapped)
        gains = gains or cooperative_hover_gains()
        self.gains = gains
        if params is None:
            params = adapter.sim.vehicle.params
        self.params = params

        self._hover_states = _batched_hover_init(n_vehicles, gains)
        self._batched_step = _make_batched_hover_step(params, gains)

        # Phase
        self.phase = CoopPhase.TAKEOFF
        self._stabilize_start: float = 0.0

        # Statistics
        self.max_swing_deg: float = 0.0
        self.transit_start_time: float = 0.0
        self.transit_end_time: float = 0.0
        self.payload_history: list[np.ndarray] = []

        self._enter_phase(CoopPhase.TAKEOFF)

    # ── payload sensing ──────────────────────────────────────────────────

    @property
    def payload_position(self) -> np.ndarray:
        """World-frame position of the payload centre."""
        return self.sim.data.xpos[self._payload_id].copy()

    @property
    def swing_angle_deg(self) -> float:
        """Average cable deviation from vertical [deg]."""
        pay_pos = self.payload_position
        angles = []
        for i in range(self.n):
            quad_pos = np.asarray(self.adapter.get_state(i).position)
            attach = quad_pos.copy()
            attach[2] -= _ATTACH_OFFSET
            cable_vec = pay_pos - attach
            cos_a = -cable_vec[2] / (np.linalg.norm(cable_vec) + 1e-8)
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        return float(np.mean(angles))

    # ── formation setpoints ──────────────────────────────────────────────

    def _formation_setpoints(self, payload_target: np.ndarray) -> jnp.ndarray:
        """Compute (N, 3) setpoints for each vehicle given desired payload pos.

        Each vehicle hovers at its formation offset horizontally, at a
        height determined by the cable's vertical reach (accounting for
        the angled cable geometry).
        """
        setpoints = np.zeros((self.n, 3))
        for i in range(self.n):
            setpoints[i, 0] = payload_target[0] + self._offsets[i, 0]
            setpoints[i, 1] = payload_target[1] + self._offsets[i, 1]
            setpoints[i, 2] = payload_target[2] + self._v_reach + _ATTACH_OFFSET
        return jnp.asarray(setpoints, dtype=jnp.float32)

    # ── phase management ─────────────────────────────────────────────────

    def _enter_phase(self, phase: CoopPhase) -> None:
        self.phase = phase

    def _payload_target(self) -> np.ndarray:
        """Current desired payload position based on phase."""
        if self.phase == CoopPhase.TAKEOFF:
            return np.array([self.point_a[0], self.point_a[1],
                             self.cruise_alt - self._v_reach - _ATTACH_OFFSET])
        elif self.phase == CoopPhase.TRANSIT:
            return np.array([self.point_b[0], self.point_b[1],
                             self.cruise_alt - self._v_reach - _ATTACH_OFFSET])
        elif self.phase in (CoopPhase.DESCEND, CoopPhase.STABILIZE, CoopPhase.DONE):
            return np.array([self.point_b[0], self.point_b[1],
                             self.payload_radius + 0.02])
        return np.array([self.point_a[0], self.point_a[1], self.cruise_alt])

    # ── main update ──────────────────────────────────────────────────────

    def update(self, dt: float) -> np.ndarray:
        """Compute motor commands for all vehicles and handle phase transitions.

        Returns (N, 4) throttle array [0, 1].
        """
        # Get batched vehicle states
        batched_state = self.adapter.get_batched_state()

        # Representative position (centroid of all quads)
        positions = np.asarray(batched_state.position)  # (N, 3)
        centroid = positions.mean(axis=0)

        pay_pos = self.payload_position
        self.payload_history.append(pay_pos.copy())

        swing = self.swing_angle_deg
        self.max_swing_deg = max(self.max_swing_deg, swing)

        t = float(self.sim.data.time)

        # ── phase transitions ────────────────────────────────────────
        if self.phase == CoopPhase.TAKEOFF:
            alt_ok = centroid[2] > self.cruise_alt - 0.20
            xy_ok = np.linalg.norm(centroid[:2] - self.point_a) < 0.5
            vel_ok = all(
                abs(float(batched_state.velocity[i, 2])) < 0.3
                for i in range(self.n)
            )
            if alt_ok and xy_ok and vel_ok:
                self.transit_start_time = t
                self._enter_phase(CoopPhase.TRANSIT)

        elif self.phase == CoopPhase.TRANSIT:
            xy_ok = np.linalg.norm(centroid[:2] - self.point_b) < 0.4
            alt_ok = abs(centroid[2] - self.cruise_alt) < 0.3
            if xy_ok and alt_ok:
                self.transit_end_time = t
                self._enter_phase(CoopPhase.DESCEND)

        elif self.phase == CoopPhase.DESCEND:
            target_quad_alt = self.payload_radius + 0.02 + self.cable_length + _ATTACH_OFFSET
            alt_ok = abs(centroid[2] - target_quad_alt) < 0.15
            if alt_ok:
                self._stabilize_start = t
                self._enter_phase(CoopPhase.STABILIZE)

        elif self.phase == CoopPhase.STABILIZE:
            pay_xy_err = np.linalg.norm(pay_pos[:2] - self.point_b)
            near = pay_xy_err < 0.20
            settled = swing < 3.0
            timeout = (t - self._stabilize_start) > 5.0
            if (near and settled) or timeout:
                self._enter_phase(CoopPhase.DONE)

        # ── compute formation setpoints ──────────────────────────────
        payload_target = self._payload_target()
        setpoints = self._formation_setpoints(payload_target)

        # ── vmapped controller step ──────────────────────────────────
        self._hover_states, throttles = self._batched_step(
            self._hover_states, batched_state, setpoints, dt)

        return np.asarray(throttles)

    # ── queries ──────────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return self.phase == CoopPhase.DONE

    @property
    def placement_error(self) -> float:
        """Horizontal distance from payload to delivery point [m]."""
        pay = self.payload_position
        return float(np.linalg.norm(pay[:2] - self.point_b))
