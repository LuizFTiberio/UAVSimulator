"""MuJoCo simulation backend — vehicle-agnostic.

Wraps a MuJoCo model + data pair and delegates force computation
to the vehicle's ``compute_wrench`` function (pure JAX).
"""

from __future__ import annotations

from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
import mujoco

from uavsim.core.types import VehicleState
from uavsim.vehicles.base import VehicleModel


class MuJoCoSimulator:
    """Vehicle-agnostic MuJoCo simulator.

    Forces are computed in JAX via the vehicle's dynamics function,
    then injected into MuJoCo via ``xfrc_applied`` each step.

    Parameters
    ----------
    vehicle : VehicleModel
        Vehicle definition (params, MJCF path, dynamics).
    """

    def __init__(self, vehicle: VehicleModel, mjcf_override: str | None = None):
        self.vehicle = vehicle
        if mjcf_override is not None:
            self.model = mujoco.MjModel.from_xml_string(mjcf_override)
        else:
            self.model = mujoco.MjModel.from_xml_path(str(vehicle.mjcf_path))
        self.data = mujoco.MjData(self.model)

        self._body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "body")

        # Resolve visual actuator IDs (optional, for prop spin)
        self._act_ids: list[int] = []
        for name in vehicle.actuator_names:
            self._act_ids.append(
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
        self._spin_signs = np.array(vehicle.spin_signs, dtype=float)

        # JIT-compile the wrench function for this vehicle's params
        self._compute_wrench_jit = jax.jit(
            partial(vehicle.compute_wrench, params=vehicle.params))

        # History for post-flight analysis
        self.state_history: list[VehicleState] = []
        self.time_history: list[float] = []

        self.reset()

    # ── properties ───────────────────────────────────────────────────────

    @property
    def dt(self) -> float:
        return self.model.opt.timestep

    @property
    def current_time(self) -> float:
        return float(self.data.time)

    @property
    def total_mass(self) -> float:
        """Total mass of all bodies in the model (read from MuJoCo)."""
        return float(sum(self.model.body_mass))

    # ── state ────────────────────────────────────────────────────────────

    def get_state(self) -> VehicleState:
        """Return the current vehicle state as a VehicleState NamedTuple."""
        return VehicleState(
            position=jnp.asarray(self.data.qpos[0:3]),
            quaternion=jnp.asarray(self.data.qpos[3:7]),
            velocity=jnp.asarray(self.data.qvel[0:3]),
            angular_velocity=jnp.asarray(self.data.qvel[3:6]),
            time=float(self.data.time),
        )

    # ── step ─────────────────────────────────────────────────────────────

    def step(self, motor_commands: np.ndarray | jnp.ndarray) -> VehicleState:
        """Advance simulation by one timestep.

        Parameters
        ----------
        motor_commands : (n_motors,) normalised throttle [0, 1].

        Returns
        -------
        VehicleState after the step.
        """
        cmds = jnp.asarray(motor_commands, dtype=jnp.float32)
        state = self.get_state()

        # Compute forces via JIT-compiled vehicle dynamics
        F_world, T_world = self._compute_wrench_jit(state, cmds)

        # Inject into MuJoCo
        bid = self._body_id
        self.data.xfrc_applied[bid, 0:3] = np.asarray(F_world)
        self.data.xfrc_applied[bid, 3:6] = np.asarray(T_world)

        # Drive visual prop spin
        self._spin_props(np.asarray(cmds))

        mujoco.mj_step(self.model, self.data)

        new_state = self.get_state()
        self.state_history.append(new_state)
        self.time_history.append(new_state.time)
        return new_state

    # ── disturbance ──────────────────────────────────────────────────────

    def apply_velocity_impulse(self, impulse: np.ndarray) -> None:
        """Add a velocity impulse (world frame, m/s) to the body."""
        self.data.qvel[0:3] += np.asarray(impulse, dtype=float)
        mujoco.mj_forward(self.model, self.data)

    # ── reset ────────────────────────────────────────────────────────────

    def reset(
        self,
        position: np.ndarray | None = None,
        yaw: float = 0.0,
    ) -> VehicleState:
        """Reset to a clean hover-ready state.

        Parameters
        ----------
        position : (3,) spawn position. Default: (0, 0, 0.9).
        yaw : initial yaw [rad]. Default: 0.

        Returns
        -------
        VehicleState after reset.
        """
        mujoco.mj_resetData(self.model, self.data)

        pos = np.array([0.0, 0.0, 0.9]) if position is None else np.asarray(position)
        self.data.qpos[0:3] = pos
        self.data.qpos[3] = np.cos(yaw / 2)   # w
        self.data.qpos[4] = 0.0               # x
        self.data.qpos[5] = 0.0               # y
        self.data.qpos[6] = np.sin(yaw / 2)   # z

        self.state_history.clear()
        self.time_history.clear()

        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    # ── private ──────────────────────────────────────────────────────────

    def _spin_props(self, throttle: np.ndarray) -> None:
        """Drive actuator signals so the props rotate visually."""
        for i, aid in enumerate(self._act_ids):
            self.data.ctrl[aid] = self._spin_signs[i] * throttle[i]
