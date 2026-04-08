"""MuJoCo simulation backend — vehicle-agnostic.

Wraps a MuJoCo model + data pair and delegates force computation
to the vehicle's ``compute_wrench`` function (pure JAX).
"""

from __future__ import annotations

import logging
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
import mujoco

from uavsim.core.types import VehicleState
from uavsim.core.gpu import gpu_info
from uavsim.vehicles.base import VehicleModel

logger = logging.getLogger(__name__)


class MuJoCoSimulator:
    """Vehicle-agnostic MuJoCo simulator.

    Forces are computed in JAX via the vehicle's dynamics function,
    then injected into MuJoCo via ``xfrc_applied`` each step.

    Parameters
    ----------
    vehicle : VehicleModel
        Vehicle definition (params, MJCF path, dynamics).
    wind_model : WindModel | None
        Optional wind model.  If provided, wind velocity is queried each
        step and passed to the vehicle's ``compute_wrench`` function.
    """

    def __init__(
        self,
        vehicle: VehicleModel,
        mjcf_override: str | None = None,
        wind_model=None,
    ):
        self.vehicle = vehicle
        self.wind_model = wind_model
        if mjcf_override is not None:
            self.model = mujoco.MjModel.from_xml_string(mjcf_override)
        else:
            self.model = mujoco.MjModel.from_xml_path(str(vehicle.mjcf_path))
        self.data = mujoco.MjData(self.model)

        self._body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "body")

        # Resolve visual actuator IDs (optional, for prop spin)
        self._act_ids: list[int] = []
        self._act_cmds: list[int] = []  # index into command array
        for i, name in enumerate(vehicle.actuator_names):
            aid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                self._act_ids.append(aid)
                self._act_cmds.append(i)
        self._spin_signs = np.array(vehicle.spin_signs, dtype=float)

        # JIT-compile the wrench function for this vehicle's params
        info = gpu_info()
        device = jax.devices(info.jax_backend.value)[0] if info.has_gpu else jax.devices("cpu")[0]
        self._jax_device = device
        logger.info("MuJoCoSimulator using JAX device: %s", device)

        self._compute_wrench_jit = jax.jit(
            partial(vehicle.compute_wrench, params=vehicle.params),
        )

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

        # Query wind model (if any)
        if self.wind_model is not None:
            altitude = float(state.position[2])
            v_wind = self.wind_model.step(altitude, self.dt)
            wind_jax = jnp.asarray(v_wind, dtype=jnp.float32)
        else:
            wind_jax = jnp.zeros(3)

        # Compute forces via JIT-compiled vehicle dynamics
        F_world, T_world = self._compute_wrench_jit(
            state, cmds, wind_velocity=wind_jax)

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

    @property
    def wind_velocity(self) -> np.ndarray:
        """Current wind velocity (world frame, m/s).  Zeros if no wind model."""
        if self.wind_model is not None:
            return np.asarray(self.wind_model.current_velocity)
        return np.zeros(3)

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

        if self.wind_model is not None:
            self.wind_model.reset()

        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    # ── private ──────────────────────────────────────────────────────────

    def _spin_props(self, throttle: np.ndarray) -> None:
        """Drive actuator signals so the props rotate visually."""
        for aid, ci in zip(self._act_ids, self._act_cmds):
            self.data.ctrl[aid] = self._spin_signs[ci] * throttle[ci]
