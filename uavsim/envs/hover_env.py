"""Hover stabilisation environment."""

from __future__ import annotations

import numpy as np

from uavsim.sensors.models import SensorConfig
from uavsim.vehicles.base import VehicleModel
from uavsim.vehicles.multirotor import quadcopter
from uavsim.envs.base_env import BaseUAVEnv


class HoverEnv(BaseUAVEnv):
    """Hover at a fixed setpoint.

    Reward: negative position error + velocity penalty.
    Terminates if the vehicle crashes (z < 0) or tilts > 60 deg.
    """

    def __init__(
        self,
        vehicle: VehicleModel | None = None,
        setpoint: np.ndarray | None = None,
        max_episode_steps: int = 5000,
        render_mode: str | None = None,
        sensor_config: SensorConfig | None = None,
        sensor_seed: int = 0,
        wind_model=None,
    ):
        if vehicle is None:
            vehicle = quadcopter()
        super().__init__(
            vehicle, max_episode_steps, render_mode,
            sensor_config=sensor_config, sensor_seed=sensor_seed,
            wind_model=wind_model,
        )
        self.setpoint = (
            np.array([0.0, 0.0, 1.0]) if setpoint is None
            else np.asarray(setpoint, dtype=np.float32)
        )

    def _compute_reward(self) -> float:
        state = self.sim.get_state()
        pos = np.asarray(state.position)
        vel = np.asarray(state.velocity)

        pos_err = float(np.linalg.norm(pos - self.setpoint))
        vel_pen = float(np.linalg.norm(vel)) * 0.1

        return -(pos_err + vel_pen)

    def _check_terminated(self) -> bool:
        state = self.sim.get_state()
        pos = np.asarray(state.position)

        # Crash: below ground
        if pos[2] < 0.0:
            return True

        # Extreme tilt (quaternion w component: cos(angle/2))
        qw = float(state.quaternion[0])
        if abs(qw) < 0.5:   # roughly > 60 deg tilt
            return True

        return False
