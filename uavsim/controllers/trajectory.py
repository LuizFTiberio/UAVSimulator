"""Waypoint-following controller built on the hover controller."""

import jax.numpy as jnp
import numpy as np

from uavsim.core.types import HoverGains, MultirotorParams, VehicleState
from uavsim.controllers.hover import (
    HoverController, default_hover_gains, hover_init, hover_step,
)


class TrajectoryController:
    """Waypoint-following controller.

    Waypoints are [x, y, z] or [x, y, z, yaw].
    Advances to the next waypoint once within ``acceptance_radius``.
    """

    def __init__(
        self,
        params: MultirotorParams,
        gains: HoverGains | None = None,
        acceptance_radius: float = 0.15,
    ):
        self.params = params
        self.gains = gains if gains is not None else default_hover_gains()
        self.radius = acceptance_radius
        self._hover_state = hover_init(self.gains)
        self._wpts: list[jnp.ndarray] = []
        self._idx: int = 0

    def set_waypoints(self, waypoints: list | np.ndarray) -> None:
        """Set the waypoint list. Each entry: [x, y, z] or [x, y, z, yaw]."""
        self._wpts = [jnp.asarray(w, dtype=jnp.float32) for w in waypoints]
        self._idx = 0
        self._hover_state = hover_init(self.gains)

    @property
    def current_waypoint_index(self) -> int:
        return self._idx

    @property
    def done(self) -> bool:
        """True when the vehicle has reached the final waypoint."""
        if not self._wpts:
            return True
        return self._idx >= len(self._wpts) - 1

    def update(self, vehicle: VehicleState, dt: float) -> jnp.ndarray:
        """Compute motor throttle. Returns (4,) throttle [0, 1]."""
        if not self._wpts:
            return jnp.zeros(4)

        wpt = self._wpts[self._idx]
        setpoint = wpt[:3]
        desired_yaw = float(wpt[3]) if wpt.shape[0] >= 4 else 0.0

        dist = jnp.linalg.norm(setpoint - vehicle.position)
        if float(dist) < self.radius and self._idx < len(self._wpts) - 1:
            self._idx += 1

        self._hover_state, throttle = hover_step(
            self._hover_state, vehicle, setpoint,
            self.params, self.gains, dt, desired_yaw)
        return throttle
