"""Quadplane waypoint-following controller (7-channel) with state machine.

Channels: [fl, fr, bl, br, pusher, flap, aileron].

Five discrete flight phases:

**CLIMB**       Hover to target altitude, stabilise, yaw toward waypoint.
**TRANSITION**  Ramp pusher while hover still has full XY authority.
                Wait until forward airspeed exceeds a threshold.
**CRUISE**      Wing-borne flight.  Hover tracks altitude only.
                Ailerons damp sideslip + cross-track error.
**DECEL**       Approaching final waypoint — ramp pusher down, gradually
                restore hover XY authority.
**HOVER_HOLD**  Full hover position hold at the final waypoint.
"""

import enum

import jax.numpy as jnp
import numpy as np

from uavsim.core.types import (
    HoverGains, MultirotorParams, PusherParams, QuadplaneParams, VehicleState,
)
from uavsim.core.math import euler_from_quaternion, quat_to_rotation_matrix, wrap_angle
from uavsim.controllers.hover import default_hover_gains, hover_init, hover_step


class Phase(enum.Enum):
    CLIMB = "CLIMB"
    TRANSITION = "TRANSITION"
    CRUISE = "CRUISE"
    DECEL = "DECEL"
    HOVER_HOLD = "HOVER_HOLD"
    LAND = "LAND"
    LANDED = "LANDED"


class QuadplaneTrajectoryController:
    """Waypoint-following controller for a quadplane.

    Outputs (7,) commands: [fl, fr, bl, br, pusher, flap, aileron].

    Parameters
    ----------
    params : QuadplaneParams
    gains : HoverGains | None
        Hover controller gains (for the quad rotors).
    cruise_speed : float
        Target forward airspeed in cruise [m/s].
    acceptance_radius : float
        Distance to waypoint to advance to the next one [m].
    decel_radius : float
        Distance at which decel phase begins for final waypoint [m].
    kp_speed : float
        Proportional gain for pusher speed controller.
    kp_xtrack : float
        Proportional gain for aileron cross-track correction.
    kp_sideslip : float
        Proportional gain for aileron sideslip damping.
    transition_speed : float
        Forward airspeed threshold to enter cruise [m/s].
    climb_settle_time : float
        Seconds to hold stable at altitude before transitioning.
    """

    def __init__(
        self,
        params: QuadplaneParams,
        gains: HoverGains | None = None,
        cruise_speed: float = 12.0,
        acceptance_radius: float = 3.0,
        decel_radius: float = 15.0,
        kp_speed: float = 0.15,
        kp_xtrack: float = 0.15,
        kp_sideslip: float = 0.3,
        transition_speed: float | None = None,
        climb_settle_time: float = 2.0,
    ):
        self.params = params
        self.gains = gains if gains is not None else default_hover_gains()
        self.cruise_speed = cruise_speed
        self.radius = acceptance_radius
        self.decel_radius = decel_radius
        self.kp_speed = kp_speed
        self.kp_xtrack = kp_xtrack
        self.kp_sideslip = kp_sideslip
        # Enter cruise once airspeed exceeds wing transition speed
        self.transition_speed = (transition_speed if transition_speed is not None
                                 else params.wing.transition_speed)
        self.climb_settle_time = climb_settle_time

        self._hover_state = hover_init(self.gains)
        self._wpts: list[jnp.ndarray] = []
        self._idx: int = 0
        self._phase = Phase.CLIMB
        self._settle_timer: float = 0.0
        # Smooth pusher ramp during transition (0→1 over time)
        self._transition_ramp: float = 0.0
        # Landing altitude target (decreases over time)
        self._land_alt_setpoint: float = 0.0

    def set_waypoints(self, waypoints: list | np.ndarray) -> None:
        """Set the waypoint list. Each entry: [x, y, z] or [x, y, z, yaw]."""
        self._wpts = [jnp.asarray(w, dtype=jnp.float32) for w in waypoints]
        self._idx = 0
        self._hover_state = hover_init(self.gains)
        self._phase = Phase.CLIMB
        self._settle_timer = 0.0
        self._transition_ramp = 0.0
        self._land_alt_setpoint = 0.0

    @property
    def current_waypoint_index(self) -> int:
        return self._idx

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def done(self) -> bool:
        if not self._wpts:
            return True
        return self._idx >= len(self._wpts) - 1 and self._phase == Phase.LANDED

    @property
    def _is_final_waypoint(self) -> bool:
        return self._idx >= len(self._wpts) - 1

    # ── helpers ──────────────────────────────────────────────────────────

    def _body_frame(self, vehicle: VehicleState):
        """Return R, v_body, v_forward."""
        R = quat_to_rotation_matrix(vehicle.quaternion)
        v_body = R.T @ vehicle.velocity
        return R, v_body, float(v_body[0])

    def _yaw_toward_waypoint(self, vehicle: VehicleState, setpoint):
        """Compute desired yaw pointing toward setpoint, and yaw error."""
        delta_xy = setpoint[:2] - vehicle.position[:2]
        dist_xy = float(jnp.linalg.norm(delta_xy))
        if dist_xy > 1.0:
            desired_yaw = float(jnp.arctan2(delta_xy[1], delta_xy[0]))
        else:
            desired_yaw = float(euler_from_quaternion(vehicle.quaternion)[2])
        current_yaw = float(euler_from_quaternion(vehicle.quaternion)[2])
        yaw_error = abs(float(wrap_angle(jnp.array(desired_yaw - current_yaw))))
        return desired_yaw, yaw_error

    def _estimate_wing_lift_N(self, vehicle: VehicleState, v_forward: float,
                              flap: float = 0.0) -> float:
        """Estimate vertical component of wing lift [N].

        Accounts for pitch angle (which changes effective AoA) and flap
        deflection, matching the aerodynamic model's sigmoid activation.
        """
        wing = self.params.wing
        V = max(v_forward, 0.0)

        # Sigmoid activation matching the aero model
        sigma = 1.0 / (1.0 + float(jnp.exp(
            -wing.transition_sharpness * (V - wing.transition_speed))))
        if sigma < 0.01:
            return 0.0

        # Pitch contributes to AoA: positive pitch = more lift
        pitch = float(euler_from_quaternion(vehicle.quaternion)[1])
        alpha = float(jnp.clip(pitch, -wing.alpha_max, wing.alpha_max))

        CL = wing.CL0 + wing.CLa * alpha
        # Flap contribution
        delta_f = max(0.0, min(flap, 1.0)) * wing.max_flap
        CL += wing.CL_delta_f * delta_f

        q = 0.5 * wing.rho * V ** 2
        lift = sigma * q * wing.wing_area * CL
        # Only count the vertical component (cos of pitch)
        return max(float(lift * jnp.cos(pitch)), 0.0)

    # ── phase logic ──────────────────────────────────────────────────────

    def _update_phase(self, vehicle: VehicleState, setpoint, dist, v_forward, yaw_error, dt):
        """Advance the state machine."""
        alt_error = abs(float(setpoint[2] - vehicle.position[2]))
        speed_total = float(jnp.linalg.norm(vehicle.velocity))
        # Horizontal speed (more reliable than body-frame during pitch changes)
        v_horiz = float(jnp.linalg.norm(vehicle.velocity[:2]))

        if self._phase == Phase.CLIMB:
            # Wait until at altitude, low velocity, yaw aligned
            if alt_error < 0.5 and speed_total < 1.0 and yaw_error < 0.3:
                self._settle_timer += dt
            else:
                self._settle_timer = 0.0
            if self._settle_timer >= self.climb_settle_time:
                # Advance to next waypoint if available
                if self._idx < len(self._wpts) - 1:
                    self._idx += 1
                    self._phase = Phase.TRANSITION
                    self._transition_ramp = 0.0
                else:
                    self._phase = Phase.HOVER_HOLD

        elif self._phase == Phase.TRANSITION:
            # Ramp pusher gradually (2 s ramp)
            self._transition_ramp = min(self._transition_ramp + dt / 2.0, 1.0)
            # Use horizontal speed for transition check (robust to pitch)
            if v_horiz >= self.transition_speed and yaw_error < 0.4:
                self._phase = Phase.CRUISE

        elif self._phase == Phase.CRUISE:
            # Check if approaching final waypoint → decel
            if self._is_final_waypoint and dist < self.decel_radius:
                self._phase = Phase.DECEL

        elif self._phase == Phase.DECEL:
            # Once slow enough, switch to hover hold
            if v_forward < 2.0 and dist < self.radius * 2.0:
                self._phase = Phase.HOVER_HOLD
                self._hover_state = hover_init(self.gains)
                self._settle_timer = 0.0

        elif self._phase == Phase.HOVER_HOLD:
            # Stabilise briefly at waypoint, then land
            if dist < self.radius * 2.0 and speed_total < 1.0:
                self._settle_timer += dt
            else:
                self._settle_timer = 0.0
            if self._settle_timer >= 2.0:
                self._phase = Phase.LAND
                self._land_alt_setpoint = float(vehicle.position[2])

        elif self._phase == Phase.LAND:
            # Touched down when altitude < min_alt and low velocity
            alt = float(vehicle.position[2])
            if alt < self.gains.min_alt + 0.05 and speed_total < 0.5:
                self._phase = Phase.LANDED

        # LANDED is terminal

    # ── main update ──────────────────────────────────────────────────────

    def update(self, vehicle: VehicleState, dt: float) -> jnp.ndarray:
        """Compute 7-channel command: [fl, fr, bl, br, pusher, flap, aileron]."""
        if not self._wpts:
            return jnp.zeros(7)

        wpt = self._wpts[self._idx]
        setpoint = wpt[:3]

        # ── waypoint advance (only during CRUISE for intermediates) ──────
        dist = float(jnp.linalg.norm(setpoint - vehicle.position))
        if (self._phase == Phase.CRUISE
                and dist < self.radius
                and self._idx < len(self._wpts) - 1):
            self._idx += 1
            setpoint = self._wpts[self._idx][:3]
            dist = float(jnp.linalg.norm(setpoint - vehicle.position))

        R, v_body, v_forward = self._body_frame(vehicle)

        # During CLIMB, yaw toward the NEXT waypoint so we're aligned
        # before transition.  Otherwise yaw toward current setpoint.
        if self._phase == Phase.CLIMB and self._idx < len(self._wpts) - 1:
            next_wpt = self._wpts[self._idx + 1][:3]
            desired_yaw, yaw_error = self._yaw_toward_waypoint(vehicle, next_wpt)
        else:
            desired_yaw, yaw_error = self._yaw_toward_waypoint(vehicle, setpoint)

        # ── advance state machine ────────────────────────────────────────
        self._update_phase(vehicle, setpoint, dist, v_forward, yaw_error, dt)

        # ── compute commands per phase ───────────────────────────────────
        if self._phase == Phase.CLIMB:
            return self._cmd_climb(vehicle, setpoint, desired_yaw, dt)

        elif self._phase == Phase.TRANSITION:
            return self._cmd_transition(vehicle, setpoint, desired_yaw,
                                        v_forward, yaw_error, dt)

        elif self._phase == Phase.CRUISE:
            return self._cmd_cruise(vehicle, setpoint, desired_yaw,
                                    R, v_body, v_forward, dist, yaw_error, dt)

        elif self._phase == Phase.DECEL:
            return self._cmd_decel(vehicle, setpoint, desired_yaw,
                                   R, v_body, v_forward, dist, yaw_error, dt)

        elif self._phase == Phase.HOVER_HOLD:
            return self._cmd_hover_hold(vehicle, setpoint, desired_yaw, dt)

        elif self._phase == Phase.LAND:
            return self._cmd_land(vehicle, setpoint, desired_yaw, dt)

        else:  # LANDED
            return jnp.zeros(7)

    # ── CLIMB: full hover, no pusher ─────────────────────────────────────

    def _cmd_climb(self, vehicle, setpoint, desired_yaw, dt):
        self._hover_state, quad = hover_step(
            self._hover_state, vehicle, setpoint,
            self.params.rotor, self.gains, dt, desired_yaw,
            xy_weight=1.0,
        )
        return jnp.concatenate([quad, jnp.array([0.0, 1.0, 0.0])])

    # ── TRANSITION: hover + gradual pusher ramp ──────────────────────────

    def _cmd_transition(self, vehicle, setpoint, desired_yaw,
                        v_forward, yaw_error, dt):
        # Fade XY weight as ramp increases — let the vehicle accelerate
        # instead of hover fighting the forward motion
        xy_weight = max(1.0 - self._transition_ramp * 0.8, 0.0)

        # Flaps retract as ramp increases
        flap_cmd = float(jnp.clip(1.0 - self._transition_ramp, 0.0, 1.0))

        # Estimate wing lift and tell hover about it
        wing_lift = self._estimate_wing_lift_N(vehicle, v_forward, flap=flap_cmd)

        self._hover_state, quad = hover_step(
            self._hover_state, vehicle, setpoint,
            self.params.rotor, self.gains, dt, desired_yaw,
            xy_weight=xy_weight,
            external_lift_N=wing_lift,
        )

        # Ramp pusher, gated on yaw alignment
        speed_target = self.cruise_speed * self._transition_ramp
        speed_error = speed_target - v_forward
        pusher_cmd = float(jnp.clip(self.kp_speed * speed_error, 0.0, 1.0))
        yaw_gate = float(jnp.clip(1.0 - (yaw_error - 0.2) / 0.4, 0.0, 1.0))
        pusher_cmd *= yaw_gate * self._transition_ramp

        return jnp.concatenate([quad, jnp.array([pusher_cmd, flap_cmd, 0.0])])

    # ── CRUISE: wing-borne, hover for altitude only ──────────────────────

    def _cmd_cruise(self, vehicle, setpoint, desired_yaw,
                    R, v_body, v_forward, dist, yaw_error, dt):
        # Aileron: sideslip + cross-track
        v_to_wpt_body = R.T @ (setpoint - vehicle.position)
        xtrack_err = float(v_to_wpt_body[1])
        sideslip = float(v_body[1])
        lateral_correction = (self.kp_xtrack * xtrack_err
                              + self.kp_sideslip * sideslip)
        aileron_cmd = float(jnp.clip(lateral_correction, -1.0, 1.0))

        # Roll override so quad rotors cooperate with aileron
        desired_roll = float(jnp.clip(
            lateral_correction,
            -float(self.gains.max_tilt), float(self.gains.max_tilt)))

        # Estimate wing lift (no flaps in cruise)
        wing_lift = self._estimate_wing_lift_N(vehicle, v_forward, flap=0.0)

        # Hover: altitude only (xy_weight=0), with roll override
        self._hover_state, quad = hover_step(
            self._hover_state, vehicle, setpoint,
            self.params.rotor, self.gains, dt, desired_yaw,
            xy_weight=0.0,
            desired_roll_override=desired_roll,
            external_lift_N=wing_lift,
        )

        # Pusher speed control
        speed_error = self.cruise_speed - v_forward
        pusher_cmd = float(jnp.clip(self.kp_speed * speed_error, 0.0, 1.0))

        # Flaps retracted in cruise
        flap_cmd = 0.0

        return jnp.concatenate([quad, jnp.array([pusher_cmd, flap_cmd, aileron_cmd])])

    # ── DECEL: slow down, restore hover authority ────────────────────────

    def _cmd_decel(self, vehicle, setpoint, desired_yaw,
                   R, v_body, v_forward, dist, yaw_error, dt):
        # Blend factor: 1.0 at decel_radius → 0.0 at acceptance radius
        decel_progress = float(jnp.clip(
            (dist - self.radius) / max(self.decel_radius - self.radius, 1.0),
            0.0, 1.0))
        # cruise_weight goes from 1→0 as we approach waypoint
        cruise_weight = decel_progress
        xy_weight = 1.0 - cruise_weight

        # Aileron with diminishing authority
        v_to_wpt_body = R.T @ (setpoint - vehicle.position)
        xtrack_err = float(v_to_wpt_body[1])
        sideslip = float(v_body[1])
        lateral_correction = (self.kp_xtrack * xtrack_err
                              + self.kp_sideslip * sideslip)
        aileron_cmd = float(jnp.clip(
            lateral_correction * cruise_weight, -1.0, 1.0))

        # Roll override blended with decel progress
        desired_roll = float(jnp.clip(
            lateral_correction * cruise_weight,
            -float(self.gains.max_tilt), float(self.gains.max_tilt)))
        roll_override = desired_roll if cruise_weight > 0.2 else None

        # Flaps deploy as speed drops
        flap_cmd = float(jnp.clip(1.0 - cruise_weight, 0.0, 1.0))

        # Estimate wing lift with current flap setting
        wing_lift = self._estimate_wing_lift_N(vehicle, v_forward, flap=flap_cmd)

        self._hover_state, quad = hover_step(
            self._hover_state, vehicle, setpoint,
            self.params.rotor, self.gains, dt, desired_yaw,
            xy_weight=xy_weight,
            desired_roll_override=roll_override,
            external_lift_N=wing_lift,
        )

        # Pusher ramps down with distance
        speed_target = self.cruise_speed * decel_progress
        speed_error = speed_target - v_forward
        pusher_cmd = float(jnp.clip(self.kp_speed * speed_error, 0.0, 1.0))

        # Flaps deploy as speed drops
        flap_cmd = float(jnp.clip(1.0 - cruise_weight, 0.0, 1.0))

        return jnp.concatenate([quad, jnp.array([pusher_cmd, flap_cmd, aileron_cmd])])

    # ── HOVER_HOLD: full hover at final waypoint ─────────────────────────

    def _cmd_hover_hold(self, vehicle, setpoint, desired_yaw, dt):
        self._hover_state, quad = hover_step(
            self._hover_state, vehicle, setpoint,
            self.params.rotor, self.gains, dt, desired_yaw,
            xy_weight=1.0,
        )
        return jnp.concatenate([quad, jnp.array([0.0, 1.0, 0.0])])

    # ── LAND: controlled descent to ground ─────────────────────────────

    def _cmd_land(self, vehicle, setpoint, desired_yaw, dt):
        # Persistent setpoint that keeps decreasing at ~1 m/s
        descent_rate = 1.0  # m/s
        self._land_alt_setpoint = max(self._land_alt_setpoint - descent_rate * dt, 0.0)
        land_setpoint = setpoint.at[2].set(jnp.float32(self._land_alt_setpoint))

        self._hover_state, quad = hover_step(
            self._hover_state, vehicle, land_setpoint,
            self.params.rotor, self.gains, dt, desired_yaw,
            xy_weight=1.0,
        )
        return jnp.concatenate([quad, jnp.array([0.0, 1.0, 0.0])])
