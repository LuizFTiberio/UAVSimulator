"""Mission guidance / reference layer for the tail-sitter (Phase C, Sec. 5).

Turns a sequence of high-level mission legs -- hold a hover, accelerate to a
cruise speed, cruise (optionally turning), decelerate back to hover -- into the
time-varying reference the INDI controller consumes each step:

    (position, velocity, acceleration, yaw, yaw_rate_ff)

The whole point of Phase C (and the paper's core claim) is that the transition
hover<->cruise is NOT a mode switch: the guidance just commands a continuously
changing velocity reference and the one INDI controller
(TailsitterINDIController) tracks whatever attitude/thrust that requires. There
is no "transition mode" here or in the controller -- only a smoothly varying
reference.

Design
------
A mission is a list of MissionLeg. Each leg carries a duration, a *target*
forward ground speed to reach by the end of the leg (linearly ramped from the
speed at the leg's start -- i.e. the previous leg's end speed), and a constant
heading turn rate for the leg. That small vocabulary expresses the whole
roadmap mission:

    hover-hold        -> target_speed 0, turn_rate 0
    transition out    -> target_speed V, turn_rate 0   (accelerate)
    cruise straight   -> target_speed V, turn_rate 0   (hold V)
    cruise + turn     -> target_speed V, turn_rate r   (coordinated turn)
    transition back   -> target_speed 0, turn_rate 0   (decelerate)

The horizontal velocity reference is  v = s(t) * [cos(psi), sin(psi), 0]  with
heading psi integrated from the leg turn rate, so the acceleration reference is
analytic:

    a = s_dot * heading_dir + s * psi_dot * left_dir

i.e. the tangential (accel/decel) term plus the centripetal turn term. The
turn rate is also handed to the controller as `yaw_rate_ff` (the coordinated-
turn feedforward, Phase B / Eq. 17), so the nose follows the turning velocity
and the vehicle holds ~zero sideslip through the turn rather than crabbing.

Altitude is held at a fixed `altitude`; this first-pass guidance flies level
transitions and turns (vertical reference is a constant -- the outer position
loop holds it). Position is integrated from the velocity reference, so the
controller's position term corrects drift rather than the guidance having to
close closed-form path integrals through the turn.

The guidance is stateful (it integrates psi and the reference position, exactly
as the closed-loop tests did inline): call reset(), then step(dt) once per
control step to advance and read the current reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


@dataclass
class MissionLeg:
    """One mission segment.

    duration     : leg length [s].
    target_speed : forward ground speed [m/s] to reach by the END of the leg,
                   linearly ramped from the speed at the leg's start (the
                   previous leg's end speed; 0 for the first leg). Hold a speed
                   by giving two consecutive legs the same target_speed.
    turn_rate    : heading rate [rad/s] held constant over the leg (coordinated
                   turn; 0 = straight). Positive = turn left (+z, right-hand).
    name         : optional label for logging/plots (e.g. "transition-out").
    """

    duration: float
    target_speed: float
    turn_rate: float = 0.0
    name: str = ""


class Reference(NamedTuple):
    """The reference handed to TailsitterINDIController.update() each step."""

    position: np.ndarray        # (3,) world position setpoint
    velocity: np.ndarray        # (3,) world velocity reference
    acceleration: np.ndarray    # (3,) world acceleration feedforward
    yaw: float                  # heading reference [rad]
    yaw_rate_ff: float          # coordinated-turn feedforward [rad/s]
    speed: float                # commanded forward speed [m/s] (for logging)
    leg_index: int              # active leg (for logging / phase annotation)
    leg_name: str               # active leg name (for logging)


@dataclass
class MissionGuidance:
    """Leg-based guidance generating a continuous reference from a mission.

    Parameters
    ----------
    legs : mission legs, flown in order.
    altitude : constant reference altitude [m].
    start_xy : initial reference (x, y) [m].
    start_yaw : initial heading [rad].
    """

    legs: list[MissionLeg]
    altitude: float = 2.0
    start_xy: tuple[float, float] = (0.0, 0.0)
    start_yaw: float = 0.0

    # integrated reference state
    _t: float = field(default=0.0, init=False)
    _psi: float = field(default=0.0, init=False)
    _pos: np.ndarray = field(default=None, init=False)
    _leg_start_speed: list[float] = field(default=None, init=False)
    _leg_start_time: list[float] = field(default=None, init=False)

    def __post_init__(self):
        # Precompute each leg's start speed (previous leg's end speed) and the
        # cumulative time at which each leg begins.
        starts, t0, prev_speed = [], [], 0.0
        acc = 0.0
        for leg in self.legs:
            starts.append(prev_speed)
            t0.append(acc)
            acc += leg.duration
            prev_speed = leg.target_speed
        self._leg_start_speed = starts
        self._leg_start_time = t0
        self._total_time = acc
        self.reset()

    @property
    def total_time(self) -> float:
        return self._total_time

    def reset(self) -> None:
        self._t = 0.0
        self._psi = float(self.start_yaw)
        self._pos = np.array([self.start_xy[0], self.start_xy[1], self.altitude],
                             dtype=float)

    def _active_leg(self, t: float) -> int:
        """Index of the leg active at time t (clamped to the last leg)."""
        for i in range(len(self.legs) - 1, -1, -1):
            if t >= self._leg_start_time[i] - 1e-12:
                return i
        return 0

    def _speed_and_rate(self, t: float) -> tuple[float, float, float, int]:
        """Return (speed, speed_dot, turn_rate, leg_index) at time t."""
        t = min(t, self._total_time)
        i = self._active_leg(t)
        leg = self.legs[i]
        s0 = self._leg_start_speed[i]
        tau = t - self._leg_start_time[i]
        if leg.duration > 0:
            s_dot = (leg.target_speed - s0) / leg.duration
        else:
            s_dot = 0.0
        speed = s0 + s_dot * min(tau, leg.duration)
        # once past the leg (only happens at the very end / clamp) hold target
        if tau >= leg.duration:
            speed, s_dot = leg.target_speed, 0.0
        return speed, s_dot, leg.turn_rate, i

    def step(self, dt: float) -> Reference:
        """Advance the reference by dt and return it (call once per control step)."""
        speed, s_dot, turn_rate, i = self._speed_and_rate(self._t)

        heading_dir = np.array([np.cos(self._psi), np.sin(self._psi), 0.0])
        left_dir = np.array([-np.sin(self._psi), np.cos(self._psi), 0.0])

        vel = speed * heading_dir
        acc = s_dot * heading_dir + speed * turn_rate * left_dir

        ref = Reference(
            position=self._pos.copy(),
            velocity=vel,
            acceleration=acc,
            yaw=float(self._psi),
            yaw_rate_ff=float(turn_rate),
            speed=float(speed),
            leg_index=i,
            leg_name=self.legs[i].name,
        )

        # integrate for the next step (semi-implicit: report the pre-step
        # position, then advance)
        self._pos = self._pos + vel * dt
        self._psi = self._psi + turn_rate * dt
        self._t += dt
        return ref

    @property
    def done(self) -> bool:
        return self._t >= self._total_time


def roadmap_mission(cruise_speed: float = 12.0, altitude: float = 10.0,
                    turn_rate: float = 0.15) -> list[MissionLeg]:
    """The roadmap Phase C mission (Sec. 156-198): hover -> transition ->
    cruise-with-turn -> transition -> hover, as a list of legs.

    A ~180 deg course reversal in the cruise leg exercises the coordinated-turn
    / sideslip controller (Phase B). Durations are sized so the transitions are
    brisk but the vehicle settles between phases.

    The default cruise speed is 12 m/s: below ~10 m/s the wing cannot support the
    weight (the props must carry it regardless of the controller), so a genuinely
    wing-borne cruise -- where the Guidance INDI outer loop pitches over and
    unloads the weight onto the wing -- needs >=12 m/s. Altitude default is 10 m
    to give the transitions vertical room.
    """
    turn_duration = float(np.pi / turn_rate)   # half-circle course reversal
    return [
        MissionLeg(2.0, 0.0, 0.0, "hover-start"),
        MissionLeg(5.0, cruise_speed, 0.0, "transition-out"),
        MissionLeg(2.0, cruise_speed, 0.0, "cruise-straight"),
        MissionLeg(turn_duration, cruise_speed, turn_rate, "cruise-turn"),
        MissionLeg(2.0, cruise_speed, 0.0, "cruise-straight-2"),
        MissionLeg(5.0, 0.0, 0.0, "transition-in"),
        MissionLeg(3.0, 0.0, 0.0, "hover-end"),
    ]
