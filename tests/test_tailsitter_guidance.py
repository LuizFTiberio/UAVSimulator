"""Tests for the tail-sitter mission guidance / reference layer (Phase C).

These are fast pure-numpy unit tests on the reference the guidance emits (no
sim). The closed-loop behaviour of the controller tracking this reference is
covered by tests/test_tailsitter_indi.py (transition + coordinated turn) and by
examples/tailsitter_mission_demo.py end-to-end.
"""

import numpy as np
import pytest

from uavsim.controllers.tailsitter_guidance import (
    MissionLeg,
    MissionGuidance,
    Reference,
    roadmap_mission,
)


def _roll_out(guid, dt=0.01):
    """Step a guidance through its whole mission, returning the reference list."""
    guid.reset()
    n = int(round(guid.total_time / dt))
    return [guid.step(dt) for _ in range(n)]


class TestLegSequencing:

    def test_total_time_is_sum_of_legs(self):
        legs = [MissionLeg(1.0, 0.0), MissionLeg(2.0, 5.0), MissionLeg(0.5, 5.0)]
        guid = MissionGuidance(legs)
        assert guid.total_time == pytest.approx(3.5)

    def test_leg_start_speed_chains_from_previous(self):
        """Each leg ramps from the previous leg's END speed, not from zero."""
        legs = [MissionLeg(1.0, 0.0), MissionLeg(1.0, 6.0), MissionLeg(1.0, 6.0),
                MissionLeg(1.0, 0.0)]
        guid = MissionGuidance(legs)
        assert guid._leg_start_speed == [0.0, 0.0, 6.0, 6.0]


class TestHoverLeg:

    def test_hover_holds_position_and_zero_velocity(self):
        guid = MissionGuidance([MissionLeg(2.0, 0.0)], altitude=3.0,
                               start_xy=(1.0, -2.0))
        refs = _roll_out(guid)
        for r in refs:
            assert np.allclose(r.velocity, 0.0)
            assert np.allclose(r.acceleration, 0.0)
            assert np.allclose(r.position, [1.0, -2.0, 3.0])
            assert r.yaw_rate_ff == 0.0


class TestAccelLeg:

    def test_speed_ramps_linearly_and_accel_is_along_heading(self):
        """A transition-out leg (0 -> V over T) gives constant tangential accel
        V/T along the heading and no lateral component (straight)."""
        V, T, dt = 6.0, 3.0, 0.01
        guid = MissionGuidance([MissionLeg(T, V)], start_yaw=0.0)
        refs = _roll_out(guid, dt)
        # midway speed ~ V/2, final speed ~ V
        assert refs[len(refs) // 2].speed == pytest.approx(V / 2, abs=0.1)
        assert refs[-1].speed == pytest.approx(V, abs=0.1)
        # acceleration along +x (heading 0), magnitude V/T, no lateral
        a = refs[len(refs) // 2].acceleration
        assert a[0] == pytest.approx(V / T, abs=1e-6)
        assert a[1] == pytest.approx(0.0, abs=1e-9)

    def test_position_integrates_velocity(self):
        """Reference position must be the integral of the reference velocity."""
        V, T, dt = 6.0, 3.0, 0.001
        guid = MissionGuidance([MissionLeg(T, V)], start_yaw=0.0, start_xy=(0.0, 0.0))
        refs = _roll_out(guid, dt)
        # analytic distance for a linear ramp 0->V over T: 0.5*V*T
        assert refs[-1].position[0] == pytest.approx(0.5 * V * T, rel=1e-2)
        assert refs[-1].position[1] == pytest.approx(0.0, abs=1e-6)


class TestCruiseTurn:

    def test_heading_integrates_turn_rate(self):
        rate, T, dt = 0.25, 4.0, 0.001
        guid = MissionGuidance([MissionLeg(0.0, 5.0)],  # seed speed... use two legs
                               start_yaw=0.0)
        # simpler: a single cruise leg at constant speed with a turn
        guid = MissionGuidance([MissionLeg(0.1, 5.0), MissionLeg(T, 5.0, rate)])
        refs = _roll_out(guid, dt)
        assert refs[-1].yaw == pytest.approx(rate * T, rel=1e-2)
        # yaw_rate_ff during the turn leg equals the commanded turn rate
        assert refs[-1].yaw_rate_ff == pytest.approx(rate)

    def test_turn_acceleration_is_centripetal(self):
        """During a constant-speed turn the accel feedforward is purely
        centripetal: magnitude s*rate, perpendicular to the velocity."""
        rate, s, dt = 0.25, 5.0, 0.001
        guid = MissionGuidance([MissionLeg(0.1, s), MissionLeg(4.0, s, rate)])
        refs = _roll_out(guid, dt)
        r = refs[len(refs) // 2]   # mid-turn
        a, v = r.acceleration, r.velocity
        assert np.linalg.norm(a) == pytest.approx(s * rate, rel=1e-2)
        assert abs(float(a @ v)) < 1e-3 * (np.linalg.norm(a) * np.linalg.norm(v) + 1e-9)


class TestCruiseStraightHoldsSpeed:

    def test_repeated_target_speed_holds_constant_velocity(self):
        """Two consecutive legs at the same target_speed = cruise: speed flat,
        near-zero tangential accel."""
        guid = MissionGuidance([MissionLeg(1.0, 6.0), MissionLeg(2.0, 6.0)])
        refs = _roll_out(guid)
        cruise = [r for r in refs if r.leg_index == 1]   # the hold leg
        assert len(cruise) > 0
        for r in cruise:
            assert r.speed == pytest.approx(6.0, abs=1e-6)
            assert np.allclose(r.acceleration, 0.0, atol=1e-6)


class TestRoadmapMission:

    def test_structure_has_all_phases(self):
        legs = roadmap_mission(cruise_speed=6.0, altitude=2.0, turn_rate=0.25)
        names = [l.name for l in legs]
        assert "hover-start" in names
        assert "transition-out" in names
        assert "cruise-turn" in names
        assert "transition-in" in names
        assert "hover-end" in names
        # a turn leg actually turns
        turn = next(l for l in legs if l.name == "cruise-turn")
        assert turn.turn_rate != 0.0
        # starts and ends at rest
        assert legs[0].target_speed == 0.0
        assert legs[-1].target_speed == 0.0

    def test_reference_returns_to_rest_at_end(self):
        guid = MissionGuidance(roadmap_mission(), altitude=2.0)
        refs = _roll_out(guid, dt=0.01)
        assert refs[-1].speed == pytest.approx(0.0, abs=1e-6)
        assert np.allclose(refs[-1].velocity, 0.0, atol=1e-6)
        assert refs[-1].position[2] == pytest.approx(2.0)
