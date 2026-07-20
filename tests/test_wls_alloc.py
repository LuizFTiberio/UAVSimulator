"""Tests for the active-set weighted least-squares allocator (wls_alloc)."""

import numpy as np
import pytest

from uavsim.controllers.wls_alloc import wls_alloc


class TestUnconstrained:

    def test_square_full_rank_matches_exact_solution(self):
        """When the exact solution is well inside the bounds, WLS must return
        it (the objective is fully achievable -> B u = v).

        WLS is regularised by the finite objective-vs-preference weight
        gamma_sq (it also minimises ||u - ud||), so it only recovers the pure
        objective solution up to O(1/gamma_sq); large gamma_sq -> exact."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            B = rng.normal(size=(4, 4))
            if abs(np.linalg.det(B)) < 1e-3:
                continue
            u_true = rng.uniform(-0.4, 0.4, size=4)
            v = B @ u_true
            u, it = wls_alloc(v, B, -np.ones(4), np.ones(4), u_guess=np.zeros(4),
                              gamma_sq=1e10)
            assert np.allclose(B @ u, v, atol=1e-6), f"objective not met ({it} it)"

    def test_matches_pinv_when_unconstrained(self):
        B = np.array([[1.0, 1.0, 0.5, 0.5],
                      [1.0, -1.0, 0.5, -0.5],
                      [0.0, 0.0, 1.0, -1.0],
                      [1.0, 1.0, 0.0, 0.0]])
        v = np.array([0.2, -0.1, 0.05, 0.3])
        # gamma_sq -> large makes the preference term negligible, so WLS -> pinv.
        u, _ = wls_alloc(v, B, -np.ones(4), np.ones(4), u_guess=np.zeros(4),
                         gamma_sq=1e10)
        u_pinv = np.linalg.pinv(B) @ v
        assert np.allclose(u, u_pinv, atol=1e-6)
        # and with the default (finite) gamma_sq, still close but regularised
        u_def, _ = wls_alloc(v, B, -np.ones(4), np.ones(4), u_guess=np.zeros(4))
        assert np.allclose(u_def, u_pinv, atol=1e-3)


class TestConstraints:

    def test_bounds_always_respected(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            B = rng.normal(size=(4, 4))
            v = rng.normal(size=4) * 5.0   # large -> forces saturation
            umin, umax = -np.ones(4), np.ones(4)
            u, _ = wls_alloc(v, B, umin, umax, u_guess=np.zeros(4))
            assert np.all(u >= umin - 1e-6) and np.all(u <= umax + 1e-6)


class TestPriority:

    def test_weight_favours_high_priority_axis(self):
        """Two directly conflicting objectives on a single actuator: obj0 wants
        u=+1, obj1 wants u=-1. The higher-Wv axis must win."""
        B = np.array([[1.0], [1.0]])
        v = np.array([1.0, -1.0])
        umin, umax = np.array([-1.0]), np.array([1.0])

        u_hi0, _ = wls_alloc(v, B, umin, umax, Wv=np.array([1000.0, 1.0]),
                             u_guess=np.zeros(1))
        u_hi1, _ = wls_alloc(v, B, umin, umax, Wv=np.array([1.0, 1000.0]),
                             u_guess=np.zeros(1))
        assert u_hi0[0] > 0.9, f"obj0 priority should push u->+1, got {u_hi0[0]}"
        assert u_hi1[0] < -0.9, f"obj1 priority should push u->-1, got {u_hi1[0]}"

    def test_pitch_priority_under_saturation(self):
        """Tail-sitter-shaped case: roll and pitch both want the elevons, which
        saturate. With the paper's weights (pitch >> roll) the pitch objective
        should be met more closely than roll when they compete."""
        # cols: [elevon_L, elevon_R]; row0 = roll (differential), row1 = pitch (common)
        B = np.array([[1.0, -1.0],
                      [1.0, 1.0]])
        v = np.array([1.5, 1.5])          # both demand more than bounds allow
        umin, umax = -np.ones(2), np.ones(2)
        Wv = np.array([100.0, 1000.0])    # roll, pitch (paper ordering)
        u, _ = wls_alloc(v, B, umin, umax, Wv=Wv, u_guess=np.zeros(2))
        roll_res = abs((B @ u)[0] - v[0])
        pitch_res = abs((B @ u)[1] - v[1])
        assert pitch_res < roll_res, (
            f"pitch (high priority) should have smaller residual: "
            f"pitch={pitch_res:.3f} roll={roll_res:.3f}")
        assert np.all(u >= umin - 1e-6) and np.all(u <= umax + 1e-6)


class TestConvergence:

    def test_terminates_within_imax(self):
        rng = np.random.default_rng(2)
        for _ in range(30):
            B = rng.normal(size=(4, 4))
            v = rng.normal(size=4) * 3.0
            u, it = wls_alloc(v, B, -np.ones(4), np.ones(4),
                              u_guess=np.zeros(4), imax=100)
            assert it < 100, "did not converge before imax"
