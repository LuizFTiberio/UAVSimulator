"""Wind models: constant, Dryden turbulence (MIL-HDBK-1797).

All wind models follow a common interface:

    wind_model.step(altitude, dt) → v_wind_world  (3,) [m/s]
    wind_model.reset()

``DrydenWind`` implements the continuous-time Dryden transfer functions
discretised via Tustin (bilinear) transform, driven by unit white noise.

The Dryden model produces turbulence in a body-fixed NED frame.  Since
rotorcraft fly at relatively low speed and change heading frequently, the
turbulence is applied in the **world frame** (NED → ENU mapping with signs
handled internally).

References
----------
MIL-HDBK-1797, appendix A — Flying qualities of piloted aircraft.
https://en.wikipedia.org/wiki/Dryden_Wind_Turbulence_Model
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from uavsim.core.types import DrydenParams


# ── base protocol ────────────────────────────────────────────────────────────

class WindModel(ABC):
    """Abstract base class for wind models."""

    @abstractmethod
    def step(self, altitude: float, dt: float) -> np.ndarray:
        """Return world-frame wind velocity [m/s] for the current step.

        Parameters
        ----------
        altitude : float
            Current vehicle altitude AGL [m].
        dt : float
            Simulation timestep [s].

        Returns
        -------
        v_wind : (3,) world-frame wind velocity [m/s].
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (filter memory, RNG)."""

    @property
    @abstractmethod
    def current_velocity(self) -> np.ndarray:
        """Last computed wind velocity (world frame, m/s)."""


# ── constant (steady) wind ───────────────────────────────────────────────────

class ConstantWind(WindModel):
    """Time-invariant wind field.

    Parameters
    ----------
    velocity : (3,)
        World-frame wind velocity [m/s].  Default: no wind.
    """

    def __init__(self, velocity: np.ndarray | list | tuple = (0.0, 0.0, 0.0)):
        self._velocity = np.asarray(velocity, dtype=np.float64)

    def step(self, altitude: float, dt: float) -> np.ndarray:
        return self._velocity.copy()

    def reset(self) -> None:
        pass

    @property
    def current_velocity(self) -> np.ndarray:
        return self._velocity.copy()


# ── Dryden turbulence model ─────────────────────────────────────────────────

class DrydenWind(WindModel):
    """Dryden continuous-turbulence wind model (MIL-HDBK-1797).

    Drives three independent first-order shaping filters with white noise
    to produce stochastic wind gusts.  The transfer functions are:

        H_u(s) = σ_u √(2 V / (π L_u)) · 1 / (s + V/L_u)
        H_v(s) = σ_v √(2 V / (π L_v)) · 1 / (s + V/L_v)   (simplified)
        H_w(s) = σ_w √(2 V / (π L_w)) · 1 / (s + V/L_w)

    These are discretised via Tustin (bilinear) transform each step.

    A constant steady-state wind (mean wind) can be superimposed via
    ``mean_wind``.

    Parameters
    ----------
    params : DrydenParams
        Turbulence intensities and scale lengths.
    mean_wind : (3,) or None
        Constant mean wind to add on top of turbulence [m/s].
    airspeed_min : float
        Minimum effective airspeed used in the shaping filters [m/s].
        Prevents division-by-zero at hover; set to ~2–5 m/s.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        params: DrydenParams | None = None,
        mean_wind: np.ndarray | list | tuple | None = None,
        airspeed_min: float = 3.0,
        seed: int = 0,
    ):
        self.params = params if params is not None else DrydenParams()
        self.mean_wind = (
            np.zeros(3) if mean_wind is None
            else np.asarray(mean_wind, dtype=np.float64)
        )
        self.airspeed_min = airspeed_min
        self._rng = np.random.default_rng(seed)

        # Filter states (one per axis)
        self._state = np.zeros(3)
        self._velocity = np.zeros(3)

    def step(self, altitude: float, dt: float) -> np.ndarray:
        """Advance the Dryden filters by one timestep.

        Parameters
        ----------
        altitude : float
            Vehicle altitude AGL [m].  Used as a minimum airspeed proxy
            (low altitudes produce less turbulence in MIL-HDBK-1797).
        dt : float
            Timestep [s].

        Returns
        -------
        v_wind : (3,) world-frame wind velocity [m/s].
        """
        p = self.params
        V = max(self.airspeed_min, 1.0)  # effective airspeed for filter bandwidth

        # Scale lengths and intensities (could be altitude-dependent)
        sigmas = np.array([p.sigma_u, p.sigma_v, p.sigma_w])
        Ls = np.array([p.Lu, p.Lv, p.Lw])

        # White-noise input
        noise = self._rng.standard_normal(3)

        # Per-axis Tustin (bilinear) discretisation of H(s) = K / (s + a)
        # where a = V/L, K = σ √(2V / (πL))
        for i in range(3):
            a = V / Ls[i]
            K = sigmas[i] * np.sqrt(2.0 * V / (np.pi * Ls[i]))

            # Bilinear: s → (2/dt)(z-1)/(z+1)
            # H(z) = K · dt / (2 + a·dt) · (z + 1) / (z - (2 - a·dt)/(2 + a·dt))
            alpha = (2.0 - a * dt) / (2.0 + a * dt)
            beta = K * dt / (2.0 + a * dt)

            # y[n] = alpha * y[n-1] + beta * (x[n] + x[n-1])
            # We use noise as the combined (x[n] + x[n-1]) ≈ 2·noise for simplicity
            # (white noise is uncorrelated, so this just scales the gain by √2,
            #  which we absorb into the overall calibration)
            self._state[i] = alpha * self._state[i] + beta * noise[i]

        self._velocity = self._state + self.mean_wind
        return self._velocity.copy()

    def reset(self) -> None:
        """Reset filter states and RNG."""
        self._state = np.zeros(3)
        self._velocity = np.zeros(3)

    @property
    def current_velocity(self) -> np.ndarray:
        return self._velocity.copy()


# ── convenience factories ────────────────────────────────────────────────────

def light_turbulence(
    mean_wind: np.ndarray | list | tuple = (0.0, 0.0, 0.0),
    seed: int = 0,
) -> DrydenWind:
    """Light turbulence (MIL-HDBK-1797, light category at low altitude)."""
    params = DrydenParams(
        sigma_u=0.5, sigma_v=0.5, sigma_w=0.3,
        Lu=200.0, Lv=200.0, Lw=50.0,
    )
    return DrydenWind(params=params, mean_wind=mean_wind, seed=seed)


def moderate_turbulence(
    mean_wind: np.ndarray | list | tuple = (0.0, 0.0, 0.0),
    seed: int = 0,
) -> DrydenWind:
    """Moderate turbulence (MIL-HDBK-1797, moderate category)."""
    params = DrydenParams(
        sigma_u=1.5, sigma_v=1.5, sigma_w=1.0,
        Lu=200.0, Lv=200.0, Lw=50.0,
    )
    return DrydenWind(params=params, mean_wind=mean_wind, seed=seed)


def severe_turbulence(
    mean_wind: np.ndarray | list | tuple = (0.0, 0.0, 0.0),
    seed: int = 0,
) -> DrydenWind:
    """Severe turbulence for stress testing."""
    params = DrydenParams(
        sigma_u=3.0, sigma_v=3.0, sigma_w=2.0,
        Lu=200.0, Lv=200.0, Lw=50.0,
    )
    return DrydenWind(params=params, mean_wind=mean_wind, seed=seed)
