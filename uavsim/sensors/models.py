"""Sensor noise and dynamics models.

Provides configurable sensor models for:
- IMU (accelerometer + gyroscope): bias random walk, white noise, scale errors
- GPS / position: white noise, latency, dropout
- Barometer / altimeter: bias drift, white noise
- Magnetometer: hard/soft iron distortion, white noise

All models accept a NumPy RNG for reproducibility and domain randomization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from uavsim.core.types import VehicleState
from uavsim.core.math import quat_to_rotation_matrix


# ── individual sensor configs ────────────────────────────────────────────────


@dataclass
class IMUConfig:
    """IMU noise and bias parameters.

    Accelerometer
    -------------
    accel_noise_std : float     White noise std dev [m/s²].  Typical MEMS: 0.01–0.1
    accel_bias_std : float      Bias random-walk std dev [m/s²/√s].
    accel_scale_error : float   Multiplicative scale error [fraction]. 0 = perfect.

    Gyroscope
    ---------
    gyro_noise_std : float      White noise std dev [rad/s].  Typical MEMS: 0.001–0.01
    gyro_bias_std : float       Bias random-walk std dev [rad/s/√s].
    gyro_scale_error : float    Multiplicative scale error [fraction]. 0 = perfect.

    rate_hz : float             Sensor update rate [Hz]. 0 = every sim step.
    """
    accel_noise_std: float = 0.05
    accel_bias_std: float = 0.001
    accel_scale_error: float = 0.0
    gyro_noise_std: float = 0.005
    gyro_bias_std: float = 0.0005
    gyro_scale_error: float = 0.0
    rate_hz: float = 0.0


@dataclass
class GPSConfig:
    """GPS / position sensor noise parameters.

    pos_noise_std : float       Position white noise std dev [m].
    vel_noise_std : float       Velocity white noise std dev [m/s].
    latency_s : float           Measurement latency [s].  0 = no latency.
    dropout_prob : float        Probability of missed reading per step [0, 1).
    rate_hz : float             Sensor update rate [Hz]. 0 = every sim step.
    """
    pos_noise_std: float = 0.3
    vel_noise_std: float = 0.1
    latency_s: float = 0.0
    dropout_prob: float = 0.0
    rate_hz: float = 0.0


@dataclass
class BarometerConfig:
    """Barometer / altimeter noise parameters.

    alt_noise_std : float       Altitude white noise std dev [m].
    bias_drift_std : float      Bias random-walk std dev [m/√s].
    rate_hz : float             Sensor update rate [Hz]. 0 = every sim step.
    """
    alt_noise_std: float = 0.1
    bias_drift_std: float = 0.005
    rate_hz: float = 0.0


@dataclass
class MagnetometerConfig:
    """Magnetometer noise parameters.

    noise_std : float           White noise std dev [μT].
    hard_iron_bias : tuple      Constant offset in body frame (x, y, z) [μT].
    soft_iron_scale : tuple     Axis scale factors (sx, sy, sz). (1,1,1) = no distortion.
    rate_hz : float             Sensor update rate [Hz]. 0 = every sim step.
    """
    noise_std: float = 0.5
    hard_iron_bias: tuple = (0.0, 0.0, 0.0)
    soft_iron_scale: tuple = (1.0, 1.0, 1.0)
    rate_hz: float = 0.0


# ── composite config ─────────────────────────────────────────────────────────


@dataclass
class SensorConfig:
    """Aggregated sensor configuration.

    Set any sub-config to ``None`` to disable that sensor channel.
    """
    imu: IMUConfig | None = field(default_factory=IMUConfig)
    gps: GPSConfig | None = field(default_factory=GPSConfig)
    barometer: BarometerConfig | None = field(default_factory=BarometerConfig)
    magnetometer: MagnetometerConfig | None = field(default_factory=MagnetometerConfig)
    enabled: bool = True


# ── sensor state (mutable, tracks bias drift + latency buffers) ──────────────


class SensorState:
    """Internal mutable state for sensor dynamics (bias walks, buffers)."""

    def __init__(self, config: SensorConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

        # IMU bias state
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

        # Barometer bias state
        self.baro_bias = 0.0

        # GPS latency buffer: stores (time, pos, vel) tuples
        self._gps_buffer: list[tuple[float, np.ndarray, np.ndarray]] = []
        self._last_gps_pos: np.ndarray | None = None
        self._last_gps_vel: np.ndarray | None = None

        # Rate-limiting: last measurement time per sensor
        self._last_imu_time = -np.inf
        self._last_gps_time = -np.inf
        self._last_baro_time = -np.inf
        self._last_mag_time = -np.inf

    def reset(self) -> None:
        """Reset all internal state."""
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.baro_bias = 0.0
        self._gps_buffer.clear()
        self._last_gps_pos = None
        self._last_gps_vel = None
        self._last_imu_time = -np.inf
        self._last_gps_time = -np.inf
        self._last_baro_time = -np.inf
        self._last_mag_time = -np.inf


# ── measurement containers ───────────────────────────────────────────────────


class IMUMeasurement(NamedTuple):
    """Accelerometer and gyroscope readings."""
    accel: np.ndarray       # (3,) body frame [m/s²]
    gyro: np.ndarray        # (3,) body frame [rad/s]
    valid: bool             # False if rate-limited (stale data)


class GPSMeasurement(NamedTuple):
    """GPS position and velocity readings."""
    position: np.ndarray    # (3,) world frame [m]
    velocity: np.ndarray    # (3,) world frame [m/s]
    valid: bool             # False if dropout or rate-limited


class BarometerMeasurement(NamedTuple):
    """Barometric altitude reading."""
    altitude: float         # [m]
    valid: bool


class MagnetometerMeasurement(NamedTuple):
    """Magnetometer reading."""
    field: np.ndarray       # (3,) body frame [μT]
    valid: bool


# ── sensor suite (the main API) ─────────────────────────────────────────────


class SensorSuite:
    """Applies configurable sensor dynamics to ground-truth ``VehicleState``.

    Usage
    -----
    >>> suite = SensorSuite(SensorConfig(), seed=42)
    >>> state = sim.get_state()
    >>> noisy_obs = suite.observe(state, dt=0.001)

    The returned observation dict can be converted to a flat array via
    ``suite.to_flat_obs(noisy_obs)`` which mirrors the original 13-element
    observation vector: [pos(3), vel(3), quat(4), ang_vel(3)].
    """

    # Earth magnetic field reference (NED, approximate mid-latitude) [μT]
    MAG_FIELD_NED = np.array([22.0, 0.0, 42.0])

    def __init__(self, config: SensorConfig | None = None, seed: int = 0):
        self.config = config or SensorConfig()
        self.rng = np.random.default_rng(seed)
        self.state = SensorState(self.config, self.rng)

    def reset(self, seed: int | None = None) -> None:
        """Reset sensor state. Optionally reseed."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.state.rng = self.rng
        self.state.reset()

    def randomize(self, factor: float = 1.0) -> None:
        """Domain-randomize sensor parameters in-place.

        Scales each noise parameter by a random factor drawn from
        ``Uniform(1/factor, factor)`` and re-randomizes biases.
        Useful for training robust RL policies.

        Parameters
        ----------
        factor : float
            Scaling range. E.g. ``factor=2.0`` means each parameter is
            randomly multiplied by something in [0.5, 2.0].
        """
        def _rand_scale():
            return self.rng.uniform(1.0 / factor, factor)

        cfg = self.config
        if cfg.imu is not None:
            cfg.imu.accel_noise_std *= _rand_scale()
            cfg.imu.accel_bias_std *= _rand_scale()
            cfg.imu.gyro_noise_std *= _rand_scale()
            cfg.imu.gyro_bias_std *= _rand_scale()
            cfg.imu.accel_scale_error = self.rng.uniform(-0.02, 0.02) * factor
            cfg.imu.gyro_scale_error = self.rng.uniform(-0.02, 0.02) * factor

        if cfg.gps is not None:
            cfg.gps.pos_noise_std *= _rand_scale()
            cfg.gps.vel_noise_std *= _rand_scale()
            cfg.gps.latency_s = abs(cfg.gps.latency_s * _rand_scale())

        if cfg.barometer is not None:
            cfg.barometer.alt_noise_std *= _rand_scale()
            cfg.barometer.bias_drift_std *= _rand_scale()

        if cfg.magnetometer is not None:
            cfg.magnetometer.noise_std *= _rand_scale()

    # ── main API ─────────────────────────────────────────────────────────

    def observe(self, true_state: VehicleState, dt: float) -> dict:
        """Produce noisy sensor measurements from ground-truth state.

        Parameters
        ----------
        true_state : VehicleState
            Ground-truth state from the simulator.
        dt : float
            Simulation timestep [s], used for bias random walks.

        Returns
        -------
        dict with keys: ``imu``, ``gps``, ``baro``, ``mag`` (or ``None``
        for disabled channels).
        """
        if not self.config.enabled:
            # Pass-through: no noise
            return {
                "imu": IMUMeasurement(
                    accel=np.zeros(3), gyro=np.asarray(true_state.angular_velocity), valid=True),
                "gps": GPSMeasurement(
                    position=np.asarray(true_state.position),
                    velocity=np.asarray(true_state.velocity), valid=True),
                "baro": BarometerMeasurement(altitude=float(true_state.position[2]), valid=True),
                "mag": None,
            }

        t = true_state.time

        return {
            "imu": self._measure_imu(true_state, dt, t),
            "gps": self._measure_gps(true_state, dt, t),
            "baro": self._measure_baro(true_state, dt, t),
            "mag": self._measure_mag(true_state, t),
        }

    def to_flat_obs(self, measurements: dict, true_state: VehicleState) -> np.ndarray:
        """Convert sensor measurements to a flat 13-element observation.

        Falls back to ground-truth for disabled / invalid channels so the
        observation shape stays consistent.

        Layout: [position(3), velocity(3), quaternion(4), angular_velocity(3)].
        """
        # Position: prefer GPS, fall back to ground truth
        gps = measurements.get("gps")
        if gps is not None and gps.valid:
            pos = gps.position
            vel = gps.velocity
        else:
            pos = np.asarray(true_state.position)
            vel = np.asarray(true_state.velocity)

        # Quaternion: kept from ground truth (would need an estimator to fuse)
        quat = np.asarray(true_state.quaternion)

        # Angular velocity: from gyro if available
        imu = measurements.get("imu")
        if imu is not None and imu.valid:
            ang_vel = imu.gyro
        else:
            ang_vel = np.asarray(true_state.angular_velocity)

        return np.concatenate([pos, vel, quat, ang_vel]).astype(np.float32)

    # ── private sensor models ────────────────────────────────────────────

    def _should_update(self, rate_hz: float, t: float, last_t: float) -> bool:
        """Check if enough time has elapsed for a rate-limited sensor."""
        if rate_hz <= 0:
            return True
        period = 1.0 / rate_hz
        return (t - last_t) >= period

    def _measure_imu(
        self, state: VehicleState, dt: float, t: float,
    ) -> IMUMeasurement | None:
        cfg = self.config.imu
        if cfg is None:
            return None

        s = self.state
        if not self._should_update(cfg.rate_hz, t, s._last_imu_time):
            # Return stale measurement
            return IMUMeasurement(accel=np.zeros(3), gyro=np.asarray(state.angular_velocity), valid=False)
        s._last_imu_time = t

        # -- Accelerometer --
        # True specific force in world frame: acceleration - gravity
        # For simplicity we use velocity difference approximation;
        # in practice the IMU measures specific force in body frame.
        # We rotate gravity into body frame and add noise.
        R = np.asarray(quat_to_rotation_matrix(state.quaternion))  # body -> world
        R_bw = R.T  # world -> body

        gravity_world = np.array([0.0, 0.0, -9.81])
        # Specific force in body frame (what an accel would measure in hover)
        # In hover: accel reads +g in z-body (thrust = mg)
        accel_true_body = R_bw @ (-gravity_world)  # simplified hover model

        # Bias random walk
        s.accel_bias += self.rng.normal(0, cfg.accel_bias_std * np.sqrt(dt), size=3)

        # Scale error + bias + white noise
        accel = accel_true_body * (1.0 + cfg.accel_scale_error) + s.accel_bias
        accel += self.rng.normal(0, cfg.accel_noise_std, size=3)

        # -- Gyroscope --
        gyro_true = np.asarray(state.angular_velocity)

        # Bias random walk
        s.gyro_bias += self.rng.normal(0, cfg.gyro_bias_std * np.sqrt(dt), size=3)

        gyro = gyro_true * (1.0 + cfg.gyro_scale_error) + s.gyro_bias
        gyro += self.rng.normal(0, cfg.gyro_noise_std, size=3)

        return IMUMeasurement(accel=accel, gyro=gyro, valid=True)

    def _measure_gps(
        self, state: VehicleState, dt: float, t: float,
    ) -> GPSMeasurement | None:
        cfg = self.config.gps
        if cfg is None:
            return None

        s = self.state

        if not self._should_update(cfg.rate_hz, t, s._last_gps_time):
            # Stale
            pos = s._last_gps_pos if s._last_gps_pos is not None else np.asarray(state.position)
            vel = s._last_gps_vel if s._last_gps_vel is not None else np.asarray(state.velocity)
            return GPSMeasurement(position=pos, velocity=vel, valid=False)
        s._last_gps_time = t

        # Dropout
        if self.rng.random() < cfg.dropout_prob:
            pos = s._last_gps_pos if s._last_gps_pos is not None else np.asarray(state.position)
            vel = s._last_gps_vel if s._last_gps_vel is not None else np.asarray(state.velocity)
            return GPSMeasurement(position=pos, velocity=vel, valid=False)

        # White noise
        pos = np.asarray(state.position) + self.rng.normal(0, cfg.pos_noise_std, size=3)
        vel = np.asarray(state.velocity) + self.rng.normal(0, cfg.vel_noise_std, size=3)

        # Latency: buffer and return delayed measurement
        if cfg.latency_s > 0:
            s._gps_buffer.append((t, pos.copy(), vel.copy()))
            # Find the most recent measurement that's old enough
            cutoff = t - cfg.latency_s
            delayed_pos, delayed_vel = pos, vel
            while len(s._gps_buffer) > 1 and s._gps_buffer[0][0] <= cutoff:
                _, delayed_pos, delayed_vel = s._gps_buffer.pop(0)
            pos, vel = delayed_pos, delayed_vel

        s._last_gps_pos = pos
        s._last_gps_vel = vel

        return GPSMeasurement(position=pos, velocity=vel, valid=True)

    def _measure_baro(
        self, state: VehicleState, dt: float, t: float,
    ) -> BarometerMeasurement | None:
        cfg = self.config.barometer
        if cfg is None:
            return None

        s = self.state
        if not self._should_update(cfg.rate_hz, t, s._last_baro_time):
            return BarometerMeasurement(altitude=float(state.position[2]), valid=False)
        s._last_baro_time = t

        # Bias drift
        s.baro_bias += self.rng.normal(0, cfg.bias_drift_std * np.sqrt(dt))

        alt = float(state.position[2]) + s.baro_bias
        alt += self.rng.normal(0, cfg.alt_noise_std)

        return BarometerMeasurement(altitude=alt, valid=True)

    def _measure_mag(
        self, state: VehicleState, t: float,
    ) -> MagnetometerMeasurement | None:
        cfg = self.config.magnetometer
        if cfg is None:
            return None

        s = self.state
        if not self._should_update(cfg.rate_hz, t, s._last_mag_time):
            return MagnetometerMeasurement(field=np.zeros(3), valid=False)
        s._last_mag_time = t

        # Rotate Earth field into body frame
        R = np.asarray(quat_to_rotation_matrix(state.quaternion))
        R_bw = R.T
        field_body = R_bw @ self.MAG_FIELD_NED

        # Soft iron distortion
        scale = np.array(cfg.soft_iron_scale)
        field_body = field_body * scale

        # Hard iron bias
        field_body += np.array(cfg.hard_iron_bias)

        # White noise
        field_body += self.rng.normal(0, cfg.noise_std, size=3)

        return MagnetometerMeasurement(field=field_body, valid=True)


# ── factory helpers ──────────────────────────────────────────────────────────


def default_sensor_suite(seed: int = 0) -> SensorSuite:
    """Create a sensor suite with conservative defaults (light noise)."""
    return SensorSuite(SensorConfig(), seed=seed)


def noisy_sensor_suite(seed: int = 0) -> SensorSuite:
    """Create a sensor suite tuned for aggressive sim-to-real training.

    Higher noise, GPS latency, moderate dropout, and bias drift.
    """
    config = SensorConfig(
        imu=IMUConfig(
            accel_noise_std=0.1,
            accel_bias_std=0.005,
            accel_scale_error=0.01,
            gyro_noise_std=0.01,
            gyro_bias_std=0.002,
            gyro_scale_error=0.005,
        ),
        gps=GPSConfig(
            pos_noise_std=0.5,
            vel_noise_std=0.2,
            latency_s=0.1,
            dropout_prob=0.05,
        ),
        barometer=BarometerConfig(
            alt_noise_std=0.3,
            bias_drift_std=0.02,
        ),
        magnetometer=MagnetometerConfig(
            noise_std=1.5,
            hard_iron_bias=(5.0, -3.0, 2.0),
            soft_iron_scale=(1.02, 0.98, 1.01),
        ),
    )
    return SensorSuite(config, seed=seed)
