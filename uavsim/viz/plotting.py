"""Post-flight telemetry plotting."""

from __future__ import annotations

import numpy as np

from uavsim.core.types import VehicleState
from uavsim.core.math import quat_to_rotation_matrix


def plot_flight_data(
    states: list[VehicleState],
    times: list[float],
    setpoint: np.ndarray,
    disturbance_time: float | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Generate a 4-panel flight analysis plot.

    Parameters
    ----------
    states : list of VehicleState from sim.state_history
    times : list of float from sim.time_history
    setpoint : (3,) desired [x, y, z]
    disturbance_time : time of disturbance event (vertical line)
    save_path : if not None, save figure to this path
    show : if True, call plt.show()
    """
    import matplotlib.pyplot as plt

    positions = np.array([np.asarray(s.position) for s in states])
    velocities = np.array([np.asarray(s.velocity) for s in states])
    t = np.array(times)
    sp = np.asarray(setpoint)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("UAV Flight Data", fontsize=14)

    kw_dist = {}
    if disturbance_time is not None:
        kw_dist = dict(color="k", linestyle=":", alpha=0.45, linewidth=1.2,
                       label=f"Disturbance (t={disturbance_time}s)")

    # Position
    ax = axes[0, 0]
    for i, (label, color) in enumerate(
            zip(["X", "Y", "Z"], ["tab:red", "tab:green", "tab:blue"])):
        ax.plot(t, positions[:, i], color=color, lw=1.8, label=label)
        ax.axhline(sp[i], color=color, ls="--", alpha=0.3)
    if kw_dist:
        ax.axvline(disturbance_time, **kw_dist)
    ax.set(xlabel="Time (s)", ylabel="Position (m)", title="Position")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Velocity
    ax = axes[0, 1]
    for i, (label, color) in enumerate(
            zip(["Vx", "Vy", "Vz"], ["tab:red", "tab:green", "tab:blue"])):
        ax.plot(t, velocities[:, i], color=color, lw=1.8, label=label)
    ax.axhline(0, color="k", lw=0.8, alpha=0.2)
    if kw_dist:
        ax.axvline(disturbance_time, **kw_dist)
    ax.set(xlabel="Time (s)", ylabel="Velocity (m/s)", title="Velocity")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # XY trajectory
    ax = axes[1, 0]
    ax.plot(positions[:, 0], positions[:, 1], lw=1.8, label="Trajectory")
    ax.scatter(*positions[0, :2], s=80, color="green", zorder=5, label="Start")
    ax.scatter(*positions[-1, :2], s=80, color="red", zorder=5,
               marker="s", label="End")
    ax.scatter(sp[0], sp[1], s=120, color="blue", zorder=4,
               marker="*", label="Setpoint")
    ax.set(xlabel="X (m)", ylabel="Y (m)", title="XY trajectory")
    ax.axis("equal")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Altitude
    ax = axes[1, 1]
    ax.plot(t, positions[:, 2], lw=2, color="tab:blue", label="Altitude")
    ax.axhline(sp[2], color="tab:red", ls="--", lw=1.5, alpha=0.7,
               label=f"Setpoint ({sp[2]:.1f} m)")
    ax.fill_between(t, 0, positions[:, 2], alpha=0.15, color="tab:blue")
    if kw_dist:
        ax.axvline(disturbance_time, **kw_dist)
    ax.set(xlabel="Time (s)", ylabel="Altitude (m)", title="Altitude",
           ylim=(0, None))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved -> {save_path}")
    if show:
        plt.show()


def plot_tailsitter_mission(
    states: list[VehicleState],
    times: list[float],
    commands: np.ndarray,
    references: list | None = None,
    wind_velocity: np.ndarray | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Six-panel Phase-C mission analysis for the tail-sitter, mirroring the
    presentation in Smeur et al. (JGCD 2019) Fig. 16-19: trajectory, altitude,
    pitch angle, speed, control-surface saturation, and sideslip over a full
    hover -> transition -> cruise-with-turn -> transition -> hover mission.

    Parameters
    ----------
    states : list of VehicleState (sim.state_history).
    times : list of float (sim.time_history).
    commands : (N, 4) applied commands [thr_L, thr_R, elevon_L, elevon_R] --
        throttles in [0, 1], elevons in [-1, 1].
    references : optional list of guidance Reference (same length/order as
        states); used to overlay the speed reference and shade mission phases.
    wind_velocity : optional (3,) constant wind; subtracted for the airspeed /
        sideslip traces (defaults to still air).
    save_path, show : as plot_flight_data.
    """
    import matplotlib.pyplot as plt

    t = np.array(times)
    pos = np.array([np.asarray(s.position) for s in states])
    vel = np.array([np.asarray(s.velocity) for s in states])
    cmd = np.asarray(commands, dtype=float)
    wind = np.zeros(3) if wind_velocity is None else np.asarray(wind_velocity, float)

    # per-step body-frame airspeed -> pitch (angle of thrust axis from vertical),
    # horizontal airspeed, and sideslip beta
    pitch_deg = np.zeros(len(states))
    airspeed = np.zeros(len(states))
    beta_deg = np.full(len(states), np.nan)
    for i, s in enumerate(states):
        R = np.asarray(quat_to_rotation_matrix(s.quaternion))
        b1 = R[:, 0]
        pitch_deg[i] = np.degrees(np.arctan2(np.hypot(b1[0], b1[1]), b1[2]))
        v_air = np.asarray(s.velocity) - wind
        airspeed[i] = np.linalg.norm(v_air[:2])
        v_body = R.T @ v_air
        if v_body[0] > 2.0:   # sideslip only meaningful in forward flight
            beta_deg[i] = np.degrees(np.arctan2(v_body[1], v_body[0]))

    # phase shading + speed reference from the guidance references (optional)
    spans = []          # (t_start, t_end, name)
    speed_ref = None
    if references is not None and len(references) == len(states):
        speed_ref = np.array([r.speed for r in references])
        idx = np.array([r.leg_index for r in references])
        change = np.where(np.diff(idx) != 0)[0]
        bounds = [0, *(change + 1), len(idx)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            spans.append((t[a], t[min(b, len(t) - 1)], references[a].leg_name))

    phase_colors = plt.cm.Pastel1(np.linspace(0, 1, max(len(spans), 1)))

    def shade(ax):
        for (a, b, name), c in zip(spans, phase_colors):
            ax.axvspan(a, b, color=c, alpha=0.35, lw=0)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Tail-sitter mission: hover -> transition -> cruise+turn -> "
                 "transition -> hover", fontsize=14)

    # XY trajectory
    ax = axes[0, 0]
    ax.plot(pos[:, 0], pos[:, 1], lw=1.8, color="tab:blue")
    ax.scatter(*pos[0, :2], s=90, color="green", zorder=5, label="Start")
    ax.scatter(*pos[-1, :2], s=90, color="red", marker="s", zorder=5, label="End")
    ax.set(xlabel="X (m)", ylabel="Y (m)", title="Ground track")
    ax.axis("equal"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Altitude
    ax = axes[0, 1]; shade(ax)
    ax.plot(t, pos[:, 2], lw=1.8, color="tab:blue", label="Altitude")
    if references is not None and len(references) == len(states):
        ax.plot(t, [r.position[2] for r in references], "--", color="k",
                alpha=0.5, lw=1.2, label="Reference")
    ax.set(xlabel="Time (s)", ylabel="Altitude (m)", title="Altitude")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Pitch angle (thrust axis from vertical: 0deg = hover, 90deg = cruise)
    ax = axes[1, 0]; shade(ax)
    ax.plot(t, pitch_deg, lw=1.8, color="tab:purple")
    ax.axhline(0, color="k", lw=0.8, alpha=0.3)
    ax.axhline(90, color="k", ls=":", lw=0.8, alpha=0.3)
    ax.set(xlabel="Time (s)", ylabel="Pitch from vertical (deg)",
           title="Attitude (0 deg = hover, 90 deg = level cruise)")
    ax.grid(alpha=0.3)

    # Speed (with reference)
    ax = axes[1, 1]; shade(ax)
    ax.plot(t, airspeed, lw=1.8, color="tab:orange", label="Horizontal airspeed")
    if speed_ref is not None:
        ax.plot(t, speed_ref, "--", color="k", alpha=0.6, lw=1.2, label="Speed ref")
    ax.set(xlabel="Time (s)", ylabel="Speed (m/s)", title="Speed tracking")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Control-surface saturation
    ax = axes[2, 0]; shade(ax)
    ax.plot(t, cmd[:, 0], lw=1.4, color="tab:red", label="throttle L")
    ax.plot(t, cmd[:, 1], lw=1.4, color="tab:pink", label="throttle R")
    ax.plot(t, cmd[:, 2], lw=1.4, color="tab:green", label="elevon L")
    ax.plot(t, cmd[:, 3], lw=1.4, color="tab:olive", label="elevon R")
    for lim in (-1.0, 0.0, 1.0):
        ax.axhline(lim, color="k", lw=0.8, alpha=0.2)
    ax.set(xlabel="Time (s)", ylabel="Command", title="Actuators (saturate at [-1, 1])",
           ylim=(-1.15, 1.15))
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    # Sideslip
    ax = axes[2, 1]; shade(ax)
    ax.plot(t, beta_deg, lw=1.6, color="tab:brown")
    ax.axhline(0, color="k", lw=0.8, alpha=0.3)
    ax.set(xlabel="Time (s)", ylabel="Sideslip beta (deg)",
           title="Sideslip (NaN below forward-flight threshold)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved -> {save_path}")
    if show:
        plt.show()
