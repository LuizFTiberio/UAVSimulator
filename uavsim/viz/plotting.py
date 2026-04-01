"""Post-flight telemetry plotting."""

from __future__ import annotations

import numpy as np

from uavsim.core.types import VehicleState


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
