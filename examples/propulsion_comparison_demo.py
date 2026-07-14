"""Propulsion model comparison — SIMPLE vs BEM_LIVE vs BEM_TABLE.

Flies a quadcopter through 2 waypoints with an MPC controller that uses
the simple (kt/km) model internally.  The same controller plan is executed
against three different plant (physics) models so we can see how the model
mismatch affects trajectory tracking.

Runs headlessly — no interactive viewer needed.
Output: propulsion_comparison.png

Install CCBlade JAX first (one-time):
    pip install -e /path/to/CCBlade_jax
"""

import sys
import time
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ── CCBlade imports ───────────────────────────────────────────────────────────
try:
    from bem.core import BladeGeom, RotorParams, load_airfoil, solve_rotor_oblique
    import bem as _bem_pkg
    _CCBLADE_ROOT = Path(_bem_pkg.__file__).parent.parent
except ImportError as exc:
    sys.exit(
        f"CCBlade JAX not found: {exc}\n"
        "  Install with:  pip install -e /path/to/CCBlade_jax"
    )

# ── UAVSimulator imports ──────────────────────────────────────────────────────
from uavsim.vehicles.multirotor import quadcopter, quadcopter_params, multirotor_wrench
from uavsim.vehicles.base import VehicleModel
from uavsim.sim.mujoco_sim import MuJoCoSimulator
from uavsim.controllers.mpc import MPCController, default_mpc_config
from uavsim.dynamics.propulsion import PropulsionModel
from uavsim.dynamics.propulsion_bem import BEMConfig, build_bem_table

# ─────────────────────────────────────────────────────────────────────────────
# Mission
# ─────────────────────────────────────────────────────────────────────────────
WAYPOINTS = [
    [0.0, 0.0, 1.5],   # take-off
    [4.0, 0.0, 2.0],   # target 1 — fly east, climb
    [4.0, 4.0, 1.5],   # target 2 — fly north, descend
]
DURATION      = 28.0   # [s] — enough for both waypoints with MPC
ACCEPTANCE_R  = 0.4    # [m] — waypoint acceptance radius for progress display

# ─────────────────────────────────────────────────────────────────────────────
# BEM blade geometry (APC 10×5-like, NACA 4412)
# ─────────────────────────────────────────────────────────────────────────────
NACA4412 = _CCBLADE_ROOT / "data" / "naca4412.dat"
if not NACA4412.exists():
    sys.exit(f"naca4412.dat not found at {NACA4412}")

af   = load_airfoil(str(NACA4412))
Rhub = 0.0254                                          # 1-inch hub [m]
Rtip = 0.127                                           # 5-inch tip [m]
N_st = 10
r     = jnp.linspace(Rhub + 1e-3, Rtip - 1e-3, N_st)
chord = jnp.linspace(0.025, 0.012, N_st)
twist = jnp.linspace(40.0, 15.0, N_st) * jnp.pi / 180.0
geom  = BladeGeom(r=r, chord=chord, twist=twist)
rotor = RotorParams(Rhub=Rhub, Rtip=Rtip, B=2)


# ─────────────────────────────────────────────────────────────────────────────
# Hover-mass helper — one BEM solve at hover to find what this blade can lift
# ─────────────────────────────────────────────────────────────────────────────

def _bem_hover_mass(
    hover_omega: float = 419.0,   # 50 % of max_omega=838 rad/s
    n_motors: int = 4,
    motor_mass: float = 0.050,    # per-motor mass in MJCF [kg]
    g: float = 9.81,
    rho: float = 1.225,
) -> tuple[float, float]:
    """Single BEM solve at hover (V_inf=0, alpha=0) to find liftable mass.

    Returns
    -------
    mass_eff  : total vehicle mass the blade can hover [kg]
    body_mass : main-body mass for the MJCF (mass_eff - n_motors * motor_mass)
    """
    result = solve_rotor_oblique(
        0.0, 0.0, float(hover_omega),
        geom, af, rotor, rho,
    )
    T_hover   = float(result.T)
    mass_eff  = n_motors * T_hover / g
    body_mass = mass_eff - n_motors * motor_mass
    return mass_eff, body_mass


# ─────────────────────────────────────────────────────────────────────────────
# Build lookup table (covers the quadcopter's operating RPM range)
# ─────────────────────────────────────────────────────────────────────────────
def _build_table():
    print("  Building BEM lookup table (25 × 12 × 12 grid) …", flush=True)
    t0 = time.perf_counter()
    omega_g = jnp.linspace(50.0,  900.0,       25)
    vinf_g  = jnp.linspace(0.0,   15.0,        12)
    alpha_g = jnp.linspace(0.0,   jnp.pi / 2., 12)
    tbl = build_bem_table(geom, af, rotor, 1.225, omega_g, vinf_g, alpha_g)
    print(f"  Table built in {time.perf_counter() - t0:.1f} s")
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle factory
# ─────────────────────────────────────────────────────────────────────────────
def _make_vehicle(
    prop_model: PropulsionModel,
    mass: float,
    mjcf_xml: str,
    bem_cfg=None,
) -> VehicleModel:
    """Return a VehicleModel with the chosen propulsion plant and mass-adjusted XML.

    The MPC is always built from simple quadcopter_params — only the plant
    (MuJoCo physics via xfrc_applied) changes between modes.
    """
    base = quadcopter()
    base_params = quadcopter_params(mass=mass)

    if prop_model is PropulsionModel.SIMPLE:
        params = base_params
    else:
        params = base_params._replace(bem_config=bem_cfg)

    wrench_fn = partial(multirotor_wrench, propulsion_model=prop_model)

    return VehicleModel(
        params=params,
        mjcf_path=base.mjcf_path,
        compute_wrench=wrench_fn,
        actuator_names=base.actuator_names,
        spin_signs=base.spin_signs,
        mjcf_xml=mjcf_xml,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run one simulation, return history and wall-clock time
# ─────────────────────────────────────────────────────────────────────────────
def _run(
    vehicle: VehicleModel,
    label: str,
    ctrl_params,
) -> dict:
    sim  = MuJoCoSimulator(vehicle)
    ctrl = MPCController(ctrl_params,
                         config=default_mpc_config(ctrl_params),
                         acceptance_radius=ACCEPTANCE_R)
    ctrl.set_waypoints(WAYPOINTS)

    n_steps = int(DURATION / sim.dt)
    print_every = max(1, int(1.0 / sim.dt))
    wall_t0 = time.perf_counter()

    for step in range(n_steps):
        state = sim.get_state()
        cmd   = ctrl.update(state, sim.dt)
        sim.step(cmd)

        if step % print_every == 0:
            p   = np.asarray(state.position)
            wpt = ctrl.current_waypoint_index
            print(
                f"  [{label}] t={sim.current_time:5.1f}s  "
                f"pos=[{p[0]:5.2f},{p[1]:5.2f},{p[2]:5.2f}]  "
                f"wpt={wpt}/{len(WAYPOINTS)-1}"
            )

        if ctrl.done:
            break

    elapsed = time.perf_counter() - wall_t0
    rt_factor = sim.current_time / elapsed if elapsed > 0 else 0.0
    print(
        f"  [{label}] finished  sim_time={sim.current_time:.1f}s  "
        f"wall={elapsed:.1f}s  ({rt_factor:.1f}x realtime)"
    )

    return {
        "positions": np.array([np.asarray(s.position) for s in sim.state_history]),
        "velocities": np.array([np.asarray(s.velocity) for s in sim.state_history]),
        "times":      np.array(sim.time_history),
        "wall_s":     elapsed,
        "sim_s":      sim.current_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plot
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "SIMPLE":    ("tab:blue",   "-"),
    "BEM_LIVE":  ("tab:orange", "--"),
    "BEM_TABLE": ("tab:green",  ":"),
}


def _plot(results: dict, waypoints: list, save_path: str) -> None:
    wpts = np.array(waypoints)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Propulsion model comparison\n"
        "MPC controller (simple internal model) · same waypoints · different physics",
        fontsize=12,
    )

    ax_xy  = axes[0, 0]
    ax_z   = axes[0, 1]
    ax_spd = axes[1, 0]
    ax_bar = axes[1, 1]

    for name, data in results.items():
        pos   = data["positions"]
        vel   = data["velocities"]
        t     = data["times"]
        speed = np.linalg.norm(vel, axis=1)
        col, ls = COLORS[name]

        ax_xy.plot(pos[:, 0], pos[:, 1], color=col, ls=ls, lw=2.2,
                   label=name, alpha=0.85)
        ax_z.plot(t, pos[:, 2], color=col, ls=ls, lw=2.2, label=name, alpha=0.85)
        ax_spd.plot(t, speed, color=col, ls=ls, lw=2.2, label=name, alpha=0.85)

    # waypoints on XY
    ax_xy.scatter(wpts[:, 0], wpts[:, 1], s=160, zorder=5, color="red",
                  marker="*", label="Waypoints")
    ax_xy.scatter(*wpts[0, :2], s=100, zorder=6, color="green", marker="o",
                  label="Start")
    ax_xy.set(xlabel="X [m]", ylabel="Y [m]", title="XY trajectory")
    ax_xy.axis("equal"); ax_xy.legend(fontsize=9); ax_xy.grid(alpha=0.3)

    # altitude target lines
    for wi, wpt in enumerate(waypoints[1:], 1):
        ax_z.axhline(wpt[2], color="grey", ls=":", lw=0.9, alpha=0.6)
    ax_z.set(xlabel="Time [s]", ylabel="Altitude [m]", title="Altitude")
    ax_z.legend(fontsize=9); ax_z.grid(alpha=0.3)

    ax_spd.set(xlabel="Time [s]", ylabel="Speed [m/s]", title="Airspeed")
    ax_spd.legend(fontsize=9); ax_spd.grid(alpha=0.3)

    # wall-clock time bar chart
    names = list(results.keys())
    wall_times = [results[n]["wall_s"] for n in names]
    bars = ax_bar.bar(names, wall_times,
                      color=[COLORS[n][0] for n in names], alpha=0.8)
    for bar, wt in zip(bars, wall_times):
        ax_bar.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 0.3,
                    f"{wt:.1f} s", ha="center", va="bottom", fontsize=10)
    ax_bar.set(ylabel="Wall-clock time [s]",
               title="Compute time (same sim duration)")
    ax_bar.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\nPlot saved → {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "=" * 65)
    print("PROPULSION MODEL COMPARISON")
    print("  MPC controller · simple internal model · 2 waypoints")
    print("=" * 65)

    # ── hover mass from BEM ───────────────────────────────────────────────
    print("\n  Computing BEM hover mass …", flush=True)
    mass_eff, body_mass = _bem_hover_mass()
    print(f"  BEM hover thrust : {mass_eff * 9.81 / 4:.3f} N/rotor")
    print(f"  Effective mass   : {mass_eff:.4f} kg  (body={body_mass:.4f} kg + 4×0.050 kg motors)")

    # ── MJCF override: replace main body mass for all three sims ─────────
    xml_template = Path(
        __file__
    ).parent.parent / "uavsim" / "models" / "quadcopter.xml"
    mjcf_xml = xml_template.read_text().replace(
        'mass="1.0"', f'mass="{body_mass:.6f}"'
    )

    # ── shared controller params (lightweight drone, simple model) ────────
    ctrl_params = quadcopter_params(mass=mass_eff)

    # ── build BEM artefacts once ──────────────────────────────────────────
    bem_cfg_live = BEMConfig(geom=geom, af=af, rotor_params=rotor, rho=1.225)
    table = _build_table()
    bem_cfg_table = BEMConfig(
        geom=geom, af=af, rotor_params=rotor, rho=1.225, bem_table=table
    )

    modes = [
        ("SIMPLE",    PropulsionModel.SIMPLE,    None),
        ("BEM_LIVE",  PropulsionModel.BEM_LIVE,  bem_cfg_live),
        ("BEM_TABLE", PropulsionModel.BEM_TABLE, bem_cfg_table),
    ]

    results = {}
    for label, prop_model, bem_cfg in modes:
        print(f"\n{'─' * 65}")
        print(f"  Running {label} …")
        if prop_model is PropulsionModel.BEM_LIVE:
            print("  (JIT compile on first step — expect ~10 s one-time delay)")
        vehicle = _make_vehicle(prop_model, mass_eff, mjcf_xml, bem_cfg)
        results[label] = _run(vehicle, label, ctrl_params)

    _plot(results, WAYPOINTS, "propulsion_comparison.png")

    print("\n" + "=" * 65)
    print("SUMMARY")
    print(f"  {'Model':<12}  {'sim [s]':>8}  {'wall [s]':>9}  {'realtime':>9}")
    print("  " + "-" * 42)
    for name, data in results.items():
        rt = data["sim_s"] / data["wall_s"] if data["wall_s"] > 0 else 0.0
        print(f"  {name:<12}  {data['sim_s']:>8.1f}  {data['wall_s']:>9.1f}  {rt:>8.1f}x")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
