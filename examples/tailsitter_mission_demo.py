"""Phase C — full tail-sitter mission demo.

Flies ONE continuous mission with the single INDI controller from Phases A+B:

    hover -> transition (accelerate) -> cruise with a 180 deg turn ->
    transition (decelerate) -> hover

There is deliberately NO mode-switching in the control law -- the transitions
fall out of the same continuous controller tracking a changing reference (the
paper's core claim). The mission guidance (uavsim/controllers/tailsitter_guidance)
turns a list of high-level legs into the time-varying position/velocity/accel/
heading reference; TailsitterINDIController tracks it, with sideslip control on
so the cruise turn is coordinated (Phase B).

Outer loop (--outer): "guidance" (default) is the Guidance INDI loop (paper
Sec. 4) -- its effectiveness includes the wing-lift derivative, so in cruise it
pitches the nose over and unloads the weight onto the wing (thrust ~20% of
weight) while holding altitude. "model" is the aerodynamically-blind thrust-
vector loop, which keeps thrust high and climbs -- run it to see the contrast.
"analytic" is the paper's fitted Eq. 28-33 backend (the sim2real path).

Wind (--wind / --turbulence) is applied to the SIMULATOR, and the same wind
vector is handed to the controller each step. That is deliberate and it is a
cheat: a real Cyclone has no wind sensor and would have to estimate the wind
(or, equivalently, reconstruct sideslip from the lateral accelerometer, paper
Eq. 13-16). It is the same ground-truth shortcut estimate_sideslip() already
takes -- fine for sim/RL, flagged here so it is not mistaken for a wind
observer. Drop the wind_velocity= argument below to see the blind case.

Note the mission does NOT currently recover hover cleanly in a strong steady
wind: the elevons saturate in the final hover leg and the INDI allocator starts
jumping between saturated vertices. --zeta-f raises the propwash elevon
effectiveness (the only source of hover elevon authority, see
default_phi_wing_params) to give it more control power; the default 0.3 is a
placeholder, not a Cyclone measurement, so raising it for this demo is honest.

Run:
    python examples/tailsitter_mission_demo.py            # live viewer + plots
    python examples/tailsitter_mission_demo.py --headless # plots only (no display)
    python examples/tailsitter_mission_demo.py --outer model    # blind outer loop
    python examples/tailsitter_mission_demo.py --wind 5 0 0     # 5 m/s steady wind
    python examples/tailsitter_mission_demo.py --wind 4 2 0 --turbulence light
    python examples/tailsitter_mission_demo.py --wind 5 0 0 --zeta-f 0.6

Produces tailsitter_mission.png (trajectory, altitude, pitch, speed, actuator
saturation, sideslip) mirroring the paper's Fig. 16-19.
"""

from __future__ import annotations

import argparse

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import mujoco

from uavsim.vehicles.tailsitter import tailsitter_params, tailsitter
from uavsim.sim.mujoco_sim import MuJoCoSimulator
from uavsim.core.math import quat_to_rotation_matrix
from uavsim.controllers.tailsitter_indi import (
    TailsitterINDIController, composite_inertia_from_model)
from uavsim.controllers.tailsitter_guidance import MissionGuidance, roadmap_mission
from uavsim.viz.viewer import SimulationVisualizer
from uavsim.viz.plotting import plot_tailsitter_mission
from uavsim.disturbances.wind import (
    ConstantWind, light_turbulence, moderate_turbulence, severe_turbulence)

# ── mission parameters ───────────────────────────────────────────────────────
ALTITUDE = 10.0
CRUISE_SPEED = 12.0       # >=12 m/s: wing can support weight -> wing-borne cruise
TURN_RATE = 0.15          # rad/s coordinated turn in the cruise leg


def _hover_quat() -> np.ndarray:
    """Nose-up hover attitude (rotate -90 deg about body y from nose-forward)."""
    axis, angle = np.array([0.0, 1.0, 0.0]), -np.pi / 2
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def _build_wind(mean_wind, turbulence: str, seed: int):
    """Wind model from the CLI options, or None for still air."""
    mean = np.asarray(mean_wind, dtype=float)
    if turbulence == "none":
        return ConstantWind(mean) if np.any(mean) else None
    factory = {"light": light_turbulence, "moderate": moderate_turbulence,
               "severe": severe_turbulence}[turbulence]
    return factory(mean_wind=mean, seed=seed)


def main(headless: bool = False, real_time_factor: float = 1.0,
         outer: str = "guidance", wind=(0.0, 0.0, 0.0),
         turbulence: str = "none", seed: int = 0, zeta_f: float = 0.3,
         substeps: int = 4):
    if headless:   # no display: force a non-interactive matplotlib backend
        import matplotlib
        matplotlib.use("Agg")

    print("\n" + "=" * 66)
    print("TAIL-SITTER FULL MISSION — hover / transition / cruise+turn / hover")
    print(f"   outer loop: {outer}")
    print("=" * 66)

    params = tailsitter_params(zeta_f_pitch=zeta_f)
    wind_model = _build_wind(wind, turbulence, seed)
    sim = MuJoCoSimulator(tailsitter(params=params), wind_model=wind_model,
                          substeps=substeps)
    I = composite_inertia_from_model(sim.model)
    ctrl_kw = dict(mass=sim.total_mass,
                   use_sideslip_control=True)   # coordinated turn (Phase B)
    if outer == "guidance":
        ctrl_kw.update(use_guidance_indi=True, guidance_effectiveness="jacobian")
    elif outer == "analytic":
        ctrl_kw.update(use_guidance_indi=True, guidance_effectiveness="analytic",
                       pitch_scaling=2.0)
    # "model": leave use_guidance_indi False -> aerodynamically-blind loop
    ctrl = TailsitterINDIController(params, inertia=I, dt=sim.dt, **ctrl_kw)

    legs = roadmap_mission(cruise_speed=CRUISE_SPEED, altitude=ALTITUDE,
                           turn_rate=TURN_RATE)
    guid = MissionGuidance(legs, altitude=ALTITUDE, start_xy=(0.0, 0.0),
                           start_yaw=0.0)

    print(f"  Cruise speed : {CRUISE_SPEED} m/s")
    print(f"  Altitude     : {ALTITUDE} m")
    print(f"  Turn rate    : {TURN_RATE} rad/s")
    print(f"  Mission time : {guid.total_time:.1f} s   ({len(legs)} legs)")
    print(f"  Mean wind    : {np.asarray(wind, dtype=float)} m/s"
          f"   turbulence: {turbulence}")
    print(f"  zeta_f pitch : {zeta_f}  (hover elevon authority; 0.3 = default placeholder)")
    print(f"  control dt   : {sim.dt * 1e3:.2f} ms    physics dt: "
          f"{sim.physics_dt * 1e6:.0f} us  ({substeps} substeps, Euler)")
    print("=" * 66 + "\n")

    # spawn at hover
    sim.reset()
    sim.data.qpos[3:7] = _hover_quat()
    sim.data.qpos[0:3] = np.array([0.0, 0.0, ALTITUDE])
    mujoco.mj_forward(sim.model, sim.data)
    state = sim.get_state()
    ctrl.reset(); guid.reset()

    # optional live viewer (degrade gracefully if no display)
    vis = None
    if not headless:
        try:
            vis = SimulationVisualizer(sim).launch()
            print("MuJoCo viewer launched — watch the mission fly.\n")
        except Exception as exc:
            print(f"Viewer unavailable ({exc}); running headless.\n")
            vis = None

    dt = sim.dt
    n_steps = int(guid.total_time / dt)
    print_every = max(1, int(0.5 / dt))

    commands, references = [], []
    try:
        for k in range(n_steps):
            ref = guid.step(dt)
            # Hand the controller the CURRENT wind (see module docstring: this is
            # ground truth, not an estimate). sim.wind_velocity is the vector the
            # simulator used on the previous step, and is zeros with no wind model.
            cmd = ctrl.update(state, jnp.asarray(ref.position),
                              setpoint_velocity=ref.velocity,
                              accel_feedforward=ref.acceleration,
                              desired_yaw=ref.yaw,
                              yaw_rate_ff=ref.yaw_rate_ff,
                              wind_velocity=sim.wind_velocity)
            state = sim.step(cmd)
            commands.append(np.asarray(cmd, dtype=float))
            references.append(ref)

            if not bool(jnp.all(jnp.isfinite(state.position))):
                print(f"DIVERGED at t={k * dt:.2f}s (leg {ref.leg_name}) — aborting.")
                break

            if vis is not None and vis.is_running:
                vis.sync(real_time_factor=real_time_factor)

            if k % print_every == 0:
                p = np.asarray(state.position)
                b1 = np.asarray(quat_to_rotation_matrix(state.quaternion)[:, 0])
                pitch = np.degrees(np.arctan2(np.hypot(b1[0], b1[1]), b1[2]))
                spd = float(np.linalg.norm(np.asarray(state.velocity)[:2]))
                print(f"t={k * dt:5.2f}s  {ref.leg_name:17s} "
                      f"pos=[{p[0]:6.1f} {p[1]:6.1f} {p[2]:5.1f}]  "
                      f"spd={spd:4.1f}  pitch={pitch:5.1f}deg")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if vis is not None:
            print("\nClose the viewer window to generate plots.")
            while vis.is_running:
                try:
                    vis.sync(real_time_factor=0.0)
                except Exception:
                    break
            vis.close()

    # ── summary ──────────────────────────────────────────────────────────────
    b1_end = np.asarray(quat_to_rotation_matrix(state.quaternion)[:, 0])
    print("\n=== mission summary ===")
    print(f"  end position   : {np.asarray(state.position)}")
    print(f"  end |velocity| : {float(np.linalg.norm(state.velocity)):.3f} m/s")
    print(f"  end |omega|    : {float(np.linalg.norm(state.angular_velocity)):.3f} rad/s")
    print(f"  end b1.z (hover -> 1.0): {b1_end[2]:.3f}")

    plot_tailsitter_mission(
        sim.state_history, sim.time_history, np.array(commands),
        references=references, save_path="tailsitter_mission.png",
        show=not headless)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headless", action="store_true",
                    help="no live viewer; save plots only")
    ap.add_argument("--rtf", type=float, default=1.0,
                    help="real-time factor for the viewer (default 1.0)")
    ap.add_argument("--outer", choices=["guidance", "model", "analytic"],
                    default="guidance",
                    help="outer-loop mode (default: guidance INDI)")
    ap.add_argument("--wind", type=float, nargs=3, metavar=("VX", "VY", "VZ"),
                    default=[0.0, 0.0, 0.0],
                    help="steady world-frame wind [m/s] (also the mean wind "
                         "under --turbulence). Default: still air.")
    ap.add_argument("--turbulence",
                    choices=["none", "light", "moderate", "severe"],
                    default="none",
                    help="Dryden turbulence on top of --wind (default: none)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the turbulence (default: 0)")
    ap.add_argument("--zeta-f", type=float, default=0.3, dest="zeta_f",
                    help="propwash elevon effectiveness = hover elevon "
                         "authority (default 0.3, a placeholder; try 0.6)")
    ap.add_argument("--substeps", type=int, default=4,
                    help="physics steps per 1 ms control step (default 4 = "
                         "250 us physics). Error falls ~linearly with this; in "
                         "this demo the controller dominates the runtime, so 4 "
                         "costs only ~17%% more wall time than 1.")
    args = ap.parse_args()
    main(headless=args.headless, real_time_factor=args.rtf, outer=args.outer,
         wind=args.wind, turbulence=args.turbulence, seed=args.seed,
         zeta_f=args.zeta_f, substeps=args.substeps)
