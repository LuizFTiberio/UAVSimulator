# Session Summary — phi-theory Tailsitter Model

## Goal of this session

Take the UAVSimulator repo (JAX + MuJoCo) and build a differentiable
tail-sitter aerodynamic model based on phi-theory (Lustosa, Defay,
Moschetta, JGCD 2019), generalized beyond the paper's fixed 2-section
worked example, then get it to a state where it can actually hover and
has a validated cruise trim -- as groundwork for RL / multi-vehicle work
later, per the stated long-term goal for this simulator.

Vehicle: **Cyclone**, a real TU Delft MAVLab hybrid tailsitter (developed
with ENAC Toulouse's dronelab) -- discovered mid-session, not initially
known. It has a dedicated control paper: Smeur, Bronz, de Croon,
*"Incremental Control and Guidance of Hybrid Aircraft Applied to a
Tailsitter Unmanned Air Vehicle,"* JGCD 2019 (arxiv.org/pdf/1802.00714),
describing an INDI + WLS-allocation controller flight-tested on this
exact airframe. This is the basis for the next phase of work (see
TODO_ROADMAP.md).

---

## Phase 0 — Core phi-theory module (`uavsim/aero/phi_theory.py`)

Built from the paper's general primitives (Eq. 14/21, 45, 67, 79),
**not** from Eqs. (97)-(98) (the paper's own hand-expanded 2-section
worked example) -- deliberately, to avoid transcribing a dense
equation block that showed signs of possible OCR/typesetting issues
when first extracted. Generalized to N wing sections / N propellers, each
section carrying its own area, position, and set of assigned propellers.

**Architecture (current, after a mid-session revision):**
1. **Whole-vehicle "dry" term** (`eq14_whole_vehicle_wrench`): the general
   `tau = -eta_norm * C @ Phi @ (C @ eta_b)` law, using the *real*
   `eta_norm = sqrt(v^2 + phi*omega^2)` (Eq. 17) -- an earlier version of
   this module used plain velocity norm here, which was a real gap, fixed
   this session. `Phi` is assembled from `Phi_fv` (velocity->force),
   `Phi_fw` (rate->force), a *geometric* `Phi_mv` (Eq. 67,
   `Phi_mv = B^-1 [Delta_r x] Phi_fv`, not a directly-fit constant -- see
   "Sign convention bug" below), and `Phi_mw` (real rate damping).
2. **Per-section propwash term** (Eq. 79): thrust-linear force/moment,
   elevon-modulated (Eq. 95), superposed per propeller-fed section.

An earlier design also computed freestream force *per section* using a
"local velocity" (rigid-body point-velocity, `v_cg + omega x a_k`) as a
proxy for rotational-rate coupling, before any real Cmq/Clp/Cnr data
existed. That mechanism is **retired** now that real rate-damping data is
used -- keeping both would double-count rotational-rate effects.

## The vehicle (`uavsim/vehicles/tailsitter.py`)

Three parameter sets, kept deliberately distinct:
- **`cyclone_params()`** -- the real, as-measured vehicle (wingspan 0.88m,
  chord 0.17m, prop diameter 0.21m, MTOW 1.5kg, AXI2212-26 + HQ-Prop 8x5).
  Thrust-to-weight ratio at MTOW is **~0.79 -- this vehicle, as measured,
  cannot hover.** MAVLab's own papers suggest Cyclone was designed for
  forward-flight efficiency (up to 90 min endurance) rather than hover
  performance, consistent with this.
- **`cyclone_hover_capable_params()`** -- same wing/mass, but with a prop
  resized via a real BEM solve (github.com/LuizFTiberio/ccblade-jax,
  validated against CCBlade.jl) to actually hover: D=11in, T/W~1.98 at
  max throttle. This is the vehicle used for all hover/cruise-trim
  validation in this session. **Not** the real Cyclone's actual
  propeller -- a resized-for-hover variant.
- Aero coefficients (`Cd0, Cda, Cla, Cl0, Cy0, Clp, Cmq, Cnr, Cmde, Clda`)
  are **real, fitted data** for a Cessna-class GA aircraft (from Leandro
  Lustosa, open-source), used as a starting point without changing
  Cyclone's geometry, per an explicit scoping decision this session. CG
  at 25% chord / elevon at trailing 10% chord were also explicit choices
  -- and turned out to match MAVLab's own stated design reasoning for
  the real Cyclone (CG at the forward-flight neutral point, i.e.
  quarter-chord) almost exactly, discovered only after reading their
  paper.

**Still placeholder / not yet real data:** `km` reaction-torque ratio for
`cyclone_params()`'s SIMPLE propulsion path (not used by
`cyclone_hover_capable_params()`, which gets real km from the BEM solve);
`prop_y_offset` (0.30m, estimated from a photo, not measured);
`zeta_f` propwash elevon effectiveness (still a placeholder shape, unlike
the freestream `Cmde`/`Clda` which are now real).

---

## Real bugs found and fixed this session (chronological)

1. **Missing pitch authority.** Original section geometry had zero
   chordwise (x) offset -> symmetric elevon deflection produced *zero*
   pitch moment, contradicting the paper's own claim (Sec. VI.A).
   Fixed by giving sections a nonzero `Delta_r` (elevon's position
   behind CG) -- the mechanism the paper itself uses (Eq. 67/68).

2. **JIT/grad footgun.** `PhiWingSection.prop_indices` is static Python
   topology (used for indexing), not a differentiable leaf. Passing
   `params` as a direct `jax.jit` argument breaks it. Fix: bind params
   via closure before jitting (the pattern already used elsewhere in
   this repo, e.g. `mujoco_sim.py`).

3. **Mixer linearized at the wrong point.** The hover control-allocation
   mixer (`compute_hover_mixer`) originally linearized at a hardcoded 50%
   throttle -- correct only by construction for the old placeholder
   vehicle (whose `kf` was itself back-derived to make 50%=hover trim).
   Once a real, independently-fit `kf` was used, real hover trim moved to
   ~75% throttle, and since thrust is convex in throttle, linearizing far
   from the true operating point caused a **real closed-loop divergence**
   (commanded hover thrust, got 13% more, runaway climb) before being
   caught and fixed (linearize at the vehicle's actual computed trim).

4. **`Phi^(mv)` appearing inside a force equation (Eq. 97)** -- initially
   suspected as OCR corruption, then the person provided a screenshot
   confirming it's real. Root-caused (not just accepted): it's the
   rigid-body point-velocity effect (`v_local = v_cg + omega x a`) in
   disguise, verified symbolically with sympy
   (`Phi_fv @ (omega x a) == Phi_mv(a)^T @ B @ omega`). This became the
   (later retired -- see Phase 0 above) local-velocity-coupling
   mechanism.

5. **`safe_norm` gradient-at-zero bug.** `jnp.linalg.norm`'s gradient is
   NaN at exactly `x=0` (a genuine mathematical cusp). This broke a
   cruise-trim Newton/LM solve (which differentiates the wrench w.r.t.
   attitude, hitting pure-axial-flow states where a lateral-velocity
   vector is exactly zero). Fixed at the source with a new
   `uavsim/core/math.py::safe_norm`, applied everywhere a velocity norm
   is taken in the aero/propulsion modules -- this matters for *any*
   future gradient-based work (MPC, RL, sysID) around hover or trimmed
   flight, not just the trim solve that surfaced it.

6. **Sign convention bug in `Phi_mw` / elevon-freestream moment.** Built
   initially from the paper's literal `+0.5*Eq.(60)`, but the person's
   real MATLAB implementation (from Leandro) negates all its moment-block
   terms. Verified directly, not just copied: fed a positive pitch rate
   with real (damping) `Cmq=-12.4` through the full formula and confirmed
   only the negative-sign convention gives a physically correct restoring
   moment. Fixed `Phi_mw` and `elevon_phi_mv_act` to match.

7. **BEM propulsion hardcoded to body-Z.** `propulsion_bem.py` assumed
   rotor thrust along body+Z (quad convention) throughout. Generalized to
   an arbitrary `rotor_axis`, regression-tested for exact backward
   compatibility on the default axis (9 tests), then used with
   `rotor_axis=[1,0,0]` for the tailsitter. This is what let real,
   velocity-dependent BEM thrust (built from the same 11in blade design
   used for prop sizing) replace the velocity-blind `kf*omega^2` model --
   which matters because the real Cyclone thrust data shows thrust
   roughly halving from 0 to 15-20 m/s at fixed RPM, an effect the old
   model literally could not represent.

8. **Several "test failures" that were real physics, not bugs** --
   caught, checked by hand, and confirmed rather than assumed either way:
   real camber (`Cl0=0.31 != 0`) means propwash alone produces lift even
   at zero elevon deflection, which couples into a genuine pitch-trim
   requirement through the elevon's own moment arm -- something the old
   idealized symmetric/uncambered placeholder model could not show.
   Tests were corrected to check what's actually still guaranteed (roll/
   yaw symmetry) rather than an idealized zero that real airfoil data
   doesn't support.

---

## What's validated, and how

- **Hover**: closed-loop in MuJoCo, including recovery from a genuine
  ~90 degree attitude error (spawn nose-forward, must reorient to
  nose-up while holding position). Rotation-matrix attitude error (not
  Euler -- Euler gimbal-locks exactly at the tailsitter's nominal hover
  attitude). Control-allocation mixer built via `jax.jacobian` on the
  real wrench at the vehicle's actual trim, not hand-derived.
- **Cruise trim**: a real Levenberg-Marquardt solve (hand-implemented;
  `jaxopt`'s LM did not behave as expected) finds genuine equilibria for
  V=5-15 m/s, converged to ~1e-15 cost, with a physically sensible
  pitch-angle-vs-speed curve. Trim genuinely fails above ~18-20 m/s --
  not a solver problem, a real consequence of sizing the prop for hover
  rather than cruise (confirmed against the BEM table's own thrust
  falloff). **Not** validated in closed-loop MuJoCo -- trim existence and
  the linearized mixer at one trim point are confirmed; an actual cruise
  velocity-hold controller was not built this session.
- **125+ / 147 tests** pass across the whole touched surface, including
  regression tests for every bug above (so they can't silently
  reappear).

## Final test count: **147 passed**, `pytest tests/ uavsim/aero/tests/ -q`
(needs `ccblade-jax` on `PYTHONPATH` for the full count; a handful of
tests are skipped otherwise via `pytest.importorskip`).
