# TODO Roadmap — INDI/WLS Control, Sideslip, Full Mission Demo

Builds on `SESSION_SUMMARY.md` and the patch in this folder. Sequencing
agreed: **INDI/WLS control allocation → sideslip control → full mission
example** (hover → transition → cruise-with-turn → transition → hover).

Reference: Smeur, Bronz, de Croon, *"Incremental Control and Guidance of
Hybrid Aircraft Applied to a Tailsitter Unmanned Air Vehicle,"* JGCD 2019
(arxiv.org/pdf/1802.00714) -- flight-validated on the real Cyclone.
Equation numbers below refer to that paper unless stated otherwise.
Reference implementation for the *algorithm pattern* (not directly
reusable -- it's for a quad, and it's fairly literal C-to-Python, not
JAX): github.com/enac-drones/dronesim/tree/master/dronesim/control
(`INDIControl.py`, `wls_alloc.py`).

**Key advantage we have that the original Cyclone work didn't**: they
had to fit control effectiveness (`G`) from real flight-test data
specifically to avoid needing an aerodynamic model. We have a validated
differentiable model (`phi_theory.py` + BEM propulsion), so `G` can be
computed directly and continuously via `jax.jacobian` on the real wrench,
anywhere in the envelope, rather than flight-test curve-fitting. This
replaces their hardest, most time-consuming implementation steps
(their guidelines' steps 3, 6, 7).

---

## Phase A — INDI attitude + rate control, with WLS allocation

**Goal:** Replace the current hover-only fixed-trim-linearized mixer
(`uavsim/controllers/tailsitter_hover.py`) with a single continuous INDI
controller that works across the whole envelope, matching the paper's
core claim (Sec. 2).

**Steps:**
1. **Rate loop (the core INDI update, Eq. 2-3).** Given current angular
   velocity `Omega` (from `state.angular_velocity`) and its time
   derivative `Omega_dot` (finite-difference across sim steps, or exact
   if available), and current thrust `T`:
   ```
   u_c = u_f + G^+ @ (nu - [Omega_dot_f; T_f])
   ```
   `G` (4x4 here: 3 angular accel axes + thrust, x 4 actuators) computed
   via `jax.jacobian` on `tailsitter_wrench` w.r.t. commands, evaluated
   at the **current** state each step (not a fixed trim point) -- this
   is what makes it INDI rather than a fixed linear mixer. Note: our
   existing `compute_hover_mixer` already does the "jacobian + invert"
   part; the INDI change is doing it fresh each step from current state
   rather than once at a fixed trim.
2. **Rate reference from angular-rate proportional control (Eq. 4):**
   `nu = [K_Omega @ (Omega_ref - Omega); T_d]`.
3. **Attitude loop (Eq. 5-6):** quaternion error feedback for `Omega_ref`.
   We already have a rotation-matrix attitude error
   (`tailsitter_hover.py::attitude_error`) that's arguably *more* robust
   than the paper's own approach (they use ZXY Euler specifically to
   avoid gimbal lock at -90deg pitch, still singular at +-90deg roll;
   ours has no Euler singularity at all) -- reuse it rather than
   reimplementing quaternion-error feedback from scratch.
4. **WLS control allocation (Sec. 2.7)**, replacing plain pseudo-inverse:
   weighted least-squares QP with per-axis priority weights (paper's
   values: `[100, 1000, 0.1, 10]` for roll/pitch/yaw/thrust -- pitch
   highest priority since it's the axis most likely to saturate and
   matters most for the vehicle's natural pitch-down tendency). Options:
   - Port `wls_alloc.py`'s active-set algorithm (works, but fairly
     literal C-to-Python, not idiomatic JAX/jittable as-is).
   - Cleaner: use `jaxopt`'s QP/OSQP solver, or a simple JAX-native
     projected-gradient/ADMM solve given the small problem size (4
     actuators). Worth a quick prototype of both before committing.
5. **Control effectiveness scheduling.** Unlike the paper (which needed
   explicit fitted functions of theta/V, Eq. 7-12, since they had no
   model), we get this "for free" by just calling `jax.jacobian` at the
   current state -- confirm this actually behaves smoothly across the
   envelope (sanity-sweep G's entries vs. theta/V, compare qualitatively
   to the paper's Eq. 7/9 shapes as a sanity check, not a match
   requirement since our vehicle differs).
6. **Position/velocity outer loop (Sec. 4, Eq. 18-30).** Also INDI-style:
   uses current measured/simulated acceleration as baseline, control
   effectiveness matrices `G_T` (thrust effect on accel) and `G_L` (lift
   effect on accel via pitch) computed via jacobian rather than the
   paper's simplified analytic forms (Eq. 28-33, which they needed since
   they had no lift model -- we have one). This replaces
   `tailsitter_hover_step`'s simpler position-PD -> desired-attitude
   construction.
7. **Non-minimum-phase elevon/lift coupling (Sec. 4.1)** -- watch for
   this once real closed-loop tests run: elevon deflection for pitch
   initially produces lift in the *wrong* direction before pitch catches
   up, can cause oscillation. Paper's two fixes: (a) deliberately
   over-state the modeled pitch-effectiveness (crude but simple), or
   (b) model the transient lift-from-elevon, high-pass filter it, and
   subtract from the feedback acceleration signal. Only worth adding if
   oscillation actually shows up in testing -- don't pre-emptively add
   complexity.
8. Test: closed-loop MuJoCo across hover, the existing cruise trim
   points, AND (new) genuine transition maneuvers -- accelerate from
   hover to cruise trim and back, checking the SAME controller handles
   it without any mode-switch logic (this is the actual point of INDI
   here).

---

## Phase B — Sideslip control (Sec. 3) — DONE (2026-07-18)

**Goal:** Cyclone has no rudder/vertical tail -- nothing currently
prevents the vehicle from picking up sideslip, which degrades wing
efficiency and (per the paper) can reduce lift. Needs active control.

**Status: complete.** Added to `uavsim/controllers/tailsitter_indi.py`
(purely additive): `estimate_sideslip()` (ground-truth beta from body-frame
airspeed, approach (a)), `coordinated_turn_rate()` guidance helper
(`g*tan(phi_ref)/V`), a `k_beta` gain (default 1.5), and a
`use_sideslip_control` flag + internal `psi_ref` heading integrator on the
controller. When enabled, `update()` integrates `psi_dot_ref = yaw_rate_ff +
k_beta*beta` (Eq. 17) into the heading fed to `desired_attitude_transition`,
gated on forward speed > `V_SIDESLIP_MIN` (2 m/s). Sign verified in
closed-loop sim (positive beta -> positive heading rate yaws the nose into the
slip).

**Key design point (matches the paper): phi in Eq. 17 is phi_REF, not the
measured bank.** The coordinated-turn feedforward must come from the commanded
turn (guidance-supplied `yaw_rate_ff`), NOT the vehicle's measured bank: in
this architecture the outer loop banks to hold lateral POSITION too (crabbing
against a crosswind), so feeding measured bank back as a turn command misfires
(verified: it pushed a crosswind's steady sideslip from ~5deg back up to
~20deg). `coordinated_turn_rate()` converts a commanded bank -> `yaw_rate_ff`
for the Phase C guidance layer.

**Validated (tests/test_tailsitter_indi.py::TestSideslipControl, +
TestSideslipEstimate, TestCoordinatedTurnRate -- 7 tests, all pass):**
- Crosswind rejection: steady +y crosswind gives sustained sideslip the
  position loop can't remove (airspeed = ground vel - wind); enabling sideslip
  control drives it ~15deg -> ~5deg mean.
- Coordinated turn: a curving velocity reference at fixed heading builds ~52deg
  sideslip; with `yaw_rate_ff` = commanded turn rate it holds ~1deg through the
  turn.

**Steps (all done):**
1. **Sideslip estimation.** Paper uses lateral specific force from
   accelerometer feedback (Eq. 13-16, `beta = c2*f_y + b2`, fit from
   real flight data). In simulation we have ground-truth state, so we
   can either (a) compute true sideslip directly from body-frame
   velocity for a first pass (simpler, "cheating" relative to a real
   vehicle but fine for sim/RL purposes per the stated goal), or (b)
   deliberately mimic their accelerometer-based estimator for realism if
   sim2real transfer matters later. Start with (a); note (b) as a
   refinement.
2. **Sideslip feedback + coordinated turn (Eq. 17):**
   `psi_dot_ref = g*tan(phi_t)/V_l + K_beta*beta`, feeding into the
   heading reference used by the attitude loop's `desired_yaw` input
   (already a parameter of `tailsitter_hover_step`/whatever replaces it
   in Phase A).
3. Test: forward flight with a deliberate lateral disturbance (wind or
   an initial sideslip condition), confirm the controller drives it back
   to near-zero sideslip and holds it through a turn.

---

## Phase C — Full mission example — DONE (2026-07-18)

**Goal:** hover → transition → cruise (with a turn) → transition → hover,
as a single scripted demonstration using Phase A+B's controller, no
mode-switching logic in the control law itself (matching the paper's
core claim -- transitions should fall out naturally from the same
continuous controller tracking a changing reference, not from an
explicit state machine).

**Status: complete.** New reusable guidance layer
`uavsim/controllers/tailsitter_guidance.py` (`MissionLeg`, `MissionGuidance`,
`roadmap_mission()`): a leg-based mission (hold speed / accelerate / cruise+turn
/ decelerate) is turned into a continuous `(position, velocity, acceleration,
yaw, yaw_rate_ff)` reference. Horizontal velocity is `s(t)*[cos psi, sin psi]`
with heading integrated from the leg turn rate, so the accel feedforward is
analytic (tangential + centripetal); the turn rate is also handed to the
controller as `yaw_rate_ff` for the coordinated turn (Phase B). `MissionGuidance`
is stateful (reset / step(dt)), integrating reference position and heading. Unit-
tested in `tests/test_tailsitter_guidance.py` (10 tests, all pass).

Demo: `examples/tailsitter_mission_demo.py` (live MuJoCo viewer + plots, or
`--headless` for plots only). Flies the full `roadmap_mission()` with the one
`TailsitterINDIController` (sideslip control on), NO mode switch. Result: reaches
6 m/s cruise, pitches to ~45 deg, holds a coordinated 180 deg turn, decelerates,
and recovers a stable nose-up hover (b1.z=1.000, |vel| 0.06, |omega| 0.01). New
6-panel `plot_tailsitter_mission()` in `uavsim/viz/plotting.py` mirrors the
paper's Fig. 16-19 (ground track, altitude, pitch, speed, actuator saturation,
sideslip; phase-shaded).

**Latent sim bug fixed in passing:** `MuJoCoSimulator.get_state()` wrapped the
`qpos`/`qvel` slices with `jnp.asarray`, which zero-copies from numpy on
CPU/x64 -- so every `state_history` entry ALIASED the live MjData buffer and
read back the final state (collapsing all recorded trajectories, incl. the
existing quad `plot_flight_data` demos). Now copies the slices (`np.array`)
before wrapping. This is why the first Phase C plots showed a single point.

**Known limitation (expected, = Phase D):** the model-based outer loop ignores
wing lift, so altitude bulges to ~3.2 m during the fast cruise leg before
returning to 2 m. Fixing that hold is exactly Phase D's measured-accel outer
loop.

**Steps (all done):**
1. **Guidance/reference layer** (Sec. 5 for a reference, though our
   needs are simpler for a first pass): position/waypoint -> velocity
   reference -> acceleration reference (Eq. 38's simple PD is a
   reasonable starting point: `xi_ddot_ref = ((xi_ref - xi)*K_xi -
   xi_dot)*K_xidot`). Don't need the paper's full efficient-turning
   heuristic (Sec. 5.1) or line-following (Sec. 5.3) for a first
   scripted demo -- straightforward waypoint sequencing is enough.
2. **Mission script:** a sequence of waypoints/legs exercising all
   phases explicitly:
   - Hover at a start point (hold position, matches Phase 0's validated
     hover controller behavior).
   - Accelerate to cruise trim speed (transition -- the interesting
     part, no explicit "transition mode," just the guidance layer
     commanding increasing velocity and the INDI controller tracking
     whatever attitude/control that requires).
   - Cruise leg with a turn (exercises the sideslip controller from
     Phase B, and cruise trim's lateral/roll authority found in the
     session's earlier trim work).
   - Decelerate back through transition to hover at an end point.
3. **Visualization:** use the existing MuJoCo viewer setup
   (`uavsim/sim/mujoco_sim.py` + whatever viewer script already exists in
   `examples/`) to actually watch the mission fly, plus the
   Phase-0-established plotting conventions (see the earlier project
   plan's Plotting section, if built) for trajectory/attitude/control
   traces -- pitch angle, airspeed, and control surface saturation over
   the mission, mirroring the paper's own Fig. 16-19 presentation.
4. **Success criteria** (informal, matching the spirit of the paper's own
   test-flight validation, Sec. 6): completes the full hover-transition-
   cruise-transition-hover sequence without diverging, tracks the
   guidance-commanded acceleration reasonably well (some tracking loss
   during actuator saturation is expected and fine, per the paper's own
   findings -- Fig. 19 shows exactly this), and returns to a stable hover
   at the end.

---

## Phase D — Guidance INDI outer loop (with wing-lift effectiveness) — DONE (2026-07-19)

**Status: complete, and solved more directly than the original plan below.** The
real missing piece was structural, not a filtering refinement: the paper's
*Guidance INDI* outer loop (Sec. 4, Eq. 18-33) controls linear acceleration with
the vector `v = [phi, theta, T]` by inverting a control-effectiveness matrix
`G = G_T + G_L` that **includes the wing-lift derivative** (`dL/dtheta`, Eq. 27-29).
That lift term is what lets the loop trade thrust for pitch -- pitch the nose
over, let the wing carry the weight, pull thrust back. The measured-accel
thrust-VECTOR loop (`use_indi_outer`) has no lift term, which is why it held
loosely; the model-based loop has neither lift nor measurement.

**Implemented** in `uavsim/controllers/tailsitter_indi.py` (`use_guidance_indi=True`):
- Outer control vector **`v = [phi, theta, T]`** (bank, pitch, thrust) -- the
  paper/Paparazzi vector. `guidance_attitude(phi,theta,psi) = Rot(h,phi) @
  desired_attitude_transition(b1(theta,psi),psi)`, h = horizontal heading axis.
  Bank `phi` about h does DOUBLE DUTY (the paper's single roll channel): at hover
  (b1 up) it tilts the thrust sideways (lateral authority); in cruise (b1 ~ h) it
  rolls the LIFT vector into the turn. `phi=0` reduces to wings-level. Heading
  `psi` stays with the Phase-B sideslip loop. (An earlier `[theta, lam=lateral-
  thrust-tilt, T]` parameterisation with FORCED wings-level flew the straight
  transition but diverged in the turn -- see "the turn fix" below.)
- Two effectiveness backends: **`"jacobian"` (default)** -- `G` from `jax.jacobian`
  of the real `phi_theory` specific force (exact `dL/dtheta` in the theta column
  AND exact lift-ROLL in the phi column, no fitting; the "we have a model, the
  Cyclone work didn't" advantage applied to the outer loop);
  **`"analytic"`** -- explicit Eq. 28-33 `G` (`col_phi = (T·(h×b1)+L·(h×lift_dir))/m`,
  `col_theta` with fitted `dL/dtheta` Eq. 33 + Sec. 4.1 `pitch_scaling`), for
  sim2real where you can't autodiff a real aircraft. Inner G/WLS allocation untouched.
- INDI law `dv = gain * pinv(G) @ (a_ref - a_meas_f)`, `[phi,theta]_cmd =
  [phi0,theta0] + [dphi,dtheta]` read off the current attitude via
  `guidance_extract_bank_pitch` (as in Cyclone/dronesim/Paparazzi guidance).

**The turn fix (checked against Paparazzi `guidance_indi_hybrid_tailsitter.c`):**
the coordinated turn is roll + yaw + sideslip that must all agree. The loop had
yaw (sideslip `psi_ref`) and sideslip (Phase B) but **no roll** -- it tried to get
lateral force by tilting the tiny cruise thrust (~20 % W), which has almost no
authority while the big lift vector sat pointing straight up (wings level). The
speed/altitude runaway in the turn was exactly that. Making **bank a native
control channel whose G-column carries the lift** (like Paparazzi's
`Gmat[0][0] = ... + cphi*spsi*lift`) means the coordinated-turn bank falls out of
the same inversion -- no feedforward. Now roll+yaw+sideslip are consistent.

**Validated (closed-loop + tests):** straight 12 m/s cruise -> pitch ~74 deg,
**thrust ~22 % (props ~5 % W, wing ~95 %), altitude held ~0.1 m** vs the model
loop's ~2.4 m climb. **FULL turning mission** (`roadmap_mission`, 12 m/s, 180 deg
turn): speed tracks 12 m/s (was runaway to 35), **altitude held [9.3, 10.2] m
through the turn**, **coordinated turn holds beta ~0.1 deg**, returns to hover
(b1z 1.00). Tests in `tests/test_tailsitter_indi.py`
(`TestGuidanceParameterization`, `TestGuidanceEffectiveness` incl.
bank-gives-lateral-via-lift-in-cruise, `TestGuidanceINDIClosedLoop`) + all prior
tests pass.

**Key physics finding:** below ~10 m/s the wing cannot support the weight at all
(stall-speed limit), so the props MUST carry it regardless of the controller --
the 6 m/s demo's ~45 deg deep-stall attitude was partly physics, not just the
blind controller. A genuine wing-borne cruise needs >=12 m/s (level trim there:
theta~74 deg, thrust ~18% W). Demo default raised accordingly.

The original narrower plan (better `a_meas` filtering on the *thrust-vector* loop)
is superseded and kept below only for reference; the lift-aware `G` is the fix.

---

### (superseded) original Phase D plan — robust measured-acceleration INDI outer loop

**Goal:** Make the *measured-acceleration* INDI outer loop
(`uavsim/controllers/tailsitter_indi.py`, `use_indi_outer=True`) hold as
tightly as the model-based default, so the outer loop gets INDI's aero-error
rejection (the paper's Sec. 4 point) instead of relying on a model of
gravity/aero. Deferred out of Phase A on purpose: the inner INDI loop is the
core contribution and the model-based outer loop already flies the full
transition, so this is a refinement, not a blocker.

**Current state (end of Phase A):** the measured-accel outer loop is *stable*
but holds loosely -- it settles with a steady-state tilt (~`b1z` 0.72 vs the
model-based loop's ~1.0) and a small position offset. Two things are already
known:
- The thrust-vector baseline MUST be the hover thrust `m*g` (the `T = 9.81`
  assumption in dronesim's `_INDIPositionControl` / Paparazzi). Using the
  *modeled* thrust as the baseline made `|omega|` diverge; `m*g` fixed the
  divergence. So the remaining looseness is NOT the thrust baseline.
- The looseness is the Sec. 4.1 non-minimum-phase / cascade-bandwidth
  problem: near hover `m*g*b1` dominates the small `m*(a_ref - a_meas)`
  correction, so the attitude reference is nearly self-referential and leans
  entirely on a timely, low-lag `a_meas` -- which a raw finite-difference +
  first-order low-pass of velocity can't provide against the aggressive inner
  loop.

**Already tried and ruled out (don't redo):**
- Frame bug in `a_meas`: no -- confirmed MuJoCo `qvel[0:3]` is world-frame.
- Elevon-lift high-pass subtraction (Phase A step 7, Sec. 4.1 option b): no
  measurable effect here, so the dominant issue is measurement lag, not the
  elevon-lift transient.
- Gain/filter sweeps (softer outer gains, larger `filt_tau`): reduce but do
  not remove the tilt.

**Steps:**
1. **Better acceleration estimation.** Replace the first-order finite-diff +
   LPF with the paper's 2nd-order Butterworth on the accelerometer signal,
   matched in cutoff/phase to the actuator/command filter (Smeur's whole
   INDI hinges on the gyro/accel and the command filters being
   *synchronised*). Consider a complementary filter that blends the noisy
   finite-difference `a_meas` with the modeled specific force (model at high
   frequency where the finite-difference is noisy, measurement at low
   frequency where it's trustworthy) -- this is the principled way to get a
   fast *and* clean `a_meas`.
2. **Thrust estimation.** Keep the stabilising `m*g` baseline near hover, but
   schedule/estimate the actual thrust magnitude as airspeed grows (in cruise
   the thrust vector is far from `m*g` in both magnitude and direction). A
   slow, low-passed thrust estimate (from the throttle->thrust map / BEM
   table, or from `m*f_meas - F_aero_model` along `b1`) that never introduces
   the fast jitter that made the raw modeled thrust diverge.
3. **Explicit timescale separation.** Set the outer-loop bandwidth well below
   the inner rate loop's, and verify it (the two were too close in Phase A).
4. **Optional model+measurement blend for the thrust-vector direction.** Near
   hover, fall back toward the model-based restoring term (which is dead-on);
   as airspeed/aero grows, weight the measurement more (where INDI's
   aero-rejection actually earns its keep). A complementary blend, not a hard
   switch.
5. **Test:** the measured-accel outer loop should (a) hold hover as tightly as
   the model-based default (`pos_err < ~0.1 m`, `b1z > 0.98`, `|omega|` small),
   AND (b) track the Phase A transition at least as well as the model-based
   loop -- ideally *better* on altitude hold during the fast leg (the
   model-based loop climbs to ~3 m because it ignores wing lift; measured
   `a_meas` should catch that and hold altitude). That altitude-hold
   improvement is the concrete win that justifies the loop.

---

## Notes / open questions to revisit

- **Aero fix (2026-07-19): camber `Cl0` sign in `phi_theory.py`.** The wing model
  had `Cl0` entering `Phi_fv` with the paper's z-DOWN sign while this codebase is
  body z-UP, so a positive-camber wing made a DOWNforce at zero AoA (CL(0) = -Cl0)
  and cruise trimmed at an unphysically high ~16 deg AoA (pitch ~74 deg). Fixed by
  negating the `Cl0` camber terms `((3,1): -Cl0, (1,3): Cda-Cl0)`; now
  `CL = +Cl0 + Cla*alpha`, cruise AoA ~7 deg (pitch ~83 deg), wing carries ~97 % of
  weight. Re-validated (aero 42 + hover/transition/basic 33 tests, full mission).
  Also flips the wing pitching moment (`Phi_mv` from `Phi_fv`) -- checked OK.
- Real `km`, `prop_y_offset`, and propwash `zeta_f` are still not real
  data (see SESSION_SUMMARY.md) -- fine to proceed with placeholders for
  control-law development, but worth closing out before treating any
  quantitative result as representative of the real vehicle.
- Cruise trim was validated via a root-find, not closed-loop MuJoCo --
  Phase A's closed-loop testing at cruise trim points will be the first
  real closed-loop cruise validation.
- Worth reading MAVLab's earlier/related Cyclone papers (Bronz et al.,
  AIAA 2017-3739, referenced in the INDI paper) for the vehicle's real
  mass/geometry across revisions, given the 1.2kg vs 1.5kg discrepancy
  noticed this session.
