# User Acceptance Run — 2026-07-27

This report reviews the five PDF transcripts in `Test-Run-Examples/` against
`docs/acceptance-scenarios.md`. It records observed behavior, not inferred or
invented numerical expectations.

## Repair status

The defects recorded below were repaired in the same 2026-07-27 development
checkpoint:

- named-corner pins now remain semantic and resolve from arbitrary rectangle
  bounds;
- deterministic readiness guarantees human clarification questions;
- input/output springs are stable first-class `I…`/`O…` entities with semantic
  selectors, dimensional force-per-length conversion, and matched-DOF approval
  evidence;
- near-incompressibility and other validation warnings are visible before
  approval and inherited by final analysis evidence;
- compliance output no longer prints the mechanism-only zero objective;
- failed-worker lifecycle messages include the typed numerical code/message; and
- the OC sign guard now uses scale-aware sensitivity-noise handling.

The exact Scenario 4 configuration completed a contained 60-iteration confirmation
through the formerly failing beta transition (`status=ok`, lifecycle
`succeeded`, normal `max_iterations_reached`). The complete local checkpoint
passes 311 tests plus 197 subtests, and `pip check` reports no broken
requirements. No billed model evaluation was run. A fresh five-scenario UI run,
including Scenario 4 Turn 3, remains the user-level confirmation.

## Executive result

| Scenario | Formulation | Approval gate | Solve/output | Assessment |
| --- | --- | --- | --- | --- |
| 1. Equipment arm | Correct | Correct | Two complete, nonconverged runs with all plots | Strong pass with result-wording issues |
| 2. Simply supported beam | Stalled on the pin | Not reached | Not run | Formulation defect |
| 3. Polymer bracket | Correct | Correct, but hid a warning | Complete, nonconverged run with all plots | Numerical path passes; warning presentation fails |
| 4. Two-load bracket | Correct through Turn 2 | Correct | Typed numerical failure after iteration 50 | Formulation passes; OC tolerance defect |
| 5. Motion inverter | Stalled on spring regions/units | Not reached | Not run | Supported mechanism journey is incomplete |

Three scenarios reached a validated proposal. Scenarios 1 and 3 then completed
contained solves and produced manifest-backed density, compliance, volume, and
design-change plots. Scenario 4 reached the solver but stopped safely on a typed
numerical error. Scenarios 2 and 5 never crossed the approval boundary, so the
system correctly avoided starting a solve with incomplete state.

## What is working

The evidence supports the following release claims:

- Ordinary engineering language can formulate useful compliance problems without
  exposing solver schemas.
- Whole-edge clamps, coordinate intervals, fractional spans, physical-width
  selectors, pressure directions, multiple stable load IDs, and total-resultant
  conversion work.
- Geometry-derived defaults, explicit `(x,y)` mesh divisions, and casual
  long-side/short-side mesh instructions are retained and disclosed.
- Requested boundary regions are resolved against the actual mesh, with visible
  extents, facet count, measure, and integrated resultant.
- A total force remains a resultant until validation computes the effective
  traction. Scenario 3 reconstructed the requested `20 N` after mesh resolution.
- A proposal does not start a solve until the user sends an explicit approval.
  A mesh revision in Scenario 1 produced a fresh proposal and required fresh
  approval.
- Solver nonconvergence and numerical failure are not disguised as success.
  Partial artifacts survive a failed run, while successful runs expose the
  expected plot gallery.

## Scenario findings

### 1. Lightweight equipment arm

The typed problem, material, units, clamp, right-edge interval, traction, and
material fraction were correct. The automatic `87 × 29` quadrilateral mesh is
close to square in physical space and close to the intended 2,500-element
resolution. Validation disclosed that the requested `80…120 mm` interval became
`82.7586…117.241 mm` on that mesh.

The first approved run and the later user-requested `120 × 40` rerun both
completed and displayed all four plots. Both honestly reported
`converged=false` after 400 iterations. The second mesh resolved the loaded
interval exactly.

Improvements:

- The rerun reply says it “can’t rerun” even while preparing the requested
  revision. It should say that the revision is ready but requires fresh approval.
- The fact-preserving explanation reports both compliance and
  `final objective: 0.0`. In compliance mode that objective field is not a second
  meaningful result. Present compliance as the objective and reserve the signed
  output objective for mechanism runs.
- Both runs ended at the move limit rather than converging. This is not a hidden
  failure, but the default continuation/convergence policy should be evaluated
  before treating these outputs as numerical reference cases.

### 2. Simply supported transfer beam

The model correctly understood the domain, material, units, triangular
`160 × 20` mesh, filter, iteration limit, roller, and inward top pressure. It
also described the requested lower-left pin correctly in prose.

The canonical support state nevertheless retained only a boundary-point pin
without a point or edge-relative center. Readiness therefore stalled on the
internal path `S1.selector.point_or_edge_center`. No focused question asked the
user where the pin was, even though “lower-left corner” and the rectangular
bounds were already known.

Required repair:

1. Retain a named corner as semantic boundary-point state.
2. Resolve that corner deterministically from the known rectangle bounds; do not
   ask the model to calculate coordinates.
3. Add an application-owned human question for any missing readiness path so an
   omitted model question never leaves the user with only an internal field name.
4. Add a regression with negative/nonzero bounds, because `[0,0]` examples can
   hide coordinate-resolution mistakes.

### 3. Near-incompressible polymer bracket

The entire formulation and resultant conversion were correct. The requested
centered `40 mm` patch resolved to `41.6667 mm` on the `100 × 60` mesh, and the
effective traction reintegrated to exactly `20 N`. All output plots were present.
The 300-iteration cap was honored, and the run honestly reported nonconvergence.

The important defect is warning visibility. Validation generated the correct
plane-strain warning for Poisson ratio `0.495`, but the approval request did not
show it. It was available only in the collapsed workflow trace and was also
absent from the final fact ledger.

Required repair:

- Put validation warnings in a prominent approval section before the user can
  approve.
- Carry applicable validation/run warnings into deterministic result evidence and
  the fact-preserving explanation.
- Add UI and approval-text regressions for the near-incompressibility warning.

### 4. Conversational bracket with two loads

Turns 1 and 2 worked well. The assistant retained the incomplete first-turn
facts, asked focused questions, then built one support and two separately
identified loads. Directions, fractional regions, defaults, resolved extents,
integrated resultants, cards, and the diagram were correct.

The transcript approved after Turn 2 and did not submit the documented Turn 3.
Consequently it did **not** test targeted `L1` correction, preservation of `L2`,
replacement of defaults, or stale-approval invalidation. Those checks still need
to be run.

The solve reached iteration 50, changed projection `beta` from 1 to 2, and then
returned `oc_non_descent_gradient`. A disposable diagnostic reproduction showed:

- before the transition, all compliance sensitivities were non-positive;
- after the transition, one of 2,485 values was slightly positive
  (`1.61e-08`) while the dominant negative value was about `-1.23`;
- the resulting OC ratio was only about `-7.27e-05`, but the current absolute
  guard rejects anything below `-1e-12`.

This is a numerical-tolerance defect, not evidence of a materially non-descent
gradient. The guard should remain for real sign failures, but insignificant
positive sensitivity noise should use a scale-aware tolerance and clipping rule.
The exact threshold must be justified with finite-difference and regression
evidence rather than chosen only to make this case pass.

The job itself was contained correctly: the durable lifecycle is `failed`, exit
code is recorded, the last completed iteration is 50, and `response.json`
contains the typed numerical error. As a usability improvement, the lifecycle
artifact could link or summarize that richer error rather than storing only
“worker returned a typed failure.”

### 5. Compliant motion inverter

The model retained the mechanism type, domain, four clamp regions, input
traction, material, units, material fraction, and compliance bound. It then
stalled on the two springs.

The user had already specified both spring regions as the central 20% of known
edges. The current mechanism-spring intent accepts only a raw region expression,
so the assistant asked the user to provide physical coordinate intervals instead
of compiling the known semantic selector. A later stiffness correction exposed a
second gap: spring stiffness is stored as a bare number, with no
dimension-bearing unit conversion between `N/m` and `N/mm`.

Required repair:

1. Give input and output springs first-class, stable semantic entities and edge
   selectors comparable to supports and loads.
2. Compile a centered fraction to the solver region deterministically.
3. Store spring stiffness as a dimensioned quantity with retained display unit
   and deterministic conversion into the active unit context.
4. Show that stiffness is applied per matched directional nodal degree of freedom,
   including matched-node evidence, because its aggregate effect is mesh
   dependent.
5. Add complete formulation, approval, and numerical mechanism acceptance tests.

Until this is implemented, the documentation should not imply that fractional
spring regions and mixed spring units already form a complete natural-language
journey.

## Prioritized repair program

### P0 — blocks a supported end-to-end problem

1. Make the OC non-descent check robust to scale-insignificant sensitivity noise
   at projection transitions, with a numerical regression based on Scenario 4.
2. Add first-class semantic spring selectors and dimension-aware spring
   stiffness so Scenario 5 can reach review and execute.

### P1 — breaks interpretation or informed approval

3. Preserve and deterministically resolve named-corner pins.
4. Guarantee human clarification questions from deterministic readiness gaps.
5. Surface validation warnings in approval and final evidence.
6. Remove the meaningless compliance-mode `final objective: 0.0` presentation.

### P2 — acceptance completeness and usability

7. Improve revision/rerun wording.
8. Run Scenario 4 exactly through Turn 3 after the P0/P1 repairs.
9. Re-run all five journeys in fresh sessions and record a second acceptance
   report.
10. Evaluate convergence defaults using multiple mechanically valid cases before
    changing them.

## Regression evidence required before closure

- Provider-independent formulation cases for named-corner pins, missing-question
  fallback, semantic spring regions, and spring unit conversions.
- Compiler/validator tests for corner-to-point resolution and spring selector
  compilation.
- Approval/UI tests proving warnings are visible without opening the trace.
- Explainer tests distinguishing compliance and mechanism objective semantics.
- A numerical OC test that accepts only scale-insignificant sign noise, plus a
  counterexample that still rejects a material positive gradient.
- A complete mechanism numerical baseline.
- A repeated five-scenario user acceptance run, including Scenario 4 Turn 3.
