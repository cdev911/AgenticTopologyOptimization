# AgenticTopologyOptimization — Living Project Spec

This file is the compact handoff for future development. The release narrative is
in `README.md`; exact tool behavior is in `docs/tool-reference.md`.

Last updated: 2026-07-27

## 0. Current Status

- **Release**: v1 numerical/execution behavior remains complete, and the post-v1
  conversational Phase 1 is now the released Streamlit entry path.
- **Verification**: 303 tests plus 197 subtests pass in the pinned Docker image,
  and `pip check` reports no broken requirements. This includes historical
  numerical baselines, exact resultant integration, deterministic selector/error
  behavior, UI approval behavior, and solver lifecycle tests.
- **Scope**: personal learning/demo workflow, not engineering design software or a
  hosted multi-user service. This scope does not authorize intentionally weak or
  toy implementations: functional quality and measured reliability are the
  learning objective.
- **Just implemented**: Streamlit now retains a typed `FormulationSession` and
  uses `ConversationFormulator` plus the live Responses adapter. It displays
  accepted facts, missing fields, unconfirmed assumptions, capability limits,
  deterministic conflicts, rejected patches, and inspectable provenance/revision
  history. A ready `FormulationStep` crosses into compilation only through the
  orchestrator's typed `prepare_formulation()` method. Requested changes revoke
  the prior approvable proposal before the new model call; explicit approval
  remains the only path to solver execution.
- **BC program**: the first user-led acceptance exposed boundary conditions as
  the main formulation bottleneck. Section 8 records the now-completed
  two-increment improvement: conversational mesh-verifiable distributed
  conditions followed by separately versioned roller/symmetry/pin physics.
- **Just implemented**: BC Work Package 0 adds a provider-independent semantic
  evaluation contract, deterministic grader, and 53-case versioned corpus. It
  includes the supplied six-turn transcript, both previously failing complete
  prompts, partial/multiple/corrected BCs, units, resultants, pressure, and
  capability boundaries. This checkpoint does not change live formulation or
  solver behavior.
- **Just implemented**: BC Work Package 1 adds provider-independent partial BC
  entities, application-owned monotonic `S…`/`L…` IDs, per-field provenance and
  revision history, atomic create plus targeted update/delete/confirm operations,
  typed pending confirmations, basic BC readiness, and explicit migration from
  legacy list facts. `ProblemDraft` and `FormulationTurn` carry the new optional
  state/patch; Package 4 owns finalization and Package 5 now populates this state
  from the live OpenAI transport.
- **Just implemented**: BC Work Package 2 pins Pint 0.25.3 in Docker and adds
  provenance-bearing length/force/stress unit facts, typed dimensional
  normalization with retained display values, deterministic global/edge-local
  direction and pressure resolution, and explicit traction versus total-resultant
  state. Resultants normalize as force and remain conversion-deferred until the
  mesh resolver supplies an authoritative boundary measure.
- **Just implemented**: BC Work Package 3 versions canonical `AgentSafeConfig`
  as 2.0 and the three-tool contract as 5.0.0. It adds stable-ID fixed,
  uniform-traction, and uniform-resultant variants; expert and rectangle-edge
  selectors; deterministic 1.1 migration; one validation/FEM facet resolver;
  requested/resolved extent, measure, centroid, normal, error, unit, and
  conversion evidence; resultant round-trip verification; a traction/resultant
  numerical-equivalence solve; and a visible plane-strain warning for
  `poisson_ratio >= 0.49`.
- **Just implemented**: BC Work Package 4 makes complete, confirmed first-class
  BC state authoritative at the conversational compiler boundary. Fraction,
  coordinate, physical-width, and corner-offset selectors compile
  deterministically; explicit-unit tractions normalize and resultants remain
  forces for mesh conversion; stable IDs survive into schema 2.0. Legacy live
  facts migrate once at this boundary. Approval fails closed without successful
  geometry evidence and shows requested versus resolved selectors, measures,
  normals, conversions, integrated resultants, units, thickness, and warnings.
  The separate explicit run-approval transition is unchanged.
- **Just implemented**: BC Work Package 5 versions the live contract as
  `formulation-system-v2` / `openai-responses-v2`. Its 6,491-character strict
  transport carries flat create/update/delete/confirm operations, compact JSON
  field values, stable-ID corrections, partial BCs, and targeted confirmations
  without exposing solver schemas. The prompt distinguishes traction, pressure,
  resultant, and point semantics; retains semantic selectors; asks
  highest-information questions; and recognizes gated supports honestly.
  Deterministic code migrates a legacy browser-session draft before its first v2
  turn. The v3 Sol/medium live gate passes 6/6 with zero solver starts.
- **Just implemented**: BC Work Package 6 adds provider-independent partial and
  validated BC cards, stable-ID correction guidance, human-readable approval
  rows, complete BC provenance/revisions in the existing expander, and a
  deterministic SVG rectangle driven only by successful validation evidence.
  Requested extents are dashed, resolved supports/loads are solid, load arrows
  use effective traction, and cards expose mesh extent/measure, conversions,
  integrated resultants, thickness, units, and warnings.
- **Just implemented**: BC Work Package 7 adds a dedicated formulation-only
  53-case billed evaluator for `boundary-condition-evals-v6`, semantic
  normalization and forbidden-behavior grading, bounded retries for connection,
  timeout, and provider internal-server errors only, and deterministic
  application normalization of constant traction uniformity and named-edge-center
  retention. The final clean
  Sol/medium run passed 53/53 with zero solver starts, zero context recoveries,
  and no transport retries. The complete deterministic/numerical gate also
  passes, closing the first BC increment.
- **Just implemented**: BC Work Package 8 versions canonical config `2.1`, tool
  contract `5.1.0`, live adapter v4, and BC corpus v7. It adds explicit
  zero-displacement components, boundary-node selection with visible snap
  evidence, component-aware rigid-body/duplicate/load validation, Dolfinx
  subspace constraints, roller/symmetry/pin compilation, UI cards/preview, and a
  real roller+pin numerical baseline. Valid 2.0 inputs migrate deterministically.
  The clean v7 Sol/medium gate passed 53/53 with zero solver starts.
- **Release bookkeeping**: Work Package 9 audits the public version/migration
  narrative, documents a mechanically valid roller-plus-pin acceptance
  conversation and the exact planar rigid-body rows, and verifies the secret,
  Compose, dependency, complete test, and UI-health boundaries. The unchanged
  live contract retains Package 8's clean 53/53 v7 result rather than spending
  API credit to repeat the same measurement.
- **Next action**: use the released workflow for broader user-led acceptance and
  turn any newly observed failure into a versioned regression case.

## 1. Product

The product is the full plain-language workflow:

```text
formulate ↔ gather/repair/reformulate → finalize typed problem and BC state
          → compile → validate → await approval → run → analyze → explain
```

Users describe a rectangular 2D topology-optimization problem in chat. The system
collects missing physics, discloses deterministic defaults, performs contained
numerical execution, and returns a fact-preserving explanation with inspectable
evidence.

This repository is a modified derivative of FEniTop by Yingqi Jia, Chao Wang, and
Xiaojia Shelly Zhang. See `NOTICE.md`, `CITATION.cff`, and `LICENSE`.

## 2. Scope and Non-Goals

Supported agent-safe physics:

- rectangular 2D plane strain with unit thickness;
- isotropic single-material compliance or compliant-mechanism optimization;
- full-vector zero clamps, zero-component rollers/symmetry boundaries, and true
  boundary-node pins;
- distributed boundary traction, pressure, uniform total resultant, and body
  force;
- volume fraction and declarative 2D selection regions;
- relative centered/spanned rectangle-edge segments for tractions; and
- explicit compliant-mechanism input/output springs.

Not supported in v1:

- agent-safe 3D, non-rectangular domains, plane stress, nonlinear/dynamic/thermal
  physics, or multiple materials;
- nonzero prescribed displacement or mathematical point loads;
- parallel tool execution, multiple concurrent solves, or browser/LLM control of
  paths, solver profiles, limits, timeouts, or safety policy.

The legacy FEniTop examples may cover capabilities outside the natural-language
contract. They do not expand the agent-safe surface.

## 3. Architecture

- **LLM roles**: live conversational formulation and constrained evidence
  organization. The one-shot v1 typed interpreter remains for regression and
  compatibility paths, not the Streamlit entry.
- **Deterministic authority**: compilation, defaults, validation, state
  transitions, side effects, retry bounds, and factual rendering.
- **Hands**: `fenitop.tools` exposes `validate_config`, `run_topopt`, and
  `analyze_results` without depending on CrewAI.
- **Interface**: Streamlit retains typed formulation/session/job state and
  displays accepted facts, unresolved items, public provenance, events, and
  evidence; it does not own readiness, validation, approval, or execution state.
- **Execution**: one pinned Docker image; every native solve runs in a separate
  credential-free child process.
- **Handoffs**: exact Pydantic objects and a checksum-verified `RunManifest`; no
  prose or LLM copying between stages.
- **Post-v1 live formulation**: Responses API turn patches merge into an
  application-owned partial draft with source-turn provenance. Provider
  continuation improves multi-turn reasoning but never replaces canonical draft
  state. Model-declared readiness is advisory; deterministic repair, readiness,
  semantic resolution, and typed finalization remain authoritative.
  This is now the live UI path.

## 4. Deterministic Defaults

`agentic-defaults-v1` owns omitted numerical preferences and shows every selected
value and reason before requesting execution approval.

For an omitted mesh:

```text
h = sqrt(domain area) / 50
nx = round(domain width / h)
ny = round(domain height / h)
```

A square is approximately `50 × 50`; ordinary rectangles remain near 2,500 cells
with nearly square elements. Extreme aspect ratios retain at least two cells
across the short direction. The default filter radius is 1.5 times the larger
element edge. Resource validation remains authoritative.

## 5. Trust Boundaries

- `AgentSafeConfig` contains physics only.
- Application code owns paths, IDs, PETSc profiles, resource ceilings, timeout,
  cancellation, rendering, and idempotency.
- The solver worker receives no `OPENAI_*` variables.
- In-process and durable idempotency prevent duplicate solves across UI reruns and
  process restarts.
- A validated proposal is not executable. Only the application-owned
  `approve()` transition, triggered by an unambiguous user green light, creates a
  runnable workflow state.
- Optional model-produced mesh/filter/iteration values are reset unless a
  deterministic text check finds that the user explicitly requested that class of
  preference.
- Formulation-only width/height/origin, relative support edges, and long/short
  mesh counts become solver intent only through deterministic resolution.
- Responses continuation IDs are application state, not execution authority.
  Stored provider state may recover conversation reasoning, but every call
  receives the canonical draft and an expired ID gets only one full-context
  fallback.
- Successful solver evidence is immutable; deterministic derived plots are not
  added to the original manifest.
- SHA-256 protects local evidence consistency, not authenticity against an actor
  who can rewrite the entire trusted results root.

## 6. LLM Configuration

The legacy v1 interpreter configuration retained for regression checks is:

- CrewAI `1.15.6`
- OpenAI `gpt-5.6-terra`
- interpreter prompt `intent-system-v3`
- low reasoning effort
- temperature unset
- strict Pydantic structured output
- two application-owned interpretation attempts with provider retries disabled

The model is environment-configurable through `OPENAI_MODEL`. Model smoke/golden
checks are manual and billed; the deterministic test suite uses canned results.

The live Streamlit formulation configuration is:

- direct OpenAI Responses API through `openai==2.48.0`;
- `gpt-5.6-sol` through `OPENAI_FORMULATION_MODEL`;
- medium reasoning through `OPENAI_FORMULATION_REASONING_EFFORT`;
- `reasoning.context=all_turns` and `previous_response_id`;
- prompt `formulation-system-v4`, composed from the released v3 base plus the
  Package 8 component-support addendum;
- strict `OpenAIFormulationTurn` output with a 7,162-character schema and flat
  first-class BC operations;
- SDK retries disabled, one application-owned deterministic patch repair, and one
  continuation-expiry full-history recovery; and
- `store=true` for response continuation under provider retention, while the local
  typed draft remains authoritative.

On the v2 six-scenario gate, both Sol/medium and Terra/medium passed. Across ten
turns, Sol recorded 30,938 input, 4,576 output, and 1,613 reasoning tokens in
75.347 seconds; Terra recorded 30,306 input, 2,784 output, and 236 reasoning tokens
in 34.688 seconds. Sol remains the default because the current suite establishes
a minimum capability bar rather than broad out-of-distribution equivalence;
Terra/medium is a measured lower-latency option.

The v3 first-class BC gate was rerun on Sol/medium after Package 5. All six
scenarios passed across ten calls with 51,726 input, 7,921 output, 3,578 reasoning,
and 43,317 cached tokens in 147.888 seconds, with no context recoveries and zero
solver starts. The earlier v2 comparison remains the evidence for the Sol/Terra
default decision; v3 measures the changed BC contract.

The Package 7 evaluator uses `boundary-condition-evals-v6` and the
`formulation-system-v3` / `openai-responses-v3` live contract. It imports no
orchestrator or solver entry point and reports `solver_executed=false`. Transport
timeouts, connection failures, and provider internal-server errors receive up to
three attempts; semantic failures receive exactly one. The final clean run passed
53/53 over 64 API calls with 393,486 input, 45,631 output, 18,314 reasoning, and
364,507 cached tokens in 804.885 seconds, with no context recoveries, no
transport retries, and zero solver starts.

The Package 8 evaluator uses `boundary-condition-evals-v7` and the
`formulation-system-v4` / `openai-responses-v4` live contract. The fixed corpus
changes only the three newly supported roller/symmetry/pin outcomes and removes
their obsolete capability limits. The clean Sol/medium run passed 53/53 over 64
API calls with 412,596 input, 45,137 output, 17,813 reasoning, and 377,843 cached
tokens in 851.383 seconds, with no context recoveries, no retries, and zero
solver starts.

## 7. Testing

Checkpoint command:

```bash
docker compose run --rm -T fenitop python -m pip check
docker compose run --rm -T fenitop pytest -q
```

Coverage includes schemas, adversarial requests, physics and mesh validation,
resource calibration, numerical baselines and sensitivities, contained process
lifecycle, secret scrubbing, idempotency, manifest/artifact integrity, CLI/MCP
transports, workflow branches, Responses continuation/recovery, deterministic
patch repair, partial semantic draft resolution, typed formulation handoff,
approval invalidation on requested changes, visible draft/capability/error states,
fact preservation, and Streamlit behavior. `scripts/formulation_live_eval.py` is
a separate billed gate and never starts the solver.

## 8. Planned Boundary-Condition Formulation Upgrade

The conversational boundary-condition upgrade is authorized for implementation.
It is split into two increments so that language/geometry reliability can be
measured independently from new finite-element support physics.

### 8.1 Required user experience

- A user may describe supports and loads in any logical order and refine them over
  multiple turns. The UI assigns stable labels such as `S1` and `L1`, so “move L1
  upward” changes that load rather than replacing every load.
- Each BC retains partial facts independently: type, boundary edge, along-edge
  location, extent, direction, quantity semantics, magnitude, units, and
  provenance. A missing magnitude must not erase a known location.
- Common spatial descriptions include whole edges; centered, upper/lower, or
  left/right fractions; coordinate intervals; physical widths; distances from a
  corner; and equivalent corrections. Ambiguous words such as “width” on a
  vertical edge require clarification when context does not resolve them.
- The conversation distinguishes traction, pressure, total resultant force, and
  a mathematical point load. It never silently treats one as another.
- Every proposed value absent from the user's description is stored as a typed
  pending assumption with its derivation. The user can confirm it, change it, or
  continue supplying facts. Formulation confirmation remains separate from the
  final run approval.
- Before approval, the UI presents human-readable BC cards and a deterministic
  rectangle preview showing support segments, load segments/arrows, stable IDs,
  requested versus mesh-resolved extents, and any load conversion.

### 8.2 Conversational BC model

`ProblemDraft` will keep ordinary scalar facts but replace the monolithic
`supports`, `support_edges`, and `tractions` formulation paths with first-class
BC entities:

- application-owned stable IDs and BC kind;
- field-level `explicit`, `derived`, `assumption`, or `confirmed` provenance;
- typed create/update/delete/confirm operations instead of replacing a whole
  list;
- semantic rectangle-edge selectors that remain meaningful before geometry or
  mesh divisions are complete; and
- explicit incomplete, ambiguous, resolved, and confirmed states.

The compact Responses transport will continue to carry JSON strings decoded by
path-specific validators; the provider will not receive the recursive final
solver schema. The application allocates IDs and rejects references to missing
entities, conflicting updates, unsupported confirmation, and destructive
whole-list replacement.

### 8.3 Deterministic boundary and load resolution

Boundary selection becomes a shared tool-layer operation rather than duplicated
agent arithmetic:

1. Preserve a requested continuous rectangle-edge segment as edge plus normalized
   or coordinate interval.
2. Enumerate and order the actual boundary facets on that edge.
3. Select the contiguous facets whose midpoints fall in the requested interval.
   If a positive interval contains no midpoint, select the single closest facet
   and issue a visible resolution warning.
4. Report the requested interval, resolved facet bounds, facet count, physical
   boundary measure, centroid, outward normal, and resolution error.
5. Use exactly the same resolver in validation and in `form_fem`; validation and
   execution must never disagree about which facets carry a BC.

The generic region DSL remains available for already-supported expert regions.
First-class rectangle-edge selectors are preferred for conversational BCs because
their discretization policy and evidence are explicit.

Uniform loads will support two input quantity kinds:

- an effective distributed traction vector; or
- a desired total resultant vector distributed uniformly over the resolved
  segment under the existing unit-thickness model.

For a resultant, deterministic code calculates
`traction = resultant / (resolved boundary measure × thickness)`. The geometry
report records both values and recomputes the integrated resultant as a
verification check. An exact point load remains unsupported; the workflow may
offer a finite patch and a uniform resultant as an explicitly confirmed
reformulation.

A typed mechanical unit context will retain the user's original units and
normalize supported length, force, and stress quantities into one solver unit
system using a Docker-pinned `Pint` dependency. Missing units inferred from
context—for example N, mm, and MPa—are assumptions, not silent conversions.
Implicit thickness remains one length unit and must be stated when converting a
total force.

### 8.4 Required tool changes

The existing three-tool boundary remains; no fourth LLM-callable preview tool is
needed. `validate_config` will be extended so the UI and approval layer can use
its authoritative evidence.

- Add versioned, strict BC identifiers, rectangle-edge selectors, and uniform
  traction/resultant load variants to `AgentSafeConfig`.
- Add one shared mesh-aware boundary resolver used by `validate_config` and
  `fenitop.fem`.
- Extend each geometry entity record with measure, centroid, requested/resolved
  extents, normal, resolution warning, effective traction, and integrated
  resultant where applicable.
- Continue rejecting empty selectors, overlapping distributed loads, loads erased
  by void zones, and under-constrained models.
- Add a calibrated plane-strain warning for Poisson ratios near `0.5`; do not
  silently reject a value inside the existing mathematical range.
- Target `AgentSafeConfig` schema `2.0` and tool contract `5.0.0` for the new
  canonical BC representation. Keep a tested, deterministic `1.1` input migration
  adapter for current full-vector clamp and traction fixtures; do not make the LLM
  perform migration.

### 8.5 Capability boundary and second increment

| User phrase or model | First increment | Tool physics change |
| --- | --- | --- |
| Full edge or finite-segment clamp | Supported | Mesh-aware selector/evidence only |
| Uniform traction on an edge segment | Supported | Selector/evidence extension |
| Uniform total resultant over a finite segment | Supported after unit/distribution confirmation | New versioned load variant |
| Normal pressure or tangential load on a named rectangle edge | Deterministically converted to a global vector | No new FEM physics |
| Mathematical point load | Not represented directly; offer a finite loaded patch | Remains unsupported |
| Roller or symmetry support | Recognized but not aliased to a clamp | Component-wise zero-DOF contract required |
| True point-node pin | Recognized but not approximated by a clamped facet | Node selector plus component-aware validation required |
| Nonzero prescribed displacement | Not part of this improvement | Remains backlog |

After the first increment passes its gate, the second increment may version
`DirichletBC` for zero-valued component constraints and boundary-node selectors.
Implementation must use component subspaces in Dolfinx, populate the existing
rigid-body rank calculation with the actual constrained components, detect
duplicate/conflicting constrained DOFs, and add numerical pin/roller baselines.
This is a physics-contract change and receives an explicit checkpoint before code
changes; the formulator must already recognize these terms honestly even while
the capability is gated.

#### Package 8 component-support checkpoint

The first-increment 53/53 gate is closed, so the following second-increment
contract is authorized:

- Canonical `AgentSafeConfig` advances from `2.0` to `2.1`; tool envelopes and
  exported schemas advance from `5.0.0` to `5.1.0`. Valid `2.0` inputs migrate
  deterministically to `2.1`, just as `1.1` inputs already migrate through the
  application-owned adapter.
- Existing `kind="fixed"` remains the full-vector zero clamp. A new
  `kind="zero_displacement"` carries an ordered, unique nonempty component list
  drawn from `["x", "y"]`. Values remain exactly zero; nonzero prescribed
  displacement remains unsupported.
- Facet supports continue using `rectangle_edge` or `expert_region`. A new
  `boundary_node` selector carries one requested physical point. It is valid only
  on the rectangle boundary and resolves to exactly one nearest boundary mesh
  node. Validation reports the requested point, resolved point, and snap
  distance; a nonzero snap is visible as a warning before approval.
- Language semantics remain outside the solver schema. A normal roller or
  symmetry condition on left/right compiles to component `x`; on bottom/top it
  compiles to `y`. Explicit `roller_x` and `roller_y` mean constrained
  displacement component, not permitted travel direction. A true pin compiles to
  both components on one `boundary_node`; it is never compiled as a clamped
  facet.
- Geometry validation expands every support into actual `(node, component)` rows.
  Rigid-body rank uses those rows. Repeated constraints on the same row are
  rejected as `duplicate_constrained_dof`; the zero-only contract has no
  contradictory values to reconcile.
- The existing `traction_on_fixed_support` rejection is retained. For component
  supports, a load is rejected only when every nonzero load component is
  constrained over all of its selected nodes; a tangential effective component
  remains valid.
- Dolfinx applies component constraints through vector subspaces. The gate
  requires schema/migration tests, node-resolution evidence, duplicate and
  rigid-body tests, FEM construction tests, real numerical roller/pin baselines,
  UI/approval coverage, and a versioned live language gate with zero solver
  starts.

### 8.6 Work packages and gates

0. **Corpus first** — turn the supplied cantilever transcript, both previously
   failing complete prompts, and at least 50 boundary-language fixtures into
   versioned expected semantic outcomes. Include multiple loads, corrections,
   coordinate/fraction/physical spans, direction synonyms, pressure, resultant
   force, unit ambiguity, point-load reformulation, roller/pin recognition, and
   deliberate contradictions.
1. **BC draft and patches** — implement stable entities, partial per-field
   provenance, confirmation records, reference/correction operations, readiness,
   migration from legacy list facts, and pure deterministic tests.
2. **Units and semantic load model** — normalize dimensional quantities, retain
   display units, represent traction versus resultant explicitly, and add
   deterministic direction/pressure resolution without requiring solver-native
   wording from the user.
3. **Tool resolver and evidence** — add the shared facet resolver, versioned safe
   schema, resultant normalization, enriched geometry report, near-incompressible
   warning, and mesh/numerical tests.
4. **Finalization and approval** — compile only complete, confirmed BC entities;
   show requested/resolved selectors and conversions in the approval facts; keep
   the existing separate green-light transition unchanged.
5. **Prompt and live adapter** — teach the formulation role to extract semantic
   BC facts, preserve uncertainty, ask the highest-information question, propose
   recorded assumptions, and refer to stable IDs. Arithmetic, facet selection,
   unit conversion, readiness, and side effects remain deterministic.
6. **UI and preview** — replace raw BC paths with human cards, draw the
   deterministic pre-run diagram from validated evidence, expose detailed
   provenance in the existing expander, and support precise correction language.
7. **First-increment release gate** — require the complete deterministic suite,
   unchanged historical numerical baselines, exact resultant integration within
   numerical tolerance, deterministic selector/error tests, and 100% passage of
   the fixed billed BC conversation gate with zero silent semantic changes and
   zero solver starts.
8. **Component-support checkpoint and increment** — review first-increment
   evidence, explicitly authorize the versioned roller/symmetry/pin contract,
   implement it, and repeat schema, geometry, rigid-body, FEM, UI, and live gates.
9. **Documentation and learning** — update `README.md` only for shipped behavior,
   update `docs/tool-reference.md` with exact contracts and formulas, record real
   failures and tradeoffs in `LEARNING.md`, refresh this spec, and commit at each
   verified boundary.

Remaining unplanned backlog: optional recorded offline UI scenarios, nonzero
prescribed displacement, agent-safe 3D/MPI, additional materials/physics, and
hosted or multi-user deployment. Each physics addition still requires a versioned
contract, prompt update, numerical tests, and an explicit decision.

## 9. Decision Log

Reverse chronological; final decisions only.

- **2026-07-27 — Close the BC program at the verified Package 9 boundary**:
  treat release bookkeeping as a contract audit rather than another physics
  increment. Keep older schema/prompt numbers where they accurately describe
  historical package boundaries, while making current 2.1/5.1.0 contracts and
  deterministic migration explicit. Publish a nonduplicating bottom-roller plus
  mid-left-pin conversation and the exact planar rigid-body constraint rows.
  Verify secret exclusion, Compose, locked dependencies, the complete suite, and
  UI health; do not rerun a billed language gate when its contract and corpus are
  unchanged from the clean Package 8 53/53 result. Reason: release claims should
  be traceable to the layer that proves them, and repeating an unchanged billed
  measurement adds cost without new evidence.
- **2026-07-27 — Version zero-component supports and true boundary-node pins**:
  advance canonical config to 2.1 and tool contracts to 5.1.0 while migrating
  valid 2.0 inputs deterministically. Keep `fixed` for full clamps; represent new
  support physics as `zero_displacement` plus canonical `x`/`y` components.
  Resolve a requested boundary point to one nearest mesh node with visible snap
  evidence, compute rigid-body rank from actual `(node, component)` rows, reject
  duplicate rows, and apply components through Dolfinx vector subspaces. Compile
  edge-normal rollers/symmetry to the normal component and true pins to both
  components at a node. Reason: language labels must not leak ambiguous roller
  terminology into solver physics, and a pin must never become a clamped facet.
  A real FEM test caught the trusted normalizer dropping `components`; preserving
  that field changed the compliance baseline from the accidentally clamped model
  to the intended component model. The final v7 live gate passed 53/53 with zero
  solver starts.
- **2026-07-27 — Distinguish named sides from bare directional locations**: an
  unqualified “right side/edge/face/boundary” denotes the complete rectangle
  edge, while “the load on the right” identifies the edge but not its extent.
  Qualifiers such as “somewhere,” “center,” “near,” or “segment” also prevent a
  whole-edge inference. Enforce this in application-owned BC normalization and
  prompt v4. Do not infer a global N-mm-MPa system solely from a
  capability-limited `N mm` moment. Reason: two 52/53-style live differences
  exposed an inconsistent selector convention and an irrelevant unit assumption;
  focused cases and the final full gate now pass.
- **2026-07-27 — Preserve named edge centers independently of selector
  completeness**: when a current-turn source quote names the center/middle of an
  already identified edge, deterministically retain `selector.center=0.5` even
  while kind and extent remain unresolved. If a later model patch repeats the
  same value as an assumption, preserve the stronger explicit/derived fact.
  Retry provider internal-server errors with the same bounded policy as
  connection/timeouts, never semantic failures. Reason: the supplied six-turn
  transcript showed that a known center could otherwise disappear until a later
  span arrived, while a single provider 5xx invalidated an otherwise 52/53 run.
  With these changes the clean billed gate passed 53/53 with zero solver starts.
- **2026-07-27 — Redundant vector direction is equivalent only when provably
  consistent**: when an axis-aligned load vector exactly implies the separately
  retained direction label, ignore that duplicate label during live grading;
  retain and fail any contradictory label. Reason: `[0,-1]` already carries
  complete direction semantics, while accepting arbitrary vector-plus-direction
  pairs would hide real conflicts. The post-fix clean gate reached 52/53 before
  this regression; the next full attempt was invalidated by a persistent provider
  rate limit rather than semantic behavior.
- **2026-07-27 — The BC release gate grades engineering meaning and separates
  provider health**: execute the fixed 53-case corpus through a dedicated
  formulation-only live evaluator. Canonicalize only proven-equivalent selector
  encodings and redundant exact-point labels; retain exact load/support facts,
  clarifications, assumptions, capability limits, forbidden transformations, and
  the no-solver rule. Retry only provider connection/timeouts, never semantic
  failures. Make constant traction uniformity an application-owned deterministic
  derivation once its finite selector is complete. Reason: the gate exposed
  evaluator crashes, stale expectations, provider outages, harmless selector
  spelling variation, and genuine model-basis variation. Release evidence must
  distinguish all five rather than prompt-tune or rerun until a convenient pass.
- **2026-07-27 — Pre-run BC presentation is a pure evidence view**: render
  partial first-class BCs as stable-ID cards with retained semantic facts,
  missing fields, pending confirmations, capability limits, and direct
  “Change S1/L1 …” guidance. After validation, replace raw selector rows with
  human-readable cards and draw a deterministic SVG rectangle from the compiled
  config joined to `GeometryReport` by BC ID. Show requested continuous spans
  separately from resolved mesh spans, use effective traction only for arrow
  direction, expose conversions and integrated resultants in text, and retain
  exact field provenance/revisions in the existing expander. Fail closed when
  successful per-BC geometry evidence is missing; do not move resolution,
  readiness, approval, or execution into Streamlit. Reason: a preview must
  explain the same facts FEM will use, stable labels make corrections precise,
  and duplicating geometry arithmetic in the UI would create a contradictory
  fourth mechanics implementation.
- **2026-07-27 — Live BC language uses flat semantic operations**: version the
  live prompt/continuation contract to v2 and replace legacy support/traction
  updates with a strict flat transport for BC create, update, delete, and
  confirmation operations. Carry arbitrary field values as compact JSON strings,
  allocate canonical IDs only in deterministic merge code, include the current
  boundary catalog and pending confirmations on every call, and migrate any
  legacy browser-session facts before the first v2 turn. Teach the model to
  preserve partial entities, distinguish traction/pressure/resultant/point
  meaning, keep fraction/physical selector semantics without arithmetic, ask the
  highest-information question, and name gated capabilities without aliasing
  them. Treat a new conflicting implication as a conflict—not a correction—unless
  the user signals correction intent or answers a prior conflict question.
  Reason: a recursive solver schema is unnecessary and expensive, model-owned IDs
  or migration would corrupt provenance, and the live v3 failure showed that
  fluent language still needs an explicit correction-authority rule. The final
  Sol/medium v3 gate passed 6/6 with zero solver starts.
- **2026-07-27 — Finalization consumes first-class BC state, approval consumes
  mesh evidence**: make a ready `ProblemDraft`, not the legacy `ProblemIntent`,
  the conversational orchestrator handoff. Compile ordinary problem facts and
  complete, confirmed BC entities through separate typed paths into canonical
  schema 2.0. If no first-class state exists, migrate legacy BC facts exactly
  once; otherwise first-class state is authoritative. Require explicit mechanical
  units for native boundary loads, preserve traction/resultant quantity kind and
  stable IDs, and deterministically resolve fraction, coordinate, physical-width,
  and corner-offset selectors. Render approval only from successful validation
  and pair every requested selector with its authoritative mesh-resolved facets,
  extent, measure, normal, conversion, integrated resultant, units, thickness,
  and warning. Do not alter the separate explicit run green-light transition.
  Reason: finite clamps and resultants cannot be represented honestly by the
  legacy intent, a temporary fake intent would corrupt the audit trail, and
  approval should show what the solver will apply rather than only what was
  requested.
- **2026-07-27 — Validation and FEM share one facet-resolution policy**: version
  canonical `AgentSafeConfig` as 2.0 and the complete tool boundary as 5.0.0.
  Represent stable-ID fixed, uniform-traction, and uniform-resultant BCs with
  expert-region or rectangle-edge selectors. Select rectangle-edge facets by
  ordered midpoint inclusion; if a positive interval contains no midpoint, use
  the single closest facet and report a warning and resolution error. Both
  validation and FEM call the same resolver. Convert a resultant using resolved
  measure times one-length-unit thickness, reintegrate it as a pre-run check, and
  require explicit mechanical units. Accept schema 1.1 through deterministic
  migration with an honest unlabeled consistent-unit sentinel, but return only
  canonical 2.0 and disallow resultants for that sentinel. Warn at plane-strain
  Poisson ratio 0.49 or above rather than rejecting values below 0.5. Reason:
  duplicated selectors can drift, valid sub-facet intent should not become a
  zero-load failure, invented migration units would be false provenance, and
  mathematical material validity does not prevent volumetric locking.
- **2026-07-27 — Unit normalization stops before mesh-dependent physics**: pin
  Pint 0.25.3 in the Docker runtime and represent length, force, and stress as
  provenance-bearing draft facts. Retain original display values/units beside
  JSON-safe normalized values; do not pass Pint objects across application
  boundaries. Treat effective traction/pressure as stress and total resultant as
  force. Resolve unambiguous global and edge-normal directions deterministically,
  but reject directions without a sign or named edge. A normalized resultant is
  semantically ready and execution-deferred until the shared mesh resolver
  supplies boundary measure; only Package 3 may compute
  `traction = resultant / (measure × one-length-unit thickness)`. Keep legacy
  finalization authoritative for this checkpoint. Reason: dimensional validity
  does not identify load meaning, inferred units need the existing confirmation
  contract, and using a requested span before facet resolution could apply the
  wrong total force.
- **2026-07-27 — BC identity and confirmation are application-owned state**:
  allocate monotonic support/load IDs in deterministic merge code from
  turn-local create references; never accept model-selected canonical IDs and
  never reuse an ID after deletion. Store each BC field with its own provenance
  and revision, and express confirmation as a typed operation against an existing
  assumption rather than restating a value. A bare generic confirmation is valid
  only with one pending BC assumption; multiple assumptions require explicit
  “confirm all” or targeted confirmation. Keep legacy facts and first-class BC
  state side by side through an explicit migration function while legacy
  finalization remains authoritative. Reason: stable references are needed for
  corrections, field-level state prevents partial-load loss, confirmation must
  not mutate proposed physics, and a staged migration avoids changing the live
  or solver path accidentally.
- **2026-07-27 — Grade BC meaning before changing the provider or solver**: add a
  provider-independent `boundary-condition-evals-v1` contract and 53-case corpus
  before integrating new live draft fields. Grade exact retained BC semantics,
  coded clarifications/assumptions/capability limits, prohibited silent changes,
  and absence of solver execution rather than assistant wording. Reason: the
  corpus must drive the intermediate representation, and the supplied transcript
  already proves that fluent prose can coexist with missing canonical load facts.
- **2026-07-27 — Boundary conditions become first-class semantic entities with
  mesh evidence**: implement stable per-BC identities, partial field provenance,
  semantic edge selectors, typed load quantity semantics, unit-aware uniform
  resultant conversion, and deterministic pre-run visualization. Move
  rectangle-edge facet selection into one shared resolver used by both validation
  and execution, and enrich `validate_config` evidence rather than adding a fourth
  LLM-callable tool. Keep mathematical point loads unsupported and offer only an
  explicitly confirmed finite-patch reformulation. Plan component-wise
  roller/symmetry and point-node pin support as a second versioned increment with
  a separate physics checkpoint. Reason: the live transcript showed that
  monolithic load lists lose known partial facts, solver-native wording makes the
  conversation repetitive, total-force conversion cannot be verified without
  units and resolved boundary measure, and a syntactically valid selector can
  still select different—or zero—mesh facets.
- **2026-07-27 — The typed conversational draft is the live UI state**:
  Streamlit retains `FormulationSession`, renders accepted facts and unresolved
  items directly, and places detailed public provenance/revisions in an expander.
  Only a `ready_for_review` `FormulationStep` can enter the orchestrator through
  `prepare_formulation()`. The workflow identity is derived from ordered user
  turns plus the compiled configuration, not variable assistant prose. Any
  requested parameter change clears the prior approvable outcome before calling
  the model. Reason: the UI must expose the conversation's canonical truth without
  becoming readiness or execution authority, harmless wording variation should
  not alter idempotency, and a failed revision must never leave stale parameters
  eligible for a later “yes.”
- **2026-07-27 — Responses API with a quality-first model default**: use the
  Responses API, `previous_response_id`, persisted all-turn reasoning, strict
  output, and the application-owned canonical draft for post-v1 formulation.
  Default to Sol/medium; expose Terra/medium as a measured faster option. Both
  passed all six v2 conversations. The full comparison measured Sol at 75.347
  seconds and Terra at 34.688 seconds over ten turns, with similar input but fewer
  Terra output/reasoning tokens. Reason: Responses directly supports the
  multi-turn reasoning behavior this role needs; the small suite proves a floor
  but is not broad enough to trade away the flagship model's capability by
  default.
- **2026-07-27 — Formulation facts must preserve unresolved semantics**: retain
  domain origin/width/height, relative support edges, and long/short mesh counts
  before mapping them into strict solver coordinates. Allow unsupported features
  to be named while a supported reformulation remains in `gathering`. Use a
  strict `value_json` transport because Pydantic's unconstrained `JsonValue`
  produced an OpenAI-invalid empty schema; decode immediately through the existing
  field validators. Reason: live evaluations showed that monolithic bounds forced
  invented origins and repeated questions, x/y mesh fields could not honestly
  retain relative preferences, negotiable point loads did not fit a terminal
  unsupported state, and the initial live call failed before inference with
  `invalid_json_schema`.
- **2026-07-27 — Solve the real problem and preserve the learning**: “learning
  project” describes the purpose and deployment scope, not a reason to keep the
  implementation artificially simple. Choose frameworks, APIs, models, and
  architecture by measured functional quality, reliability, safety, and
  understandability. Keep a dedicated `LEARNING.md` beginning with agent-safe tool
  preparation and continuing through orchestration, real failures, and design
  changes. Reason: the useful learning is how a functioning system was reached,
  including why simpler or earlier approaches failed; CrewAI and other components
  are means to that outcome rather than goals that must be preserved at the
  expense of it.
- **2026-07-27 — Provider-independent formulation core first**: implement the
  partial draft, small turn-patch contract, provenance validation, revision
  history, deterministic readiness, strict finalization, and evaluation seeds
  before choosing CrewAI versus Responses API state. Reason: conversation behavior
  and safety invariants should be testable without API credit, and a stable core
  lets provider/model experiments compare the same task rather than different
  architectures.
- **2026-07-27 — Conversational problem formulation is required**: Phase 1 must
  accept ordinary logical descriptions rather than requiring a near-schema-shaped
  prompt. It must accumulate a visible problem draft over multiple turns, explain
  its current understanding, ask useful questions, accept corrections, and recover
  from encoding/validation failures without presenting them as user failures.
  Creativity is allowed in interpretation and proposing explicitly labeled
  assumptions, never in silently inventing physics or numerical facts. The exact
  API/model/state design remains an open question.
- **2026-07-26 — Explicit approval, optional-value provenance, and filter
  roundoff**: stop every validated request in an `awaiting_run_approval` state;
  recognize only unambiguous whole-message approval; reinterpret changes and
  require fresh approval. Strip model-produced mesh, cell, filter, and iteration
  preferences unless the user text explicitly mentions them. After a converged
  filter solve, accept at most `1e-5` density roundoff, clip it to `[0,1]`, and
  continue rejecting larger violations. Reason: an expensive solve requires a
  user green light, strict schema shape does not prove optional-value provenance,
  and a `-4.83e-7` discretization undershoot is not a physical density failure.
  This supersedes the earlier no-confirmation decision.
- **2026-07-26 — Normalize strict-output edge sentinels**: accept `region=none` as
  the unused nullable sentinel only when a valid `edge_segment` is present, and
  allow `span_fraction=1.0` for a whole edge. Reject a standalone `none` region and
  any real region combined with an edge segment. Reason: strict structured output
  can materialize nullable union alternatives, while semantic authority still
  requires one effective traction location. Both reported prompts now pass billed
  live interpretation plus deterministic mesh validation.
- **2026-07-26 — Relative edge segments and result gallery**: represent relative
  traction spans as typed `edge`, `center_fraction`, and `span_fraction` intent;
  deterministically compile them through domain bounds into the existing region
  DSL. Display only known analysis PNG roles that resolve inside the verified run
  directory, with downloads; add signed mechanism-objective history while treating
  compliance as the objective for compliance minimization. Reason: models should
  interpret percentage language but not own geometry arithmetic or filesystem
  presentation authority.
- **2026-07-26 — Release cleanup**: consolidate documentation around the shipped
  v1 architecture, remove superseded implementation/production planning, and add
  the upstream GPL-3.0 license, authors, and paper citation. Reason: a release
  should explain the product and obligations without carrying a build diary.
- **2026-07-26 — Thin Streamlit UI**: keep typed workflow/job state in
  `st.session_state`, poll durable lifecycle state, and derive cancellation
  identity in trusted application code. Reason: UI reruns must not own side
  effects.
- **2026-07-26 — Fact-preserving explainer**: the model returns an evidence-ID
  section plan only; deterministic code validates completeness and renders
  immutable fact text. Reason: prompt-only anti-hallucination is insufficient.
- **2026-07-26 — Deterministic orchestration**: replace the proposed three-agent
  tool pipeline with typed application-owned transitions. Reason: fixed tool
  handoffs and expensive side effects do not require model judgment.
- **2026-07-26 — Visible geometry-derived defaults**: derive near-square meshes
  around 2,500 cells and disclose every compiler choice. The original decision to
  proceed without confirmation was superseded by the explicit-approval decision
  above.
- **2026-07-26 — Contained execution**: use one pinned Docker image but launch each
  solve in a credential-free child process. Reason: Python exceptions cannot
  contain native PETSc/MPI failures.
- **2026-07-26 — Model/runtime pin**: CrewAI `1.15.6`, Pydantic `2.12.5`,
  Streamlit `1.60.0`, and `gpt-5.6-terra`; accept dependency compatibility changes
  only after the complete numerical suite passes.
- **2026-07-25 — Learning/demo scope**: this is not a production engineering or
  hosted service. Favor clear, testable learning boundaries over platform scale.

## 10. Open Questions

- No architecture question blocks the released BC increments.
- Nonzero prescribed displacement, mathematical point loads, applied moments,
  and varying tractions remain separate future physics contracts.

## 11. Implementation Checklist

- [x] Stage 0 — pinned Docker model/runtime and protected API secret.
- [x] Stage 1 — typed intent, compiler, orchestrator, tools, and explainer.
- [x] Stage 2 — chat UI, progress/cancellation, and inspectable evidence.
- [x] Stage 3 — release narrative, scenarios, limits, rationale, and commands.
- [x] Release cleanup — compact docs, upstream attribution/license, metadata,
      verification, and clean tracked tree.
- [x] Post-release fix — deterministic relative edge segments and verified final
      design/objective plot gallery.
- [x] Live regression — nullable `none` sentinel normalization and whole-edge
      100% spans.
- [x] Post-release safety fix — explicit pre-run approval, optional-preference
      provenance, and filtered-density roundoff handling.
- [x] Post-v1 Phase 1 foundation — conversational typed draft, provenance-aware
      merge, revision history, readiness, strict final conversion, and eval seeds.
- [x] Learning record — tool preparation, numerical containment, orchestration,
      real regression lessons, and quality-first development principle.
- [x] Post-v1 live formulation — conversation-state adapter, structured repair
      feedback, and measured Terra/Sol comparison.
- [x] Post-v1 UI migration — conversational draft display, error-specific UX,
      final compile/validate handoff, and approval regression verification.
- [x] BC improvement planning — language taxonomy, first-class draft design,
      mesh-aware tool changes, unit/resultant semantics, preview, phased physics
      expansion, and acceptance gates.
- [x] BC Work Package 0 — 53-case versioned boundary-language corpus, strict
      semantic observation contract, and deterministic safety grader.
- [x] BC Work Package 1 — stable partial BC entities, monotonic application IDs,
      per-field provenance/revisions, typed create/update/delete/confirm patches,
      readiness, pending confirmations, and explicit legacy migration.
- [x] BC Work Package 2 — typed unit context and semantic traction/resultant
      resolution.
- [x] BC Work Package 3 — shared mesh boundary resolver, versioned tool contract,
      enriched evidence, and numerical warnings.
- [x] BC Work Package 4 — first-class finalization, deterministic BC compilation,
      authoritative approval evidence, and unchanged explicit run gate.
- [x] BC Work Package 5 — versioned live prompt, compact first-class BC adapter,
      legacy-session migration, mocked regressions, and passing v3 live gate.
- [x] BC Work Package 6 — human BC cards, requested/resolved preview, and precise
      correction UX.
- [x] BC Work Package 7 — complete deterministic, numerical, UI, and 53/53 billed
      live first-increment gate with zero solver starts.
- [x] BC Work Package 8 — versioned component-support checkpoint,
      roller/symmetry/point-pin implementation, numerical/UI coverage, and 53/53
      v7 live gate.
- [x] BC Work Package 9 — shipped-behavior documentation, learning record, and
      release verification.
