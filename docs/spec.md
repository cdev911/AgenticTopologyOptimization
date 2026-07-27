# AgenticTopologyOptimization — Living Project Spec

This file is the compact handoff for future development. The release narrative is
in `README.md`; exact tool behavior is in `docs/tool-reference.md`.

Last updated: 2026-07-27

## 0. Current Status

- **Release**: v1 numerical/execution behavior remains complete, and the post-v1
  conversational Phase 1 is now the released Streamlit entry path.
- **Verification**: 202 tests plus 109 numerical subtests pass in the pinned Docker
  image. Compose config, dependency checks, Streamlit health, the no-credit
  idempotent harness, and documentation links have also passed.
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
- **Next action**: perform user-led live UI acceptance with varied natural problem
  descriptions and turn every observed conversation or presentation failure into
  a versioned evaluation and deterministic regression test before adding physics.

## 1. Product

The product is the full plain-language workflow:

```text
formulate ↔ gather/repair/reformulate → finalize strict intent
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
- full-vector zero clamps;
- distributed boundary traction and body force;
- volume fraction and declarative 2D selection regions;
- relative centered/spanned rectangle-edge segments for tractions; and
- explicit compliant-mechanism input/output springs.

Not supported in v1:

- agent-safe 3D, non-rectangular domains, plane stress, nonlinear/dynamic/thermal
  physics, or multiple materials;
- roller/component supports, nonzero prescribed displacement, point loads, or
  total-force semantics;
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
  semantic resolution, and final strict intent validation remain authoritative.
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
- prompt `formulation-system-v1`;
- strict `OpenAIFormulationTurn` output with a 2,729-character schema;
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

## 8. Post-v1 Backlog

No item below is authorized or designed yet:

- optional recorded UI scenario for offline/live-demo reliability;
- roller supports or nonzero prescribed displacement;
- agent-safe 3D and MPI execution;
- additional materials/physics; and
- hosted or multi-user deployment.

Any physics addition requires a new versioned contract, prompt update, numerical
tests, and explicit user decision.

## 9. Decision Log

Reverse chronological; final decisions only.

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

- No architecture question blocks live UI acceptance.
- The provenance presentation decision is resolved: accepted facts, assumptions,
  capability limits, conflicts, selected defaults, and the final approval question
  are directly visible; exact source quotes and revision records are in an
  inspectable expander.

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
