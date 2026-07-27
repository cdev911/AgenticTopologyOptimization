# AgenticTopologyOptimization — Project Spec

This is the compact engineering handoff for the portfolio release. The public
story is in `README.md`, the case study is in `docs/project-story.md`, and the
exact contract is in `docs/tool-reference.md`. The complete package-by-package
development record is preserved on branch
`archive/development-record-2026-07-27`.

Last updated: 2026-07-27

## 0. Current Status

- **Release:** portfolio-ready local demonstration of the implemented rectangular
  2D topology-optimization scope.
- **User path:** conversational Streamlit formulation → deterministic compilation
  and mesh validation → explicit approval → contained FEniCSx execution →
  manifest verification → fact-preserving explanation.
- **Verification:** 311 tests plus 197 parameterized subtests passed in the pinned
  Docker environment; `pip check` reported no broken requirements.
- **Language evidence:** the released 53-case boundary-condition gate passed
  53/53 with zero solver starts.
- **Acceptance evidence:** the first five-scenario manual run produced two
  complete solve galleries and exposed named-corner, spring-contract, warning,
  explanation, and projection-transition defects. All were repaired with local
  regressions. The exact former numerical failure completed 60 iterations across
  its beta transition.
- **Presentation:** selected real plots are tracked under `docs/assets/`; raw
  results, `.env`, and user-owned acceptance PDFs remain ignored.
- **Scope:** learning/portfolio software, not certified engineering software or a
  hosted multi-user service.
- **Required work:** none for the portfolio release. Future physics or a repeated
  paid acceptance campaign is a separate, explicitly authorized project.

## 1. Product Contract

The product accepts ordinary descriptions of supported topology-optimization
problems and develops them over multiple turns:

```text
formulate ↔ gather/repair/reformulate
          → finalize typed problem
          → compile
          → validate against the mesh
          → await explicit approval
          → run
          → verify/analyze
          → explain from evidence
```

### Supported agent-safe physics

- rectangular 2D plane strain with unit thickness;
- isotropic single-material compliance and compliant-mechanism optimization;
- full-vector zero clamps;
- zero-component roller and symmetry constraints;
- true boundary-node pins;
- uniform traction, pressure, uniform total resultant, and body force;
- fractional, coordinate, physical-width, and corner-offset edge selectors;
- explicit mechanism input/output springs; and
- volume fraction plus declarative solid/void regions.

### Explicitly unsupported

- agent-safe 3D, non-rectangular domains, and plane stress;
- nonlinear, dynamic, or thermal physics;
- multiple materials;
- mathematical point loads and applied moments;
- varying distributed tractions;
- nonzero prescribed displacements; and
- browser/LLM control of paths, solver profiles, limits, timeouts, or policy.

The legacy FEniTop examples do not expand the conversational contract.

## 2. Architecture and Authority

### Language-model roles

- The live OpenAI Responses formulator interprets ordinary language, proposes a
  small typed turn patch, identifies conflicts, and asks focused questions.
- The retained CrewAI path supports the earlier typed interpreter and constrained
  evidence organization.
- The evidence explainer may organize known fact IDs but may not author numerical
  facts.

### Deterministic roles

- `ProblemDraft` is the canonical multi-turn state.
- Application code owns stable BC/spring IDs, provenance, revision history,
  assumptions, readiness, defaults, compilation, and migrations.
- Validation owns physics, resources, mesh entities, unit conversions, warnings,
  and approval evidence.
- The orchestrator owns transitions, idempotency, approval, execution authority,
  and lifecycle state.
- Analysis owns manifest verification and result facts.

### Tool boundary

`fenitop.tools` exposes exactly:

1. `validate_config`;
2. `run_topopt`; and
3. `analyze_results`.

The Python API, CLI, and MCP transport share Pydantic contracts. The solver layer
does not depend on CrewAI.

## 3. Key Invariants

- Model output is a proposal, never canonical state or execution authority.
- Model-declared readiness is advisory; deterministic readiness is authoritative.
- Every inferred value is visible as a pending assumption until confirmed.
- Optional mesh/filter/iteration values are accepted only when explicit user text
  supports their provenance.
- A validated proposal cannot run without a separate unambiguous approval.
- Any requested change revokes the older proposal before the next model call.
- Validation and FEM assembly use the same boundary resolver.
- Resultant force becomes traction only after authoritative mesh measure and
  thickness are known; reintegration must reproduce the requested resultant.
- Solver workers receive no `OPENAI_*` environment variables.
- Run manifests use relative contained paths and SHA-256 artifact checks.
- Successful solver evidence is immutable; later analysis is derived evidence.
- Explanation fails closed on unknown, missing, or duplicated required fact IDs.

## 4. Defaults

When mesh resolution is omitted:

```text
h = sqrt(domain area) / 50
nx = round(domain width / h)
ny = round(domain height / h)
```

A square is approximately 50 × 50. Ordinary rectangles remain near 2,500 cells
with nearly square elements; extreme aspect ratios retain at least two cells
across the short direction. The default filter radius is 1.5 times the larger
element edge. Every default and its reason is shown before approval.

## 5. Runtime Configuration

Pinned application components include:

- CrewAI `1.15.6`;
- OpenAI Python SDK `2.48.0`;
- Pydantic `2.12.5`;
- Streamlit `1.60.0`; and
- Pint `0.25.3`.

The live formulator defaults to:

- model from `OPENAI_FORMULATION_MODEL` (Compose default `gpt-5.6-sol`);
- medium reasoning;
- strict structured output;
- `previous_response_id` plus application-owned canonical context;
- SDK retries disabled;
- one deterministic patch-repair attempt; and
- one full-context recovery for expired continuation state.

Live calls are billed and are never part of the normal test suite.

## 6. Verification

Local non-billed checkpoint:

```bash
docker compose run --rm -T fenitop python -m pip check
docker compose run --rm -T fenitop pytest -q
```

Coverage includes schemas, adversarial requests, deterministic merging,
provenance, migration, BC semantics, unit normalization, resource admission,
mesh-backed validation, rigid-body constraints, FEM construction, numerical
baselines, solver lifecycle, secret scrubbing, idempotency, manifest integrity,
workflow branches, Responses continuation/recovery, approval invalidation,
fact-preserving explanations, and Streamlit behavior.

The separate live evaluator is formulation-only and imports no solver entry
point. Its fixed 53-case gate is the released evidence; do not spend API credit
repeating it unless the language contract, prompt, model claim, or corpus changes.

## 7. Decision Log

Reverse chronological; only release-level decisions are retained here.

- **2026-07-27 — Curate a portfolio release and archive the build diary:** lead
  the public repository with the useful engineering problem, architecture,
  results, evidence, and transferable lessons. Track selected real plots; keep
  raw runs, secrets, and user acceptance PDFs ignored. Preserve the full
  package-by-package record on
  `archive/development-record-2026-07-27`. Do not claim the unperformed second
  billed/UI acceptance pass.
- **2026-07-27 — Repair acceptance failures at their owning boundaries:** retain
  named corners and springs as first-class semantic state; normalize spring
  stiffness as force/length; provide readiness-owned fallback questions; inherit
  warnings into approval/final evidence; distinguish compliance and mechanism
  explanations; and use a scale-aware OC sign-noise threshold while preserving a
  material-positive rejection.
- **2026-07-27 — Make boundary conditions first-class:** use stable identities,
  partial per-field provenance, semantic edge selectors, typed load quantities,
  unit-aware resultants, component constraints, true boundary-node pins, and
  deterministic preview evidence.
- **2026-07-27 — Make the typed conversational draft the live UI state:** retain
  canonical facts outside the provider, compile only through a ready typed
  handoff, derive workflow identity from user turns plus config, and revoke stale
  approval on any requested change.
- **2026-07-27 — Use the Responses API for live multi-turn formulation:** keep
  provider continuation for reasoning quality while sending canonical state on
  every turn. Use CrewAI where it remains useful rather than forcing one
  framework across all roles.
- **2026-07-27 — Build for real function, not artificial simplicity:** treat
  “learning project” as deployment scope, not permission for a toy. Let measured
  failures and quality determine architecture.
- **2026-07-26 — Require explicit run approval:** a valid formulation stops
  before execution; edits require fresh validation and approval.
- **2026-07-26 — Use deterministic orchestration and evidence-bound
  explanation:** fixed handoffs and expensive side effects do not require model
  judgment; language generation may structure but not alter facts.
- **2026-07-26 — Contain native numerical execution:** use a credential-free
  child process because PETSc/MPI failures exceed ordinary Python exception
  boundaries.
- **2026-07-25 — Bound the product:** ship a local learning/demo workflow, not a
  production engineering or hosted service.

## 8. Implementation Checklist

- [x] Agent-safe solver preparation and three-tool contract.
- [x] Pinned Docker runtime and protected `.env` secret.
- [x] Typed conversational draft, provenance, revisions, and repair loop.
- [x] First-class BCs and mechanism springs with stable identities.
- [x] Dimensional quantities and mesh-dependent resultant conversion.
- [x] Shared mesh resolver and component-aware structural validation.
- [x] Deterministic compiler, defaults, workflow, and explicit approval.
- [x] Contained execution, lifecycle, cancellation, and idempotency.
- [x] Checksum-verified manifests, analysis, and result gallery.
- [x] Fact-preserving LLM explanation.
- [x] Streamlit interface with inspectable state and evidence.
- [x] Deterministic, numerical, live-language, and manual acceptance gates.
- [x] Regression-backed repairs from the first user acceptance run.
- [x] Why-first README, selected real results, concise learning record, and
      portfolio case study.
- [x] Detailed development snapshot preserved on the archive branch.

## 9. Attribution

This repository is a modified derivative of FEniTop by Yingqi Jia, Chao Wang,
and Xiaojia Shelly Zhang. See `NOTICE.md`, `CITATION.cff`, and `LICENSE`.
