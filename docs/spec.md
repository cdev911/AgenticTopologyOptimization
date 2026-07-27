# AgenticTopologyOptimization — Living Project Spec

This file is the compact handoff for future development. The release narrative is
in `README.md`; exact tool behavior is in `docs/tool-reference.md`.

Last updated: 2026-07-26

## 0. Current Status

- **Release**: v1 complete and release-ready.
- **Verification**: 151 tests plus 103 numerical subtests pass in the pinned Docker
  image. Compose config, dependency checks, Streamlit health, the no-credit
  idempotent harness, and documentation links have also passed.
- **Scope**: personal learning/demo workflow, not engineering design software or a
  hosted multi-user service.
- **Just finished**: release cleanup. Documentation was consolidated around the
  final architecture, superseded implementation chronology was removed, upstream
  FEniTop GPL-3.0 attribution/citation was restored, and release metadata was
  aligned.
- **Next action**: no planned v1 work. Select and specify a post-v1 objective before
  changing physics or workflow authority.

## 1. Product

The product is the full plain-language workflow:

```text
interpret → clarify | unsupported | compile
          → validate → run → analyze → explain
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
- volume fraction and declarative 2D selection regions; and
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

- **LLM roles**: typed intent interpretation and constrained evidence
  organization only.
- **Deterministic authority**: compilation, defaults, validation, state
  transitions, side effects, retry bounds, and factual rendering.
- **Hands**: `fenitop.tools` exposes `validate_config`, `run_topopt`, and
  `analyze_results` without depending on CrewAI.
- **Interface**: Streamlit retains session/job state and displays events/evidence;
  it does not own execution state.
- **Execution**: one pinned Docker image; every native solve runs in a separate
  credential-free child process.
- **Handoffs**: exact Pydantic objects and a checksum-verified `RunManifest`; no
  prose or LLM copying between stages.

## 4. Deterministic Defaults

`agentic-defaults-v1` owns omitted numerical preferences and shows every selected
value and reason before execution continues.

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
- Successful solver evidence is immutable; deterministic derived plots are not
  added to the original manifest.
- SHA-256 protects local evidence consistency, not authenticity against an actor
  who can rewrite the entire trusted results root.

## 6. LLM Configuration

The recorded v1 configuration is:

- CrewAI `1.15.6`
- OpenAI `gpt-5.6-terra`
- low reasoning effort
- temperature unset
- strict Pydantic structured output
- two application-owned interpretation attempts with provider retries disabled

The model is environment-configurable through `OPENAI_MODEL`. Model smoke/golden
checks are manual and billed; the deterministic test suite uses canned results.

## 7. Testing

Checkpoint command:

```bash
docker compose run --rm -T fenitop python -m pip check
docker compose run --rm -T fenitop pytest -q
```

Coverage includes schemas, adversarial requests, physics and mesh validation,
resource calibration, numerical baselines and sensitivities, contained process
lifecycle, secret scrubbing, idempotency, manifest/artifact integrity, CLI/MCP
transports, workflow branches, fact preservation, and Streamlit behavior.

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
  around 2,500 cells, disclose every compiler choice, and proceed without a
  confirmation gate. Reason: reduce user tuning while keeping assumptions visible.
- **2026-07-26 — Contained execution**: use one pinned Docker image but launch each
  solve in a credential-free child process. Reason: Python exceptions cannot
  contain native PETSc/MPI failures.
- **2026-07-26 — Model/runtime pin**: CrewAI `1.15.6`, Pydantic `2.12.5`,
  Streamlit `1.60.0`, and `gpt-5.6-terra`; accept dependency compatibility changes
  only after the complete numerical suite passes.
- **2026-07-25 — Learning/demo scope**: this is not a production engineering or
  hosted service. Favor clear, testable learning boundaries over platform scale.

## 10. Open Questions

None for v1. Post-v1 objectives must be selected before implementation.

## 11. Implementation Checklist

- [x] Stage 0 — pinned Docker model/runtime and protected API secret.
- [x] Stage 1 — typed intent, compiler, orchestrator, tools, and explainer.
- [x] Stage 2 — chat UI, progress/cancellation, and inspectable evidence.
- [x] Stage 3 — release narrative, scenarios, limits, rationale, and commands.
- [x] Release cleanup — compact docs, upstream attribution/license, metadata,
      verification, and clean tracked tree.
