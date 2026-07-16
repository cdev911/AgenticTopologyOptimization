# AgenticTopologyOptimization — Project Spec

This is a living working document, not a one-time design doc. Every time we make a
decision or hit a checkpoint, we add to it. When a past decision changes, we go back
and edit the relevant section rather than leaving stale info in place — the Decision
Log is the only place history is preserved on purpose.

Last updated: 2026-07-26

## 0. Current Status (read this first; update it last, every session)

- **Stage**: Stage 0 model environment/secrets complete. Deterministic agentic
  Stage 1 is now unblocked.
- **As of**: 2026-07-26.
- **Just finished**: Stage 0 bookkeeping and learning documentation. The README
  now contains the fresh-clone Docker/CrewAI recipe, separate Git/build-context/
  runtime secret boundaries, version and API checks, billed smoke/golden commands,
  common failure interpretation, and the tested Pydantic/Rich rollback recipe.
  The committed `.env.example` is blank; the real `.env` remains ignored by Git
  and excluded from Docker builds.
- **Architecture decisions updated** (§3, §6, §6a): deterministic orchestration
  replaces the three-agent tool-calling pipeline; solver execution stays in the
  same image/container but moves to a child process without the API key;
  clarification is allowed for incomplete/ambiguous requests without adding a
  pre-run confirmation gate; and `gpt-5.6-terra` replaces the old
  `gpt-4.1-mini` default.
- **Next action**: begin Stage 1 (§11) with `agentic/intent.py` and its tests, then
  implement the interpreter and deterministic orchestrator in checklist order.
- **If you're an AI assistant picking this up cold**: read this whole file before
  doing anything, then summarize your understanding of current state + proposed
  next step back to the user before acting. See `CLAUDE.md`/`AGENTS.md` at the
  repo root for the full protocol.

## 1. Vision & Scope

This is a learning project. The goal is to build an **agentic workflow** where an LLM
acts as the "brain" that operates a topology-optimization solver as its "hands." The
topology optimization code (`fenitop`) is not the product — it is the trusted tool
layer the deterministic workflow operates after LLM interpretation. The product is
the whole workflow: taking a plain-English structural design request, clarifying
missing physics, turning it into a valid solver config, running the solve, and
explaining the result back to the user.

Primary use: personal learning + demoing to others. Not a production service, not
multi-tenant, not optimized for scale.

## 2. Non-Goals (for now)

- **This is explicitly not a production workflow.** It is a demonstration /
  learning repo whose purpose is to show what was understood and built, not to be
  hardened, scaled, or operated as a real service. Every design choice in this spec
  should be read against that bar — "good enough to demo and explain clearly" beats
  "production-grade."
- Not building a hosted/multi-user product.
- Not optimizing for large-scale or high-throughput solving.
- Not committing to a single agent framework long-term — CrewAI is the v1 choice for
  learning purposes; the architecture (see §3) keeps the door open to swapping it.
- Not choosing the cheapest model at the expense of interpretation/tool-use
  reliability. This is still a personal demo, so retries and execution remain
  bounded and usage remains observable, but the original fixed $5/model-cost
  optimization is no longer an architectural constraint (see §6).

## 2a. Documentation Philosophy: spec.md vs README.md

Two documents, two audiences, deliberately not merged:

- **`docs/spec.md` (this file)** — the working decision log, for us. Chronological,
  allowed to contain half-finished thinking, options considered and rejected, open
  questions. Optimized for "why did we decide X," not for readability by a stranger.
- **`README.md`** — the showcase document, for anyone visiting the repo (reviewers,
  other learners, demo viewers). Should read as a clear narrative explanation of
  what this project is, why it's built the way it is, and what was learned —
  architecture, key design decisions and their reasoning, how to run the demo. It
  should demonstrate understanding, not just list setup commands.

As stages complete in this spec, the corresponding understanding should get
distilled into README.md in narrative form. README.md is not rewritten yet — it
predates the `agentic/` layer and should be revisited once that layer exists, so it
describes the whole picture rather than being edited twice.

## 3. Architecture Overview: Brain vs Hands

- **Hands** — `fenitop/` (complete for the supported v1 scope). The domain library:
  FEM/topology optimization solver, plus `fenitop/tools/` — three hardened,
  framework-agnostic operations (`validate_config`, `run_topopt`,
  `analyze_results`). The contract is typed, failure-contained, numerically
  explicit, and ready for a CrewAI adapter.
- **Brain** — `agentic/` (new, not yet created — see Decision Log 2026-07-25). The
  natural-language interpretation and optional result-explanation layer. It does
  not decide whether validation/run/analysis happen or hand those deterministic
  transitions from one LLM agent to another.
- **Orchestrator** — planned deterministic state machine in `agentic/` (CrewAI Flow
  where useful, plain typed Python for domain state). It owns:
  `interpret → clarify-or-compile → validate → launch worker → analyze → explain`.
  Exact Pydantic objects/manifests move between stages; no LLM retypes a normalized
  config, output path, or run envelope.

Design principle: `fenitop` should never need to know an agent framework exists.
`agentic/` depends on `fenitop.tools`, never the other way around.

**Runtime placement (updated 2026-07-26): one Docker image/container, separate
processes.** Streamlit/CrewAI and Dolfinx share the pinned image for simple demo
deployment, but every expensive/native solve runs in a child worker process. The
parent assigns trusted paths/limits, strips `OPENAI_API_KEY` from the worker
environment, captures progress, enforces timeout/cancellation, and translates
exit/signal/crash state into a job lifecycle manifest. This supersedes the earlier
same-container/**same-process** decision: Python exception handling cannot contain
PETSc/MPI native aborts or OOM, and the current solver's stdout would corrupt a
stdio protocol.

Agent workflow v1 is serial. Existing legacy scripts may still demonstrate MPI,
but `run_topopt` does not advertise MPI until run IDs, output ownership, and
rank-zero response behavior have an MPI-specific implementation and test.

Practical implications, not yet done:
- CrewAI `1.15.6` is pinned in `pyproject.toml` and the complete closure is pinned
  in `requirements/runtime.lock`; the image was rebuilt successfully.
- The container reaches api.openai.com through CrewAI, and
  `scripts/stage0_model_smoke.py` proves a billed, schema-constrained Terra call.
  Terra rejects an explicit `temperature=0`, so its default temperature is left
  unset; with `response_format` CrewAI `1.15.6` returns the validated Pydantic
  instance directly.
- `OPENAI_API_KEY` is supplied to the container via `docker-compose.yml`'s
  `env_file:` pointing at an untracked `.env` file — never baked into the image or
  committed. Local `.env` files are excluded from Git and Docker build contexts.
  The solver worker receives a sanitized environment without this key.

## 4. Repository Layout

### Current (as of 2026-07-26)
```
fenitop/                  # domain library: solver + tools/
  tools/                  # hardened validate/run/analyze tool layer
config/                   # example configs (beam_2d, mechanism_2d)
scripts/                  # example CLI entry points (config-driven + legacy hardcoded)
tests/                    # unittest-based, split dolfinx-free vs Docker-only
results/                  # gitignored solver outputs
docs/spec.md              # this file
docs/tool-reference.md    # final tool capabilities, contracts, and operations
```

### Planned addition (not yet created)
```
agentic/
  intent.py                # typed ProblemIntent and ready/clarify/unsupported result
  interpreter.py           # LLM-backed natural-language interpretation only
  orchestrator.py          # deterministic state machine / CrewAI Flow
  explainer.py             # optional LLM explanation over deterministic analysis
  llm.py                   # env-driven, pinned model/provider selection
  prompts/                 # versioned interpretation/explanation prompts
tests/agentic/             # mocked-LLM + deterministic-flow integration tests
```

Nothing under "Planned addition" exists yet. We create it deliberately after the
Stage 0 environment/model checkpoint—not just as a folder shell.

## 5. Tool Contract (the "hands" API)

Source of truth: `fenitop/tools/contracts.py`,
`fenitop/tools/config_models.py`, and the three tool implementations.

- Public contract version is `4.0.0`; agent-safe config schema is `1.1`. Every
  request and response is a strict Pydantic model with unknown fields forbidden
  and structured issue records.
- The LLM-visible inputs are exactly `validate_config(config)`,
  `run_topopt(config)`, and `analyze_results(run_manifest)`.
  `AgentSafeConfig` describes physics only. `TrustedValidationPolicy`,
  `TrustedRunPolicy`, and `TrustedAnalysisPolicy` are Python/application-owned and
  are not exported in the MCP schema.
- Serialized regions are a bounded, recursive, discriminated 2D DSL; positional
  mechanism springs were replaced with named models. All vectors are exactly two
  finite values. JSON/source strings are never evaluated.
- Supported physics is explicit: rectangular 2D plane strain, unit thickness,
  distributed boundary traction, consistent user units, and full-vector zero
  clamps. Nonzero prescribed displacements and component-wise supports are
  rejected for v1.
- `validate_config_tool` performs structural/cross-field physics checks,
  independent pure-arithmetic resource admission, then trusted mesh-backed checks
  for all support/load/spring/passive entities and their relevant conflicts. Its
  typed geometry report includes entity counts and 2D bounds.
- `ResourceLimits` independently bounds elements (250,000), nodes (300,000),
  displacement DOFs (600,000), iterations (2,000), solver-weighted work
  (500 million units), estimated peak memory (2,048 MiB), output (1,024 MiB), and
  wall time (900 s). These defaults are application-owned. Estimates incorporate
  the selected iterative/direct profile and compliance/mechanism solve count and
  are calibrated in `tests/fixtures/resource_calibration.json`.
- `run_topopt_tool` always re-validates and internally compiles safe physics with
  trusted PETSc/output settings. Run paths, identifiers, rendering, timeout, and
  safety overrides cannot be supplied in its public request.
- A successful run checks every PETSc elasticity/adjoint/filter solve, finite
  fields/metrics/gradients/updates, density bounds, and explicit optimizer status.
  Iteration zero and every update are evaluated states. Final metrics/artifacts
  share one state and include grayness, binarization, beta, and continuation.
- A successful `run_topopt_tool` response includes a canonical `RunManifest`;
  `analyze_results_tool` accepts only that manifest. It recovers normalized config
  and analysis evidence without duplicated caller fields. Its relative artifact
  inventory is checked for trusted-root containment, symlinks, existence,
  completeness, size, and SHA-256 before any content is parsed.
- Real MCP input and output schemas are hash-snapshotted. Because pinned MCP 1.28.1
  generated the outer function argument model with extra fields ignored, server
  registration explicitly switches that generated model to `extra="forbid"` and
  tests both its schema and runtime behavior.

The final hardened boundary is built around six guarantees:

1. **Two capability levels**: a strict `AgentSafeConfig` (physics only, region DSL
   only) and trusted application-owned run policy (paths, solver profile, limits,
   timeout, rendering, idempotency). Lambda strings are removed from JSON config
   loading; hardcoded Python examples may still use internal callables.
2. **Typed/versioned boundaries**: strict Pydantic requests, structured
   warnings/errors, response models, and a self-contained `RunManifest`. Every
   JSON-shaped public request returns a schema-valid envelope; no exception crosses.
3. **Complete validation**: pure arithmetic rejects unsafe mesh/work sizes before
   Dolfinx; mesh-backed checks cover supports, loads, springs, passive zones,
   overlaps, rigid motion, and supported physics.
4. **Numerical truthfulness**: elasticity/adjoint/filter convergence and finite
   values are checked; initial/final artifacts and metrics describe one explicit
   evaluated state; optimizer failure/continuation status is surfaced.
5. **Contained execution**: fresh path-contained run directories, no agent safety
   override, idempotent kickoff, serial v1 worker subprocess, timeout/cancellation,
   sanitized environment without the API key, and atomic lifecycle manifests.
6. **Clean composition/transports**: Tool 2 emits everything Tool 3 needs; Tool 3
   never asks an LLM to retype a config/path; stdout is one JSON response; real CLI
   and MCP integration tests verify framing.

These guarantees are covered by the pinned 107-test suite. The detailed operational
and field-level contract is in `docs/tool-reference.md`.

## 6. Brain: LLM Provider Strategy

**Decision updated 2026-07-26: OpenAI API, default model `gpt-5.6-terra`,
environment-configurable and pinned/recorded for reproducible demo runs.**

Why:

- CrewAI (the open-source pip package) has no usage limits or cost of its own — it's
  local orchestration code. All spend risk comes from the LLM API it calls, not from
  CrewAI. (CrewAI's separate hosted "AMP" cloud product, with its own free/paid
  execution tiers, is not something we opted into.)
- A ChatGPT Plus subscription does **not** include API credits — API billing is
  entirely separate from the consumer subscription.
- The user explicitly prefers a stronger current-generation model over minimizing
  token cost for this personal project. `gpt-5.6-terra` is the current balanced
  intelligence/cost candidate rather than the flagship/highest-cost choice. The
  model will be set through `OPENAI_MODEL`, recorded in each workflow trace, and
  changed only through config — never hard-wired throughout prompts/tasks.
- The Stage 0 golden intent gate passed on both the selected model and cheaper
  current alternative. The final recorded configuration is OpenAI
  `gpt-5.6-terra` through CrewAI `1.15.6`, low reasoning effort, temperature
  unset, and strict Pydantic structured output. The public model catalog exposes
  the `gpt-5.6-terra` ID without a distinct dated snapshot, so that ID is the
  recorded pin. `gpt-5.6-luna` also passed but remains only a lower-cost comparison.
- Local LLM (Ollama) was considered and deliberately deferred, not rejected — small
  local models are meaningfully less reliable at strict JSON-schema tool-calling and
  multi-step self-correction, which would shift effort into prompt-engineering
  workarounds rather than saving effort. May be added later as a $0/offline fallback
  profile; CrewAI's LiteLLM backend makes this a config change, not a redesign.

Budget is no longer the architecture driver, but operational bounds still matter:
pin exact CrewAI/model configuration once implemented; cap interpretation retries;
use structured outputs and low reasoning effort for intent extraction (Terra does
not accept `temperature=0`, so temperature remains unset); log token usage; and
never let an LLM retry an expensive solver side effect directly.
Solver idempotency/resource limits are tool policy, not LLM-cost policy.

## 6a. Agentic Layer Design: Agents, Tasks, Process

**Decisions updated 2026-07-26:**

- **Orchestration shape**: deterministic state machine, not three agents mapped 1:1
  to three tools. The fixed path is:
  `interpret → clarify/unsupported/ready → compile → validate → run worker →
  analyze → explain`. CrewAI Flow may host the state transitions, but typed Python
  domain state is authoritative and remains testable without an LLM.
- **LLM roles**:
  - Intent Interpreter — the essential LLM step. Produces a typed `ProblemIntent`
    or a clarification/unsupported result from free text.
  - Result Explainer — optional LLM step over Tool 3's deterministic evidence.
    It may improve presentation but cannot change metrics or quality flags.
  There is no Solver Operator agent. Launching an expensive deterministic function
  does not need judgment, and an extra agent would add mutation/retry risk.
- **Exact handoffs**: tools/manifests are passed as Pydantic/Python objects. No task
  output is reparsed from prose and no LLM is asked to copy a normalized config,
  filesystem path, safety setting, or prior tool envelope.
- **Clarification without confirmation**: there is still no approval gate for a
  ready, supported request, but the interpreter must not invent missing
  problem-defining physics. It returns one of:
  - `ready` — enough explicit/safely-derived information to compile the supported
    problem;
  - `needs_clarification` — asks focused chat questions and does not solve;
  - `unsupported` — explains the tool capability mismatch and does not solve.
  This supersedes the earlier “fully autonomous with accepted
  semantically-wrong-but-valid risk” posture. Clarification is required-input
  collection, not a confirm-every-run human gate.
- **Execution authority**: application code owns run IDs, filesystem roots, solver
  profiles, timeout/cancellation, idempotency, and resource ceilings. These are not
  LLM tools/arguments. The worker process never receives the API key (§3, §5).
- **Interface**: a minimal **Streamlit** web UI, not a CLI. (Gradio was the other
  candidate; Streamlit picked for broader familiarity in data/ML-style demos — this
  is a low-stakes, reversible pick.) Build order remains: verify the deterministic
  orchestrator with a plain harness and mocked LLM before the UI. The UI shows an
  inspectable event trace (intent, clarification, normalized config, validation,
  resource estimate, progress, analysis evidence), not hidden chain-of-thought.
  Long solver work is represented as job state so a Streamlit rerun/refresh does
  not duplicate it.
- **Input shape (decided 2026-07-25)**: **free-text chat only**, not a structured
  form and not a hybrid with a JSON-paste escape hatch. The user describes the
  problem in plain English; clarification remains chat, not a fallback form.
  Interpretation targets the smaller `ProblemIntent`, while a deterministic
  compiler supplies trusted numerical/execution defaults and creates
  `AgentSafeConfig`. This preserves the natural-language learning goal without
  asking the model to author PETSc options, paths, or other non-semantic details.

## 7. Testing Strategy

Existing (unittest-based, already in place):
- Dolfinx-free unit tests: bounded region DSL and adversarial source/shape/numeric
  cases, strict agent-safe/config/response contracts, real generated MCP schema
  snapshots, independently calibrated resource estimates/limits, Tool 1
  structural/cross-field checks, narrative generation, and typed Tool 2→Tool 3
  handoff against committed fixture logs.
- Docker-only tests (need dolfinx/PETSc/MPI): geometry checks, Tool 2 end-to-end
  smoke runs, root-discovered serial compliance/mechanism numerical baselines,
  complete support/load/spring/passive-zone entity/conflict checks, filter/stop
  semantics, central directional finite differences, PETSc/filter/optimizer fault
  injection, configured/passive initialization, final-grid metric consistency,
  cleanup, config-hash guards, artifact consistency, isolated worker execution,
  credential scrubbing, path/idempotency/capacity behavior, and real process-group
  timeout/cancellation/crash recovery, checksum-verified Tool 2→Tool 3
  composition, CLI JSON purity, and actual stdio MCP composition.
- Test entry point: `docker compose run --rm -T fenitop python -m unittest discover -v`.
  `tests/__init__.py` makes nested discovery reliable; zero collection exits 5.
  Current result: all 107 tests pass with no expected failures
  (`docker compose run --rm -T fenitop python -m unittest discover -v`, 18.706
  seconds at the final tool checkpoint).

Remaining additions for agent-workflow compatibility:
- **Hardened-tool suite (passed)**: contract/schema, generated
  adversarial JSON, path/security, geometry/numerics/fault injection, subprocess
  lifecycle, CLI stdout, actual MCP stdio, artifact integrity, and
  manifest-driven Tool 2→Tool 3 composition are implemented.
- **Measured resource calibration**: frozen in
  `tests/fixtures/resource_calibration.json`; refresh deliberately if the runtime,
  estimator, solver profiles, or output state model changes.
- **Mocked-LLM deterministic-flow integration test**: canned
  ready/clarification/unsupported interpreter outputs exercise the whole state
  machine without a real API call or solver duplication.
- **Golden-scenario smoke test** (manual/occasional, real API call, not in CI): one
  real selected-model run across a small set of supported, ambiguous, and
  unsupported requests, to catch prompt/model drift over time.

## 8. Reviewer Notes / Backlog (things to come back to)

Flagged during initial planning (2026-07-25), not yet actioned:
- Observability: show a structured event/evidence trace for demo storytelling
  (intent, clarification, validation, resource estimate, progress, analysis), not
  private chain-of-thought or raw verbose reasoning.
- Confirm `docker-compose.yml` allows outbound internet access from the container
  for the LLM API call.
- Demo-day reliability: consider caching/recording a known-good run so a live
  presentation doesn't depend on a live LLM call working under pressure.
- License/attribution check: confirm the origin/license of the `fenitop` codebase
  before presenting this publicly as a demo.

Final tool review findings (2026-07-26):

- No architectural reversal is needed. Deterministic orchestration,
  same-container/separate-process solves, clarification without confirmation,
  application-owned execution authority, and manifest-only analysis remain the
  right fit for this personal learning/demo scope.
- Mechanism spring stiffness is applied per matched directional nodal DOF and the
  output functional sums matched nodal displacements. It is therefore
  mesh/region-dependent, not a mesh-independent total spring constant. The future
  intent compiler must preserve this explicit contract and show matched counts;
  changing the formulation would be a deliberate future physics/version change.
- Manifest/artifact SHA-256 provides local evidence integrity, not authenticity
  against an actor who can rewrite the trusted results root. The child process
  contains native crashes but shares the parent's container-level OOM boundary;
  independent memory admission remains the primary defense.
- Solver evidence in a successful manifest is immutable. Analyzer-created plots
  are deterministic derived outputs, are not part of the original solve manifest,
  and may be regenerated.
- Use the focused test tiers in `docs/tool-reference.md` during implementation and
  the full pinned suite only at checkpoints or after cross-cutting executable
  changes.

## 9. Decision Log

Reverse-chronological. Each entry: date, decision, why, status.

- **2026-07-26** — Close Stage 0 with a documented, Docker-only learning recipe.
  Keep `.env` outside both Git history and the Docker build context, inject it only
  into the parent at runtime, and document verification, billed model checks, and
  dependency rollback in README. Reason: make the checkpoint reproducible and
  explain the security/runtime boundaries before agentic code is added. Status:
  done.
- **2026-07-26** — Complete Stage 0 and retain `gpt-5.6-terra` as the default.
  The reproducible golden gate in `scripts/stage0_golden_intents.py` evaluated
  supported-ready, ambiguous/needs-clarification, and unsupported requests.
  Terra and the cheaper Luna alternative both matched all expected statuses;
  Terra remains selected because this project prioritizes interpretation
  reliability over minimum token price. Final configuration: CrewAI `1.15.6`,
  OpenAI provider, model `gpt-5.6-terra`, low reasoning effort, temperature unset,
  strict Pydantic output. Status: done; Stage 1 unblocked.
- **2026-07-26** — Pin CrewAI `1.15.6` in the Docker runtime only. To satisfy its
  published dependency bounds, move Pydantic `2.13.4` → `2.12.5`,
  pydantic-core `2.46.4` → `2.41.5`, and Rich `15.0.0` → `14.2.0`; regenerate the
  full lock and accept the compatibility change only with the complete hardened
  suite passing. Rollback recipe if later checks fail: remove CrewAI from
  `pyproject.toml`, restore those three original pins, regenerate
  `requirements/runtime.lock`, rebuild, and rerun all 107 tests. Reason: isolate
  the learning framework in the reproducible project image without changing the
  Mac's global Python environment. Status: installed; all 107 tests pass. A live
  `gpt-5.6-terra` structured-output call passes with low reasoning effort,
  temperature unset, and a Pydantic instance returned directly by CrewAI.
- **2026-07-26** — Remove the completed tool-hardening workstream plan and
  phase-by-phase history. Preserve the final contract in `docs/tool-reference.md`
  and keep one consolidated completion decision here. Reason: completed
  implementation scaffolding was obscuring the current Stage 0/1 handover.
  Status: done.
- **2026-07-26** — Freeze the hardened tool boundary at contract `4.0.0`, config
  schema `1.1`, and manifest `1.0`. The agent surface is physics-only and typed;
  validation is semantic, resource-aware, and mesh-backed; numerical success
  requires converged finite sub-solves and consistent evaluated state; execution
  uses contained credential-scrubbed child processes with lifecycle/idempotency;
  direct/CLI/MCP transports are total and clean; and analysis consumes a
  checksum-verified success manifest. Reason: give deterministic agentic
  orchestration a trustworthy, exact handoff instead of compensating with prompts.
  Status: implemented; all 107 pinned tests pass.
- **2026-07-26** — Replace the 3-agent sequential tool pipeline with deterministic
  orchestration. The LLM interprets `ProblemIntent` and may explain deterministic
  analysis; typed application code owns compile→validate→run→analyze and exact
  handoffs. Reason: tool boundaries are fixed dependencies, not independent
  judgment tasks; LLM handoffs add mutation/retry risk. Status: decided, not
  implemented.
- **2026-07-26** — Keep one Docker image/container for demo simplicity but run each
  solver invocation in a separate child process, with timeout/cancellation,
  trusted paths/limits, captured transports, and no API key in the worker.
  Status: implemented in the hardened tool layer.
- **2026-07-26** — Free-text remains the only UI input, with
  `ready | needs_clarification | unsupported` interpretation. Clarification for
  missing/ambiguous physics is allowed and required; there is still no confirmation
  gate for a ready request. Status: decided, not implemented.
- **2026-07-26** — Replace the fixed `gpt-4.1-mini` default with current-generation
  `gpt-5.6-terra`, environment-configurable and recorded/pinned for demo
  reproducibility. Reliability is prioritized over minimizing token price for this
  personal project. Status: decided, not implemented; verify with golden scenarios
  before prompt freeze.
- **2026-07-25** — Streamlit input is free-text chat only (no structured form, no
  JSON-paste hybrid) — matches §1's vision and the tool layer's existing
  natural-language design intent. Status: input-shape decision retained; its
  accepted ambiguity risk was superseded 2026-07-26 by required clarification.
- **2026-07-25** — Agentic layer design: 3 specialist agents (1:1 with the 3
  tools), `Process.sequential`, no human-in-the-loop gate (relies on the existing
  `safety.py` cost ceiling), interface will be a minimal Streamlit UI rather than a
  CLI. Status: three-agent/process/safety portion superseded 2026-07-26;
  Streamlit/no-confirmation portion retained with clarification.
- **2026-07-25** — CrewAI runs inside the existing Docker container (same image as
  the dolfinx solver), not as a split local-process setup. See §3 for full
  reasoning. Status: same image/container retained; same-process implication
  superseded 2026-07-26 by child-process execution.
- **2026-07-25** — Made explicit that this repo is a demonstration/learning project,
  not a production workflow (§2), and that `README.md` (not `docs/spec.md`) is the
  document responsible for narrating understanding/architecture/reasoning to
  outside readers (§2a). Status: principle recorded; README.md rewrite itself
  deferred until the `agentic/` layer exists (tracked in §11 Stage 3).
- **2026-07-25** — Create `docs/spec.md` as the living planning doc for this project.
  Status: done.
- **2026-07-25** — LLM brain: OpenAI `gpt-4.1-mini`, $5 prepaid cap, auto-recharge
  off. Status: model/budget-as-driver decision superseded 2026-07-26; OpenAI
  provider decision retained.
- **2026-07-25** — New top-level package `agentic/` will hold all CrewAI
  orchestration code, kept separate from `fenitop/` so the domain library stays
  framework-agnostic. Status: decided; agent/task design now spec'd in §6a, folder
  creation tracked in §11 Stage 1.

## 10. Open Questions / Next Checkpoint

The next checkpoint is deterministic agentic Stage 1. Items intentionally resolved
by measurement/implementation rather than guessed now:

- The trusted iterative compliance and direct mechanism PETSc profiles pass
  convergence/residual checks, finite differences, and isolated-worker
  calibration; retain them unless later testing finds a profile-specific issue.
- Component-wise/roller supports and nonzero prescribed displacement are explicitly
  rejected in agent-safe v1. A future implementation must add correct
  lifting/subspace behavior and tests before changing that capability.
- The public catalog currently exposes `gpt-5.6-terra` without a distinct dated
  snapshot. That exact ID and the rest of the recorded Stage 0 configuration in
  §6 are the reproducibility baseline.

Prompt wording is deliberately deferred until Stage 1; prompts must document the
final contract in `docs/tool-reference.md`.

## 11. Implementation Checklist

Completed tool capabilities and their verification commands are maintained in
`docs/tool-reference.md` and README. This checklist tracks only remaining work.

**Stage 0 — model environment & secrets** (complete):

- [x] Create/configure the OpenAI API account and generate an API key; cost alerts
      or prepaid limits are optional personal-account policy, not architecture.
- [x] Exclude local `.env` files from Git and Docker build contexts.
- [x] Create an untracked `.env` and committed `.env.example` containing
      `OPENAI_API_KEY=` and `OPENAI_MODEL=gpt-5.6-terra`.
- [x] Add/pin CrewAI only after its exact deterministic Flow/tool API is verified
      against the pinned Pydantic/runtime stack; rebuild the image once.
- [x] Wire `.env` into the parent application process and prove the solver worker
      environment does not inherit `OPENAI_API_KEY`.
- [x] Verify outbound API access and run a throwaway structured-output smoke test.
- [x] Run golden intent scenarios and record/pin the final model ID/config (§6).

**Stage 1 — deterministic `agentic/` build** (next):

- [ ] `intent.py` — typed `ProblemIntent` and
      `ready | needs_clarification | unsupported`.
- [ ] `interpreter.py` and prompts — LLM interpretation only; structured output,
      bounded retries, capability-aware clarification.
- [ ] `orchestrator.py` — deterministic typed state machine/CrewAI Flow; exact
      compile→validate→worker→analyze handoffs and idempotent resume.
- [ ] Optional `explainer.py` — explains Tool 3 evidence without changing facts.
- [ ] `tests/agentic/` — canned interpreter outputs, clarification/resume,
      unsupported case, no duplicate solve, and full mocked-LLM flow.
- [ ] Verify via a plain harness before touching Streamlit.

**Stage 2 — Streamlit UI** (blocked on Stage 1):

- [ ] Single free-text chat input; clarification stays in chat; no form/JSON escape
      hatch and no confirmation gate for `ready`.
- [ ] Thin UI over orchestrator/job state; refresh/rerun cannot duplicate a solve.
- [ ] Show structured event/evidence trace and progress/cancel state, not hidden
      chain-of-thought.

**Stage 3 — documentation/showcase**:

- [ ] Rewrite README narrative once the hardened tools and `agentic/` flow exist.
- [ ] Add demo scenarios, known capability limits, architecture rationale, and
      reproducible run/test instructions.
