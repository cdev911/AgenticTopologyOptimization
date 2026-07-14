# AgenticTopologyOptimization — Project Spec

This is a living working document, not a one-time design doc. Every time we make a
decision or hit a checkpoint, we add to it. When a past decision changes, we go back
and edit the relevant section rather than leaving stale info in place — the Decision
Log is the only place history is preserved on purpose.

Last updated: 2026-07-26

## 0. Current Status (read this first; update it last, every session)

- **Stage**: Tool Hardening implementation; TH-0 through TH-4 are complete and TH-5 is next.
  Agentic work remains deliberately gated on the full Stage TH gate (§11).
- **As of**: 2026-07-26.
- **Just finished**: TH-4 contained execution. Contract `3.0.0` (config schema
  remains `1.1`) gives every public solve a fresh, exclusively created run
  directory under a trusted root; safe identifiers; canonical request hashing and
  idempotent replay; an atomic one-solve capacity lock; and a durable lifecycle
  record covering `queued | running | succeeded | failed | timed_out | cancelled |
  orphaned`. The numerical solve runs serially in a separate process group with a
  fixed working directory, captured streams, a credential-scrubbed environment,
  timeout/cancel TERM→KILL escalation, and parent-side crash/signal translation.
  Disk admission, symlink/path containment, incomplete partial artifacts, explicit
  MPI rejection, stale-job recovery, and setup-failure lock release are tested.
  Refreshed child-RSS/wall/output calibration remains beneath the existing
  conservative estimator. All 95 pinned tests and 54 subtests pass.
- **Architecture decisions updated** (§3, §6, §6a): deterministic orchestration
  replaces the three-agent tool-calling pipeline; solver execution stays in the
  same image/container but moves to a child process without the API key;
  clarification is allowed for incomplete/ambiguous requests without adding a
  pre-run confirmation gate; and `gpt-5.6-terra` replaces the old
  `gpt-4.1-mini` default.
- **Next action**: Stage TH-5 (§11 and `docs/tool-hardening-plan.md` §4) — make
  every public JSON-shaped call total, keep tracebacks local, reserve CLI stdout
  for exactly one response, and verify real CLI and MCP stdio framing. Do not start
  `agentic/` yet.
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

- **Hands** — `fenitop/` (existing; now entering hardening). The domain library:
  FEM/topology optimization solver, plus `fenitop/tools/` — three framework-agnostic
  operations (`validate_config`, `run_topopt`, `analyze_results`). The 2026-07-26
  audit proved the happy path but also proved this is not yet a stable agent-facing
  contract. Stage TH makes it typed, failure-contained, numerically explicit, and
  safe before any CrewAI adapter exists.
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
- Add `crewai` to `pyproject.toml` and the `Dockerfile`; one image rebuild needed
  after that (code changes after this don't need rebuilds — `docker-compose.yml`
  already bind-mounts the repo).
- Verify the container has outbound internet access to reach api.openai.com.
- `OPENAI_API_KEY` is supplied to the container via `docker-compose.yml`'s
  `env_file:` pointing at an untracked `.env` file — never baked into the image or
  committed. `.env` needs adding to `.gitignore` (not yet checked). The solver
  worker receives a sanitized environment without this key.

## 4. Repository Layout

### Current (as of 2026-07-26)
```
fenitop/                  # domain library: solver + tools/
  tools/                  # tool layer under Stage TH (validate_config, run_topopt, analyze_results)
config/                   # example configs (beam_2d, mechanism_2d)
scripts/                  # example CLI entry points (config-driven + legacy hardcoded)
tests/                    # unittest-based, split dolfinx-free vs Docker-only
results/                  # gitignored solver outputs
docs/spec.md              # this file
docs/tool-hardening-plan.md # detailed Stage TH workstreams, risks, and exit gates
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

Nothing under "Planned addition" exists yet. We create it deliberately, in a later
checkpoint, after the Stage TH tool gate passes — not just the folder shell.

## 5. Tool Contract (the "hands" API)

Source of truth: `fenitop/tools/contracts.py`,
`fenitop/tools/config_models.py`, and the three tool implementations.

- Public contract version is `3.0.0`; agent-safe config schema is `1.1`. Every
  request and response is a strict Pydantic model with unknown fields forbidden
  and structured issue records.
- The LLM-visible inputs are exactly `validate_config(config)`,
  `run_topopt(config)`, and `analyze_results(run_topopt_envelope)`.
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
- `analyze_results_tool` accepts the typed Tool 2 envelope and recovers the
  normalized config from it; legacy caller-supplied folder/prefix mode is absent
  from the public request. Every artifact path is resolved and must remain beneath
  application-owned allowed roots, including through symlinks.
- Real MCP input and output schemas are hash-snapshotted. Because pinned MCP 1.28.1
  generated the outer function argument model with extra fields ignored, server
  registration explicitly switches that generated model to `extra="forbid"` and
  tests both its schema and runtime behavior.

**Historical audit correction (2026-07-26): the earlier “errors never raise” and
“stable contract” claims were too strong.** Targeted checks found uncaught
request/I/O/geometry errors, mixed CLI stdout, executable strings, path authority,
incomplete physical validation, unverified PETSc convergence/final-state
consistency, and lossy Tool 2→Tool 3 composition. TH-1 removed executable strings
and public execution controls, added typed direct composition, and contained Tool
3 reads; TH-2 completed semantic/resource validation; TH-3 completed numerical
and evaluated-state correctness; TH-4 added process/filesystem/lifecycle
containment. The remaining findings are assigned to TH-5 and TH-6.

The detailed remediation source is `docs/tool-hardening-plan.md`. TH-1 implements
the capability split and typed-envelope portions of the first two boundary
decisions below; the complete analysis manifest remains a TH-6 gate:

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

This Stage TH gate blocks any CrewAI wrapper. We do not wrap the current unsafe/raw
surface and plan to fix it later.

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
- Before the first real agent stage, run the same golden intent/config scenarios
  against the selected model and one cheaper current alternative. If the exact
  provider snapshot/alias needed for reproducibility differs from
  `gpt-5.6-terra`, update this section and §9 rather than silently changing it.
- Local LLM (Ollama) was considered and deliberately deferred, not rejected — small
  local models are meaningfully less reliable at strict JSON-schema tool-calling and
  multi-step self-correction, which would shift effort into prompt-engineering
  workarounds rather than saving effort. May be added later as a $0/offline fallback
  profile; CrewAI's LiteLLM backend makes this a config change, not a redesign.

Budget is no longer the architecture driver, but operational bounds still matter:
pin exact CrewAI/model configuration once implemented; cap interpretation retries;
use structured outputs and low/non-creative settings for intent extraction; log
token usage; and never let an LLM retry an expensive solver side effect directly.
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
  timeout/cancellation/crash recovery.
- Test entry point: `docker compose run --rm -T fenitop python -m unittest discover -v`.
  `tests/__init__.py` makes nested discovery reliable; zero collection exits 5.
  Current result: all 95 tests and 54 subtests pass with no expected failures.

Remaining additions for agent-workflow compatibility:
- **Tool-hardening suite (blocks agent work)**: contract/schema, path/security,
  geometry/numerics/fault injection, and subprocess lifecycle coverage are in
  place. TH-5/TH-6 still add generated adversarial JSON, CLI stdout, actual MCP
  stdio, complete artifact integrity, and manifest-driven Tool 2→Tool 3
  composition. Full matrix: `docs/tool-hardening-plan.md` §5–§6.
- **Measured resource calibration**: completed in TH-2 and frozen in
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

## 9. Decision Log

Reverse-chronological. Each entry: date, decision, why, status.

- **2026-07-26** — TH-4 advances the public contract to `3.0.0` while retaining
  config schema `1.1`. The application allocates immutable per-run directories and
  owns safe IDs, request hashes, idempotency, disk admission, and a single active
  solve. Each solve executes serially in a credential-scrubbed child process group;
  atomic job state records timeout, cancellation, signal/exit status, partial
  artifacts, and restart orphan recovery. A duplicate idempotency key replays the
  matching state/result and conflicts on changed request content. Reason: Python
  exception handling in the API-bearing parent cannot contain native PETSc/MPI
  failure or make concurrent retries safe. Status: implemented and verified by 94
  tests and 54 subtests, including real timeout/cancel/signal cases.
- **2026-07-26** — TH-3 advances the public contract to `2.0.0` while retaining
  config schema `1.1`. A topology state is public only after filter, projection,
  elasticity/adjoint evaluation, and metric checks complete; iteration zero and
  every update use that same contract. Conventional grayness is
  `mean(4ρ(1-ρ))` over physical density and binarization is its complement.
  Design-change tolerance counts as convergence only after beta continuation is
  complete and one update has run at the final beta. PETSc divergence,
  non-finite/bounds failures, invalid OC assumptions, and failed MMA subproblems
  are typed numerical errors. Reason: an agent must never receive success for an
  unevaluated, stale, divergent, or ambiguously described design. Status:
  implemented and verified by 80 tests, including finite differences and fault
  injection.
- **2026-07-26** — TH-2 advances the public contract/config versions to
  `1.1.0`/`1.1`. Tool 1 now treats absent/zero loads, overlapping tractions,
  tractions on full clamps, unmatched/overlapping/constrained springs,
  unmatched/overlapping passive zones, voided required neighborhoods,
  forced-solid volume infeasibility, and extreme spring/material scaling as hard
  errors. Overlapping tractions are rejected rather than implicitly summed because
  the current solver silently keeps the first facet tag. Resource admission uses
  eight independent application-owned limits and a solver-aware estimate calibrated
  against fresh medium serial compliance/mechanism runs. Reason: syntactically
  valid input must still be physically meaningful and bounded before Dolfinx or an
  expensive solve is reached. Status: implemented and verified.
- **2026-07-26** — TH-1 replaces contract `0.1.0` with strict contract `1.0.0`
  and config schema `1.0`. Agent requests contain only `AgentSafeConfig` physics;
  trusted Python policies own execution authority. Serialized geometry is the
  bounded 2D discriminated DSL only, mechanism springs are named and positive,
  vectors are exact/finite, and v1 explicitly rejects nonzero or component-wise
  supports. Tool 3 accepts Tool 2's typed envelope. Actual MCP input/output schemas
  and outer-extra rejection are snapshot/runtime tested; artifact reads are
  contained beneath trusted application roots. Reason: make invalid and
  over-authorized states unrepresentable before an LLM is connected. Status:
  implemented and verified; numerical/resource/process/manifest gates remain.
- **2026-07-26** — TH-0 pins the immutable Dolfinx image digest, installed Ubuntu
  package versions, Python 3.12 minor line, direct Python dependencies, and their
  PyPI dependency closure. The current dict envelope is frozen as contract
  `0.1.0`; incompatible typed-contract changes in TH-1 must increment it. Serial
  compliance and mechanism reference cases use relative/absolute tolerances and
  config hashes rather than byte-identical output. Reason: make later failures
  attributable without treating floating-point artifacts as portable bytes.
  Status: implemented and verified.
- **2026-07-26** — Tool hardening is a blocking stage before any `agentic/`
  implementation. The tools have a real tested happy path (59 intended-runtime
  tests passed) but the audit found executable-string/path capabilities, incomplete
  validation/exception containment, unverified numerical convergence,
  inconsistent state/artifacts, stdout transport pollution, and lossy
  Tool 2→Tool 3 composition. Detailed workstreams and exit gates are in
  `docs/tool-hardening-plan.md`. Status: decided/planned, not implemented.
- **2026-07-26** — Replace the 3-agent sequential tool pipeline with deterministic
  orchestration. The LLM interprets `ProblemIntent` and may explain deterministic
  analysis; typed application code owns compile→validate→run→analyze and exact
  handoffs. Reason: tool boundaries are fixed dependencies, not independent
  judgment tasks; LLM handoffs add mutation/retry risk. Status: decided, not
  implemented.
- **2026-07-26** — Keep one Docker image/container for demo simplicity but run each
  solver invocation in a separate child process, with timeout/cancellation,
  trusted paths/limits, captured transports, and no API key in the worker.
  Status: implemented in Stage TH-4.
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
- **2026-07-25** — Confirmed the existing `fenitop/tools/` layer (from commit
  `f04f2e4`) is suitable as-is for wrapping with CrewAI, with the four changes
  listed in §5. Status: superseded by the deeper 2026-07-26 audit; do not wrap
  before Stage TH passes.

## 10. Open Questions / Next Checkpoint

The hardening architecture and order are decided. The next checkpoint is Stage
TH-5 total boundaries and clean transports. Items intentionally resolved by
measurement/implementation rather than guessed now:

- The trusted iterative compliance and direct mechanism PETSc profiles passed
  TH-3 convergence/residual checks and finite differences and TH-4 isolated-worker
  calibration; retain them unless later testing finds a profile-specific issue.
- Component-wise/roller supports and nonzero prescribed displacement are explicitly
  rejected in agent-safe v1. A future implementation must add correct
  lifting/subspace behavior and tests before changing that capability.
- The exact immutable OpenAI model snapshot/alias after golden intent tests; the
  current default decision is `gpt-5.6-terra` (§6).

Prompt wording is deliberately deferred until the tool schemas/capabilities pass
Stage TH; prompts must document the final contract, not today's unsafe surface.

## 11. Implementation Checklist

The detailed Stage TH actions, failure matrix, and per-workstream exit criteria are
in `docs/tool-hardening-plan.md`. This checklist is the high-level handover/status
index; check items here only when the corresponding plan exit gate passes.

**Stage TH — tool hardening** (blocks every agentic stage):

- [x] **TH-0 Characterization/reproducibility** — pin Dolfinx/dependencies; make
      zero-test discovery impossible; add fast compliance+mechanism numerical
      baselines; record runtime/config/contract versions; declare agent v1 serial.
- [x] **TH-1 Typed agent-safe contracts** — versioned Pydantic requests/responses;
      strict discriminated 2D region DSL and named spring model; exact vectors;
      separate AgentSafeConfig from trusted run policy; migrate reference configs;
      remove source/path/PETSc/safety capabilities from LLM schema.
- [x] **TH-2 Complete validation/resource policy** — finite/physical/cross-field
      rules; load/support/spring/passive-zone entity checks and overlap/conflict
      handling; independent pre-mesh memory/DOF ceiling; calibrated trusted work,
      output, and timeout estimates.
- [x] **TH-3 Numerical/state correctness** — PETSc/filter/adjoint convergence and
      finite checks; explicit MMA/OC failure status; honor initial density; correct
      iteration-0 and final evaluated state; consistent artifacts/metrics;
      grayness+binarization naming; cleanup warnings eliminated.
- [x] **TH-4 Contained execution** — fixed contained run root/slug IDs; no
      untrusted deletion/overwrite; idempotent job lifecycle; child solver process
      without API key; timeout/cancel/crash/orphan handling; atomic manifests;
      serial-only agent surface.
- [ ] **TH-5 Total boundaries/transports** — no public exceptions for JSON-shaped
      input; structured error codes/retryability; tracebacks local only; stderr
      progress and exactly-one-JSON stdout; real CLI and MCP framing tests.
- [ ] **TH-6 Manifest-driven analysis** — self-contained verified RunManifest;
      Tool 3 direct handoff without duplicate config/path; reject incomplete/corrupt
      runs; constraint/convergence/continuation diagnostics; calibrated,
      mesh-aware quality heuristics and deterministic narrative.
- [ ] **TH-7 Documentation/final gate** — capability/physics semantics, lifecycle,
      artifacts, commands, event trace, README truthfulness, full failure matrix,
      both end-to-end baselines, and final hardening review documented.

**Stage 0 — model environment & secrets** (may proceed alongside late Stage TH, but
does not unblock agent work by itself):

- [ ] Create/configure the OpenAI API account and generate an API key; cost alerts
      or prepaid limits are optional personal-account policy, not architecture.
- [ ] Add `.env` to `.gitignore`; create an untracked `.env` and committed
      `.env.example` containing `OPENAI_API_KEY=` and `OPENAI_MODEL=gpt-5.6-terra`.
- [ ] Add/pin CrewAI only after its exact deterministic Flow/tool API is verified
      against the pinned Pydantic/runtime stack; rebuild the image once.
- [ ] Wire `.env` into the parent application process and prove the Stage TH worker
      environment does not inherit `OPENAI_API_KEY`.
- [ ] Verify outbound API access and run a throwaway structured-output smoke test.
- [ ] Run golden intent scenarios and record/pin the final model ID/config (§6).

**Stage 1 — deterministic `agentic/` build** (blocked on Stage TH + Stage 0):

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
