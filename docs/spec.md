# AgenticTopologyOptimization — Project Spec

This is a living working document, not a one-time design doc. Every time we make a
decision or hit a checkpoint, we add to it. When a past decision changes, we go back
and edit the relevant section rather than leaving stale info in place — the Decision
Log is the only place history is preserved on purpose.

Last updated: 2026-07-25

## 0. Current Status (read this first; update it last, every session)

- **Stage**: Nothing implemented yet — all work so far is spec/design-only.
- **As of**: 2026-07-25.
- **Just finished**: all agentic-layer design decisions closed out (§6a) — 3
  specialist agents (Config Specialist / Solver Operator / Analyst), sequential
  process, no human-in-the-loop gate, Streamlit UI with free-text chat input only
  (no form, no JSON-paste escape hatch).
- **Next action**: Stage 0 bootstrap (§11) — create the OpenAI account/API key,
  add `crewai` to `pyproject.toml`/`Dockerfile`, wire secrets via `.env` +
  `docker-compose.yml`. Deliberately deferred by the user for now — not blocked on
  any open design question.
- **Nothing else in progress.** No code written for `agentic/` yet; no repo files
  besides `docs/spec.md` have been touched.
- **If you're an AI assistant picking this up cold**: read this whole file before
  doing anything, then summarize your understanding of current state + proposed
  next step back to the user before acting. See `CLAUDE.md`/`AGENTS.md` at the
  repo root for the full protocol.

## 1. Vision & Scope

This is a learning project. The goal is to build an **agentic workflow** where an LLM
acts as the "brain" that operates a topology-optimization solver as its "hands." The
topology optimization code (`fenitop`) is not the product — it is one tool the agent
calls. The product is the agent workflow itself: taking a plain-English structural
design request, turning it into a valid solver config, running the solve, and
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
- Not spending real money beyond an explicit, small, capped budget (see §6).

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

- **Hands** — `fenitop/` (existing, unchanged). The domain library: FEM/topology
  optimization solver, plus `fenitop/tools/` — three plain `dict -> dict` functions
  (`validate_config_tool`, `run_topopt_tool`, `analyze_results_tool`) that are already
  framework-agnostic (no CrewAI/LangChain/MCP imports inside the tool functions
  themselves — only the separate `mcp_server.py` wrapper imports MCP). This layer is
  the stable contract other layers build on.
- **Brain** — `agentic/` (new, not yet created — see Decision Log 2026-07-25). The
  CrewAI orchestration layer: agent definitions, task definitions, crew assembly,
  prompt templates, and the LLM provider config. This is the only place CrewAI is
  imported, so the domain library stays reusable if the framework ever changes.

Design principle: `fenitop` should never need to know an agent framework exists.
`agentic/` depends on `fenitop.tools`, never the other way around.

**Runtime placement (decided 2026-07-25): CrewAI runs inside the same Docker
container as the dolfinx solver**, not as a separate local process. Reasoning:
`run_topopt_tool` (and the geometry-check half of `validate_config_tool`) import
dolfinx, which only exists in the Docker image. Running everything in one container
means the agent calls the tool functions in-process, directly — no MCP/network
boundary needed for v1. This was chosen over a split (CrewAI local + dolfinx tools
reached via the existing MCP server) specifically for demo reliability: fewer moving
parts that can fail while presenting. The split remains a valid later upgrade if
agent-code iteration speed becomes a real friction point (see §10).

Practical implications, not yet done:
- Add `crewai` to `pyproject.toml` and the `Dockerfile`; one image rebuild needed
  after that (code changes after this don't need rebuilds — `docker-compose.yml`
  already bind-mounts the repo).
- Verify the container has outbound internet access to reach api.openai.com.
- `OPENAI_API_KEY` is supplied to the container via `docker-compose.yml`'s
  `env_file:` pointing at an untracked `.env` file — never baked into the image or
  committed. `.env` needs adding to `.gitignore` (not yet checked).

## 4. Repository Layout

### Current (as of 2026-07-25)
```
fenitop/                  # domain library: solver + tools/
  tools/                  # agent-callable tool layer (validate_config, run_topopt, analyze_results)
config/                   # example configs (beam_2d, mechanism_2d)
scripts/                  # example CLI entry points (config-driven + legacy hardcoded)
tests/                    # unittest-based, split dolfinx-free vs Docker-only
results/                  # gitignored solver outputs
docs/spec.md              # this file
```

### Planned addition (not yet created)
```
agentic/
  agents.py               # Agent definitions
  tasks.py                # Task definitions
  crew.py                 # Crew assembly (process type: sequential, see Decision Log)
  tools_adapter.py         # thin CrewAI Tool wrappers around fenitop.tools.*_tool
  llm.py                   # LLM provider selection (env-driven)
  prompts/                 # versioned prompt templates
tests/agentic/              # contract tests + mocked-LLM integration tests
```

Nothing under "Planned addition" exists yet. We create it deliberately, in a later
checkpoint, once the agent/task design itself is spec'd — not just the folder shell.

## 5. Tool Contract (the "hands" API)

Source of truth: `fenitop/tools/{validate_config,run_topopt,analyze_results}.py`
(added in commit `f04f2e4`, 2026-07-25).

- All three tools take and return plain JSON-serializable dicts. Errors never raise —
  they come back as a structured error envelope (`schema.py`: `ok_envelope` /
  `error_envelope`, always `status: "ok"|"error"`).
- `validate_config_tool`: structural (Pydantic `ConfigModel`) + optional geometry
  checks (real mesh build, facet-matching, rigid-body rank). Returns normalized
  config, warnings, estimated cost.
- `run_topopt_tool`: always re-validates internally (never trusts the caller already
  did). Enforces a safety cost ceiling (`safety.py`, `num_elements * max_iter`
  threshold) before running. Never lets a solver exception escape uncaught.
- `analyze_results_tool`: dolfinx/MPI-free. Reads run log + summary + a pre-exported
  `.npz` density grid (written by Tool 2 so Tool 3 never needs mesh reconstruction).
  Produces convergence diagnostics, design-quality flags, plots, and a deterministic
  narrative.

Changes planned before wrapping these for CrewAI (not yet done):
1. Add a CrewAI `args_schema` per tool, reusing `config_models.py` rather than a raw
   untyped dict.
2. Rewrite tool descriptions for an LLM audience ("when should I call this / what do
   I get back"), current docstrings are dev-oriented.
3. ~~Add an explicit confirm/estimate-cost step before `run_topopt`~~ — superseded
   2026-07-25: the crew runs fully autonomously, no human-in-the-loop gate (see
   §6a). The estimated cost should still be surfaced in the agent's narrative
   output for transparency, but it does not block execution — the sole guardrail
   is the existing `safety.py` cost ceiling.
4. Verify `output_folder`/`output_prefix` handling in `run_topopt_tool` is safe now
   that an LLM (not just a developer) can populate those strings — check for path
   traversal.

## 6. Brain: LLM Provider Strategy

**Decision (2026-07-25): OpenAI API, model `gpt-4.1-mini`, $5 prepaid credit,
auto-recharge OFF.**

Why this and not alternatives:
- CrewAI (the open-source pip package) has no usage limits or cost of its own — it's
  local orchestration code. All spend risk comes from the LLM API it calls, not from
  CrewAI. (CrewAI's separate hosted "AMP" cloud product, with its own free/paid
  execution tiers, is not something we opted into.)
- A ChatGPT Plus subscription does **not** include API credits — API billing is
  entirely separate from the consumer subscription.
- OpenAI's dashboard "spend limit" no longer hard-stops requests (as of early 2026)
  — it only alerts by email while continuing to bill. The only real hard cap is
  **prepaid credit with auto-recharge turned off**, which is what we're using.
- `gpt-4.1-mini` chosen over `gpt-4o-mini` for stronger reasoning/self-correction
  during the multi-step validate→run→analyze tool-calling loop, at ~3x the token
  cost ($0.40/$1.60 vs $0.15/$0.60 per 1M tokens). At an estimated ~30k in / 5k out
  tokens per full demo run, that's roughly $0.02-0.03/run — the $5 cap comfortably
  covers well over 100 full runs. If cost ever becomes a real constraint, dropping
  to `gpt-4o-mini` is a one-line config change (see `agentic/llm.py`, planned).
- Local LLM (Ollama) was considered and deliberately deferred, not rejected — small
  local models are meaningfully less reliable at strict JSON-schema tool-calling and
  multi-step self-correction, which would shift effort into prompt-engineering
  workarounds rather than saving effort. May be added later as a $0/offline fallback
  profile; CrewAI's LiteLLM backend makes this a config change, not a redesign.

Guardrails still to design (not yet implemented):
- Max tool-call/retry count per crew run, to bound spend if the agent loops.
- Low temperature for the config-authoring step (favor determinism over creativity).
- Pin exact CrewAI version and model ID in `pyproject.toml` once `agentic/` exists —
  CrewAI has a history of breaking changes between versions.

## 6a. Agentic Layer Design: Agents, Tasks, Process

**Decided 2026-07-25:**

- **Agent shape**: three specialist agents, one per tool, not one generalist agent
  holding all three.
  - Config Specialist — owns `validate_config`
  - Solver Operator — owns `run_topopt`
  - Analyst — owns `analyze_results`
  Reasoning: matches the existing tool boundaries 1:1, keeps each agent's role/goal
  narrow (easier to prompt reliably than one agent juggling three concerns), and
  actually exercises CrewAI's multi-agent coordination — a single generalist agent
  with three tools is functionally closer to a plain function-calling loop and
  wouldn't need CrewAI's multi-agent features at all.
- **Process type**: `Process.sequential`. Fixed validate → run → analyze pipeline,
  each task's output feeds the next as context. A hierarchical manager-agent setup
  was considered and rejected for v1: not worth the extra LLM call, cost, and
  unpredictability when the realistic "analysis found a problem" case can simply be
  reported to the user rather than auto-retried.
- **Human-in-the-loop**: **none**. The crew runs fully autonomously,
  validate → run → analyze without pausing for confirmation. The only guardrail
  against a runaway/expensive run is the existing cost ceiling in
  `fenitop/tools/safety.py` (`num_elements * max_iter` threshold) — a conscious
  tradeoff, not an oversight. This supersedes the "add an explicit confirm step"
  item originally listed in §5 (see that section's strikethrough note).
  **Correction (2026-07-25)**: this section originally cited CrewAI's built-in
  `human_input=True` as "the mechanism we'd use if we ever wanted a gate." That's
  a blocking terminal `input()` prompt, which doesn't map cleanly onto a Streamlit
  app (the crew runs inside a web request/button-click, not an interactive
  terminal). If a gate is ever added later, the real implementation would be a
  Streamlit-level two-step flow (show the interpreted config, wait for a button
  click, then proceed) — a UI state machine, not this CrewAI feature.
- **Interface**: a minimal **Streamlit** web UI, not a CLI. (Gradio was the other
  candidate; Streamlit picked for broader familiarity in data/ML-style demos — this
  is a low-stakes, easily-reversible pick, same category as the `agentic/` folder
  name.) Build order matters here: get `agentic/crew.py` working and verified via a
  plain script/test harness first, then wrap it in Streamlit — the UI is a thin
  presentation layer on top of the crew, the crew's logic should never depend on it.
- **Input shape (decided 2026-07-25)**: **free-text chat only**, not a structured
  form and not a hybrid with a JSON-paste escape hatch. The user describes the
  design problem in plain English; the Config Specialist agent does the full
  interpretation into a config from scratch — no pre-structured fields. Reasoning:
  matches §1's stated vision ("plain-English structural design request") and the
  tool layer's own existing design intent (`regions.py`'s docstring explicitly
  frames the region DSL as being for "an agent authoring a config from natural
  language"; `config_models.py` field descriptions are written to double as
  LLM-facing documentation). A form would leave that design investment unused and
  shrink the agent's role to orchestration + narration only, undermining the
  project's actual point.
  **Named risk, not a defect**: this compounds with the "no human-in-the-loop"
  decision above. `validate_config` catches structurally invalid configs and the
  agent self-corrects on those, but it cannot catch a config that is validly
  formed yet models the *wrong* problem because the agent misread intent — with no
  confirmation gate, such a run proceeds and the mismatch only surfaces in the
  final narrative. Accepted for this project's scope; would need revisiting if the
  demo starts producing confidently-wrong results.

## 7. Testing Strategy

Existing (unittest-based, already in place):
- Dolfinx-free unit tests: region DSL, Pydantic model rules, safety cost estimation,
  Tool 1 structural half, narrative generation, Tool 3 against committed fixture logs.
- Docker-only tests (need dolfinx/PETSc/MPI): geometry checks, Tool 2 end-to-end
  smoke runs.

Planned additions for agent-workflow compatibility (not yet implemented):
- **Contract tests**: feed each tool plausible-but-malformed LLM-shaped input
  (missing keys, wrong types, hallucinated extra fields) and assert a structured
  error envelope comes back, never an exception.
- **Schema introspection tests**: assert each tool's `args_schema` /
  `model_json_schema()` renders valid JSON Schema consumable by CrewAI/MCP.
- **Mocked-LLM full-crew integration test**: run the whole crew with a
  deterministic canned fake LLM (no real API call), so CI can verify pipeline
  wiring for free on every run.
- **Golden-scenario smoke test** (manual/occasional, real API call, not in CI): one
  real `gpt-4.1-mini` run against `beam_2d`, to catch prompt drift over time.

## 8. Reviewer Notes / Backlog (things to come back to)

Flagged during initial planning (2026-07-25), not yet actioned:
- Observability: enable CrewAI verbose/step logging for demo storytelling (show the
  agent's reasoning trail, not just the final answer).
- Confirm `docker-compose.yml` allows outbound internet access from the container
  for the LLM API call.
- Demo-day reliability: consider caching/recording a known-good run so a live
  presentation doesn't depend on a live LLM call working under pressure.
- License/attribution check: confirm the origin/license of the `fenitop` codebase
  before presenting this publicly as a demo.

## 9. Decision Log

Reverse-chronological. Each entry: date, decision, why, status.

- **2026-07-25** — Streamlit input is free-text chat only (no structured form, no
  JSON-paste hybrid) — matches §1's vision and the tool layer's existing
  natural-language design intent. Named, accepted risk: combined with "no
  human-in-the-loop," a semantically-wrong-but-valid config can run without
  anyone catching the misinterpretation before the fact. Full reasoning in §6a.
  Status: decided, not yet implemented.
- **2026-07-25** — Agentic layer design: 3 specialist agents (1:1 with the 3
  tools), `Process.sequential`, no human-in-the-loop gate (relies on the existing
  `safety.py` cost ceiling), interface will be a minimal Streamlit UI rather than a
  CLI. Full reasoning in §6a. Status: decided, not yet implemented (tracked in
  §11 Stage 1).
- **2026-07-25** — CrewAI runs inside the existing Docker container (same image as
  the dolfinx solver), not as a split local-process setup. See §3 for full
  reasoning (mainly: fewer moving parts to fail during a live demo). Status:
  decided, not yet wired into `pyproject.toml`/`Dockerfile`.
- **2026-07-25** — Made explicit that this repo is a demonstration/learning project,
  not a production workflow (§2), and that `README.md` (not `docs/spec.md`) is the
  document responsible for narrating understanding/architecture/reasoning to
  outside readers (§2a). Status: principle recorded; README.md rewrite itself
  deferred until the `agentic/` layer exists (tracked in §11 Stage 3).
- **2026-07-25** — Create `docs/spec.md` as the living planning doc for this project.
  Status: done.
- **2026-07-25** — LLM brain: OpenAI `gpt-4.1-mini`, $5 prepaid cap, auto-recharge
  off. See §6 for full reasoning. Status: decided, not yet wired into code
  (`agentic/llm.py` doesn't exist yet).
- **2026-07-25** — New top-level package `agentic/` will hold all CrewAI
  orchestration code, kept separate from `fenitop/` so the domain library stays
  framework-agnostic. Status: decided; agent/task design now spec'd in §6a, folder
  creation tracked in §11 Stage 1.
- **2026-07-25** — Confirmed the existing `fenitop/tools/` layer (from commit
  `f04f2e4`) is suitable as-is for wrapping with CrewAI, with the four changes
  listed in §5. Status: assessed, changes not yet made.

## 10. Open Questions / Next Checkpoint

Agent shape, process type, human-in-the-loop, interface framework, and input shape
are all resolved (§6a) — no longer listed here. Remaining opens:

- Exact task descriptions/prompt wording for the three agents — deferred to
  implementation time (Stage 1, §11 below), not a spec-level decision.

## 11. Implementation Checklist (bootstrap work — decided, not yet done)

Everything here is already *decided* (see §3, §6, §2a, §6a) — it's pure
setup/execution work, not a design decision, so it's tracked as a checklist rather
than mixed into the Decision Log or Open Questions. Check items off in place as
they're completed.

**Stage 0 — environment & secrets** (can happen anytime, blocks nothing else):
- [ ] Create OpenAI platform account, prepay $5, disable auto-recharge, generate an
      API key (§6)
- [ ] Add `.env` to `.gitignore`; create an untracked `.env` (and a committed
      `.env.example` template) holding `OPENAI_API_KEY=`
- [ ] Add `crewai` to `pyproject.toml` and the `Dockerfile`; rebuild the image once
- [ ] Verify the container has outbound internet access to `api.openai.com`
- [ ] Wire `OPENAI_API_KEY` into the container via `docker-compose.yml`'s
      `env_file:`
- [ ] Smoke-test: one throwaway single-agent CrewAI script to confirm the key and
      billing actually work end to end

**Stage 1 — agentic/ package build** (blocked on Stage 0; design decided in §6a):
- [ ] `agentic/tools_adapter.py` — wrap the 3 `fenitop.tools` functions as CrewAI
      tools
- [ ] `agentic/agents.py`, `tasks.py`, `crew.py` (3 specialist agents, sequential
      process, no human-in-the-loop gate)
- [ ] `tests/agentic/` — contract tests + mocked-LLM integration test (§7)
- [ ] Verify the crew runs correctly via a plain script/test harness before
      touching the UI

**Stage 2 — Streamlit UI** (blocked on Stage 1 being verified working):
- [ ] Single free-text chat input (no structured form, no JSON-paste escape hatch —
      see §6a)
- [ ] Thin presentation layer only — no crew logic lives in the UI code

**Stage 3 — documentation**:
- [ ] Rewrite `README.md` narrative once `agentic/` exists (§2a)
