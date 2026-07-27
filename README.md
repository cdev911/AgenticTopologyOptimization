# FEniTop

FEniTop is a FEniCSx-based topology optimization package for 2D and 3D problems. It combines a finite-element formulation with density filtering, Heaviside projection, and optimization routines such as OC and MMA.

## Docker-based setup

The recommended way to run this project is in the pinned Docker image. The
`Dockerfile` uses an immutable Dolfinx image digest and exact Python dependency
versions so numerical baselines do not drift when an upstream `stable` tag moves.

### Prerequisites

- Docker Engine
- Docker Compose
- An OpenAI Platform API key with API billing/credit enabled for the agentic
  workflow checks. A ChatGPT subscription does not include API credit.

CrewAI is installed inside the project image at its pinned version. No global
CrewAI or Python package installation is needed on the host Mac. Keeping the
framework in Docker prevents one project's CrewAI/Pydantic versions from changing
another project's environment.

### Configure the local API environment

Create the local environment file from the safe committed template:

```bash
cp .env.example .env
```

Edit `.env` and set the key:

```dotenv
OPENAI_API_KEY=your-real-api-key
OPENAI_MODEL=gpt-5.6-terra
```

Do not commit `.env` or paste its key into issues, logs, or chat. The secret has
two separate protections:

1. `.gitignore` prevents `.env` and `.env.*` files from entering Git, while
   explicitly allowing the blank `.env.example` template.
2. `.dockerignore` prevents those files from entering the Docker build context,
   so `COPY . /workspace` cannot bake the key into an image layer.

At runtime, Compose's `env_file:` injects the values into the parent application
container. Solver jobs run as child processes with all `OPENAI_*` variables
removed, because numerical code does not need model credentials.

### Pull the base image (optional)

```bash
docker pull dolfinx/dolfinx@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8
```

`docker compose build` pulls this digest automatically if it is not already
available, so the explicit pull is normally unnecessary.

### Build the project image

```bash
docker compose build
```

This installs the complete pinned dependency closure from
`requirements/runtime.lock`, including CrewAI, inside `fenitop:local`. Rebuild
after changing `pyproject.toml`, the lock file, Dockerfile, or base-image digest.
Ordinary source changes are visible immediately through the repository bind mount.

Check the installed environment without exposing the API key:

```bash
docker compose run --rm -T fenitop python -m pip check
docker compose run --rm -T fenitop python -c \
  'from importlib.metadata import version; print("crewai", version("crewai")); print("pydantic", version("pydantic"))'
```

Expected Stage 0 pins are CrewAI `1.15.6` and Pydantic `2.12.5`.

### Verify the model environment

Run the small structured-output smoke test:

```bash
docker compose run --rm -T fenitop python scripts/stage0_model_smoke.py
```

Expected output:

```text
crewai_structured_output=ok model=gpt-5.6-terra status=ok answer=4
```

Run the occasional golden intent comparison:

```bash
docker compose run --rm -T fenitop python scripts/stage0_golden_intents.py
```

The golden check exercises `ready`, `needs_clarification`, and `unsupported`
classification against Terra and the cheaper Luna comparison. Both commands make
real, billed API calls; they are manual checkpoints, not part of the default test
suite or CI.

If OpenAI reports `insufficient_quota`, the key can still be valid—the API account
needs billing or prepaid credit. If the model rejects `temperature=0`, leave
temperature unset; the pinned Terra integration uses low reasoning effort and
strict Pydantic output instead.

### CrewAI compatibility and rollback

CrewAI `1.15.6` currently requires Pydantic below 2.13 and its structured-output
stack requires Rich below 15. The verified image therefore pins Pydantic `2.12.5`,
`pydantic-core 2.41.5`, and Rich `14.2.0`. All 107 hardened solver/tool tests pass
with this combination.

If a future CrewAI change breaks the hardened layer, roll back by removing CrewAI
from `pyproject.toml`, restoring Pydantic `2.13.4`, `pydantic-core 2.46.4`, and
Rich `15.0.0`, regenerating `requirements/runtime.lock`, rebuilding the image, and
rerunning the full test suite. Never bypass dependency checks or accept changed
contract/schema snapshots just to force an installation.

### Run the examples

Run the 2D beam example:

```bash
docker compose run --rm fenitop python scripts/beam_2d.py --config config/beam_2d.json
```

Run the compliant mechanism example:

```bash
docker compose run --rm fenitop python scripts/mechanism_2d.py --config config/mechanism_2d.json
```

Run with MPI:

```bash
docker compose run --rm fenitop mpirun -n 2 python scripts/beam_2d.py --config config/beam_2d.json
```

## Configuration-driven runs

The example entry points now read a JSON configuration file. The repository includes example configs in the config directory:

- config/beam_2d.json
- config/mechanism_2d.json

The config file controls:

- mesh generation and boundary conditions
- material properties and the supported plane-strain physics
- objective and constraint settings
- filter parameters and optimizer settings

The versioned agent-safe schema deliberately does not include output paths, PETSc
options, mesh ghost modes, rendering switches, or safety overrides. Application
code owns those execution capabilities. The loader validates the config before a
run begins, so invalid settings such as negative iteration counts, malformed
vectors, non-finite values, or volume fractions outside the supported range fail
fast with field-level errors.

For a much more thorough check — a strict schema, physical sanity checks, and a real mesh build confirming boundary conditions actually apply — use the `validate_config` agent tool described below.

## Outputs

Each run writes FEniCSx-native XDMF time series output with HDF5 sidecar data:

- `<prefix>_density_history.xdmf` and `<prefix>_density_history.h5` contain the scalar physical density history.
- `<prefix>_displacement_history.xdmf` and `<prefix>_displacement_history.h5` contain the vector displacement history.
- `<prefix>_run.log` contains flushed per-iteration logging and structured `history` JSON records.
- `<prefix>_summary.json` contains final evaluated compliance, volume, objective,
  conventional grayness, binarization, design change, beta/continuation state,
  optimizer status, and iteration count.

The XDMF files are intended to be opened in ParaView. Density and displacement are written as separate time series so each file has a clean, selectable field while keeping all time steps for that field in one place.

Example scripts write these files under `results/` beneath the repository root.
The tool layer assigns a fresh per-run directory through trusted application
policy. Generated output files are ignored by Git.

## Agentic workflow (Stage 1 complete)

The first agentic boundary is implemented in `agentic/intent.py`. It converts the
idea of a free-text request into one of three strict, mutually exclusive outcomes:

- `ready`: all problem-defining physics is present in a typed `ProblemIntent`.
- `needs_clarification`: required physics is missing or ambiguous, so the workflow
  returns focused questions and does not solve.
- `unsupported`: the request needs physics outside the v1 tool contract, so the
  workflow explains the mismatch and does not solve.

`ProblemIntent` separates structural meaning from numerical tuning. Geometry,
material, supports, loads, volume fraction, and mechanism-specific fields are
physics and must be present for `ready`. Mesh divisions/cell type, filter radius,
and iteration limit are optional preferences. When they are omitted, deterministic
application code—not the LLM—will apply versioned defaults and show each selected
value and reason to the user before proceeding. This is a transparency notice, not
a confirmation gate: the user can request changes, while an unchanged supported
request continues automatically.

This separation makes an important agent-design rule concrete: use the LLM where
language judgment is needed, and use typed deterministic code for defaults,
validation, execution authority, and side effects.

`agentic/interpreter.py` now provides the LLM boundary. It loads the versioned
capability prompt from `agentic/prompts/intent_system_v1.txt`, sends the free-text
request as JSON-escaped untrusted data, and accepts only the strict outcome schema.
CrewAI is configured with low reasoning effort and hidden SDK retries disabled;
the application owns a bounded two-attempt retry policy and reports only sanitized
failure metadata. The interpreter cannot compile configs, choose defaults, call
solver tools, or launch work.

The OpenAI structured-output API accepts a stricter JSON Schema subset than
Pydantic normally emits for fixed tuples and recursive region expressions. A
transport-only schema adapter therefore converts tuple `prefixItems` to compatible
`items` schemas and ensures every object rejects extra properties. The semantic
Pydantic models remain unchanged, so this compatibility layer does not weaken
runtime validation. Tests snapshot these transport invariants, and a billed smoke
check exercises ready, clarification, and unsupported outcomes.

`agentic/compiler.py` handles the next deterministic boundary. If mesh resolution
is omitted, it derives a target element size as `sqrt(domain area) / 50`, then
rounds each axis count from its physical length. A square becomes `50×50`; a
rectangle remains close to 2,500 elements with nearly square cells. Extremely
slender domains keep at least two cells across the short direction and refine the
long direction to preserve cell shape; the existing resource validator then
decides whether that mesh is admissible. The default filter radius is 1.5 times
the larger derived element edge.

Compilation returns both the exact `AgentSafeConfig` and a versioned ledger of
every application-selected setting. This includes omitted mesh/filter/iteration
preferences plus optimizer, initialization, tolerance, continuation, move-limit,
plane-strain, thickness, units, and passive-zone settings. The ledger generates
the explicit user notice that the values were not provided and were selected by
the deterministic compiler; the user may request changes, otherwise execution can
continue without a confirmation gate.

The first `agentic/orchestrator.py` slice is a typed deterministic state machine
through validation. It returns one of four explicit states:
`awaiting_clarification`, `unsupported`, `validation_failed`, or `validated`.
Clarification and unsupported outcomes cannot compile or validate. A clarification
resume persists the original request, the exact missing fields/questions, and the
user's answer before asking the interpreter again; it does not depend on hidden
model memory.

For a ready intent, the orchestrator emits the defaults notice before invoking the
validator, then passes a `ValidateConfigRequest` containing the exact compiled
Pydantic config. Its callback and persisted event trace expose stage facts without
chain-of-thought.

Execution continues from a `validated` state using two idempotency layers. An
in-memory cache prevents a repeated transition in one application process from
calling the runner again. A stable application-owned key derived from conversation,
compiled config, and defaults profile lets the run tool replay the same durable
result after a UI/process restart. A successful typed `RunManifest` is passed
directly into `AnalyzeResultsRequest`; no model or application code copies its
paths or metrics. Run and analysis failures are separate typed states, and an
analysis retry reuses the stored manifest without revisiting the solver.

Run the small no-API Stage 1 harness:

```bash
docker compose run --rm -T fenitop python scripts/stage1_workflow_harness.py
```

It uses a canned schema-validated `8×4`, five-iteration intent, then performs real
validation, contained execution, and deterministic analysis. The first invocation
reports `idempotent_replay=false`; repeating it returns the same run ID with
`idempotent_replay=true`.

`agentic/explainer.py` adds the optional final LLM step without giving the model
permission to rewrite results. Deterministic code converts Tool 3 output into an
immutable evidence ledger with stable IDs such as `F001`, marks convergence,
metrics, constraints, and quality facts as required, and excludes the run
directory. The model returns only a structured plan of allowed section headings
and fact IDs. Unknown IDs, omitted required facts, duplicates, and repeated
headings are rejected. Deterministic code then renders the original fact text and
citations; no model-generated prose or recalculated value reaches the explanation.

This is a deliberately constrained use of an LLM: it contributes presentation
judgment—selection and organization of supporting evidence—while deterministic
analysis retains factual authority. A live Terra check successfully organized the
real harness evidence, including its exact non-convergence and quality findings.

## Agent-tool layer

Beyond the example scripts, `fenitop/tools/` exposes the config-driven workflow as three composable tools for an agent (or any script) to call, each usable as a plain Python function, a `--input`/`--output` JSON CLI, or an MCP tool — one implementation behind all three. Logs and solver progress stay on stderr or in captured run files; CLI stdout is exactly one JSON response and real stdio MCP composition is tested.

> **Tool readiness (2026-07-26):** The tool layer is hardened and ready for the
> deterministic agentic workflow. The runtime
> and numerical baselines are pinned; the public tools now use contract `4.0.0`
> and physics-only config schema `1.1`. Source strings and execution controls are
> absent from the agent surface, semantic/mesh-backed validation covers every
> current solver input, independent resource ceilings are calibrated in the
> pinned image, and successful solves now require checked linear solves, finite
> bounded states, explicit optimizer success, and consistent evaluated artifacts.
> Solves now run serially in credential-scrubbed child process groups with contained
> per-run paths, idempotent lifecycle state, one-solve admission, and tested
> timeout/cancel/crash/orphan handling. Public boundaries are total and
> transport-clean, and successful runs carry a durable, checksum-verified
> `RunManifest` consumed directly by deterministic analysis. Deterministic agentic
> development may begin after the model/secrets environment checkpoint.
> `docs/spec.md` remains the live status source.

The [hardened tool reference](docs/tool-reference.md) documents exact physics,
every result field, lifecycle/artifact semantics, error and retry behavior, test
tiers, and the trust limits of local integrity checks.

### `validate_config`

Pedantically validates a config before it reaches the solver against
`AgentSafeConfig` in `fenitop/tools/config_models.py`. Unknown fields are rejected,
all vectors are exactly 2D and finite, conditional compliance/mechanism fields are
expressed as a discriminated union, and mechanism springs use named
`region`/`direction`/`stiffness` fields. The schema explicitly supports rectangular
2D meshes, plane strain, unit out-of-plane thickness, consistent user units,
distributed boundary traction, and full-vector zero clamps. Nonzero prescribed
displacements and component-wise roller supports are rejected in v1.

Beyond structural checks, it runs logical/physical checks an agent is likely to get wrong — all hard errors, not warnings, since a config that fails them cannot produce a meaningful result:

- `opt.filter_radius` must be smaller than the domain's smallest extent (derived from `mesh.bounds`) — a filter radius at or beyond the domain size would smooth the density field across the entire domain into a uniformly gray design. (Also warns, non-fatally, if the filter radius is smaller than one mesh element — no real smoothing effect.)
- Volume fraction and optional initial density are strictly between zero and one;
  move and SIMP epsilon are positive; beta continuation uses a power-of-two cap.
  Both problem modes require a nonzero external load.
- `fem.dirichlet_bcs` must be non-empty. Trusted validation builds the real mesh
  and checks support/load facets, rigid-body rank, spring nodes, passive-zone
  cells, traction/support/spring overlaps, solid/void conflicts, required material
  neighborhoods, and forced-solid feasibility against the volume budget.
- Mechanism spring stiffness is checked relative to Young's modulus. Entity counts
  and physical bounds are returned in a typed geometry report.
- Before any mesh build, pure arithmetic independently estimates elements, nodes,
  displacement DOFs, iterations, solver-weighted work, peak memory, output, and
  wall time. The application-owned defaults are calibrated against committed
  medium compliance and mechanism measurements.

Errors and warnings are structured records containing `code`, `path`, `message`,
`severity`, and `retryable`. The response includes the normalized config (defaults
filled in, still JSON-safe and re-runnable), resource estimate, and geometry
entity report.

### `run_topopt` and `analyze_results`

- **`run_topopt`** — accepts only `{"config": AgentSafeConfig}`. It always
  re-validates, performs a cheap arithmetic safety pre-check, then applies an
  application-owned `TrustedRunPolicy` for run IDs, paths, rendering, solver
  profile, and safety ceilings. The MCP/LLM schema cannot override that policy.
  The parent exclusively allocates a contained run directory, deduplicates
  matching idempotency keys, admits only one serial solve, and launches a
  credential-free worker process group. Its atomic lifecycle distinguishes queued,
  running, succeeded, failed, timed-out, cancelled, and orphaned jobs; parent-side
  crash translation records exit/signal state and marks partial artifacts
  incomplete. On success it returns convergence state, key metrics, validation
  evidence, lifecycle state, typed artifact records, and a self-contained
  `RunManifest`. The manifest embeds normalized config/evidence/runtime versions
  and uses relative artifact paths with sizes and SHA-256 checksums.
- **`analyze_results`** — accepts only the exact successful `RunManifest`, so no
  model copies paths or resupplies a config. Before reading results it verifies the
  manifest hash, durable manifest file, trusted run root, and every artifact's
  path, size, completeness, and checksum. It rejects empty/inconsistent history,
  summary mismatches, and malformed density grids; reports convergence,
  continuation, constraints, checkerboards, disconnected material, and per-load/
  spring connectivity; and produces a deterministic narrative without dolfinx/MPI
  for its core metrics.

All serialized boundary/load markers, solid/void zones, and mechanism spring
regions use the strict declarative JSON region DSL:

```json
{"op": "plane", "axis": "x", "value": 0}
{"op": "range", "axis": "y", "min": 8, "max": 12}
{"op": "circle", "center": [2.5, 3.0], "radius": 0.5}
{"op": "and", "regions": [ {"op": "plane", ...}, {"op": "range", ...} ]}
```

The full op set is `plane`, `range`, `circle`, `all`, `none`, `and`, `or`, and
`not`. Unknown fields, non-finite values, 3D axes, invalid radii/tolerances, and
excessive recursion/node counts are rejected. JSON lambda/source strings are not
supported and are never evaluated; trusted hardcoded Python callers may still use
callables internally.

Run a tool from the CLI (JSON request in, JSON response out):

```bash
python -c 'import json; print(json.dumps({"config": json.load(open("config/beam_2d.json"))}))' > request.json
docker compose run --rm -T fenitop python -m fenitop.tools.validate_config --input request.json
docker compose run --rm -T fenitop python -m fenitop.tools.run_topopt --input request.json --output run-response.json
python -c 'import json; r=json.load(open("run-response.json")); print(json.dumps({"run_manifest": r["run_manifest"]}))' > analysis-request.json
docker compose run --rm -T fenitop python -m fenitop.tools.analyze_results --input analysis-request.json
```

The analyzer request's `run_manifest` field contains the exact manifest returned by
Tool 2. Direct Python, CLI, and stdio MCP surfaces all validate the same models.

Or start the MCP server (the `-T` disables the pty; stdio-transport MCP needs clean, unbuffered JSON-RPC framing on stdin/stdout, which `docker-compose.yml`'s default `tty: true` would otherwise corrupt):

```bash
docker compose run --rm -T fenitop python -m fenitop.tools.mcp_server
```

Scope note: this layer currently covers the two config-driven, 2D problem types (`beam_2d` "minimize compliance" and `mechanism_2d` "compliant mechanism"). The 3D/legacy hardcoded scripts (`beam_3d.py`, `disk_2d.py`, `shell_3d.py`) are not yet part of it.

## Tests

Build the pinned image, then run the complete suite from the repository root:

```bash
docker compose build
docker compose run --rm -T fenitop python -m unittest discover -v
```

The package marker at `tests/__init__.py` is intentional: without it Python's
default unittest discovery can silently skip nested test modules. A zero-test run
now exits nonzero. The suite includes fast real solves for compliance and
compliant-mechanism modes with tolerance-based references in
`tests/fixtures/numerical_baselines.json`, directional finite-difference
sensitivity checks, injected numerical failures, initial/final-state consistency,
cleanup checks, path/idempotency adversarial cases, real isolated-worker timeout/
cancellation/signal recovery, generated malformed inputs, corrupt artifacts,
calibrated heuristics, CLI purity, and real MCP composition. Focused development
commands are listed in the [tool reference](docs/tool-reference.md); the full
command above remains the checkpoint gate. All 107 tests pass in the pinned image.
The completed Stage 1 workflow brings the current checkpoint to 148 passing tests
plus 103 passing subtests.

## Repository layout

- scripts/: example entry points
- agentic/: typed natural-language interpretation and deterministic workflow
  modules (Stage 1 complete)
- fenitop/: core FEM, sensitivity, optimization, and utility modules
  - fenitop/regions.py: declarative boundary/load region DSL
  - fenitop/tools/: the agent-tool layer (validate_config, run_topopt, analyze_results, mcp_server)
- config/: JSON config files for the examples
- tests/: unit tests; tests/fixtures/ holds small configs and committed run artifacts used by the tools' tests
- docs/tool-reference.md: exact hardened tool capabilities and result interpretation
