# FEniTop

FEniTop is a FEniCSx-based topology optimization package for 2D and 3D problems. It combines a finite-element formulation with density filtering, Heaviside projection, and optimization routines such as OC and MMA.

## Docker-based setup

The recommended way to run this project is in the pinned Docker image. The
`Dockerfile` uses an immutable Dolfinx image digest and exact Python dependency
versions so numerical baselines do not drift when an upstream `stable` tag moves.

### Prerequisites

- Docker Engine
- Docker Compose

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

## Agent-tool layer

Beyond the example scripts, `fenitop/tools/` exposes the config-driven workflow as three composable tools for an agent (or any script) to call, each usable as a plain Python function, a `--input`/`--output` JSON CLI, or an MCP tool — one implementation behind all three. The tool wrappers send their own logging to stderr so stdout can be reserved for machine-readable responses; the hardening note below records a remaining solver-level violation of that contract.

> **Hardening status (2026-07-26):** TH-0 through TH-4 are complete. The runtime
> and numerical baselines are pinned; the public tools now use contract `3.0.0`
> and physics-only config schema `1.1`. Source strings and execution controls are
> absent from the agent surface, semantic/mesh-backed validation covers every
> current solver input, independent resource ceilings are calibrated in the
> pinned image, and successful solves now require checked linear solves, finite
> bounded states, explicit optimizer success, and consistent evaluated artifacts.
> Solves now run serially in credential-scrubbed child process groups with contained
> per-run paths, idempotent lifecycle state, one-solve admission, and tested
> timeout/cancel/crash/orphan handling. Clean public exception/CLI/MCP boundaries
> and the final verified analysis manifest remain. Agentic development stays paused
> until the full blocking [tool-hardening plan](docs/tool-hardening-plan.md) passes;
> `docs/spec.md` is the live status source.

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
  evidence, lifecycle state, and typed artifact records.
- **`analyze_results`** — accepts the exact typed successful `run_topopt` envelope,
  so no model copies filesystem paths or resupplies a config. It derives
  convergence diagnostics, design-quality flags, plots, and a deterministic
  narrative without dolfinx/MPI for its core metrics. Artifact paths are resolved
  and rejected unless they remain under application-owned allowed roots. A checksum-verified,
  self-contained `RunManifest` and stronger artifact integrity checks remain TH-6
  work.

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
docker compose run --rm -T fenitop python -m fenitop.tools.run_topopt --input request.json
docker compose run --rm -T fenitop python -m fenitop.tools.analyze_results --input request.json
```

The analyzer request is different: its `run_topopt_envelope` field must contain the
exact Tool 2 response. Clean stdout/stdio framing is still a TH-5 gate, so direct
Python calls are the verified integration surface at this checkpoint.

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
and cleanup checks, path/idempotency adversarial cases, and real isolated-worker
timeout/cancellation/signal recovery. The TH-4 checkpoint runs all 95 tests and 54
subtests successfully.

## Repository layout

- scripts/: example entry points
- fenitop/: core FEM, sensitivity, optimization, and utility modules
  - fenitop/regions.py: declarative boundary/load region DSL
  - fenitop/tools/: the agent-tool layer (validate_config, run_topopt, analyze_results, mcp_server)
- config/: JSON config files for the examples
- tests/: unit tests; tests/fixtures/ holds small configs and committed run artifacts used by the tools' tests
