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
- constitutive law and FEM solver parameters
- objective and constraint settings
- filter parameters and optimizer settings
- output folder and output prefix

Each config also carries a parameter_guidance block with human-readable descriptions and validation rules. The loader validates the values before the run begins, so invalid settings such as negative iteration counts, decimal iteration counts, or volume fractions outside $[0, 1]$ fail fast with a clear error.

For a much more thorough check — a strict schema, physical sanity checks, and a real mesh build confirming boundary conditions actually apply — use the `validate_config` agent tool described below.

## Outputs

Each run writes FEniCSx-native XDMF time series output with HDF5 sidecar data:

- `<prefix>_density_history.xdmf` and `<prefix>_density_history.h5` contain the scalar physical density history.
- `<prefix>_displacement_history.xdmf` and `<prefix>_displacement_history.h5` contain the vector displacement history.
- `<prefix>_run.log` contains flushed per-iteration logging and structured `history` JSON records.
- `<prefix>_summary.json` contains the final compliance, volume, objective, grayness, and iteration count.

The XDMF files are intended to be opened in ParaView. Density and displacement are written as separate time series so each file has a clean, selectable field while keeping all time steps for that field in one place.

These files are written under the output folder defined in the config file, typically `results/` beneath the repository root. Generated output files are ignored by Git.

## Agent-tool layer

Beyond the example scripts, `fenitop/tools/` exposes the config-driven workflow as three composable tools for an agent (or any script) to call, each usable as a plain Python function, a `--input`/`--output` JSON CLI, or an MCP tool — one implementation behind all three. The tool wrappers send their own logging to stderr so stdout can be reserved for machine-readable responses; the hardening note below records a remaining solver-level violation of that contract.

> **Hardening status (2026-07-26):** TH-0 is complete: the runtime is pinned,
> default test discovery is reliable, and both supported problem modes have serial
> numerical baselines. The deeper audit gaps in validation, path/capability
> separation, numerical state, subprocess containment, transport, and result
> composition remain. Do not treat the CLI/MCP solve surface or lambda-string
> config support as agent-safe yet. Agentic development stays paused until the
> blocking [tool-hardening plan](docs/tool-hardening-plan.md) passes;
> `docs/spec.md` is the live status source.

### `validate_config`

Pedantically validates a config before it ever reaches the solver, against a strict Pydantic schema (`fenitop/tools/config_models.py`) designed for an LLM-authored config: every field carries a `description` in the exported JSON schema, and unknown fields are rejected outright (`extra="forbid"`, so a typo like `"otp"` instead of `"opt"` is reported directly instead of silently ignored). Fields are split deliberately:

- **Required, no default** — values that define *what problem this is*, and must be a deliberate choice: `mesh.bounds`, `mesh.divisions`, `opt.max_iter`, `opt.vol_frac`, `opt.filter_radius`, `opt.opt_compliance`, `fem."poisson's ratio"`, and `fem.dirichlet_bcs` (must be non-empty — see below).
- **Defaulted** — numerical/solver tuning knobs with literature-standard values both reference configs already agree on: `opt.opt_tol` (1e-5), `opt.penalty` (3.0), `opt.epsilon` (1e-6), `opt.beta_interval`/`beta_max` (50/128), `opt.move` (0.02), `opt.use_oc` (true), `fem."young's modulus"` (100 — for single-material SIMP compliance minimization the optimized *topology* is invariant to E, so it's safe to default), and `fem.petsc_options` (conjugate-gradient + algebraic multigrid).

Beyond structural checks, it runs logical/physical checks an agent is likely to get wrong — all hard errors, not warnings, since a config that fails them cannot produce a meaningful result:

- `opt.filter_radius` must be smaller than the domain's smallest extent (derived from `mesh.bounds`) — a filter radius at or beyond the domain size would smooth the density field across the entire domain into a uniformly gray design. (Also warns, non-fatally, if the filter radius is smaller than one mesh element — no real smoothing effect.)
- `fem.dirichlet_bcs` must be non-empty, and — when `check_geometry=true` builds a real mesh — every boundary/load marker must match at least one facet (a marker matching zero facets is otherwise silently dropped by `fem.py`), and the matched Dirichlet constraints must resist all 3 planar rigid-body modes (x-translation, y-translation, rotation). If they don't, the assembled stiffness matrix is mathematically guaranteed to have a null space and the FEM solve will fail or return a meaningless result.
- Warns (non-fatally) if there's no load at all — `fem.traction_bcs` empty and `fem.body_force` effectively zero — in compliance-minimization mode: there's nothing for the optimizer to act against.

Errors carry dotted field paths (e.g. `fem.dirichlet_bcs[0].marker`) an agent can map straight back to the JSON it generated. The response includes the normalized config (defaults filled in, still JSON-safe and re-runnable) plus an estimated problem size/cost.

### `run_topopt` and `analyze_results`

- **`run_topopt`** — runs `fenitop.topopt.topopt` with an orchestration layer: always re-validates first (a cheap arithmetic-only pre-check rejects an absurdly oversized mesh before ever building it, then the full `validate_config` check runs), enforces a cost-based safety ceiling before spending CPU (`allow_large_run`/`max_complexity_override` to proceed anyway, or invoke under `mpirun` directly for a legitimately large job), writes to a fresh timestamped output directory by default (`scoped_output=false` for the legacy flat-overwrite layout), and never raises past the tool boundary — solver failures come back as a structured error with the last known-good metrics. On success: `converged`/`stop_reason` (not just iteration count), key metrics, and every output artifact including a rendered density-field PNG and a coordinate-binned `.npz` density grid.
- **`analyze_results`** — summarizes a completed run (from `run_topopt`'s own result, or an older run's `output_folder`/`output_prefix`): convergence diagnostics (including *why* a run didn't converge, e.g. pinned at the move limit), design-quality flags (grayness/binarization, disconnected material via a connected-component check, a checkerboard heuristic, and — if the config is supplied — whether the support and load regions are connected through the same material), convergence plots, and a deterministic English narrative. Needs no dolfinx/MPI for its core metrics.

Boundary/load markers, solid/void zones, and mechanism springs can also be authored as a declarative JSON region DSL instead of a Python lambda string:

```json
{"op": "plane", "axis": "x", "value": 0}
{"op": "range", "axis": "y", "min": 8, "max": 12}
{"op": "circle", "center": [2.5, 3.0], "radius": 0.5}
{"op": "and", "regions": [ {"op": "plane", ...}, {"op": "range", ...} ]}
```

See `fenitop/regions.py` for the full op list. Existing lambda-string configs keep working unchanged; the DSL is an additional, agent-friendlier authoring surface, not a replacement.

Run a tool from the CLI (JSON request in, JSON response out):

```bash
docker compose run --rm fenitop python -m fenitop.tools.validate_config --input config/beam_2d.json
docker compose run --rm fenitop python -m fenitop.tools.run_topopt --input request.json
docker compose run --rm fenitop python -m fenitop.tools.analyze_results --input request.json
```

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
`tests/fixtures/numerical_baselines.json`. One expected failure records the known
`initial_density` bug until TH-3 fixes it.

## Repository layout

- scripts/: example entry points
- fenitop/: core FEM, sensitivity, optimization, and utility modules
  - fenitop/regions.py: declarative boundary/load region DSL
  - fenitop/tools/: the agent-tool layer (validate_config, run_topopt, analyze_results, mcp_server)
- config/: JSON config files for the examples
- tests/: unit tests; tests/fixtures/ holds small configs and committed run artifacts used by the tools' tests
