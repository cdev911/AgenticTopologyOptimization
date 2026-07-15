# Hardened Tool Reference

This is the human-readable reference for the agent-facing tool contract. The
machine sources of truth are `fenitop/tools/contracts.py`,
`fenitop/tools/config_models.py`, and the generated MCP schemas tested in
`tests/fixtures/mcp_schema_hashes.json`.

Current versions:

- tool contract: `4.0.0`
- agent-safe config schema: `1.1`
- successful-run manifest: `1.0`

## 1. Scope and trust boundary

The public surface contains exactly three operations:

```text
validate_config({"config": AgentSafeConfig})
run_topopt({"config": AgentSafeConfig})
analyze_results({"run_manifest": RunManifest})
```

All three are available as direct Python functions, JSON CLIs, and MCP tools. The
same Pydantic request/response models are used by every transport.

`AgentSafeConfig` contains problem physics only. It cannot contain source strings,
callables, paths, run IDs, output names, PETSc options, MPI size, timeouts,
rendering flags, resource overrides, or overwrite controls. Trusted application
code—not the model—constructs validation, run, and analysis policies.

The agent workflow is serial. Legacy scripts can still demonstrate MPI, 3D, or
hardcoded Python regions, but those are not agent-tool capabilities.

## 2. Exact physics represented by the config

| Area | Supported meaning | Important limitation |
|---|---|---|
| Domain | One axis-aligned 2D rectangle, quadrilateral or triangular cells | No arbitrary CAD, holes in the mesh, 3D, shells, or remeshing |
| Constitutive model | Isotropic linear elasticity under plane strain | Unit out-of-plane thickness is fixed |
| Units | Any one self-consistent user unit system | The tool does not convert or infer units |
| Material | SIMP interpolation using `young_modulus`, `poisson_ratio`, `penalty`, and positive ersatz `epsilon` | Single isotropic material plus void approximation |
| Support | Full-vector zero displacement on matched boundary facets | No roller/component-only support and no nonzero prescribed displacement |
| Traction | Constant distributed boundary traction vector integrated over every matched boundary facet, per unit out-of-plane thickness | It is not a total force; overlapping traction regions are rejected rather than summed |
| Body force | Constant force-per-volume vector integrated over the domain, with unit thickness | It is not a total force |
| Passive zones | Cell-center regions forced near solid (`0.99`) or void (`0.01`) | Regions must match cells, remain disjoint, and preserve required neighborhoods |
| Density filter | Helmholtz length-scale filter; `filter_radius` uses the same length unit as mesh coordinates | It must be smaller than the smallest domain extent; sub-element radii warn |
| Projection | Heaviside continuation from beta 1 to the configured power-of-two `beta_max` | A low design change is not called converged until continuation completes |

The supported optimization modes are:

- `minimize_compliance`: minimizes strain-energy compliance subject to the volume
  fraction. OC is the default; MMA is also representable.
- `compliant_mechanism`: minimizes the signed output-displacement functional
  subject to volume and compliance bounds, using MMA.

Mechanism springs deserve special care. A configured stiffness is placed on each
matched directional nodal degree of freedom; it is not automatically divided
across the selected region. The output objective is likewise the sum of signed
directional displacements at matched output nodes. Consequently, region width,
mesh resolution, stiffness, load, and Young's modulus are coupled. Validation
checks the matched node counts and rejects an extreme
`stiffness / young_modulus` ratio, but it cannot infer the user's intended
physical spring assembly. The future intent compiler must preserve this meaning
and expose the geometry report rather than pretending the value is a
mesh-independent total spring constant.

The region DSL supports `plane`, `range`, `circle`, `all`, `none`, `and`, `or`,
and `not`. It is strictly 2D, finite, bounded in depth and node count, and rejects
unknown fields. Regions describe geometric selection; they do not execute code.

## 3. Tool sequence and result envelopes

Every response has:

- `contract_version`: version of the complete public boundary.
- `tool`: `validate_config`, `run_topopt`, or `analyze_results`.
- `status`: exactly `ok` or `error`.
- `warnings` / `errors`: records with `code`, dotted `path`, human `message`,
  `severity`, and `retryable`.
- `stage`: the finer failure phase; normally `null` on success.

Callers branch on `status`, not on missing optional fields. They must pass the
exact normalized objects/manifests in memory; an LLM must never retype them.

### `validate_config`

Validation performs, in order:

1. strict request and config parsing;
2. cross-field physical checks;
3. pure arithmetic resource estimation/admission before mesh construction;
4. real mesh-backed entity and conflict checks.

Key success fields:

- `checked`: whether structural, resource, and geometry checks ran.
- `problem_type`: selected supported mode.
- `normalized_config`: defaults-filled, JSON-safe config suitable for a run.
- `estimated_cost`: element/node/DOF/design counts, evaluated states, linear solves,
  work, memory, output, wall estimate, solver profile, and risk band.
- `geometry_report`: total mesh entities, rigid-body rank, and a record for every
  support/load/spring/passive region with entity kind, count, and physical bounds.

An `ok` result means the config is valid under the current trusted policy. It is
not a guarantee that iterative optimization will converge to a good topology.

### `run_topopt`

Tool 2 always revalidates. The parent allocates a contained run directory,
deduplicates the idempotency key when supplied by trusted code, acquires the
single-solve lock, and launches a sanitized serial child process.

Important fields:

- `run_id`: application-owned identity.
- `converged`: whether the topology loop met its design-change criterion after
  completing projection continuation.
- `stop_reason`: `tolerance_met`, `max_iterations_reached`, or
  `continuation_incomplete`.
- `iterations`: number of design updates; iteration zero is the evaluated initial
  design.
- `metrics`: final compliance, volume, signed objective, conventional grayness
  `mean(4 rho (1-rho))`, complementary binarization, design change, tolerance,
  beta, and continuation state.
- `optimizer_status`: success and work of the last OC/MMA update. This is distinct
  from outer topology `converged`.
- `validation` / `estimated_cost`: evidence used immediately before launch.
- `lifecycle`: durable process/job state and exit evidence.
- `artifacts`: convenient response inventory using absolute local paths.
- `run_manifest`: present only on success; this is the authoritative Tool 3 input.
- `last_known_good_metrics`: last durable evaluated state on a failed run, when
  available. It is diagnostic evidence, never a success result.
- `error`: sanitized numerical/worker context and a local debug artifact role.

A successful solve may legitimately have `converged=false` if it reached the
iteration cap. It is still numerically valid and analyzable; analysis explains the
stop condition.

### `analyze_results`

Tool 3 accepts only the exact successful `RunManifest`. It verifies the canonical
manifest hash, durable copy, trusted root, symlinks, file existence/completeness,
size, and SHA-256 of every listed artifact before parsing content. It then rejects
malformed/empty/non-monotonic history, inconsistent summary facts, and invalid or
oversized density grids.

Important fields:

- `source`: verified run identity, directory, prefix, and manifest hash.
- `convergence`: outer convergence, iteration cap, continuation, move-limit
  pinning, plateau/oscillation heuristics, and optimizer warning count.
- `metrics.constraints`: signed volume error and tolerance result, optional
  mechanism compliance-bound result, and density-bound result.
- `quality_flags`: recomputed grayness/binarization plus checkerboard,
  disconnected-component, and per-traction/spring support-connectivity heuristics.
- `plots`: verified solver image and/or deterministic derived plots.
- `narrative`: deterministic prose generated only from the structured evidence.

Checkerboard method `binary_2x2_alternation_v1` and connectivity method
`component_labels_filter_scaled_dilation_v1` are diagnostics, not proofs of
manufacturability or structural load paths. Connectivity thresholds scale with
mesh/filter length and each relevant region is reported separately.

## 4. Lifecycle and artifacts

Lifecycle states are:

- `queued`: directory and durable job record exist; worker has not started.
- `running`: the child PID is recorded and the serial capacity lock is held.
- `succeeded`: worker returned a valid success and a verified manifest was built.
- `failed`: validation/worker/result processing ended unsuccessfully.
- `timed_out`: the parent terminated the process group after the trusted deadline.
- `cancelled`: a trusted cancellation request terminated the process group.
- `orphaned`: restart recovery found a stale running job whose parent no longer
  exists.

Terminal failures retain partial artifacts with `complete=false`. Tool 3 cannot
accept them because it requires a success-only manifest.

Successful manifest artifact records use paths relative to `run_directory` and
include role, format, byte size, existence/completeness literals, and SHA-256.
Common roles are:

| Role | Meaning |
|---|---|
| `density_history` / `_data` | Physical-density XDMF timeline and HDF5 sidecar |
| `displacement_history` / `_data` | Displacement XDMF timeline and HDF5 sidecar |
| `run_log` | Human progress plus flushed structured `history` records |
| `summary` | Atomic final facts for the same evaluated design |
| `density_grid` | Optional bounded NPZ used for deterministic topology heuristics |
| `density_snapshot_png` | Optional renderer output; failure does not invalidate a solve |
| `worker_stdout` / `worker_stderr` | Captured child transports and local debugging |
| `job_manifest` | Durable lifecycle/process record |
| `run_manifest` | Durable success manifest; listed in the response, not recursively inside itself |

The manifest hash and artifact hashes detect accidental/stale/tampered local
evidence; they are not signatures and do not defend against an attacker who can
rewrite the entire trusted results root. Solver evidence listed by the manifest is
immutable. Tool 3 may create derived plot files beside it; those derived plots are
not solver evidence and are not added to the original manifest.

## 5. Errors and retryability

`stage` tells the orchestrator what kind of action is possible:

| Stage family | Typical codes | Normal handling |
|---|---|---|
| Request/structural | Pydantic field codes, `malformed_request`, `malformed_json` | Correct the request; do not retry unchanged |
| Semantic/geometry | `external_load_required`, `region_matches_no_facets`, `rigid_body_modes_unconstrained`, traction/spring/passive-zone conflict codes | Clarify or deterministically recompile physics |
| Resource/safety | `element_limit`, `memory_limit`, `work_limit`, `estimated_timeout` | Reduce mesh/iterations or change trusted policy outside the LLM |
| Filesystem/idempotency/capacity | path/symlink/disk errors, `idempotency_conflict`, `job_already_active` | Inspect lifecycle; resume/replay rather than launch a duplicate |
| Numerical/solve | `linear_solve_diverged`, non-finite/bounds codes, OC/MMA failure codes, `unexpected_solver_error` | Preserve evidence; do not have the LLM blindly rerun |
| Timeout/cancel/crash | `worker_timed_out`, `worker_cancelled`, `worker_crashed` | Treat lifecycle as terminal; retry only under explicit application policy |
| Artifact analysis | manifest/hash/size/checksum/history/summary/grid codes | Reject the narrative; repair or rerun from trusted inputs |
| Internal | `internal_error`, `worker_result_invalid` | Use local logs/debug artifacts; no traceback is public |

`retryable=true` is a hint that external state or trusted application policy may
make a retry reasonable. It never authorizes an automatic expensive rerun, and it
does not mean the identical request will succeed immediately. The deterministic
orchestrator owns retry counts and idempotency; the LLM does not call Tool 2 in a
free-running retry loop.

CLI exits are `0` for tool success, `1` for a valid tool error envelope, and `2`
for transport read/write failure. CLI stdout is one JSON object unless `--output`
is used, in which case stdout is empty. Logs and progress use stderr/captured files.

## 6. Commands and test tiers

See the README for the complete validate/run/analyze example and MCP server
command. The direct Python functions are:

```python
from fenitop.tools.validate_config import validate_config_tool
from fenitop.tools.run_topopt import run_topopt_tool
from fenitop.tools.analyze_results import analyze_results_tool
```

Use focused tests during development and the complete gate before a checkpoint:

```bash
# Fast contract + deterministic analysis checks
docker compose run --rm -T fenitop python -m unittest \
  tests.test_config_validation tests.tools.test_analyze_results \
  tests.tools.test_total_boundaries -v

# Process/transport checks
docker compose run --rm -T fenitop python -m unittest \
  tests.tools.test_worker_lifecycle tests.tools.test_transports -v

# Full pinned gate, including both numerical baselines
docker compose run --rm -T fenitop python -m unittest discover -v
```

The root `tests/__init__.py` is intentional; it prevents nested tests from being
silently skipped by unittest discovery.

## 7. Event trace for the future agentic layer

The UI/orchestrator should expose an inspectable event/evidence trace:

```text
user request
→ interpreted intent or focused clarification
→ deterministic normalized config
→ validation findings + entity report + resource estimate
→ run ID and queued/running/terminal lifecycle progress
→ verified manifest identity
→ structured analysis evidence
→ optional explanation
```

This trace contains decisions, inputs, tool results, and evidence. It must not
request, store, or display hidden chain-of-thought. Exact typed objects—not prose
copies—are the resume checkpoints.
