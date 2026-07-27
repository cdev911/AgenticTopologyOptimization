# Tool Hardening Plan

Status: TH-0 and TH-1 complete; TH-2 is next
Created: 2026-07-26
Owner/source of truth for status: `docs/spec.md` §0 and §11

## 1. Purpose and readiness rule

The topology-optimization tools are the trust boundary between natural-language
interpretation and an expensive numerical solve. A stronger model or a better
prompt cannot compensate for a tool that accepts an unsafe path, silently models
the wrong physics, reports a diverged solve as successful, or hands the analyzer
inconsistent artifacts.

For that reason, no `agentic/` implementation starts until the hardening gate in
this document is complete. The target is not a production multi-tenant service.
It is a deterministic, explainable, failure-contained personal/demo workflow with
strong contracts at the places where an LLM can make mistakes.

The gate is passed only when:

1. Every public tool accepts a typed request and returns a versioned,
   JSON-serializable success/error envelope for every JSON-shaped input; malformed
   input never escapes as an exception.
2. The agent-facing surface cannot execute source strings, select arbitrary
   filesystem paths, change PETSc settings, or bypass cost/resource limits.
3. A successful run means the numerical solves converged and all reported final
   metrics/artifacts describe the same evaluated design state.
4. `analyze_results` can consume Tool 2's result directly, without an LLM copying
   paths or resupplying the config.
5. The solver runs in a child process with a timeout and without the LLM API key.
6. CLI stdout is exactly one JSON document and stdio MCP framing is not polluted.
7. The security, adversarial, numerical, subprocess, and run→analyze tests in
   §5–§6 pass in the pinned Docker environment.

## 2. Confirmed audit findings

The 2026-07-26 audit ran the intended Docker test modules explicitly: 59 tests
passed, including real Dolfinx geometry checks and end-to-end smoke solves. A real
Tool 2→Tool 3 run also succeeded. The happy path is therefore genuine.

The same audit confirmed the following gaps; these are plan inputs, not speculative
future enhancements:

- Solver progress logging and XDMF status prints go to stdout. A real CLI run
  produced mixed log+JSON output that could not be parsed as one JSON document;
  stdio MCP is exposed to the same pollution.
- Agent-authored marker strings can reach `eval()`. The existing declarative region
  DSL is safer, but its schema uses `Any`, permits unknown fields, and does not
  constrain recursion, dimensionality, or all numeric edge cases.
- `run_topopt` accepts agent-controllable path components and exposes
  `allow_large_run`, `max_complexity_override`, `scoped_output`, and PETSc/output
  tuning that should be application-owned capabilities.
- Missing config paths and invalid mesh ghost modes can raise past Tool 1; malformed
  analyzer inputs can also raise. Tool 2's catch boundary covers the solver call,
  not every setup/postprocessing failure.
- A one-component traction, mechanism springs matching no node, fully overlapping
  solid/void zones, and unknown region fields can pass current validation.
- Mechanism-mode load presence, positive spring stiffness, passive-zone conflicts,
  overlapping loads, and all solver-relevant geometry are not completely checked.
- The current cost score (`num_elements * max_iter`) does not independently cap
  mesh memory. A huge mesh with one iteration can reach real mesh construction.
  It also ignores solver type, primal/adjoint count, and artifact memory/I/O.
- `initial_density` is accepted and normalized but the solver initializes from
  `vol_frac`.
- The initial density XDMF snapshot is written before filter/projection has
  populated the physical-density field.
- Iterations update the design variable and then save the pre-update physical
  density. At termination the density artifact, compliance/volume, design change,
  and grayness can refer to different state boundaries.
- Elasticity and density-filter PETSc convergence reasons are not inspected, and
  finite-value checks are absent. MMA inner-iteration exhaustion is only logged.
- Nonzero full-vector Dirichlet values are accepted, but the assembled RHS does not
  perform the lifting needed for a correct nonzero prescribed displacement.
- The 2D tool supports full-vector clamps, not component-wise/roller constraints;
  the formulation is plane strain with implicit unit thickness. These capability
  limits are not explicit enough for a natural-language interpreter.
- Repeated in-process runs leak/retain logger and PETSc resources; tests emitted
  unclosed file-handler warnings.
- MPI execution is not a sound Tool 2 contract today: each rank can generate a
  different default run ID/output directory and multiple ranks can emit responses.
- Tool 2's preferred result envelope contains the normalized config indirectly,
  but Tool 3 does not reuse it automatically. Target volume, move limit, and
  load-path connectivity are lost unless the caller supplies the config again.
- Tool 3 can return `status="ok"` for an empty history. Several quality heuristics
  are uncalibrated or mesh-dependent, and constraint satisfaction is not reported.
- Default test discovery from the repo root found zero tests, so a green command
  can accidentally mean that nothing ran.
- The Docker base image and most dependencies are moving/unbounded, which weakens
  numerical reproducibility and demo reliability.

## 3. Target boundary and data flow

The LLM does not receive a general-purpose solver function. It produces a typed
problem intent; deterministic code owns compilation, validation, execution, and
artifact routing.

```text
free text
   |
   v
ProblemIntent interpreter (LLM)
   |-- needs_clarification --> chat question, no solve
   |-- unsupported ---------> capability explanation, no solve
   `-- ready ----------------> deterministic config compiler
                                   |
                                   v
                         AgentSafeConfig (DSL only)
                                   |
                                   v
                     validate_config (pure + geometry)
                                   |
                                   v
                    trusted RunRequest (application-owned
                    run ID, output root, limits, solver profile)
                                   |
                                   v
                       isolated solver subprocess
                                   |
                                   v
                    self-contained, versioned RunManifest
                                   |
                                   v
                      deterministic analyze_results
                                   |
                                   v
                       optional LLM explanation layer
```

There are deliberately two configuration surfaces:

- **AgentSafeConfig**: JSON-only physics/problem definition. Region DSL only. No
  paths, API/runtime controls, raw Python, PETSc dictionary, MPI size, safety
  override, or output naming.
- **TrustedRunPolicy/RunRequest**: constructed by application code. Chooses the
  pinned solver profile, output root, timeout, memory/mesh ceilings, rendering
  policy, and idempotency key. This object is not exposed as an LLM tool schema.

JSON config files do not retain a lambda-string escape hatch: migrate the reference
configs and remove string `eval()` from the JSON loader. Legacy hardcoded Python
examples may still construct callable markers directly in Python; those callables
never enter a JSON/tool/LLM contract.

## 4. Workstreams and implementation order

### TH-0 — Characterization, contract freeze, and reproducibility

Goal: capture current intended behavior before changing it and make every later
failure attributable.

Actions:

- Pin the Dolfinx base image by version/digest and current Python dependencies,
  including Pydantic, MCP, and plotting; use the same exact-pin policy when CrewAI
  is added after the tool gate.
- Record Python, Dolfinx, PETSc, MPI, NumPy/SciPy, git commit, config hash, and tool
  contract version in every run manifest.
- Make the test entry point unambiguous (`tests/__init__.py`, a documented command,
  and/or a small test runner); fail when zero tests are collected.
- Add baseline serial smoke results for compliance and mechanism modes. Compare
  with physically meaningful tolerances, not byte-for-byte floating-point output.
- Add small unit characterizations for initial/final state semantics, current
  grayness meaning, filter behavior, and optimizer stop conditions before editing
  those areas.
- Decide and document v1 execution scope: the agent workflow is serial. Legacy MPI
  examples remain separate until Tool 2 has a deliberately designed MPI worker.

Exit criteria:

- One pinned container build runs the complete named suite reproducibly.
- Zero-test collection is an error.
- Both problem modes have a fast numerical baseline and expected artifact manifest.

Completion (2026-07-26):

- Pinned the Dolfinx base by digest, the Ubuntu packages installed on top, Python
  3.12 compatibility, all direct dependencies, and their PyPI dependency closure.
- Froze the existing envelope as tool contract `0.1.0`; the typed redesign in TH-1
  will version any incompatible replacement.
- Added reliable root discovery, serial compliance/mechanism fixtures and
  tolerance-based references with runtime/config hashes, a uniform-filter
  characterization, stop-condition characterization, and an expected-failure test
  for the known ignored-`initial_density` defect.
- Verified the rebuilt image and all 64 root-discovered tests (one intentional
  expected failure for the TH-3 `initial_density` fix). Agent workflow execution
  scope remains serial as declared in `docs/spec.md` §3.

### TH-1 — Typed, agent-safe contracts

Goal: make invalid states difficult to express and remove capabilities the model
does not need.

Actions:

- Add `schema_version`/`contract_version` and strict Pydantic request/response
  models for all three tools.
- Replace `Any` region fields with a discriminated union:
  `plane`, `range`, `circle`, `all`, `none`, `and`, `or`, and `not`. Forbid extra
  fields; enforce 2D axes, finite values, positive tolerances/radii where
  appropriate, bounded nesting depth/node count, and `rtol=0` plane matching.
- Replace positional spring arrays with a named `MechanismSpring` model containing
  `region`, `direction`, and positive `stiffness`.
- Constrain every 2D vector to exactly two finite numbers.
- Separate problem physics from execution settings. Remove output folder/prefix,
  ghost mode, PETSc options, safety overrides, and rendering controls from the
  agent-authored schema.
- Make supported physics explicit: rectangular 2D mesh, plane strain, unit
  thickness, distributed traction (not total force), consistent user-supplied
  units, and full-vector zero clamps for v1.
- Either reject nonzero prescribed displacement and component-wise supports for v1,
  or implement and test correct lifting/subspace BCs before advertising them.
- Migrate the two reference configurations to the DSL and remove lambda-string
  `eval()` from JSON materialization. Hardcoded Python callables remain an internal
  library capability, not a serialized config feature.
- Export and snapshot-test the actual CrewAI/MCP-facing JSON Schema. A generic
  `additionalProperties: true` config is a failure.
- Use structured warning/error records (`code`, `path`, `message`, `severity`,
  `retryable`) rather than free-form strings alone.

Exit criteria:

- No executable string is representable in AgentSafeConfig.
- Tool schemas fully describe regions, springs, vectors, and required conditional
  fields.
- Execution/safety/path capabilities are absent from the LLM-visible schema.
- Reference configs validate and solve through the new safe path.

Completion (2026-07-26):

- Introduced strict contract `1.0.0` and config schema `1.0`, with typed requests,
  responses, nested metrics/analysis/error records, and structured issues for all
  three tools.
- Split physics-only `AgentSafeConfig` from trusted validation/run/analysis
  policies. The actual LLM/MCP schema has no path, PETSc, ghost-mode, rendering,
  timeout, output, or safety authority.
- Replaced serialized marker strings and positional springs with the bounded,
  finite, exact-2D discriminated region DSL and named positive spring models.
  Removed string evaluation from JSON materialization and migrated all reference
  and smoke configs.
- Made the supported plane-strain/unit-thickness/distributed-traction/full-vector
  zero-clamp semantics explicit; nonzero and component-wise displacement
  constraints are rejected for agent-safe v1.
- Snapshot-tested the real MCP input/output schemas and runtime unknown-argument
  rejection, including the pinned MCP 1.28.1 outer-model strictness workaround.
- Contained Tool 3 artifact reads beneath application-owned allowed roots, with
  resolved-path checks that also reject symlink escapes from fabricated envelopes.
- Verified 50 root-discovered tests (one intentional TH-3 expected failure) and
  both real compliance/mechanism smoke solves with unchanged TH-0 numerical
  metrics.

### TH-2 — Complete semantic and geometry validation

Goal: reject configurations that are syntactically valid but cannot produce a
meaningful supported solve.

Pure/structural checks:

- Finite mesh bounds and material/optimizer values; positive extents and divisions.
- Independent limits for elements, nodes/DOFs, max iterations, estimated artifact
  size, and total complexity.
- Reasonable element aspect ratio and resolution warnings; filter radius checked
  against both element axes and domain extent.
- `0 < vol_frac < 1`, `0 < move <= 1`, positive `epsilon`, meaningful
  `beta_interval/beta_max`, and initial density consistent with passive zones.
- A nonzero external load for both compliance and mechanism modes.
- Mechanism-only requirements: positive compliance bound, positive spring
  stiffness, required input/output regions, and a material/spring scaling policy.
  Young's modulus cannot be treated as topology-invariant when springs introduce an
  external stiffness scale.
- Reject solid/void overlap and contradictory settings.
- Reject unsupported partial supports/nonzero displacement until implemented.

Mesh-backed checks:

- Every support and traction region matches the intended boundary entities.
- Supports resist all rigid-body modes and do not prescribe conflicting values.
- Detect overlapping traction regions; either sum them deliberately or reject them.
  Never silently keep the first.
- Input/output spring regions match degrees of freedom, are not unintentionally
  identical/overlapping, and are compatible with supports.
- Solid/void zones match cells when specified, remain disjoint, and do not erase
  required support/load/spring neighborhoods.
- Report entity counts and physical locations in the normalized validation result
  so interpretation mistakes are inspectable.

Cost/resource policy:

- Reject an oversized mesh before any Dolfinx mesh build, even with one iteration.
- Estimate separate memory and work components: mesh/DOFs, cell type, number of
  primal/adjoint/filter solves, solver profile, iterations, snapshots, and render
  overhead.
- Calibrate thresholds from measured smoke/medium runs in the pinned container.
  Thresholds are trusted application policy and never agent-overridable.

Exit criteria:

- Every known bad case in §2 has a stable field-level error.
- Huge-mesh/one-iteration input is rejected by pure arithmetic.
- Both supported reference problems pass full mesh-backed validation.

### TH-3 — Numerical correctness and explicit solver state

Goal: make `status="ok"` mean the numerics are trustworthy within the supported
scope.

Actions:

- Check PETSc convergence reason and residual norms after every elasticity,
  adjoint, and density-filter solve. Convert divergence into a typed numerical
  failure with iteration/solver context.
- Check all fields, objectives, constraints, sensitivities, optimizer updates, and
  summary metrics for NaN/Inf and valid density bounds.
- Add finite-difference checks on tiny meshes for compliance, volume, and mechanism
  objective/constraint sensitivities through the filter/projection chain. An
  optimizer that runs without crashing is not sufficient evidence that gradients
  drive the right problem.
- Make MMA subproblem status explicit. Repeated inner-iteration caps, singular
  subproblems, failed line searches, or non-finite updates must not be silently
  accepted.
- Guard OC square-root/bisection assumptions and report infeasible/non-descent
  updates clearly.
- Honor `initial_density`, initialize the physical-density field before writing
  iteration 0, and test passive-zone initialization.
- Define an iteration-state contract. Recommended:
  1. evaluate the current design,
  2. log/save that evaluated state,
  3. compute the update,
  4. on termination, apply filter/projection and evaluate the final updated design
     once more,
  5. produce summary, density, displacement, and analysis inputs from that same
     final evaluation.
- Compute clearly named metrics from physical density. Replace the ambiguous
  current “grayness” field with both a conventional grayness score (higher means
  more intermediate material) and a binarization score (higher means more binary).
- Record final beta and whether projection continuation completed. Do not label a
  low-change design fully converged if the required continuation stage is
  incomplete.
- Close log handlers and PETSc objects deterministically; child-process isolation is
  a second line of defense, not a substitute for correct cleanup.

Exit criteria:

- Fault-injected PETSc/filter/optimizer failures return numerical error envelopes.
- Initial and final artifacts match their logged metrics.
- No unclosed-file warnings occur in repeated runs.
- Compliance and mechanism baselines remain within documented tolerances.
- Tiny-mesh analytical/adjoint sensitivities agree with finite differences within
  documented numerical tolerances.

### TH-4 — Filesystem safety, idempotency, and subprocess containment

Goal: contain expensive/native execution and make retries safe.

Parent/orchestrator responsibilities:

- Generate a slugged run ID and fixed output prefix; resolve a configured results
  root once and prove every run path remains beneath it.
- Reject absolute paths, traversal, separators, glob metacharacters, control
  characters, reserved names, and excessive lengths in any remaining identifier.
- Create run directories itself with exclusive semantics and reject symlinked run
  components/artifacts before containment checks or writes.
- Do not perform glob-based deletion from untrusted text. Fresh run directories are
  immutable; overwrite mode is developer-only and outside the agent workflow.
- Add a canonical-JSON config/request hash and idempotency key. Duplicate kickoff
  returns the existing run state/result rather than starting a second solve; use a
  lock/atomic-create operation so concurrent UI events cannot race past the check.
- Enforce one active solve (or an explicit small trusted limit) for the demo.
- Check available disk space and estimated output size before launch.

Worker responsibilities:

- Add a dedicated solver-worker entry point with typed JSON input and a result
  manifest written atomically.
- Launch it as a child process in the same container, with a fixed working
  directory, sanitized environment, no `OPENAI_API_KEY`, captured stdout/stderr,
  wall-clock timeout, and process-group termination on timeout/cancel.
- Record exit code, terminating signal, timeout/cancel state, and the last durable
  iteration. Native crashes/OOM cannot be converted by Python inside the worker, so
  the parent translates process death into a stable envelope when it survives.
  A hard cgroup/container OOM can also kill the parent; independent pre-mesh memory
  ceilings are therefore the primary defense. Per-run containers remain a later
  escalation if subprocess isolation proves insufficient for the demo.
- Maintain an atomic lifecycle manifest:
  `queued | running | succeeded | failed | timed_out | cancelled | orphaned`.
- Preserve partial artifacts but mark them incomplete. Never pass incomplete XDMF
  or summaries to analysis as completed outputs.
- On application restart, detect stale `running` manifests and mark them orphaned.

MPI:

- Agent v1 rejects `mpi_processes != 1`.
- If MPI support is later added, the parent creates/broadcasts one run ID and path,
  only rank 0 writes the response manifest, and an MPI-specific integration test is
  mandatory.

Exit criteria:

- Traversal/glob/safety-override attempts cannot affect files outside the run root.
- The worker environment does not contain the API key.
- Timeout, cancellation, crash, duplicate request, and restart recovery tests pass.

### TH-5 — Clean transports and total tool boundaries

Goal: one implementation with trustworthy direct, CLI, and MCP behavior.

Actions:

- Route all logs/progress to stderr or run log; replace solver `print()` calls.
- Reserve worker/CLI stdout for exactly one JSON response.
- Put a final exception boundary around each public tool while retaining precise
  local catches for expected validation, I/O, numerical, and artifact errors.
- Validate the request itself before calling `.get()` or opening paths.
- Do not expose raw tracebacks to the LLM/UI. Return a stable public error and a
  debug/log reference; keep full tracebacks in local logs.
- Verify every success/error response against its response model and JSON encoding.
- Add CLI tests for stdin, file input, stdout, output file, malformed JSON, and exit
  codes.
- Add an actual stdio MCP client/server integration test that calls all three tools
  and proves progress output cannot corrupt framing.

Exit criteria:

- Captured CLI stdout parses as exactly one response object.
- All fuzz/adversarial JSON-shaped requests return schema-valid envelopes.
- MCP calls survive a real solver run with progress logging enabled.

### TH-6 — Self-contained artifacts and deterministic analysis

Goal: make Tool 3 a reliable consumer of Tool 2, not another place for an LLM to
reconstruct state.

RunManifest:

- Include contract/schema version, run/config IDs and hashes, normalized safe
  config (or immutable local reference plus embedded fields needed for analysis),
  solver/runtime versions, lifecycle/numerical status, convergence state, final
  evaluated metrics, warnings, and artifact records.
- Each artifact record includes role, path relative to the run directory, format,
  size, existence/completeness, and checksum where practical.
- Write summary/manifest atomically. Verify listed artifacts before returning
  success.

Analyzer:

- Accept a successful RunManifest as the primary input. Standalone legacy
  folder/prefix mode remains developer-only and path-contained.
- Reject missing/empty/corrupt histories and inconsistent summary/manifest data.
- Read `.npz` with pickle disabled; validate expected keys, bounded shapes/file
  sizes, monotonic coordinates, finite values, and config-consistent grid
  dimensions before quality analysis.
- Automatically use the run config for volume target, move limit, compliance bound,
  beta schedule, and support/load/spring regions.
- Report constraint satisfaction explicitly: volume error, compliance-bound result
  for mechanism mode, density bounds, and numerical warnings.
- Improve convergence diagnostics: tolerance, iteration cap, final beta/continuation,
  move-limit pinning, oscillation/plateau, MMA warnings, and solver divergence.
- Compute grayness/binarization from the exported physical density using named
  formulas.
- Treat checkerboard/connectivity as heuristics with method/version and thresholds,
  not facts. Calibrate on synthetic uniform, binary, checkerboard, disconnected,
  and known-good designs.
- Make connectivity physical/mesh-aware: evaluate each load/spring region against
  supports, use a distance tied to element/filter scale rather than a fixed
  two-pixel dilation, and report a per-region result rather than one union boolean.
- Plot filenames come from the trusted manifest, never an LLM-supplied prefix.
- Keep deterministic narrative generation; an optional LLM may explain the
  structured result later but cannot change its facts.

Exit criteria:

- `analyze_results({"run_manifest": manifest})` requires no duplicated config/path.
- Empty/corrupt/incomplete runs fail cleanly.
- Synthetic heuristic tests and both real baselines produce expected flags.

### TH-7 — Documentation and tool usability

Goal: make supported semantics and failure behavior clear to humans and models.

Actions:

- Update README tool claims only when the corresponding behavior is true.
- Document exact physical semantics and limitations: 2D rectangle, plane strain,
  unit thickness, traction/body-force meaning, support capability, mechanism spring
  scaling, filter length units, and supported problem types.
- Document request/response models, error codes, retryability, run lifecycle, and
  artifact meanings.
- Replace “reasoning trail” observability language with an inspectable event trace:
  interpreted intent, clarification, normalized config, validation findings,
  resource estimate, run progress, and analysis evidence. Do not depend on hidden
  chain-of-thought.
- Add handover notes to `docs/spec.md` after every hardening checkpoint and keep
  this plan's item IDs in commit/decision notes.

Exit criteria:

- A new contributor can identify capabilities, run the full test suite, interpret
  every result field, and resume from `docs/spec.md` without chat history.

## 5. Failure-scenario matrix

Each row requires at least one automated test before the gate passes.

| Scenario | Expected behavior |
|---|---|
| Missing/wrong/extra config fields | All field errors returned together; no exception |
| Unknown/deep/oversized region DSL | Typed validation error before compilation |
| Lambda/source string in agent config | Rejected as an invalid region type |
| NaN/Inf/extreme numeric values | Rejected before mesh construction |
| Huge mesh, one iteration | Rejected by independent memory/DOF ceiling |
| Tiny mesh, enormous iterations | Rejected by work/timeout policy |
| No load | Hard validation error for both problem modes |
| Marker/spring/zone matches nothing | Entity-specific geometry error |
| Overlapping traction or solid/void regions | Explicit reject or documented combine rule |
| Nonzero/partial prescribed displacement | Rejected until correctly implemented |
| PETSc/filter divergence | Numerical failure with solver reason and iteration |
| NaN sensitivity/optimizer update | Numerical failure; no success manifest |
| MMA inner solver fails repeatedly | Non-converged numerical status, not a quiet warning |
| Renderer unavailable | Solve remains valid; structured optional-artifact warning |
| Disk full/read-only output | I/O failure with no false success |
| Timeout/cancel/native crash | Parent returns terminal lifecycle state and partial inventory |
| Duplicate kickoff/UI rerun | Existing idempotent run returned; no duplicate solve |
| Traversal/absolute/glob identifier | Rejected; no filesystem effect outside run root |
| Worker environment inspection | API key absent |
| CLI run with progress | Stdout remains one JSON document |
| MCP run with progress | JSON-RPC remains valid |
| Empty/truncated/corrupt artifacts | Analyzer returns error, not a narrative |
| Tool 2→Tool 3 direct handoff | Full target/constraint/connectivity context retained |
| Repeated serial runs | No open-handler/PETSc resource warnings |
| MPI request through agent surface | Explicit unsupported error in v1 |

## 6. Test layers

1. **Pure contract/unit tests** — Pydantic schemas, region DSL, config compiler,
   envelopes, cost/memory estimates, path containment, manifests, narratives.
2. **Generated adversarial tests** — nested JSON-shaped values, wrong types,
   extreme numbers, unexpected nulls/lists/dicts, and round-trip JSON serialization.
   The invariant is “never raise past the public boundary.”
3. **Docker geometry tests** — real entity matching for supports, loads, springs,
   passive zones, overlaps, and rigid-body modes.
4. **Tiny numerical tests** — compliance and mechanism solves, finite-difference
   sensitivity checks, initial/final state, filter/solver convergence checks,
   finite metrics, and expected constraints.
5. **Fault-injection tests** — forced worker exception, PETSc failure, render failure,
   timeout, cancellation, malformed output, disk error, and child termination.
6. **Transport tests** — direct function, CLI JSON purity/exit codes, and actual MCP
   stdio calls.
7. **Composition tests** — validate→run worker→manifest→analyze with no LLM and no
   duplicated config.
8. **Manual calibrated runs** — reference beam and mechanism in the pinned image;
   record runtime/memory and update resource thresholds.

CI/default local tests should run fast pure tests automatically. Docker numerical,
fault, transport, and calibrated suites may be separate named jobs, but the command
and last known result must be visible in `docs/spec.md`.

## 7. Implementation sequence and review checkpoints

Implement in this order to avoid building later work on an unstable contract:

1. TH-0 characterization and dependency/runtime pinning.
2. TH-1 typed safe schema and execution-capability split.
3. TH-2 complete validation and resource policy.
4. TH-3 numerical/final-state correctness.
5. TH-4 subprocess, filesystem, lifecycle, and idempotency.
6. TH-5 clean CLI/MCP and total boundaries.
7. TH-6 manifest-driven analysis and calibrated heuristics.
8. TH-7 documentation, full regression run, and hardening gate review.

After each workstream:

- run its pure and Docker tests;
- update `docs/spec.md` §0 and §11;
- add a §9 decision entry only if implementation required a new design choice;
- update this file if the accepted behavior or exit criterion changed;
- do not begin the next workstream with unexplained failures or stale documentation.

## 8. Definition of done

Tool hardening is complete when all workstream exit criteria pass and a final review
demonstrates:

- a valid compliance request and a valid mechanism request complete through
  manifest-driven analysis;
- unsupported config capabilities cannot cross the agent-safe typed boundary;
- every failure class in §5 returns a stable, useful result;
- the application—not the LLM—owns execution authority;
- final numerical facts and artifacts are internally consistent;
- the child solve cannot expose the API key or corrupt the parent transport;
- the pinned environment and documented commands reproduce the result.

Only then does `docs/spec.md` move the project to the deterministic agentic
orchestration stage.
