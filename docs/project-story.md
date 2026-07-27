# Project Story

## The problem worth solving

An engineer thinks:

> Clamp the left edge, put a downward total load near the middle of the right
> edge, use 30% material, and make the mesh reasonably fine.

A topology-optimization solver needs:

- exact domain bounds and mesh divisions;
- a material model and analysis type;
- boundary entities that resolve to real nodes or facets;
- a distinction between traction, pressure, and total resultant;
- consistent dimensions and plane-strain thickness;
- stable optimizer settings; and
- a controlled place to write expensive results.

The gap between those two descriptions is not a text-generation problem alone.
It is an agentic systems problem: interpret uncertain input, accumulate state,
call strict tools, manage side effects, preserve evidence, and keep a human in
control.

This project implements that complete loop.

## The system I built

The workflow has four cooperating layers.

### 1. Conversational formulation

The formulator accepts facts in any logical order and returns a small structured
patch rather than a complete solver configuration. The application merges that
patch into a canonical `ProblemDraft`.

Each fact records:

- its typed value;
- the source conversation turn;
- whether it was explicit, derived, assumed, or confirmed; and
- its revision history.

Supports, loads, and mechanism springs are first-class entities with stable IDs.
This lets a user say “change L2 to 15 N” without replacing every boundary
condition.

### 2. Deterministic compilation and validation

The compiler owns defaults and converts the conversational representation into a
versioned `AgentSafeConfig`. Validation then checks:

- schema and supported physics;
- units and cross-field consistency;
- rigid-body constraints and duplicate degrees of freedom;
- estimated memory, work, and output size;
- actual mesh nodes and boundary facets;
- requested versus resolved boundary extents;
- traction/resultant conversion; and
- load/support conflicts.

Validation produces evidence used by the approval UI and by the FEM assembly.
The same resolver selects boundary facets in both places.

### 3. Approval and contained execution

The workflow state machine separates:

```text
ready for review → validated → awaiting approval → approved → running
```

Only the explicit `approved` state can start a worker. The worker runs in a child
process without OpenAI credentials. Application policy owns paths, run IDs,
timeouts, cancellation, capacity, and idempotency.

### 4. Verified analysis and explanation

The worker publishes a checksum-bearing manifest. Analysis verifies it before
reading artifacts or deriving plots.

The model does not write a free-form numerical report. It receives immutable fact
IDs and may only organize them. Deterministic code validates the plan and renders
the original fact text.

## Acceptance case study

I wrote five realistic user journeys before treating the interface as finished.
They deliberately varied problem type, phrasing, units, boundary conditions, and
numerical customization.

| Scenario | Main behaviors exercised | First manual run |
| --- | --- | --- |
| Lightweight equipment arm | Casual engineering language, clamp, finite traction patch, automatic mesh | Formulated, approved, solved, and plotted |
| Simply supported transfer beam | Negative origin, true corner pin, roller, pressure, exact triangular mesh | Exposed lost named-corner semantics |
| Near-incompressible polymer bracket | Total resultant, mesh-dependent distribution, warning, casual long/short mesh | Formulated, approved, solved, and plotted |
| Two-load machine bracket | Multiple stable load IDs, correction after proposal, beta continuation | Exposed a numerical tolerance failure after transition |
| Compliant gripper | Different objective, input/output springs, sparse load description | Exposed an under-specified spring interface |

The important outcome was not a convenient “5/5” claim. The run exposed defects
that isolated tests had missed.

### Repairs driven by the run

**Named corners**

A pin described as “lower-left” was retained as language but lost before solver
finalization. Named corners now remain semantic until arbitrary domain bounds are
known, then resolve deterministically to a boundary node with visible snap
evidence.

**Mechanism springs**

The first interface exposed raw solver regions. Springs are now first-class
`I…`/`O…` entities with stable IDs, shared selectors, per-field provenance,
force-per-length dimensional validation, and matched-degree-of-freedom approval
evidence.

**Projection-transition failure**

A real two-load run failed after its projection parameter changed because a
useful sensitivity was treated as a forbidden positive gradient. The safeguard
was preserved but changed to a scale-aware sign-noise threshold. A
material-positive counterexample still fails. The exact former failure then ran
60 iterations across the transition successfully.

**Warnings and explanation**

Mesh-resolution and near-incompressibility warnings now survive into approval
and final evidence. Compliance explanations omit the mechanism-only zero
objective. Solver failures identify their lifecycle stage and last known metrics.

Each repair was added at the layer that owned the invariant, not hidden in a
larger prompt.

## Evidence

### Deterministic and numerical gate

The pinned Docker environment passes:

```text
311 tests + 197 parameterized subtests
pip check: no broken requirements
```

Coverage includes contracts, adversarial input, state transitions, provenance,
approval invalidation, secret scrubbing, idempotency, manifest integrity,
Streamlit behavior, FEM assembly, numerical baselines, and the exact
projection-transition regression.

### Language gate

The released boundary-condition corpus contains 53 fixed cases covering:

- partial and corrected conditions;
- multiple supports and loads;
- coordinate, fractional, physical-width, and corner-offset spans;
- pressure, traction, resultant, units, and directions;
- pins, rollers, and symmetry;
- ambiguous and contradictory descriptions; and
- capability-limit behavior.

The released live formulation contract passed 53/53 with zero solver starts.
Language evaluation is deliberately formulation-only so it cannot spend
numerical compute.

### Real result

The main README gallery comes from a real cantilever run:

| Metric | Value |
| --- | ---: |
| Domain | 600 mm × 200 mm |
| Automatically selected mesh | 87 × 29 quadrilaterals |
| Requested material | 35% |
| Final material fraction | 0.3479 |
| Final compliance | 8.894 |
| Binarization score | 0.9905 |
| Projection continuation | Completed |
| Termination | Configured iteration limit |

The status is intentionally reported as `converged=false`: producing a useful
design is not permission to redefine the convergence contract.

## Design decisions I can defend

### Why not use an autonomous agent for every stage?

Compilation, approval, and execution have exact rules. Adding model judgment
would make them more expensive and less testable. The LLM is used where ambiguity
exists: conversational interpretation and evidence organization.

### Why keep canonical state outside the model?

Provider conversation state improves continuity but cannot provide field-level
provenance, deterministic revisions, migration, or a trustworthy execution
boundary. The model proposes changes; the application owns truth.

### Why three tools?

Validation, execution, and analysis differ in cost and authority. Separating them
enables cheap repair loops and prevents a language mistake from becoming a solve.

### Why validate on a real mesh?

A valid coordinate expression can still select no facets or the wrong segment.
The solver must approve the discretized condition, not only the continuous
description.

### Why retain CrewAI and also use the Responses API?

CrewAI was useful for the first typed interpreter and evidence-explanation role.
The Responses API provided a better primitive for the live multi-turn formulator.
Framework consistency was less important than a clear contract at each boundary.

## What this project demonstrates

- designing agent-facing tools rather than merely wrapping functions;
- structured output, versioned schemas, and provider-independent state;
- multi-turn correction and provenance;
- deterministic guardrails around costly side effects;
- human-in-the-loop approval and stale-approval prevention;
- numerical process containment and secret isolation;
- evidence-bound generation and artifact integrity;
- eval design that separates language, workflow, and numerical quality; and
- using end-to-end failures to improve architecture.

## A concise project pitch

> I built a conversational topology-optimization system that lets an engineer
> describe a structural problem naturally, but keeps the LLM away from numerical
> and execution authority. The model develops a typed draft over multiple turns;
> deterministic code resolves units and mesh boundary conditions, validates the
> physics, and waits for explicit approval. The solver runs in an isolated
> credential-free process, and the final explanation is bound to verified
> evidence. I used a 53-case language evaluation, more than 300 tests, numerical
> baselines, and five realistic acceptance journeys to find and fix integration
> failures.

## Honest scope

This is a local portfolio application, not certified structural-design software
or a hosted multi-user service. It supports a deliberately bounded rectangular
2D plane-strain contract. The first manual five-scenario run was followed by
regression-backed repairs; a second complete billed/UI acceptance pass is not
claimed in this release.

That limitation is part of the engineering story: claims in an agentic system
should stop where the evidence stops.
