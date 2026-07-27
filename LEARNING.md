# Engineering Lessons

This project began with a simple idea: let an LLM operate a topology-optimization
solver. The useful learning came from discovering why a prompt plus a Python
function was not enough, then repairing each weak boundary until the complete
workflow behaved like a dependable tool.

## 1. The agent is only as capable as its tools

The first priority was not orchestration. It was making the scientific solver
safe and legible to an agent.

The final tool surface has three operations:

1. `validate_config` checks the schema, physics, resources, and actual mesh
   entities without starting an optimization.
2. `run_topopt` performs one validated solve in a contained process.
3. `analyze_results` verifies the run manifest and derives result facts.

This separation is important. Validation is cheap and repeatable; execution is
expensive and state-changing; analysis consumes evidence. A single “run anything”
tool would hide those different trust levels.

The model receives a declarative physics contract, not operational control. Paths,
run IDs, timeouts, retry policy, concurrency, PETSc settings, and resource ceilings
remain application policy.

## 2. Structured output does not guarantee semantic truth

A JSON response can match a Pydantic schema and still be physically wrong. The
system therefore validates in layers:

- strict types and versioned contracts;
- cross-field physics rules;
- resource admission before mesh construction;
- mesh-backed checks for actual nodes and boundary facets; and
- numerical invariants during the solve.

One early request described a centered load segment correctly but matched zero
facets after discretization. The repair was not more prompting. It was a shared
mesh-aware resolver used by both validation and FEM assembly, with requested and
resolved extents shown to the user.

The broader lesson: a tool should validate the meaning of an action in the world,
not only the shape of the request.

## 3. Conversation needs application-owned memory

The first formulator expected a nearly complete, schema-shaped prompt. Real users
describe a problem over several turns, correct themselves, and use different
words for the same boundary condition.

The live workflow now stores a typed partial draft outside the model. Every field
records its value, source turn, basis (`explicit`, `derived`, `assumption`, or
`confirmed`), and revision history. Boundary conditions are first-class entities
with stable IDs such as `S1` and `L1`, so “move L1 upward” changes one load rather
than replacing the entire list.

Provider continuation helps the model reason naturally, but the provider's hidden
conversation state is never canonical. Every turn also receives the application
draft, and deterministic code alone decides readiness.

## 4. Preserve semantics until the right layer can resolve them

The model is good at understanding that “the middle 10% of the right edge” means a
centered segment. It should not calculate its coordinates or guess which mesh
facets represent it.

The intermediate representation therefore keeps:

- named rectangle edges;
- fraction, coordinate, physical-width, and corner-offset selectors;
- traction, pressure, and total-resultant meaning;
- global and edge-relative directions; and
- user units plus normalized units.

Only deterministic geometry code converts those semantics using known bounds,
mesh divisions, thickness, and units. This preserves intent and makes the
conversion inspectable.

## 5. Boundary conditions deserve first-class design

Boundary descriptions were the hardest part of the language interface. A useful
system had to understand clamps, rollers, symmetry, pins, pressures, tractions,
resultants, partial edge spans, corners, corrections, and ambiguous words such as
“width.”

The main design changes were:

- stable support/load identities;
- partial per-field state instead of monolithic lists;
- explicit quantity semantics rather than a generic load vector;
- true component constraints for rollers and symmetry;
- a boundary-node selector for pins instead of approximating one with a tiny
  clamp;
- deterministic direction and pressure resolution;
- unit-aware total-force distribution over the resolved boundary measure; and
- approval evidence that shows what the mesh will actually receive.

The resulting pattern applies to many agent systems: complex domain objects
should have identities, partial state, provenance, and targeted operations.

## 6. Human approval is a state transition, not a phrase in a prompt

A validated problem is not executable. The workflow stops in
`awaiting_run_approval` and presents all defaults, warnings, resolved selectors,
load conversions, and estimated cost.

Only an unambiguous whole-message approval causes the application-owned
transition to an executable state. A requested edit immediately revokes the
previous proposal, even if the new model call later fails. Durable idempotency
prevents UI reruns from launching a duplicate solve.

This is a general pattern for agents that send messages, spend money, modify
records, or control physical/numerical tools: authorization belongs below the
chat layer.

## 7. Native computation needs containment

PETSc and MPI failures are not ordinary Python exceptions. Every optimization
runs in a separate child process with:

- a sanitized environment and no `OPENAI_*` credentials;
- application-owned timeout and cancellation;
- durable queued/running/terminal lifecycle state;
- crash and orphan detection; and
- success publication only after artifacts are complete.

Docker provides the pinned scientific environment; the child-process boundary
contains each solve inside that environment.

## 8. Evidence must survive the handoff

A directory path is not a trustworthy result. Successful runs produce a manifest
with relative paths, sizes, hashes, and completion metadata. Analysis verifies
containment and SHA-256 before deriving plots or facts.

The explainer is also evidence-bound. The model may organize known fact IDs into
allowed sections, but deterministic code checks completeness and renders the
original fact text. Unknown, duplicated, or omitted required facts fail closed.

The useful split is:

- deterministic systems establish facts;
- models organize and communicate those facts.

## 9. Evaluate layers separately

No single test type proved the workflow.

- deterministic tests covered schemas, merges, state transitions, validation,
  approval, evidence, UI behavior, and adversarial inputs;
- numerical baselines checked FEM construction and optimization behavior;
- a fixed live corpus measured language understanding without solver execution;
- realistic multi-turn acceptance scenarios tested the entire user journey.

The first five-scenario acceptance run was especially valuable. Two scenarios
completed end to end, while the others exposed a lost named-corner pin, an
under-specified spring contract, warning/explanation gaps, and a numerical failure
after a projection transition. These issues were repaired at their owning layers
and converted into regressions. The exact numerical failure then completed 60
iterations across the previously failing transition.

Passing unit tests was evidence, not proof that the whole product worked.

## 10. Report numerical status honestly

A solver run can be valid without meeting its convergence tolerance. The result
contract therefore distinguishes:

- successful execution;
- termination at the configured iteration limit;
- completed continuation;
- optimizer convergence; and
- final quality metrics.

The showcased cantilever run is a good example: it produced a strong binary
topology and completed projection continuation, but it reached its iteration
limit, so the result remains `converged=false`.

Numerical tolerances also need measured justification. Tiny density undershoots
from a converged filter solve are clipped only inside a narrow tested tolerance;
larger violations still fail. A sensitivity safeguard uses a scale-aware
noise threshold rather than deleting a physically important sign check.

## 11. Frameworks are means, not the architecture

CrewAI was useful for the first interpreter and the evidence-explanation
integration. The OpenAI Responses API was a better fit for the live multi-turn
formulation role. Deterministic Python remained the right tool for compilation,
workflow authority, and safety.

The important question was never “How can every step be an agent?” It was “Where
does language judgment improve the system, and where must behavior remain exact?”

That decision produced a hybrid architecture with a clearer trust boundary and
better testability than a chain of autonomous agents.

## 12. The final lesson

Calling this a learning project should not justify toy behavior. The learning is
in solving a real problem, measuring failures, choosing the right abstraction,
and documenting why the system changed.

The most transferable outcome is a development method:

1. prepare a small, safe tool contract;
2. create typed intermediate state;
3. keep side-effect authority deterministic;
4. preserve uncertainty and provenance;
5. require human approval for consequential actions;
6. bind explanations to evidence; and
7. let realistic acceptance failures drive the next repair.

That is the agentic engineering demonstrated by this repository.
