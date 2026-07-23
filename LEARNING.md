# What We Learned Building Agentic Topology Optimization

This is the engineering learning record for the project. `README.md` explains
how to use the current system, `docs/spec.md` records current decisions and next
steps, and `docs/tool-reference.md` defines the exact tool contract. This file
explains what we learned while reaching the current state, including the
approaches that failed and the reasons for changing them.

The project is a learning project, but that does not mean deliberately building a
toy. The learning comes from solving the real problem as well as we can,
measuring the result, and documenting the architecture, failures, and tradeoffs.
A framework or a simpler implementation is not a goal by itself. A dependable,
understandable tool is the goal.

## 1. The first lesson: the agent is only as capable as its tools

Our first important realization was that the success of the agentic workflow
depended on the quality of the tools available to it. A language model cannot make
an unsafe, ambiguous scientific library into a reliable agent tool merely by
receiving a better prompt.

We therefore prepared the topology-optimization solver before building the
orchestration around it. The preparation was not just an API wrapper. It required
an explicit contract, validation against real mesh entities, numerical-state
checks, process isolation, durable execution state, and verifiable result
artifacts.

### 1.1 Expose a small operation surface

The prepared tool has exactly three public operations:

1. `validate_config` checks whether a proposed physical problem is meaningful,
   supported, and affordable before a solve.
2. `run_topopt` performs one validated, contained optimization and produces an
   immutable run manifest.
3. `analyze_results` verifies that manifest and derives deterministic facts and
   plots from it.

Separating these operations matters. Validation is cheap and safe to repeat.
Execution is expensive and has side effects. Analysis consumes evidence but must
not silently change the original result. Giving all three jobs to one unconstrained
“run anything” function would hide these different trust levels.

The Python API, JSON command-line interface, and MCP interface use the same
Pydantic request and response models. Transport-specific code does not redefine
the physics. Schema-hash and parity tests protect that promise.

### 1.2 Give the model physics, not operational authority

`AgentSafeConfig` describes physical intent. It does not allow an agent to choose:

- filesystem paths or overwrite behavior;
- run identifiers;
- PETSc or MPI options;
- resource ceilings;
- timeouts or cancellation policy;
- concurrency;
- rendering policy; or
- retry policy.

Those values belong to trusted application policy. This reduced the prompt and
schema surface while preventing a model-generated configuration from becoming an
operational command.

### 1.3 Replace executable selectors with a declarative region language

The original scientific code could express boundary selections with Python
callables. That is convenient for a human developer but unsuitable for an LLM
tool boundary: generated code is hard to validate, serialize, compare, and
sandbox.

We replaced it at the agent boundary with a bounded two-dimensional region
language: planes, ranges, circles, Boolean combinations, `all`, and `none`.
Application code compiles this declarative form into the lower-level selector.
The lesson is broader than geometry: agent-facing tools should accept data whose
meaning can be validated, not code whose behavior must be trusted.

### 1.4 Make failures machine-readable

Tool failures carry a stage, a stable error code, a message, and whether retry
could be useful. This lets the workflow distinguish invalid physics, resource
rejection, native-solver failure, timeout, cancellation, and damaged evidence.

“Retryable” is diagnostic information, not permission to spend more compute. The
application still controls whether another run is allowed.

## 2. Schema validation is necessary but far from sufficient

A configuration can have the correct JSON shape and still describe an impossible
or dangerous problem. We learned to validate in layers:

1. strict parsing rejects unknown fields, invalid types, and non-finite values;
2. cross-field checks enforce supported physics and consistent choices;
3. pure admission checks estimate mesh size, degrees of freedom, memory, output,
   and work before constructing a mesh;
4. mesh-backed checks verify actual nodes, cells, boundary facets, loads,
   constraints, and conflicts.

The mesh-backed stage caught a particularly important class of failure: a region
can be syntactically valid but match zero boundary facets. Without this check a
request may reach the solver with no applied load or insufficient constraints.

Resource validation also belongs before execution. An agent should not discover
that its mesh request is unreasonable only after allocating the mesh or starting
the optimizer.

## 3. Numerical correctness must be part of the tool contract

Wrapping a numerical solver does not automatically make the wrapper correct. We
needed pinned numerical baselines and checks for the solver state itself:

- finite displacement, objective, volume, and sensitivity values;
- successful linear-solver convergence;
- bounded physical density;
- consistent filtering and sensitivity calculations;
- last-known-good metrics when a later step fails; and
- regression baselines for representative configurations.

Two distinctions proved important.

First, a run can finish successfully at its iteration limit without satisfying
the optimizer's convergence tolerance. The tool therefore distinguishes a valid
completed run from `converged=true`. Hiding this distinction would turn an
optimization status into a false claim.

Second, numerical tolerances must reflect the algorithms being used. One real run
produced a filtered density of approximately `-4.83e-7`. Mathematically the
density should be in `[0, 1]`, but this tiny undershoot was discretization
roundoff, not a failed physical state. We now accept and clip at most `1e-5` after
a converged filter solve, while larger violations still fail. The lesson is not
to relax validation casually; it is to encode a measured numerical tolerance and
keep rejecting values outside it.

## 4. Native numerical work needs process containment

PETSc and MPI failures are not ordinary Python exceptions. A native library can
abort the interpreter, exhaust memory, or leave a process unusable. A `try/except`
around the solve is therefore not a sufficient safety boundary.

Every solve now runs in a separate child process. The trusted parent:

- revalidates the request;
- allocates the run ID and paths;
- enforces a single-solve capacity limit;
- records queued and running state;
- applies timeout and cancellation policy;
- detects crashes and orphaned jobs; and
- publishes success only after all evidence is complete.

The worker is serial, receives a sanitized environment, and has all `OPENAI_*`
credentials removed. The LLM process may know how to formulate a problem, but the
native solver process never needs the model key.

Durable idempotency was also necessary. Streamlit reruns its script frequently,
so duplicate prevention cannot live only in a button handler or in process
memory. It belongs below the interface, alongside lifecycle state.

## 5. Results need a trustworthy evidence handoff

A path to an output directory is not a reliable result contract. Files may be
missing, partially written, replaced, or outside the trusted result root.

A successful run therefore produces an immutable manifest containing relative
artifact paths, sizes, SHA-256 hashes, and completion metadata. Analysis verifies:

- the manifest hash;
- containment under the trusted results root;
- symlink and path behavior;
- each file's presence, size, and checksum; and
- the expected content needed for deterministic analysis.

Only then does analysis calculate convergence summaries, final metrics, quality
flags, the final design image, and the objective-history plot.

Checksums prove local evidence consistency, not authorship or protection against
an actor who can rewrite the entire trusted root. Recording that limit is as
important as recording what the check does guarantee.

We also learned to separate original and derived evidence. The solver's success
manifest remains immutable. Later plots are derived, verified outputs and are not
silently inserted into the original manifest.

## 6. Reproducibility is part of correctness

CrewAI and the solver are installed in the project Docker image rather than in the
MacBook's shared Python environment. This gives the numerical and agentic layers
one repeatable runtime and prevents dependency changes for another project from
altering this one.

This was not dependency-free. CrewAI required compatible Pydantic and Rich
versions; Streamlit changed the selected PyArrow version. We pinned the complete
environment and accepted compatibility changes only after rebuilding and running
the full numerical and application test suite.

The model key lives in an untracked `.env` file. Both Git and Docker build context
rules exclude it. Compose injects the key into the application service at
runtime; it is neither baked into the image nor passed to the numerical worker.

The practical lesson is that “install the package” is only the beginning.
Reproducibility includes the lock, image, secret boundary, dependency check,
smoke test, deterministic tests, and numerical regression tests.

## 7. Orchestration learning: use model judgment only where it helps

The original idea of several autonomous agents passing tool outputs to one
another was attractive, but it gave model judgment to steps that did not need it.
Compilation, defaults, validation, lifecycle transitions, and expensive side
effects have exact rules. They are safer and easier to test as deterministic
application code.

The current architecture therefore uses:

- an LLM to understand ordinary language and organize already-approved evidence;
- deterministic code to compile intent and calculate geometry;
- the prepared tools to validate, run, and analyze; and
- an application-owned state machine to decide which transition is legal.

Exact Pydantic objects cross the boundaries. The LLM does not retype a normalized
configuration or copy a run manifest into prose for the next component. This
“judgment at the edges, authority in the middle” pattern kept creativity where it
is useful without letting it control numerical truth or side effects.

CrewAI remains a useful orchestration technology to learn, but it is not a
requirement that overrides functional quality. If another supported API or model
produces materially better conversational formulation, we should measure it
against the same evaluation set and use the better architecture. What we learn
from that comparison is more valuable than preserving a framework for its own
sake.

## 8. Defaults must be deterministic, visible, and editable

The user should not need to specify every numerical preference. The application
can select suitable defaults, but it must say which values were absent and which
values it selected.

For an omitted mesh, the compiler targets nearly square elements at a resolution
similar to `50 × 50` for a square:

```text
h = sqrt(domain area) / 50
nx = round(domain width / h)
ny = round(domain height / h)
```

This produces roughly 2,500 cells for ordinary rectangles while respecting
aspect ratio. The default filter radius is 1.5 times the larger element edge.
These are deterministic, versioned rules rather than values invented afresh by
the model.

Before execution, the workflow must present the interpreted problem and say, in
substance: these values were not given, so the application selected them; ask for
changes or approve the run.

## 9. Human approval is a real state transition

At first, a validated request could proceed directly to execution. That was wrong
for an expensive numerical action and did not match the intended conversation.

A valid proposal now stops in `awaiting_run_approval`. Only an unambiguous
whole-message approval such as `yes` or `start` moves it into a runnable state. A
rejection stays stopped. A requested change returns to formulation, triggers
fresh validation, and requires fresh approval.

Approval is not merely UI text. It is an application-owned transition tested at
the orchestration layer so that another interface cannot bypass it accidentally.

## 10. Fact-preserving explanation requires structural enforcement

Asking a model to “use only these facts” is not enough. A fluent model can still
change a number, omit an important warning, or add an unsupported engineering
claim.

Deterministic analysis now assigns stable identifiers to immutable facts. The
model may organize only those identifiers under allowed headings. Code then:

- rejects unknown or duplicate identifiers;
- requires all mandatory facts;
- renders the original fact text and numbers; and
- fails closed if the plan is invalid.

This preserves the model's useful ability to organize an explanation without
letting it rewrite numerical evidence. The resulting analysis is still not proof
of manufacturability, a load path, or suitability for real engineering use.

## 11. Real failures taught us where the abstractions leaked

The most useful learning came from examples that failed after the first release.

| Observed failure | Root cause | Change made | General lesson |
| --- | --- | --- | --- |
| “Centered 10% of the right edge” matched zero boundary facets. | The model was effectively expected to translate relative language into exact coordinates, and mesh discretization made a narrow absolute selector fragile. | Added typed semantic edge segments with `edge`, `center_fraction`, and `span_fraction`; deterministic compilation performs the geometry arithmetic. | Let the model understand meaning, but let code calculate geometry. Validate against the actual mesh. |
| A whole right-edge traction failed structured interpretation. | Strict output materialized `none` for an unused nullable region, and the contract rejected a span of exactly `1.0`. | Normalize `none` only as the unused sentinel beside a valid edge segment and allow `span_fraction=1.0`. | Strict structured output still needs semantic normalization at union boundaries. |
| The model filled mesh, filter, or iteration preferences the user never stated. | Nullable fields and a strict schema guaranteed shape, not provenance. | Added a deterministic text/provenance guard and reset unrequested preferences so the compiler supplies and discloses them. | Structured shape is not evidence that a fact came from the user. |
| A numerically valid run failed on a tiny negative filtered density. | An exact bound ignored measured floating-point/filter roundoff. | Added a narrow, tested post-filter tolerance and clipping rule. | Numerical validation needs justified tolerances, not blind exactness or broad relaxation. |
| Users could not see the final design or objective history. | The tool contract preserved raw artifacts, but the interface had no constrained presentation path for derived plots. | Added verified design and objective plots with trusted path/role checks and downloads. | A functioning tool includes understandable outputs, not only a successful solver exit. |
| Network/provider problems appeared as “could not interpret.” | Different failure stages were collapsed into one UI message. | Began separating provider, schema, semantic, validation, and execution failure semantics. | Error messages should identify the component that can actually repair the problem. |
| Retrying the same failed interpretation produced the same result. | The retry had no structured feedback or changed state. | The conversational redesign includes typed draft state and is designed for error-specific repair patches. | A retry without new information is repetition, not reasoning. |

Each failure improved the contract. We should continue treating real user examples
as regression cases rather than fixing only the prompt that exposed them.

## 12. Why the first conversational phase felt unintelligent

The API model did not have the same environment or interaction pattern as a
general web assistant or Codex session. The v1 interpreter received:

- a one-shot request rather than a persistent problem-solving conversation;
- an approximately 235,234-character generated schema;
- a small structured answer target;
- low reasoning effort;
- two near-identical attempts; and
- no retained partial draft or structured repair feedback.

The measured trace showed that the model spent almost all of the interaction
processing a very large contract while producing a tiny answer, with no useful
reasoning loop. Calling the result “the API model is less smart” would hide the
architectural cause. We asked the model to fill a huge form in one shot, then
called the same operation again when it failed.

Prompt tuning alone cannot solve that design.

## 13. The conversational redesign

The new provider-independent formulation core is the first step toward an actual
collaborative conversation.

`ProblemDraft` retains partial facts across turns. Every retained fact records:

- its value;
- the source turn;
- a short modeling rationale; and
- whether it is explicit, derived, assumed, or confirmed.

A compact `FormulationTurn` carries the assistant's natural response, changed
fields, up to three focused questions, and a state hint. Deterministic code
validates the patch, records corrections, identifies missing or unconfirmed
facts, and decides whether the draft is ready. A model cannot declare itself
ready and bypass these checks.

Assumptions are allowed because useful problem formulation sometimes requires
creativity. They remain visible and block finalization until the user confirms
them. Once complete, the draft converts into the existing strict
`ProblemIntent`; the compiler, tool validation, and approval gate remain
unchanged.

The strict live transport is 2,729 characters rather than 235,234—about a 99%
reduction. Six multi-turn evaluation seeds cover disordered descriptions,
corrections, load negotiation, unsupported reformulation, conflicts, and casually
stated numerical preferences.

The provider-independent foundation and live Responses adapter are tested and now
drive the released Streamlit entry path.

## 14. What the live formulation implementation taught us

### 14.1 A small Pydantic schema can still be an invalid provider schema

The first live Responses call failed with HTTP 400 before the model ran.
`DraftUpdate.value` used Pydantic's unconstrained `JsonValue`, which generated an
empty `{}` definition. That was convenient for deterministic per-path validation,
but OpenAI strict structured output requires a real type at every schema node.

Expanding every possible support, traction, spring, and recursive region value
back into the transport would have recreated the original huge schema. We instead
added a separate strict transport: `value_json` is a string containing one JSON
value. Application code decodes it immediately and sends the result through the
existing path-specific Pydantic validator. This kept the API schema small without
weakening the domain boundary.

The general lesson is that domain models and provider transport models need not
be identical. Their conversion must be explicit and tested.

### 14.2 Conversation state needs a canonical truth anchor

The live adapter uses the [OpenAI Responses API conversation-state
mechanism](https://developers.openai.com/api/docs/guides/conversation-state) with
`previous_response_id` and all-turn reasoning context. That lets later turns
benefit from earlier model work. It does not make provider memory authoritative.

Every request also contains the application-owned canonical draft. The response
ID is stored as opaque adapter state. If it expires or is unavailable, the adapter
makes exactly one recovery call with full application history. Generic SDK retries
remain disabled.

Responses are stored to support continuation under the provider's documented
retention behavior. This is a conscious data-lifecycle tradeoff for the local
learning tool; the typed local draft remains the durable project state.

### 14.3 A conversational draft must be more granular than final solver input

The first live disordered conversation exposed a design flaw. The user supplied
“ten long and half as tall” on one turn and the lower-left origin on the next.
Because the draft had only `domain.bounds`, the model had to either lose the
dimensions or invent an origin. It invented an origin as an assumption and then
asked the user to reconfirm dimensions and a clamp already supplied.

The fix was to add formulation-only facts:

- `domain.origin`, `domain.width`, and `domain.height`;
- `support_edges` for relative full-vector clamps; and
- `mesh.long_short_divisions` for orientation-independent mesh preferences.

Deterministic finalization resolves these into the unchanged strict
`ProblemIntent` only when enough geometry exists. It also detects conflicts
between complete bounds and component geometry and detects ambiguous long/short
counts for a square.

The lesson is that the conversation's intermediate representation must match the
order in which people naturally provide information. A strict final schema is not
automatically a good partial-draft schema.

### 14.4 Unsupported features are not always a terminal conversation

For a requested point load, the model correctly explained that point and
total-force semantics are unsupported, offered a finite distributed traction
segment, and asked for width and traction magnitude. Our initial turn invariant
rejected that response because it allowed `unsupported_features` only when
`declared_state="unsupported"`.

That confused “this feature is unsupported” with “this conversation must stop.”
The draft now allows capability limits to remain visible while status stays
`gathering`. A truly incompatible request can still use the terminal unsupported
state, and a later user turn can reformulate it.

### 14.5 Repair must receive new information

When deterministic merge rejects a value, provenance, or confirmation, the model
now receives:

- the rejected complete turn;
- stable issue codes and field paths;
- deterministic error messages;
- the same current user message; and
- the unchanged canonical draft.

It gets one bounded attempt to return a complete replacement patch for that user
turn. This differs from the v1 retry, which repeated the identical request and
therefore gave the model no reason to produce a better result.

### 14.6 Evaluations must grade meaning, not internal storage

After semantic `support_edges` were added, both models produced a valid left-edge
clamp and passed real mesh validation. The first evaluator still reported failure
because it looked only for the old absolute `supports` draft path. The grader was
testing an implementation detail rather than the finalized strict intent.

We corrected the grader to inspect semantic outcomes after deterministic
resolution. Evaluation code can become stale just like application code; a
failed check must be diagnosed before it is treated as a model failure.

### 14.7 Model selection needs measured conversations

The versioned v2 gate makes ten turns across six scenarios and never calls the
solver. Ready drafts must compile and pass mesh-backed validation.

Both GPT-5.6 Sol at medium reasoning and GPT-5.6 Terra at medium reasoning passed
all six. The full comparison measured:

| Setting | Input tokens | Output tokens | Reasoning tokens | Total latency |
| --- | ---: | ---: | ---: | ---: |
| Sol / medium | 30,938 | 4,576 | 1,613 | 75.347 s |
| Terra / medium | 30,306 | 2,784 | 236 | 34.688 s |

Terra is the measured lower-latency option on this suite. Sol remains the default
because six scenarios establish a capability floor, not broad equivalence on
unseen engineering conversations. This decision can change when a larger
evaluation provides stronger evidence.

This implementation follows the official guidance to use
[Responses for multi-turn reasoning](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.6#migrate-to-gpt-56),
[structured Pydantic output](https://developers.openai.com/api/docs/guides/structured-outputs),
lean prompts, and explicit approval boundaries.

## 15. What the UI migration taught us

### 15.1 Connecting two correct subsystems still needs a typed boundary

The formulator could already produce a strict `ProblemIntent`, and the
orchestrator could already compile and validate one. Calling compiler functions
directly from Streamlit would nevertheless have made the interface a new source
of workflow authority.

We added `DeterministicOrchestrator.prepare_formulation()`. It accepts a complete
`FormulationStep`, verifies that deterministic status is `ready_for_review` and
that a strict intent exists, then owns compilation and validation through the
existing path. A partial, conflicted, assumed, or repair-needed draft cannot cross
that method.

The lesson is that integration code is a trust boundary, not just glue. Its input
contract should express the state transition it is allowed to perform.

### 15.2 “Ready for review” and “approved to execute” must remain different states

The new UI can reach a ready formulation automatically after any user turn. It
still only produces `AwaitingRunApproval`; it never submits a job. The later user
message must independently match the deterministic whole-message approval
classifier before the orchestrator creates `ValidatedWorkflow`.

A subtler case appears when the user describes a change while an old proposal is
awaiting approval. The UI now clears that approvable outcome before asking the
model to interpret the revision. If the API times out or returns unusable output,
the old parameters do not silently remain eligible for a later “yes.”

The lesson is that invalidating stale authority is as important as granting new
authority. A failed edit must fail closed.

### 15.3 The canonical draft should be visible at the user's level

Streamlit now shows accepted paths, values, and basis directly. Missing fields,
unconfirmed assumptions, capability limits, semantic conflicts, and rejected
patches are also visible without opening a diagnostic panel. Exact source quotes,
concise modeling rationales, and revision records live in an inspectable expander.

This gives the user enough information to correct the engineering formulation
without presenting private chain-of-thought. Public provenance and private model
reasoning are different things.

### 15.4 Model prose should not determine run identity

Responses can phrase the same question differently across otherwise equivalent
conversations. If assistant prose were included in the durable idempotency
material, those wording differences could create different run IDs for the same
user turns and compiled configuration.

`ConversationContext.from_formulation_session()` therefore retains ordered user
turns for workflow identity and excludes assistant text. The normalized compiled
configuration and defaults profile remain part of the idempotency key.

The lesson is to keep nondeterministic presentation out of deterministic identity
unless it carries required semantics.

### 15.5 UI conversation behavior can be tested without API billing

Streamlit application tests inject a canned `ConversationFormulator` while using
the same typed turns, merge rules, readiness checks, and session state as the live
path. They verify multi-turn retention, visible assumptions and capability limits,
field-specific failures, ready-intent handoff, absence of job submission before
approval, and stale-proposal invalidation.

The model itself remains covered by the separate billed formulation evaluation.
This separation keeps ordinary regression tests fast and deterministic while
retaining a real-model quality gate.

## 16. Boundary conditions are three coupled problems

The first extended cantilever conversation showed that “understand the boundary
condition” is not one task. It crosses three different interfaces:

1. **Language semantics** — “tip load,” “middle of the right face,” “10 percent,”
   “downward,” “total force,” “pressure,” “fixed,” “pinned,” and “roller” are not
   interchangeable fields. They describe location, extent, direction, quantity
   type, distribution, and constrained displacement components.
2. **Continuum physics** — a total resultant is not a traction, a mathematical
   point load is not a finite loaded patch, and a roller is not a full clamp. A
   conversion needs the missing physical facts and explicit user agreement.
3. **Mesh realization** — even a valid continuous interval can match zero facets
   or resolve to a different physical length on a discrete mesh. The solver and
   validator must use one selector implementation and show what was actually
   selected.

Our generic region DSL is a good safe execution language, but the monolithic
`supports` and `tractions` formulation fields are a poor conversational memory.
In the observed transcript the system knew the load edge, center, span, direction,
and total magnitude at different times, yet none of those partial facts survived
in the canonical draft because a complete traction object could not be formed.
The assistant then repeated questions and proposed a numerical conversion only in
prose, where it could not be confirmed reliably.

The planned correction is to make each BC a stable, partially populated entity
with field-level provenance. Semantic edge segments will remain continuous until
a shared mesh resolver selects actual facets and reports their measure. Uniform
resultant conversion will use that resolved measure and the declared
unit-thickness model, then verify the integrated force. A deterministic diagram
will show the interpretation before the separate run approval.

This also clarified when a tool change is necessary. Better prompting is enough
to recognize paraphrases, but it cannot create missing solver semantics.
Segmented full clamps and uniform tractions already fit the physics; they need
better selection and evidence. Rollers, symmetry constraints, and true
point-node pins need component- or node-level Dirichlet support in the FEM
contract and corresponding rigid-body/numerical tests. We must recognize those
terms honestly before supporting them, and never approximate them as clamps
without an explicit reformulation.

Building the corpus before the runtime model exposed another useful boundary. A
point described as “the center of the right edge” cannot honestly be stored as an
absolute coordinate while domain bounds are still missing. The evaluation
contract therefore permits a boundary point to retain `edge=right` plus a
relative center until geometry is known. It also grades BC semantic state
separately from whole-problem readiness: a boundary condition can be complete
while material or optimization facts remain missing.

The deterministic grader compares normalized BC entities and stable codes for
clarifications, assumptions, and capability limits. It does not compare assistant
sentences. This lets future prompts and models vary their wording while still
failing on lost partial facts, invented units/directions/extents, silent
point/resultant conversion, roller/pin-to-clamp substitution, unrelated BC
overwrites, or any solver start. The initial version contains 53 cases, including
the supplied six-turn transcript and both earlier complete-prompt failures.

Implementing the partial draft showed why canonical BC IDs cannot come from the
model. A model needs a temporary reference to group fields in one response, but
the application allocates `S1`, `L1`, and later IDs monotonically. Deleting `L1`
does not make that name available again; the next load is `L2`. This prevents a
later correction or audit record from silently referring to a different physical
condition.

Confirmation is also a state operation, not another value update. It targets an
existing assumption and preserves the exact value while changing its basis to
confirmed. A bare “yes” is safe when one BC assumption is pending. When several
are pending, the user must say “confirm all” or identify the intended fact. This
avoids turning a vague acknowledgment into authority for several unrelated
modeling choices.

The field model distinguishes fractional span from physical offset and length.
That distinction emerged while implementing readiness: “the centered 10%” and
“start 1 mm from the corner and continue for 2 mm” cannot share one numeric field
without losing dimensional meaning. Corpus-first development forced this issue
to surface before unit conversion or mesh selection was implemented.

At the Package 1 checkpoint, migration was explicit and non-destructive.
Existing `supports`, `support_edges`, and `tractions` could be copied into
first-class BC state with their provenance, while finalization still read the
legacy facts. Running both representations side by side at that checkpoint gave
us tests for the new state without silently changing live or numerical behavior.

Package 2 made the distinction between unit syntax and physical meaning concrete.
A library can prove that `kN` is a force and `MPa` is a stress, but it cannot
decide whether the number the user supplied was a total force or a distributed
traction. We therefore use Pint only inside deterministic normalization code and
keep `traction` versus `resultant` as an explicit semantic type. JSON-safe models
retain the original value and unit beside the normalized value; Pint objects
never cross the formulation, approval, or solver boundaries.

The unit system is itself conversational state. Length, force, and stress are
three ordinary draft facts with the same explicit/derived/assumption/confirmed
provenance as other parameters. A plausible N-mm-MPa system inferred from context
is still pending until the user confirms it. Once confirmed, an omitted per-load
unit can refer visibly to that mechanical context instead of triggering a silent
conversion.

Direction words also need deterministic semantics after language interpretation.
Global `up`, `down`, `left`, and `right` map directly to vectors. `inward` and
`outward` require a named rectangle edge and use its normal. Bare `normal`, `x`,
`y`, or `tangential` remain ambiguous because they omit a sign or along-edge
sense. Rejecting these states is more reliable than letting model prose conceal a
guessed vector.

Most importantly, normalization stops at the correct knowledge boundary. A
traction has stress dimensions under the current unit-thickness plane-strain
contract and can be normalized immediately. A total resultant has force
dimensions; converting it requires the actual selected boundary measure times
the one-length-unit thickness. Package 2 therefore marks it semantically ready
but execution-deferred. Package 3 must provide the mesh-resolved measure and
verify the integrated force before execution can become ready.

Package 3 showed that “use the same selector” is stronger than “write the same
selection logic twice.” Validation previously compiled a region and located
facets in the tool layer, while FEM independently located them again. Even when
both implementations looked equivalent, there was no structural guarantee that a
future tolerance or discretization fix would reach both. The shared resolver now
returns the exact facet indices and the evidence derived from them; validation
and execution both call that one policy.

Continuous intent and discrete realization must remain separate. A requested
edge interval can be narrower than one facet and contain no facet midpoint. The
old behavior rejected the load as matching zero facets, even though the user had
described a valid positive segment. The new policy chooses the one closest facet
and reports both extents plus the resolution error. This is not pretending the
mesh exactly represents the request: the warning and evidence make the
approximation reviewable before approval.

Versioning also required separating compatibility from canonical meaning.
Existing 1.1 configs contain unlabeled “consistent user units” and expert-region
tractions. Inventing N, mm, and MPa during migration would create false physical
facts. The migration therefore preserves an explicit `legacy_consistent`
sentinel, assigns stable IDs in list order, and wraps old regions without changing
their numerical meaning. Canonical 2.0 can express explicit mechanical units and
total resultants, but a migrated unlabeled config cannot use resultant conversion.

The resultant numerical gate compares two actual solves: one driven by an
effective traction and one by the equivalent total force over the same resolved
segment. Matching objective, compliance, and volume demonstrates more than a
unit-function test—it verifies that schema, migration, resolver, compiler, FEM
measure, and solver assembly agree end to end. The geometry report independently
reintegrates the effective traction to the requested force, giving a second,
cheaper check before execution.

Finally, mathematical validity is not numerical reassurance. A Poisson ratio just
below `0.5` is legal for the constitutive model, but low-order displacement
elements in plane strain may lock as incompressibility is approached. We retained
the strict `<0.5` range and added a visible warning from `0.49`, including the
Lamé ratio, instead of silently rejecting the user's material or silently
presenting a potentially over-stiff result.

Package 4 exposed a less obvious migration trap: the old strict `ProblemIntent`
cannot honestly represent the new finalization surface. A finite support segment
does not fit its whole-region support abstraction, and a total resultant is
explicitly not a traction. We considered reusing the legacy compiler with a
temporary placeholder intent and replacing its BCs afterward. That would have
produced the right final JSON in simple cases, but the intermediate audit object
would state physics the user never requested. Instead, compilation now starts
from a BC-independent typed problem definition and combines it with a separately
finalized first-class BC list.

Compatibility still belongs at a named boundary. Existing live prompts continue
to emit legacy support/traction lists until the prompt package changes. When a
draft has no first-class BC state, finalization migrates those lists once and
then compiles the resulting stable entities. When first-class state exists, it is
authoritative; stale list facts cannot overwrite a correction to `L1` or `S1`.
This lets us ship and test the new deterministic authority before asking a model
to produce it.

Selector arithmetic also belongs in finalization, not in the model response.
Centered fractions stay fractional, while centered physical widths and
corner-offset lengths become coordinate intervals only after domain bounds are
known. Finalization checks that a corner really lies on the named edge and that
the resulting positive interval stays inside it. The mesh resolver then performs
the distinct second mapping from continuous interval to actual facets. Keeping
those two transformations separate preserves both requested and realized
geometry.

Approval became an evidence join keyed by stable BC ID. A config alone can show
what was requested, but it cannot show which facets will carry the load. A
geometry report alone can show selected facets, but without the config it loses
the original selector and quantity semantics. The approval renderer now requires
both successful objects, pairs them by ID, and fails closed if evidence is
missing. For a resultant it shows the original force, effective traction, and
reintegrated force together. This is a stronger review boundary than displaying
a serialized input list, and it leaves the later explicit “yes” transition
unchanged.

## 17. How we will continue learning

We will use these rules for the remaining work:

1. Start from a real user outcome, not from a framework feature.
2. Preserve numerical and side-effect authority in deterministic, tested code.
3. Give the model enough context, state, and reasoning budget for the judgment it
   is expected to perform.
4. Compare adapters and models using the same representative conversations.
5. Turn every real failure into a regression test when possible.
6. Keep assumptions visible and require confirmation before finalization.
7. Require explicit approval before an expensive solve.
8. Record why an architecture changed, not merely which files changed.
9. Keep `README.md` about the current product, `docs/spec.md` about current
   decisions, and this file about accumulated learning.

The next lesson should come from user-led live UI conversations and the failures
they reveal—not from keeping the design artificially simple.

## 18. Milestone trail

This condensed trail connects the lessons above to the implementation order:

- **2026-07-25** — restored and verified the numerical baseline; separated density
  and displacement output behavior.
- **2026-07-25 to 2026-07-26** — built and hardened the three agent-safe tools,
  including contracts, validation, resource admission, numerical checks, process
  containment, lifecycle state, transports, manifests, and analysis.
- **2026-07-26** — pinned CrewAI/OpenAI inside Docker, protected `.env`, and added
  smoke and golden model checks.
- **2026-07-26** — implemented typed interpretation, deterministic compilation,
  visible defaults, validation orchestration, contained execution, analysis, and
  fact-preserving explanation.
- **2026-07-26** — added the Streamlit interface, release documentation, explicit
  run approval, relative edge segments, verified result plots, strict-output
  normalization, provenance checks, and the measured numerical tolerance fix.
- **2026-07-27** — diagnosed the limitations of the one-shot conversational
  phase and added the provider-independent multi-turn draft, provenance, revision,
  readiness, finalization, and evaluation foundation.
- **2026-07-27** — implemented the live Responses adapter, persisted continuation,
  bounded structured repair, strict transport conversion, semantic partial facts,
  v2 live graders, and the measured Sol/Terra comparison.
- **2026-07-27** — migrated Streamlit to the typed multi-turn formulation session,
  added the ready-step orchestrator bridge, visible draft/provenance/error states,
  stable user-turn workflow identity, and stale-approval invalidation.
- **2026-07-27** — used the first extended live conversation to design the
  first-class BC draft, mesh-aware selector evidence, unit/resultant semantics,
  pre-run preview, and a separately gated component-support increment.
- **2026-07-27** — added the provider-independent BC evaluation contract,
  deterministic semantic/safety grader, and 53-case versioned corpus before
  changing runtime behavior.
- **2026-07-27** — implemented stable application-owned BC identities, partial
  field facts, revisions, readiness, pending confirmations, typed patch
  operations, and explicit legacy migration without switching the live path.
- **2026-07-27** — pinned Pint in Docker, added provenance-bearing mechanical
  unit facts, retained display/normalized quantities, deterministic
  direction/pressure resolution, and explicit traction versus deferred-resultant
  state without switching the live finalizer.
- **2026-07-27** — versioned the safe config/tool contracts, added deterministic
  1.1 migration, one validation/FEM facet resolver, enriched boundary evidence,
  verified resultant conversion, near-incompressibility warnings, and a
  traction/resultant numerical-equivalence gate.
- **2026-07-27** — made complete confirmed first-class BCs authoritative at
  finalization, compiled semantic selectors and explicit-unit
  traction/resultants directly into schema 2.0, migrated legacy live facts once,
  and joined requested/resolved boundary evidence in the unchanged approval gate.
- **Next** — teach the live Responses prompt and compact adapter to create,
  refine, and confirm first-class BC patches directly.
