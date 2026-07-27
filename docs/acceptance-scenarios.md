# User Acceptance Scenarios

These five scenarios are realistic, plain-language journeys for learning the
released workflow and testing it beyond the fixed semantic corpus. They are not
golden numerical benchmarks yet: their prompts and expected interpretations must
be reviewed before billed formulation calls or solver runs establish recorded
outcomes.

Run each scenario in a fresh Streamlit conversation. During formulation, stop at
the approval request and compare the visible draft, boundary-condition cards,
defaults, and mesh-resolved preview with the expectations below. Do not reply
`yes` until that review is complete.

## Why these scenarios are structured this way

The first draft already covered compliance and mechanism optimization, clamps,
pins, rollers, traction, pressure, total resultants, clarification, correction,
and explicit approval. Its mesh coverage was narrower than it first appeared:
Scenario 2 did request a triangular `160 × 20` mesh, a `0.075 m` filter radius,
and 250 iterations, but the other four relied on defaults.

This revision strengthens the set without making every prompt look like a config
file:

- Scenario 1 intentionally omits numerical preferences and tests visible
  geometry-derived defaults.
- Scenario 2 uses exact axis-specific mesh divisions and triangular cells.
- Scenario 3 uses casual long-side/short-side mesh language plus an explicit
  filter and iteration limit.
- Scenario 4 adds two separately identified loads, then changes only one load and
  adds numerical preferences after a proposal already exists.
- Scenario 5 leaves the mechanism mesh unspecified so the workflow must disclose
  its defaults and the user can see that spring meaning is mesh dependent.

The five runnable journeys do not try to cover every refusal path. A separate
capability-limit probe section teaches the important unsupported boundaries
without mixing them into problems intended eventually to solve.

## Review protocol

### Phase A — formulation and validation

For each scenario:

1. Start a fresh conversation and submit only the stated turn.
2. When the assistant asks a question, answer only with the next supplied turn or
   with information explicitly called for by the scenario.
3. Inspect the accepted facts, their basis, missing items, assumptions, stable
   `S…`/`L…` labels, and revision history.
4. If the formulation reaches review, inspect the requested and mesh-resolved
   boundary extents, pin snap evidence, load conversions, units, warnings,
   defaults, and estimated cost.
5. Record any semantic difference before approving. Do not repair a wrong
   interpretation by silently adapting the expected result.
6. Confirm that no solver process starts before an unambiguous whole-message
   approval.

### Phase B — execution and evidence

After the prompts and expected interpretations have been finalized, run one
scenario at a time:

1. Approve the exact validated proposal with `yes`.
2. Observe queued, running, and terminal lifecycle states.
3. Verify that a successful run produces a manifest-backed result gallery.
4. For compliance problems, inspect final density, compliance objective, volume,
   and design-change histories.
5. For the mechanism, also inspect the signed output-objective history.
6. Compare the fact-preserving explanation with the deterministic analysis
   evidence rather than judging it only by fluency.

Until Phase B is completed and recorded, these scenarios establish semantic
expectations, not expected objective values or topology images.

## Scenario 1 — Lightweight equipment arm

### User prompt

```text
I need a lightweight mounting arm for a small piece of equipment. The available
design space is a rectangle 600 mm long and 200 mm high, with its lower-left
corner at [0, 0]. The whole left-hand face is built into a rigid wall.

The equipment bears on the right-hand face between heights 80 mm and 120 mm and
produces a uniform downward distributed traction of 1.5 MPa there. Treat the arm
as an aluminium-like material with Young's modulus 69000 MPa and Poisson ratio
0.33. Use mm, N, and MPa as the unit system, and allow 35 percent of the design
region to contain material.

Choose reasonable numerical settings for me and minimize compliance.
```

### What it teaches

- A cantilever can be described using application language rather than solver
  field names.
- “Built into a rigid wall” means a full-vector zero clamp.
- A distributed traction is force per boundary length per unit plane-strain
  thickness, not a total force.
- Coordinate endpoints on a vertical edge can be retained directly.
- Omitted numerical settings are selected deterministically, disclosed, and still
  require approval.

### Expected formulation and review evidence

- `minimize_compliance` on bounds `[[0,0],[600,200]]`.
- One full clamp on the entire left edge.
- One uniform downward traction on the right-edge coordinate interval
  `y = 80…120 mm`; it must not become the entire edge.
- Explicit `mm`, `N`, and `MPa` unit context and volume fraction `0.35`.
- No user-specified cell type, mesh divisions, filter radius, or iteration limit.
- A default near-square mesh around 2,500 cells, with every default and reason
  shown before approval.

### Stress target

This case checks coordinate-interval preservation and whether ordinary engineering
language reaches review without unnecessary clarification.

## Scenario 2 — Simply supported transfer beam

### User prompt

```text
Design a simply supported transfer beam inside a rectangular region 10 m long
and 1 m deep. Put the origin at the middle of its length on the bottom edge, so
the rectangle runs from [-5, 0] to [5, 1].

Use a pin at the lower-left corner. Near the other end, support the last 5 percent
of the bottom edge on a normal roller. The middle 60 percent of the top surface
carries a uniform inward pressure of 0.15 MPa.

Use steel with Young's modulus 200 GPa and Poisson ratio 0.30. My working units
are metres, kilonewtons, and MPa. Keep 45 percent material. Use a triangular mesh
with 160 divisions along x and 20 along y, a filter radius of 0.075 m, and no
more than 250 optimization iterations. Minimize compliance.
```

### What it teaches

- Domain coordinates may be negative and need not start at zero.
- A true pin constrains both displacement components at one boundary mesh node;
  it is not a short clamped edge.
- A bottom-edge normal roller constrains vertical displacement only.
- Inward pressure on the top edge resolves to a downward global traction.
- User mesh, cell, filter, and iteration preferences override defaults only when
  explicitly requested.

### Expected formulation and review evidence

- Bounds `[[-5,0],[5,1]]`, with the pin requested at `[-5,0]`.
- A normal roller on the rightmost 5% of the bottom edge, separate from the pin.
- A uniform inward pressure on the centered 60% of the top edge.
- Triangle cells, divisions `(160,20)`, filter radius `0.075 m`, and maximum 250
  iterations retained with explicit provenance.
- Unit conversion of `200 GPa` into the selected stress context without changing
  the material value physically.
- Component-aware rigid-body rank three, no duplicate constrained degree of
  freedom, and visible pin requested/resolved point evidence.

### Stress target

This case checks the most detailed one-turn customization, component supports,
unit conversion, a nonzero origin, pressure direction, and exact numerical
preference provenance.

## Scenario 3 — Near-incompressible polymer bracket

### User prompt

```text
Optimize a soft polymer mounting bracket that fits in a 400 mm by 250 mm
rectangular design envelope starting at [0, 0]. The complete left edge is bonded
to a rigid frame.

A cable attachment spreads a total downward force of 20 N uniformly over a
40 mm-high patch centered on the right edge. This is the total force carried by
the patch, not a traction value and not a mathematical point load.

Model the material with Young's modulus 10 MPa and Poisson ratio 0.495 under
plane strain. Use mm, N, and MPa. Limit the design to 30 percent material and
minimize compliance.

For the mesh, use roughly 100 cells along the long side and 60 along the short
side, with quadrilateral elements. Use a 7.5 mm filter radius and stop after at
most 300 iterations.
```

### What it teaches

- A total resultant force is different from an effective traction.
- A physical patch width stays semantic until the mesh resolver determines the
  actual loaded boundary measure.
- Under unit thickness, validation computes
  `traction = resultant / (resolved measure × thickness)` and reintegrates it as
  a check.
- Near-incompressible plane strain is mathematically allowed but can produce
  volumetric-locking risk.
- Casual long-side/short-side mesh language is a supported numerical preference.

### Expected formulation and review evidence

- Entire left edge clamped.
- One uniform downward total resultant of `20 N`, centered on a physical
  `40 mm`-high patch on the right edge.
- The load must remain a resultant until mesh resolution; it must never be copied
  directly into the traction field.
- Requested and resolved patch extents, measure, one-unit thickness, effective
  traction, and reconstructed `20 N` resultant shown before approval.
- Long/short mesh request deterministically resolved to `(100,60)` because the
  x direction is the longer side; quadrilateral cells, `7.5 mm` filter, and 300
  iterations retained.
- A visible near-incompressibility/volumetric-locking warning for
  Poisson ratio `0.495`, not an unexplained rejection.

### Stress target

This case checks resultants, physical widths, mesh-dependent conversion, casual
relative mesh language, explicit filter control, and a numerical warning path.

## Scenario 4 — Conversational bracket with two loads and a correction

This scenario deliberately starts incomplete. A useful conversation—not
one-shot success—is the expected behavior.

### Turn 1

```text
I need a lightweight bracket in a rectangular space that is twice as long as it
is high. The left side is fixed to a machine frame. Something pushes downward
near the upper part of the right side, and I would like to use roughly one third
of the available material.
```

The assistant should retain the aspect ratio, support, approximate load location,
and material fraction while asking focused questions about absolute dimensions,
material properties, load quantity/magnitude/units, and exact extent. It must not
invent those facts.

### Turn 2

```text
Make the rectangle 300 mm long and 150 mm high, starting at [0, 0]. Use an
aluminium alloy with Young's modulus 70000 MPa and Poisson ratio 0.33. Use mm,
N, and MPa.

The first load is a uniform downward traction of 5 MPa over the upper quarter of
the right edge. Add a second, uniform rightward tangential traction of 1 MPa over
the middle third of the top edge. Keep the material fraction at one third and
minimize compliance. Let the program choose the numerical settings for now.
```

The workflow should retain one support and two distinct loads with stable labels,
normally `S1`, `L1`, and `L2`, then reach a validated proposal.

### Turn 3 — correction before approval

Use the actual stable label displayed for the first load; if it is `L1`, submit:

```text
Before running, change L1. Move it to the centered 20 percent of the right edge
and reduce its traction to 3 MPa. Keep the other top-edge load unchanged. Change
the material allowance to 40 percent.

Also replace the proposed numerical defaults with a 90 by 45 quadrilateral mesh,
a 4 mm filter radius, and a maximum of 180 iterations.
```

### What it teaches

- Partial structural facts can accumulate over several turns.
- Stable IDs permit one load to change without replacing another load or the
  support.
- A requested change revokes the older approvable proposal immediately.
- Numerical defaults are proposals, not hidden constants; the user can replace
  them before execution.
- A tangential top-edge traction remains horizontal rather than becoming normal
  pressure.

### Expected formulation and review evidence

- After Turn 1, status remains gathering and no config or solver run exists.
- After Turn 2, the right-edge downward load and top-edge rightward load have
  separate stable IDs and nonoverlapping resolved regions.
- After Turn 3, only the first load changes to a centered 20% span and `3 MPa`;
  the second load remains `1 MPa` rightward over the middle third of the top.
- Volume fraction becomes `0.40`.
- Mesh `(90,45)`, quadrilateral cells, filter `4 mm`, and 180 iterations replace
  the earlier defaults with user-explicit provenance.
- The superseded proposal cannot be approved; fresh validation and fresh run
  approval are required.

### Stress target

This is the strongest state-management test: incomplete facts, two loads,
targeted correction, fact retention, defaults replacement, and stale-approval
invalidation.

## Scenario 5 — Compliant motion inverter

### User prompt

```text
Create a compliant mechanism inside a 100 mm wide by 60 mm high rectangle running
from [0, -30] to [100, 30]. I want a push to the right at the middle of the left
side to make the middle of the right side move to the left.

Clamp the upper 20 percent and lower 20 percent of both vertical sides. Leave the
middle portions free for the input and output. Apply a uniform rightward traction
of 1 MPa over the central 20 percent of the left edge.

Place the input spring over that same central 20 percent of the left edge, acting
in the x direction with stiffness 0.2 N/mm. Put the output spring over the central
20 percent of the right edge, also in the x direction with stiffness 0.2 N/mm.
Keep compliance below 0.5 and use 25 percent material.

Use Young's modulus 100 MPa, Poisson ratio 0.25, and mm, N, and MPa as the unit
system. Choose the mesh and optimization settings.
```

### What it teaches

- Compliant-mechanism optimization is a distinct supported problem type.
- Input loading, input spring, output spring, compliance bound, and desired output
  direction all need to be understood together.
- Finite support and spring regions can be described as fractions of named edges.
- A configured spring stiffness is applied per matched directional nodal degree
  of freedom; it is not automatically divided across the spring region.
- Mechanism results include a signed output-objective history in addition to the
  common density, volume, and design-change plots.

### Expected formulation and review evidence

- Problem type `compliant_mechanism`, volume fraction `0.25`, and compliance
  bound `0.5`.
- Four distinct full-vector clamp segments: upper and lower 20% of both vertical
  edges.
- One rightward uniform traction on the central 20% of the left edge.
- Distinct input and output springs on the central 20% of the left and right
  edges, both acting in `x`, each with stiffness `0.2`.
- The intended signed output is leftward motion on the right when the input is
  pushed right. If the sign convention cannot be established from the supported
  contract, the assistant should clarify rather than silently promise inversion.
- Mechanism/MMA numerical defaults and their reasons shown before approval.

### Stress target

This is the highest-risk formulation and numerical case. It tests four support
segments, coincident input load/spring regions, a distinct output region, signed
motion reasoning, mechanism-only fields, and mesh-dependent spring semantics.

## Coverage matrix

| Capability | S1 | S2 | S3 | S4 | S5 |
| --- | --- | --- | --- | --- | --- |
| Compliance minimization | ✓ | ✓ | ✓ | ✓ | — |
| Compliant mechanism | — | — | — | — | ✓ |
| Full clamp | ✓ | — | ✓ | ✓ | ✓ |
| True boundary-node pin | — | ✓ | — | — | — |
| Component roller | — | ✓ | — | — | — |
| Multiple supports | — | ✓ | — | — | ✓ |
| Multiple loads with stable IDs | — | — | — | ✓ | — |
| Distributed traction | ✓ | — | — | ✓ | ✓ |
| Normal pressure | — | ✓ | — | — | — |
| Uniform total resultant | — | — | ✓ | — | — |
| Coordinate interval | ✓ | — | — | — | — |
| Fractional interval | — | ✓ | — | ✓ | ✓ |
| Physical-width selector | — | — | ✓ | — | — |
| Automatic mesh/default disclosure | ✓ | — | — | initially | ✓ |
| Exact axis-specific mesh | — | ✓ | — | after correction | — |
| Casual long/short mesh | — | — | ✓ | — | — |
| Triangle cells | — | ✓ | — | — | — |
| Explicit filter/iteration controls | — | ✓ | ✓ | after correction | — |
| Unit conversion | — | ✓ | ✓ | — | — |
| Near-incompressibility warning | — | — | ✓ | — | — |
| Multi-turn clarification | — | — | — | ✓ | — |
| Targeted correction/stale approval | — | — | — | ✓ | — |
| Mechanism springs and signed output | — | — | — | — | ✓ |

## Honest gaps in the five runnable journeys

The set is broad, but it is not complete coverage:

- It does not exercise constant body force.
- It uses a normal roller but not the word “symmetry,” even though both compile
  to explicit normal-component constraints.
- It does not exercise expert region expressions, circular regions, passive solid
  or void zones, or mechanism spring variants beyond edge intervals.
- It does not test cancellation, timeout, crash recovery, idempotent replay,
  manifest tampering, or concurrent-solve rejection; those are lifecycle/tool
  tests rather than prompt-only scenarios.
- It does not establish golden numerical objective values or topology images.
  Those can be recorded only after the prompts are approved and real runs are
  reviewed for physical plausibility.

The existing automated suite remains authoritative for lower-level safety,
numerical baselines, lifecycle behavior, transport parity, and artifact
integrity. These five scenarios add user-level acceptance evidence; they do not
replace that suite.

## Supplemental capability-limit probes

Run these only as formulation checks. They should never reach solver approval:

```text
Optimize a three-dimensional mounting bracket with a non-rectangular bolt hole.
```

Expected: identify unsupported agent-safe 3D/non-rectangular geometry and offer a
supported 2D rectangular reformulation only if the user wants one.

```text
Apply a 100 N downward mathematical point load exactly at [10, 2].
```

Expected: retain the point-force meaning, explain that exact point loads are
unsupported, and offer a finite patch carrying a uniform resultant for explicit
confirmation. Never silently convert it.

```text
Move the right edge upward by a prescribed displacement of 2 mm.
```

Expected: report nonzero prescribed displacement as unsupported. Do not alias it
to a zero support or a force.

```text
Apply a linearly varying traction and a 50 N mm boundary moment on the top edge.
```

Expected: identify both unsupported load contracts. Do not replace either with a
constant traction.

These probes teach the capability boundary and verify honest negotiation, while
the five main scenarios remain candidates for full end-to-end runs.
