"""Deterministically compile semantic intent into the agent-safe tool contract."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic.boundary_draft import (
    BoundaryConditionDraft,
    BoundaryDraftState,
    assess_boundary_state,
)
from agentic.formulation import (
    ProblemDraft,
    _resolved_bounds,
    _resolved_mesh_divisions,
    assess_mechanical_units,
    migrate_legacy_boundary_facts,
)
from agentic.intent import (
    MaterialIntent,
    MechanismSpringIntent,
    MechanismProblemIntent,
    MeshPreferences,
    OptimizationPreferences,
    ProblemIntent,
    RectangularDomainIntent,
)
from agentic.load_semantics import resolve_boundary_load_state
from fenitop.tools.config_models import AgentSafeConfig
from fenitop.tools.mechanical_units import (
    MechanicalUnitContext,
    normalize_scalar,
)

DEFAULT_PROFILE_VERSION = "agentic-defaults-v1"
TARGET_SQUARE_DIVISIONS = 50
FILTER_CELL_MULTIPLIER = 1.5


class AppliedDefault(BaseModel):
    """One application-owned choice that was absent from user intent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    value: Any
    reason: str


class CompilationResult(BaseModel):
    """Exact tool config plus an inspectable account of compiler choices."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config: AgentSafeConfig
    defaults_profile: Literal["agentic-defaults-v1"] = DEFAULT_PROFILE_VERSION
    applied_defaults: tuple[AppliedDefault, ...]
    defaults_notice: str
    spring_evidence: tuple["SpringCompilationEvidence", ...] = ()


class SpringCompilationEvidence(BaseModel):
    """Review evidence retained while semantic springs compile to solver regions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spring_id: str
    role: Literal["input", "output"]
    location: str
    direction: Literal["x", "y"]
    original_stiffness: float
    original_unit: str
    normalized_stiffness: float
    normalized_unit: str


class CompilationProblem(BaseModel):
    """BC-independent, strictly validated problem facts used by both front doors."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    problem_type: Literal["minimize_compliance", "compliant_mechanism"]
    domain: RectangularDomainIntent
    material: MaterialIntent
    body_force: tuple[float, float] = (0.0, 0.0)
    volume_fraction: float = Field(gt=0, lt=1)
    mesh: MeshPreferences = Field(default_factory=MeshPreferences)
    optimization: OptimizationPreferences = Field(
        default_factory=OptimizationPreferences
    )
    compliance_bound: float | None = Field(default=None, gt=0)
    input_spring: MechanismSpringIntent | None = None
    output_spring: MechanismSpringIntent | None = None

    @model_validator(mode="after")
    def _problem_specific_fields(self):
        mechanism_values = (
            self.compliance_bound,
            self.input_spring,
            self.output_spring,
        )
        if self.problem_type == "compliant_mechanism":
            if self.compliance_bound is None:
                raise ValueError(
                    "compliant mechanisms require compliance_bound."
                )
        elif any(item is not None for item in mechanism_values):
            raise ValueError(
                "compliance problems cannot contain mechanism-only fields."
            )
        return self


class FormulationFinalizationError(ValueError):
    """A conversational draft cannot safely cross the compiler boundary."""


def _mesh_divisions(bounds) -> tuple[int, int]:
    """Target ~2,500 near-square cells for any rectangular aspect ratio."""
    (x0, y0), (x1, y1) = bounds
    width = float(x1 - x0)
    height = float(y1 - y0)
    target_h = math.sqrt(width * height) / TARGET_SQUARE_DIVISIONS
    nx, ny = (
        max(1, math.floor(width / target_h + 0.5)),
        max(1, math.floor(height / target_h + 0.5)),
    )
    # A filter radius of 1.5 cells must remain smaller than either domain axis.
    # For extreme aspect ratios, preserve square cells by refining the long axis
    # after enforcing at least two cells on the short axis. Resource validation
    # remains responsible for rejecting an excessively large resulting mesh.
    if nx < 2:
        nx = 2
        ny = max(2, math.floor(height / (width / nx) + 0.5))
    if ny < 2:
        ny = 2
        nx = max(2, math.floor(width / (height / ny) + 0.5))
    return nx, ny


def _default_filter_radius(bounds, divisions: tuple[int, int]) -> float:
    (x0, y0), (x1, y1) = bounds
    dx = float(x1 - x0) / divisions[0]
    dy = float(y1 - y0) / divisions[1]
    return float(f"{FILTER_CELL_MULTIPLIER * max(dx, dy):.12g}")


def _record(
    defaults: list[AppliedDefault],
    path: str,
    value: Any,
    reason: str,
) -> Any:
    defaults.append(AppliedDefault(path=path, value=value, reason=reason))
    return value


def _source_from_draft(draft: ProblemDraft) -> CompilationProblem:
    values = draft.values()
    unconfirmed = [
        fact.path for fact in draft.facts if fact.basis == "assumption"
    ]
    if unconfirmed:
        raise FormulationFinalizationError(
            "Unconfirmed ordinary facts: " + ", ".join(unconfirmed) + "."
        )
    bounds, bounds_errors = _resolved_bounds(values)
    required = (
        "problem_type",
        "material.young_modulus",
        "material.poisson_ratio",
        "volume_fraction",
    )
    missing = [path for path in required if path not in values]
    if bounds is None:
        missing.append("domain.bounds")
    if missing:
        raise FormulationFinalizationError(
            "Missing ordinary facts: " + ", ".join(missing) + "."
        )
    divisions, mesh_errors = _resolved_mesh_divisions(values, bounds)
    errors = [*bounds_errors, *mesh_errors]
    if errors:
        raise FormulationFinalizationError("; ".join(errors))

    payload: dict[str, Any] = {
        "problem_type": values["problem_type"],
        "domain": {"bounds": bounds},
        "material": {
            "young_modulus": values["material.young_modulus"],
            "poisson_ratio": values["material.poisson_ratio"],
        },
        "body_force": values.get("body_force", (0.0, 0.0)),
        "volume_fraction": values["volume_fraction"],
        "mesh": {
            "divisions": divisions,
            "cell_type": values.get("mesh.cell_type"),
        },
        "optimization": {
            "filter_radius": values.get("optimization.filter_radius"),
            "max_iter": values.get("optimization.max_iter"),
        },
    }
    if values["problem_type"] == "compliant_mechanism":
        if "compliance_bound" not in values:
            raise FormulationFinalizationError(
                "Missing mechanism fact: compliance_bound."
            )
        payload["compliance_bound"] = values["compliance_bound"]
        for path in ("input_spring", "output_spring"):
            if path in values:
                payload[path] = values[path]
    return CompilationProblem.model_validate(payload)


def _edge_limits(
    edge: str,
    bounds: tuple[tuple[int | float, int | float], tuple[int | float, int | float]],
) -> tuple[float, float]:
    (x0, y0), (x1, y1) = bounds
    return (
        (float(y0), float(y1))
        if edge in {"left", "right"}
        else (float(x0), float(x1))
    )


def _rectangle_selector(
    condition: BoundaryConditionDraft,
    bounds,
) -> dict:
    values = condition.values()
    kind = values["selector.kind"]
    if kind == "expert_region":
        return {
            "kind": "expert_region",
            "region": values["selector.region"],
        }
    if kind in {"boundary_point", "unspecified", "unspecified_extent"}:
        raise FormulationFinalizationError(
            f"{condition.bc_id} selector {kind!r} cannot select finite facets."
        )

    edge = str(values["selector.edge"])
    low, high = _edge_limits(edge, bounds)
    if kind == "whole_edge":
        interval = {"kind": "fraction", "start": 0.0, "end": 1.0}
    elif kind == "centered_fraction":
        center = float(values["selector.center"])
        half_span = float(values["selector.span"]) / 2.0
        interval = {
            "kind": "fraction",
            "start": center - half_span,
            "end": center + half_span,
        }
    elif kind == "fraction_interval":
        interval = {
            "kind": "fraction",
            "start": float(values["selector.start"]),
            "end": float(values["selector.end"]),
        }
    elif kind == "coordinate_interval":
        interval = {
            "kind": "coordinate",
            "start": float(values["selector.start"]),
            "end": float(values["selector.end"]),
        }
    elif kind == "centered_width":
        center = low + float(values["selector.center"]) * (high - low)
        half_width = float(values["selector.width"]) / 2.0
        interval = {
            "kind": "coordinate",
            "start": center - half_width,
            "end": center + half_width,
        }
    elif kind == "distance_from_corner":
        corner = str(values["selector.from_corner"])
        valid_corners = {
            "left": {"lower_left": "low", "upper_left": "high"},
            "right": {"lower_right": "low", "upper_right": "high"},
            "bottom": {"lower_left": "low", "lower_right": "high"},
            "top": {"upper_left": "low", "upper_right": "high"},
        }[edge]
        if corner not in valid_corners:
            raise FormulationFinalizationError(
                f"{condition.bc_id} corner {corner!r} is not on the {edge} edge."
            )
        offset = float(values["selector.offset"])
        length = float(values["selector.length"])
        if valid_corners[corner] == "low":
            start, end = low + offset, low + offset + length
        else:
            start, end = high - offset - length, high - offset
        interval = {"kind": "coordinate", "start": start, "end": end}
    else:
        raise FormulationFinalizationError(
            f"{condition.bc_id} has unsupported selector kind {kind!r}."
        )

    start, end = float(interval["start"]), float(interval["end"])
    allowed_low, allowed_high = (
        (0.0, 1.0) if interval["kind"] == "fraction" else (low, high)
    )
    if start < allowed_low or end > allowed_high or end <= start:
        raise FormulationFinalizationError(
            f"{condition.bc_id} selector interval [{start:g}, {end:g}] "
            f"must be a positive interval inside [{allowed_low:g}, "
            f"{allowed_high:g}]."
        )
    return {"kind": "rectangle_edge", "edge": edge, "interval": interval}


def _boundary_node_selector(
    condition: BoundaryConditionDraft,
    bounds,
) -> dict:
    values = condition.values()
    if values.get("selector.kind") != "boundary_point":
        raise FormulationFinalizationError(
            f"{condition.bc_id} point pin requires one boundary-point selector."
        )
    point = values.get("selector.point")
    if point is None:
        corner = values.get("selector.from_corner")
        if corner is not None:
            (x0, y0), (x1, y1) = bounds
            point = {
                "lower_left": (x0, y0),
                "upper_left": (x0, y1),
                "lower_right": (x1, y0),
                "upper_right": (x1, y1),
            }[str(corner)]
        else:
            edge = values.get("selector.edge")
            center = values.get("selector.center")
            if edge is None or center is None:
                raise FormulationFinalizationError(
                    f"{condition.bc_id} boundary point is incomplete."
                )
            (x0, y0), (x1, y1) = bounds
            fraction = float(center)
            point = {
                "left": (x0, y0 + fraction * (y1 - y0)),
                "right": (x1, y0 + fraction * (y1 - y0)),
                "bottom": (x0 + fraction * (x1 - x0), y0),
                "top": (x0 + fraction * (x1 - x0), y1),
            }[str(edge)]
    return {
        "kind": "boundary_node",
        "point": tuple(float(value) for value in point),
    }


def _first_class_boundary_conditions(
    state: BoundaryDraftState,
    *,
    bounds,
    context: MechanicalUnitContext | None,
    legacy_migration: bool,
) -> list[dict]:
    readiness = assess_boundary_state(state)
    failures = [item for item in readiness.conditions if not item.ready]
    if failures:
        details = []
        for item in failures:
            reasons = (
                *item.missing_fields,
                *item.unconfirmed_fields,
                *item.semantic_errors,
                *item.capability_limits,
            )
            details.append(f"{item.bc_id}: {', '.join(reasons)}")
        raise FormulationFinalizationError(
            "Boundary conditions are not complete and confirmed ("
            + "; ".join(details)
            + ")."
        )
    if not any(item.kind == "support" for item in state.conditions):
        raise FormulationFinalizationError(
            "At least one complete fixed support is required."
        )

    resolved_loads = {}
    if not legacy_migration:
        if context is None and any(
            item.kind == "load" for item in state.conditions
        ):
            raise FormulationFinalizationError(
                "First-class boundary loads require complete, confirmed "
                "length, force, and stress units."
            )
        if context is not None:
            load_state = resolve_boundary_load_state(state, context)
            if not load_state.semantic_ready:
                issues = [
                    f"{resolution.load.bc_id if resolution.load else 'load'}: "
                    + ", ".join(issue.message for issue in resolution.issues)
                    for resolution in load_state.loads
                    if resolution.status not in {"resolved", "deferred"}
                ]
                raise FormulationFinalizationError(
                    "Boundary load semantics are unresolved ("
                    + "; ".join(issues)
                    + ")."
                )
            resolved_loads = {
                resolution.load.bc_id: resolution.load
                for resolution in load_state.loads
                if resolution.load is not None
            }

    compiled = []
    for condition in state.conditions:
        if condition.kind in {"input_spring", "output_spring"}:
            continue
        values = condition.values()
        if condition.kind == "support":
            support_kind = values["support.kind"]
            if support_kind == "pin":
                compiled.append({
                    "bc_id": condition.bc_id,
                    "kind": "zero_displacement",
                    "selector": _boundary_node_selector(condition, bounds),
                    "components": ("x", "y"),
                })
                continue
            selector = _rectangle_selector(condition, bounds)
            if support_kind == "fixed_all":
                compiled.append({
                    "bc_id": condition.bc_id,
                    "kind": "fixed",
                    "selector": selector,
                    "value": (0.0, 0.0),
                })
                continue
            if support_kind in {"roller_normal", "symmetry"}:
                edge = values.get("selector.edge")
                if edge is None:
                    raise FormulationFinalizationError(
                        f"{condition.bc_id} normal component requires a named "
                        "rectangle edge."
                    )
                component = "x" if edge in {"left", "right"} else "y"
            elif support_kind == "roller_x":
                component = "x"
            elif support_kind == "roller_y":
                component = "y"
            else:
                raise FormulationFinalizationError(
                    f"{condition.bc_id} support kind is outside the current "
                    "zero-displacement contract."
                )
            compiled.append({
                "bc_id": condition.bc_id,
                "kind": "zero_displacement",
                "selector": selector,
                "components": (component,),
            })
            continue

        selector = _rectangle_selector(condition, bounds)
        if legacy_migration:
            compiled.append({
                "bc_id": condition.bc_id,
                "kind": "uniform_traction",
                "selector": selector,
                "traction": values["load.vector"],
            })
            continue
        load = resolved_loads[condition.bc_id]
        vector = load.vector.normalized_value
        compiled.append({
            "bc_id": condition.bc_id,
            "kind": (
                "uniform_traction"
                if load.quantity_kind == "traction"
                else "uniform_resultant"
            ),
            "selector": selector,
            (
                "traction"
                if load.quantity_kind == "traction"
                else "resultant"
            ): vector,
        })
    return compiled


def _selector_region(condition: BoundaryConditionDraft, bounds) -> tuple[dict, str]:
    """Compile one semantic spring selector to the existing safe region DSL."""
    selector = _rectangle_selector(condition, bounds)
    if selector["kind"] == "expert_region":
        return selector["region"], "expert region"
    edge = selector["edge"]
    interval = selector["interval"]
    low, high = _edge_limits(edge, bounds)
    if interval["kind"] == "fraction":
        start = low + float(interval["start"]) * (high - low)
        end = low + float(interval["end"]) * (high - low)
    else:
        start, end = float(interval["start"]), float(interval["end"])
    (x0, y0), (x1, y1) = bounds
    fixed_axis, fixed_value, span_axis = {
        "left": ("x", float(x0), "y"),
        "right": ("x", float(x1), "y"),
        "bottom": ("y", float(y0), "x"),
        "top": ("y", float(y1), "x"),
    }[edge]
    return (
        {
            "op": "and",
            "regions": [
                {"op": "plane", "axis": fixed_axis, "value": fixed_value},
                {
                    "op": "range",
                    "axis": span_axis,
                    "min": start,
                    "max": end,
                },
            ],
        },
        f"{edge} edge from {start:g} to {end:g}",
    )


def _first_class_mechanism_springs(
    state: BoundaryDraftState,
    *,
    bounds,
    context: MechanicalUnitContext,
) -> tuple[dict[str, MechanismSpringIntent], tuple[SpringCompilationEvidence, ...]]:
    compiled: dict[str, MechanismSpringIntent] = {}
    evidence: list[SpringCompilationEvidence] = []
    for kind, config_name, role in (
        ("input_spring", "input_spring", "input"),
        ("output_spring", "output_spring", "output"),
    ):
        matches = [item for item in state.conditions if item.kind == kind]
        if len(matches) != 1:
            raise FormulationFinalizationError(
                f"Exactly one {kind.replace('_', ' ')} is required."
            )
        condition = matches[0]
        values = condition.values()
        region, location = _selector_region(condition, bounds)
        normalized = normalize_scalar(
            values["spring.stiffness"],
            values["spring.unit"],
            "spring_stiffness",
            context,
        )
        compiled[config_name] = MechanismSpringIntent(
            region=region,
            direction=values["spring.direction"],
            stiffness=normalized.normalized_value,
        )
        evidence.append(SpringCompilationEvidence(
            spring_id=condition.bc_id,
            role=role,
            location=location,
            direction=values["spring.direction"],
            original_stiffness=float(values["spring.stiffness"]),
            original_unit=str(values["spring.unit"]),
            normalized_stiffness=normalized.normalized_value,
            normalized_unit=normalized.normalized_unit,
        ))
    return compiled, tuple(evidence)


def format_defaults_notice(
    profile: str,
    defaults: list[AppliedDefault] | tuple[AppliedDefault, ...],
) -> str:
    """Create the required editable transparency notice."""
    lines = [
        "The following values were not provided, so the deterministic compiler "
        f"selected them for you using {profile}:"
    ]
    lines.extend(
        f"- {item.path} = {item.value!r} — {item.reason}" for item in defaults
    )
    lines.append(
        "Review these choices before approving the run. You can request changes "
        "instead of approving."
    )
    return "\n".join(lines)


def _source_from_intent(intent: ProblemIntent) -> CompilationProblem:
    payload = {
        "problem_type": intent.problem_type,
        "domain": intent.domain,
        "material": intent.material,
        "body_force": intent.body_force,
        "volume_fraction": intent.volume_fraction,
        "mesh": intent.mesh,
        "optimization": intent.optimization,
    }
    if isinstance(intent, MechanismProblemIntent):
        payload.update({
            "compliance_bound": intent.compliance_bound,
            "input_spring": intent.input_spring,
            "output_spring": intent.output_spring,
        })
    return CompilationProblem.model_validate(payload)


def _legacy_boundary_conditions(intent: ProblemIntent) -> list[dict]:
    conditions = [
        {
            "bc_id": f"S{index}",
            "kind": "fixed",
            "selector": {
                "kind": "expert_region",
                "region": support.region,
            },
            "value": (0.0, 0.0),
        }
        for index, support in enumerate(intent.supports, start=1)
    ]
    for index, traction in enumerate(intent.tractions, start=1):
        if traction.edge_segment is not None:
            segment = traction.edge_segment
            start = segment.center_fraction - segment.span_fraction / 2
            end = segment.center_fraction + segment.span_fraction / 2
            selector = {
                "kind": "rectangle_edge",
                "edge": segment.edge,
                "interval": {
                    "kind": "fraction",
                    "start": start,
                    "end": end,
                },
            }
        else:
            selector = {
                "kind": "expert_region",
                "region": traction.region,
            }
        conditions.append({
            "bc_id": f"L{index}",
            "kind": "uniform_traction",
            "selector": selector,
            "traction": traction.vector,
        })
    return conditions


def _compile_problem(
    source: CompilationProblem,
    *,
    boundary_conditions: list[dict],
    units: dict,
    mechanism_springs: dict[str, MechanismSpringIntent] | None = None,
    spring_evidence: tuple[SpringCompilationEvidence, ...] = (),
) -> CompilationResult:
    """Compile already-finalized facts without consulting an LLM."""
    defaults: list[AppliedDefault] = []
    bounds = source.domain.bounds

    divisions = source.mesh.divisions
    if divisions is None:
        divisions = _record(
            defaults,
            "mesh.divisions",
            _mesh_divisions(bounds),
            "targets about 2,500 near-square elements (50×50 for a square domain)",
        )

    cell_type = source.mesh.cell_type
    if cell_type is None:
        cell_type = _record(
            defaults,
            "mesh.cell_type",
            "quadrilateral",
            "the pinned v1 structured-mesh profile",
        )

    filter_radius = source.optimization.filter_radius
    if filter_radius is None:
        filter_radius = _record(
            defaults,
            "opt.filter_radius",
            _default_filter_radius(bounds, divisions),
            "1.5 times the larger element edge length",
        )

    is_compliance = source.problem_type == "minimize_compliance"
    max_iter = source.optimization.max_iter
    if max_iter is None:
        max_iter = _record(
            defaults,
            "opt.max_iter",
            400 if is_compliance else 500,
            "the pinned convergence budget for this problem type",
        )

    optimizer = _record(
        defaults,
        "opt.optimizer",
        "oc" if is_compliance else "mma",
        "the solver's trusted optimizer for this problem type",
    )
    initial_density = _record(
        defaults,
        "opt.initial_density",
        source.volume_fraction,
        "starts from a uniform design satisfying the requested volume fraction",
    )
    opt_tol = _record(
        defaults, "opt.opt_tol", 1e-5, "the pinned v1 convergence tolerance"
    )
    penalty = _record(
        defaults, "opt.penalty", 3.0, "the pinned v1 SIMP penalization"
    )
    epsilon = _record(
        defaults, "opt.epsilon", 1e-6, "the pinned numerical regularization"
    )
    beta_interval = _record(
        defaults,
        "opt.beta_interval",
        50,
        "the pinned Heaviside continuation interval",
    )
    beta_max = _record(
        defaults,
        "opt.beta_max",
        128.0,
        "the pinned Heaviside continuation cap",
    )
    move = _record(
        defaults,
        "opt.move",
        0.02 if is_compliance else 0.05,
        "the pinned update limit for this problem type",
    )
    analysis_type = _record(
        defaults,
        "fem.analysis_type",
        "plane_strain",
        "the supported v1 constitutive assumption",
    )
    thickness = _record(
        defaults,
        "units.thickness_value",
        1.0,
        "the supported implicit out-of-plane thickness",
    )
    if units["kind"] == "legacy_consistent":
        _record(
            defaults,
            "units.kind",
            "legacy_consistent",
            "preserves legacy values without inventing physical unit labels",
        )
    solid_zone = _record(
        defaults,
        "opt.solid_zone",
        {"op": "none"},
        "no mandatory solid region was specified by the intent contract",
    )
    void_zone = _record(
        defaults,
        "opt.void_zone",
        {"op": "none"},
        "no mandatory void region was specified by the intent contract",
    )

    mesh = {
        "bounds": bounds,
        "divisions": divisions,
        "cell_type": cell_type,
    }
    fem = {
        "analysis_type": analysis_type,
        "young_modulus": source.material.young_modulus,
        "poisson_ratio": source.material.poisson_ratio,
        "boundary_conditions": boundary_conditions,
        "body_force": source.body_force,
    }
    opt: dict[str, Any] = {
        "problem_type": source.problem_type,
        "optimizer": optimizer,
        "max_iter": max_iter,
        "opt_tol": opt_tol,
        "vol_frac": source.volume_fraction,
        "initial_density": initial_density,
        "penalty": penalty,
        "epsilon": epsilon,
        "filter_radius": filter_radius,
        "beta_interval": beta_interval,
        "beta_max": beta_max,
        "move": move,
        "solid_zone": solid_zone,
        "void_zone": void_zone,
    }
    if source.problem_type == "compliant_mechanism":
        springs = mechanism_springs or {
            "input_spring": source.input_spring,
            "output_spring": source.output_spring,
        }
        if any(value is None for value in springs.values()):
            raise FormulationFinalizationError(
                "A compliant mechanism requires complete input and output springs."
            )
        opt.update(
            {
                "compliance_bound": source.compliance_bound,
                "in_spring": {
                    "region": springs["input_spring"].region,
                    "direction": springs["input_spring"].direction,
                    "stiffness": springs["input_spring"].stiffness,
                },
                "out_spring": {
                    "region": springs["output_spring"].region,
                    "direction": springs["output_spring"].direction,
                    "stiffness": springs["output_spring"].stiffness,
                },
            }
        )

    config = AgentSafeConfig.model_validate({
        "units": {**units, "thickness_value": thickness},
        "mesh": mesh,
        "fem": fem,
        "opt": opt,
    })
    applied = tuple(defaults)
    return CompilationResult(
        config=config,
        applied_defaults=applied,
        defaults_notice=format_defaults_notice(DEFAULT_PROFILE_VERSION, applied),
        spring_evidence=spring_evidence,
    )


def compile_intent(intent: ProblemIntent) -> CompilationResult:
    """Compile the retained legacy intent through the canonical 2.1 contract."""
    return _compile_problem(
        _source_from_intent(intent),
        boundary_conditions=_legacy_boundary_conditions(intent),
        units={"kind": "legacy_consistent"},
    )


def compile_formulation_draft(draft: ProblemDraft) -> CompilationResult:
    """Finalize a conversational draft through first-class BC entities.

    Legacy list facts are migrated once at this boundary. If first-class state is
    already present, it is authoritative and legacy BC lists are ignored.
    """
    source = _source_from_draft(draft)
    legacy_migration = not draft.boundary_state.conditions
    finalized = (
        migrate_legacy_boundary_facts(draft)
        if legacy_migration
        else draft
    )
    state = finalized.boundary_state
    has_boundary_load = any(item.kind == "load" for item in state.conditions)
    has_semantic_spring = any(
        item.kind in {"input_spring", "output_spring"}
        for item in state.conditions
    )
    body_force_nonzero = any(float(item) != 0.0 for item in source.body_force)
    if not has_boundary_load and not body_force_nonzero:
        raise FormulationFinalizationError(
            "At least one boundary load or nonzero body force is required."
        )

    context = None
    if legacy_migration:
        units = {"kind": "legacy_consistent"}
    else:
        unit_readiness = assess_mechanical_units(draft)
        if (has_boundary_load or has_semantic_spring) and not unit_readiness.ready:
            details = [
                *(f"missing {path}" for path in unit_readiness.missing_fields),
                *(
                    f"unconfirmed {path}"
                    for path in unit_readiness.unconfirmed_fields
                ),
                *unit_readiness.semantic_errors,
            ]
            raise FormulationFinalizationError(
                "Mechanical units are not ready (" + "; ".join(details) + ")."
            )
        context = unit_readiness.context
        units = (
            {
                "kind": "explicit",
                "length_unit": context.length_unit,
                "force_unit": context.force_unit,
                "stress_unit": context.stress_unit,
            }
            if context is not None
            else {"kind": "legacy_consistent"}
        )

    conditions = _first_class_boundary_conditions(
        state,
        bounds=source.domain.bounds,
        context=context,
        legacy_migration=legacy_migration,
    )
    mechanism_springs = None
    spring_evidence = ()
    if source.problem_type == "compliant_mechanism" and any(
        item.kind in {"input_spring", "output_spring"}
        for item in state.conditions
    ):
        if context is None:
            raise FormulationFinalizationError(
                "First-class mechanism springs require explicit mechanical units."
            )
        mechanism_springs, spring_evidence = _first_class_mechanism_springs(
            state,
            bounds=source.domain.bounds,
            context=context,
        )
    return _compile_problem(
        source,
        boundary_conditions=conditions,
        units=units,
        mechanism_springs=mechanism_springs,
        spring_evidence=spring_evidence,
    )
