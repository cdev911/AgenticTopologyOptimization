"""Deterministic semantic resolution for first-class boundary loads.

This layer understands user-facing load meaning but deliberately knows nothing
about mesh facets.  In particular, a total resultant remains deferred until
Package 3 supplies an authoritative resolved boundary measure.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict

from agentic.boundary_draft import (
    BoundaryConditionDraft,
    BoundaryDraftReadiness,
    BoundaryDraftState,
    Direction,
    Edge,
    assess_boundary_state,
)
from agentic.mechanical_units import (
    MechanicalUnitContext,
    NormalizedVector2D,
    normalize_vector,
)


class StrictLoadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


LoadResolutionStatus = Literal["resolved", "deferred", "invalid", "unsupported"]
ResolvedQuantityKind = Literal["traction", "resultant"]
UnitSource = Literal["load", "mechanical_context"]


class LoadResolutionIssue(StrictLoadModel):
    code: Literal[
        "not_a_load",
        "missing_field",
        "invalid_unit",
        "invalid_direction",
        "zero_load",
        "unsupported_distribution",
        "unsupported_load_kind",
    ]
    field: str
    message: str


class SemanticLoad(StrictLoadModel):
    bc_id: str
    source_kind: str
    quantity_kind: ResolvedQuantityKind
    vector: NormalizedVector2D
    unit_source: UnitSource
    edge: Edge | None
    direction: Direction | None
    distribution: Literal["uniform"]
    effective_traction: NormalizedVector2D | None = None
    requires_boundary_measure: bool = False
    thickness_value: float = 1.0
    thickness_unit: str


class LoadResolution(StrictLoadModel):
    status: LoadResolutionStatus
    load: SemanticLoad | None = None
    issues: tuple[LoadResolutionIssue, ...] = ()


class BoundaryLoadState(StrictLoadModel):
    """Combined Package 1 completeness and Package 2 semantic readiness."""

    semantic_ready: bool
    execution_ready: bool
    boundary_readiness: BoundaryDraftReadiness
    loads: tuple[LoadResolution, ...]


_GLOBAL_DIRECTIONS: dict[str, tuple[float, float]] = {
    "up": (0.0, 1.0),
    "down": (0.0, -1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
}
_OUTWARD_NORMALS: dict[str, tuple[float, float]] = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "bottom": (0.0, -1.0),
    "top": (0.0, 1.0),
}


def resolve_direction(
    direction: Direction,
    edge: Edge | None,
) -> tuple[float, float]:
    """Convert an unambiguous semantic direction into global x/y components."""
    if direction in _GLOBAL_DIRECTIONS:
        return _GLOBAL_DIRECTIONS[direction]
    if direction in {"inward", "outward"}:
        if edge is None:
            raise ValueError(
                f"{direction} requires a named rectangle edge."
            )
        outward = _OUTWARD_NORMALS[edge]
        factor = -1.0 if direction == "inward" else 1.0
        return (factor * outward[0], factor * outward[1])
    if direction in {"x", "y"}:
        raise ValueError(f"{direction} does not specify a positive or negative sense.")
    if direction == "normal":
        raise ValueError("normal does not specify inward or outward.")
    raise ValueError("tangential does not specify either along-edge sense.")


def _issue(
    code,
    field: str,
    message: str,
    *,
    status: LoadResolutionStatus = "invalid",
) -> LoadResolution:
    return LoadResolution(
        status=status,
        issues=(LoadResolutionIssue(code=code, field=field, message=message),),
    )


def resolve_semantic_load(
    condition: BoundaryConditionDraft,
    context: MechanicalUnitContext,
) -> LoadResolution:
    """Normalize one uniform boundary load without performing mesh arithmetic."""
    if condition.kind != "load":
        return _issue("not_a_load", "kind", "Boundary condition is not a load.")

    values = condition.values()
    load_kind = values.get("load.kind")
    if load_kind is None:
        return _issue("missing_field", "load.kind", "Load kind is required.")
    if load_kind in {"point_force", "moment", "varying_traction"}:
        return _issue(
            "unsupported_load_kind",
            "load.kind",
            f"{load_kind} is outside the current solver capability.",
            status="unsupported",
        )

    distribution = values.get("load.distribution")
    if distribution is None:
        return _issue(
            "missing_field",
            "load.distribution",
            "Load distribution must be confirmed.",
        )
    if distribution != "uniform":
        return _issue(
            "unsupported_distribution",
            "load.distribution",
            "Only a uniform finite-segment load is currently supported.",
            status="unsupported",
        )

    quantity_kind = (
        "resultant" if str(load_kind).startswith("resultant") else "traction"
    )
    mechanical_kind = "force" if quantity_kind == "resultant" else "stress"
    explicit_unit = values.get("load.unit")
    unit_name = (
        str(explicit_unit)
        if explicit_unit is not None
        else context.unit_for(mechanical_kind)
    )
    unit_source: UnitSource = (
        "load" if explicit_unit is not None else "mechanical_context"
    )
    edge_value = values.get("selector.edge")
    edge: Edge | None = str(edge_value) if edge_value is not None else None
    direction: Direction | None = None

    if load_kind in {"traction_vector", "resultant_vector"}:
        raw_vector = values.get("load.vector")
        if raw_vector is None:
            return _issue("missing_field", "load.vector", "Load vector is required.")
        vector = (float(raw_vector[0]), float(raw_vector[1]))
    else:
        raw_magnitude = values.get("load.magnitude")
        raw_direction = values.get("load.direction")
        if raw_magnitude is None:
            return _issue(
                "missing_field", "load.magnitude", "Load magnitude is required."
            )
        if raw_direction is None:
            return _issue(
                "missing_field", "load.direction", "Load direction is required."
            )
        direction = str(raw_direction)
        try:
            direction_vector = resolve_direction(direction, edge)
        except ValueError as exc:
            return _issue(
                "invalid_direction", "load.direction", str(exc)
            )
        magnitude = float(raw_magnitude)
        vector = (
            magnitude * direction_vector[0],
            magnitude * direction_vector[1],
        )

    if math.isclose(math.hypot(*vector), 0.0, abs_tol=0.0):
        return _issue("zero_load", "load.vector", "Load vector must be nonzero.")
    try:
        normalized = normalize_vector(
            vector,
            unit_name,
            mechanical_kind,
            context,
        )
    except ValueError as exc:
        return _issue("invalid_unit", "load.unit", str(exc))

    deferred = quantity_kind == "resultant"
    load = SemanticLoad(
        bc_id=condition.bc_id,
        source_kind=str(load_kind),
        quantity_kind=quantity_kind,
        vector=normalized,
        unit_source=unit_source,
        edge=edge,
        direction=direction,
        distribution="uniform",
        effective_traction=None if deferred else normalized,
        requires_boundary_measure=deferred,
        thickness_value=context.thickness_value,
        thickness_unit=context.thickness_unit,
    )
    return LoadResolution(
        status="deferred" if deferred else "resolved",
        load=load,
    )


def resolve_boundary_load_state(
    state: BoundaryDraftState,
    context: MechanicalUnitContext,
) -> BoundaryLoadState:
    """Assess all first-class BCs without changing legacy finalization.

    A normalized resultant is semantically ready but not execution-ready until
    the Package 3 mesh resolver supplies the segment measure.
    """
    boundary_readiness = assess_boundary_state(state)
    loads = tuple(
        resolve_semantic_load(condition, context)
        for condition in state.conditions
        if condition.kind == "load"
    )
    semantic_statuses = {"resolved", "deferred"}
    semantic_ready = (
        boundary_readiness.ready
        and bool(loads)
        and all(item.status in semantic_statuses for item in loads)
    )
    return BoundaryLoadState(
        semantic_ready=semantic_ready,
        execution_ready=(
            semantic_ready and all(item.status == "resolved" for item in loads)
        ),
        boundary_readiness=boundary_readiness,
        loads=loads,
    )
