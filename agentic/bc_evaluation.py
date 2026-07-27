"""Provider-independent contracts and graders for boundary-condition language.

This module deliberately does not participate in the live formulation path yet.
It defines the semantic target that the next implementation work packages must
meet, so prompt or model changes can be compared against one stable corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BC_EVALUATION_VERSION = "boundary-condition-evals-v7"

ScenarioFamily = Literal[
    "support_aliases",
    "support_segments",
    "distributed_loads",
    "resultant_loads",
    "multiple_and_corrections",
    "ambiguity_and_conflicts",
    "capability_boundaries",
]
SemanticState = Literal[
    "complete",
    "incomplete",
    "ambiguous",
    "needs_reformulation",
    "requires_capability",
]
BoundaryKind = Literal["support", "load"]
SupportKind = Literal[
    "fixed_all",
    "roller_normal",
    "roller_x",
    "roller_y",
    "symmetry",
    "pin",
]
LoadKind = Literal[
    "traction_vector",
    "traction_magnitude",
    "resultant_vector",
    "resultant_magnitude",
    "pressure",
    "point_force",
    "moment",
    "varying_traction",
]
Edge = Literal["left", "right", "bottom", "top"]
SelectorKind = Literal[
    "whole_edge",
    "centered_fraction",
    "fraction_interval",
    "coordinate_interval",
    "centered_width",
    "distance_from_corner",
    "boundary_point",
    "unspecified_extent",
    "unspecified",
]
Direction = Literal[
    "up",
    "down",
    "left",
    "right",
    "inward",
    "outward",
    "normal",
    "tangential",
    "x",
    "y",
]
Completeness = Literal["complete", "partial", "capability_limited"]
ClarificationCode = Literal[
    "boundary_edge",
    "boundary_extent",
    "boundary_coordinates",
    "clarify_width_axis",
    "conflicting_boundary_location",
    "conflicting_support_kind",
    "load_direction",
    "load_magnitude",
    "load_quantity_kind",
    "pressure_orientation",
    "support_kind",
    "target_boundary_condition",
    "unit_system",
    "confirm_finite_patch",
    "confirm_uniform_distribution",
    "confirm_resultant_conversion",
]
AssumptionCode = Literal[
    "finite_patch_span",
    "uniform_distribution",
    "unit_system_n_mm_mpa",
]
CapabilityCode = Literal[
    "component_support",
    "point_node_pin",
    "mathematical_point_load",
    "nonzero_prescribed_displacement",
    "applied_moment",
    "varying_traction",
]
BehaviorCode = Literal[
    "solver_start",
    "discard_partial_boundary_condition",
    "invent_boundary_extent",
    "invent_load_direction",
    "invent_load_magnitude",
    "invent_units",
    "merge_distinct_boundary_conditions",
    "overwrite_unrelated_boundary_condition",
    "pin_to_clamp",
    "point_to_traction_silently",
    "resultant_to_traction_silently",
    "roller_to_clamp",
]


class StrictEvaluationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


FinitePair = tuple[float, float]


class ExpectedBoundaryCondition(StrictEvaluationModel):
    """One final or partial semantic BC expected after the scenario turns."""

    bc_id: str = Field(pattern=r"^[SL][1-9][0-9]*$")
    kind: BoundaryKind
    completeness: Completeness = "complete"
    support_kind: SupportKind | None = None
    load_kind: LoadKind | None = None
    edge: Edge | None = None
    selector_kind: SelectorKind = "unspecified"
    start: float | None = None
    end: float | None = None
    center: float | None = None
    span: float | None = None
    width: float | None = None
    from_corner: Literal[
        "lower_left", "upper_left", "lower_right", "upper_right"
    ] | None = None
    point: FinitePair | None = None
    vector: FinitePair | None = None
    magnitude: float | None = None
    direction: Direction | None = None
    unit: str | None = None
    distribution: Literal["uniform", "point", "varying"] | None = None

    @field_validator("unit")
    @classmethod
    def _nonblank_unit(cls, value):
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("unit must not be blank.")
        return stripped

    @model_validator(mode="after")
    def _kind_and_selector_are_consistent(self):
        expected_prefix = "S" if self.kind == "support" else "L"
        if not self.bc_id.startswith(expected_prefix):
            raise ValueError(
                f"{self.kind} boundary IDs must start with {expected_prefix}."
            )
        if self.kind == "support":
            if self.completeness == "complete" and self.support_kind is None:
                raise ValueError("support_kind is required for a complete support.")
            if self.load_kind is not None:
                raise ValueError("a support cannot carry load_kind.")
        else:
            if self.completeness == "complete" and self.load_kind is None:
                raise ValueError("load_kind is required for a complete load.")
            if self.support_kind is not None:
                raise ValueError("a load cannot carry support_kind.")

        if self.selector_kind in {
            "centered_fraction",
            "fraction_interval",
        }:
            for name in ("start", "end", "center", "span"):
                value = getattr(self, name)
                if value is not None and not 0 <= value <= 1:
                    raise ValueError(f"{name} must lie in [0,1] for a fraction selector.")
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("selector end must be greater than or equal to start.")
        if self.span is not None and self.span <= 0:
            raise ValueError("selector span must be positive.")
        if self.width is not None and self.width <= 0:
            raise ValueError("selector width must be positive.")
        if self.magnitude is not None and self.magnitude <= 0:
            raise ValueError("magnitude must be positive.")
        if self.selector_kind == "whole_edge" and self.edge is None:
            raise ValueError("whole_edge requires an edge.")
        if self.selector_kind == "centered_fraction":
            if self.edge is None or self.center is None or self.span is None:
                raise ValueError(
                    "centered_fraction requires edge, center, and span."
                )
        if self.selector_kind in {"fraction_interval", "coordinate_interval"}:
            if self.edge is None or self.start is None or self.end is None:
                raise ValueError(
                    f"{self.selector_kind} requires edge, start, and end."
                )
        if self.selector_kind == "centered_width":
            if self.edge is None or self.center is None or self.width is None:
                raise ValueError("centered_width requires edge, center, and width.")
        if self.selector_kind == "distance_from_corner":
            if (
                self.edge is None
                or self.from_corner is None
                or self.start is None
                or self.span is None
            ):
                raise ValueError(
                    "distance_from_corner requires edge, corner, start, and span."
                )
        if (
            self.selector_kind == "boundary_point"
            and self.point is None
            and self.from_corner is None
        ):
            if self.edge is None or self.center is None:
                raise ValueError(
                    "boundary_point requires an absolute point, named corner, "
                    "or edge plus relative center."
                )
            if not 0 <= self.center <= 1:
                raise ValueError(
                    "relative boundary-point center must lie in [0,1]."
                )
        if self.completeness == "complete":
            if self.selector_kind in {"unspecified", "unspecified_extent"}:
                raise ValueError(
                    "a complete boundary condition requires a complete selector."
                )
            if self.kind == "load":
                if self.load_kind in {
                    "traction_vector",
                    "resultant_vector",
                } and self.vector is None:
                    raise ValueError(f"{self.load_kind} requires vector.")
                if self.load_kind in {
                    "traction_magnitude",
                    "resultant_magnitude",
                    "pressure",
                } and self.magnitude is None:
                    raise ValueError(f"{self.load_kind} requires magnitude.")
                if self.load_kind in {
                    "traction_magnitude",
                    "resultant_magnitude",
                    "pressure",
                } and self.direction is None:
                    raise ValueError(f"{self.load_kind} requires direction.")
        return self


class ExpectedBoundaryOutcome(StrictEvaluationModel):
    semantic_state: SemanticState
    boundary_conditions: tuple[ExpectedBoundaryCondition, ...] = ()
    clarifications: tuple[ClarificationCode, ...] = ()
    assumptions: tuple[AssumptionCode, ...] = ()
    capability_limits: tuple[CapabilityCode, ...] = ()
    forbidden_behaviors: tuple[BehaviorCode, ...] = ()

    @model_validator(mode="after")
    def _state_has_required_evidence(self):
        ids = [bc.bc_id for bc in self.boundary_conditions]
        if len(ids) != len(set(ids)):
            raise ValueError("expected boundary-condition IDs must be unique.")
        if self.semantic_state in {"incomplete", "ambiguous"} and not self.clarifications:
            raise ValueError(
                f"{self.semantic_state} outcomes require at least one clarification."
            )
        if (
            self.semantic_state in {"needs_reformulation", "requires_capability"}
            and not self.capability_limits
        ):
            raise ValueError(
                f"{self.semantic_state} outcomes require a capability limit."
            )
        return self


class BoundaryScenario(StrictEvaluationModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    family: ScenarioFamily
    purpose: str
    context: str | None = None
    turns: Annotated[tuple[str, ...], Field(min_length=1)]
    expected: ExpectedBoundaryOutcome
    tags: tuple[str, ...] = ()

    @field_validator("purpose", "context")
    @classmethod
    def _strip_optional_text(cls, value):
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be blank.")
        return stripped

    @field_validator("turns", "tags")
    @classmethod
    def _nonblank_items(cls, values):
        stripped = tuple(value.strip() for value in values)
        if any(not value for value in stripped):
            raise ValueError("items must not be blank.")
        return stripped


class BoundaryEvaluationSuite(StrictEvaluationModel):
    version: Literal[BC_EVALUATION_VERSION]
    description: str
    global_forbidden_behaviors: tuple[BehaviorCode, ...]
    scenarios: tuple[BoundaryScenario, ...]

    @field_validator("description")
    @classmethod
    def _nonblank_description(cls, value):
        if not value.strip():
            raise ValueError("description must not be blank.")
        return value.strip()

    @model_validator(mode="after")
    def _unique_scenario_ids(self):
        ids = [scenario.id for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario IDs must be unique.")
        return self


class BoundaryObservation(StrictEvaluationModel):
    """Provider/application result normalized for deterministic corpus grading."""

    semantic_state: SemanticState
    boundary_conditions: tuple[ExpectedBoundaryCondition, ...] = ()
    clarifications: tuple[ClarificationCode, ...] = ()
    assumptions: tuple[AssumptionCode, ...] = ()
    capability_limits: tuple[CapabilityCode, ...] = ()
    behavior_violations: tuple[BehaviorCode, ...] = ()
    solver_started: bool = False


class GradeCheck(StrictEvaluationModel):
    name: str
    passed: bool
    detail: str


class BoundaryGrade(StrictEvaluationModel):
    scenario_id: str
    passed: bool
    checks: tuple[GradeCheck, ...]


def load_boundary_evaluation_suite(
    path: str | Path,
) -> BoundaryEvaluationSuite:
    """Load and strictly validate a versioned BC language corpus."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return BoundaryEvaluationSuite.model_validate(payload)


def _check(name: str, passed: bool, detail: str) -> GradeCheck:
    return GradeCheck(name=name, passed=passed, detail=detail)


def _canonical_boundary_condition(
    condition: ExpectedBoundaryCondition,
) -> dict:
    """Canonicalize mathematically equivalent fractional selector forms."""
    payload = condition.model_dump(mode="python")
    if payload["selector_kind"] == "centered_fraction":
        center = payload["center"]
        span = payload["span"]
        payload["selector_kind"] = "fraction_interval"
        payload["start"] = round(center - span / 2.0, 12)
        payload["end"] = round(center + span / 2.0, 12)
        payload["center"] = None
        payload["span"] = None
    if payload["point"] is not None:
        payload["from_corner"] = None
    vector = payload["vector"]
    direction = payload["direction"]
    if vector is not None and direction is not None:
        x_value, y_value = vector
        implied_direction = None
        if x_value == 0 and y_value < 0:
            implied_direction = "down"
        elif x_value == 0 and y_value > 0:
            implied_direction = "up"
        elif y_value == 0 and x_value < 0:
            implied_direction = "left"
        elif y_value == 0 and x_value > 0:
            implied_direction = "right"
        if direction == implied_direction:
            payload["direction"] = None
    return payload


def _boundary_conditions_equal(
    expected: tuple[ExpectedBoundaryCondition, ...],
    observed: tuple[ExpectedBoundaryCondition, ...],
) -> bool:
    return tuple(map(_canonical_boundary_condition, expected)) == tuple(
        map(_canonical_boundary_condition, observed)
    )


def grade_boundary_observation(
    suite: BoundaryEvaluationSuite,
    scenario: BoundaryScenario,
    observation: BoundaryObservation,
) -> BoundaryGrade:
    """Grade semantic equality and non-negotiable safety behavior."""
    expected = scenario.expected
    forbidden = set(suite.global_forbidden_behaviors) | set(
        expected.forbidden_behaviors
    )
    observed_violations = set(observation.behavior_violations)
    checks = (
        _check(
            "semantic_state",
            observation.semantic_state == expected.semantic_state,
            (
                f"expected={expected.semantic_state} "
                f"observed={observation.semantic_state}"
            ),
        ),
        _check(
            "boundary_conditions",
            _boundary_conditions_equal(
                expected.boundary_conditions,
                observation.boundary_conditions,
            ),
            (
                f"expected={expected.boundary_conditions!r} "
                f"observed={observation.boundary_conditions!r}"
            ),
        ),
        _check(
            "clarifications",
            set(observation.clarifications) == set(expected.clarifications),
            (
                f"expected={sorted(expected.clarifications)} "
                f"observed={sorted(observation.clarifications)}"
            ),
        ),
        _check(
            "assumptions",
            set(observation.assumptions) == set(expected.assumptions),
            (
                f"expected={sorted(expected.assumptions)} "
                f"observed={sorted(observation.assumptions)}"
            ),
        ),
        _check(
            "capability_limits",
            set(observation.capability_limits)
            == set(expected.capability_limits),
            (
                f"expected={sorted(expected.capability_limits)} "
                f"observed={sorted(observation.capability_limits)}"
            ),
        ),
        _check(
            "forbidden_behaviors",
            not bool(observed_violations & forbidden),
            (
                "violations="
                f"{sorted(observed_violations & forbidden)}"
            ),
        ),
        _check(
            "solver_not_started",
            not observation.solver_started,
            f"solver_started={observation.solver_started}",
        ),
    )
    return BoundaryGrade(
        scenario_id=scenario.id,
        passed=all(check.passed for check in checks),
        checks=checks,
    )
