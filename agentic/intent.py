"""Typed semantic boundary between free text and deterministic compilation."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from fenitop.regions import RegionSpec, parse_region


def _finite(value: int | float) -> int | float:
    if not math.isfinite(float(value)):
        raise ValueError("must be finite.")
    return value


FiniteNumber = Annotated[
    Union[StrictInt, StrictFloat],
    AfterValidator(_finite),
]
PositiveFiniteNumber = Annotated[FiniteNumber, Field(gt=0)]
OpenFraction = Annotated[FiniteNumber, Field(gt=0, lt=1)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
Vector2D = tuple[FiniteNumber, FiniteNumber]
Point2D = tuple[FiniteNumber, FiniteNumber]
Bounds2D = tuple[Point2D, Point2D]


class StrictIntentModel(BaseModel):
    """Forbid silent acceptance of misspelled or invented model fields."""

    model_config = ConfigDict(extra="forbid")


class RectangularDomainIntent(StrictIntentModel):
    """Problem-defining 2D geometry; meshing is a separate preference."""

    bounds: Bounds2D

    @model_validator(mode="after")
    def _ordered_bounds(self):
        (x0, y0), (x1, y1) = self.bounds
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bounds must satisfy x1>x0 and y1>y0.")
        return self


class MaterialIntent(StrictIntentModel):
    young_modulus: PositiveFiniteNumber
    poisson_ratio: FiniteNumber

    @field_validator("poisson_ratio")
    @classmethod
    def _poisson_range(cls, value):
        if not -1.0 < value < 0.5:
            raise ValueError("must satisfy -1 < poisson_ratio < 0.5.")
        return value


class FixedSupportIntent(StrictIntentModel):
    """A full-vector zero clamp; component-wise supports are not representable."""

    region: RegionSpec

    @field_validator("region")
    @classmethod
    def _bounded_region(cls, value):
        return parse_region(value)


class TractionIntent(StrictIntentModel):
    """A distributed boundary traction, not a mesh-independent total force."""

    region: RegionSpec
    vector: Vector2D

    @field_validator("region")
    @classmethod
    def _bounded_region(cls, value):
        return parse_region(value)

    @field_validator("vector")
    @classmethod
    def _nonzero_vector(cls, value):
        if all(float(component) == 0.0 for component in value):
            raise ValueError("traction vector must be nonzero.")
        return value


class MechanismSpringIntent(StrictIntentModel):
    region: RegionSpec
    direction: Literal["x", "y"]
    stiffness: PositiveFiniteNumber

    @field_validator("region")
    @classmethod
    def _bounded_region(cls, value):
        return parse_region(value)


class MeshPreferences(StrictIntentModel):
    """Optional user choices; ``None`` means the compiler must disclose a default."""

    divisions: tuple[PositiveInt, PositiveInt] | None = None
    cell_type: Literal["quadrilateral", "triangle"] | None = None


class OptimizationPreferences(StrictIntentModel):
    """Optional numerical tuning, deliberately separate from structural physics."""

    filter_radius: PositiveFiniteNumber | None = None
    max_iter: PositiveInt | None = None


class ProblemIntentBase(StrictIntentModel):
    domain: RectangularDomainIntent
    material: MaterialIntent
    supports: Annotated[list[FixedSupportIntent], Field(min_length=1)]
    tractions: list[TractionIntent] = Field(default_factory=list)
    body_force: Vector2D = (0.0, 0.0)
    volume_fraction: OpenFraction
    mesh: MeshPreferences = Field(default_factory=MeshPreferences)
    optimization: OptimizationPreferences = Field(
        default_factory=OptimizationPreferences
    )

    @model_validator(mode="after")
    def _external_load_required(self):
        if not self.tractions and all(
            float(component) == 0.0 for component in self.body_force
        ):
            raise ValueError(
                "at least one distributed traction or a nonzero body force is required."
            )
        return self


class ComplianceProblemIntent(ProblemIntentBase):
    problem_type: Literal["minimize_compliance"]


class MechanismProblemIntent(ProblemIntentBase):
    problem_type: Literal["compliant_mechanism"]
    compliance_bound: PositiveFiniteNumber
    input_spring: MechanismSpringIntent
    output_spring: MechanismSpringIntent


ProblemIntent = Annotated[
    Union[ComplianceProblemIntent, MechanismProblemIntent],
    Field(discriminator="problem_type"),
]


class ReadyInterpretation(StrictIntentModel):
    status: Literal["ready"]
    intent: ProblemIntent


class NeedsClarificationInterpretation(StrictIntentModel):
    status: Literal["needs_clarification"]
    missing_fields: Annotated[list[str], Field(min_length=1)]
    questions: Annotated[list[str], Field(min_length=1)]

    @field_validator("missing_fields", "questions")
    @classmethod
    def _nonblank_items(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError("items must not be blank.")
        return values


class UnsupportedInterpretation(StrictIntentModel):
    status: Literal["unsupported"]
    unsupported_features: Annotated[list[str], Field(min_length=1)]
    explanation: Annotated[str, Field(min_length=1)]

    @field_validator("unsupported_features")
    @classmethod
    def _nonblank_features(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError("items must not be blank.")
        return values


InterpretationResult = Annotated[
    Union[
        ReadyInterpretation,
        NeedsClarificationInterpretation,
        UnsupportedInterpretation,
    ],
    Field(discriminator="status"),
]


class InterpretationEnvelope(StrictIntentModel):
    """Transport wrapper required by CrewAI's structured-output interface."""

    result: InterpretationResult
