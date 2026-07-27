"""Strict, JSON-only region DSL used by the agent-safe tool contract.

The serialized surface is a discriminated Pydantic union. It can describe
geometry, but it cannot carry Python source or an arbitrary callable. The compiler
below is deliberately the only conversion from a validated region to the
``f(x) -> bool array`` convention used by Dolfinx.
"""
from __future__ import annotations

import math
from typing import Annotated, Literal, Union

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    TypeAdapter,
    field_validator,
)

MAX_REGION_DEPTH = 8
MAX_REGION_NODES = 64
MAX_BOOLEAN_CHILDREN = 16


def _finite_number(value):
    if not math.isfinite(float(value)):
        raise ValueError("must be finite.")
    return value


FiniteNumber = Annotated[
    Union[StrictInt, StrictFloat],
    AfterValidator(_finite_number),
]
PositiveFiniteNumber = Annotated[FiniteNumber, Field(gt=0)]
Axis2D = Literal["x", "y"]
Point2D = tuple[FiniteNumber, FiniteNumber]


class RegionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlaneRegion(RegionModel):
    op: Literal["plane"]
    axis: Axis2D
    value: FiniteNumber
    tol: PositiveFiniteNumber = 1e-8


class RangeRegion(RegionModel):
    op: Literal["range"]
    axis: Axis2D
    min: FiniteNumber
    max: FiniteNumber
    min_inclusive: bool = True
    max_inclusive: bool = True

    @field_validator("max")
    @classmethod
    def _ordered(cls, value, info):
        lower = info.data.get("min")
        if lower is not None and value < lower:
            raise ValueError(f"must be >= min ({lower}); got {value}.")
        return value


class CircleRegion(RegionModel):
    op: Literal["circle"]
    center: Point2D
    radius: PositiveFiniteNumber
    inside: bool = True


class AllRegion(RegionModel):
    op: Literal["all"]


class NoneRegion(RegionModel):
    op: Literal["none"]


class AndRegion(RegionModel):
    op: Literal["and"]
    regions: Annotated[list["RegionSpec"], Field(min_length=1, max_length=MAX_BOOLEAN_CHILDREN)]


class OrRegion(RegionModel):
    op: Literal["or"]
    regions: Annotated[list["RegionSpec"], Field(min_length=1, max_length=MAX_BOOLEAN_CHILDREN)]


class NotRegion(RegionModel):
    op: Literal["not"]
    region: "RegionSpec"


RegionSpec = Annotated[
    Union[
        PlaneRegion,
        RangeRegion,
        CircleRegion,
        AllRegion,
        NoneRegion,
        AndRegion,
        OrRegion,
        NotRegion,
    ],
    Field(discriminator="op"),
]

AndRegion.model_rebuild()
OrRegion.model_rebuild()
NotRegion.model_rebuild()
_REGION_ADAPTER = TypeAdapter(RegionSpec)
_KNOWN_OPS = {"plane", "range", "circle", "all", "none", "and", "or", "not"}


class RegionError(ValueError):
    """Raised when a region is invalid or exceeds contract complexity limits."""


def is_region_spec(value) -> bool:
    return isinstance(value, (RegionModel, dict)) and (
        not isinstance(value, dict) or value.get("op") in _KNOWN_OPS
    )


def _count_region(region: RegionModel, depth: int = 1) -> tuple[int, int]:
    if isinstance(region, (AndRegion, OrRegion)):
        child_counts = [_count_region(child, depth + 1) for child in region.regions]
    elif isinstance(region, NotRegion):
        child_counts = [_count_region(region.region, depth + 1)]
    else:
        child_counts = []
    nodes = 1 + sum(item[0] for item in child_counts)
    deepest = max([depth, *(item[1] for item in child_counts)])
    return nodes, deepest


def parse_region(value) -> RegionSpec:
    """Validate a JSON-shaped region and enforce bounded recursive complexity."""
    try:
        region = value if isinstance(value, RegionModel) else _REGION_ADAPTER.validate_python(value)
    except Exception as exc:
        raise RegionError(str(exc)) from exc
    nodes, depth = _count_region(region)
    if nodes > MAX_REGION_NODES:
        raise RegionError(
            f"Region contains {nodes} nodes; maximum is {MAX_REGION_NODES}."
        )
    if depth > MAX_REGION_DEPTH:
        raise RegionError(
            f"Region nesting depth is {depth}; maximum is {MAX_REGION_DEPTH}."
        )
    return region


def region_to_dict(value) -> dict:
    return parse_region(value).model_dump(mode="json")


def compile_region(value):
    """Compile a validated 2D region into a vectorized NumPy predicate."""
    region = parse_region(value)

    if isinstance(region, PlaneRegion):
        axis = 0 if region.axis == "x" else 1
        return lambda x: np.isclose(
            x[axis], float(region.value), atol=float(region.tol), rtol=0.0
        )

    if isinstance(region, RangeRegion):
        axis = 0 if region.axis == "x" else 1

        def region_range(x):
            coordinate = x[axis]
            lower = (
                coordinate >= region.min
                if region.min_inclusive
                else coordinate > region.min
            )
            upper = (
                coordinate <= region.max
                if region.max_inclusive
                else coordinate < region.max
            )
            return lower & upper

        return region_range

    if isinstance(region, CircleRegion):
        center = np.asarray(region.center, dtype=float).reshape(2, 1)
        radius_squared = float(region.radius) ** 2

        def region_circle(x):
            distance_squared = np.sum((x[:2] - center) ** 2, axis=0)
            return (
                distance_squared <= radius_squared
                if region.inside
                else distance_squared > radius_squared
            )

        return region_circle

    if isinstance(region, AllRegion):
        return lambda x: np.full(x.shape[1], True, dtype=bool)

    if isinstance(region, NoneRegion):
        return lambda x: np.full(x.shape[1], False, dtype=bool)

    if isinstance(region, (AndRegion, OrRegion)):
        children = [compile_region(child) for child in region.regions]
        initial = isinstance(region, AndRegion)

        def region_boolean(x):
            result = np.full(x.shape[1], initial, dtype=bool)
            for child in children:
                if initial:
                    result &= child(x)
                else:
                    result |= child(x)
            return result

        return region_boolean

    child = compile_region(region.region)
    return lambda x: ~child(x)
