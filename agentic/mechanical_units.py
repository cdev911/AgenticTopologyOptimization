"""Deterministic mechanical-unit validation and normalization.

Pint is deliberately kept behind JSON-safe Pydantic models.  No Pint quantity
crosses the formulation, approval, or solver boundaries.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

import pint
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


QuantityKind = Literal["length", "force", "stress"]


def _nonblank_unit(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("unit must not be blank.")
    if len(stripped) > 64:
        raise ValueError("unit must contain at most 64 characters.")
    return stripped


UnitName = Annotated[str, AfterValidator(_nonblank_unit)]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]


class StrictUnitModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


_REGISTRY = pint.UnitRegistry(autoconvert_offset_to_baseunit=False)
_REFERENCE_UNITS = {
    "length": _REGISTRY.meter,
    "force": _REGISTRY.newton,
    "stress": _REGISTRY.pascal,
}


def _unit(unit_name: str):
    try:
        return _REGISTRY.Unit(unit_name)
    except (pint.errors.PintError, ValueError) as exc:
        raise ValueError(f"unknown or invalid unit {unit_name!r}.") from exc


def _require_dimension(unit_name: str, quantity_kind: QuantityKind) -> None:
    candidate = _unit(unit_name)
    expected = _REFERENCE_UNITS[quantity_kind]
    if candidate.dimensionality != expected.dimensionality:
        raise ValueError(
            f"{unit_name!r} is not a {quantity_kind} unit."
        )


def _length_unit(value: str) -> str:
    _require_dimension(value, "length")
    return value


def _force_unit(value: str) -> str:
    _require_dimension(value, "force")
    return value


def _stress_unit(value: str) -> str:
    _require_dimension(value, "stress")
    return value


LengthUnitName = Annotated[UnitName, AfterValidator(_length_unit)]
ForceUnitName = Annotated[UnitName, AfterValidator(_force_unit)]
StressUnitName = Annotated[UnitName, AfterValidator(_stress_unit)]


class MechanicalUnitContext(StrictUnitModel):
    """One internally consistent display and solver unit system.

    The numerical model remains the existing 2D plane-strain model with an
    implicit out-of-plane thickness of one ``length_unit``.
    """

    length_unit: LengthUnitName
    force_unit: ForceUnitName
    stress_unit: StressUnitName
    thickness_value: FiniteNumber = 1.0

    @field_validator("thickness_value")
    @classmethod
    def _fixed_thickness(cls, value):
        if not math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                "the current solver contract requires implicit thickness 1."
            )
        return float(value)

    @model_validator(mode="after")
    def _mechanically_consistent(self):
        try:
            (1 * _unit(self.stress_unit)).to(
                _unit(self.force_unit) / _unit(self.length_unit) ** 2
            )
        except pint.errors.DimensionalityError as exc:
            raise ValueError(
                "stress_unit must be compatible with force_unit/length_unit^2."
            ) from exc
        return self

    @property
    def thickness_unit(self) -> str:
        return self.length_unit

    def unit_for(self, quantity_kind: QuantityKind) -> str:
        return {
            "length": self.length_unit,
            "force": self.force_unit,
            "stress": self.stress_unit,
        }[quantity_kind]


class NormalizedScalar(StrictUnitModel):
    quantity_kind: QuantityKind
    original_value: FiniteNumber
    original_unit: UnitName
    normalized_value: FiniteNumber
    normalized_unit: UnitName


class NormalizedVector2D(StrictUnitModel):
    quantity_kind: QuantityKind
    original_value: tuple[FiniteNumber, FiniteNumber]
    original_unit: UnitName
    normalized_value: tuple[FiniteNumber, FiniteNumber]
    normalized_unit: UnitName


def normalize_scalar(
    value: int | float,
    unit_name: str,
    quantity_kind: QuantityKind,
    context: MechanicalUnitContext,
) -> NormalizedScalar:
    """Normalize a scalar while retaining exactly what the user supplied."""
    source = _nonblank_unit(unit_name)
    _require_dimension(source, quantity_kind)
    target = context.unit_for(quantity_kind)
    try:
        normalized = (float(value) * _unit(source)).to(_unit(target)).magnitude
    except (pint.errors.PintError, ValueError) as exc:
        raise ValueError(
            f"cannot convert {source!r} to configured {target!r}."
        ) from exc
    return NormalizedScalar(
        quantity_kind=quantity_kind,
        original_value=float(value),
        original_unit=source,
        normalized_value=float(normalized),
        normalized_unit=target,
    )


def normalize_vector(
    value: tuple[int | float, int | float],
    unit_name: str,
    quantity_kind: QuantityKind,
    context: MechanicalUnitContext,
) -> NormalizedVector2D:
    """Normalize both components using one mechanical quantity unit."""
    first = normalize_scalar(value[0], unit_name, quantity_kind, context)
    second = normalize_scalar(value[1], unit_name, quantity_kind, context)
    return NormalizedVector2D(
        quantity_kind=quantity_kind,
        original_value=(float(value[0]), float(value[1])),
        original_unit=first.original_unit,
        normalized_value=(
            first.normalized_value,
            second.normalized_value,
        ),
        normalized_unit=first.normalized_unit,
    )
