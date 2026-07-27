"""Typed partial boundary-condition state for conversational formulation.

The application allocates stable IDs and owns every merge. This module is
provider-independent and intentionally does not compile BCs into solver physics.
"""

from __future__ import annotations

import math
import re
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from fenitop.regions import RegionSpec


BoundaryKind = Literal["support", "load"]
BoundaryFactBasis = Literal["explicit", "derived", "assumption", "confirmed"]
BoundaryUpdateBasis = Literal["explicit", "derived", "assumption"]
BoundaryField = Literal[
    "support.kind",
    "support.direction",
    "support.magnitude",
    "support.unit",
    "load.kind",
    "load.vector",
    "load.magnitude",
    "load.direction",
    "load.unit",
    "load.distribution",
    "selector.kind",
    "selector.edge",
    "selector.start",
    "selector.end",
    "selector.center",
    "selector.span",
    "selector.width",
    "selector.offset",
    "selector.length",
    "selector.from_corner",
    "selector.point",
    "selector.region",
]
BOUNDARY_FIELDS: tuple[BoundaryField, ...] = (
    "support.kind",
    "support.direction",
    "support.magnitude",
    "support.unit",
    "load.kind",
    "load.vector",
    "load.magnitude",
    "load.direction",
    "load.unit",
    "load.distribution",
    "selector.kind",
    "selector.edge",
    "selector.start",
    "selector.end",
    "selector.center",
    "selector.span",
    "selector.width",
    "selector.offset",
    "selector.length",
    "selector.from_corner",
    "selector.point",
    "selector.region",
)


def _id_order(bc_id: str) -> tuple[int, int]:
    return (0 if bc_id.startswith("S") else 1, int(bc_id[1:]))


class StrictBoundaryModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


def _finite(value):
    if not math.isfinite(float(value)):
        raise ValueError("must be finite.")
    return value


def _nonblank(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("must not be blank.")
    return stripped


FiniteNumber = Annotated[int | float, AfterValidator(_finite)]
PositiveFiniteNumber = Annotated[FiniteNumber, Field(gt=0)]
UnitText = Annotated[str, AfterValidator(_nonblank)]
Fraction = Annotated[FiniteNumber, Field(ge=0, le=1)]
PositiveFraction = Annotated[FiniteNumber, Field(gt=0, le=1)]
Vector2D = tuple[FiniteNumber, FiniteNumber]
Point2D = tuple[FiniteNumber, FiniteNumber]
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
SelectorKind = Literal[
    "whole_edge",
    "centered_fraction",
    "fraction_interval",
    "coordinate_interval",
    "centered_width",
    "distance_from_corner",
    "boundary_point",
    "expert_region",
    "unspecified_extent",
    "unspecified",
]
Edge = Literal["left", "right", "bottom", "top"]
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
Corner = Literal[
    "lower_left", "upper_left", "lower_right", "upper_right"
]
Distribution = Literal["uniform", "point", "varying"]

_FIELD_ADAPTERS: dict[BoundaryField, TypeAdapter] = {
    "support.kind": TypeAdapter(SupportKind),
    "support.direction": TypeAdapter(Direction),
    "support.magnitude": TypeAdapter(PositiveFiniteNumber),
    "support.unit": TypeAdapter(UnitText),
    "load.kind": TypeAdapter(LoadKind),
    "load.vector": TypeAdapter(Vector2D),
    "load.magnitude": TypeAdapter(PositiveFiniteNumber),
    "load.direction": TypeAdapter(Direction),
    "load.unit": TypeAdapter(UnitText),
    "load.distribution": TypeAdapter(Distribution),
    "selector.kind": TypeAdapter(SelectorKind),
    "selector.edge": TypeAdapter(Edge),
    "selector.start": TypeAdapter(FiniteNumber),
    "selector.end": TypeAdapter(FiniteNumber),
    "selector.center": TypeAdapter(Fraction),
    "selector.span": TypeAdapter(PositiveFraction),
    "selector.width": TypeAdapter(PositiveFiniteNumber),
    "selector.offset": TypeAdapter(
        Annotated[FiniteNumber, Field(ge=0)]
    ),
    "selector.length": TypeAdapter(PositiveFiniteNumber),
    "selector.from_corner": TypeAdapter(Corner),
    "selector.point": TypeAdapter(Point2D),
    "selector.region": TypeAdapter(RegionSpec),
}


def canonical_boundary_value(
    field: BoundaryField,
    value: JsonValue,
) -> JsonValue:
    adapter = _FIELD_ADAPTERS[field]
    validated = adapter.validate_python(value)
    return adapter.dump_python(validated, mode="json")


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _quote_is_supported(quote: str | None, user_message: str) -> bool:
    return bool(
        quote
        and _normalized_text(quote) in _normalized_text(user_message)
    )


def _validation_message(error: ValidationError) -> str:
    return "; ".join(
        (
            (".".join(str(part) for part in item["loc"]) + ": ")
            if item["loc"]
            else ""
        )
        + item["msg"]
        for item in error.errors(include_url=False)
    )


class BoundaryFieldInput(StrictBoundaryModel):
    field: BoundaryField
    value: JsonValue
    basis: BoundaryUpdateBasis
    source_quote: str | None = None
    rationale: str

    @field_validator("source_quote")
    @classmethod
    def _strip_quote(cls, value):
        return _nonblank(value) if value is not None else None

    @field_validator("rationale")
    @classmethod
    def _rationale(cls, value):
        return _nonblank(value)


class BoundaryCreate(StrictBoundaryModel):
    local_ref: str = Field(pattern=r"^new_[a-z0-9][a-z0-9_]*$")
    kind: BoundaryKind
    fields: Annotated[tuple[BoundaryFieldInput, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _unique_fields(self):
        names = [item.field for item in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("a create operation may set each field once.")
        return self


class BoundaryUpdate(BoundaryFieldInput):
    bc_id: str = Field(pattern=r"^[SL][1-9][0-9]*$")


class BoundaryDelete(StrictBoundaryModel):
    bc_id: str = Field(pattern=r"^[SL][1-9][0-9]*$")
    source_quote: str
    rationale: str

    @field_validator("source_quote", "rationale")
    @classmethod
    def _text(cls, value):
        return _nonblank(value)


class BoundaryConfirm(StrictBoundaryModel):
    bc_id: str = Field(pattern=r"^[SL][1-9][0-9]*$")
    field: BoundaryField
    source_quote: str
    rationale: str

    @field_validator("source_quote", "rationale")
    @classmethod
    def _text(cls, value):
        return _nonblank(value)


class BoundaryPatch(StrictBoundaryModel):
    creates: tuple[BoundaryCreate, ...] = ()
    updates: tuple[BoundaryUpdate, ...] = ()
    deletes: tuple[BoundaryDelete, ...] = ()
    confirmations: tuple[BoundaryConfirm, ...] = ()

    @model_validator(mode="after")
    def _nonconflicting_operations(self):
        refs = [create.local_ref for create in self.creates]
        if len(refs) != len(set(refs)):
            raise ValueError("create local_ref values must be unique.")
        update_keys = [(item.bc_id, item.field) for item in self.updates]
        confirm_keys = [
            (item.bc_id, item.field) for item in self.confirmations
        ]
        if len(update_keys) != len(set(update_keys)):
            raise ValueError("a patch may update each BC field once.")
        if len(confirm_keys) != len(set(confirm_keys)):
            raise ValueError("a patch may confirm each BC field once.")
        overlap = set(update_keys) & set(confirm_keys)
        if overlap:
            raise ValueError("a patch cannot update and confirm the same BC field.")
        deleted = [item.bc_id for item in self.deletes]
        if len(deleted) != len(set(deleted)):
            raise ValueError("a patch may delete each BC once.")
        touched = {item.bc_id for item in self.updates} | {
            item.bc_id for item in self.confirmations
        }
        if set(deleted) & touched:
            raise ValueError("a patch cannot delete and modify the same BC.")
        return self


class BoundaryFieldFact(StrictBoundaryModel):
    field: BoundaryField
    value: JsonValue
    basis: BoundaryFactBasis
    source_turn: int = Field(ge=1)
    source_quote: str | None = None
    rationale: str


class BoundaryConditionDraft(StrictBoundaryModel):
    bc_id: str = Field(pattern=r"^[SL][1-9][0-9]*$")
    kind: BoundaryKind
    created_turn: int = Field(ge=1)
    facts: tuple[BoundaryFieldFact, ...] = ()

    @model_validator(mode="after")
    def _canonical_facts_and_id(self):
        prefix = "S" if self.kind == "support" else "L"
        if not self.bc_id.startswith(prefix):
            raise ValueError(f"{self.kind} IDs must start with {prefix}.")
        names = [fact.field for fact in self.facts]
        if len(names) != len(set(names)):
            raise ValueError("BC facts must have unique fields.")
        if names != sorted(names, key=BOUNDARY_FIELDS.index):
            raise ValueError("BC facts must use canonical field ordering.")
        disallowed_prefix = "load." if self.kind == "support" else "support."
        if any(name.startswith(disallowed_prefix) for name in names):
            raise ValueError(f"{self.kind} BC contains incompatible fields.")
        return self

    def fact(self, field: BoundaryField) -> BoundaryFieldFact | None:
        return next((fact for fact in self.facts if fact.field == field), None)

    def values(self) -> dict[BoundaryField, JsonValue]:
        return {fact.field: fact.value for fact in self.facts}


class BoundaryRevision(StrictBoundaryModel):
    bc_id: str
    turn: int = Field(ge=1)
    action: Literal["create", "set", "delete", "confirm"]
    field: BoundaryField | None = None
    previous_value: JsonValue | None = None
    new_value: JsonValue | None = None
    basis: BoundaryFactBasis | None = None
    source_quote: str = ""
    rationale: str


class BoundaryDraftState(StrictBoundaryModel):
    conditions: tuple[BoundaryConditionDraft, ...] = ()
    revisions: tuple[BoundaryRevision, ...] = ()
    next_support_number: int = Field(default=1, ge=1)
    next_load_number: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _unique_ids_and_monotonic_counters(self):
        ids = [condition.bc_id for condition in self.conditions]
        if len(ids) != len(set(ids)):
            raise ValueError("boundary-condition IDs must be unique.")
        if ids != sorted(ids, key=_id_order):
            raise ValueError("boundary conditions must use canonical ID ordering.")
        for prefix, next_number in (
            ("S", self.next_support_number),
            ("L", self.next_load_number),
        ):
            used = [int(item[1:]) for item in ids if item.startswith(prefix)]
            if used and next_number <= max(used):
                raise ValueError(f"next {prefix} number must exceed existing IDs.")
        return self

    def condition(self, bc_id: str) -> BoundaryConditionDraft | None:
        return next(
            (condition for condition in self.conditions if condition.bc_id == bc_id),
            None,
        )

    def pending_confirmations(self) -> tuple["PendingBoundaryConfirmation", ...]:
        return tuple(
            PendingBoundaryConfirmation(
                bc_id=condition.bc_id,
                field=fact.field,
                value=fact.value,
                rationale=fact.rationale,
            )
            for condition in self.conditions
            for fact in condition.facts
            if fact.basis == "assumption"
        )


class PendingBoundaryConfirmation(StrictBoundaryModel):
    bc_id: str
    field: BoundaryField
    value: JsonValue
    rationale: str


class BoundaryPatchIssue(StrictBoundaryModel):
    code: Literal[
        "duplicate_target",
        "invalid_confirmation",
        "invalid_field",
        "invalid_value",
        "missing_target",
        "unsupported_provenance",
    ]
    path: str
    message: str


class AcceptedBoundaryOperation(StrictBoundaryModel):
    action: Literal["create", "set", "delete", "confirm"]
    bc_id: str
    field: BoundaryField | None = None
    local_ref: str | None = None


class BoundaryMergeResult(StrictBoundaryModel):
    state: BoundaryDraftState
    accepted: tuple[AcceptedBoundaryOperation, ...]
    issues: tuple[BoundaryPatchIssue, ...]


class BoundaryConditionReadiness(StrictBoundaryModel):
    bc_id: str
    ready: bool
    missing_fields: tuple[str, ...]
    unconfirmed_fields: tuple[BoundaryField, ...]
    semantic_errors: tuple[str, ...]
    capability_limits: tuple[str, ...]


class BoundaryDraftReadiness(StrictBoundaryModel):
    ready: bool
    conditions: tuple[BoundaryConditionReadiness, ...]


_CAPABILITY_SUPPORTS = {
    "roller_normal": "component_support",
    "roller_x": "component_support",
    "roller_y": "component_support",
    "symmetry": "component_support",
    "pin": "point_node_pin",
}
_CAPABILITY_LOADS = {
    "point_force": "mathematical_point_load",
    "moment": "applied_moment",
    "varying_traction": "varying_traction",
}


def assess_boundary_state(state: BoundaryDraftState) -> BoundaryDraftReadiness:
    """Assess only BC entity completeness; whole-problem readiness stays elsewhere."""
    results: list[BoundaryConditionReadiness] = []
    for condition in state.conditions:
        values = condition.values()
        missing: list[str] = []
        errors: list[str] = []
        capabilities: list[str] = []

        selector_kind = values.get("selector.kind")
        if selector_kind is None:
            missing.append("selector.kind")
        elif selector_kind in {"unspecified", "unspecified_extent"}:
            missing.append("selector.extent")
        elif selector_kind == "whole_edge":
            if "selector.edge" not in values:
                missing.append("selector.edge")
        elif selector_kind == "centered_fraction":
            for field in ("selector.edge", "selector.center", "selector.span"):
                if field not in values:
                    missing.append(field)
        elif selector_kind in {"fraction_interval", "coordinate_interval"}:
            for field in ("selector.edge", "selector.start", "selector.end"):
                if field not in values:
                    missing.append(field)
            if (
                "selector.start" in values
                and "selector.end" in values
                and float(values["selector.end"]) < float(values["selector.start"])
            ):
                errors.append("selector.end must be greater than or equal to start.")
            if selector_kind == "fraction_interval":
                for field in ("selector.start", "selector.end"):
                    if field in values and not 0 <= float(values[field]) <= 1:
                        errors.append(f"{field} must lie in [0,1].")
        elif selector_kind == "centered_width":
            for field in ("selector.edge", "selector.center", "selector.width"):
                if field not in values:
                    missing.append(field)
        elif selector_kind == "distance_from_corner":
            for field in (
                "selector.edge",
                "selector.from_corner",
                "selector.offset",
                "selector.length",
            ):
                if field not in values:
                    missing.append(field)
        elif selector_kind == "boundary_point":
            has_absolute = "selector.point" in values
            has_relative = (
                "selector.edge" in values and "selector.center" in values
            )
            if not has_absolute and not has_relative:
                missing.append("selector.point_or_edge_center")
        elif selector_kind == "expert_region":
            if "selector.region" not in values:
                missing.append("selector.region")

        if condition.kind == "support":
            support_kind = values.get("support.kind")
            if support_kind is None:
                missing.append("support.kind")
            elif support_kind in _CAPABILITY_SUPPORTS:
                capabilities.append(_CAPABILITY_SUPPORTS[support_kind])
            if any(
                field in values
                for field in (
                    "support.direction",
                    "support.magnitude",
                    "support.unit",
                )
            ):
                capabilities.append("nonzero_prescribed_displacement")
        else:
            load_kind = values.get("load.kind")
            if load_kind is None:
                missing.append("load.kind")
            elif load_kind in _CAPABILITY_LOADS:
                capabilities.append(_CAPABILITY_LOADS[load_kind])
            if load_kind in {"traction_vector", "resultant_vector"}:
                if "load.vector" not in values:
                    missing.append("load.vector")
            elif load_kind in {
                "traction_magnitude",
                "resultant_magnitude",
                "pressure",
            }:
                for field in ("load.magnitude", "load.direction"):
                    if field not in values:
                        missing.append(field)
            if load_kind is not None and "load.distribution" not in values:
                missing.append("load.distribution")

        unconfirmed = tuple(
            fact.field
            for fact in condition.facts
            if fact.basis == "assumption"
        )
        ready = not missing and not unconfirmed and not errors and not capabilities
        results.append(BoundaryConditionReadiness(
            bc_id=condition.bc_id,
            ready=ready,
            missing_fields=tuple(dict.fromkeys(missing)),
            unconfirmed_fields=unconfirmed,
            semantic_errors=tuple(errors),
            capability_limits=tuple(dict.fromkeys(capabilities)),
        ))

    return BoundaryDraftReadiness(
        ready=bool(results) and all(item.ready for item in results),
        conditions=tuple(results),
    )


def _field_allowed(kind: BoundaryKind, field: BoundaryField) -> bool:
    return not (
        (kind == "support" and field.startswith("load."))
        or (kind == "load" and field.startswith("support."))
    )


def _selector_is_complete(values: dict[str, object]) -> bool:
    required = {
        "whole_edge": {"selector.edge"},
        "centered_fraction": {
            "selector.edge",
            "selector.center",
            "selector.span",
        },
        "fraction_interval": {
            "selector.edge",
            "selector.start",
            "selector.end",
        },
        "coordinate_interval": {
            "selector.edge",
            "selector.start",
            "selector.end",
        },
        "centered_width": {
            "selector.edge",
            "selector.center",
            "selector.width",
        },
        "distance_from_corner": {
            "selector.edge",
            "selector.from_corner",
            "selector.offset",
            "selector.length",
        },
    }.get(values.get("selector.kind"))
    return required is not None and required.issubset(values)


def _derive_constant_traction_distribution(
    condition: BoundaryConditionDraft,
    *,
    turn_number: int,
) -> BoundaryConditionDraft:
    """Make uniformity deterministic for a constant traction on a finite region."""
    values = condition.values()
    if (
        condition.kind != "load"
        or values.get("load.kind")
        not in {"traction_vector", "traction_magnitude"}
        or not _selector_is_complete(values)
    ):
        return condition
    existing = condition.fact("load.distribution")
    if existing is not None and not (
        existing.basis == "assumption" and existing.value == "uniform"
    ):
        return condition
    quoted = next(
        (
            fact
            for field in ("load.vector", "load.magnitude", "load.kind")
            if (fact := condition.fact(field)) is not None
            and fact.source_quote
        ),
        None,
    )
    facts = {fact.field: fact for fact in condition.facts}
    facts["load.distribution"] = BoundaryFieldFact(
        field="load.distribution",
        value="uniform",
        basis="derived",
        source_turn=quoted.source_turn if quoted else turn_number,
        source_quote=quoted.source_quote if quoted else "",
        rationale=(
            "A constant traction over a complete finite selector is uniform."
        ),
    )
    return condition.model_copy(update={
        "facts": tuple(
            facts[name] for name in BOUNDARY_FIELDS if name in facts
        )
    })


def _retain_named_edge_center(
    condition: BoundaryConditionDraft,
    *,
    turn_number: int,
) -> BoundaryConditionDraft:
    """Retain a named edge center even while selector extent is unresolved."""
    if condition.fact("selector.center") is not None:
        return condition
    source = next(
        (
            fact
            for fact in condition.facts
            if fact.source_turn == turn_number
            and fact.source_quote
            and re.search(
                r"\b(?:center|centre|midpoint|middle)\b",
                fact.source_quote,
                re.IGNORECASE,
            )
        ),
        None,
    )
    if source is None or condition.fact("selector.edge") is None:
        return condition
    facts = {fact.field: fact for fact in condition.facts}
    facts["selector.center"] = BoundaryFieldFact(
        field="selector.center",
        value=0.5,
        basis="derived",
        source_turn=turn_number,
        source_quote=source.source_quote,
        rationale=(
            "A named edge center fixes the along-edge fractional position at 0.5."
        ),
    )
    return condition.model_copy(update={
        "facts": tuple(
            facts[name] for name in BOUNDARY_FIELDS if name in facts
        )
    })


def _fact_from_input(
    item: BoundaryFieldInput,
    *,
    user_message: str,
    turn_number: int,
) -> tuple[BoundaryFieldFact | None, BoundaryPatchIssue | None]:
    if item.basis in {"explicit", "derived"}:
        if not _quote_is_supported(item.source_quote, user_message):
            return None, BoundaryPatchIssue(
                code="unsupported_provenance",
                path=item.field,
                message=(
                    f"{item.basis} boundary updates require an exact quote "
                    "from the current user turn."
                ),
            )
    elif item.source_quote is not None:
        return None, BoundaryPatchIssue(
            code="unsupported_provenance",
            path=item.field,
            message="Boundary assumptions must not carry a user quote.",
        )
    try:
        value = canonical_boundary_value(item.field, item.value)
    except ValidationError as exc:
        return None, BoundaryPatchIssue(
            code="invalid_value",
            path=item.field,
            message=_validation_message(exc),
        )
    return (
        BoundaryFieldFact(
            field=item.field,
            value=value,
            basis=item.basis,
            source_turn=turn_number,
            source_quote=item.source_quote,
            rationale=item.rationale,
        ),
        None,
    )


_GENERIC_CONFIRMATIONS = {
    "yes",
    "confirm",
    "confirmed",
    "correct",
    "looks good",
    "use that",
}


def merge_boundary_patch(
    state: BoundaryDraftState,
    patch: BoundaryPatch,
    *,
    user_message: str,
    turn_number: int,
) -> BoundaryMergeResult:
    """Validate and merge BC operations while application code owns identity."""
    if turn_number < 1:
        raise ValueError("turn_number must be positive.")
    if not user_message.strip():
        raise ValueError("user_message must not be blank.")

    conditions = {item.bc_id: item for item in state.conditions}
    revisions = list(state.revisions)
    accepted: list[AcceptedBoundaryOperation] = []
    issues: list[BoundaryPatchIssue] = []
    next_support = state.next_support_number
    next_load = state.next_load_number

    for create in patch.creates:
        facts: list[BoundaryFieldFact] = []
        create_issues: list[BoundaryPatchIssue] = []
        for item in create.fields:
            if not _field_allowed(create.kind, item.field):
                create_issues.append(BoundaryPatchIssue(
                    code="invalid_field",
                    path=f"{create.local_ref}.{item.field}",
                    message=f"{item.field} is incompatible with {create.kind}.",
                ))
                continue
            fact, issue = _fact_from_input(
                item,
                user_message=user_message,
                turn_number=turn_number,
            )
            if issue is not None:
                create_issues.append(issue.model_copy(
                    update={"path": f"{create.local_ref}.{item.field}"}
                ))
            else:
                facts.append(fact)
        if create_issues:
            issues.extend(create_issues)
            continue
        if create.kind == "support":
            bc_id = f"S{next_support}"
            next_support += 1
        else:
            bc_id = f"L{next_load}"
            next_load += 1
        condition = BoundaryConditionDraft(
            bc_id=bc_id,
            kind=create.kind,
            created_turn=turn_number,
            facts=tuple(sorted(facts, key=lambda fact: BOUNDARY_FIELDS.index(fact.field))),
        )
        conditions[bc_id] = condition
        revisions.append(BoundaryRevision(
            bc_id=bc_id,
            turn=turn_number,
            action="create",
            new_value=condition.model_dump(mode="json"),
            source_quote="",
            rationale=f"Created from {create.local_ref}.",
        ))
        accepted.append(AcceptedBoundaryOperation(
            action="create",
            bc_id=bc_id,
            local_ref=create.local_ref,
        ))

    for update in patch.updates:
        condition = conditions.get(update.bc_id)
        path = f"{update.bc_id}.{update.field}"
        if condition is None:
            issues.append(BoundaryPatchIssue(
                code="missing_target",
                path=path,
                message=f"{update.bc_id} does not exist.",
            ))
            continue
        if not _field_allowed(condition.kind, update.field):
            issues.append(BoundaryPatchIssue(
                code="invalid_field",
                path=path,
                message=f"{update.field} is incompatible with {condition.kind}.",
            ))
            continue
        fact, issue = _fact_from_input(
            update,
            user_message=user_message,
            turn_number=turn_number,
        )
        if issue is not None:
            issues.append(issue.model_copy(update={"path": path}))
            continue
        facts = {item.field: item for item in condition.facts}
        previous = facts.get(update.field)
        if (
            previous is not None
            and previous.basis != "assumption"
            and fact.basis == "assumption"
            and previous.value == fact.value
        ):
            fact = previous
        facts[update.field] = fact
        conditions[update.bc_id] = condition.model_copy(update={
            "facts": tuple(
                facts[name] for name in BOUNDARY_FIELDS if name in facts
            )
        })
        revisions.append(BoundaryRevision(
            bc_id=update.bc_id,
            turn=turn_number,
            action="set",
            field=update.field,
            previous_value=previous.value if previous else None,
            new_value=fact.value,
            basis=fact.basis,
            source_quote=fact.source_quote or "",
            rationale=fact.rationale,
        ))
        accepted.append(AcceptedBoundaryOperation(
            action="set",
            bc_id=update.bc_id,
            field=update.field,
        ))

    pending = {
        (item.bc_id, item.field): item
        for item in BoundaryDraftState(
            conditions=tuple(sorted(
                conditions.values(),
                key=lambda item: _id_order(item.bc_id),
            )),
            revisions=tuple(revisions),
            next_support_number=next_support,
            next_load_number=next_load,
        ).pending_confirmations()
    }
    generic_confirmation = (
        _normalized_text(user_message) in _GENERIC_CONFIRMATIONS
    )
    for confirmation in patch.confirmations:
        path = f"{confirmation.bc_id}.{confirmation.field}"
        if not _quote_is_supported(confirmation.source_quote, user_message):
            issues.append(BoundaryPatchIssue(
                code="unsupported_provenance",
                path=path,
                message="Confirmation requires an exact current-turn quote.",
            ))
            continue
        if generic_confirmation and len(pending) != 1:
            issues.append(BoundaryPatchIssue(
                code="invalid_confirmation",
                path=path,
                message=(
                    "A generic confirmation is ambiguous when more than one "
                    "boundary assumption is pending."
                ),
            ))
            continue
        condition = conditions.get(confirmation.bc_id)
        previous = (
            condition.fact(confirmation.field) if condition is not None else None
        )
        if previous is None or previous.basis != "assumption":
            issues.append(BoundaryPatchIssue(
                code="invalid_confirmation",
                path=path,
                message="Confirmation must target an existing assumption.",
            ))
            continue
        confirmed = previous.model_copy(update={
            "basis": "confirmed",
            "source_turn": turn_number,
            "source_quote": confirmation.source_quote,
            "rationale": confirmation.rationale,
        })
        facts = {item.field: item for item in condition.facts}
        facts[confirmation.field] = confirmed
        conditions[confirmation.bc_id] = condition.model_copy(update={
            "facts": tuple(
                facts[name] for name in BOUNDARY_FIELDS if name in facts
            )
        })
        revisions.append(BoundaryRevision(
            bc_id=confirmation.bc_id,
            turn=turn_number,
            action="confirm",
            field=confirmation.field,
            previous_value=previous.value,
            new_value=previous.value,
            basis="confirmed",
            source_quote=confirmation.source_quote,
            rationale=confirmation.rationale,
        ))
        accepted.append(AcceptedBoundaryOperation(
            action="confirm",
            bc_id=confirmation.bc_id,
            field=confirmation.field,
        ))

    for delete in patch.deletes:
        path = delete.bc_id
        if not _quote_is_supported(delete.source_quote, user_message):
            issues.append(BoundaryPatchIssue(
                code="unsupported_provenance",
                path=path,
                message="Delete requires an exact current-turn quote.",
            ))
            continue
        previous = conditions.pop(delete.bc_id, None)
        if previous is None:
            issues.append(BoundaryPatchIssue(
                code="missing_target",
                path=path,
                message=f"{delete.bc_id} does not exist.",
            ))
            continue
        revisions.append(BoundaryRevision(
            bc_id=delete.bc_id,
            turn=turn_number,
            action="delete",
            previous_value=previous.model_dump(mode="json"),
            source_quote=delete.source_quote,
            rationale=delete.rationale,
        ))
        accepted.append(AcceptedBoundaryOperation(
            action="delete",
            bc_id=delete.bc_id,
        ))

    conditions = {
        bc_id: _derive_constant_traction_distribution(
            _retain_named_edge_center(
                condition,
                turn_number=turn_number,
            ),
            turn_number=turn_number,
        )
        for bc_id, condition in conditions.items()
    }
    merged = BoundaryDraftState(
        conditions=tuple(sorted(
            conditions.values(),
            key=lambda item: _id_order(item.bc_id),
        )),
        revisions=tuple(revisions),
        next_support_number=next_support,
        next_load_number=next_load,
    )
    return BoundaryMergeResult(
        state=merged,
        accepted=tuple(accepted),
        issues=tuple(issues),
    )
