"""Typed, model-independent core for conversational problem formulation.

The language model proposes a small patch on each turn. This module keeps the
durable draft, provenance, merge rules, readiness decision, and strict final
conversion in deterministic application code.
"""

from __future__ import annotations

from typing import Annotated, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from agentic.boundary_draft import (
    BOUNDARY_FIELDS,
    BoundaryConditionDraft,
    BoundaryDraftState,
    BoundaryFieldFact,
    BoundaryMergeResult,
    BoundaryPatch,
    BoundaryRevision,
    assess_boundary_state,
    canonical_boundary_value,
    merge_boundary_patch,
)
from agentic.intent import (
    Bounds2D,
    FixedSupportIntent,
    FiniteNumber,
    MechanismSpringIntent,
    MeshPreferences,
    OpenFraction,
    OptimizationPreferences,
    Point2D,
    PositiveFiniteNumber,
    PositiveInt,
    ProblemIntent,
    RectangularDomainIntent,
    TractionIntent,
    Vector2D,
)
from agentic.load_semantics import (
    BoundaryLoadState,
    resolve_boundary_load_state,
)
from agentic.mechanical_units import (
    ForceUnitName,
    LengthUnitName,
    MechanicalUnitContext,
    StressUnitName,
)


DraftPath = Literal[
    "problem_type",
    "domain.bounds",
    "domain.origin",
    "domain.width",
    "domain.height",
    "material.young_modulus",
    "material.poisson_ratio",
    "units.length",
    "units.force",
    "units.stress",
    "supports",
    "support_edges",
    "tractions",
    "body_force",
    "volume_fraction",
    "compliance_bound",
    "input_spring",
    "output_spring",
    "mesh.divisions",
    "mesh.long_short_divisions",
    "mesh.cell_type",
    "optimization.filter_radius",
    "optimization.max_iter",
]
FactBasis = Literal["explicit", "derived", "assumption", "confirmed"]
DeclaredTurnState = Literal["gathering", "reviewing", "ready", "unsupported"]
SessionStatus = Literal[
    "gathering",
    "repair_needed",
    "unsupported",
    "ready_for_review",
]

DRAFT_PATHS: tuple[DraftPath, ...] = (
    "problem_type",
    "domain.bounds",
    "domain.origin",
    "domain.width",
    "domain.height",
    "material.young_modulus",
    "material.poisson_ratio",
    "units.length",
    "units.force",
    "units.stress",
    "supports",
    "support_edges",
    "tractions",
    "body_force",
    "volume_fraction",
    "compliance_bound",
    "input_spring",
    "output_spring",
    "mesh.divisions",
    "mesh.long_short_divisions",
    "mesh.cell_type",
    "optimization.filter_radius",
    "optimization.max_iter",
)
COMMON_REQUIRED_PATHS: tuple[DraftPath, ...] = (
    "problem_type",
    "material.young_modulus",
    "material.poisson_ratio",
    "volume_fraction",
)
MECHANISM_REQUIRED_PATHS: tuple[DraftPath, ...] = (
    "compliance_bound",
    "input_spring",
    "output_spring",
)


class StrictFormulationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DraftUpdate(StrictFormulationModel):
    """One model-proposed fact update for the current user turn."""

    path: DraftPath
    value: JsonValue
    basis: FactBasis
    source_quote: str | None = None
    rationale: str

    @field_validator("source_quote")
    @classmethod
    def _strip_quote(cls, value):
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("rationale")
    @classmethod
    def _nonblank_rationale(cls, value):
        if not value.strip():
            raise ValueError("rationale must not be blank.")
        return value.strip()


class DraftClear(StrictFormulationModel):
    """Remove a previously stored fact only when the current user supports it."""

    path: DraftPath
    source_quote: str
    rationale: str

    @field_validator("source_quote", "rationale")
    @classmethod
    def _nonblank_text(cls, value):
        if not value.strip():
            raise ValueError("text must not be blank.")
        return value.strip()


class FormulationTurn(StrictFormulationModel):
    """Small structured response requested from the conversational model."""

    assistant_message: str
    updates: tuple[DraftUpdate, ...] = ()
    clears: tuple[DraftClear, ...] = ()
    boundary_patch: BoundaryPatch = Field(default_factory=BoundaryPatch)
    questions: Annotated[tuple[str, ...], Field(max_length=3)] = ()
    declared_state: DeclaredTurnState = "gathering"
    unsupported_features: tuple[str, ...] = ()

    @field_validator("assistant_message")
    @classmethod
    def _nonblank_message(cls, value):
        if not value.strip():
            raise ValueError("assistant_message must not be blank.")
        return value.strip()

    @field_validator("questions", "unsupported_features")
    @classmethod
    def _nonblank_items(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError("items must not be blank.")
        return tuple(value.strip() for value in values)

    @model_validator(mode="after")
    def _consistent_patch(self):
        update_paths = [item.path for item in self.updates]
        clear_paths = [item.path for item in self.clears]
        if len(update_paths) != len(set(update_paths)):
            raise ValueError("a turn may update each draft path at most once.")
        if len(clear_paths) != len(set(clear_paths)):
            raise ValueError("a turn may clear each draft path at most once.")
        overlap = set(update_paths) & set(clear_paths)
        if overlap:
            raise ValueError(
                "a turn cannot update and clear the same path: "
                + ", ".join(sorted(overlap))
            )
        if self.declared_state == "unsupported" and not self.unsupported_features:
            raise ValueError(
                "unsupported turns must identify at least one unsupported feature."
            )
        return self


class DraftFact(StrictFormulationModel):
    path: DraftPath
    value: JsonValue
    basis: FactBasis
    source_turn: int = Field(ge=1)
    source_quote: str | None = None
    rationale: str


class DraftRevision(StrictFormulationModel):
    path: DraftPath
    turn: int = Field(ge=1)
    action: Literal["set", "clear"]
    previous_value: JsonValue | None = None
    new_value: JsonValue | None = None
    basis: FactBasis | None = None
    source_quote: str
    rationale: str


class ProblemDraft(StrictFormulationModel):
    """Current canonical facts plus an inspectable revision history."""

    facts: tuple[DraftFact, ...] = ()
    revisions: tuple[DraftRevision, ...] = ()
    boundary_state: BoundaryDraftState = Field(
        default_factory=BoundaryDraftState
    )
    turn_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _unique_ordered_facts(self):
        paths = [fact.path for fact in self.facts]
        if len(paths) != len(set(paths)):
            raise ValueError("draft facts must have unique paths.")
        expected = sorted(paths, key=DRAFT_PATHS.index)
        if paths != expected:
            raise ValueError("draft facts must use canonical path ordering.")
        return self

    def fact(self, path: DraftPath) -> DraftFact | None:
        return next((fact for fact in self.facts if fact.path == path), None)

    def values(self) -> dict[DraftPath, JsonValue]:
        return {fact.path: fact.value for fact in self.facts}


class PatchIssue(StrictFormulationModel):
    code: Literal[
        "duplicate_target",
        "invalid_field",
        "invalid_value",
        "missing_target",
        "unsupported_provenance",
        "invalid_confirmation",
    ]
    path: str
    message: str


class DraftMergeResult(StrictFormulationModel):
    draft: ProblemDraft
    accepted_paths: tuple[DraftPath, ...]
    boundary_merge: BoundaryMergeResult
    issues: tuple[PatchIssue, ...]


class DraftReadiness(StrictFormulationModel):
    ready: bool
    missing_fields: tuple[str, ...]
    unconfirmed_fields: tuple[str, ...]
    semantic_errors: tuple[str, ...]


class MechanicalUnitReadiness(StrictFormulationModel):
    """Readiness of the optional Package 2 mechanical-unit subdraft."""

    ready: bool
    missing_fields: tuple[DraftPath, ...]
    unconfirmed_fields: tuple[DraftPath, ...]
    semantic_errors: tuple[str, ...]
    context: MechanicalUnitContext | None = None


class SemanticBoundaryReadiness(StrictFormulationModel):
    """Package 2 view over unit facts and first-class boundary entities."""

    semantic_ready: bool
    execution_ready: bool
    units: MechanicalUnitReadiness
    boundary_loads: BoundaryLoadState | None = None


class ConversationMessage(StrictFormulationModel):
    turn: int = Field(ge=1)
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _nonblank_content(cls, value):
        if not value.strip():
            raise ValueError("content must not be blank.")
        return value.strip()


class FormulationRepair(StrictFormulationModel):
    """Deterministic feedback for a corrected attempt on the same user turn."""

    attempt: int = Field(ge=2, le=3)
    rejected_turn: FormulationTurn
    issues: Annotated[tuple[PatchIssue, ...], Field(min_length=1)]


class FormulationModelState(StrictFormulationModel):
    """Opaque continuation owned by a model adapter, never by solver logic."""

    adapter: str
    continuation_id: str

    @field_validator("adapter", "continuation_id")
    @classmethod
    def _nonblank_state(cls, value):
        if not value.strip():
            raise ValueError("model-state values must not be blank.")
        return value.strip()


class FormulationRequest(StrictFormulationModel):
    turn_number: int = Field(ge=1)
    user_message: str
    draft: ProblemDraft
    history: tuple[ConversationMessage, ...]
    model_state: FormulationModelState | None = None
    repair: FormulationRepair | None = None


class FormulationAgentResponse(StrictFormulationModel):
    turn: FormulationTurn
    model_state: FormulationModelState | None = None


class FormulationSession(StrictFormulationModel):
    draft: ProblemDraft = Field(default_factory=ProblemDraft)
    messages: tuple[ConversationMessage, ...] = ()
    model_state: FormulationModelState | None = None
    status: SessionStatus = "gathering"
    questions: tuple[str, ...] = ()
    unsupported_features: tuple[str, ...] = ()


class FormulationStep(StrictFormulationModel):
    session: FormulationSession
    turn: FormulationTurn
    merge: DraftMergeResult
    readiness: DraftReadiness
    intent: ProblemIntent | None = None
    finalized_draft: ProblemDraft | None = None


class FormulationAgent(Protocol):
    def formulate(
        self,
        request: FormulationRequest,
    ) -> FormulationAgentResponse | FormulationTurn: ...


class DraftNotReadyError(ValueError):
    """The partial draft cannot yet cross into strict solver intent."""

    def __init__(self, readiness: DraftReadiness):
        self.readiness = readiness
        parts = []
        if readiness.missing_fields:
            parts.append("missing: " + ", ".join(readiness.missing_fields))
        if readiness.unconfirmed_fields:
            parts.append(
                "unconfirmed: " + ", ".join(readiness.unconfirmed_fields)
            )
        if readiness.semantic_errors:
            parts.append("invalid: " + "; ".join(readiness.semantic_errors))
        super().__init__("Draft is not ready (" + " | ".join(parts) + ").")


SupportList = Annotated[list[FixedSupportIntent], Field(min_length=1)]
TractionList = list[TractionIntent]
MeshDivisions = tuple[PositiveInt, PositiveInt]
CellType = Literal["quadrilateral", "triangle"]
RectangleEdge = Literal["left", "right", "bottom", "top"]
SupportEdges = Annotated[list[RectangleEdge], Field(min_length=1)]
ProblemType = Literal["minimize_compliance", "compliant_mechanism"]
PoissonRatio = Annotated[FiniteNumber, Field(gt=-1.0, lt=0.5)]

_FIELD_ADAPTERS: dict[DraftPath, TypeAdapter] = {
    "problem_type": TypeAdapter(ProblemType),
    "domain.bounds": TypeAdapter(Bounds2D),
    "domain.origin": TypeAdapter(Point2D),
    "domain.width": TypeAdapter(PositiveFiniteNumber),
    "domain.height": TypeAdapter(PositiveFiniteNumber),
    "material.young_modulus": TypeAdapter(PositiveFiniteNumber),
    "material.poisson_ratio": TypeAdapter(PoissonRatio),
    "units.length": TypeAdapter(LengthUnitName),
    "units.force": TypeAdapter(ForceUnitName),
    "units.stress": TypeAdapter(StressUnitName),
    "supports": TypeAdapter(SupportList),
    "support_edges": TypeAdapter(SupportEdges),
    "tractions": TypeAdapter(TractionList),
    "body_force": TypeAdapter(Vector2D),
    "volume_fraction": TypeAdapter(OpenFraction),
    "compliance_bound": TypeAdapter(PositiveFiniteNumber),
    "input_spring": TypeAdapter(MechanismSpringIntent),
    "output_spring": TypeAdapter(MechanismSpringIntent),
    "mesh.divisions": TypeAdapter(MeshDivisions),
    "mesh.long_short_divisions": TypeAdapter(MeshDivisions),
    "mesh.cell_type": TypeAdapter(CellType),
    "optimization.filter_radius": TypeAdapter(PositiveFiniteNumber),
    "optimization.max_iter": TypeAdapter(PositiveInt),
}
_PROBLEM_ADAPTER = TypeAdapter(ProblemIntent)


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _quote_is_supported(quote: str | None, user_message: str) -> bool:
    if quote is None:
        return False
    return _normalized_text(quote) in _normalized_text(user_message)


def _canonical_value(path: DraftPath, value: JsonValue) -> JsonValue:
    adapter = _FIELD_ADAPTERS[path]
    validated = adapter.validate_python(value)
    if path == "domain.bounds":
        validated = RectangularDomainIntent(bounds=validated).bounds
    return adapter.dump_python(validated, mode="json")


def _validation_message(error: ValidationError) -> str:
    details = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        prefix = f"{location}: " if location else ""
        details.append(prefix + item["msg"])
    return "; ".join(details)


def merge_formulation_turn(
    draft: ProblemDraft,
    turn: FormulationTurn,
    *,
    user_message: str,
    turn_number: int,
) -> DraftMergeResult:
    """Validate provenance and values, then apply accepted changes atomically."""
    if turn_number != draft.turn_count + 1:
        raise ValueError(
            f"turn_number must be {draft.turn_count + 1}, got {turn_number}."
        )
    if not user_message.strip():
        raise ValueError("user_message must not be blank.")

    facts = {fact.path: fact for fact in draft.facts}
    revisions = list(draft.revisions)
    accepted: list[DraftPath] = []
    issues: list[PatchIssue] = []

    for clear in turn.clears:
        if not _quote_is_supported(clear.source_quote, user_message):
            issues.append(
                PatchIssue(
                    code="unsupported_provenance",
                    path=clear.path,
                    message=(
                        "The clear operation is not supported by an exact quote "
                        "from the current user turn."
                    ),
                )
            )
            continue
        previous = facts.pop(clear.path, None)
        if previous is not None:
            revisions.append(
                DraftRevision(
                    path=clear.path,
                    turn=turn_number,
                    action="clear",
                    previous_value=previous.value,
                    source_quote=clear.source_quote,
                    rationale=clear.rationale,
                )
            )
        accepted.append(clear.path)

    for update in turn.updates:
        previous = facts.get(update.path)
        if update.basis in ("explicit", "derived", "confirmed"):
            if not _quote_is_supported(update.source_quote, user_message):
                issues.append(
                    PatchIssue(
                        code="unsupported_provenance",
                        path=update.path,
                        message=(
                            f"{update.basis} updates require an exact quote from "
                            "the current user turn."
                        ),
                    )
                )
                continue
        if update.basis == "assumption" and update.source_quote is not None:
            issues.append(
                PatchIssue(
                    code="unsupported_provenance",
                    path=update.path,
                    message="Assumptions must not be attributed to a user quote.",
                )
            )
            continue

        try:
            value = _canonical_value(update.path, update.value)
        except ValidationError as exc:
            issues.append(
                PatchIssue(
                    code="invalid_value",
                    path=update.path,
                    message=_validation_message(exc),
                )
            )
            continue

        if update.basis == "confirmed":
            if previous is None or previous.basis != "assumption":
                issues.append(
                    PatchIssue(
                        code="invalid_confirmation",
                        path=update.path,
                        message=(
                            "A confirmation must match an existing unconfirmed "
                            "assumption."
                        ),
                    )
                )
                continue
            if previous.value != value:
                issues.append(
                    PatchIssue(
                        code="invalid_confirmation",
                        path=update.path,
                        message=(
                            "A confirmation cannot change the assumed value; use "
                            "an explicit correction instead."
                        ),
                    )
                )
                continue

        facts[update.path] = DraftFact(
            path=update.path,
            value=value,
            basis=update.basis,
            source_turn=turn_number,
            source_quote=update.source_quote,
            rationale=update.rationale,
        )
        revisions.append(
            DraftRevision(
                path=update.path,
                turn=turn_number,
                action="set",
                previous_value=previous.value if previous else None,
                new_value=value,
                basis=update.basis,
                source_quote=update.source_quote or "",
                rationale=update.rationale,
            )
        )
        accepted.append(update.path)

    ordered_facts = tuple(
        facts[path] for path in DRAFT_PATHS if path in facts
    )
    boundary_merge = merge_boundary_patch(
        draft.boundary_state,
        turn.boundary_patch,
        user_message=user_message,
        turn_number=turn_number,
    )
    issues.extend(
        PatchIssue(
            code=issue.code,
            path=issue.path,
            message=issue.message,
        )
        for issue in boundary_merge.issues
    )
    merged = ProblemDraft(
        facts=ordered_facts,
        revisions=tuple(revisions),
        boundary_state=boundary_merge.state,
        turn_count=turn_number,
    )
    return DraftMergeResult(
        draft=merged,
        accepted_paths=tuple(accepted),
        boundary_merge=boundary_merge,
        issues=tuple(issues),
    )


def _resolved_bounds(
    values: dict[DraftPath, JsonValue],
) -> tuple[list[list[int | float]] | None, list[str]]:
    stored = values.get("domain.bounds")
    component_paths = (
        "domain.origin",
        "domain.width",
        "domain.height",
    )
    has_all_components = all(path in values for path in component_paths)
    derived = None
    if has_all_components:
        origin = values["domain.origin"]
        width = values["domain.width"]
        height = values["domain.height"]
        derived = [
            [origin[0], origin[1]],
            [origin[0] + width, origin[1] + height],
        ]

    errors = []
    if stored is not None and derived is not None and stored != derived:
        errors.append(
            "domain.bounds conflicts with domain.origin/width/height."
        )
    return stored or derived, errors


_MECHANICAL_UNIT_PATHS: tuple[DraftPath, ...] = (
    "units.length",
    "units.force",
    "units.stress",
)


def assess_mechanical_units(draft: ProblemDraft) -> MechanicalUnitReadiness:
    """Build a context only from complete, confirmed dimensional facts.

    Unit facts are not required by the legacy live finalizer during the staged
    migration.  Once any unit fact exists, however, this function reports the
    complete and independently usable state of that unit subdraft.
    """
    facts = {
        path: draft.fact(path)
        for path in _MECHANICAL_UNIT_PATHS
    }
    missing = tuple(path for path, fact in facts.items() if fact is None)
    unconfirmed = tuple(
        path
        for path, fact in facts.items()
        if fact is not None and fact.basis == "assumption"
    )
    errors: list[str] = []
    context = None
    if not missing and not unconfirmed:
        try:
            context = MechanicalUnitContext(
                length_unit=facts["units.length"].value,
                force_unit=facts["units.force"].value,
                stress_unit=facts["units.stress"].value,
            )
        except (ValidationError, ValueError) as exc:
            errors.append(
                _validation_message(exc)
                if isinstance(exc, ValidationError)
                else str(exc)
            )
    return MechanicalUnitReadiness(
        ready=not missing and not unconfirmed and not errors,
        missing_fields=missing,
        unconfirmed_fields=unconfirmed,
        semantic_errors=tuple(errors),
        context=context,
    )


def assess_semantic_boundary_loads(
    draft: ProblemDraft,
) -> SemanticBoundaryReadiness:
    """Resolve Package 2 state without changing current live finalization."""
    units = assess_mechanical_units(draft)
    if units.context is None:
        return SemanticBoundaryReadiness(
            semantic_ready=False,
            execution_ready=False,
            units=units,
        )
    loads = resolve_boundary_load_state(draft.boundary_state, units.context)
    return SemanticBoundaryReadiness(
        semantic_ready=loads.semantic_ready,
        execution_ready=loads.execution_ready,
        units=units,
        boundary_loads=loads,
    )


def _support_for_edge(
    edge: str,
    bounds: list[list[int | float]],
) -> dict:
    (x0, y0), (x1, y1) = bounds
    axis, value = {
        "left": ("x", x0),
        "right": ("x", x1),
        "bottom": ("y", y0),
        "top": ("y", y1),
    }[edge]
    return {
        "region": {
            "op": "plane",
            "axis": axis,
            "value": value,
        }
    }


def migrate_legacy_boundary_facts(draft: ProblemDraft) -> ProblemDraft:
    """Populate first-class BC state from current monolithic draft facts.

    Migration is explicit during Work Package 1. Legacy facts remain untouched
    and therefore remain authoritative for current finalization until a later
    work package deliberately switches the live path.
    """
    if draft.boundary_state.conditions:
        raise ValueError("boundary_state is already populated.")

    conditions: list[BoundaryConditionDraft] = []
    revisions: list[BoundaryRevision] = []
    support_number = 1
    load_number = 1

    def make_fact(
        source: DraftFact,
        field,
        value,
        rationale: str,
    ) -> BoundaryFieldFact:
        return BoundaryFieldFact(
            field=field,
            value=canonical_boundary_value(field, value),
            basis=source.basis,
            source_turn=source.source_turn,
            source_quote=source.source_quote,
            rationale=rationale,
        )

    def add_condition(
        *,
        bc_id: str,
        kind: str,
        source: DraftFact,
        field_values: list[tuple],
    ) -> None:
        facts = tuple(sorted(
            (
                make_fact(
                    source,
                    field,
                    value,
                    f"Migrated from legacy {source.path}: {source.rationale}",
                )
                for field, value in field_values
            ),
            key=lambda fact: BOUNDARY_FIELDS.index(fact.field),
        ))
        condition = BoundaryConditionDraft(
            bc_id=bc_id,
            kind=kind,
            created_turn=source.source_turn,
            facts=facts,
        )
        conditions.append(condition)
        revisions.append(BoundaryRevision(
            bc_id=bc_id,
            turn=source.source_turn,
            action="create",
            new_value=condition.model_dump(mode="json"),
            source_quote=source.source_quote or "",
            rationale=f"Migrated legacy {source.path}.",
        ))

    supports = draft.fact("supports")
    if supports is not None:
        for support in supports.value:
            add_condition(
                bc_id=f"S{support_number}",
                kind="support",
                source=supports,
                field_values=[
                    ("support.kind", "fixed_all"),
                    ("selector.kind", "expert_region"),
                    ("selector.region", support["region"]),
                ],
            )
            support_number += 1

    support_edges = draft.fact("support_edges")
    if support_edges is not None:
        for edge in support_edges.value:
            add_condition(
                bc_id=f"S{support_number}",
                kind="support",
                source=support_edges,
                field_values=[
                    ("support.kind", "fixed_all"),
                    ("selector.kind", "whole_edge"),
                    ("selector.edge", edge),
                ],
            )
            support_number += 1

    tractions = draft.fact("tractions")
    if tractions is not None:
        for traction in tractions.value:
            selector = traction.get("edge_segment")
            if selector:
                selector_fields = [
                    ("selector.kind", "centered_fraction"),
                    ("selector.edge", selector["edge"]),
                    ("selector.center", selector["center_fraction"]),
                    ("selector.span", selector["span_fraction"]),
                ]
            else:
                selector_fields = [
                    ("selector.kind", "expert_region"),
                    ("selector.region", traction["region"]),
                ]
            add_condition(
                bc_id=f"L{load_number}",
                kind="load",
                source=tractions,
                field_values=[
                    ("load.kind", "traction_vector"),
                    ("load.vector", traction["vector"]),
                    ("load.distribution", "uniform"),
                    *selector_fields,
                ],
            )
            load_number += 1

    state = BoundaryDraftState(
        conditions=tuple(conditions),
        revisions=tuple(revisions),
        next_support_number=support_number,
        next_load_number=load_number,
    )
    return draft.model_copy(update={"boundary_state": state})


def _resolved_mesh_divisions(
    values: dict[DraftPath, JsonValue],
    bounds: list[list[int | float]] | None,
) -> tuple[JsonValue | None, list[str]]:
    direct = values.get("mesh.divisions")
    relative = values.get("mesh.long_short_divisions")
    if relative is None:
        return direct, []
    if bounds is None:
        return direct, []

    (x0, y0), (x1, y1) = bounds
    width = float(x1) - float(x0)
    height = float(y1) - float(y0)
    long_count, short_count = relative
    if width > height:
        mapped = [long_count, short_count]
    elif height > width:
        mapped = [short_count, long_count]
    elif long_count == short_count:
        mapped = [long_count, short_count]
    else:
        return direct, [
            "mesh.long_short_divisions is ambiguous for a square domain; "
            "specify x/y divisions."
        ]

    if direct is not None and direct != mapped:
        return direct, [
            "mesh.divisions conflicts with mesh.long_short_divisions."
        ]
    return direct or mapped, []


def _intent_payload(
    draft: ProblemDraft,
    *,
    bounds: list[list[int | float]] | None = None,
) -> dict:
    values = draft.values()
    if bounds is None:
        bounds, _ = _resolved_bounds(values)
    payload: dict = {}
    if "problem_type" in values:
        payload["problem_type"] = values["problem_type"]
    if bounds is not None:
        payload["domain"] = {"bounds": bounds}
    material = {}
    if "material.young_modulus" in values:
        material["young_modulus"] = values["material.young_modulus"]
    if "material.poisson_ratio" in values:
        material["poisson_ratio"] = values["material.poisson_ratio"]
    if material:
        payload["material"] = material
    supports = list(values.get("supports", []))
    if bounds is not None:
        supports.extend(
            _support_for_edge(edge, bounds)
            for edge in values.get("support_edges", [])
        )
    if supports:
        payload["supports"] = supports

    for path in (
        "tractions",
        "body_force",
        "volume_fraction",
        "compliance_bound",
        "input_spring",
        "output_spring",
    ):
        if path in values:
            payload[path] = values[path]

    mesh = {}
    divisions, _ = _resolved_mesh_divisions(values, bounds)
    if divisions is not None:
        mesh["divisions"] = divisions
    if "mesh.cell_type" in values:
        mesh["cell_type"] = values["mesh.cell_type"]
    if mesh:
        payload["mesh"] = MeshPreferences.model_validate(mesh).model_dump(
            mode="json"
        )

    optimization = {}
    if "optimization.filter_radius" in values:
        optimization["filter_radius"] = values["optimization.filter_radius"]
    if "optimization.max_iter" in values:
        optimization["max_iter"] = values["optimization.max_iter"]
    if optimization:
        payload["optimization"] = OptimizationPreferences.model_validate(
            optimization
        ).model_dump(mode="json")
    return payload


def assess_draft(draft: ProblemDraft) -> DraftReadiness:
    """Determine readiness independently of the model's declared turn state."""
    values = draft.values()
    missing = [path for path in COMMON_REQUIRED_PATHS if path not in values]
    bounds, semantic_errors = _resolved_bounds(values)
    if bounds is None:
        missing.append("domain.bounds")
    first_class = bool(draft.boundary_state.conditions)
    if first_class:
        boundary_readiness = assess_boundary_state(draft.boundary_state)
        if not any(
            item.kind == "support"
            for item in draft.boundary_state.conditions
        ):
            missing.append("supports")
        for condition in boundary_readiness.conditions:
            missing.extend(
                f"{condition.bc_id}.{field}"
                for field in condition.missing_fields
            )
            semantic_errors.extend(
                f"{condition.bc_id}: {message}"
                for message in (
                    *condition.semantic_errors,
                    *condition.capability_limits,
                )
            )
    elif "supports" not in values and "support_edges" not in values:
        missing.append("supports")

    problem_type = values.get("problem_type")
    if problem_type == "compliant_mechanism":
        missing.extend(
            path for path in MECHANISM_REQUIRED_PATHS if path not in values
        )

    tractions = values.get("tractions", [])
    boundary_loads = [
        item
        for item in draft.boundary_state.conditions
        if item.kind == "load"
    ]
    body_force = values.get("body_force", [0.0, 0.0])
    has_body_force = bool(
        isinstance(body_force, list)
        and len(body_force) == 2
        and any(float(component) != 0.0 for component in body_force)
    )
    if not tractions and not boundary_loads and not has_body_force:
        missing.append("external_load")

    unconfirmed = [
        fact.path for fact in draft.facts if fact.basis == "assumption"
    ]
    if first_class:
        unconfirmed.extend(
            f"{condition.bc_id}.{fact.field}"
            for condition in draft.boundary_state.conditions
            for fact in condition.facts
            if fact.basis == "assumption"
        )
        if boundary_loads:
            unit_readiness = assess_mechanical_units(draft)
            missing.extend(unit_readiness.missing_fields)
            unconfirmed.extend(unit_readiness.unconfirmed_fields)
            semantic_errors.extend(unit_readiness.semantic_errors)
            if unit_readiness.context is not None:
                load_state = resolve_boundary_load_state(
                    draft.boundary_state,
                    unit_readiness.context,
                )
                for condition, resolution in zip(
                    boundary_loads,
                    load_state.loads,
                    strict=True,
                ):
                    semantic_errors.extend(
                        f"{condition.bc_id}.{issue.field}: {issue.message}"
                        for issue in resolution.issues
                    )
    _, mesh_errors = _resolved_mesh_divisions(values, bounds)
    semantic_errors.extend(mesh_errors)
    if not first_class and not missing and not unconfirmed:
        try:
            _PROBLEM_ADAPTER.validate_python(
                _intent_payload(draft, bounds=bounds)
            )
        except ValidationError as exc:
            semantic_errors.append(_validation_message(exc))

    return DraftReadiness(
        ready=not missing and not unconfirmed and not semantic_errors,
        missing_fields=tuple(dict.fromkeys(missing)),
        unconfirmed_fields=tuple(dict.fromkeys(unconfirmed)),
        semantic_errors=tuple(semantic_errors),
    )


def finalize_draft(draft: ProblemDraft) -> ProblemIntent:
    """Cross from conversational state into the existing strict intent contract."""
    readiness = assess_draft(draft)
    if not readiness.ready:
        raise DraftNotReadyError(readiness)
    return _PROBLEM_ADAPTER.validate_python(_intent_payload(draft))


class ConversationFormulator:
    """Advance real user/assistant turns while deterministic state remains canonical."""

    def __init__(
        self,
        agent: FormulationAgent,
        *,
        max_repair_attempts: int = 1,
    ):
        if not 0 <= max_repair_attempts <= 2:
            raise ValueError("max_repair_attempts must be between 0 and 2.")
        self._agent = agent
        self._max_repair_attempts = max_repair_attempts

    def start(self, user_message: str) -> FormulationStep:
        return self.advance(FormulationSession(), user_message)

    def advance(
        self,
        session: FormulationSession,
        user_message: str,
    ) -> FormulationStep:
        message = user_message.strip()
        if not message:
            raise ValueError("user_message must not be blank.")
        starting_draft = session.draft
        if (
            getattr(self._agent, "first_class_boundary_patches", False)
            and not starting_draft.boundary_state.conditions
            and any(
                starting_draft.fact(path) is not None
                for path in ("supports", "support_edges", "tractions")
            )
        ):
            starting_draft = migrate_legacy_boundary_facts(starting_draft)
        turn_number = starting_draft.turn_count + 1
        model_state = session.model_state
        repair = None
        for attempt in range(1, self._max_repair_attempts + 2):
            request = FormulationRequest(
                turn_number=turn_number,
                user_message=message,
                draft=starting_draft,
                history=session.messages,
                model_state=model_state,
                repair=repair,
            )
            raw_response = self._agent.formulate(request)
            response = (
                raw_response
                if isinstance(raw_response, FormulationAgentResponse)
                else FormulationAgentResponse(
                    turn=raw_response,
                    model_state=model_state,
                )
            )
            turn = response.turn
            model_state = response.model_state
            merge = merge_formulation_turn(
                starting_draft,
                turn,
                user_message=message,
                turn_number=turn_number,
            )
            if not merge.issues or attempt > self._max_repair_attempts:
                break
            repair = FormulationRepair(
                attempt=attempt + 1,
                rejected_turn=turn,
                issues=merge.issues,
            )

        readiness = assess_draft(merge.draft)

        intent = None
        finalized_draft = None
        if merge.issues:
            status: SessionStatus = "repair_needed"
        elif turn.declared_state == "unsupported":
            status = "unsupported"
        elif readiness.ready:
            status = "ready_for_review"
            finalized_draft = merge.draft
            if not merge.draft.boundary_state.conditions:
                intent = finalize_draft(merge.draft)
        else:
            status = "gathering"

        messages = (
            *session.messages,
            ConversationMessage(
                turn=turn_number,
                role="user",
                content=message,
            ),
            ConversationMessage(
                turn=turn_number,
                role="assistant",
                content=turn.assistant_message,
            ),
        )
        next_session = FormulationSession(
            draft=merge.draft,
            messages=messages,
            model_state=model_state,
            status=status,
            questions=turn.questions,
            unsupported_features=turn.unsupported_features,
        )
        return FormulationStep(
            session=next_session,
            turn=turn,
            merge=merge,
            readiness=readiness,
            intent=intent,
            finalized_draft=finalized_draft,
        )
