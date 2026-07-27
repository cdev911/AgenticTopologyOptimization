"""Typed, model-independent foundation for conversational problem formulation.

The language model will eventually propose a small patch on each turn.  This
module keeps the durable draft, provenance, merge rules, readiness decision, and
strict final conversion in deterministic application code.
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

from agentic.intent import (
    Bounds2D,
    FixedSupportIntent,
    FiniteNumber,
    MechanismSpringIntent,
    MeshPreferences,
    OpenFraction,
    OptimizationPreferences,
    PositiveFiniteNumber,
    PositiveInt,
    ProblemIntent,
    RectangularDomainIntent,
    TractionIntent,
    Vector2D,
)


DraftPath = Literal[
    "problem_type",
    "domain.bounds",
    "material.young_modulus",
    "material.poisson_ratio",
    "supports",
    "tractions",
    "body_force",
    "volume_fraction",
    "compliance_bound",
    "input_spring",
    "output_spring",
    "mesh.divisions",
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
    "material.young_modulus",
    "material.poisson_ratio",
    "supports",
    "tractions",
    "body_force",
    "volume_fraction",
    "compliance_bound",
    "input_spring",
    "output_spring",
    "mesh.divisions",
    "mesh.cell_type",
    "optimization.filter_radius",
    "optimization.max_iter",
)
COMMON_REQUIRED_PATHS: tuple[DraftPath, ...] = (
    "problem_type",
    "domain.bounds",
    "material.young_modulus",
    "material.poisson_ratio",
    "supports",
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
        if self.declared_state != "unsupported" and self.unsupported_features:
            raise ValueError(
                "unsupported_features are allowed only for an unsupported turn."
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
        "invalid_value",
        "unsupported_provenance",
        "invalid_confirmation",
    ]
    path: DraftPath
    message: str


class DraftMergeResult(StrictFormulationModel):
    draft: ProblemDraft
    accepted_paths: tuple[DraftPath, ...]
    issues: tuple[PatchIssue, ...]


class DraftReadiness(StrictFormulationModel):
    ready: bool
    missing_fields: tuple[str, ...]
    unconfirmed_fields: tuple[DraftPath, ...]
    semantic_errors: tuple[str, ...]


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


class FormulationRequest(StrictFormulationModel):
    turn_number: int = Field(ge=1)
    user_message: str
    draft: ProblemDraft
    history: tuple[ConversationMessage, ...]


class FormulationSession(StrictFormulationModel):
    draft: ProblemDraft = Field(default_factory=ProblemDraft)
    messages: tuple[ConversationMessage, ...] = ()
    status: SessionStatus = "gathering"
    questions: tuple[str, ...] = ()
    unsupported_features: tuple[str, ...] = ()


class FormulationStep(StrictFormulationModel):
    session: FormulationSession
    turn: FormulationTurn
    merge: DraftMergeResult
    readiness: DraftReadiness
    intent: ProblemIntent | None = None


class FormulationAgent(Protocol):
    def formulate(self, request: FormulationRequest) -> FormulationTurn: ...


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
ProblemType = Literal["minimize_compliance", "compliant_mechanism"]
PoissonRatio = Annotated[FiniteNumber, Field(gt=-1.0, lt=0.5)]

_FIELD_ADAPTERS: dict[DraftPath, TypeAdapter] = {
    "problem_type": TypeAdapter(ProblemType),
    "domain.bounds": TypeAdapter(Bounds2D),
    "material.young_modulus": TypeAdapter(PositiveFiniteNumber),
    "material.poisson_ratio": TypeAdapter(PoissonRatio),
    "supports": TypeAdapter(SupportList),
    "tractions": TypeAdapter(TractionList),
    "body_force": TypeAdapter(Vector2D),
    "volume_fraction": TypeAdapter(OpenFraction),
    "compliance_bound": TypeAdapter(PositiveFiniteNumber),
    "input_spring": TypeAdapter(MechanismSpringIntent),
    "output_spring": TypeAdapter(MechanismSpringIntent),
    "mesh.divisions": TypeAdapter(MeshDivisions),
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
    merged = ProblemDraft(
        facts=ordered_facts,
        revisions=tuple(revisions),
        turn_count=turn_number,
    )
    return DraftMergeResult(
        draft=merged,
        accepted_paths=tuple(accepted),
        issues=tuple(issues),
    )


def _intent_payload(draft: ProblemDraft) -> dict:
    values = draft.values()
    payload: dict = {}
    if "problem_type" in values:
        payload["problem_type"] = values["problem_type"]
    if "domain.bounds" in values:
        payload["domain"] = {"bounds": values["domain.bounds"]}
    material = {}
    if "material.young_modulus" in values:
        material["young_modulus"] = values["material.young_modulus"]
    if "material.poisson_ratio" in values:
        material["poisson_ratio"] = values["material.poisson_ratio"]
    if material:
        payload["material"] = material
    for path in (
        "supports",
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
    if "mesh.divisions" in values:
        mesh["divisions"] = values["mesh.divisions"]
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
    problem_type = values.get("problem_type")
    if problem_type == "compliant_mechanism":
        missing.extend(
            path for path in MECHANISM_REQUIRED_PATHS if path not in values
        )

    tractions = values.get("tractions", [])
    body_force = values.get("body_force", [0.0, 0.0])
    has_body_force = bool(
        isinstance(body_force, list)
        and len(body_force) == 2
        and any(float(component) != 0.0 for component in body_force)
    )
    if not tractions and not has_body_force:
        missing.append("external_load")

    unconfirmed = tuple(
        fact.path for fact in draft.facts if fact.basis == "assumption"
    )
    semantic_errors: list[str] = []
    if not missing and not unconfirmed:
        try:
            _PROBLEM_ADAPTER.validate_python(_intent_payload(draft))
        except ValidationError as exc:
            semantic_errors.append(_validation_message(exc))

    return DraftReadiness(
        ready=not missing and not unconfirmed and not semantic_errors,
        missing_fields=tuple(missing),
        unconfirmed_fields=unconfirmed,
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

    def __init__(self, agent: FormulationAgent):
        self._agent = agent

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
        turn_number = session.draft.turn_count + 1
        request = FormulationRequest(
            turn_number=turn_number,
            user_message=message,
            draft=session.draft,
            history=session.messages,
        )
        turn = self._agent.formulate(request)
        merge = merge_formulation_turn(
            session.draft,
            turn,
            user_message=message,
            turn_number=turn_number,
        )
        readiness = assess_draft(merge.draft)

        intent = None
        if merge.issues:
            status: SessionStatus = "repair_needed"
        elif turn.declared_state == "unsupported":
            status = "unsupported"
        elif readiness.ready:
            status = "ready_for_review"
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
        )
