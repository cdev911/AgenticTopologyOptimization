"""Live OpenAI Responses adapter for conversational problem formulation."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from importlib.resources import files
from typing import Annotated, Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic.boundary_draft import (
    BoundaryConfirm,
    BoundaryCreate,
    BoundaryDelete,
    BoundaryField,
    BoundaryFieldInput,
    BoundaryKind,
    BoundaryPatch,
    BoundaryUpdate,
    BoundaryUpdateBasis,
)
from agentic.formulation import (
    DeclaredTurnState,
    DraftClear,
    DraftUpdate,
    FactBasis,
    FormulationAgentResponse,
    FormulationModelState,
    FormulationRequest,
    FormulationTurn,
    assess_draft,
)


PROMPT_VERSION = "formulation-system-v2"
ADAPTER_ID = "openai-responses-v2"
DEFAULT_MODEL = "gpt-5.6-sol"
ReasoningEffort = Literal["low", "medium", "high", "xhigh", "max"]
ErrorKind = Literal["provider", "invalid_response", "refusal"]
LiveDraftPath = Literal[
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


class StrictOpenAIFormulationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OpenAIDraftUpdate(StrictOpenAIFormulationModel):
    """Compact transport update; value_json avoids an unconstrained JSON schema."""

    path: LiveDraftPath
    value_json: str
    basis: FactBasis
    source_quote: str | None
    rationale: str

    @field_validator("value_json", "rationale")
    @classmethod
    def _nonblank_text(cls, value):
        if not value.strip():
            raise ValueError("text must not be blank.")
        return value.strip()

    def to_domain(self) -> DraftUpdate:
        value = _decode_json_value(self.value_json)
        return DraftUpdate(
            path=self.path,
            value=value,
            basis=self.basis,
            source_quote=self.source_quote,
            rationale=self.rationale,
        )


class OpenAIDraftClear(StrictOpenAIFormulationModel):
    path: LiveDraftPath
    source_quote: str
    rationale: str

    def to_domain(self) -> DraftClear:
        return DraftClear(**self.model_dump())


class OpenAIBoundaryFieldInput(StrictOpenAIFormulationModel):
    """One compact create-field transport decoded by the domain validator."""

    field: BoundaryField
    value_json: str
    basis: BoundaryUpdateBasis
    source_quote: str | None
    rationale: str

    @field_validator("value_json", "rationale")
    @classmethod
    def _nonblank_text(cls, value):
        if not value.strip():
            raise ValueError("text must not be blank.")
        return value.strip()

    def to_domain(self) -> BoundaryFieldInput:
        return BoundaryFieldInput(
            field=self.field,
            value=_decode_json_value(self.value_json),
            basis=self.basis,
            source_quote=self.source_quote,
            rationale=self.rationale,
        )


class OpenAIBoundaryCreate(StrictOpenAIFormulationModel):
    local_ref: str = Field(pattern=r"^new_[a-z0-9][a-z0-9_]*$")
    kind: BoundaryKind
    fields: Annotated[
        tuple[OpenAIBoundaryFieldInput, ...],
        Field(min_length=1),
    ]

    def to_domain(self) -> BoundaryCreate:
        return BoundaryCreate(
            local_ref=self.local_ref,
            kind=self.kind,
            fields=tuple(field.to_domain() for field in self.fields),
        )


class OpenAIBoundaryUpdate(StrictOpenAIFormulationModel):
    bc_id: str = Field(pattern=r"^[SL][1-9][0-9]*$")
    field: BoundaryField
    value_json: str
    basis: BoundaryUpdateBasis
    source_quote: str | None
    rationale: str

    @field_validator("value_json", "rationale")
    @classmethod
    def _nonblank_text(cls, value):
        if not value.strip():
            raise ValueError("text must not be blank.")
        return value.strip()

    def to_domain(self) -> BoundaryUpdate:
        return BoundaryUpdate(
            bc_id=self.bc_id,
            field=self.field,
            value=_decode_json_value(self.value_json),
            basis=self.basis,
            source_quote=self.source_quote,
            rationale=self.rationale,
        )


class OpenAIBoundaryDelete(StrictOpenAIFormulationModel):
    bc_id: str = Field(pattern=r"^[SL][1-9][0-9]*$")
    source_quote: str
    rationale: str

    def to_domain(self) -> BoundaryDelete:
        return BoundaryDelete(**self.model_dump())


class OpenAIBoundaryConfirm(StrictOpenAIFormulationModel):
    bc_id: str = Field(pattern=r"^[SL][1-9][0-9]*$")
    field: BoundaryField
    source_quote: str
    rationale: str

    def to_domain(self) -> BoundaryConfirm:
        return BoundaryConfirm(**self.model_dump())


class OpenAIBoundaryPatch(StrictOpenAIFormulationModel):
    """Flat strict transport; canonical IDs remain application-owned."""

    creates: tuple[OpenAIBoundaryCreate, ...]
    updates: tuple[OpenAIBoundaryUpdate, ...]
    deletes: tuple[OpenAIBoundaryDelete, ...]
    confirmations: tuple[OpenAIBoundaryConfirm, ...]

    def to_domain(self) -> BoundaryPatch:
        return BoundaryPatch(
            creates=tuple(item.to_domain() for item in self.creates),
            updates=tuple(item.to_domain() for item in self.updates),
            deletes=tuple(item.to_domain() for item in self.deletes),
            confirmations=tuple(
                item.to_domain() for item in self.confirmations
            ),
        )


class OpenAIFormulationTurn(StrictOpenAIFormulationModel):
    """Strict API transport converted immediately to the domain turn contract."""

    assistant_message: str
    updates: tuple[OpenAIDraftUpdate, ...]
    clears: tuple[OpenAIDraftClear, ...]
    boundary_patch: OpenAIBoundaryPatch
    questions: Annotated[tuple[str, ...], Field(max_length=3)]
    declared_state: DeclaredTurnState
    unsupported_features: tuple[str, ...]

    def to_domain(self) -> FormulationTurn:
        return FormulationTurn(
            assistant_message=self.assistant_message,
            updates=tuple(update.to_domain() for update in self.updates),
            clears=tuple(clear.to_domain() for clear in self.clears),
            boundary_patch=self.boundary_patch.to_domain(),
            questions=self.questions,
            declared_state=self.declared_state,
            unsupported_features=self.unsupported_features,
        )


def _decode_json_value(value_json: str):
    try:
        return json.loads(value_json)
    except json.JSONDecodeError:
        # No ordinary or boundary field accepts null. Converting malformed inner
        # JSON to null routes it through deterministic invalid_value feedback and
        # the bounded same-turn repair instead of aborting the conversation.
        return None


class ResponsesAPI(Protocol):
    def parse(self, **kwargs) -> object: ...


class OpenAIClient(Protocol):
    responses: ResponsesAPI


class FormulationAPIError(RuntimeError):
    """Sanitized model-boundary failure; the canonical draft remains unchanged."""

    def __init__(self, kind: ErrorKind, provider_error_type: str):
        self.kind = kind
        self.provider_error_type = provider_error_type
        super().__init__(
            "Conversational formulation failed at the model boundary "
            f"({kind}; {provider_error_type})."
        )


@dataclass(frozen=True)
class OpenAIFormulationConfig:
    model: str = DEFAULT_MODEL
    reasoning_effort: ReasoningEffort = "medium"
    timeout_seconds: int = 90
    max_output_tokens: int = 5_000
    safety_identifier: str = "agentic-topopt-local-user"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be blank.")
        if self.reasoning_effort not in {
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("unsupported reasoning_effort.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if not 512 <= self.max_output_tokens <= 20_000:
            raise ValueError("max_output_tokens must be between 512 and 20000.")
        if not self.safety_identifier.strip():
            raise ValueError("safety_identifier must not be blank.")


@dataclass(frozen=True)
class FormulationCallRecord:
    response_id: str
    recovered_context: bool
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cached_tokens: int


def config_from_environment() -> OpenAIFormulationConfig:
    return OpenAIFormulationConfig(
        model=os.getenv("OPENAI_FORMULATION_MODEL", DEFAULT_MODEL),
        reasoning_effort=os.getenv(
            "OPENAI_FORMULATION_REASONING_EFFORT",
            "medium",
        ),
        timeout_seconds=int(
            os.getenv("OPENAI_FORMULATION_TIMEOUT_SECONDS", "90")
        ),
        max_output_tokens=int(
            os.getenv("OPENAI_FORMULATION_MAX_OUTPUT_TOKENS", "5000")
        ),
    )


def load_system_prompt() -> str:
    prompt = (
        files("agentic.prompts")
        .joinpath("formulation_system_v2.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    if not prompt:
        raise RuntimeError("Formulation system prompt is empty.")
    return prompt


def build_openai_client(
    config: OpenAIFormulationConfig,
) -> OpenAIClient:
    """Create a client with hidden provider retries disabled."""
    return OpenAI(
        timeout=config.timeout_seconds,
        max_retries=0,
    )


class OpenAIResponsesFormulationAgent:
    """Return a small typed patch while deterministic code owns canonical state."""

    first_class_boundary_patches = True

    def __init__(
        self,
        client: OpenAIClient,
        *,
        config: OpenAIFormulationConfig,
        system_prompt: str | None = None,
    ):
        self._client = client
        self.config = config
        self.system_prompt = system_prompt or load_system_prompt()
        self.call_records: list[FormulationCallRecord] = []

    @classmethod
    def from_environment(cls) -> "OpenAIResponsesFormulationAgent":
        config = config_from_environment()
        return cls(build_openai_client(config), config=config)

    def formulate(
        self,
        request: FormulationRequest,
    ) -> FormulationAgentResponse:
        previous_response_id = self._continuation_id(request)
        kwargs = self._request_kwargs(
            request,
            previous_response_id=previous_response_id,
        )
        started = time.monotonic()
        recovered_context = False
        try:
            response = self._client.responses.parse(**kwargs)
        except Exception as exc:
            if previous_response_id and _continuation_is_unavailable(exc):
                recovered_context = True
                recovery_kwargs = self._request_kwargs(
                    request,
                    previous_response_id=None,
                    recovering_context=True,
                )
                try:
                    response = self._client.responses.parse(**recovery_kwargs)
                except Exception as recovery_exc:
                    raise FormulationAPIError(
                        "provider",
                        type(recovery_exc).__name__,
                    ) from None
            else:
                raise FormulationAPIError(
                    "provider",
                    type(exc).__name__,
                ) from None

        response_id = getattr(response, "id", None)
        if not isinstance(response_id, str) or not response_id.strip():
            raise FormulationAPIError("invalid_response", "MissingResponseId")
        self.call_records.append(
            _call_record(
                response,
                response_id=response_id,
                recovered_context=recovered_context,
                latency_seconds=time.monotonic() - started,
            )
        )
        turn = self._parsed_turn(response)
        return FormulationAgentResponse(
            turn=turn,
            model_state=FormulationModelState(
                adapter=ADAPTER_ID,
                continuation_id=response_id,
            ),
        )

    def _continuation_id(
        self,
        request: FormulationRequest,
    ) -> str | None:
        state = request.model_state
        if state is None or state.adapter != ADAPTER_ID:
            return None
        return state.continuation_id

    def _request_kwargs(
        self,
        request: FormulationRequest,
        *,
        previous_response_id: str | None,
        recovering_context: bool = False,
    ) -> dict:
        payload = {
            "turn_number": request.turn_number,
            "user_message": request.user_message,
            "canonical_draft": request.draft.model_dump(mode="json"),
            "boundary_catalog": [
                condition.model_dump(mode="json")
                for condition in request.draft.boundary_state.conditions
            ],
            "pending_boundary_confirmations": [
                item.model_dump(mode="json")
                for item in (
                    request.draft.boundary_state.pending_confirmations()
                )
            ],
            "readiness_before_turn": assess_draft(request.draft).model_dump(
                mode="json"
            ),
            "history": (
                [
                    message.model_dump(mode="json")
                    for message in request.history
                ]
                if previous_response_id is None
                else []
            ),
            "repair_feedback": (
                request.repair.model_dump(mode="json")
                if request.repair is not None
                else None
            ),
            "context_recovery": recovering_context,
        }
        kwargs = {
            "model": self.config.model,
            "instructions": self.system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            ],
            "text_format": OpenAIFormulationTurn,
            "reasoning": {
                "effort": self.config.reasoning_effort,
                "context": "all_turns",
            },
            "store": True,
            "max_output_tokens": self.config.max_output_tokens,
            "safety_identifier": self.config.safety_identifier,
            "timeout": self.config.timeout_seconds,
        }
        if previous_response_id is not None:
            kwargs["previous_response_id"] = previous_response_id
        return kwargs

    @staticmethod
    def _parsed_turn(response: object) -> FormulationTurn:
        parsed = getattr(response, "output_parsed", None)
        if parsed is not None:
            if isinstance(parsed, FormulationTurn):
                return parsed
            transport = (
                parsed
                if isinstance(parsed, OpenAIFormulationTurn)
                else OpenAIFormulationTurn.model_validate(parsed)
            )
            try:
                return transport.to_domain()
            except FormulationAPIError:
                raise
            except Exception as exc:
                raise FormulationAPIError(
                    "invalid_response",
                    type(exc).__name__,
                ) from None
        for output in getattr(response, "output", ()) or ():
            for content in getattr(output, "content", ()) or ():
                if getattr(content, "type", None) == "refusal":
                    raise FormulationAPIError("refusal", "ModelRefusal")
        status = getattr(response, "status", None)
        error_type = (
            "IncompleteResponse"
            if status == "incomplete"
            else "MissingParsedOutput"
        )
        raise FormulationAPIError("invalid_response", error_type)


def _continuation_is_unavailable(error: Exception) -> bool:
    """Recognize only a missing/expired continuation before one full-context retry."""
    status_code = getattr(error, "status_code", None)
    if status_code == 404:
        return True
    if status_code != 400:
        return False
    text = str(error).casefold()
    return (
        "previous_response_id" in text
        or "previous response" in text
    )


def _call_record(
    response: object,
    *,
    response_id: str,
    recovered_context: bool,
    latency_seconds: float,
) -> FormulationCallRecord:
    usage = getattr(response, "usage", None)
    output_details = getattr(usage, "output_tokens_details", None)
    input_details = getattr(usage, "input_tokens_details", None)
    return FormulationCallRecord(
        response_id=response_id,
        recovered_context=recovered_context,
        latency_seconds=latency_seconds,
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        reasoning_tokens=int(
            getattr(output_details, "reasoning_tokens", 0) or 0
        ),
        cached_tokens=int(
            getattr(input_details, "cached_tokens", 0) or 0
        ),
    )
