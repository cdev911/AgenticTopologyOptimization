"""Typed deterministic orchestration through the validation boundary."""

from __future__ import annotations

from typing import Annotated, Callable, Literal, Protocol, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic.compiler import CompilationResult, compile_intent
from agentic.intent import (
    InterpretationResult,
    ProblemIntent,
)
from fenitop.tools.contracts import ValidateConfigRequest, ValidateConfigResponse
from fenitop.tools.validate_config import validate_config_tool


class Interpreter(Protocol):
    def interpret(self, user_request: str) -> InterpretationResult: ...


class Compiler(Protocol):
    def __call__(self, intent: ProblemIntent) -> CompilationResult: ...


class Validator(Protocol):
    def __call__(
        self, request: ValidateConfigRequest
    ) -> dict | ValidateConfigResponse: ...


class StrictWorkflowModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClarificationExchange(StrictWorkflowModel):
    missing_fields: tuple[str, ...]
    questions: tuple[str, ...]
    answer: str

    @field_validator("answer")
    @classmethod
    def _nonblank_answer(cls, value):
        if not value.strip():
            raise ValueError("clarification answer must not be blank.")
        return value.strip()


class ConversationContext(StrictWorkflowModel):
    original_request: str
    clarifications: tuple[ClarificationExchange, ...] = ()

    @field_validator("original_request")
    @classmethod
    def _nonblank_original(cls, value):
        if not value.strip():
            raise ValueError("original_request must not be blank.")
        return value.strip()

    def interpreter_request(self) -> str:
        if not self.clarifications:
            return self.original_request
        lines = [f"Original request:\n{self.original_request}"]
        for index, exchange in enumerate(self.clarifications, start=1):
            fields = ", ".join(exchange.missing_fields)
            questions = "\n".join(f"- {item}" for item in exchange.questions)
            lines.append(
                f"Clarification exchange {index}:\n"
                f"Missing fields: {fields}\n"
                f"Questions:\n{questions}\n"
                f"User answer: {exchange.answer}"
            )
        return "\n\n".join(lines)


class WorkflowEvent(StrictWorkflowModel):
    sequence: int = Field(ge=1)
    stage: Literal[
        "interpreted",
        "clarification_requested",
        "unsupported",
        "defaults_applied",
        "validation_failed",
        "validated",
    ]
    message: str


class AwaitingClarification(StrictWorkflowModel):
    status: Literal["awaiting_clarification"] = "awaiting_clarification"
    conversation: ConversationContext
    missing_fields: tuple[str, ...]
    questions: tuple[str, ...]
    events: tuple[WorkflowEvent, ...]


class UnsupportedWorkflow(StrictWorkflowModel):
    status: Literal["unsupported"] = "unsupported"
    conversation: ConversationContext
    unsupported_features: tuple[str, ...]
    explanation: str
    events: tuple[WorkflowEvent, ...]


class ValidationFailedWorkflow(StrictWorkflowModel):
    status: Literal["validation_failed"] = "validation_failed"
    conversation: ConversationContext
    compilation: CompilationResult
    validation: ValidateConfigResponse
    events: tuple[WorkflowEvent, ...]


class ValidatedWorkflow(StrictWorkflowModel):
    status: Literal["validated"] = "validated"
    conversation: ConversationContext
    compilation: CompilationResult
    validation: ValidateConfigResponse
    events: tuple[WorkflowEvent, ...]


WorkflowOutcome = Annotated[
    Union[
        AwaitingClarification,
        UnsupportedWorkflow,
        ValidationFailedWorkflow,
        ValidatedWorkflow,
    ],
    Field(discriminator="status"),
]


class DeterministicOrchestrator:
    """Advance a conversation without granting the LLM side-effect authority."""

    def __init__(
        self,
        interpreter: Interpreter,
        *,
        compiler: Compiler = compile_intent,
        validator: Validator = validate_config_tool,
        event_callback: Callable[[WorkflowEvent], None] | None = None,
    ):
        self._interpreter = interpreter
        self._compiler = compiler
        self._validator = validator
        self._event_callback = event_callback

    def start(self, user_request: str) -> WorkflowOutcome:
        return self._advance(ConversationContext(original_request=user_request))

    def resume(
        self,
        prior: AwaitingClarification,
        clarification_answer: str,
    ) -> WorkflowOutcome:
        answer = clarification_answer.strip()
        if not answer:
            raise ValueError("clarification_answer must not be blank.")
        conversation = ConversationContext(
            original_request=prior.conversation.original_request,
            clarifications=(
                *prior.conversation.clarifications,
                ClarificationExchange(
                    missing_fields=prior.missing_fields,
                    questions=prior.questions,
                    answer=answer,
                ),
            ),
        )
        return self._advance(conversation, prior_events=prior.events)

    def _advance(
        self,
        conversation: ConversationContext,
        *,
        prior_events: Sequence[WorkflowEvent] = (),
    ) -> WorkflowOutcome:
        events = list(prior_events)

        def emit(stage, message) -> None:
            event = WorkflowEvent(
                sequence=len(events) + 1,
                stage=stage,
                message=message,
            )
            events.append(event)
            if self._event_callback is not None:
                self._event_callback(event)

        interpretation = self._interpreter.interpret(
            conversation.interpreter_request()
        )
        emit("interpreted", f"Interpreter returned {interpretation.status}.")

        if interpretation.status == "needs_clarification":
            emit(
                "clarification_requested",
                "Required problem-defining information is still missing.",
            )
            return AwaitingClarification(
                conversation=conversation,
                missing_fields=tuple(interpretation.missing_fields),
                questions=tuple(interpretation.questions),
                events=tuple(events),
            )

        if interpretation.status == "unsupported":
            emit("unsupported", interpretation.explanation)
            return UnsupportedWorkflow(
                conversation=conversation,
                unsupported_features=tuple(interpretation.unsupported_features),
                explanation=interpretation.explanation,
                events=tuple(events),
            )

        compilation = self._compiler(interpretation.intent)
        emit("defaults_applied", compilation.defaults_notice)

        request = ValidateConfigRequest(config=compilation.config)
        raw_validation = self._validator(request)
        validation = (
            raw_validation
            if isinstance(raw_validation, ValidateConfigResponse)
            else ValidateConfigResponse.model_validate(raw_validation)
        )
        if validation.status == "error":
            emit(
                "validation_failed",
                f"Validation returned {len(validation.errors)} error(s).",
            )
            return ValidationFailedWorkflow(
                conversation=conversation,
                compilation=compilation,
                validation=validation,
                events=tuple(events),
            )

        emit("validated", "The compiled configuration passed validation.")
        return ValidatedWorkflow(
            conversation=conversation,
            compilation=compilation,
            validation=validation,
            events=tuple(events),
        )
