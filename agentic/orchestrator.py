"""Typed deterministic orchestration through the validation boundary."""

from __future__ import annotations

from typing import Annotated, Callable, Literal, Protocol, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic.compiler import CompilationResult, compile_intent
from agentic.explainer import ExplanationResult
from agentic.intent import (
    InterpretationResult,
    ProblemIntent,
)
from fenitop.tools.contracts import ValidateConfigRequest, ValidateConfigResponse
from fenitop.tools.contracts import (
    AnalyzeResultsRequest,
    AnalyzeResultsResponse,
    RunTopoptRequest,
    RunTopoptResponse,
    TrustedRunPolicy,
)
from fenitop.tools.analyze_results import analyze_results_tool
from fenitop.tools.lifecycle import canonical_json_hash
from fenitop.tools.run_topopt import run_topopt_tool
from fenitop.tools.validate_config import validate_config_tool


class Interpreter(Protocol):
    def interpret(self, user_request: str) -> InterpretationResult: ...


class Compiler(Protocol):
    def __call__(self, intent: ProblemIntent) -> CompilationResult: ...


class Validator(Protocol):
    def __call__(
        self, request: ValidateConfigRequest
    ) -> dict | ValidateConfigResponse: ...


class Runner(Protocol):
    def __call__(
        self,
        request: RunTopoptRequest,
        *,
        policy: TrustedRunPolicy,
    ) -> dict | RunTopoptResponse: ...


class Analyzer(Protocol):
    def __call__(
        self, request: AnalyzeResultsRequest
    ) -> dict | AnalyzeResultsResponse: ...


class Explainer(Protocol):
    def explain(self, analysis: AnalyzeResultsResponse) -> ExplanationResult: ...


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
        "run_started",
        "run_failed",
        "run_succeeded",
        "analysis_failed",
        "completed",
        "explained",
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


class RunFailedWorkflow(StrictWorkflowModel):
    status: Literal["run_failed"] = "run_failed"
    conversation: ConversationContext
    compilation: CompilationResult
    validation: ValidateConfigResponse
    idempotency_key: str
    run: RunTopoptResponse
    events: tuple[WorkflowEvent, ...]


class AnalysisFailedWorkflow(StrictWorkflowModel):
    status: Literal["analysis_failed"] = "analysis_failed"
    conversation: ConversationContext
    compilation: CompilationResult
    validation: ValidateConfigResponse
    idempotency_key: str
    run: RunTopoptResponse
    analysis: AnalyzeResultsResponse
    events: tuple[WorkflowEvent, ...]


class CompletedWorkflow(StrictWorkflowModel):
    status: Literal["completed"] = "completed"
    conversation: ConversationContext
    compilation: CompilationResult
    validation: ValidateConfigResponse
    idempotency_key: str
    run: RunTopoptResponse
    analysis: AnalyzeResultsResponse
    events: tuple[WorkflowEvent, ...]


class ExplainedWorkflow(StrictWorkflowModel):
    status: Literal["explained"] = "explained"
    conversation: ConversationContext
    compilation: CompilationResult
    validation: ValidateConfigResponse
    idempotency_key: str
    run: RunTopoptResponse
    analysis: AnalyzeResultsResponse
    explanation: ExplanationResult
    events: tuple[WorkflowEvent, ...]


WorkflowOutcome = Annotated[
    Union[
        AwaitingClarification,
        UnsupportedWorkflow,
        ValidationFailedWorkflow,
        ValidatedWorkflow,
        RunFailedWorkflow,
        AnalysisFailedWorkflow,
        CompletedWorkflow,
        ExplainedWorkflow,
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
        runner: Runner = run_topopt_tool,
        analyzer: Analyzer = analyze_results_tool,
        explainer: Explainer | None = None,
        event_callback: Callable[[WorkflowEvent], None] | None = None,
    ):
        self._interpreter = interpreter
        self._compiler = compiler
        self._validator = validator
        self._runner = runner
        self._analyzer = analyzer
        self._explainer = explainer
        self._event_callback = event_callback
        self._run_cache: dict[str, RunTopoptResponse] = {}

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

    def execute(
        self,
        state: ValidatedWorkflow | AnalysisFailedWorkflow | CompletedWorkflow,
    ) -> RunFailedWorkflow | AnalysisFailedWorkflow | CompletedWorkflow:
        """Run and analyze once; resume analysis without repeating the solve."""
        if isinstance(state, CompletedWorkflow):
            return state
        if isinstance(state, AnalysisFailedWorkflow):
            return self._analyze(
                conversation=state.conversation,
                compilation=state.compilation,
                validation=state.validation,
                idempotency_key=state.idempotency_key,
                run=state.run,
                prior_events=state.events,
            )

        events = list(state.events)
        idempotency_key = self._idempotency_key(state)
        self._emit(
            events,
            "run_started",
            "Launching the contained solver with an application-owned "
            "idempotency key.",
        )
        run = self._run_cache.get(idempotency_key)
        if run is None:
            request = RunTopoptRequest(config=state.compilation.config)
            raw_run = self._runner(
                request,
                policy=TrustedRunPolicy(idempotency_key=idempotency_key),
            )
            run = (
                raw_run
                if isinstance(raw_run, RunTopoptResponse)
                else RunTopoptResponse.model_validate(raw_run)
            )
            self._run_cache[idempotency_key] = run

        if run.status == "error":
            self._emit(events, "run_failed", run.message or "Solver run failed.")
            return RunFailedWorkflow(
                conversation=state.conversation,
                compilation=state.compilation,
                validation=state.validation,
                idempotency_key=idempotency_key,
                run=run,
                events=tuple(events),
            )
        if run.run_manifest is None:
            raise RuntimeError("Successful run response is missing run_manifest.")

        self._emit(
            events,
            "run_succeeded",
            f"Solver run {run.run_manifest.run_id} succeeded.",
        )
        return self._analyze(
            conversation=state.conversation,
            compilation=state.compilation,
            validation=state.validation,
            idempotency_key=idempotency_key,
            run=run,
            prior_events=events,
        )

    def explain(
        self,
        state: CompletedWorkflow | ExplainedWorkflow,
    ) -> ExplainedWorkflow:
        """Optionally organize immutable analysis evidence for presentation."""
        if isinstance(state, ExplainedWorkflow):
            return state
        if self._explainer is None:
            raise RuntimeError("No result explainer is configured.")
        explanation = self._explainer.explain(state.analysis)
        events = list(state.events)
        self._emit(
            events,
            "explained",
            "The LLM organized immutable evidence IDs; deterministic code "
            "rendered their original facts.",
        )
        return ExplainedWorkflow(
            conversation=state.conversation,
            compilation=state.compilation,
            validation=state.validation,
            idempotency_key=state.idempotency_key,
            run=state.run,
            analysis=state.analysis,
            explanation=explanation,
            events=tuple(events),
        )

    def _analyze(
        self,
        *,
        conversation: ConversationContext,
        compilation: CompilationResult,
        validation: ValidateConfigResponse,
        idempotency_key: str,
        run: RunTopoptResponse,
        prior_events: Sequence[WorkflowEvent],
    ) -> AnalysisFailedWorkflow | CompletedWorkflow:
        if run.run_manifest is None:
            raise RuntimeError("Analysis requires a successful run manifest.")
        events = list(prior_events)
        request = AnalyzeResultsRequest(run_manifest=run.run_manifest)
        raw_analysis = self._analyzer(request)
        analysis = (
            raw_analysis
            if isinstance(raw_analysis, AnalyzeResultsResponse)
            else AnalyzeResultsResponse.model_validate(raw_analysis)
        )
        if analysis.status == "error":
            self._emit(
                events,
                "analysis_failed",
                analysis.message or "Deterministic analysis failed.",
            )
            return AnalysisFailedWorkflow(
                conversation=conversation,
                compilation=compilation,
                validation=validation,
                idempotency_key=idempotency_key,
                run=run,
                analysis=analysis,
                events=tuple(events),
            )

        self._emit(
            events,
            "completed",
            "The successful run manifest was analyzed deterministically.",
        )
        return CompletedWorkflow(
            conversation=conversation,
            compilation=compilation,
            validation=validation,
            idempotency_key=idempotency_key,
            run=run,
            analysis=analysis,
            events=tuple(events),
        )

    @staticmethod
    def _idempotency_key(state: ValidatedWorkflow) -> str:
        material = {
            "conversation": state.conversation.model_dump(mode="json"),
            "config": state.compilation.config.model_dump(mode="json"),
            "defaults_profile": state.compilation.defaults_profile,
        }
        return f"agentic-workflow-v1:{canonical_json_hash(material)}"

    def _emit(self, events: list[WorkflowEvent], stage, message) -> None:
        event = WorkflowEvent(
            sequence=len(events) + 1,
            stage=stage,
            message=message,
        )
        events.append(event)
        if self._event_callback is not None:
            self._event_callback(event)

    def _advance(
        self,
        conversation: ConversationContext,
        *,
        prior_events: Sequence[WorkflowEvent] = (),
    ) -> WorkflowOutcome:
        events = list(prior_events)

        interpretation = self._interpreter.interpret(
            conversation.interpreter_request()
        )
        self._emit(
            events,
            "interpreted",
            f"Interpreter returned {interpretation.status}.",
        )

        if interpretation.status == "needs_clarification":
            self._emit(
                events,
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
            self._emit(events, "unsupported", interpretation.explanation)
            return UnsupportedWorkflow(
                conversation=conversation,
                unsupported_features=tuple(interpretation.unsupported_features),
                explanation=interpretation.explanation,
                events=tuple(events),
            )

        compilation = self._compiler(interpretation.intent)
        self._emit(events, "defaults_applied", compilation.defaults_notice)

        request = ValidateConfigRequest(config=compilation.config)
        raw_validation = self._validator(request)
        validation = (
            raw_validation
            if isinstance(raw_validation, ValidateConfigResponse)
            else ValidateConfigResponse.model_validate(raw_validation)
        )
        if validation.status == "error":
            self._emit(
                events,
                "validation_failed",
                f"Validation returned {len(validation.errors)} error(s).",
            )
            return ValidationFailedWorkflow(
                conversation=conversation,
                compilation=compilation,
                validation=validation,
                events=tuple(events),
            )

        self._emit(
            events,
            "validated",
            "The compiled configuration passed validation.",
        )
        return ValidatedWorkflow(
            conversation=conversation,
            compilation=compilation,
            validation=validation,
            events=tuple(events),
        )
