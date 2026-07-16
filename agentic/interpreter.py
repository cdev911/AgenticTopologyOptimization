"""Bounded LLM adapter for free-text structural intent interpretation."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any, Protocol, Sequence

from crewai import LLM
from crewai.utilities.pydantic_schema_utils import generate_model_description
from pydantic import ValidationError

from agentic.intent import InterpretationEnvelope, InterpretationResult

PROMPT_VERSION = "intent-system-v1"
DEFAULT_MAX_ATTEMPTS = 2


class StructuredLLM(Protocol):
    """The only model capability the interpreter is allowed to use."""

    def call(self, messages: Sequence[dict[str, str]]) -> object: ...


class InterpretationError(RuntimeError):
    """A model/transport failure, distinct from a semantic unsupported result."""

    def __init__(self, attempts: int, last_error_type: str):
        self.attempts = attempts
        self.last_error_type = last_error_type
        super().__init__(
            f"Intent interpretation failed after {attempts} attempt(s) "
            f"({last_error_type})."
        )


@dataclass(frozen=True)
class InterpreterConfig:
    model: str
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    reasoning_effort: str = "low"
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be blank.")
        if not 1 <= self.max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3.")
        if self.reasoning_effort != "low":
            raise ValueError("Stage 1 pins reasoning_effort to 'low'.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")


def load_system_prompt() -> str:
    prompt = (
        files("agentic.prompts")
        .joinpath("intent_system_v1.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    if not prompt:
        raise RuntimeError("Intent system prompt is empty.")
    return prompt


def _normalize_openai_schema(value: Any) -> Any:
    """Make Pydantic's schema satisfy OpenAI's strict-schema subset."""
    if isinstance(value, list):
        return [_normalize_openai_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized = {
        key: _normalize_openai_schema(child)
        for key, child in value.items()
        if key != "prefixItems"
    }
    if normalized.get("type") == "object" or "properties" in normalized:
        normalized["additionalProperties"] = False

    prefix_items = value.get("prefixItems")
    if prefix_items is not None:
        choices = [_normalize_openai_schema(item) for item in prefix_items]
        if all(choice == choices[0] for choice in choices[1:]):
            normalized["items"] = choices[0]
        else:
            normalized["items"] = {"anyOf": choices}
    return normalized


@lru_cache(maxsize=1)
def _dereferenced_openai_schema() -> dict[str, Any]:
    """Build the normalized inner schema once from the semantic model."""
    generated = generate_model_description(
        InterpretationEnvelope,
        strip_null_types=False,
    )
    return _normalize_openai_schema(generated["json_schema"]["schema"])


class OpenAIInterpretationEnvelope(InterpretationEnvelope):
    """Typed transport whose schema is stable under CrewAI post-processing."""

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        # Return a copy because CrewAI mutates schemas while preparing requests.
        return deepcopy(_dereferenced_openai_schema())


def openai_response_format() -> dict[str, Any]:
    """Return the final schema CrewAI will send to OpenAI."""
    generated = generate_model_description(
        OpenAIInterpretationEnvelope,
        strip_null_types=False,
    )
    return _normalize_openai_schema(generated)


def build_crewai_llm(config: InterpreterConfig) -> StructuredLLM:
    """Build the pinned provider adapter with hidden SDK retries disabled.

    The transport subclass is intentional: CrewAI requires a model class at
    provider construction, while its ordinary recursive dereferencing can
    introduce invalid strict-schema nodes after normalization.
    """
    return LLM(
        model=f"openai/{config.model}",
        response_format=OpenAIInterpretationEnvelope,
        reasoning_effort=config.reasoning_effort,
        timeout=config.timeout_seconds,
        max_retries=0,
    )


def config_from_environment() -> InterpreterConfig:
    return InterpreterConfig(model=os.getenv("OPENAI_MODEL", "gpt-5.6-terra"))


class IntentInterpreter:
    """Interpret one user message; never validate, compile, or launch a solve."""

    def __init__(
        self,
        llm: StructuredLLM,
        *,
        config: InterpreterConfig,
        system_prompt: str | None = None,
    ):
        self._llm = llm
        self.config = config
        self.system_prompt = system_prompt or load_system_prompt()

    @classmethod
    def from_environment(cls) -> "IntentInterpreter":
        config = config_from_environment()
        return cls(build_crewai_llm(config), config=config)

    def interpret(self, user_request: str) -> InterpretationResult:
        request = user_request.strip()
        if not request:
            raise ValueError("user_request must not be blank.")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"user_request": request},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]

        last_error_type = "unknown_error"
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                raw = self._llm.call(messages)
                envelope = self._validate_envelope(raw)
                return envelope.result
            except Exception as exc:
                last_error_type = type(exc).__name__
                if attempt == self.config.max_attempts:
                    raise InterpretationError(
                        attempts=attempt,
                        last_error_type=last_error_type,
                    ) from None

        raise AssertionError("unreachable")

    @staticmethod
    def _validate_envelope(raw: object) -> InterpretationEnvelope:
        if isinstance(raw, InterpretationEnvelope):
            return raw
        if isinstance(raw, (str, bytes, bytearray)):
            return InterpretationEnvelope.model_validate_json(raw)
        try:
            return InterpretationEnvelope.model_validate(raw)
        except ValidationError:
            raise
