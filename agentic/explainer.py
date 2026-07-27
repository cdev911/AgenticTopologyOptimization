"""Fact-preserving LLM presentation over deterministic analysis evidence."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal, Protocol, Sequence

from crewai import LLM
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fenitop.tools.contracts import AnalyzeResultsResponse

PROMPT_VERSION = "explainer-system-v1"
DEFAULT_MAX_ATTEMPTS = 2


class StructuredLLM(Protocol):
    def call(self, messages: Sequence[dict[str, str]]) -> object: ...


class StrictExplanationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceFact(StrictExplanationModel):
    fact_id: str = Field(pattern=r"^F[0-9]{3}$")
    category: Literal[
        "run", "convergence", "metrics", "constraints", "quality", "warning"
    ]
    priority: Literal["required", "supporting"]
    text: str

    @field_validator("text")
    @classmethod
    def _nonblank_text(cls, value):
        if not value.strip():
            raise ValueError("evidence text must not be blank.")
        return value.strip()


class EvidenceLedger(StrictExplanationModel):
    facts: tuple[EvidenceFact, ...]


class ExplanationSection(StrictExplanationModel):
    heading: Literal[
        "Outcome", "Convergence", "Metrics", "Constraints", "Quality", "Warnings"
    ]
    fact_ids: tuple[str, ...] = Field(min_length=1, max_length=12)


class ExplanationPlan(StrictExplanationModel):
    sections: tuple[ExplanationSection, ...] = Field(min_length=1, max_length=6)


class ExplanationResult(StrictExplanationModel):
    prompt_version: Literal["explainer-system-v1"] = PROMPT_VERSION
    evidence: EvidenceLedger
    plan: ExplanationPlan
    markdown: str


class ExplanationError(RuntimeError):
    def __init__(self, attempts: int, last_error_type: str):
        self.attempts = attempts
        self.last_error_type = last_error_type
        super().__init__(
            f"Result explanation failed after {attempts} attempt(s) "
            f"({last_error_type})."
        )


@dataclass(frozen=True)
class ExplainerConfig:
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
        .joinpath("explainer_system_v1.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    if not prompt:
        raise RuntimeError("Explainer system prompt is empty.")
    return prompt


def build_crewai_llm(config: ExplainerConfig) -> StructuredLLM:
    return LLM(
        model=f"openai/{config.model}",
        response_format=ExplanationPlan,
        reasoning_effort=config.reasoning_effort,
        timeout=config.timeout_seconds,
        max_retries=0,
    )


def config_from_environment() -> ExplainerConfig:
    return ExplainerConfig(model=os.getenv("OPENAI_MODEL", "gpt-5.6-terra"))


def build_evidence_ledger(analysis: AnalyzeResultsResponse) -> EvidenceLedger:
    """Convert verified analysis into immutable human-readable facts."""
    if analysis.status != "ok":
        raise ValueError("Only successful deterministic analysis can be explained.")
    if analysis.source is None:
        raise ValueError("Successful analysis is missing source evidence.")
    if analysis.convergence is None:
        raise ValueError("Successful analysis is missing convergence evidence.")
    if analysis.metrics is None:
        raise ValueError("Successful analysis is missing metric evidence.")
    if analysis.quality_flags is None:
        raise ValueError("Successful analysis is missing quality evidence.")

    facts: list[EvidenceFact] = []

    def add(category, priority, text) -> None:
        facts.append(
            EvidenceFact(
                fact_id=f"F{len(facts) + 1:03d}",
                category=category,
                priority=priority,
                text=text,
            )
        )

    source = analysis.source
    convergence = analysis.convergence
    metrics = analysis.metrics
    constraints = metrics.constraints
    quality = analysis.quality_flags

    add("run", "required", f"Run ID: {source.run_id}.")
    add(
        "convergence",
        "required",
        f"Converged: {convergence.converged}; stop reason: "
        f"{convergence.stop_reason}; iterations: {convergence.iterations}.",
    )
    if constraints.compliance_bound is None:
        metric_text = (
            f"Final compliance objective: {metrics.final_compliance}; "
            f"final volume: {metrics.final_volume}."
        )
    else:
        metric_text = (
            f"Final compliance: {metrics.final_compliance}; final volume: "
            f"{metrics.final_volume}; signed output objective: "
            f"{metrics.final_objective}."
        )
    add("metrics", "required", metric_text)
    add(
        "constraints",
        "required",
        f"Volume target: {constraints.volume_target}; volume error: "
        f"{constraints.volume_error}; volume satisfied: "
        f"{constraints.volume_satisfied}; density bounds satisfied: "
        f"{constraints.density_bounds_satisfied}.",
    )
    if constraints.compliance_bound is not None:
        add(
            "constraints",
            "required",
            f"Compliance bound: {constraints.compliance_bound}; bound satisfied: "
            f"{constraints.compliance_bound_satisfied}.",
        )
    add(
        "quality",
        "required",
        f"High-grayness warning: {quality.high_grayness_warning}; checkerboard "
        f"detected: {quality.checkerboard_detected}; disconnected material: "
        f"{quality.has_disconnected_material}; load path connected: "
        f"{quality.load_path_connected}.",
    )
    add(
        "convergence",
        "supporting",
        f"Final design change: {convergence.final_change}; tolerance: "
        f"{convergence.opt_tol}; final beta: {convergence.final_beta}; "
        f"continuation completed: {convergence.continuation_completed}.",
    )
    add(
        "quality",
        "supporting",
        f"Grayness: {quality.grayness}; binarization score: "
        f"{quality.binarization_score}; component count: "
        f"{quality.num_components}; largest component fraction: "
        f"{quality.largest_component_fraction}.",
    )
    if analysis.narrative:
        add(
            "metrics",
            "supporting",
            f"Deterministic analyzer summary: {analysis.narrative}",
        )
    for warning in analysis.warnings:
        add(
            "warning",
            "supporting",
            f"Analyzer warning {warning.code} at {warning.path}: "
            f"{warning.message}",
        )
    return EvidenceLedger(facts=tuple(facts))


def validate_plan(plan: ExplanationPlan, ledger: EvidenceLedger) -> None:
    available = {fact.fact_id: fact for fact in ledger.facts}
    selected = [
        fact_id for section in plan.sections for fact_id in section.fact_ids
    ]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"plan referenced unknown fact IDs: {unknown}.")
    if len(selected) != len(set(selected)):
        raise ValueError("each evidence fact may appear at most once.")
    missing_required = [
        fact.fact_id
        for fact in ledger.facts
        if fact.priority == "required" and fact.fact_id not in selected
    ]
    if missing_required:
        raise ValueError(
            f"plan omitted required evidence facts: {missing_required}."
        )
    headings = [section.heading for section in plan.sections]
    if len(headings) != len(set(headings)):
        raise ValueError("section headings must not repeat.")


def render_explanation(
    plan: ExplanationPlan,
    ledger: EvidenceLedger,
) -> str:
    validate_plan(plan, ledger)
    by_id = {fact.fact_id: fact for fact in ledger.facts}
    lines = ["# Result explanation"]
    for section in plan.sections:
        lines.extend(("", f"## {section.heading}"))
        lines.extend(
            f"- {by_id[fact_id].text} `[{fact_id}]`"
            for fact_id in section.fact_ids
        )
    return "\n".join(lines)


class FactPreservingExplainer:
    """Let an LLM organize evidence IDs; deterministic code renders the facts."""

    def __init__(
        self,
        llm: StructuredLLM,
        *,
        config: ExplainerConfig,
        system_prompt: str | None = None,
    ):
        self._llm = llm
        self.config = config
        self.system_prompt = system_prompt or load_system_prompt()

    @classmethod
    def from_environment(cls) -> "FactPreservingExplainer":
        config = config_from_environment()
        return cls(build_crewai_llm(config), config=config)

    def explain(self, analysis: AnalyzeResultsResponse) -> ExplanationResult:
        evidence = build_evidence_ledger(analysis)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"evidence_ledger": evidence.model_dump(mode="json")},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]

        last_error_type = "unknown_error"
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                raw = self._llm.call(messages)
                plan = self._validate_plan_output(raw)
                markdown = render_explanation(plan, evidence)
                return ExplanationResult(
                    evidence=evidence,
                    plan=plan,
                    markdown=markdown,
                )
            except Exception as exc:
                last_error_type = type(exc).__name__
                if attempt == self.config.max_attempts:
                    raise ExplanationError(attempt, last_error_type) from None
        raise AssertionError("unreachable")

    @staticmethod
    def _validate_plan_output(raw: object) -> ExplanationPlan:
        if isinstance(raw, ExplanationPlan):
            return raw
        if isinstance(raw, (str, bytes, bytearray)):
            return ExplanationPlan.model_validate_json(raw)
        return ExplanationPlan.model_validate(raw)
