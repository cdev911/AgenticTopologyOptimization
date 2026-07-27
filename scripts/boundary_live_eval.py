"""Billed 53-case first-increment boundary-conversation release gate.

This executable calls only the live formulation adapter.  It normalizes the
application-owned draft through ``agentic.bc_live_evaluation`` and grades it
against the fixed provider-independent corpus.  It imports no orchestrator or
solver entry point.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

from agentic.bc_evaluation import (
    grade_boundary_observation,
    load_boundary_evaluation_suite,
)
from agentic.bc_live_evaluation import observation_from_steps
from agentic.formulation import ConversationFormulator, FormulationSession
from agentic.formulation_openai import (
    FormulationAPIError,
    OpenAIResponsesFormulationAgent,
    build_openai_client,
    config_from_environment,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "boundary_condition_scenarios.json"
)


def _usage(agent: OpenAIResponsesFormulationAgent) -> dict:
    records = agent.call_records
    return {
        "api_calls": len(records),
        "input_tokens": sum(item.input_tokens for item in records),
        "output_tokens": sum(item.output_tokens for item in records),
        "reasoning_tokens": sum(item.reasoning_tokens for item in records),
        "cached_tokens": sum(item.cached_tokens for item in records),
        "latency_seconds": round(
            sum(item.latency_seconds for item in records), 3
        ),
        "context_recoveries": sum(item.recovered_context for item in records),
    }


def _messages(scenario) -> tuple[str, ...]:
    turns = list(scenario.turns)
    context = scenario.context
    if context and "no canonical facts" not in context.casefold():
        if any(
            token in context
            for token in ("L1 is", "L2 is", "S1 clamps")
        ):
            turns.insert(0, context)
        else:
            turns[0] = (
                "Established conversation context:\n"
                f"{context}\n\nCurrent user request:\n{turns[0]}"
            )
    return tuple(turns)


def _evaluate(suite, scenario, *, model: str, reasoning_effort: str) -> dict:
    config = replace(
        config_from_environment(),
        model=model,
        reasoning_effort=reasoning_effort,
    )
    agent = OpenAIResponsesFormulationAgent(
        build_openai_client(config),
        config=config,
    )
    formulator = ConversationFormulator(agent, max_repair_attempts=1)
    session = FormulationSession()
    steps = []
    messages = _messages(scenario)
    try:
        for message in messages:
            step = formulator.advance(session, message)
            steps.append(step)
            session = step.session
    except FormulationAPIError as exc:
        return {
            "scenario_id": scenario.id,
            "passed": False,
            "error": {
                "kind": exc.kind,
                "provider_error_type": exc.provider_error_type,
            },
            "grade": None,
            "observation": None,
            "usage": _usage(agent),
        }

    try:
        observation = observation_from_steps(scenario, steps)
    except Exception as exc:
        return {
            "scenario_id": scenario.id,
            "passed": False,
            "error": {
                "kind": "normalization_error",
                "provider_error_type": type(exc).__name__,
                "message": str(exc),
            },
            "grade": None,
            "observation": None,
            "turns": [
                {
                    "user": message,
                    "assistant": step.turn.assistant_message,
                    "questions": list(step.turn.questions),
                    "status": step.session.status,
                }
                for message, step in zip(messages, steps)
            ],
            "usage": _usage(agent),
        }
    grade = grade_boundary_observation(suite, scenario, observation)
    return {
        "scenario_id": scenario.id,
        "passed": grade.passed,
        "error": None,
        "grade": grade.model_dump(mode="json"),
        "observation": observation.model_dump(mode="json"),
        "turns": [
            {
                "user": message,
                "assistant": step.turn.assistant_message,
                "questions": list(step.turn.questions),
                "status": step.session.status,
                "boundary_state": step.session.draft.boundary_state.model_dump(
                    mode="json"
                ),
                "unsupported_features": list(
                    step.session.unsupported_features
                ),
                "merge_issues": [
                    issue.model_dump(mode="json")
                    for issue in step.merge.issues
                ],
            }
            for message, step in zip(messages, steps)
        ],
        "usage": _usage(agent),
    }


def _evaluate_with_transient_retries(
    suite,
    scenario,
    *,
    model: str,
    reasoning_effort: str,
    attempts: int,
) -> dict:
    """Retry transport failures, never semantic failures."""
    usage_keys = (
        "api_calls",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "latency_seconds",
        "context_recoveries",
    )
    total_usage = {key: 0 for key in usage_keys}
    result = None
    for attempt in range(1, attempts + 1):
        result = _evaluate(
            suite,
            scenario,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        for key in usage_keys:
            total_usage[key] += result["usage"][key]
        error = result.get("error") or {}
        transient = (
            error.get("kind") == "provider"
            and error.get("provider_error_type")
            in {"APIConnectionError", "APITimeoutError"}
        )
        if not transient or attempt == attempts:
            break
    result["usage"] = {
        key: round(value, 3)
        for key, value in total_usage.items()
    }
    result["transient_attempts"] = attempt
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=config_from_environment().model,
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default=config_from_environment().reasoning_effort,
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Optional fixed scenario IDs; default is all 53.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit passing transcripts and observations.",
    )
    parser.add_argument(
        "--transient-attempts",
        type=int,
        default=3,
        help="Attempts per scenario for connection/timeouts only (default: 3).",
    )
    args = parser.parse_args()
    if args.transient_attempts < 1:
        raise SystemExit("--transient-attempts must be at least 1.")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY is required for this billed evaluation.")

    suite = load_boundary_evaluation_suite(SCENARIOS_PATH)
    scenarios = list(suite.scenarios)
    if args.scenarios:
        requested = set(args.scenarios)
        scenarios = [item for item in scenarios if item.id in requested]
        missing = requested - {item.id for item in scenarios}
        if missing:
            raise SystemExit(
                "Unknown scenario IDs: " + ", ".join(sorted(missing))
            )

    results = [
        _evaluate_with_transient_retries(
            suite,
            scenario,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            attempts=args.transient_attempts,
        )
        for scenario in scenarios
    ]
    usage = {
        key: round(sum(item["usage"][key] for item in results), 3)
        for key in (
            "api_calls",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_tokens",
            "latency_seconds",
            "context_recoveries",
        )
    }
    payload = {
        "fixture_version": suite.version,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "passed": all(item["passed"] for item in results),
        "scenarios_passed": sum(item["passed"] for item in results),
        "scenarios_total": len(results),
        "solver_executed": False,
        "usage": usage,
        "results": results,
    }
    if args.summary_only:
        payload["results"] = [
            (
                {
                    "scenario_id": item["scenario_id"],
                    "passed": True,
                    "transient_attempts": item["transient_attempts"],
                    "usage": item["usage"],
                }
                if item["passed"]
                else item
            )
            for item in results
        ]
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
