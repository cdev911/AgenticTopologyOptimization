"""Billed multi-turn evaluation for the live formulation adapter.

This script calls the OpenAI Responses API, but it never starts the numerical
solver. Drafts that reach review are compiled and validated deterministically.

Run inside the project container:

    docker compose run --rm -T fenitop \
      python scripts/formulation_live_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from agentic.compiler import compile_formulation_draft
from agentic.formulation import (
    ConversationFormulator,
    FormulationSession,
    FormulationStep,
)
from agentic.formulation_openai import (
    FormulationAPIError,
    OpenAIResponsesFormulationAgent,
    build_openai_client,
    config_from_environment,
)
from fenitop.tools.contracts import (
    ValidateConfigRequest,
    ValidateConfigResponse,
)
from fenitop.tools.validate_config import validate_config_tool


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "formulation_scenarios.json"
)


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _fact(step: FormulationStep, path: str):
    fact = step.session.draft.fact(path)
    return fact.value if fact is not None else None


def _bc(step: FormulationStep, bc_id: str):
    return step.session.draft.boundary_state.condition(bc_id)


def _contains_all(text: str, words: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return all(word.casefold() in lowered for word in words)


def _grade(
    scenario: dict,
    steps: list[FormulationStep],
) -> list[dict]:
    final = steps[-1]
    checks = [
        _check(
            "expected_final_status",
            final.session.status == scenario["expected_final_status"],
            (
                f"expected={scenario['expected_final_status']} "
                f"observed={final.session.status}"
            ),
        )
    ]
    scenario_id = scenario["id"]

    if scenario_id == "disordered_cantilever_over_two_turns":
        support = _bc(final, "S1")
        load = _bc(final, "L1")
        load_values = load.values() if load is not None else {}
        checks.extend(
            [
                _check(
                    "domain",
                    (
                        _fact(final, "domain.bounds") == [[0, 0], [10, 5]]
                        or (
                            _fact(final, "domain.origin") == [0, 0]
                            and _fact(final, "domain.width") == 10
                            and _fact(final, "domain.height") == 5
                        )
                    ),
                    f"facts={final.session.draft.values()}",
                ),
                _check(
                    "physics",
                    (
                        _fact(final, "problem_type") == "minimize_compliance"
                        and _fact(final, "material.young_modulus") == 10
                        and _fact(final, "material.poisson_ratio") == 0.3
                        and abs(
                            float(_fact(final, "volume_fraction")) - (1.0 / 3.0)
                        )
                        < 0.01
                    ),
                    "objective, material, and volume must be retained",
                ),
                _check(
                    "support_and_load",
                    support is not None
                    and support.values().get("support.kind") == "fixed_all"
                    and load_values.get("load.kind") == "traction_vector"
                    and load_values.get("selector.kind")
                    == "centered_fraction"
                    and load_values.get("selector.span") == 0.1,
                    "left support and centered 10% traction are required",
                ),
            ]
        )
    elif scenario_id == "correction_overrides_prior_dimension":
        bounds = _fact(final, "domain.bounds")
        if bounds is not None:
            domain = tuple(map(tuple, bounds))
        elif (
            _fact(final, "domain.origin") is not None
            and _fact(final, "domain.width") is not None
            and _fact(final, "domain.height") is not None
        ):
            origin = _fact(final, "domain.origin")
            domain = (
                tuple(origin),
                (
                    origin[0] + _fact(final, "domain.width"),
                    origin[1] + _fact(final, "domain.height"),
                ),
            )
        else:
            domain = None
        related = (
            _fact(final, "material.young_modulus"),
            _fact(final, "material.poisson_ratio"),
            _fact(final, "volume_fraction"),
            _bc(final, "S1") is not None,
            _bc(final, "L1") is not None,
        )
        domain_revisions = [
            revision
            for revision in final.session.draft.revisions
            if revision.path in {"domain.bounds", "domain.width"}
        ]
        checks.extend(
            [
                _check(
                    "corrected_domain",
                    domain == ((0, 0), (12, 5)),
                    f"observed={domain}",
                ),
                _check(
                    "unrelated_facts_retained",
                    related == (100, 0.3, 0.4, True, True),
                    f"observed={related}",
                ),
                _check(
                    "revision_provenance",
                    (
                        len(domain_revisions) >= 2
                        and (
                            (
                                domain_revisions[-1].previous_value
                                == [[0, 0], [10, 5]]
                                and domain_revisions[-1].new_value
                                == [[0, 0], [12, 5]]
                            )
                            or (
                                domain_revisions[-1].previous_value == 10
                                and domain_revisions[-1].new_value == 12
                            )
                        )
                    ),
                    "the domain correction must retain old and new values",
                ),
            ]
        )
    elif scenario_id == "point_load_requires_supported_reformulation":
        response_text = " ".join(
            [
                final.turn.assistant_message,
                *final.turn.questions,
            ]
        )
        checks.extend(
            [
                _check(
                    "no_silent_point_load_conversion",
                    all(
                        condition.values().get("load.kind")
                        not in {
                            "traction_vector",
                            "traction_magnitude",
                            "resultant_vector",
                            "resultant_magnitude",
                        }
                        for condition in final.session.draft.boundary_state.conditions
                        if condition.kind == "load"
                    ),
                    "no finite load may be created before the patch is agreed",
                ),
                _check(
                    "supported_alternative_explained",
                    _contains_all(
                        response_text,
                        ("point", "distributed", "resultant"),
                    ),
                    response_text,
                ),
                _check(
                    "reformulation_question",
                    bool(final.turn.questions)
                    and (
                        "width" in response_text.casefold()
                        or "segment" in response_text.casefold()
                        or "patch" in response_text.casefold()
                    )
                    and (
                        "force" in response_text.casefold()
                        or "resultant" in response_text.casefold()
                    ),
                    response_text,
                ),
            ]
        )
    elif scenario_id == "unsupported_3d_can_be_reformulated":
        checks.extend(
            [
                _check(
                    "first_turn_identifies_3d",
                    steps[0].session.status in {"gathering", "unsupported"}
                    and any(
                        "3d" in feature.casefold()
                        or "three-dimensional" in feature.casefold()
                        for feature in steps[0].session.unsupported_features
                    ),
                    (
                        f"status={steps[0].session.status} "
                        f"features={steps[0].session.unsupported_features}"
                    ),
                ),
                _check(
                    "supported_reformulation_continues",
                    final.session.status == "gathering"
                    and not final.session.unsupported_features,
                    (
                        f"status={final.session.status} "
                        f"features={final.session.unsupported_features}"
                    ),
                ),
            ]
        )
    elif scenario_id == "conflicting_geometry_needs_resolution":
        response_text = " ".join(
            [
                final.turn.assistant_message,
                *final.turn.questions,
            ]
        )
        checks.extend(
            [
                _check(
                    "conflict_explained",
                    (
                        "conflict" in response_text.casefold()
                        or "inconsistent" in response_text.casefold()
                        or _contains_all(response_text, ("10", "12"))
                    ),
                    response_text,
                ),
                _check(
                    "authoritative_value_requested",
                    bool(final.turn.questions)
                    and (
                        "which" in response_text.casefold()
                        or "whether" in response_text.casefold()
                        or "confirm" in response_text.casefold()
                    ),
                    response_text,
                ),
                _check(
                    "prior_width_not_silently_overwritten",
                    (
                        _fact(final, "domain.width") == 10
                        or (
                            _fact(final, "domain.bounds") is not None
                            and (
                                _fact(final, "domain.bounds")[1][0]
                                - _fact(final, "domain.bounds")[0][0]
                            )
                            == 10
                        )
                    ),
                    f"facts={final.session.draft.values()}",
                ),
            ]
        )
    elif scenario_id == "optional_mesh_preference_in_casual_language":
        fact = final.session.draft.fact("mesh.long_short_divisions")
        checks.extend(
            [
                _check(
                    "mesh_preference",
                    fact is not None and fact.value == [60, 30],
                    f"observed={fact.value if fact else None}",
                ),
                _check(
                    "mesh_provenance",
                    fact is not None
                    and fact.basis in {"explicit", "derived"}
                    and fact.source_turn == 1
                    and fact.source_quote is not None,
                    f"observed={fact}",
                ),
                _check(
                    "missing_physics_not_invented",
                    final.finalized_draft is None
                    and _fact(final, "problem_type") is None,
                    f"facts={final.session.draft.values()}",
                ),
            ]
        )

    if final.session.status == "ready_for_review":
        if final.finalized_draft is None:
            checks.append(
                _check(
                    "finalized_draft",
                    False,
                    "ready_for_review is missing finalized draft",
                )
            )
        else:
            compilation = compile_formulation_draft(final.finalized_draft)
            validation = ValidateConfigResponse.model_validate(
                validate_config_tool(
                    ValidateConfigRequest(config=compilation.config)
                )
            )
            checks.append(
                _check(
                    "compile_and_validate",
                    validation.status == "ok",
                    (
                        "validated"
                        if validation.status == "ok"
                        else json.dumps(validation.model_dump(mode="json"))
                    ),
                )
            )
    return checks


def _evaluate_scenario(
    scenario: dict,
    *,
    model: str,
    reasoning_effort: str,
) -> dict:
    base_config = config_from_environment()
    config = replace(
        base_config,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    agent = OpenAIResponsesFormulationAgent(
        build_openai_client(config),
        config=config,
    )
    formulator = ConversationFormulator(agent, max_repair_attempts=1)
    session = FormulationSession()
    steps: list[FormulationStep] = []
    try:
        for user_message in scenario["turns"]:
            step = formulator.advance(session, user_message)
            steps.append(step)
            session = step.session
    except FormulationAPIError as exc:
        return {
            "scenario_id": scenario["id"],
            "passed": False,
            "error": {
                "kind": exc.kind,
                "provider_error_type": exc.provider_error_type,
            },
            "turns": [],
            "checks": [],
            "usage": _usage(agent),
        }

    checks = _grade(scenario, steps)
    transcript = [
        {
            "user": user_message,
            "assistant": step.turn.assistant_message,
            "questions": list(step.turn.questions),
            "status": step.session.status,
            "accepted_paths": list(step.merge.accepted_paths),
            "repair_issues": [
                issue.model_dump(mode="json")
                for issue in step.merge.issues
            ],
            "draft_values": step.session.draft.values(),
        }
        for user_message, step in zip(scenario["turns"], steps)
    ]
    return {
        "scenario_id": scenario["id"],
        "passed": all(check["passed"] for check in checks),
        "error": None,
        "turns": transcript,
        "checks": checks,
        "usage": _usage(agent),
    }


def _usage(agent: OpenAIResponsesFormulationAgent) -> dict:
    records = agent.call_records
    return {
        "api_calls": len(records),
        "input_tokens": sum(item.input_tokens for item in records),
        "output_tokens": sum(item.output_tokens for item in records),
        "reasoning_tokens": sum(item.reasoning_tokens for item in records),
        "cached_tokens": sum(item.cached_tokens for item in records),
        "latency_seconds": round(
            sum(item.latency_seconds for item in records),
            3,
        ),
        "context_recoveries": sum(
            item.recovered_context for item in records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=[config_from_environment().model],
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default=config_from_environment().reasoning_effort,
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Optional scenario IDs; default is the complete fixture.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print aggregate and per-scenario pass/usage without transcripts.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY is required for this billed evaluation.")

    fixture = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    scenarios = fixture["scenarios"]
    if args.scenarios:
        requested = set(args.scenarios)
        scenarios = [
            item for item in scenarios if item["id"] in requested
        ]
        missing = requested - {item["id"] for item in scenarios}
        if missing:
            raise SystemExit(
                "Unknown scenario IDs: " + ", ".join(sorted(missing))
            )

    evaluations = []
    for model in args.models:
        results = [
            _evaluate_scenario(
                scenario,
                model=model,
                reasoning_effort=args.reasoning_effort,
            )
            for scenario in scenarios
        ]
        evaluations.append(
            {
                "model": model,
                "reasoning_effort": args.reasoning_effort,
                "passed": all(result["passed"] for result in results),
                "scenarios_passed": sum(
                    result["passed"] for result in results
                ),
                "scenarios_total": len(results),
                "usage": {
                    key: round(
                        sum(result["usage"][key] for result in results),
                        3,
                    )
                    for key in (
                        "api_calls",
                        "input_tokens",
                        "output_tokens",
                        "reasoning_tokens",
                        "cached_tokens",
                        "latency_seconds",
                        "context_recoveries",
                    )
                },
                "results": results,
            }
        )

    payload = {
        "fixture_version": fixture["version"],
        "solver_executed": False,
        "evaluations": evaluations,
    }
    if args.summary_only:
        payload["evaluations"] = [
            {
                **{
                    key: value
                    for key, value in evaluation.items()
                    if key != "results"
                },
                "results": [
                    {
                        "scenario_id": result["scenario_id"],
                        "passed": result["passed"],
                        "error": result["error"],
                        "failed_checks": [
                            check
                            for check in result["checks"]
                            if not check["passed"]
                        ],
                        "usage": result["usage"],
                    }
                    for result in evaluation["results"]
                ],
            }
            for evaluation in evaluations
        ]
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all(item["passed"] for item in evaluations):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
