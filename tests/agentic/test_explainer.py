from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from agentic.explainer import (
    EvidenceLedger,
    ExplanationError,
    ExplanationPlan,
    ExplainerConfig,
    FactPreservingExplainer,
    build_crewai_llm,
    build_evidence_ledger,
    config_from_environment,
    load_system_prompt,
    render_explanation,
    validate_plan,
)
from fenitop.tools.contracts import AnalyzeResultsResponse
from fenitop.tools.schema import TOOL_CONTRACT_VERSION


def analysis_response():
    return AnalyzeResultsResponse.model_validate(
        {
            "contract_version": TOOL_CONTRACT_VERSION,
            "tool": "analyze_results",
            "status": "ok",
            "warnings": [],
            "errors": [],
            "source": {
                "run_directory": "/workspace/results/example",
                "output_prefix": "beam",
                "run_id": "example",
                "manifest_hash": "a" * 64,
            },
            "convergence": {
                "converged": True,
                "stop_reason": "tolerance_met",
                "iterations": 20,
                "final_change": 0.001,
                "opt_tol": 0.01,
                "fraction_iterations_at_move_limit": 0.1,
                "move_limit": 0.02,
                "final_beta": 128,
                "continuation_completed": True,
                "iteration_cap_reached": False,
                "move_limit_pinned": False,
                "oscillation_detected": False,
                "plateau_detected": False,
                "optimizer_warning_count": 0,
            },
            "quality_flags": {
                "grayness": 0.1,
                "binarization_score": 0.9,
                "grayness_threshold": 0.4,
                "high_grayness_warning": False,
                "checkerboard_detected": False,
                "checkerboard_score": 0.0,
                "num_components": 1,
                "largest_component_fraction": 1.0,
                "has_disconnected_material": False,
                "load_path_connected": True,
                "checkerboard_method": "fixture",
                "connectivity_method": "fixture",
                "connectivity": [],
            },
            "metrics": {
                "final_compliance": 1.25,
                "final_volume": 0.4,
                "final_objective": 0.0,
                "constraints": {
                    "volume_target": 0.4,
                    "volume_error": 0.0,
                    "volume_tolerance": 0.01,
                    "volume_satisfied": True,
                    "compliance_bound": None,
                    "compliance_bound_satisfied": None,
                    "density_bounds_satisfied": True,
                },
            },
            "narrative": "The deterministic result passed its checks.",
        }
    )


def valid_plan():
    return {
        "sections": [
            {"heading": "Outcome", "fact_ids": ["F001", "F003"]},
            {"heading": "Convergence", "fact_ids": ["F002", "F006"]},
            {"heading": "Constraints", "fact_ids": ["F004"]},
            {"heading": "Quality", "fact_ids": ["F005", "F007"]},
        ]
    }


class FakeLLM:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.messages = []

    def call(self, messages):
        self.messages.append(messages)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class ExplainerTests(unittest.TestCase):
    def config(self, max_attempts=2):
        return ExplainerConfig(model="test-model", max_attempts=max_attempts)

    def test_evidence_ledger_contains_exact_required_analysis_facts(self):
        ledger = build_evidence_ledger(analysis_response())

        self.assertIsInstance(ledger, EvidenceLedger)
        self.assertEqual(
            [fact.fact_id for fact in ledger.facts],
            [f"F{index:03d}" for index in range(1, 9)],
        )
        self.assertEqual(
            [fact.fact_id for fact in ledger.facts if fact.priority == "required"],
            ["F001", "F002", "F003", "F004", "F005"],
        )
        self.assertEqual(ledger.facts[2].text, (
            "Final compliance objective: 1.25; final volume: 0.4."
        ))
        self.assertNotIn("/workspace/results", ledger.model_dump_json())

    def test_model_selects_ids_and_deterministic_renderer_owns_all_prose(self):
        llm = FakeLLM([valid_plan()])
        result = FactPreservingExplainer(
            llm,
            config=self.config(),
            system_prompt="organize IDs only",
        ).explain(analysis_response())

        self.assertIn("# Result explanation", result.markdown)
        self.assertIn(
            "- Final compliance objective: 1.25; final volume: 0.4. `[F003]`",
            result.markdown,
        )
        self.assertNotIn("final objective: 0.0", result.markdown)
        sent = json.loads(llm.messages[0][1]["content"])
        self.assertIn("evidence_ledger", sent)
        self.assertNotIn("run_directory", sent)

    def test_mechanism_ledger_names_the_signed_output_objective(self):
        analysis = analysis_response()
        metrics = analysis.metrics.model_copy(update={
            "final_objective": -0.125,
            "constraints": analysis.metrics.constraints.model_copy(update={
                "compliance_bound": 2.0,
                "compliance_bound_satisfied": True,
            }),
        })
        ledger = build_evidence_ledger(
            analysis.model_copy(update={"metrics": metrics})
        )
        self.assertIn("signed output objective: -0.125", ledger.facts[2].text)
        self.assertNotIn("compliance objective", ledger.facts[2].text)

    def test_unknown_or_omitted_required_ids_are_rejected(self):
        ledger = build_evidence_ledger(analysis_response())
        unknown = ExplanationPlan.model_validate(
            {"sections": [{"heading": "Outcome", "fact_ids": ["F999"]}]}
        )
        omitted = ExplanationPlan.model_validate(
            {"sections": [{"heading": "Outcome", "fact_ids": ["F001"]}]}
        )

        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_plan(unknown, ledger)
        with self.assertRaisesRegex(ValueError, "omitted required"):
            validate_plan(omitted, ledger)

    def test_invalid_plan_retries_once_then_succeeds(self):
        llm = FakeLLM(
            [
                {"sections": [{"heading": "Outcome", "fact_ids": ["F999"]}]},
                ExplanationPlan.model_validate(valid_plan()),
            ]
        )
        result = FactPreservingExplainer(
            llm,
            config=self.config(),
            system_prompt="contract",
        ).explain(analysis_response())

        self.assertEqual(len(llm.messages), 2)
        self.assertIn("Final compliance", result.markdown)

    def test_failures_are_bounded_and_provider_details_are_sanitized(self):
        llm = FakeLLM(
            [RuntimeError("secret provider detail"), RuntimeError("another secret")]
        )
        explainer = FactPreservingExplainer(
            llm,
            config=self.config(),
            system_prompt="contract",
        )

        with self.assertRaises(ExplanationError) as caught:
            explainer.explain(analysis_response())

        self.assertEqual(caught.exception.attempts, 2)
        self.assertEqual(caught.exception.last_error_type, "RuntimeError")
        self.assertNotIn("secret", str(caught.exception))

    def test_prompt_and_crewai_configuration_limit_model_authority(self):
        prompt = load_system_prompt()
        self.assertIn("Select fact", prompt)
        self.assertIn("IDs only", prompt)
        self.assertIn("Do not rewrite facts", prompt)
        self.assertIn('priority is "required"', prompt)

        with mock.patch.dict(os.environ, {"OPENAI_MODEL": "chosen-model"}):
            config = config_from_environment()
        with mock.patch("agentic.explainer.LLM") as llm_factory:
            build_crewai_llm(config)

        kwargs = llm_factory.call_args.kwargs
        self.assertEqual(kwargs["model"], "openai/chosen-model")
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertIs(kwargs["response_format"], ExplanationPlan)
        self.assertNotIn("temperature", kwargs)

    def test_unsuccessful_analysis_cannot_be_explained(self):
        failed = analysis_response().model_copy(
            update={"status": "error", "source": None}
        )
        with self.assertRaises(ValueError):
            build_evidence_ledger(failed)


if __name__ == "__main__":
    unittest.main()
