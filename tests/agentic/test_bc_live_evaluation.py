from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agentic.bc_evaluation import (
    grade_boundary_observation,
    load_boundary_evaluation_suite,
)
from agentic.bc_live_evaluation import observation_from_steps
from agentic.boundary_draft import (
    BOUNDARY_FIELDS,
    BoundaryConditionDraft,
    BoundaryDraftState,
    BoundaryFieldFact,
)
from agentic.formulation import ProblemDraft
from scripts.boundary_live_eval import (
    _evaluate_with_transient_retries,
    _messages,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests/fixtures/boundary_condition_scenarios.json"


def _condition(bc_id, kind, values):
    return BoundaryConditionDraft(
        bc_id=bc_id,
        kind=kind,
        created_turn=1,
        facts=tuple(
            BoundaryFieldFact(
                field=field,
                value=value,
                basis="explicit",
                source_turn=1,
                source_quote="fixture",
                rationale="Live normalization fixture.",
            )
            for field, value in sorted(
                values.items(),
                key=lambda item: BOUNDARY_FIELDS.index(item[0]),
            )
        ),
    )


def _step(*conditions, assistant="", questions=(), unsupported=()):
    draft = ProblemDraft(
        boundary_state=BoundaryDraftState(
            conditions=conditions,
            next_support_number=(
                max(
                    [int(item.bc_id[1:]) for item in conditions if item.kind == "support"],
                    default=0,
                )
                + 1
            ),
            next_load_number=(
                max(
                    [int(item.bc_id[1:]) for item in conditions if item.kind == "load"],
                    default=0,
                )
                + 1
            ),
        ),
        turn_count=1,
    )
    return SimpleNamespace(
        session=SimpleNamespace(
            draft=draft,
            unsupported_features=unsupported,
        ),
        turn=SimpleNamespace(
            assistant_message=assistant,
            questions=questions,
        ),
    )


class LiveBoundaryNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = load_boundary_evaluation_suite(FIXTURE)
        cls.scenarios = {item.id: item for item in cls.suite.scenarios}

    def test_complete_resultant_normalizes_to_exact_fixed_contract(self):
        scenario = self.scenarios[
            "resultant_ten_newtons_centered_ten_percent"
        ]
        load = _condition(
            "L1",
            "load",
            {
                "load.kind": "resultant_magnitude",
                "load.magnitude": 10,
                "load.direction": "down",
                "load.unit": "N",
                "load.distribution": "uniform",
                "selector.kind": "centered_fraction",
                "selector.edge": "right",
                "selector.center": 0.5,
                "selector.span": 0.1,
            },
        )

        observation = observation_from_steps(scenario, [_step(load)])
        grade = grade_boundary_observation(
            self.suite, scenario, observation
        )

        self.assertTrue(grade.passed, grade)
        self.assertFalse(observation.solver_started)

    def test_point_load_reformulation_is_capability_limited_not_converted(self):
        scenario = self.scenarios["mathematical_point_load_capability"]
        load = _condition(
            "L1",
            "load",
            {
                "load.kind": "point_force",
                "load.vector": [0, -10],
                "load.unit": "N",
                "load.distribution": "point",
                "selector.kind": "boundary_point",
                "selector.point": [10, 2.5],
            },
        )
        step = _step(
            load,
            assistant=(
                "This exact point load is unsupported. I can offer a finite "
                "patch with a uniform distribution."
            ),
            questions=("Do you confirm that finite patch reformulation?",),
            unsupported=("mathematical_point_load",),
        )

        observation = observation_from_steps(scenario, [step])
        grade = grade_boundary_observation(
            self.suite, scenario, observation
        )

        self.assertTrue(grade.passed, grade)
        self.assertNotIn(
            "point_to_traction_silently", observation.behavior_violations
        )

    def test_nonzero_displacement_details_survive_as_unsupported_semantics(self):
        scenario = self.scenarios[
            "nonzero_prescribed_displacement_remains_unsupported"
        ]
        support = _condition(
            "S1",
            "support",
            {
                "support.direction": "x",
                "support.magnitude": 0.5,
                "support.unit": "mm",
                "selector.kind": "whole_edge",
                "selector.edge": "right",
            },
        )

        observation = observation_from_steps(
            scenario,
            [
                _step(
                    support,
                    unsupported=("nonzero_prescribed_displacement",),
                )
            ],
        )
        grade = grade_boundary_observation(
            self.suite, scenario, observation
        )

        self.assertTrue(grade.passed, grade)
        self.assertEqual(
            observation.boundary_conditions[0].magnitude,
            0.5,
        )

    def test_context_is_supplied_without_an_extra_model_call(self):
        scenario = self.scenarios["move_l1_upward"]
        messages = _messages(scenario)

        self.assertEqual(len(messages), 2)
        self.assertIn("L1 is traction", messages[0])
        self.assertIn("Move L1 upward", messages[1])

    def test_partial_centered_selector_cannot_crash_release_normalization(self):
        scenario = self.scenarios["centered_ten_percent_right_traction"]
        load = _condition(
            "L1",
            "load",
            {
                "load.kind": "traction_vector",
                "load.vector": [0, -1],
                "load.distribution": "uniform",
                "selector.kind": "centered_fraction",
                "selector.edge": "right",
                "selector.span": 0.1,
            },
        )

        observation = observation_from_steps(scenario, [_step(load)])

        self.assertEqual(
            observation.boundary_conditions[0].selector_kind,
            "unspecified_extent",
        )
        self.assertEqual(observation.boundary_conditions[0].span, 0.1)

    def test_boundary_point_can_use_edge_relative_center(self):
        scenario = self.scenarios["point_force_requires_finite_patch"]
        load = _condition(
            "L1",
            "load",
            {
                "load.kind": "point_force",
                "load.magnitude": 100,
                "load.direction": "down",
                "load.unit": "N",
                "load.distribution": "point",
                "selector.kind": "boundary_point",
                "selector.edge": "right",
                "selector.center": 0.5,
            },
        )

        observation = observation_from_steps(
            scenario,
            [
                _step(
                    load,
                    assistant="The exact point load needs a finite uniform patch.",
                    questions=("Would you accept a finite patch?",),
                    unsupported=("mathematical_point_load",),
                )
            ],
        )

        self.assertEqual(
            observation.boundary_conditions[0].selector_kind,
            "boundary_point",
        )

    def test_unknown_edge_canonicalizes_unknown_extent_to_unspecified(self):
        scenario = self.scenarios["support_near_lower_left_needs_extent"]
        support = _condition(
            "S1",
            "support",
            {
                "support.kind": "fixed_all",
                "selector.kind": "unspecified_extent",
                "selector.from_corner": "lower_left",
            },
        )

        observation = observation_from_steps(scenario, [_step(support)])

        self.assertEqual(
            observation.boundary_conditions[0].selector_kind,
            "unspecified",
        )
        self.assertEqual(
            observation.boundary_conditions[0].from_corner,
            "lower_left",
        )

    def test_release_executable_has_no_solver_or_orchestrator_import(self):
        source = (
            REPO_ROOT / "scripts/boundary_live_eval.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("run_topopt", source)
        self.assertNotIn("agentic.orchestrator", source)
        self.assertIn('"solver_executed": False', source)

    def test_release_gate_retries_transport_but_not_semantic_failure(self):
        usage = {
            "api_calls": 1,
            "input_tokens": 10,
            "output_tokens": 2,
            "reasoning_tokens": 1,
            "cached_tokens": 5,
            "latency_seconds": 0.5,
            "context_recoveries": 0,
        }
        transient = {
            "passed": False,
            "error": {
                "kind": "provider",
                "provider_error_type": "APITimeoutError",
            },
            "usage": usage,
        }
        success = {"passed": True, "error": None, "usage": usage}
        with patch(
            "scripts.boundary_live_eval._evaluate",
            side_effect=(transient, success),
        ) as evaluate:
            result = _evaluate_with_transient_retries(
                self.suite,
                self.scenarios["clamp_entire_left_edge"],
                model="test",
                reasoning_effort="medium",
                attempts=3,
            )

        self.assertTrue(result["passed"])
        self.assertEqual(result["transient_attempts"], 2)
        self.assertEqual(result["usage"]["api_calls"], 2)
        self.assertEqual(evaluate.call_count, 2)

        semantic = {"passed": False, "error": None, "usage": usage}
        with patch(
            "scripts.boundary_live_eval._evaluate",
            return_value=semantic,
        ) as evaluate:
            result = _evaluate_with_transient_retries(
                self.suite,
                self.scenarios["clamp_entire_left_edge"],
                model="test",
                reasoning_effort="medium",
                attempts=3,
            )

        self.assertFalse(result["passed"])
        self.assertEqual(result["transient_attempts"], 1)
        self.assertEqual(evaluate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
