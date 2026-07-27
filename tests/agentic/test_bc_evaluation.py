from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from agentic.bc_evaluation import (
    BC_EVALUATION_VERSION,
    BoundaryEvaluationSuite,
    BoundaryObservation,
    ExpectedBoundaryCondition,
    grade_boundary_observation,
    load_boundary_evaluation_suite,
)


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "boundary_condition_scenarios.json"
)


def exact_observation(scenario, **updates) -> BoundaryObservation:
    expected = scenario.expected
    values = {
        "semantic_state": expected.semantic_state,
        "boundary_conditions": expected.boundary_conditions,
        "clarifications": expected.clarifications,
        "assumptions": expected.assumptions,
        "capability_limits": expected.capability_limits,
        "behavior_violations": (),
        "solver_started": False,
    }
    values.update(updates)
    return BoundaryObservation(**values)


class BoundaryCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = load_boundary_evaluation_suite(FIXTURE)

    def test_versioned_corpus_has_planned_size_and_family_coverage(self):
        self.assertEqual(self.suite.version, BC_EVALUATION_VERSION)
        self.assertGreaterEqual(len(self.suite.scenarios), 50)
        counts = Counter(scenario.family for scenario in self.suite.scenarios)
        self.assertEqual(
            set(counts),
            {
                "support_aliases",
                "support_segments",
                "distributed_loads",
                "resultant_loads",
                "multiple_and_corrections",
                "ambiguity_and_conflicts",
                "capability_boundaries",
            },
        )
        self.assertTrue(all(count >= 6 for count in counts.values()), counts)

    def test_corpus_contains_real_failures_and_multi_turn_corrections(self):
        by_id = {scenario.id: scenario for scenario in self.suite.scenarios}
        required = {
            "reported_coordinate_segment_complete_prompt",
            "reported_whole_edge_complete_prompt",
            "supplied_cantilever_transcript_retains_partial_resultant",
            "resultant_correction_one_to_ten",
            "move_l1_upward",
            "roller_support_requires_component_contract",
            "true_corner_pin_requires_node_selector",
        }
        self.assertTrue(required <= set(by_id))
        transcript = by_id[
            "supplied_cantilever_transcript_retains_partial_resultant"
        ]
        self.assertGreaterEqual(len(transcript.turns), 6)
        load = transcript.expected.boundary_conditions[1]
        self.assertEqual(load.bc_id, "L1")
        self.assertEqual(load.load_kind, "resultant_magnitude")
        self.assertEqual(load.magnitude, 10)
        self.assertIsNone(load.direction)
        self.assertIn("load_direction", transcript.expected.clarifications)

    def test_global_safety_rules_forbid_every_observed_failure_class(self):
        forbidden = set(self.suite.global_forbidden_behaviors)
        self.assertTrue(
            {
                "solver_start",
                "discard_partial_boundary_condition",
                "invent_boundary_extent",
                "invent_load_direction",
                "invent_units",
                "point_to_traction_silently",
                "resultant_to_traction_silently",
                "roller_to_clamp",
                "pin_to_clamp",
                "overwrite_unrelated_boundary_condition",
            }
            <= forbidden
        )

    def test_scenario_and_boundary_ids_are_unique_and_stable(self):
        scenario_ids = [scenario.id for scenario in self.suite.scenarios]
        self.assertEqual(len(scenario_ids), len(set(scenario_ids)))
        for scenario in self.suite.scenarios:
            with self.subTest(scenario=scenario.id):
                boundary_ids = [
                    bc.bc_id for bc in scenario.expected.boundary_conditions
                ]
                self.assertEqual(len(boundary_ids), len(set(boundary_ids)))
                for bc in scenario.expected.boundary_conditions:
                    prefix = "S" if bc.kind == "support" else "L"
                    self.assertTrue(bc.bc_id.startswith(prefix))

    def test_loader_rejects_duplicate_scenario_ids(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["scenarios"][1]["id"] = payload["scenarios"][0]["id"]
        with self.assertRaisesRegex(ValidationError, "scenario IDs must be unique"):
            BoundaryEvaluationSuite.model_validate(payload)

    def test_boundary_schema_rejects_wrong_id_kind_and_invalid_fraction(self):
        with self.assertRaisesRegex(ValidationError, "must start with S"):
            ExpectedBoundaryCondition(
                bc_id="L1",
                kind="support",
                support_kind="fixed_all",
                edge="left",
                selector_kind="whole_edge",
            )
        with self.assertRaisesRegex(ValidationError, r"must lie in \[0,1\]"):
            ExpectedBoundaryCondition(
                bc_id="L1",
                kind="load",
                load_kind="traction_vector",
                edge="right",
                selector_kind="centered_fraction",
                center=0.5,
                span=1.2,
                vector=(0, -1),
            )


class BoundaryGraderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = load_boundary_evaluation_suite(FIXTURE)
        cls.by_id = {
            scenario.id: scenario for scenario in cls.suite.scenarios
        }

    def test_exact_semantic_observation_passes(self):
        scenario = self.by_id["resultant_ten_newtons_centered_ten_percent"]
        result = grade_boundary_observation(
            self.suite,
            scenario,
            exact_observation(scenario),
        )
        self.assertTrue(result.passed, result.checks)
        self.assertTrue(all(check.passed for check in result.checks))

    def test_grader_detects_lost_partial_load_fact(self):
        scenario = self.by_id[
            "supplied_cantilever_transcript_retains_partial_resultant"
        ]
        observation = exact_observation(
            scenario,
            boundary_conditions=(scenario.expected.boundary_conditions[0],),
        )
        result = grade_boundary_observation(
            self.suite, scenario, observation
        )
        self.assertFalse(result.passed)
        failed = {check.name for check in result.checks if not check.passed}
        self.assertEqual(failed, {"boundary_conditions"})

    def test_grader_detects_silent_resultant_conversion(self):
        scenario = self.by_id[
            "resultant_ten_newtons_centered_ten_percent"
        ]
        observation = exact_observation(
            scenario,
            behavior_violations=("resultant_to_traction_silently",),
        )
        result = grade_boundary_observation(
            self.suite, scenario, observation
        )
        self.assertFalse(result.passed)
        self.assertFalse(
            next(
                check
                for check in result.checks
                if check.name == "forbidden_behaviors"
            ).passed
        )

    def test_grader_detects_extra_or_missing_clarification(self):
        scenario = self.by_id["scalar_load_ten_has_unknown_quantity"]
        observation = exact_observation(
            scenario,
            clarifications=("load_quantity_kind",),
        )
        result = grade_boundary_observation(
            self.suite, scenario, observation
        )
        self.assertFalse(result.passed)
        self.assertFalse(
            next(
                check
                for check in result.checks
                if check.name == "clarifications"
            ).passed
        )

    def test_grader_always_rejects_solver_start(self):
        scenario = self.by_id["whole_right_traction_vector"]
        result = grade_boundary_observation(
            self.suite,
            scenario,
            exact_observation(scenario, solver_started=True),
        )
        self.assertFalse(result.passed)
        self.assertFalse(
            next(
                check
                for check in result.checks
                if check.name == "solver_not_started"
            ).passed
        )


if __name__ == "__main__":
    unittest.main()
