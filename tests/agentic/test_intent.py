from __future__ import annotations

import unittest

from pydantic import TypeAdapter, ValidationError

from agentic.intent import InterpretationResult


RESULT_ADAPTER = TypeAdapter(InterpretationResult)


def ready_compliance() -> dict:
    return {
        "status": "ready",
        "intent": {
            "problem_type": "minimize_compliance",
            "domain": {"bounds": [[0, 0], [10, 4]]},
            "material": {"young_modulus": 1.0, "poisson_ratio": 0.3},
            "supports": [
                {
                    "region": {
                        "op": "plane",
                        "axis": "x",
                        "value": 0,
                    }
                }
            ],
            "tractions": [
                {
                    "region": {
                        "op": "plane",
                        "axis": "x",
                        "value": 10,
                    },
                    "vector": [0, -1],
                }
            ],
            "volume_fraction": 0.4,
        },
    }


class IntentResultTests(unittest.TestCase):
    def test_ready_intent_keeps_numerical_preferences_unset(self):
        result = RESULT_ADAPTER.validate_python(ready_compliance())

        self.assertEqual(result.status, "ready")
        self.assertIsNone(result.intent.mesh.divisions)
        self.assertIsNone(result.intent.mesh.cell_type)
        self.assertIsNone(result.intent.optimization.filter_radius)
        self.assertIsNone(result.intent.optimization.max_iter)

        serialized = RESULT_ADAPTER.dump_python(result, mode="json")
        self.assertEqual(serialized["intent"]["domain"]["bounds"], [[0, 0], [10, 4]])
        self.assertEqual(serialized["intent"]["tractions"][0]["vector"], [0, -1])

    def test_optional_preferences_are_preserved_when_user_supplies_them(self):
        payload = ready_compliance()
        payload["intent"]["mesh"] = {
            "divisions": [40, 16],
            "cell_type": "quadrilateral",
        }
        payload["intent"]["optimization"] = {
            "filter_radius": 0.3,
            "max_iter": 100,
        }

        result = RESULT_ADAPTER.validate_python(payload)

        self.assertEqual(result.intent.mesh.divisions, (40, 16))
        self.assertEqual(result.intent.optimization.filter_radius, 0.3)

    def test_relative_edge_segment_is_semantic_and_must_fit_on_edge(self):
        payload = ready_compliance()
        payload["intent"]["tractions"][0].pop("region")
        payload["intent"]["tractions"][0]["edge_segment"] = {
            "edge": "right",
            "center_fraction": 0.5,
            "span_fraction": 0.1,
        }

        result = RESULT_ADAPTER.validate_python(payload)

        segment = result.intent.tractions[0].edge_segment
        self.assertEqual(segment.edge, "right")
        self.assertEqual(segment.center_fraction, 0.5)
        self.assertEqual(segment.span_fraction, 0.1)

        payload["intent"]["tractions"][0]["edge_segment"] = {
            "edge": "right",
            "center_fraction": 0.05,
            "span_fraction": 0.2,
        }
        with self.assertRaises(ValidationError):
            RESULT_ADAPTER.validate_python(payload)

    def test_traction_requires_exactly_one_location_representation(self):
        payload = ready_compliance()
        payload["intent"]["tractions"][0]["edge_segment"] = {
            "edge": "right",
            "center_fraction": 0.5,
            "span_fraction": 0.1,
        }
        with self.assertRaises(ValidationError):
            RESULT_ADAPTER.validate_python(payload)

    def test_mechanism_requires_its_problem_defining_fields(self):
        payload = ready_compliance()
        payload["intent"]["problem_type"] = "compliant_mechanism"

        with self.assertRaises(ValidationError) as caught:
            RESULT_ADAPTER.validate_python(payload)

        locations = {error["loc"] for error in caught.exception.errors()}
        self.assertIn(("ready", "intent", "compliant_mechanism", "input_spring"), locations)
        self.assertIn(("ready", "intent", "compliant_mechanism", "output_spring"), locations)
        self.assertIn(
            ("ready", "intent", "compliant_mechanism", "compliance_bound"),
            locations,
        )

    def test_ready_rejects_invalid_or_incomplete_supported_physics(self):
        cases = []

        missing_support = ready_compliance()
        missing_support["intent"]["supports"] = []
        cases.append(missing_support)

        zero_load = ready_compliance()
        zero_load["intent"]["tractions"][0]["vector"] = [0, 0]
        cases.append(zero_load)

        bad_material = ready_compliance()
        bad_material["intent"]["material"]["poisson_ratio"] = 0.5
        cases.append(bad_material)

        reversed_domain = ready_compliance()
        reversed_domain["intent"]["domain"]["bounds"] = [[10, 0], [0, 4]]
        cases.append(reversed_domain)

        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                RESULT_ADAPTER.validate_python(payload)

    def test_nonzero_body_force_is_a_supported_external_load(self):
        payload = ready_compliance()
        payload["intent"]["tractions"] = []
        payload["intent"]["body_force"] = [0, -0.1]

        result = RESULT_ADAPTER.validate_python(payload)

        self.assertEqual(result.intent.tractions, [])
        self.assertEqual(result.intent.body_force, (0, -0.1))

        payload["intent"]["body_force"] = [0, 0]
        with self.assertRaises(ValidationError):
            RESULT_ADAPTER.validate_python(payload)

    def test_clarification_and_unsupported_results_are_distinct(self):
        clarification = RESULT_ADAPTER.validate_python(
            {
                "status": "needs_clarification",
                "missing_fields": ["material.young_modulus", "tractions[0].vector"],
                "questions": [
                    "What Young's modulus should be used?",
                    "What traction vector acts on the loaded boundary?",
                ],
            }
        )
        unsupported = RESULT_ADAPTER.validate_python(
            {
                "status": "unsupported",
                "unsupported_features": ["3D domain", "roller support"],
                "explanation": "Agent-safe v1 supports only rectangular 2D domains "
                "with full-vector zero clamps.",
            }
        )

        self.assertEqual(clarification.status, "needs_clarification")
        self.assertEqual(unsupported.status, "unsupported")

    def test_all_models_forbid_unknown_fields_and_blank_failure_details(self):
        extra = ready_compliance()
        extra["intent"]["solver_profile"] = "direct"
        with self.assertRaises(ValidationError):
            RESULT_ADAPTER.validate_python(extra)

        with self.assertRaises(ValidationError):
            RESULT_ADAPTER.validate_python(
                {
                    "status": "needs_clarification",
                    "missing_fields": [""],
                    "questions": ["What is missing?"],
                }
            )

        with self.assertRaises(ValidationError):
            RESULT_ADAPTER.validate_python(
                {
                    "status": "unsupported",
                    "unsupported_features": [" "],
                    "explanation": "Unsupported.",
                }
            )


if __name__ == "__main__":
    unittest.main()
