from __future__ import annotations

import json
import unittest
from pathlib import Path


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "formulation_scenarios.json"
)


class FormulationScenarioTests(unittest.TestCase):
    def test_eval_seed_contains_diverse_multi_turn_requirements(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        scenarios = payload["scenarios"]
        ids = [item["id"] for item in scenarios]

        self.assertEqual(payload["version"], "formulation-evals-v2")
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(scenarios), 6)
        self.assertTrue(any(len(item["turns"]) > 1 for item in scenarios))
        self.assertIn("point_load_requires_supported_reformulation", ids)
        self.assertIn("conflicting_geometry_needs_resolution", ids)
        self.assertIn("correction_overrides_prior_dimension", ids)
        for scenario in scenarios:
            with self.subTest(scenario=scenario["id"]):
                self.assertTrue(scenario["purpose"].strip())
                self.assertTrue(all(turn.strip() for turn in scenario["turns"]))
                self.assertTrue(scenario["required_behaviors"])


if __name__ == "__main__":
    unittest.main()
