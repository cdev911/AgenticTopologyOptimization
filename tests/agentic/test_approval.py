from __future__ import annotations

import unittest

from agentic.approval import (
    classify_run_approval,
    format_run_approval_request,
)
from agentic.compiler import compile_intent
from agentic.intent import ComplianceProblemIntent
from fenitop.tools.contracts import ValidateConfigResponse
from fenitop.tools.validate_config import validate_config_tool


def _compilation():
    intent = ComplianceProblemIntent.model_validate(
        {
            "problem_type": "minimize_compliance",
            "domain": {"bounds": [[0, 0], [10, 4]]},
            "material": {"young_modulus": 100, "poisson_ratio": 0.3},
            "supports": [
                {"region": {"op": "plane", "axis": "x", "value": 0}}
            ],
            "tractions": [
                {
                    "edge_segment": {
                        "edge": "right",
                        "center_fraction": 0.5,
                        "span_fraction": 1.0,
                    },
                    "vector": [0, -1],
                }
            ],
            "volume_fraction": 0.4,
        }
    )
    return compile_intent(intent)


class ApprovalTests(unittest.TestCase):
    def test_only_unambiguous_whole_message_is_a_green_light(self):
        for message in ("yes", "Yes!", "start the run", "go ahead", "run it"):
            self.assertEqual(classify_run_approval(message), "approve")
        for message in ("no", "not yet", "do not run"):
            self.assertEqual(classify_run_approval(message), "reject")
        for message in (
            "yes, but use 20 iterations",
            "make the mesh 40 x 20",
            "maybe",
        ):
            self.assertEqual(
                classify_run_approval(message),
                "request_changes",
            )

    def test_approval_prompt_shows_exact_parameters_and_explicit_question(self):
        compilation = _compilation()
        validation = ValidateConfigResponse.model_validate(
            validate_config_tool({"config": compilation.config})
        )

        message = format_run_approval_request(compilation, validation)

        self.assertIn("values were not provided", message)
        self.assertIn("Mesh: 79 × 32", message)
        self.assertIn("Material: E=100", message)
        self.assertIn("Material fraction: 0.4", message)
        self.assertIn("Do you approve these parameters", message)
        self.assertIn("Reply **yes** to start", message)


if __name__ == "__main__":
    unittest.main()
