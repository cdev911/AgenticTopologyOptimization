"""Generated JSON-shape and injected-fault checks for total direct boundaries."""
from __future__ import annotations

import json
import math
import unittest
from unittest import mock

from fenitop.tools.contracts import (
    AnalyzeResultsResponse,
    RunTopoptResponse,
    ValidateConfigResponse,
)


class TotalBoundaryTests(unittest.TestCase):
    def test_adversarial_json_shapes_never_escape_or_emit_non_json_values(self):
        from fenitop.tools.analyze_results import analyze_results_tool
        from fenitop.tools.run_topopt import run_topopt_tool
        from fenitop.tools.validate_config import validate_config_tool

        tools = (
            (validate_config_tool, ValidateConfigResponse),
            (run_topopt_tool, RunTopoptResponse),
            (analyze_results_tool, AnalyzeResultsResponse),
        )
        values = (
            None,
            True,
            0,
            1.5,
            "config",
            [],
            [None, {"unexpected": []}],
            {},
            {"config": None},
            {"config": []},
            {"unexpected": {"deep": [{"value": math.nan}]}},
        )
        for tool, response_model in tools:
            for value in values:
                with self.subTest(tool=tool.__name__, value=repr(value)):
                    response = tool(value)
                    response_model.model_validate(response)
                    json.dumps(response, allow_nan=False)
                    self.assertEqual(response["status"], "error")

    def test_unexpected_faults_become_local_log_references_not_tracebacks(self):
        from fenitop.tools.analyze_results import analyze_results_tool
        from fenitop.tools.run_topopt import run_topopt_tool
        from fenitop.tools.validate_config import validate_config_tool

        injections = (
            (
                "fenitop.tools.validate_config._validate_config_impl",
                validate_config_tool,
                {},
            ),
            (
                "fenitop.tools.run_topopt._run_topopt_impl",
                run_topopt_tool,
                {},
            ),
            (
                "fenitop.tools.analyze_results._analyze_results_impl",
                analyze_results_tool,
                {},
            ),
        )
        for target, tool, request in injections:
            with self.subTest(tool=tool.__name__), mock.patch(
                target, side_effect=RuntimeError("private failure details")
            ):
                response = tool(request)
            self.assertEqual(response["status"], "error")
            self.assertEqual(response["errors"][0]["code"], "internal_error")
            serialized = json.dumps(response)
            self.assertNotIn("Traceback", serialized)
            self.assertNotIn("private failure details", serialized)


if __name__ == "__main__":
    unittest.main()
