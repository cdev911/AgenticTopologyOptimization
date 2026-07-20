from __future__ import annotations

import base64
import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path

from streamlit.testing.v1 import AppTest

from agentic.compiler import compile_intent
from agentic.intent import ComplianceProblemIntent
from agentic.orchestrator import AwaitingRunApproval
from fenitop.tools.contracts import AnalyzeResultsResponse
from fenitop.tools.contracts import ValidateConfigResponse
from fenitop.tools.validate_config import validate_config_tool


REPO_ROOT = Path(__file__).resolve().parents[2]


def _awaiting_approval():
    compilation = compile_intent(
        ComplianceProblemIntent.model_validate(
            {
                "problem_type": "minimize_compliance",
                "domain": {"bounds": [[0, 0], [4, 2]]},
                "material": {"young_modulus": 100, "poisson_ratio": 0.3},
                "supports": [
                    {"region": {"op": "plane", "axis": "x", "value": 0}}
                ],
                "tractions": [
                    {
                        "region": {"op": "plane", "axis": "x", "value": 4},
                        "vector": [0, -1],
                    }
                ],
                "volume_fraction": 0.4,
            }
        )
    )
    validation = ValidateConfigResponse.model_validate(
        validate_config_tool({"config": compilation.config})
    )
    return AwaitingRunApproval(
        conversation={"original_request": "test request"},
        compilation=compilation,
        validation=validation,
        events=(),
    )


class ApprovalOnlyOrchestrator:
    def __init__(self, outcome):
        self.outcome = outcome
        self.start_calls = 0

    def start(self, _message):
        self.start_calls += 1
        return self.outcome


class StreamlitAppTests(unittest.TestCase):
    def test_initial_ui_is_chat_only_and_exposes_no_config_form(self):
        app = AppTest.from_file(
            REPO_ROOT / "streamlit_app.py",
            default_timeout=10,
        ).run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "Agentic Topology Optimization")
        self.assertEqual(len(app.chat_input), 1)
        self.assertEqual(len(app.text_input), 0)
        self.assertEqual(len(app.text_area), 0)
        self.assertEqual(len(app.json), 0)
        self.assertEqual(len(app.chat_message), 1)

    def test_reset_keeps_a_clean_initial_conversation(self):
        app = AppTest.from_file(
            REPO_ROOT / "streamlit_app.py",
            default_timeout=10,
        ).run()

        app.button[0].click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.chat_input), 1)
        self.assertEqual(len(app.chat_message), 1)

    def test_valid_request_waits_in_chat_without_submitting_a_job(self):
        app = AppTest.from_file(
            REPO_ROOT / "streamlit_app.py",
            default_timeout=10,
        ).run()
        orchestrator = ApprovalOnlyOrchestrator(_awaiting_approval())
        app.session_state["orchestrator"] = orchestrator

        app.chat_input[0].set_value("test request").run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(orchestrator.start_calls, 1)
        self.assertEqual(
            app.session_state["outcome"].status,
            "awaiting_run_approval",
        )
        self.assertIsNone(app.session_state["job_future"])
        self.assertIn(
            "Do you approve these parameters",
            app.chat_message[-1].markdown[0].value,
        )

    def test_completed_analysis_renders_verified_plot_gallery(self):
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            density = run_dir / "density.png"
            compliance = run_dir / "compliance.png"
            png = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lE"
                "QVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            )
            density.write_bytes(png)
            compliance.write_bytes(png)
            analysis = AnalyzeResultsResponse.model_validate(
                {
                    "contract_version": "4.0.0",
                    "tool": "analyze_results",
                    "status": "ok",
                    "warnings": [],
                    "errors": [],
                    "source": {
                        "run_directory": str(run_dir),
                        "output_prefix": "compliance_2d",
                        "run_id": "compliance_2d_test",
                        "manifest_hash": "a" * 64,
                    },
                    "plots": [
                        {"role": "density_field", "path": str(density)},
                        {
                            "role": "compliance_vs_iteration",
                            "path": str(compliance),
                        },
                    ],
                }
            )
            app = AppTest.from_file(
                REPO_ROOT / "streamlit_app.py",
                default_timeout=10,
            ).run()
            app.session_state["outcome"] = SimpleNamespace(
                events=(),
                analysis=analysis,
            )

            app.run()

            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.image), 2)
            self.assertEqual(len(app.download_button), 2)
            self.assertEqual(
                [button.label for button in app.download_button],
                [
                    "Download Final optimized design",
                    "Download Compliance objective vs. iteration",
                ],
            )


if __name__ == "__main__":
    unittest.main()
