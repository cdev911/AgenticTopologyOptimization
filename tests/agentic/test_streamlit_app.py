from __future__ import annotations

import base64
import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path

from streamlit.testing.v1 import AppTest

from fenitop.tools.contracts import AnalyzeResultsResponse


REPO_ROOT = Path(__file__).resolve().parents[2]


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
