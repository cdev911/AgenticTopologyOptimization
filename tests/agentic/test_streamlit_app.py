from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


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


if __name__ == "__main__":
    unittest.main()
