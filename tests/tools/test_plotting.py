from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fenitop.tools.plotting import plot_convergence


class PlottingTests(unittest.TestCase):
    def test_mechanism_history_gets_signed_objective_plot(self):
        history = [
            {
                "state": "iterate",
                "iteration": 1,
                "compliance": 2.0,
                "objective": -0.1,
                "volume": 0.3,
                "change": 0.05,
            },
            {
                "state": "iterate",
                "iteration": 2,
                "compliance": 1.8,
                "objective": -0.2,
                "volume": 0.3,
                "change": 0.02,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            plots = plot_convergence(history, temporary, "mechanism")

            roles = {plot["role"] for plot in plots}
            self.assertIn("objective_vs_iteration", roles)
            objective = next(
                Path(plot["path"])
                for plot in plots
                if plot["role"] == "objective_vs_iteration"
            )
            self.assertTrue(objective.is_file())

    def test_zero_compliance_placeholder_does_not_get_objective_plot(self):
        history = [
            {
                "state": "iterate",
                "iteration": 1,
                "compliance": 2.0,
                "objective": 0.0,
                "volume": 0.4,
                "change": 0.02,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            plots = plot_convergence(history, temporary, "compliance")

        self.assertNotIn(
            "objective_vs_iteration",
            {plot["role"] for plot in plots},
        )


if __name__ == "__main__":
    unittest.main()
