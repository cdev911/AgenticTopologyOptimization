"""Dolfinx-free tests for Tool 3 (fenitop.tools.analyze_results), using
committed static fixtures rather than results/ (gitignored, won't exist in a
fresh checkout) or a live dolfinx run.

tests/fixtures/non_convergent_run/ is a real run.log + summary.json copied
verbatim from an actual completed beam_2d run (config/beam_2d.json,
max_iter=400) that hit the iteration cap without converging -- change stayed
pinned at the move-limit (0.02) the entire time, never approaching
opt_tol=1e-5. tests/fixtures/convergent_run/ is a real run generated via
Tool 2 against tests/fixtures/smoke_beam_2d.json with a loosened opt_tol=0.05
so it genuinely converges (iterations=3 of a 20 budget) -- a positive control
proving the detector doesn't just always say "not converged".

Runs anywhere (no dolfinx/MPI needed): python -m unittest tests.tools.test_analyze_results
"""
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NON_CONVERGENT_DIR = REPO_ROOT / "tests" / "fixtures" / "non_convergent_run"
CONVERGENT_DIR = REPO_ROOT / "tests" / "fixtures" / "convergent_run"


class AnalyzeResultsFixtureTests(unittest.TestCase):
    def test_non_convergent_fixture_is_correctly_flagged(self):
        from fenitop.tools.analyze_results import analyze_results_tool

        result = analyze_results_tool({
            "output_folder": str(NON_CONVERGENT_DIR), "output_prefix": "beam_2d",
            "opt_tol": 1e-5, "move_limit": 0.02, "make_plots": False,
        })
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["convergence"]["converged"])
        self.assertEqual(result["convergence"]["stop_reason"], "max_iterations_reached")
        self.assertEqual(result["convergence"]["iterations"], 400)
        self.assertGreater(result["convergence"]["fraction_iterations_at_move_limit"], 0.9)

        narrative = result["narrative"]
        self.assertIn("did NOT converge", narrative)
        self.assertIn("move limit", narrative)

    def test_convergent_fixture_has_no_false_alarm(self):
        from fenitop.tools.analyze_results import analyze_results_tool

        result = analyze_results_tool({
            "output_folder": str(CONVERGENT_DIR), "output_prefix": "beam_2d",
            "opt_tol": 0.05, "move_limit": 0.1, "make_plots": False,
        })
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["convergence"]["converged"])
        self.assertEqual(result["convergence"]["stop_reason"], "tolerance_met")
        self.assertEqual(result["convergence"]["iterations"], 3)

        narrative = result["narrative"]
        self.assertNotIn("did NOT converge", narrative)
        self.assertIn("converged:", narrative)

    def test_missing_density_grid_degrades_gracefully(self):
        # Neither fixture has a .npz grid (they predate/bypass Tool 2's
        # render_snapshot step) -- the design-quality checks must skip with a
        # clear warning, not fail the whole analysis.
        from fenitop.tools.analyze_results import analyze_results_tool

        result = analyze_results_tool({
            "output_folder": str(NON_CONVERGENT_DIR), "output_prefix": "beam_2d",
            "opt_tol": 1e-5, "make_plots": False,
        })
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["quality_flags"]["num_components"])
        self.assertTrue(any("density grid" in w for w in result["warnings"]))

    def test_missing_run_log_is_a_clean_request_error(self):
        from fenitop.tools.analyze_results import analyze_results_tool

        result = analyze_results_tool({
            "output_folder": "/nonexistent/path", "output_prefix": "nope", "make_plots": False,
        })
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "request")

    def test_envelope_mode_trusts_tool2s_own_converged_flag(self):
        from fenitop.tools.analyze_results import analyze_results_tool

        envelope = {
            "tool": "run_topopt", "status": "ok", "run_id": "beam_2d_fixture",
            "problem_type": "minimize_compliance", "converged": False,
            "stop_reason": "max_iterations_reached", "iterations": 400,
            "metrics": {"final_compliance": 0.7151, "final_volume": 0.4989,
                        "final_objective": 0.0, "grayness": 0.4775,
                        "final_change": 0.02, "opt_tol": 1e-5},
            "artifacts": [
                {"role": "run_log", "path": str(NON_CONVERGENT_DIR / "beam_2d_run.log")},
                {"role": "summary", "path": str(NON_CONVERGENT_DIR / "beam_2d_summary.json")},
            ],
        }
        result = analyze_results_tool({"run_topopt_envelope": envelope, "make_plots": False})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["convergence"]["converged"], False)
        self.assertEqual(result["convergence"]["iterations"], 400)
        self.assertEqual(result["source"]["run_id"], "beam_2d_fixture")

    def test_make_plots_produces_files_on_disk(self):
        import tempfile
        import shutil
        from fenitop.tools.analyze_results import analyze_results_tool

        tmp_dir = tempfile.mkdtemp(prefix="fenitop_analyze_test_")
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        for name in ("beam_2d_run.log", "beam_2d_summary.json"):
            shutil.copy(NON_CONVERGENT_DIR / name, Path(tmp_dir) / name)

        result = analyze_results_tool({
            "output_folder": tmp_dir, "output_prefix": "beam_2d",
            "opt_tol": 1e-5, "move_limit": 0.02, "make_plots": True,
        })
        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(len(result["plots"]), 3)
        for plot in result["plots"]:
            self.assertTrue(Path(plot["path"]).is_file(), plot)


if __name__ == "__main__":
    unittest.main()
