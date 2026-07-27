"""Dolfinx-free Tool 3 tests using typed Tool 2 envelopes over static artifacts."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from fenitop.tools.contracts import TrustedAnalysisPolicy, TrustedValidationPolicy
from fenitop.tools.schema import TOOL_CONTRACT_VERSION
from fenitop.tools.validate_config import validate_config_tool

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
NON_CONVERGENT_DIR = FIXTURES / "non_convergent_run"
CONVERGENT_DIR = FIXTURES / "convergent_run"


def _config(converged=False):
    with open(FIXTURES / "smoke_beam_2d.json", encoding="utf-8") as handle:
        config = json.load(handle)
    if converged:
        config["opt"].update(max_iter=20, opt_tol=0.05, move=0.1)
    else:
        config["opt"].update(max_iter=400, opt_tol=1e-5, move=0.02)
    return config


def _envelope(directory, *, converged, iterations, run_id="beam_fixture"):
    config = _config(converged=converged)
    validation = validate_config_tool(
        {"config": config},
        policy=TrustedValidationPolicy(check_geometry=False),
    )
    prefix = "beam_2d"
    return {
        "contract_version": TOOL_CONTRACT_VERSION,
        "tool": "run_topopt",
        "status": "ok",
        "warnings": [],
        "errors": [],
        "run_id": run_id,
        "problem_type": "minimize_compliance",
        "converged": converged,
        "stop_reason": "tolerance_met" if converged else "max_iterations_reached",
        "iterations": iterations,
        "metrics": {
            "final_compliance": 0.7151,
            "final_volume": 0.4989,
            "final_objective": 0.0,
            "grayness": 0.4775,
            "binarization_score": 0.5225,
            "final_change": 0.04 if converged else 0.02,
            "opt_tol": config["opt"]["opt_tol"],
            "final_beta": 128.0 if converged else 4.0,
            "continuation_completed": converged,
        },
        "mma_inner_iteration_warnings": 0,
        "wall_time_seconds": 1.0,
        "artifacts": [
            {
                "role": "run_log",
                "format": "text+jsonlines",
                "path": str(directory / f"{prefix}_run.log"),
            },
            {
                "role": "summary",
                "format": "json",
                "path": str(directory / f"{prefix}_summary.json"),
            },
        ],
        "validation": validation,
    }


class AnalyzeResultsFixtureTests(unittest.TestCase):
    def _analyze(self, envelope, *, plots=False, allowed_roots=None):
        from fenitop.tools.analyze_results import analyze_results_tool

        roots = allowed_roots or tuple(
            {Path(artifact["path"]).parent for artifact in envelope["artifacts"]}
        )
        return analyze_results_tool(
            {"run_topopt_envelope": envelope},
            policy=TrustedAnalysisPolicy(allowed_roots=roots, make_plots=plots),
        )

    def test_non_convergent_fixture_is_correctly_flagged(self):
        result = self._analyze(
            _envelope(
                NON_CONVERGENT_DIR,
                converged=False,
                iterations=400,
            )
        )
        self.assertEqual(result["status"], "ok", result["errors"])
        self.assertFalse(result["convergence"]["converged"])
        self.assertEqual(result["convergence"]["iterations"], 400)
        self.assertGreater(
            result["convergence"]["fraction_iterations_at_move_limit"], 0.9
        )
        self.assertIn("did NOT converge", result["narrative"])

    def test_convergent_fixture_has_no_false_alarm(self):
        result = self._analyze(
            _envelope(CONVERGENT_DIR, converged=True, iterations=3)
        )
        self.assertEqual(result["status"], "ok", result["errors"])
        self.assertTrue(result["convergence"]["converged"])
        self.assertEqual(result["convergence"]["iterations"], 3)
        self.assertNotIn("did NOT converge", result["narrative"])

    def test_missing_density_grid_degrades_with_structured_warning(self):
        result = self._analyze(
            _envelope(NON_CONVERGENT_DIR, converged=False, iterations=400)
        )
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["quality_flags"]["num_components"])
        self.assertTrue(
            any("density grid" in warning["message"] for warning in result["warnings"])
        )

    def test_missing_run_log_is_a_clean_request_error(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="fenitop_missing_log_"))
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        envelope = _envelope(tmp_dir, converged=False, iterations=1)
        result = self._analyze(envelope, allowed_roots=(tmp_dir,))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["errors"][0]["code"], "run_log_missing")

    def test_artifacts_outside_trusted_root_are_rejected(self):
        envelope = _envelope(
            NON_CONVERGENT_DIR, converged=False, iterations=400
        )
        envelope["artifacts"][0]["path"] = "/etc/passwd"
        result = self._analyze(
            envelope, allowed_roots=(NON_CONVERGENT_DIR,)
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["errors"][0]["code"], "artifact_outside_trusted_root"
        )

    def test_envelope_values_are_authoritative(self):
        result = self._analyze(
            _envelope(
                NON_CONVERGENT_DIR,
                converged=False,
                iterations=400,
                run_id="authoritative",
            )
        )
        self.assertFalse(result["convergence"]["converged"])
        self.assertEqual(result["source"]["run_id"], "authoritative")

    def test_make_plots_produces_files(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="fenitop_analyze_test_"))
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        for name in ("beam_2d_run.log", "beam_2d_summary.json"):
            shutil.copy(NON_CONVERGENT_DIR / name, tmp_dir / name)
        result = self._analyze(
            _envelope(tmp_dir, converged=False, iterations=400),
            plots=True,
        )
        self.assertEqual(result["status"], "ok", result["errors"])
        self.assertGreaterEqual(len(result["plots"]), 2)
        for plot in result["plots"]:
            self.assertTrue(Path(plot["path"]).is_file(), plot)

    def test_legacy_path_request_is_rejected(self):
        from fenitop.tools.analyze_results import analyze_results_tool

        result = analyze_results_tool(
            {
                "output_folder": str(NON_CONVERGENT_DIR),
                "output_prefix": "beam_2d",
            }
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "request")


if __name__ == "__main__":
    unittest.main()
