"""Docker-only smoke test for Tool 2 (fenitop.tools.run_topopt).

Needs the full dolfinx/PETSc/MPI stack -- run via:
    docker compose run --rm fenitop python -m unittest tests.tools.test_run_topopt -v

Uses tests/fixtures/smoke_beam_2d.json (32 elements, max_iter=5) rather than
the full example configs specifically so this suite runs in well under a
second instead of the ~90s+ a real 400-iteration example takes.
"""
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_smoke_config():
    with open(REPO_ROOT / "tests" / "fixtures" / "smoke_beam_2d.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


class RunTopoptSmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="fenitop_test_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def test_smoke_run_succeeds_with_expected_artifacts(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        result = run_topopt_tool({
            "config": _load_smoke_config(),
            "output_root": self.tmp_dir,
            "render_snapshot": True,
        })
        self.assertEqual(result["status"], "ok", result.get("error"))
        self.assertIn("run_id", result)
        self.assertEqual(result["iterations"], 5)
        self.assertIsNotNone(result["metrics"]["final_compliance"])

        artifact_roles = {a["role"] for a in result["artifacts"]}
        self.assertIn("run_log", artifact_roles)
        self.assertIn("summary", artifact_roles)
        self.assertIn("density_history", artifact_roles)
        for artifact in result["artifacts"]:
            self.assertTrue(Path(artifact["path"]).is_file(), f"missing artifact: {artifact}")

    def test_two_runs_get_distinct_scoped_output_directories(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        config = _load_smoke_config()
        r1 = run_topopt_tool({"config": config, "output_root": self.tmp_dir, "render_snapshot": False})
        r2 = run_topopt_tool({"config": config, "output_root": self.tmp_dir, "render_snapshot": False})
        self.assertEqual(r1["status"], "ok")
        self.assertEqual(r2["status"], "ok")
        self.assertNotEqual(r1["run_id"], r2["run_id"])
        dir1 = Path(r1["artifacts"][0]["path"]).parent
        dir2 = Path(r2["artifacts"][0]["path"]).parent
        self.assertNotEqual(dir1, dir2)
        self.assertTrue(dir1.is_dir())
        self.assertTrue(dir2.is_dir())

    def test_scoped_output_false_reproduces_flat_overwrite_layout(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        config = _load_smoke_config()
        result = run_topopt_tool({
            "config": config, "output_root": self.tmp_dir, "scoped_output": False,
            "render_snapshot": False,
        })
        self.assertEqual(result["status"], "ok")
        run_log = next(a for a in result["artifacts"] if a["role"] == "run_log")
        self.assertEqual(Path(run_log["path"]).parent, Path(self.tmp_dir))

    def test_degenerate_config_returns_error_envelope_not_a_crash(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        config = _load_smoke_config()
        config["mesh"]["bounds"] = [[0, 0], [0, 0]]  # zero-size domain
        result = run_topopt_tool({"config": config, "output_root": self.tmp_dir})
        self.assertEqual(result["status"], "error")
        self.assertIn(result["stage"], ("pre_flight_validation", "solve"))

    def test_oversized_config_is_rejected_near_instantly(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        config = _load_smoke_config()
        config["mesh"]["divisions"] = [2000, 2000]
        config["opt"]["max_iter"] = 50000
        start = time.perf_counter()
        result = run_topopt_tool({"config": config, "output_root": self.tmp_dir})
        elapsed = time.perf_counter() - start
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "safety_check")
        self.assertLess(elapsed, 5.0, "safety_check rejection should be near-instant; topopt() must not run")

    def test_max_complexity_override_can_reject_even_a_tiny_problem(self):
        # Deliberately does NOT inflate mesh/max_iter -- lowering the ceiling
        # below the smoke config's own tiny complexity_score (160) is enough,
        # and keeps this test fast regardless of outcome.
        from fenitop.tools.run_topopt import run_topopt_tool

        result = run_topopt_tool({
            "config": _load_smoke_config(), "output_root": self.tmp_dir,
            "max_complexity_override": 1,
        })
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "safety_check")

    def test_allow_large_run_overrides_an_artificially_lowered_ceiling(self):
        # allow_large_run=True bypasses the (artificially lowered) ceiling,
        # and the underlying problem is still the tiny 32-element/5-iteration
        # smoke config -- this exercises the override path without ever
        # attempting to actually solve a genuinely huge problem.
        from fenitop.tools.run_topopt import run_topopt_tool

        result = run_topopt_tool({
            "config": _load_smoke_config(), "output_root": self.tmp_dir,
            "max_complexity_override": 1, "allow_large_run": True, "render_snapshot": False,
        })
        self.assertEqual(result["status"], "ok", result.get("error"))


if __name__ == "__main__":
    unittest.main()
