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
from unittest import mock

from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_smoke_config():
    with open(REPO_ROOT / "tests" / "fixtures" / "smoke_beam_2d.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


class RunTopoptSmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="fenitop_test_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def _policy(self, **updates):
        from fenitop.tools.contracts import TrustedRunPolicy

        return TrustedRunPolicy(
            output_root=Path(self.tmp_dir),
            output_prefix="smoke_beam_2d",
            **updates,
        )

    def test_smoke_run_succeeds_with_expected_artifacts(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        result = run_topopt_tool(
            {"config": _load_smoke_config()},
            policy=self._policy(render_snapshot=True),
        )
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
        r1 = run_topopt_tool(
            {"config": config}, policy=self._policy(render_snapshot=False)
        )
        r2 = run_topopt_tool(
            {"config": config}, policy=self._policy(render_snapshot=False)
        )
        self.assertEqual(r1["status"], "ok")
        self.assertEqual(r2["status"], "ok")
        self.assertNotEqual(r1["run_id"], r2["run_id"])
        dir1 = Path(r1["artifacts"][0]["path"]).parent
        dir2 = Path(r2["artifacts"][0]["path"]).parent
        self.assertNotEqual(dir1, dir2)
        self.assertTrue(dir1.is_dir())
        self.assertTrue(dir2.is_dir())

    def test_flat_overwrite_policy_is_no_longer_representable(self):
        with self.assertRaises(ValidationError):
            self._policy(scoped_output=False, render_snapshot=False)

    def test_degenerate_config_returns_error_envelope_not_a_crash(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        config = _load_smoke_config()
        config["mesh"]["bounds"] = [[0, 0], [0, 0]]  # zero-size domain
        result = run_topopt_tool({"config": config}, policy=self._policy())
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "request")

    def test_oversized_config_is_rejected_near_instantly(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        config = _load_smoke_config()
        config["mesh"]["divisions"] = [2000, 2000]
        config["opt"]["max_iter"] = 1
        start = time.perf_counter()
        result = run_topopt_tool({"config": config}, policy=self._policy())
        elapsed = time.perf_counter() - start
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "safety_check")
        self.assertLess(elapsed, 5.0, "safety_check rejection should be near-instant; topopt() must not run")

    def test_trusted_resource_policy_can_reject_even_a_tiny_problem(self):
        # Deliberately does NOT inflate mesh/max_iter -- lowering the work limit
        # below the smoke config's own tiny estimate is enough,
        # and keeps this test fast regardless of outcome.
        from fenitop.tools.run_topopt import run_topopt_tool

        from fenitop.tools.contracts import ResourceLimits

        result = run_topopt_tool(
            {"config": _load_smoke_config()},
            policy=self._policy(
                resource_limits=ResourceLimits(max_work_units=1)
            ),
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "safety_check")

    def test_large_run_safety_override_is_no_longer_representable(self):
        from fenitop.tools.contracts import ResourceLimits

        with self.assertRaises(ValidationError):
            self._policy(
                resource_limits=ResourceLimits(max_work_units=1),
                allow_large_run=True,
                render_snapshot=False,
            )

    def test_timeout_budget_participates_in_preflight_admission(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        result = run_topopt_tool(
            {"config": _load_smoke_config()},
            policy=self._policy(timeout_seconds=0.1),
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "safety_check")
        self.assertIn("estimated_timeout", {e["code"] for e in result["errors"]})

    def test_optional_renderer_failure_does_not_invalidate_the_solve(self):
        from fenitop.tools.run_topopt import _run_topopt_in_process

        with mock.patch(
            "fenitop.tools.run_topopt._render_snapshot",
            return_value=(
                {"density_snapshot_png": None, "density_grid_npz": None},
                ["Failed to render optional snapshot (injected)."],
            ),
        ):
            result = _run_topopt_in_process(
                {"config": _load_smoke_config()},
                policy=self._policy(render_snapshot=True),
            )

        self.assertEqual(result["status"], "ok", result.get("errors"))
        self.assertTrue(
            any("optional snapshot" in warning["message"]
                for warning in result["warnings"])
        )
        roles = {artifact["role"] for artifact in result["artifacts"]}
        self.assertNotIn("density_snapshot_png", roles)
        self.assertNotIn("density_grid", roles)


if __name__ == "__main__":
    unittest.main()
