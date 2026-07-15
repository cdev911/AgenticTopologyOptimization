"""Dolfinx-free manifest verification and deterministic Tool 3 tests."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from fenitop.tools.contracts import TrustedAnalysisPolicy, TrustedValidationPolicy
from fenitop.tools.lifecycle import canonical_json_hash
from fenitop.tools.manifest import build_run_manifest, write_run_manifest
from fenitop.tools.validate_config import validate_config_tool

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
NON_CONVERGENT_DIR = FIXTURES / "non_convergent_run"
CONVERGENT_DIR = FIXTURES / "convergent_run"


def _config(converged: bool = False) -> dict:
    with (FIXTURES / "smoke_beam_2d.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if converged:
        config["opt"].update(max_iter=20, opt_tol=0.05, move=0.1)
    else:
        config["opt"].update(max_iter=400, opt_tol=1e-5, move=0.02)
    return config


def _lifecycle(run_id: str, request_hash: str) -> dict:
    return {
        "state": "succeeded",
        "run_id": run_id,
        "request_hash": request_hash,
        "idempotency_key_hash": None,
        "created_at": "2026-07-26T00:00:00+00:00",
        "updated_at": "2026-07-26T00:00:01+00:00",
        "parent_pid": 1,
        "worker_pid": 2,
        "exit_code": 0,
        "terminating_signal": None,
        "timed_out": False,
        "cancelled": False,
        "last_iteration": None,
        "worker_api_key_present": False,
        "message": "fixture",
    }


class AnalyzeResultsManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="fenitop_manifest_analysis_"))
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def _manifest(
        self,
        *,
        converged: bool,
        iterations: int,
        run_id: str = "beam_fixture",
        grid=None,
        grid_payload: dict | None = None,
        empty_history: bool = False,
        malformed_history: bool = False,
        summary_override: dict | None = None,
        metric_override: dict | None = None,
    ) -> dict:
        source = CONVERGENT_DIR if converged else NON_CONVERGENT_DIR
        run_dir = self.tmp_dir / run_id
        run_dir.mkdir()
        log_path = run_dir / "beam_2d_run.log"
        if empty_history:
            log_path.write_text("no history records\n", encoding="utf-8")
        else:
            shutil.copy(source / "beam_2d_run.log", log_path)
            if malformed_history:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write("2026-07-26 INFO history {\"iteration\":\n")
            if metric_override:
                lines = log_path.read_text(encoding="utf-8").splitlines()
                for index in range(len(lines) - 1, -1, -1):
                    marker = "history "
                    if marker not in lines[index]:
                        continue
                    prefix, payload = lines[index].split(marker, 1)
                    record = json.loads(payload)
                    for history_field, metric_field in (
                        ("compliance", "final_compliance"),
                        ("volume", "final_volume"),
                        ("objective", "final_objective"),
                        ("grayness", "grayness"),
                        ("binarization_score", "binarization_score"),
                        ("change", "final_change"),
                    ):
                        if metric_field in metric_override:
                            record[history_field] = metric_override[metric_field]
                    lines[index] = prefix + marker + json.dumps(
                        record, sort_keys=True
                    )
                    break
                log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary = json.loads((source / "beam_2d_summary.json").read_text())
        summary["binarization_score"] = 1.0 - summary["grayness"]
        summary.update(summary_override or {})
        summary_path = run_dir / "beam_2d_summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        artifacts = [
            {
                "role": "run_log",
                "format": "text+jsonlines",
                "path": str(log_path),
                "complete": True,
            },
            {
                "role": "summary",
                "format": "json",
                "path": str(summary_path),
                "complete": True,
            },
        ]
        if grid is not None or grid_payload is not None:
            import numpy as np

            grid_path = run_dir / "beam_2d_density_grid.npz"
            xs = np.linspace(0, 4, 9)
            ys = np.linspace(0, 2, 5)
            np.savez(
                grid_path,
                **(
                    grid_payload
                    if grid_payload is not None
                    else {"density": grid, "x": xs, "y": ys}
                ),
            )
            artifacts.append({
                "role": "density_grid",
                "format": "npz",
                "path": str(grid_path),
                "complete": True,
            })

        config = _config(converged=converged)
        validation = validate_config_tool(
            {"config": config},
            policy=TrustedValidationPolicy(check_geometry=False),
        )
        validation["geometry_report"] = {
            "total_boundary_facets": 24,
            "total_nodes": 45,
            "total_cells": 32,
            "rigid_body_rank": 3,
            "entities": [],
        }
        metrics = {
            "final_compliance": summary["final_compliance"],
            "final_volume": summary["final_volume"],
            "final_objective": summary["final_objective"],
            "grayness": summary["grayness"],
            "binarization_score": summary["binarization_score"],
            "final_change": (
                0.04423493332868389 if converged else 0.020000000000000018
            ),
            "opt_tol": config["opt"]["opt_tol"],
            "final_beta": 128.0 if converged else 4.0,
            "continuation_completed": converged,
        }
        metrics.update(metric_override or {})
        request_hash = canonical_json_hash({"fixture": run_id})
        response = {
            "run_id": run_id,
            "problem_type": "minimize_compliance",
            "converged": converged,
            "stop_reason": (
                "tolerance_met" if converged else "max_iterations_reached"
            ),
            "iterations": iterations,
            "metrics": metrics,
            "optimizer_status": {
                "method": "oc",
                "converged": True,
                "outer_iterations": 1,
                "newton_iterations": 0,
                "line_search_iterations": 0,
                "residual_norm": None,
                "residual_max": None,
            },
            "mma_inner_iteration_warnings": 0,
            "warnings": [],
            "validation": validation,
        }
        manifest = build_run_manifest(
            run_dir=run_dir,
            output_prefix="beam_2d",
            request_hash=request_hash,
            response=response,
            lifecycle=_lifecycle(run_id, request_hash),
            artifacts=artifacts,
        )
        write_run_manifest(run_dir, manifest)
        return manifest.model_dump(mode="json")

    def _analyze(self, manifest: dict, *, plots: bool = False, roots=None):
        from fenitop.tools.analyze_results import analyze_results_tool

        return analyze_results_tool(
            {"run_manifest": manifest},
            policy=TrustedAnalysisPolicy(
                allowed_roots=roots or (self.tmp_dir,),
                make_plots=plots,
            ),
        )

    def test_non_convergent_manifest_is_correctly_diagnosed(self):
        result = self._analyze(
            self._manifest(converged=False, iterations=400)
        )
        self.assertEqual(result["status"], "ok", result["errors"])
        self.assertFalse(result["convergence"]["converged"])
        self.assertTrue(result["convergence"]["iteration_cap_reached"])
        self.assertGreater(
            result["convergence"]["fraction_iterations_at_move_limit"], 0.9
        )
        self.assertTrue(result["convergence"]["move_limit_pinned"])
        self.assertIn("did NOT converge", result["narrative"])

    def test_convergent_manifest_has_constraints_and_no_false_alarm(self):
        result = self._analyze(
            self._manifest(converged=True, iterations=3, run_id="convergent")
        )
        self.assertEqual(result["status"], "ok", result["errors"])
        self.assertTrue(result["convergence"]["converged"])
        self.assertTrue(result["metrics"]["constraints"]["volume_satisfied"])
        self.assertNotIn("did NOT converge", result["narrative"])

    def test_missing_optional_density_grid_is_a_structured_warning(self):
        result = self._analyze(
            self._manifest(converged=False, iterations=400)
        )
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["quality_flags"]["num_components"])
        self.assertTrue(
            any("density grid" in warning["message"].lower()
                for warning in result["warnings"])
        )

    def test_missing_or_checksum_changed_artifact_is_rejected(self):
        for mode in ("missing", "changed"):
            with self.subTest(mode=mode):
                manifest = self._manifest(
                    converged=False,
                    iterations=400,
                    run_id=f"artifact_{mode}",
                )
                log_path = Path(manifest["run_directory"]) / "beam_2d_run.log"
                if mode == "missing":
                    log_path.unlink()
                    expected = "artifact_missing"
                else:
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write("tampered\n")
                    expected = "artifact_size_mismatch"
                result = self._analyze(manifest)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["errors"][0]["code"], expected)

    def test_manifest_hash_and_trusted_root_are_enforced(self):
        manifest = self._manifest(converged=False, iterations=400)
        changed = json.loads(json.dumps(manifest))
        changed["iterations"] = 399
        result = self._analyze(changed)
        self.assertEqual(result["errors"][0]["code"], "manifest_hash_mismatch")

        result = self._analyze(manifest, roots=(self.tmp_dir / "elsewhere",))
        self.assertEqual(result["status"], "error")
        self.assertIn(
            result["errors"][0]["code"],
            {"run_outside_trusted_root", "artifact_validation_failed"},
        )

    def test_empty_history_and_summary_mismatch_fail_cleanly(self):
        empty = self._manifest(
            converged=False,
            iterations=400,
            run_id="empty",
            empty_history=True,
        )
        result = self._analyze(empty)
        self.assertEqual(result["errors"][0]["code"], "history_empty")

        mismatch = self._manifest(
            converged=False,
            iterations=400,
            run_id="mismatch",
            summary_override={"final_compliance": 999.0},
            metric_override={"final_compliance": 0.715146720663673},
        )
        result = self._analyze(mismatch)
        self.assertEqual(result["errors"][0]["code"], "summary_metric_mismatch")

    def test_malformed_history_is_not_silently_skipped(self):
        manifest = self._manifest(
            converged=False,
            iterations=400,
            run_id="malformed_history",
            malformed_history=True,
        )
        result = self._analyze(manifest)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["errors"][0]["code"], "history_record_invalid")

    def test_synthetic_checkerboard_and_connected_design_are_calibrated(self):
        import numpy as np

        checkerboard = np.indices((5, 9)).sum(axis=0) % 2
        result = self._analyze(
            self._manifest(
                converged=False,
                iterations=400,
                run_id="checkerboard",
                grid=checkerboard.astype(float),
                summary_override={"grayness": 0.0, "binarization_score": 1.0},
                metric_override={"grayness": 0.0, "binarization_score": 1.0},
            )
        )
        self.assertEqual(result["status"], "ok", result["errors"])
        self.assertTrue(result["quality_flags"]["checkerboard_detected"])
        self.assertEqual(
            result["quality_flags"]["checkerboard_method"],
            "binary_2x2_alternation_v1",
        )

        connected = np.ones((5, 9), dtype=float)
        result = self._analyze(
            self._manifest(
                converged=False,
                iterations=400,
                run_id="connected",
                grid=connected,
                summary_override={"grayness": 0.0, "binarization_score": 1.0},
                metric_override={"grayness": 0.0, "binarization_score": 1.0},
            )
        )
        self.assertFalse(result["quality_flags"]["checkerboard_detected"])
        self.assertTrue(result["quality_flags"]["load_path_connected"])
        self.assertEqual(len(result["quality_flags"]["connectivity"]), 1)

    def test_npz_keys_shapes_dtypes_coordinates_and_finiteness_are_bounded(self):
        import numpy as np

        valid_grid = np.ones((5, 9), dtype=float)
        xs = np.linspace(0, 4, 9)
        ys = np.linspace(0, 2, 5)
        cases = (
            (
                "keys",
                {"density": valid_grid, "x": xs, "unexpected": ys},
                "density_grid_keys_invalid",
            ),
            (
                "shape",
                {"density": np.ones((4, 9)), "x": xs, "y": ys},
                "density_grid_shape_mismatch",
            ),
            (
                "dtype",
                {"density": np.full((5, 9), "solid"), "x": xs, "y": ys},
                "density_grid_dtype_invalid",
            ),
            (
                "coordinate",
                {"density": valid_grid, "x": xs + 1, "y": ys},
                "density_grid_values_invalid",
            ),
            (
                "nonfinite",
                {
                    "density": np.where(
                        np.indices((5, 9))[0] == 0, np.nan, valid_grid
                    ),
                    "x": xs,
                    "y": ys,
                },
                "density_grid_values_invalid",
            ),
        )
        for name, payload, code in cases:
            with self.subTest(name=name):
                manifest = self._manifest(
                    converged=False,
                    iterations=400,
                    run_id=f"npz_{name}",
                    grid_payload=payload,
                )
                result = self._analyze(manifest)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["errors"][0]["code"], code)

    def test_make_plots_uses_trusted_manifest_prefix(self):
        manifest = self._manifest(
            converged=False, iterations=400, run_id="plots"
        )
        result = self._analyze(manifest, plots=True)
        self.assertEqual(result["status"], "ok", result["errors"])
        self.assertGreaterEqual(len(result["plots"]), 2)
        for plot in result["plots"]:
            path = Path(plot["path"])
            self.assertTrue(path.is_file())
            self.assertTrue(path.name.startswith("beam_2d_"))

    def test_legacy_envelope_and_path_requests_are_rejected(self):
        from fenitop.tools.analyze_results import analyze_results_tool

        for request in (
            {"run_topopt_envelope": {}},
            {"output_folder": str(NON_CONVERGENT_DIR), "output_prefix": "beam_2d"},
        ):
            with self.subTest(request=request):
                result = analyze_results_tool(request)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["stage"], "request")


if __name__ == "__main__":
    unittest.main()
