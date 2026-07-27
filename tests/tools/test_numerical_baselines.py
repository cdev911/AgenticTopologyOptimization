"""Pinned, serial characterization tests for both supported problem modes.

These tests intentionally compare numerical quantities with tolerances. They
guard the current solver behavior while the tool boundary is hardened without
pretending that floating-point output is byte-stable across CPU architectures.
"""
import copy
import gc
import hashlib
import json
import logging
import math
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

from fenitop.tools.logreader import read_history
from fenitop.tools.schema import TOOL_CONTRACT_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def _load_json(name):
    with open(FIXTURES / name, "r", encoding="utf-8") as handle:
        return json.load(handle)


class SerialNumericalBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baselines = _load_json("numerical_baselines.json")

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="fenitop_baseline_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def _policy(self, prefix):
        from fenitop.tools.contracts import TrustedRunPolicy

        return TrustedRunPolicy(
            output_root=Path(self.tmp_dir),
            output_prefix=prefix,
            render_snapshot=False,
        )

    def _run_case(self, case_name):
        from fenitop.tools.run_topopt import run_topopt_tool

        expected = self.baselines["cases"][case_name]
        fixture_path = FIXTURES / expected["fixture"]
        self.assertEqual(
            hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            expected["config_sha256"],
            "The fixture changed; review the physics and deliberately refresh its baseline.",
        )
        result = run_topopt_tool(
            {"config": _load_json(expected["fixture"])},
            policy=self._policy(Path(expected["fixture"]).stem),
        )
        self.assertEqual(result["status"], "ok", result.get("error"))
        self.assertEqual(result["contract_version"], self.baselines["tool_contract_version"])
        self.assertEqual(result["contract_version"], TOOL_CONTRACT_VERSION)
        self.assertEqual(result["problem_type"], expected["problem_type"])
        self.assertEqual(result["iterations"], expected["iterations"])
        self.assertEqual(result["stop_reason"], expected["stop_reason"])
        self.assertEqual(result["mma_inner_iteration_warnings"], 0)
        self.assertTrue(result["optimizer_status"]["converged"])
        self.assertIn(result["optimizer_status"]["method"], {"oc", "mma"})
        self.assertEqual(
            result["metrics"]["continuation_completed"],
            expected["continuation_completed"],
        )
        self.assertEqual(result["metrics"]["final_beta"], expected["final_beta"])

        for name, reference in expected["metrics"].items():
            observed = result["metrics"][name]
            self.assertTrue(math.isfinite(observed), f"{name} is not finite: {observed!r}")
            self.assertTrue(
                math.isclose(
                    observed,
                    reference,
                    rel_tol=expected["relative_tolerance"],
                    abs_tol=expected["absolute_tolerance"],
                ),
                f"{case_name}.{name}: observed {observed!r}, expected {reference!r}",
            )

        artifact_by_role = {item["role"]: Path(item["path"]) for item in result["artifacts"]}
        expected_roles = {
            "density_history",
            "density_history_data",
            "displacement_history",
            "displacement_history_data",
            "run_log",
            "summary",
        }
        self.assertEqual(set(artifact_by_role), expected_roles)
        for role, path in artifact_by_role.items():
            self.assertTrue(path.is_file(), f"{role} artifact is missing: {path}")

        history = read_history(artifact_by_role["run_log"])
        self.assertEqual(len(history), expected["iterations"] + 1)
        self.assertEqual(history[0]["state"], "initial")
        self.assertEqual(history[-1]["state"], "iterate")
        self.assertAlmostEqual(
            history[-1]["compliance"], result["metrics"]["final_compliance"], places=12
        )
        self.assertAlmostEqual(
            history[-1]["volume"], result["metrics"]["final_volume"], places=12
        )
        self.assertAlmostEqual(
            history[-1]["objective"], result["metrics"]["final_objective"], places=12
        )
        return result

    def test_compliance_baseline(self):
        result = self._run_case("compliance")
        self.assertLess(abs(result["metrics"]["final_volume"] - 0.4), 1e-4)

    def test_mechanism_baseline(self):
        result = self._run_case("mechanism")
        self.assertLess(result["metrics"]["final_compliance"], 10.0)
        self.assertLess(abs(result["metrics"]["final_volume"] - 0.3), 0.03)

    def test_zero_move_limit_is_rejected_before_solver_execution(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        config = _load_json("smoke_beam_2d.json")
        config["opt"]["move"] = 0.0
        result = run_topopt_tool(
            {"config": config}, policy=self._policy("smoke_beam_2d")
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "request")
        self.assertTrue(
            any(
                error["path"].endswith("opt.minimize_compliance.move")
                for error in result["errors"]
            )
        )

    def test_helmholtz_filter_preserves_a_uniform_density_field(self):
        import numpy as np
        from dolfinx.fem import Function, functionspace
        from dolfinx.mesh import CellType, create_rectangle
        from mpi4py import MPI

        from fenitop.parameterize import DensityFilter

        mesh = create_rectangle(
            MPI.COMM_SELF, [[0.0, 0.0], [2.0, 1.0]], [4, 2], CellType.quadrilateral
        )
        design_space = functionspace(mesh, ("DG", 0))
        physical_space = functionspace(mesh, ("CG", 1))
        design = Function(design_space)
        physical = Function(physical_space)
        design.x.array[:] = 0.37
        design.x.scatter_forward()

        density_filter = DensityFilter(
            MPI.COMM_SELF,
            design,
            physical,
            R=0.2,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        density_filter.forward()
        self.assertTrue(np.all(np.isfinite(physical.x.array)))
        self.assertTrue(np.allclose(physical.x.array, 0.37, rtol=1e-10, atol=1e-10))

    def test_configured_initial_density_is_used(self):
        """The configured design is filtered/projected and evaluated at iteration zero."""
        from fenitop.tools.run_topopt import run_topopt_tool

        config = copy.deepcopy(_load_json("smoke_beam_2d.json"))
        config["opt"]["max_iter"] = 1
        config["opt"]["initial_density"] = 0.2
        result = run_topopt_tool(
            {"config": config}, policy=self._policy("smoke_beam_2d")
        )
        self.assertEqual(result["status"], "ok", result.get("error"))
        run_log = next(
            Path(item["path"]) for item in result["artifacts"] if item["role"] == "run_log"
        )
        initial = read_history(run_log)[0]
        self.assertAlmostEqual(initial["initial_density"], 0.2, places=12)
        self.assertIsNotNone(initial["compliance"])
        self.assertIsNotNone(initial["volume"])
        self.assertLess(initial["volume"], 0.3)

    def test_passive_zones_override_initial_density_and_bounds(self):
        import numpy as np

        from fenitop.topopt import _initial_design_values

        centers = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
        rho, lower, upper = _initial_design_values(
            centers,
            {
                "initial_density": 0.25,
                "solid_zone": lambda x: x[0] == 0.0,
                "void_zone": lambda x: x[0] == 2.0,
            },
        )
        np.testing.assert_allclose(rho, [0.995, 0.25, 0.005])
        np.testing.assert_allclose(lower, [0.99, 0.0, 0.0])
        np.testing.assert_allclose(upper, [1.0, 1.0, 0.01])

    def test_final_density_grid_matches_reported_material_metrics(self):
        import numpy as np

        from fenitop.tools.contracts import TrustedRunPolicy
        from fenitop.tools.run_topopt import run_topopt_tool

        result = run_topopt_tool(
            {"config": _load_json("smoke_beam_2d.json")},
            policy=TrustedRunPolicy(
                output_root=Path(self.tmp_dir),
                output_prefix="state_consistency",
                render_snapshot=True,
            ),
        )
        self.assertEqual(result["status"], "ok", result.get("error"))
        grid_path = next(
            Path(item["path"])
            for item in result["artifacts"]
            if item["role"] == "density_grid"
        )
        with np.load(grid_path, allow_pickle=False) as payload:
            density = payload["density"]
        grayness = float(np.nanmean(4.0 * density * (1.0 - density)))
        self.assertAlmostEqual(grayness, result["metrics"]["grayness"], places=12)
        self.assertAlmostEqual(
            1.0 - grayness,
            result["metrics"]["binarization_score"],
            places=12,
        )

    def test_run_logger_handlers_are_closed_after_repeated_runs(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ResourceWarning)
            for _ in range(2):
                result = run_topopt_tool(
                    {"config": _load_json("smoke_beam_2d.json")},
                    policy=self._policy("cleanup"),
                )
                self.assertEqual(result["status"], "ok", result.get("error"))
                self.assertEqual(logging.getLogger("fenitop_cleanup").handlers, [])
            gc.collect()
        resource_warnings = [
            warning for warning in captured
            if issubclass(warning.category, ResourceWarning)
        ]
        self.assertEqual(resource_warnings, [])

    def test_convergence_requires_an_update_after_final_beta_is_reached(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        config = copy.deepcopy(_load_json("smoke_beam_2d.json"))
        config["opt"].update(
            max_iter=2,
            opt_tol=1.0,
            beta_interval=1,
            beta_max=2,
        )
        result = run_topopt_tool(
            {"config": config}, policy=self._policy("continuation")
        )
        self.assertEqual(result["status"], "ok", result.get("error"))
        self.assertTrue(result["converged"])
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(result["metrics"]["final_beta"], 2.0)
        self.assertTrue(result["metrics"]["continuation_completed"])


if __name__ == "__main__":
    unittest.main()
