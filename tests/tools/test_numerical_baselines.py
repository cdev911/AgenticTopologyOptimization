"""Pinned, serial characterization tests for both supported problem modes.

These tests intentionally compare numerical quantities with tolerances. They
guard the current solver behavior while the tool boundary is hardened without
pretending that floating-point output is byte-stable across CPU architectures.
"""
import copy
import hashlib
import json
import math
import shutil
import tempfile
import unittest
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
        self.assertLess(abs(result["metrics"]["final_volume"] - 0.3), 0.01)

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

    @unittest.expectedFailure
    def test_configured_initial_density_is_used(self):
        """Desired semantics; expected to fail until the known TH-3 bug is fixed."""
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


if __name__ == "__main__":
    unittest.main()
