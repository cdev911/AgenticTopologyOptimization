"""TH-3 numerical trust checks in the pinned Dolfinx/PETSc runtime."""
from __future__ import annotations

import copy
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def _load_config(name):
    with (FIXTURES / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


class NumericalFailureEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="fenitop_numerical_failure_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def _run_with_failure(self, patch_target, failure, fixture="smoke_beam_2d.json"):
        from fenitop.tools.contracts import TrustedRunPolicy
        from fenitop.tools.run_topopt import _run_topopt_in_process

        policy = TrustedRunPolicy(
            output_root=Path(self.tmp_dir),
            output_prefix="fault",
            render_snapshot=False,
        )
        with mock.patch(patch_target, side_effect=failure):
            return _run_topopt_in_process(
                {"config": _load_config(fixture)}, policy=policy
            )

    def test_elasticity_divergence_is_a_typed_numerical_failure(self):
        from fenitop.numerics import NumericalError, NumericalFailure

        result = self._run_with_failure(
            "fenitop.utility.check_ksp",
            NumericalError(NumericalFailure(
                code="linear_solve_diverged",
                component="elasticity",
                iteration=0,
                reason=-3,
                residual_norm=math.inf,
                message="injected elasticity divergence",
            )),
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "numerical")
        self.assertEqual(result["error"]["code"], "linear_solve_diverged")
        self.assertEqual(result["error"]["component"], "elasticity")
        self.assertEqual(result["error"]["iteration"], 0)

    def test_filter_divergence_is_a_typed_numerical_failure(self):
        from fenitop.numerics import NumericalError, NumericalFailure

        result = self._run_with_failure(
            "fenitop.parameterize.check_ksp",
            NumericalError(NumericalFailure(
                code="linear_solve_diverged",
                component="density_filter_forward",
                iteration=0,
                reason=-5,
                message="injected filter divergence",
            )),
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "numerical")
        self.assertEqual(result["error"]["component"], "density_filter_forward")

    def test_optimizer_nonfinite_failure_is_typed(self):
        from fenitop.numerics import NumericalError, NumericalFailure

        result = self._run_with_failure(
            "fenitop.topopt.optimality_criteria",
            NumericalError(NumericalFailure(
                code="nonfinite_value",
                component="optimality_criteria",
                iteration=1,
                message="injected optimizer failure",
            )),
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "numerical")
        self.assertEqual(result["error"]["component"], "optimality_criteria")

    def test_adjoint_divergence_is_a_typed_numerical_failure(self):
        from fenitop.numerics import NumericalError, NumericalFailure

        def fail_adjoint(_solver, *, component, iteration):
            if component == "adjoint":
                raise NumericalError(NumericalFailure(
                    code="linear_solve_diverged",
                    component=component,
                    iteration=iteration,
                    reason=-7,
                    message="injected adjoint divergence",
                ))

        result = self._run_with_failure(
            "fenitop.utility.check_ksp",
            fail_adjoint,
            fixture="smoke_mechanism_2d.json",
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "numerical")
        self.assertEqual(result["error"]["component"], "adjoint")

    def test_filter_adjoint_divergence_is_a_typed_numerical_failure(self):
        from fenitop.numerics import NumericalError, NumericalFailure

        def fail_filter_adjoint(_solver, *, component, iteration):
            if component == "density_filter_adjoint":
                raise NumericalError(NumericalFailure(
                    code="linear_solve_diverged",
                    component=component,
                    iteration=iteration,
                    reason=-8,
                    message="injected filter-adjoint divergence",
                ))

        result = self._run_with_failure(
            "fenitop.parameterize.check_ksp", fail_filter_adjoint
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "numerical")
        self.assertEqual(result["error"]["component"], "density_filter_adjoint")

    def test_mma_singular_subproblem_is_a_typed_numerical_failure(self):
        result = self._run_with_failure(
            "fenitop.optimize.solve",
            np.linalg.LinAlgError("injected singular matrix"),
            fixture="smoke_mechanism_2d.json",
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "numerical")
        self.assertEqual(result["error"]["code"], "mma_singular_subproblem")

    def test_mma_inner_iteration_cap_is_a_typed_numerical_failure(self):
        from fenitop.numerics import NumericalError, NumericalFailure

        result = self._run_with_failure(
            "fenitop.topopt.mma_optimizer",
            NumericalError(NumericalFailure(
                code="mma_inner_iteration_cap",
                component="mma_subproblem",
                iteration=1,
                residual_norm=1.0,
                message="injected MMA inner-iteration cap",
            )),
            fixture="smoke_mechanism_2d.json",
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "numerical")
        self.assertEqual(result["error"]["code"], "mma_inner_iteration_cap")

    def test_unexpected_solver_exception_is_sanitized(self):
        result = self._run_with_failure(
            "fenitop.topopt.topopt",
            RuntimeError("private injected solver details"),
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "solve")
        self.assertEqual(result["errors"][0]["code"], "unexpected_solver_error")
        serialized = json.dumps(result)
        self.assertNotIn("private injected solver details", serialized)
        self.assertNotIn("Traceback", serialized)


class DirectionalSensitivityTests(unittest.TestCase):
    """Compare filtered/projected analytical gradients to central differences."""

    def _check_case(self, fixture, names):
        from mpi4py import MPI

        from fenitop.config import build_fem_opt
        from fenitop.fem import form_fem
        from fenitop.parameterize import DensityFilter, Heaviside
        from fenitop.sensitivity import Sensitivity
        from fenitop.tools.config_models import compile_solver_config

        config = _load_config(fixture)
        solver_config = compile_solver_config(
            config, output_folder="/tmp", output_prefix="gradient_check"
        )
        fem, opt = build_fem_opt(solver_config, comm=MPI.COMM_SELF)
        linear_problem = density_filter = sensitivity = None
        try:
            linear_problem, u, lam, rho, rho_phys = form_fem(fem, opt)
            density_filter = DensityFilter(
                MPI.COMM_SELF, rho, rho_phys, opt["filter_radius"],
                fem["petsc_options"],
            )
            heaviside = Heaviside(rho_phys)
            sensitivity = Sensitivity(
                MPI.COMM_SELF, opt, linear_problem, u, lam, rho_phys
            )
            size = rho.x.array.size
            base = np.linspace(0.28, 0.52, size)
            direction = np.cos(np.arange(size, dtype=float) + 0.3)
            direction /= np.linalg.norm(direction)
            beta = 2.0

            def evaluate(design, *, gradients=False):
                rho.x.array[:] = design
                rho.x.scatter_forward()
                density_filter.forward(iteration=0)
                heaviside.forward(beta, iteration=0)
                linear_problem.solve_fem(iteration=0)
                values, vectors = sensitivity.evaluate(iteration=0)
                if not gradients:
                    return np.asarray(values, dtype=float)
                heaviside.backward(vectors, iteration=0)
                return values, density_filter.backward(vectors, iteration=0)

            values, analytical = evaluate(base, gradients=True)
            step = 2e-5
            plus = evaluate(base + step * direction)
            minus = evaluate(base - step * direction)
            finite_difference = (plus - minus) / (2.0 * step)

            for index, name in names:
                predicted = float(np.dot(analytical[index], direction))
                observed = float(finite_difference[index])
                self.assertTrue(
                    math.isclose(predicted, observed, rel_tol=3e-3, abs_tol=2e-6),
                    (
                        f"{fixture} {name} directional sensitivity mismatch: "
                        f"analytical={predicted}, finite_difference={observed}, "
                        f"value={values[index]}"
                    ),
                )
        finally:
            if sensitivity is not None:
                sensitivity.close()
            if density_filter is not None:
                density_filter.close()
            if linear_problem is not None:
                linear_problem.close()

    def test_compliance_and_volume_sensitivities(self):
        self._check_case("smoke_beam_2d.json", [(0, "compliance"), (1, "volume")])

    def test_mechanism_objective_constraint_and_volume_sensitivities(self):
        self._check_case(
            "smoke_mechanism_2d.json",
            [(0, "compliance constraint"), (1, "volume"), (2, "mechanism objective")],
        )


class OptimizerGuardTests(unittest.TestCase):
    def test_oc_rejects_a_non_descent_square_root_update(self):
        from fenitop.numerics import NumericalError
        from fenitop.optimize import optimality_criteria

        with self.assertRaises(NumericalError) as caught:
            optimality_criteria(
                rho=np.array([0.5]),
                rho_min=np.array([0.0]),
                rho_max=np.array([1.0]),
                V=0.0,
                dCdrho=np.array([1.0]),
                dVdrho=np.array([1.0]),
                move=0.1,
            )
        self.assertEqual(caught.exception.failure.code, "oc_non_descent_gradient")

    def test_oc_rejects_an_infeasible_volume_linearization(self):
        from fenitop.numerics import NumericalError
        from fenitop.optimize import optimality_criteria

        with self.assertRaises(NumericalError) as caught:
            optimality_criteria(
                rho=np.array([0.5]),
                rho_min=np.array([0.49]),
                rho_max=np.array([1.0]),
                V=0.1,
                dCdrho=np.array([-1.0]),
                dVdrho=np.array([1.0]),
                move=0.1,
            )
        self.assertEqual(
            caught.exception.failure.code, "oc_infeasible_volume_update"
        )


if __name__ == "__main__":
    unittest.main()
