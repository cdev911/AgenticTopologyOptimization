"""Numerical integrity checks shared by FEM, filtering, and optimization.

The public tool layer treats :class:`NumericalError` as a stable, expected
failure class.  Keeping the checks beside the native solver operations ensures
that library callers receive the same guarantees as the agent-facing wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from mpi4py import MPI


@dataclass
class NumericalFailure:
    code: str
    component: str
    message: str
    iteration: int | None = None
    reason: int | None = None
    residual_norm: float | None = None


class NumericalError(RuntimeError):
    """A solver/optimizer result failed a numerical trust check."""

    def __init__(self, failure: NumericalFailure):
        super().__init__(failure.message)
        self.failure = failure


def require_finite(
    name: str,
    values: Any,
    *,
    component: str,
    iteration: int | None = None,
    comm=MPI.COMM_WORLD,
) -> None:
    """Reject a scalar/array containing NaN or infinity on any rank."""
    array = np.asarray(values)
    local_ok = bool(np.all(np.isfinite(array)))
    globally_ok = bool(comm.allreduce(local_ok, op=MPI.LAND))
    if not globally_ok:
        raise NumericalError(NumericalFailure(
            code="nonfinite_value",
            component=component,
            iteration=iteration,
            message=f"{component} produced a non-finite value in {name}.",
        ))


def require_density_bounds(
    name: str,
    values: Any,
    *,
    component: str,
    iteration: int | None = None,
    lower: float = 0.0,
    upper: float = 1.0,
    tolerance: float = 1e-10,
    comm=MPI.COMM_WORLD,
) -> None:
    """Reject non-finite density values outside the supported physical range."""
    require_finite(
        name, values, component=component, iteration=iteration, comm=comm
    )
    array = np.asarray(values)
    local_ok = bool(
        np.all(array >= lower - tolerance) and np.all(array <= upper + tolerance)
    )
    globally_ok = bool(comm.allreduce(local_ok, op=MPI.LAND))
    if not globally_ok:
        local_min = float(np.min(array, initial=np.inf))
        local_max = float(np.max(array, initial=-np.inf))
        global_min = float(comm.allreduce(local_min, op=MPI.MIN))
        global_max = float(comm.allreduce(local_max, op=MPI.MAX))
        raise NumericalError(NumericalFailure(
            code="density_out_of_bounds",
            component=component,
            iteration=iteration,
            message=(
                f"{component} produced {name} outside [{lower}, {upper}] "
                f"(observed [{global_min}, {global_max}])."
            ),
        ))


def check_ksp(
    solver: Any,
    *,
    component: str,
    iteration: int | None = None,
) -> None:
    """Require a positive PETSc KSP convergence reason and finite residual."""
    reason = int(solver.getConvergedReason())
    residual_norm = float(solver.getResidualNorm())
    if reason <= 0:
        raise NumericalError(NumericalFailure(
            code="linear_solve_diverged",
            component=component,
            iteration=iteration,
            reason=reason,
            residual_norm=residual_norm if np.isfinite(residual_norm) else None,
            message=(
                f"{component} linear solve did not converge "
                f"(PETSc reason={reason}, residual_norm={residual_norm!r})."
            ),
        ))
    if not np.isfinite(residual_norm):
        raise NumericalError(NumericalFailure(
            code="nonfinite_residual",
            component=component,
            iteration=iteration,
            reason=reason,
            message=f"{component} linear solve reported a non-finite residual norm.",
        ))
