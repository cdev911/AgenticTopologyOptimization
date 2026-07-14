"""Pure resource estimation and trusted admission policy for serial tool runs.

The estimates are deliberately conservative planning values, not promises about
PETSc's exact allocator behavior. They independently bound mesh size, degrees of
freedom, iteration count, solver work, peak memory, output, and wall time before
Dolfinx constructs a mesh. Thresholds are application-owned and absent from the
agent request schema.
"""
from __future__ import annotations

from typing import Any, Literal

from fenitop.tools.contracts import ResourceLimits
from fenitop.tools.schema import FieldError

DEFAULT_RESOURCE_LIMITS = ResourceLimits()


def _resolved_solver_profile(
    problem_type: str,
    solver_profile: Literal["auto", "iterative", "direct"],
) -> Literal["iterative", "direct"]:
    if solver_profile != "auto":
        return solver_profile
    return "iterative" if problem_type == "minimize_compliance" else "direct"


def estimate_cost(
    mesh_spec: dict[str, Any],
    max_iter: int,
    *,
    problem_type: str = "minimize_compliance",
    solver_profile: Literal["auto", "iterative", "direct"] = "auto",
    output_interval: int = 20,
) -> dict[str, Any]:
    """Estimate serial resource demand without importing or constructing Dolfinx."""
    nx, ny = mesh_spec["divisions"]
    cell_type = mesh_spec.get("cell_type", "quadrilateral")
    cell_multiplier = 1 if cell_type == "quadrilateral" else 2
    num_elements = nx * ny * cell_multiplier
    num_nodes = (nx + 1) * (ny + 1)
    displacement_dofs = num_nodes * 2
    num_design_variables = num_elements
    profile = _resolved_solver_profile(problem_type, solver_profile)

    # One elasticity solve plus density-filter forward/backward. Mechanism mode
    # adds an adjoint elasticity solve. Sparse direct factorization receives a
    # conservative 3x work multiplier relative to the iterative profile.
    linear_solves = 3 if problem_type == "minimize_compliance" else 4
    solver_work_multiplier = 1 if profile == "iterative" else 3
    complexity_score = float(num_elements * max(max_iter, 1))
    # The evaluated-state contract solves iteration zero plus every updated
    # design, so max_iter updates produce max_iter + 1 complete states.
    evaluated_states = max(max_iter, 1) + 1
    work_units = (
        float(num_elements * evaluated_states)
        * linear_solves
        * solver_work_multiplier
    )

    # Sparse matrix/vector + mesh/field planning model. The 12x direct factor
    # multiplier accounts for fill-in in the supported 2D problem sizes.
    matrix_mb = displacement_dofs * 30 * 16 / (1024**2)
    matrix_multiplier = 3 if profile == "iterative" else 12
    mesh_mb = (num_elements * 256 + num_nodes * 128) / (1024**2)
    field_mb = (
        num_design_variables * 10 + num_nodes * 10 + displacement_dofs * 4
    ) * 8 / (1024**2)
    # The pinned Dolfinx/PETSc/MPI/Python stack itself reached roughly 220-232 MiB
    # RSS in medium serial measurements. Include a conservative fixed 256 MiB
    # runtime floor before adding problem-scaled storage.
    estimated_peak_memory_mb = (
        256.0 + mesh_mb + field_mb + matrix_mb * matrix_multiplier
    )

    snapshots = min(max(max_iter, 1) + 1, max(output_interval, 1) + 2)
    per_snapshot_bytes = (num_nodes + displacement_dofs) * 8
    estimated_output_mb = (
        per_snapshot_bytes * snapshots * 2.0
        + num_nodes * 32
        + num_elements * 16
    ) / (1024**2)

    # Calibrated conservatively from pinned serial smoke/medium runs. A fixed
    # setup floor covers imports/mesh creation; work units scale the solve budget.
    estimated_wall_time_seconds = 3.0 + work_units * 2.0e-6

    limits = DEFAULT_RESOURCE_LIMITS
    ratios = [
        num_elements / limits.max_elements,
        num_nodes / limits.max_nodes,
        displacement_dofs / limits.max_displacement_dofs,
        max_iter / limits.max_iterations,
        work_units / limits.max_work_units,
        estimated_peak_memory_mb / limits.max_peak_memory_mb,
        estimated_output_mb / limits.max_output_mb,
        estimated_wall_time_seconds / limits.max_estimated_wall_time_seconds,
    ]
    peak_ratio = max(ratios)
    risk_level = "high" if peak_ratio > 1 else "medium" if peak_ratio >= 0.5 else "low"

    return {
        "cell_type": cell_type,
        "solver_profile": profile,
        "num_elements": num_elements,
        "num_nodes": num_nodes,
        "num_design_variables": num_design_variables,
        "displacement_dofs": displacement_dofs,
        "max_iter": max_iter,
        "evaluated_states": evaluated_states,
        "linear_solves_per_iteration": linear_solves,
        "complexity_score": complexity_score,
        "work_units": work_units,
        "estimated_peak_memory_mb": estimated_peak_memory_mb,
        "estimated_output_mb": estimated_output_mb,
        "estimated_wall_time_seconds": estimated_wall_time_seconds,
        "risk_level": risk_level,
        "exceeds_default_safety_ceiling": peak_ratio > 1,
    }


_LIMIT_FIELDS = (
    ("num_elements", "max_elements", "config.mesh.divisions", "mesh_element_limit"),
    ("num_nodes", "max_nodes", "config.mesh.divisions", "mesh_node_limit"),
    (
        "displacement_dofs",
        "max_displacement_dofs",
        "config.mesh.divisions",
        "displacement_dof_limit",
    ),
    ("max_iter", "max_iterations", "config.opt.max_iter", "iteration_limit"),
    ("work_units", "max_work_units", "config.opt.max_iter", "work_limit"),
    (
        "estimated_peak_memory_mb",
        "max_peak_memory_mb",
        "config.mesh.divisions",
        "memory_limit",
    ),
    ("estimated_output_mb", "max_output_mb", "config.opt.max_iter", "output_limit"),
    (
        "estimated_wall_time_seconds",
        "max_estimated_wall_time_seconds",
        "config.opt.max_iter",
        "estimated_timeout",
    ),
)


def resource_limit_errors(
    estimate: dict[str, Any], limits: ResourceLimits
) -> list[FieldError]:
    """Return stable field-level errors for every exceeded trusted limit."""
    errors: list[FieldError] = []
    for estimate_field, limit_field, path, code in _LIMIT_FIELDS:
        value = estimate[estimate_field]
        limit = getattr(limits, limit_field)
        if value > limit:
            errors.append(
                FieldError(
                    path,
                    code,
                    f"{estimate_field}={value:.6g} exceeds trusted "
                    f"{limit_field}={limit:.6g}.",
                )
            )
    return errors
