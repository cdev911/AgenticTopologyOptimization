"""Cost estimation and safety-ceiling policy for agent-triggered runs.

Affordability isn't a validity property, so Tool 1 (validate_config) only
ever *warns* using this module's numbers -- it never blocks status="ok".
Tool 2 (run_topopt) is what actually spends wall-clock/CPU, so it *enforces*
the ceiling by default, with an explicit opt-out for legitimate large runs
(or the documented `mpirun -n N ...` escape hatch for genuinely large jobs).

Thresholds are calibrated against the two real example configs so neither is
rejected by default: beam_2d = 12,000 elements x 400 iters = 4.8e6
(comfortably "low"); mechanism_2d = 40,000 elements x 500 iters = 2.0e7
("medium", still runs). These are a reasoned starting calibration, not a
wall-clock measurement -- revisit once real timing data is available.
"""
from typing import Any, Dict

_WARN_THRESHOLD = 1.0e7
_REJECT_THRESHOLD = 1.0e8
DEFAULT_MAX_COMPLEXITY = _REJECT_THRESHOLD


def estimate_cost(mesh_spec: Dict[str, Any], max_iter: int) -> Dict[str, Any]:
    """Estimate problem size/cost from a config's mesh spec and iteration budget.

    Pure arithmetic on the mesh spec -- does not build a mesh or need dolfinx,
    so it's cheap enough to call from both Tool 1 and Tool 2 on every request.
    """
    nx, ny = mesh_spec["divisions"]
    cell_type = mesh_spec.get("cell_type", "quadrilateral")
    num_elements = nx * ny if cell_type == "quadrilateral" else 2 * nx * ny
    num_nodes = (nx + 1) * (ny + 1)
    displacement_dofs = num_nodes * 2  # 2D vector field, CG1

    complexity_score = num_elements * max(max_iter, 1)
    if complexity_score >= _REJECT_THRESHOLD:
        risk_level = "high"
    elif complexity_score >= _WARN_THRESHOLD:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "num_elements": num_elements,
        "num_nodes": num_nodes,
        "num_design_variables": num_elements,
        "displacement_dofs": displacement_dofs,
        "max_iter": max_iter,
        "complexity_score": complexity_score,
        "risk_level": risk_level,
        "exceeds_default_safety_ceiling": complexity_score >= _REJECT_THRESHOLD,
    }


def effective_ceiling(max_complexity_override: float = None) -> float:
    """The complexity_score above which Tool 2 hard-rejects by default."""
    if max_complexity_override is not None:
        return float(max_complexity_override)
    return _REJECT_THRESHOLD
