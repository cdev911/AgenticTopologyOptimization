"""Tool 1: strict structural and optional Dolfinx-backed geometry validation."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from fenitop.tools.config_models import compute_warnings, translate_validation_error
from fenitop.tools.contracts import (
    TrustedValidationPolicy,
    ValidateConfigRequest,
    ValidateConfigResponse,
)
from fenitop.tools.logging_config import get_logger
from fenitop.tools.safety import estimate_cost
from fenitop.tools.schema import FieldError, error_envelope, ok_envelope

logger = get_logger(__name__)


def load_raw_config(config: Any) -> dict:
    """Return a JSON-shaped config without granting filesystem read authority."""
    if isinstance(config, BaseModel):
        return config.model_dump(mode="json")
    if isinstance(config, dict):
        return config
    raise TypeError("config must be a JSON object; paths and source strings are not accepted.")


def _rigid_body_rank(constrained_rows: list[tuple[float, float, int]]) -> int:
    """Rank of the three planar rigid-body modes resisted by fixed-vector rows."""
    import numpy as np

    if not constrained_rows:
        return 0
    xs = np.array([row[0] for row in constrained_rows])
    ys = np.array([row[1] for row in constrained_rows])
    center_x, center_y = xs.mean(), ys.mean()
    matrix = [
        (
            [1.0, 0.0, -(y - center_y)]
            if component == 0
            else [0.0, 1.0, x - center_x]
        )
        for x, y, component in constrained_rows
    ]
    singular_values = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    tolerance = 1e-8 * max(float(singular_values[0]), 1.0)
    return int(np.sum(singular_values > tolerance))


def _check_geometry(config: dict) -> tuple[list[FieldError], list[str]]:
    import numpy as np
    from dolfinx.mesh import locate_entities_boundary
    from mpi4py import MPI

    from fenitop.config import build_mesh
    from fenitop.regions import compile_region

    errors: list[FieldError] = []
    warnings: list[str] = []
    mesh = build_mesh(config["mesh"], comm=MPI.COMM_SELF)
    facet_dimension = mesh.topology.dim - 1
    mesh.topology.create_connectivity(facet_dimension, 0)
    facet_to_vertex = mesh.topology.connectivity(facet_dimension, 0)
    total_facets = mesh.topology.index_map(facet_dimension).size_local

    def facet_vertex_coords(facets):
        indices = set()
        for facet in facets:
            indices.update(
                int(vertex) for vertex in facet_to_vertex.links(int(facet))
            )
        return mesh.geometry.x[sorted(indices)]

    def check_marker(path: str, marker_spec):
        try:
            marker = compile_region(marker_spec)
            facets = locate_entities_boundary(mesh, facet_dimension, marker)
        except Exception as exc:
            errors.append(
                FieldError(
                    path,
                    "geometry_marker_error",
                    f"Region failed while matching boundary facets: {exc}",
                )
            )
            return None
        if len(facets) == 0:
            errors.append(
                FieldError(
                    path,
                    "region_matches_no_facets",
                    f"Region matched 0 of {total_facets} boundary facets.",
                )
            )
            return None
        return facets

    constrained_rows = []
    for index, bc in enumerate(config["fem"]["dirichlet_bcs"]):
        facets = check_marker(
            f"config.fem.dirichlet_bcs[{index}].marker", bc["marker"]
        )
        if facets is not None:
            for x, y, *_ in facet_vertex_coords(facets):
                constrained_rows.extend([(x, y, 0), (x, y, 1)])

    for index, bc in enumerate(config["fem"].get("traction_bcs", [])):
        check_marker(
            f"config.fem.traction_bcs[{index}].locator", bc["locator"]
        )

    if constrained_rows:
        rank = _rigid_body_rank(constrained_rows)
        if rank < 3:
            errors.append(
                FieldError(
                    "config.fem.dirichlet_bcs",
                    "rigid_body_modes_unconstrained",
                    f"Fixed supports resist only rank {rank}/3 planar rigid-body modes.",
                )
            )
    elif not errors:
        errors.append(
            FieldError(
                "config.fem.dirichlet_bcs",
                "no_support_facets",
                "No fixed support matched a boundary facet.",
            )
        )
    return errors, warnings


def _response(data: dict) -> dict:
    ValidateConfigResponse.model_validate(data)
    return data


def validate_config_tool(
    request: dict | ValidateConfigRequest,
    *,
    policy: TrustedValidationPolicy | None = None,
) -> dict:
    """Validate an AgentSafeConfig; execution controls come from trusted code."""
    validation_policy = policy or TrustedValidationPolicy()
    logger.info(
        "validate_config: starting (check_geometry=%s)",
        validation_policy.check_geometry,
    )
    try:
        parsed_request = (
            request
            if isinstance(request, ValidateConfigRequest)
            else ValidateConfigRequest.model_validate(request)
        )
    except ValidationError as exc:
        result = error_envelope(
            "validate_config",
            translate_validation_error(exc),
            stage="structural_validation",
            checked={"structural": True, "geometry": False},
            normalized_config=None,
        )
        return _response(result)
    except Exception as exc:
        result = error_envelope(
            "validate_config",
            [FieldError("<root>", "malformed_request", str(exc))],
            stage="structural_validation",
            checked={"structural": True, "geometry": False},
            normalized_config=None,
        )
        return _response(result)

    model = parsed_request.config
    normalized = model.model_dump(mode="json")
    warnings = compute_warnings(model)
    cost = estimate_cost(normalized["mesh"], model.opt.max_iter)
    if cost["risk_level"] != "low":
        warnings.append(
            f"estimated_cost.complexity_score={cost['complexity_score']:.3g} "
            f"is in the '{cost['risk_level']}' risk band."
        )

    errors: list[FieldError] = []
    geometry_ran = False
    if validation_policy.check_geometry:
        geometry_ran = True
        geometry_errors, geometry_warnings = _check_geometry(normalized)
        errors.extend(geometry_errors)
        warnings.extend(geometry_warnings)

    checked = {"structural": True, "geometry": geometry_ran}
    if errors:
        return _response(
            error_envelope(
                "validate_config",
                errors,
                stage="geometry_validation",
                warnings=warnings,
                checked=checked,
                problem_type=model.opt.problem_type,
                normalized_config=None,
                estimated_cost=cost,
            )
        )
    return _response(
        ok_envelope(
            "validate_config",
            warnings=warnings,
            checked=checked,
            problem_type=model.opt.problem_type,
            normalized_config=normalized,
            estimated_cost=cost,
        )
    )


def main() -> int:
    from fenitop.tools.cli import run_cli

    return run_cli(
        validate_config_tool,
        "Validate a versioned AgentSafeConfig JSON request.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
