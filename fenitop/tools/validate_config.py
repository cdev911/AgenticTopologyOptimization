"""Tool 1: strict structural, resource, semantic, and mesh-backed validation."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from fenitop.tools.config_models import (
    AgentSafeConfig,
    compute_warnings,
    translate_validation_error,
)
from fenitop.tools.contracts import (
    TrustedValidationPolicy,
    ValidateConfigRequest,
    ValidateConfigResponse,
)
from fenitop.tools.logging_config import get_logger
from fenitop.tools.safety import estimate_cost, resource_limit_errors
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


def _semantic_errors(model: AgentSafeConfig) -> list[FieldError]:
    """Cross-field rules that need more than one independently valid field."""
    errors: list[FieldError] = []
    nonzero_tractions = [
        index
        for index, bc in enumerate(model.fem.traction_bcs)
        if any(float(component) != 0.0 for component in bc.value)
    ]
    if (
        not nonzero_tractions
        and all(float(component) == 0.0 for component in model.fem.body_force)
    ):
        errors.append(
            FieldError(
                "config.fem",
                "external_load_required",
                "At least one nonzero distributed traction or body force is required.",
            )
        )
    for index, bc in enumerate(model.fem.traction_bcs):
        if all(float(component) == 0.0 for component in bc.value):
            errors.append(
                FieldError(
                    f"config.fem.traction_bcs[{index}].value",
                    "zero_traction",
                    "A distributed traction entry must have a nonzero vector.",
                )
            )
    if model.opt.problem_type == "compliant_mechanism":
        for name in ("in_spring", "out_spring"):
            spring = getattr(model.opt, name)
            ratio = float(spring.stiffness) / float(model.fem.young_modulus)
            if ratio < 1e-8 or ratio > 100:
                errors.append(
                    FieldError(
                        f"config.opt.{name}.stiffness",
                        "spring_material_scale",
                        f"stiffness / young_modulus = {ratio:.6g} is outside "
                        "the supported numerical range [1e-8, 100].",
                    )
                )
    return errors


def _check_geometry(
    config: dict,
) -> tuple[list[FieldError], list[str], dict]:
    """Validate every solver-relevant region against the exact mesh entities."""
    import numpy as np
    from dolfinx.fem import functionspace
    from dolfinx.mesh import locate_entities_boundary
    from mpi4py import MPI
    from scipy.spatial import cKDTree

    from fenitop.config import build_mesh
    from fenitop.regions import compile_region

    errors: list[FieldError] = []
    warnings: list[str] = []
    entities: list[dict] = []
    mesh = build_mesh(config["mesh"], comm=MPI.COMM_SELF)
    cell_dimension = mesh.topology.dim
    facet_dimension = cell_dimension - 1
    mesh.topology.create_connectivity(facet_dimension, 0)
    facet_to_vertex = mesh.topology.connectivity(facet_dimension, 0)
    boundary_facets = locate_entities_boundary(
        mesh,
        facet_dimension,
        lambda x: np.full(x.shape[1], True, dtype=bool),
    )
    total_facets = len(boundary_facets)
    total_cells = mesh.topology.index_map(cell_dimension).size_local

    scalar_space = functionspace(mesh, ("CG", 1))
    total_nodes = scalar_space.dofmap.index_map.size_local
    node_coords = scalar_space.tabulate_dof_coordinates()[:total_nodes]
    cell_space = functionspace(mesh, ("DG", 0))
    cell_coords = cell_space.tabulate_dof_coordinates()[:total_cells]

    def bounds(coords) -> tuple[tuple[float, float], tuple[float, float]]:
        xy = np.asarray(coords)[:, :2]
        return (
            (float(xy[:, 0].min()), float(xy[:, 1].min())),
            (float(xy[:, 0].max()), float(xy[:, 1].max())),
        )

    def add_record(path: str, kind: str, coords, *, count: int | None = None) -> None:
        entities.append(
            {
                "path": path,
                "entity_kind": kind,
                "count": int(len(coords) if count is None else count),
                "bounds": bounds(coords) if len(coords) else ((0.0, 0.0), (0.0, 0.0)),
            }
        )

    def facet_vertex_indices(facets) -> np.ndarray:
        indices: set[int] = set()
        for facet in facets:
            indices.update(int(v) for v in facet_to_vertex.links(int(facet)))
        return np.asarray(sorted(indices), dtype=np.int32)

    def check_facets(path: str, spec):
        try:
            facets = locate_entities_boundary(
                mesh, facet_dimension, compile_region(spec)
            )
        except Exception as exc:
            errors.append(
                FieldError(
                    path,
                    "geometry_marker_error",
                    f"Region failed while matching boundary facets: {exc}",
                )
            )
            return None
        vertices = facet_vertex_indices(facets)
        coords = mesh.geometry.x[vertices] if vertices.size else np.empty((0, 3))
        add_record(path, "facet", coords, count=len(facets))
        if len(facets) == 0:
            errors.append(
                FieldError(
                    path,
                    "region_matches_no_facets",
                    f"Region matched 0 of {total_facets} boundary facets.",
                )
            )
            return None
        return np.asarray(facets, dtype=np.int32)

    support_facets: list[np.ndarray] = []
    support_node_masks: list[np.ndarray] = []
    constrained_rows: list[tuple[float, float, int]] = []
    for index, bc in enumerate(config["fem"]["dirichlet_bcs"]):
        path = f"config.fem.dirichlet_bcs[{index}].marker"
        facets = check_facets(path, bc["marker"])
        mask = compile_region(bc["marker"])(node_coords.T)
        support_node_masks.append(mask)
        if facets is not None:
            support_facets.append(facets)
            vertices = facet_vertex_indices(facets)
            for x, y, *_ in mesh.geometry.x[vertices]:
                constrained_rows.extend([(x, y, 0), (x, y, 1)])

    traction_facets: list[tuple[int, np.ndarray]] = []
    traction_node_masks: list[np.ndarray] = []
    for index, bc in enumerate(config["fem"].get("traction_bcs", [])):
        path = f"config.fem.traction_bcs[{index}].locator"
        facets = check_facets(path, bc["locator"])
        traction_node_masks.append(compile_region(bc["locator"])(node_coords.T))
        if facets is not None:
            traction_facets.append((index, facets))

    for right in range(len(traction_facets)):
        for left in range(right):
            left_index, left_facets = traction_facets[left]
            right_index, right_facets = traction_facets[right]
            overlap = np.intersect1d(
                left_facets, right_facets
            )
            if overlap.size:
                errors.append(
                    FieldError(
                        f"config.fem.traction_bcs[{right_index}].locator",
                        "traction_regions_overlap",
                        f"Overlaps traction_bcs[{left_index}] on {overlap.size} facets; "
                        "overlapping distributed tractions are rejected.",
                    )
                )

    all_support_facets = (
        np.unique(np.concatenate(support_facets)) if support_facets else np.array([])
    )
    for index, facets in traction_facets:
        overlap = np.intersect1d(all_support_facets, facets)
        if overlap.size:
            errors.append(
                FieldError(
                    f"config.fem.traction_bcs[{index}].locator",
                    "traction_on_fixed_support",
                    f"Traction overlaps fully fixed support on {overlap.size} facets.",
                )
            )

    rank = _rigid_body_rank(constrained_rows)
    if constrained_rows and rank < 3:
        errors.append(
            FieldError(
                "config.fem.dirichlet_bcs",
                "rigid_body_modes_unconstrained",
                f"Fixed supports resist only rank {rank}/3 planar rigid-body modes.",
            )
        )
    elif not constrained_rows and not errors:
        errors.append(
            FieldError(
                "config.fem.dirichlet_bcs",
                "no_support_facets",
                "No fixed support matched a boundary facet.",
            )
        )

    spring_masks: dict[str, np.ndarray] = {}
    if config["opt"]["problem_type"] == "compliant_mechanism":
        support_mask = (
            np.logical_or.reduce(support_node_masks)
            if support_node_masks
            else np.zeros(total_nodes, dtype=bool)
        )
        for name in ("in_spring", "out_spring"):
            path = f"config.opt.{name}.region"
            mask = compile_region(config["opt"][name]["region"])(node_coords.T)
            spring_masks[name] = mask
            add_record(path, "node", node_coords[mask])
            if not mask.any():
                errors.append(
                    FieldError(
                        path,
                        "spring_region_matches_no_nodes",
                        f"Spring region matched 0 of {total_nodes} displacement nodes.",
                    )
                )
            elif np.any(mask & support_mask):
                errors.append(
                    FieldError(
                        path,
                        "spring_overlaps_fixed_support",
                        f"Spring region overlaps {int(np.sum(mask & support_mask))} "
                        "fully constrained nodes.",
                    )
                )
        overlap = spring_masks["in_spring"] & spring_masks["out_spring"]
        if overlap.any():
            errors.append(
                FieldError(
                    "config.opt.out_spring.region",
                    "spring_regions_overlap",
                    f"Input and output springs overlap on {int(overlap.sum())} nodes.",
                )
            )

    passive_masks: dict[str, np.ndarray] = {}
    for name in ("solid_zone", "void_zone"):
        path = f"config.opt.{name}"
        spec = config["opt"][name]
        mask = compile_region(spec)(cell_coords.T)
        passive_masks[name] = mask
        add_record(path, "cell", cell_coords[mask])
        if spec.get("op") != "none" and not mask.any():
            errors.append(
                FieldError(
                    path,
                    "passive_zone_matches_no_cells",
                    f"Passive zone matched 0 of {total_cells} design cells.",
                )
            )

    passive_overlap = passive_masks["solid_zone"] & passive_masks["void_zone"]
    if passive_overlap.any():
        errors.append(
            FieldError(
                "config.opt.void_zone",
                "passive_zones_overlap",
                f"Solid and void zones overlap on {int(passive_overlap.sum())} cells.",
            )
        )

    forced_solid_fraction = (
        0.99 * float(passive_masks["solid_zone"].sum()) / max(total_cells, 1)
    )
    if forced_solid_fraction > float(config["opt"]["vol_frac"]):
        errors.append(
            FieldError(
                "config.opt.solid_zone",
                "solid_zone_exceeds_volume_budget",
                f"Forced-solid minimum volume fraction {forced_solid_fraction:.6g} "
                f"exceeds opt.vol_frac={config['opt']['vol_frac']:.6g}.",
            )
        )

    void_coords = cell_coords[passive_masks["void_zone"]]
    if len(void_coords):
        required_masks = [*support_node_masks, *traction_node_masks, *spring_masks.values()]
        required_mask = (
            np.logical_or.reduce(required_masks)
            if required_masks
            else np.zeros(total_nodes, dtype=bool)
        )
        required_coords = node_coords[required_mask]
        if len(required_coords):
            (x0, y0), (x1, y1) = config["mesh"]["bounds"]
            nx, ny = config["mesh"]["divisions"]
            neighborhood = 0.51 * np.hypot((x1 - x0) / nx, (y1 - y0) / ny)
            distances, _ = cKDTree(void_coords[:, :2]).query(required_coords[:, :2])
            erased = int(np.sum(distances <= neighborhood))
            if erased:
                errors.append(
                    FieldError(
                        "config.opt.void_zone",
                        "void_zone_erases_required_neighborhood",
                        f"Void zone removes cells adjacent to {erased} support/load/"
                        "spring nodes.",
                    )
                )

    report = {
        "total_boundary_facets": int(total_facets),
        "total_nodes": int(total_nodes),
        "total_cells": int(total_cells),
        "rigid_body_rank": rank,
        "entities": entities,
    }
    return errors, warnings, report


def _response(data: dict) -> dict:
    ValidateConfigResponse.model_validate(data)
    return data


def _validate_config_impl(
    request: dict | ValidateConfigRequest,
    *,
    policy: TrustedValidationPolicy | None = None,
) -> dict:
    """Validate safe physics under application-owned geometry/resource policy."""
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
        return _response(
            error_envelope(
                "validate_config",
                translate_validation_error(exc),
                stage="structural_validation",
                checked={"structural": True, "resource": False, "geometry": False},
                normalized_config=None,
            )
        )
    except Exception as exc:
        return _response(
            error_envelope(
                "validate_config",
                [FieldError("<root>", "malformed_request", str(exc))],
                stage="structural_validation",
                checked={"structural": True, "resource": False, "geometry": False},
                normalized_config=None,
            )
        )

    model = parsed_request.config
    normalized = model.model_dump(mode="json")
    warnings = compute_warnings(model)
    semantic_errors = _semantic_errors(model)
    cost = estimate_cost(
        normalized["mesh"],
        model.opt.max_iter,
        problem_type=model.opt.problem_type,
        solver_profile=validation_policy.solver_profile,
        output_interval=validation_policy.output_interval,
    )
    admission_errors = (
        resource_limit_errors(cost, validation_policy.resource_limits)
        if validation_policy.enforce_resource_limits
        else []
    )
    if semantic_errors or admission_errors:
        stage = "semantic_validation" if semantic_errors else "resource_validation"
        return _response(
            error_envelope(
                "validate_config",
                [*semantic_errors, *admission_errors],
                stage=stage,
                warnings=warnings,
                checked={"structural": True, "resource": True, "geometry": False},
                problem_type=model.opt.problem_type,
                normalized_config=None,
                estimated_cost=cost,
            )
        )

    if cost["risk_level"] != "low":
        warnings.append(
            f"estimated resource demand is in the '{cost['risk_level']}' trusted "
            "policy band."
        )

    errors: list[FieldError] = []
    geometry_report = None
    geometry_ran = False
    if validation_policy.check_geometry:
        geometry_ran = True
        geometry_errors, geometry_warnings, geometry_report = _check_geometry(normalized)
        errors.extend(geometry_errors)
        warnings.extend(geometry_warnings)

    checked = {"structural": True, "resource": True, "geometry": geometry_ran}
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
                geometry_report=geometry_report,
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
            geometry_report=geometry_report,
        )
    )


def validate_config_tool(
    request: dict | ValidateConfigRequest,
    *,
    policy: TrustedValidationPolicy | None = None,
) -> dict:
    """Total public boundary for structural/resource/geometry validation."""
    try:
        return _validate_config_impl(request, policy=policy)
    except Exception:
        logger.exception("validate_config: unexpected public-boundary failure")
        return _response(error_envelope(
            "validate_config",
            [FieldError(
                "<root>",
                "internal_error",
                "Validation failed unexpectedly; inspect local logs.",
                retryable=True,
            )],
            stage="internal",
            checked={"structural": False, "resource": False, "geometry": False},
            normalized_config=None,
        ))


def main() -> int:
    from fenitop.tools.cli import run_cli

    return run_cli(
        validate_config_tool,
        "Validate a versioned AgentSafeConfig JSON request.",
        tool_name="validate_config",
    )


if __name__ == "__main__":
    raise SystemExit(main())
