"""Tool 1: strict structural, resource, semantic, and mesh-backed validation."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from fenitop.tools.config_models import (
    AgentSafeConfig,
    compute_warnings,
    parse_agent_safe_config,
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
    loads = [
        bc
        for bc in model.fem.boundary_conditions
        if bc.kind in {"uniform_traction", "uniform_resultant"}
    ]
    nonzero_loads = [
        bc
        for bc in loads
        if any(
            float(component) != 0.0
            for component in (
                bc.traction
                if bc.kind == "uniform_traction"
                else bc.resultant
            )
        )
    ]
    if (
        not nonzero_loads
        and all(float(component) == 0.0 for component in model.fem.body_force)
    ):
        errors.append(
            FieldError(
                "config.fem",
                "external_load_required",
                "At least one nonzero distributed traction or body force is required.",
            )
        )
    for bc in loads:
        value = (
            bc.traction if bc.kind == "uniform_traction" else bc.resultant
        )
        if all(float(component) == 0.0 for component in value):
            errors.append(
                FieldError(
                    f"config.fem.boundary_conditions.{bc.bc_id}",
                    (
                        "zero_traction"
                        if bc.kind == "uniform_traction"
                        else "zero_resultant"
                    ),
                    "A distributed boundary load must have a nonzero vector.",
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
    from mpi4py import MPI
    from scipy.spatial import cKDTree

    from fenitop.boundary_resolver import (
        BoundaryResolutionError,
        resolve_boundary,
    )
    from fenitop.config import build_mesh
    from fenitop.regions import compile_region
    from fenitop.tools.mechanical_units import (
        MechanicalUnitContext,
        resultant_to_traction,
        traction_to_resultant,
    )

    errors: list[FieldError] = []
    warnings: list[str] = []
    entities: list[dict] = []
    mesh = build_mesh(config["mesh"], comm=MPI.COMM_SELF)
    cell_dimension = mesh.topology.dim
    facet_dimension = cell_dimension - 1
    from dolfinx.mesh import locate_entities_boundary

    boundary_facets = locate_entities_boundary(
        mesh, facet_dimension, lambda x: np.full(x.shape[1], True, dtype=bool)
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

    def check_facets(path: str, selector, *, bc=None):
        try:
            resolved = resolve_boundary(mesh, selector)
        except BoundaryResolutionError as exc:
            errors.append(
                FieldError(
                    path,
                    exc.code,
                    str(exc),
                )
            )
            return None
        record = {
            "path": path,
            "entity_kind": (
                "node" if selector["kind"] == "boundary_node" else "facet"
            ),
            "count": int(
                len(resolved.node_indices)
                if selector["kind"] == "boundary_node"
                else len(resolved.facets)
            ),
            "bounds": resolved.bounds,
            "bc_id": bc["bc_id"] if bc else None,
            "selector_kind": selector["kind"],
            "requested_extent": resolved.requested_extent,
            "resolved_extent": resolved.resolved_extent,
            "measure": resolved.measure,
            "centroid": resolved.centroid,
            "outward_normal": resolved.outward_normal,
            "resolution_error": resolved.resolution_error,
            "resolution_warning": resolved.warning,
            "requested_point": resolved.requested_point,
            "resolved_point": resolved.resolved_point,
            "constrained_components": None,
            "quantity_kind": None,
            "input_vector": None,
            "effective_traction": None,
            "integrated_resultant": None,
            "length_unit": None,
            "force_unit": None,
            "stress_unit": None,
            "thickness_value": None,
            "thickness_unit": None,
        }
        if resolved.warning:
            warnings.append(f"{bc['bc_id']}: {resolved.warning}")
        entities.append(record)
        return resolved, record

    def resolved_node_mask(resolved) -> np.ndarray:
        selected_coords = np.asarray(
            mesh.geometry.x[resolved.node_indices, :2], dtype=float
        )
        if not len(selected_coords):
            return np.zeros(total_nodes, dtype=bool)
        (x0, y0), (x1, y1) = config["mesh"]["bounds"]
        tolerance = 1e-9 * max(1.0, x1 - x0, y1 - y0)
        distances, _ = cKDTree(selected_coords).query(node_coords[:, :2])
        return distances <= tolerance

    support_facets: list[np.ndarray] = []
    support_node_masks: list[np.ndarray] = []
    support_component_masks = {
        0: np.zeros(total_nodes, dtype=bool),
        1: np.zeros(total_nodes, dtype=bool),
    }
    constrained_rows: list[tuple[float, float, int]] = []
    conditions = config["fem"]["boundary_conditions"]
    supports = [
        bc for bc in conditions
        if bc["kind"] in {"fixed", "zero_displacement"}
    ]
    loads = [
        bc for bc in conditions
        if bc["kind"] in {"uniform_traction", "uniform_resultant"}
    ]
    constrained_keys: dict[tuple[int, int], str] = {}
    for index, bc in enumerate(supports):
        path = f"config.fem.boundary_conditions.{bc['bc_id']}.selector"
        checked = check_facets(path, bc["selector"], bc=bc)
        mask = np.zeros(total_nodes, dtype=bool)
        if checked is not None:
            resolved, _ = checked
            mask = resolved_node_mask(resolved)
        support_node_masks.append(mask)
        if checked is not None:
            resolved, record = checked
            components = (
                ("x", "y")
                if bc["kind"] == "fixed"
                else tuple(bc["components"])
            )
            record["constrained_components"] = components
            if bc["kind"] == "fixed":
                support_facets.append(resolved.facets)
            component_indices = tuple(
                0 if component == "x" else 1 for component in components
            )
            for component in component_indices:
                support_component_masks[component][mask] = True
            for node_index, (x, y, *_) in zip(
                resolved.node_indices,
                mesh.geometry.x[resolved.node_indices],
            ):
                for component in component_indices:
                    key = (int(node_index), component)
                    previous = constrained_keys.get(key)
                    if previous is not None:
                        errors.append(
                            FieldError(
                                path,
                                "duplicate_constrained_dof",
                                f"{bc['bc_id']} repeats component "
                                f"{'x' if component == 0 else 'y'} at a node "
                                f"already constrained by {previous}.",
                            )
                        )
                        continue
                    constrained_keys[key] = bc["bc_id"]
                    constrained_rows.append((x, y, component))

    traction_facets: list[tuple[int, np.ndarray]] = []
    traction_node_masks: list[np.ndarray] = []
    unit_spec = config["units"]
    unit_context = (
        MechanicalUnitContext(
            length_unit=unit_spec["length_unit"],
            force_unit=unit_spec["force_unit"],
            stress_unit=unit_spec["stress_unit"],
            thickness_value=unit_spec["thickness_value"],
        )
        if unit_spec["kind"] == "explicit"
        else None
    )
    for index, bc in enumerate(loads):
        path = f"config.fem.boundary_conditions.{bc['bc_id']}.selector"
        checked = check_facets(path, bc["selector"], bc=bc)
        mask = np.zeros(total_nodes, dtype=bool)
        if checked is not None:
            resolved, record = checked
            mask = resolved_node_mask(resolved)
            traction_facets.append((index, resolved.facets))
            if bc["kind"] == "uniform_resultant":
                effective, integrated = resultant_to_traction(
                    tuple(bc["resultant"]),
                    boundary_measure=resolved.measure,
                    context=unit_context,
                )
                record.update({
                    "quantity_kind": "resultant",
                    "input_vector": bc["resultant"],
                    "effective_traction": effective,
                    "integrated_resultant": integrated,
                    "length_unit": unit_context.length_unit,
                    "force_unit": unit_context.force_unit,
                    "stress_unit": unit_context.stress_unit,
                    "thickness_value": unit_context.thickness_value,
                    "thickness_unit": unit_context.thickness_unit,
                })
            else:
                traction = tuple(float(v) for v in bc["traction"])
                integrated = (
                    traction_to_resultant(
                        traction,
                        boundary_measure=resolved.measure,
                        context=unit_context,
                    )
                    if unit_context is not None
                    else tuple(
                        component * resolved.measure
                        for component in traction
                    )
                )
                record.update({
                    "quantity_kind": "traction",
                    "input_vector": traction,
                    "effective_traction": traction,
                    "integrated_resultant": integrated,
                    "length_unit": (
                        unit_context.length_unit if unit_context else None
                    ),
                    "force_unit": (
                        unit_context.force_unit if unit_context else None
                    ),
                    "stress_unit": (
                        unit_context.stress_unit if unit_context else None
                    ),
                    "thickness_value": (
                        unit_context.thickness_value if unit_context else 1.0
                    ),
                    "thickness_unit": (
                        unit_context.thickness_unit if unit_context else None
                    ),
                })
        traction_node_masks.append(mask)

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
                        (
                            "config.fem.boundary_conditions."
                            f"{loads[right_index]['bc_id']}.selector"
                        ),
                        "traction_regions_overlap",
                        f"Overlaps {loads[left_index]['bc_id']} on "
                        f"{overlap.size} facets; "
                        "overlapping distributed tractions are rejected.",
                    )
                )

    all_support_facets = (
        np.unique(np.concatenate(support_facets)) if support_facets else np.array([])
    )
    fixed_overlap_loads: set[int] = set()
    for index, facets in traction_facets:
        overlap = np.intersect1d(all_support_facets, facets)
        if overlap.size:
            fixed_overlap_loads.add(index)
            errors.append(
                FieldError(
                    (
                        "config.fem.boundary_conditions."
                        f"{loads[index]['bc_id']}.selector"
                    ),
                    "traction_on_fixed_support",
                    f"Traction overlaps fully fixed support on {overlap.size} facets.",
                )
            )
    for index, (bc, load_mask) in enumerate(zip(loads, traction_node_masks)):
        if index in fixed_overlap_loads or not load_mask.any():
            continue
        vector = (
            bc["traction"]
            if bc["kind"] == "uniform_traction"
            else bc["resultant"]
        )
        active_components = [
            component
            for component, value in enumerate(vector)
            if float(value) != 0.0
        ]
        if active_components and all(
            np.all(support_component_masks[component][load_mask])
            for component in active_components
        ):
            errors.append(
                FieldError(
                    (
                        "config.fem.boundary_conditions."
                        f"{bc['bc_id']}.selector"
                    ),
                    "load_components_fully_constrained",
                    "Every nonzero load component is constrained at all "
                    "selected boundary nodes, so the load can do no work.",
                )
            )

    rank = _rigid_body_rank(constrained_rows)
    if constrained_rows and rank < 3:
        errors.append(
            FieldError(
                "config.fem.boundary_conditions",
                "rigid_body_modes_unconstrained",
                f"Fixed supports resist only rank {rank}/3 planar rigid-body modes.",
            )
        )
    elif not constrained_rows and not errors:
        errors.append(
            FieldError(
                "config.fem.boundary_conditions",
                "no_support_facets",
                "No fixed support matched a boundary facet.",
            )
        )

    spring_masks: dict[str, np.ndarray] = {}
    if config["opt"]["problem_type"] == "compliant_mechanism":
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
            spring_component = (
                0 if config["opt"][name]["direction"] == "x" else 1
            )
            overlap = mask & support_component_masks[spring_component]
            if np.any(overlap):
                errors.append(
                    FieldError(
                        path,
                        "spring_overlaps_fixed_support",
                        f"Spring region overlaps {int(np.sum(overlap))} nodes "
                        "constrained in its active component.",
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
    migrated_legacy = False
    source_version = None
    try:
        if isinstance(request, ValidateConfigRequest):
            source_version = request.config.schema_version
            parsed_config, migrated_legacy = parse_agent_safe_config(
                request.config
            )
            parsed_request = ValidateConfigRequest(config=parsed_config)
        elif isinstance(request, dict) and isinstance(request.get("config"), dict):
            source_version = request["config"].get("schema_version")
            parsed_config, migrated_legacy = parse_agent_safe_config(
                request["config"]
            )
            parsed_request = ValidateConfigRequest.model_validate({
                **request,
                "config": parsed_config,
            })
        else:
            parsed_request = ValidateConfigRequest.model_validate(request)
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
    if migrated_legacy:
        if source_version in {None, "1.1"}:
            warnings.append(
                "AgentSafeConfig 1.1 was deterministically migrated to "
                "canonical 2.1; legacy consistent units remain unspecified "
                "and resultants are therefore unavailable."
            )
        else:
            warnings.append(
                f"AgentSafeConfig {source_version} was deterministically "
                "migrated to canonical 2.1 without changing its physics."
            )
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
