"""Tool 1: validate_config.

Pedantically validates a fenitop JSON config before it's ever handed to the
solver. Layers on top of, never replaces, fenitop.config.validate_config()
(tests/test_config_validation.py's existing tests keep passing verbatim
against that function).

Two halves with different dependency footprints:
  - structural (pydantic model + the legacy hand-rolled validator + cost
    estimate): pure Python, no dolfinx/MPI needed, safe on a bare host.
  - geometry (locate_entities_boundary + a rigid-body-motion sanity check):
    needs a real dolfinx mesh build, so it's gated behind check_geometry=True
    and only runs inside the docker image. It's also only attempted once
    structural validation has fully passed, since it needs a well-typed
    mesh/BC shape to operate on at all.

All structural errors are collected (not fail-fast) so an agent gets every
problem in one round trip instead of fix-one/resubmit thrashing.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from fenitop.config import normalize_boundary_conditions
from fenitop.config import validate_config as _legacy_validate_config
from fenitop.tools.config_models import ConfigModel, compute_warnings, translate_validation_error
from fenitop.tools.logging_config import get_logger
from fenitop.tools.safety import estimate_cost
from fenitop.tools.schema import FieldError, error_envelope, ok_envelope

_LEGACY_FEM_KEYS_TO_DROP = ("disp_bc", "neumann_bcs")
logger = get_logger(__name__)


def load_raw_config(config: Any) -> Dict[str, Any]:
    """Accept either a config dict directly or a path to a JSON file.

    Shared by Tool 1 and Tool 2 so both resolve a `config` request field
    identically, and so Tool 2 can peek at the raw mesh spec for its cheap
    pre-safety-check (see run_topopt.py) without duplicating this.
    """
    if isinstance(config, str):
        with open(config, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return config


def _shape_for_pydantic(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize BC shapes (reusing normalize_boundary_conditions) into the
    dict-of-dicts shape ConfigModel expects, without touching lambda strings
    or region-DSL dicts -- markers/locators stay exactly as authored.
    """
    fem = dict(raw_config.get("fem", {}))
    dirichlet_bcs, traction_bcs = normalize_boundary_conditions(fem, dim=2)
    for key in _LEGACY_FEM_KEYS_TO_DROP:
        fem.pop(key, None)
    fem["dirichlet_bcs"] = dirichlet_bcs
    shaped_traction = []
    for value, locator in traction_bcs:
        # normalize_boundary_conditions() injects a literal `lambda x: False`
        # callable when an entry has neither "locator" nor "marker" -- turn
        # that into an explicit rejection instead of a silently-accepted,
        # JSON-unsafe no-op (see config_models._validate_marker's docstring).
        shaped_traction.append({"value": value, "locator": None if callable(locator) else locator})
    fem["traction_bcs"] = shaped_traction

    opt = dict(raw_config.get("opt", {}))
    opt.setdefault("solid_zone", {"op": "none"})
    opt.setdefault("void_zone", {"op": "none"})

    shaped = dict(raw_config)
    shaped["mesh"] = dict(raw_config.get("mesh", {}))
    shaped["fem"] = fem
    shaped["opt"] = opt
    return shaped


def _guess_legacy_error_path(message: str) -> str:
    match = re.search(r"Parameter '([\w.]+)'", message)
    if match:
        return f"opt.{match.group(1)}"
    if "body_force" in message:
        return "fem.body_force"
    return "<root>"


def _rigid_body_rank(constrained_rows: List[Tuple[float, float, int]]) -> int:
    """Rank (0-3) of the rigid-body mode matrix for a set of (x, y, component)
    Dirichlet constraints, where component 0/1 = x/y-translation constrained
    at that node. Rank 3 means all 3 planar rigid-body modes (x-translation,
    y-translation, rotation) are resisted -- a necessary-not-sufficient
    condition for a well-posed (non-singular) elasticity problem.

    Pure numpy, no dolfinx -- kept standalone specifically so it's directly
    unit-testable with synthetic data, independent of dolfinx's facet-
    matching semantics. Deliberately counts singular values above tolerance
    rather than inspecting only the smallest one: a matrix built from fewer
    than 2 constrained points has fewer than 3 rows and so returns fewer
    than 3 singular values in the first place (numpy.linalg.svd returns
    min(rows, cols) values) -- checking only "the last one" would silently
    ignore that a single pinned point can never reach rank 3, since there's
    no third singular value there to even inspect.
    """
    import numpy as np

    if not constrained_rows:
        return 0
    xs = np.array([r[0] for r in constrained_rows])
    ys = np.array([r[1] for r in constrained_rows])
    cx, cy = xs.mean(), ys.mean()
    rows = [([1.0, 0.0, -(y - cy)] if comp == 0 else [0.0, 1.0, (x - cx)])
            for x, y, comp in constrained_rows]
    singular_values = np.linalg.svd(np.array(rows), compute_uv=False)
    tol = 1e-8 * max(float(singular_values[0]), 1.0)
    return int(np.sum(singular_values > tol))


def _check_geometry(normalized_config: Dict[str, Any]) -> Tuple[List[FieldError], List[str]]:
    """Build a real (serial) mesh and confirm every marker matches >=1 facet,
    plus a necessary-condition rigid-body-motion check on the Dirichlet BCs.

    This is the only part of Tool 1 that needs dolfinx. Deliberately uses
    mesh topology/geometry (facet -> vertex -> coordinate) rather than the
    function-space dofmap, since every Dirichlet BC in this codebase is
    applied over the full vector space (fem.py's dirichletbc() call is never
    given a sub-space), so every matched node constrains both components --
    no need to reason about blocked-dofmap component indexing at all.
    """
    import numpy as np
    from mpi4py import MPI
    from dolfinx.mesh import locate_entities_boundary

    from fenitop.config import _materialize, build_mesh

    errors: List[FieldError] = []
    warnings: List[str] = []

    mesh = build_mesh(normalized_config["mesh"], comm=MPI.COMM_SELF)
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)
    total_facets = mesh.topology.index_map(fdim).size_local

    def facet_vertex_coords(facets) -> "np.ndarray":
        vertex_indices = set()
        for f in facets:
            vertex_indices.update(int(v) for v in f2v.links(int(f)))
        return mesh.geometry.x[sorted(vertex_indices)]

    def check_marker(path: str, marker_spec: Any):
        if marker_spec is None:
            errors.append(FieldError(path, "geometry", "No marker/locator was provided."))
            return None
        marker = _materialize(marker_spec)
        if not callable(marker):
            errors.append(FieldError(
                path, "geometry", f"Could not compile a marker from {marker_spec!r}."))
            return None
        try:
            facets = locate_entities_boundary(mesh, fdim, marker)
        except Exception as exc:  # noqa: BLE001 - surfacing solver-side failures to the agent
            errors.append(FieldError(path, "geometry", f"Marker raised while matching facets: {exc}"))
            return None
        if len(facets) == 0:
            errors.append(FieldError(
                path, "geometry",
                f"Marker matched 0 of {total_facets} boundary facets; this boundary condition "
                "would be silently dropped (fenitop/fem.py skips markers matching zero facets "
                "without raising)."))
            return None
        return facets

    constrained_rows = []
    for i, bc in enumerate(normalized_config["fem"].get("dirichlet_bcs", [])):
        facets = check_marker(f"fem.dirichlet_bcs[{i}].marker", bc.get("marker"))
        if facets is None:
            continue
        for x, y, *_ in facet_vertex_coords(facets):
            constrained_rows.append((x, y, 0))
            constrained_rows.append((x, y, 1))

    for i, bc in enumerate(normalized_config["fem"].get("traction_bcs", [])):
        check_marker(f"fem.traction_bcs[{i}].locator", bc.get("locator"))

    if constrained_rows:
        rank = _rigid_body_rank(constrained_rows)
        if rank < 3:
            # A necessary-not-sufficient check, but a hard one where it does
            # fire: if the Dirichlet constraints don't span all 3 planar
            # rigid-body modes, the assembled stiffness matrix is mathematically
            # guaranteed to have a null space on the missing mode(s) -- the FEM
            # solve will fail or silently return a meaningless result. This is
            # exactly the "boundary conditions must be sufficient to actually
            # define the problem" requirement, not just "must be present" --
            # so it's an error, not a warning.
            errors.append(FieldError(
                "fem.dirichlet_bcs", "geometry",
                "The Dirichlet boundary conditions do not constrain all rigid-body modes "
                f"(x-translation, y-translation, rotation) -- only rank {rank}/3 was resisted. "
                "The assembled system is singular; add or extend supports so translation in "
                "both directions and rotation are all resisted."))
    elif not errors:
        errors.append(FieldError(
            "fem.dirichlet_bcs", "geometry",
            "No Dirichlet boundary conditions matched any facets; the model has no supports "
            "and the assembled system is singular."))

    return errors, warnings


def validate_config_tool(request: Dict[str, Any]) -> Dict[str, Any]:
    """Tool 1 entry point. request = {"config": dict|path, "check_geometry": bool=True}.

    A bare config dict (no "config" key) is also accepted as the whole
    request, so `validate_config_tool(json.load(open("config/beam_2d.json")))`
    works directly with check_geometry defaulting to True. That shorthand
    cannot be combined with sibling request options in the same flat dict
    (e.g. {**config, "check_geometry": False}) -- with the strict
    extra="forbid" config schema, "check_geometry" would be indistinguishable
    from a genuine (invalid) top-level config key and rejected as such. Use
    the explicit {"config": ..., "check_geometry": ...} wrapper whenever you
    need to pass anything beyond the config itself.
    """
    config = request.get("config", request) if isinstance(request, dict) else request
    check_geometry = request.get("check_geometry", True) if isinstance(request, dict) else True
    raw_config = load_raw_config(config)
    logger.info("validate_config: starting (check_geometry=%s)", check_geometry)

    try:
        shaped = _shape_for_pydantic(raw_config)
    except Exception as exc:  # noqa: BLE001 - malformed input, report don't crash
        logger.error("validate_config: could not interpret config structure: %s", exc)
        return error_envelope(
            "validate_config",
            [FieldError("<root>", "malformed", f"Could not interpret the config structure: {exc}")],
            stage="structural_validation", checked={"structural": True, "geometry": False},
            normalized_config=None)

    errors: List[FieldError] = []
    warnings: List[str] = []

    model: Optional[ConfigModel] = None
    try:
        model = ConfigModel.model_validate(shaped)
    except ValidationError as exc:
        errors.extend(translate_validation_error(exc))

    try:
        _legacy_validate_config(copy.deepcopy(shaped))
    except ValueError as exc:
        errors.append(FieldError(_guess_legacy_error_path(str(exc)), "legacy_rule", str(exc)))

    if errors or model is None:
        logger.warning("validate_config: structural validation failed with %d error(s): %s",
                        len(errors), "; ".join(f"{e.path}: {e.message}" for e in errors))
        return error_envelope(
            "validate_config", errors, stage="structural_validation", warnings=warnings,
            checked={"structural": True, "geometry": False}, normalized_config=None)
    logger.info("validate_config: structural validation passed (problem_type=%s)",
                "minimize_compliance" if model.opt.opt_compliance else "compliant_mechanism")

    warnings.extend(compute_warnings(model))
    problem_type = "minimize_compliance" if model.opt.opt_compliance else "compliant_mechanism"
    # exclude_none=True matters beyond tidiness: normalized_config is meant to
    # be re-runnable, including as the "config" of a *second* validate_config_tool
    # call (run_topopt_tool always re-validates). fenitop.config.validate_config()
    # -- the untouched legacy validator this function layers on top of -- does
    # `config.get("parameter_guidance", {})`, which only falls back to {} when
    # the key is *absent*; an explicit "parameter_guidance": null (which a bare
    # model_dump() would emit for any config that never set it) would round-trip
    # back in as a real None and crash that dict.get fallback on the next pass.
    normalized_config = model.model_dump(by_alias=True, exclude_none=True)
    cost = estimate_cost(normalized_config["mesh"], model.opt.max_iter)
    if cost["risk_level"] != "low":
        warnings.append(
            f"estimated_cost.complexity_score={cost['complexity_score']:.3g} is in the "
            f"'{cost['risk_level']}' risk band.")
    logger.info("validate_config: estimated_cost=%s risk_level=%s",
                cost["complexity_score"], cost["risk_level"])

    geometry_ran = False
    if check_geometry:
        logger.info("validate_config: running geometry checks (dolfinx mesh build)...")
        geometry_ran = True
        geo_errors, geo_warnings = _check_geometry(normalized_config)
        errors.extend(geo_errors)
        warnings.extend(geo_warnings)

    checked = {"structural": True, "geometry": geometry_ran}
    if errors:
        logger.warning("validate_config: geometry validation failed with %d error(s): %s",
                        len(errors), "; ".join(f"{e.path}: {e.message}" for e in errors))
        return error_envelope(
            "validate_config", errors, stage="geometry_validation", warnings=warnings,
            checked=checked, problem_type=problem_type, normalized_config=None,
            estimated_cost=cost)

    logger.info("validate_config: status=ok (%d warning(s))", len(warnings))
    return ok_envelope(
        "validate_config", warnings=warnings, checked=checked, problem_type=problem_type,
        errors=[], normalized_config=normalized_config, estimated_cost=cost)


def main() -> int:
    from fenitop.tools.cli import run_cli
    return run_cli(validate_config_tool, "Pedantically validate a fenitop JSON config.")


if __name__ == "__main__":
    raise SystemExit(main())
