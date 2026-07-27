import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from fenitop.regions import compile_region, is_region_spec

try:
    from mpi4py import MPI
except Exception:  # pragma: no cover
    MPI = None


def _materialize(value):
    if isinstance(value, str):
        if value.startswith("lambda"):
            try:
                return eval(value, {"np": np, "__builtins__": {}}, {})
            except Exception:
                return value
        return value
    if isinstance(value, list):
        return [_materialize(item) for item in value]
    if isinstance(value, dict):
        if is_region_spec(value):
            return compile_region(value)
        return {key: _materialize(item) for key, item in value.items()}
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    opt = config.setdefault("opt", {})
    fem = config.setdefault("fem", {})
    guidance = config.get("parameter_guidance", {})

    opt.setdefault("initial_density", opt.get("vol_frac", 0.5))
    opt.setdefault("output_interval", 20)

    rules = {
        "max_iter": {"type": "integer", "minimum": 1},
        "output_interval": {"type": "integer", "minimum": 1},
        "beta_interval": {"type": "integer", "minimum": 1},
        "vol_frac": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "initial_density": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "move": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "filter_radius": {"type": "number", "minimum": 0.0},
        "penalty": {"type": "number", "minimum": 0.0},
        "epsilon": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "opt_tol": {"type": "number", "minimum": 0.0},
        "beta_max": {"type": "number", "minimum": 0.0},
    }

    for name, spec in rules.items():
        if name not in opt:
            continue
        if name in guidance:
            spec = {**spec, **guidance[name]}
        value = opt[name]
        if spec.get("type") == "integer":
            if not _is_integer(value):
                raise ValueError(f"Parameter '{name}' must be an integer; got {value!r}.")
            if "minimum" in spec and value < spec["minimum"]:
                raise ValueError(f"Parameter '{name}' must be >= {spec['minimum']}; got {value}.")
        else:
            if not _is_number(value):
                raise ValueError(f"Parameter '{name}' must be numeric; got {value!r}.")
            if "minimum" in spec and value < spec["minimum"]:
                raise ValueError(f"Parameter '{name}' must be >= {spec['minimum']}; got {value}.")
            if "maximum" in spec and value > spec["maximum"]:
                raise ValueError(f"Parameter '{name}' must be <= {spec['maximum']}; got {value}.")

    if not isinstance(fem.get("body_force", [0.0, 0.0]), (list, tuple, np.ndarray)):
        raise ValueError("Parameter 'fem.body_force' must be a sequence of numbers.")

    return config


def normalize_boundary_conditions(fem_config: Dict[str, Any], dim: int = 2):
    dirichlet_bcs = []
    legacy_disp_bc = fem_config.get("disp_bc")
    for entry in fem_config.get("dirichlet_bcs", []):
        if isinstance(entry, dict):
            marker = entry.get("marker", entry.get("locator", legacy_disp_bc))
            value = entry.get("value", [0.0] * dim)
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            marker = entry[0]
            value = entry[1]
        else:
            marker = entry
            value = [0.0] * dim
        dirichlet_bcs.append({"marker": marker, "value": value})

    if not dirichlet_bcs and legacy_disp_bc is not None:
        dirichlet_bcs.append({"marker": legacy_disp_bc, "value": [0.0] * dim})

    traction_bcs = []
    for entry in fem_config.get("traction_bcs", fem_config.get("neumann_bcs", [])):
        if isinstance(entry, dict):
            value = entry.get("value", [0.0] * dim)
            locator = entry.get("locator", entry.get("marker", lambda x: False))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            value, locator = entry[0], entry[1]
        else:
            value, locator = [0.0] * dim, entry
        traction_bcs.append([value, locator])

    return dirichlet_bcs, traction_bcs


def load_config(config_path: Optional[str] = None, default_case: str = "beam_2d") -> Dict[str, Any]:
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent.parent / "config" / f"{default_case}.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    config = _materialize(raw_config)
    return validate_config(config)


def build_mesh(mesh_spec: Dict[str, Any], comm=None):
    if comm is None:
        if MPI is None:
            raise RuntimeError("mpi4py is required to build the mesh.")
        comm = MPI.COMM_WORLD

    from dolfinx.mesh import create_rectangle, CellType
    import dolfinx as dfx

    if mesh_spec.get("kind", "rectangle") != "rectangle":
        raise NotImplementedError("Only rectangle meshes are supported in this config loader.")

    bounds = mesh_spec["bounds"]
    divisions = mesh_spec["divisions"]
    cell_type_name = mesh_spec.get("cell_type", "quadrilateral")
    # 2D rectangle domains only -- hexahedron (3D) is out of scope.
    cell_type = {
        "quadrilateral": CellType.quadrilateral,
        "triangle": CellType.triangle,
    }[cell_type_name]

    ghost_mode = mesh_spec.get("ghost_mode", None)
    if ghost_mode is None:
        return create_rectangle(comm, bounds, divisions, cell_type)
    if isinstance(ghost_mode, str):
        ghost_mode = getattr(dfx.cpp.mesh.GhostMode, ghost_mode)
    return create_rectangle(comm, bounds, divisions, cell_type, ghost_mode=ghost_mode)


def build_fem_opt(config: Dict[str, Any], comm=None):
    """Build the (fem, opt) dicts consumed by topopt() from a JSON-safe,
    not-yet-materialized config (e.g. the "normalized_config" a validate_config
    call returns -- markers/locators still lambda strings or region-DSL
    dicts, never compiled callables).

    Factors out the logic duplicated between scripts/beam_2d.py::build_problem()
    and scripts/mechanism_2d.py::build_problem() so agent-tool callers (and any
    future entry point) don't need to reimplement it. Those two scripts are
    unaffected and keep working exactly as before -- they materialize via
    load_config() themselves before ever touching normalize_boundary_conditions,
    a distinct code path this function does not alter.
    """
    if comm is None:
        if MPI is None:
            raise RuntimeError("mpi4py is required to build the mesh.")
        comm = MPI.COMM_WORLD

    config = _materialize(config)
    mesh_spec = config.get("mesh", {})
    mesh = build_mesh(mesh_spec, comm)
    mesh_serial = build_mesh(mesh_spec, MPI.COMM_SELF) if comm.rank == 0 else None

    fem_config = config.get("fem", {})
    dirichlet_bcs, traction_bcs = normalize_boundary_conditions(fem_config, dim=2)

    fem = {
        "mesh": mesh,
        "mesh_serial": mesh_serial,
        "young's modulus": fem_config.get("young's modulus", 100),
        "poisson's ratio": fem_config.get("poisson's ratio", 0.25),
        "disp_bc": fem_config.get("disp_bc", lambda x: np.isclose(x[0], 0)),
        "dirichlet_bcs": dirichlet_bcs,
        "traction_bcs": traction_bcs,
        "body_force": tuple(fem_config.get("body_force", [0.0, 0.0])),
        "quadrature_degree": fem_config.get("quadrature_degree", 2),
        "petsc_options": fem_config.get("petsc_options", {}),
    }

    opt = dict(config.get("opt", {}))
    opt.setdefault("output_folder", "results")
    opt.setdefault("output_prefix", "fenitop")
    opt.setdefault("output_interval", 20)
    opt.setdefault("initial_density", opt.get("vol_frac", 0.5))
    return fem, opt
