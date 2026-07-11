"""Pydantic structural validation for fenitop JSON configs.

Validates the *normalized, pre-materialized* config shape (see
fenitop.config.normalize_boundary_conditions / _materialize): boundary/load
markers here are still raw JSON (lambda strings or region-DSL dicts), never
compiled callables, so this model stays fully JSON-schema-exportable -- its
model_json_schema() doubles as the declared input schema for the MCP tool
layer and as documentation for whatever agent is authoring configs.

Every field carries a `description` for exactly that reason: the exported
JSON schema is the primary interface an LLM sees, so the schema itself should
say what a field means, whether it's required, and what a sane value looks
like -- not just enforce it after the fact.

Fields are split deliberately into two groups:
  - REQUIRED, no default: values that define *what problem this is*
    (domain, target volume fraction, filter length scale, problem type,
    Poisson's ratio, boundary conditions). Silently defaulting any of these
    would let an agent produce a config that "runs" but solves a different
    problem than intended.
  - Defaulted: numerical/solver tuning knobs with literature-standard values
    that both real example configs (config/beam_2d.json, config/mechanism_2d.json)
    happen to agree on exactly (opt_tol=1e-5, penalty=3.0, epsilon=1e-6,
    beta_interval=50, beta_max=128), plus young's modulus (topology-invariant
    for single-material SIMP compliance minimization -- it only rescales the
    reported compliance value, never the optimized shape, unlike Poisson's
    ratio which does affect the result and stays required) and petsc_options
    (an empty dict "works" via PETSc's own GMRES+ILU default, but a real
    elasticity system converges far more reliably with an explicit SPD-aware
    choice -- CG+GAMG, matching config/beam_2d.json).

This module deliberately needs no dolfinx/MPI import and stays importable on
a bare host; only fenitop.tools.validate_config's geometry-check path (which
must build a real mesh) requires the full stack.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import field_validator as _field_validator

from fenitop.regions import RegionError, compile_region
from fenitop.tools.schema import FieldError

_AXES_2D = ("x", "y")

# Robust default for a 2D SPD elasticity system: conjugate gradient (the
# stiffness matrix is symmetric positive definite) with algebraic multigrid.
# Matches config/beam_2d.json. An empty dict would still "work" -- PETSc
# falls back to GMRES+ILU -- but converges far less reliably at scale.
_DEFAULT_PETSC_OPTIONS = {"ksp_type": "cg", "pc_type": "gamg"}


def _validate_marker(value: Any) -> Any:
    """Accept a lambda string or a region-DSL dict; reject everything else.

    Notably this rejects a bare Python callable too: normalize_boundary_conditions()
    can inject a literal `lambda x: False` default for a traction entry with
    neither a locator nor a marker (fenitop/config.py:105). The Tool 1
    orchestrator (fenitop/tools/validate_config.py::_shape_for_pydantic)
    replaces that injected callable with None before this validator ever sees
    it, specifically so a config that omits both keys gets a clear structural
    error here instead of a JSON-unsafe callable silently reaching
    normalized_config, or a "no-op, matches nothing" load silently vanishing.
    """
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("must be a non-empty lambda string.")
        return value
    if isinstance(value, dict):
        try:
            compile_region(value)
        except RegionError as exc:
            raise ValueError(str(exc)) from exc
        return value
    raise ValueError(
        "must be a lambda string or a region-DSL dict (got neither a locator "
        "nor a marker fenitop could interpret).")


class MeshModel(BaseModel):
    """A 2D structured rectangular domain -- the only mesh kind this agent-tool
    layer supports. 3D/other mesh kinds are explicitly out of scope."""
    model_config = ConfigDict(extra="forbid")

    kind: Literal["rectangle"] = Field(
        default="rectangle",
        description="Mesh topology. Only \"rectangle\" is supported in this 2D-scoped tool.")
    bounds: List[List[float]] = Field(
        description="Domain corners as [[x0, y0], [x1, y1]] with x1>x0 and y1>y0. "
                    "Required -- domain size is inherently problem-specific.")
    divisions: List[int] = Field(
        description="Element counts along each axis as [nx, ny], both positive integers. "
                    "Required -- mesh resolution is inherently problem-specific.")
    cell_type: Literal["quadrilateral", "triangle"] = Field(
        default="quadrilateral",
        description="2D element shape. Defaults to \"quadrilateral\" (the common case).")
    ghost_mode: Optional[str] = Field(
        default=None,
        description="MPI ghost-cell mode for parallel runs (e.g. \"shared_facet\"). "
                    "Optional; None uses dolfinx's own default.")

    @_field_validator("bounds")
    @classmethod
    def _check_bounds(cls, v):
        if len(v) != 2 or any(len(pt) != 2 for pt in v):
            raise ValueError("must be [[x0, y0], [x1, y1]].")
        (x0, y0), (x1, y1) = v
        if not x1 > x0:
            raise ValueError(f"x1 must be > x0; got x0={x0}, x1={x1}.")
        if not y1 > y0:
            raise ValueError(f"y1 must be > y0; got y0={y0}, y1={y1}.")
        return v

    @_field_validator("divisions")
    @classmethod
    def _check_divisions(cls, v):
        if len(v) != 2:
            raise ValueError("must be [nx, ny].")
        if any(n <= 0 for n in v):
            raise ValueError(f"entries must be positive integers; got {v}.")
        return v


class DirichletBC(BaseModel):
    """Mirrors normalize_boundary_conditions()'s {"marker","value"} shape."""
    model_config = ConfigDict(extra="forbid")

    marker: Any = Field(
        description="Where this constraint applies: a region-DSL dict (preferred, e.g. "
                    "{\"op\": \"plane\", \"axis\": \"x\", \"value\": 0}) or a lambda-string "
                    "for backward compatibility. Must match at least one mesh boundary facet.")
    value: List[float] = Field(
        default_factory=lambda: [0.0, 0.0],
        description="Prescribed [ux, uy] displacement at the matched boundary. Defaults to "
                    "[0, 0] (a fixed/clamped support), the overwhelmingly common case.")

    @_field_validator("marker")
    @classmethod
    def _check_marker(cls, v):
        return _validate_marker(v)


class TractionBC(BaseModel):
    """Mirrors normalize_boundary_conditions()'s [value, locator] pair, recast
    as a dict by the Tool 1 orchestrator before validation for convenience."""
    model_config = ConfigDict(extra="forbid")

    value: List[float] = Field(description="Applied traction [tx, ty] on the matched boundary.")
    locator: Any = Field(
        description="Where this load applies: a region-DSL dict (preferred) or a lambda-string. "
                    "Must match at least one mesh boundary facet.")

    @_field_validator("locator")
    @classmethod
    def _check_locator(cls, v):
        return _validate_marker(v)


class FemModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    young_modulus: float = Field(
        alias="young's modulus", default=100.0, gt=0,
        description="Elastic modulus E. Defaults to 100 (a normalized value used throughout the "
                    "topology-optimization literature). For single-material SIMP compliance "
                    "minimization the optimized *topology* is invariant to E -- it only rescales "
                    "the reported compliance value -- so this is safe to leave at its default "
                    "unless you specifically need physically-dimensioned compliance values.")
    poisson_ratio: float = Field(
        alias="poisson's ratio",
        description="Poisson's ratio nu. Required -- unlike E, nu materially affects the "
                    "optimized result and must be a deliberate choice. Must satisfy -1 < nu < 0.5.")
    dirichlet_bcs: List[DirichletBC] = Field(
        min_length=1,
        description="Displacement (support) boundary conditions. Required and must be "
                    "non-empty -- a model with no supports has unconstrained rigid-body motion "
                    "and cannot be solved.")
    traction_bcs: List[TractionBC] = Field(
        default_factory=list,
        description="Applied load boundary conditions. Optional (a body_force-only or "
                    "compliant-mechanism/spring-driven problem may have none), but a "
                    "compliance-minimization problem with no traction_bcs AND no body_force "
                    "has nothing to optimize against (flagged as a warning).")
    body_force: List[float] = Field(
        default_factory=lambda: [0.0, 0.0],
        description="Distributed body force [bx, by] (e.g. gravity). Defaults to [0, 0] -- "
                    "most problems have none.")
    quadrature_degree: int = Field(
        default=2, ge=1,
        description="Numerical integration order. Defaults to 2, sufficient for the bilinear/"
                    "trilinear elements this tool uses; rarely needs changing.")
    petsc_options: Dict[str, Any] = Field(
        default_factory=lambda: dict(_DEFAULT_PETSC_OPTIONS),
        description="PETSc KSP/PC options for the linear elasticity solve. Defaults to "
                    "conjugate-gradient with algebraic multigrid ({\"ksp_type\": \"cg\", "
                    "\"pc_type\": \"gamg\"}), a robust choice for the SPD systems this tool "
                    "produces.")

    @_field_validator("poisson_ratio")
    @classmethod
    def _check_poisson(cls, v):
        if not (-1.0 < v < 0.5):
            raise ValueError(
                "must satisfy -1 < nu < 0.5 (division by zero in the Lame constants "
                f"at the endpoints, fem.py:73); got {v}.")
        return v

    @_field_validator("body_force")
    @classmethod
    def _check_body_force(cls, v):
        if len(v) != 2:
            raise ValueError(f"must have length 2 for a 2D problem; got length {len(v)}.")
        return v


class OptModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iter: int = Field(
        gt=0,
        description="Maximum optimization iterations. Required -- directly determines "
                    "computational cost, must be a deliberate choice.")
    opt_tol: float = Field(
        default=1e-5, gt=0,
        description="Convergence tolerance on the design-change norm. Defaults to 1e-5, the "
                    "standard value used by both reference examples.")
    vol_frac: float = Field(
        ge=0.0, le=1.0,
        description="Target volume fraction (material budget) in [0, 1]. Required -- this is "
                    "the key design target and must never be silently defaulted.")
    initial_density: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Initial uniform density before optimization begins. Defaults to vol_frac "
                    "if omitted (the standard starting point).")
    penalty: float = Field(
        default=3.0, ge=1.0,
        description="SIMP penalization power. Defaults to 3.0, the conventional literature "
                    "value used by both reference examples.")
    epsilon: float = Field(
        default=1e-6, ge=0.0, le=1.0,
        description="SIMP stiffness floor (prevents a fully-void element from having exactly "
                    "zero stiffness). Defaults to 1e-6, used by both reference examples.")
    filter_radius: float = Field(
        gt=0.0,
        description="Physical length scale of the density (Helmholtz PDE) filter. Required -- "
                    "inherently domain-specific, and must be smaller than the domain's smallest "
                    "extent (checked against mesh.bounds) or the filter would smooth across the "
                    "entire design.")
    beta_interval: int = Field(
        default=50, gt=0,
        description="Iterations between Heaviside projection sharpening steps. Defaults to 50, "
                    "used by both reference examples.")
    beta_max: float = Field(
        default=128.0, gt=0.0,
        description="Maximum Heaviside projection sharpness. Defaults to 128, used by both "
                    "reference examples.")
    use_oc: bool = Field(
        default=True,
        description="Use the Optimality Criteria optimizer instead of MMA. Defaults to true. "
                    "Only meaningful when opt_compliance=true (compliant-mechanism mode always "
                    "uses MMA regardless of this flag), so the default is safe either way.")
    move: float = Field(
        default=0.02, ge=0.0, le=1.0,
        description="Maximum per-iteration design-variable change (move limit). Defaults to "
                    "0.02 (config/beam_2d.json's value); a looser problem may converge faster "
                    "with a larger value such as 0.05 (config/mechanism_2d.json's value).")
    opt_compliance: bool = Field(
        description="true = minimize compliance subject to a volume constraint; false = "
                    "compliant-mechanism design (requires compliance_bound, in_spring, and "
                    "out_spring). Required -- defines what kind of problem this is.")
    compliance_bound: Optional[float] = Field(
        default=None,
        description="Compliance constraint upper bound. Required if and only if "
                    "opt_compliance=false.")
    in_spring: Optional[List[Any]] = Field(
        default=None,
        description="[locator, direction, stiffness] input spring for compliant-mechanism mode. "
                    "direction must be \"x\" or \"y\". Required if and only if opt_compliance=false.")
    out_spring: Optional[List[Any]] = Field(
        default=None,
        description="[locator, direction, stiffness] output spring for compliant-mechanism mode. "
                    "Required if and only if opt_compliance=false.")
    solid_zone: Optional[Any] = Field(
        default=None,
        description="Region-DSL dict or lambda-string marking elements forced to full density. "
                    "Optional; omitted means no passive solid zone.")
    void_zone: Optional[Any] = Field(
        default=None,
        description="Region-DSL dict or lambda-string marking elements forced to zero density. "
                    "Optional; omitted means no passive void zone.")
    output_folder: str = Field(default="results", description="Output directory (execution detail, not physics).")
    output_prefix: str = Field(default="fenitop", description="Output filename prefix (execution detail, not physics).")
    output_interval: int = Field(
        default=20, gt=0,
        description="Approximate number of history snapshots to save across the run. Defaults to 20.")

    @_field_validator("solid_zone", "void_zone")
    @classmethod
    def _check_zone(cls, v):
        if v is None:
            return v
        return _validate_marker(v)

    @_field_validator("in_spring", "out_spring")
    @classmethod
    def _check_spring(cls, v):
        if v is None:
            return v
        if len(v) != 3:
            raise ValueError(f"must be [locator, direction, value]; got {v!r}.")
        locator, direction, value = v
        _validate_marker(locator)
        if direction not in _AXES_2D:
            raise ValueError(
                f"direction must be one of {_AXES_2D} for this 2D-scoped config "
                "(fenitop/utility.py's create_mechanism_vectors silently computes the "
                f"wrong dof index for 'z' in a 2D problem instead of raising); got {direction!r}.")
        if not isinstance(value, (int, float)):
            raise ValueError(f"spring value must be numeric; got {value!r}.")
        return v


class ConfigModel(BaseModel):
    """Top-level structural model for a fenitop config. Strict: any key other
    than mesh/fem/opt/parameter_guidance is rejected outright, so a typo'd
    top-level key (e.g. "otp" instead of "opt") is reported directly instead
    of silently ignored."""
    model_config = ConfigDict(extra="forbid")

    mesh: MeshModel
    fem: FemModel
    opt: OptModel
    parameter_guidance: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional human-readable per-parameter documentation block, passed through "
                    "unvalidated. Legacy/informational only -- has no effect on the solve.")

    @model_validator(mode="after")
    def _check_mechanism_fields(self):
        if not self.opt.opt_compliance:
            missing = [name for name in ("compliance_bound", "in_spring", "out_spring")
                       if getattr(self.opt, name) is None]
            if missing:
                raise ValueError(
                    "opt.opt_compliance=false (compliant-mechanism mode) requires "
                    + ", ".join(f"opt.{m}" for m in missing) + " to be set.")
        return self

    @model_validator(mode="after")
    def _check_filter_radius_vs_domain(self):
        (x0, y0), (x1, y1) = self.mesh.bounds
        min_extent = min(x1 - x0, y1 - y0)
        if self.opt.filter_radius >= min_extent:
            raise ValueError(
                f"opt.filter_radius={self.opt.filter_radius:g} must be smaller than the "
                f"domain's smallest extent ({min_extent:g}, derived from mesh.bounds); a filter "
                "radius at or beyond the domain size would smooth the density field across the "
                "entire domain, producing a degenerate (uniformly gray) design.")
        return self


def _format_loc(loc: tuple) -> str:
    if not loc:
        return "<root>"  # model-level validator (e.g. conditional mechanism-mode fields)
    out = ""
    for item in loc:
        if isinstance(item, int):
            out += f"[{item}]"
        else:
            out += f".{item}" if out else str(item)
    return out


def translate_validation_error(exc) -> List[FieldError]:
    """Convert a pydantic ValidationError into our shared FieldError list."""
    errors = []
    for err in exc.errors():
        errors.append(FieldError(path=_format_loc(err["loc"]), kind=err["type"], message=err["msg"]))
    return errors


def compute_warnings(model: ConfigModel) -> List[str]:
    """Non-fatal structural observations that don't block status="ok"."""
    warnings: List[str] = []
    nx, ny = model.mesh.divisions
    (x0, y0), (x1, y1) = model.mesh.bounds

    if nx < 4 or ny < 4:
        warnings.append(
            f"mesh.divisions={model.mesh.divisions} has an axis with fewer than 4 elements; "
            "this is unlikely to produce a meaningful topology.")

    element_size = min((x1 - x0) / nx, (y1 - y0) / ny)
    if model.opt.filter_radius < element_size:
        warnings.append(
            f"opt.filter_radius={model.opt.filter_radius:g} is smaller than the mesh's "
            f"smallest element size (~{element_size:.3g}); the density filter will have little "
            "to no smoothing effect and small-scale checkerboarding is more likely.")

    if model.fem.poisson_ratio < 0:
        warnings.append(
            f"fem.poisson's ratio={model.fem.poisson_ratio} is negative (auxetic material); "
            "physically valid but unusual -- confirm this is intentional.")

    if (model.opt.opt_compliance and not model.fem.traction_bcs
            and all(abs(b) < 1e-12 for b in model.fem.body_force)):
        warnings.append(
            "fem.traction_bcs is empty and fem.body_force is effectively zero; there is no "
            "external load for the optimizer to act against.")

    return warnings
