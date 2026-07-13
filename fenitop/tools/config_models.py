"""Versioned, physics-only configuration models for the agent-safe surface."""
from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from fenitop.regions import NoneRegion, RegionSpec, parse_region
from fenitop.tools.schema import FieldError

CONFIG_SCHEMA_VERSION = "1.1"


def _finite(value):
    if not math.isfinite(float(value)):
        raise ValueError("must be finite.")
    return value


FiniteNumber = Annotated[
    Union[StrictInt, StrictFloat],
    AfterValidator(_finite),
]
PositiveFiniteNumber = Annotated[FiniteNumber, Field(gt=0)]
OpenFraction = Annotated[FiniteNumber, Field(gt=0, lt=1)]
PositiveFraction = Annotated[FiniteNumber, Field(gt=0, le=1)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
Vector2D = tuple[FiniteNumber, FiniteNumber]
Point2D = tuple[FiniteNumber, FiniteNumber]
Bounds2D = tuple[Point2D, Point2D]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MeshModel(StrictModel):
    kind: Literal["rectangle"] = "rectangle"
    bounds: Bounds2D = Field(
        description="Opposite corners [[x0,y0],[x1,y1]] of the rectangular 2D domain."
    )
    divisions: tuple[PositiveInt, PositiveInt] = Field(
        description="Structured element counts [nx,ny]."
    )
    cell_type: Literal["quadrilateral", "triangle"] = "quadrilateral"

    @model_validator(mode="after")
    def _ordered_bounds(self):
        (x0, y0), (x1, y1) = self.bounds
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"bounds must satisfy x1>x0 and y1>y0; got {self.bounds!r}."
            )
        return self


class DirichletBC(StrictModel):
    marker: RegionSpec = Field(
        description="DSL region for a full-vector fixed support."
    )
    value: Vector2D = Field(
        default=(0.0, 0.0),
        description="Prescribed [ux,uy]. Agent-safe v1 supports only [0,0].",
    )

    @field_validator("marker")
    @classmethod
    def _bounded_region(cls, value):
        return parse_region(value)

    @field_validator("value")
    @classmethod
    def _zero_only(cls, value):
        if any(float(component) != 0.0 for component in value):
            raise ValueError(
                "agent-safe v1 supports only full-vector zero clamps [0,0]; "
                "nonzero and component-wise displacement constraints are unsupported."
            )
        return value


class TractionBC(StrictModel):
    value: Vector2D = Field(
        description="Distributed boundary traction [tx,ty] per unit out-of-plane thickness."
    )
    locator: RegionSpec

    @field_validator("locator")
    @classmethod
    def _bounded_region(cls, value):
        return parse_region(value)

class FemModel(StrictModel):
    analysis_type: Literal["plane_strain"] = Field(
        default="plane_strain",
        description="The v1 constitutive assumption is 2D plane strain.",
    )
    thickness: Literal[1.0] = Field(
        default=1.0,
        description="The v1 model uses implicit unit out-of-plane thickness.",
    )
    units: Literal["consistent_user_units"] = Field(
        default="consistent_user_units",
        description="All geometry, material, and load values must use one consistent unit system.",
    )
    young_modulus: PositiveFiniteNumber = 100.0
    poisson_ratio: FiniteNumber
    dirichlet_bcs: Annotated[list[DirichletBC], Field(min_length=1)]
    traction_bcs: list[TractionBC] = Field(default_factory=list)
    body_force: Vector2D = (0.0, 0.0)

    @field_validator("poisson_ratio")
    @classmethod
    def _poisson_range(cls, value):
        if not -1.0 < value < 0.5:
            raise ValueError(f"must satisfy -1 < nu < 0.5; got {value}.")
        return value


class MechanismSpring(StrictModel):
    region: RegionSpec
    direction: Literal["x", "y"]
    stiffness: PositiveFiniteNumber

    @field_validator("region")
    @classmethod
    def _bounded_region(cls, value):
        return parse_region(value)


class OptimizationBase(StrictModel):
    max_iter: PositiveInt
    opt_tol: PositiveFiniteNumber = 1e-5
    vol_frac: OpenFraction
    initial_density: OpenFraction | None = None
    penalty: Annotated[FiniteNumber, Field(ge=1)] = 3.0
    epsilon: PositiveFraction = 1e-6
    filter_radius: PositiveFiniteNumber
    beta_interval: PositiveInt = 50
    beta_max: PositiveFiniteNumber = 128.0
    move: PositiveFraction = 0.02
    solid_zone: RegionSpec = Field(default_factory=lambda: NoneRegion(op="none"))
    void_zone: RegionSpec = Field(default_factory=lambda: NoneRegion(op="none"))

    @field_validator("solid_zone", "void_zone")
    @classmethod
    def _bounded_regions(cls, value):
        return parse_region(value)

    @field_validator("beta_max")
    @classmethod
    def _valid_beta_schedule(cls, value):
        exponent = math.log2(float(value))
        if value < 1 or not math.isclose(exponent, round(exponent), abs_tol=1e-12):
            raise ValueError("must be a power of two greater than or equal to 1.")
        return value


class ComplianceOptimization(OptimizationBase):
    problem_type: Literal["minimize_compliance"]
    optimizer: Literal["oc", "mma"] = "oc"


class MechanismOptimization(OptimizationBase):
    problem_type: Literal["compliant_mechanism"]
    optimizer: Literal["mma"] = "mma"
    compliance_bound: PositiveFiniteNumber
    in_spring: MechanismSpring
    out_spring: MechanismSpring


OptimizationModel = Annotated[
    Union[ComplianceOptimization, MechanismOptimization],
    Field(discriminator="problem_type"),
]


class AgentSafeConfig(StrictModel):
    schema_version: Literal[CONFIG_SCHEMA_VERSION] = CONFIG_SCHEMA_VERSION
    mesh: MeshModel
    fem: FemModel
    opt: OptimizationModel

    @model_validator(mode="after")
    def _filter_smaller_than_domain(self):
        (x0, y0), (x1, y1) = self.mesh.bounds
        extent = min(x1 - x0, y1 - y0)
        if self.opt.filter_radius >= extent:
            raise ValueError(
                f"opt.filter_radius={self.opt.filter_radius:g} must be smaller "
                f"than the domain's minimum extent ({extent:g})."
            )
        return self

# Transitional import name used by existing internal callers. It now denotes the
# strict agent-safe schema; it is not the former lambda/path-capable model.
ConfigModel = AgentSafeConfig


def compile_solver_config(
    config: AgentSafeConfig | dict,
    *,
    solver_profile: Literal["auto", "iterative", "direct"] = "auto",
    output_folder: str = "results",
    output_prefix: str | None = None,
    output_interval: int = 20,
) -> dict:
    """Deterministically compile safe physics into the legacy solver dictionary."""
    model = (
        config
        if isinstance(config, AgentSafeConfig)
        else AgentSafeConfig.model_validate(config)
    )
    problem_type = model.opt.problem_type
    if solver_profile == "auto":
        solver_profile = (
            "iterative" if problem_type == "minimize_compliance" else "direct"
        )
    petsc_options = (
        {"ksp_type": "cg", "pc_type": "gamg"}
        if solver_profile == "iterative"
        else {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    )
    prefix = output_prefix or (
        "compliance_2d"
        if problem_type == "minimize_compliance"
        else "mechanism_2d"
    )

    opt = model.opt.model_dump(mode="json")
    opt.pop("problem_type")
    optimizer = opt.pop("optimizer")
    opt["opt_compliance"] = problem_type == "minimize_compliance"
    opt["use_oc"] = optimizer == "oc"
    if isinstance(model.opt, MechanismOptimization):
        for name in ("in_spring", "out_spring"):
            spring = getattr(model.opt, name)
            opt[name] = [
                spring.region.model_dump(mode="json"),
                spring.direction,
                spring.stiffness,
            ]
    opt.update(
        {
            "output_folder": output_folder,
            "output_prefix": prefix,
            "output_interval": output_interval,
        }
    )

    return {
        "mesh": {
            **model.mesh.model_dump(mode="json"),
            "ghost_mode": "shared_facet",
        },
        "fem": {
            "young's modulus": model.fem.young_modulus,
            "poisson's ratio": model.fem.poisson_ratio,
            "dirichlet_bcs": [
                {
                    "marker": bc.marker.model_dump(mode="json"),
                    "value": list(bc.value),
                }
                for bc in model.fem.dirichlet_bcs
            ],
            "traction_bcs": [
                {
                    "value": list(bc.value),
                    "locator": bc.locator.model_dump(mode="json"),
                }
                for bc in model.fem.traction_bcs
            ],
            "body_force": list(model.fem.body_force),
            "quadrature_degree": 2,
            "petsc_options": petsc_options,
        },
        "opt": opt,
    }


def _format_loc(loc: tuple) -> str:
    if not loc:
        return "<root>"
    out = ""
    for item in loc:
        if isinstance(item, int):
            out += f"[{item}]"
        else:
            out += f".{item}" if out else str(item)
    return out


def translate_validation_error(exc) -> list[FieldError]:
    return [
        FieldError(
            path=_format_loc(error["loc"]),
            code=error["type"],
            message=error["msg"],
        )
        for error in exc.errors()
    ]


def compute_warnings(model: AgentSafeConfig) -> list[str]:
    warnings: list[str] = []
    nx, ny = model.mesh.divisions
    (x0, y0), (x1, y1) = model.mesh.bounds
    if nx < 4 or ny < 4:
        warnings.append(
            f"mesh.divisions={model.mesh.divisions} has an axis with fewer than "
            "4 elements; this is unlikely to produce a meaningful topology."
        )
    hx, hy = (x1 - x0) / nx, (y1 - y0) / ny
    if model.opt.filter_radius < max(hx, hy):
        warnings.append(
            f"opt.filter_radius={model.opt.filter_radius:g} is smaller than the "
            f"larger element axis ({max(hx, hy):.3g}); filtering may be ineffective "
            "or directionally inconsistent."
        )
    if model.fem.poisson_ratio < 0:
        warnings.append(
            f"fem.poisson_ratio={model.fem.poisson_ratio} is auxetic; confirm this is intentional."
        )
    aspect_ratio = max(hx / hy, hy / hx)
    if aspect_ratio > 5:
        warnings.append(
            f"mesh element aspect ratio is {aspect_ratio:.3g}:1; values above 5:1 "
            "can reduce accuracy and make filtering anisotropic."
        )
    for name, spring in (
        ("in_spring", getattr(model.opt, "in_spring", None)),
        ("out_spring", getattr(model.opt, "out_spring", None)),
    ):
        if spring is None:
            continue
        ratio = float(spring.stiffness) / float(model.fem.young_modulus)
        if ratio < 1e-5 or ratio > 1:
            warnings.append(
                f"opt.{name}.stiffness / fem.young_modulus = {ratio:.3g}; "
                "confirm the mechanism spring/material scale is intentional."
            )
    if model.opt.beta_max > 1 and model.opt.beta_interval > model.opt.max_iter:
        warnings.append(
            "opt.beta_interval exceeds opt.max_iter, so Heaviside continuation "
            "cannot advance beyond beta=1 in this run."
        )
    return warnings
