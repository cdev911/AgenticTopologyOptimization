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
from fenitop.tools.mechanical_units import (
    ForceUnitName,
    LengthUnitName,
    MechanicalUnitContext,
    StressUnitName,
)
from fenitop.tools.schema import FieldError

CONFIG_SCHEMA_VERSION = "2.1"
PREVIOUS_CONFIG_SCHEMA_VERSION = "2.0"
LEGACY_CONFIG_SCHEMA_VERSION = "1.1"


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


class LegacyDirichletBC(StrictModel):
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


class LegacyTractionBC(StrictModel):
    value: Vector2D = Field(
        description="Distributed boundary traction [tx,ty] per unit out-of-plane thickness."
    )
    locator: RegionSpec

    @field_validator("locator")
    @classmethod
    def _bounded_region(cls, value):
        return parse_region(value)

class LegacyFemModel(StrictModel):
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
    dirichlet_bcs: Annotated[list[LegacyDirichletBC], Field(min_length=1)]
    traction_bcs: list[LegacyTractionBC] = Field(default_factory=list)
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


class LegacyAgentSafeConfig(StrictModel):
    """Read-only parser used solely by the deterministic 1.1 migration."""

    schema_version: Literal[LEGACY_CONFIG_SCHEMA_VERSION]
    mesh: MeshModel
    fem: LegacyFemModel
    opt: OptimizationModel


class LegacyConsistentUnits(StrictModel):
    kind: Literal["legacy_consistent"] = "legacy_consistent"
    thickness_value: Literal[1.0] = 1.0


class ExplicitMechanicalUnits(StrictModel):
    kind: Literal["explicit"] = "explicit"
    length_unit: LengthUnitName
    force_unit: ForceUnitName
    stress_unit: StressUnitName
    thickness_value: Literal[1.0] = 1.0

    def context(self) -> MechanicalUnitContext:
        return MechanicalUnitContext(
            length_unit=self.length_unit,
            force_unit=self.force_unit,
            stress_unit=self.stress_unit,
            thickness_value=self.thickness_value,
        )


MechanicalUnits = Annotated[
    Union[LegacyConsistentUnits, ExplicitMechanicalUnits],
    Field(discriminator="kind"),
]


class FractionEdgeInterval(StrictModel):
    kind: Literal["fraction"] = "fraction"
    start: Annotated[FiniteNumber, Field(ge=0, le=1)]
    end: Annotated[FiniteNumber, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def _positive_ordered_interval(self):
        if self.end <= self.start:
            raise ValueError("end must be greater than start.")
        return self


class CoordinateEdgeInterval(StrictModel):
    kind: Literal["coordinate"] = "coordinate"
    start: FiniteNumber
    end: FiniteNumber

    @model_validator(mode="after")
    def _positive_ordered_interval(self):
        if self.end <= self.start:
            raise ValueError("end must be greater than start.")
        return self


EdgeInterval = Annotated[
    Union[FractionEdgeInterval, CoordinateEdgeInterval],
    Field(discriminator="kind"),
]


class RectangleEdgeSelector(StrictModel):
    kind: Literal["rectangle_edge"] = "rectangle_edge"
    edge: Literal["left", "right", "bottom", "top"]
    interval: EdgeInterval


class ExpertRegionSelector(StrictModel):
    kind: Literal["expert_region"] = "expert_region"
    region: RegionSpec

    @field_validator("region")
    @classmethod
    def _bounded_region(cls, value):
        return parse_region(value)


BoundarySelector = Annotated[
    Union[RectangleEdgeSelector, ExpertRegionSelector],
    Field(discriminator="kind"),
]


class BoundaryNodeSelector(StrictModel):
    kind: Literal["boundary_node"] = "boundary_node"
    point: Point2D = Field(
        description="Requested physical point on the rectangular domain boundary."
    )


SupportSelector = Annotated[
    Union[RectangleEdgeSelector, ExpertRegionSelector, BoundaryNodeSelector],
    Field(discriminator="kind"),
]


class FixedBoundaryCondition(StrictModel):
    bc_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    kind: Literal["fixed"] = "fixed"
    selector: BoundarySelector
    value: Vector2D = (0.0, 0.0)

    @field_validator("value")
    @classmethod
    def _zero_only(cls, value):
        if any(float(component) != 0.0 for component in value):
            raise ValueError(
                "agent-safe 2.1 supports only full-vector zero clamps [0,0]."
            )
        return value


class ZeroDisplacementBoundaryCondition(StrictModel):
    bc_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    kind: Literal["zero_displacement"]
    selector: SupportSelector
    components: Annotated[
        tuple[Literal["x", "y"], ...],
        Field(min_length=1, max_length=2),
    ]

    @field_validator("components")
    @classmethod
    def _canonical_components(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("components must be unique.")
        if tuple(sorted(value)) != value:
            raise ValueError("components must use canonical order ['x', 'y'].")
        return value


class UniformTractionBoundaryCondition(StrictModel):
    bc_id: str = Field(pattern=r"^L[1-9][0-9]*$")
    kind: Literal["uniform_traction"]
    selector: BoundarySelector
    traction: Vector2D


class UniformResultantBoundaryCondition(StrictModel):
    bc_id: str = Field(pattern=r"^L[1-9][0-9]*$")
    kind: Literal["uniform_resultant"]
    selector: BoundarySelector
    resultant: Vector2D


BoundaryCondition = Annotated[
    Union[
        FixedBoundaryCondition,
        ZeroDisplacementBoundaryCondition,
        UniformTractionBoundaryCondition,
        UniformResultantBoundaryCondition,
    ],
    Field(discriminator="kind"),
]


class FemModel(StrictModel):
    analysis_type: Literal["plane_strain"] = "plane_strain"
    young_modulus: PositiveFiniteNumber = 100.0
    poisson_ratio: FiniteNumber
    boundary_conditions: Annotated[list[BoundaryCondition], Field(min_length=1)]
    body_force: Vector2D = (0.0, 0.0)

    @field_validator("poisson_ratio")
    @classmethod
    def _poisson_range(cls, value):
        if not -1.0 < value < 0.5:
            raise ValueError(f"must satisfy -1 < nu < 0.5; got {value}.")
        return value

    @model_validator(mode="after")
    def _boundary_identity_and_support(self):
        ids = [bc.bc_id for bc in self.boundary_conditions]
        if len(ids) != len(set(ids)):
            raise ValueError("boundary-condition IDs must be unique.")
        if not any(
            bc.kind in {"fixed", "zero_displacement"}
            for bc in self.boundary_conditions
        ):
            raise ValueError(
                "at least one zero-displacement boundary condition is required."
            )
        return self


class AgentSafeConfig(StrictModel):
    schema_version: Literal[CONFIG_SCHEMA_VERSION] = CONFIG_SCHEMA_VERSION
    units: MechanicalUnits
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
        if (
            isinstance(self.units, LegacyConsistentUnits)
            and any(
                bc.kind == "uniform_resultant"
                for bc in self.fem.boundary_conditions
            )
        ):
            raise ValueError(
                "uniform_resultant requires explicit length, force, and stress "
                "units; legacy_consistent units cannot verify the conversion."
            )
        return self


class PreviousFemModel(FemModel):
    boundary_conditions: Annotated[
        list[
            Annotated[
                Union[
                    FixedBoundaryCondition,
                    UniformTractionBoundaryCondition,
                    UniformResultantBoundaryCondition,
                ],
                Field(discriminator="kind"),
            ]
        ],
        Field(min_length=1),
    ]


class PreviousAgentSafeConfig(AgentSafeConfig):
    """Read-only parser for deterministic canonical 2.0 migration."""

    schema_version: Literal[PREVIOUS_CONFIG_SCHEMA_VERSION]
    fem: PreviousFemModel


AgentSafeConfigInput = Annotated[
    Union[AgentSafeConfig, PreviousAgentSafeConfig, LegacyAgentSafeConfig],
    Field(discriminator="schema_version"),
]

# Transitional import name used by existing internal callers. It now denotes the
# strict agent-safe schema; it is not the former lambda/path-capable model.
ConfigModel = AgentSafeConfig


def migrate_legacy_config(config: dict) -> dict:
    """Convert a validated 1.1 config to canonical 2.1 without model help."""
    legacy = LegacyAgentSafeConfig.model_validate({
        **config,
        "schema_version": config.get(
            "schema_version", LEGACY_CONFIG_SCHEMA_VERSION
        ),
    })
    conditions: list[dict] = []
    for index, bc in enumerate(legacy.fem.dirichlet_bcs, start=1):
        conditions.append({
            "bc_id": f"S{index}",
            "kind": "fixed",
            "selector": {
                "kind": "expert_region",
                "region": bc.marker.model_dump(mode="json"),
            },
            "value": list(bc.value),
        })
    for index, bc in enumerate(legacy.fem.traction_bcs, start=1):
        conditions.append({
            "bc_id": f"L{index}",
            "kind": "uniform_traction",
            "selector": {
                "kind": "expert_region",
                "region": bc.locator.model_dump(mode="json"),
            },
            "traction": list(bc.value),
        })
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "units": {"kind": "legacy_consistent"},
        "mesh": legacy.mesh.model_dump(mode="json"),
        "fem": {
            "analysis_type": legacy.fem.analysis_type,
            "young_modulus": legacy.fem.young_modulus,
            "poisson_ratio": legacy.fem.poisson_ratio,
            "boundary_conditions": conditions,
            "body_force": list(legacy.fem.body_force),
        },
        "opt": legacy.opt.model_dump(mode="json"),
    }


def parse_agent_safe_config(
    config: (
        AgentSafeConfig
        | PreviousAgentSafeConfig
        | LegacyAgentSafeConfig
        | dict
    ),
) -> tuple[AgentSafeConfig, bool]:
    """Parse canonical 2.1 or deterministically migrate 2.0/legacy 1.1."""
    if isinstance(config, PreviousAgentSafeConfig):
        config = config.model_dump(mode="json")
    elif isinstance(config, AgentSafeConfig):
        return config, False
    if isinstance(config, LegacyAgentSafeConfig):
        config = config.model_dump(mode="json")
    version = config.get("schema_version")
    if version == PREVIOUS_CONFIG_SCHEMA_VERSION:
        previous = PreviousAgentSafeConfig.model_validate(config)
        migrated = previous.model_dump(mode="json")
        migrated["schema_version"] = CONFIG_SCHEMA_VERSION
        return AgentSafeConfig.model_validate(migrated), True
    looks_legacy = (
        version == LEGACY_CONFIG_SCHEMA_VERSION
        or (
            version is None
            and isinstance(config.get("fem"), dict)
            and "dirichlet_bcs" in config["fem"]
        )
    )
    if looks_legacy:
        return AgentSafeConfig.model_validate(migrate_legacy_config(config)), True
    return AgentSafeConfig.model_validate(config), False


def compile_solver_config(
    config: AgentSafeConfig | dict,
    *,
    solver_profile: Literal["auto", "iterative", "direct"] = "auto",
    output_folder: str = "results",
    output_prefix: str | None = None,
    output_interval: int = 20,
) -> dict:
    """Deterministically compile safe physics into the legacy solver dictionary."""
    model, _ = parse_agent_safe_config(config)
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
            "unit_context": model.units.model_dump(mode="json"),
            "dirichlet_bcs": [
                {
                    "bc_id": bc.bc_id,
                    "selector": bc.selector.model_dump(mode="json"),
                    **(
                        {"value": list(bc.value)}
                        if bc.kind == "fixed"
                        else {"components": list(bc.components)}
                    ),
                }
                for bc in model.fem.boundary_conditions
                if bc.kind in {"fixed", "zero_displacement"}
            ],
            "traction_bcs": [
                {
                    "bc_id": bc.bc_id,
                    "quantity_kind": (
                        "traction"
                        if bc.kind == "uniform_traction"
                        else "resultant"
                    ),
                    "value": list(
                        bc.traction
                        if bc.kind == "uniform_traction"
                        else bc.resultant
                    ),
                    "selector": bc.selector.model_dump(mode="json"),
                }
                for bc in model.fem.boundary_conditions
                if bc.kind in {"uniform_traction", "uniform_resultant"}
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
    if model.fem.poisson_ratio >= 0.49:
        ratio = (
            2 * float(model.fem.poisson_ratio)
            / (1 - 2 * float(model.fem.poisson_ratio))
        )
        warnings.append(
            f"fem.poisson_ratio={model.fem.poisson_ratio} is near "
            "incompressible in plane strain "
            f"(lambda/mu={ratio:.3g}); low-order displacement elements may "
            "exhibit volumetric locking, so interpret the result cautiously."
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
