"""Deterministically compile semantic intent into the agent-safe tool contract."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from agentic.intent import (
    ComplianceProblemIntent,
    MechanismProblemIntent,
    ProblemIntent,
)
from fenitop.tools.config_models import AgentSafeConfig

DEFAULT_PROFILE_VERSION = "agentic-defaults-v1"
TARGET_SQUARE_DIVISIONS = 50
FILTER_CELL_MULTIPLIER = 1.5


class AppliedDefault(BaseModel):
    """One application-owned choice that was absent from user intent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    value: Any
    reason: str


class CompilationResult(BaseModel):
    """Exact tool config plus an inspectable account of compiler choices."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config: AgentSafeConfig
    defaults_profile: Literal["agentic-defaults-v1"] = DEFAULT_PROFILE_VERSION
    applied_defaults: tuple[AppliedDefault, ...]
    defaults_notice: str


def _mesh_divisions(bounds) -> tuple[int, int]:
    """Target ~2,500 near-square cells for any rectangular aspect ratio."""
    (x0, y0), (x1, y1) = bounds
    width = float(x1 - x0)
    height = float(y1 - y0)
    target_h = math.sqrt(width * height) / TARGET_SQUARE_DIVISIONS
    nx, ny = (
        max(1, math.floor(width / target_h + 0.5)),
        max(1, math.floor(height / target_h + 0.5)),
    )
    # A filter radius of 1.5 cells must remain smaller than either domain axis.
    # For extreme aspect ratios, preserve square cells by refining the long axis
    # after enforcing at least two cells on the short axis. Resource validation
    # remains responsible for rejecting an excessively large resulting mesh.
    if nx < 2:
        nx = 2
        ny = max(2, math.floor(height / (width / nx) + 0.5))
    if ny < 2:
        ny = 2
        nx = max(2, math.floor(width / (height / ny) + 0.5))
    return nx, ny


def _default_filter_radius(bounds, divisions: tuple[int, int]) -> float:
    (x0, y0), (x1, y1) = bounds
    dx = float(x1 - x0) / divisions[0]
    dy = float(y1 - y0) / divisions[1]
    return float(f"{FILTER_CELL_MULTIPLIER * max(dx, dy):.12g}")


def _record(
    defaults: list[AppliedDefault],
    path: str,
    value: Any,
    reason: str,
) -> Any:
    defaults.append(AppliedDefault(path=path, value=value, reason=reason))
    return value


def format_defaults_notice(
    profile: str,
    defaults: list[AppliedDefault] | tuple[AppliedDefault, ...],
) -> str:
    """Create the required editable transparency notice."""
    lines = [
        "The following values were not provided, so the deterministic compiler "
        f"selected them for you using {profile}:"
    ]
    lines.extend(
        f"- {item.path} = {item.value!r} — {item.reason}" for item in defaults
    )
    lines.append(
        "Review these choices before approving the run. You can request changes "
        "instead of approving."
    )
    return "\n".join(lines)


def compile_intent(intent: ProblemIntent) -> CompilationResult:
    """Compile validated semantic intent without consulting an LLM."""
    defaults: list[AppliedDefault] = []
    bounds = intent.domain.bounds

    divisions = intent.mesh.divisions
    if divisions is None:
        divisions = _record(
            defaults,
            "mesh.divisions",
            _mesh_divisions(bounds),
            "targets about 2,500 near-square elements (50×50 for a square domain)",
        )

    cell_type = intent.mesh.cell_type
    if cell_type is None:
        cell_type = _record(
            defaults,
            "mesh.cell_type",
            "quadrilateral",
            "the pinned v1 structured-mesh profile",
        )

    filter_radius = intent.optimization.filter_radius
    if filter_radius is None:
        filter_radius = _record(
            defaults,
            "opt.filter_radius",
            _default_filter_radius(bounds, divisions),
            "1.5 times the larger element edge length",
        )

    is_compliance = isinstance(intent, ComplianceProblemIntent)
    max_iter = intent.optimization.max_iter
    if max_iter is None:
        max_iter = _record(
            defaults,
            "opt.max_iter",
            400 if is_compliance else 500,
            "the pinned convergence budget for this problem type",
        )

    optimizer = _record(
        defaults,
        "opt.optimizer",
        "oc" if is_compliance else "mma",
        "the solver's trusted optimizer for this problem type",
    )
    initial_density = _record(
        defaults,
        "opt.initial_density",
        intent.volume_fraction,
        "starts from a uniform design satisfying the requested volume fraction",
    )
    opt_tol = _record(
        defaults, "opt.opt_tol", 1e-5, "the pinned v1 convergence tolerance"
    )
    penalty = _record(
        defaults, "opt.penalty", 3.0, "the pinned v1 SIMP penalization"
    )
    epsilon = _record(
        defaults, "opt.epsilon", 1e-6, "the pinned numerical regularization"
    )
    beta_interval = _record(
        defaults,
        "opt.beta_interval",
        50,
        "the pinned Heaviside continuation interval",
    )
    beta_max = _record(
        defaults,
        "opt.beta_max",
        128.0,
        "the pinned Heaviside continuation cap",
    )
    move = _record(
        defaults,
        "opt.move",
        0.02 if is_compliance else 0.05,
        "the pinned update limit for this problem type",
    )
    analysis_type = _record(
        defaults,
        "fem.analysis_type",
        "plane_strain",
        "the supported v1 constitutive assumption",
    )
    thickness = _record(
        defaults,
        "units.thickness_value",
        1.0,
        "the supported implicit out-of-plane thickness",
    )
    units = _record(
        defaults,
        "units.kind",
        "legacy_consistent",
        "preserves legacy values without inventing physical unit labels",
    )
    solid_zone = _record(
        defaults,
        "opt.solid_zone",
        {"op": "none"},
        "no mandatory solid region was specified by the intent contract",
    )
    void_zone = _record(
        defaults,
        "opt.void_zone",
        {"op": "none"},
        "no mandatory void region was specified by the intent contract",
    )

    mesh = {
        "bounds": bounds,
        "divisions": divisions,
        "cell_type": cell_type,
    }
    boundary_conditions = [
        {
            "bc_id": f"S{index}",
            "kind": "fixed",
            "selector": {
                "kind": "expert_region",
                "region": support.region,
            },
            "value": (0.0, 0.0),
        }
        for index, support in enumerate(intent.supports, start=1)
    ]
    for index, traction in enumerate(intent.tractions, start=1):
        if traction.edge_segment is not None:
            segment = traction.edge_segment
            start = segment.center_fraction - segment.span_fraction / 2
            end = segment.center_fraction + segment.span_fraction / 2
            selector = {
                "kind": "rectangle_edge",
                "edge": segment.edge,
                "interval": {
                    "kind": "fraction",
                    "start": start,
                    "end": end,
                },
            }
        else:
            selector = {
                "kind": "expert_region",
                "region": traction.region,
            }
        boundary_conditions.append({
            "bc_id": f"L{index}",
            "kind": "uniform_traction",
            "selector": selector,
            "traction": traction.vector,
        })

    fem = {
        "analysis_type": analysis_type,
        "young_modulus": intent.material.young_modulus,
        "poisson_ratio": intent.material.poisson_ratio,
        "boundary_conditions": boundary_conditions,
        "body_force": intent.body_force,
    }
    opt: dict[str, Any] = {
        "problem_type": intent.problem_type,
        "optimizer": optimizer,
        "max_iter": max_iter,
        "opt_tol": opt_tol,
        "vol_frac": intent.volume_fraction,
        "initial_density": initial_density,
        "penalty": penalty,
        "epsilon": epsilon,
        "filter_radius": filter_radius,
        "beta_interval": beta_interval,
        "beta_max": beta_max,
        "move": move,
        "solid_zone": solid_zone,
        "void_zone": void_zone,
    }
    if isinstance(intent, MechanismProblemIntent):
        opt.update(
            {
                "compliance_bound": intent.compliance_bound,
                "in_spring": {
                    "region": intent.input_spring.region,
                    "direction": intent.input_spring.direction,
                    "stiffness": intent.input_spring.stiffness,
                },
                "out_spring": {
                    "region": intent.output_spring.region,
                    "direction": intent.output_spring.direction,
                    "stiffness": intent.output_spring.stiffness,
                },
            }
        )

    config = AgentSafeConfig.model_validate({
        "units": {
            "kind": units,
            "thickness_value": thickness,
        },
        "mesh": mesh,
        "fem": fem,
        "opt": opt,
    })
    applied = tuple(defaults)
    return CompilationResult(
        config=config,
        applied_defaults=applied,
        defaults_notice=format_defaults_notice(DEFAULT_PROFILE_VERSION, applied),
    )
