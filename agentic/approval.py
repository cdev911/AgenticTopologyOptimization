"""Deterministic run-approval language and parameter presentation."""

from __future__ import annotations

import re
from typing import Literal

from agentic.boundary_presentation import validated_boundary_cards
from agentic.compiler import CompilationResult
from fenitop.tools.contracts import ValidateConfigResponse


ApprovalDecision = Literal["approve", "reject", "request_changes"]

_APPROVALS = {
    "approve",
    "approved",
    "go ahead",
    "proceed",
    "run it",
    "start",
    "start the run",
    "yes",
    "yes approve",
    "yes proceed",
    "yes run it",
    "yes start",
    "yes start the run",
}
_REJECTIONS = {
    "cancel",
    "do not run",
    "don't run",
    "no",
    "no do not run",
    "no don't run",
    "not yet",
    "stop",
}


def classify_run_approval(message: str) -> ApprovalDecision:
    """Recognize only an unambiguous whole-message green light."""
    normalized = re.sub(r"[.!?]+$", "", " ".join(message.casefold().split()))
    if normalized in _APPROVALS:
        return "approve"
    if normalized in _REJECTIONS:
        return "reject"
    return "request_changes"


def format_run_approval_request(
    compilation: CompilationResult,
    validation: ValidateConfigResponse,
) -> str:
    """Render the exact validated proposal before any solver side effect."""
    if validation.status != "ok" or validation.geometry_report is None:
        raise ValueError(
            "Run approval can only be rendered from successful mesh validation."
        )
    config = compilation.config
    bounds = config.mesh.bounds
    divisions = config.mesh.divisions
    boundary_lines = [
        "- "
        + " — ".join((f"{card.bc_id} {card.title}", card.physics))
        + "; "
        + "; ".join((card.location, *card.details))
        + (f"; Warning: {card.warning}" if card.warning else "")
        for card in validated_boundary_cards(config, validation)
    ]
    spring_lines = []
    if compilation.spring_evidence:
        records = {
            item.path: item for item in validation.geometry_report.entities
        }
        for item in compilation.spring_evidence:
            path = (
                "config.opt.in_spring.region"
                if item.role == "input"
                else "config.opt.out_spring.region"
            )
            record = records.get(path)
            matched = (
                f"{record.count} matched directional nodal DOFs"
                if record is not None
                else "matched-node evidence unavailable"
            )
            spring_lines.append(
                f"- {item.spring_id} {item.role.title()} spring — "
                f"{item.direction}-direction at {item.location}; "
                f"{item.original_stiffness:g} {item.original_unit} → "
                f"{item.normalized_stiffness:g} {item.normalized_unit} per "
                f"matched directional DOF; {matched}"
            )
    warning_lines = [
        f"- `{warning.code}`: {warning.message}"
        for warning in validation.warnings
    ]
    estimated = validation.estimated_cost
    cost_line = (
        f"- Estimated run: {estimated.num_elements} elements, "
        f"{estimated.max_iter} maximum iterations, "
        f"about {estimated.estimated_wall_time_seconds:.1f} seconds "
        f"(`{estimated.risk_level}` risk)"
        if estimated is not None
        else "- Estimated run: unavailable"
    )
    return "\n".join(
        [
            compilation.defaults_notice,
            "",
            "Validated parameter summary:",
            f"- Problem: `{config.opt.problem_type}` / "
            f"`{config.fem.analysis_type}`",
            f"- Domain: {bounds[0]} to {bounds[1]}",
            f"- Mesh: {divisions[0]} × {divisions[1]} "
            f"{config.mesh.cell_type} elements",
            f"- Material: E={config.fem.young_modulus}, "
            f"ν={config.fem.poisson_ratio}",
            f"- Material fraction: {config.opt.vol_frac}",
            f"- Filter radius: {config.opt.filter_radius}",
            (
                "- Mechanical units: unlabeled legacy-consistent values; "
                "implicit thickness=1"
                if config.units.kind == "legacy_consistent"
                else (
                    f"- Mechanical units: length={config.units.length_unit}, "
                    f"force={config.units.force_unit}, "
                    f"stress={config.units.stress_unit}; implicit thickness=1 "
                    f"{config.units.length_unit}"
                )
            ),
            "- Boundary conditions (requested → mesh-resolved):",
            *boundary_lines,
            *(
                ("- Mechanism springs:", *spring_lines)
                if spring_lines
                else ()
            ),
            *(
                (
                    "",
                    "Validation warnings requiring review:",
                    *warning_lines,
                )
                if warning_lines
                else ()
            ),
            cost_line,
            "",
            "Do you approve these parameters and want me to start the run? "
            "Reply **yes** to start, **no** to keep it stopped, or describe "
            "the changes you want.",
        ]
    )
