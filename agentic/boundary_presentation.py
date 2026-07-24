"""Deterministic, provider-independent presentation of boundary conditions.

This module deliberately consumes typed draft state or validated geometry
evidence.  It does not decide readiness, resolve facets, or authorize execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
import math

from agentic.boundary_draft import (
    BoundaryConditionDraft,
    BoundaryDraftState,
    assess_boundary_state,
)
from fenitop.tools.config_models import AgentSafeConfig, BoundaryCondition
from fenitop.tools.contracts import EntityMatchRecord, ValidateConfigResponse


@dataclass(frozen=True)
class BoundaryCard:
    """Small human-facing summary for one stable boundary-condition entity."""

    bc_id: str
    title: str
    status: str
    physics: str
    location: str
    details: tuple[str, ...]
    correction_hint: str
    warning: str | None = None


def _number(value: float | int) -> str:
    return f"{float(value):.6g}"


def _vector(value: tuple[float, float] | list[float]) -> str:
    return f"[{_number(value[0])}, {_number(value[1])}]"


def _draft_location(condition: BoundaryConditionDraft) -> str:
    values = condition.values()
    kind = values.get("selector.kind")
    edge = values.get("selector.edge")
    edge_text = f"{edge} edge" if edge else "boundary edge not yet specified"
    if kind == "whole_edge":
        return f"whole {edge_text}"
    if kind == "centered_fraction":
        span = values.get("selector.span")
        return (
            f"centered {_number(float(span) * 100)}% of the {edge_text}"
            if span is not None
            else f"centered segment of the {edge_text}; span not yet specified"
        )
    if kind == "fraction_interval":
        start, end = values.get("selector.start"), values.get("selector.end")
        if start is not None and end is not None:
            return (
                f"{edge_text}, from {_number(float(start) * 100)}% to "
                f"{_number(float(end) * 100)}% along the edge"
            )
        return f"fractional interval on the {edge_text}; endpoints incomplete"
    if kind == "coordinate_interval":
        start, end = values.get("selector.start"), values.get("selector.end")
        if start is not None and end is not None:
            coordinate = "y" if edge in {"left", "right"} else "x"
            return (
                f"{edge_text}, {coordinate}={_number(start)} to "
                f"{_number(end)}"
            )
        return f"coordinate interval on the {edge_text}; endpoints incomplete"
    if kind == "centered_width":
        width = values.get("selector.width")
        return (
            f"centered width {_number(width)} on the {edge_text}"
            if width is not None
            else f"centered physical segment on the {edge_text}; width incomplete"
        )
    if kind == "distance_from_corner":
        corner = str(values.get("selector.from_corner", "unspecified corner"))
        offset, length = values.get("selector.offset"), values.get("selector.length")
        if offset is not None and length is not None:
            return (
                f"{edge_text}, length {_number(length)} starting "
                f"{_number(offset)} from {corner.replace('_', ' ')}"
            )
        return f"corner-relative segment on the {edge_text}; extent incomplete"
    if kind == "boundary_point":
        point = values.get("selector.point")
        return (
            f"boundary point {_vector(point)}"
            if point is not None
            else f"point on the {edge_text}; location incomplete"
        )
    if kind == "expert_region":
        return (
            "expert region "
            + json.dumps(values.get("selector.region"), separators=(",", ":"))
        )
    if kind in {"unspecified_extent", "unspecified"}:
        return f"{edge_text}; extent not yet specified"
    return "location not yet specified"


def _draft_physics(condition: BoundaryConditionDraft) -> str:
    values = condition.values()
    if condition.kind == "support":
        names = {
            "fixed_all": "full-vector zero clamp",
            "roller_normal": "normal roller support",
            "roller_x": "x roller support",
            "roller_y": "y roller support",
            "symmetry": "symmetry support",
            "pin": "point-node pin",
        }
        return names.get(values.get("support.kind"), "support type not yet specified")

    kind = values.get("load.kind")
    vector = values.get("load.vector")
    magnitude = values.get("load.magnitude")
    direction = values.get("load.direction")
    unit = values.get("load.unit")
    distribution = values.get("load.distribution")
    names = {
        "traction_vector": "traction",
        "traction_magnitude": "traction",
        "resultant_vector": "total resultant",
        "resultant_magnitude": "total resultant",
        "pressure": "pressure",
        "point_force": "mathematical point force",
        "moment": "moment",
        "varying_traction": "varying traction",
    }
    parts = [names.get(kind, "load type not yet specified")]
    if vector is not None:
        parts.append(_vector(vector))
    elif magnitude is not None:
        parts.append(_number(magnitude))
        if direction is not None:
            parts.append(str(direction))
    if unit is not None:
        parts.append(str(unit))
    if distribution is not None:
        parts.append(f"{distribution} distribution")
    return " · ".join(parts)


def draft_boundary_cards(state: BoundaryDraftState) -> tuple[BoundaryCard, ...]:
    """Present partial BCs without pretending they are mesh-resolved."""
    readiness = {
        item.bc_id: item for item in assess_boundary_state(state).conditions
    }
    cards = []
    for condition in state.conditions:
        check = readiness[condition.bc_id]
        details = []
        if check.missing_fields:
            details.append("Still needed: " + ", ".join(check.missing_fields))
        if check.unconfirmed_fields:
            details.append(
                "Needs confirmation: " + ", ".join(check.unconfirmed_fields)
            )
        if check.capability_limits:
            details.append(
                "Current capability limit: " + ", ".join(check.capability_limits)
            )
        details.extend(check.semantic_errors)
        status = "ready for deterministic compilation" if check.ready else "in progress"
        title = "Support" if condition.kind == "support" else "Load"
        cards.append(
            BoundaryCard(
                bc_id=condition.bc_id,
                title=title,
                status=status,
                physics=_draft_physics(condition),
                location=_draft_location(condition),
                details=tuple(details),
                correction_hint=(
                    f'Say “Change {condition.bc_id} …” to revise only this '
                    f'{condition.kind}.'
                ),
            )
        )
    return tuple(cards)


def _selector_text(
    bc: BoundaryCondition,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> str:
    selector = bc.selector
    if selector.kind == "rectangle_edge":
        interval = selector.interval
        axis = "y" if selector.edge in {"left", "right"} else "x"
        if interval.kind == "fraction":
            if interval.start == 0 and interval.end == 1:
                return f"whole {selector.edge} edge"
            return (
                f"{selector.edge} edge, {_number(interval.start * 100)}% to "
                f"{_number(interval.end * 100)}% along the edge"
            )
        return (
            f"{selector.edge} edge, {axis}={_number(interval.start)} to "
            f"{_number(interval.end)}"
        )
    region = selector.region.model_dump(mode="json")
    if region.get("op") == "plane":
        axis = region["axis"]
        return f"boundary where {axis}={_number(region['value'])}"
    return "expert declarative boundary region (see detailed provenance)"


def _resolved_text(record: EntityMatchRecord) -> str:
    base = f"{record.count} mesh facets"
    if record.resolved_extent is not None:
        base += (
            f", extent {_number(record.resolved_extent[0])} to "
            f"{_number(record.resolved_extent[1])}"
        )
    if record.measure is not None:
        base += f", measure {_number(record.measure)}"
    return base


def validated_boundary_cards(
    config: AgentSafeConfig,
    validation: ValidateConfigResponse,
) -> tuple[BoundaryCard, ...]:
    """Present the exact proposal paired with authoritative mesh evidence."""
    if validation.status != "ok" or validation.geometry_report is None:
        raise ValueError("Validated BC cards require successful geometry evidence.")
    evidence = {
        item.bc_id: item
        for item in validation.geometry_report.entities
        if item.bc_id is not None
    }
    cards = []
    for bc in config.fem.boundary_conditions:
        record = evidence.get(bc.bc_id)
        if record is None:
            raise ValueError(f"Geometry evidence is missing for {bc.bc_id}.")
        if bc.kind == "fixed":
            title = "Support"
            physics = "full-vector zero clamp"
            details = (f"Mesh-resolved: {_resolved_text(record)}",)
        else:
            if record.input_vector is None:
                raise ValueError(f"Validated load evidence is incomplete for {bc.bc_id}.")
            title = "Load"
            input_unit = (
                record.stress_unit
                if bc.kind == "uniform_traction"
                else record.force_unit
            )
            physics = (
                "uniform effective traction "
                if bc.kind == "uniform_traction"
                else "uniform total resultant "
            ) + _vector(record.input_vector)
            if input_unit:
                physics += f" {input_unit}"
            detail_items = [f"Mesh-resolved: {_resolved_text(record)}"]
            if record.effective_traction is not None:
                unit = f" {record.stress_unit}" if record.stress_unit else ""
                detail_items.append(
                    "Applied traction: "
                    f"{_vector(record.effective_traction)}{unit}"
                )
            if record.integrated_resultant is not None:
                unit = f" {record.force_unit}" if record.force_unit else ""
                detail_items.append(
                    "Integrated resultant: "
                    f"{_vector(record.integrated_resultant)}{unit}"
                )
            if record.thickness_value is not None:
                unit = f" {record.thickness_unit}" if record.thickness_unit else ""
                detail_items.append(
                    f"Implicit thickness: {_number(record.thickness_value)}{unit}"
                )
            details = tuple(detail_items)
        cards.append(
            BoundaryCard(
                bc_id=bc.bc_id,
                title=title,
                status="validated",
                physics=physics,
                location="Requested: " + _selector_text(bc, config.mesh.bounds),
                details=details,
                correction_hint=(
                    f'Say “Change {bc.bc_id} …” to revise only this '
                    f'{"support" if bc.kind == "fixed" else "load"}; '
                    "the proposal will be revalidated before approval."
                ),
                warning=record.resolution_warning,
            )
        )
    return tuple(cards)


def _edge_segment(
    edge: str,
    extent: tuple[float, float],
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    (x0, y0), (x1, y1) = bounds
    if edge == "left":
        return ((x0, extent[0]), (x0, extent[1]))
    if edge == "right":
        return ((x1, extent[0]), (x1, extent[1]))
    if edge == "bottom":
        return ((extent[0], y0), (extent[1], y0))
    return ((extent[0], y1), (extent[1], y1))


def _inferred_edge(
    record: EntityMatchRecord,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> str | None:
    (x0, y0), (x1, y1) = bounds
    (rx0, ry0), (rx1, ry1) = record.bounds
    scale = max(1.0, x1 - x0, y1 - y0)
    candidates = {
        "left": abs(rx0 - x0) + abs(rx1 - x0),
        "right": abs(rx0 - x1) + abs(rx1 - x1),
        "bottom": abs(ry0 - y0) + abs(ry1 - y0),
        "top": abs(ry0 - y1) + abs(ry1 - y1),
    }
    edge, error = min(candidates.items(), key=lambda item: item[1])
    return edge if error <= 1e-7 * scale else None


def boundary_preview_svg(
    config: AgentSafeConfig,
    validation: ValidateConfigResponse,
) -> str:
    """Draw a deterministic rectangle preview from validated geometry evidence."""
    if validation.status != "ok" or validation.geometry_report is None:
        raise ValueError("Boundary preview requires successful geometry evidence.")
    bounds = config.mesh.bounds
    (x0, y0), (x1, y1) = bounds
    width, height = x1 - x0, y1 - y0
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise ValueError("Preview bounds must be finite.")

    canvas_w, canvas_h = 760.0, 420.0
    plot_w, plot_h = 560.0, 270.0
    scale = min(plot_w / width, plot_h / height)
    drawn_w, drawn_h = width * scale, height * scale
    left = (canvas_w - drawn_w) / 2
    top = 58.0 + (plot_h - drawn_h) / 2

    def point(x: float, y: float) -> tuple[float, float]:
        return (left + (x - x0) * scale, top + (y1 - y) * scale)

    evidence = {
        item.bc_id: item
        for item in validation.geometry_report.entities
        if item.bc_id is not None
    }
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 420" '
        'role="img" aria-label="Validated boundary-condition preview">',
        "<defs>",
        '<marker id="arrow" markerWidth="8" markerHeight="8" refX="6" '
        'refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" '
        'fill="#b42318"/></marker>',
        "</defs>",
        '<rect width="760" height="420" fill="#ffffff"/>',
        '<text x="24" y="30" font-family="sans-serif" font-size="17" '
        'font-weight="600" fill="#182230">Validated boundary conditions</text>',
        (
            f'<rect x="{left:.3f}" y="{top:.3f}" width="{drawn_w:.3f}" '
            f'height="{drawn_h:.3f}" fill="#f8fafc" stroke="#344054" '
            'stroke-width="2"/>'
        ),
    ]

    for bc in config.fem.boundary_conditions:
        record = evidence.get(bc.bc_id)
        if record is None:
            raise ValueError(f"Geometry evidence is missing for {bc.bc_id}.")
        edge = (
            bc.selector.edge
            if bc.selector.kind == "rectangle_edge"
            else _inferred_edge(record, bounds)
        )
        resolved_extent = record.resolved_extent
        if resolved_extent is None and edge is not None:
            (rx0, ry0), (rx1, ry1) = record.bounds
            resolved_extent = (
                (ry0, ry1) if edge in {"left", "right"} else (rx0, rx1)
            )
        if edge is None or resolved_extent is None:
            continue
        resolved = _edge_segment(edge, resolved_extent, bounds)
        requested_extent = record.requested_extent
        if requested_extent is not None:
            requested = _edge_segment(edge, requested_extent, bounds)
            (ax, ay), (bx, by) = (point(*requested[0]), point(*requested[1]))
            parts.append(
                f'<line x1="{ax:.3f}" y1="{ay:.3f}" x2="{bx:.3f}" '
                f'y2="{by:.3f}" stroke="#f79009" stroke-width="9" '
                'stroke-dasharray="7 5" opacity="0.65"/>'
            )
        (ax, ay), (bx, by) = (point(*resolved[0]), point(*resolved[1]))
        color = "#175cd3" if bc.kind == "fixed" else "#b42318"
        parts.append(
            f'<line x1="{ax:.3f}" y1="{ay:.3f}" x2="{bx:.3f}" '
            f'y2="{by:.3f}" stroke="{color}" stroke-width="6" '
            'stroke-linecap="round"/>'
        )
        cx, cy = (ax + bx) / 2, (ay + by) / 2
        if bc.kind == "fixed":
            # Compact clamp teeth point out from the supported edge.
            outward = {
                "left": (-1, 0),
                "right": (1, 0),
                "bottom": (0, 1),
                "top": (0, -1),
            }[edge]
            for fraction in (0.15, 0.35, 0.55, 0.75, 0.95):
                tx, ty = ax + (bx - ax) * fraction, ay + (by - ay) * fraction
                parts.append(
                    f'<line x1="{tx:.3f}" y1="{ty:.3f}" '
                    f'x2="{tx + outward[0] * 12:.3f}" '
                    f'y2="{ty + outward[1] * 12:.3f}" '
                    'stroke="#175cd3" stroke-width="2"/>'
                )
        else:
            vector = record.effective_traction or record.input_vector
            if vector and math.hypot(*vector) > 0:
                norm = math.hypot(*vector)
                dx, dy = 36 * vector[0] / norm, -36 * vector[1] / norm
                parts.append(
                    f'<line x1="{cx - dx:.3f}" y1="{cy - dy:.3f}" '
                    f'x2="{cx:.3f}" y2="{cy:.3f}" stroke="#b42318" '
                    'stroke-width="3" marker-end="url(#arrow)"/>'
                )
        label_x = cx + (14 if edge in {"right", "top"} else -34)
        label_y = cy - 10
        parts.append(
            f'<text x="{label_x:.3f}" y="{label_y:.3f}" '
            f'font-family="sans-serif" font-size="15" font-weight="700" '
            f'fill="{color}">{escape(bc.bc_id)}</text>'
        )

    parts.extend(
        [
            '<line x1="28" y1="365" x2="62" y2="365" stroke="#f79009" '
            'stroke-width="7" stroke-dasharray="7 5" opacity="0.65"/>',
            '<text x="72" y="370" font-family="sans-serif" font-size="13" '
            'fill="#475467">requested continuous extent</text>',
            '<line x1="282" y1="365" x2="316" y2="365" stroke="#175cd3" '
            'stroke-width="6"/>',
            '<text x="326" y="370" font-family="sans-serif" font-size="13" '
            'fill="#475467">resolved support</text>',
            '<line x1="480" y1="365" x2="514" y2="365" stroke="#b42318" '
            'stroke-width="6"/>',
            '<text x="524" y="370" font-family="sans-serif" font-size="13" '
            'fill="#475467">resolved load</text>',
            '<text x="28" y="402" font-family="sans-serif" font-size="12" '
            'fill="#667085">Diagram is schematic; card values are authoritative.</text>',
            "</svg>",
        ]
    )
    return "".join(parts)
