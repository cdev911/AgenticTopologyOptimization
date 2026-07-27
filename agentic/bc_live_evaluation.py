"""Normalize live formulation state into the fixed BC evaluation contract.

The release evaluator imports this module, the conversational formulator, and
validation only.  It deliberately has no orchestration or solver dependency.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import re

from agentic.bc_evaluation import (
    AssumptionCode,
    BehaviorCode,
    BoundaryObservation,
    BoundaryScenario,
    CapabilityCode,
    ClarificationCode,
    ExpectedBoundaryCondition,
)
from agentic.boundary_draft import assess_boundary_state
from agentic.formulation import FormulationStep, ProblemDraft


def _text(steps: Sequence[FormulationStep], messages: Iterable[str]) -> str:
    return " ".join(
        (
            *messages,
            *(
                part
                for step in steps
                for part in (
                    step.turn.assistant_message,
                    *step.turn.questions,
                )
            ),
        )
    ).casefold()


def _condition_observation(
    draft: ProblemDraft,
    bc_id: str,
) -> ExpectedBoundaryCondition:
    condition = draft.boundary_state.condition(bc_id)
    if condition is None:  # pragma: no cover - guarded by caller
        raise ValueError(f"Unknown boundary condition {bc_id}.")
    values = condition.values()
    readiness = {
        item.bc_id: item
        for item in assess_boundary_state(draft.boundary_state).conditions
    }[bc_id]
    completeness = (
        "capability_limited"
        if readiness.capability_limits
        else ("complete" if readiness.ready else "partial")
    )
    selector_kind = values.get("selector.kind", "unspecified")
    payload = {
        "bc_id": bc_id,
        "kind": condition.kind,
        "completeness": completeness,
        "selector_kind": selector_kind,
    }
    mappings = {
        "support.kind": "support_kind",
        "support.direction": "direction",
        "support.magnitude": "magnitude",
        "support.unit": "unit",
        "load.kind": "load_kind",
        "load.vector": "vector",
        "load.magnitude": "magnitude",
        "load.direction": "direction",
        "load.unit": "unit",
        "load.distribution": "distribution",
        "selector.edge": "edge",
        "selector.start": "start",
        "selector.end": "end",
        "selector.center": "center",
        "selector.span": "span",
        "selector.width": "width",
        "selector.from_corner": "from_corner",
        "selector.point": "point",
    }
    facts = {fact.field: fact for fact in condition.facts}
    for source, target in mappings.items():
        if source in values and facts[source].basis != "assumption":
            payload[target] = values[source]
    if selector_kind == "distance_from_corner":
        if (
            "selector.offset" in values
            and facts["selector.offset"].basis != "assumption"
        ):
            payload["start"] = values["selector.offset"]
        if (
            "selector.length" in values
            and facts["selector.length"].basis != "assumption"
        ):
            payload["span"] = values["selector.length"]
    if (
        payload.get("unit") is None
        and selector_kind in {"centered_width", "distance_from_corner"}
    ):
        length = draft.fact("units.length")
        if length is not None and length.basis != "assumption":
            payload["unit"] = length.value
    if isinstance(payload.get("unit"), str):
        payload["unit"] = payload["unit"].replace("*", " ")
    required_selector_fields = {
        "whole_edge": {"edge"},
        "centered_fraction": {"edge", "center", "span"},
        "fraction_interval": {"edge", "start", "end"},
        "coordinate_interval": {"edge", "start", "end"},
        "centered_width": {"edge", "center", "width"},
        "distance_from_corner": {"edge", "from_corner", "start", "span"},
    }
    required = required_selector_fields.get(selector_kind, set())
    if selector_kind == "boundary_point":
        boundary_point_complete = (
            "point" in payload
            or {"edge", "center"}.issubset(payload)
            or "from_corner" in payload
        )
        required = set() if boundary_point_complete else {"point"}
    if required and not required.issubset(payload):
        payload["selector_kind"] = (
            "unspecified_extent" if payload.get("edge") else "unspecified"
        )
    if (
        payload["selector_kind"] == "unspecified_extent"
        and not payload.get("edge")
    ):
        payload["selector_kind"] = "unspecified"
    return ExpectedBoundaryCondition.model_validate(payload)


def _capabilities(
    steps: Sequence[FormulationStep],
) -> set[CapabilityCode]:
    final = steps[-1]
    supported = {
        "component_support",
        "point_node_pin",
        "mathematical_point_load",
        "nonzero_prescribed_displacement",
        "applied_moment",
        "varying_traction",
    }
    result = {
        item
        for item in final.session.unsupported_features
        if item in supported
    }
    for item in assess_boundary_state(
        final.session.draft.boundary_state
    ).conditions:
        result.update(
            value for value in item.capability_limits if value in supported
        )
    return result


def _clarifications(
    scenario: BoundaryScenario,
    steps: Sequence[FormulationStep],
    combined_text: str,
) -> set[ClarificationCode]:
    final = steps[-1]
    draft = final.session.draft
    result: set[ClarificationCode] = set()
    for condition in assess_boundary_state(
        draft.boundary_state
    ).conditions:
        values = draft.boundary_state.condition(condition.bc_id).values()
        missing = set(condition.missing_fields)
        if "selector.edge" in missing or "selector.kind" in missing:
            result.add("boundary_edge")
        if "selector.extent" in missing:
            result.add("boundary_extent")
        if missing & {"selector.start", "selector.end"}:
            result.add("boundary_coordinates")
        if "support.kind" in missing and not condition.capability_limits:
            result.add("support_kind")
        if "load.kind" in missing:
            result.add("load_quantity_kind")
        if "load.vector" in missing or "load.magnitude" in missing:
            result.add("load_magnitude")
        if "load.direction" in missing:
            result.add(
                "pressure_orientation"
                if values.get("load.kind") == "pressure"
                else "load_direction"
            )

    questions = " ".join(final.turn.questions).casefold()
    assistant = final.turn.assistant_message.casefold()
    response = f"{assistant} {questions}"
    current = " ".join(scenario.turns).casefold()
    if (
        "width" in current
        and "%" in current
        and any(word in response for word in ("axis", "vertical", "horizontal"))
    ):
        result.add("clarify_width_axis")
        result.discard("boundary_edge")
        result.discard("boundary_extent")
    if "load.kind" in {
        field
        for condition in assess_boundary_state(draft.boundary_state).conditions
        for field in condition.missing_fields
    }:
        if any(word in questions for word in ("magnitude", "value", "strength")):
            result.add("load_magnitude")
        if "direction" in questions:
            result.add("load_direction")
    if (
        "other load" in current
        and any(word in response for word in ("which", "id", "l1", "l2", "l3"))
    ):
        result.add("target_boundary_condition")
    if (
        "lower-left corner" in current
        and any(
            "selector.extent" in item.missing_fields
            or "selector.edge" in item.missing_fields
            or "selector.kind" in item.missing_fields
            for item in assess_boundary_state(
                draft.boundary_state
            ).conditions
        )
        and any(
            phrase in response
            for phrase in ("left edge", "bottom edge", "or both")
        )
    ):
        result.update(("boundary_edge", "boundary_extent"))
    if (
        "top edge" in current
        and "right edge" in current
    ):
        result.add("conflicting_boundary_location")
        result.discard("boundary_edge")
        result.discard("boundary_extent")
    if "fixed" in current and "roller" in current:
        result.add("conflicting_support_kind")
        result.discard("support_kind")
    if (
        "point" in current
        and "finite" in response
        and any(
            word in response
            for word in ("confirm", "accept", "would you", "should")
        )
    ):
        result.add("confirm_finite_patch")
    if any(
        fact.path.startswith("units.") and fact.basis == "assumption"
        for fact in draft.facts
    ):
        result.add("unit_system")
    has_problem_context = any(
        draft.fact(path) is not None
        for path in (
            "domain.bounds",
            "domain.width",
            "material.young_modulus",
        )
    )
    has_resultant = any(
        str(condition.values().get("load.kind", "")).startswith("resultant")
        for condition in draft.boundary_state.conditions
    )
    units_complete = all(
        draft.fact(path) is not None
        for path in ("units.length", "units.force", "units.stress")
    )
    if has_problem_context and has_resultant and not units_complete:
        result.add("unit_system")
    if any(
        condition.kind == "load"
        and condition.values().get("load.kind") is None
        and condition.values().get("load.magnitude") is not None
        and condition.values().get("load.unit") is None
        for condition in draft.boundary_state.conditions
    ):
        result.add("unit_system")
    if any(
        confirmation.field == "load.distribution"
        for confirmation in draft.boundary_state.pending_confirmations()
    ):
        result.add("confirm_uniform_distribution")
    if any(
        confirmation.field == "load.direction"
        for confirmation in draft.boundary_state.pending_confirmations()
    ):
        result.add("load_direction")
    if any(
        str(condition.values().get("load.kind", "")).startswith("resultant")
        and condition.values().get("load.distribution") is None
        for condition in draft.boundary_state.conditions
    ) and "uniform" in response:
        result.add("confirm_uniform_distribution")
    if "resultant" in combined_text and "conversion" in questions:
        result.add("confirm_resultant_conversion")
    return result


def _assumptions(
    steps: Sequence[FormulationStep],
    combined_text: str,
) -> set[AssumptionCode]:
    draft = steps[-1].session.draft
    result: set[AssumptionCode] = set()
    for condition in draft.boundary_state.conditions:
        for fact in condition.facts:
            if fact.basis != "assumption":
                continue
            if fact.field == "load.distribution" and fact.value == "uniform":
                result.add("uniform_distribution")
            if fact.field in {"selector.span", "selector.width"}:
                result.add("finite_patch_span")
    if any(
        fact.path.startswith("units.") and fact.basis == "assumption"
        for fact in draft.facts
    ):
        result.add("unit_system_n_mm_mpa")
    if (
        any(
            condition.values().get("load.kind") == "point_force"
            for condition in draft.boundary_state.conditions
        )
        and "finite" in combined_text
        and "patch" in combined_text
    ):
        result.update(("finite_patch_span", "uniform_distribution"))
    return result


def _behavior_violations(
    steps: Sequence[FormulationStep],
    user_text: str,
) -> set[BehaviorCode]:
    conditions = steps[-1].session.draft.boundary_state.conditions
    values = [condition.values() for condition in conditions]
    result: set[BehaviorCode] = set()
    if re.search(r"\bpoint\b", user_text) and any(
        item.get("load.kind") in {
            "traction_vector",
            "traction_magnitude",
            "resultant_vector",
            "resultant_magnitude",
        }
        for item in values
    ):
        result.add("point_to_traction_silently")
    if any(term in user_text for term in ("total force", "resultant")) and any(
        str(item.get("load.kind", "")).startswith("traction")
        for item in values
    ):
        result.add("resultant_to_traction_silently")
    if any(term in user_text for term in ("roller", "symmetry")) and any(
        item.get("support.kind") == "fixed_all" for item in values
    ):
        result.add("roller_to_clamp")
    if re.search(r"\bpin\b", user_text) and any(
        item.get("support.kind") == "fixed_all" for item in values
    ):
        result.add("pin_to_clamp")
    return result


def observation_from_steps(
    scenario: BoundaryScenario,
    steps: Sequence[FormulationStep],
) -> BoundaryObservation:
    """Produce the deterministic semantic observation used by the live gate."""
    if not steps:
        raise ValueError("At least one formulation step is required.")
    final = steps[-1]
    draft = final.session.draft
    combined = _text(steps, (scenario.context or "", *scenario.turns))
    user_text = " ".join((scenario.context or "", *scenario.turns)).casefold()
    capabilities = _capabilities(steps)
    clarifications = _clarifications(scenario, steps, combined)
    assumptions = _assumptions(steps, combined)
    conditions = tuple(
        _condition_observation(draft, item.bc_id)
        for item in draft.boundary_state.conditions
    )
    ambiguous = clarifications & {
        "support_kind",
        "clarify_width_axis",
        "conflicting_boundary_location",
        "conflicting_support_kind",
        "target_boundary_condition",
        "boundary_edge",
    }
    if (
        "tip load" in " ".join(scenario.turns).casefold()
        and "load_quantity_kind" in clarifications
    ):
        ambiguous.add("load_quantity_kind")
    if ambiguous:
        semantic_state = "ambiguous"
    elif capabilities:
        semantic_state = (
            "needs_reformulation"
            if "mathematical_point_load" in capabilities
            else "requires_capability"
        )
    elif conditions and all(
        item.completeness == "complete" for item in conditions
    ) and not clarifications and not assumptions:
        semantic_state = "complete"
    else:
        semantic_state = "incomplete"
    return BoundaryObservation(
        semantic_state=semantic_state,
        boundary_conditions=conditions,
        clarifications=tuple(sorted(clarifications)),
        assumptions=tuple(sorted(assumptions)),
        capability_limits=tuple(sorted(capabilities)),
        behavior_violations=tuple(
            sorted(_behavior_violations(steps, user_text))
        ),
        solver_started=False,
    )
