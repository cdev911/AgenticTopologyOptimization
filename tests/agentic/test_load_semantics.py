"""Package 2 unit and semantic-load regression tests."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from agentic.boundary_draft import (
    BOUNDARY_FIELDS,
    BoundaryConditionDraft,
    BoundaryDraftState,
    BoundaryFieldFact,
)
from agentic.formulation import (
    DraftUpdate,
    FormulationTurn,
    ProblemDraft,
    assess_mechanical_units,
    assess_semantic_boundary_loads,
    merge_formulation_turn,
)
from agentic.load_semantics import (
    resolve_boundary_load_state,
    resolve_direction,
    resolve_semantic_load,
)
from agentic.mechanical_units import (
    MechanicalUnitContext,
    normalize_scalar,
    normalize_vector,
    resultant_to_traction,
)


def load_condition(**values) -> BoundaryConditionDraft:
    facts = tuple(
        BoundaryFieldFact(
            field=field,
            value=value,
            basis="explicit",
            source_turn=1,
            source_quote="load",
            rationale="Test load fact.",
        )
        for field, value in sorted(
            values.items(), key=lambda item: BOUNDARY_FIELDS.index(item[0])
        )
    )
    return BoundaryConditionDraft(
        bc_id="L1",
        kind="load",
        created_turn=1,
        facts=facts,
    )


class MechanicalUnitTests(unittest.TestCase):
    def setUp(self):
        self.context = MechanicalUnitContext(
            length_unit="mm",
            force_unit="N",
            stress_unit="MPa",
        )

    def test_engineering_context_records_implicit_unit_thickness(self):
        self.assertEqual(self.context.thickness_value, 1.0)
        self.assertEqual(self.context.thickness_unit, "mm")

    def test_context_rejects_wrong_dimensions_and_unknown_units(self):
        for field, value in (
            ("length_unit", "N"),
            ("force_unit", "MPa"),
            ("stress_unit", "N/mm"),
            ("length_unit", "not_a_unit"),
        ):
            values = {
                "length_unit": "mm",
                "force_unit": "N",
                "stress_unit": "MPa",
            }
            values[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValidationError):
                    MechanicalUnitContext(**values)

    def test_current_contract_rejects_nonunit_thickness(self):
        with self.assertRaises(ValidationError):
            MechanicalUnitContext(
                length_unit="mm",
                force_unit="N",
                stress_unit="MPa",
                thickness_value=2,
            )

    def test_scalar_normalization_retains_original_display_value(self):
        value = normalize_scalar(1, "m", "length", self.context)
        self.assertEqual(value.original_value, 1.0)
        self.assertEqual(value.original_unit, "m")
        self.assertEqual(value.normalized_value, 1000.0)
        self.assertEqual(value.normalized_unit, "mm")

    def test_force_and_stress_normalization(self):
        force = normalize_scalar(2.5, "kN", "force", self.context)
        stress = normalize_scalar(750, "kPa", "stress", self.context)
        self.assertEqual(force.normalized_value, 2500.0)
        self.assertAlmostEqual(stress.normalized_value, 0.75)

    def test_vector_normalization_is_json_safe(self):
        value = normalize_vector((1, -2), "GPa", "stress", self.context)
        self.assertEqual(value.normalized_value, (1000.0, -2000.0))
        dumped = value.model_dump(mode="json")
        self.assertEqual(dumped["original_unit"], "GPa")
        self.assertNotIn("Quantity", repr(dumped))

    def test_normalization_rejects_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "not a stress unit"):
            normalize_scalar(10, "N", "stress", self.context)

    def test_resultant_conversion_respects_unit_scale_and_round_trips(self):
        context = MechanicalUnitContext(
            length_unit="cm",
            force_unit="kN",
            stress_unit="MPa",
        )
        traction, integrated = resultant_to_traction(
            (0, -1),
            boundary_measure=1,
            context=context,
        )
        self.assertEqual(traction, (0.0, -10.0))
        self.assertEqual(integrated, (0.0, -1.0))


class MechanicalUnitDraftTests(unittest.TestCase):
    def test_unit_context_uses_existing_fact_provenance_and_ordering(self):
        message = "Use mm for length, N for force, and MPa for stress."
        turn = FormulationTurn(
            assistant_message="Recorded the unit system.",
            updates=(
                DraftUpdate(
                    path="units.length",
                    value="mm",
                    basis="explicit",
                    source_quote="mm",
                    rationale="Explicit length unit.",
                ),
                DraftUpdate(
                    path="units.force",
                    value="N",
                    basis="explicit",
                    source_quote="N",
                    rationale="Explicit force unit.",
                ),
                DraftUpdate(
                    path="units.stress",
                    value="MPa",
                    basis="explicit",
                    source_quote="MPa",
                    rationale="Explicit stress unit.",
                ),
            ),
        )
        merged = merge_formulation_turn(
            ProblemDraft(),
            turn,
            user_message=message,
            turn_number=1,
        )
        self.assertEqual(merged.issues, ())
        readiness = assess_mechanical_units(merged.draft)
        self.assertTrue(readiness.ready)
        self.assertEqual(readiness.context.length_unit, "mm")
        self.assertEqual(
            [fact.path for fact in merged.draft.facts],
            ["units.length", "units.force", "units.stress"],
        )
        condition = load_condition(**{
            "load.kind": "traction_vector",
            "load.vector": [0, -1],
            "load.distribution": "uniform",
            "selector.kind": "whole_edge",
            "selector.edge": "right",
        })
        draft = merged.draft.model_copy(update={
            "boundary_state": BoundaryDraftState(
                conditions=(condition,),
                next_load_number=2,
            )
        })
        semantic = assess_semantic_boundary_loads(draft)
        self.assertTrue(semantic.semantic_ready)
        self.assertTrue(semantic.execution_ready)
        self.assertIsNotNone(semantic.boundary_loads)

    def test_inferred_unit_remains_pending_assumption(self):
        turn = FormulationTurn(
            assistant_message="I inferred a compatible unit system.",
            updates=(
                DraftUpdate(
                    path="units.length",
                    value="mm",
                    basis="explicit",
                    source_quote="mm",
                    rationale="Explicit.",
                ),
                DraftUpdate(
                    path="units.force",
                    value="N",
                    basis="explicit",
                    source_quote="N",
                    rationale="Explicit.",
                ),
                DraftUpdate(
                    path="units.stress",
                    value="MPa",
                    basis="assumption",
                    rationale="N/mm^2 is MPa.",
                ),
            ),
        )
        merged = merge_formulation_turn(
            ProblemDraft(),
            turn,
            user_message="Use mm and N.",
            turn_number=1,
        )
        readiness = assess_mechanical_units(merged.draft)
        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.unconfirmed_fields, ("units.stress",))
        self.assertIsNone(readiness.context)

    def test_partial_context_lists_missing_facts(self):
        readiness = assess_mechanical_units(ProblemDraft())
        self.assertFalse(readiness.ready)
        self.assertEqual(
            readiness.missing_fields,
            ("units.length", "units.force", "units.stress"),
        )
        semantic = assess_semantic_boundary_loads(ProblemDraft())
        self.assertFalse(semantic.semantic_ready)
        self.assertIsNone(semantic.boundary_loads)

    def test_wrong_dimension_is_rejected_at_its_draft_path(self):
        turn = FormulationTurn(
            assistant_message="Recorded the proposed unit.",
            updates=(
                DraftUpdate(
                    path="units.length",
                    value="N",
                    basis="explicit",
                    source_quote="N",
                    rationale="User called N the length unit.",
                ),
            ),
        )
        merged = merge_formulation_turn(
            ProblemDraft(),
            turn,
            user_message="Use N as the length unit.",
            turn_number=1,
        )
        self.assertEqual(merged.accepted_paths, ())
        self.assertEqual(merged.issues[0].code, "invalid_value")
        self.assertEqual(merged.issues[0].path, "units.length")


class DirectionResolutionTests(unittest.TestCase):
    def test_global_directions(self):
        expected = {
            "up": (0.0, 1.0),
            "down": (0.0, -1.0),
            "left": (-1.0, 0.0),
            "right": (1.0, 0.0),
        }
        for direction, vector in expected.items():
            with self.subTest(direction=direction):
                self.assertEqual(resolve_direction(direction, None), vector)

    def test_edge_local_inward_and_outward(self):
        outward = {
            "left": (-1.0, 0.0),
            "right": (1.0, 0.0),
            "bottom": (0.0, -1.0),
            "top": (0.0, 1.0),
        }
        for edge, vector in outward.items():
            with self.subTest(edge=edge):
                self.assertEqual(resolve_direction("outward", edge), vector)
                self.assertEqual(
                    resolve_direction("inward", edge),
                    (-vector[0], -vector[1]),
                )

    def test_ambiguous_local_directions_are_rejected(self):
        for direction in ("x", "y", "normal", "tangential"):
            with self.subTest(direction=direction):
                with self.assertRaises(ValueError):
                    resolve_direction(direction, "right")
        with self.assertRaisesRegex(ValueError, "named rectangle edge"):
            resolve_direction("inward", None)


class SemanticLoadTests(unittest.TestCase):
    def setUp(self):
        self.context = MechanicalUnitContext(
            length_unit="mm",
            force_unit="N",
            stress_unit="MPa",
        )

    def test_traction_vector_resolves_and_normalizes(self):
        result = resolve_semantic_load(
            load_condition(**{
                "load.kind": "traction_vector",
                "load.vector": [0, -750],
                "load.unit": "kPa",
                "load.distribution": "uniform",
                "selector.kind": "whole_edge",
                "selector.edge": "right",
            }),
            self.context,
        )
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.load.quantity_kind, "traction")
        self.assertEqual(result.load.vector.normalized_value, (0.0, -0.75))
        self.assertEqual(result.load.effective_traction, result.load.vector)
        self.assertFalse(result.load.requires_boundary_measure)

    def test_magnitude_and_global_direction_resolve(self):
        result = resolve_semantic_load(
            load_condition(**{
                "load.kind": "traction_magnitude",
                "load.magnitude": 3,
                "load.direction": "down",
                "load.unit": "MPa",
                "load.distribution": "uniform",
            }),
            self.context,
        )
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.load.vector.normalized_value, (0.0, -3.0))

    def test_pressure_resolves_against_edge_normal(self):
        result = resolve_semantic_load(
            load_condition(**{
                "load.kind": "pressure",
                "load.magnitude": 5,
                "load.direction": "inward",
                "load.unit": "MPa",
                "load.distribution": "uniform",
                "selector.kind": "whole_edge",
                "selector.edge": "right",
            }),
            self.context,
        )
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.load.vector.normalized_value, (-5.0, 0.0))

    def test_resultant_is_normalized_but_mesh_conversion_is_deferred(self):
        condition = load_condition(**{
                "load.kind": "resultant_magnitude",
                "load.magnitude": 2,
                "load.direction": "down",
                "load.unit": "kN",
                "load.distribution": "uniform",
                "selector.kind": "centered_width",
                "selector.edge": "right",
                "selector.center": 0.5,
                "selector.width": 1,
            })
        result = resolve_semantic_load(condition, self.context)
        self.assertEqual(result.status, "deferred")
        self.assertEqual(result.load.quantity_kind, "resultant")
        self.assertEqual(result.load.vector.normalized_value, (0.0, -2000.0))
        self.assertTrue(result.load.requires_boundary_measure)
        self.assertIsNone(result.load.effective_traction)
        self.assertEqual(result.load.thickness_value, 1.0)
        self.assertEqual(result.load.thickness_unit, "mm")
        combined = resolve_boundary_load_state(
            BoundaryDraftState(
                conditions=(condition,),
                next_load_number=2,
            ),
            self.context,
        )
        self.assertTrue(combined.semantic_ready)
        self.assertFalse(combined.execution_ready)

    def test_resolved_traction_is_semantically_and_execution_ready(self):
        condition = load_condition(**{
            "load.kind": "traction_vector",
            "load.vector": [0, -1],
            "load.distribution": "uniform",
            "selector.kind": "whole_edge",
            "selector.edge": "right",
        })
        combined = resolve_boundary_load_state(
            BoundaryDraftState(
                conditions=(condition,),
                next_load_number=2,
            ),
            self.context,
        )
        self.assertTrue(combined.semantic_ready)
        self.assertTrue(combined.execution_ready)

    def test_context_unit_is_used_when_load_has_no_separate_unit(self):
        result = resolve_semantic_load(
            load_condition(**{
                "load.kind": "traction_vector",
                "load.vector": [1, 0],
                "load.distribution": "uniform",
            }),
            self.context,
        )
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.load.unit_source, "mechanical_context")
        self.assertEqual(result.load.vector.original_unit, "MPa")

    def test_dimensional_mismatch_is_visible(self):
        result = resolve_semantic_load(
            load_condition(**{
                "load.kind": "traction_vector",
                "load.vector": [0, -1],
                "load.unit": "N",
                "load.distribution": "uniform",
            }),
            self.context,
        )
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.issues[0].code, "invalid_unit")

    def test_ambiguous_direction_is_visible(self):
        result = resolve_semantic_load(
            load_condition(**{
                "load.kind": "pressure",
                "load.magnitude": 2,
                "load.direction": "normal",
                "load.distribution": "uniform",
                "selector.edge": "top",
            }),
            self.context,
        )
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.issues[0].code, "invalid_direction")

    def test_unsupported_physics_is_not_silently_reinterpreted(self):
        for kind in ("point_force", "moment", "varying_traction"):
            with self.subTest(kind=kind):
                result = resolve_semantic_load(
                    load_condition(**{
                        "load.kind": kind,
                        "load.distribution": "uniform",
                    }),
                    self.context,
                )
                self.assertEqual(result.status, "unsupported")
                self.assertEqual(
                    result.issues[0].code, "unsupported_load_kind"
                )

    def test_nonuniform_distribution_is_unsupported(self):
        result = resolve_semantic_load(
            load_condition(**{
                "load.kind": "traction_vector",
                "load.vector": [0, -1],
                "load.distribution": "varying",
            }),
            self.context,
        )
        self.assertEqual(result.status, "unsupported")
        self.assertEqual(
            result.issues[0].code, "unsupported_distribution"
        )

    def test_zero_vector_is_invalid(self):
        result = resolve_semantic_load(
            load_condition(**{
                "load.kind": "traction_vector",
                "load.vector": [0, 0],
                "load.distribution": "uniform",
            }),
            self.context,
        )
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.issues[0].code, "zero_load")


if __name__ == "__main__":
    unittest.main()
