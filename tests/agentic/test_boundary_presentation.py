from __future__ import annotations

import unittest

from agentic.boundary_draft import (
    BOUNDARY_FIELDS,
    BoundaryConditionDraft,
    BoundaryDraftState,
    BoundaryFieldFact,
)
from agentic.boundary_presentation import (
    boundary_preview_svg,
    draft_boundary_cards,
    validated_boundary_cards,
)
from agentic.compiler import compile_formulation_draft
from agentic.formulation import DRAFT_PATHS, DraftFact, ProblemDraft
from fenitop.tools.contracts import ValidateConfigResponse
from fenitop.tools.validate_config import validate_config_tool


def _condition(bc_id, kind, fields, *, assumption=None):
    return BoundaryConditionDraft(
        bc_id=bc_id,
        kind=kind,
        created_turn=1,
        facts=tuple(
            BoundaryFieldFact(
                field=field,
                value=value,
                basis="assumption" if field == assumption else "explicit",
                source_turn=1,
                source_quote=None if field == assumption else "fixture",
                rationale="Presentation fixture.",
            )
            for field, value in sorted(
                fields.items(), key=lambda item: BOUNDARY_FIELDS.index(item[0])
            )
        ),
    )


def _draft():
    ordinary = {
        "problem_type": "minimize_compliance",
        "domain.bounds": [[0, 0], [10, 4]],
        "material.young_modulus": 100,
        "material.poisson_ratio": 0.3,
        "units.length": "m",
        "units.force": "N",
        "units.stress": "Pa",
        "volume_fraction": 0.4,
        "mesh.divisions": [10, 4],
    }
    support = _condition(
        "S4",
        "support",
        {
            "support.kind": "fixed_all",
            "selector.kind": "centered_width",
            "selector.edge": "left",
            "selector.center": 0.5,
            "selector.width": 2.0,
        },
    )
    load = _condition(
        "L7",
        "load",
        {
            "load.kind": "resultant_vector",
            "load.vector": [0, -100],
            "load.unit": "N",
            "load.distribution": "uniform",
            "selector.kind": "distance_from_corner",
            "selector.edge": "right",
            "selector.from_corner": "lower_right",
            "selector.offset": 0.5,
            "selector.length": 1.0,
        },
    )
    return ProblemDraft(
        facts=tuple(
            DraftFact(
                path=path,
                value=ordinary[path],
                basis="explicit",
                source_turn=1,
                source_quote="fixture",
                rationale="Presentation fixture.",
            )
            for path in DRAFT_PATHS
            if path in ordinary
        ),
        boundary_state=BoundaryDraftState(
            conditions=(support, load),
            next_support_number=5,
            next_load_number=8,
        ),
        turn_count=1,
    )


class BoundaryPresentationTests(unittest.TestCase):
    def test_partial_card_preserves_known_facts_and_names_missing_fields(self):
        load = _condition(
            "L2",
            "load",
            {
                "load.kind": "resultant_magnitude",
                "load.magnitude": 10,
                "load.unit": "N",
                "selector.kind": "unspecified_extent",
                "selector.edge": "right",
            },
        )

        card = draft_boundary_cards(
            BoundaryDraftState(conditions=(load,), next_load_number=3)
        )[0]

        self.assertEqual(card.bc_id, "L2")
        self.assertIn("total resultant", card.physics)
        self.assertIn("10", card.physics)
        self.assertIn("right edge", card.location)
        self.assertIn("extent not yet specified", card.location)
        self.assertIn("load.direction", card.details[0])
        self.assertIn("Change L2", card.correction_hint)

    def test_validated_cards_show_mesh_resolution_and_resultant_conversion(self):
        compilation = compile_formulation_draft(_draft())
        validation = ValidateConfigResponse.model_validate(
            validate_config_tool({"config": compilation.config})
        )

        cards = validated_boundary_cards(compilation.config, validation)
        self.assertEqual([card.bc_id for card in cards], ["S4", "L7"])
        load = cards[1]
        self.assertIn("total resultant [0, -100]", load.physics)
        self.assertIn("right edge", load.location)
        self.assertTrue(
            any("Applied traction: [0, -50] Pa" in item for item in load.details)
        )
        self.assertTrue(
            any("Integrated resultant: [0, -100] N" in item for item in load.details)
        )
        self.assertTrue(any("extent 0 to 2" in item for item in load.details))

    def test_svg_uses_stable_ids_requested_resolved_styles_and_load_arrow(self):
        compilation = compile_formulation_draft(_draft())
        validation = ValidateConfigResponse.model_validate(
            validate_config_tool({"config": compilation.config})
        )

        first = boundary_preview_svg(compilation.config, validation)
        second = boundary_preview_svg(compilation.config, validation)

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("<svg"))
        self.assertIn(">S4</text>", first)
        self.assertIn(">L7</text>", first)
        self.assertIn('stroke-dasharray="7 5"', first)
        self.assertIn('marker-end="url(#arrow)"', first)
        self.assertIn("requested continuous extent", first)
        self.assertNotIn("<script", first)

    def test_validated_presentation_fails_closed_without_geometry_evidence(self):
        compilation = compile_formulation_draft(_draft())
        failed = ValidateConfigResponse.model_validate(
            {
                "contract_version": "5.0.0",
                "tool": "validate_config",
                "status": "error",
                "warnings": [],
                "errors": [],
                "checked": {
                    "structural": True,
                    "resource": False,
                    "geometry": False,
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "successful geometry"):
            validated_boundary_cards(compilation.config, failed)
        with self.assertRaisesRegex(ValueError, "successful geometry"):
            boundary_preview_svg(compilation.config, failed)


if __name__ == "__main__":
    unittest.main()
