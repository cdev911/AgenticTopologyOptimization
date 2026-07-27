from __future__ import annotations

import unittest

from agentic.boundary_draft import (
    BOUNDARY_FIELDS,
    BoundaryConditionDraft,
    BoundaryDraftState,
    BoundaryFieldFact,
)
from agentic.approval import format_run_approval_request
from agentic.compiler import (
    DEFAULT_PROFILE_VERSION,
    FormulationFinalizationError,
    compile_formulation_draft,
    compile_intent,
)
from agentic.formulation import (
    DRAFT_PATHS,
    DraftFact,
    ProblemDraft,
    assess_draft,
)
from agentic.intent import (
    ComplianceProblemIntent,
    MechanismProblemIntent,
)
from fenitop.tools.config_models import (
    ComplianceOptimization,
    MechanismOptimization,
)
from fenitop.tools.contracts import ValidateConfigResponse
from fenitop.tools.validate_config import validate_config_tool


def compliance_data(bounds=((0, 0), (10, 10))):
    return {
        "problem_type": "minimize_compliance",
        "domain": {"bounds": bounds},
        "material": {"young_modulus": 100, "poisson_ratio": 0.3},
        "supports": [
            {"region": {"op": "plane", "axis": "x", "value": bounds[0][0]}}
        ],
        "tractions": [
            {
                "region": {"op": "plane", "axis": "x", "value": bounds[1][0]},
                "vector": [0, -1],
            }
        ],
        "volume_fraction": 0.4,
    }


def _draft_fact(path, value, *, basis="explicit"):
    return DraftFact(
        path=path,
        value=value,
        basis=basis,
        source_turn=1,
        source_quote="fixture",
        rationale="Package 4 fixture.",
    )


def _condition(bc_id, kind, fields, *, assumed_field=None):
    return BoundaryConditionDraft(
        bc_id=bc_id,
        kind=kind,
        created_turn=1,
        facts=tuple(
            BoundaryFieldFact(
                field=field,
                value=value,
                basis="assumption" if field == assumed_field else "explicit",
                source_turn=1,
                source_quote=None if field == assumed_field else "fixture",
                rationale="Package 4 fixture.",
            )
            for field, value in sorted(
                fields.items(), key=lambda item: BOUNDARY_FIELDS.index(item[0])
            )
        ),
    )


def first_class_draft(*, load_kind="resultant_vector", assumed_field=None):
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
            "load.kind": load_kind,
            "load.vector": [0, -100] if load_kind.startswith("resultant") else [0, -2],
            "load.distribution": "uniform",
            "selector.kind": "distance_from_corner",
            "selector.edge": "right",
            "selector.from_corner": "lower_right",
            "selector.offset": 0.5,
            "selector.length": 1.0,
        },
        assumed_field=assumed_field,
    )
    return ProblemDraft(
        facts=tuple(
            _draft_fact(path, ordinary[path])
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


class CompilerTests(unittest.TestCase):
    def test_named_corner_pin_resolves_from_nonzero_negative_domain_bounds(self):
        draft = first_class_draft()
        pin = _condition(
            "S4",
            "support",
            {
                "support.kind": "pin",
                "selector.kind": "boundary_point",
                "selector.from_corner": "lower_left",
            },
        )
        load = draft.boundary_state.condition("L7")
        facts = tuple(
            fact.model_copy(update={"value": [[-5, -2], [5, 2]]})
            if fact.path == "domain.bounds"
            else fact
            for fact in draft.facts
        )
        draft = draft.model_copy(update={
            "facts": facts,
            "boundary_state": BoundaryDraftState(
                conditions=(pin, load),
                next_support_number=5,
                next_load_number=8,
            ),
        })

        result = compile_formulation_draft(draft)
        self.assertEqual(
            result.config.fem.boundary_conditions[0].selector.point,
            (-5.0, -2.0),
        )

    def test_semantic_mechanism_springs_compile_regions_and_convert_units(self):
        draft = first_class_draft(load_kind="traction_vector")
        ordinary = {
            fact.path: fact for fact in draft.facts
        }
        ordinary["problem_type"] = _draft_fact(
            "problem_type", "compliant_mechanism"
        )
        ordinary["compliance_bound"] = _draft_fact("compliance_bound", 10)
        ordinary["units.length"] = _draft_fact("units.length", "mm")
        ordinary["units.force"] = _draft_fact("units.force", "N")
        ordinary["units.stress"] = _draft_fact("units.stress", "MPa")
        spring_fields = {
            "spring.direction": "x",
            "spring.stiffness": 10,
            "spring.unit": "N/m",
            "selector.kind": "centered_fraction",
            "selector.center": 0.5,
            "selector.span": 0.2,
        }
        input_spring = _condition(
            "I1", "input_spring", {**spring_fields, "selector.edge": "left"}
        )
        output_spring = _condition(
            "O1", "output_spring", {**spring_fields, "selector.edge": "right"}
        )
        supports = tuple(
            _condition(
                f"S{index}",
                "support",
                {
                    "support.kind": "fixed_all",
                    "selector.kind": "fraction_interval",
                    "selector.edge": edge,
                    "selector.start": start,
                    "selector.end": end,
                },
            )
            for index, (edge, start, end) in enumerate(
                (
                    ("left", 0.0, 0.2),
                    ("left", 0.8, 1.0),
                    ("right", 0.0, 0.2),
                    ("right", 0.8, 1.0),
                ),
                start=1,
            )
        )
        load = _condition(
            "L1",
            "load",
            {
                "load.kind": "traction_vector",
                "load.vector": [1, 0],
                "load.distribution": "uniform",
                "selector.kind": "centered_fraction",
                "selector.edge": "left",
                "selector.center": 0.5,
                "selector.span": 0.2,
            },
        )
        draft = draft.model_copy(update={
            "facts": tuple(
                ordinary[path] for path in DRAFT_PATHS if path in ordinary
            ),
            "boundary_state": BoundaryDraftState(
                conditions=(
                    *supports,
                    load,
                    input_spring,
                    output_spring,
                ),
                next_support_number=5,
                next_load_number=2,
                next_input_spring_number=2,
                next_output_spring_number=2,
            ),
        })

        self.assertTrue(assess_draft(draft).ready, assess_draft(draft))
        result = compile_formulation_draft(draft)
        self.assertAlmostEqual(result.config.opt.in_spring.stiffness, 0.01)
        self.assertEqual(
            result.config.opt.in_spring.region.model_dump(mode="json"),
            {
                "op": "and",
                "regions": [
                    {
                        "op": "plane",
                        "axis": "x",
                        "value": 0,
                        "tol": 1e-8,
                    },
                    {
                        "op": "range",
                        "axis": "y",
                        "min": 1.6,
                        "max": 2.4,
                        "min_inclusive": True,
                        "max_inclusive": True,
                    },
                ],
            },
        )
        self.assertEqual(result.spring_evidence[0].spring_id, "I1")
        self.assertEqual(result.spring_evidence[0].original_unit, "N/m")
        self.assertEqual(result.spring_evidence[0].normalized_unit, "(N)/(mm)")
        validation = validate_config_tool({"config": result.config})
        self.assertEqual(validation["status"], "ok", validation["errors"])
        spring_records = [
            item
            for item in validation["geometry_report"]["entities"]
            if ".spring.region" in item["path"]
            or "_spring.region" in item["path"]
        ]
        self.assertEqual([item["count"] for item in spring_records], [1, 1])
        approval = format_run_approval_request(
            result,
            ValidateConfigResponse.model_validate(validation),
        )
        self.assertIn("I1 Input spring", approval)
        self.assertIn("10 N/m → 0.01 (N)/(mm)", approval)
        self.assertIn("1 matched directional nodal DOFs", approval)

    def test_roller_and_true_pin_compile_to_component_and_node_constraints(self):
        draft = first_class_draft()
        roller = _condition(
            "S4",
            "support",
            {
                "support.kind": "roller_normal",
                "selector.kind": "whole_edge",
                "selector.edge": "bottom",
            },
        )
        pin = _condition(
            "S5",
            "support",
            {
                "support.kind": "pin",
                "selector.kind": "boundary_point",
                "selector.point": [0, 4],
            },
        )
        load = draft.boundary_state.condition("L7")
        draft = draft.model_copy(update={
            "boundary_state": BoundaryDraftState(
                conditions=(roller, pin, load),
                next_support_number=6,
                next_load_number=8,
            )
        })

        result = compile_formulation_draft(draft)
        roller_config, pin_config, _ = result.config.fem.boundary_conditions
        self.assertEqual(roller_config.kind, "zero_displacement")
        self.assertEqual(roller_config.components, ("y",))
        self.assertEqual(roller_config.selector.kind, "rectangle_edge")
        self.assertEqual(pin_config.kind, "zero_displacement")
        self.assertEqual(pin_config.components, ("x", "y"))
        self.assertEqual(pin_config.selector.kind, "boundary_node")
        self.assertEqual(pin_config.selector.point, (0.0, 4.0))

        validation = validate_config_tool({"config": result.config})
        self.assertEqual(validation["status"], "ok", validation["errors"])
        pin_record = next(
            record
            for record in validation["geometry_report"]["entities"]
            if record["bc_id"] == "S5"
        )
        self.assertEqual(pin_record["entity_kind"], "node")
        self.assertEqual(pin_record["resolved_point"], [0.0, 4.0])
        self.assertEqual(pin_record["constrained_components"], ["x", "y"])
        self.assertEqual(validation["geometry_report"]["rigid_body_rank"], 3)

    def test_first_class_resultant_compiles_stable_ids_units_and_selectors(self):
        draft = first_class_draft()
        self.assertTrue(assess_draft(draft).ready)

        result = compile_formulation_draft(draft)

        self.assertEqual(result.config.units.kind, "explicit")
        self.assertEqual(
            [item.bc_id for item in result.config.fem.boundary_conditions],
            ["S4", "L7"],
        )
        support, load = result.config.fem.boundary_conditions
        self.assertEqual(
            support.selector.interval.model_dump(mode="json"),
            {"kind": "coordinate", "start": 1.0, "end": 3.0},
        )
        self.assertEqual(load.kind, "uniform_resultant")
        self.assertEqual(load.resultant, (0.0, -100.0))
        self.assertEqual(
            load.selector.interval.model_dump(mode="json"),
            {"kind": "coordinate", "start": 0.5, "end": 1.5},
        )

        validation = validate_config_tool({"config": result.config})
        self.assertEqual(validation["status"], "ok", validation["errors"])
        evidence = next(
            item
            for item in validation["geometry_report"]["entities"]
            if item["bc_id"] == "L7"
        )
        self.assertEqual(evidence["quantity_kind"], "resultant")
        self.assertAlmostEqual(evidence["integrated_resultant"][1], -100.0)

    def test_first_class_traction_is_normalized_to_context_stress_unit(self):
        draft = first_class_draft(load_kind="traction_vector")
        load = draft.boundary_state.condition("L7")
        facts = tuple(
            fact.model_copy(update={"value": "kPa"})
            if fact.field == "load.unit"
            else fact
            for fact in load.facts
        )
        if not any(fact.field == "load.unit" for fact in facts):
            facts = tuple(sorted(
                (
                    *facts,
                    BoundaryFieldFact(
                        field="load.unit",
                        value="kPa",
                        basis="explicit",
                        source_turn=1,
                        source_quote="fixture",
                        rationale="Package 4 fixture.",
                    ),
                ),
                key=lambda fact: BOUNDARY_FIELDS.index(fact.field),
            ))
        conditions = (
            draft.boundary_state.conditions[0],
            load.model_copy(update={"facts": facts}),
        )
        draft = draft.model_copy(update={
            "boundary_state": draft.boundary_state.model_copy(
                update={"conditions": conditions}
            )
        })

        result = compile_formulation_draft(draft)
        compiled_load = result.config.fem.boundary_conditions[1]

        self.assertEqual(compiled_load.kind, "uniform_traction")
        self.assertEqual(compiled_load.traction, (0.0, -2000.0))

    def test_unconfirmed_first_class_bc_cannot_cross_finalization(self):
        with self.assertRaisesRegex(
            FormulationFinalizationError,
            "not complete and confirmed",
        ):
            compile_formulation_draft(
                first_class_draft(assumed_field="selector.length")
            )

    def test_square_domain_defaults_to_50_by_50(self):
        result = compile_intent(
            ComplianceProblemIntent.model_validate(compliance_data())
        )

        self.assertEqual(result.config.mesh.divisions, (50, 50))
        self.assertEqual(result.config.opt.filter_radius, 0.3)
        self.assertEqual(result.defaults_profile, DEFAULT_PROFILE_VERSION)

    def test_rectangle_keeps_elements_nearly_square_and_area_near_target(self):
        result = compile_intent(
            ComplianceProblemIntent.model_validate(
                compliance_data(bounds=((0, 0), (60, 20)))
            )
        )
        nx, ny = result.config.mesh.divisions
        dx, dy = 60 / nx, 20 / ny

        self.assertEqual((nx, ny), (87, 29))
        self.assertAlmostEqual(dx, dy)
        self.assertLess(abs(nx * ny - 2500), 100)

    def test_extreme_aspect_ratio_preserves_square_cells_for_later_admission(self):
        result = compile_intent(
            ComplianceProblemIntent.model_validate(
                compliance_data(bounds=((0, 0), (10000, 1)))
            )
        )
        nx, ny = result.config.mesh.divisions

        self.assertEqual((nx, ny), (20000, 2))
        self.assertAlmostEqual(10000 / nx, 1 / ny)
        self.assertLess(result.config.opt.filter_radius, 1)

    def test_user_preferences_override_defaults_and_are_not_misattributed(self):
        data = compliance_data()
        data["mesh"] = {"divisions": [40, 40], "cell_type": "triangle"}
        data["optimization"] = {"filter_radius": 0.75, "max_iter": 123}

        result = compile_intent(ComplianceProblemIntent.model_validate(data))
        paths = {item.path for item in result.applied_defaults}

        self.assertEqual(result.config.mesh.divisions, (40, 40))
        self.assertEqual(result.config.mesh.cell_type, "triangle")
        self.assertEqual(result.config.opt.filter_radius, 0.75)
        self.assertEqual(result.config.opt.max_iter, 123)
        self.assertNotIn("mesh.divisions", paths)
        self.assertNotIn("mesh.cell_type", paths)
        self.assertNotIn("opt.filter_radius", paths)
        self.assertNotIn("opt.max_iter", paths)

    def test_centered_percentage_edge_segment_compiles_and_matches_facets(self):
        data = compliance_data(bounds=((0, 0), (10, 5)))
        data["material"] = {"young_modulus": 10, "poisson_ratio": 0.499}
        data["volume_fraction"] = 0.33
        data["tractions"] = [
            {
                "edge_segment": {
                    "edge": "right",
                    "center_fraction": 0.5,
                    "span_fraction": 0.1,
                },
                "vector": [0, -1],
            }
        ]

        result = compile_intent(ComplianceProblemIntent.model_validate(data))
        load = next(
            bc
            for bc in result.config.fem.boundary_conditions
            if bc.kind == "uniform_traction"
        )
        selector = load.selector.model_dump(mode="json")

        self.assertEqual(selector["kind"], "rectangle_edge")
        self.assertEqual(
            selector,
            {
                "kind": "rectangle_edge",
                "edge": "right",
                "interval": {
                    "kind": "fraction",
                    "start": 0.45,
                    "end": 0.55,
                },
            },
        )
        validation = validate_config_tool({"config": result.config})
        self.assertEqual(validation["status"], "ok", validation["errors"])
        traction_record = next(
            item
            for item in validation["geometry_report"]["entities"]
            if item["bc_id"] == "L1"
        )
        self.assertGreater(traction_record["count"], 0)
        self.assertEqual(traction_record["requested_extent"], [2.25, 2.75])
        self.assertEqual(traction_record["outward_normal"], [1.0, 0.0])

    def test_whole_edge_segment_compiles_to_full_boundary(self):
        data = compliance_data(bounds=((0, 0), (10, 4)))
        data["tractions"] = [
            {
                "region": {"op": "none"},
                "edge_segment": {
                    "edge": "right",
                    "center_fraction": 0.5,
                    "span_fraction": 1.0,
                },
                "vector": [0, -1],
            }
        ]

        result = compile_intent(ComplianceProblemIntent.model_validate(data))
        validation = validate_config_tool({"config": result.config})

        self.assertEqual(validation["status"], "ok", validation["errors"])
        traction_record = next(
            item
            for item in validation["geometry_report"]["entities"]
            if item["bc_id"] == "L1"
        )
        self.assertEqual(
            traction_record["count"],
            result.config.mesh.divisions[1],
        )

    def test_compliance_mapping_and_default_notice_are_explicit(self):
        result = compile_intent(
            ComplianceProblemIntent.model_validate(compliance_data())
        )

        self.assertIsInstance(result.config.opt, ComplianceOptimization)
        self.assertEqual(result.config.opt.optimizer, "oc")
        self.assertEqual(result.config.opt.max_iter, 400)
        self.assertEqual(result.config.opt.initial_density, 0.4)
        self.assertIn("values were not provided", result.defaults_notice)
        self.assertIn("mesh.divisions = (50, 50)", result.defaults_notice)
        self.assertIn("You can request changes", result.defaults_notice)
        self.assertIn("Review these choices before approving", result.defaults_notice)

    def test_mechanism_mapping_uses_mma_profile_and_exact_springs(self):
        data = compliance_data()
        data.update(
            {
                "problem_type": "compliant_mechanism",
                "compliance_bound": 2.5,
                "input_spring": {
                    "region": {"op": "plane", "axis": "x", "value": 0},
                    "direction": "x",
                    "stiffness": 0.2,
                },
                "output_spring": {
                    "region": {"op": "plane", "axis": "x", "value": 10},
                    "direction": "y",
                    "stiffness": 0.3,
                },
            }
        )

        result = compile_intent(MechanismProblemIntent.model_validate(data))

        self.assertIsInstance(result.config.opt, MechanismOptimization)
        self.assertEqual(result.config.opt.optimizer, "mma")
        self.assertEqual(result.config.opt.max_iter, 500)
        self.assertEqual(result.config.opt.move, 0.05)
        self.assertEqual(result.config.opt.in_spring.direction, "x")
        self.assertEqual(result.config.opt.out_spring.stiffness, 0.3)


if __name__ == "__main__":
    unittest.main()
