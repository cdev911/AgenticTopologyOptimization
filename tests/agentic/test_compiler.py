from __future__ import annotations

import unittest

from agentic.compiler import DEFAULT_PROFILE_VERSION, compile_intent
from agentic.intent import (
    ComplianceProblemIntent,
    MechanismProblemIntent,
)
from fenitop.tools.config_models import (
    ComplianceOptimization,
    MechanismOptimization,
)
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


class CompilerTests(unittest.TestCase):
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
        locator = result.config.fem.traction_bcs[0].locator.model_dump(mode="json")

        self.assertEqual(locator["op"], "and")
        self.assertEqual(
            locator["regions"],
            [
                {
                    "op": "plane",
                    "axis": "x",
                    "value": 10,
                    "tol": 1e-8,
                },
                {
                    "op": "range",
                    "axis": "y",
                    "min": 2.25,
                    "max": 2.75,
                    "min_inclusive": True,
                    "max_inclusive": True,
                },
            ],
        )
        validation = validate_config_tool({"config": result.config})
        self.assertEqual(validation["status"], "ok", validation["errors"])
        traction_record = next(
            item
            for item in validation["geometry_report"]["entities"]
            if item["path"] == "config.fem.traction_bcs[0].locator"
        )
        self.assertGreater(traction_record["count"], 0)

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
        self.assertIn("Tell me if you want to change", result.defaults_notice)
        self.assertIn("otherwise the workflow will proceed", result.defaults_notice)

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
