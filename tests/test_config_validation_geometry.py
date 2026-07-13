"""Docker-only geometry tests for Tool 1 (fenitop.tools.validate_config).

Needs the full dolfinx/MPI stack -- run via:
    docker compose run --rm fenitop python -m unittest tests.test_config_validation_geometry -v

Scope note: the rigid-body-motion rank check itself (does a set of
constrained points resist all 3 planar rigid-body modes) is pure numpy and is
unit-tested directly with synthetic data in
tests/test_config_validation.py::RigidBodyRankTests, dolfinx-free. This file
deliberately does not try to force a real config into tripping that specific
warning through facet-matching alone: reliably constructing a config whose
marker matches facets belonging to exactly one mesh vertex (the only
degenerate case that mathematically produces rank < 3, since pinning any two
distinct points always fully constrains a 2D rigid body -- see that test
file's docstring) depends on dolfinx's locate_entities_boundary vertex
quantifier, which isn't worth coupling a test's correctness to. What
genuinely needs dolfinx -- and is covered here -- is real mesh construction
and real facet matching: confirming a marker matching zero facets is
rejected, and confirming a well-posed config's geometry check passes.
"""
import json
import unittest
from pathlib import Path

from fenitop.tools.validate_config import validate_config_tool
from fenitop.tools.contracts import TrustedValidationPolicy

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, from_fixtures: bool = False):
    directory = REPO_ROOT / "tests" / "fixtures" if from_fixtures else REPO_ROOT / "config"
    with open(directory / f"{name}.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


class GeometryValidationTests(unittest.TestCase):
    policy = TrustedValidationPolicy(check_geometry=True)

    def test_beam_2d_passes_full_validation(self):
        result = validate_config_tool({"config": _load("beam_2d")}, policy=self.policy)
        self.assertEqual(result["status"], "ok", result.get("errors"))
        self.assertEqual(
            result["checked"],
            {"structural": True, "resource": True, "geometry": True},
        )
        report = result["geometry_report"]
        self.assertEqual(report["rigid_body_rank"], 3)
        self.assertGreater(report["total_boundary_facets"], 0)
        self.assertTrue(
            any(
                entity["path"] == "config.fem.traction_bcs[0].locator"
                and entity["count"] > 0
                for entity in report["entities"]
            )
        )
        # A real, well-posed support (the full left edge) must not trip the
        # rigid-body warning.
        self.assertFalse(any("rigid-body" in w["message"] for w in result["warnings"]))

    def test_mechanism_2d_passes_full_validation(self):
        result = validate_config_tool({"config": _load("mechanism_2d")}, policy=self.policy)
        self.assertEqual(result["status"], "ok", result.get("errors"))
        self.assertEqual(
            result["checked"],
            {"structural": True, "resource": True, "geometry": True},
        )

    def test_smoke_fixture_passes_full_validation(self):
        result = validate_config_tool(
            {"config": _load("smoke_beam_2d", from_fixtures=True)}, policy=self.policy)
        self.assertEqual(result["status"], "ok", result.get("errors"))

    def test_dirichlet_marker_matching_zero_facets_is_rejected(self):
        # beam_2d's mesh spans x in [0, 60]; a marker at x=9999 cannot match
        # anything -- this is the exact silent-drop hole (fem.py:89-91) Tool 1
        # exists to close.
        raw = _load("beam_2d")
        raw["fem"]["dirichlet_bcs"][0]["marker"] = {"op": "plane", "axis": "x", "value": 9999}
        result = validate_config_tool({"config": raw}, policy=self.policy)
        self.assertEqual(result["status"], "error")
        self.assertTrue(any(
            e["path"] == "config.fem.dirichlet_bcs[0].marker" and "matched 0 of" in e["message"]
            for e in result["errors"]))

    def test_traction_locator_matching_zero_facets_is_rejected(self):
        raw = _load("beam_2d")
        raw["fem"]["traction_bcs"][0]["locator"] = {"op": "plane", "axis": "y", "value": 9999}
        result = validate_config_tool({"config": raw}, policy=self.policy)
        self.assertEqual(result["status"], "error")
        self.assertTrue(any(
            e["path"] == "config.fem.traction_bcs[0].locator" and "matched 0 of" in e["message"]
            for e in result["errors"]))

    def test_zero_facet_error_takes_priority_over_estimated_cost(self):
        # A rejected geometry check must not report a normalized_config or
        # claim to have a usable estimated_cost-backed "ok" result.
        raw = _load("beam_2d")
        raw["fem"]["dirichlet_bcs"][0]["marker"] = {"op": "plane", "axis": "x", "value": 9999}
        result = validate_config_tool({"config": raw}, policy=self.policy)
        self.assertIsNone(result["normalized_config"])

    def test_overlapping_tractions_are_rejected(self):
        raw = _load("beam_2d")
        raw["fem"]["traction_bcs"].append(
            json.loads(json.dumps(raw["fem"]["traction_bcs"][0]))
        )
        result = validate_config_tool({"config": raw}, policy=self.policy)
        self.assertIn(
            "traction_regions_overlap", {error["code"] for error in result["errors"]}
        )

    def test_traction_on_fixed_support_is_rejected(self):
        raw = _load("beam_2d")
        raw["fem"]["traction_bcs"][0]["locator"] = {
            "op": "plane",
            "axis": "x",
            "value": 0,
        }
        result = validate_config_tool({"config": raw}, policy=self.policy)
        self.assertIn(
            "traction_on_fixed_support", {error["code"] for error in result["errors"]}
        )

    def test_spring_regions_must_match_distinct_unconstrained_nodes(self):
        missing = _load("mechanism_2d")
        missing["opt"]["out_spring"]["region"] = {
            "op": "plane",
            "axis": "x",
            "value": 999,
        }
        result = validate_config_tool({"config": missing}, policy=self.policy)
        self.assertIn(
            "spring_region_matches_no_nodes",
            {error["code"] for error in result["errors"]},
        )

        overlapping = _load("mechanism_2d")
        overlapping["opt"]["out_spring"]["region"] = json.loads(
            json.dumps(overlapping["opt"]["in_spring"]["region"])
        )
        result = validate_config_tool({"config": overlapping}, policy=self.policy)
        self.assertIn(
            "spring_regions_overlap", {error["code"] for error in result["errors"]}
        )

        constrained = _load("mechanism_2d")
        constrained["opt"]["in_spring"]["region"] = json.loads(
            json.dumps(constrained["fem"]["dirichlet_bcs"][0]["marker"])
        )
        result = validate_config_tool({"config": constrained}, policy=self.policy)
        self.assertIn(
            "spring_overlaps_fixed_support",
            {error["code"] for error in result["errors"]},
        )

    def test_passive_zones_must_match_disjoint_cells(self):
        missing = _load("beam_2d")
        missing["opt"]["solid_zone"] = {
            "op": "circle",
            "center": [999, 999],
            "radius": 1,
        }
        result = validate_config_tool({"config": missing}, policy=self.policy)
        self.assertIn(
            "passive_zone_matches_no_cells",
            {error["code"] for error in result["errors"]},
        )

        overlapping = _load("mechanism_2d")
        overlapping["opt"]["void_zone"] = json.loads(
            json.dumps(overlapping["opt"]["solid_zone"])
        )
        result = validate_config_tool({"config": overlapping}, policy=self.policy)
        self.assertIn(
            "passive_zones_overlap", {error["code"] for error in result["errors"]}
        )

    def test_void_zone_cannot_erase_load_neighborhood(self):
        raw = _load("beam_2d")
        raw["opt"]["void_zone"] = {
            "op": "circle",
            "center": [59.8, 10],
            "radius": 0.5,
        }
        result = validate_config_tool({"config": raw}, policy=self.policy)
        self.assertIn(
            "void_zone_erases_required_neighborhood",
            {error["code"] for error in result["errors"]},
        )

    def test_forced_solid_zone_must_fit_volume_budget(self):
        raw = _load("beam_2d")
        raw["opt"]["solid_zone"] = {"op": "all"}
        result = validate_config_tool({"config": raw}, policy=self.policy)
        self.assertIn(
            "solid_zone_exceeds_volume_budget",
            {error["code"] for error in result["errors"]},
        )


if __name__ == "__main__":
    unittest.main()
