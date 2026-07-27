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
        self.assertEqual(result["checked"], {"structural": True, "geometry": True})
        # A real, well-posed support (the full left edge) must not trip the
        # rigid-body warning.
        self.assertFalse(any("rigid-body" in w["message"] for w in result["warnings"]))

    def test_mechanism_2d_passes_full_validation(self):
        result = validate_config_tool({"config": _load("mechanism_2d")}, policy=self.policy)
        self.assertEqual(result["status"], "ok", result.get("errors"))
        self.assertEqual(result["checked"], {"structural": True, "geometry": True})

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


if __name__ == "__main__":
    unittest.main()
