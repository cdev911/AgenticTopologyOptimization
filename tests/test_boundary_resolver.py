"""Package 3 mesh-aware selector, migration, and evidence tests."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from fenitop.tools.config_models import (
    AgentSafeConfig,
    migrate_legacy_config,
    parse_agent_safe_config,
)
from fenitop.tools.validate_config import validate_config_tool


REPO_ROOT = Path(__file__).resolve().parents[1]


def _legacy_smoke():
    with (REPO_ROOT / "tests/fixtures/smoke_beam_2d.json").open(
        encoding="utf-8"
    ) as handle:
        return json.load(handle)


def _canonical():
    config = migrate_legacy_config(_legacy_smoke())
    config["units"] = {
        "kind": "explicit",
        "length_unit": "mm",
        "force_unit": "N",
        "stress_unit": "MPa",
    }
    config["fem"]["boundary_conditions"] = [
        {
            "bc_id": "S1",
            "kind": "fixed",
            "selector": {
                "kind": "rectangle_edge",
                "edge": "left",
                "interval": {"kind": "fraction", "start": 0, "end": 1},
            },
        },
        {
            "bc_id": "L1",
            "kind": "uniform_traction",
            "selector": {
                "kind": "rectangle_edge",
                "edge": "right",
                "interval": {
                    "kind": "fraction",
                    "start": 0.25,
                    "end": 0.75,
                },
            },
            "traction": [0, -1],
        },
    ]
    return config


class ConfigMigrationTests(unittest.TestCase):
    def test_legacy_migration_is_deterministic_and_identity_preserving(self):
        raw = _legacy_smoke()
        first, migrated = parse_agent_safe_config(raw)
        second, second_migrated = parse_agent_safe_config(copy.deepcopy(raw))
        self.assertTrue(migrated)
        self.assertTrue(second_migrated)
        self.assertEqual(first, second)
        self.assertEqual(first.schema_version, "2.1")
        self.assertEqual(first.units.kind, "legacy_consistent")
        self.assertEqual(
            [bc.bc_id for bc in first.fem.boundary_conditions],
            ["S1", "L1"],
        )
        self.assertEqual(
            [bc.kind for bc in first.fem.boundary_conditions],
            ["fixed", "uniform_traction"],
        )

    def test_validation_returns_only_canonical_normalized_config(self):
        result = validate_config_tool({"config": _legacy_smoke()})
        self.assertEqual(result["status"], "ok", result["errors"])
        self.assertEqual(result["normalized_config"]["schema_version"], "2.1")
        self.assertIn(
            "boundary_conditions", result["normalized_config"]["fem"]
        )
        self.assertNotIn(
            "dirichlet_bcs", result["normalized_config"]["fem"]
        )
        self.assertTrue(
            any("migrated" in warning["message"] for warning in result["warnings"])
        )

    def test_canonical_20_migrates_to_21_without_semantic_change(self):
        previous = _canonical()
        previous["schema_version"] = "2.0"

        current, migrated = parse_agent_safe_config(previous)

        self.assertTrue(migrated)
        self.assertEqual(current.schema_version, "2.1")
        self.assertEqual(
            [condition.kind for condition in current.fem.boundary_conditions],
            ["fixed", "uniform_traction"],
        )

    def test_canonical_contract_enforces_ids_intervals_and_resultant_units(self):
        mutations = []

        duplicate = _canonical()
        duplicate["fem"]["boundary_conditions"][1]["bc_id"] = "S1"
        mutations.append(duplicate)

        wrong_prefix = _canonical()
        wrong_prefix["fem"]["boundary_conditions"][1]["bc_id"] = "S2"
        mutations.append(wrong_prefix)

        reversed_interval = _canonical()
        reversed_interval["fem"]["boundary_conditions"][1]["selector"][
            "interval"
        ] = {"kind": "fraction", "start": 0.8, "end": 0.2}
        mutations.append(reversed_interval)

        out_of_fraction = _canonical()
        out_of_fraction["fem"]["boundary_conditions"][1]["selector"][
            "interval"
        ] = {"kind": "fraction", "start": 0.2, "end": 1.2}
        mutations.append(out_of_fraction)

        legacy_resultant = _canonical()
        legacy_resultant["units"] = {"kind": "legacy_consistent"}
        legacy_resultant["fem"]["boundary_conditions"][1] = {
            **legacy_resultant["fem"]["boundary_conditions"][1],
            "kind": "uniform_resultant",
            "resultant": [0, -10],
        }
        legacy_resultant["fem"]["boundary_conditions"][1].pop("traction")
        mutations.append(legacy_resultant)

        for config in mutations:
            with self.subTest(config=config), self.assertRaises(ValidationError):
                AgentSafeConfig.model_validate(config)

    def test_component_contract_requires_canonical_unique_zero_components(self):
        for components in ([], ["x", "x"], ["y", "x"]):
            config = _canonical()
            config["fem"]["boundary_conditions"][0] = {
                "bc_id": "S1",
                "kind": "zero_displacement",
                "selector": {
                    "kind": "boundary_node",
                    "point": [0, 0],
                },
                "components": components,
            }
            with self.subTest(components=components), self.assertRaises(
                ValidationError
            ):
                AgentSafeConfig.model_validate(config)


class BoundaryResolverTests(unittest.TestCase):
    def _mesh(self, *, cell_type="quadrilateral"):
        from mpi4py import MPI

        from fenitop.config import build_mesh

        return build_mesh(
            {
                "kind": "rectangle",
                "bounds": [[0, 0], [4, 2]],
                "divisions": [8, 4],
                "cell_type": cell_type,
            },
            comm=MPI.COMM_SELF,
        )

    def test_whole_edges_report_count_measure_centroid_and_normal(self):
        from fenitop.boundary_resolver import resolve_boundary

        expected = {
            "left": (4, 2.0, (0.0, 1.0), (-1.0, 0.0)),
            "right": (4, 2.0, (4.0, 1.0), (1.0, 0.0)),
            "bottom": (8, 4.0, (2.0, 0.0), (0.0, -1.0)),
            "top": (8, 4.0, (2.0, 2.0), (0.0, 1.0)),
        }
        for cell_type in ("quadrilateral", "triangle"):
            mesh = self._mesh(cell_type=cell_type)
            for edge, values in expected.items():
                with self.subTest(cell_type=cell_type, edge=edge):
                    result = resolve_boundary(mesh, {
                        "kind": "rectangle_edge",
                        "edge": edge,
                        "interval": {
                            "kind": "fraction",
                            "start": 0,
                            "end": 1,
                        },
                    })
                    self.assertEqual(len(result.facets), values[0])
                    self.assertAlmostEqual(result.measure, values[1])
                    self.assertEqual(result.centroid, values[2])
                    self.assertEqual(result.outward_normal, values[3])
                    self.assertEqual(result.resolution_error, 0.0)

    def test_fraction_and_coordinate_intervals_have_same_resolution(self):
        from fenitop.boundary_resolver import resolve_boundary

        mesh = self._mesh()
        common = {"kind": "rectangle_edge", "edge": "right"}
        fractional = resolve_boundary(mesh, {
            **common,
            "interval": {"kind": "fraction", "start": 0.25, "end": 0.75},
        })
        coordinate = resolve_boundary(mesh, {
            **common,
            "interval": {"kind": "coordinate", "start": 0.5, "end": 1.5},
        })
        self.assertEqual(fractional.facets.tolist(), coordinate.facets.tolist())
        self.assertEqual(fractional.requested_extent, (0.5, 1.5))
        self.assertEqual(fractional.resolved_extent, (0.5, 1.5))
        self.assertEqual(fractional.measure, 1.0)

    def test_positive_subfacet_interval_selects_closest_facet_with_warning(self):
        from fenitop.boundary_resolver import resolve_boundary

        result = resolve_boundary(self._mesh(), {
            "kind": "rectangle_edge",
            "edge": "right",
            "interval": {"kind": "coordinate", "start": 0.9, "end": 1.1},
        })
        self.assertEqual(len(result.facets), 1)
        self.assertEqual(result.measure, 0.5)
        self.assertIn("single closest facet", result.warning)
        self.assertGreater(result.resolution_error, 0)

    def test_boundary_node_is_exact_or_visibly_snapped(self):
        from fenitop.boundary_resolver import resolve_boundary

        mesh = self._mesh()
        exact = resolve_boundary(mesh, {
            "kind": "boundary_node",
            "point": [0.0, 2.0],
        })
        self.assertEqual(len(exact.node_indices), 1)
        self.assertEqual(exact.requested_point, (0.0, 2.0))
        self.assertEqual(exact.resolved_point, (0.0, 2.0))
        self.assertEqual(exact.resolution_error, 0.0)
        self.assertIsNone(exact.warning)

        snapped = resolve_boundary(mesh, {
            "kind": "boundary_node",
            "point": [0.0, 1.13],
        })
        self.assertEqual(snapped.requested_point, (0.0, 1.13))
        self.assertEqual(snapped.resolved_point, (0.0, 1.0))
        self.assertAlmostEqual(snapped.resolution_error, 0.13)
        self.assertIn("snapped", snapped.warning)

    def test_boundary_node_rejects_interior_point(self):
        from fenitop.boundary_resolver import (
            BoundaryResolutionError,
            resolve_boundary,
        )

        with self.assertRaises(BoundaryResolutionError) as captured:
            resolve_boundary(self._mesh(), {
                "kind": "boundary_node",
                "point": [2.0, 1.0],
            })
        self.assertEqual(captured.exception.code, "point_not_on_boundary")

    def test_coordinate_interval_outside_edge_is_rejected(self):
        from fenitop.boundary_resolver import (
            BoundaryResolutionError,
            resolve_boundary,
        )

        with self.assertRaises(BoundaryResolutionError) as captured:
            resolve_boundary(self._mesh(), {
                "kind": "rectangle_edge",
                "edge": "right",
                "interval": {
                    "kind": "coordinate",
                    "start": 2.1,
                    "end": 2.2,
                },
            })
        self.assertEqual(captured.exception.code, "selector_outside_edge")


class BoundaryEvidenceTests(unittest.TestCase):
    def test_resultant_conversion_and_round_trip_are_reported(self):
        config = _canonical()
        config["fem"]["boundary_conditions"][1] = {
            "bc_id": "L1",
            "kind": "uniform_resultant",
            "selector": {
                "kind": "rectangle_edge",
                "edge": "right",
                "interval": {
                    "kind": "coordinate",
                    "start": 0.9,
                    "end": 1.1,
                },
            },
            "resultant": [0, -100],
        }
        result = validate_config_tool({"config": config})
        self.assertEqual(result["status"], "ok", result["errors"])
        record = next(
            item
            for item in result["geometry_report"]["entities"]
            if item.get("bc_id") == "L1"
        )
        self.assertEqual(record["quantity_kind"], "resultant")
        self.assertEqual(record["measure"], 0.5)
        self.assertEqual(record["effective_traction"], [0.0, -200.0])
        self.assertEqual(record["integrated_resultant"], [0.0, -100.0])
        self.assertEqual(record["stress_unit"], "MPa")
        self.assertEqual(record["force_unit"], "N")
        self.assertEqual(record["thickness_value"], 1.0)
        self.assertEqual(record["thickness_unit"], "mm")

    def test_near_incompressible_plane_strain_is_warning_not_rejection(self):
        config = _canonical()
        config["fem"]["poisson_ratio"] = 0.499
        result = validate_config_tool({"config": config})
        self.assertEqual(result["status"], "ok", result["errors"])
        self.assertTrue(any(
            "volumetric locking" in warning["message"]
            for warning in result["warnings"]
        ))

    def test_rectangle_selector_overlap_uses_resolved_facets(self):
        config = _canonical()
        duplicate = copy.deepcopy(config["fem"]["boundary_conditions"][1])
        duplicate["bc_id"] = "L2"
        config["fem"]["boundary_conditions"].append(duplicate)
        result = validate_config_tool({"config": config})
        self.assertIn(
            "traction_regions_overlap",
            {error["code"] for error in result["errors"]},
        )


if __name__ == "__main__":
    unittest.main()
