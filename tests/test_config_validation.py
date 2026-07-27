import json
import unittest
from pathlib import Path

import numpy as np

from fenitop.config import normalize_boundary_conditions, validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigValidationTests(unittest.TestCase):
    def test_rejects_invalid_iteration_count(self):
        config = {"opt": {"max_iter": 0}, "fem": {}, "parameter_guidance": {}}
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_rejects_volume_fraction_outside_bounds(self):
        config = {"opt": {"vol_frac": 1.2}, "fem": {}, "parameter_guidance": {}}
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_normalizes_multiple_boundary_conditions(self):
        fem_config = {
            "dirichlet_bcs": [
                {"marker": lambda x: True, "value": [1.0, 0.0]},
                {"marker": lambda x: False, "value": [0.0, 1.0]},
            ],
            "traction_bcs": [
                {"value": [0.0, -1.0], "locator": lambda x: True},
            ],
        }
        dirichlet_bcs, traction_bcs = normalize_boundary_conditions(fem_config, dim=2)
        self.assertEqual(len(dirichlet_bcs), 2)
        self.assertEqual(traction_bcs[0][0], [0.0, -1.0])


# --- Agent-tool layer: pure Python, no dolfinx/MPI needed for anything below. ---


class RegionDSLTests(unittest.TestCase):
    """fenitop/regions.py -- the declarative boundary/load marker DSL."""

    def setUp(self):
        # dolfinx always carries 3 coordinate rows internally, even for 2D.
        self.x = np.array([
            [0.0, 0.0, 60.0, 60.0, 30.0],
            [0.0, 20.0, 0.0, 20.0, 10.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ])

    def test_plane(self):
        from fenitop.regions import compile_region
        region = compile_region({"op": "plane", "axis": "x", "value": 0.0})
        self.assertEqual(region(self.x).tolist(), [True, True, False, False, False])

    def test_range(self):
        from fenitop.regions import compile_region
        region = compile_region({"op": "range", "axis": "y", "min": 8, "max": 12})
        self.assertEqual(region(self.x).tolist(), [False, False, False, False, True])

    def test_circle(self):
        from fenitop.regions import compile_region
        region = compile_region({"op": "circle", "center": [30, 10], "radius": 1})
        self.assertEqual(region(self.x).tolist(), [False, False, False, False, True])

    def test_all_and_none(self):
        from fenitop.regions import compile_region
        self.assertEqual(compile_region({"op": "all"})(self.x).tolist(), [True] * 5)
        self.assertEqual(compile_region({"op": "none"})(self.x).tolist(), [False] * 5)

    def test_and_or_not_composition(self):
        from fenitop.regions import compile_region
        both = compile_region({"op": "and", "regions": [
            {"op": "plane", "axis": "x", "value": 60},
            {"op": "range", "axis": "y", "min": 8, "max": 12},
        ]})
        self.assertEqual(both(self.x).tolist(), [False, False, False, False, False])

        either = compile_region({"op": "or", "regions": [
            {"op": "plane", "axis": "x", "value": 0},
            {"op": "plane", "axis": "x", "value": 60},
        ]})
        self.assertEqual(either(self.x).tolist(), [True, True, True, True, False])

        inverted = compile_region({"op": "not", "region": {"op": "plane", "axis": "x", "value": 0}})
        self.assertEqual(inverted(self.x).tolist(), [False, False, True, True, True])

    def test_unknown_axis_raises_region_error(self):
        from fenitop.regions import RegionError, compile_region
        with self.assertRaises(RegionError):
            compile_region({"op": "plane", "axis": "q", "value": 0})

    def test_missing_required_field_raises_region_error(self):
        from fenitop.regions import RegionError, compile_region
        with self.assertRaises(RegionError):
            compile_region({"op": "plane", "value": 0})  # missing "axis"

    def test_is_region_spec(self):
        from fenitop.regions import is_region_spec
        self.assertTrue(is_region_spec({"op": "plane", "axis": "x", "value": 0}))
        self.assertFalse(is_region_spec({"marker": "lambda x: True", "value": [0, 0]}))
        self.assertFalse(is_region_spec("lambda x: True"))

    def test_materialize_compiles_nested_region_dicts(self):
        from fenitop.config import _materialize
        raw = {"dirichlet_bcs": [{"marker": {"op": "plane", "axis": "x", "value": 0},
                                   "value": [0.0, 0.0]}]}
        out = _materialize(raw)
        marker = out["dirichlet_bcs"][0]["marker"]
        self.assertTrue(callable(marker))
        self.assertEqual(marker(self.x).tolist(), [True, True, False, False, False])


class ConfigModelStructuralTests(unittest.TestCase):
    """fenitop/tools/config_models.py -- pydantic structural validation."""

    def _base_config(self):
        return {
            "mesh": {"bounds": [[0, 0], [10, 10]], "divisions": [20, 20]},
            "fem": {
                "young's modulus": 100, "poisson's ratio": 0.25,
                "dirichlet_bcs": [{"marker": {"op": "plane", "axis": "x", "value": 0}, "value": [0, 0]}],
                "traction_bcs": [{"value": [0, -1], "locator": {"op": "plane", "axis": "x", "value": 10}}],
            },
            "opt": {
                "max_iter": 100, "opt_tol": 1e-5, "vol_frac": 0.5, "penalty": 3.0, "epsilon": 1e-6,
                "filter_radius": 1.0, "beta_interval": 50, "beta_max": 128, "use_oc": True,
                "move": 0.02, "opt_compliance": True,
            },
        }

    def _shaped(self, raw):
        from fenitop.tools.validate_config import _shape_for_pydantic
        return _shape_for_pydantic(raw)

    def test_accepts_well_formed_config(self):
        from fenitop.tools.config_models import ConfigModel
        ConfigModel.model_validate(self._shaped(self._base_config()))  # must not raise

    def test_rejects_poisson_ratio_at_singular_endpoints(self):
        from fenitop.tools.config_models import ConfigModel
        from pydantic import ValidationError
        for bad_nu in (0.5, -1.0, 0.6):
            with self.subTest(nu=bad_nu):
                cfg = self._base_config()
                cfg["fem"]["poisson's ratio"] = bad_nu
                with self.assertRaises(ValidationError):
                    ConfigModel.model_validate(self._shaped(cfg))

    def test_rejects_nonpositive_youngs_modulus(self):
        from fenitop.tools.config_models import ConfigModel
        from pydantic import ValidationError
        cfg = self._base_config()
        cfg["fem"]["young's modulus"] = 0
        with self.assertRaises(ValidationError):
            ConfigModel.model_validate(self._shaped(cfg))

    def test_rejects_hexahedron_cell_type_in_2d_scope(self):
        from fenitop.tools.config_models import ConfigModel
        from pydantic import ValidationError
        cfg = self._base_config()
        cfg["mesh"]["cell_type"] = "hexahedron"
        with self.assertRaises(ValidationError):
            ConfigModel.model_validate(self._shaped(cfg))

    def test_rejects_missing_mesh_bounds(self):
        from fenitop.tools.config_models import ConfigModel
        from pydantic import ValidationError
        cfg = self._base_config()
        del cfg["mesh"]["bounds"]
        with self.assertRaises(ValidationError):
            ConfigModel.model_validate(self._shaped(cfg))

    def test_rejects_nonpositive_divisions(self):
        from fenitop.tools.config_models import ConfigModel
        from pydantic import ValidationError
        cfg = self._base_config()
        cfg["mesh"]["divisions"] = [0, 10]
        with self.assertRaises(ValidationError):
            ConfigModel.model_validate(self._shaped(cfg))

    def test_mechanism_mode_requires_compliance_bound_and_springs(self):
        from fenitop.tools.config_models import ConfigModel, translate_validation_error
        from pydantic import ValidationError
        cfg = self._base_config()
        cfg["opt"]["opt_compliance"] = False
        with self.assertRaises(ValidationError) as ctx:
            ConfigModel.model_validate(self._shaped(cfg))
        errors = translate_validation_error(ctx.exception)
        message = " ".join(e.message for e in errors)
        self.assertIn("compliance_bound", message)
        self.assertIn("in_spring", message)
        self.assertIn("out_spring", message)

    def test_rejects_z_direction_spring_in_2d_scope(self):
        from fenitop.tools.config_models import ConfigModel
        from pydantic import ValidationError
        cfg = self._base_config()
        cfg["opt"]["opt_compliance"] = False
        cfg["opt"]["compliance_bound"] = 0.5
        cfg["opt"]["in_spring"] = [{"op": "plane", "axis": "x", "value": 0}, "z", 0.2]
        cfg["opt"]["out_spring"] = [{"op": "plane", "axis": "x", "value": 10}, "x", 0.2]
        with self.assertRaises(ValidationError):
            ConfigModel.model_validate(self._shaped(cfg))

    def test_rejects_unknown_field_typo(self):
        from fenitop.tools.config_models import ConfigModel
        from pydantic import ValidationError
        cfg = self._base_config()
        cfg["fem"]["yungs modulus"] = 100
        with self.assertRaises(ValidationError):
            ConfigModel.model_validate(self._shaped(cfg))

    def test_zero_facet_shaped_marker_is_structurally_fine(self):
        # Geometry validity (does this marker match any facet) is Tool 1's
        # separate, dolfinx-dependent check -- structural validation must not
        # reject a geometrically-plausible-looking marker on its own.
        from fenitop.tools.config_models import ConfigModel
        cfg = self._base_config()
        cfg["fem"]["dirichlet_bcs"][0]["marker"] = {"op": "plane", "axis": "x", "value": 9999}
        ConfigModel.model_validate(self._shaped(cfg))  # must not raise

    def test_json_schema_uses_apostrophe_aliases(self):
        from fenitop.tools.config_models import ConfigModel
        schema = ConfigModel.model_json_schema()
        fem_props = schema["$defs"]["FemModel"]["properties"]
        self.assertIn("young's modulus", fem_props)
        self.assertIn("poisson's ratio", fem_props)


class SafetyEstimateTests(unittest.TestCase):
    """fenitop/tools/safety.py -- cost estimation, calibrated against the
    two real example configs so a change here can't silently start
    rejecting them."""

    def _load(self, name):
        with open(REPO_ROOT / "config" / f"{name}.json", "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_beam_2d_is_low_risk_and_not_rejected(self):
        from fenitop.tools.safety import estimate_cost
        raw = self._load("beam_2d")
        cost = estimate_cost(raw["mesh"], raw["opt"]["max_iter"])
        self.assertEqual(cost["num_elements"], 12000)
        self.assertEqual(cost["complexity_score"], 4_800_000)
        self.assertEqual(cost["risk_level"], "low")
        self.assertFalse(cost["exceeds_default_safety_ceiling"])

    def test_mechanism_2d_is_medium_risk_and_not_rejected(self):
        from fenitop.tools.safety import estimate_cost
        raw = self._load("mechanism_2d")
        cost = estimate_cost(raw["mesh"], raw["opt"]["max_iter"])
        self.assertEqual(cost["num_elements"], 40000)
        self.assertEqual(cost["complexity_score"], 20_000_000)
        self.assertEqual(cost["risk_level"], "medium")
        self.assertFalse(cost["exceeds_default_safety_ceiling"])

    def test_huge_problem_exceeds_ceiling(self):
        from fenitop.tools.safety import estimate_cost
        cost = estimate_cost({"divisions": [2000, 2000], "cell_type": "quadrilateral"}, 50000)
        self.assertTrue(cost["exceeds_default_safety_ceiling"])
        self.assertEqual(cost["risk_level"], "high")


class ValidateConfigToolTests(unittest.TestCase):
    """fenitop/tools/validate_config.py -- Tool 1's structural half
    (check_geometry=False; the geometry half needs dolfinx, see
    tests/test_config_validation_geometry.py, docker-only)."""

    def _load(self, name):
        with open(REPO_ROOT / "config" / f"{name}.json", "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_beam_2d_passes(self):
        from fenitop.tools.validate_config import validate_config_tool
        result = validate_config_tool({"config": self._load("beam_2d"), "check_geometry": False})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["problem_type"], "minimize_compliance")
        self.assertEqual(result["errors"], [])

    def test_mechanism_2d_passes_with_medium_risk_warning(self):
        from fenitop.tools.validate_config import validate_config_tool
        result = validate_config_tool({"config": self._load("mechanism_2d"), "check_geometry": False})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["problem_type"], "compliant_mechanism")
        self.assertTrue(any("risk band" in w for w in result["warnings"]))

    def test_bad_config_reports_field_path(self):
        from fenitop.tools.validate_config import validate_config_tool
        raw = self._load("beam_2d")
        raw["fem"]["poisson's ratio"] = 0.5
        result = validate_config_tool({"config": raw, "check_geometry": False})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["normalized_config"], None)
        self.assertTrue(any("poisson" in e["path"] for e in result["errors"]))

    def test_normalized_config_round_trips_through_validate_config_again(self):
        # Regression test: run_topopt_tool always re-validates by feeding
        # Tool 1's own normalized_config back into validate_config_tool.
        # normalized_config used to include explicit "parameter_guidance":
        # null for any config that never set it (a plain model_dump() emits
        # every field, including unset Optional ones) -- and on that SECOND
        # pass, fenitop.config.validate_config()'s `config.get(
        # "parameter_guidance", {})` no longer falls back to {}, since the
        # key now genuinely exists (holding None), crashing with
        # "argument of type 'NoneType' is not iterable". normalized_config
        # must be safe to feed back into this same function indefinitely.
        from fenitop.tools.validate_config import validate_config_tool
        raw = self._load("beam_2d")
        raw.pop("parameter_guidance", None)  # exercise the "source never set it" path
        first = validate_config_tool({"config": raw, "check_geometry": False})
        self.assertEqual(first["status"], "ok", first.get("errors"))
        self.assertNotIn("parameter_guidance", first["normalized_config"])

        second = validate_config_tool({"config": first["normalized_config"], "check_geometry": False})
        self.assertEqual(second["status"], "ok", second.get("errors"))
        self.assertEqual(second["normalized_config"], first["normalized_config"])

    def test_bare_config_dict_accepted_as_whole_request(self):
        # The "bare config, no request wrapper" convenience is for exactly
        # that -- just a config, defaulting check_geometry=True. It is not
        # meant to be combined with sibling request-level options spread
        # into the same flat dict: with the strict extra="forbid" schema, a
        # "check_geometry" key smuggled in that way is correctly rejected as
        # an unrecognized config field, since there's no way to distinguish
        # it from a genuine (if oddly named) top-level config key.
        from fenitop.tools.validate_config import validate_config_tool
        result = validate_config_tool(self._load("beam_2d"))
        self.assertEqual(result["status"], "ok", result.get("errors"))

    def test_check_geometry_as_sibling_key_requires_the_config_wrapper(self):
        from fenitop.tools.validate_config import validate_config_tool
        bare = validate_config_tool({**self._load("beam_2d"), "check_geometry": False})
        self.assertEqual(bare["status"], "error")
        self.assertTrue(any(e["path"] == "check_geometry" for e in bare["errors"]))

        wrapped = validate_config_tool({"config": self._load("beam_2d"), "check_geometry": False})
        self.assertEqual(wrapped["status"], "ok", wrapped.get("errors"))
        self.assertEqual(wrapped["checked"], {"structural": True, "geometry": False})

    def test_missing_traction_locator_is_rejected_not_silently_dropped(self):
        # normalize_boundary_conditions() would inject a `lambda x: False`
        # fallback here; _shape_for_pydantic must neutralize that into an
        # explicit rejection rather than let a JSON-unsafe callable through.
        from fenitop.tools.validate_config import validate_config_tool
        raw = self._load("beam_2d")
        raw["fem"]["traction_bcs"] = [{"value": [0, -0.2]}]  # no locator, no marker
        result = validate_config_tool({"config": raw, "check_geometry": False})
        self.assertEqual(result["status"], "error")
        self.assertTrue(any("traction_bcs" in e["path"] for e in result["errors"]))


class SchemaHelperTests(unittest.TestCase):
    def test_get_or_falls_back_on_explicit_none(self):
        from fenitop.tools.schema import get_or
        self.assertEqual(get_or({"a": None}, "a", "default"), "default")
        self.assertEqual(get_or({}, "a", "default"), "default")
        self.assertEqual(get_or({"a": 5}, "a", "default"), 5)
        self.assertIs(get_or({"a": False}, "a", True), False)
        self.assertEqual(get_or({"a": 0}, "a", 99), 0)


class RigidBodyRankTests(unittest.TestCase):
    """fenitop/tools/validate_config.py::_rigid_body_rank -- pure numpy, no
    dolfinx. Kept directly unit-testable with synthetic (x, y, component)
    data specifically so this math can be verified without depending on
    dolfinx's locate_entities_boundary facet-matching quantifier semantics
    (see tests/test_config_validation_geometry.py's module docstring)."""

    def test_no_constraints_is_rank_zero(self):
        from fenitop.tools.validate_config import _rigid_body_rank
        self.assertEqual(_rigid_body_rank([]), 0)

    def test_single_pinned_point_leaves_rotation_free(self):
        # Both x,y translation are resisted at one point, but nothing resists
        # rotation about that point -- rank must be 2, not 3.
        from fenitop.tools.validate_config import _rigid_body_rank
        self.assertEqual(_rigid_body_rank([(0, 0, 0), (0, 0, 1)]), 2)

    def test_two_distinct_points_fully_constrain_rigid_body(self):
        # Classical 2D statics: pinning any two distinct points (each fixed
        # in both x and y) fully resists translation AND rotation, however
        # close together the two points are.
        from fenitop.tools.validate_config import _rigid_body_rank
        close = [(0, 0, 0), (0, 0, 1), (0, 0.001, 0), (0, 0.001, 1)]
        self.assertEqual(_rigid_body_rank(close), 3)
        far = [(0, 0, 0), (0, 0, 1), (0, 20, 0), (0, 20, 1)]
        self.assertEqual(_rigid_body_rank(far), 3)

    def test_full_edge_of_real_beam_2d_support_is_rank_three(self):
        # Mirrors the real beam_2d.json support: the whole left edge (x=0,
        # y in [0,20]) constrained -- comfortably rank 3.
        from fenitop.tools.validate_config import _rigid_body_rank
        rows = []
        for y in [0.0, 5.0, 10.0, 15.0, 20.0]:
            rows.append((0.0, y, 0))
            rows.append((0.0, y, 1))
        self.assertEqual(_rigid_body_rank(rows), 3)


class NarrativeTests(unittest.TestCase):
    """fenitop/tools/narrative.py -- deterministic, keyword-testable text."""

    def test_non_convergent_run_narrative(self):
        from fenitop.tools.narrative import build_narrative
        convergence = {"converged": False, "iterations": 400, "final_change": 0.02, "opt_tol": 1e-5,
                       "fraction_iterations_at_move_limit": 0.998, "move_limit": 0.02}
        quality = {"grayness": 0.4775, "low_grayness_warning": False}
        metrics = {"final_compliance": 0.7151, "final_volume": 0.4989, "vol_frac_target": 0.5}
        text = build_narrative(convergence, quality, metrics, "minimize_compliance")
        self.assertIn("did NOT converge", text)
        self.assertIn("move limit", text)
        self.assertIn("compliance=0.7151", text)

    def test_convergent_run_narrative_has_no_false_alarm(self):
        from fenitop.tools.narrative import build_narrative
        convergence = {"converged": True, "iterations": 210, "final_change": 9e-6, "opt_tol": 1e-5}
        quality = {"grayness": 0.49, "low_grayness_warning": False}
        text = build_narrative(convergence, quality, {}, "minimize_compliance")
        self.assertIn("converged:", text)
        self.assertNotIn("did NOT", text)


if __name__ == "__main__":
    unittest.main()
