import copy
import asyncio
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from fenitop.config import normalize_boundary_conditions, validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name="beam_2d"):
    with open(REPO_ROOT / "config" / f"{name}.json", encoding="utf-8") as handle:
        return json.load(handle)


class LegacyInternalConfigTests(unittest.TestCase):
    """Internal Python APIs remain usable by hardcoded, trusted examples."""

    def test_rejects_invalid_iteration_count(self):
        with self.assertRaises(ValueError):
            validate_config({"opt": {"max_iter": 0}, "fem": {}})

    def test_rejects_volume_fraction_outside_bounds(self):
        with self.assertRaises(ValueError):
            validate_config({"opt": {"vol_frac": 1.2}, "fem": {}})

    def test_normalizes_callable_boundary_conditions(self):
        fem = {
            "dirichlet_bcs": [{"marker": lambda x: True, "value": [0, 0]}],
            "traction_bcs": [{"value": [0, -1], "locator": lambda x: True}],
        }
        supports, tractions = normalize_boundary_conditions(fem)
        self.assertEqual(len(supports), 1)
        self.assertEqual(tractions[0][0], [0, -1])


class RegionDSLTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array(
            [
                [0.0, 0.0, 60.0, 60.0, 30.0],
                [0.0, 20.0, 0.0, 20.0, 10.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

    def test_all_region_operations(self):
        from fenitop.regions import compile_region

        self.assertEqual(
            compile_region({"op": "plane", "axis": "x", "value": 0})(self.x).tolist(),
            [True, True, False, False, False],
        )
        self.assertEqual(
            compile_region(
                {"op": "range", "axis": "y", "min": 8, "max": 12}
            )(self.x).tolist(),
            [False, False, False, False, True],
        )
        self.assertEqual(
            compile_region(
                {"op": "circle", "center": [30, 10], "radius": 1}
            )(self.x).tolist(),
            [False, False, False, False, True],
        )
        self.assertTrue(compile_region({"op": "all"})(self.x).all())
        self.assertFalse(compile_region({"op": "none"})(self.x).any())

        either = compile_region(
            {
                "op": "or",
                "regions": [
                    {"op": "plane", "axis": "x", "value": 0},
                    {"op": "plane", "axis": "x", "value": 60},
                ],
            }
        )
        self.assertEqual(either(self.x).tolist(), [True, True, True, True, False])
        inverted = compile_region(
            {
                "op": "not",
                "region": {"op": "plane", "axis": "x", "value": 0},
            }
        )
        self.assertEqual(inverted(self.x).tolist(), [False, False, True, True, True])

    def test_plane_uses_absolute_tolerance_only(self):
        from fenitop.regions import compile_region

        points = np.array([[1e9 + 1, 1e9], [0, 0], [0, 0]], dtype=float)
        matched = compile_region(
            {"op": "plane", "axis": "x", "value": 1e9, "tol": 1e-8}
        )(points)
        self.assertEqual(matched.tolist(), [False, True])

    def test_rejects_unknown_extra_z_nonfinite_and_nonpositive_geometry(self):
        from fenitop.regions import RegionError, parse_region

        invalid = [
            {"op": "plane", "axis": "z", "value": 0},
            {"op": "plane", "axis": "x", "value": 0, "typo": 1},
            {"op": "plane", "axis": "x", "value": float("nan")},
            {"op": "plane", "axis": "x", "value": 0, "tol": 0},
            {"op": "circle", "center": [0, 0], "radius": 0},
            {"op": "range", "axis": "x", "min": 2, "max": 1},
            {"op": "and", "regions": []},
        ]
        for region in invalid:
            with self.subTest(region=region), self.assertRaises(RegionError):
                parse_region(region)

    def test_rejects_excessive_recursion_and_node_count(self):
        from fenitop.regions import RegionError, parse_region

        nested = {"op": "plane", "axis": "x", "value": 0}
        for _ in range(9):
            nested = {"op": "not", "region": nested}
        with self.assertRaises(RegionError):
            parse_region(nested)

        many = {
            "op": "and",
            "regions": [
                {
                    "op": "or",
                    "regions": [
                        {"op": "plane", "axis": "x", "value": index}
                        for index in range(16)
                    ],
                }
                for _ in range(5)
            ],
        }
        with self.assertRaises(RegionError):
            parse_region(many)

    def test_materialize_never_evaluates_source_strings(self):
        from fenitop.config import _materialize

        source = "lambda x: __import__('os').system('false')"
        self.assertEqual(_materialize(source), source)
        self.assertFalse(callable(_materialize(source)))


class AgentSafeConfigTests(unittest.TestCase):
    def test_reference_configs_validate(self):
        from fenitop.tools.config_models import AgentSafeConfig

        self.assertEqual(
            AgentSafeConfig.model_validate(_load()).opt.problem_type,
            "minimize_compliance",
        )
        self.assertEqual(
            AgentSafeConfig.model_validate(_load("mechanism_2d")).opt.problem_type,
            "compliant_mechanism",
        )

    def test_rejects_lambda_markers_and_execution_capabilities(self):
        from fenitop.tools.config_models import AgentSafeConfig

        for path, value in [
            (("fem", "dirichlet_bcs", 0, "marker"), "lambda x: True"),
            (("mesh", "ghost_mode"), "shared_facet"),
            (("fem", "petsc_options"), {"ksp_type": "preonly"}),
            (("opt", "output_folder"), "/tmp/escape"),
            (("opt", "output_prefix"), "../escape"),
        ]:
            config = _load()
            target = config
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValidationError):
                AgentSafeConfig.model_validate(config)

    def test_rejects_nonzero_supports_bad_vectors_and_nonfinite_values(self):
        from fenitop.tools.config_models import AgentSafeConfig

        mutations = [
            ("support", lambda c: c["fem"]["dirichlet_bcs"][0].update(value=[0, 1])),
            ("traction", lambda c: c["fem"]["traction_bcs"][0].update(value=[1])),
            ("body", lambda c: c["fem"].update(body_force=[0, 0, 0])),
            ("nan", lambda c: c["fem"].update(young_modulus=float("inf"))),
        ]
        for name, mutate in mutations:
            config = _load()
            mutate(config)
            with self.subTest(name=name), self.assertRaises(ValidationError):
                AgentSafeConfig.model_validate(config)

    def test_mechanism_springs_are_named_positive_and_conditional(self):
        from fenitop.tools.config_models import AgentSafeConfig

        positional = _load("mechanism_2d")
        positional["opt"]["in_spring"] = [
            {"op": "plane", "axis": "x", "value": 0},
            "x",
            0.2,
        ]
        with self.assertRaises(ValidationError):
            AgentSafeConfig.model_validate(positional)

        nonpositive = _load("mechanism_2d")
        nonpositive["opt"]["out_spring"]["stiffness"] = 0
        with self.assertRaises(ValidationError):
            AgentSafeConfig.model_validate(nonpositive)

        missing = _load("mechanism_2d")
        del missing["opt"]["in_spring"]
        with self.assertRaises(ValidationError):
            AgentSafeConfig.model_validate(missing)

    def test_compiler_adds_only_trusted_solver_settings(self):
        from fenitop.tools.config_models import compile_solver_config

        compiled = compile_solver_config(
            _load(), output_folder="/trusted/run", output_prefix="trusted"
        )
        self.assertEqual(compiled["fem"]["petsc_options"]["ksp_type"], "cg")
        self.assertEqual(compiled["mesh"]["ghost_mode"], "shared_facet")
        self.assertEqual(compiled["opt"]["output_folder"], "/trusted/run")
        self.assertNotIn("schema_version", compiled)


class ToolContractTests(unittest.TestCase):
    def test_agent_request_schemas_exclude_execution_authority(self):
        from fenitop.tools.contracts import agent_tool_schemas

        schemas = agent_tool_schemas()
        config_schema_text = json.dumps(
            schemas["run_topopt"]["$defs"]["AgentSafeConfig"], sort_keys=True
        )
        for forbidden in (
            "lambda",
            "ghost_mode",
            "petsc",
            "output_folder",
            "output_prefix",
            "output_root",
            "run_id",
            "render_snapshot",
            "allow_large_run",
            "max_complexity",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, config_schema_text)
        self.assertFalse(schemas["run_topopt"]["additionalProperties"])
        self.assertEqual(set(schemas["run_topopt"]["properties"]), {"config"})

    def test_agent_safe_config_defs_forbid_unknown_fields(self):
        from fenitop.tools.config_models import AgentSafeConfig

        schema = AgentSafeConfig.model_json_schema()
        self.assertFalse(schema["additionalProperties"])
        for definition in schema["$defs"].values():
            if definition.get("type") == "object":
                self.assertIs(
                    definition.get("additionalProperties"),
                    False,
                    definition.get("title"),
                )

    def test_public_requests_reject_trusted_policy_fields(self):
        from fenitop.tools.contracts import RunTopoptRequest

        for key, value in {
            "output_root": "/tmp",
            "run_id": "chosen",
            "render_snapshot": False,
            "allow_large_run": True,
            "max_complexity": 1e30,
        }.items():
            with self.subTest(key=key), self.assertRaises(ValidationError):
                RunTopoptRequest.model_validate({"config": _load(), key: value})

    def test_tool_errors_and_warnings_are_structured(self):
        from fenitop.tools.contracts import (
            TrustedValidationPolicy,
            ValidateConfigResponse,
        )
        from fenitop.tools.validate_config import validate_config_tool

        invalid = _load()
        invalid["fem"]["poisson_ratio"] = 0.5
        result = validate_config_tool(
            {"config": invalid},
            policy=TrustedValidationPolicy(check_geometry=False),
        )
        ValidateConfigResponse.model_validate(result)
        self.assertEqual(result["status"], "error")
        issue = result["errors"][0]
        self.assertEqual(
            set(issue),
            {"code", "path", "message", "severity", "retryable"},
        )

    def test_actual_mcp_input_schemas_match_reviewed_snapshot(self):
        from fenitop.tools.mcp_server import mcp

        def assert_all_objects_are_closed(node, path="$"):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertIs(
                        node.get("additionalProperties"),
                        False,
                        f"open object schema at {path}",
                    )
                for key, value in node.items():
                    assert_all_objects_are_closed(value, f"{path}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    assert_all_objects_are_closed(value, f"{path}[{index}]")

        with open(
            REPO_ROOT / "tests" / "fixtures" / "mcp_schema_hashes.json",
            encoding="utf-8",
        ) as handle:
            snapshot = json.load(handle)

        tools = asyncio.run(mcp.list_tools())
        observed_inputs = {}
        observed_outputs = {}
        for tool in tools:
            canonical_input = json.dumps(
                tool.inputSchema, sort_keys=True, separators=(",", ":")
            ).encode()
            canonical_output = json.dumps(
                tool.outputSchema, sort_keys=True, separators=(",", ":")
            ).encode()
            observed_inputs[tool.name] = hashlib.sha256(canonical_input).hexdigest()
            observed_outputs[tool.name] = hashlib.sha256(canonical_output).hexdigest()
            assert_all_objects_are_closed(tool.inputSchema)
            assert_all_objects_are_closed(tool.outputSchema)
        self.assertEqual(observed_inputs, snapshot["input_sha256"])
        self.assertEqual(observed_outputs, snapshot["output_sha256"])
        run_tool = mcp._tool_manager._tools["run_topopt"]
        with self.assertRaises(ValidationError):
            run_tool.fn_metadata.arg_model.model_validate(
                {"config": _load(), "output_root": "/tmp/not-allowed"}
            )


class SafetyEstimateTests(unittest.TestCase):
    def test_reference_costs_remain_calibrated(self):
        from fenitop.tools.safety import estimate_cost

        beam = _load()
        mechanism = _load("mechanism_2d")
        self.assertEqual(
            estimate_cost(beam["mesh"], beam["opt"]["max_iter"])["complexity_score"],
            4_800_000,
        )
        self.assertEqual(
            estimate_cost(
                mechanism["mesh"], mechanism["opt"]["max_iter"]
            )["complexity_score"],
            20_000_000,
        )


class ValidateConfigToolTests(unittest.TestCase):
    def test_structural_validation_and_round_trip(self):
        from fenitop.tools.contracts import TrustedValidationPolicy
        from fenitop.tools.validate_config import validate_config_tool

        policy = TrustedValidationPolicy(check_geometry=False)
        first = validate_config_tool({"config": _load()}, policy=policy)
        self.assertEqual(first["status"], "ok", first["errors"])
        second = validate_config_tool(
            {"config": first["normalized_config"]}, policy=policy
        )
        self.assertEqual(second["status"], "ok", second["errors"])
        self.assertEqual(second["normalized_config"], first["normalized_config"])

    def test_requires_the_typed_request_wrapper(self):
        from fenitop.tools.contracts import TrustedValidationPolicy
        from fenitop.tools.validate_config import validate_config_tool

        result = validate_config_tool(
            _load(), policy=TrustedValidationPolicy(check_geometry=False)
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "structural_validation")

    def test_medium_risk_is_a_structured_warning(self):
        from fenitop.tools.contracts import TrustedValidationPolicy
        from fenitop.tools.validate_config import validate_config_tool

        result = validate_config_tool(
            {"config": _load("mechanism_2d")},
            policy=TrustedValidationPolicy(check_geometry=False),
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(
            any("risk band" in warning["message"] for warning in result["warnings"])
        )


class RigidBodyRankTests(unittest.TestCase):
    def test_rank_characterization(self):
        from fenitop.tools.validate_config import _rigid_body_rank

        self.assertEqual(_rigid_body_rank([]), 0)
        self.assertEqual(_rigid_body_rank([(0, 0, 0), (0, 0, 1)]), 2)
        rows = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]
        self.assertEqual(_rigid_body_rank(rows), 3)


class NarrativeTests(unittest.TestCase):
    def test_non_convergent_and_convergent_narratives(self):
        from fenitop.tools.narrative import build_narrative

        failed = build_narrative(
            {
                "converged": False,
                "iterations": 400,
                "final_change": 0.02,
                "opt_tol": 1e-5,
                "fraction_iterations_at_move_limit": 0.998,
                "move_limit": 0.02,
            },
            {"grayness": 0.47, "low_grayness_warning": False},
            {"final_compliance": 0.7},
            "minimize_compliance",
        )
        self.assertIn("did NOT converge", failed)
        passed = build_narrative(
            {"converged": True, "iterations": 3, "final_change": 1e-6},
            {"grayness": 0.49, "low_grayness_warning": False},
            {},
            "minimize_compliance",
        )
        self.assertNotIn("did NOT", passed)


if __name__ == "__main__":
    unittest.main()
