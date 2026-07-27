"""Numerical equivalence gate for Package 3 resultant conversion."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from fenitop.tools.config_models import migrate_legacy_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def _base_config():
    with (REPO_ROOT / "tests/fixtures/smoke_beam_2d.json").open(
        encoding="utf-8"
    ) as handle:
        config = migrate_legacy_config(json.load(handle))
    config["units"] = {
        "kind": "explicit",
        "length_unit": "mm",
        "force_unit": "N",
        "stress_unit": "MPa",
    }
    config["opt"]["max_iter"] = 2
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
    ]
    return config


class ResultantNumericalTests(unittest.TestCase):
    def setUp(self):
        self.output_root = Path(tempfile.mkdtemp(prefix="resultant_gate_"))
        self.addCleanup(shutil.rmtree, self.output_root, ignore_errors=True)

    def _run(self, config, prefix):
        from fenitop.tools.contracts import TrustedRunPolicy
        from fenitop.tools.run_topopt import _run_topopt_in_process

        result = _run_topopt_in_process(
            {"config": config},
            policy=TrustedRunPolicy(
                output_root=self.output_root,
                output_prefix=prefix,
                render_snapshot=False,
            ),
        )
        self.assertEqual(result["status"], "ok", result.get("errors"))
        return result

    def test_resultant_matches_equivalent_uniform_traction_numerically(self):
        selector = {
            "kind": "rectangle_edge",
            "edge": "right",
            "interval": {"kind": "fraction", "start": 0.25, "end": 0.75},
        }
        traction = _base_config()
        traction["fem"]["boundary_conditions"].append({
            "bc_id": "L1",
            "kind": "uniform_traction",
            "selector": selector,
            "traction": [0, -1],
        })
        resultant = _base_config()
        resultant["fem"]["boundary_conditions"].append({
            "bc_id": "L1",
            "kind": "uniform_resultant",
            "selector": selector,
            "resultant": [0, -1],
        })

        traction_result = self._run(traction, "traction")
        resultant_result = self._run(resultant, "resultant")
        for metric in ("final_compliance", "final_volume", "final_objective"):
            self.assertTrue(math.isclose(
                traction_result["metrics"][metric],
                resultant_result["metrics"][metric],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ))


if __name__ == "__main__":
    unittest.main()
