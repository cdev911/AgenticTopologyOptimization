"""Real CLI and stdio MCP integration tests against the final tool contract."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def _request() -> dict:
    return {
        "config": json.loads((FIXTURES / "smoke_beam_2d.json").read_text())
    }


class CliTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="fenitop_cli_transport_"))
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.environment = dict(os.environ)
        prior = self.environment.get("PYTHONPATH")
        self.environment["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{prior}" if prior else str(REPO_ROOT)
        )

    def _run(self, module: str, *, stdin: str = "", args=()):
        return subprocess.run(
            [sys.executable, "-m", module, *args],
            input=stdin,
            text=True,
            capture_output=True,
            cwd=self.tmp_dir,
            env=self.environment,
            timeout=30,
            check=False,
        )

    def test_cli_stdin_file_output_malformed_json_and_exit_codes(self):
        request_text = json.dumps(_request())
        valid = self._run("fenitop.tools.validate_config", stdin=request_text)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertEqual(json.loads(valid.stdout)["status"], "ok")

        invalid = self._run("fenitop.tools.validate_config", stdin="{}")
        self.assertEqual(invalid.returncode, 1)
        self.assertEqual(json.loads(invalid.stdout)["status"], "error")

        malformed = self._run("fenitop.tools.validate_config", stdin="{bad")
        self.assertEqual(malformed.returncode, 2)
        malformed_response = json.loads(malformed.stdout)
        self.assertEqual(malformed_response["errors"][0]["code"], "malformed_json")

        input_path = self.tmp_dir / "request.json"
        output_path = self.tmp_dir / "response.json"
        input_path.write_text(request_text, encoding="utf-8")
        file_result = self._run(
            "fenitop.tools.validate_config",
            args=("--input", str(input_path), "--output", str(output_path)),
        )
        self.assertEqual(file_result.returncode, 0, file_result.stderr)
        self.assertEqual(file_result.stdout, "")
        self.assertEqual(json.loads(output_path.read_text())["status"], "ok")

    def test_cli_solver_progress_and_manifest_analysis_do_not_pollute_stdout(self):
        run = self._run(
            "fenitop.tools.run_topopt",
            stdin=json.dumps(_request()),
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        run_response = json.loads(run.stdout)
        self.assertEqual(run_response["status"], "ok", run_response["errors"])
        self.assertIsNotNone(run_response["run_manifest"])

        analysis = self._run(
            "fenitop.tools.analyze_results",
            stdin=json.dumps({"run_manifest": run_response["run_manifest"]}),
        )
        self.assertEqual(analysis.returncode, 0, analysis.stderr)
        analysis_response = json.loads(analysis.stdout)
        self.assertEqual(
            analysis_response["source"]["manifest_hash"],
            run_response["run_manifest"]["manifest_hash"],
        )


class McpStdioTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_stdio_validate_run_analyze_composition(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="fenitop_mcp_transport_"))
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        environment = dict(os.environ)
        prior = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{prior}" if prior else str(REPO_ROOT)
        )
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "fenitop.tools.mcp_server"],
            env=environment,
            cwd=tmp_dir,
        )
        error_log = tmp_dir / "mcp.stderr.log"
        with error_log.open("w", encoding="utf-8") as stderr:
            async with stdio_client(parameters, errlog=stderr) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    validation = await session.call_tool("validate_config", _request())
                    self.assertFalse(validation.isError)
                    self.assertEqual(
                        validation.structuredContent["status"], "ok"
                    )

                    run = await session.call_tool("run_topopt", _request())
                    self.assertFalse(run.isError)
                    run_response = run.structuredContent
                    self.assertEqual(run_response["status"], "ok")
                    manifest = run_response["run_manifest"]

                    analysis = await session.call_tool(
                        "analyze_results", {"run_manifest": manifest}
                    )
                    self.assertFalse(analysis.isError)
                    self.assertEqual(
                        analysis.structuredContent["source"]["manifest_hash"],
                        manifest["manifest_hash"],
                    )


if __name__ == "__main__":
    unittest.main()
