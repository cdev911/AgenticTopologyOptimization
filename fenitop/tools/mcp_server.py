"""MCP server exposing validate_config, run_topopt, and analyze_results as
MCP tools -- thin wrappers over the exact same functions the CLI and direct
Python callers use (fenitop.tools.validate_config.validate_config_tool,
run_topopt.run_topopt_tool, analyze_results.analyze_results_tool), so there
is one implementation behind all three call surfaces.

Run via:
    docker compose run --rm -T fenitop python -m fenitop.tools.mcp_server

The -T is required, not optional: docker-compose.yml's default
tty: true / stdin_open: true is right for an interactive `docker compose run
--rm fenitop bash` session, but stdio-transport MCP needs clean, unbuffered
JSON-RPC framing on stdin/stdout, and a pty's line-buffering/escape sequences
would corrupt that stream. -T disables the pty for this one invocation.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from fenitop.tools.config_models import AgentSafeConfig
from fenitop.tools.contracts import (
    AnalyzeResultsRequest,
    AnalyzeResultsResponse,
    RunTopoptRequest,
    RunTopoptResponse,
    ValidateConfigRequest,
    ValidateConfigResponse,
)

mcp = FastMCP("fenitop")


@mcp.tool()
def validate_config(config: AgentSafeConfig) -> ValidateConfigResponse:
    """Pedantically validate a fenitop topology-optimization JSON config before running it.

    Runs structural checks (mesh/material/optimizer parameters, physical
    ranges such as -1 < Poisson's ratio < 0.5, conditional-required fields
    for compliant-mechanism mode) and builds a
    real mesh to confirm every boundary/load marker matches at least one
    facet (a marker matching zero facets would otherwise be silently
    dropped) plus a rigid-body-motion sanity check. Returns errors with
    dotted field paths (e.g. "fem.dirichlet_bcs[0].marker") an agent can map
    straight back to the JSON it generated, plus warnings, the normalized
    config (defaults filled in, still JSON-safe and re-runnable), and an
    estimated problem size/cost.
    """
    from fenitop.tools.validate_config import validate_config_tool
    return validate_config_tool(ValidateConfigRequest(config=config))


@mcp.tool()
def run_topopt(config: AgentSafeConfig) -> RunTopoptResponse:
    """Run a fenitop topology optimization from a config.

    Supports 2D "minimize compliance" and "compliant mechanism" problems.
    Always re-validates the config first. Run IDs, output roots, rendering,
    solver profiles, and safety ceilings are application-owned policy and are
    intentionally absent from this schema.
    Never raises past this boundary: solver failures come back as a
    structured error with the last known-good metrics rather than a bare
    traceback. On success, returns converged/stop_reason (and *why* a run
    stopped, not just that it did), the key metrics, and every output
    artifact -- XDMF/H5 time series, the run log, summary.json, a rendered
    density-field PNG, and a coordinate-binned density .npz grid for
    analyze_results to consume without needing dolfinx or h5py.
    """
    from fenitop.tools.run_topopt import run_topopt_tool
    return run_topopt_tool(RunTopoptRequest(config=config))


@mcp.tool()
def analyze_results(run_topopt_envelope: RunTopoptResponse) -> AnalyzeResultsResponse:
    """Analyze a completed fenitop run and summarize it for a human.

    Pass the exact result envelope from run_topopt. Reports convergence
    (including *why* a run didn't converge, e.g.
    the design change was pinned at the move limit rather than genuinely
    stalling), design-quality flags (grayness/binarization score,
    disconnected material via a connected-component check, a checkerboard
    heuristic, and -- if the original config is also supplied -- whether the
    support and load regions are connected through the same material),
    convergence plots, and a deterministic English-language narrative. Needs
    no dolfinx/MPI for its core metrics; the design-quality checks degrade
    gracefully (with a warning, not a failure) if no density-grid artifact is
    available.
    """
    from fenitop.tools.analyze_results import analyze_results_tool
    return analyze_results_tool(
        AnalyzeResultsRequest(run_topopt_envelope=run_topopt_envelope)
    )


def _forbid_unknown_mcp_arguments() -> None:
    """Close FastMCP's generated outer argument models.

    MCP 1.28.1 creates strict nested Pydantic fields but defaults the generated
    function-argument model to ignoring extra top-level keys. The dependency is
    pinned, and the schema/runtime behavior is snapshot-tested, so make that
    generated model explicitly forbid extras before the server starts.
    """
    for registered in mcp._tool_manager._tools.values():
        argument_model = registered.fn_metadata.arg_model
        argument_model.model_config["extra"] = "forbid"
        argument_model.model_rebuild(force=True)
        registered.parameters = argument_model.model_json_schema(by_alias=True)


_forbid_unknown_mcp_arguments()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
