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

from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fenitop")


@mcp.tool()
def validate_config(config: Dict[str, Any], check_geometry: bool = True) -> Dict[str, Any]:
    """Pedantically validate a fenitop topology-optimization JSON config before running it.

    Runs structural checks (mesh/material/optimizer parameters, physical
    ranges such as -1 < Poisson's ratio < 0.5, conditional-required fields
    for compliant-mechanism mode) and, when check_geometry is true, builds a
    real mesh to confirm every boundary/load marker matches at least one
    facet (a marker matching zero facets would otherwise be silently
    dropped) plus a rigid-body-motion sanity check. Returns errors with
    dotted field paths (e.g. "fem.dirichlet_bcs[0].marker") an agent can map
    straight back to the JSON it generated, plus warnings, the normalized
    config (defaults filled in, still JSON-safe and re-runnable), and an
    estimated problem size/cost.
    """
    from fenitop.tools.validate_config import validate_config_tool
    return validate_config_tool({"config": config, "check_geometry": check_geometry})


@mcp.tool()
def run_topopt(config: Dict[str, Any], run_id: Optional[str] = None,
               output_root: Optional[str] = None, scoped_output: bool = True,
               allow_large_run: bool = False, max_complexity_override: Optional[float] = None,
               render_snapshot: bool = True) -> Dict[str, Any]:
    """Run a fenitop topology optimization from a config.

    Supports 2D "minimize compliance" and "compliant mechanism" problems.
    Always re-validates the config first (never trusts that validate_config
    was already called). Rejects problems above a default cost ceiling
    unless allow_large_run is set or max_complexity_override raises it --
    the ceiling only guards this default single-process invocation; a
    legitimately large job should be run under mpirun directly instead.
    Writes outputs to a fresh, timestamped run directory by default
    (scoped_output=false reproduces the legacy flat-overwrite behavior).
    Never raises past this boundary: solver failures come back as a
    structured error with the last known-good metrics rather than a bare
    traceback. On success, returns converged/stop_reason (and *why* a run
    stopped, not just that it did), the key metrics, and every output
    artifact -- XDMF/H5 time series, the run log, summary.json, a rendered
    density-field PNG, and a coordinate-binned density .npz grid for
    analyze_results to consume without needing dolfinx or h5py.
    """
    from fenitop.tools.run_topopt import run_topopt_tool
    return run_topopt_tool({
        "config": config, "run_id": run_id, "output_root": output_root,
        "scoped_output": scoped_output, "allow_large_run": allow_large_run,
        "max_complexity_override": max_complexity_override, "render_snapshot": render_snapshot,
    })


@mcp.tool()
def analyze_results(run_topopt_envelope: Optional[Dict[str, Any]] = None,
                     output_folder: Optional[str] = None, output_prefix: Optional[str] = None,
                     config: Optional[Dict[str, Any]] = None, opt_tol: Optional[float] = None,
                     move_limit: Optional[float] = None, grayness_threshold: float = 0.4,
                     checkerboard_threshold: float = 0.15, density_threshold: float = 0.5,
                     make_plots: bool = True) -> Dict[str, Any]:
    """Analyze a completed fenitop run and summarize it for a human.

    Pass either the full result envelope from run_topopt (preferred -- reuses
    its already-computed converged/stop_reason and artifact paths directly),
    or an output_folder/output_prefix pointing at an older run's artifacts on
    disk. Reports convergence (including *why* a run didn't converge, e.g.
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
    return analyze_results_tool({
        "run_topopt_envelope": run_topopt_envelope, "output_folder": output_folder,
        "output_prefix": output_prefix, "config": config, "opt_tol": opt_tol,
        "move_limit": move_limit, "grayness_threshold": grayness_threshold,
        "checkerboard_threshold": checkerboard_threshold, "density_threshold": density_threshold,
        "make_plots": make_plots,
    })


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
