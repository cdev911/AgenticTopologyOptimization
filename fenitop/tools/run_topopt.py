"""Tool 2: run_topopt.

Orchestrates fenitop.topopt.topopt(fem, opt) without touching its physics:
always re-validates first (never trusts the caller already called Tool 1),
enforces a safety ceiling on problem size before spending any CPU, scopes
output into a fresh per-run directory by default, catches every exception
(including PETSc/dolfinx ones) into a structured failure envelope instead of
a bare traceback, derives converged/stop_reason, surfaces the MMA optimizer's
inner-iteration-cap warning, and renders a density-field PNG plus a
coordinate-binned .npz grid while the dolfinx Function/mesh objects are still
live in memory -- so Tool 3 never needs h5py or mesh reconstruction.

Needs the full dolfinx/PETSc/MPI stack; only runs inside the docker image.
"""
from __future__ import annotations

import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from fenitop.tools.config_models import compile_solver_config, translate_validation_error
from fenitop.tools.contracts import (
    RunTopoptRequest,
    RunTopoptResponse,
    TrustedRunPolicy,
    TrustedValidationPolicy,
)
from fenitop.tools.logging_config import get_logger
from fenitop.tools.logreader import count_warnings, read_history
from fenitop.tools.safety import estimate_cost, resource_limit_errors
from fenitop.tools.schema import FieldError, error_envelope, ok_envelope
from fenitop.tools.validate_config import validate_config_tool

_MMA_WARNING_MARKER = "mma_inner_iteration_cap_reached"
logger = get_logger(__name__)


def _make_run_id(output_prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{output_prefix}_{stamp}_{uuid.uuid4().hex[:6]}"


def _truncated_traceback(exc: BaseException, limit_chars: int = 4000) -> str:
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return text[-limit_chars:] if len(text) > limit_chars else text


def _list_partial_artifacts(output_dir: Path) -> List[Dict[str, str]]:
    if not output_dir.is_dir():
        return []
    return [{"role": path.stem, "path": str(path), "format": path.suffix.lstrip(".")}
            for path in sorted(output_dir.iterdir())]


def _render_snapshot(result: Dict[str, Any], opt: Dict[str, Any]
                      ) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """Render a density-field PNG (pyvista, off-screen) and export a
    coordinate-binned .npz grid, both from the serial mesh + gathered density
    values topopt() just returned -- while those live dolfinx objects are
    still in memory, so Tool 3 never needs h5py or mesh reconstruction.

    Reconstructs the CG1 scalar function space on mesh_serial itself (rather
    than using mesh_serial.geometry.x directly) because `density_values_serial`
    is ordered to match that function space's dof order (via S_comm.gather in
    topopt.py), which dolfinx does not guarantee matches raw mesh vertex
    order. Best-effort: a rendering failure is reported as a warning, not a
    tool failure, since the actual optimization already succeeded.
    """
    import numpy as np

    paths: Dict[str, Optional[str]] = {"density_snapshot_png": None, "density_grid_npz": None}
    warnings: List[str] = []

    mesh_serial = result.get("mesh_serial")
    values = result.get("density_values_serial")
    if mesh_serial is None or values is None:
        warnings.append("Density snapshot/grid skipped: no serial density values on this rank.")
        return paths, warnings

    from dolfinx.fem import Function, functionspace

    output_dir = Path(opt["output_folder"])
    prefix = opt["output_prefix"]
    S_serial = functionspace(mesh_serial, ("CG", 1))
    rho_serial = Function(S_serial)
    try:
        rho_serial.x.array[:] = np.asarray(values)
    except ValueError as exc:
        warnings.append(f"Density snapshot/grid skipped: gathered values did not match the "
                         f"serial mesh's dof count ({exc}).")
        return paths, warnings
    coords = S_serial.tabulate_dof_coordinates()

    try:
        xs, ys = coords[:, 0], coords[:, 1]
        ux = np.unique(np.round(xs, 9))
        uy = np.unique(np.round(ys, 9))
        ix = np.searchsorted(ux, np.round(xs, 9))
        iy = np.searchsorted(uy, np.round(ys, 9))
        grid = np.full((uy.size, ux.size), np.nan)
        grid[iy, ix] = rho_serial.x.array
        npz_path = output_dir / f"{prefix}_density_grid.npz"
        np.savez(npz_path, density=grid, x=ux, y=uy)
        paths["density_grid_npz"] = str(npz_path)
    except Exception as exc:  # noqa: BLE001 - best-effort artifact, report don't fail the run
        warnings.append(f"Failed to export density grid (.npz): {exc}")

    try:
        import dolfinx.plot
        import pyvista as pv

        try:
            pv.start_xvfb()
        except Exception:
            pass  # already running, or a real display is available

        topology, cell_types, geometry = dolfinx.plot.vtk_mesh(S_serial)
        grid_pv = pv.UnstructuredGrid(topology, cell_types, geometry)
        grid_pv.point_data["density"] = rho_serial.x.array
        grid_pv.set_active_scalars("density")

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(grid_pv, cmap="gray_r", clim=[0.0, 1.0], show_scalar_bar=True)
        plotter.view_xy()
        plotter.camera.parallel_projection = True
        png_path = output_dir / f"{prefix}_density_snapshot.png"
        plotter.screenshot(str(png_path))
        plotter.close()
        paths["density_snapshot_png"] = str(png_path)
    except Exception as exc:  # noqa: BLE001 - best-effort artifact, report don't fail the run
        warnings.append(f"Failed to render density snapshot (.png): {exc}")

    return paths, warnings


def _reject_oversized(
    cost: Dict[str, Any],
    policy: TrustedRunPolicy,
    validation: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if policy.allow_large_run:
        return None
    limits = policy.resource_limits.model_copy(
        update={
            "max_estimated_wall_time_seconds": min(
                policy.resource_limits.max_estimated_wall_time_seconds,
                policy.timeout_seconds,
            )
        }
    )
    errors = resource_limit_errors(cost, limits)
    if not errors:
        return None
    return error_envelope(
        "run_topopt",
        errors,
        stage="safety_check",
        estimated_cost=cost,
        validation=validation,
        message=(
            "Estimated resource demand exceeds application-owned limits. "
            "The agent-facing request cannot override this policy."
        ),
    )


def _response(data: dict) -> dict:
    RunTopoptResponse.model_validate(data)
    return data


def run_topopt_tool(
    request: Dict[str, Any] | RunTopoptRequest,
    *,
    policy: TrustedRunPolicy | None = None,
) -> Dict[str, Any]:
    """Run a safe config under an application-owned execution policy."""
    run_policy = policy or TrustedRunPolicy()
    try:
        parsed_request = (
            request
            if isinstance(request, RunTopoptRequest)
            else RunTopoptRequest.model_validate(request)
        )
    except ValidationError as exc:
        return _response(error_envelope(
            "run_topopt",
            translate_validation_error(exc),
            stage="request",
            message="Request did not match RunTopoptRequest.",
        ))
    except Exception as exc:
        return _response(error_envelope(
            "run_topopt",
            [FieldError("<root>", "malformed_request", str(exc))],
            stage="request",
        ))
    config = parsed_request.config

    # Cheap pre-check on raw arithmetic (divisions x max_iter, no mesh build)
    # BEFORE the full validate_config_tool call, which -- with
    # check_geometry=True -- builds a real mesh. Without this, an absurdly
    # oversized config (e.g. a 2000x2000 mesh) would pay the cost of actually
    # constructing that huge mesh just to be told it's too big; this fast
    # path rejects it on arithmetic alone first. Any error here (malformed
    # mesh spec, etc.) is swallowed -- the full validation below reports it
    # properly with field paths instead.
    try:
        quick_cost = estimate_cost(
            config.mesh.model_dump(mode="json"),
            config.opt.max_iter,
            problem_type=config.opt.problem_type,
            solver_profile=run_policy.solver_profile,
            output_interval=run_policy.output_interval,
        )
        quick_reject = _reject_oversized(quick_cost, run_policy, validation=None)
        if quick_reject is not None:
            logger.warning("run_topopt: rejected on cheap pre-check, complexity_score=%s",
                            quick_cost["complexity_score"])
            return _response(quick_reject)
    except Exception:  # noqa: BLE001 - fall through to full validation for a proper diagnosis
        pass

    logger.info("run_topopt: re-validating config (full structural + geometry check)...")
    check = validate_config_tool(
        {"config": config},
        policy=TrustedValidationPolicy(
            check_geometry=True,
            enforce_resource_limits=not run_policy.allow_large_run,
            solver_profile=run_policy.solver_profile,
            output_interval=run_policy.output_interval,
            resource_limits=run_policy.resource_limits,
        ),
    )
    if check["status"] == "error":
        logger.warning("run_topopt: pre-flight validation failed with %d error(s)", len(check["errors"]))
        return _response(error_envelope(
            "run_topopt", check["errors"], stage="pre_flight_validation",
            warnings=check.get("warnings", []), validation=check))

    normalized_config = check["normalized_config"]
    reject = _reject_oversized(check["estimated_cost"], run_policy, validation=check)
    if reject is not None:
        logger.warning("run_topopt: rejected by safety ceiling, complexity_score=%s",
                        check["estimated_cost"]["complexity_score"])
        return _response(reject)

    from mpi4py import MPI

    from fenitop.config import build_fem_opt
    from fenitop.topopt import topopt

    comm = MPI.COMM_WORLD
    problem_type = config.opt.problem_type
    output_prefix = run_policy.output_prefix or (
        "compliance_2d" if problem_type == "minimize_compliance" else "mechanism_2d"
    )
    run_id = run_policy.run_id or _make_run_id(output_prefix)
    scoped = run_policy.scoped_output
    output_root = run_policy.output_root
    output_dir = (output_root / run_id) if scoped else output_root

    solver_config = compile_solver_config(
        config,
        solver_profile=run_policy.solver_profile,
        output_folder=str(output_dir),
        output_prefix=output_prefix,
        output_interval=run_policy.output_interval,
    )
    fem, opt = build_fem_opt(solver_config, comm=comm)
    run_log_path = output_dir / f"{output_prefix}_run.log"
    logger.info("run_topopt: run_id=%s output_dir=%s max_iter=%s", run_id, output_dir, opt["max_iter"])

    start = time.perf_counter()
    try:
        result = topopt(fem, opt)
    except Exception as exc:  # noqa: BLE001 - never let a solver failure cross as a bare traceback
        logger.exception("run_topopt: solver raised during run_id=%s", run_id)
        last_known_good = read_history(run_log_path)[-1:]
        return _response(error_envelope(
            "run_topopt", [], stage="solve", run_id=run_id, validation=check,
            error={"exception_type": type(exc).__name__, "message": str(exc),
                   "traceback": _truncated_traceback(exc)},
            last_known_good_metrics=(last_known_good[0] if last_known_good else None),
            artifacts=_list_partial_artifacts(output_dir)))
    wall_time = time.perf_counter() - start

    if comm.rank != 0:
        return _response(ok_envelope(
            "run_topopt", run_id=run_id, rank=comm.rank,
            note="Non-root MPI rank; the full result envelope is produced by rank 0."))

    converged_raw = result["converged_raw"]
    converged = converged_raw["opt_iter"] < opt["max_iter"] or converged_raw["change"] <= opt["opt_tol"]
    stop_reason = "tolerance_met" if converged else "max_iterations_reached"
    mma_warnings = count_warnings(run_log_path, _MMA_WARNING_MARKER)
    logger.info("run_topopt: solve finished in %.2fs, iterations=%d, converged=%s, stop_reason=%s",
                wall_time, converged_raw["opt_iter"], converged, stop_reason)
    if mma_warnings:
        logger.warning("run_topopt: MMA inner-iteration cap was reached %d time(s)", mma_warnings)

    render_paths: Dict[str, Optional[str]] = {"density_snapshot_png": None, "density_grid_npz": None}
    render_warnings: List[str] = []
    if run_policy.render_snapshot:
        render_paths, render_warnings = _render_snapshot(result, opt)

    summary = result.get("summary") or {}
    artifacts = [
        {"role": "density_history", "format": "xdmf",
         "path": str(output_dir / f"{output_prefix}_density_history.xdmf")},
        {"role": "density_history_data", "format": "hdf5",
         "path": str(output_dir / f"{output_prefix}_density_history.h5")},
        {"role": "displacement_history", "format": "xdmf",
         "path": str(output_dir / f"{output_prefix}_displacement_history.xdmf")},
        {"role": "displacement_history_data", "format": "hdf5",
         "path": str(output_dir / f"{output_prefix}_displacement_history.h5")},
        {"role": "run_log", "format": "text+jsonlines", "path": str(run_log_path)},
        {"role": "summary", "format": "json", "path": str(output_dir / f"{output_prefix}_summary.json")},
    ]
    if render_paths.get("density_snapshot_png"):
        artifacts.append({"role": "density_snapshot_png", "format": "png",
                           "path": render_paths["density_snapshot_png"]})
    if render_paths.get("density_grid_npz"):
        artifacts.append({"role": "density_grid", "format": "npz",
                           "path": render_paths["density_grid_npz"]})

    logger.info("run_topopt: status=ok run_id=%s (%d artifact(s))", run_id, len(artifacts))
    return _response(ok_envelope(
        "run_topopt", warnings=[*check.get("warnings", []), *render_warnings],
        run_id=run_id, problem_type=check["problem_type"], converged=converged,
        stop_reason=stop_reason, iterations=converged_raw["opt_iter"],
        metrics={
            "final_compliance": summary.get("final_compliance"),
            "final_volume": summary.get("final_volume"),
            "final_objective": summary.get("final_objective"),
            "grayness": summary.get("grayness"),
            "final_change": converged_raw["change"],
            "opt_tol": opt["opt_tol"],
        },
        mma_inner_iteration_warnings=mma_warnings, wall_time_seconds=wall_time,
        artifacts=artifacts, validation=check, error=None))


def main() -> int:
    from fenitop.tools.cli import run_cli
    return run_cli(run_topopt_tool, "Run a fenitop topology optimization from a validated config.")


if __name__ == "__main__":
    raise SystemExit(main())
