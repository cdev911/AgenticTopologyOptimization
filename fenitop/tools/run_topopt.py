"""Tool 2: validate, contain, and run one topology-optimization solve.

The public entry point owns trusted paths, idempotency, capacity, and lifecycle
state, then launches the serial numerical implementation in a credential-scrubbed
child process group. The worker writes native artifacts plus a typed result; the
parent translates timeout, cancellation, signals, and missing/invalid results.

The numerical implementation re-validates before solving, enforces resource
ceilings, reports checked optimizer/convergence state, and renders the density
artifacts while Dolfinx objects are live. It requires the pinned Docker runtime.
"""
from __future__ import annotations

import math
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from fenitop.tools.config_models import (
    compile_solver_config,
    parse_agent_safe_config,
    translate_validation_error,
)
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


def _parse_run_request(request) -> RunTopoptRequest:
    if isinstance(request, RunTopoptRequest):
        config, _ = parse_agent_safe_config(request.config)
        return RunTopoptRequest(config=config)
    if isinstance(request, dict) and isinstance(request.get("config"), dict):
        config, _ = parse_agent_safe_config(request["config"])
        return RunTopoptRequest.model_validate({**request, "config": config})
    return RunTopoptRequest.model_validate(request)


def _make_run_id(output_prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{output_prefix}_{stamp}_{uuid.uuid4().hex[:6]}"


def _list_partial_artifacts(output_dir: Path) -> List[Dict[str, Any]]:
    if not output_dir.is_dir():
        return []
    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if path.is_symlink() or not path.is_file():
            continue
        artifacts.append({
            "role": path.stem,
            "path": str(path),
            "format": path.suffix.lstrip("."),
            "complete": False,
        })
    return artifacts


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


def _agent_world_size() -> int:
    try:
        from mpi4py import MPI
    except ImportError:
        return 1
    return int(MPI.COMM_WORLD.size)


def _run_topopt_in_process(
    request: Dict[str, Any] | RunTopoptRequest,
    *,
    policy: TrustedRunPolicy | None = None,
) -> Dict[str, Any]:
    """Execute Tool 2 in the current process.

    The public entry point invokes this only inside ``solver_worker``.
    It remains importable for focused numerical fault injection tests.
    """
    run_policy = policy or TrustedRunPolicy()
    try:
        parsed_request = _parse_run_request(request)
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
            enforce_resource_limits=True,
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
    output_root = run_policy.output_root.resolve()
    output_dir = output_root / run_id

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

    from fenitop.numerics import NumericalError

    start = time.perf_counter()
    try:
        result = topopt(fem, opt)
    except NumericalError as exc:
        failure = exc.failure
        residual_norm = (
            failure.residual_norm
            if failure.residual_norm is not None
            and math.isfinite(failure.residual_norm)
            else None
        )
        logger.error(
            "run_topopt: numerical failure run_id=%s code=%s component=%s "
            "iteration=%s reason=%s residual=%s",
            run_id, failure.code, failure.component, failure.iteration,
            failure.reason, residual_norm,
        )
        last_known_good = read_history(run_log_path)[-1:]
        return _response(error_envelope(
            "run_topopt",
            [FieldError(
                f"solver.{failure.component}",
                failure.code,
                failure.message,
                severity="error",
                retryable=False,
            )],
            stage="numerical", run_id=run_id, validation=check,
            error={
                "exception_type": type(exc).__name__,
                "message": failure.message,
                "code": failure.code,
                "component": failure.component,
                "iteration": failure.iteration,
                "reason": failure.reason,
                "residual_norm": residual_norm,
                "debug_artifact_role": "run_log",
            },
            last_known_good_metrics=(
                last_known_good[0] if last_known_good else None
            ),
            artifacts=_list_partial_artifacts(output_dir),
        ))
    except Exception as exc:  # noqa: BLE001 - never let a solver failure cross as a bare traceback
        logger.exception("run_topopt: solver raised during run_id=%s", run_id)
        last_known_good = read_history(run_log_path)[-1:]
        return _response(error_envelope(
            "run_topopt",
            [FieldError(
                "solver",
                "unexpected_solver_error",
                "The solver failed unexpectedly; inspect the local run log.",
                retryable=True,
            )],
            stage="solve", run_id=run_id, validation=check,
            error={
                "exception_type": "SolverExecutionError",
                "message": "The solver failed unexpectedly.",
                "code": "unexpected_solver_error",
                "debug_artifact_role": "run_log",
            },
            last_known_good_metrics=(last_known_good[0] if last_known_good else None),
            artifacts=_list_partial_artifacts(output_dir)))
    wall_time = time.perf_counter() - start

    if comm.rank != 0:
        return _response(ok_envelope(
            "run_topopt", run_id=run_id, rank=comm.rank,
            note="Non-root MPI rank; the full result envelope is produced by rank 0."))

    converged_raw = result["converged_raw"]
    converged = converged_raw["converged"]
    stop_reason = converged_raw["stop_reason"]
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
         "path": str(output_dir / f"{output_prefix}_density_history.xdmf"), "complete": True},
        {"role": "density_history_data", "format": "hdf5",
         "path": str(output_dir / f"{output_prefix}_density_history.h5"), "complete": True},
        {"role": "displacement_history", "format": "xdmf",
         "path": str(output_dir / f"{output_prefix}_displacement_history.xdmf"), "complete": True},
        {"role": "displacement_history_data", "format": "hdf5",
         "path": str(output_dir / f"{output_prefix}_displacement_history.h5"), "complete": True},
        {"role": "run_log", "format": "text+jsonlines", "path": str(run_log_path), "complete": True},
        {"role": "summary", "format": "json",
         "path": str(output_dir / f"{output_prefix}_summary.json"), "complete": True},
    ]
    if render_paths.get("density_snapshot_png"):
        artifacts.append({"role": "density_snapshot_png", "format": "png",
                           "path": render_paths["density_snapshot_png"], "complete": True})
    if render_paths.get("density_grid_npz"):
        artifacts.append({"role": "density_grid", "format": "npz",
                           "path": render_paths["density_grid_npz"], "complete": True})

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
            "binarization_score": summary.get("binarization_score"),
            "final_change": converged_raw["change"],
            "opt_tol": opt["opt_tol"],
            "final_beta": converged_raw["final_beta"],
            "continuation_completed": converged_raw["continuation_completed"],
        },
        optimizer_status=converged_raw["optimizer_status"],
        mma_inner_iteration_warnings=mma_warnings, wall_time_seconds=wall_time,
        artifacts=artifacts, validation=check, error=None))


def _terminal_process_error(
    *,
    run_id: str,
    stage: str,
    code: str,
    message: str,
    lifecycle: dict[str, Any],
    output_dir: Path,
    output_prefix: str,
    validation: dict[str, Any],
    retryable: bool,
) -> dict[str, Any]:
    history = read_history(output_dir / f"{output_prefix}_run.log")
    return _response(error_envelope(
        "run_topopt",
        [FieldError("worker", code, message, retryable=retryable)],
        stage=stage,
        run_id=run_id,
        validation=validation,
        lifecycle=lifecycle,
        error={
            "exception_type": code,
            "message": message,
            "code": code,
            "debug_artifact_role": "worker_stderr",
        },
        last_known_good_metrics=(history[-1] if history else None),
        artifacts=_list_partial_artifacts(output_dir),
    ))


def _validate_worker_artifacts(response: dict[str, Any], run_dir: Path) -> None:
    for artifact in response.get("artifacts", []):
        path = Path(artifact["path"])
        if path.is_symlink():
            raise RuntimeError(f"Worker artifact is a symlink: {path.name}")
        resolved = path.resolve(strict=response["status"] == "ok")
        if not resolved.is_relative_to(run_dir):
            raise RuntimeError(f"Worker artifact escaped its run directory: {path}")
        if response["status"] == "ok" and not resolved.is_file():
            raise RuntimeError(f"Worker success artifact is missing: {path}")


def _existing_job_response(
    *,
    run_dir: Path,
    request_hash: str,
    validation: dict[str, Any],
) -> dict[str, Any]:
    from fenitop.tools.lifecycle import (
        RESPONSE_NAME,
        LifecycleError,
        read_json,
        read_lifecycle,
    )

    try:
        lifecycle = read_lifecycle(run_dir)
    except Exception as exc:
        return _response(error_envelope(
            "run_topopt",
            [FieldError(
                "run_id", "run_directory_collision",
                f"Existing run directory has no valid lifecycle manifest: {exc}",
            )],
            stage="lifecycle",
            validation=validation,
        ))
    if lifecycle["request_hash"] != request_hash:
        return _response(error_envelope(
            "run_topopt",
            [FieldError(
                "idempotency_key",
                "idempotency_conflict",
                "The idempotency key/run ID already belongs to a different request.",
            )],
            stage="idempotency",
            run_id=lifecycle["run_id"],
            lifecycle=lifecycle,
            validation=validation,
        ))

    response_path = run_dir / RESPONSE_NAME
    if response_path.is_file() and not response_path.is_symlink():
        try:
            response = read_json(response_path)
            _validate_worker_artifacts(response, run_dir)
            if response["status"] == "ok":
                from fenitop.tools.contracts import RunManifest
                from fenitop.tools.manifest import verify_durable_manifest

                verify_durable_manifest(
                    RunManifest.model_validate(response.get("run_manifest")),
                    expected_run_dir=run_dir,
                )
            response["idempotent_replay"] = True
            response["lifecycle"] = lifecycle
            return _response(response)
        except Exception as exc:
            return _response(error_envelope(
                "run_topopt",
                [FieldError(
                    "response", "stored_response_invalid",
                    f"Stored idempotent response is invalid: {exc}",
                )],
                stage="idempotency",
                run_id=lifecycle["run_id"],
                lifecycle=lifecycle,
                validation=validation,
            ))

    retryable = lifecycle["state"] in {"queued", "running"}
    code = "job_already_active" if retryable else "terminal_response_missing"
    return _response(error_envelope(
        "run_topopt",
        [FieldError(
            "idempotency_key",
            code,
            (
                "The same job is already queued/running."
                if retryable
                else "The existing terminal job has no durable response."
            ),
            retryable=retryable,
        )],
        stage="idempotency",
        run_id=lifecycle["run_id"],
        lifecycle=lifecycle,
        validation=validation,
        idempotent_replay=True,
    ))


def _run_topopt_impl(
    request: Dict[str, Any] | RunTopoptRequest,
    *,
    policy: TrustedRunPolicy | None = None,
) -> Dict[str, Any]:
    """Validate in the parent, then execute one contained serial worker."""
    from fenitop.tools.lifecycle import (
        CANCEL_NAME,
        JOB_MANIFEST_NAME,
        RESPONSE_NAME,
        WORKER_REQUEST_NAME,
        WORKER_RESULT_NAME,
        LifecycleError,
        acquire_active_lock,
        allocate_run_directory,
        atomic_write_json,
        canonical_json_hash,
        check_disk_capacity,
        idempotency_hash,
        new_lifecycle,
        read_json,
        recover_orphaned_jobs,
        release_active_lock,
        resolve_output_root,
        update_lifecycle,
        validate_identifier,
        write_lifecycle,
    )
    from fenitop.tools.worker_process import (
        launch_worker_process,
        sanitized_worker_environment,
    )
    from fenitop.tools.worker_protocol import SolverWorkerRequest, SolverWorkerResult
    from fenitop.tools.manifest import build_run_manifest, write_run_manifest

    run_policy = policy or TrustedRunPolicy()
    try:
        parsed_request = _parse_run_request(request)
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

    if _agent_world_size() != 1:
        return _response(error_envelope(
            "run_topopt",
            [FieldError(
                "mpi_processes",
                "mpi_unsupported",
                "The agent-facing solver worker is serial-only.",
            )],
            stage="execution_policy",
        ))

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
            return _response(quick_reject)
    except Exception:
        pass

    check = validate_config_tool(
        {"config": config},
        policy=TrustedValidationPolicy(
            check_geometry=True,
            enforce_resource_limits=True,
            solver_profile=run_policy.solver_profile,
            output_interval=run_policy.output_interval,
            resource_limits=run_policy.resource_limits,
        ),
    )
    if check["status"] == "error":
        return _response(error_envelope(
            "run_topopt",
            check["errors"],
            stage="pre_flight_validation",
            warnings=check.get("warnings", []),
            validation=check,
        ))
    reject = _reject_oversized(check["estimated_cost"], run_policy, validation=check)
    if reject is not None:
        return _response(reject)

    problem_type = config.opt.problem_type
    output_prefix = run_policy.output_prefix or (
        "compliance_2d" if problem_type == "minimize_compliance" else "mechanism_2d"
    )
    try:
        validate_identifier(output_prefix, field="output_prefix")
        root = resolve_output_root(run_policy.output_root)
        recover_orphaned_jobs(root)
        check_disk_capacity(
            root,
            estimated_output_mb=check["estimated_cost"]["estimated_output_mb"],
            minimum_free_mb=run_policy.min_free_disk_mb,
        )
    except (LifecycleError, OSError) as exc:
        code = getattr(exc, "code", "output_root_error")
        return _response(error_envelope(
            "run_topopt",
            [FieldError("output_root", code, str(exc), retryable=True)],
            stage="filesystem",
            validation=check,
        ))

    key_hash = idempotency_hash(run_policy.idempotency_key)
    if run_policy.run_id is not None:
        run_id = run_policy.run_id
    elif key_hash is not None:
        run_id = f"{output_prefix[:54]}_{key_hash[:16]}"
    else:
        run_id = _make_run_id(output_prefix[:54])
    request_material = {
        "contract_version": check["contract_version"],
        "config": check["normalized_config"],
        "solver_profile": run_policy.solver_profile,
        "output_interval": run_policy.output_interval,
        "render_snapshot": run_policy.render_snapshot,
    }
    request_hash = canonical_json_hash(request_material)

    try:
        run_dir, created = allocate_run_directory(root, run_id)
    except (LifecycleError, OSError) as exc:
        code = getattr(exc, "code", "run_directory_error")
        return _response(error_envelope(
            "run_topopt",
            [FieldError("run_id", code, str(exc))],
            stage="filesystem",
            validation=check,
        ))
    if not created:
        return _existing_job_response(
            run_dir=run_dir, request_hash=request_hash, validation=check
        )

    lifecycle = new_lifecycle(
        run_id=run_id,
        request_hash=request_hash,
        idempotency_key_hash=key_hash,
    )
    try:
        lifecycle = write_lifecycle(run_dir, lifecycle)
    except Exception as exc:
        return _response(error_envelope(
            "run_topopt",
            [FieldError("lifecycle", "manifest_write_failed", str(exc))],
            stage="filesystem",
            run_id=run_id,
            validation=check,
        ))

    try:
        active_lock = acquire_active_lock(root, run_id)
    except LifecycleError as exc:
        lifecycle = update_lifecycle(
            run_dir, lifecycle, state="failed", message=str(exc)
        )
        response = _terminal_process_error(
            run_id=run_id,
            stage="capacity",
            code=exc.code,
            message=str(exc),
            lifecycle=lifecycle,
            output_dir=run_dir,
            output_prefix=output_prefix,
            validation=check,
            retryable=True,
        )
        atomic_write_json(run_dir / RESPONSE_NAME, response)
        return response

    request_path = run_dir / WORKER_REQUEST_NAME
    result_path = run_dir / WORKER_RESULT_NAME
    stdout_path = run_dir / "worker.stdout.log"
    stderr_path = run_dir / "worker.stderr.log"
    cancel_path = run_dir / CANCEL_NAME

    def worker_started(pid: int) -> None:
        nonlocal lifecycle
        lifecycle = update_lifecycle(
            run_dir,
            lifecycle,
            state="running",
            worker_pid=pid,
            message="Solver worker started.",
        )

    command = [
        sys.executable,
        "-m",
        "fenitop.tools.solver_worker",
        "--request",
        str(request_path),
    ]
    repository_root = Path(__file__).resolve().parents[2]
    try:
        worker_request = SolverWorkerRequest(
            run_id=run_id,
            request_hash=request_hash,
            output_dir=run_dir,
            output_prefix=output_prefix,
            config=config,
            validation=check,
            render_snapshot=run_policy.render_snapshot,
            solver_profile=run_policy.solver_profile,
            output_interval=run_policy.output_interval,
        )
        atomic_write_json(request_path, worker_request.model_dump(mode="json"))
        outcome = launch_worker_process(
            command,
            cwd=repository_root,
            environment=sanitized_worker_environment(),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cancel_path=cancel_path,
            timeout_seconds=run_policy.timeout_seconds,
            termination_grace_seconds=run_policy.termination_grace_seconds,
            poll_interval_seconds=run_policy.poll_interval_seconds,
            on_started=worker_started,
        )
    except Exception as exc:
        lifecycle = update_lifecycle(
            run_dir,
            lifecycle,
            state="failed",
            message=f"Worker launch failed: {exc}",
        )
        response = _terminal_process_error(
            run_id=run_id,
            stage="worker_launch",
            code="worker_launch_failed",
            message=str(exc),
            lifecycle=lifecycle,
            output_dir=run_dir,
            output_prefix=output_prefix,
            validation=check,
            retryable=True,
        )
        atomic_write_json(run_dir / RESPONSE_NAME, response)
        return response
    finally:
        release_active_lock(active_lock, run_id)

    history = read_history(run_dir / f"{output_prefix}_run.log")
    last_iteration = history[-1]["iteration"] if history else None
    common_terminal = {
        "worker_pid": outcome.worker_pid,
        "exit_code": outcome.exit_code,
        "terminating_signal": outcome.terminating_signal,
        "last_iteration": last_iteration,
    }
    if outcome.cancelled:
        lifecycle = update_lifecycle(
            run_dir,
            lifecycle,
            state="cancelled",
            cancelled=True,
            message="Cancellation was requested; the worker process group was terminated.",
            **common_terminal,
        )
        response = _terminal_process_error(
            run_id=run_id,
            stage="cancelled",
            code="worker_cancelled",
            message=lifecycle["message"],
            lifecycle=lifecycle,
            output_dir=run_dir,
            output_prefix=output_prefix,
            validation=check,
            retryable=False,
        )
    elif outcome.timed_out:
        lifecycle = update_lifecycle(
            run_dir,
            lifecycle,
            state="timed_out",
            timed_out=True,
            message=f"Worker exceeded the {run_policy.timeout_seconds:g}s timeout.",
            **common_terminal,
        )
        response = _terminal_process_error(
            run_id=run_id,
            stage="timeout",
            code="worker_timed_out",
            message=lifecycle["message"],
            lifecycle=lifecycle,
            output_dir=run_dir,
            output_prefix=output_prefix,
            validation=check,
            retryable=True,
        )
    elif outcome.exit_code != 0 or not result_path.is_file():
        lifecycle = update_lifecycle(
            run_dir,
            lifecycle,
            state="failed",
            message=(
                f"Worker exited with code {outcome.exit_code} "
                "without a valid result."
            ),
            **common_terminal,
        )
        response = _terminal_process_error(
            run_id=run_id,
            stage="worker_crash",
            code="worker_crashed",
            message=lifecycle["message"],
            lifecycle=lifecycle,
            output_dir=run_dir,
            output_prefix=output_prefix,
            validation=check,
            retryable=True,
        )
    else:
        try:
            worker_result = SolverWorkerResult.model_validate(read_json(result_path))
            response = worker_result.response.model_dump(mode="json")
            _validate_worker_artifacts(response, run_dir)
            if worker_result.worker_api_key_present:
                raise RuntimeError("Solver worker inherited an API key.")
            terminal_state = "succeeded" if response["status"] == "ok" else "failed"
            lifecycle = update_lifecycle(
                run_dir,
                lifecycle,
                state=terminal_state,
                worker_api_key_present=worker_result.worker_api_key_present,
                message=(
                    "Solver worker completed successfully."
                    if terminal_state == "succeeded"
                    else "Solver worker returned a typed failure."
                ),
                **common_terminal,
            )
            response["lifecycle"] = lifecycle
            response["idempotent_replay"] = False
            response["wall_time_seconds"] = outcome.wall_time_seconds
            response["artifacts"].extend([
                {
                    "role": "worker_stdout",
                    "format": "text",
                    "path": str(stdout_path),
                    "complete": True,
                },
                {
                    "role": "worker_stderr",
                    "format": "text",
                    "path": str(stderr_path),
                    "complete": True,
                },
                {
                    "role": "job_manifest",
                    "format": "json",
                    "path": str(run_dir / JOB_MANIFEST_NAME),
                    "complete": True,
                },
            ])
            if response["status"] == "ok":
                manifest = build_run_manifest(
                    run_dir=run_dir,
                    output_prefix=output_prefix,
                    request_hash=request_hash,
                    response=response,
                    lifecycle=lifecycle,
                    artifacts=response["artifacts"],
                )
                manifest_path = write_run_manifest(run_dir, manifest)
                response["run_manifest"] = manifest.model_dump(mode="json")
                response["artifacts"].append(
                    {
                        "role": "run_manifest",
                        "format": "json",
                        "path": str(manifest_path),
                        "complete": True,
                    }
                )
            response = _response(response)
        except Exception as exc:
            lifecycle = update_lifecycle(
                run_dir,
                lifecycle,
                state="failed",
                message=f"Worker result validation failed: {exc}",
                **common_terminal,
            )
            response = _terminal_process_error(
                run_id=run_id,
                stage="worker_result",
                code="worker_result_invalid",
                message=lifecycle["message"],
                lifecycle=lifecycle,
                output_dir=run_dir,
                output_prefix=output_prefix,
                validation=check,
                retryable=True,
            )

    atomic_write_json(run_dir / RESPONSE_NAME, response)
    return response


def run_topopt_tool(
    request: Dict[str, Any] | RunTopoptRequest,
    *,
    policy: TrustedRunPolicy | None = None,
) -> Dict[str, Any]:
    """Total public boundary for one contained topology-optimization run."""
    try:
        return _run_topopt_impl(request, policy=policy)
    except Exception:
        logger.exception("run_topopt: unexpected public-boundary failure")
        return _response(error_envelope(
            "run_topopt",
            [FieldError(
                "<root>",
                "internal_error",
                "Solver orchestration failed unexpectedly; inspect local logs.",
                retryable=True,
            )],
            stage="internal",
        ))


def main() -> int:
    from fenitop.tools.cli import run_cli
    return run_cli(
        run_topopt_tool,
        "Run a fenitop topology optimization from a validated config.",
        tool_name="run_topopt",
    )


if __name__ == "__main__":
    raise SystemExit(main())
