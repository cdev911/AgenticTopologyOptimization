"""Tool 3: verify a RunManifest and derive deterministic analysis evidence."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fenitop.tools.config_models import translate_validation_error
from fenitop.tools.contracts import (
    AnalyzeResultsRequest,
    AnalyzeResultsResponse,
    IterationMetrics,
    RunManifest,
    TrustedAnalysisPolicy,
)
from fenitop.tools.logging_config import get_logger
from fenitop.tools.manifest import (
    ManifestError,
    verify_manifest_artifacts,
)
from fenitop.tools.narrative import build_narrative
from fenitop.tools.plotting import plot_convergence, plot_density_grid_fallback
from fenitop.tools.schema import FieldError, error_envelope, ok_envelope

logger = get_logger(__name__)
_CHECKERBOARD_METHOD = "binary_2x2_alternation_v1"
_CONNECTIVITY_METHOD = "component_labels_filter_scaled_dilation_v1"


def _response(data: dict[str, Any]) -> dict[str, Any]:
    AnalyzeResultsResponse.model_validate(data)
    return data


def _read_json_object(path: Path, *, max_bytes: int = 2 * 1024 * 1024) -> dict[str, Any]:
    if path.stat().st_size > max_bytes:
        raise ManifestError("summary_too_large", "Summary exceeds the analysis limit.")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ManifestError("summary_invalid", "Summary must contain a JSON object.")
    return value


def _read_history_strict(path: Path) -> list[dict[str, Any]]:
    from fenitop.tools.logreader import HistoryParseError, read_history

    try:
        raw = read_history(path, strict=True)
    except HistoryParseError as exc:
        raise ManifestError("history_record_invalid", str(exc)) from exc
    if not raw:
        raise ManifestError("history_empty", "Run history contains no evaluated states.")
    records: list[dict[str, Any]] = []
    previous_iteration = -1
    for index, item in enumerate(raw):
        try:
            record = IterationMetrics.model_validate(item).model_dump(mode="json")
        except ValidationError as exc:
            raise ManifestError(
                "history_record_invalid", f"History record {index} is invalid: {exc}"
            ) from exc
        if record["iteration"] <= previous_iteration:
            raise ManifestError(
                "history_not_monotonic", "History iterations must be strictly increasing."
            )
        previous_iteration = record["iteration"]
        records.append(record)
    if records[0]["state"] != "initial" or records[0]["iteration"] != 0:
        raise ManifestError(
            "history_initial_state_missing",
            "History must begin with evaluated initial state iteration zero.",
        )
    return records


def _verify_summary(
    summary: dict[str, Any],
    manifest: RunManifest,
    history: list[dict[str, Any]],
) -> None:
    if summary.get("iterations") != manifest.iterations:
        raise ManifestError(
            "summary_iteration_mismatch",
            "Summary iteration count does not match the RunManifest.",
        )
    if history[-1]["iteration"] != manifest.iterations:
        raise ManifestError(
            "history_iteration_mismatch",
            "Final history iteration does not match the RunManifest.",
        )
    comparisons = {
        "final_compliance": manifest.metrics.final_compliance,
        "final_volume": manifest.metrics.final_volume,
        "final_objective": manifest.metrics.final_objective,
        "grayness": manifest.metrics.grayness,
        "binarization_score": manifest.metrics.binarization_score,
    }
    for field, expected in comparisons.items():
        observed = summary.get(field)
        if expected is None and observed is None:
            continue
        if (
            expected is None
            or observed is None
            or not math.isfinite(float(observed))
            or not math.isclose(float(observed), expected, rel_tol=1e-9, abs_tol=1e-11)
        ):
            raise ManifestError(
                "summary_metric_mismatch",
                f"Summary field {field} does not match the RunManifest.",
            )
    final_history = history[-1]
    for history_field, metric_field in (
        ("compliance", "final_compliance"),
        ("volume", "final_volume"),
        ("objective", "final_objective"),
        ("grayness", "grayness"),
        ("binarization_score", "binarization_score"),
        ("change", "final_change"),
    ):
        observed = final_history.get(history_field)
        expected = getattr(manifest.metrics, metric_field)
        if observed is None:
            continue
        if not math.isclose(
            float(observed), float(expected), rel_tol=1e-9, abs_tol=1e-11
        ):
            raise ManifestError(
                "history_metric_mismatch",
                f"Final history field {history_field} does not match the RunManifest.",
            )


def _derive_convergence(
    history: list[dict[str, Any]], manifest: RunManifest
) -> dict[str, Any]:
    opt = manifest.normalized_config.opt
    iterate = [record for record in history if record["state"] == "iterate"]
    changes = [
        float(record["change"])
        for record in iterate
        if record.get("change") is not None
    ]
    fraction_pinned = None
    move_limit_pinned = None
    if changes:
        fraction_pinned = sum(
            math.isclose(value, float(opt.move), rel_tol=0, abs_tol=1e-9)
            for value in changes
        ) / len(changes)
        move_limit_pinned = fraction_pinned > 0.5

    tail_changes = changes[-8:]
    plateau = None
    if len(tail_changes) >= 4:
        plateau = (
            max(tail_changes) - min(tail_changes)
            <= max(float(opt.opt_tol) * 0.1, 1e-12)
            and tail_changes[-1] > float(opt.opt_tol)
        )

    objectives = [
        float(record["objective"])
        for record in iterate[-10:]
        if record.get("objective") is not None
    ]
    oscillation = None
    if len(objectives) >= 5:
        deltas = [
            right - left for left, right in zip(objectives, objectives[1:])
            if not math.isclose(right, left, rel_tol=1e-12, abs_tol=1e-14)
        ]
        sign_changes = sum(
            left * right < 0 for left, right in zip(deltas, deltas[1:])
        )
        oscillation = sign_changes >= 3

    return {
        "converged": manifest.converged,
        "stop_reason": manifest.stop_reason,
        "iterations": manifest.iterations,
        "final_change": manifest.metrics.final_change,
        "opt_tol": manifest.metrics.opt_tol,
        "fraction_iterations_at_move_limit": fraction_pinned,
        "move_limit": float(opt.move),
        "final_beta": manifest.metrics.final_beta,
        "continuation_completed": manifest.metrics.continuation_completed,
        "iteration_cap_reached": (
            manifest.iterations >= opt.max_iter and not manifest.converged
        ),
        "move_limit_pinned": move_limit_pinned,
        "oscillation_detected": oscillation,
        "plateau_detected": plateau,
        "optimizer_warning_count": manifest.mma_inner_iteration_warnings,
    }


def _grid_region_connectivity(
    xs,
    ys,
    labeled,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool | None, str]:
    import numpy as np
    from scipy.ndimage import binary_dilation

    from fenitop.regions import compile_region

    x_spacing = float(np.min(np.diff(xs)))
    y_spacing = float(np.min(np.diff(ys)))
    cell_scale = min(x_spacing, y_spacing)
    filter_radius = float(config["opt"]["filter_radius"])
    dilation_cells = max(1, min(32, math.ceil(filter_radius / cell_scale)))
    method = f"{_CONNECTIVITY_METHOD}:dilation_cells={dilation_cells}"

    X, Y = np.meshgrid(xs, ys)
    coords_xyz = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=0)

    def labels_near(spec) -> tuple[int, list[int]]:
        mask = compile_region(spec)(coords_xyz).reshape(labeled.shape)
        matched = int(mask.sum())
        if not matched:
            return 0, []
        near = binary_dilation(mask, iterations=dilation_cells)
        labels = sorted({int(value) for value in labeled[near] if value > 0})
        return matched, labels

    def labels_near_selector(selector) -> tuple[int, list[int]]:
        if selector["kind"] == "expert_region":
            return labels_near(selector["region"])
        (x0, y0), (x1, y1) = config["mesh"]["bounds"]
        edge = selector["edge"]
        interval = selector["interval"]
        low, high = (
            (y0, y1) if edge in {"left", "right"} else (x0, x1)
        )
        start, end = float(interval["start"]), float(interval["end"])
        if interval["kind"] == "fraction":
            start, end = (
                low + start * (high - low),
                low + end * (high - low),
            )
        if edge in {"left", "right"}:
            edge_value = x0 if edge == "left" else x1
            normal = np.abs(X - edge_value) <= 0.51 * x_spacing
            along = (Y >= start) & (Y <= end)
        else:
            edge_value = y0 if edge == "bottom" else y1
            normal = np.abs(Y - edge_value) <= 0.51 * y_spacing
            along = (X >= start) & (X <= end)
        mask = normal & along
        matched = int(mask.sum())
        if not matched:
            return 0, []
        near = binary_dilation(mask, iterations=dilation_cells)
        labels = sorted({int(value) for value in labeled[near] if value > 0})
        return matched, labels

    support_labels: set[int] = set()
    boundaries = config["fem"]["boundary_conditions"]
    for boundary in boundaries:
        if boundary["kind"] != "fixed":
            continue
        _, labels = labels_near_selector(boundary["selector"])
        support_labels.update(labels)

    regions: list[tuple[str, int, dict[str, Any]]] = [
        ("traction", index, boundary["selector"])
        for index, boundary in enumerate(
            bc for bc in boundaries if bc["kind"] != "fixed"
        )
    ]
    if config["opt"]["problem_type"] == "compliant_mechanism":
        regions.extend([
            ("spring", 0, config["opt"]["in_spring"]["region"]),
            ("spring", 1, config["opt"]["out_spring"]["region"]),
        ])

    records: list[dict[str, Any]] = []
    determinate: list[bool] = []
    for kind, index, spec in regions:
        matched, nearby = (
            labels_near(spec)
            if kind == "spring"
            else labels_near_selector(spec)
        )
        connected = None if matched == 0 else bool(support_labels.intersection(nearby))
        if connected is not None:
            determinate.append(connected)
        records.append({
            "region_kind": kind,
            "region_index": index,
            "matched_grid_points": matched,
            "connected_to_support": connected,
            "nearby_component_labels": nearby,
        })
    aggregate = all(determinate) if len(determinate) == len(records) and records else None
    return records, aggregate, method


def _analyze_density_grid(
    grid_path: Path,
    manifest: RunManifest,
    policy: TrustedAnalysisPolicy,
) -> tuple[dict[str, Any], list[str], bool]:
    import numpy as np
    from scipy import ndimage

    if grid_path.stat().st_size > int(policy.max_grid_mb * 1024**2):
        raise ManifestError("density_grid_too_large", "Density grid exceeds analysis limit.")
    try:
        with np.load(grid_path, allow_pickle=False) as data:
            if set(data.files) != {"density", "x", "y"}:
                raise ManifestError(
                    "density_grid_keys_invalid",
                    "Density grid must contain exactly density, x, and y arrays.",
                )
            grid = np.asarray(data["density"])
            xs = np.asarray(data["x"])
            ys = np.asarray(data["y"])
    except (ValueError, OSError) as exc:
        raise ManifestError("density_grid_invalid", f"Cannot read density grid: {exc}") from exc

    nx, ny = manifest.normalized_config.mesh.divisions
    (x0, y0), (x1, y1) = manifest.normalized_config.mesh.bounds
    if (
        grid.ndim != 2
        or xs.ndim != 1
        or ys.ndim != 1
        or grid.shape != (ny + 1, nx + 1)
        or grid.shape != (ys.size, xs.size)
    ):
        raise ManifestError(
            "density_grid_shape_mismatch",
            "Density grid dimensions do not match the normalized mesh config.",
        )
    if not all(np.issubdtype(array.dtype, np.number) for array in (grid, xs, ys)):
        raise ManifestError(
            "density_grid_dtype_invalid",
            "Density grid arrays must use numeric, non-object dtypes.",
        )
    if (
        not np.isfinite(grid).all()
        or not np.isfinite(xs).all()
        or not np.isfinite(ys).all()
        or not np.all(np.diff(xs) > 0)
        or not np.all(np.diff(ys) > 0)
        or not np.isclose(xs[0], float(x0))
        or not np.isclose(xs[-1], float(x1))
        or not np.isclose(ys[0], float(y0))
        or not np.isclose(ys[-1], float(y1))
    ):
        raise ManifestError(
            "density_grid_values_invalid",
            "Density grid coordinates/densities must be finite and monotonic.",
        )
    density_bounds_satisfied = bool(
        np.all(grid >= -1e-10) and np.all(grid <= 1.0 + 1e-10)
    )
    if not density_bounds_satisfied:
        raise ManifestError(
            "density_grid_bounds_invalid", "Density grid contains values outside [0,1]."
        )
    computed_grayness = float(np.mean(4.0 * grid * (1.0 - grid)))
    computed_binarization = 1.0 - computed_grayness
    if (
        not math.isclose(
            computed_grayness,
            float(manifest.metrics.grayness),
            rel_tol=1e-9,
            abs_tol=1e-11,
        )
        or not math.isclose(
            computed_binarization,
            float(manifest.metrics.binarization_score),
            rel_tol=1e-9,
            abs_tol=1e-11,
        )
    ):
        raise ManifestError(
            "density_grid_metric_mismatch",
            "Density grid grayness/binarization does not match the RunManifest.",
        )

    binary = grid >= policy.density_threshold
    labeled, num_components = ndimage.label(binary)
    total_solid = int(binary.sum())
    if num_components and total_solid:
        sizes = ndimage.sum(binary, labeled, index=range(1, num_components + 1))
        largest_fraction = float(np.max(sizes)) / total_solid
    else:
        largest_fraction = 0.0

    if grid.shape[0] >= 2 and grid.shape[1] >= 2:
        alternating = (
            (binary[:-1, :-1] == binary[1:, 1:])
            & (binary[:-1, 1:] == binary[1:, :-1])
            & (binary[:-1, :-1] != binary[:-1, 1:])
        )
        checkerboard_score = float(alternating.mean())
    else:
        checkerboard_score = 0.0

    config = manifest.normalized_config.model_dump(mode="json")
    connectivity, load_path_connected, connectivity_method = (
        _grid_region_connectivity(xs, ys, labeled, config)
    )
    flags = {
        "grayness": computed_grayness,
        "binarization_score": computed_binarization,
        "checkerboard_detected": (
            checkerboard_score > policy.checkerboard_threshold
        ),
        "checkerboard_score": checkerboard_score,
        "num_components": int(num_components),
        "largest_component_fraction": largest_fraction,
        "has_disconnected_material": num_components > 1,
        "load_path_connected": load_path_connected,
        "checkerboard_method": _CHECKERBOARD_METHOD,
        "connectivity_method": connectivity_method,
        "connectivity": connectivity,
    }
    return flags, [], density_bounds_satisfied


def _constraint_analysis(
    manifest: RunManifest,
    *,
    density_bounds_satisfied: bool,
    volume_tolerance: float,
) -> dict[str, Any]:
    opt = manifest.normalized_config.opt
    final_volume = manifest.metrics.final_volume
    if final_volume is None:
        raise ManifestError("final_volume_missing", "Manifest has no final volume.")
    target = float(opt.vol_frac)
    volume_error = float(final_volume) - target
    compliance_bound = getattr(opt, "compliance_bound", None)
    final_compliance = manifest.metrics.final_compliance
    compliance_satisfied = None
    if compliance_bound is not None:
        compliance_satisfied = bool(
            final_compliance is not None
            and final_compliance <= float(compliance_bound) * (1 + 1e-9)
        )
    return {
        "volume_target": target,
        "volume_error": volume_error,
        "volume_tolerance": volume_tolerance,
        "volume_satisfied": abs(volume_error) <= volume_tolerance,
        "compliance_bound": (
            float(compliance_bound) if compliance_bound is not None else None
        ),
        "compliance_bound_satisfied": compliance_satisfied,
        "density_bounds_satisfied": density_bounds_satisfied,
    }


def _analyze_results_impl(
    request: dict[str, Any] | AnalyzeResultsRequest,
    *,
    policy: TrustedAnalysisPolicy | None = None,
) -> dict[str, Any]:
    analysis_policy = policy or TrustedAnalysisPolicy()
    try:
        parsed = (
            request
            if isinstance(request, AnalyzeResultsRequest)
            else AnalyzeResultsRequest.model_validate(request)
        )
    except ValidationError as exc:
        return _response(error_envelope(
            "analyze_results",
            translate_validation_error(exc),
            stage="request",
            message="Request did not match AnalyzeResultsRequest.",
        ))

    manifest = parsed.run_manifest
    try:
        paths = verify_manifest_artifacts(
            manifest,
            analysis_policy.allowed_roots,
            max_total_bytes=int(analysis_policy.max_total_artifact_mb * 1024**2),
        )
        if "run_log" not in paths or "summary" not in paths:
            raise ManifestError(
                "required_artifact_missing",
                "Manifest must include complete run_log and summary artifacts.",
            )
        history = _read_history_strict(paths["run_log"])
        summary = _read_json_object(paths["summary"])
        _verify_summary(summary, manifest, history)
    except (ManifestError, OSError, json.JSONDecodeError) as exc:
        code = getattr(exc, "code", "artifact_validation_failed")
        return _response(error_envelope(
            "analyze_results",
            [FieldError("run_manifest", code, str(exc))],
            stage="artifact_validation",
        ))

    convergence = _derive_convergence(history, manifest)
    warnings: list[str] = []
    grayness = manifest.metrics.grayness
    quality_flags: dict[str, Any] = {
        "grayness": grayness,
        "binarization_score": manifest.metrics.binarization_score,
        "grayness_threshold": analysis_policy.grayness_threshold,
        "high_grayness_warning": bool(
            grayness is not None and grayness > analysis_policy.grayness_threshold
        ),
        "checkerboard_detected": None,
        "checkerboard_score": None,
        "num_components": None,
        "largest_component_fraction": None,
        "has_disconnected_material": None,
        "load_path_connected": None,
        "checkerboard_method": None,
        "connectivity_method": None,
        "connectivity": [],
    }
    density_bounds_satisfied = True
    grid_path = paths.get("density_grid")
    if grid_path is not None:
        try:
            grid_flags, grid_warnings, density_bounds_satisfied = (
                _analyze_density_grid(grid_path, manifest, analysis_policy)
            )
        except ManifestError as exc:
            return _response(error_envelope(
                "analyze_results",
                [FieldError("run_manifest.artifacts", exc.code, str(exc))],
                stage="artifact_validation",
            ))
        quality_flags.update(grid_flags)
        warnings.extend(grid_warnings)
    else:
        warnings.append(
            "No density grid artifact is present; optional topology heuristics "
            "were skipped."
        )

    constraints = _constraint_analysis(
        manifest,
        density_bounds_satisfied=density_bounds_satisfied,
        volume_tolerance=analysis_policy.volume_tolerance,
    )
    metrics = {
        "final_compliance": manifest.metrics.final_compliance,
        "final_volume": manifest.metrics.final_volume,
        "final_objective": manifest.metrics.final_objective,
        "constraints": constraints,
    }

    plots: list[dict[str, str]] = []
    run_dir = Path(manifest.run_directory).resolve(strict=True)
    if analysis_policy.make_plots:
        plots = plot_convergence(
            history,
            run_dir,
            manifest.output_prefix,
            opt_tol=manifest.metrics.opt_tol,
            move_limit=float(manifest.normalized_config.opt.move),
        )
    density_png = paths.get("density_snapshot_png")
    if density_png is not None:
        plots.append({
            "role": "density_field",
            "path": str(density_png),
            "source": "verified_run_artifact",
        })
    elif grid_path is not None and analysis_policy.make_plots:
        fallback_path = (
            run_dir / f"{manifest.output_prefix}_density_field_fallback.png"
        )
        try:
            plot_density_grid_fallback(grid_path, fallback_path)
        except Exception:
            logger.exception("analyze_results: fallback rendering failed")
            warnings.append(
                "Failed to render the optional fallback density image; inspect "
                "local logs."
            )
        else:
            plots.append({
                "role": "density_field",
                "path": str(fallback_path),
                "source": "matplotlib_fallback",
            })

    narrative = build_narrative(
        convergence,
        quality_flags,
        metrics,
        manifest.problem_type,
    )
    return _response(ok_envelope(
        "analyze_results",
        warnings=warnings,
        source={
            "run_directory": str(run_dir),
            "output_prefix": manifest.output_prefix,
            "run_id": manifest.run_id,
            "manifest_hash": manifest.manifest_hash,
        },
        convergence=convergence,
        quality_flags=quality_flags,
        metrics=metrics,
        plots=plots,
        narrative=narrative,
    ))


def analyze_results_tool(
    request: dict[str, Any] | AnalyzeResultsRequest,
    *,
    policy: TrustedAnalysisPolicy | None = None,
) -> dict[str, Any]:
    """Analyze only a verified successful RunManifest."""
    try:
        return _analyze_results_impl(request, policy=policy)
    except Exception as exc:
        logger.exception("analyze_results: unexpected public-boundary failure")
        return _response(error_envelope(
            "analyze_results",
            [FieldError(
                "<root>",
                "internal_error",
                "Analysis failed unexpectedly; inspect local logs.",
                retryable=True,
            )],
            stage="internal",
        ))


def main() -> int:
    from fenitop.tools.cli import run_cli

    return run_cli(
        analyze_results_tool,
        "Analyze a completed fenitop run and summarize the results.",
        tool_name="analyze_results",
    )


if __name__ == "__main__":
    raise SystemExit(main())
