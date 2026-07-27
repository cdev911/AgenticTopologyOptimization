"""Tool 3: analyze_results.

Deliberately dolfinx/MPI-free for its core metrics: <prefix>_run.log's
"history" JSON lines and <prefix>_summary.json are both plain text/JSON. The
only "mesh-shaped" data this tool touches is Tool 2's coordinate-binned .npz
density grid, never the raw XDMF/H5 (which would need h5py and mesh
reconstruction) -- that's the whole point of Tool 2 exporting it up front.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fenitop.tools.logging_config import get_logger
from fenitop.tools.logreader import read_history
from fenitop.tools.narrative import build_narrative
from fenitop.tools.plotting import plot_convergence, plot_density_grid_fallback
from fenitop.tools.schema import error_envelope, get_or, ok_envelope

logger = get_logger(__name__)
_DEFAULT_GRAYNESS_THRESHOLD = 0.4
_DEFAULT_CHECKERBOARD_THRESHOLD = 0.15
_DEFAULT_DENSITY_THRESHOLD = 0.5


def _artifact_paths_from_envelope(envelope: Dict[str, Any]) -> Dict[str, str]:
    return {a["role"]: a["path"] for a in envelope.get("artifacts", []) if a.get("path")}


def _artifact_paths_from_prefix(output_folder: str, output_prefix: str) -> Dict[str, str]:
    base = Path(output_folder)
    return {
        "run_log": str(base / f"{output_prefix}_run.log"),
        "summary": str(base / f"{output_prefix}_summary.json"),
        "density_grid": str(base / f"{output_prefix}_density_grid.npz"),
        "density_snapshot_png": str(base / f"{output_prefix}_density_snapshot.png"),
    }


def _read_summary(path: Optional[str]) -> Dict[str, Any]:
    if not path or not Path(path).is_file():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _derive_convergence(history: List[Dict[str, Any]], opt_tol: Optional[float],
                         move_limit: Optional[float]) -> Dict[str, Any]:
    """Best-effort convergence derivation from the log alone. Note run.log /
    summary.json never record max_iter, so without a `config` this can only
    compare final_change to opt_tol -- it cannot distinguish "hit the
    iteration cap while still above tolerance" from "the config's max_iter
    happened to equal this run's length right as it converged". When a Tool 2
    envelope is available, its own live-computed converged/stop_reason (which
    does know max_iter) is used instead -- see analyze_results_tool.
    """
    iterate = [r for r in history if r.get("state") == "iterate"]
    if not iterate or opt_tol is None:
        return {
            "converged": None, "stop_reason": "unknown",
            "iterations": (iterate[-1]["iteration"] if iterate else None),
            "final_change": (iterate[-1].get("change") if iterate else None),
            "opt_tol": opt_tol, "fraction_iterations_at_move_limit": None, "move_limit": move_limit,
        }

    final_change = iterate[-1].get("change")
    converged = final_change is not None and final_change <= opt_tol

    fraction_pinned = None
    if move_limit is not None:
        changes = [r.get("change") for r in iterate if r.get("change") is not None]
        if changes:
            fraction_pinned = sum(1 for c in changes if abs(c - move_limit) < 1e-9) / len(changes)

    return {
        "converged": converged,
        "stop_reason": "tolerance_met" if converged else "max_iterations_reached",
        "iterations": iterate[-1]["iteration"],
        "final_change": final_change,
        "opt_tol": opt_tol,
        "fraction_iterations_at_move_limit": fraction_pinned,
        "move_limit": move_limit,
    }


def _check_load_path_connected(coords_xyz, labeled, config: Dict[str, Any]) -> Optional[bool]:
    """Reuses fenitop.config._materialize (the same sandboxed lambda-string/
    region-DSL compiler used everywhere else) to find which labeled
    connected component the Dirichlet (support) and traction (load) regions
    land in. Dilates each region's grid mask by a couple of cells first,
    since the exact boundary nodes are not guaranteed to be solid even in a
    perfectly valid design -- what matters is solid material *near* the BC,
    not literally at it.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation

    from fenitop.config import _materialize

    fem_cfg = config.get("fem", {})
    dirichlet_bcs = fem_cfg.get("dirichlet_bcs", [])
    traction_bcs = fem_cfg.get("traction_bcs", [])
    if not dirichlet_bcs or not traction_bcs:
        return None

    def region_labels(spec) -> Optional[set]:
        """None means "could not evaluate this marker/locator on the grid at
        all" (matched zero grid points -- Tool 1's geometry check should
        already reject this at the facet level). An empty set is a real,
        meaningful result: the marker/locator geometrically matched grid
        points, but none of them (even after dilating outward a couple of
        cells) are near any solid material -- i.e. genuinely no material
        near that support/load region, which is exactly the failure this
        check exists to catch, not an "undetermined" case.
        """
        if spec is None:
            return None
        region_fn = spec if callable(spec) else _materialize(spec)
        if not callable(region_fn):
            return None
        mask = region_fn(coords_xyz).reshape(labeled.shape)
        if not mask.any():
            return None
        mask = binary_dilation(mask, iterations=2)
        return set(int(v) for v in labeled[mask] if v > 0)

    support_label_sets = [region_labels(bc.get("marker")) for bc in dirichlet_bcs]
    load_label_sets = [region_labels(bc.get("locator")) for bc in traction_bcs]
    if any(s is None for s in support_label_sets) or any(s is None for s in load_label_sets):
        return None

    support_labels: set = set().union(*support_label_sets) if support_label_sets else set()
    load_labels: set = set().union(*load_label_sets) if load_label_sets else set()
    return bool(support_labels & load_labels)


def _analyze_density_grid(grid_path: str, density_threshold: float, checkerboard_threshold: float,
                           config: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    """scipy.ndimage-based connected-component + checkerboard heuristics on
    Tool 2's coordinate-binned grid. scipy is already a hard dependency."""
    import numpy as np
    from scipy import ndimage

    warnings: List[str] = []
    data = np.load(grid_path)
    grid, xs, ys = data["density"], data["x"], data["y"]
    finite = np.isfinite(grid)
    binary = np.zeros_like(grid, dtype=bool)
    binary[finite] = grid[finite] >= density_threshold

    labeled, num_components = ndimage.label(binary)
    total_solid = float(binary.sum())
    if num_components > 0 and total_solid > 0:
        sizes = ndimage.sum(binary, labeled, index=range(1, num_components + 1))
        largest_fraction = float(np.max(sizes)) / total_solid
    else:
        largest_fraction = 0.0

    # Checkerboard heuristic: 2x2 alternation kernel, normalized against the
    # grid's own variance. Coarse -- can false-positive near sharp, legitimate
    # gradients at point loads/supports; a heuristic, not a rigorous QA metric.
    fill_value = float(np.nanmean(grid[finite])) if finite.any() else 0.0
    filled = np.where(finite, grid, fill_value)
    kernel = np.array([[1.0, -1.0], [-1.0, 1.0]])
    conv = ndimage.convolve(filled, kernel, mode="nearest")
    variance = float(np.var(filled)) or 1.0
    checkerboard_score = float(np.mean(conv**2)) / (4.0 * variance)

    flags: Dict[str, Any] = {
        "checkerboard_detected": checkerboard_score > checkerboard_threshold,
        "checkerboard_score": checkerboard_score,
        "num_components": int(num_components),
        "largest_component_fraction": largest_fraction,
        "has_disconnected_material": num_components > 1,
        "load_path_connected": None,
    }

    if config is not None:
        try:
            X, Y = np.meshgrid(xs, ys)  # shape (len(ys), len(xs)), matches grid/labeled
            coords_xyz = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=0)
            flags["load_path_connected"] = _check_load_path_connected(coords_xyz, labeled, config)
        except Exception as exc:  # noqa: BLE001 - optional deeper check, degrade gracefully
            warnings.append(f"Load-path connectivity check failed: {exc}")

    return flags, warnings


def analyze_results_tool(request: Dict[str, Any]) -> Dict[str, Any]:
    """request = {"run_topopt_envelope"?: dict} OR {"output_folder", "output_prefix"},
    plus optional "config", "opt_tol", "move_limit", "grayness_threshold",
    "checkerboard_threshold", "density_threshold", "make_plots"=True."""
    envelope = request.get("run_topopt_envelope")
    config = request.get("config")
    opt_cfg = config.get("opt", {}) if config else {}

    prior_converged = prior_stop_reason = prior_iterations = None
    run_id = None

    if envelope is not None:
        if envelope.get("status") != "ok":
            logger.error("analyze_results: run_topopt_envelope.status was %r, not 'ok'", envelope.get("status"))
            return error_envelope("analyze_results", [], stage="request",
                                   message="run_topopt_envelope.status was not 'ok'; nothing to analyze.")
        paths = _artifact_paths_from_envelope(envelope)
        output_prefix = request.get("output_prefix") or envelope.get("run_id") or "fenitop"
        run_id = envelope.get("run_id")
        problem_type = get_or(envelope, "problem_type", "minimize_compliance")
        opt_tol = get_or(request, "opt_tol", envelope.get("metrics", {}).get("opt_tol"))
        prior_converged = envelope.get("converged")
        prior_stop_reason = envelope.get("stop_reason")
        prior_iterations = envelope.get("iterations")
    else:
        output_folder = request.get("output_folder")
        output_prefix = request.get("output_prefix")
        if not output_folder or not output_prefix:
            return error_envelope(
                "analyze_results", [], stage="request",
                message="request must include either 'run_topopt_envelope' or both "
                        "'output_folder' and 'output_prefix'.")
        paths = _artifact_paths_from_prefix(output_folder, output_prefix)
        problem_type = get_or(request, "problem_type", "minimize_compliance")
        opt_tol = get_or(request, "opt_tol", opt_cfg.get("opt_tol"))

    logger.info("analyze_results: analyzing output_prefix=%s run_id=%s", output_prefix, run_id)
    move_limit = get_or(request, "move_limit", opt_cfg.get("move"))
    grayness_threshold = get_or(request, "grayness_threshold", _DEFAULT_GRAYNESS_THRESHOLD)
    checkerboard_threshold = get_or(request, "checkerboard_threshold", _DEFAULT_CHECKERBOARD_THRESHOLD)
    density_threshold = get_or(request, "density_threshold", _DEFAULT_DENSITY_THRESHOLD)
    make_plots = get_or(request, "make_plots", True)

    run_log_path = paths.get("run_log")
    if not run_log_path or not Path(run_log_path).is_file():
        logger.error("analyze_results: run_log not found at %r", run_log_path)
        return error_envelope("analyze_results", [], stage="request",
                               message=f"run_log not found at {run_log_path!r}; cannot analyze.")

    history = read_history(run_log_path)
    summary = _read_summary(paths.get("summary"))

    convergence = _derive_convergence(history, opt_tol, move_limit)
    if prior_converged is not None:
        # Trust Tool 2's own live-computed values (it knows max_iter) over a
        # log-only re-derivation, which can only compare against opt_tol.
        convergence["converged"] = prior_converged
        convergence["stop_reason"] = prior_stop_reason
        convergence["iterations"] = prior_iterations

    warnings: List[str] = []

    grayness = summary.get("grayness")
    quality_flags: Dict[str, Any] = {
        "grayness": grayness,
        "binarization_score": grayness,
        "grayness_threshold": grayness_threshold,
        "low_grayness_warning": bool(grayness is not None and grayness < grayness_threshold),
        "checkerboard_detected": None,
        "checkerboard_score": None,
        "num_components": None,
        "largest_component_fraction": None,
        "has_disconnected_material": None,
        "load_path_connected": None,
    }

    grid_path = paths.get("density_grid")
    if grid_path and Path(grid_path).is_file():
        grid_flags, grid_warnings = _analyze_density_grid(
            grid_path, density_threshold, checkerboard_threshold, config)
        quality_flags.update(grid_flags)
        warnings.extend(grid_warnings)
    else:
        logger.warning("analyze_results: no density grid (.npz) found for %s; "
                        "design-quality checks skipped", output_prefix)
        warnings.append(
            "No density grid (.npz) artifact found; connected-component and checkerboard "
            "checks were skipped. Re-run Tool 2 with render_snapshot=true to enable them.")

    plots: List[Dict[str, str]] = []
    if make_plots:
        plot_dir = Path(run_log_path).parent
        plots = plot_convergence(history, plot_dir, output_prefix, opt_tol=opt_tol, move_limit=move_limit)

    density_png = paths.get("density_snapshot_png")
    if density_png and Path(density_png).is_file():
        plots.append({"role": "density_field", "path": density_png, "source": "reused_from_run_topopt"})
    elif grid_path and Path(grid_path).is_file():
        try:
            fallback_path = Path(run_log_path).parent / f"{output_prefix}_density_field_fallback.png"
            plot_density_grid_fallback(grid_path, fallback_path)
            plots.append({"role": "density_field", "path": str(fallback_path), "source": "matplotlib_fallback"})
        except Exception as exc:  # noqa: BLE001 - a plot failure shouldn't fail the whole analysis
            warnings.append(f"Failed to render fallback density image: {exc}")

    metrics = {
        "final_compliance": summary.get("final_compliance"),
        "final_volume": summary.get("final_volume"),
        "final_objective": summary.get("final_objective"),
        "vol_frac_target": opt_cfg.get("vol_frac"),
    }
    narrative = build_narrative(convergence, quality_flags, metrics, problem_type)
    logger.info("analyze_results: status=ok converged=%s stop_reason=%s (%d warning(s))",
                convergence["converged"], convergence["stop_reason"], len(warnings))

    return ok_envelope(
        "analyze_results", warnings=warnings,
        source={"output_folder": str(Path(run_log_path).parent), "output_prefix": output_prefix, "run_id": run_id},
        convergence=convergence, quality_flags=quality_flags, metrics=metrics,
        plots=plots, narrative=narrative)


def main() -> int:
    from fenitop.tools.cli import run_cli
    return run_cli(analyze_results_tool, "Analyze a completed fenitop run and summarize the results.")


if __name__ == "__main__":
    raise SystemExit(main())
