"""Matplotlib convergence plots for Tool 3.

Agg backend set at import time so this is headless-safe regardless of the
caller's environment (doesn't rely on an MPLBACKEND env var being set).
Pure numpy/matplotlib -- no dolfinx needed.

Each chart is a single-series line (compliance, volume, or design-change vs.
iteration) -- one axis, no dual-axis combination, no legend (a single series
needs none, the title names it), threshold/reference lines rendered as
recessive muted dashes with a direct text label rather than a second
competing series. Colors are the validated default palette's sequential blue
for the data line and its neutral chrome tokens for everything else
(references/palette.md in the dataviz skill).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Union  # noqa: E402

_SURFACE = "#fcfcfb"
_LINE = "#2a78d6"       # validated sequential/categorical-slot-1 blue
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"


def _line_plot(iterations: List[float], values: List[Optional[float]], ylabel: str, title: str,
                output_path: Union[str, Path], *, log_y: bool = False,
                hlines: Optional[List[tuple]] = None) -> None:
    xs = [i for i, v in zip(iterations, values) if v is not None]
    ys = [v for v in values if v is not None]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.plot(xs, ys, color=_LINE, linewidth=2)
    if log_y:
        ax.set_yscale("log")
    if hlines:
        x_right = xs[-1] if xs else 0
        for y, label in hlines:
            ax.axhline(y, color=_BASELINE, linestyle="--", linewidth=1.0, zorder=0)
            ax.annotate(label, xy=(x_right, y), xytext=(-4, 4), textcoords="offset points",
                        ha="right", va="bottom", color=_INK_MUTED, fontsize=8)

    ax.set_xlabel("iteration", color=_INK_SECONDARY)
    ax.set_ylabel(ylabel, color=_INK_SECONDARY)
    ax.set_title(title, color=_INK_PRIMARY, fontsize=11, loc="left")
    ax.tick_params(colors=_INK_MUTED, labelsize=8)
    ax.grid(True, axis="y", color=_GRIDLINE, linewidth=0.8, zorder=-1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_BASELINE)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def plot_convergence(history: List[Dict[str, Any]], output_dir: Union[str, Path], prefix: str,
                      opt_tol: Optional[float] = None, move_limit: Optional[float] = None
                      ) -> List[Dict[str, str]]:
    """Render compliance/volume/change-vs-iteration plots from Tool 3's
    parsed run.log history. Returns an artifact list (role + path)."""
    iterate = [r for r in history if r.get("state") == "iterate"]
    output_dir = Path(output_dir)
    plots: List[Dict[str, str]] = []
    if not iterate:
        return plots

    iters = [r["iteration"] for r in iterate]

    compliance = [r.get("compliance") for r in iterate]
    if any(c is not None for c in compliance):
        path = output_dir / f"{prefix}_compliance_vs_iteration.png"
        _line_plot(iters, compliance, "compliance", "Compliance vs. iteration", path)
        plots.append({"role": "compliance_vs_iteration", "path": str(path)})

    volume = [r.get("volume") for r in iterate]
    if any(v is not None for v in volume):
        path = output_dir / f"{prefix}_volume_vs_iteration.png"
        _line_plot(iters, volume, "volume fraction", "Volume fraction vs. iteration", path)
        plots.append({"role": "volume_vs_iteration", "path": str(path)})

    change = [r.get("change") for r in iterate]
    if any(c is not None for c in change):
        hlines = []
        if opt_tol is not None:
            hlines.append((opt_tol, f"opt_tol={opt_tol:.1e}"))
        if move_limit is not None:
            hlines.append((move_limit, f"move limit={move_limit:g}"))
        path = output_dir / f"{prefix}_change_vs_iteration.png"
        _line_plot(iters, change, "design change (log scale)", "Design change vs. iteration",
                   path, log_y=True, hlines=hlines)
        plots.append({"role": "change_vs_iteration", "path": str(path)})

    return plots


def plot_density_grid_fallback(npz_path: Union[str, Path], output_path: Union[str, Path]) -> str:
    """Cheap matplotlib.imshow fallback for the density field when Tool 2's
    pyvista PNG isn't available (e.g. Tool 3 invoked standalone against an
    older run, or render_snapshot=false was passed to Tool 2)."""
    import numpy as np

    data = np.load(npz_path)
    grid = data["density"]
    height = 6.0 * grid.shape[0] / max(grid.shape[1], 1)
    fig, ax = plt.subplots(figsize=(6, max(height, 1.0)), facecolor=_SURFACE)
    ax.imshow(grid, cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    return str(output_path)
