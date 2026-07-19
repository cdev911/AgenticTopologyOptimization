"""Deterministic UI policy for verified analysis plot records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from fenitop.tools.contracts import PlotRecord


PLOT_LABELS = {
    "density_field": "Final optimized design",
    "compliance_vs_iteration": "Compliance objective vs. iteration",
    "objective_vs_iteration": "Mechanism objective vs. iteration",
    "volume_vs_iteration": "Volume fraction vs. iteration",
    "change_vs_iteration": "Design change vs. iteration",
}


@dataclass(frozen=True)
class DisplayPlot:
    role: str
    label: str
    path: Path


def verified_display_plots(
    run_directory: str,
    plots: Sequence[PlotRecord],
) -> tuple[DisplayPlot, ...]:
    """Select known PNGs that still resolve inside the analyzed run directory."""
    try:
        root = Path(run_directory).resolve(strict=True)
    except (OSError, RuntimeError):
        return ()

    selected: list[DisplayPlot] = []
    seen_roles: set[str] = set()
    for plot in plots:
        if plot.role not in PLOT_LABELS or plot.role in seen_roles:
            continue
        candidate = Path(plot.path)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            continue
        if (
            candidate.is_symlink()
            or not resolved.is_file()
            or resolved.suffix.lower() != ".png"
        ):
            continue
        selected.append(
            DisplayPlot(
                role=plot.role,
                label=PLOT_LABELS[plot.role],
                path=resolved,
            )
        )
        seen_roles.add(plot.role)
    return tuple(selected)
