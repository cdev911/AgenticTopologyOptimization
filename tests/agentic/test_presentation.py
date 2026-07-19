from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic.presentation import verified_display_plots
from fenitop.tools.contracts import PlotRecord


class PresentationTests(unittest.TestCase):
    def test_selects_known_pngs_inside_run_and_rejects_untrusted_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            density = run_dir / "density.png"
            density.write_bytes(b"png")
            outside = root / "outside.png"
            outside.write_bytes(b"png")
            text = run_dir / "plot.txt"
            text.write_text("not an image")

            selected = verified_display_plots(
                str(run_dir),
                [
                    PlotRecord(role="density_field", path=str(density)),
                    PlotRecord(role="density_field", path=str(density)),
                    PlotRecord(role="compliance_vs_iteration", path=str(outside)),
                    PlotRecord(role="volume_vs_iteration", path=str(text)),
                    PlotRecord(role="unknown", path=str(density)),
                ],
            )

            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].label, "Final optimized design")
            self.assertEqual(selected[0].path, density.resolve())

    def test_missing_run_directory_produces_no_display_records(self):
        selected = verified_display_plots(
            "/missing/run",
            [PlotRecord(role="density_field", path="/missing/design.png")],
        )

        self.assertEqual(selected, ())


if __name__ == "__main__":
    unittest.main()
