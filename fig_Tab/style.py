from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    # Reuse the centralised publication defaults where available
    from fedchem.visualization.styles import set_plot_style as _base_set_plot_style
except Exception:  # pragma: no cover - fallback when fedchem is not importable
    _base_set_plot_style = None  # type: ignore[assignment]


@dataclass
class PlotStyle:
    """
    Lightweight configuration for figure styling.

    Instances of this class can be passed around and applied before
    creating figures so that visual tweaks (font size, line width, etc.)
    do not require changes to the plotting logic itself.
    """

    dpi: int = 300
    font_size: float = 10.0
    title_size: float = 11.0
    label_size: float = 10.0
    tick_size: float = 9.0
    legend_size: float = 9.0
    line_width: float = 1.5
    marker_size: float = 5.0
    figure_size: Tuple[float, float] = (6.4, 4.8)
    palette: str = "colorblind"

    def apply(self) -> None:
        """
        Apply this style to matplotlib's global rcParams.
        """
        try:
            if _base_set_plot_style is not None:
                # Use the central helper to ensure consistency across scripts
                _base_set_plot_style(dpi=self.dpi, palette=self.palette)
            else:
                plt.style.use("seaborn-whitegrid")
                mpl.rcParams["figure.dpi"] = self.dpi
        except OSError:
            # Fallback when seaborn styles are not registered
            plt.style.use("default")
            mpl.rcParams["figure.dpi"] = self.dpi

        # Override specific typography / line settings
        mpl.rcParams["font.size"] = self.font_size
        mpl.rcParams["axes.titlesize"] = self.title_size
        mpl.rcParams["axes.labelsize"] = self.label_size
        mpl.rcParams["xtick.labelsize"] = self.tick_size
        mpl.rcParams["ytick.labelsize"] = self.tick_size
        mpl.rcParams["legend.fontsize"] = self.legend_size
        mpl.rcParams["lines.linewidth"] = self.line_width
        mpl.rcParams["lines.markersize"] = self.marker_size
        mpl.rcParams["figure.figsize"] = self.figure_size


def style_from_args(args) -> PlotStyle:
    """
    Convenience helper to build a PlotStyle from argparse-style args.

    Expected optional attributes on `args`:
      - figure_dpi
      - font_size
      - line_width
      - marker_size
    """
    dpi = getattr(args, "figure_dpi", None) or 300
    font_size = getattr(args, "font_size", None) or 10.0
    line_width = getattr(args, "line_width", None) or 1.5
    marker_size = getattr(args, "marker_size", None) or 5.0
    return PlotStyle(
        dpi=int(dpi),
        font_size=float(font_size),
        line_width=float(line_width),
        marker_size=float(marker_size),
    )
