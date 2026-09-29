"""Matplotlib style: Times New Roman, English-friendly defaults."""

from __future__ import annotations

import matplotlib as mpl


def apply_plot_style(font: str = "Times New Roman") -> None:
    """Apply serif plot style (Times New Roman) for titles, ticks, and colorbars."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font, "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "mathtext.fontset": "custom",
            "mathtext.rm": font,
            "mathtext.it": f"{font}:italic",
            "mathtext.bf": f"{font}:bold",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
