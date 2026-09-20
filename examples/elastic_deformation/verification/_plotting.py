"""Shared plotting setup for the verification figures in this directory.

Every figure is saved as ``.pgf`` (for direct LaTeX ``\\input``), ``.pdf``,
and ``.png`` (for quick preview) side by side in this file's ``figures/``
directory.
"""

from __future__ import annotations

import os

import numpy as np

from crystallite.visualization import configure_pgf

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt = configure_pgf()

from matplotlib.lines import Line2D  # noqa: E402  (needs matplotlib, checked above)


def analytic_numeric_curve(
    ax, x, analytic, numeric, label, color, downsample=4, x_numeric=None, markersize=3
):
    """Plot one component: a solid analytic line and numerical markers, both
    in `color`.

    The legend gets a single entry per component (a line with a marker), not
    one each for analytical and numerical; say in the caption that the solid
    line is analytical and the markers are numerical. Build the legend with
    :func:`component_legend`. `x_numeric` gives the marker positions when they
    differ from the line's `x`; `downsample` thins the numerical markers.
    """
    x = np.asarray(x)
    x_numeric = x if x_numeric is None else np.asarray(x_numeric)
    ax.plot(x, np.asarray(analytic), "-", color=color)
    ax.plot(
        x_numeric[::downsample], np.asarray(numeric)[::downsample], "o", color=color,
        markersize=markersize,
    )
    entries = ax.__dict__.setdefault("_component_entries", [])
    entries.append(
        Line2D([], [], color=color, linestyle="-", marker="o", markersize=markersize, label=label)
    )


def component_legend(ax, **kwargs):
    """Legend with one entry per component drawn by :func:`analytic_numeric_curve`."""
    handles = ax.__dict__.get("_component_entries", [])
    kwargs.setdefault("fontsize", 7)
    kwargs.setdefault("ncol", min(len(handles), 2))
    return ax.legend(handles=handles, numpoints=1, **kwargs)


def save_all(fig, name):
    """Save ``fig`` as ``<name>.pgf``, ``.pdf``, and ``.png`` in figures/."""
    for ext in ("pgf", "pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"saved figures/{name}.{{pgf,pdf,png}}")
