"""Shared plotting setup for the verification figures in this directory.

Every figure is saved as ``.pgf`` (for direct LaTeX ``\\input``), ``.pdf``,
and ``.png`` (for quick preview) side by side in this file's ``figures/``
directory.
"""

from __future__ import annotations

import os

from crystallite.visualization import configure_pgf

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt = configure_pgf()


def analytic_numeric_curve(ax, xi, analytic, numeric, label, color, downsample=4):
    """Plot one stress/strain component: a solid analytic curve and
    downsampled numerical markers, both in `color`, distinguished by a
    legend entry each ("`label` analytical" / "`label` numerical")."""
    ax.plot(xi, analytic, "-", color=color, label=f"{label} analytical")
    ax.plot(
        xi[::downsample], numeric[::downsample], "o", color=color, markersize=3,
        label=f"{label} numerical",
    )


def save_all(fig, name):
    """Save ``fig`` as ``<name>.pgf``, ``.pdf``, and ``.png`` in figures/."""
    for ext in ("pgf", "pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"saved figures/{name}.{{pgf,pdf,png}}")
