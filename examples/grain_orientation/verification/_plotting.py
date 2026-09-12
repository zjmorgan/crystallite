"""Shared plotting setup for the verification figures in this directory.

Mirrors the reference presentation's own grain-orientation figures (the
homogeneous free-energy landscape, the planar-interface analytical vs.
numerical comparison) using crystallite's actual ``GrainOrientation``
solver. Every figure is saved as ``.pgf`` (for direct LaTeX ``\\input``),
``.pdf``, and ``.png`` (for quick preview) side by side in this file's
``figures/`` directory.
"""

from __future__ import annotations

import os

from crystallite.visualization import configure_pgf

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt = configure_pgf()


def save_all(fig, name):
    """Save ``fig`` as ``<name>.pgf``, ``.pdf``, and ``.png`` in figures/."""
    for ext in ("pgf", "pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"saved figures/{name}.{{pgf,pdf,png}}")
