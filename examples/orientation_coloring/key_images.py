"""One inverse-pole-figure colour key per point group, for figure making: each
key twice, with its corners labelled by plane normals ``(hkl)`` and by
lattice directions ``[uvw]``, in ``keys/`` as PDF, as PGF for LaTeX
(``\\input{<group>_<indices>.pgf}``, with the colour fill in the
``-img0.png`` file written beside it) and as a 300 dpi PNG with a
transparent background.

The keys are those of :class:`crystallite.visualization.orientation.InversePoleFigure`
at the ``"point_group"`` level (a crystal's key at the default ``"laue"``
level is that of its Laue class). No lattice is given, so the labels hold for
any cell: every corner lies along a crystal axis, in a basal or prism plane,
or along a cube diagonal, where plane and direction indices coincide.
Monoclinic keys label the crystal axes that are exact for any monoclinic
angle instead of their corners: [001] and [010], or (100) and (010) (see
:meth:`~crystallite.visualization.orientation.InversePoleFigure.label_points`).

Plots use the project's shared style,
:func:`crystallite.visualization.configure_pgf`.
"""

from __future__ import annotations

from pathlib import Path

from crystallite.material.symmetry import _POINT_GROUPS_BY_SYSTEM
from crystallite.visualization.orientation import InversePoleFigure, plot_ipf_key

INDICES = ("planes", "directions")


def file_stem(point_group):
    """File-name-safe point group symbol: ``m-3m`` -> ``mbar3m``,
    ``4/mmm`` -> ``4_mmm``."""
    return point_group.replace("-", "bar").replace("/", "_")


def run_example(out_dir="keys", resolution=400, formats=("pdf", "pgf", "png"), height=2.2):
    """Write the colour key of every point group with each kind of label.

    Returns
    -------
    list of Path
        The files written.
    """
    try:
        from crystallite.visualization import configure_pgf

        plt = configure_pgf()
    except ImportError:  # pragma: no cover
        print("Matplotlib is not installed; skipping plot output for this run.")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for groups in _POINT_GROUPS_BY_SYSTEM.values():
        for point_group in groups:
            for indices in INDICES:
                ipf = InversePoleFigure(point_group, "point_group", indices=indices)
                fig, ax = plt.subplots(figsize=(height, height))
                plot_ipf_key(ipf, ax=ax, resolution=resolution)
                (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
                fig.set_size_inches(height * (x1 - x0) / (y1 - y0), height)  # one scale for all panels
                for fmt in formats:
                    path = out_dir / f"{file_stem(point_group)}_{indices}.{fmt}"
                    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=True)
                    written.append(path)
                plt.close(fig)
    print(f"Wrote {len(written)} files to {out_dir}/")
    return written


if __name__ == "__main__":
    run_example(out_dir=Path(__file__).parent / "keys")
