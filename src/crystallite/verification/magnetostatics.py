r"""Verification case for :class:`crystallite.magnetostatics.Magnetostatics`: a
uniformly magnetized cylinder.

For a uniform in-plane magnetization :math:`M` in a disk of radius :math:`a`,
isolated, the field inside is uniform, :math:`H_\mathrm{in}=-M/2`, and outside
a dipole,

.. math::

    H_\mathrm{out}=-\frac{a^2}{2}\left[\frac{M}{r^2}
    -\frac{2(M\cdot x)\,x}{r^4}\right],

with :math:`B=\mu_0(H+M)` inside and :math:`\mu_0H` outside (the potential and
the normal :math:`B` are continuous at :math:`r=a`). The periodic array is
images of that field, with the exact interior :math:`-\tfrac12(1-f)M` (area
fraction :math:`f`) and a zero cell-mean field. It is the same mathematics as
the uniformly polarized cylinder of
:mod:`crystallite.verification.electrostatics` (:math:`\varepsilon=1`,
:math:`P\to M`, :math:`E\to H`, :math:`D\to B/\mu_0`), exposed here with the
magnetic quantities and units.
"""

from __future__ import annotations

from crystallite.verification.electrostatics import polarized_disk_fields, smoothed_disk

__all__ = ["magnetized_disk_fields", "smoothed_disk"]


def magnetized_disk_fields(
    grid, radius, magnetization, permeability=1.0, center=(0.5, 0.5), n_images=8
):
    """``(H, B)``, each ``(3,) + grid.shape``, of a periodic array of cylinders
    uniformly magnetized by `magnetization` ``(Mx, My[, Mz])`` (zero cell-mean
    field). ``B = permeability (H + M)`` inside the disk, ``permeability H``
    outside. Axial ``Mz`` produces no field."""
    field, displacement = polarized_disk_fields(
        grid, radius, magnetization, 1.0, center=center, n_images=n_images
    )
    return field, permeability * displacement
