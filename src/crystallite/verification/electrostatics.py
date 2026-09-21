r"""Verification cases for :class:`crystallite.electrostatics.Electrostatics`:
a uniformly charged cylinder and a uniformly polarized cylinder. (The uniformly
magnetized cylinder is the same mathematics with :math:`\varepsilon=1`,
:math:`P\to M`, :math:`E\to H`, :math:`D\to B/\mu_0`, and has its own module,
:mod:`crystallite.verification.magnetostatics`.)

Both are exact and analytic -- no solve, no pixelated mask.

* **Charged cylinder** (radius :math:`a`, density :math:`\rho`): isolated, the
  field is radial, :math:`E_r=\rho r/(2\varepsilon)` inside and
  :math:`\rho a^2/(2\varepsilon r)` outside. The periodic array, with the
  :math:`k=0` mode dropped, is the exact Fourier series
  :math:`\hat E=-i\mathbf k\,\hat\rho\,\hat s(k)/(\varepsilon k^2)` with the
  analytic Bessel transform of the disk,
  :math:`\hat s=2\pi aJ_1(ka)/k`.
* **Polarized cylinder** (uniform in-plane :math:`P` in a disk): the
  bound charge is a surface charge :math:`P\cdot n`, and isolated,

  .. math::

      E_\mathrm{in}=-\frac{P}{2\varepsilon},\qquad
      E_\mathrm{out}=-\frac{a^2}{2\varepsilon}\left[\frac{P}{r^2}
      -\frac{2(P\cdot x)\,x}{r^4}\right],

  (both the potential and the normal :math:`D=\varepsilon E+P` are continuous
  at :math:`r=a`). The periodic array is images of that field with the
  interior replaced by the exact region mean :math:`-S^\mathrm{per}P/
  \varepsilon` and the constant fixed by a zero cell-mean field (the image sum
  of an :math:`r^{-2}` field is only conditionally convergent). For a circle in
  a square cell the depolarization tensor is isotropic by symmetry and
  Parseval gives its trace, :math:`1-f` with :math:`f` the area fraction, so
  :math:`S^\mathrm{per}=\tfrac12(1-f)I`, exactly. Axial :math:`P_z` produces
  no field.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j1

from crystallite.grid import Grid


def smoothed_disk(grid, radius, center=(0.5, 0.5)):
    """Lanczos-smoothed (band-limited, area preserving) indicator of a disk,
    shape ``grid.shape``, for building a numeric source."""
    x, y = np.asarray(grid.x[0]), np.asarray(grid.x[1])
    indicator = np.where(np.hypot(x - center[0], y - center[1]) < radius, 1.0, 0.0)
    return np.real(np.asarray(grid.ifft(grid.fft(indicator.astype(np.float32)) * grid.lanczos_filter)))


def _disk_transform(grid, radius, center):
    """Fourier transform of a disk in the unnormalized-FFT convention of
    ``grid.fft``: ``(N / V) 2 pi a J1(ka) / k`` times the shift phase, over the
    active axes."""
    k = np.sqrt(np.asarray(grid.k2, dtype=float))
    safe = np.where(k == 0, 1.0, k)
    continuum = np.where(
        k == 0, np.pi * radius**2, 2.0 * np.pi * radius * j1(safe * radius) / safe
    )
    phase = np.exp(-1j * sum(np.asarray(grid.k[i], dtype=float) * center[i] for i in range(2)))
    points, volume = 1, 1.0
    for axis in grid.fft_axes:
        points *= grid.shape[axis]
        volume *= grid.lengths[axis]
    return (points / volume) * continuum * phase


def charged_disk_field(
    grid, radius, charge_density, permittivity=1.0, center=(0.5, 0.5), refine=4
):
    """Electric field ``(3,) + grid.shape`` of a periodic array of uniformly
    charged cylinders (density `charge_density`, radius `radius`) in a
    homogeneous medium, with the uniform neutralizing background (``k = 0``
    dropped).

    Exact in Fourier space with the analytic Bessel transform of the disk (no
    pixelated mask), synthesized on a grid `refine` times finer and sampled at
    the points of `grid`, so its Gibbs ringing sits far from the sampled
    field. Isolated, the field is radial, ``rho r / 2 eps`` inside and
    ``rho a^2 / 2 eps r`` outside."""
    fine = Grid(shape=(grid.shape[0] * refine, grid.shape[1] * refine, 1), lengths=grid.lengths)
    k = [np.asarray(kk, dtype=float) for kk in fine.k]
    k2 = np.asarray(fine.k2, dtype=float)
    safe = np.where(k2 == 0, 1.0, k2)
    density_hat = charge_density * _disk_transform(fine, radius, center)
    field = np.empty((3,) + grid.shape)
    for i in range(3):
        e_hat = np.where(k2 == 0, 0.0, -1j * k[i] * density_hat / (permittivity * safe))
        field[i] = np.real(np.asarray(fine.ifft(e_hat)))[::refine, ::refine, :]
    return field


def polarized_disk_fields(
    grid, radius, polarization, permittivity=1.0, center=(0.5, 0.5), n_images=8
):
    """``(E, D)``, each ``(3,) + grid.shape``, of a periodic array of
    cylinders uniformly polarized by `polarization` ``(Px, Py[, Pz])`` in a
    homogeneous medium (zero cell-mean field). `D` is ``eps E + P`` inside the
    disk, ``eps E`` outside."""
    if grid.lengths[0] != grid.lengths[1]:
        raise ValueError("the polarized-cylinder reference needs a square cell")
    polarization = np.zeros(3) + np.asarray(polarization, dtype=float)
    in_plane = polarization[:2]
    area_fraction = np.pi * radius**2 / (grid.lengths[0] * grid.lengths[1])
    interior = -0.5 * (1.0 - area_fraction) * in_plane / permittivity
    dipole = -0.5 * radius**2 * in_plane / permittivity

    x = np.asarray(grid.x[0], dtype=float) - center[0]
    y = np.asarray(grid.x[1], dtype=float) - center[1]
    x, y = np.broadcast_to(x, grid.shape), np.broadcast_to(y, grid.shape)
    length_x, length_y = grid.lengths[:2]
    raw = np.zeros((2,) + grid.shape)
    for n1 in range(-n_images, n_images + 1):
        for n2 in range(-n_images, n_images + 1):
            dx, dy = x - n1 * length_x, y - n2 * length_y
            r2 = np.where((dx == 0) & (dy == 0), 1.0, dx**2 + dy**2)
            dot = dipole[0] * dx + dipole[1] * dy
            raw[0] += dipole[0] / r2 - 2.0 * dot * dx / r2**2
            raw[1] += dipole[1] / r2 - 2.0 * dot * dy / r2**2

    inside = np.hypot(x, y) < radius
    n_inside, n_outside = inside.sum(), (~inside).sum()
    field = np.zeros((3,) + grid.shape)
    for i in range(2):
        background = -(n_inside * interior[i] + raw[i][~inside].sum()) / n_outside
        field[i] = np.where(inside, interior[i], background + raw[i])
    displacement = permittivity * field
    displacement += np.where(inside[None], polarization[:, None, None, None], 0.0)
    return field, displacement
