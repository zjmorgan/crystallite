r"""Reference results for coherent (misfit-driven) diffusion,
:class:`crystallite.chemomechanics.CoherentDiffusion`.

For a homogeneous stiffness :math:`C` and eigenstrain :math:`\varepsilon^*=
\eta\,(c-c_\mathrm{ref})`, mechanical equilibrium in a clamped periodic cell
gives Khachaturyan's elastic energy, a sum over composition modes
:math:`E_\mathrm{el}=\tfrac12\sum_{k\neq0}B(\hat n)\,|\hat c_k|^2`, with
:math:`\hat n=k/|k|` and

.. math::

    B(\hat n)=\eta:C:\eta-\hat n\cdot(C:\eta)\cdot
    \left(\hat n\cdot C\cdot\hat n\right)^{-1}\cdot(C:\eta)\cdot\hat n .

The elastic chemical potential of a mode is then :math:`\mu_\mathrm{el}=
B(\hat n)\,c_k`, so :math:`B` is the elastic stiffness of a modulation along
:math:`\hat n`. For a dilatational misfit in a cubic crystal it is smallest along
:math:`\langle100\rangle` when the Zener ratio :math:`A_\mathrm{Z}>1` and along
:math:`\langle111\rangle` when :math:`A_\mathrm{Z}<1` (in a plane,
:math:`\langle110\rangle`), and independent of direction for
:math:`A_\mathrm{Z}=1`, which is what selects the morphology when the misfit
alone is the driver.
"""

from __future__ import annotations

import numpy as np


def khachaturyan_b(stiffness, eigenstrain, direction):
    """Elastic energy per unit modulation amplitude squared, ``B(n)``, of a
    composition wave along `direction` (any vector, normalized): stiffness
    ``(3, 3, 3, 3)``, eigenstrain per unit composition ``(3, 3)``."""
    n = np.asarray(direction, dtype=float)
    n = n / np.linalg.norm(n)
    stiffness = np.asarray(stiffness, dtype=float)
    eigenstrain = np.asarray(eigenstrain, dtype=float)
    sigma = np.einsum("ijkl,kl->ij", stiffness, eigenstrain)
    acoustic = np.einsum("ijkl,j,l->ik", stiffness, n, n)
    t = sigma @ n
    return np.einsum("ij,ij", eigenstrain, sigma) - t @ np.linalg.solve(acoustic, t)


def axis_diagonal_anisotropy(composition, k_min=2.0, k_max=None):
    """How much a 2D composition field's structure factor lies along the grid
    axes rather than the diagonals: ``(W_axes - W_diagonals) / (W_axes +
    W_diagonals)``, the weights of ``|c_k|^2`` in the sectors within 22.5
    degrees of an axis and of a diagonal, for ``k_min < |k| < k_max`` (in
    units of the fundamental wavenumber; default ``k_max = n / 4``). +1 is
    pure axis modulation, -1 pure diagonal, 0 no preference.

    Parameters
    ----------
    composition : array_like of shape (n, n) or (n, n, 1)
    """
    composition = np.asarray(composition, dtype=float)
    composition = composition.reshape(composition.shape[:2])
    n = composition.shape[0]
    k_max = n / 4.0 if k_max is None else k_max
    structure = np.abs(np.fft.fftshift(np.fft.fft2(composition - composition.mean()))) ** 2
    k = np.fft.fftshift(np.fft.fftfreq(n)) * n
    kx, ky = np.meshgrid(k, k, indexing="ij")
    radius = np.hypot(kx, ky)
    angle = np.degrees(np.arctan2(ky, kx)) % 90.0
    band = (radius > k_min) & (radius < k_max)
    near_axis = (angle < 22.5) | (angle > 67.5)
    axes, diagonals = structure[band & near_axis].sum(), structure[band & ~near_axis].sum()
    return float((axes - diagonals) / (axes + diagonals))


def interface_orientation(composition, weight=None):
    r"""Orientation of the interfaces of a 2D composition field, modulo the
    90-degree cubic symmetry: the phase of the four-fold order parameter
    :math:`\langle|\nabla c|^2e^{4i\phi}\rangle/\langle|\nabla c|^2\rangle`,
    with :math:`\phi` the direction of :math:`\nabla c` (the interface normal)
    and the average over the (optionally `weight`-ed, e.g. one grain's interior)
    points.

    For a modulation along the cube axes rotated by :math:`\theta` this returns
    :math:`\theta`; along the diagonals, :math:`\theta+45^\circ`.

    Parameters
    ----------
    composition : array_like of shape (n, m) or (n, m, 1)
        A periodic field on a square grid (angles in grid units).
    weight : array_like of the same shape, optional

    Returns
    -------
    angle : float
        Degrees, in ``[0, 90)``.
    strength : float
        The magnitude of the order parameter, in ``[0, 1]``: 1 for interfaces
        all along one cubic orientation, near 0 for none preferred.
    """
    composition = np.asarray(composition, dtype=float)
    composition = composition.reshape(composition.shape[:2])
    nx, ny = composition.shape
    transform = np.fft.fft2(composition)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx)[:, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(ny)[None, :]
    gx = np.real(np.fft.ifft2(1j * kx * transform))
    gy = np.real(np.fft.ifft2(1j * ky * transform))
    power = gx**2 + gy**2
    if weight is not None:
        power = power * np.asarray(weight, dtype=float).reshape(composition.shape)
    order = np.sum(power * np.exp(4j * np.arctan2(gy, gx))) / np.sum(power)
    return float((np.degrees(np.angle(order)) / 4.0) % 90.0), float(abs(order))
