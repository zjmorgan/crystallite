r"""Reference results for coherent (misfit-driven) diffusion,
:class:`crystallite.chemomechanics.CoherentDiffusion`.

For a homogeneous stiffness :math:`C` and eigenstrain :math:`\varepsilon^*=
\eta\,(c-c_\mathrm{ref})`, mechanical equilibrium in a clamped periodic cell
gives Khachaturyan's elastic energy (A. G. Khachaturyan, *Theory of Structural
Transformations in Solids*, Wiley, 1983), a sum over composition modes
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
from scipy.optimize import brentq
from scipy.special import ellipe


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
    """How much a 2D composition field's spectral power lies along the grid
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


def _pore_perimeter_energy(aspect_ratio, surface_energy, radius):
    r"""Surface energy :math:`T(b/a)=4\gamma b\,E(k^2)` of an elliptical pore
    of area :math:`\pi\,`\ `radius`\ :math:`^2` (semi-axes :math:`a\le b`,
    :math:`k^2=1-a^2/b^2`), `radius` being the equal-area circle's radius."""
    rho = np.asarray(aspect_ratio, dtype=float)
    b = radius * np.sqrt(rho)
    return 4.0 * surface_energy * b * ellipe(1.0 - 1.0 / rho**2)


def _pore_elastic_slope(shear_modulus, poisson_ratio, radius):
    r""":math:`dK/d(b/a)`, `K` the coefficient of :math:`\sigma_\infty^2` in the
    elastic strain energy :math:`U(b/a)=K(b/a)\,\sigma_\infty^2` of a coherent
    (misfit-strain, homogeneous-stiffness) elliptical pore at fixed area
    (McCartney 1977; Heald and Speight 1977): exactly constant, since
    :math:`K=\pi b[a+2b(1-\nu^2)]/[4\mu(1+\nu)]` and :math:`ab` is fixed, so
    the :math:`a` term drops out of the derivative."""
    return np.pi * radius**2 * (1.0 - poisson_ratio) / (2.0 * shear_modulus)


def pore_critical_aspect_ratio():
    r"""Aspect ratio :math:`(b/a)^*` at which a coherent (misfit-strain,
    homogeneous-stiffness) elliptical pore under remote uniaxial tension stops
    having a stable equilibrium shape and grows without bound into a crack
    (McCartney 1977; Heald and Speight 1977) -- about 2.817.

    Because the elastic energy coefficient :math:`K(b/a)` is exactly linear in
    :math:`b/a` at fixed pore area (:func:`_pore_elastic_slope`), the
    saddle-node condition that sets the critical ratio,
    :math:`d(dT/d\rho)/d\rho=d^2T/d\rho^2=0`, involves only the surface energy
    :math:`T(b/a)`: :math:`(b/a)^*` is therefore an inflection point of
    :math:`T` alone, independent of the shear modulus, surface energy,
    Poisson's ratio, and pore size.

    References
    ----------
    .. [1] L. N. McCartney, "A model to describe the creep-deformation and
       fracture of Type 316 stainless steel", Acta Metall. 25, 221 (1977).
    .. [2] P. T. Heald and M. V. Speight, "Steady state creep and cavitation of
       grain boundaries", Mater. Sci. Eng. 29, 271 (1977).
    """
    def curvature(rho, h=1e-4):
        return (
            _pore_perimeter_energy(rho + h, 1.0, 1.0)
            - 2.0 * _pore_perimeter_energy(rho, 1.0, 1.0)
            + _pore_perimeter_energy(rho - h, 1.0, 1.0)
        ) / h**2

    return float(brentq(curvature, 1.5, 6.0, xtol=1e-10))


def pore_critical_stress(shear_modulus, surface_energy, radius, poisson_ratio):
    r"""Remote uniaxial stress :math:`\sigma_\infty^*` above which a coherent
    elliptical pore (:func:`pore_critical_aspect_ratio`) has no stable
    equilibrium shape and grows unboundedly into a crack."""
    rho_c = pore_critical_aspect_ratio()
    h = 1.0e-6
    dt = (
        _pore_perimeter_energy(rho_c + h, surface_energy, radius)
        - _pore_perimeter_energy(rho_c - h, surface_energy, radius)
    ) / (2.0 * h)
    dk = _pore_elastic_slope(shear_modulus, poisson_ratio, radius)
    return float(np.sqrt(dt / dk))


def pore_equilibrium_aspect_ratio(remote_stress, shear_modulus, surface_energy, radius, poisson_ratio):
    r"""Equilibrium aspect ratio :math:`b/a` of a coherent elliptical pore
    (:func:`pore_critical_aspect_ratio`) under remote uniaxial stress
    `remote_stress`, on the stable branch :math:`1<b/a\le(b/a)^*`: the root of
    :math:`dT/d\rho=(dK/d\rho)\,\sigma_\infty^2` there (the free-energy
    stationarity condition), found by bisection since :math:`dG/d\rho<0` at
    :math:`\rho=1` and :math:`\ge0` at :math:`\rho=(b/a)^*` for any
    sub-critical stress.

    Raises
    ------
    ValueError
        If ``abs(remote_stress)`` is at or above
        :func:`pore_critical_stress`: the pore then has no stable shape.
    """
    sigma_c = pore_critical_stress(shear_modulus, surface_energy, radius, poisson_ratio)
    if abs(remote_stress) >= sigma_c:
        raise ValueError(
            f"remote_stress={remote_stress!r} is at or above the critical stress "
            f"{sigma_c:g}: the pore has no stable equilibrium shape"
        )
    rho_c = pore_critical_aspect_ratio()
    dk = _pore_elastic_slope(shear_modulus, poisson_ratio, radius)

    def imbalance(rho, h=1.0e-6):
        dt = (
            _pore_perimeter_energy(rho + h, surface_energy, radius)
            - _pore_perimeter_energy(rho - h, surface_energy, radius)
        ) / (2.0 * h)
        return dt - dk * remote_stress**2

    return float(brentq(imbalance, 1.0 + 1.0e-6, rho_c, xtol=1.0e-10))
