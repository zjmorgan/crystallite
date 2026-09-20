"""Verification case for a uniform eigenstrain in an *anisotropic* matrix.

An ellipse (a circle when ``semi_axis_a == semi_axis_b``) carrying a uniform
eigenstrain in an otherwise homogeneous, generally anisotropic matrix -- the
classical Eshelby problem with a general stiffness. The stress inside is
uniform and depends on the shape, the stiffness tensor and the eigenstrain.

The analytic references need no FFT and no grid. Both come from the same
object, the Green-operator response of the matrix to a unit eigenstrain,

.. math::

    \\hat\\varepsilon_{ij}(\\hat\\xi)=\\tfrac12(\\xi_i g_j+\\xi_j g_i),\\qquad
    g=N(\\hat\\xi)^{-1}\\,(C:\\varepsilon^*)\\,\\hat\\xi,\\qquad
    N_{ik}(\\hat\\xi)=C_{ijkl}\\hat\\xi_j\\hat\\xi_l,

with :math:`\\hat\\xi=(\\cos\\theta,\\sin\\theta,0)` (the axis of the cylinder is
:math:`x_3`, so plane strain is built in and the in-plane and antiplane
components are handled together, including any coupling a rotated stiffness
introduces):

- the *isolated* ellipse Eshelby tensor is the angular average
  :math:`S=\\frac{1}{2\\pi}\\int_0^{2\\pi}F(\\theta)\\,
  \\frac{ab}{a^2\\cos^2\\theta+b^2\\sin^2\\theta}\\,d\\theta` (Mura's
  formula; a smooth periodic integrand, so a uniform-angle sum is
  spectrally accurate);
- the *periodic* tensor is the lattice sum :math:`S=\\frac{1}{A|\\Omega|}
  \\sum_{k\\ne0}\\hat s(k)^2 F(\\hat k)` with the exact Bessel transform of the
  ellipse, the anisotropic counterpart of
  :func:`crystallite.verification.elastic_deformation._lattice_eshelby_tensor`.

The interior stress is then :math:`\\sigma=C:(S:\\varepsilon^*-\\varepsilon^*)`.
The numeric counterpart is :class:`crystallite.elastic_deformation.
ElasticDeformation` with its ``stiffness`` argument and the eigenstrain field of
:meth:`AnisotropicInclusionCase.eigenstrain_field`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
from scipy.special import j1

from crystallite.backend import xp
from crystallite.elastic_deformation import ElasticDeformation
from crystallite.grid import Grid

# the six independent (symmetric) index pairs, and the response ordering
_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


# ---------------------------------------------------------------------------
# Stiffness builders
# ---------------------------------------------------------------------------


def isotropic_tensor(lam, mu):
    """Rank-4 isotropic stiffness (host array)."""
    delta = np.eye(3)
    return lam * np.einsum("ij,kl->ijkl", delta, delta) + mu * (
        np.einsum("ik,jl->ijkl", delta, delta) + np.einsum("il,jk->ijkl", delta, delta)
    )


def cubic_stiffness(c11, c12, c44):
    """Rank-4 cubic stiffness with the crystal axes along x, y, z."""
    c = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    if i == j == k == l:
                        c[i, j, k, l] = c11
                    elif i == j and k == l:
                        c[i, j, k, l] = c12
                    elif (i == k and j == l) or (i == l and j == k):
                        c[i, j, k, l] = c44
    return c


def cubic_from_zener(zener, lam=1.0, mu=0.7):
    r"""Cubic stiffness with Zener ratio `zener` :math:`=2c_{44}/(c_{11}-c_{12})`
    at the *fixed* bulk modulus :math:`K=\lambda+2\mu/3` and fixed
    :math:`c'=(c_{11}-c_{12})/2=\mu`, so only :math:`c_{44}` changes and
    ``zener=1`` is exactly the isotropic medium (`lam`, `mu`)."""
    c_prime = mu
    bulk = lam + 2.0 * mu / 3.0
    return cubic_stiffness(
        bulk + 4.0 * c_prime / 3.0, bulk - 2.0 * c_prime / 3.0, zener * c_prime
    )


def antiplane_orthotropic_stiffness(ratio, lam=1.0, mu=0.7):
    r"""Stiffness that is isotropic (`lam`, `mu`) except for the two antiplane
    shear moduli, :math:`c_{44}=\mu\sqrt{r}` and :math:`c_{55}=\mu/\sqrt{r}` with
    `ratio` :math:`r=c_{44}/c_{55}`, so the geometric mean
    :math:`\sqrt{c_{44}c_{55}}=\mu` is fixed and ``ratio=1`` is isotropic.
    (:math:`c_{44}` is the :math:`x_2x_3` shear, :math:`c_{55}` the :math:`x_1x_3`.)"""
    c = isotropic_tensor(lam, mu)
    c44, c55 = mu * np.sqrt(ratio), mu / np.sqrt(ratio)
    for (i, j, value) in ((1, 2, c44), (0, 2, c55)):
        c[i, j, i, j] = c[i, j, j, i] = c[j, i, i, j] = c[j, i, j, i] = value
    return c


def rotate_stiffness(stiffness, angle):
    """Rotate a rank-4 stiffness by `angle` (radians) about the x3 axis."""
    cos, sin = np.cos(angle), np.sin(angle)
    r = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return np.einsum("ia,jb,kc,ld,abcd->ijkl", r, r, r, r, stiffness)


# ---------------------------------------------------------------------------
# Green-operator response of the matrix to a unit eigenstrain
# ---------------------------------------------------------------------------


def _acoustic_matrices(stiffness):
    """``N(xi) = Axx xi_x^2 + Axy xi_x xi_y + Ayy xi_y^2`` for in-plane xi."""
    axx = stiffness[:, 0, :, 0]
    axy = stiffness[:, 0, :, 1] + stiffness[:, 1, :, 0]
    ayy = stiffness[:, 1, :, 1]
    return axx, axy, ayy


def _unit_response(stiffness, xi_x, xi_y):
    """Response ``R[m][n]`` (``m``: eigenstrain pair, ``n``: strain pair, both
    in the order of ``_PAIRS``) at the unit in-plane directions
    ``(xi_x, xi_y)`` (arrays of one shape): the strain
    ``sym(xi (x) g)``, ``g = N^{-1} (C:E) xi``, of a unit eigenstrain E."""
    axx, axy, ayy = _acoustic_matrices(stiffness)
    n = [
        [axx[i, k] * xi_x**2 + axy[i, k] * xi_x * xi_y + ayy[i, k] * xi_y**2 for k in range(3)]
        for i in range(3)
    ]
    # adjugate inverse of the 3x3 acoustic tensor, on whole arrays
    cof = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            i1, i2 = [x for x in range(3) if x != i]
            j1_, j2 = [x for x in range(3) if x != j]
            minor = n[i1][j1_] * n[i2][j2] - n[i1][j2] * n[i2][j1_]
            cof[i][j] = minor if (i + j) % 2 == 0 else -minor
    det = n[0][0] * cof[0][0] + n[0][1] * cof[0][1] + n[0][2] * cof[0][2]
    xi = (xi_x, xi_y, 0.0 * xi_x)
    response = []
    for (k, l) in _PAIRS:
        unit = np.zeros((3, 3))
        unit[k, l] = 1.0
        sigma_star = np.einsum("ijmn,mn->ij", stiffness, unit)
        t = [sigma_star[i, 0] * xi_x + sigma_star[i, 1] * xi_y for i in range(3)]
        # g = N^{-1} t = adj(N) t / det, adj = cofactor transpose
        g = [sum(cof[j][i] * t[j] for j in range(3)) / det for i in range(3)]
        response.append(
            [0.5 * (xi[i] * g[j] + xi[j] * g[i]) for (i, j) in _PAIRS]
        )
    return response


def _full_tensor(pair_response):
    """Assemble ``S[i, j, k, l]`` from ``pair_response[m][n]`` scalars."""
    tensor = np.zeros((3, 3, 3, 3))
    for m, (k, l) in enumerate(_PAIRS):
        for n, (i, j) in enumerate(_PAIRS):
            value = pair_response[m][n]
            for (a, b) in {(i, j), (j, i)}:
                for (c, d) in {(k, l), (l, k)}:
                    tensor[a, b, c, d] = value
    return tensor


def isolated_eshelby_tensor(stiffness, semi_axis_a, semi_axis_b, n_theta=4096):
    """Eshelby tensor ``S[i, j, k, l]`` of an isolated ellipse cylinder in the
    anisotropic matrix `stiffness` -- Mura's angular-average formula (see the
    module docstring), a uniform-angle sum of a smooth periodic function."""
    theta = 2.0 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    a, b = semi_axis_a, semi_axis_b
    weight = a * b / (a**2 * np.cos(theta) ** 2 + b**2 * np.sin(theta) ** 2)
    response = _unit_response(np.asarray(stiffness, dtype=float), np.cos(theta), np.sin(theta))
    return _full_tensor([[np.mean(weight * r) for r in row] for row in response])


def _lattice_sum(stiffness, a, b, length_x, length_y, n_modes, chunk=32):
    cell_area, region_area = length_x * length_y, np.pi * a * b
    ky = (2.0 * np.pi * np.arange(-n_modes, n_modes + 1) / length_y)[None, :]
    total = [[0.0] * 6 for _ in range(6)]
    for start in range(-n_modes, n_modes + 1, chunk):
        m = np.arange(start, min(start + chunk, n_modes + 1))
        kx = (2.0 * np.pi * m / length_x)[:, None]
        kmag = np.sqrt(kx**2 + ky**2)
        origin = kmag == 0
        safe = np.where(origin, 1.0, kmag)
        q = np.sqrt((a * kx) ** 2 + (b * ky) ** 2)
        q_safe = np.where(q == 0, 1.0, q)
        shape2 = np.where(origin, 0.0, (2.0 * np.pi * a * b * j1(q_safe) / q_safe) ** 2)
        # at k = 0 use a dummy direction (its shape factor is zero) so the acoustic
        # tensor stays invertible
        xi_x = np.where(origin, 1.0, kx / safe)
        xi_y = np.where(origin, 0.0, ky / safe)
        response = _unit_response(stiffness, xi_x + 0.0 * ky, xi_y + 0.0 * kx)
        for row in range(6):
            for col in range(6):
                total[row][col] += float(np.sum(shape2 * response[row][col]))
    scale = 1.0 / (cell_area * region_area)
    return [[value * scale for value in row] for row in total]


@functools.lru_cache(maxsize=64)
def _periodic_items(stiffness_key, a, b, length_x, length_y, n_modes):
    stiffness = np.array(stiffness_key).reshape(3, 3, 3, 3)
    coarse = _lattice_sum(stiffness, a, b, length_x, length_y, n_modes // 2)
    fine = _lattice_sum(stiffness, a, b, length_x, length_y, n_modes)
    # truncation error ~ 1/n_modes: one Richardson step removes the leading term
    return tuple(tuple(2.0 * f - c for f, c in zip(fr, cr)) for fr, cr in zip(fine, coarse))


def periodic_eshelby_tensor(
    stiffness, semi_axis_a, semi_axis_b, length_x=1.0, length_y=1.0, n_modes=512
):
    """Exact periodic Eshelby tensor ``S[i, j, k, l]`` of an ellipse cylinder in a
    `length_x` by `length_y` cell of the anisotropic matrix `stiffness`, by the
    analytic lattice sum (Richardson-extrapolated from ``n_modes/2`` and
    ``n_modes``). No FFT, no grid; approaches :func:`isolated_eshelby_tensor`
    as the area fraction goes to zero."""
    key = tuple(np.asarray(stiffness, dtype=float).ravel().tolist())
    items = _periodic_items(
        key, float(semi_axis_a), float(semi_axis_b), float(length_x), float(length_y),
        int(n_modes),
    )
    return _full_tensor([list(row) for row in items])


def interior_stress(stiffness, eshelby, eigenstrain):
    """Uniform interior stress ``C:(S:eps* - eps*)`` for the Eshelby tensor
    `eshelby` (isolated or periodic)."""
    stiffness = np.asarray(stiffness, dtype=float)
    eigenstrain = np.asarray(eigenstrain, dtype=float)
    strain = np.einsum("ijkl,kl->ij", eshelby, eigenstrain)
    return np.einsum("ijkl,kl->ij", stiffness, strain - eigenstrain)


# ---------------------------------------------------------------------------
# The case
# ---------------------------------------------------------------------------


@dataclass
class AnisotropicInclusionCase:
    """An ellipse (a circle for ``semi_axis_a == semi_axis_b``) with a uniform
    eigenstrain in a homogeneous anisotropic matrix.

    Parameters
    ----------
    grid : Grid
        A 2D grid (``grid.shape[2] == 1``).
    stiffness : array_like
        Rank-4 matrix stiffness, shape ``(3, 3, 3, 3)`` (see
        :func:`cubic_from_zener`, :func:`antiplane_orthotropic_stiffness`,
        :func:`rotate_stiffness`).
    semi_axis_a, semi_axis_b : float
        Semi-axes along the grid's first and second coordinates.
    center : tuple of float, default=(0.5, 0.5)
    """

    grid: Grid
    stiffness: object
    semi_axis_a: float
    semi_axis_b: float
    center: tuple = (0.5, 0.5)

    def __post_init__(self):
        if self.grid.shape[2] != 1:
            raise ValueError("AnisotropicInclusionCase requires a 2D grid (shape[2] == 1)")
        self.stiffness = np.asarray(self.stiffness, dtype=float)

    @property
    def elliptical_radius(self):
        """Dimensionless ``sqrt((x/a)^2 + (y/b)^2)`` about the center, shape
        ``grid.shape`` (exactly 1 on the boundary)."""
        x = self.grid.x[0] - self.center[0]
        y = self.grid.x[1] - self.center[1]
        rho = xp.sqrt((x / self.semi_axis_a) ** 2 + (y / self.semi_axis_b) ** 2)
        return xp.broadcast_to(rho, self.grid.shape)

    def eigenstrain_field(self, eigenstrain_tensor, band_limited=True):
        """Real-space eigenstrain field ``(3, 3) + grid.shape``: the tensor over
        the ellipse, zero elsewhere. With `band_limited` the indicator is
        Lanczos-smoothed (area preserving), which also removes the Nyquist
        content on which the discrete solver operator differs from its
        reference Green operator -- so the numeric solve converges in one
        iteration -- and avoids a pixelated mask's area error."""
        tensor = xp.asarray(eigenstrain_tensor, dtype=self.grid.real_dtype)
        indicator = xp.where(self.elliptical_radius < 1.0, 1.0, 0.0).astype(self.grid.real_dtype)
        field = tensor[:, :, None, None, None] * indicator[None, None, ...]
        if band_limited:
            field = self.grid.ifft(self.grid.fft(field) * self.grid.lanczos_filter)
            field = xp.real(field)
        return field

    def solver(self):
        """The CG solver for this homogeneous anisotropic matrix (its reference
        Green operator is the exact one, so it converges in one iteration for a
        band-limited source)."""
        mean_mu = float(self.stiffness[0, 1, 0, 1])  # placeholder Lame values only
        return ElasticDeformation(
            self.grid, lame_lambda=1.0, lame_mu=mean_mu,
            reference_lame_lambda=1.0, reference_lame_mu=mean_mu,
            stiffness=self.stiffness,
        )

    def isolated_interior_stress(self, eigenstrain_tensor):
        """Interior stress ``(3, 3)`` of the isolated inclusion (no periodic
        images)."""
        s = isolated_eshelby_tensor(self.stiffness, self.semi_axis_a, self.semi_axis_b)
        return interior_stress(self.stiffness, s, eigenstrain_tensor)

    def periodic_interior_stress(self, eigenstrain_tensor, n_modes=512):
        """Exact interior stress ``(3, 3)`` of the periodic array (zero
        macroscopic strain), from the lattice-sum Eshelby tensor."""
        s = periodic_eshelby_tensor(
            self.stiffness, self.semi_axis_a, self.semi_axis_b,
            float(self.grid.lengths[0]), float(self.grid.lengths[1]), n_modes=n_modes,
        )
        return interior_stress(self.stiffness, s, eigenstrain_tensor)
