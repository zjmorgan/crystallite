r"""Verification cases for steady conduction: an inclusion (hole, soft, hard or
perfectly conducting, circular or elliptical, isotropic or anisotropic matrix)
under a mean driving force, and a cylinder generating heat uniformly.

Everything is analytic -- no FFT solve, no pixelated mask. With
:math:`q=\kappa X`, :math:`X=-\nabla T`, an inclusion of conductivity
:math:`\kappa_1` in a matrix :math:`\kappa_0` is replaced by the matrix
material carrying an *equivalent* driving force :math:`X^*` (a bookkeeping
device, not a physical stress-free strain): :math:`q=\kappa_0(X-X^*\chi)`.
Conservation gives the response of the matrix to :math:`X^*`, in Fourier space

.. math::

    \delta\hat X_i=\frac{k_i\,(k_j\kappa^0_{jl})}{k\cdot\kappa^0\cdot k}\,
    \hat X^*_l ,

whose average over the (ellipse-shaped) region is the depolarization tensor
:math:`S`, so the uniform interior field :math:`X_\mathrm{in}` solves
:math:`\kappa_1X_\mathrm{in}=\kappa_0(X_\mathrm{in}-X^*)`,
:math:`X_\mathrm{in}=\bar X+S X^*`:

.. math::

    \left[I-S\,(I-\kappa_0^{-1}\kappa_1)\right]X_\mathrm{in}=\bar X .

* the *isolated* :math:`S` is Mura's angular average (a smooth periodic
  integrand, so a uniform-angle sum is spectrally accurate);
* the *periodic* :math:`S` is the analytic lattice sum with the exact Bessel
  transform of the ellipse, the conduction counterpart of
  :mod:`crystallite.verification.anisotropic_inclusion`.

Mathematically this is the antiplane (out-of-plane) part of the elastic Eshelby
solution, under :math:`\kappa_{il}=C_{i3l3}`, :math:`X^*_i=2\varepsilon^*_{i3}`,
:math:`q_i=\sigma_{i3}`, :math:`S_{il}=2S_{i3l3}` (:math:`i,l` in-plane;
:math:`\kappa=\mu` isotropic, :math:`\kappa_{11}=c_{55}` and
:math:`\kappa_{22}=c_{44}` orthotropic). These functions are written
independently of the elastic ones, and ``tests/test_conduction_eshelby_mapping.py``
checks that they coincide: isolated and periodic depolarization tensors
(isotropic, orthotropic and rotated matrices), circular and elliptical interiors
(including the perfect-conductor limit) and the exterior fields.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
from scipy.special import j1

from crystallite.backend import xp
from crystallite.conduction import SteadyConduction
from crystallite.grid import Grid
from crystallite.verification.elastic_deformation import _ellipse_fourier_transform

_NUMERIC_CONTRAST_CAP = 1.0e3


def _tensor(conductivity):
    conductivity = np.asarray(conductivity, dtype=float)
    return conductivity * np.eye(3) if conductivity.ndim == 0 else conductivity


# ---------------------------------------------------------------------------
# Isolated circular inhomogeneity (closed form)
# ---------------------------------------------------------------------------


def circular_inhomogeneity_driving_force(x, y, radius, contrast, driving_force):
    r"""Driving force :math:`X=-\nabla T` of an isolated circular inclusion
    (conductivity ratio `contrast`, ``float('inf')`` for a perfect conductor)
    in an isotropic plate under a far-field `driving_force` ``(X1, X2)``.

    With :math:`c=(1-\beta)/(1+\beta)`, the exterior potential is
    :math:`T=-\bar X\cdot x\,(1+c\,a^2/r^2)` and the interior driving force is
    uniform, :math:`2\bar X/(1+\beta)`; both conditions of continuity (of
    :math:`T` and of :math:`\kappa\,\partial_rT`) hold at ``r = a``.

    Returns
    -------
    tuple of ndarray
        ``(X1, X2)`` at the points `x`, `y` (relative to the center).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x1, x2 = float(driving_force[0]), float(driving_force[1])
    c = -1.0 if contrast == float("inf") else (1.0 - contrast) / (1.0 + contrast)
    inner = 0.0 if contrast == float("inf") else 2.0 / (1.0 + contrast)
    r2 = x**2 + y**2
    outside = r2 >= radius**2
    r2 = np.where(outside, r2, 1.0)
    p1, p2 = c * radius**2 * x1, c * radius**2 * x2
    dot = p1 * x + p2 * y
    ext1 = x1 + p1 / r2 - 2.0 * dot * x / r2**2
    ext2 = x2 + p2 / r2 - 2.0 * dot * y / r2**2
    return np.where(outside, ext1, inner * x1), np.where(outside, ext2, inner * x2)


def elliptical_perturbation(x, y, semi_axis_a, semi_axis_b, contrast, exciting_field):
    r"""Exterior perturbation :math:`\delta X=X-X_\mathrm{exc}` of an isolated
    ellipse cylinder in an isotropic plate, for the exciting uniform field
    `exciting_field` ``(X1, X2)``, at points `x`, `y` relative to the center.

    In elliptic coordinates :math:`z=c\cosh\zeta`, :math:`c^2=a^2-b^2`
    (:math:`a>b`, the boundary at :math:`\tanh\xi_0=b/a`) the exterior
    potential is :math:`T=-c\,\mathrm{Re}[X_1(\cosh\zeta+\alpha e^{-\zeta})
    -iX_2(\cosh\zeta-\gamma e^{-\zeta})]`; continuity of :math:`T` and of
    :math:`\kappa\,\partial_\xi T` at :math:`\xi_0` gives
    :math:`\alpha e^{-\xi_0}=\sinh\xi_0(1-\beta)/(1+\beta\tanh\xi_0)` and
    :math:`\gamma e^{-\xi_0}=\cosh\xi_0(1-\beta)/(1+\beta\coth\xi_0)`
    (their :math:`\beta\to\infty` limits, :math:`-\cosh\xi_0` and
    :math:`-\sinh\xi_0`, for a perfect conductor), and

    .. math::

        \delta X_x - i\,\delta X_y=-\frac{e^{-\zeta}\,(\alpha X_1 +
        i\gamma X_2)}{\sinh\zeta}.

    A thin ellipse (:math:`b\ll a`) is a crack. Requires ``a != b``; a circle
    is :func:`circular_inhomogeneity_driving_force`.

    Returns
    -------
    tuple of ndarray
        ``(dX1, dX2)``. Only meaningful outside the ellipse.
    """
    a, b = float(semi_axis_a), float(semi_axis_b)
    x1, x2 = float(exciting_field[0]), float(exciting_field[1])
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    swap = a < b
    if swap:
        a, b, x, y, x1, x2 = b, a, y, x, x2, x1
    c = np.sqrt(a * a - b * b)
    xi0 = np.arctanh(b / a)
    tanh0, coth0 = b / a, a / b
    sinh0, cosh0 = np.sinh(xi0), np.cosh(xi0)
    if contrast == float("inf"):
        u, v = -cosh0, -sinh0
    else:
        u = sinh0 * (1.0 - contrast) / (1.0 + contrast * tanh0)
        v = cosh0 * (1.0 - contrast) / (1.0 + contrast * coth0)
    alpha, gamma = u * np.exp(xi0), v * np.exp(xi0)
    zeta = np.arccosh((x + 1j * y) / c + 0j)
    sinh_zeta = np.sinh(zeta)
    sinh_zeta = np.where(np.abs(sinh_zeta) < 1e-12, 1.0, sinh_zeta)
    w = -np.exp(-zeta) * (alpha * x1 + 1j * gamma * x2) / sinh_zeta
    d1, d2 = np.real(w), -np.imag(w)
    return (d2, d1) if swap else (d1, d2)


# ---------------------------------------------------------------------------
# Depolarization tensors
# ---------------------------------------------------------------------------


def isolated_depolarization_tensor(conductivity, semi_axis_a, semi_axis_b, n_theta=4096):
    r"""Depolarization tensor :math:`S_{il}` of an isolated ellipse cylinder
    (semi-axes along :math:`x_1`, :math:`x_2`) in the matrix `conductivity`
    -- Mura's angular average
    :math:`\frac1{2\pi}\int\frac{\xi_i(\kappa\xi)_l}{\xi\cdot\kappa\cdot\xi}
    \frac{ab}{a^2\cos^2\theta+b^2\sin^2\theta}d\theta`, :math:`\xi` the
    in-plane unit direction. Shape ``(3, 3)``, row 3 zero (no
    :math:`x_3` variation). Isotropic matrix: ``diag(b, a)/(a+b)``."""
    kappa = _tensor(conductivity)
    theta = 2.0 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    a, b = semi_axis_a, semi_axis_b
    weight = a * b / (a**2 * np.cos(theta) ** 2 + b**2 * np.sin(theta) ** 2)
    xi = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    k_xi = kappa @ xi
    denominator = np.sum(xi * k_xi, axis=0)
    s = np.zeros((3, 3))
    for i in range(2):
        for l in range(3):
            s[i, l] = np.mean(weight * xi[i] * k_xi[l] / denominator)
    return s


def _lattice_sum(kappa, a, b, length_x, length_y, n_modes, chunk=64):
    cell_area, region_area = length_x * length_y, np.pi * a * b
    ky = (2.0 * np.pi * np.arange(-n_modes, n_modes + 1) / length_y)[None, :]
    total = np.zeros((3, 3))
    for start in range(-n_modes, n_modes + 1, chunk):
        m = np.arange(start, min(start + chunk, n_modes + 1))
        kx = (2.0 * np.pi * m / length_x)[:, None]
        origin = (kx == 0) & (ky == 0)
        q = np.sqrt((a * kx) ** 2 + (b * ky) ** 2)
        q_safe = np.where(q == 0, 1.0, q)
        shape2 = np.where(origin, 0.0, (2.0 * np.pi * a * b * j1(q_safe) / q_safe) ** 2)
        k = (kx + 0.0 * ky, ky + 0.0 * kx)
        k_k = [kappa[l, 0] * k[0] + kappa[l, 1] * k[1] for l in range(3)]
        denominator = np.where(origin, 1.0, k[0] * k_k[0] + k[1] * k_k[1])
        for i in range(2):
            for l in range(3):
                total[i, l] += np.sum(shape2 * k[i] * k_k[l] / denominator)
    return total / (cell_area * region_area)


@functools.lru_cache(maxsize=64)
def _periodic_items(kappa_key, a, b, length_x, length_y, n_modes):
    kappa = np.array(kappa_key).reshape(3, 3)
    coarse = _lattice_sum(kappa, a, b, length_x, length_y, n_modes // 2)
    fine = _lattice_sum(kappa, a, b, length_x, length_y, n_modes)
    # truncation error ~ 1/n_modes: one Richardson step removes the leading term
    return tuple(map(tuple, 2.0 * fine - coarse))


def periodic_depolarization_tensor(
    conductivity, semi_axis_a, semi_axis_b, length_x=1.0, length_y=1.0, n_modes=512
):
    """Exact periodic depolarization tensor of an ellipse cylinder in a
    `length_x` by `length_y` cell, by the analytic lattice sum
    (Richardson-extrapolated from ``n_modes/2`` and ``n_modes``). No grid.
    Approaches :func:`isolated_depolarization_tensor` as the area fraction
    goes to zero. Shape ``(3, 3)``."""
    key = tuple(_tensor(conductivity).ravel().tolist())
    items = _periodic_items(
        key, float(semi_axis_a), float(semi_axis_b), float(length_x), float(length_y),
        int(n_modes),
    )
    return np.array(items)


def interior_driving_force(contrast, depolarization, macro_field):
    r"""Uniform interior driving force :math:`X_\mathrm{in}` of an inclusion
    :math:`\kappa_1=` `contrast` :math:`\times\,\kappa_0` (a perfect
    conductor for ``float('inf')``, where :math:`X_\mathrm{in}=0` exactly)
    given the depolarization tensor and the mean driving force."""
    if contrast == float("inf"):
        return np.zeros(3)
    system = np.eye(3) - (1.0 - contrast) * depolarization  # I - S (I - kappa_0^-1 kappa_1)
    return np.linalg.solve(system, np.asarray(macro_field, dtype=float))


def equivalent_driving_force(contrast, depolarization, macro_field, interior):
    r"""Equivalent driving force :math:`X^*=(I-\kappa_0^{-1}\kappa_1)X_\mathrm{in}`
    (for a perfect conductor, from :math:`\bar X+SX^*=0` on the in-plane
    block, with :math:`X^*_3=0`)."""
    if contrast != float("inf"):
        return (1.0 - contrast) * np.asarray(interior)
    star = np.zeros(3)
    star[:2] = np.linalg.solve(depolarization[:2, :2], -np.asarray(macro_field, dtype=float)[:2])
    return star


# ---------------------------------------------------------------------------
# The inclusion case
# ---------------------------------------------------------------------------


@dataclass
class ConductionInclusionCase:
    """An ellipse (a circle for ``semi_axis_a == semi_axis_b``) of
    conductivity `contrast` times the matrix's, in a homogeneous matrix.

    Parameters
    ----------
    grid : Grid
        A 2D grid (``grid.shape[2] == 1``).
    matrix_conductivity : float or array_like of shape (3, 3)
    semi_axis_a, semi_axis_b : float
        Semi-axes along the grid's first and second coordinates.
    contrast : float, default=0.0
        Inclusion-to-matrix conductivity ratio: 0 is a hole (insulator),
        below 1 soft, above 1 hard, ``float('inf')`` perfectly conducting.
        Use a small positive value (e.g. 1e-3) for the numeric solve of a
        hole. The numeric field caps ``inf`` at 1e3: float32 arithmetic
        develops interior noise beyond that.
    center : tuple of float, default=(0.5, 0.5)
    smoothing_width : float, optional
        Numeric interface width in grid spacings. Default ``None`` is the
        Lanczos-smoothed indicator; a value gives a ``tanh`` profile in the
        distance to the boundary instead. A high-contrast (hard or perfectly
        conducting) inclusion needs a *narrow* one: an arithmetic-mean
        interface lets the conductor's effective radius grow by roughly the
        width, and the exterior dipole with its square.
    """

    grid: Grid
    matrix_conductivity: object
    semi_axis_a: float
    semi_axis_b: float
    contrast: float = 0.0
    center: tuple = (0.5, 0.5)
    smoothing_width: object = None

    def __post_init__(self):
        if self.grid.shape[2] != 1:
            raise ValueError("ConductionInclusionCase requires a 2D grid (shape[2] == 1)")
        self.matrix_conductivity = np.asarray(self.matrix_conductivity, dtype=float)

    @property
    def isotropic_matrix(self):
        return self.matrix_conductivity.ndim == 0

    @property
    def elliptical_radius(self):
        """Dimensionless ``sqrt((x/a)^2 + (y/b)^2)`` about the center, shape
        ``grid.shape`` (exactly 1 on the boundary)."""
        x = self.grid.x[0] - self.center[0]
        y = self.grid.x[1] - self.center[1]
        rho = xp.sqrt((x / self.semi_axis_a) ** 2 + (y / self.semi_axis_b) ** 2)
        return xp.broadcast_to(rho, self.grid.shape)

    def conductivity_field(self, band_limited=True):
        """Real-space conductivity: a scalar field (isotropic matrix) or a
        ``(3, 3) + grid.shape`` tensor field. With `band_limited` the
        inclusion indicator is Lanczos-smoothed (area preserving), which
        removes the Nyquist content on which the discrete operator differs
        from its reference Green operator."""
        if self.smoothing_width is not None:
            # distance to the boundary ~ (rho - 1) * sqrt(a b) for a near-circle
            scale = (self.semi_axis_a * self.semi_axis_b) ** 0.5
            width = self.smoothing_width * min(self.grid.spacing[:2])
            indicator = 0.5 * (1.0 - xp.tanh((self.elliptical_radius - 1.0) * scale / width))
            indicator = indicator.astype(self.grid.real_dtype)
        else:
            indicator = xp.where(self.elliptical_radius < 1.0, 1.0, 0.0).astype(self.grid.real_dtype)
        if band_limited and self.smoothing_width is None:
            indicator = xp.real(self.grid.ifft(self.grid.fft(indicator) * self.grid.lanczos_filter))
            # Lanczos ringing overshoots [0, 1], which would push a hole's
            # conductivity negative and break positive-definiteness
            indicator = xp.clip(indicator, 0.0, 1.0)
        contrast = min(self.contrast, _NUMERIC_CONTRAST_CAP)
        scale = 1.0 + (contrast - 1.0) * indicator
        matrix = xp.asarray(self.matrix_conductivity, dtype=self.grid.real_dtype)
        if self.isotropic_matrix:
            return matrix * scale
        return matrix[:, :, None, None, None] * scale[None, None, ...]

    def solver(self, **kwargs):
        """The CG solver for this case, referenced to the matrix."""
        return SteadyConduction(
            self.grid, self.conductivity_field(), reference_conductivity=self.matrix_conductivity,
            **kwargs,
        )

    def isolated_depolarization_tensor(self):
        return isolated_depolarization_tensor(
            self.matrix_conductivity, self.semi_axis_a, self.semi_axis_b
        )

    def periodic_depolarization_tensor(self, n_modes=512):
        return periodic_depolarization_tensor(
            self.matrix_conductivity, self.semi_axis_a, self.semi_axis_b,
            float(self.grid.lengths[0]), float(self.grid.lengths[1]), n_modes=n_modes,
        )

    def isolated_interior_driving_force(self, macro_field):
        """Interior driving force ``(3,)`` of the isolated inclusion under the
        far-field `macro_field`."""
        return interior_driving_force(
            self.contrast, self.isolated_depolarization_tensor(), macro_field
        )

    def periodic_interior_driving_force(self, macro_field, n_modes=512):
        """Exact interior driving force ``(3,)`` of the periodic array under
        the mean driving force `macro_field`."""
        return interior_driving_force(
            self.contrast, self.periodic_depolarization_tensor(n_modes), macro_field
        )

    def periodic_interior_flux(self, macro_field, n_modes=512):
        """Interior flux ``(3,)``, ``kappa_1 X_in``. For a perfect conductor
        this is the finite limit ``kappa_1 -> inf``, ``X_in -> 0`` with
        ``kappa_1 X_in -> -kappa_0 X*``, uniform inside (for a circle,
        ``2 kappa_0 X / (1 - f)``, exactly 2 for an isolated cylinder)."""
        s = self.periodic_depolarization_tensor(n_modes)
        macro = np.asarray(macro_field, dtype=float)
        interior = interior_driving_force(self.contrast, s, macro)
        if self.contrast == float("inf"):
            star = equivalent_driving_force(self.contrast, s, macro, interior)
            return -_tensor(self.matrix_conductivity) @ star
        return self.contrast * _tensor(self.matrix_conductivity) @ interior

    def periodic_exterior_field(self, macro_field, n_images=8, n_modes=512):
        """Driving force ``(3,) + grid.shape`` of the periodic array of
        inclusions in an isotropic matrix: the exact uniform interior, and the
        exterior from images of the isolated perturbation field (each image
        excited by the local field :math:`X_\\mathrm{in}-S X^*` its periodic
        environment gives it), recentered so the domain mean is exactly
        `macro_field` (the image sum of a :math:`r^{-2}` field is only
        conditionally convergent, so the mean is what fixes its constant).
        A thin ellipse is a crack."""
        if not self.isotropic_matrix:
            raise ValueError("the image sum needs an isotropic matrix")
        macro = np.asarray(macro_field, dtype=float)
        s = self.periodic_depolarization_tensor(n_modes)
        interior = interior_driving_force(self.contrast, s, macro)
        star = equivalent_driving_force(self.contrast, s, macro, interior)
        circle = self.semi_axis_a == self.semi_axis_b
        if circle:
            dipole = 0.5 * self.semi_axis_a**2 * star[:2]
        else:
            isolated = self.isolated_depolarization_tensor()
            exciting = (interior - isolated @ star)[:2]

        x = np.asarray(self.grid.x[0], dtype=float) - self.center[0]
        y = np.asarray(self.grid.x[1], dtype=float) - self.center[1]
        x, y = np.broadcast_to(x, self.grid.shape), np.broadcast_to(y, self.grid.shape)
        length_x, length_y = self.grid.lengths[:2]
        raw = np.zeros((2,) + self.grid.shape)
        for n1 in range(-n_images, n_images + 1):
            for n2 in range(-n_images, n_images + 1):
                dx, dy = x - n1 * length_x, y - n2 * length_y
                if circle:
                    r2 = np.where((dx == 0) & (dy == 0), 1.0, dx**2 + dy**2)
                    dot = dipole[0] * dx + dipole[1] * dy
                    raw[0] += dipole[0] / r2 - 2.0 * dot * dx / r2**2
                    raw[1] += dipole[1] / r2 - 2.0 * dot * dy / r2**2
                else:
                    d1, d2 = elliptical_perturbation(
                        dx, dy, self.semi_axis_a, self.semi_axis_b, self.contrast, exciting
                    )
                    raw[0] += d1
                    raw[1] += d2

        outside = np.asarray(self.elliptical_radius) >= 1.0
        n_outside, n_inside = outside.sum(), (~outside).sum()
        field = np.empty((3,) + self.grid.shape)
        for i in range(2):
            background = (
                outside.size * macro[i] - n_inside * interior[i] - raw[i][outside].sum()
            ) / n_outside
            field[i] = np.where(outside, background + raw[i], interior[i])
        field[2] = macro[2]
        return field


    def periodic_flux_field(self, macro_field, n_images=8, n_modes=512):
        """Flux ``(3,) + grid.shape`` of the periodic array in an isotropic
        matrix: ``kappa_0 X`` outside and the uniform interior flux
        (:meth:`periodic_interior_flux`) inside the inclusion."""
        driving = self.periodic_exterior_field(macro_field, n_images, n_modes)
        inside = (np.asarray(self.elliptical_radius) < 1.0)[None]
        interior = self.periodic_interior_flux(macro_field, n_modes)[:, None, None, None]
        return np.where(inside, interior, float(self.matrix_conductivity) * driving)


# ---------------------------------------------------------------------------
# Uniform heat generation in a cylinder
# ---------------------------------------------------------------------------


def disk_source_flux(grid, radius, generation, center=(0.5, 0.5), conductivity=1.0, refine=4):
    r"""Heat flux ``(3,) + grid.shape`` of a periodic array of cylinders each
    generating heat uniformly (`generation`, per unit area) in a homogeneous
    matrix, with the uniform sink that makes a periodic steady state exist
    (:math:`k=0` dropped).

    Exact in Fourier space, :math:`\hat q=-\kappa\,i k\,\hat\varphi/
    (k\cdot\kappa\cdot k)`, with the analytic Bessel transform of the disk
    (no pixelated mask) synthesized on a grid `refine` times finer and sampled
    at the points of `grid`, so its Gibbs ringing sits far from the sampled
    field. Isolated, the flux is radial, :math:`\varphi r/2` inside and
    :math:`\varphi a^2/(2r)` outside.

    Parameters
    ----------
    conductivity : float or array_like of shape (3, 3), default=1
    refine : int, default=4
    """
    fine = Grid(
        shape=(grid.shape[0] * refine, grid.shape[1] * refine, 1), lengths=grid.lengths
    )
    kappa = _tensor(conductivity)
    k = [np.asarray(kk, dtype=float) for kk in fine.k]
    k_kappa = [sum(kappa[l, j] * k[j] for j in range(3)) for l in range(3)]
    denominator = sum(k[i] * k_kappa[i] for i in range(3))
    safe = np.where(denominator == 0, 1.0, denominator)
    source_hat = generation * np.asarray(
        _ellipse_fourier_transform(fine, radius, radius, center=(center[0], center[1], 0.0))
    )
    flux = np.empty((3,) + grid.shape)
    for l in range(3):
        q_hat = np.where(denominator == 0, 0.0, -1j * k_kappa[l] * source_hat / safe)
        flux[l] = np.real(np.asarray(fine.ifft(q_hat)))[::refine, ::refine, :]
    return flux
