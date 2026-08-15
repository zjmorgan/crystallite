"""Long-range spectral operators (e.g. elastic Green's functions)."""

from crystallite.backend import xp
from crystallite.spectral.short_range import DifferentialOperators


_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
"""Index pairs (i, j) for Voigt components 1..6 (11, 22, 33, 23, 13, 12)."""


def _voigt_stiffness(medium):
    r"""Rank-4 elastic tensor -> (6, 6) Voigt stiffness matrix.

    Direct reindex, no scaling: with the standard engineering-shear
    strain vector (shear components x2) and an unscaled stress vector,

    .. math::

        \sigma_{\mathrm{V}} = C_{\mathrm{V}} \, \varepsilon_{\mathrm{V}}

    holds using `medium`'s raw :math:`C_{ijkl}` components at the
    Voigt-paired indices.
    """
    i = xp.asarray([p[0] for p in _VOIGT_PAIRS])
    j = xp.asarray([p[1] for p in _VOIGT_PAIRS])
    return medium[i[:, None], j[:, None], i[None, :], j[None, :]]


def _voigt_compliance_to_tensor(s_voigt):
    r"""(6, 6) Voigt compliance matrix -> (3, 3, 3, 3) compliance tensor.

    Voigt compliance carries factors of 2 (normal-shear entries) and 4
    (shear-shear entries) relative to the raw tensor components
    :math:`S_{ijkl}` in :math:`\varepsilon_{ij} = S_{ijkl}\sigma_{kl}`
    -- dividing them back out is what makes plain matrix inversion of
    `_voigt_stiffness`'s output equal the correct tensor compliance.
    """
    scale = xp.where(xp.arange(6) < 3, 1.0, 2.0).astype(s_voigt.dtype)
    s_scaled = s_voigt / (scale[:, None] * scale[None, :])

    result = xp.zeros((3, 3, 3, 3), dtype=s_voigt.dtype)
    for row, (i, j) in enumerate(_VOIGT_PAIRS):
        for col, (k, l) in enumerate(_VOIGT_PAIRS):
            value = s_scaled[row, col]
            result[i, j, k, l] = value
            result[j, i, k, l] = value
            result[i, j, l, k] = value
            result[j, i, l, k] = value

    return result


def _add_uniform_at_dc(grid, field, value):
    """Add a uniform value to `field`'s k=0 mode, scaled for the
    unnormalized forward FFT."""
    n_points = 1
    for n in grid.fft_shape:
        n_points *= n

    rank = field.ndim - len(grid.k2.shape)
    dc = (slice(None),) * rank + (0,) * len(grid.k2.shape)
    field[dc] += xp.asarray(value) * n_points
    return field


class CoulombOperator:
    r"""Long-range 1/r interaction: solves the Poisson equation for a
    potential from a Fourier-space source term.

    Caller builds `source`, e.g. :math:`-\rho` for charge density or
    :math:`\nabla \cdot \mathbf{m}` for dipole density.

    Parameters
    ----------
    grid : Grid
    """

    def __init__(self, grid):
        self.grid = grid
        self.ops = DifferentialOperators(grid)

    def potential(self, source):
        r""":math:`\hat\phi = -\hat{f} / k^2`, zero at k=0.

        Parameters
        ----------
        source : ndarray

        Returns
        -------
        ndarray
        """
        return -source * self.grid.inv_k2

    def field(self, source, external=None):
        r""":math:`-\nabla \phi`.

        Parameters
        ----------
        source : ndarray
        external : array_like of shape (3,), optional
            Uniform far field, added at the k=0 mode.

        Returns
        -------
        ndarray
        """
        field = -self.ops.grad(self.potential(source))

        if external is not None:
            field = _add_uniform_at_dc(self.grid, field, external)

        return field


class GreenOperator:
    r"""Long-range Green's operator for a homogeneous reference medium.

    Solves :math:`A(\mathbf{k}) \cdot \hat{u}(\mathbf{k}) =
    \hat{f}(\mathbf{k})`, where :math:`A(\mathbf{k})` contracts the
    medium tensor with the wavevector twice:

    - Conduction: `medium` (3, 3) :math:`\kappa_{ij}`;
      :math:`A(\mathbf{k}) = \kappa_{ij} k_i k_j` (scalar);
      source/output scalar.
    - Elasticity: `medium` (3, 3, 3, 3) :math:`C_{ijkl}`;
      :math:`A_{ik}(\mathbf{k}) = C_{ijkl} k_j k_l` (3, 3);
      source/output vectors, tensor axis leading.

    Typical source is an eigenstrain (see `apply_eigenstrain`), not a
    raw force. k=0 mode is always zero.

    Parameters
    ----------
    grid : Grid
    medium : ndarray
        Shape (3, 3) for conduction, or (3, 3, 3, 3) for elasticity.

    Raises
    ------
    ValueError
        If `medium` is not shape (3, 3) or (3, 3, 3, 3).
    """

    def __init__(self, grid, medium):
        if medium.shape not in ((3, 3), (3, 3, 3, 3)):
            raise ValueError(
                "medium must have shape (3, 3) for conduction or "
                "(3, 3, 3, 3) for elasticity."
            )

        self.grid = grid
        self.medium = medium
        self.ops = DifferentialOperators(grid)

    def _wavevector(self):
        k0, k1, k2 = self.grid.k
        shape = self.grid.k2.shape
        return xp.stack(
            tuple(xp.broadcast_to(k, shape) for k in (k0, k1, k2)),
            axis=0,
        )

    def _acoustic_tensor(self):
        k = self._wavevector()

        if self.medium.ndim == 2:
            return xp.einsum("ij,i...,j...->...", self.medium, k, k)

        return xp.einsum("ijkl,j...,l...->ik...", self.medium, k, k)

    def tensor(self):
        r"""Green's function :math:`G(\mathbf{k}) = A(\mathbf{k})^{-1}`.

        Returns
        -------
        ndarray
            Scalar (conduction) or (3, 3, ...) tensor (elasticity),
            matrix axes leading.
        """
        a = self._acoustic_tensor()
        is_dc = self.grid.k2 == 0

        if self.medium.ndim == 2:
            safe_a = xp.where(is_dc, 1.0, a)
            return xp.where(is_dc, 0.0, 1.0 / safe_a)

        # xp.linalg.inv batches over leading axes, so move the (3, 3)
        # matrix axes from the front (this module's convention) to
        # the back for the inversion, then move them back.
        a = xp.moveaxis(a, (0, 1), (-2, -1))
        is_dc = is_dc[..., None, None]
        safe_a = xp.where(is_dc, xp.eye(3, dtype=a.dtype), a)
        g = xp.where(is_dc, 0.0, xp.linalg.inv(safe_a))
        return xp.moveaxis(g, (-2, -1), (0, 1))

    def apply(self, source):
        r""":math:`G(\mathbf{k}) \cdot \hat{f}(\mathbf{k})`.

        Parameters
        ----------
        source : ndarray
            Scalar (conduction) or vector, tensor axis leading
            (elasticity).

        Returns
        -------
        ndarray
            Potential (conduction) or displacement (elasticity).
        """
        g = self.tensor()

        if self.medium.ndim == 2:
            return g * source

        return xp.einsum("ik...,k...->i...", g, source)

    def apply_eigenstrain(self, eigenstrain):
        r"""Response to an eigenstrain source.

        .. math::

            \sigma^{*} = C : \varepsilon^{*}, \qquad
            \hat{f} = -i\, \mathbf{k} \cdot \sigma^{*}

        then :math:`\mathrm{apply}(\hat{f})`.

        Parameters
        ----------
        eigenstrain : ndarray
            (3, 3) strain, tensor axes leading (elasticity), or (3,)
            eigen-gradient (conduction).

        Returns
        -------
        ndarray
        """
        k = self._wavevector()

        if self.medium.ndim == 2:
            flux = xp.einsum("ij,j...->i...", self.medium, eigenstrain)
            source = -1j * xp.einsum("i...,i...->...", k, flux)
        else:
            stress = xp.einsum(
                "ijkl,kl...->ij...", self.medium, eigenstrain
            )
            source = -1j * xp.einsum("j...,ij...->i...", k, stress)

        return self.apply(source)

    def strain(self, displacement, external=None):
        r""":math:`\varepsilon = \tfrac{1}{2}(\nabla u + (\nabla u)^{\mathsf{T}})`.

        Parameters
        ----------
        displacement : ndarray
            Tensor axis immediately before the spatial axes.
        external : array_like of shape (3, 3), optional
            Uniform far-field strain, added at the k=0 mode.

        Returns
        -------
        ndarray
            (3, 3) strain, tensor axes leading.
        """
        grad_u = self.ops.grad(displacement)
        strain = 0.5 * (grad_u + xp.swapaxes(grad_u, 0, 1))

        if external is not None:
            strain = _add_uniform_at_dc(self.grid, strain, external)

        return strain

    def stress(self, strain, external=None):
        r"""Hooke's law: :math:`\sigma = C : \varepsilon`.

        Parameters
        ----------
        strain : ndarray
            (3, 3), tensor axes leading.
        external : array_like of shape (3, 3), optional
            Uniform far-field stress, added directly at the k=0 mode
            (independent of the medium, unlike `strain`'s far field).

        Returns
        -------
        ndarray

        Raises
        ------
        ValueError
            If the reference medium is not rank-4 (elastic).
        """
        if self.medium.ndim != 4:
            raise ValueError("stress requires a rank-4 (elastic) medium.")

        stress = xp.einsum("ijkl,kl...->ij...", self.medium, strain)

        if external is not None:
            stress = _add_uniform_at_dc(self.grid, stress, external)

        return stress

    def compliance(self):
        r"""Medium's functional inverse (once, not per k-point).

        Resistivity :math:`\kappa^{-1}` (conduction), or compliance
        :math:`S` with :math:`S : C : \varepsilon = \varepsilon` via a
        Voigt-convention 6x6 inverse (elasticity).

        Returns
        -------
        ndarray
            (3, 3) resistivity, or (3, 3, 3, 3) compliance.
        """
        if self.medium.ndim == 2:
            return xp.linalg.inv(self.medium)

        c_voigt = _voigt_stiffness(self.medium)
        s_voigt = xp.linalg.inv(c_voigt)
        return _voigt_compliance_to_tensor(s_voigt)

    def apply_compliance(self, flux):
        r""":math:`\mathrm{compliance}() : \mathrm{flux}`: far-field
        strain/gradient equivalent to a far-field stress/flux.

        Parameters
        ----------
        flux : array_like
            (3,) flux (conduction) or (3, 3) stress (elasticity).

        Returns
        -------
        ndarray
        """
        s = self.compliance()
        flux = xp.asarray(flux)

        if self.medium.ndim == 2:
            return s @ flux

        return xp.einsum("ijkl,kl->ij", s, flux)
