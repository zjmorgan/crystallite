"""Long-range spectral operators (e.g. elastic Green's functions)."""

from crystallite.backend import xp
from crystallite.material.properties import (
    voigt_compliance_to_tensor,
    voigt_stiffness,
)
from crystallite.spectral.short_range import DifferentialOperators


def _add_uniform_constant(grid, field, value):
    """Add a uniform value to `field`'s k=0 mode, scaled for the
    unnormalized forward FFT."""
    n_points = 1
    for n in grid.fft_shape:
        n_points *= n

    rank = field.ndim - len(grid.k2.shape)
    dc = (slice(None),) * rank + (0,) * len(grid.k2.shape)
    field[dc] += xp.asarray(value) * n_points
    return field


def _add_uniform_gradient(grid, field, gradient):
    r"""Add a spatially-linear term with a constant gradient, built
    directly in reciprocal space -- no real-space ramp is ever formed.

    Adds :math:`\sum_j \mathrm{gradient}[\ldots, j] \, x_j` to `field`.
    A periodic ramp along one axis has Fourier content only on the
    pencil of modes where the other two axes are at k=0 (by
    separability); away from k=0 on that pencil,

    .. math::

        \mathcal{F}[x_j]_{k_j} = \frac{-L_j N_{\mathrm{other}}}
        {1 - e^{-i k_j \Delta x_j}}

    (:math:`N_{\mathrm{other}}` the product of the other two axes'
    point counts, matching the unnormalized forward FFT), and the k=0
    mode gets :math:`\langle x_j \rangle` -- `grid.x`'s actual mean,
    since `grid.x` runs ``0, dx, ..., (n-1)*dx`` rather than a
    centered range.

    Parameters
    ----------
    grid : Grid
    field : ndarray
        Modified in place.
    gradient : array_like
        Shape ``field``'s tensor shape + ``(3,)``: `gradient[..., j]`
        is the tensor to add times :math:`x_j`, for each of the 3
        spatial directions j.

    Returns
    -------
    ndarray
    """
    gradient = xp.asarray(gradient)
    spatial_ndim = len(grid.k2.shape)
    dc = (slice(None),) * (field.ndim - spatial_ndim) + (0,) * spatial_ndim

    n_points = 1
    for n in grid.fft_shape:
        n_points *= n

    dc_term = xp.zeros(gradient.shape[:-1], dtype=field.dtype)

    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        k_axis = xp.broadcast_to(grid.k[axis], grid.k2.shape)
        k_other0 = xp.broadcast_to(grid.k[other[0]], grid.k2.shape)
        k_other1 = xp.broadcast_to(grid.k[other[1]], grid.k2.shape)

        is_dc_axis = k_axis == 0
        on_pencil = (k_other0 == 0) & (k_other1 == 0) & ~is_dc_axis

        length = grid.fft_shape[axis] * grid.spacing[axis]
        n_other = grid.fft_shape[other[0]] * grid.fft_shape[other[1]]

        denom = xp.where(is_dc_axis, 1.0, 1 - xp.exp(-1j * k_axis * grid.spacing[axis]))
        ramp = xp.where(on_pencil, -length * n_other / denom, 0.0)
        ramp = ramp.astype(field.dtype)

        term = gradient[..., axis]
        field += term.reshape(term.shape + (1,) * spatial_ndim) * ramp

        x_mean = xp.asarray(grid.x[axis]).mean()
        dc_term = dc_term + gradient[..., axis] * x_mean

    field[dc] += dc_term * n_points
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

    def field(self, source, uniform_field=None, uniform_field_gradient=None):
        r""":math:`-\nabla \phi`.

        Parameters
        ----------
        source : ndarray
        uniform_field : array_like of shape (3,), optional
            Uniform far field, added at the k=0 mode.
        uniform_field_gradient : array_like of shape (3, 3), optional
            Constant far-field gradient (component, direction), added
            directly in reciprocal space -- see `_add_uniform_gradient`.

        Returns
        -------
        ndarray
        """
        field = -self.ops.grad(self.potential(source))

        if uniform_field is not None:
            field = _add_uniform_constant(self.grid, field, uniform_field)

        if uniform_field_gradient is not None:
            field = _add_uniform_gradient(
                self.grid, field, uniform_field_gradient
            )

        return field


class GreenOperator:
    r"""Long-range Green's operator for a fixed, homogeneous reference
    medium.

    Solves :math:`A(\mathbf{k}) \cdot \hat{u}(\mathbf{k}) =
    \hat{f}(\mathbf{k})`, :math:`A(\mathbf{k})` the medium contracted
    with the wavevector twice:

    - Conduction: `medium` (3, 3) :math:`\kappa_{ij}`;
      :math:`A(\mathbf{k}) = \kappa_{ij} k_i k_j` (scalar).
    - Deformation: `medium` (3, 3, 3, 3) :math:`C_{ijkl}`;
      :math:`A_{ik}(\mathbf{k}) = C_{ijkl} k_j k_l` (3, 3), vectors
      with the tensor axis leading.

    Parameters
    ----------
    grid : Grid
    medium : ndarray
        Shape (3, 3) for conduction, or (3, 3, 3, 3) for deformation.

    Raises
    ------
    ValueError
        If `medium` is not shape (3, 3) or (3, 3, 3, 3).
    """

    def __init__(self, grid, medium):
        if medium.shape not in ((3, 3), (3, 3, 3, 3)):
            raise ValueError(
                "medium must have shape (3, 3) for conduction or "
                "(3, 3, 3, 3) for deformation."
            )

        self.grid = grid
        self.medium = medium
        self.ops = DifferentialOperators(grid)

        self._tensor = self._compute_tensor()
        self._compliance = self._compute_compliance()

    def _compute_tensor(self):
        k = xp.stack(
            tuple(
                xp.broadcast_to(k, self.grid.k2.shape) for k in self.grid.k
            ),
            axis=0,
        )
        is_dc = self.grid.k2 == 0

        if self.medium.ndim == 2:
            a = xp.einsum("ij,i...,j...->...", self.medium, k, k)
            safe_a = xp.where(is_dc, 1.0, a)
            return xp.where(is_dc, 0.0, 1.0 / safe_a)

        a = xp.einsum("ijkl,j...,l...->ik...", self.medium, k, k)
        # xp.linalg.inv batches over leading axes, so move the (3, 3)
        # matrix axes from the front (this module's convention) to
        # the back for the inversion, then move them back.
        a = xp.moveaxis(a, (0, 1), (-2, -1))
        is_dc = is_dc[..., None, None]
        safe_a = xp.where(is_dc, xp.eye(3, dtype=a.dtype), a)
        g = xp.where(is_dc, 0.0, xp.linalg.inv(safe_a))
        return xp.moveaxis(g, (-2, -1), (0, 1))

    def _compute_compliance(self):
        if self.medium.ndim == 2:
            return xp.linalg.inv(self.medium)

        c_voigt = voigt_stiffness(self.medium)
        s_voigt = xp.linalg.inv(c_voigt)
        return voigt_compliance_to_tensor(s_voigt)

    def tensor(self):
        r"""Green's function :math:`G(\mathbf{k}) = A(\mathbf{k})^{-1}`.

        Returns
        -------
        ndarray
            Scalar (conduction) or (3, 3, ...) tensor (deformation),
            matrix axes leading.
        """
        return self._tensor

    def potential(self, source):
        r""":math:`G(\mathbf{k}) \cdot \hat{f}(\mathbf{k})`.

        Parameters
        ----------
        source : ndarray
            Scalar (conduction) or vector, tensor axis leading
            (deformation).

        Returns
        -------
        ndarray
            Potential (conduction) or displacement (deformation).
        """
        g = self.tensor()

        if self.medium.ndim == 2:
            return g * source

        return xp.einsum("ik...,k...->i...", g, source)

    def field(
        self, source, uniform_field=None, uniform_field_gradient=None
    ):
        r""":math:`\varepsilon = \tfrac{1}{2}(\nabla u + (\nabla
        u)^{\mathsf{T}})` of :math:`u = \mathrm{potential}(source)`,
        parallel to `CoulombOperator.potential`/`field`.

        Parameters
        ----------
        source : ndarray
            Tensor axis immediately before the spatial axes.
        uniform_field : array_like of shape (3, 3), optional
            Uniform far-field strain, added at the k=0 mode.
        uniform_field_gradient : array_like of shape (3, 3, 3), optional
            Constant far-field strain gradient (i, j, direction), added
            directly in reciprocal space -- see `_add_uniform_gradient`.

        Returns
        -------
        ndarray
            (3, 3) strain, tensor axes leading.
        """
        grad_u = self.ops.grad(self.potential(source))
        field = 0.5 * (grad_u + xp.swapaxes(grad_u, 0, 1))

        if uniform_field is not None:
            field = _add_uniform_constant(self.grid, field, uniform_field)

        if uniform_field_gradient is not None:
            field = _add_uniform_gradient(
                self.grid, field, uniform_field_gradient
            )

        return field

    def compliance(self):
        r"""Medium's functional inverse: resistivity :math:`\kappa^{-1}`
        (conduction), or compliance :math:`S^{-1}` (deformation).

        Returns
        -------
        ndarray
            (3, 3) resistivity, or (3, 3, 3, 3) compliance.
        """
        return self._compliance

    def apply_compliance(self, flux):
        r"""Strain/field for a far-field stress/flux, based on the medium.

        Broadcasts over extra trailing axes on `flux` (e.g. a
        direction axis for a stress/flux *gradient*).

        Parameters
        ----------
        flux : array_like
            (3,) flux (conduction) or (3, 3) stress (deformation), with
            optional extra trailing axes.

        Returns
        -------
        ndarray
        """
        s = self.compliance()
        flux = xp.asarray(flux)

        if self.medium.ndim == 2:
            return s @ flux

        return xp.einsum("ijkl,kl...->ij...", s, flux)
