"""Long-range spectral operators (e.g. elastic Green's functions)."""

from crystallite.backend import xp
from crystallite.spectral.short_range import DifferentialOperators


def _mandel_basis(dtype):
    """Orthonormal (Mandel) basis, shape (6, 3, 3), for symmetric
    3x3 tensors."""
    basis = xp.zeros((6, 3, 3), dtype=dtype)
    basis[0, 0, 0] = 1.0
    basis[1, 1, 1] = 1.0
    basis[2, 2, 2] = 1.0

    inv_sqrt2 = 1.0 / xp.sqrt(xp.asarray(2.0, dtype=dtype))
    basis[3, 1, 2] = basis[3, 2, 1] = inv_sqrt2
    basis[4, 0, 2] = basis[4, 2, 0] = inv_sqrt2
    basis[5, 0, 1] = basis[5, 1, 0] = inv_sqrt2

    return basis


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
    """Long-range 1/r interaction: solves the Poisson equation for a
    potential from a Fourier-space source term.

    Caller builds `source`, e.g. ``-rho`` for charge density or
    ``div(m)`` for dipole density.

    Parameters
    ----------
    grid : Grid
    """

    def __init__(self, grid):
        self.grid = grid
        self.ops = DifferentialOperators(grid)

    def potential(self, source):
        """``phi_hat = -source_hat / k^2``, zero at k=0.

        Parameters
        ----------
        source : ndarray

        Returns
        -------
        ndarray
        """
        return -source * self.grid.inv_k2

    def field(self, source, external=None):
        """``-grad(potential)``.

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
    """Long-range Green's operator for a homogeneous reference medium.

    Solves ``A(k) . output(k) = source(k)``, where `A(k)` contracts
    the medium tensor with the wavevector twice:

    - Conduction: `medium` (3, 3) ``kappa_ij``; ``A(k) = kappa_ij
      k_i k_j`` (scalar); source/output scalar.
    - Elasticity: `medium` (3, 3, 3, 3) ``C_ijkl``; ``A_ik(k) =
      C_ijkl k_j k_l`` (3, 3); source/output vectors, tensor axis
      leading.

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
        """Green's function ``G(k) = A(k)^-1``.

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
        """``G(k) . source``.

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
        """Response to an eigenstrain source.

        ``generalized_stress = medium . eigenstrain``, ``source =
        -i k . generalized_stress``, then ``apply(source)``.

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
        """``0.5 * (grad(u) + grad(u)^T)``.

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
        """Hooke's law: ``C : strain``.

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
        """Medium's functional inverse (once, not per k-point).

        Resistivity ``kappa^-1`` (conduction), or compliance `S`
        with ``S : C : epsilon = epsilon`` via a Mandel-basis 6x6
        inverse (elasticity).

        Returns
        -------
        ndarray
            (3, 3) resistivity, or (3, 3, 3, 3) compliance.
        """
        if self.medium.ndim == 2:
            return xp.linalg.inv(self.medium)

        basis = _mandel_basis(self.medium.dtype)
        c_mandel = xp.einsum("aij,ijkl,bkl->ab", basis, self.medium, basis)
        s_mandel = xp.linalg.inv(c_mandel)
        return xp.einsum("ab,aij,bkl->ijkl", s_mandel, basis, basis)

    def apply_compliance(self, flux):
        """``compliance() : flux``: far-field strain/gradient
        equivalent to a far-field stress/flux.

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
