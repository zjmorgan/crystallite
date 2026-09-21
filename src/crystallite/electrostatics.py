"""Periodic electrostatics (and, by relabeling, magnetostatics) from free
charge and polarization, solved directly in Fourier space."""

from dataclasses import dataclass

from crystallite.backend import xp
from crystallite.spectral.long_range import CoulombOperator
from crystallite.spectral.short_range import DifferentialOperators


@dataclass(frozen=True)
class ElectrostaticSolution:
    """Result of :meth:`Electrostatics.solve`.

    Attributes
    ----------
    potential : ndarray
        Electrostatic potential :math:`\\phi` (zero mean), shape
        ``grid.shape``.
    electric_field : ndarray
        :math:`E=E_\\infty-\\nabla\\phi`, shape ``(3,) + grid.shape``.
    displacement : ndarray
        :math:`D=\\varepsilon E+P`, shape ``(3,) + grid.shape``.
    """

    potential: object
    electric_field: object
    displacement: object


@dataclass(frozen=True)
class Electrostatics:
    r"""Gauss's law in a homogeneous medium with free charge and polarization.

    Solves :math:`\nabla\cdot D=\rho` with :math:`D=\varepsilon E+P` and
    :math:`E=E_\infty-\nabla\phi`, i.e. the Poisson equation

    .. math::

        -\varepsilon\nabla^2\phi=\rho-\nabla\cdot P,
        \qquad \hat\phi=\frac{\hat\rho-i\mathbf{k}\cdot\hat P}
        {\varepsilon k^2},

    so a polarization acts through its bound charge
    :math:`\rho_b=-\nabla\cdot P` (a surface charge :math:`P\cdot n` where it
    ends). Direct, no iteration: the operator is diagonal in Fourier space.

    A periodic cell holds no net charge, so the :math:`k=0` mode of the
    source is dropped, as if a uniform neutralizing background balanced it.
    Give band-limited sources (e.g. Lanczos-smoothed, see
    ``grid.lanczos_filter``): a sharp edge rings at the grid scale.

    **Magnetostatics is the same problem** with :math:`\varepsilon=1`, no free
    charge and the magnetization :math:`M` as `polarization`: :math:`H=-\nabla
    \psi` is `electric_field` and :math:`B/\mu_0=H+M` is `displacement`. It is
    exposed with magnetic vocabulary as
    :class:`crystallite.magnetostatics.Magnetostatics`.

    Parameters
    ----------
    grid : Grid
    permittivity : float, default=1
        Homogeneous :math:`\varepsilon` (vacuum :math:`\varepsilon_0`, or
        1 in nondimensional units). A heterogeneous dielectric is
        :class:`crystallite.conduction.SteadyConduction` with
        ``conductivity=permittivity_field`` and ``source=charge_density``
        (its ``flux`` is :math:`D`).
    """

    grid: object
    permittivity: float = 1.0

    def __post_init__(self):
        permittivity = float(xp.asarray(self.permittivity))
        if permittivity <= 0.0:
            raise ValueError(f"permittivity must be positive, got {permittivity}")
        object.__setattr__(self, "permittivity", permittivity)

    def solve(self, charge_density=None, polarization=None, far_field=None):
        """Solve for the potential and fields of the given sources.

        Parameters
        ----------
        charge_density : array_like, optional
            Free charge :math:`\\rho`, shape ``grid.shape``.
        polarization : array_like, optional
            :math:`P`, shape ``(3,) + grid.shape``.
        far_field : array_like of shape (3,), optional
            Uniform applied field :math:`E_\\infty`, added at the k=0 mode.

        Returns
        -------
        ElectrostaticSolution
        """
        grid = self.grid
        source_hat = xp.zeros(grid.k2.shape, dtype=grid.complex_dtype)
        if charge_density is not None:
            charge_density = xp.asarray(charge_density, dtype=grid.real_dtype)
            if charge_density.shape != grid.shape:
                raise ValueError(
                    f"charge_density must have shape grid.shape={grid.shape}, "
                    f"got {charge_density.shape}"
                )
            source_hat = source_hat - grid.fft(charge_density) / self.permittivity
        polarization_field = xp.zeros((3,) + grid.shape, dtype=grid.real_dtype)
        if polarization is not None:
            polarization_field = xp.asarray(polarization, dtype=grid.real_dtype)
            if polarization_field.shape != (3,) + grid.shape:
                raise ValueError(
                    f"polarization must have shape (3,) + grid.shape={(3,) + grid.shape}, "
                    f"got {polarization_field.shape}"
                )
            divergence = DifferentialOperators(grid).div(grid.fft(polarization_field))
            source_hat = source_hat + divergence / self.permittivity
        if far_field is not None:
            far_field = xp.asarray(far_field, dtype=grid.real_dtype)
            if far_field.shape != (3,):
                raise ValueError(f"far_field must have shape (3,), got {far_field.shape}")

        coulomb = CoulombOperator(grid)
        potential = xp.real(grid.ifft(coulomb.potential(source_hat)))
        field = xp.real(grid.ifft(coulomb.field(source_hat, uniform_field=far_field)))
        return ElectrostaticSolution(
            potential=potential,
            electric_field=field,
            displacement=self.permittivity * field + polarization_field,
        )
