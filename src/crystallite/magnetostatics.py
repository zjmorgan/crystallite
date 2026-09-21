"""Periodic magnetostatics from a magnetization, solved directly in Fourier
space."""

from dataclasses import dataclass

from crystallite.electrostatics import Electrostatics


@dataclass(frozen=True)
class MagnetostaticSolution:
    """Result of :meth:`Magnetostatics.solve`.

    Attributes
    ----------
    potential : ndarray
        Magnetic scalar potential :math:`\\psi` (zero mean), shape
        ``grid.shape``.
    magnetic_field : ndarray
        :math:`H=H_\\infty-\\nabla\\psi`, shape ``(3,) + grid.shape``.
    flux_density : ndarray
        :math:`B=\\mu_0(H+M)`, shape ``(3,) + grid.shape``.
    """

    potential: object
    magnetic_field: object
    flux_density: object


@dataclass(frozen=True)
class Magnetostatics:
    r"""Magnetostatics of a magnetization :math:`M` with no free current.

    Solves :math:`\nabla\cdot B=0`, :math:`\nabla\times H=0` with
    :math:`B=\mu_0(H+M)`, :math:`H=H_\infty-\nabla\psi`, i.e.
    :math:`-\nabla^2\psi=-\nabla\cdot M` with the bound "magnetic charge"
    :math:`\rho_m=-\nabla\cdot M`. This is
    :class:`crystallite.electrostatics.Electrostatics` with
    :math:`\varepsilon=1`, :math:`P\to M`, :math:`E\to H` and
    :math:`D\to B/\mu_0`; this class only fixes that mapping and the
    magnetic vocabulary. The k=0 mode is dropped and sources should be
    band-limited, as there.

    Parameters
    ----------
    grid : Grid
    permeability : float, default=1
        :math:`\mu_0`, only scaling :math:`B` (1 in nondimensional units).
    """

    grid: object
    permeability: float = 1.0

    def __post_init__(self):
        if float(self.permeability) <= 0.0:
            raise ValueError(f"permeability must be positive, got {self.permeability}")

    def solve(self, magnetization, far_field=None):
        """Solve for the potential and fields of `magnetization`.

        Parameters
        ----------
        magnetization : array_like
            :math:`M`, shape ``(3,) + grid.shape``.
        far_field : array_like of shape (3,), optional
            Uniform applied field :math:`H_\\infty`.

        Returns
        -------
        MagnetostaticSolution
        """
        solution = Electrostatics(self.grid, 1.0).solve(
            polarization=magnetization, far_field=far_field
        )
        return MagnetostaticSolution(
            potential=solution.potential,
            magnetic_field=solution.electric_field,
            flux_density=self.permeability * solution.displacement,
        )
