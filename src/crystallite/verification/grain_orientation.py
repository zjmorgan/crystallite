"""Verification cases for grain-orientation (Allen-Cahn) dynamics."""

from dataclasses import dataclass

from crystallite.backend import xp
from crystallite.grid import Grid
from crystallite.mass_diffusion import (
    cahn_hilliard_periodic_profile,
    double_well_equilibrium_width,
)


@dataclass(frozen=True)
class PlanarGrainBoundaryCase:
    r"""A stationary planar boundary between two grain-orientation states.

    One order parameter runs from :math:`-\eta_0` to :math:`+\eta_0` and
    back across a periodic domain (a kink-antikink pair, exactly
    :func:`crystallite.mass_diffusion.cahn_hilliard_periodic_profile`),
    the rest held at zero. The single-order-parameter homogeneous term
    :math:`-\tfrac{a}{2}\eta^2+\tfrac{b}{4}\eta^4` is algebraically
    :func:`crystallite.mass_diffusion.double_well_derivative` with
    ``a_param = b/4``, ``c_alpha = -eta0``, ``c_beta = eta0``,
    :math:`\eta_0=\sqrt{a/b}` -- so the exact equilibrium profile and its
    width reuse that existing machinery directly, with no new interface
    math needed.

    Parameters
    ----------
    grid : Grid
        A grid with one active (1D) axis, matching
        :func:`crystallite.mass_diffusion.cahn_hilliard_periodic_profile`'s
        own requirement.
    barrier_coefficient : float, default=1.0
        Coefficient ``a``.
    quartic_coefficient : float, default=1.0
        Coefficient ``b``.
    gradient_energy : float, default=1.0
        Gradient-energy coefficient ``kappa`` for the active orientation.
    n_orientations : int, default=2
        Total number of order parameters in the field returned by
        :meth:`profile` (only one is active; the rest are held at zero).
    active_orientation : int, default=0
        Index of the active order parameter.
    center : float, default=0.0
        Position of the kink; the antikink sits half a period away.
    """

    grid: Grid
    barrier_coefficient: float = 1.0
    quartic_coefficient: float = 1.0
    gradient_energy: float = 1.0
    n_orientations: int = 2
    active_orientation: int = 0
    center: float = 0.0

    @property
    def eta0(self):
        """Equilibrium single-grain magnitude, ``sqrt(a / b)``."""
        return xp.sqrt(self.barrier_coefficient / self.quartic_coefficient)

    @property
    def _a_param(self):
        return self.quartic_coefficient / 4.0

    @property
    def width(self):
        """Analytic equilibrium interface width."""
        return double_well_equilibrium_width(
            self._a_param, -self.eta0, self.eta0, self.gradient_energy
        )

    def profile(self):
        """Return the exact equilibrium field, shape
        ``(n_orientations,) + grid.shape``."""
        eta0 = self.eta0
        active = cahn_hilliard_periodic_profile(
            self.grid.x[0],
            self.grid.lengths[0],
            self._a_param,
            -eta0,
            eta0,
            self.gradient_energy,
            center=self.center,
        )
        eta = xp.zeros((self.n_orientations,) + self.grid.shape)
        eta[self.active_orientation] = xp.broadcast_to(active, self.grid.shape)
        return eta
