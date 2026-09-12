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


@dataclass(frozen=True)
class CircularGrainCase:
    r"""A shrinking circular grain under curvature-driven Allen-Cahn dynamics.

    A single order parameter is :math:`+\eta_0` inside a disk of radius
    :math:`R` and :math:`-\eta_0` outside (the rest held at zero, the
    radial analog of :class:`PlanarGrainBoundaryCase`'s kink profile).

    The equilibrium radial profile is taken to be the flat-interface
    tanh profile evaluated at ``r - R`` -- exact on an infinite plane,
    and a good approximation here whenever the interface width is small
    relative to :math:`R`. Substituting this ansatz into the Allen-Cahn
    equation and expanding the radial Laplacian's curvature term,
    :math:`\nabla^2\eta = \eta'' + \eta'/r`, to leading order in
    :math:`1/R` gives an exact cancellation of every term *except* the
    curvature term, because the flat-interface profile already solves
    :math:`\kappa\Phi'' = f'(\Phi)` identically
    (:func:`crystallite.mass_diffusion.double_well_equilibrium_width`).
    What remains is

    .. math::

        R \frac{dR}{dt} = -L \kappa,
        \qquad
        R(t)^2 = R_0^2 - 2 L \kappa t,

    where :math:`L` is ``mobility`` and :math:`\kappa` is
    ``gradient_energy`` -- notably the *gradient-energy coefficient
    itself*, not the interfacial (line-tension) energy :attr:`surface_energy`.
    The two are easy to conflate (the standard curvature-flow heuristic
    "velocity = mobility x surface energy x curvature" from sharp-interface
    theory) but are only proportional to each other up to a factor of
    :math:`\eta_0^2/\text{width}^2` here, which is not of order 1 in
    general -- using :attr:`surface_energy` in place of ``gradient_energy``
    in the rate law overstates or understates the true shrink rate by
    that factor. :attr:`surface_energy` remains the correct quantity for
    the *static* excess energy of the boundary, :math:`F = 2\pi R\,\sigma`
    in 2D (verified directly against :meth:`GrainOrientation.free_energy`).

    Parameters
    ----------
    grid : Grid
        A 2D (or 3D) grid.
    barrier_coefficient : float, default=1.0
        Coefficient ``a``.
    quartic_coefficient : float, default=1.0
        Coefficient ``b``.
    gradient_energy : float, default=1.0
        Gradient-energy coefficient ``kappa`` for the active orientation.
    mobility : float, default=1.0
        Kinetic coefficient ``L`` for the active orientation.
    n_orientations : int, default=2
        Total number of order parameters in the field returned by
        :meth:`profile` (only one is active; the rest are held at zero).
    active_orientation : int, default=0
        Index of the active order parameter.
    center : tuple of float, default=(0.0, 0.0)
        Center of the disk in the grid's first two coordinates.
    """

    grid: Grid
    barrier_coefficient: float = 1.0
    quartic_coefficient: float = 1.0
    gradient_energy: float = 1.0
    mobility: float = 1.0
    n_orientations: int = 2
    active_orientation: int = 0
    center: tuple = (0.0, 0.0)

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

    @property
    def surface_energy(self):
        r"""Interfacial (line-tension) energy per unit boundary length.

        Equal to :math:`\int \kappa (\Phi')^2\,dx = \tfrac{4}{3}\kappa
        \eta_0^2/\text{width}`, using equipartition
        (:math:`\kappa(\Phi')^2/2 = f(\Phi) - f(\eta_0)` on the exact
        static profile) and :math:`\int \mathrm{sech}^4 = 4/3`. Sets the
        static excess energy :math:`F=2\pi R\sigma`, not the shrink rate
        (see class docstring).
        """
        return (4.0 / 3.0) * self.gradient_energy * self.eta0**2 / self.width

    def radius(self, t, initial_radius):
        """Return the analytic radius at time ``t``, clipped at zero."""
        value = initial_radius**2 - 2.0 * self.mobility * self.gradient_energy * t
        return xp.sqrt(xp.clip(value, 0.0, None))

    def vanishing_time(self, initial_radius):
        """Return the time at which the analytic radius reaches zero."""
        return initial_radius**2 / (2.0 * self.mobility * self.gradient_energy)

    def profile(self, initial_radius):
        """Return the initial field, shape ``(n_orientations,) + grid.shape``."""
        x = self.grid.x[0] - self.center[0]
        y = self.grid.x[1] - self.center[1]
        r = xp.sqrt(x**2 + y**2)
        eta0 = self.eta0
        active = eta0 * xp.tanh((initial_radius - r) / self.width)
        eta = xp.zeros((self.n_orientations,) + self.grid.shape)
        eta[self.active_orientation] = xp.broadcast_to(active, self.grid.shape)
        return eta
