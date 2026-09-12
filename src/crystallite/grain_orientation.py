"""Multi-order-parameter Allen-Cahn grain-orientation time integration."""

from dataclasses import dataclass

from crystallite.backend import xp
from crystallite.spectral.short_range import DifferentialOperators


def _broadcast_per_orientation(value, n_orientations):
    """Reshape a scalar or ``(n_orientations,)`` array so it broadcasts
    against a leading orientation axis, ``(n_orientations, *grid.shape)``.
    """
    value = xp.asarray(value)
    if value.ndim == 0:
        return value
    if value.shape == (n_orientations,):
        return value.reshape((n_orientations, 1, 1, 1))
    raise ValueError(
        f"value must be a scalar or a ({n_orientations},) array, "
        f"got shape {value.shape}"
    )


@dataclass(frozen=True)
class GrainOrientation:
    r"""Multi-order-parameter Allen-Cahn grain-orientation solver.

    Evolves ``n_orientations`` non-conserved order-parameter fields
    :math:`\eta_i` (grain-orientation indicators) according to the
    time-dependent Ginzburg-Landau (Allen-Cahn) equation

    .. math::

        \frac{\partial \eta_i}{\partial t} = -L_i \frac{\delta F}{\delta \eta_i},

    with free energy

    .. math::

        F = \int_\Omega \left\{ f(\eta) + \sum_i \frac{\kappa_i}{2}
        (\nabla \eta_i)^2 \right\} d^3 r,
        \qquad
        f(\eta) = \sum_i \left[-\frac{a}{2}\eta_i^2 + \frac{b}{4}\eta_i^4\right]
        + \gamma \sum_{i<j} \eta_i^2 \eta_j^2,

    whose minima are the ``n_orientations`` single-grain states
    :math:`f(\pm\eta_0, 0, \ldots, 0) = \cdots = 0` for
    :math:`\eta_0 = \sqrt{a/b}`.

    Time-stepped with the first-order semi-implicit scheme

    .. math::

        \left[1 + h L_i \kappa_i k^2\right] \hat\eta_i^{\,n+1}
        = \hat\eta_i^{\,n} - h L_i \hat N_i^{\,n},

    which treats the (exactly linear) gradient term implicitly -- damping
    the stiff high-:math:`k` modes unconditionally -- while the nonlinear
    remainder

    .. math::

        N_i = -a\eta_i + b\eta_i^3 + 2\gamma\eta_i \sum_{j\neq i}\eta_j^2

    stays explicit. Unlike :class:`crystallite.mass_diffusion.MassDiffusion`,
    dividing the *whole* update by the implicit denominator is correct here
    (not just a flux-divergence term): with ``N=0`` this reduces exactly to
    backward Euler for the linear ODE
    :math:`d\hat\eta/dt = -L\kappa k^2 \hat\eta`, since :math:`\delta F
    /\delta\eta_i` is itself the evolution rate here, with no extra spatial
    divergence between the potential and the equation of motion the way
    Cahn-Hilliard's :math:`\nabla\cdot(M\nabla\mu)` has.

    Parameters
    ----------
    grid : Grid
        Spatial grid used by the solver.
    mobility : float or array_like, default=1.0
        Kinetic coefficient :math:`L_i`. Scalar (shared by every
        orientation) or a ``(n_orientations,)`` array.
    gradient_energy : float or array_like, default=1.0
        Gradient-energy coefficient :math:`\kappa_i`, with the same
        scalar-or-``(n_orientations,)`` shape rule as ``mobility``.
    barrier_coefficient : float, default=1.0
        Quadratic (barrier) coefficient ``a``.
    quartic_coefficient : float, default=1.0
        Quartic coefficient ``b``.
    cross_coefficient : float, default=1.0
        Cross-coupling coefficient ``gamma`` between distinct orientations.
    operator : object, optional
        Differential operators for the Laplacian. Defaults to
        :class:`crystallite.spectral.short_range.DifferentialOperators`.
    """

    grid: object
    mobility: object = 1.0
    gradient_energy: object = 1.0
    barrier_coefficient: float = 1.0
    quartic_coefficient: float = 1.0
    cross_coefficient: float = 1.0
    operator: object = None

    def __post_init__(self):
        if self.operator is None:
            object.__setattr__(self, "operator", DifferentialOperators(self.grid))

    def _cross_term(self, eta):
        """Return ``sum_{j != i} eta_j**2`` for every orientation ``i``."""
        squared = eta**2
        total = xp.sum(squared, axis=0, keepdims=True)
        return total - squared

    def _nonlinear_term(self, eta):
        """Return the nonlinear remainder ``N_i`` (no gradient term)."""
        a = self.barrier_coefficient
        b = self.quartic_coefficient
        gamma = self.cross_coefficient
        return -a * eta + b * eta**3 + 2.0 * gamma * eta * self._cross_term(eta)

    def homogeneous_energy(self, eta):
        r"""Return the local (non-gradient) free-energy density ``f(eta)``.

        Uses :math:`\sum_{i<j}\eta_i^2\eta_j^2 = \tfrac{1}{2}\left[\left(
        \sum_i \eta_i^2\right)^2 - \sum_i \eta_i^4\right]` to avoid an
        explicit double loop (or an :math:`O(n^2)` intermediate array)
        over orientation pairs.
        """
        eta = xp.asarray(eta)
        a = self.barrier_coefficient
        b = self.quartic_coefficient
        gamma = self.cross_coefficient
        local = xp.sum(-0.5 * a * eta**2 + 0.25 * b * eta**4, axis=0)
        squared = eta**2
        total_squared = xp.sum(squared, axis=0)
        sum_of_fourth = xp.sum(squared**2, axis=0)
        cross = 0.5 * gamma * (total_squared**2 - sum_of_fourth)
        return local + cross

    def driving_force(self, eta):
        """Return the real-space functional derivative ``dF/deta_i``,
        including the gradient (Laplacian) term."""
        eta = xp.asarray(eta)
        kappa = _broadcast_per_orientation(self.gradient_energy, eta.shape[0])
        laplacian_hat = self.operator.laplacian(self.grid.fft(eta))
        gradient_term = self.grid.ifft(-kappa * laplacian_hat)
        return self._nonlinear_term(eta) + gradient_term

    def free_energy(self, eta):
        """Return the mean total free-energy density (bulk + gradient)."""
        eta = xp.asarray(eta)
        bulk = self.homogeneous_energy(eta)
        grad = self.grid.ifft(self.operator.grad(self.grid.fft(eta)))
        grad_sq = xp.sum(grad**2, axis=1)
        kappa = _broadcast_per_orientation(self.gradient_energy, eta.shape[0])
        gradient_density = xp.sum(0.5 * kappa * grad_sq, axis=0)
        return xp.mean(bulk + gradient_density)

    def step(self, eta, time_step):
        """Advance every orientation field by one semi-implicit step."""
        eta = xp.asarray(eta)
        if eta.shape[1:] != self.grid.shape:
            raise ValueError(
                "eta shape must be (n_orientations,) + grid.shape: "
                f"expected (*, {self.grid.shape}), got {eta.shape}"
            )
        if time_step < 0:
            raise ValueError("time_step must be nonnegative")

        n = eta.shape[0]
        mobility = _broadcast_per_orientation(self.mobility, n)
        kappa = _broadcast_per_orientation(self.gradient_energy, n)

        eta_hat = self.grid.fft(eta)
        nonlinear_hat = self.grid.fft(self._nonlinear_term(eta))
        linear_operator = kappa * self.grid.k2
        denominator = 1.0 + time_step * mobility * linear_operator
        updated_hat = (eta_hat - time_step * mobility * nonlinear_hat) / denominator
        return self.grid.ifft(updated_hat)
