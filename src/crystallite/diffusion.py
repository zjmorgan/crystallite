"""Linear spectral diffusion and nonlinear mass-diffusion time integration."""

from dataclasses import dataclass

from crystallite.backend import xp
from crystallite.spectral.short_range import DifferentialOperators


def _property_factor(property_value, k, k2, name):
    """Return ``k_i P_ij k_j`` for a scalar or 3 by 3 property."""
    property_value = xp.asarray(property_value)
    if property_value.ndim == 0:
        return property_value * k2
    if property_value.shape != (3, 3):
        raise ValueError(f"{name} must be a scalar or a 3 by 3 tensor")
    return xp.einsum("i...,ij,j...->...", k, property_value, k)


def double_well_derivative(c, a=1.0, c_alpha=-1.0, c_beta=1.0):
    r"""Return the derivative of the quartic two-well free energy.

    The general barrier form from the PDF is

    .. math::

        f(c) = a (c - c_\alpha)^2 (c - c_\beta)^2,
        \qquad
        f'(c) = 2a (c - c_\alpha)(c - c_\beta)
        (2c - c_\alpha - c_\beta).

    The symmetric special case is recovered by setting
    ``c_alpha = -c0`` and ``c_beta = c0``.
    """
    c = xp.asarray(c)
    return 2.0 * a * (c - c_alpha) * (c - c_beta) * (
        2.0 * c - c_alpha - c_beta
    )


def symmetric_double_well_derivative(c, a=1.0, c0=1.0):
    """Convenience wrapper for the symmetric quartic case."""
    return double_well_derivative(c, a=a, c_alpha=-c0, c_beta=c0)


def double_well_curvature(c, a=1.0, c_alpha=-1.0, c_beta=1.0):
    r"""Return the second derivative of the quartic two-well free energy.

    .. math::

        f''(c) = 2a \left[(2c - c_\alpha - c_\beta)^2
        + 2(c - c_\alpha)(c - c_\beta)\right].

    At the midpoint :math:`c = (c_\alpha + c_\beta)/2` this reduces to
    :math:`f''=-a(c_\beta - c_\alpha)^2`, the (negative, spinodal) local
    curvature used to linearize :class:`LinearSpectralDiffusion` about
    the unstable composition halfway between the two wells.
    """
    c = xp.asarray(c)
    return 2.0 * a * (
        (2.0 * c - c_alpha - c_beta) ** 2
        + 2.0 * (c - c_alpha) * (c - c_beta)
    )


def double_well_equilibrium_width(a, c_alpha, c_beta, gradient_energy):
    r"""Return the analytic equilibrium tanh interface width.

    For the quartic barrier ``f(c) = a (c - c_alpha)^2 (c - c_beta)^2``
    with an isotropic gradient-energy coefficient ``kappa``, the 1D
    Cahn-Hilliard equilibrium profile solving
    ``kappa * c'' = f'(c)`` with ``c -> c_alpha, c_beta`` and
    ``c' -> 0`` as ``x -> -inf, +inf`` is the tanh profile produced by
    :meth:`crystallite.verification.InterfaceCase.tanh_profile`,

    .. math::

        c(x) = m + w \tanh\!\left(\frac{x - x_0}{\delta}\right),
        \qquad
        \delta = \sqrt{\frac{\kappa}{2 a w^2}},

    with midpoint ``m = (c_alpha + c_beta) / 2`` and half-span
    ``w = (c_beta - c_alpha) / 2``. This first integral (and hence
    ``delta``) follows from multiplying the ODE by ``c'`` and
    integrating once, using that ``f(c_alpha) = f(c_beta) = 0``.

    Parameters
    ----------
    a : float
        Quartic barrier coefficient.
    c_alpha, c_beta : float
        The two stable equilibrium compositions.
    gradient_energy : float
        Isotropic gradient-energy coefficient ``kappa``.

    Returns
    -------
    float
        The analytic tanh width ``delta``.
    """
    half_span = 0.5 * (c_beta - c_alpha)
    return xp.sqrt(gradient_energy / (2.0 * a * half_span**2))


@dataclass(frozen=True)
class LinearSpectralDiffusion:
    r"""Exact spectral stepping for a linear diffusion equation.

    The model evolves a scalar concentration according to

    .. math::

        \partial_t c = \nabla \cdot M \nabla \mu,
        \qquad
        \mu = a c - \nabla \cdot K \nabla c.

    ``a`` is the local free-energy curvature, ``M`` is the mobility, and
    ``K`` is the concentration-gradient energy tensor. The resulting
    Fourier modes decay as ``exp(-lambda * t)`` with

    .. math::

        \lambda = (k_i M_{ij} k_j)
        (a + k_i K_{ij} k_j).

    Parameters
    ----------
    grid : Grid
        Spatial grid used for the real and Fourier transforms.
    mobility : float or array_like
        Constant scalar mobility or a constant ``(3, 3)`` mobility tensor.
    gradient_energy : float or array_like, default=0.0
        Scalar or ``(3, 3)`` concentration-gradient energy coefficient.
    free_energy_curvature : float, default=1.0
        Linearized local free-energy curvature ``a``. May be negative for
        a mode linearized about an unstable (spinodal) composition; the
        mode is still correctly resolved as growing rather than decaying,
        since this stepper is the exact solution of the linear ODE for
        each Fourier mode.
    """

    grid: object
    mobility: object
    gradient_energy: object = 0.0
    free_energy_curvature: float = 1.0

    def __post_init__(self):
        _property_factor(
            self.mobility,
            self._wavevector,
            self.grid.k2,
            "mobility",
        )
        _property_factor(
            self.gradient_energy,
            self._wavevector,
            self.grid.k2,
            "gradient_energy",
        )

    @property
    def _wavevector(self):
        """Wavevector with a leading component axis."""
        components = xp.broadcast_arrays(*self.grid.k)
        return xp.stack(components, axis=0)

    @property
    def decay_rate(self):
        """Fourier-space decay rate for every mode on the grid."""
        mobility_factor = _property_factor(
            self.mobility,
            self._wavevector,
            self.grid.k2,
            "mobility",
        )
        gradient_factor = _property_factor(
            self.gradient_energy,
            self._wavevector,
            self.grid.k2,
            "gradient_energy",
        )
        return mobility_factor * (
            self.free_energy_curvature + gradient_factor
        )

    def step(self, field, time_step):
        """Advance a real-space field by one exact spectral time step.

        Parameters
        ----------
        field : array_like
            Real scalar field with shape equal to ``grid.shape``.
        time_step : float
            Nonnegative time increment.

        Returns
        -------
        ndarray
            The advanced real-space field.
        """
        field = xp.asarray(field)
        if field.shape != self.grid.shape:
            raise ValueError(
                "field shape must match grid shape: "
                f"expected {self.grid.shape}, got {field.shape}"
            )
        if time_step < 0:
            raise ValueError("time_step must be nonnegative")

        field_hat = self.grid.fft(field)
        field_hat *= xp.exp(-time_step * self.decay_rate)
        return self.grid.ifft(field_hat)

    def run(self, field, times):
        """Return fields at a sequence of times from a common initial state.

        Parameters
        ----------
        field : array_like
            Initial real scalar field at time zero.
        times : array_like
            Nondecreasing times at which to evaluate the solution. The
            first time must be zero.

        Returns
        -------
        ndarray
            Fields stacked along a new leading time axis.
        """
        times = xp.asarray(times)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times must be a nonempty one-dimensional array")
        if times[0] != 0 or xp.any(times[1:] < times[:-1]):
            raise ValueError("times must be nondecreasing and start at zero")

        field = xp.asarray(field)
        if field.shape != self.grid.shape:
            raise ValueError(
                "field shape must match grid shape: "
                f"expected {self.grid.shape}, got {field.shape}"
            )

        field_hat = self.grid.fft(field)
        fields = tuple(
            self.grid.ifft(field_hat * xp.exp(-time * self.decay_rate))
            for time in times
        )
        return xp.stack(fields, axis=0)


@dataclass(frozen=True)
class MassDiffusion:
    r"""Nonlinear mass-diffusion step for a scalar composition field.

    The solver advances

    .. math::

        \partial_t c = -\nabla \cdot J,
        \qquad
        J_i = -M_{ij} \partial_j \mu,
        \qquad
        \mu = f'(c) - \partial_i (K_{ij} \partial_j c).

    Parameters
    ----------
    grid : Grid
        Spatial grid used by the solver.
    mobility : float or array_like, default=1.0
        Constant scalar mobility or a constant ``(3, 3)`` mobility tensor.
    gradient_energy : float or array_like, default=0.0
        Scalar or ``(3, 3)`` concentration-gradient energy coefficient.
    free_energy_derivative : callable, optional
        Function returning the derivative of the local free-energy density.
        If omitted, the PDF quartic two-well form is used,
        ``f'(c) = 2a (c - c_alpha)(c - c_beta)(2c - c_alpha - c_beta)``.
    operator : object, optional
        Differential operators for gradient/divergence operations.
    bulk_free_energy_coefficient : float, default=1.0
        Coefficient ``a`` in the quartic barrier model.
    left_well : float, default=-1.0
        Left equilibrium composition in the default barrier form.
    right_well : float, default=1.0
        Right equilibrium composition in the default barrier form.
    """

    grid: object
    mobility: object = 1.0
    gradient_energy: object = 0.0
    free_energy_derivative: object = None
    operator: object = None
    bulk_free_energy_coefficient: float = 1.0
    left_well: float = -1.0
    right_well: float = 1.0

    def __post_init__(self):
        if self.operator is None:
            object.__setattr__(self, "operator", DifferentialOperators(self.grid))
        if self.free_energy_derivative is None:
            object.__setattr__(
                self,
                "free_energy_derivative",
                lambda c: double_well_derivative(
                    c,
                    a=self.bulk_free_energy_coefficient,
                    c_alpha=self.left_well,
                    c_beta=self.right_well,
                ),
            )

    def _mobility_tensor(self):
        """Normalizes mobility to a 3 by 3 tensor."""
        mobility = xp.asarray(self.mobility)
        if mobility.ndim == 0:
            return mobility * xp.eye(3)
        if mobility.shape != (3, 3):
            raise ValueError("mobility must be a scalar or a 3 by 3 tensor")
        return mobility

    def _gradient_energy_tensor(self):
        """Normalizes gradient stiffness to a 3 by 3 tensor."""
        kappa = xp.asarray(self.gradient_energy)
        if kappa.ndim == 0:
            return kappa * xp.eye(3)
        if kappa.shape != (3, 3):
            raise ValueError(
                "gradient_energy must be a scalar or a 3 by 3 tensor"
            )
        return kappa

    def _gradient(self, field):
        """Return the spectral gradient of a scalar field."""
        field = xp.asarray(field)
        if field.shape == self.grid.shape:
            field = self.grid.fft(field)
        return self.operator.grad(field)

    def _flux(self, chemical_potential):
        """Return the diffusive flux J_i = -M_ij d_j mu."""
        grad_mu = self._gradient(chemical_potential)
        mobility = self._mobility_tensor()
        return -xp.tensordot(mobility, grad_mu, axes=(1, 0))

    def _chemical_potential(self, field):
        """Return the chemical potential mu = f'(c) - div(K grad c)."""
        field = xp.asarray(field)
        derivative = xp.asarray(self.free_energy_derivative(field))
        grad_c = self._gradient(field)
        kappa = self._gradient_energy_tensor()
        k_grad = xp.tensordot(kappa, grad_c, axes=(1, 0))
        return self.grid.fft(derivative) - self.operator.div(k_grad)

    def chemical_potential(self, field):
        """Return the real-space chemical potential ``mu = f'(c) - div(K grad c)``.

        A spatially uniform (constant) result indicates ``field`` is a
        diffusive equilibrium for this solver's free-energy and
        gradient-energy terms; this is useful for verifying analytic
        equilibrium profiles (e.g. the tanh interface) without timestepping.
        """
        field = xp.asarray(field)
        if field.shape != self.grid.shape:
            raise ValueError(
                "field shape must match grid shape: "
                f"expected {self.grid.shape}, got {field.shape}"
            )
        return self.grid.ifft(self._chemical_potential(field))

    def step(self, field, time_step):
        """Advance the field by one explicit mass-diffusion step."""
        field = xp.asarray(field)
        if field.shape != self.grid.shape:
            raise ValueError(
                "field shape must match grid shape: "
                f"expected {self.grid.shape}, got {field.shape}"
            )
        if time_step < 0:
            raise ValueError("time_step must be nonnegative")

        field_hat = self.grid.fft(field)
        chemical_potential = self._chemical_potential(field)
        flux = self._flux(chemical_potential)
        updated_hat = field_hat - time_step * self.operator.div(flux)
        return self.grid.ifft(updated_hat)

    def energy(self, field):
        """Return a simple bulk-plus-gradient energy estimate."""
        field = xp.asarray(field)
        bulk = 0.5 * field**2 * (1.0 - field) ** 2
        grad = self._gradient(field)
        grad_sq = xp.sum(grad * grad, axis=0)
        return xp.mean(bulk + 0.5 * self.gradient_energy * grad_sq)