"""Linear spectral diffusion and Cahn-Hilliard time integration."""

from dataclasses import dataclass

from crystallite.backend import xp


def _property_factor(property_value, k, k2, name):
    """Return ``k_i P_ij k_j`` for a scalar or 3 by 3 property."""
    property_value = xp.asarray(property_value)
    if property_value.ndim == 0:
        return property_value * k2
    if property_value.shape != (3, 3):
        raise ValueError(f"{name} must be a scalar or a 3 by 3 tensor")
    return xp.einsum("i...,ij,j...->...", k, property_value, k)


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
        Linearized local free-energy curvature ``a``.
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
        if self.free_energy_curvature < 0:
            raise ValueError("free_energy_curvature must be nonnegative")

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