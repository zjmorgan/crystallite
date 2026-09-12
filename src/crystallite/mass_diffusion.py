"""Linear spectral diffusion and nonlinear mass-diffusion time integration."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipj, ellipk

from crystallite.backend import xp
from crystallite.material.properties import as_tensor
from crystallite.mixture import arithmetic, harmonic
from crystallite.spectral.short_range import DifferentialOperators


def _property_factor(property_value, k, k2):
    """Return ``k_i P_ij k_j`` for a scalar or 3 by 3 property."""
    property_value = xp.asarray(property_value)
    if property_value.ndim == 0:
        return property_value * k2
    return xp.einsum("i...,ij,j...->...", k, as_tensor(property_value), k)


def _apply_property(value, grid_shape, vector, name):
    """Contract a material property against a component-first vector field.

    ``value`` may be a scalar or ``(3, 3)`` tensor (spatially constant), or
    a per-point field of one -- shape ``grid_shape`` for an isotropic field
    or ``(3, 3) + grid_shape`` for a full anisotropic tensor field -- to be
    applied pointwise in real space.
    """
    value = xp.asarray(value)
    if value.ndim == 0:
        return value * vector
    if value.shape == (3, 3):
        return xp.einsum("ij,j...->i...", value, vector)
    if value.shape == grid_shape:
        return value[None, ...] * vector
    if value.shape == (3, 3) + grid_shape:
        return xp.einsum("ij...,j...->i...", value, vector)
    raise ValueError(
        f"{name} must be a scalar, a 3 by 3 tensor, a {grid_shape} field, "
        f"or a (3, 3) + {grid_shape} tensor field"
    )


@dataclass(frozen=True)
class GradientEnergy:
    """A reusable two-phase gradient-energy (concentration-gradient
    coefficient :math:`\\kappa`) mixing model.

    Just like a diffusivity, :math:`\\kappa` is properly a (possibly
    anisotropic) ``(3, 3)`` tensor rather than a bare scalar; this mixes a
    parent and product phase's ``kappa`` by composition, reusing the same
    mixing rules as :func:`crystallite.mobility.diffusivity`.

    Parameters
    ----------
    parent, product : float or array_like
        Gradient-energy coefficient of the parent and product phases.
        Scalars and ``(3, 3)`` tensors are supported.
    model : {"arithmetic", "harmonic"}, default="arithmetic"
        Mixing rule; see :func:`crystallite.mixture.arithmetic` and
        :func:`crystallite.mixture.harmonic`.
    """

    parent: object
    product: object
    model: str = "arithmetic"

    def __post_init__(self):
        if self.model not in {"arithmetic", "harmonic"}:
            raise ValueError("model must be 'arithmetic' or 'harmonic'")

    def gradient_energy(self, fraction):
        """Evaluate the mixed gradient-energy coefficient at ``fraction``
        (the product-phase fraction, e.g. a composition field)."""
        mix = arithmetic if self.model == "arithmetic" else harmonic
        return mix(self.product, self.parent, fraction)


def double_well_derivative(c, a=1.0, c_alpha=-1.0, c_beta=1.0):
    r"""Return the derivative of the quartic two-well free energy.

    The general quartic barrier form is

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


def chemical_free_energy(c, barrier_height, c_alpha, c_beta):
    r"""Return the fourth-order (quartic) bulk free energy.

    A fourth-order Taylor approximation about a reference composition
    :math:`X_0`,

    .. math::

        f_0(X) \approx 16 f_0(X_0)
        \left[\frac{c_\beta - c}{c_\beta - c_\alpha}\right]^2
        \left[\frac{c - c_\alpha}{c_\beta - c_\alpha}\right]^2,

    parametrized by the free-energy value :math:`f_0(X_0)` at the
    reference composition (e.g. read off CALPHAD data) rather than the
    raw quartic coefficient ``a`` used by :func:`double_well_derivative`
    -- the two are the same functional form, related by
    ``a = 16 * barrier_height / (c_beta - c_alpha)**4``.
    """
    span = c_beta - c_alpha
    return (
        16.0
        * barrier_height
        * ((c_beta - c) / span) ** 2
        * ((c - c_alpha) / span) ** 2
    )


def chemical_free_energy_derivative(c, barrier_height, c_alpha, c_beta):
    """Return the derivative of :func:`chemical_free_energy`.

    Equal to :func:`double_well_derivative` with
    ``a = 16 * barrier_height / (c_beta - c_alpha)**4``.
    """
    a = 16.0 * barrier_height / (c_beta - c_alpha) ** 4
    return double_well_derivative(c, a=a, c_alpha=c_alpha, c_beta=c_beta)


@dataclass(frozen=True)
class ChemicalFreeEnergy:
    """A reusable fourth-order (quartic) bulk free-energy model.

    Wraps :func:`chemical_free_energy`/:func:`chemical_free_energy_derivative` with
    fixed parameters so ``derivative`` can be passed directly as
    :class:`MassDiffusion`'s ``free_energy_derivative``.

    Parameters
    ----------
    barrier_height : float
        Free-energy value :math:`f_0(X_0)` at the reference composition.
    c_alpha, c_beta : float, default=-1.0, 1.0
        The two stable equilibrium compositions.
    """

    barrier_height: float
    c_alpha: float = -1.0
    c_beta: float = 1.0

    def value(self, c):
        """Evaluate the free energy at composition ``c``."""
        return chemical_free_energy(c, self.barrier_height, self.c_alpha, self.c_beta)

    def derivative(self, c):
        """Evaluate the free-energy derivative at composition ``c``."""
        return chemical_free_energy_derivative(
            c, self.barrier_height, self.c_alpha, self.c_beta
        )


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


def cahn_hilliard_period_range(a, c_alpha, c_beta, gradient_energy):
    r"""Return the minimum period admitting a periodic equilibrium profile.

    :func:`cahn_hilliard_periodic_profile` needs a period at least this
    large; smaller periods have no periodic double-well equilibrium (the
    elliptic modulus would have to be negative). The minimum occurs at
    modulus 0, the small-amplitude sinusoidal limit, where the period is
    :math:`2\pi\delta_0` for a length scale :math:`\delta_0` built the
    same way as :func:`double_well_equilibrium_width`'s ``delta`` but
    from the half-span rather than the full span.
    """
    half_span = 0.5 * (c_beta - c_alpha)
    delta_0 = xp.sqrt(gradient_energy / a) / (2.0 * half_span)
    return 2.0 * xp.pi * delta_0


def cahn_hilliard_periodic_profile(
    x, length, a, c_alpha, c_beta, gradient_energy, center=0.0
):
    r"""Return the exact periodic 1D Cahn-Hilliard equilibrium profile.

    A single tanh interface (:func:`double_well_equilibrium_width`) is
    only a valid equilibrium on an infinite domain: a periodic domain's
    left and right boundaries are identified, so a single kink from
    ``c_alpha`` to ``c_beta`` cannot close up on itself -- the field must
    come back via a second, opposite-signed interface (an "antikink"),
    giving one kink-antikink pair per period. The exact solution of
    ``kappa * c'' = f'(c)`` with this periodicity is a Jacobi elliptic
    function,

    .. math::

        c(x) = m_0 + A \, \mathrm{sn}\!\left(\frac{x - x_0}{s}, k\right),
        \qquad
        m_0 = \frac{c_\alpha + c_\beta}{2},

    with modulus ``k`` chosen so the profile's exact period (``4 K(k)
    s``, ``K`` the complete elliptic integral of the first kind) equals
    ``length``; ``k`` is solved for numerically since the length scale
    ``s`` itself depends on it. As ``k -> 1`` (period much larger than
    the tanh width), ``sn -> tanh`` and this reduces to two well-separated
    tanh interfaces, each of width :func:`double_well_equilibrium_width`;
    as ``k -> 0`` (period near the minimum in
    :func:`cahn_hilliard_period_range`), it reduces to a small-amplitude
    sinusoid.

    Parameters
    ----------
    x : array_like
        Positions at which to evaluate the profile.
    length : float
        Domain period. Must be at least
        :func:`cahn_hilliard_period_range`'s value, below which no
        periodic equilibrium exists.
    a : float
        Quartic barrier coefficient.
    c_alpha, c_beta : float
        The two stable equilibrium compositions.
    gradient_energy : float
        Isotropic gradient-energy coefficient ``kappa``.
    center : float, default=0.0
        Position of the kink (composition rising through the midpoint);
        the antikink sits half a period away.

    Returns
    -------
    ndarray
        The periodic equilibrium profile at ``x``.

    Raises
    ------
    ValueError
        If ``length`` is below the minimum admissible period.
    """
    minimum_length = float(cahn_hilliard_period_range(a, c_alpha, c_beta, gradient_energy))
    if length < minimum_length:
        raise ValueError(
            f"length must be at least {minimum_length:.6g} "
            "for a periodic equilibrium to exist at this a/gradient_energy"
        )

    half_span = 0.5 * (c_beta - c_alpha)
    midpoint = 0.5 * (c_beta + c_alpha)
    length_scale = np.sqrt(gradient_energy / a) / (2.0 * half_span)

    def scale(modulus):
        return length_scale * np.sqrt(1.0 + modulus)

    def period_residual(modulus):
        return 4.0 * ellipk(modulus) * scale(modulus) - length

    modulus = brentq(period_residual, 0.0, 1.0 - 1e-14, xtol=1e-14, rtol=1e-14)
    profile_scale = scale(modulus)
    amplitude = half_span * np.sqrt(2.0 * modulus / (1.0 + modulus))

    x_cpu = x.get() if hasattr(x, "get") else np.asarray(x)
    sn, _, _, _ = ellipj((x_cpu - center) / profile_scale, modulus)
    return xp.asarray(midpoint + amplitude * sn)


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
        _property_factor(self.mobility, self._wavevector, self.grid.k2)
        _property_factor(self.gradient_energy, self._wavevector, self.grid.k2)

    @property
    def _wavevector(self):
        """Wavevector with a leading component axis."""
        components = xp.broadcast_arrays(*self.grid.k)
        return xp.stack(components, axis=0)

    @property
    def decay_rate(self):
        """Fourier-space decay rate for every mode on the grid."""
        mobility_factor = _property_factor(
            self.mobility, self._wavevector, self.grid.k2
        )
        gradient_factor = _property_factor(
            self.gradient_energy, self._wavevector, self.grid.k2
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
    mobility : float, array_like, or callable, default=1.0
        Constant scalar mobility, a constant ``(3, 3)`` mobility tensor, or
        a callable ``mobility(field)`` returning either at every grid point
        (e.g. a scalar field of shape ``grid.shape`` for a spatially
        varying isotropic mobility, or a ``(3, 3) + grid.shape`` tensor
        field for a spatially varying anisotropic one) -- evaluated at the
        current composition each step. :func:`crystallite.mobility.mobility`
        and :class:`crystallite.mobility.Mobility` already produce
        composition-dependent scalar fields of this form.
    gradient_energy : float, array_like, or callable, default=0.0
        Scalar or ``(3, 3)`` concentration-gradient energy coefficient, or
        a callable with the same signature and return shapes as
        ``mobility``, e.g. mixed from phase properties with
        :func:`crystallite.mixture.arithmetic` or
        :func:`crystallite.mixture.harmonic`.
    free_energy_derivative : callable, optional
        Function returning the derivative of the local free-energy density.
        If omitted, the quartic two-well form is used,
        ``f'(c) = 2a (c - c_alpha)(c - c_beta)(2c - c_alpha - c_beta)``.
    operator : object, optional
        Differential operators for gradient/divergence operations.
    bulk_free_energy_coefficient : float, default=1.0
        Coefficient ``a`` in the quartic barrier model.
    left_well : float, default=-1.0
        Left equilibrium composition in the default barrier form.
    right_well : float, default=1.0
        Right equilibrium composition in the default barrier form.
    scheme : {"explicit", "semi_implicit"}, default="explicit"
        Time-stepping scheme. ``"explicit"`` is plain forward Euler, whose
        stable time step is limited by the shortest-wavelength (Nyquist)
        mode -- typically far smaller than the time scale of real
        spinodal growth, especially with a small gradient-energy
        coefficient. ``"semi_implicit"`` treats a homogeneous reference
        linearization of the dynamics implicitly (dividing the explicit
        update by ``1 + alpha * dt * L(k)``, where ``L(k)`` is exactly
        :class:`LinearSpectralDiffusion`'s ``decay_rate`` evaluated at
        ``reference_mobility``/``reference_gradient_energy``/
        ``reference_curvature``), which damps stiff high-``k`` modes
        unconditionally while leaving the full nonlinear physics in the
        explicit numerator -- allowing far larger, still-stable time
        steps.
    reference_mobility : float or array_like, optional
        Homogeneous mobility used only to build the ``semi_implicit``
        stabilizer. Defaults to ``mobility`` if that is not callable;
        required (and must be a scalar or ``(3, 3)`` tensor) if it is.
    reference_gradient_energy : float or array_like, optional
        Homogeneous gradient-energy coefficient for the ``semi_implicit``
        stabilizer, with the same default/callable rule as
        ``reference_mobility``.
    reference_curvature : float, optional
        Homogeneous free-energy curvature ``f''`` at the linearization
        point, for the ``semi_implicit`` stabilizer. Always required for
        that scheme -- unlike ``reference_mobility``/
        ``reference_gradient_energy`` it is never inferred, since
        ``free_energy_derivative`` may be an arbitrary callable. For the
        default quartic form, use :func:`double_well_curvature` at the
        chosen reference composition.
    alpha : float, default=1.0
        Stabilization weight in the ``semi_implicit`` denominator.
    dealias : bool, default=False
        If True, multiply the chemical potential's Fourier transform by
        ``grid.lanczos_filter`` each step -- a smooth taper to exactly 0
        at the Nyquist frequency, suppressing the spurious high-``k``
        content a nonlinear term (``f'(c)``, or a composition-dependent
        mobility/gradient energy) generates via aliasing when evaluated
        pseudo-spectrally. Standard practice for nonlinear pseudo-spectral
        solvers; ported from the reference Fortran solver's own
        ``sf_lanczos_filter``/``sf_filter``.
    """

    grid: object
    mobility: object = 1.0
    gradient_energy: object = 0.0
    free_energy_derivative: object = None
    operator: object = None
    bulk_free_energy_coefficient: float = 1.0
    left_well: float = -1.0
    right_well: float = 1.0
    scheme: str = "explicit"
    reference_mobility: object = None
    reference_gradient_energy: object = None
    reference_curvature: object = None
    alpha: float = 1.0
    dealias: bool = False

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
        if self.scheme not in ("explicit", "semi_implicit"):
            raise ValueError("scheme must be 'explicit' or 'semi_implicit'")
        if self.scheme == "semi_implicit":
            reference_mobility = self.reference_mobility
            if reference_mobility is None:
                if callable(self.mobility):
                    raise ValueError(
                        "semi_implicit scheme needs reference_mobility "
                        "when mobility is callable"
                    )
                reference_mobility = self.mobility
            reference_gradient_energy = self.reference_gradient_energy
            if reference_gradient_energy is None:
                if callable(self.gradient_energy):
                    raise ValueError(
                        "semi_implicit scheme needs reference_gradient_energy "
                        "when gradient_energy is callable"
                    )
                reference_gradient_energy = self.gradient_energy
            if self.reference_curvature is None:
                raise ValueError(
                    "semi_implicit scheme needs reference_curvature"
                )
            object.__setattr__(
                self,
                "_reference_stepper",
                LinearSpectralDiffusion(
                    self.grid,
                    mobility=reference_mobility,
                    gradient_energy=reference_gradient_energy,
                    free_energy_curvature=self.reference_curvature,
                ),
            )

    @staticmethod
    def _evaluate(value, field):
        """Evaluate a constant-or-callable material property at ``field``."""
        return value(field) if callable(value) else value

    def _real_gradient(self, field):
        """Real-space gradient of a real-space scalar field (single round trip)."""
        return self.grid.ifft(self.operator.grad(self.grid.fft(field)))

    def _chemical_potential(self, field):
        r"""Return the Fourier-space chemical potential
        :math:`\widehat{\mu} = \widehat{f'(c)} - \widehat{\nabla \cdot (K(c) \nabla c)}`.

        A constant ``gradient_energy`` stays entirely in Fourier space (one
        multiply, exactly as before); a callable one is only evaluated
        pointwise in real space -- unavoidable for a spatially varying
        coefficient -- with a single extra FFT/IFFT round trip.
        """
        derivative = xp.asarray(self.free_energy_derivative(field))
        grad_c_hat = self.operator.grad(self.grid.fft(field))
        kappa = self.gradient_energy
        if callable(kappa):
            kappa_value = kappa(field)
            grad_c = self.grid.ifft(grad_c_hat)
            k_grad = _apply_property(
                kappa_value, self.grid.shape, grad_c, "gradient_energy"
            )
            k_grad_hat = self.grid.fft(k_grad)
        else:
            k_grad_hat = _apply_property(
                kappa, self.grid.shape, grad_c_hat, "gradient_energy"
            )
        mu_hat = self.grid.fft(derivative) - self.operator.div(k_grad_hat)
        if self.dealias:
            mu_hat = mu_hat * self.grid.lanczos_filter
        return mu_hat

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

    def _flux(self, field, mu_hat):
        """Return the Fourier-space diffusive flux ``FFT(J_i) = FFT(-M_ij(c) d_j mu)``.

        Same constant-stays-in-Fourier-space, callable-drops-to-real-space
        split as :meth:`_chemical_potential`.
        """
        grad_mu_hat = self.operator.grad(mu_hat)
        mobility = self.mobility
        if callable(mobility):
            mobility_value = mobility(field)
            grad_mu = self.grid.ifft(grad_mu_hat)
            flux = -_apply_property(mobility_value, self.grid.shape, grad_mu, "mobility")
            return self.grid.fft(flux)
        return -_apply_property(mobility, self.grid.shape, grad_mu_hat, "mobility")

    def step(self, field, time_step):
        """Advance the field by one mass-diffusion step (see ``scheme``)."""
        field = xp.asarray(field)
        if field.shape != self.grid.shape:
            raise ValueError(
                "field shape must match grid shape: "
                f"expected {self.grid.shape}, got {field.shape}"
            )
        if time_step < 0:
            raise ValueError("time_step must be nonnegative")

        field_hat = self.grid.fft(field)
        mu_hat = self._chemical_potential(field)
        flux_hat = self._flux(field, mu_hat)
        divergence_hat = self.operator.div(flux_hat)
        if self.scheme == "semi_implicit":
            divergence_hat = divergence_hat / (
                1.0 + self.alpha * time_step * self._reference_stepper.decay_rate
            )
        updated_hat = field_hat - time_step * divergence_hat
        return self.grid.ifft(updated_hat)

    def energy(self, field):
        """Return a simple bulk-plus-gradient energy estimate."""
        field = xp.asarray(field)
        bulk = 0.5 * field**2 * (1.0 - field) ** 2
        grad = self._real_gradient(field)
        kappa = self._evaluate(self.gradient_energy, field)
        k_grad = _apply_property(kappa, self.grid.shape, grad, "gradient_energy")
        grad_term = xp.sum(grad * k_grad, axis=0)
        return xp.mean(bulk + 0.5 * grad_term)