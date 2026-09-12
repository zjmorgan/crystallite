"""Composition-dependent diffusivity and mobility models."""

from dataclasses import dataclass

from crystallite.backend import xp
from crystallite.mixture import arithmetic, harmonic


def diffusivity(x, d_a, d_b, model="darken"):
    """Return the composition-dependent interdiffusion coefficient.

    Parameters
    ----------
    x : array_like
        Fraction of species B. The fraction of species A is ``1 - x``.
    d_a, d_b : array_like
        Species A and B diffusivities. Scalars and square diffusivity
        tensors are supported.
    model : {"darken", "collective"}, default="darken"
        Mixing rule. ``"darken"`` uses the arithmetic rule
        ``x * d_b + (1 - x) * d_a``. ``"collective"`` uses the harmonic
        rule for the same two species diffusivities.

    Returns
    -------
    array_like
        The effective diffusivity.

    Raises
    ------
    ValueError
        If ``model`` is not recognized.
    """
    if model == "darken":
        return arithmetic(d_b, d_a, x)
    if model == "collective":
        return harmonic(d_b, d_a, x)
    raise ValueError("model must be 'darken' or 'collective'")


def mobility(
    x,
    d_a,
    d_b,
    temperature,
    model="darken",
    c0=1.0,
    gas_constant=1.0,
    degeneracy=False,
):
    """Return mobility from composition-dependent diffusivity.

    Parameters
    ----------
    x : array_like
        Fraction of species B. The fraction of species A is ``1 - x``.
    d_a, d_b : array_like
        Species A and B diffusivities. Scalars and square diffusivity
        tensors are supported.
    temperature : float or array_like
        Absolute temperature in the units used by ``gas_constant``.
    model : {"darken", "collective"}, default="darken"
        Diffusivity mixing rule passed to :func:`diffusivity`.
    c0 : float or array_like, default=1.0
        Reference concentration in the mobility relation.
    gas_constant : float, default=1.0
        Gas constant in the chosen energy units.
    degeneracy : bool, default=False
        If true, multiply the mobility by ``x * (1 - x)``.

    Returns
    -------
    array_like
        The mobility, using ``M = c0 * D / (R * T)``.
    """
    temperature = xp.asarray(temperature)
    if xp.any(temperature <= 0):
        raise ValueError("temperature must be positive")
    if gas_constant <= 0:
        raise ValueError("gas_constant must be positive")

    result = c0 * diffusivity(x, d_a, d_b, model=model)
    result = result / (gas_constant * temperature)
    if degeneracy:
        result = result * xp.asarray(x) * (1.0 - xp.asarray(x))
    return result


@dataclass(frozen=True)
class Mobility:
    """A reusable two-species mobility model.

    Parameters
    ----------
    d_a, d_b : array_like
        Species A and B diffusivities.
    temperature : float or array_like
        Absolute temperature in the units used by ``gas_constant``.
    model : {"darken", "collective"}, default="darken"
        Diffusivity mixing rule.
    c0 : float or array_like, default=1.0
        Reference concentration.
    gas_constant : float, default=1.0
        Gas constant in the chosen energy units.
    degeneracy : bool, default=False
        Whether to apply the ``x * (1 - x)`` concentration factor.
    """

    d_a: object
    d_b: object
    temperature: object
    model: str = "darken"
    c0: object = 1.0
    gas_constant: float = 1.0
    degeneracy: bool = False

    def __post_init__(self):
        if self.model not in {"darken", "collective"}:
            raise ValueError("model must be 'darken' or 'collective'")

    def diffusivity(self, x):
        """Evaluate effective diffusivity at composition ``x``."""
        return diffusivity(x, self.d_a, self.d_b, model=self.model)

    def mobility(self, x):
        """Evaluate mobility at composition ``x``."""
        return mobility(
            x,
            self.d_a,
            self.d_b,
            self.temperature,
            model=self.model,
            c0=self.c0,
            gas_constant=self.gas_constant,
            degeneracy=self.degeneracy,
        )


def binary_mobility(
    x_b,
    d_a_in_a,
    d_a_in_b,
    d_b_in_a,
    d_b_in_b,
    temperature,
    gas_constant=1.0,
    c0=1.0,
):
    r"""Return the binary A-B interdiffusion mobility.

    Each species' own diffusivity is first linearly interpolated between
    its value in the two pure end members,

    .. math::

        D_A(X_B) = X_A D_{A|A} + X_B D_{A|B}, \qquad
        D_B(X_B) = X_A D_{B|A} + X_B D_{B|B},

    then Darken-combined into the interdiffusion coefficient

    .. math::

        \tilde{D}(X_B) = X_B D_A(X_B) + X_A D_B(X_B),

    and converted to a phase-field mobility with the usual composition
    degeneracy factor,

    .. math::

        M(X_B) = \frac{X_A X_B}{RT} \tilde{D}(X_B).

    Parameters
    ----------
    x_b : array_like
        Composition (mole fraction) of species B, ``X_B``. The fraction of
        species A is ``X_A = 1 - X_B``. May be a scalar or a spatial field.
    d_a_in_a, d_a_in_b : float or array_like
        Diffusivity of species A in the A-rich (``X_B=0``) and B-rich
        (``X_B=1``) end members. Scalars and ``(3, 3)`` diffusivity
        tensors are supported.
    d_b_in_a, d_b_in_b : float or array_like
        Diffusivity of species B in the A-rich and B-rich end members.
    temperature : float or array_like
        Absolute temperature in the units used by ``gas_constant``.
    gas_constant : float, default=1.0
        Gas constant in the chosen energy units.
    c0 : float or array_like, default=1.0
        Reference concentration in the mobility relation.

    Returns
    -------
    array_like
        The interdiffusion mobility ``M(X_B)``, ``M = c0 * X_A * X_B *
        D_dark / (R * T)``.

    Notes
    -----
    ``x_b`` is clipped to ``[0, 1]`` before use: a mole fraction outside
    that range is unphysical, and would otherwise flip the sign of the
    ``X_A * X_B`` degeneracy factor -- turning diffusion into unstable
    anti-diffusion -- for any caller whose composition field transiently
    overshoots ``0`` or ``1`` (routine for an explicit or semi-implicit
    solver evolving a field that only asymptotically approaches those
    bounds).
    """
    x_b = xp.clip(xp.asarray(x_b), 0.0, 1.0)
    temperature = xp.asarray(temperature)
    if xp.any(temperature <= 0):
        raise ValueError("temperature must be positive")
    if gas_constant <= 0:
        raise ValueError("gas_constant must be positive")

    d_a_in_a = xp.asarray(d_a_in_a)
    d_a_in_b = xp.asarray(d_a_in_b)
    d_b_in_a = xp.asarray(d_b_in_a)
    d_b_in_b = xp.asarray(d_b_in_b)
    tensor_ndim = max(d_a_in_a.ndim, d_a_in_b.ndim, d_b_in_a.ndim, d_b_in_b.ndim)

    # Broadcast the (possibly field-valued) composition weight against the
    # end-member diffusivities' leading tensor axes once, up front, rather
    # than re-inferring a tensor rank at each interpolation stage below --
    # by the Darken step, d_a/d_b are themselves already tensor fields, so
    # re-deriving "the" tensor rank from their shape there would double
    # count the spatial axes already folded in.
    if tensor_ndim and x_b.ndim:
        pad = (1,) * x_b.ndim
        if d_a_in_a.ndim:
            d_a_in_a = d_a_in_a.reshape(d_a_in_a.shape + pad)
        if d_a_in_b.ndim:
            d_a_in_b = d_a_in_b.reshape(d_a_in_b.shape + pad)
        if d_b_in_a.ndim:
            d_b_in_a = d_b_in_a.reshape(d_b_in_a.shape + pad)
        if d_b_in_b.ndim:
            d_b_in_b = d_b_in_b.reshape(d_b_in_b.shape + pad)
        weight_b = x_b.reshape((1,) * tensor_ndim + x_b.shape)
    else:
        weight_b = x_b
    weight_a = 1.0 - weight_b

    d_a = weight_a * d_a_in_a + weight_b * d_a_in_b
    d_b = weight_a * d_b_in_a + weight_b * d_b_in_b
    d_dark = weight_b * d_a + weight_a * d_b
    degeneracy = weight_a * weight_b

    return c0 * degeneracy * d_dark / (gas_constant * temperature)


@dataclass(frozen=True)
class BinaryMobility:
    """A reusable binary A-B interdiffusion mobility model.

    Parameters
    ----------
    d_a_in_a, d_a_in_b : float or array_like
        Diffusivity of species A in the A-rich and B-rich end members.
    d_b_in_a, d_b_in_b : float or array_like
        Diffusivity of species B in the A-rich and B-rich end members.
    temperature : float or array_like
        Absolute temperature in the units used by ``gas_constant``.
    gas_constant : float, default=1.0
        Gas constant in the chosen energy units.
    c0 : float or array_like, default=1.0
        Reference concentration.
    """

    d_a_in_a: object
    d_a_in_b: object
    d_b_in_a: object
    d_b_in_b: object
    temperature: object
    gas_constant: float = 1.0
    c0: object = 1.0

    def mobility(self, x_b):
        """Evaluate the interdiffusion mobility at composition ``x_b``."""
        return binary_mobility(
            x_b,
            self.d_a_in_a,
            self.d_a_in_b,
            self.d_b_in_a,
            self.d_b_in_b,
            self.temperature,
            gas_constant=self.gas_constant,
            c0=self.c0,
        )