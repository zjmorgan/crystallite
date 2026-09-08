"""Composition-dependent diffusivity and mobility models."""

from dataclasses import dataclass

from crystallite.backend import xp
from crystallite.mixture import arithmetic, harmonic


def diffusivity(x, d_a, d_b, model="darken"):
    """Return the composition-dependent interdiffusion coefficient.

    Parameters
    ----------
    x : array_like
        Fraction of species A. The fraction of species B is ``1 - x``.
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
        Fraction of species A. The fraction of species B is ``1 - x``.
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