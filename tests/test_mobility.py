import numpy as np
import pytest

from crystallite.mobility import Mobility, diffusivity, mobility


def test_darken_diffusivity_uses_species_fractions():
    # PDF convention: x = X_B and X_A = 1 - x.
    assert diffusivity(0.25, 2.0, 10.0) == pytest.approx(4.0)
    assert diffusivity(0.75, 2.0, 10.0) == pytest.approx(8.0)


def test_collective_diffusivity_uses_harmonic_rule():
    assert diffusivity(0.25, 2.0, 10.0, model="collective") == pytest.approx(2.5)


def test_darken_supports_anisotropic_diffusivity():
    d_a = np.diag([2.0, 4.0, 8.0])
    d_b = np.diag([10.0, 20.0, 40.0])

    np.testing.assert_allclose(
        diffusivity(0.25, d_a, d_b), np.diag([4.0, 8.0, 16.0])
    )


def test_mobility_converts_diffusivity():
    assert mobility(0.25, 2.0, 10.0, 2.0, c0=4.0, gas_constant=2.0) == pytest.approx(4.0)


def test_mobility_applies_optional_degeneracy():
    plain = mobility(0.25, 2.0, 10.0, 2.0)
    degenerate = mobility(0.25, 2.0, 10.0, 2.0, degeneracy=True)

    assert degenerate == pytest.approx(plain * 0.25 * 0.75)


def test_mobility_class_reuses_model_parameters():
    model = Mobility(2.0, 10.0, 2.0, c0=4.0, gas_constant=2.0)

    assert model.diffusivity(0.25) == pytest.approx(4.0)
    assert model.mobility(0.25) == pytest.approx(4.0)


def test_unknown_diffusivity_model_raises():
    with pytest.raises(ValueError, match="model"):
        diffusivity(0.5, 1.0, 1.0, model="unknown")


def test_darken_degeneracy_matches_cahn_taylor_peak_location():
    # fick.pdf, "Fick's law of diffusion": for the composition-dependent
    # interdiffusion mobility M = X(1-X)(X*M_j + (1-X)*M_k), the peak
    # occurs at a closed-form X depending only on zeta = M_k / M_j (Cahn,
    # J. et al. Eur. J. Appl. Math. 1996; Hillert 2008). model="darken"
    # with degeneracy=True is exactly this form with M_j, M_k passed as
    # d_b, d_a (c0 = gas_constant = 1 so mobility reduces to diffusivity).
    m_j, m_k = 1.0, 3.0
    zeta = m_k / m_j
    expected = (
        (4 * zeta - 2) - np.sqrt((2 - 4 * zeta) ** 2 - 4 * zeta * (3 * zeta - 3))
    ) / (2 * (3 * zeta - 3))

    grid = np.linspace(1e-4, 1 - 1e-4, 20001)
    values = mobility(
        grid, m_k, m_j, temperature=1.0, gas_constant=1.0, c0=1.0,
        degeneracy=True,
    )

    assert grid[np.argmax(values)] == pytest.approx(expected, abs=1e-3)


def test_darken_degeneracy_peak_at_half_for_equal_species_mobility():
    # zeta = 1 special case from the same closed-form result.
    grid = np.linspace(1e-4, 1 - 1e-4, 2001)
    values = mobility(
        grid, 2.0, 2.0, temperature=1.0, gas_constant=1.0, c0=1.0,
        degeneracy=True,
    )

    assert grid[np.argmax(values)] == pytest.approx(0.5, abs=1e-3)