import numpy as np
import pytest

from crystallite.mobility import BinaryMobility, Mobility, binary_mobility, diffusivity, mobility


def test_darken_diffusivity_uses_species_fractions():
    # Convention: x = X_B and X_A = 1 - x.
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
    # For the composition-dependent interdiffusion mobility
    # M = X(1-X)(X*M_j + (1-X)*M_k), the peak occurs at a closed-form X
    # depending only on zeta = M_k / M_j (Cahn et al., Eur. J. Appl. Math.
    # 1996; Hillert 2008). model="darken" with degeneracy=True is exactly
    # this form with M_j, M_k passed as d_b, d_a (c0 = gas_constant = 1 so
    # mobility reduces to diffusivity).
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


def test_binary_mobility_matches_hand_computed_cascade():
    x_b = 0.3
    x_a = 0.7
    d_a = x_a * 1.0 + x_b * 2.0  # D_A(xB) interpolated between D_A|A, D_A|B
    d_b = x_a * 3.0 + x_b * 4.0  # D_B(xB) interpolated between D_B|A, D_B|B
    d_dark = x_b * d_a + x_a * d_b  # Darken combination
    expected = x_a * x_b * d_dark  # RT = 1

    assert binary_mobility(
        x_b, 1.0, 2.0, 3.0, 4.0, temperature=1.0
    ) == pytest.approx(expected)


def test_binary_mobility_reduces_to_darken_mobility_for_fixed_species_diffusivities():
    # When each species' own diffusivity doesn't depend on composition
    # (D_A|A = D_A|B, D_B|A = D_B|B), the two-stage cascade collapses to
    # the plain Darken mobility() -- note mobility()'s (d_a, d_b) label the
    # species by their *own* fraction weight, the opposite convention from
    # binary_mobility's Darken weighting (xB weights D_A, xA weights D_B), so the
    # equivalent call swaps the two diffusivities.
    x_b = 0.3
    d_a, d_b = 1.5, 2.5

    cascade = binary_mobility(
        x_b, d_a, d_a, d_b, d_b, temperature=2.0, gas_constant=3.0
    )
    plain = mobility(
        x_b, d_b, d_a, temperature=2.0, gas_constant=3.0, degeneracy=True
    )

    assert cascade == pytest.approx(plain)


def test_binary_mobility_supports_tensor_diffusivities_with_a_field_composition():
    d_a = np.diag([1.0, 2.0, 3.0])
    d_b = np.diag([4.0, 5.0, 6.0])
    x_b = np.full((5, 5, 1), 0.3)

    result = binary_mobility(x_b, d_a, d_a, d_b, d_b, temperature=1.0)

    expected = binary_mobility(0.3, d_a, d_a, d_b, d_b, temperature=1.0)
    assert result.shape == (3, 3) + x_b.shape
    np.testing.assert_allclose(result[:, :, 0, 0, 0], expected)
    np.testing.assert_allclose(result[:, :, 4, 2, 0], expected)


def test_binary_mobility_class_reuses_model_parameters():
    model = BinaryMobility(
        d_a_in_a=1.0, d_a_in_b=2.0, d_b_in_a=3.0, d_b_in_b=4.0, temperature=1.0
    )

    assert model.mobility(0.3) == pytest.approx(
        binary_mobility(0.3, 1.0, 2.0, 3.0, 4.0, temperature=1.0)
    )