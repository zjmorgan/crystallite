import numpy as np
import pytest

from crystallite.mobility import Mobility, diffusivity, mobility


def test_darken_diffusivity_uses_species_fractions():
    assert diffusivity(0.25, 2.0, 10.0) == pytest.approx(4.0)


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