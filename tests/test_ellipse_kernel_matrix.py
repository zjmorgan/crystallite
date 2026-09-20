"""Stress-test matrix for the isolated elliptical-inhomogeneity kernels.

Every in-plane load family (uniform ``tension``/``biaxial``/``shear`` via the
equivalent-eigenstrain H/T-tensor, and the ``moment`` stress gradient via
the Muskhelishvili solution) is exposed through one adapter,
:func:`total_stress`, returning the *total* isolated-ellipse stress at any
point (interior and exterior, background included). The same physical
invariants are then checked over one shared grid of aspect ratios and
contrasts, so a bug in any family shows up as a failing cell, and a future
refactor onto a single kernel can be judged against the identical matrix.

Three further families are covered by the same style of matrix: antiplane
shear (the scalar conduction-type problem), and directly prescribed
uniform eigenstrains (dilatation, normal, shear, screw), which include
point defects and dislocation cores as the small ``a == b`` case -- see
:func:`test_scale_invariance_of_every_uniform_family`.

Invariants: identity at unit contrast, traction continuity across the
interface, pointwise equilibrium, far-field decay, traction-free void limit,
and reflection covariance about the diagonal. Independent oracles: the
Inglis void closed form (any aspect ratio) and the circular closed forms
(``a == b``).
"""

import functools

import numpy as np
import pytest

from crystallite.verification.elastic_deformation import (
    _antiplane_inhomogeneity_polar_stress,
    _antiplane_periodic_equivalent_eigenstrain,
    _dispatch_tension_polar_stress,
    _ellipse_inhomogeneity_antiplane_exterior_stress,
    _ellipse_inhomogeneity_antiplane_interior_stress,
    _ellipse_void_antiplane_cartesian_stress,
    polar_to_cartesian_antiplane_stress,
    _ellipse_equivalent_eigenstrain,
    _ellipse_gradient_inhomogeneity_correction_cartesian,
    _ellipse_gradient_inhomogeneity_solution,
    _ellipse_inhomogeneity_exterior_stress,
    _ellipse_inhomogeneity_interior_stress,
    _ellipse_void_cartesian_stress,
    _gradient_void_hole_correction_cartesian,
    _inhomogeneity_polar_stress,
    polar_to_cartesian_stress,
)

LAM0, MU0 = 1.0, 0.7
NU = LAM0 / (2.0 * (LAM0 + MU0))
CHARACTERISTIC = 0.05  # sqrt(a*b), fixed across aspect ratios

UNIFORM_LOADS = ("tension", "biaxial", "shear")
LOADS = UNIFORM_LOADS + ("moment",)
# a/b, spanning circle, mild, and slender (crack- and needle-like) in both orientations
ASPECT_RATIOS = (1.0, 0.5, 2.0, 0.125, 8.0, 30.0, 1.0 / 30.0, 100.0, 1.0 / 100.0, 1000.0)
# 0 is a void, 1 is "no inhomogeneity", inf is rigid, the rest soft/stiff
CONTRASTS = (0.0, 0.1, 0.5, 3.0, 100.0, float("inf"))


def semi_axes(ratio):
    return CHARACTERISTIC * np.sqrt(ratio), CHARACTERISTIC / np.sqrt(ratio)


def _magnitude(a, b):
    """Load scale giving O(1) stresses: uniform loads use 1, the gradient
    uses 1/sqrt(a*b) so ``magnitude * length`` is O(1) too."""
    return 1.0


def _gradient_magnitude(a, b):
    return 1.0 / np.sqrt(a * b)


@functools.lru_cache(maxsize=None)
def _moment_solution(a, b, contrast):
    return _ellipse_gradient_inhomogeneity_solution(
        a, b, contrast, NU, _gradient_magnitude(a, b)
    )


def background_stress(load, a, b, x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    zero = np.zeros_like(x)
    if load == "tension":
        return zero + 1.0, zero, zero
    if load == "biaxial":
        return zero + 1.0, zero + 1.0, zero
    if load == "shear":
        return zero, zero, zero + 1.0
    if load == "moment":
        return _gradient_magnitude(a, b) * y, zero, zero
    raise ValueError(load)


def total_stress(load, a, b, contrast, x, y):
    """Total isolated-ellipse stress ``(sxx, syy, sxy)`` at ``(x, y)`` (ellipse
    centered at the origin, semi-axis `a` along x), interior and exterior."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    inside = (x / a) ** 2 + (y / b) ** 2 < 1.0
    bx, by, bxy = background_stress(load, a, b, x, y)
    if load == "moment":
        cxx, cyy, cxy = (
            np.asarray(c)
            for c in _ellipse_gradient_inhomogeneity_correction_cartesian(
                x, y, _moment_solution(a, b, contrast)
            )
        )
        return bx + cxx, by + cyy, bxy + cxy

    eps_star = _ellipse_equivalent_eigenstrain(1.0, contrast, LAM0, MU0, load, a, b)
    # evaluate the exterior formula only at exterior points
    far = 10.0 * (a + b)
    xe, ye = np.where(inside, far, x), np.where(inside, far, y)
    ext = [np.asarray(c) for c in _ellipse_inhomogeneity_exterior_stress(xe, ye, eps_star, a, b, MU0, NU)]
    inn = [float(c) for c in _ellipse_inhomogeneity_interior_stress(eps_star, a, b, MU0, NU)]
    corr = [np.where(inside, i, e) for i, e in zip(inn, ext)]
    return bx + corr[0], by + corr[1], bxy + corr[2]


def correction_stress(load, a, b, contrast, x, y):
    t = total_stress(load, a, b, contrast, x, y)
    bg = background_stress(load, a, b, x, y)
    return tuple(ti - bi for ti, bi in zip(t, bg))


def _tip_radius(a, b):
    """Radius of curvature at the sharper tips, ``min^2/max``: the length over
    which the field varies there. Offsets and finite-difference steps must be
    small against it, not against the short semi-axis, or they straddle a real
    gradient (at 1000:1 the two differ by a factor of 1000)."""
    return min(a, b) ** 2 / max(a, b)


def _boundary(a, b, n=61):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + 0.013
    x, y = a * np.cos(t), b * np.sin(t)
    nx, ny = x / a**2, y / b**2
    norm = np.hypot(nx, ny)
    return x, y, nx / norm, ny / norm


def _traction(stress, nx, ny):
    sxx, syy, sxy = stress
    return sxx * nx + sxy * ny, sxy * nx + syy * ny


def _stress_scale(load, a, b, contrast):
    x, y, _, _ = _boundary(a, b)
    return max(np.abs(c).max() for c in total_stress(load, a, b, contrast, x * 1.02, y * 1.02))


# ---------------------------------------------------------------------------
# Invariants over the full (load, aspect ratio, contrast) matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("load", LOADS)
def test_unit_contrast_leaves_the_background_untouched(load, ratio):
    a, b = semi_axes(ratio)
    rng = np.random.default_rng(3)
    x, y = rng.uniform(-0.3, 0.3, 300), rng.uniform(-0.3, 0.3, 300)
    for c in correction_stress(load, a, b, 1.0, x, y):
        np.testing.assert_allclose(c, 0.0, atol=1e-9)


@pytest.mark.parametrize("contrast", CONTRASTS)
@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("load", LOADS)
def test_traction_is_continuous_across_the_interface(load, ratio, contrast):
    a, b = semi_axes(ratio)
    x, y, nx, ny = _boundary(a, b)
    eps = 1e-8 * _tip_radius(a, b)
    out = total_stress(load, a, b, contrast, x + eps * nx, y + eps * ny)
    inn = total_stress(load, a, b, contrast, x - eps * nx, y - eps * ny)
    scale = _stress_scale(load, a, b, contrast)
    # The rigid moment solve (free rigid-body motion, extra unknowns) reaches
    # ~3e-5 of the stress scale at 1000:1 -- a measured limit, not resolution
    # (4x the terms and collocation points does not improve it) -- so that
    # one corner has its own tolerance; everything else holds to 2e-5.
    extreme_rigid_moment = (
        load == "moment" and contrast == float("inf") and max(a / b, b / a) >= 300.0
    )
    tolerance = 1e-4 if extreme_rigid_moment else 2e-5
    for t_out, t_in in zip(_traction(out, nx, ny), _traction(inn, nx, ny)):
        np.testing.assert_allclose(t_out, t_in, atol=tolerance * scale)


@pytest.mark.parametrize("contrast", CONTRASTS)
@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("load", LOADS)
def test_field_is_in_pointwise_equilibrium(load, ratio, contrast):
    a, b = semi_axes(ratio)
    h = 1e-3 * min(a, b)  # noise ~ 1/h, so not tiny; truncation is negligible at this size
    rng = np.random.default_rng(7)
    ang = rng.uniform(0.0, 2.0 * np.pi, 8)
    # exterior points at several distances, plus interior points
    radii = np.array([1.3, 1.6, 2.0, 3.0, 5.0, 0.3, 0.5, 0.7])
    px, py = radii * a * np.cos(ang), radii * b * np.sin(ang)

    def s(dx, dy):
        return total_stress(load, a, b, contrast, px + dx, py + dy)

    div_x = (s(h, 0)[0] - s(-h, 0)[0] + s(0, h)[2] - s(0, -h)[2]) / (2 * h)
    div_y = (s(h, 0)[2] - s(-h, 0)[2] + s(0, h)[1] - s(0, -h)[1]) / (2 * h)
    scale = _stress_scale(load, a, b, contrast) / min(a, b)
    np.testing.assert_allclose(div_x, 0.0, atol=1e-5 * scale)
    np.testing.assert_allclose(div_y, 0.0, atol=1e-5 * scale)


@pytest.mark.parametrize("contrast", CONTRASTS)
@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("load", LOADS)
def test_correction_decays_far_from_the_inhomogeneity(load, ratio, contrast):
    a, b = semi_axes(ratio)
    x, y, _, _ = _boundary(a, b)
    near = max(np.abs(c).max() for c in correction_stress(load, a, b, contrast, 1.1 * x, 1.1 * y))
    far_r = 200.0 * max(a, b)
    ang = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False) + 0.1
    far = max(np.abs(c).max() for c in correction_stress(load, a, b, contrast, far_r * np.cos(ang), far_r * np.sin(ang)))
    # correction decays at least like r^-2: 200x farther -> < 1e-3 of the near value
    assert far < 1e-3 * near


@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("load", LOADS)
def test_void_limit_is_traction_free(load, ratio):
    a, b = semi_axes(ratio)
    x, y, nx, ny = _boundary(a, b)
    eps = 1e-8 * _tip_radius(a, b)
    out = total_stress(load, a, b, 0.0, x + eps * nx, y + eps * ny)
    scale = _stress_scale(load, a, b, 0.0)
    for t in _traction(out, nx, ny):
        np.testing.assert_allclose(t, 0.0, atol=2e-5 * scale)


@pytest.mark.parametrize("contrast", CONTRASTS)
@pytest.mark.parametrize("ratio", (1.0, 0.5, 2.0))
@pytest.mark.parametrize("load", ("biaxial", "shear"))
def test_covariance_under_reflection_about_the_diagonal(load, ratio, contrast):
    # reflecting x <-> y maps an (a, b) ellipse to (b, a); biaxial and shear
    # loads are invariant under it, so sxx <-> syy and sxy is unchanged
    a, b = semi_axes(ratio)
    rng = np.random.default_rng(11)
    x, y = rng.uniform(-0.2, 0.2, 200), rng.uniform(-0.2, 0.2, 200)
    sxx, syy, sxy = total_stress(load, a, b, contrast, x, y)
    rxx, ryy, rxy = total_stress(load, b, a, contrast, y, x)
    scale = max(np.abs(sxx).max(), np.abs(syy).max(), np.abs(sxy).max())
    np.testing.assert_allclose(sxx, ryy, atol=1e-9 * scale)
    np.testing.assert_allclose(syy, rxx, atol=1e-9 * scale)
    np.testing.assert_allclose(sxy, rxy, atol=1e-9 * scale)


# ---------------------------------------------------------------------------
# Independent oracles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("load", UNIFORM_LOADS)
def test_void_limit_matches_the_inglis_closed_form(load, ratio):
    a, b = semi_axes(ratio)
    ang = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False) + 0.05
    r = 2.0 * max(a, b)
    x, y = r * np.cos(ang), r * np.sin(ang)
    got = total_stress(load, a, b, 0.0, x, y)
    ref = _ellipse_void_cartesian_stress(x, y, a, b, load, 1.0)
    scale = max(np.abs(np.real(c)).max() for c in ref)
    for g, e in zip(got, ref):
        np.testing.assert_allclose(g, np.real(e), atol=1e-9 * scale)


@pytest.mark.parametrize("contrast", CONTRASTS)
@pytest.mark.parametrize("load", ("tension", "biaxial"))
def test_circle_matches_the_independent_circular_closed_form(load, contrast):
    a = b = CHARACTERISTIC
    ang = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False) + 0.05
    r = 2.5 * a
    x, y = r * np.cos(ang), r * np.sin(ang)
    got = total_stress(load, a, b, contrast, x, y)

    def formula(r_, theta_, radius, magnitude):
        return _inhomogeneity_polar_stress(r_, theta_, radius, magnitude, contrast, NU)

    srr, stt, srt = _dispatch_tension_polar_stress(formula, r, ang, a, 1.0, load)
    ref = polar_to_cartesian_stress(srr, stt, srt, ang)
    scale = max(np.abs(c).max() for c in ref)
    for g, e in zip(got, ref):
        np.testing.assert_allclose(g, e, atol=1e-9 * scale)


def test_circular_moment_void_matches_the_independent_closed_form():
    a = b = CHARACTERISTIC
    ang = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False) + 0.05
    r = 2.5 * a
    x, y = r * np.cos(ang), r * np.sin(ang)
    got = correction_stress("moment", a, b, 0.0, x, y)
    ref = _gradient_void_hole_correction_cartesian(x, y, a, _gradient_magnitude(a, b))
    scale = max(np.abs(np.asarray(c)).max() for c in ref)
    for g, e in zip(got, ref):
        np.testing.assert_allclose(g, np.asarray(e), atol=1e-8 * scale)


# ---------------------------------------------------------------------------
# Scale invariance: the isolated field depends only on x/a, y/b and a/b, so a
# point defect is simply the same solution at small a = b
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", (0.01, 30.0))
@pytest.mark.parametrize("contrast", (0.0, 0.5, 3.0, float("inf")))
@pytest.mark.parametrize("ratio", (1.0, 0.5, 8.0))
@pytest.mark.parametrize("load", LOADS)
def test_scale_invariance_of_every_uniform_family(load, ratio, contrast, scale):
    a, b = semi_axes(ratio)
    rng = np.random.default_rng(5)
    x, y = rng.uniform(-0.2, 0.2, 200), rng.uniform(-0.2, 0.2, 200)
    ref = total_stress(load, a, b, contrast, x, y)
    got = total_stress(load, scale * a, scale * b, contrast, scale * x, scale * y)
    norm = max(np.abs(c).max() for c in ref)
    for g, e in zip(got, ref):
        np.testing.assert_allclose(g, e, atol=1e-8 * norm)


# ---------------------------------------------------------------------------
# Antiplane shear (scalar, conduction-type): sigma_13 = magnitude far away
# ---------------------------------------------------------------------------

ANTIPLANE_CONTRASTS = CONTRASTS


def antiplane_total(a, b, contrast, x, y):
    """Total ``(sigma_13, sigma_23)`` of the isolated antiplane inhomogeneity
    under remote ``sigma_13 = 1`` (interior and exterior)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    inside = (x / a) ** 2 + (y / b) ** 2 < 1.0
    eps_star = _antiplane_periodic_equivalent_eigenstrain(1.0, contrast, MU0, b / (2.0 * (a + b)))
    far = 10.0 * (a + b)
    xe, ye = np.where(inside, far, x), np.where(inside, far, y)
    e13, e23 = (np.real(np.asarray(c)) for c in _ellipse_inhomogeneity_antiplane_exterior_stress(xe, ye, eps_star, a, b, MU0))
    i13, i23 = (float(c) for c in _ellipse_inhomogeneity_antiplane_interior_stress(eps_star, a, b, MU0))
    return 1.0 + np.where(inside, i13, e13), np.where(inside, i23, e23)


def _antiplane_correction(a, b, contrast, x, y):
    s13, s23 = antiplane_total(a, b, contrast, x, y)
    return s13 - 1.0, s23


def _antiplane_traction(stress, nx, ny):
    return stress[0] * nx + stress[1] * ny


def _antiplane_scale(a, b, contrast):
    x, y, _, _ = _boundary(a, b)
    return max(np.abs(c).max() for c in antiplane_total(a, b, contrast, 1.02 * x, 1.02 * y))


@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
def test_antiplane_unit_contrast_leaves_the_background_untouched(ratio):
    a, b = semi_axes(ratio)
    rng = np.random.default_rng(3)
    x, y = rng.uniform(-0.3, 0.3, 300), rng.uniform(-0.3, 0.3, 300)
    for c in _antiplane_correction(a, b, 1.0, x, y):
        np.testing.assert_allclose(c, 0.0, atol=1e-9)


@pytest.mark.parametrize("contrast", ANTIPLANE_CONTRASTS)
@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
def test_antiplane_traction_is_continuous_across_the_interface(ratio, contrast):
    a, b = semi_axes(ratio)
    x, y, nx, ny = _boundary(a, b)
    eps = 1e-8 * _tip_radius(a, b)
    out = antiplane_total(a, b, contrast, x + eps * nx, y + eps * ny)
    inn = antiplane_total(a, b, contrast, x - eps * nx, y - eps * ny)
    scale = _antiplane_scale(a, b, contrast)
    np.testing.assert_allclose(
        _antiplane_traction(out, nx, ny), _antiplane_traction(inn, nx, ny), atol=2e-5 * scale
    )


@pytest.mark.parametrize("contrast", ANTIPLANE_CONTRASTS)
@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
def test_antiplane_field_is_in_pointwise_equilibrium(ratio, contrast):
    a, b = semi_axes(ratio)
    h = 1e-3 * min(a, b)  # noise ~ 1/h, so not tiny; truncation is negligible at this size
    rng = np.random.default_rng(7)
    ang = rng.uniform(0.0, 2.0 * np.pi, 8)
    radii = np.array([1.3, 1.6, 2.0, 3.0, 5.0, 0.3, 0.5, 0.7])
    px, py = radii * a * np.cos(ang), radii * b * np.sin(ang)

    def s(dx, dy):
        return antiplane_total(a, b, contrast, px + dx, py + dy)

    div = (s(h, 0)[0] - s(-h, 0)[0] + s(0, h)[1] - s(0, -h)[1]) / (2 * h)
    np.testing.assert_allclose(div, 0.0, atol=1e-5 * _antiplane_scale(a, b, contrast) / min(a, b))


@pytest.mark.parametrize("contrast", ANTIPLANE_CONTRASTS)
@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
def test_antiplane_correction_decays_far_from_the_inhomogeneity(ratio, contrast):
    a, b = semi_axes(ratio)
    x, y, _, _ = _boundary(a, b)
    near = max(np.abs(c).max() for c in _antiplane_correction(a, b, contrast, 1.1 * x, 1.1 * y))
    far_r = 200.0 * max(a, b)
    ang = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False) + 0.1
    far = max(np.abs(c).max() for c in _antiplane_correction(a, b, contrast, far_r * np.cos(ang), far_r * np.sin(ang)))
    assert far < 1e-3 * near


@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
def test_antiplane_void_limit_is_traction_free(ratio):
    a, b = semi_axes(ratio)
    x, y, nx, ny = _boundary(a, b)
    eps = 1e-8 * _tip_radius(a, b)
    out = antiplane_total(a, b, 0.0, x + eps * nx, y + eps * ny)
    np.testing.assert_allclose(_antiplane_traction(out, nx, ny), 0.0, atol=2e-5 * _antiplane_scale(a, b, 0.0))


@pytest.mark.parametrize("contrast", ANTIPLANE_CONTRASTS)
@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
def test_antiplane_field_has_the_mirror_symmetries_of_the_load(ratio, contrast):
    # remote sigma_13: sigma_13 is even and sigma_23 odd under x -> -x and y -> -y
    a, b = semi_axes(ratio)
    rng = np.random.default_rng(13)
    x, y = rng.uniform(-0.2, 0.2, 200), rng.uniform(-0.2, 0.2, 200)
    s13, s23 = antiplane_total(a, b, contrast, x, y)
    for sx, sy in ((-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)):
        m13, m23 = antiplane_total(a, b, contrast, sx * x, sy * y)
        scale = max(np.abs(s13).max(), np.abs(s23).max())
        np.testing.assert_allclose(m13, s13, atol=1e-9 * scale)
        np.testing.assert_allclose(m23, sx * sy * s23, atol=1e-9 * scale)


@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
def test_antiplane_void_limit_matches_the_closed_form(ratio):
    a, b = semi_axes(ratio)
    ang = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False) + 0.05
    r = 2.0 * max(a, b)
    x, y = r * np.cos(ang), r * np.sin(ang)
    got = antiplane_total(a, b, 0.0, x, y)
    ref = _ellipse_void_antiplane_cartesian_stress(x, y, a, b, 1.0)
    scale = max(np.abs(np.real(c)).max() for c in ref)
    for g, e in zip(got, ref):
        np.testing.assert_allclose(g, np.real(e), atol=1e-9 * scale)


@pytest.mark.parametrize("contrast", CONTRASTS)
def test_antiplane_circle_matches_the_independent_circular_closed_form(contrast):
    a = b = CHARACTERISTIC
    ang = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False) + 0.05
    r = 2.5 * a
    x, y = r * np.cos(ang), r * np.sin(ang)
    got = antiplane_total(a, b, contrast, x, y)
    sr3, st3 = _antiplane_inhomogeneity_polar_stress(r, ang, a, 1.0, contrast)
    ref = polar_to_cartesian_antiplane_stress(sr3, st3, ang)
    scale = max(np.abs(c).max() for c in ref)
    for g, e in zip(got, ref):
        np.testing.assert_allclose(g, e, atol=1e-9 * scale)


@pytest.mark.parametrize("contrast", ANTIPLANE_CONTRASTS)
@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
def test_antiplane_interior_matches_the_classical_depolarization_formula(ratio, contrast):
    # conduction analogy: an ellipse of relative conductivity beta in a uniform
    # flux q along x carries the uniform flux q*beta*(a+b)/(a+beta*b)
    a, b = semi_axes(ratio)
    s13, s23 = antiplane_total(a, b, contrast, 0.1 * a, 0.2 * b)
    # beta -> inf limit of q*beta*(a+b)/(a+beta*b) is q*(a+b)/b
    expected = (a + b) / b if contrast == float("inf") else contrast * (a + b) / (a + contrast * b)
    np.testing.assert_allclose(s13, expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(s23, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Directly prescribed uniform eigenstrain (point defects, dislocation cores)
# ---------------------------------------------------------------------------

EIGEN_AMPLITUDE = 1.0e-3
EIGENSTRAINS = ("dilatation", "normal_x", "normal_y", "shear", "screw_x", "screw_y")


def eigenstrain_tensor(name):
    e = np.zeros((3, 3))
    if name == "dilatation":
        e[0, 0] = e[1, 1] = EIGEN_AMPLITUDE
    elif name == "normal_x":
        e[0, 0] = EIGEN_AMPLITUDE
    elif name == "normal_y":
        e[1, 1] = EIGEN_AMPLITUDE
    elif name == "shear":
        e[0, 1] = e[1, 0] = EIGEN_AMPLITUDE
    elif name == "screw_x":
        e[0, 2] = e[2, 0] = EIGEN_AMPLITUDE
    elif name == "screw_y":
        e[1, 2] = e[2, 1] = EIGEN_AMPLITUDE
    else:
        raise ValueError(name)
    return e


def eigen_stress(eps, a, b, x, y):
    """``(sxx, syy, sxy, s13, s23)`` of a uniform eigenstrain `eps` over an
    isolated ellipse in a homogeneous matrix (interior and exterior)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    inside = (x / a) ** 2 + (y / b) ** 2 < 1.0
    far = 10.0 * (a + b)
    xe, ye = np.where(inside, far, x), np.where(inside, far, y)
    ext_p = [np.real(np.asarray(c)) for c in _ellipse_inhomogeneity_exterior_stress(xe, ye, eps, a, b, MU0, NU)]
    int_p = [float(np.real(c)) for c in _ellipse_inhomogeneity_interior_stress(eps, a, b, MU0, NU)]
    ext_a = [np.real(np.asarray(c)) for c in _ellipse_inhomogeneity_antiplane_exterior_stress(xe, ye, eps, a, b, MU0)]
    int_a = [float(np.real(c)) for c in _ellipse_inhomogeneity_antiplane_interior_stress(eps, a, b, MU0)]
    return tuple(np.where(inside, i, e) for i, e in zip(int_p + int_a, ext_p + ext_a))


def _eigen_tractions(stress, nx, ny):
    sxx, syy, sxy, s13, s23 = stress
    return sxx * nx + sxy * ny, sxy * nx + syy * ny, s13 * nx + s23 * ny


def _eigen_scale(eps, a, b):
    x, y, _, _ = _boundary(a, b)
    return max(np.abs(c).max() for c in eigen_stress(eps, a, b, 1.02 * x, 1.02 * y))


@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("name", EIGENSTRAINS)
def test_eigenstrain_traction_is_continuous_across_the_interface(name, ratio):
    a, b = semi_axes(ratio)
    eps_t = eigenstrain_tensor(name)
    x, y, nx, ny = _boundary(a, b)
    eps = 1e-8 * _tip_radius(a, b)
    out = eigen_stress(eps_t, a, b, x + eps * nx, y + eps * ny)
    inn = eigen_stress(eps_t, a, b, x - eps * nx, y - eps * ny)
    scale = _eigen_scale(eps_t, a, b)
    for t_out, t_in in zip(_eigen_tractions(out, nx, ny), _eigen_tractions(inn, nx, ny)):
        np.testing.assert_allclose(t_out, t_in, atol=2e-5 * scale)


@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("name", EIGENSTRAINS)
def test_eigenstrain_field_is_in_pointwise_equilibrium(name, ratio):
    a, b = semi_axes(ratio)
    eps_t = eigenstrain_tensor(name)
    h = 1e-3 * min(a, b)  # noise ~ 1/h, so not tiny; truncation is negligible at this size
    rng = np.random.default_rng(7)
    ang = rng.uniform(0.0, 2.0 * np.pi, 8)
    radii = np.array([1.3, 1.6, 2.0, 3.0, 5.0, 0.3, 0.5, 0.7])
    px, py = radii * a * np.cos(ang), radii * b * np.sin(ang)

    def s(dx, dy):
        return eigen_stress(eps_t, a, b, px + dx, py + dy)

    div_x = (s(h, 0)[0] - s(-h, 0)[0] + s(0, h)[2] - s(0, -h)[2]) / (2 * h)
    div_y = (s(h, 0)[2] - s(-h, 0)[2] + s(0, h)[1] - s(0, -h)[1]) / (2 * h)
    div_z = (s(h, 0)[3] - s(-h, 0)[3] + s(0, h)[4] - s(0, -h)[4]) / (2 * h)
    scale = _eigen_scale(eps_t, a, b) / min(a, b)
    for d in (div_x, div_y, div_z):
        np.testing.assert_allclose(d, 0.0, atol=1e-5 * scale)


@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("name", EIGENSTRAINS)
def test_eigenstrain_field_decays_far_from_the_region(name, ratio):
    a, b = semi_axes(ratio)
    eps_t = eigenstrain_tensor(name)
    x, y, _, _ = _boundary(a, b)
    near = max(np.abs(c).max() for c in eigen_stress(eps_t, a, b, 1.1 * x, 1.1 * y))
    far_r = 200.0 * max(a, b)
    ang = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False) + 0.1
    far = max(np.abs(c).max() for c in eigen_stress(eps_t, a, b, far_r * np.cos(ang), far_r * np.sin(ang)))
    assert far < 1e-3 * near


@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
def test_eigenstrain_field_is_linear_in_the_eigenstrain(ratio):
    a, b = semi_axes(ratio)
    rng = np.random.default_rng(17)
    x, y = rng.uniform(-0.2, 0.2, 200), rng.uniform(-0.2, 0.2, 200)
    combo = sum(w * eigenstrain_tensor(n) for w, n in zip((1.0, -0.6, 0.3, 2.0, 0.7, -1.1), EIGENSTRAINS))
    got = eigen_stress(combo, a, b, x, y)
    parts = [eigen_stress(w * eigenstrain_tensor(n), a, b, x, y) for w, n in zip((1.0, -0.6, 0.3, 2.0, 0.7, -1.1), EIGENSTRAINS)]
    for k in range(5):
        expected = sum(p[k] for p in parts)
        np.testing.assert_allclose(got[k], expected, atol=1e-9 * max(np.abs(expected).max(), 1e-12))


@pytest.mark.parametrize("scale", (0.001, 0.01, 30.0))
@pytest.mark.parametrize("ratio", (1.0, 0.5, 8.0))
@pytest.mark.parametrize("name", EIGENSTRAINS)
def test_eigenstrain_field_is_scale_invariant(name, ratio, scale):
    a, b = semi_axes(ratio)
    eps_t = eigenstrain_tensor(name)
    rng = np.random.default_rng(5)
    x, y = rng.uniform(-0.2, 0.2, 200), rng.uniform(-0.2, 0.2, 200)
    ref = eigen_stress(eps_t, a, b, x, y)
    got = eigen_stress(eps_t, scale * a, scale * b, scale * x, scale * y)
    norm = max(np.abs(c).max() for c in ref)
    for g, e in zip(got, ref):
        np.testing.assert_allclose(g, e, atol=1e-8 * norm)


@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("name", ("dilatation", "normal_x", "normal_y", "shear"))
def test_eigenstrain_interior_matches_the_mura_s_tensor(name, ratio):
    # independent oracle: transcribed Eshelby S-tensor of an elliptical
    # cylinder (Mura, sec. 11.2), sigma_in = C0:(S:eps* - eps*), plane strain
    a, b = semi_axes(ratio)
    e = eigenstrain_tensor(name)
    d = 2.0 * (1.0 - NU)
    s1111 = ((b**2 + 2 * a * b) / (a + b) ** 2 + (1 - 2 * NU) * b / (a + b)) / d
    s2222 = ((a**2 + 2 * a * b) / (a + b) ** 2 + (1 - 2 * NU) * a / (a + b)) / d
    s1122 = (b**2 / (a + b) ** 2 - (1 - 2 * NU) * b / (a + b)) / d
    s2211 = (a**2 / (a + b) ** 2 - (1 - 2 * NU) * a / (a + b)) / d
    s1212 = ((a**2 + b**2) / (2 * (a + b) ** 2) + (1 - 2 * NU) / 2) / d
    in11 = s1111 * e[0, 0] + s1122 * e[1, 1]
    in22 = s2211 * e[0, 0] + s2222 * e[1, 1]
    in12 = 2.0 * s1212 * e[0, 1]
    tr = (in11 - e[0, 0]) + (in22 - e[1, 1])
    expected = (
        LAM0 * tr + 2 * MU0 * (in11 - e[0, 0]),
        LAM0 * tr + 2 * MU0 * (in22 - e[1, 1]),
        2 * MU0 * (in12 - e[0, 1]),
    )
    got = eigen_stress(e, a, b, np.array([0.1 * a]), np.array([0.2 * b]))[:3]
    for g, x_ in zip(got, expected):
        np.testing.assert_allclose(g, x_, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
def test_eigenstrain_antiplane_interior_matches_the_depolarization_factors(ratio):
    a, b = semi_axes(ratio)
    got_x = eigen_stress(eigenstrain_tensor("screw_x"), a, b, np.array([0.1 * a]), np.array([0.2 * b]))
    got_y = eigen_stress(eigenstrain_tensor("screw_y"), a, b, np.array([0.1 * a]), np.array([0.2 * b]))
    np.testing.assert_allclose(got_x[3], -2 * MU0 * EIGEN_AMPLITUDE * a / (a + b), rtol=1e-9)
    np.testing.assert_allclose(got_y[4], -2 * MU0 * EIGEN_AMPLITUDE * b / (a + b), rtol=1e-9)


@pytest.mark.parametrize("radius", (1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2))
def test_point_defect_dilatation_matches_the_elementary_circular_inclusion(radius):
    # plane-strain circle with in-plane dilatation e: interior sigma = -mu*e/(1-nu),
    # exterior sigma_rr = -sigma_tt = -mu*e*a^2/((1-nu)*r^2); elementary
    # displacement/traction matching, independent of the H/T-tensor code
    a = b = radius
    e = EIGEN_AMPLITUDE
    eps_t = eigenstrain_tensor("dilatation")
    p = MU0 * e / (1.0 - NU)
    inner = eigen_stress(eps_t, a, b, np.array([0.3 * a]), np.array([0.4 * a]))
    np.testing.assert_allclose(inner[0], -p, rtol=1e-9)
    np.testing.assert_allclose(inner[1], -p, rtol=1e-9)
    np.testing.assert_allclose(inner[2], 0.0, atol=1e-9 * p)

    ang = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False) + 0.1
    for mult in (1.5, 3.0, 10.0):
        r = mult * a
        sxx, syy, sxy, _, _ = eigen_stress(eps_t, a, b, r * np.cos(ang), r * np.sin(ang))
        c, s_ = np.cos(ang), np.sin(ang)
        srr = sxx * c**2 + syy * s_**2 + 2 * sxy * s_ * c
        stt = sxx * s_**2 + syy * c**2 - 2 * sxy * s_ * c
        np.testing.assert_allclose(srr, -p * (a / r) ** 2, rtol=1e-8)
        np.testing.assert_allclose(stt, p * (a / r) ** 2, rtol=1e-8)


# ---------------------------------------------------------------------------
# Rigid limit: contrast = inf must be the limit of large finite contrasts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", (1.0, 0.5, 8.0))
@pytest.mark.parametrize("load", LOADS)
def test_rigid_inclusion_is_the_limit_of_a_very_large_finite_contrast(load, ratio):
    a, b = semi_axes(ratio)
    rng = np.random.default_rng(2)
    x, y = rng.uniform(-0.2, 0.2, 100), rng.uniform(-0.2, 0.2, 100)
    rigid = total_stress(load, a, b, float("inf"), x, y)
    stiff = total_stress(load, a, b, 1.0e9, x, y)
    scale = max(np.abs(c).max() for c in stiff)
    for r, f in zip(rigid, stiff):
        np.testing.assert_allclose(r, f, atol=1e-6 * scale)


@pytest.mark.parametrize("ratio", (1.0, 0.5))
def test_moment_solution_converges_to_the_rigid_limit_like_one_over_contrast(ratio):
    # regression: a plain beta*D_e - D_i displacement row loses the traction
    # rows to round-off near beta ~ 1e9 and the solution blew up
    a, b = semi_axes(ratio)
    rng = np.random.default_rng(2)
    x, y = rng.uniform(-0.2, 0.2, 100), rng.uniform(-0.2, 0.2, 100)
    rigid = total_stress("moment", a, b, float("inf"), x, y)
    scale = max(np.abs(c).max() for c in rigid)
    previous = None
    for contrast in (1.0e2, 1.0e4, 1.0e6, 1.0e9, 1.0e12):
        stiff = total_stress("moment", a, b, contrast, x, y)
        error = max(np.abs(u - v).max() for u, v in zip(stiff, rigid)) / scale
        assert error < 6.0 / contrast + 1e-7
        previous = error
    assert previous < 1e-9
