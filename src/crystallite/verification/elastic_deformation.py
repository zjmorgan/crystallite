"""Verification case for heterogeneous elasticity: the Kirsch problem."""

import functools
from dataclasses import dataclass

import numpy as np
from scipy.special import j1

from crystallite.backend import xp
from crystallite.elastic_deformation import ElasticDeformation
from crystallite.grid import Grid


def _tension_polar_stress(r, theta, hole_radius, magnitude):
    r"""Kirsch's closed-form stress field around a circular hole of radius
    `hole_radius` in an infinite isotropic plate under remote uniaxial
    tension `magnitude` along the polar axis :math:`\theta=0`.

    Independent of the elastic constants -- the classic, remarkable
    property of this solution. A negative `magnitude` gives the
    compression case directly (same closed form).
    """
    a_over_r_sq = (hole_radius / r) ** 2
    a_over_r_4 = a_over_r_sq**2
    cos2 = xp.cos(2.0 * theta)
    sin2 = xp.sin(2.0 * theta)

    sigma_rr = 0.5 * magnitude * (1.0 - a_over_r_sq) + 0.5 * magnitude * (
        1.0 - 4.0 * a_over_r_sq + 3.0 * a_over_r_4
    ) * cos2
    sigma_theta_theta = 0.5 * magnitude * (1.0 + a_over_r_sq) - 0.5 * magnitude * (
        1.0 + 3.0 * a_over_r_4
    ) * cos2
    sigma_r_theta = -0.5 * magnitude * (
        1.0 + 2.0 * a_over_r_sq - 3.0 * a_over_r_4
    ) * sin2

    return sigma_rr, sigma_theta_theta, sigma_r_theta


def _inhomogeneity_polar_stress(r, theta, hole_radius, magnitude, contrast, nu):
    r"""Exact closed-form stress field around an isolated, finite-contrast
    circular inhomogeneity (not just the void limit) in an infinite
    isotropic plate under remote uniaxial tension `magnitude`.

    `contrast` is the inhomogeneity-to-matrix *shear-modulus* ratio (this
    project's own convention of scaling both Lame parameters by the same
    factor keeps Poisson's ratio identical in both phases, exactly the
    assumption this classical solution needs). Reduces to
    :func:`_tension_polar_stress` exactly at ``contrast=0`` -- verified
    directly (both by hand and numerically): the ``beta=0`` limit of `A`,
    `B`, `C` below is ``2, 1, -1`` (not ``2/(kappa-1)``, an arithmetic slip
    caught by checking `sigma_rr(hole_radius, *)` comes out identically
    zero, as it must for a traction-free void, for every `kappa`).

    ``contrast=float('inf')`` (a rigid inclusion) is also accepted, using
    the closed-form ``beta -> infinity`` limit directly (``a,b,c ->
    -2/kappa, (1-kappa)/2, 1/kappa``) rather than substituting a literal
    ``inf`` into the finite-contrast formula above, which hits an
    indeterminate ``inf/inf``. Independently re-derived from scratch (a
    rigid inclusion's exact bonded-displacement solution, not a limit of
    this formula) and checked three ways: the displacement matches the
    ``u_r=u_theta=0`` rigid boundary condition exactly; the field is in
    full pointwise equilibrium (finite differences); and it matches *this*
    function evaluated at a merely large finite `contrast` (``1e8``) to 4
    decimal places -- confirming the two independent derivations agree.
    """
    beta = contrast
    kappa = 3.0 - 4.0 * nu
    if beta == float("inf"):
        a = -2.0 / kappa
        b = (1.0 - kappa) / 2.0
        c = 1.0 / kappa
    else:
        a = 2.0 * (1.0 - beta) / (beta * kappa + 1.0)
        b = (kappa - 1.0) * (1.0 - beta) / (2.0 * beta + kappa - 1.0)
        c = (beta - 1.0) / (beta * kappa + 1.0)

    a_over_r_sq = (hole_radius / r) ** 2
    a_over_r_4 = a_over_r_sq**2
    cos2 = xp.cos(2.0 * theta)
    sin2 = xp.sin(2.0 * theta)

    sigma_rr = 0.5 * magnitude * (1.0 - b * a_over_r_sq) + 0.5 * magnitude * (
        1.0 - 2.0 * a * a_over_r_sq - 3.0 * c * a_over_r_4
    ) * cos2
    sigma_theta_theta = 0.5 * magnitude * (1.0 + b * a_over_r_sq) - 0.5 * magnitude * (
        1.0 - 3.0 * c * a_over_r_4
    ) * cos2
    sigma_r_theta = -0.5 * magnitude * (
        1.0 + a * a_over_r_sq + 3.0 * c * a_over_r_4
    ) * sin2

    return sigma_rr, sigma_theta_theta, sigma_r_theta


def _dispatch_tension_polar_stress(formula, r, theta, hole_radius, magnitude, load):
    r"""Combine an isolated uniaxial-tension polar stress solution
    ``formula(r, theta, hole_radius, magnitude)`` into any of the four
    uniform load cases via the standard tension-superposition trick
    (rotate/negate and add: pure shear is +/-tension at +/-45 degrees,
    biaxial tension is tension at 0 and 90 degrees) -- shared by
    :func:`_tension_polar_stress`-based
    :meth:`HoleInPlateCase.analytic_stress` (the void limit) and
    :func:`_inhomogeneity_polar_stress`-based
    :meth:`HoleInPlateCase.inhomogeneity_analytic_stress` (any finite
    `contrast`), so the dispatch logic itself is written once.
    """
    if load == "tension":
        return formula(r, theta, hole_radius, magnitude)
    if load == "compression":
        return formula(r, theta, hole_radius, -magnitude)
    if load == "shear":
        quarter_pi = xp.pi / 4.0
        rr1, tt1, rt1 = formula(r, theta - quarter_pi, hole_radius, magnitude)
        rr2, tt2, rt2 = formula(r, theta + quarter_pi, hole_radius, -magnitude)
        return rr1 + rr2, tt1 + tt2, rt1 + rt2
    if load == "biaxial":
        rr1, tt1, rt1 = formula(r, theta, hole_radius, magnitude)
        rr2, tt2, rt2 = formula(r, theta - xp.pi / 2.0, hole_radius, magnitude)
        return rr1 + rr2, tt1 + tt2, rt1 + rt2
    raise ValueError('load must be "tension", "compression", "shear", or "biaxial"')


def _inhomogeneity_interior_stress(magnitude, contrast, nu):
    r"""Uniform interior Cartesian stress inside a circular inhomogeneity
    under remote uniaxial tension `magnitude` (tension-aligned frame) --
    the classical result that the internal state of a circular (or
    elliptical) inhomogeneity is exactly uniform, regardless of position.

    ``sigma_zz`` is the plane-strain out-of-plane stress, fixed by
    ``eps_zz=0``. ``contrast=float('inf')`` (rigid inclusion) uses the
    closed-form ``beta -> infinity`` limit directly (``a_i,c_i ->
    (kappa+1)/2, (kappa+1)/kappa``), avoiding an ``inf/inf``
    indeterminate -- see :func:`_inhomogeneity_polar_stress`.
    """
    beta = contrast
    kappa = 3.0 - 4.0 * nu
    if beta == float("inf"):
        a_i = (kappa + 1.0) / 2.0
        c_i = (kappa + 1.0) / kappa
    else:
        a_i = beta * (kappa + 1.0) / (2.0 * beta + kappa - 1.0)
        c_i = beta * (kappa + 1.0) / (beta * kappa + 1.0)
    sigma_xx = 0.5 * magnitude * (a_i + c_i)
    sigma_yy = 0.5 * magnitude * (a_i - c_i)
    sigma_zz = nu * (sigma_xx + sigma_yy)
    return sigma_xx, sigma_yy, sigma_zz


def _isotropic_compliance_apply(sigma, lam, mu):
    """Return ``eps = S:sigma`` for an isotropic material, ``sigma`` a
    ``(3, 3)`` tensor."""
    sigma = xp.asarray(sigma)
    young = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    poisson = lam / (2.0 * (lam + mu))
    trace = sigma[0, 0] + sigma[1, 1] + sigma[2, 2]
    return (1.0 + poisson) / young * sigma - poisson / young * trace * xp.eye(
        3, dtype=sigma.dtype
    )


def _isotropic_stress_apply(strain, lam, mu):
    """Return ``sigma = C:strain`` for an isotropic material, ``strain`` a
    ``(3, 3)`` tensor."""
    strain = xp.asarray(strain)
    trace = strain[0, 0] + strain[1, 1] + strain[2, 2]
    return lam * trace * xp.eye(3, dtype=strain.dtype) + 2.0 * mu * strain


def _rotate_tensor_2d(tensor, angle):
    """Rotate a ``(3, 3)`` tensor by `angle` about the z-axis."""
    cos_a, sin_a = xp.cos(angle), xp.sin(angle)
    rotation = xp.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=tensor.dtype,
    )
    return rotation @ tensor @ rotation.T


def _inhomogeneity_interior_cartesian(magnitude, contrast, nu, load):
    r"""Uniform interior Cartesian stress tensor inside a circular
    inhomogeneity under `load`/`magnitude`, from
    :func:`_inhomogeneity_interior_stress` (the tension-aligned closed
    form) combined via the same tension-superposition trick as
    :func:`_dispatch_tension_polar_stress` -- shared by
    :func:`_equivalent_eigenstrain` and
    :func:`_inhomogeneity_correction_cartesian`'s interior term.

    Rotating/negating and summing the *stress* tensor this way, rather
    than solving each load case's equivalent-inclusion problem directly,
    is valid because the tension-to-eigenstrain map an isotropic
    compliance defines is linear and rotationally equivariant -- exactly
    why :func:`_equivalent_eigenstrain` used to build this same
    combination out of *eigenstrains* instead and got an identical
    answer; this function just does the combining one step earlier, on
    the interior stress the eigenstrain is derived from.
    """

    def tension_sigma_in(signed_magnitude):
        sigma_xx, sigma_yy, sigma_zz = _inhomogeneity_interior_stress(
            signed_magnitude, contrast, nu
        )
        return xp.array(
            [[sigma_xx, 0.0, 0.0], [0.0, sigma_yy, 0.0], [0.0, 0.0, sigma_zz]]
        )

    if load == "tension":
        return tension_sigma_in(magnitude)
    if load == "compression":
        return tension_sigma_in(-magnitude)
    if load == "shear":
        quarter_pi = xp.pi / 4.0
        return _rotate_tensor_2d(
            tension_sigma_in(magnitude), quarter_pi
        ) + _rotate_tensor_2d(tension_sigma_in(-magnitude), -quarter_pi)
    if load == "biaxial":
        sigma_x = tension_sigma_in(magnitude)
        return sigma_x + _rotate_tensor_2d(sigma_x, xp.pi / 2.0)
    raise ValueError('load must be "tension", "compression", "shear", or "biaxial"')


def _equivalent_eigenstrain(magnitude, contrast, matrix_lame_lambda, matrix_lame_mu, load):
    r"""Eshelby equivalent eigenstrain for a circular inhomogeneity: the
    uniform eigenstrain :math:`\varepsilon^*` a *matrix-material* disk
    would need, in place of the true (different-stiffness) inhomogeneity,
    to reproduce the identical stress state everywhere.

    From the equivalent-inclusion condition
    :math:`\sigma_{\mathrm{in}} = C_0:(\varepsilon_{\mathrm{in}} -
    \varepsilon^*)`: :math:`\varepsilon^* = \varepsilon_{\mathrm{in}} -
    S_0:\sigma_{\mathrm{in}}`, using the *true* interior strain (from the
    inhomogeneity's own compliance) and the matrix's compliance ``S_0``
    applied to that same (closed-form, uniform) interior stress
    (:func:`_inhomogeneity_interior_cartesian`, already combined for
    `load`).
    """
    nu = matrix_lame_lambda / (2.0 * (matrix_lame_lambda + matrix_lame_mu))
    sigma_in = _inhomogeneity_interior_cartesian(magnitude, contrast, nu, load)
    if contrast == float("inf"):
        # A rigid inclusion has zero strain under any finite stress -- the
        # eps_in/S0:sigma_in limit directly (substituting a literal inf
        # into the compliance below hits inf/inf).
        true_strain = xp.zeros((3, 3))
    else:
        true_strain = _isotropic_compliance_apply(
            sigma_in, contrast * matrix_lame_lambda, contrast * matrix_lame_mu
        )
    matrix_strain = _isotropic_compliance_apply(
        sigma_in, matrix_lame_lambda, matrix_lame_mu
    )
    return true_strain - matrix_strain


def _ellipse_equivalent_eigenstrain(
    magnitude, contrast, matrix_lame_lambda, matrix_lame_mu, load, semi_axis_a, semi_axis_b
):
    r"""Eshelby equivalent eigenstrain for a general (:math:`a\ne b`)
    elliptical inhomogeneity -- the elliptical counterpart of
    :func:`_equivalent_eigenstrain`, needed because a non-circular shape
    has no rotational symmetry to exploit (the circular function's
    ``"shear"``/``"biaxial"`` cases are built by rotating a single
    ``"tension"`` solution; that trick is unavailable here, so each load
    case is solved directly).

    Uses the classical elliptical-cylinder (plane strain) Eshelby
    S-tensor (`semi_axis_a` along x1, `semi_axis_b` along x2; e.g. Mura,
    *Micromechanics of Defects in Solids*, sec. 11.2, and this project's
    own reference derivation in ``equivalent.inhomogeneity.sage.py``):

    .. math::

        S_{1111}=\frac{1}{2(1-\nu)}\!\left[\frac{b^2+2ab}{(a+b)^2}+(1-2\nu)\frac{b}{a+b}\right],
        \quad S_{2222}=\frac{1}{2(1-\nu)}\!\left[\frac{a^2+2ab}{(a+b)^2}+(1-2\nu)\frac{a}{a+b}\right]

        S_{1122}=\frac{1}{2(1-\nu)}\!\left[\frac{b^2}{(a+b)^2}-(1-2\nu)\frac{b}{a+b}\right],
        \quad S_{2211}=\frac{1}{2(1-\nu)}\!\left[\frac{a^2}{(a+b)^2}-(1-2\nu)\frac{a}{a+b}\right]

        S_{1212}=\frac{1}{2(1-\nu)}\!\left[\frac{a^2+b^2}{2(a+b)^2}+\frac{1-2\nu}{2}\right]

    with :math:`S_{3jkl}=S_{i3kl}=0` (plane strain: the transformation
    strain has no direct :math:`\varepsilon_{33}` component, and the
    remote/interior :math:`\varepsilon_{33}=0` constraint is enforced
    directly rather than through the S-tensor).

    The equivalent-inclusion condition
    :math:`\sigma_{\mathrm{true}}(\varepsilon_{\mathrm{in}}) =
    \sigma_{\mathrm{equiv}}(\varepsilon_{\mathrm{in}}-\varepsilon^*)`,
    with :math:`\varepsilon_{\mathrm{in}}=\bar\varepsilon+S:\varepsilon^*`,
    gives a small linear system for :math:`\varepsilon^*` -- solved here
    numerically (a 3x3 solve for ``"tension"``/``"biaxial"`` in
    :math:`\varepsilon^*_{11},\varepsilon^*_{22},\varepsilon^*_{33}`,
    decoupled by symmetry from a scalar solve for ``"shear"``'s
    :math:`\varepsilon^*_{12}`) rather than transcribed symbolically --
    the closed forms are unwieldy rational polynomials in ``a``, ``b``,
    `contrast`, `nu` (obtained and inspected via sympy while deriving
    this), too easy to mistranscribe, whereas a 3x3 (or 1x1) linear solve
    at call time carries no such risk and costs nothing.

    Verified several ways: at ``semi_axis_a == semi_axis_b`` this
    reproduces :func:`_equivalent_eigenstrain` exactly (all three load
    cases, several `contrast` values spanning soft/hard/rigid-adjacent) --
    the two functions share no code, so this is a genuine independent
    check, not a tautology. Also: :math:`\varepsilon^*_{33}` comes out
    (numerically) exactly zero for every ``a``, ``b``, `contrast` tried,
    as it must by the problem's up-down and left-right mirror symmetry
    (an in-plane-only load on an axis-aligned ellipse cannot prefer a
    nonzero out-of-plane transformation strain) -- confirmed as an
    emergent property of the solve, not assumed.

    Unlike :func:`_equivalent_eigenstrain`, `load="compression"` is not
    offered: :class:`EllipticalHoleInPlateCase` (the only caller) has no
    such case, since a non-circular shape has no rotational symmetry to
    make "compression" anything other than "tension" with a sign flip
    the caller can already apply directly to `magnitude`.
    """
    nu = matrix_lame_lambda / (2.0 * (matrix_lame_lambda + matrix_lame_mu))
    lam0, mu0 = matrix_lame_lambda, matrix_lame_mu
    rigid = contrast == float("inf")
    lam1, mu1 = (0.0, 0.0) if rigid else (contrast * lam0, contrast * mu0)
    a, b = semi_axis_a, semi_axis_b

    denom = 2.0 * (1.0 - nu)
    s1111 = ((b**2 + 2.0 * a * b) / (a + b) ** 2 + (1.0 - 2.0 * nu) * b / (a + b)) / denom
    s2222 = ((a**2 + 2.0 * a * b) / (a + b) ** 2 + (1.0 - 2.0 * nu) * a / (a + b)) / denom
    s1122 = (b**2 / (a + b) ** 2 - (1.0 - 2.0 * nu) * b / (a + b)) / denom
    s2211 = (a**2 / (a + b) ** 2 - (1.0 - 2.0 * nu) * a / (a + b)) / denom
    s1212 = ((a**2 + b**2) / (2.0 * (a + b) ** 2) + (1.0 - 2.0 * nu) / 2.0) / denom

    def normal_eigenstrain(sbar11, sbar22):
        sbar33 = nu * (sbar11 + sbar22)
        young0 = 2.0 * mu0 * (1.0 + nu)
        eps_bar_11 = (sbar11 - nu * sbar22 - nu * sbar33) / young0
        eps_bar_22 = (sbar22 - nu * sbar11 - nu * sbar33) / young0
        eps_bar = xp.array([eps_bar_11, eps_bar_22, 0.0])

        s_mat = xp.array([[s1111, s1122, 0.0], [s2211, s2222, 0.0], [0.0, 0.0, 0.0]])
        if rigid:
            # contrast -> inf: the interior strain vanishes, eps_bar + S:eps* = 0
            # (eps*_33 stays 0, as in the finite-contrast solve)
            es_in_plane = xp.linalg.solve(s_mat[:2, :2], -eps_bar[:2])
            return xp.array([es_in_plane[0], es_in_plane[1], 0.0])
        identity3 = xp.eye(3)
        ones3 = xp.ones((3, 3))
        c1_mat = lam1 * ones3 + 2.0 * mu1 * identity3
        c0_mat = lam0 * ones3 + 2.0 * mu0 * identity3
        a_mat = c1_mat @ s_mat - c0_mat @ (s_mat - identity3)
        rhs = (c0_mat - c1_mat) @ eps_bar
        return xp.linalg.solve(a_mat, rhs)

    if load == "tension":
        es = normal_eigenstrain(magnitude, 0.0)
    elif load == "biaxial":
        es = normal_eigenstrain(magnitude, magnitude)
    elif load == "shear":
        eps_bar_12 = magnitude / (2.0 * mu0)
        s_eff = 2.0 * s1212
        if rigid:
            es12 = -eps_bar_12 / s_eff
        else:
            es12 = eps_bar_12 * (mu0 - mu1) / (mu1 * s_eff - mu0 * s_eff + mu0)
        eps_star = xp.zeros((3, 3))
        eps_star[0, 1] = eps_star[1, 0] = es12
        return eps_star
    else:
        raise ValueError('load must be "tension", "biaxial", or "shear"')

    eps_star = xp.zeros((3, 3))
    eps_star[0, 0], eps_star[1, 1], eps_star[2, 2] = es[0], es[1], es[2]
    return eps_star


def _periodic_eshelby_tensor(prescribed_eigenstrain_solution, inside_mask):
    r"""Numerically probe the *periodic* (lattice-corrected) Eshelby
    tensor components ``s1111, s2211, s1122, s2222, s1212`` for whatever
    actual domain/shape/spacing `prescribed_eigenstrain_solution` solves
    -- the periodic analogue of the closed-form isolated S-tensor used by
    :func:`_ellipse_equivalent_eigenstrain`.

    Unlike the isolated case (a universal closed form depending only on
    the shape and Poisson's ratio), a periodic array's Eshelby tensor
    also depends on how large the region is relative to the periodic
    cell -- each copy of the region feels the stress radiated by its own
    periodic images, not just a uniform remote field. That self-
    interaction is exactly the "periodic extension" a dilute-limit
    (isolated-formula) calibration misses once the region is not small
    compared to the cell (see :func:`_equivalent_eigenstrain`'s use in
    :meth:`HoleInPlateCase.periodic_analytic_solution` prior to this
    function's introduction, which used the isolated closed form
    regardless of how dense the array actually was).

    Obtained by reusing this project's own exact periodic
    Khachaturyan-Shatalov solve (`prescribed_eigenstrain_solution`, e.g.
    :meth:`HoleInPlateCase.periodic_prescribed_eigenstrain_solution`) for
    a unit probe eigenstrain, then averaging the resulting strain over
    `inside_mask` -- by the periodic generalization of Eshelby's
    uniformity theorem (a periodic array of identical uniform-eigenstrain
    ellipsoidal regions in an otherwise homogeneous medium still has
    *exactly* uniform strain inside each region, not just approximately;
    the averaging here is purely to suppress the sharp-boundary Gibbs
    ringing of the discrete Fourier representation, not because the
    continuum answer actually varies over the region).

    Only ``s1111, s2211, s1122, s2222, s1212`` are returned --
    ``s1133, s2233, s3311, s3322`` are not probed, assumed zero by the
    same plane-strain decoupling :func:`_ellipse_equivalent_eigenstrain`
    documents for the isolated case (``S3jkl=Si3kl=0``): this project's
    grid has no z-extent (`grid.shape[2] == 1`), so every Fourier mode
    has ``k_z=0`` identically, periodic or not, and the decoupling
    argument is exactly as valid here as in the continuum isolated
    derivation.
    """
    inside_mask = xp.asarray(inside_mask)
    count = xp.sum(inside_mask.astype(xp.float64))

    def mean_strain(component):
        probe = xp.zeros((3, 3))
        probe[component[0], component[1]] = 1.0
        probe[component[1], component[0]] = 1.0
        strain, _ = prescribed_eigenstrain_solution(probe)
        strain = xp.asarray(strain)
        return xp.array(
            [
                [
                    xp.sum(xp.where(inside_mask, strain[i, j], 0.0)) / count
                    for j in range(3)
                ]
                for i in range(3)
            ]
        )

    response_11 = mean_strain((0, 0))
    response_22 = mean_strain((1, 1))
    response_12 = mean_strain((0, 1))
    return {
        "s1111": float(response_11[0, 0]),
        "s2211": float(response_11[1, 1]),
        "s1122": float(response_22[0, 0]),
        "s2222": float(response_22[1, 1]),
        # response_12 was probed with eps*_12=eps*_21=1 (tensor convention),
        # which the s_eff=2*s1212 convention below expects to come back as
        # eps_in_12 = 2*s1212*eps*_12 -- see _periodic_equivalent_eigenstrain.
        "s1212": float(response_12[0, 1]) / 2.0,
    }


def _periodic_equivalent_eigenstrain(magnitude, contrast, matrix_lame_lambda, matrix_lame_mu, load, s):
    r"""Eshelby equivalent eigenstrain, periodic-lattice-corrected version
    of :func:`_equivalent_eigenstrain` -- same equivalent-inclusion
    linear algebra :func:`_ellipse_equivalent_eigenstrain` already uses
    (a reduced 3-variable solve for the normal/dilatational components,
    decoupled by symmetry from a scalar solve for the shear component),
    generalized to accept *any* numerically supplied Eshelby-tensor
    components `s` (a dict with keys ``s1111, s2211, s1122, s2222,
    s1212``, e.g. from :func:`_periodic_eshelby_tensor`) rather than the
    isolated closed form -- this is what makes it correct for a
    non-dilute periodic array, unlike :func:`_equivalent_eigenstrain`.

    Does not use the isolated case's rotate-the-tension-solution trick
    for ``"biaxial"``/``"shear"`` (valid there only because the isolated
    circular Eshelby tensor is fully isotropic) -- a periodic square
    lattice only has 4-fold symmetry, not full rotational symmetry, so
    each load case is solved directly against the supplied (possibly
    anisotropic-looking) `s`, the same way
    :func:`_ellipse_equivalent_eigenstrain` already does for its
    non-circular, symmetry-poor geometry.

    ``contrast=float('inf')`` (rigid) is handled by the exact physical
    limit ``eps_in=0`` directly (``eps_bar + S:eps*=0``) rather than
    substituting a literal ``inf`` into the finite-contrast linear system
    below, which hits ``inf/inf``.
    """
    nu = matrix_lame_lambda / (2.0 * (matrix_lame_lambda + matrix_lame_mu))
    lam0, mu0 = matrix_lame_lambda, matrix_lame_mu
    s1111, s2211 = s["s1111"], s["s2211"]
    s1122, s2222 = s["s1122"], s["s2222"]
    s1212 = s["s1212"]

    def normal_eigenstrain(sbar11, sbar22):
        sbar33 = nu * (sbar11 + sbar22)
        young0 = 2.0 * mu0 * (1.0 + nu)
        eps_bar_11 = (sbar11 - nu * sbar22 - nu * sbar33) / young0
        eps_bar_22 = (sbar22 - nu * sbar11 - nu * sbar33) / young0
        eps_bar = xp.array([eps_bar_11, eps_bar_22])

        s_mat = xp.array([[s1111, s1122], [s2211, s2222]])
        if contrast == float("inf"):
            es01 = xp.linalg.solve(s_mat, -eps_bar)
            return es01[0], es01[1], 0.0

        lam1, mu1 = contrast * lam0, contrast * mu0
        identity2 = xp.eye(2)
        ones2 = xp.ones((2, 2))
        c1_mat = lam1 * ones2 + 2.0 * mu1 * identity2
        c0_mat = lam0 * ones2 + 2.0 * mu0 * identity2
        a_mat = c1_mat @ s_mat - c0_mat @ (s_mat - identity2)
        rhs = (c0_mat - c1_mat) @ eps_bar
        es01 = xp.linalg.solve(a_mat, rhs)
        return es01[0], es01[1], 0.0

    def shear_eigenstrain():
        eps_bar_12 = magnitude / (2.0 * mu0)
        s_eff = 2.0 * s1212
        if contrast == float("inf"):
            return -eps_bar_12 / s_eff
        mu1 = contrast * mu0
        return eps_bar_12 * (mu0 - mu1) / (mu1 * s_eff - mu0 * s_eff + mu0)

    eps_star = xp.zeros((3, 3))
    if load == "tension":
        es0, es1, es2 = normal_eigenstrain(magnitude, 0.0)
    elif load == "compression":
        es0, es1, es2 = normal_eigenstrain(-magnitude, 0.0)
    elif load == "biaxial":
        es0, es1, es2 = normal_eigenstrain(magnitude, magnitude)
    elif load == "shear":
        eps_star[0, 1] = eps_star[1, 0] = shear_eigenstrain()
        return eps_star
    else:
        raise ValueError('load must be "tension", "compression", "biaxial", or "shear"')
    eps_star[0, 0], eps_star[1, 1], eps_star[2, 2] = es0, es1, es2
    return eps_star


def _periodic_eshelby_antiplane_tensor(prescribed_eigenstrain_solution, inside_mask):
    r"""Numerically probe the *periodic* (lattice-corrected) antiplane
    Eshelby tensor component ``s1313`` -- the antiplane counterpart of
    :func:`_periodic_eshelby_tensor`, needed to calibrate an antiplane
    equivalent eigenstrain (:func:`_antiplane_periodic_equivalent_eigenstrain`)
    against how densely packed the periodic array actually is, exactly as
    :func:`_periodic_eshelby_tensor` does for the in-plane loads.

    Only one component is needed (unlike the four normal/shear components
    :func:`_periodic_eshelby_tensor` probes): antiplane shear decouples
    completely from in-plane deformation for an isotropic reference medium
    with no x3-variation (probing with a pure :math:`\varepsilon^*_{13}`
    eigenstrain excites a body-force source only in the antiplane
    (:math:`i=3`) equilibrium equation -- confirmed directly from
    :func:`crystallite.spectral.short_range.DifferentialOperators.div`'s
    contraction: ``source_hat[0]`` and ``source_hat[1]`` come out exactly
    zero for a probe with only ``eps_star[0,2]=eps_star[2,0]`` nonzero,
    since the shape function has no :math:`x_3`-dependence on this
    project's ``grid.shape[2] == 1`` grids), so only ``s1313`` -- not a
    full 5-component analog of ``s1111``/etc. -- ever enters
    :func:`_antiplane_periodic_equivalent_eigenstrain`'s linear algebra.

    Same probe-and-average recipe as :func:`_periodic_eshelby_tensor`
    (reusing :meth:`HoleInPlateCase.periodic_prescribed_eigenstrain_solution`
    for a unit probe eigenstrain, then averaging the resulting strain over
    `inside_mask`), and the same ``eps_in_13 = 2*s1313*eps*_13`` halving
    convention as that function's own ``s1212`` (probed with
    ``eps*_13=eps*_31=1``, tensor-symmetric convention).
    """
    inside_mask = xp.asarray(inside_mask)
    count = xp.sum(inside_mask.astype(xp.float64))

    probe = xp.zeros((3, 3))
    probe[0, 2] = probe[2, 0] = 1.0
    strain, _ = prescribed_eigenstrain_solution(probe)
    strain = xp.asarray(strain)
    response_13 = xp.sum(xp.where(inside_mask, strain[0, 2], 0.0)) / count
    return float(response_13) / 2.0


def _antiplane_periodic_equivalent_eigenstrain(magnitude, contrast, matrix_lame_mu, s1313):
    r"""Eshelby equivalent eigenstrain for antiplane shear, periodic-
    lattice-corrected version of the isolated closed form derived in
    :func:`_antiplane_inhomogeneity_polar_stress`'s own docstring (whose
    isolated dilute-limit self-interaction, ``s1313=1/4``, this reduces to
    exactly when `s1313` is that isolated value rather than a numerically
    probed periodic one -- verified directly) -- the antiplane
    counterpart of :func:`_periodic_equivalent_eigenstrain`'s
    ``shear_eigenstrain`` branch, same ``s_eff=2*s1313`` convention and
    identical algebraic form (only one component here, unlike the coupled
    3-variable normal-stress solve that function also handles, since
    antiplane shear has no analog of the dilatational/biaxial coupling
    in-plane tension has).

    ``contrast=float('inf')`` (rigid) uses the exact ``eps_in_13=0``
    limit directly (``eps_bar_13 + s_eff*eps*_13=0``), same reasoning as
    :func:`_periodic_equivalent_eigenstrain`.
    """
    mu0 = matrix_lame_mu
    eps_bar_13 = magnitude / (2.0 * mu0)
    s_eff = 2.0 * s1313

    eps_star = xp.zeros((3, 3))
    if contrast == float("inf"):
        es13 = -eps_bar_13 / s_eff
    else:
        mu1 = contrast * mu0
        es13 = eps_bar_13 * (mu0 - mu1) / (mu1 * s_eff - mu0 * s_eff + mu0)
    eps_star[0, 2] = eps_star[2, 0] = es13
    return eps_star


def _periodic_eshelby_dipole_tensor(prescribed_gradient_eigenstrain_solution, inside_mask, offset):
    r"""Numerically probe the periodic *dipole* (gradient-order) Eshelby
    tensor components ``s1111, s2211, s1122, s2222, s1212`` -- the
    gradient-loading counterpart of :func:`_periodic_eshelby_tensor`,
    needed to calibrate an equivalent eigenstrain against a remote
    *stress-gradient* ("moment"/bending) background rather than a uniform
    remote stress.

    A linear (dipole) eigenstrain :math:`\varepsilon^*(x)=\varepsilon^*_0
    \cdot \mathrm{offset}(x)` prescribed over a periodic array of regions
    produces an *exactly linear* interior strain response, one order up
    from the uniform-eigenstrain case's uniform response -- the
    polynomial generalization of Eshelby's uniformity theorem (a periodic
    array of identical regions carrying a spatially linear eigenstrain
    still responds linearly inside each region, not just approximately).

    Probed the same way as :func:`_periodic_eshelby_tensor` (reusing this
    project's own exact periodic solve, here
    `prescribed_gradient_eigenstrain_solution`, e.g.
    :meth:`HoleInPlateCase.periodic_prescribed_gradient_eigenstrain_solution`),
    with the slope extracted from the *difference* of the region's mean
    response above vs. below ``offset=0`` (`offset` matching the
    background's own convention, typically ``x2-center``) -- a
    construction that exactly cancels any constant (zeroth-order) offset
    from discretization noise, since the true continuum response has
    none (an odd function of `offset`, by the same up/down mirror
    symmetry :func:`_gradient_void_hole_correction_cartesian` relies on
    for the true-void case).

    Returns the same key convention as :func:`_periodic_eshelby_tensor`
    (``s1111, s2211, s1122, s2222, s1212``) so the result can be fed
    directly into :func:`_periodic_equivalent_eigenstrain` unchanged --
    that function's linear algebra does not care whether the tensor it is
    given is the zeroth- or first-order Eshelby tensor, only that it
    correctly maps a trial eigenstrain (of whichever order) to the
    matching-order interior strain response.
    """
    inside_mask = xp.asarray(inside_mask)
    offset = xp.asarray(offset)
    upper = inside_mask & (offset > 0)
    lower = inside_mask & (offset < 0)
    n_upper = xp.sum(upper.astype(xp.float64))
    n_lower = xp.sum(lower.astype(xp.float64))
    mean_offset_upper = xp.sum(xp.where(upper, offset, 0.0)) / n_upper
    mean_offset_lower = xp.sum(xp.where(lower, offset, 0.0)) / n_lower
    denom = mean_offset_upper - mean_offset_lower

    def slope(component):
        probe = xp.zeros((3, 3))
        probe[component[0], component[1]] = 1.0
        probe[component[1], component[0]] = 1.0
        strain, _ = prescribed_gradient_eigenstrain_solution(probe)
        strain = xp.asarray(strain)
        return xp.array(
            [
                [
                    (
                        xp.sum(xp.where(upper, strain[i, j], 0.0)) / n_upper
                        - xp.sum(xp.where(lower, strain[i, j], 0.0)) / n_lower
                    )
                    / denom
                    for j in range(3)
                ]
                for i in range(3)
            ]
        )

    response_11 = slope((0, 0))
    response_22 = slope((1, 1))
    response_12 = slope((0, 1))
    return {
        "s1111": float(response_11[0, 0]),
        "s2211": float(response_11[1, 1]),
        "s1122": float(response_22[0, 0]),
        "s2222": float(response_22[1, 1]),
        "s1212": float(response_12[0, 1]) / 2.0,
    }


def _disk_fourier_transform(grid, hole_radius, center=(0.0, 0.0, 0.0)):
    r"""Fourier transform of a disk of radius `hole_radius` centered at
    `center`, in the unnormalized-FFT convention `grid.fft`/`grid.ifft` use.

    The continuum transform of a disk *centered at the origin* is the
    standard closed form :math:`2\pi a J_1(ka)/k` (`scipy.special.j1`),
    :math:`\pi a^2` at :math:`k=0`; matching this project's unnormalized
    ``rfftn`` convention requires an extra factor of (number of grid
    points) / (domain volume) -- pinned down empirically against a
    real-space FFT of an actual disk indicator array before relying on it
    here (see this module's tests), rather than trusting the
    discrete/continuum correspondence by derivation alone. A disk centered
    anywhere else picks up the standard Fourier shift-theorem phase factor
    :math:`e^{-i\mathbf{k}\cdot\mathbf{center}}` -- easy to forget (as a
    first version of this function did) since `grid.x` runs from 0 to the
    domain length rather than being centered at 0, so *every* off-origin
    feature needs this phase, not just this one; the omission was caught
    by the resulting field being centered on the domain corner instead of
    `center`.
    """
    k = xp.sqrt(grid.k2)
    safe_k = xp.where(k == 0, 1.0, k)
    continuum = xp.where(
        k == 0,
        xp.pi * hole_radius**2,
        2.0 * xp.pi * hole_radius * j1(safe_k * hole_radius) / safe_k,
    )
    phase = xp.exp(
        -1j * (grid.k[0] * center[0] + grid.k[1] * center[1] + grid.k[2] * center[2])
    )
    n_points = 1
    for n in grid.fft_shape:
        n_points *= n
    volume = 1.0
    for length in grid.lengths:
        volume *= length
    return (n_points / volume) * continuum * phase


def _ellipse_fourier_transform(grid, semi_axis_a, semi_axis_b, center=(0.0, 0.0, 0.0)):
    r"""Fourier transform of an ellipse (semi-axis `semi_axis_a` along x1,
    `semi_axis_b` along x2) centered at `center`, same convention as
    :func:`_disk_fourier_transform` (which this reduces to exactly at
    ``semi_axis_a=semi_axis_b``).

    An ellipse is a disk under the linear map :math:`(x,y)\mapsto(ax,by)`;
    for :math:`f(x)=g(M^{-1}x)`, :math:`\hat f(k)=|\det M|\,\hat
    g(M^{\mathsf T}k)`, so the ellipse's transform is the unit disk's own
    :math:`2\pi J_1(k)/k` closed form (:func:`_disk_fourier_transform`,
    scaled to radius 1) evaluated at :math:`k\to\sqrt{(a k_x)^2+(b
    k_y)^2}` and scaled by the area factor :math:`ab`. Verified directly
    against a brute-force 2D numerical integration of the ellipse
    indicator's Fourier integral (not just the disk limit), matching to
    the integrator's own tolerance at several sample wavevectors.
    """
    k = xp.sqrt((semi_axis_a * grid.k[0]) ** 2 + (semi_axis_b * grid.k[1]) ** 2)
    safe_k = xp.where(k == 0, 1.0, k)
    continuum = xp.where(
        k == 0,
        xp.pi * semi_axis_a * semi_axis_b,
        2.0 * xp.pi * semi_axis_a * semi_axis_b * j1(safe_k) / safe_k,
    )
    phase = xp.exp(
        -1j * (grid.k[0] * center[0] + grid.k[1] * center[1] + grid.k[2] * center[2])
    )
    n_points = 1
    for n in grid.fft_shape:
        n_points *= n
    volume = 1.0
    for length in grid.lengths:
        volume *= length
    return (n_points / volume) * continuum * phase


def _lattice_eshelby_sum(a, b, length_x, length_y, lam, mu, n_modes, chunk=64):
    r"""Periodic Eshelby tensor of an elliptical region (semi-axes `a`, `b`)
    in a `length_x` by `length_y` cell, as the truncated analytic k-space
    lattice sum over reciprocal vectors :math:`k=2\pi(m/L_x, n/L_y)`,
    :math:`0<\max(|m|,|n|)\le` `n_modes`:

    .. math::

        S^{\mathrm{per}}_{ijkl}=\frac{1}{A\,|\Omega|}\sum_{k\ne 0}
        \hat s(k)^2\,\hat\varepsilon_{ij}(k)\big[\varepsilon^*=e_{kl}\big],

    with :math:`\hat s=2\pi ab\,J_1(q)/q`, :math:`q=\sqrt{(ak_x)^2+(bk_y)^2}`
    the exact Fourier transform of the ellipse, and :math:`\hat\varepsilon`
    the plane-strain isotropic Green operator response to a unit
    eigenstrain. Every term is analytic: no grid, no pixelated mask, no
    numerical probing. Returns the in-plane components in the same
    convention as :func:`_periodic_eshelby_tensor` (``s1212`` is half the
    tensor-shear response) plus the antiplane ``s1313`` and ``s2323``.

    The truncation error falls like ``1/n_modes``; see
    :func:`_lattice_eshelby_tensor` for the extrapolated version.
    """
    c = (lam + mu) / (lam + 2.0 * mu)
    cell_area, region_area = length_x * length_y, np.pi * a * b
    ky = (2.0 * np.pi * np.arange(-n_modes, n_modes + 1) / length_y)[None, :]
    unit = {
        "11": (1.0, 0.0, 0.0),
        "22": (0.0, 1.0, 0.0),
        "12": (0.0, 0.0, 1.0),
    }
    acc = {name: np.zeros(3) for name in unit}
    acc_13 = 0.0
    acc_23 = 0.0
    for start in range(-n_modes, n_modes + 1, chunk):
        m = np.arange(start, min(start + chunk, n_modes + 1))
        kx = (2.0 * np.pi * m / length_x)[:, None]
        k2 = kx**2 + ky**2
        origin = k2 == 0
        k2_safe = np.where(origin, 1.0, k2)
        q = np.sqrt((a * kx) ** 2 + (b * ky) ** 2)
        q_safe = np.where(q == 0, 1.0, q)
        shape2 = np.where(origin, 0.0, (2.0 * np.pi * a * b * j1(q_safe) / q_safe) ** 2)
        for name, (e11, e22, e12) in unit.items():
            trace = e11 + e22
            sxx = lam * trace + 2.0 * mu * e11
            syy = lam * trace + 2.0 * mu * e22
            sxy = 2.0 * mu * e12
            tx, ty = sxx * kx + sxy * ky, sxy * kx + syy * ky
            kt = (kx * tx + ky * ty) / k2_safe
            gx = (tx - c * kx * kt) / (mu * k2_safe)
            gy = (ty - c * ky * kt) / (mu * k2_safe)
            acc[name] += np.array(
                [
                    np.sum(shape2 * kx * gx),
                    np.sum(shape2 * ky * gy),
                    np.sum(shape2 * 0.5 * (kx * gy + ky * gx)),
                ]
            )
        acc_13 += np.sum(shape2 * kx**2 / k2_safe)
        acc_23 += np.sum(shape2 * ky**2 / k2_safe)
    scale = 1.0 / (cell_area * region_area)
    return {
        "s1111": acc["11"][0] * scale,
        "s2211": acc["11"][1] * scale,
        "s1122": acc["22"][0] * scale,
        "s2222": acc["22"][1] * scale,
        "s1212": acc["12"][2] * scale / 2.0,
        "s1313": acc_13 * scale / 2.0,
        "s2323": acc_23 * scale / 2.0,
    }


@functools.lru_cache(maxsize=128)
def _lattice_eshelby_items(a, b, length_x, length_y, lam, mu, n_modes):
    coarse = _lattice_eshelby_sum(a, b, length_x, length_y, lam, mu, n_modes // 2)
    fine = _lattice_eshelby_sum(a, b, length_x, length_y, lam, mu, n_modes)
    # error ~ 1/n_modes: one Richardson step removes the leading term
    return tuple((k, 2.0 * fine[k] - coarse[k]) for k in fine)


def _lattice_eshelby_tensor(a, b, length_x, length_y, lam, mu, n_modes=1024):
    r"""Exact periodic Eshelby tensor of an ellipse in a rectangular cell by
    the analytic lattice sum :func:`_lattice_eshelby_sum`, Richardson-
    extrapolated from `n_modes/2` and `n_modes` (the truncation error
    falls like ``1/n_modes``; extrapolated values agree between
    ``(256, 512)`` and ``(512, 1024)`` to ~1e-5). Independent of any grid:
    unlike :func:`_periodic_eshelby_tensor`, which probes an FFT solve and
    inherits its mask pixelization (measured 3% off at an ellipse ~6 grid
    cells across), this approaches the closed-form isolated tensor as the
    area fraction goes to zero. Returns a dict with keys ``s1111, s2211,
    s1122, s2222, s1212`` (the convention
    :func:`_periodic_equivalent_eigenstrain` takes) and ``s1313``.
    """
    return dict(
        _lattice_eshelby_items(
            float(a), float(b), float(length_x), float(length_y), float(lam), float(mu),
            int(n_modes),
        )
    )


def _load_macro_strain(load, magnitude, lam, mu):
    """Plane-strain macroscopic strain a uniform remote `load` produces in
    the matrix (``eps_33 = 0``), the same convention
    :func:`_periodic_equivalent_eigenstrain` uses internally."""
    nu = lam / (2.0 * (lam + mu))
    young = 2.0 * mu * (1.0 + nu)
    eps = np.zeros((3, 3))
    if load == "shear":
        eps[0, 1] = eps[1, 0] = magnitude / (2.0 * mu)
        return eps
    sign = {"tension": (1.0, 0.0), "compression": (-1.0, 0.0), "biaxial": (1.0, 1.0)}
    if load not in sign:
        raise ValueError('load must be "tension", "compression", "biaxial", or "shear"')
    s11, s22 = (sign[load][0] * magnitude, sign[load][1] * sign[load][0] * magnitude)
    s33 = nu * (s11 + s22)
    eps[0, 0] = (s11 - nu * s22 - nu * s33) / young
    eps[1, 1] = (s22 - nu * s11 - nu * s33) / young
    return eps


def _periodic_mean_stress(load, magnitude, contrast, lam, mu, a, b, length_x, length_y):
    r"""Exact domain-mean stress of a periodic array of elliptical
    inhomogeneities (`contrast`, semi-axes `a`, `b`, cell `length_x` by
    `length_y`) under macroscopic strain control -- the boundary condition
    this project's solver uses -- as the ``(3, 3)`` tensor
    :math:`\bar\sigma=C_0:(\bar\varepsilon-f\,\varepsilon^*)`, with
    :math:`f=\pi ab/(L_xL_y)` and :math:`\varepsilon^*` the periodic
    equivalent eigenstrain from the lattice-sum Eshelby tensor
    (:func:`_lattice_eshelby_tensor`). No FFT and no grid. This is the
    constant an image sum of isolated fields cannot determine on its own
    (the sum of the ~:math:`r^{-2}` corrections is only conditionally
    convergent), so it is what the periodic image-sum constructions
    recenter to.
    """
    s = _lattice_eshelby_tensor(a, b, length_x, length_y, lam, mu)
    eps_star = np.asarray(_periodic_equivalent_eigenstrain(magnitude, contrast, lam, mu, load, s))
    eps_bar = _load_macro_strain(load, magnitude, lam, mu)
    difference = eps_bar - (np.pi * a * b / (length_x * length_y)) * eps_star
    trace = difference[0, 0] + difference[1, 1] + difference[2, 2]
    return lam * trace * np.eye(3) + 2.0 * mu * difference


def _periodic_antiplane_mean_stress(magnitude, contrast, mu, a, b, length_x, length_y):
    r"""Antiplane counterpart of :func:`_periodic_mean_stress`: the domain-
    mean ``(3, 3)`` stress (only ``[0, 2]``/``[2, 0]`` populated) of a
    periodic array under remote antiplane shear `magnitude`, from the
    lattice-sum ``s1313``."""
    s1313 = _lattice_eshelby_tensor(a, b, length_x, length_y, 1.0, mu)["s1313"]
    eps_star = np.asarray(_antiplane_periodic_equivalent_eigenstrain(magnitude, contrast, mu, s1313))
    eps_bar_13 = magnitude / (2.0 * mu)
    mean_13 = 2.0 * mu * (eps_bar_13 - (np.pi * a * b / (length_x * length_y)) * eps_star[0, 2])
    stress = np.zeros((3, 3))
    stress[0, 2] = stress[2, 0] = mean_13
    return stress


def _periodic_interior_stress(load, magnitude, contrast, lam, mu, a, b, length_x, length_y):
    r"""Exact uniform interior stress ``(3, 3)`` of the elliptical
    inhomogeneities in the periodic array under strain control:
    :math:`\sigma_{\mathrm{in}}=C_0:(\bar\varepsilon+S^{\mathrm{per}}:\varepsilon^*-\varepsilon^*)`
    (the matrix-material equivalent-inclusion form, which also covers
    ``contrast=inf``), with the lattice-sum tensor
    (:func:`_lattice_eshelby_tensor`). Eshelby's uniformity theorem holds
    for the periodic array, so this is the exact stress everywhere inside
    each region, not just its mean -- it differs from the isolated
    (home-image) interior by O(area fraction). Checked against the
    interior mean of a converged CG solve (within ~0.5% of the load, where
    the isolated value is up to 11% off at 12% area fraction).
    """
    s = _lattice_eshelby_tensor(a, b, length_x, length_y, lam, mu)
    eps_star = np.asarray(_periodic_equivalent_eigenstrain(magnitude, contrast, lam, mu, load, s))
    eps_bar = _load_macro_strain(load, magnitude, lam, mu)
    eps_in = eps_bar.copy()
    eps_in[0, 0] += s["s1111"] * eps_star[0, 0] + s["s1122"] * eps_star[1, 1]
    eps_in[1, 1] += s["s2211"] * eps_star[0, 0] + s["s2222"] * eps_star[1, 1]
    eps_in[0, 1] = eps_in[1, 0] = eps_bar[0, 1] + 2.0 * s["s1212"] * eps_star[0, 1]
    difference = eps_in - eps_star
    trace = difference[0, 0] + difference[1, 1] + difference[2, 2]
    return lam * trace * np.eye(3) + 2.0 * mu * difference


def _periodic_antiplane_interior_stress(magnitude, contrast, mu, a, b, length_x, length_y):
    """Antiplane counterpart of :func:`_periodic_interior_stress`
    (``sigma_13`` only; ``sigma_23`` vanishes by symmetry)."""
    s1313 = _lattice_eshelby_tensor(a, b, length_x, length_y, 1.0, mu)["s1313"]
    eps_star = np.asarray(_antiplane_periodic_equivalent_eigenstrain(magnitude, contrast, mu, s1313))
    eps_in_13 = magnitude / (2.0 * mu) + 2.0 * s1313 * eps_star[0, 2]
    stress = np.zeros((3, 3))
    stress[0, 2] = stress[2, 0] = 2.0 * mu * (eps_in_13 - eps_star[0, 2])
    return stress


def _inplane_eigenstrain_correction(x, y, eps_star, a, b, mu, nu):
    r"""In-plane stress ``(sigma_xx, sigma_yy, sigma_xy)`` of a single
    isolated ellipse carrying the uniform eigenstrain `eps_star` in a
    homogeneous matrix -- exterior from
    :func:`_ellipse_inhomogeneity_exterior_stress`, interior from
    :func:`_ellipse_inhomogeneity_interior_stress` -- at coordinates
    `x`, `y` relative to the ellipse center. Pure perturbation (no
    background). Valid at every point: interior points are projected onto
    the boundary before the exterior formula is evaluated (its value there
    is discarded), and the exact center, which has no radial direction to
    project along, uses an arbitrary boundary point.
    """
    rho = xp.sqrt((x / a) ** 2 + (y / b) ** 2)
    outside = rho >= 1.0 - 1.0e-9
    rho_safe = xp.where(rho == 0, 1.0, rho)
    x_scaled = xp.where(outside, x, x / rho_safe)
    y_scaled = xp.where(outside, y, y / rho_safe)
    x_safe = xp.where(rho == 0, a, x_scaled)
    y_safe = xp.where(rho == 0, 0.0, y_scaled)
    ext = _ellipse_inhomogeneity_exterior_stress(x_safe, y_safe, eps_star, a, b, mu, nu)
    inn = _ellipse_inhomogeneity_interior_stress(eps_star, a, b, mu, nu)
    return tuple(xp.where(outside, e, i) for e, i in zip(ext, inn))


def _exterior_shift(raw, outside, interior_value, target):
    """Constant to subtract from the exterior of the raw image sum so the
    *final* field (exterior shifted, interior set to the exact
    `interior_value`) has domain mean exactly `target`:
    ``mean(where(outside, raw - shift, interior_value)) == target``."""
    n_total = raw.size
    n_outside = xp.sum(outside.astype(raw.dtype))
    n_inside = n_total - n_outside
    exterior_total = xp.sum(xp.where(outside, raw, 0.0))
    return (exterior_total + n_inside * interior_value - n_total * target) / n_outside


def _antiplane_eigenstrain_correction(x, y, eps_star, a, b, mu):
    """Antiplane counterpart of :func:`_inplane_eigenstrain_correction`:
    ``(sigma_13, sigma_23)`` of a single isolated ellipse carrying the
    uniform antiplane eigenstrain `eps_star` (``[0, 2]``/``[1, 2]``),
    exterior and interior, at coordinates relative to its center."""
    rho = xp.sqrt((x / a) ** 2 + (y / b) ** 2)
    outside = rho >= 1.0 - 1.0e-9
    rho_safe = xp.where(rho == 0, 1.0, rho)
    x_scaled = xp.where(outside, x, x / rho_safe)
    y_scaled = xp.where(outside, y, y / rho_safe)
    x_safe = xp.where(rho == 0, a, x_scaled)
    y_safe = xp.where(rho == 0, 0.0, y_scaled)
    ext = _ellipse_inhomogeneity_antiplane_exterior_stress(x_safe, y_safe, eps_star, a, b, mu)
    inn = _ellipse_inhomogeneity_antiplane_interior_stress(eps_star, a, b, mu)
    return tuple(xp.where(outside, e, i) for e, i in zip(ext, inn))


def _periodic_gradient_inhomogeneity_stress(
    grid, center, a, b, contrast, lam, mu, magnitude, n_images
):
    """Periodic-array stress ``(3, 3) + grid.shape`` of an elliptical
    inhomogeneity (semi-axes `a`, `b`, `contrast`) under the "moment" remote
    stress gradient, as the background plus the image sum of the isolated
    solution (:func:`_ellipse_gradient_inhomogeneity_solution`) -- shared by
    both case classes (a circle is ``a == b``). Only the in-plane
    ``sigma_xx``, ``sigma_yy``, ``sigma_xy`` are populated."""
    nu = lam / (2.0 * (lam + mu))
    solution = _ellipse_gradient_inhomogeneity_solution(a, b, contrast, nu, magnitude)
    x, y = grid.x[0], grid.x[1]
    length_x, length_y = grid.lengths[0], grid.lengths[1]

    sigma_xx = magnitude * (y - center[1]) * xp.ones(grid.shape)
    sigma_yy = xp.zeros(grid.shape)
    sigma_xy = xp.zeros(grid.shape)
    for n1 in range(-n_images, n_images + 1):
        for n2 in range(-n_images, n_images + 1):
            dx = x - (center[0] + n1 * length_x)
            dy = y - (center[1] + n2 * length_y)
            sxx, syy, sxy = _ellipse_gradient_inhomogeneity_correction_cartesian(
                xp.broadcast_to(dx, grid.shape), xp.broadcast_to(dy, grid.shape), solution
            )
            sigma_xx = sigma_xx + sxx
            sigma_yy = sigma_yy + syy
            sigma_xy = sigma_xy + sxy

    stress = xp.zeros((3, 3) + grid.shape, dtype=sigma_xx.dtype)
    stress[0, 0] = sigma_xx
    stress[1, 1] = sigma_yy
    stress[0, 1] = stress[1, 0] = sigma_xy
    return stress


def _periodic_prescribed_interior_stress(eps_star, lam, mu, a, b, length_x, length_y):
    r"""Interior stress ``(3, 3)`` of a periodic array of ellipses (semi-axes
    `a`, `b`, cell `length_x` by `length_y`) each carrying the uniform
    eigenstrain `eps_star` in an otherwise homogeneous matrix, at zero
    macroscopic strain: :math:`\sigma_{\mathrm{in}}=C_0:(S^{\mathrm{per}}:
    \varepsilon^*-\varepsilon^*)`, with the exact lattice-sum Eshelby tensor
    (:func:`_lattice_eshelby_tensor`) -- no FFT and no grid. Covers the
    in-plane components (``[0, 0]``, ``[1, 1]``, ``[0, 1]``), the antiplane
    ones (``[0, 2]``, ``[1, 2]``) and a plane-strain ``[2, 2]`` eigenstrain
    (which only enters through the trace). In the dilute limit this reduces
    to the classical isolated Eshelby interior stress.
    """
    eps_star = np.asarray(eps_star, dtype=float)
    s = _lattice_eshelby_tensor(a, b, length_x, length_y, lam, mu)
    eps_in = np.zeros((3, 3))
    eps_in[0, 0] = s["s1111"] * eps_star[0, 0] + s["s1122"] * eps_star[1, 1]
    eps_in[1, 1] = s["s2211"] * eps_star[0, 0] + s["s2222"] * eps_star[1, 1]
    eps_in[0, 1] = eps_in[1, 0] = 2.0 * s["s1212"] * eps_star[0, 1]
    eps_in[0, 2] = eps_in[2, 0] = 2.0 * s["s1313"] * eps_star[0, 2]
    eps_in[1, 2] = eps_in[2, 1] = 2.0 * s["s2323"] * eps_star[1, 2]
    difference = eps_in - eps_star
    trace = difference[0, 0] + difference[1, 1] + difference[2, 2]
    return lam * trace * np.eye(3) + 2.0 * mu * difference


def cartesian_to_polar_stress(sigma_xx, sigma_yy, sigma_xy, theta):
    """Rotate a 2D Cartesian stress state into polar (r, theta) components.

    Parameters
    ----------
    sigma_xx, sigma_yy, sigma_xy : array_like
    theta : array_like
        Polar angle, same shape (or broadcastable).

    Returns
    -------
    sigma_rr, sigma_theta_theta, sigma_r_theta : ndarray
    """
    cos_t, sin_t = xp.cos(theta), xp.sin(theta)
    cos2, sin2 = cos_t**2, sin_t**2
    sin_cos = sin_t * cos_t

    sigma_rr = sigma_xx * cos2 + sigma_yy * sin2 + 2.0 * sigma_xy * sin_cos
    sigma_theta_theta = sigma_xx * sin2 + sigma_yy * cos2 - 2.0 * sigma_xy * sin_cos
    sigma_r_theta = (sigma_yy - sigma_xx) * sin_cos + sigma_xy * (cos2 - sin2)

    return sigma_rr, sigma_theta_theta, sigma_r_theta


def polar_to_cartesian_stress(sigma_rr, sigma_theta_theta, sigma_r_theta, theta):
    """Inverse of :func:`cartesian_to_polar_stress`."""
    cos_t, sin_t = xp.cos(theta), xp.sin(theta)
    cos2, sin2 = cos_t**2, sin_t**2
    sin_cos = sin_t * cos_t

    sigma_xx = sigma_rr * cos2 + sigma_theta_theta * sin2 - 2.0 * sigma_r_theta * sin_cos
    sigma_yy = sigma_rr * sin2 + sigma_theta_theta * cos2 + 2.0 * sigma_r_theta * sin_cos
    sigma_xy = (sigma_rr - sigma_theta_theta) * sin_cos + sigma_r_theta * (cos2 - sin2)

    return sigma_xx, sigma_yy, sigma_xy


def polar_to_cartesian_antiplane_stress(sigma_r3, sigma_theta3, theta):
    r"""Rotate a 2D polar antiplane shear state
    (:math:`\sigma_{r3},\sigma_{\theta 3}`) into Cartesian
    (:math:`\sigma_{13},\sigma_{23}`) -- the antiplane counterpart of
    :func:`polar_to_cartesian_stress`, simpler because the "3" index is
    rotation-invariant: :math:`(\sigma_{13},\sigma_{23})` transforms as an
    ordinary 2D vector under an in-plane rotation, not a rank-2 tensor's
    usual double rotation.

    Parameters
    ----------
    sigma_r3, sigma_theta3 : array_like
    theta : array_like
        Polar angle, same shape (or broadcastable).

    Returns
    -------
    sigma_13, sigma_23 : ndarray
    """
    cos_t, sin_t = xp.cos(theta), xp.sin(theta)
    sigma_13 = sigma_r3 * cos_t - sigma_theta3 * sin_t
    sigma_23 = sigma_r3 * sin_t + sigma_theta3 * cos_t
    return sigma_13, sigma_23


def _antiplane_polar_stress(r, theta, hole_radius, magnitude):
    r"""Exact closed-form antiplane stress field
    (:math:`\sigma_{r3},\sigma_{\theta 3}`) around a traction-free
    circular hole in an infinite isotropic medium under remote antiplane
    shear `magnitude` (:math:`\sigma_{13}=\text{magnitude}`,
    :math:`\sigma_{23}=0`, at infinity -- :math:`\theta=0` along the
    *loaded* axis, matching this project's own convention elsewhere of
    measuring :math:`\theta` from the axis the in-plane loads act along,
    e.g. :func:`_tension_polar_stress`'s remote :math:`\sigma_{11}`) --
    the antiplane counterpart of :func:`_tension_polar_stress`, and
    considerably simpler: antiplane elasticity reduces to a scalar
    Laplace problem for :math:`u_3(x_1, x_2)`
    (:math:`\sigma_{i3}=\mu\,\partial_i u_3`, equilibrium
    :math:`\nabla^2 u_3=0`), not the biharmonic in-plane problem.

    Derived from :math:`u_3=\gamma(r+a^2/r)\cos\theta` (:math:`\gamma
    =\text{magnitude}/\mu`, dropped below since stress is
    :math:`\mu`-independent here just as in the in-plane case): the
    :math:`a^2/r` term is the unique (decaying, harmonic) correction that
    cancels the remote field's radial derivative at :math:`r=a`,
    exactly the same image-term idea :func:`_tension_polar_stress`'s own
    derivation uses. Verified three ways (harmonic equilibrium,
    traction-free boundary condition, and the Cartesian closed form
    against direct finite-difference derivatives of :math:`u_3`, all to
    numerical precision) before relying on it here, plus one classical
    sanity check: the hoop stress concentration factor this gives at
    :math:`r=a,\theta=\pi/2` (the pole perpendicular to the load, same
    relative geometry as :func:`_tension_polar_stress`'s own peak) has
    magnitude exactly 2, the well-known antiplane result (vs. 3 for
    :func:`_tension_polar_stress`'s in-plane tension case).

    Returns
    -------
    sigma_r3, sigma_theta3 : ndarray
    """
    a_over_r_sq = (hole_radius / r) ** 2
    sigma_r3 = magnitude * (1.0 - a_over_r_sq) * xp.cos(theta)
    sigma_theta3 = -magnitude * (1.0 + a_over_r_sq) * xp.sin(theta)
    return sigma_r3, sigma_theta3


def _antiplane_inhomogeneity_polar_stress(r, theta, hole_radius, magnitude, contrast):
    r"""Exact closed-form antiplane stress field
    (:math:`\sigma_{r3},\sigma_{\theta 3}`) around an isolated,
    finite-`contrast` circular inhomogeneity (not just the void limit) in
    an infinite isotropic medium under remote antiplane shear `magnitude`
    -- the antiplane counterpart of :func:`_inhomogeneity_polar_stress`,
    matching this project's original (pre-crystallite) ``eshelby.cpp``
    reference's own recipe of solving for and image-summing the
    *finite-contrast* field rather than only its void-limit worked
    example (see :meth:`HoleInPlateCase._antiplane_inhomogeneity_correction_cartesian`).

    `contrast` is the inhomogeneity-to-matrix *shear-modulus* ratio, same
    convention as :func:`_inhomogeneity_polar_stress`. Antiplane
    elasticity's scalar Laplace problem for :math:`u_3` makes this
    considerably simpler to derive than the in-plane biharmonic case: with
    :math:`u_3=\gamma(r+k a^2/r)\cos\theta` outside (:math:`\gamma=
    \mathrm{magnitude}/\mu_0`, :math:`\mu_0` the matrix shear modulus) and
    :math:`u_3=\gamma\,\tfrac{2}{1+\beta}\,r\cos\theta` inside
    (:math:`\beta=` `contrast`), matching :math:`u_3` and the traction
    :math:`\sigma_{r3}=\mu\,\partial u_3/\partial r` at :math:`r=a` gives
    :math:`k=(1-\beta)/(1+\beta)` -- the same displacement-and-traction
    continuity conditions :func:`_inhomogeneity_polar_stress` solves, just
    for a scalar rather than a biharmonic potential. Reduces to
    :func:`_antiplane_polar_stress` exactly at ``contrast=0`` (``k=1``);
    the interior field (:func:`_antiplane_inhomogeneity_interior_stress`)
    is then exactly zero there too, matching a true traction-free void.

    Verified independently of the derivation: traction continuous with
    the interior's uniform stress
    (:func:`_antiplane_inhomogeneity_interior_stress`) at every
    :math:`\theta` for a finite `contrast`, in full pointwise equilibrium
    (finite differences), and reduces to the exact remote `magnitude` far
    from the inhomogeneity.
    """
    beta = contrast
    k = -1.0 if beta == float("inf") else (1.0 - beta) / (1.0 + beta)
    a_over_r_sq = (hole_radius / r) ** 2
    sigma_r3 = magnitude * (1.0 - k * a_over_r_sq) * xp.cos(theta)
    sigma_theta3 = -magnitude * (1.0 + k * a_over_r_sq) * xp.sin(theta)
    return sigma_r3, sigma_theta3


def _antiplane_inhomogeneity_interior_stress(magnitude, contrast):
    r"""Uniform interior Cartesian antiplane stress
    (:math:`\sigma_{13},\sigma_{23}`) inside a circular inhomogeneity
    under remote antiplane shear `magnitude` -- the antiplane counterpart
    of :func:`_inhomogeneity_interior_stress`, and, like that function,
    exactly uniform regardless of position (Eshelby's uniformity theorem
    applies here too, since the antiplane problem is still a linear
    circular-inclusion problem, just for a scalar potential).

    :math:`\sigma_{13}=\mathrm{magnitude}\cdot 2\beta/(1+\beta)` with
    :math:`\beta=` `contrast`, from :math:`\sigma_{13,\mathrm{in}}=\mu_2\,
    \partial u_3/\partial x_1=\mu_2\cdot\gamma\,2/(1+\beta)` (see
    :func:`_antiplane_inhomogeneity_polar_stress`'s docstring for the
    interior displacement) and :math:`\mu_2\gamma=\beta\cdot\mathrm{magnitude}`.
    Vanishes exactly at ``contrast=0``, unlike the in-plane
    :func:`_inhomogeneity_interior_stress` (whose ``sigma_zz`` stays
    nonzero at ``contrast=0`` via the plane-strain constraint) -- a true
    void carries no antiplane stress at all, interior or otherwise.
    """
    beta = contrast
    # beta -> inf (rigid): 2*beta/(1+beta) -> 2
    sigma_13 = magnitude * (2.0 if beta == float("inf") else 2.0 * beta / (1.0 + beta))
    sigma_23 = 0.0 * magnitude
    return sigma_13, sigma_23


def _gradient_void_polar_stress(r, theta, hole_radius, magnitude):
    r"""Closed-form stress field around a traction-free circular void in
    an infinite isotropic plate under a remote stress *gradient*
    :math:`d\sigma_{11}/dx_2=` `magnitude` (i.e. :math:`\sigma_{11} \to
    \mathrm{magnitude}\cdot x_2` far from the hole, :math:`\sigma_{22} =
    \sigma_{12} = 0` far away) -- the "moment"/bending-type loading case.

    Independent of the elastic constants, like :func:`_tension_polar_stress`
    (both are true-void, traction-free-boundary closed forms, not
    equivalent-inclusion approximations). Verified, not just transcribed
    from the reference derivation: traction-free at ``r=hole_radius`` for
    every theta (confirmed algebraically -- the bracketed factor multiplying
    both ``sin(3*theta)`` and ``sin(theta)`` in `sigma_rr`, and both
    ``cos(3*theta)`` and ``cos(theta)`` in `sigma_r_theta`, vanishes
    identically at ``r=hole_radius``); in full pointwise mechanical
    equilibrium (checked by finite differences against the Cartesian form);
    and its far-field limit reduces -- exactly, not just as ``r ->
    infinity``, for every ``(r, theta)`` once the ``1/r**3``/``1/r**5``
    terms are dropped -- to precisely :math:`\sigma_{11}=\mathrm{magnitude}
    \cdot x_2`, :math:`\sigma_{22}=\sigma_{12}=0` with no residual angular
    dependence, confirmed by converting to Cartesian.
    """
    a = hole_radius
    s = -magnitude
    a_over_r_cubed = (a / r) ** 3
    a_over_r_fifth = (a / r) ** 5
    r_over_a = r / a
    sin_t, cos_t = xp.sin(theta), xp.cos(theta)
    sin_3t, cos_3t = xp.sin(3.0 * theta), xp.cos(3.0 * theta)

    sigma_rr = -s * a * (
        (0.25 * r_over_a - 1.25 * a_over_r_cubed + a_over_r_fifth) * sin_3t
        + 0.25 * (r_over_a - a_over_r_cubed) * sin_t
    )
    sigma_theta_theta = s * a * (
        (0.25 * r_over_a - 0.25 * a_over_r_cubed + a_over_r_fifth) * sin_3t
        - 0.25 * (3.0 * r_over_a + a_over_r_cubed) * sin_t
    )
    sigma_r_theta = -s * a * (
        (0.25 * r_over_a + 0.75 * a_over_r_cubed - a_over_r_fifth) * cos_3t
        - 0.25 * (r_over_a - a_over_r_cubed) * cos_t
    )
    return sigma_rr, sigma_theta_theta, sigma_r_theta


def _gradient_void_hole_correction_cartesian(x, y, hole_radius, magnitude):
    r"""The pure hole-induced *correction* to the "moment" background field
    :math:`\sigma_{11}=\mathrm{magnitude}\cdot x_2`, i.e.
    :func:`_gradient_void_polar_stress` with the (exactly, pointwise)
    coincident background subtracted back out -- decays as
    :math:`(a/r)^3`, unlike the full field, so it can be summed over
    periodic images of the hole to build the periodic solution (see
    :meth:`HoleInPlateCase.periodic_gradient_stress`) without ever double
    counting the shared background.
    """
    r = xp.sqrt(x**2 + y**2)
    # Small relative tolerance rather than a bare >=: a point meant to
    # sit exactly on the boundary can round to just inside it after a
    # center-add/center-subtract round trip.
    outside = r >= hole_radius * (1.0 - 1.0e-9)
    r_safe = xp.where(outside, r, hole_radius)  # avoid the 1/r singularity inside the void
    theta = xp.arctan2(y, x)
    srr, stt, srt = _gradient_void_polar_stress(r_safe, theta, hole_radius, magnitude)
    sigma_xx, sigma_yy, sigma_xy = polar_to_cartesian_stress(srr, stt, srt, theta)
    sigma_xx = sigma_xx - magnitude * y
    # Zero inside the void, like _analytic_line's isolated curves elsewhere
    # in this module: a true traction-free hole carries no stress, and
    # (unlike outside) theta is degenerate right on the dx=0 probe line
    # through the disk center, where a clamped-but-nonzero r would
    # otherwise produce a spurious jump at theta=+/-pi/2.
    return (
        xp.where(outside, sigma_xx, 0.0),
        xp.where(outside, sigma_yy, 0.0),
        xp.where(outside, sigma_xy, 0.0),
    )


def _ellipse_gamma(load, magnitude):
    r"""Return the Muskhelishvili remote-loading constants
    :math:`(\Gamma, \Gamma')` for `load`, generally complex --
    :math:`\Gamma=(N_1+N_2)/4`, :math:`\Gamma'=(N_2-N_1)e^{2i\beta}/2` for
    principal remote stresses :math:`N_1,N_2` at angle :math:`\beta` to
    x1. ``"shear"`` (:math:`N_1=\mathrm{magnitude}` at
    :math:`\beta=45^\circ`, :math:`N_2=-\mathrm{magnitude}` at
    :math:`\beta=-45^\circ`) gives a purely imaginary :math:`\Gamma'` --
    the general complex form is what makes a single formula
    (:func:`_ellipse_void_cartesian_stress`) cover both this and
    ``"tension"``/``"biaxial"`` at once, unlike :class:`HoleInPlateCase`'s
    circular-only real-coefficient shortcut.
    """
    if load == "tension":
        return magnitude / 4.0 + 0j, -magnitude / 2.0 + 0j
    if load == "biaxial":
        return magnitude / 2.0 + 0j, 0j
    if load == "shear":
        return 0j, 1j * magnitude
    raise ValueError('load must be "tension", "biaxial", or "shear"')


def _ellipse_exterior_zeta(z, semi_axis_a, semi_axis_b):
    r"""Invert the conformal map :math:`z=\omega(\zeta)=R(\zeta+m/\zeta)`,
    :math:`R=(a+b)/2`, :math:`m=(a-b)/(a+b)`, for exterior point `z`,
    picking the root with :math:`|\zeta|\ge 1` (the exterior of the unit
    disk maps to the exterior of the ellipse) -- shared by
    :func:`_ellipse_void_cartesian_stress` and
    :func:`_ellipse_void_antiplane_cartesian_stress`, the two closed forms
    built on this same mapping.
    """
    semi_r = (semi_axis_a + semi_axis_b) / 2.0
    m = (semi_axis_a - semi_axis_b) / (semi_axis_a + semi_axis_b)
    discriminant = xp.sqrt(z**2 - 4.0 * semi_r**2 * m + 0j)
    zeta_plus = (z + discriminant) / (2.0 * semi_r)
    zeta_minus = (z - discriminant) / (2.0 * semi_r)
    zeta = xp.where(xp.abs(zeta_plus) >= xp.abs(zeta_minus), zeta_plus, zeta_minus)
    # Only exactly zero at the circular special case (m=0) exactly at the
    # ellipse center (z=0) -- always outside this function's domain of
    # validity (it describes the exterior field only), but guarded here
    # anyway to avoid a noisy divide-by-zero warning on every call.
    return xp.where(zeta == 0, 1.0 + 0j, zeta)


def _pole_basis_terms(m, n_power, k_pole):
    r"""Exterior potential basis in :math:`\zeta` (all decaying at
    infinity): the powers :math:`\zeta^{-n}`, :math:`n=1..` `n_power`, plus the
    pole terms :math:`(\zeta^2-m)^{-k}` and :math:`\zeta(\zeta^2-m)^{-k}`,
    :math:`k=1..` `k_pole`. Returns a list of ``(F, dF, ddF)`` callables.

    The exact solution has poles at :math:`\zeta=\pm\sqrt m`, just inside
    the unit disk for a slender ellipse (:math:`|m|\to 1`); a pure power
    series then converges only like :math:`|m|^{n/2}`. Carrying the poles
    explicitly (order 2 suffices; order >= 3 makes the columns near-
    degenerate and hurts) leaves the powers only smooth structure.
    """
    terms = []
    for n in range(1, n_power + 1):
        terms.append((
            lambda z, n=n: z ** (-n),
            lambda z, n=n: -n * z ** (-n - 1),
            lambda z, n=n: n * (n + 1) * z ** (-n - 2),
        ))
    for k in range(1, k_pole + 1):
        terms.append((
            lambda z, k=k: (z**2 - m) ** (-k),
            lambda z, k=k: -2 * k * z * (z**2 - m) ** (-k - 1),
            lambda z, k=k: -2 * k * (z**2 - m) ** (-k - 1)
            + 4 * k * (k + 1) * z**2 * (z**2 - m) ** (-k - 2),
        ))
        terms.append((
            lambda z, k=k: z * (z**2 - m) ** (-k),
            lambda z, k=k: (z**2 - m) ** (-k) - 2 * k * z**2 * (z**2 - m) ** (-k - 1),
            lambda z, k=k: -6 * k * z * (z**2 - m) ** (-k - 1)
            + 4 * k * (k + 1) * z**3 * (z**2 - m) ** (-k - 2),
        ))
    return terms


def _ellipse_gradient_inhomogeneity_solution(
    semi_axis_a, semi_axis_b, contrast, nu, magnitude, n_terms=24, n_collocation=1024,
    n_pole=2,
):
    r"""Coefficients of the exact isolated-ellipse solution for a
    finite-`contrast` elliptical inhomogeneity under the "moment" remote
    stress gradient :math:`\sigma_{11}\to\mathrm{magnitude}\cdot x_2`
    (:math:`\sigma_{22}=\sigma_{12}=0` far away) -- the gradient-loading
    counterpart of :func:`_ellipse_void_cartesian_stress`, extended to
    finite `contrast` (the inhomogeneity-to-matrix shear-modulus ratio,
    ``0`` a void, ``inf`` rigid; both phases share `nu`, as everywhere in
    this module).

    Muskhelishvili plane-strain potentials
    (:math:`\sigma_{11}+\sigma_{22}=4\,\mathrm{Re}\,\Phi`,
    :math:`\sigma_{22}-\sigma_{11}+2i\sigma_{12}=2(\bar z\Phi'+\Psi)`,
    :math:`2\mu(u_1+iu_2)=\kappa\varphi-z\overline{\varphi'}-\overline\psi`,
    :math:`\kappa=3-4\nu`). The remote field has
    :math:`\varphi_0=-i g z^2/8`, :math:`\psi_0=i g z^2/8`. Outside, the
    correction potentials are expanded in :func:`_pole_basis_terms` under the
    conformal map of :func:`_ellipse_exterior_zeta`; a linear remote field
    makes the interior stress exactly linear (Eshelby's polynomial
    uniformity), so the interior potentials are exactly quadratic in
    :math:`z`. Continuity of traction
    (:math:`\varphi+z\overline{\varphi'}+\overline\psi`) and displacement
    at :math:`|\zeta|=1` gives a linear system for the coefficients, solved
    in the least-squares sense on `n_collocation` boundary points, with
    the columns equilibrated (the pole columns are thousands of times
    larger than the power columns for a slender ellipse).

    For ``contrast=inf`` the inclusion is a free rigid body: the exterior
    boundary displacement is :math:`2(w_0+i\omega z)` with the translation
    :math:`w_0` and rotation :math:`\omega` solved for (three extra
    unknowns); clamping it to zero leaves an inconsistent system for
    slender ellipses. Contrasts above ``1e8`` are treated as rigid (the
    difference is below ``1e-8``, and the finite solve sits on a ~1e-6
    round-off floor there). The solve is nondimensionalized by the mean
    semi-axis, since otherwise the interior coefficients scale like
    size\ :sup:`-2` and round-off breaks scale invariance.

    Accuracy: with the defaults (24 power terms, poles to order 2) the
    interface traction is continuous to ~1e-6 (relative to the stress
    scale) for every contrast from a void to rigid and every aspect ratio
    tested from 1/1000 to 1000 (versus 1e-2 to 1e-1 with a pure power series
    of 384 terms beyond about 60:1). Also checked: at ``contrast=1`` the
    correction vanishes identically; at ``a=b`` and ``contrast -> 0`` it
    reproduces :func:`_gradient_void_hole_correction_cartesian`
    (~1e-11); the field is in pointwise equilibrium.
    """
    if contrast > 1.0e8:
        contrast = float("inf")
    rigid = contrast == float("inf")
    length = (semi_axis_a + semi_axis_b) / 2.0
    a, b = semi_axis_a / length, semi_axis_b / length
    m = (a - b) / (a + b)
    scaled_magnitude = magnitude * length  # sigma_11 = g*y = (g*length)*(y/length)
    kappa = 3.0 - 4.0 * nu
    theta = xp.linspace(0.0, 2.0 * xp.pi, n_collocation, endpoint=False)
    s = xp.exp(1j * theta)
    z = s + m / s  # mean semi-axis is 1 in these units
    omega_prime = 1.0 - m / s**2
    terms = _pole_basis_terms(m, n_terms, n_pole)
    n_basis = len(terms)
    values = xp.stack([f(s) for f, _, _ in terms], axis=1)
    slopes = xp.stack([d(s) for _, d, _ in terms], axis=1)
    n_complex = 2 * (2 * n_basis + 6)
    n_unknown = n_complex + (3 if rigid else 0)

    def residual(x, remote):
        c = x[:n_complex][0::2] + 1j * x[:n_complex][1::2]
        a_ext, c_ext = c[:n_basis], c[n_basis : 2 * n_basis]
        a_int, b_int = c[2 * n_basis : 2 * n_basis + 3], c[2 * n_basis + 3 :]
        scale = 1.0 if remote else 0.0
        phi_e = scale * (-1j * scaled_magnitude * z**2 / 8.0) + values @ a_ext
        dphi_e = scale * (-1j * scaled_magnitude * z / 4.0) + (slopes @ a_ext) / omega_prime
        psi_e = scale * (1j * scaled_magnitude * z**2 / 8.0) + values @ c_ext
        phi_i = a_int[0] + a_int[1] * z + a_int[2] * z**2
        dphi_i = a_int[1] + 2.0 * a_int[2] * z
        psi_i = b_int[0] + b_int[1] * z + b_int[2] * z**2
        traction = (phi_e + z * xp.conj(dphi_e) + xp.conj(psi_e)) - (
            phi_i + z * xp.conj(dphi_i) + xp.conj(psi_i)
        )
        disp_e = kappa * phi_e - z * xp.conj(dphi_e) - xp.conj(psi_e)
        disp_i = kappa * phi_i - z * xp.conj(dphi_i) - xp.conj(psi_i)
        # Displacement continuity D_e/mu_0 = D_i/mu_1, scaled to stay well
        # conditioned at both extremes (beta*D_e - D_i for beta <= 1,
        # D_e - D_i/beta for beta > 1); rigid: the boundary displacement is a
        # free rigid-body motion.
        if rigid:
            w0 = x[n_complex] + 1j * x[n_complex + 1]
            spin = x[n_complex + 2]
            disp = disp_e - 2.0 * (w0 + 1j * spin * z)
        elif contrast > 1.0:
            disp = disp_e - disp_i / contrast
        else:
            disp = contrast * disp_e - disp_i
        return xp.concatenate([traction.real, traction.imag, disp.real, disp.imag])

    r0 = residual(xp.zeros(n_unknown), True)
    matrix = xp.empty((r0.size, n_unknown))
    for k in range(n_unknown):
        unit = xp.zeros(n_unknown)
        unit[k] = 1.0
        matrix[:, k] = residual(unit, False)
    norms = xp.linalg.norm(matrix, axis=0)
    norms = xp.where(norms == 0, 1.0, norms)
    x = (xp.linalg.lstsq(matrix / norms, -r0, rcond=None)[0] / norms)[:n_complex]
    c = x[0::2] + 1j * x[1::2]
    return {
        "a": a, "b": b, "m": m, "magnitude": magnitude, "length": length,
        "n_power": n_terms, "n_pole": n_pole,
        "a_ext": c[:n_basis], "c_ext": c[n_basis : 2 * n_basis],
        "a_int": c[2 * n_basis : 2 * n_basis + 3], "b_int": c[2 * n_basis + 3 :],
    }


def _ellipse_gradient_inhomogeneity_correction_cartesian(x, y, solution):
    r"""Cartesian stress *correction* ``(sigma_xx, sigma_yy, sigma_xy)`` of
    the isolated elliptical inhomogeneity solution
    (:func:`_ellipse_gradient_inhomogeneity_solution`) at coordinates
    `x`, `y` relative to the ellipse center: the full field minus the
    coincident background :math:`\sigma_{11}=\mathrm{magnitude}\cdot y`,
    so it decays outside (~:math:`r^{-2}` or faster) and can be summed
    over periodic images without double counting the background --
    exactly the role :func:`_gradient_void_hole_correction_cartesian`
    plays for the true-void circle.
    """
    a, b = solution["a"], solution["b"]  # in units of the mean semi-axis
    m = solution["m"]
    length = solution["length"]
    g = solution["magnitude"] * length  # background g*y == g_scaled * (y/length)
    x, y = xp.asarray(x) / length, xp.asarray(y) / length
    inside = (x / a) ** 2 + (y / b) ** 2 < 1.0
    far = 2.0 * (a + b) + 0j
    z = x + 1j * y
    z_safe = xp.where(inside, far, z)
    zeta = _ellipse_exterior_zeta(z_safe, a, b)
    inv = 1.0 / zeta
    n_power = solution["n_power"]
    power = inv
    phi_z = xp.zeros_like(zeta)
    phi_zz = xp.zeros_like(zeta)
    psi_z = xp.zeros_like(zeta)
    for k in range(1, n_power + 1):
        a_k, c_k = solution["a_ext"][k - 1], solution["c_ext"][k - 1]
        shifted = power * inv
        phi_z = phi_z - k * a_k * shifted
        phi_zz = phi_zz + k * (k + 1) * a_k * shifted * inv
        psi_z = psi_z - k * c_k * shifted
        power = power * inv
    pole_terms = _pole_basis_terms(m, 0, solution["n_pole"])
    for j, (_, d, dd) in enumerate(pole_terms):
        a_j, c_j = solution["a_ext"][n_power + j], solution["c_ext"][n_power + j]
        phi_z = phi_z + a_j * d(zeta)
        phi_zz = phi_zz + a_j * dd(zeta)
        psi_z = psi_z + c_j * d(zeta)
    omega_p = 1.0 - m * inv**2
    omega_pp = 2.0 * m * inv**3
    phi = phi_z / omega_p
    dphi = (phi_zz / omega_p - phi_z * omega_pp / omega_p**2) / omega_p
    psi = psi_z / omega_p

    def cartesian(phi, dphi, psi, zc):
        total = 4.0 * phi.real
        deviator = 2.0 * (xp.conj(zc) * dphi + psi)
        return (
            0.5 * (total - deviator.real),
            0.5 * (total + deviator.real),
            0.5 * deviator.imag,
        )

    ext = cartesian(phi, dphi, psi, z_safe)
    a_int, b_int = solution["a_int"], solution["b_int"]
    inn = cartesian(
        a_int[1] + 2.0 * a_int[2] * z,
        2.0 * a_int[2] * xp.ones_like(z),
        b_int[1] + 2.0 * b_int[2] * z,
        z,
    )
    inn = (inn[0] - g * y, inn[1], inn[2])
    return tuple(xp.where(inside, i, e) for i, e in zip(inn, ext))


def _ellipse_void_antiplane_cartesian_stress(x, y, semi_axis_a, semi_axis_b, magnitude):
    r"""Closed-form antiplane shear stress field
    :math:`(\sigma_{13},\sigma_{23})` around a traction-free elliptical
    void (semi-axis `semi_axis_a` along x1, `semi_axis_b` along x2) in an
    infinite isotropic plate under remote antiplane shear `magnitude`
    (:math:`\sigma_{13}^\infty=` `magnitude`, matching
    :meth:`HoleInPlateCase.remote_antiplane_stress`'s convention) -- the
    antiplane counterpart of :func:`_ellipse_void_cartesian_stress`, via
    the same conformal map (:func:`_ellipse_exterior_zeta`).

    Antiplane elasticity reduces to a scalar Laplace problem for
    :math:`u_3`, considerably simpler than the in-plane biharmonic
    problem :func:`_ellipse_void_cartesian_stress` solves: writing
    :math:`u_3=\mathrm{Re}[F(z)]`, the remote condition
    :math:`u_3\to(\sigma_\infty/\mu)\mathrm{Re}(z)` and the traction-free
    condition :math:`\partial u_3/\partial n=0` on the ellipse boundary
    (equivalently, by conformal invariance, :math:`\partial_r\,\mathrm{Re}[F(\zeta)]=0`
    on :math:`|\zeta|=1`, independent of :math:`\omega`'s own form) are
    solved exactly by :math:`F(\zeta)=(\sigma_\infty R/\mu)(\zeta+1/\zeta)`
    -- checked directly (the boundary radial derivative vanishes for
    every :math:`\theta` only when the ``1/\zeta`` coefficient is
    exactly 1, with no dependence on :math:`m` at all: the ellipse's
    eccentricity enters only through :math:`\omega'(\zeta)` when
    converting back to :math:`\sigma_{i3}=\mu\,\partial_i u_3`, not
    through :math:`F` itself). This gives, after
    :math:`\sigma_{13}-i\sigma_{23}=\mu\,dF/dz=\mu\,F'(\zeta)/\omega'(\zeta)`:

    .. math::

        \sigma_{13}-i\sigma_{23}=\sigma_\infty\,\frac{\zeta^2-1}{\zeta^2-m}

    Independently verified, not just derived: reduces at the circular
    limit `semi_axis_a` == `semi_axis_b` (:math:`m=0`) to the hoop stress
    concentration factor of exactly 2 at the boundary point perpendicular
    to the load (:math:`\theta=\pi/2`) -- the same classical antiplane
    result :func:`_antiplane_polar_stress` independently reproduces (see
    its own docstring), checked here by evaluating this formula directly
    rather than assumed from the analogy.
    """
    z = x + 1j * y
    m = (semi_axis_a - semi_axis_b) / (semi_axis_a + semi_axis_b)
    zeta = _ellipse_exterior_zeta(z, semi_axis_a, semi_axis_b)
    response = magnitude * (zeta**2 - 1.0) / (zeta**2 - m)
    sigma_13 = xp.real(response)
    sigma_23 = -xp.imag(response)
    return sigma_13, sigma_23


def _ellipse_void_cartesian_stress(x, y, semi_axis_a, semi_axis_b, load, magnitude):
    r"""Closed-form stress field around a traction-free elliptical void
    (semi-axis `semi_axis_a` along x1, `semi_axis_b` along x2) in an
    infinite isotropic plate under remote `load` (see
    :func:`_ellipse_gamma`) -- the classical Kirsch/Inglis problem, via
    Muskhelishvili's complex potentials and the conformal map
    :math:`z=\omega(\zeta)=R(\zeta+m/\zeta)`, :math:`R=(a+b)/2`,
    :math:`m=(a-b)/(a+b)`, which sends the exterior of the ellipse
    (:math:`|\zeta|\ge 1`) to the exterior of the unit disk.

    Unlike :func:`_tension_polar_stress` and :func:`_gradient_void_polar_stress`,
    no ready-made closed form for this case was found in this project's
    reference material, so this one was derived from scratch: solving the
    traction-free boundary condition
    :math:`\Phi(\sigma)+\omega(\sigma)\overline{\Phi'(\sigma)}/\overline{\omega'(\sigma)}+\overline{\Psi(\sigma)}=0`
    on :math:`|\sigma|=1` for
    :math:`\Phi(\zeta)=\Gamma R\zeta+P/\zeta` and
    :math:`\Psi(\zeta)=\Gamma'R\zeta+Q_1/\zeta+Q_2/(\zeta^3-m\zeta)`,
    allowing :math:`\Gamma,\Gamma',P,Q_1,Q_2` all complex (needed for
    ``"shear"``'s purely imaginary :math:`\Gamma'`), gives, by exact
    symbolic matching of every power of :math:`\sigma` (checked with
    sympy, not by hand):

    .. math::

        P = -R(m\overline{\Gamma}+\overline{\Gamma'}),\quad
        Q_1 = -R\!\left((m^2{+}1)\Gamma+m^2\overline{\Gamma}+m\overline{\Gamma'}+\overline{\Gamma}\right),\quad
        Q_2 = -R(m^2+1)\!\left(m\Gamma+m\overline{\Gamma}+\overline{\Gamma'}\right)

    (for real :math:`\Gamma,\Gamma'`, i.e. ``"tension"``/``"biaxial"``,
    this reduces to the simpler real-coefficient form
    :math:`P=-R(\Gamma m+\Gamma')` etc. originally derived and checked
    against this same boundary condition).

    Independently verified, not just derived: traction-free at the ellipse
    boundary to machine precision (checked pointwise, using the true
    elliptical outward normal, not just at special points, for every
    `load`); in full pointwise mechanical equilibrium (finite
    differences); reduces to :func:`_tension_polar_stress`'s exact
    ``"tension"`` hoop stress (``-magnitude`` at the tip of the loaded
    axis, ``+3*magnitude`` at the tip of the perpendicular axis) *and* its
    ``"shear"`` field (built independently there via superposed rotated
    tensions, to machine-precision agreement) in the circular limit
    `semi_axis_a` == `semi_axis_b`; and a numerical boundary-collocation
    solve for ``"tension"`` (independent of this closed form, unbounded
    ansatz order) agrees with it to the collocation's own convergence
    tolerance.

    Note the rational (not polynomial) :math:`\zeta^{-1}` dependence of
    :math:`Q_2`'s term: unlike every other closed form in this module, no
    finite-order Laurent polynomial in :math:`1/\zeta` satisfies the
    boundary condition here (confirmed by a numerical collocation sweep
    over increasing polynomial order, whose residual kept shrinking rather
    than vanishing at any finite order) -- an inherent feature of the
    elliptical (non-circular) geometry, not a gap in this derivation.
    """
    semi_r = (semi_axis_a + semi_axis_b) / 2.0
    m = (semi_axis_a - semi_axis_b) / (semi_axis_a + semi_axis_b)
    gamma, gamma_p = _ellipse_gamma(load, magnitude)
    gamma_c, gamma_p_c = xp.conj(gamma), xp.conj(gamma_p)

    z = x + 1j * y
    zeta = _ellipse_exterior_zeta(z, semi_axis_a, semi_axis_b)

    p = -semi_r * (m * gamma_c + gamma_p_c)
    q1 = -semi_r * ((m**2 + 1.0) * gamma + m**2 * gamma_c + m * gamma_p_c + gamma_c)
    q2 = -semi_r * (m**2 + 1.0) * (m * gamma + m * gamma_c + gamma_p_c)

    phi_p = gamma * semi_r - p / zeta**2
    phi_pp = 2.0 * p / zeta**3
    omega_p = semi_r * (1.0 - m / zeta**2)
    omega_pp = 2.0 * semi_r * m / zeta**3
    cubic = zeta**3 - m * zeta
    psi_p = gamma_p * semi_r - q1 / zeta**2 - q2 * (3.0 * zeta**2 - m) / cubic**2

    phi_prime_z = phi_p / omega_p
    phi_pprime_z = (phi_pp * omega_p - phi_p * omega_pp) / omega_p**3
    psi_prime_z = psi_p / omega_p

    trace = 4.0 * xp.real(phi_prime_z)
    deviator = 2.0 * (xp.conj(z) * phi_pprime_z + psi_prime_z)
    sigma_xx = 0.5 * (trace - xp.real(deviator))
    sigma_yy = 0.5 * (trace + xp.real(deviator))
    sigma_xy = 0.5 * xp.imag(deviator)
    return sigma_xx, sigma_yy, sigma_xy


def _ellipse_confocal_lambda(x, y, semi_axis_a, semi_axis_b):
    r"""Confocal-ellipse parameter :math:`\lambda\ge 0` for exterior point
    ``(x, y)``: the unique :math:`\lambda` such that
    :math:`x^2/(a^2+\lambda)+y^2/(b^2+\lambda)=1`, i.e. the semi-axes
    :math:`(\sqrt{a^2+\lambda},\sqrt{b^2+\lambda})` of the ellipse,
    confocal with the base ellipse `semi_axis_a`, `semi_axis_b`, that
    passes through ``(x, y)`` -- the standard 2D elliptic-coordinate
    substitution the classical (plane-strain) elliptical-inhomogeneity
    Eshelby field is built on (the 2D analogue of the ellipsoidal
    coordinates 3D Eshelby theory uses), solving the same quadratic this
    project's original (pre-crystallite) ``eshelby.cpp`` reference's own
    ``f()`` does. Only valid (non-negative, real) outside or on the base
    ellipse; undefined (and never evaluated, guarded by the caller)
    inside it.
    """
    a2, b2 = semi_axis_a**2, semi_axis_b**2
    x2, y2 = x**2, y**2
    return 0.5 * (
        x2 + y2 - a2 - b2
        + xp.sqrt((x2 + y2 - a2 + b2) ** 2 + 4.0 * (a2 - b2) * y2)
    )


def _ellipse_inhomogeneity_exterior_stress(x, y, eps_star, semi_axis_a, semi_axis_b, matrix_lame_mu, nu):
    r"""Stress `(sigma_xx, sigma_yy, sigma_xy)` at exterior point ``(x, y)``
    induced by a uniform eigenstrain `eps_star` (a ``(3, 3)`` tensor, e.g.
    from :func:`_ellipse_equivalent_eigenstrain`) prescribed over an
    elliptical region (`semi_axis_a`, `semi_axis_b`) in an otherwise
    homogeneous matrix (`matrix_lame_mu`, `nu`) -- the classical elliptical
    Eshelby "H-tensor" field, a direct port of this project's original
    (pre-crystallite) ``eshelby.cpp`` reference's ``f()`` (its confocal-
    coordinate H-tensor formulas, restricted to the in-plane Voigt
    components ``0,1,2,5`` -- ``eps_star``'s ``(0,2)``/``(1,2)`` antiplane
    components are always zero for the uniform in-plane loads this project
    uses `eps_star` for, so ``eshelby.cpp``'s own ``lamda3``/``lamda4``
    terms, and its ``sigma3``/``sigma4`` outputs, are dropped here).

    Already exactly the *correction* this project's own convention wants
    (see :meth:`HoleInPlateCase._inhomogeneity_correction_cartesian`'s
    docstring for that convention): by the equivalent-inclusion relation
    :math:`\sigma(x)=\sigma_\infty+C_0:(S\text{-field}-I):\varepsilon^*`,
    this H-tensor contraction is exactly the second (`eps_star`-only) term
    -- it depends on `eps_star` alone, not on any remote background, and
    decays to zero far from the ellipse -- so no separate background
    subtraction is needed here, unlike
    :meth:`HoleInPlateCase._inhomogeneity_correction_cartesian`'s circular
    counterpart (built from :func:`_inhomogeneity_polar_stress`, which is
    parameterized directly by the remote magnitude and *does* need one).

    Note ``eshelby.cpp``'s own ``lamda2`` (the plane-strain
    :math:`\varepsilon^*_{33}` component, generally nonzero even for a
    purely in-plane remote load -- see
    :func:`_ellipse_equivalent_eigenstrain`) still couples back into
    `sigma_xx`/`sigma_yy` through ``H0022``/``H1122`` below, so it cannot
    be dropped the way the antiplane components can.

    Verified for ``"tension"``/``"biaxial"`` three ways (matches
    :func:`_inhomogeneity_polar_stress` in the circular limit
    ``semi_axis_a==semi_axis_b`` to machine precision; matches
    :func:`_ellipse_void_cartesian_stress` at the void-limit eigenstrain;
    traction-continuous with :func:`_ellipse_inhomogeneity_interior_stress`
    at the boundary for a general ``a != b`` ellipse, to machine
    precision) -- and ``"shear"`` now the same way, after a fix (see
    below).

    **Fixed bug, kept documented since the original H0100/H0111/H0101
    formulas below are still ported byte-for-byte from ``eshelby.cpp``'s
    ``f()``.** ``load="shear"`` (nonzero ``eps_star[0, 1]``) used to fail
    the traction-continuity check above by a wide margin (~20% of the
    applied magnitude, not floating-point noise) -- the same latent bug in
    both original (pre-crystallite) C++ references (identical in
    ``eshelby.cpp``'s sibling ``eigenstrain.cpp``, which hardcodes
    ``lamda5=0`` throughout its own ``main()`` and so never actually
    exercises this branch either). Diagnosed directly, not just patched
    until the symptom went away: comparing against an independent
    reference built by rotating the already-correct diagonal/tension case
    45 degrees (exact, by the isolated circular problem's own rotational
    equivariance -- see
    :func:`test_ellipse_inhomogeneity_exterior_shear_matches_the_rotated_diagonal_reference`
    in this project's test suite) showed a clean, exact factor of 2 at
    every point checked, not a fuzzy ~20%. The root cause: ``h0100`` etc.
    below are H-tensor values for a single ``(0,1)`` index pair, but the
    eigenstrain contraction needs *both* symmetric pairs, ``(0,1)`` and
    ``(1,0)`` -- correctly automatic for a diagonal entry like
    ``eps_star[0, 0]`` (no second pair to add), silently wrong for the
    off-diagonal ``eps_star[0, 1]`` alone (its equal partner
    ``eps_star[1, 0]`` was never added in). Fixed by contracting against
    ``2*eps_star[0, 1]`` instead.
    """
    a, b = semi_axis_a, semi_axis_b
    mu = matrix_lame_mu
    lam = _ellipse_confocal_lambda(x, y, a, b)

    rho0 = a / xp.sqrt(a**2 + lam)
    rho1 = b / xp.sqrt(b**2 + lam)

    m0 = x / (a**2 + lam)
    m1 = y / (b**2 + lam)
    m_norm = xp.sqrt(m0**2 + m1**2)
    n0 = m0 / m_norm
    n1 = m1 / m_norm

    denom = a * rho1 + b * rho0
    j0 = rho0**2 * rho1 * b / denom
    j1 = rho1**2 * rho0 * a / denom
    j01 = rho0**3 * rho1**3 / denom**2
    j00 = rho0**4 * rho1 * b * (2.0 * a * rho1 + b * rho0) / (3.0 * a**2 * denom**2)
    j11 = rho1**4 * rho0 * a * (2.0 * b * rho0 + a * rho1) / (3.0 * b**2 * denom**2)

    t5 = rho0**2 + rho1**2 - 4.0 * rho0**2 * n0**2 - 4.0 * rho1**2 * n1**2 - 4.0

    h0000 = mu / (1.0 - nu) * (
        j0 + 3.0 * a**2 * j00
        + rho0 * rho1 * n0**2 * (2.0 - 6.0 * rho0**2 + (8.0 * rho0**2 + t5) * n0**2)
    )
    h1111 = mu / (1.0 - nu) * (
        j1 + 3.0 * b**2 * j11
        + rho0 * rho1 * n1**2 * (2.0 - 6.0 * rho1**2 + (8.0 * rho1**2 + t5) * n1**2)
    )
    h1100 = mu / (1.0 - nu) * (
        a**2 * j01 - j1
        + rho0 * rho1 * (
            1.0 - rho1**2 * n0**2 - rho0**2 * n1**2
            + (4.0 * rho0**2 + 4.0 * rho1**2 + t5) * n0**2 * n1**2
        )
    )
    h0100 = 2.0 * mu * (
        rho0 * rho1 * n0 * n1 / (2.0 * (1.0 - nu))
        * (1.0 - 3.0 * rho0**2 + (6.0 * rho0**2 + 2.0 * rho1**2 + t5) * n0**2)
    )
    h0111 = 2.0 * mu * (
        rho0 * rho1 * n0 * n1 / (2.0 * (1.0 - nu))
        * (1.0 - 3.0 * rho1**2 + (2.0 * rho0**2 + 6.0 * rho1**2 + t5) * n1**2)
    )
    h0122 = 2.0 * mu * (-nu / (1.0 - nu) * rho0 * rho1 * n0 * n1)
    h0022 = 2.0 * mu * (nu / (1.0 - nu) * (j0 - rho0 * rho1 * n0**2))
    h1122 = 2.0 * mu * (nu / (1.0 - nu) * (j1 - rho0 * rho1 * n1**2))
    h0101 = 2.0 * mu * (
        ((1.0 - 2.0 * nu) * (j0 + j1) + (a**2 + b**2) * j01) / (4.0 * (1.0 - nu))
        + rho0 * rho1 / (2.0 * (1.0 - nu)) * (
            (nu - rho1**2) * n0**2 + (nu - rho0**2) * n1**2
            + (4.0 * rho0**2 + 4.0 * rho1**2 + t5) * n0**2 * n1**2
        )
    )

    e0, e1, e2 = eps_star[0, 0], eps_star[1, 1], eps_star[2, 2]
    # The h0100/h0111/h0101 terms contract against the *off-diagonal*
    # (0,1)/(1,0) index pair of the symmetric eigenstrain tensor, unlike
    # e0/e1/e2's diagonal (single-index-pair) terms -- eps_star[0, 1] alone
    # double-counts nothing when it stands for a diagonal entry, but here it
    # must stand for *both* eps_star[0, 1] and eps_star[1, 0] (equal, by
    # symmetry), i.e. 2*eps_star[0, 1], not eps_star[0, 1] alone. Missing
    # this factor of 2 was exactly the shear-coupling bug this function's
    # own docstring used to document (~20% traction-continuity residual,
    # in fact a clean, confirmed factor of 2 -- not ~20% -- once checked
    # against an independent reference built by rotating the already-
    # correct diagonal/tension case 45 degrees rather than assumed).
    e5 = 2.0 * eps_star[0, 1]
    sigma_xx = h0000 * e0 + h1100 * e1 + h0022 * e2 + h0100 * e5
    sigma_yy = h1100 * e0 + h1111 * e1 + h1122 * e2 + h0111 * e5
    sigma_xy = h0100 * e0 + h0111 * e1 + h0122 * e2 + h0101 * e5
    return sigma_xx, sigma_yy, sigma_xy


def _ellipse_inhomogeneity_interior_stress(eps_star, semi_axis_a, semi_axis_b, matrix_lame_mu, nu):
    r"""Uniform interior stress `(sigma_xx, sigma_yy, sigma_xy)` for the
    same equivalent-inclusion problem
    :func:`_ellipse_inhomogeneity_exterior_stress` solves -- a direct port
    of ``eshelby.cpp``'s ``g()`` (its own "T-tensor", the interior
    counterpart of that function's H-tensor), restricted to the in-plane
    Voigt components for the same reason
    :func:`_ellipse_inhomogeneity_exterior_stress` is. Like that function,
    already exactly this project's own *correction* convention (depends
    only on `eps_star`, not on any remote background) -- see its docstring.

    ``T0101`` (the shear-coupling term below) already matched an
    independent reference (:func:`_inhomogeneity_interior_cartesian`'s
    ``"shear"`` branch, itself built by rotating tension rather than via
    an eigenstrain at all) exactly, at the circular limit, even before
    :func:`_ellipse_inhomogeneity_exterior_stress`'s own shear-coupling
    fix (see that function's docstring) -- this side of the boundary-value
    problem was never the buggy half.
    """
    a, b = semi_axis_a, semi_axis_b
    mu = matrix_lame_mu
    denom = (a + b) ** 2

    t0000 = -mu / (1.0 - nu) * a * (2.0 * a + b) / denom
    t1111 = -mu / (1.0 - nu) * b * (2.0 * b + a) / denom
    t1100 = -mu / (1.0 - nu) * a * b / denom
    t0022 = -2.0 * mu * nu / (1.0 - nu) * a / (a + b)
    t1122 = -2.0 * mu * nu / (1.0 - nu) * b / (a + b)
    t0101 = 2.0 * t1100

    e0, e1, e2, e5 = eps_star[0, 0], eps_star[1, 1], eps_star[2, 2], eps_star[0, 1]
    sigma_xx = t0000 * e0 + t1100 * e1 + t0022 * e2
    sigma_yy = t1100 * e0 + t1111 * e1 + t1122 * e2
    sigma_xy = t0101 * e5
    return sigma_xx, sigma_yy, sigma_xy


def _ellipse_inhomogeneity_antiplane_exterior_stress(x, y, eps_star, semi_axis_a, semi_axis_b, matrix_lame_mu):
    r"""Stress `(sigma_13, sigma_23)` at exterior point ``(x, y)`` induced
    by a uniform antiplane eigenstrain (``eps_star[0, 2]``/``eps_star[1, 2]``)
    prescribed over an elliptical region in an otherwise homogeneous
    matrix -- the antiplane counterpart of
    :func:`_ellipse_inhomogeneity_exterior_stress`, needed for a screw
    dislocation's own eps_13 misfit (that function drops the antiplane
    Voigt components entirely -- see its own docstring).

    Not a port of ``eshelby.cpp``'s own antiplane ``H1212``/``H2012``/
    ``H2020`` terms: those turn out to be a genuine (not just missing-a-
    factor-of-2) derivation error, caught the same way the in-plane
    shear-coupling bug was -- checked against the trusted FFT reference
    (:meth:`EllipticalHoleInPlateCase.periodic_prescribed_eigenstrain_solution`)
    over a dilution sweep (``semi_axis_a`` shrinking at fixed aspect ratio,
    to rule out periodicity bias) that converges to a clean, exact 4/3
    ratio against ``eshelby.cpp``'s own ``T2020`` at the circular limit,
    not to 1 -- a periodicity artifact would instead have converged to 1.

    Derived instead from scratch via separation of variables in confocal
    elliptic coordinates (:math:`x=c\cosh\mu\cos\nu`,
    :math:`y=c\sinh\mu\sin\nu`, :math:`c=\sqrt{a^2-b^2}`), the natural
    coordinate system for this antiplane (scalar Laplace) problem, unlike
    the biharmonic in-plane one the Muskhelishvili complex-potential
    machinery elsewhere in this module solves: Eshelby's theorem
    guarantees a uniform interior gradient (linear :math:`u_3`), whose
    boundary trace in these coordinates is *exactly* a single n=1 harmonic
    mode (:math:`\cosh\mu\cos\nu` and :math:`\sinh\mu\sin\nu` only, no
    higher harmonics, since :math:`z(\mu_0,\nu)` is itself a pure n=1
    trig function of :math:`\nu` on the boundary) -- so, unlike the
    in-plane void problem's own need for a rational (not polynomial)
    Laurent term, continuity forces the *entire* exterior field to be the
    same single n=1 mode too, matched exactly (not just approximately) by
    imposing continuity of :math:`u_3` and of the traction jump implied by
    the eigenstrain at :math:`\mu=\mu_0`. Solving that 2-unknown linear
    system and converting back to :math:`z=x+iy` (via
    :math:`z=c\cosh w`, :math:`w=\mu+i\nu`) gives, for
    :math:`g_1=2\varepsilon^*_{13}`, :math:`g_2=2\varepsilon^*_{23}`:

    .. math::

        \sigma_{13}-i\sigma_{23} = -\mu_0\,(g_1+ig_2)\,\frac{ab}{a^2-b^2}
        \left(\frac{z}{\sqrt{z^2-(a^2-b^2)}}-1\right)
        = \frac{-\mu_0\,(g_1+ig_2)\,ab}{\sqrt{z^2-(a^2-b^2)}\left(z+\sqrt{z^2-(a^2-b^2)}\right)}

    (the second, rationalized form -- what this function actually
    evaluates -- removes the first's removable :math:`0/0` singularity at
    the circular limit :math:`a=b`, reducing there to
    :math:`\sigma_{13}-i\sigma_{23}=-\mu_0(g_1+ig_2)ab/(2z^2)`, matched
    directly against a Taylor expansion of the first form).

    Independently verified, not just derived: matches the trusted FFT
    reference to within a few percent (the same residual this project's
    other closed-form-vs-FFT cross-checks show, from the FFT side's own
    periodicity/resolution, not this formula) at a genuinely non-circular
    ellipse, along both the major- and minor-axis directions, and with
    either axis assigned to `semi_axis_a`. No dependence on Poisson ratio
    at all (antiplane elasticity never has one), matching
    ``eshelby.cpp``'s own (otherwise buggy) ``T1212``/``T2020``, which are
    also ``nu``-free.

    Returns
    -------
    sigma_13, sigma_23 : ndarray
    """
    a, b = semi_axis_a, semi_axis_b
    g1, g2 = 2.0 * eps_star[0, 2], 2.0 * eps_star[1, 2]
    z = x + 1j * y
    c2 = a**2 - b**2
    discriminant = xp.sqrt(z**2 - c2 + 0j)
    # Branch matching the large-|z| asymptotic discriminant ~ z (exterior
    # root) -- same style of selection _ellipse_exterior_zeta uses.
    discriminant = xp.where(
        xp.real(discriminant * xp.conj(z)) < 0.0, -discriminant, discriminant
    )
    # (z/discriminant - 1) rationalized to c**2 / [discriminant*(z+discriminant)]
    # (multiply by (z+discriminant)/(z+discriminant)) -- avoids the removable
    # 0/0 singularity the direct form has at the circular limit (c2 -> 0):
    # this rationalized form instead reduces cleanly there to
    # -mu*(g1+i*g2)*a*b/(2*z**2), matching a direct Taylor-expansion check.
    response = (
        -matrix_lame_mu * (g1 + 1j * g2) * a * b / (discriminant * (z + discriminant))
    )
    sigma_13 = xp.real(response)
    sigma_23 = -xp.imag(response)
    return sigma_13, sigma_23


def _ellipse_inhomogeneity_antiplane_interior_stress(eps_star, semi_axis_a, semi_axis_b, matrix_lame_mu):
    r"""Uniform interior stress `(sigma_13, sigma_23)` for the same
    antiplane equivalent-inclusion problem
    :func:`_ellipse_inhomogeneity_antiplane_exterior_stress` solves -- see
    that function's own docstring for the derivation (the same confocal-
    elliptic-coordinate solve gives this directly, as the interior linear
    :math:`u_3`'s own uniform gradient):

    .. math::

        \sigma_{13}=-2\mu_0\frac{a}{a+b}\varepsilon^*_{13},\quad
        \sigma_{23}=-2\mu_0\frac{b}{a+b}\varepsilon^*_{23}

    No cross-coupling between the two components (unlike the in-plane
    ``H0100``-style terms) -- confirmed directly in the derivation itself,
    not assumed: solving the pure-:math:`\varepsilon^*_{13}` and pure-
    :math:`\varepsilon^*_{23}` boundary-value problems independently gives
    an interior gradient depending on only one prescribed component each,
    matching ``eshelby.cpp``'s own ``g()`` having no ``T2012``-style
    interior cross term either.
    """
    a, b = semi_axis_a, semi_axis_b
    mu = matrix_lame_mu
    sigma_13 = -2.0 * mu * a / (a + b) * eps_star[0, 2]
    sigma_23 = -2.0 * mu * b / (a + b) * eps_star[1, 2]
    return sigma_13, sigma_23


@dataclass(frozen=True)
class HoleInPlateCase:
    r"""A circular hole (soft inclusion) in a periodic 2D isotropic plate,
    with the analytic Kirsch stress field for validation.

    The hole is represented as a region of much softer material
    (`contrast` times the matrix's Lame parameters, not literally zero --
    a true void makes the elasticity operator singular there) centered in
    the grid's first two coordinates -- the same "diffuse/soft
    approximation of a sharp feature, quantified rather than assumed"
    idea as :class:`crystallite.verification.CircularGrainCase`. Valid
    against Kirsch's *infinite*-plate solution only for `hole_radius`
    small relative to the domain, since the FFT solver actually sees a
    periodic array of holes.

    Parameters
    ----------
    grid : Grid
        A 2D grid (``grid.shape[2] == 1``).
    matrix_lame_lambda, matrix_lame_mu : float
        Lame parameters of the matrix material.
    hole_radius : float
    contrast : float, default=1.0e-3
        Ratio of the hole's Lame parameters to the matrix's.
    center : tuple of float, default=(0.5, 0.5)
        Hole center in the grid's first two coordinates.
    """

    grid: Grid
    matrix_lame_lambda: float
    matrix_lame_mu: float
    hole_radius: float
    contrast: float = 1.0e-3
    center: tuple = (0.5, 0.5)
    smoothing_width: float = 0.0
    dealias: bool = False

    def __post_init__(self):
        if self.grid.shape[2] != 1:
            raise ValueError("HoleInPlateCase requires a 2D grid (shape[2] == 1)")

    @property
    def _relative_x(self):
        x = self.grid.x[0] - self.center[0]
        y = self.grid.x[1] - self.center[1]
        return x, y

    @property
    def radius(self):
        """Distance from the hole center, shape ``grid.shape``."""
        x, y = self._relative_x
        return xp.broadcast_to(xp.sqrt(x**2 + y**2), self.grid.shape)

    @property
    def angle(self):
        """Polar angle from the hole center, shape ``grid.shape``."""
        x, y = self._relative_x
        return xp.broadcast_to(xp.arctan2(y, x), self.grid.shape)

    def lame_fields(self):
        """Return ``(lam(x), mu(x))``: matrix constants outside the hole,
        ``contrast`` times them inside.

        Three ways to handle the sharp material discontinuity, in
        increasing order of how close to the boundary the *numerical*
        solve stays trustworthy:

        - `smoothing_width=0`, `dealias=False` (both defaults): a literal
          sharp cutoff, which leaves real Gibbs-ringing right at the
          boundary in the spectral stress field.
        - `smoothing_width>0`: blend the two values with a tanh profile
          over that many length units -- the same "diffuse rather than
          sharp" treatment used for every other interface in this
          project. Robust and always in ``[contrast, 1]``, but smears the
          transition over a width you choose, which is *also* roughly how
          close to the boundary a comparison against a sharp-boundary
          analytic solution can be trusted.
        - `dealias=True`: keep the indicator sharp, but suppress the
          spectral ringing directly by multiplying its Fourier transform
          by ``grid.lanczos_filter`` (the same dealiasing filter
          :class:`crystallite.mass_diffusion.MassDiffusion` uses) instead
          of smearing it in real space first. Only ``smoothing_width`` or
          `dealias` should be set at a time -- `dealias` is checked first.
          Clipped to ``[0, 1]`` after filtering: an unclipped
          Lanczos-filtered step rings slightly negative right at the
          jump, which pushed the local stiffness negative and broke the
          CG solver's positive-definiteness assumption outright (observed
          directly -- it failed to converge, residual stalled around 0.4
          for 8000 iterations, not merely a small accuracy loss) before
          this clip was added.
        """
        r = self.radius
        if self.dealias:
            indicator = xp.where(r < self.hole_radius, 0.0, 1.0).astype(self.grid.real_dtype)
            blend = self.grid.ifft(self.grid.fft(indicator) * self.grid.lanczos_filter)
            blend = xp.clip(blend, 0.0, 1.0)
        elif self.smoothing_width > 0:
            blend = 0.5 * (1.0 + xp.tanh((r - self.hole_radius) / self.smoothing_width))
        else:
            blend = xp.where(r < self.hole_radius, 0.0, 1.0)
        # A literal contrast=inf (rigid inclusion) hits inf-inf in the
        # formula below at blend=1 (the matrix, where it should reduce
        # exactly to matrix_lame_lambda/mu) -- substitute a large but
        # finite practical value for the *numerical* field only; the
        # analytic methods (analytic_stress, periodic_analytic_solution)
        # use the exact contrast -> infinity closed-form limit instead
        # (see _inhomogeneity_polar_stress/_inhomogeneity_interior_stress).
        # 1e3, not something more extreme: checked directly (a 1e3/1e4/
        # 1e5/1e6 sweep) that this project's float32 grid arithmetic
        # develops real, growing interior-stress noise above this --
        # 1e6 gave a max interior |sigma| of ~10 (vs. a well-behaved ~1.5
        # analytically), clearly numerical ill-conditioning, not signal.
        contrast = 1.0e3 if self.contrast == float("inf") else self.contrast
        lam = (contrast + (1.0 - contrast) * blend) * self.matrix_lame_lambda
        mu = (contrast + (1.0 - contrast) * blend) * self.matrix_lame_mu
        return lam.astype(self.grid.real_dtype), mu.astype(self.grid.real_dtype)

    def solver(self):
        """Return the :class:`ElasticDeformation` solver for this case."""
        lam, mu = self.lame_fields()
        return ElasticDeformation(
            self.grid,
            lame_lambda=lam,
            lame_mu=mu,
            reference_lame_lambda=self.matrix_lame_lambda,
            reference_lame_mu=self.matrix_lame_mu,
        )

    def remote_stress(self, load, magnitude):
        """Return the ``(3, 3)`` remote stress tensor for `load`.

        Parameters
        ----------
        load : {"tension", "compression", "shear", "biaxial"}
        magnitude : float
            ``sigma_inf`` (tension/compression/biaxial) or ``tau_inf`` (shear).
        """
        sigma = xp.zeros((3, 3))
        if load == "tension":
            sigma[0, 0] = magnitude
        elif load == "compression":
            sigma[0, 0] = -magnitude
        elif load == "shear":
            sigma[0, 1] = sigma[1, 0] = magnitude
        elif load == "biaxial":
            sigma[0, 0] = sigma[1, 1] = magnitude
        else:
            raise ValueError('load must be "tension", "compression", "shear", or "biaxial"')
        return sigma

    def macro_strain(self, load, magnitude, solver=None):
        """Return the macroscopic strain reproducing `remote_stress` far
        from the hole, via the matrix's own compliance.

        Exact here (not an approximate homogenization) because the
        reference medium *is* the actual matrix material.
        """
        solver = solver if solver is not None else self.solver()
        sigma = self.remote_stress(load, magnitude)
        return solver.reference_green.apply_compliance(sigma)

    def analytic_stress(self, r, theta, load, magnitude):
        """Return the analytic ``(sigma_rr, sigma_theta_theta,
        sigma_r_theta)`` at polar position ``(r, theta)`` (center-relative)
        -- the void (``contrast=0``) limit, via
        :func:`_dispatch_tension_polar_stress` combining
        :func:`_tension_polar_stress`. See :meth:`inhomogeneity_analytic_stress`
        for this case's actual, finite-`contrast` counterpart.
        """
        return _dispatch_tension_polar_stress(
            _tension_polar_stress, r, theta, self.hole_radius, magnitude, load
        )

    def inhomogeneity_analytic_stress(self, r, theta, load, magnitude):
        r"""Contrast-aware analog of :meth:`analytic_stress`: the exact
        isolated closed form for this case's actual, finite `contrast`
        (:func:`_inhomogeneity_polar_stress`) rather than the void-only
        :func:`_tension_polar_stress` -- reduces to :meth:`analytic_stress`
        exactly at ``contrast=0`` (already verified directly by
        ``test_inhomogeneity_formula_reduces_to_the_void_formula_at_zero_contrast``).
        """
        nu = self.matrix_lame_lambda / (2.0 * (self.matrix_lame_lambda + self.matrix_lame_mu))

        def formula(r, theta, hole_radius, magnitude):
            return _inhomogeneity_polar_stress(r, theta, hole_radius, magnitude, self.contrast, nu)

        return _dispatch_tension_polar_stress(
            formula, r, theta, self.hole_radius, magnitude, load
        )

    def hoop_stress_at_hole(self, theta, load, magnitude):
        """Analytic hoop stress ``sigma_theta_theta`` at ``r=hole_radius``."""
        _, sigma_theta_theta, _ = self.analytic_stress(
            self.hole_radius, theta, load, magnitude
        )
        return sigma_theta_theta

    def remote_antiplane_stress(self, magnitude):
        r"""Return the ``(3, 3)`` remote antiplane shear stress tensor:
        :math:`\sigma_{13}=\sigma_{31}=` `magnitude`, :math:`\sigma_{23}=0`
        -- the antiplane counterpart of :meth:`remote_stress`, kept
        separate rather than added as another `load` option there: unlike
        the four in-plane loads, this involves genuinely different tensor
        components (:math:`\sigma_{i3}`, not :math:`\sigma_{\alpha\beta}`
        for in-plane :math:`\alpha,\beta`), and the correction/periodic
        machinery built around :meth:`remote_stress` (:meth:`_void_correction_cartesian`,
        :meth:`_periodic_image_sum`, etc.) is hardcoded to the in-plane
        ``(0,0)``/``(1,1)``/``(0,1)`` components -- see
        :meth:`_antiplane_void_correction_cartesian`/
        :meth:`periodic_antiplane_void_stress` for the parallel antiplane
        versions of that machinery instead of trying to reuse it here.

        :math:`\sigma_{13}` (not :math:`\sigma_{23}`) is the loaded
        component to match this project's own convention of measuring the
        in-plane loads' own :math:`\theta=0` along the axis they act on
        (:meth:`remote_stress`'s :math:`\sigma_{11}` for "tension") --
        confirmed against this project's original (pre-crystallite)
        reference material's own "Mode III crack" illustration, which
        loads a vertical crack the same horizontal direction (axis 1) as
        its "Mode I"/"Mode II" tension/shear illustrations do, not the
        perpendicular one.
        """
        sigma = xp.zeros((3, 3))
        sigma[0, 2] = sigma[2, 0] = magnitude
        return sigma

    def macro_antiplane_strain(self, magnitude, solver=None):
        """Antiplane counterpart of :meth:`macro_strain`: the macroscopic
        strain reproducing :meth:`remote_antiplane_stress` far from the
        hole, via the matrix's own compliance."""
        solver = solver if solver is not None else self.solver()
        sigma = self.remote_antiplane_stress(magnitude)
        return solver.reference_green.apply_compliance(sigma)

    def antiplane_analytic_stress(self, r, theta, magnitude):
        """Return the analytic ``(sigma_r3, sigma_theta3)`` at polar
        position ``(r, theta)`` (center-relative) for the circular hole
        under remote antiplane shear `magnitude` -- the antiplane
        counterpart of :meth:`analytic_stress`. Only one load case exists
        here (unlike :meth:`analytic_stress`'s four), so there is no
        tension-superposition dispatch to do.
        """
        return _antiplane_polar_stress(r, theta, self.hole_radius, magnitude)

    def antiplane_inhomogeneity_analytic_stress(self, r, theta, magnitude):
        r"""Contrast-aware analog of :meth:`antiplane_analytic_stress`: the
        exact isolated closed form for this case's actual `contrast`
        (:func:`_antiplane_inhomogeneity_polar_stress`) rather than the
        void-only :func:`_antiplane_polar_stress` -- reduces to
        :meth:`antiplane_analytic_stress` exactly at ``contrast=0``.
        """
        return _antiplane_inhomogeneity_polar_stress(
            r, theta, self.hole_radius, magnitude, self.contrast
        )

    def _antiplane_void_correction_cartesian(self, x, y, magnitude):
        r"""Local (image-frame) correction to the uniform remote
        `magnitude` antiplane background for a single copy of the
        traction-free void centered at the origin of ``(x, y)`` -- the
        antiplane counterpart of :meth:`_void_correction_cartesian`.
        Zero net stress inside the void, same reasoning as that method.
        """
        background = self.remote_antiplane_stress(magnitude)
        r = xp.sqrt(x**2 + y**2)
        outside = r >= self.hole_radius * (1.0 - 1.0e-9)
        r_safe = xp.where(outside, r, self.hole_radius)
        theta = xp.arctan2(y, x)
        sigma_r3, sigma_theta3 = self.antiplane_analytic_stress(r_safe, theta, magnitude)
        sigma_13, sigma_23 = polar_to_cartesian_antiplane_stress(sigma_r3, sigma_theta3, theta)
        sigma_13 = sigma_13 - background[0, 2]
        sigma_23 = sigma_23 - background[1, 2]
        return (
            xp.where(outside, sigma_13, -background[0, 2]),
            xp.where(outside, sigma_23, -background[1, 2]),
        )

    def periodic_antiplane_void_stress(self, magnitude, n_images=1):
        r"""Periodic-array antiplane stress field for a traction-free void
        under uniform remote antiplane shear `magnitude`, built by
        summing the exact isolated closed form
        (:meth:`antiplane_analytic_stress`) over periodic images -- the
        antiplane counterpart of :meth:`periodic_void_stress`, with the
        same two-pass structure :meth:`_periodic_image_sum` uses (see
        that method's docstring): the domain mean of the *raw* image sum
        feeds the outside recentering, and interior points are separately
        replaced by the uncontaminated home-image-only value, rather than
        summed and recentered the same way as the exterior.

        Not built on top of :meth:`_periodic_image_sum` itself: that
        method's recentering and interior-overwrite logic is hardcoded to
        three named in-plane components (``sigma_xx``, ``sigma_yy``,
        ``sigma_xy``, i.e. tensor indices ``(0,0)``, ``(1,1)``, ``(0,1)``)
        threaded through :meth:`remote_stress`; forcing the two antiplane
        components (``(0,2)``, ``(1,2)``) through that same shape would
        obscure more than it would save.

        Parameters
        ----------
        magnitude : float
        n_images : int, default=1

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape`` -- only ``sigma_13``/``sigma_31``
            and ``sigma_23``/``sigma_32`` are populated.
        """
        x, y = self.grid.x[0], self.grid.x[1]
        length_x, length_y = self.grid.lengths[0], self.grid.lengths[1]
        background = self.remote_antiplane_stress(magnitude)

        sigma_13 = background[0, 2] * xp.ones(self.grid.shape)
        sigma_23 = background[1, 2] * xp.ones(self.grid.shape)

        home_dx = x - self.center[0]
        home_dy = y - self.center[1]
        home_13, home_23 = self._antiplane_void_correction_cartesian(home_dx, home_dy, magnitude)

        for n1 in range(-n_images, n_images + 1):
            for n2 in range(-n_images, n_images + 1):
                dx = x - (self.center[0] + n1 * length_x)
                dy = y - (self.center[1] + n2 * length_y)
                s13, s23 = self._antiplane_void_correction_cartesian(dx, dy, magnitude)
                sigma_13 = sigma_13 + s13
                sigma_23 = sigma_23 + s23

        outside = self.radius >= self.hole_radius
        recentered_13 = sigma_13 - (xp.mean(sigma_13) - background[0, 2])
        recentered_23 = sigma_23 - (xp.mean(sigma_23) - background[1, 2])

        sigma_13 = xp.where(outside, recentered_13, background[0, 2] + home_13)
        sigma_23 = xp.where(outside, recentered_23, background[1, 2] + home_23)

        stress = xp.zeros((3, 3) + self.grid.shape, dtype=sigma_13.dtype)
        stress[0, 2] = stress[2, 0] = sigma_13
        stress[1, 2] = stress[2, 1] = sigma_23
        return stress

    def _antiplane_inhomogeneity_correction_cartesian(self, x, y, magnitude, eps_star=None):
        r"""Contrast-aware analog of
        :meth:`_antiplane_void_correction_cartesian`: local correction for
        a single isolated copy of this case's *actual*, finite-`contrast`
        inhomogeneity, not the void limit -- exterior from
        :meth:`antiplane_inhomogeneity_analytic_stress`, interior from the
        uniform :func:`_antiplane_inhomogeneity_interior_stress` tensor
        (generally nonzero, unlike the void case's exactly-zero interior).
        Used by :meth:`periodic_antiplane_inhomogeneity_stress`, the
        antiplane counterpart of :meth:`_inhomogeneity_correction_cartesian`
        -- this project's original (pre-crystallite) ``eshelby.cpp``
        reference's own recipe (its ``e()``/``f()``/``g()`` functions
        image-summed over a lattice), generalized here from that
        reference's void-only worked example to this case's actual
        `contrast`, same as :meth:`_inhomogeneity_correction_cartesian`
        already does for the in-plane loads.
        """
        if eps_star is not None:
            return _antiplane_eigenstrain_correction(
                x, y, eps_star, self.hole_radius, self.hole_radius, self.matrix_lame_mu
            )
        background = self.remote_antiplane_stress(magnitude)
        r = xp.sqrt(x**2 + y**2)
        outside = r >= self.hole_radius * (1.0 - 1.0e-9)
        r_safe = xp.where(outside, r, self.hole_radius)
        theta = xp.arctan2(y, x)
        sigma_r3, sigma_theta3 = self.antiplane_inhomogeneity_analytic_stress(
            r_safe, theta, magnitude
        )
        sigma_13, sigma_23 = polar_to_cartesian_antiplane_stress(sigma_r3, sigma_theta3, theta)
        sigma_13 = sigma_13 - background[0, 2]
        sigma_23 = sigma_23 - background[1, 2]

        interior_13, interior_23 = _antiplane_inhomogeneity_interior_stress(
            magnitude, self.contrast
        )
        interior_13 = interior_13 - background[0, 2]
        interior_23 = interior_23 - background[1, 2]
        return (
            xp.where(outside, sigma_13, interior_13),
            xp.where(outside, sigma_23, interior_23),
        )

    def periodic_antiplane_inhomogeneity_stress(self, magnitude, n_images=1):
        r"""Periodic-array antiplane stress field for this case's actual,
        finite-`contrast` circular inhomogeneity under uniform remote
        antiplane shear `magnitude`, built by summing the exact isolated
        closed form (:meth:`antiplane_inhomogeneity_analytic_stress`
        outside, :func:`_antiplane_inhomogeneity_interior_stress` inside)
        over periodic images -- the antiplane counterpart of
        :meth:`periodic_inhomogeneity_stress`, sharing
        :meth:`periodic_antiplane_void_stress`'s own two-pass structure
        (not :meth:`_periodic_image_sum`, for the same reason that
        method's docstring gives: its recentering/interior-overwrite logic
        is hardcoded to the in-plane ``sigma_xx``/``sigma_yy``/``sigma_xy``
        components, not the antiplane ``(0,2)``/``(1,2)`` ones).

        The contrast-aware generalization of
        :meth:`periodic_antiplane_void_stress`, fixing *two* things,
        exactly paralleling :meth:`periodic_inhomogeneity_stress`: the
        void-vs-actual-`contrast` mismatch (this case's own diffuse
        numerical boundary is never a literal void), and the recentering
        target itself -- rather than forcing the far field to the nominal
        `magnitude` (correct only under a stress-controlled boundary
        condition, ``eshelby.cpp``'s own literal recipe), this recenters
        to :meth:`periodic_antiplane_analytic_solution`'s own domain-mean
        stress, which matches this solver's actual strain-controlled
        boundary condition -- see :meth:`_periodic_image_sum`'s docstring
        for the direct numeric confirmation of this same gap for the
        in-plane loads.

        Parameters
        ----------
        magnitude : float
        n_images : int, default=1

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape`` -- only ``sigma_13``/``sigma_31``
            and ``sigma_23``/``sigma_32`` are populated.
        """
        x, y = self.grid.x[0], self.grid.x[1]
        length_x, length_y = self.grid.lengths[0], self.grid.lengths[1]
        background = self.remote_antiplane_stress(magnitude)

        mean_stress = self.periodic_antiplane_mean_stress(magnitude)
        eps_star = self.periodic_antiplane_equivalent_eigenstrain(magnitude)
        target_13 = mean_stress[0, 2]
        target_23 = mean_stress[1, 2]

        sigma_13 = background[0, 2] * xp.ones(self.grid.shape)
        sigma_23 = background[1, 2] * xp.ones(self.grid.shape)


        for n1 in range(-n_images, n_images + 1):
            for n2 in range(-n_images, n_images + 1):
                dx = x - (self.center[0] + n1 * length_x)
                dy = y - (self.center[1] + n2 * length_y)
                s13, s23 = self._antiplane_inhomogeneity_correction_cartesian(
                    dx, dy, magnitude, eps_star=eps_star
                )
                sigma_13 = sigma_13 + s13
                sigma_23 = sigma_23 + s23

        outside = self.radius >= self.hole_radius
        interior = self.periodic_antiplane_interior_stress(magnitude)
        recentered_13 = sigma_13 - _exterior_shift(sigma_13, outside, interior[0, 2], target_13)
        recentered_23 = sigma_23 - _exterior_shift(sigma_23, outside, interior[1, 2], target_23)
        sigma_13 = xp.where(outside, recentered_13, interior[0, 2])
        sigma_23 = xp.where(outside, recentered_23, interior[1, 2])

        stress = xp.zeros((3, 3) + self.grid.shape, dtype=sigma_13.dtype)
        stress[0, 2] = stress[2, 0] = sigma_13
        stress[1, 2] = stress[2, 1] = sigma_23
        return stress

    def periodic_eshelby_tensor(self):
        """Numerically probed periodic Eshelby tensor for this case's
        actual `hole_radius`/domain ratio -- see :func:`_periodic_eshelby_tensor`.

        Not cached: cheap relative to the CG-iterated numerical solve
        elsewhere in this class (three direct, non-iterative periodic
        Fourier solves), and this project favors that simplicity over a
        cache that would need invalidating if a frozen dataclass field
        were ever replaced via ``dataclasses.replace``.
        """
        return _periodic_eshelby_tensor(
            self.periodic_prescribed_eigenstrain_solution, self.radius < self.hole_radius
        )

    def periodic_analytic_solution(self, load, magnitude):
        r"""Exact analytic solution for the *periodic* array problem this
        grid actually represents -- not the isolated-hole approximation
        :meth:`analytic_stress` gives.

        Replaces the true finite-contrast inhomogeneity with an Eshelby
        equivalent eigenstrain (:func:`_periodic_equivalent_eigenstrain`,
        calibrated against this case's own numerically probed
        :meth:`periodic_eshelby_tensor` rather than the isolated closed
        form -- see that function's docstring for why the isolated
        calibration is wrong whenever the hole is not small compared to
        the periodic cell) over a matrix-material disk, then solves the
        resulting periodic (Khachaturyan-Shatalov) eigenstrain problem
        exactly in Fourier space, reusing
        :meth:`ElasticDeformation.reference_green` directly (a
        homogeneous-medium body-force solve is exactly what a fixed
        eigenstrain distribution needs) with a purely analytic source term
        built from :func:`_disk_fourier_transform` -- no discretized
        real-space eigenstrain array, no CG iteration.

        Returns
        -------
        strain, stress : ndarray
            Shape ``(3, 3) + grid.shape``, in the same convention as
            :class:`crystallite.elastic_deformation.ElasticSolution`.
        """
        solver = self.solver()
        green = solver.reference_green
        operator = solver.operator

        eps_star = _periodic_equivalent_eigenstrain(
            magnitude, self.contrast, self.matrix_lame_lambda, self.matrix_lame_mu, load,
            self.periodic_eshelby_tensor(),
        )
        sigma_star = _isotropic_stress_apply(
            eps_star, self.matrix_lame_lambda, self.matrix_lame_mu
        )
        shape_hat = _disk_fourier_transform(
            self.grid, self.hole_radius, center=(self.center[0], self.center[1], 0.0)
        )

        eps0_hat = eps_star[:, :, None, None, None] * shape_hat[None, None, ...]
        sigma_star_hat = sigma_star[:, :, None, None, None] * shape_hat[None, None, ...]
        source_hat = -operator.div(sigma_star_hat)

        eps_bar = self.macro_strain(load, magnitude, solver=solver)
        eps_hat = green.field(source_hat, uniform_field=eps_bar)

        # sigma(x) = C0:(eps(x) - eps0(x)) everywhere, by the equivalent
        # inclusion construction -- same isotropic Hooke's law pattern as
        # ElasticDeformation.stress(), applied to the eigenstrain-corrected
        # strain instead of the true heterogeneous field.
        difference = eps_hat - eps0_hat
        trace = xp.einsum("ii...->...", difference)
        delta = xp.eye(3, dtype=difference.dtype).reshape((3, 3, 1, 1, 1))
        sigma_hat = self.matrix_lame_lambda * trace[None, None, ...] * delta + (
            2.0 * self.matrix_lame_mu * difference
        )

        strain = self.grid.ifft(eps_hat)
        stress = self.grid.ifft(sigma_hat)
        return strain, stress

    def periodic_eshelby_antiplane_tensor(self):
        """Numerically probed periodic antiplane Eshelby tensor component
        ``s1313`` for this case's actual `hole_radius`/domain ratio -- see
        :func:`_periodic_eshelby_antiplane_tensor`. The antiplane
        counterpart of :meth:`periodic_eshelby_tensor`; not cached, same
        reasoning as that method.
        """
        return _periodic_eshelby_antiplane_tensor(
            self.periodic_prescribed_eigenstrain_solution, self.radius < self.hole_radius
        )

    def periodic_antiplane_analytic_solution(self, magnitude):
        r"""Exact analytic solution for the *periodic* antiplane array
        problem this grid actually represents -- the antiplane counterpart
        of :meth:`periodic_analytic_solution`, sharing that method's exact
        construction (equivalent eigenstrain over a matrix-material disk,
        solved exactly in Fourier space via
        :meth:`ElasticDeformation.reference_green`), just with
        :func:`_antiplane_periodic_equivalent_eigenstrain`/
        :meth:`periodic_eshelby_antiplane_tensor` in place of
        :func:`_periodic_equivalent_eigenstrain`/:meth:`periodic_eshelby_tensor`,
        and :meth:`macro_antiplane_strain` in place of :meth:`macro_strain`.

        Nothing in :meth:`ElasticDeformation.reference_green` or
        :func:`~crystallite.spectral.short_range.DifferentialOperators.div`
        is actually specific to the four in-plane load cases
        :meth:`periodic_analytic_solution` was written for -- both operate
        on a fully general ``(3, 3)`` eigenstrain/stress tensor, so an
        eigenstrain with only the ``(0,2)``/``(2,0)`` antiplane components
        populated is solved exactly the same way, with the in-plane
        displacement response coming back out exactly zero (verified
        directly, not just argued): an isotropic reference medium with no
        :math:`x_3`-variation has no elastic coupling between a pure
        :math:`\varepsilon^*_{13}` eigenstrain's body-force source (which
        lands entirely in the :math:`i=3` equilibrium equation -- see
        :func:`_periodic_eshelby_antiplane_tensor`'s docstring) and the
        in-plane (:math:`i=1,2`) equations.

        Returns
        -------
        strain, stress : ndarray
            Shape ``(3, 3) + grid.shape`` -- only ``sigma_13``/``sigma_31``
            and ``sigma_23``/``sigma_32`` (and correspondingly ``eps_13``/
            ``eps_23``) come back nonzero.
        """
        solver = self.solver()
        green = solver.reference_green
        operator = solver.operator

        s1313 = self.periodic_eshelby_antiplane_tensor()
        eps_star = _antiplane_periodic_equivalent_eigenstrain(
            magnitude, self.contrast, self.matrix_lame_mu, s1313
        )
        sigma_star = _isotropic_stress_apply(
            eps_star, self.matrix_lame_lambda, self.matrix_lame_mu
        )
        shape_hat = _disk_fourier_transform(
            self.grid, self.hole_radius, center=(self.center[0], self.center[1], 0.0)
        )

        eps0_hat = eps_star[:, :, None, None, None] * shape_hat[None, None, ...]
        sigma_star_hat = sigma_star[:, :, None, None, None] * shape_hat[None, None, ...]
        source_hat = -operator.div(sigma_star_hat)

        eps_bar = self.macro_antiplane_strain(magnitude, solver=solver)
        eps_hat = green.field(source_hat, uniform_field=eps_bar)

        difference = eps_hat - eps0_hat
        trace = xp.einsum("ii...->...", difference)
        delta = xp.eye(3, dtype=difference.dtype).reshape((3, 3, 1, 1, 1))
        sigma_hat = self.matrix_lame_lambda * trace[None, None, ...] * delta + (
            2.0 * self.matrix_lame_mu * difference
        )

        strain = self.grid.ifft(eps_hat)
        stress = self.grid.ifft(sigma_hat)
        return strain, stress

    def eigenstrain_field(self, eigenstrain_tensor):
        """Real-space eigenstrain field for :meth:`periodic_prescribed_eigenstrain_solution`'s
        numeric counterpart: `eigenstrain_tensor` inside the hole radius,
        zero outside, sharp cutoff (matching that method's own sharp-disk
        construction) -- shape ``(3, 3) + grid.shape``.
        """
        eigenstrain_tensor = xp.asarray(eigenstrain_tensor, dtype=self.grid.real_dtype)
        indicator = xp.where(self.radius < self.hole_radius, 1.0, 0.0).astype(
            self.grid.real_dtype
        )
        return eigenstrain_tensor[:, :, None, None, None] * indicator[None, None, ...]

    def body_force_field(self, force_vector):
        """Real-space body force field for
        :meth:`periodic_body_force_solution`'s numeric counterpart
        (:meth:`crystallite.elastic_deformation.ElasticDeformation.solve`'s
        `body_force` parameter): `force_vector` inside the hole radius,
        zero outside, sharp cutoff -- shape ``(3,) + grid.shape``. Same
        construction as :meth:`eigenstrain_field`, one tensor order down.
        """
        force_vector = xp.asarray(force_vector, dtype=self.grid.real_dtype)
        indicator = xp.where(self.radius < self.hole_radius, 1.0, 0.0).astype(
            self.grid.real_dtype
        )
        return force_vector[:, None, None, None] * indicator[None, ...]

    def periodic_prescribed_eigenstrain_solution(self, eigenstrain_tensor):
        r"""Exact analytic periodic solution for a uniform eigenstrain
        `eigenstrain_tensor` prescribed directly within the hole region --
        unlike :meth:`periodic_analytic_solution`, not derived from
        matching a remote-loaded, finite-contrast inhomogeneity's interior
        stress. Physically a *homogeneous* matrix with a stress-free
        transformation strain in a disk (the classical Eshelby
        "inclusion" problem -- no stiffness contrast involved at all,
        despite the reference material's own "cylindrical inhomogeneity"
        naming for this case), zero remote strain.

        Same Fourier construction as :meth:`periodic_analytic_solution`
        (:func:`_disk_fourier_transform` for the disk's shape, then
        `reference_green.field` for the exact periodic
        Khachaturyan-Shatalov solve), just with `eigenstrain_tensor`
        supplied directly in place of :func:`_equivalent_eigenstrain`'s
        remote-load-derived one, and no `uniform_field` (zero remote
        strain, since there is no remote load here).

        Returns
        -------
        strain, stress : ndarray
            Shape ``(3, 3) + grid.shape``.
        """
        eigenstrain_tensor = xp.asarray(eigenstrain_tensor)
        solver = self.solver()
        green = solver.reference_green
        operator = solver.operator

        sigma_star = _isotropic_stress_apply(
            eigenstrain_tensor, self.matrix_lame_lambda, self.matrix_lame_mu
        )
        shape_hat = _disk_fourier_transform(
            self.grid, self.hole_radius, center=(self.center[0], self.center[1], 0.0)
        )
        eps0_hat = eigenstrain_tensor[:, :, None, None, None] * shape_hat[None, None, ...]
        sigma_star_hat = sigma_star[:, :, None, None, None] * shape_hat[None, None, ...]
        source_hat = -operator.div(sigma_star_hat)

        eps_hat = green.field(source_hat)

        difference = eps_hat - eps0_hat
        trace = xp.einsum("ii...->...", difference)
        delta = xp.eye(3, dtype=difference.dtype).reshape((3, 3, 1, 1, 1))
        sigma_hat = self.matrix_lame_lambda * trace[None, None, ...] * delta + (
            2.0 * self.matrix_lame_mu * difference
        )
        strain = self.grid.ifft(eps_hat)
        stress = self.grid.ifft(sigma_hat)
        return strain, stress

    def periodic_body_force_solution(self, force_vector):
        r"""Exact analytic periodic solution for a uniform, genuinely
        *applied* body force `force_vector` within the hole region, zero
        outside -- unlike :meth:`periodic_prescribed_eigenstrain_solution`,
        not a kinematic misfit entering through Hooke's law, but a real
        force density entering equilibrium directly as
        :math:`\nabla\cdot\sigma(x)+f(x)=0` (see
        :meth:`crystallite.elastic_deformation.ElasticDeformation.solve`'s
        `body_force` parameter for the numeric side of this same
        equation). Physically: a circular region whose *density* (not
        stiffness) differs from the matrix, so gravity (or any other body
        force) pulls on it by a different amount than it does the matrix
        it displaces -- the excess/deficit is `force_vector`.

        Needs no equivalent-inclusion construction and no divergence of
        an eigenstress: :math:`f(x)` (via :func:`_disk_fourier_transform`
        for its disk shape) is already exactly the source
        :meth:`~crystallite.spectral.long_range.GreenOperator.field`
        expects, one differentiation order below
        :meth:`periodic_prescribed_eigenstrain_solution`'s own
        ``-operator.div(sigma_star_hat)`` construction.

        Exact for *this* problem specifically because there is no
        stiffness contrast at all (only `force_vector` distinguishes the
        disk from the matrix) -- `reference_green` already *is* the exact
        Green's function for the true (uniform) medium, not merely a
        preconditioner, so no equivalent-inclusion/Eshelby-tensor
        calibration step is needed the way a genuine stiffness contrast
        would require.

        A uniform `force_vector` (nonzero net force over the disk's area)
        has no admissible periodic solution on its own -- same reasoning
        as :meth:`ElasticDeformation.solve`'s own `body_force` parameter:
        a periodic operator cannot balance a net force. ``green.field``
        drops it silently (zero response at the k=0 Fourier mode), giving
        the correct periodic solution for a *localized* force perturbation
        with everything else undisturbed, which is what this method (and
        the disk's own finite area) already assumes -- not an nonphysical
        approximation specific to this method.

        Returns
        -------
        strain, stress : ndarray
            Shape ``(3, 3) + grid.shape``.
        """
        force_vector = xp.asarray(force_vector)
        solver = self.solver()
        green = solver.reference_green

        shape_hat = _disk_fourier_transform(
            self.grid, self.hole_radius, center=(self.center[0], self.center[1], 0.0)
        )
        force_hat = force_vector[:, None, None, None] * shape_hat[None, ...]
        eps_hat = green.field(force_hat)

        trace = xp.einsum("ii...->...", eps_hat)
        delta = xp.eye(3, dtype=eps_hat.dtype).reshape((3, 3, 1, 1, 1))
        sigma_hat = self.matrix_lame_lambda * trace[None, None, ...] * delta + (
            2.0 * self.matrix_lame_mu * eps_hat
        )
        strain = self.grid.ifft(eps_hat)
        stress = self.grid.ifft(sigma_hat)
        return strain, stress

    def _surface_force_hat(self, traction):
        shape_hat = _disk_fourier_transform(
            self.grid, self.hole_radius, center=(self.center[0], self.center[1], 0.0)
        )
        return xp.stack([1j * self.grid.k[i] * shape_hat for i in range(3)]) * traction

    def surface_force_field(self, traction):
        r"""Real-space body force :math:`f=q\nabla\chi` for
        :meth:`periodic_surface_force_solution`, with :math:`\chi` the
        disk indicator and `traction` :math:`q`: a radial line force of
        magnitude `q` on the disk boundary, directed *inward* for
        ``q > 0`` (:math:`\nabla\chi` points into the disk). Built from
        the same band-limited Fourier transform the analytic solution
        uses, so numeric and analytic solves share exactly one source.
        Shape ``(3,) + grid.shape``."""
        return self.grid.ifft(self._surface_force_hat(traction))

    def periodic_surface_force_solution(self, traction):
        r"""Exact periodic solution for the radial boundary line force
        :meth:`surface_force_field` in a homogeneous material.

        In the isolated (plane-strain) limit the disk interior is
        uniformly, hydrostatically stressed with
        :math:`\sigma=-p\,I`, :math:`p=q(\lambda+\mu)/(\lambda+2\mu)`,
        and the exterior is the Lame field
        :math:`\sigma_{rr}=-\sigma_{\theta\theta}=A(R/r)^2` with
        :math:`A=p\mu/(\lambda+\mu)` -- *not* the pressurized-cavity
        exterior (that needs an eigenstrain, not a force).

        Returns
        -------
        strain, stress : ndarray
            Shape ``(3, 3) + grid.shape``.
        """
        green = self.solver().reference_green
        eps_hat = green.field(self._surface_force_hat(traction))
        trace = xp.einsum("ii...->...", eps_hat)
        delta = xp.eye(3, dtype=eps_hat.dtype).reshape((3, 3, 1, 1, 1))
        sigma_hat = self.matrix_lame_lambda * trace[None, None, ...] * delta + (
            2.0 * self.matrix_lame_mu * eps_hat
        )
        return self.grid.ifft(eps_hat), self.grid.ifft(sigma_hat)

    def periodic_pressurized_stress(self, magnitude):
        r"""Periodic-array stress field for the hole loaded by uniform
        internal pressure `magnitude` (zero remote stress), built from
        :meth:`periodic_analytic_solution`'s ``"biaxial"`` field rather
        than re-deriving a new eigenstrain.

        A spatially uniform stress tensor trivially satisfies equilibrium
        :math:`\nabla\cdot\sigma=0` everywhere regardless of the
        heterogeneous stiffness field (its divergence is exactly zero at
        every Fourier mode, including on this project's discrete
        operators, since a uniform field has no k != 0 content and the k=0
        mode of a divergence is always zero) -- so it may be freely
        superposed onto *any* equilibrium solution on this grid, numerical
        or analytic alike. Subtracting the uniform stress
        :math:`-\text{magnitude}\cdot I` from the "traction-free hole
        under remote biaxial tension `magnitude`" state turns it into the
        "hole boundary carrying radial traction ``-magnitude``, zero
        remote stress" state, i.e. exactly a hole pressurized from inside
        with pressure `magnitude`: at the hole boundary the biaxial
        solution is traction-free (:math:`\sigma_{rr}=0`) so the result is
        :math:`\sigma_{rr}=-\text{magnitude}`, while far away the biaxial
        remote stress :math:`\text{magnitude}\cdot I` is exactly cancelled.

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape``.
        """
        _, stress = self.periodic_analytic_solution("biaxial", magnitude)
        delta = xp.eye(3, dtype=stress.dtype).reshape((3, 3, 1, 1, 1))
        return stress - magnitude * delta

    def _void_correction_cartesian(self, x, y, load, magnitude):
        r"""Local (image-frame) correction to the uniform remote
        `magnitude`/`load` background for a single copy of the
        traction-free void centered at the origin of ``(x, y)`` -- the
        uniform-load counterpart of
        :func:`_gradient_void_hole_correction_cartesian`, built from
        :meth:`analytic_stress` (the exact isolated Kirsch closed form)
        rather than that function's "moment" one. Used by
        :meth:`periodic_void_stress` (via :meth:`_periodic_image_sum`) to
        image-sum the isolated solution as an approximate, Gibbs-ringing-
        free alternative to :meth:`periodic_analytic_solution`'s Eshelby-
        eigenstrain FFT construction -- see that method's docstring for
        why it is only approximate, not exact, unlike the "moment" case.
        See :meth:`_inhomogeneity_correction_cartesian` for the
        finite-`contrast` counterpart.

        Zero net stress inside the void (``-background``, so
        ``background + correction == 0`` there): unlike the "moment"
        background (which vanishes at its own origin by construction),
        the uniform background here is nonzero even at the void center,
        so the correction must actively cancel it, not just vanish.
        """
        background = self.remote_stress(load, magnitude)
        r = xp.sqrt(x**2 + y**2)
        outside = r >= self.hole_radius * (1.0 - 1.0e-9)
        r_safe = xp.where(outside, r, self.hole_radius)
        theta = xp.arctan2(y, x)
        srr, stt, srt = self.analytic_stress(r_safe, theta, load, magnitude)
        sigma_xx, sigma_yy, sigma_xy = polar_to_cartesian_stress(srr, stt, srt, theta)
        sigma_xx = sigma_xx - background[0, 0]
        sigma_yy = sigma_yy - background[1, 1]
        sigma_xy = sigma_xy - background[0, 1]
        return (
            xp.where(outside, sigma_xx, -background[0, 0]),
            xp.where(outside, sigma_yy, -background[1, 1]),
            xp.where(outside, sigma_xy, -background[0, 1]),
        )

    def _inhomogeneity_correction_cartesian(self, x, y, load, magnitude, eps_star=None):
        r"""Contrast-aware analog of :meth:`_void_correction_cartesian`:
        local correction for a single isolated copy of this case's
        *actual*, finite-`contrast` inhomogeneity, not the void limit --
        exterior from :meth:`inhomogeneity_analytic_stress`, interior from
        the uniform :func:`_inhomogeneity_interior_cartesian` tensor
        (which is generally nonzero, unlike the void case's exactly-zero
        interior). Used by :meth:`periodic_inhomogeneity_stress` (via
        :meth:`_periodic_image_sum`) -- this project's original
        (pre-crystallite) ``eshelby.cpp`` reference implementation's own
        recipe (its ``e()``/``f()``/``g()`` functions image-summed over a
        lattice), generalized here from that reference's void-only worked
        example to this case's actual `contrast`.

        With `eps_star` given, the isolated equivalent eigenstrain is
        replaced by that one (e.g. :meth:`periodic_equivalent_eigenstrain`)
        and the field is the equivalent-inclusion H/T-tensor field at
        ``a == b`` instead of the polar closed form.
        """
        nu = self.matrix_lame_lambda / (2.0 * (self.matrix_lame_lambda + self.matrix_lame_mu))
        if eps_star is not None:
            return _inplane_eigenstrain_correction(
                x, y, eps_star, self.hole_radius, self.hole_radius, self.matrix_lame_mu, nu
            )
        background = self.remote_stress(load, magnitude)
        r = xp.sqrt(x**2 + y**2)
        outside = r >= self.hole_radius * (1.0 - 1.0e-9)
        r_safe = xp.where(outside, r, self.hole_radius)
        theta = xp.arctan2(y, x)
        srr, stt, srt = self.inhomogeneity_analytic_stress(r_safe, theta, load, magnitude)
        sigma_xx, sigma_yy, sigma_xy = polar_to_cartesian_stress(srr, stt, srt, theta)
        sigma_xx = sigma_xx - background[0, 0]
        sigma_yy = sigma_yy - background[1, 1]
        sigma_xy = sigma_xy - background[0, 1]

        interior = _inhomogeneity_interior_cartesian(magnitude, self.contrast, nu, load)
        interior_xx = interior[0, 0] - background[0, 0]
        interior_yy = interior[1, 1] - background[1, 1]
        interior_xy = interior[0, 1] - background[0, 1]
        return (
            xp.where(outside, sigma_xx, interior_xx),
            xp.where(outside, sigma_yy, interior_yy),
            xp.where(outside, sigma_xy, interior_xy),
        )

    def _periodic_image_sum(
        self, correction, load, magnitude, n_images, recenter_target=None, interior_target=None
    ):
        r"""Shared image-summation machinery for
        :meth:`periodic_void_stress` and
        :meth:`periodic_inhomogeneity_stress`: sum `correction`
        (:meth:`_void_correction_cartesian` or
        :meth:`_inhomogeneity_correction_cartesian`) over periodic images,
        then (outside every hole) recenter the result to `recenter_target`
        (default: `magnitude` itself, via `remote_stress`), and (inside
        every hole) replace it with the uncontaminated home-image-only
        value -- ``eshelby.cpp``'s own exact two-pass recipe, not a
        single pass doing both at once (see below for why that distinction
        matters and was originally missed here).

        The raw image sum's own domain mean is not exactly the target
        (each hole's presence measurably perturbs its neighbors'
        effective loading -- a real periodic self-interaction, not a
        bug), so, outside every hole, this recenters it by construction:
        ``field -= mean(field) - target``. This is not a self-consistent
        local-field solve -- it is a direct substitution, exactly the
        recipe this project's own original (pre-crystallite) reference
        implementation used (``eshelby.cpp``'s ``sigma[i][j] -=
        s0_-s0xt`` after averaging over the whole domain, `s0xt` its
        remote-stress target).

        Inside every hole, `correction` is instead evaluated *only* for
        the home image (``n1=n2=0``) and used as-is, discarding whatever
        the raw sum over every image produced there. This matters because
        `correction` itself is piecewise: for a non-home image, a point
        inside the *true* (home) hole is generally *outside* that other
        image's own, far-away hole, so it hits `correction`'s exterior
        branch and contributes a small but nonzero perturbation there --
        exactly the same way a home image's own field decays into
        neighboring cells. Left in place, this leaks neighbor-image
        perturbations into the home hole's interior, which should stay
        exactly uniform by the isolated-inhomogeneity closed form
        (Eshelby's uniformity theorem) `correction`'s own interior branch
        already returns. An earlier version of this method summed every
        image's `correction` unconditionally and left interior points
        untouched by the outside-only recentering, silently keeping that
        leakage in the final result -- confirmed directly to be a real
        (if numerically small at this module's own dilute hole/domain
        ratio) effect, not a hypothetical one: at ``n_images=1`` the
        interior stress came out non-uniform and even *sign-flipped*
        relative to the true closed-form value, whereas ``n_images=0``
        (home image only, nothing to leak) matched it exactly. This is
        exactly why ``eshelby.cpp`` itself unconditionally overwrites
        every interior pixel with a single fixed value (its own
        ``sigma0_``/etc. from ``e()``) in a *second* pass, entirely
        independent of the neighbor-image sum its *first* pass built --
        it never blends the two the way summing `correction` over every
        image and only recentering outside would.

        What `recenter_target` *should* be depends on the boundary
        condition the comparison is actually against. ``eshelby.cpp``'s
        own ``s0xt`` is exactly `magnitude` (the default here) because it
        was validated against a spectral solver driven by a *prescribed
        remote stress*: for that boundary condition, the periodic cell's
        domain-mean stress trivially *equals* the applied stress by
        definition, so recentering to `magnitude` is not an approximation
        at all there. This project's own
        :class:`crystallite.elastic_deformation.ElasticDeformation` is
        driven by a prescribed macro *strain* instead (:meth:`macro_strain`,
        via the *matrix's own* compliance) -- a genuinely different
        boundary condition for a cell that actually contains a hole: the
        softer composite carries measurably *less* mean stress than a
        pure-matrix cell would for that same imposed strain, so `magnitude`
        is the wrong recentering target there, not just an approximate
        one. Confirmed directly, two independent ways, for this module's
        own hole geometry (``HOLE_RADIUS=0.1``, ``CONTRAST=1e-3``,
        tension): the true numeric CG solve's own domain-mean stress comes
        out to ``0.9101 x magnitude``, and :meth:`periodic_analytic_solution`
        (the exact-periodicity FFT/Eshelby-tensor construction, itself
        built from the same strain-controlled `macro_strain`) independently
        gives ``0.9114 x magnitude`` -- agreeing with the numeric solve to
        0.14%, not with the naive ``1.0 x magnitude`` this method defaults
        to. :meth:`periodic_inhomogeneity_stress` passes
        :meth:`periodic_analytic_solution`'s own domain mean as
        `recenter_target` for exactly this reason;
        :meth:`periodic_void_stress` still defaults to `magnitude` itself
        (matching ``eshelby.cpp`` literally, and its own long-standing
        documented expectation of a residual gap against this solver's
        strain-controlled numeric output).

        Converges quickly in `n_images` (a handful of images already
        stabilizes the domain mean to 4+ significant figures at this
        module's own example geometry -- checked directly against
        ``eshelby.cpp``'s own ``n=16``, not just larger values of
        `n_images` within this project: ``n_images=1`` already agrees with
        ``n_images=16`` to within 0.002, far smaller than the boundary-
        condition gap above). `load="moment"`
        (:meth:`periodic_gradient_stress`) needs no such recentering:
        that background's leading-order (uniform) equivalent-eigenstrain
        response is exactly zero by symmetry, so its own domain mean is
        already exact regardless of boundary condition.

        Neither `correction` option is a self-consistent periodic solve
        (see :meth:`periodic_void_stress`/:meth:`periodic_inhomogeneity_stress`
        for what each one does and does not fix relative to
        :meth:`periodic_analytic_solution`): image-summing an isolated
        closed form, even the correct finite-contrast one, cannot
        recalibrate the inhomogeneity's own equivalent response for how
        densely packed the periodic array actually is -- only
        :meth:`periodic_analytic_solution`, calibrated against the
        numerically probed *periodic* Eshelby tensor
        (:meth:`periodic_eshelby_tensor`), does that. Passing that same
        method's domain mean as `recenter_target` (as
        :meth:`periodic_inhomogeneity_stress` does) fixes the *overall
        level* to match it exactly, but not the near-hole *shape* --
        image-summing the isolated field still leaves a genuine, smaller
        residual gap there.

        Parameters
        ----------
        correction : callable
            ``correction(dx, dy, load, magnitude) -> (sigma_xx, sigma_yy, sigma_xy)``
        load : {"tension", "compression", "shear", "biaxial"}
        magnitude : float
        n_images : int
            Sum periodic images in ``[-n_images, n_images]`` along each
            in-plane axis.
        recenter_target : ndarray of shape (3, 3), optional
            Domain-mean stress to recenter the exterior field to. Defaults
            to ``remote_stress(load, magnitude)`` (`magnitude` itself).
        interior_target : ndarray of shape (3, 3), optional
            Uniform stress to assign inside the hole. Defaults to the
            home-image value (``background`` plus the isolated
            correction); :meth:`periodic_inhomogeneity_stress` passes the
            exact periodic interior (:meth:`periodic_interior_stress`).

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape`` -- only the in-plane
            ``sigma_xx``, ``sigma_yy``, ``sigma_xy`` components are
            populated, as with :meth:`periodic_gradient_stress`.
        """
        x, y = self.grid.x[0], self.grid.x[1]
        length_x, length_y = self.grid.lengths[0], self.grid.lengths[1]
        background = self.remote_stress(load, magnitude)
        target = background if recenter_target is None else recenter_target

        sigma_xx = background[0, 0] * xp.ones(self.grid.shape)
        sigma_yy = background[1, 1] * xp.ones(self.grid.shape)
        sigma_xy = background[0, 1] * xp.ones(self.grid.shape)

        home_dx = x - self.center[0]
        home_dy = y - self.center[1]
        home_xx, home_yy, home_xy = correction(home_dx, home_dy, load, magnitude)

        for n1 in range(-n_images, n_images + 1):
            for n2 in range(-n_images, n_images + 1):
                dx = x - (self.center[0] + n1 * length_x)
                dy = y - (self.center[1] + n2 * length_y)
                sxx, syy, sxy = correction(dx, dy, load, magnitude)
                sigma_xx = sigma_xx + sxx
                sigma_yy = sigma_yy + syy
                sigma_xy = sigma_xy + sxy

        outside = self.radius >= self.hole_radius
        # The domain mean feeding the recentering below uses this raw sum
        # as-is (neighbor images' exterior field leaking into the home
        # hole's own interior included) -- matching eshelby.cpp's own
        # domain-mean accumulation, which likewise sums the *raw*,
        # not-yet-corrected per-pixel array (see this method's docstring).
        # Only *after* that mean is taken do interior points get replaced
        # below with the uncontaminated home-image-only value -- exactly
        # eshelby.cpp's own two-pass structure (accumulate the mean over
        # the raw array first, overwrite the interior unconditionally
        # after), not simultaneous with the outside-only recentering.
        if interior_target is None:
            interior_xx = background[0, 0] + home_xx
            interior_yy = background[1, 1] + home_yy
            interior_xy = background[0, 1] + home_xy
            recentered_xx = sigma_xx - (xp.mean(sigma_xx) - target[0, 0])
            recentered_yy = sigma_yy - (xp.mean(sigma_yy) - target[1, 1])
            recentered_xy = sigma_xy - (xp.mean(sigma_xy) - target[0, 1])
        else:
            # exact interior: shift the exterior so the final field's own
            # domain mean is exactly the target
            interior_xx = interior_target[0, 0]
            interior_yy = interior_target[1, 1]
            interior_xy = interior_target[0, 1]
            recentered_xx = sigma_xx - _exterior_shift(sigma_xx, outside, interior_xx, target[0, 0])
            recentered_yy = sigma_yy - _exterior_shift(sigma_yy, outside, interior_yy, target[1, 1])
            recentered_xy = sigma_xy - _exterior_shift(sigma_xy, outside, interior_xy, target[0, 1])
        sigma_xx = xp.where(outside, recentered_xx, interior_xx)
        sigma_yy = xp.where(outside, recentered_yy, interior_yy)
        sigma_xy = xp.where(outside, recentered_xy, interior_xy)

        stress = xp.zeros((3, 3) + self.grid.shape, dtype=sigma_xx.dtype)
        stress[0, 0] = sigma_xx
        stress[1, 1] = sigma_yy
        stress[0, 1] = stress[1, 0] = sigma_xy
        return stress

    def periodic_void_stress(self, load, magnitude, n_images=1):
        r"""Periodic-array stress field for a traction-free void under
        uniform remote `load`/`magnitude`, built by summing the exact
        isolated closed form (:meth:`analytic_stress`) over periodic
        images -- see :meth:`_periodic_image_sum` for the mechanics
        (image summation and recentering) shared with
        :meth:`periodic_inhomogeneity_stress`.

        Ignores `contrast` entirely: :meth:`analytic_stress` is the exact
        ``contrast=0`` void solution regardless of what `contrast` this
        case's diffuse numerical boundary actually uses. Valid as a
        reference only when `contrast` is small enough that the numerical
        hole is itself a good void approximation -- when `contrast` is
        not negligible, prefer :meth:`periodic_inhomogeneity_stress`
        (exact isolated field, still only dilute-limit periodicity) or
        :meth:`periodic_analytic_solution` (exact periodicity too, at the
        cost of Gibbs ringing -- see ``cylindrical_inclusion.py``).

        Parameters
        ----------
        load : {"tension", "compression", "shear", "biaxial"}
        magnitude : float
        n_images : int, default=1

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape``.
        """
        return self._periodic_image_sum(self._void_correction_cartesian, load, magnitude, n_images)

    def periodic_inhomogeneity_stress(self, load, magnitude, n_images=1):
        r"""Periodic-array stress field for this case's actual, finite-
        `contrast` circular inhomogeneity under uniform remote
        `load`/`magnitude`, built by summing the exact isolated closed
        form (:meth:`inhomogeneity_analytic_stress` outside,
        :func:`_inhomogeneity_interior_cartesian` inside) over periodic
        images -- see :meth:`_periodic_image_sum` for the shared
        mechanics.

        The contrast-aware generalization of :meth:`periodic_void_stress`,
        matching this project's original (pre-crystallite) ``eshelby.cpp``
        reference implementation's own recipe more closely than that
        method does: ``eshelby.cpp``'s ``e()``/``f()``/``g()`` functions
        already solve for and image-sum the *finite-contrast* Eshelby
        field (its worked example happened to set the inhomogeneity's
        modulus to zero, a literal void, but the underlying construction
        is general) -- this method is that same construction, using
        :func:`_inhomogeneity_polar_stress`/:func:`_inhomogeneity_interior_stress`
        in place of ``eshelby.cpp``'s own closed-form ``H``-tensor
        (equivalent content, this project's own independent derivation).

        Fixes, relative to :meth:`periodic_void_stress`, *two* things: the
        void-vs-actual-`contrast` mismatch (this case's own diffuse
        numerical boundary is never a literal void), and -- unlike that
        method -- the recentering target itself: rather than forcing the
        far field to the nominal `magnitude` (correct only under a
        stress-controlled boundary condition ``eshelby.cpp`` was written
        against), this recenters to :meth:`periodic_analytic_solution`'s
        own domain-mean stress, which matches this solver's actual
        strain-controlled boundary condition (see
        :meth:`_periodic_image_sum`'s docstring for the direct numeric
        confirmation this closes most of the level mismatch, though a
        smaller shape-level gap remains -- image-summing the isolated
        field still does not recalibrate the inhomogeneity's own
        equivalent response for how densely packed the array actually is,
        only :meth:`periodic_analytic_solution` itself does that, at the
        cost of Gibbs ringing).

        Parameters
        ----------
        load : {"tension", "compression", "shear", "biaxial"}
        magnitude : float
        n_images : int, default=1

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape``.
        """
        recenter_target = self.periodic_mean_stress(load, magnitude)
        eps_star = self.periodic_equivalent_eigenstrain(load, magnitude)

        def correction(dx, dy, load_, magnitude_):
            return self._inhomogeneity_correction_cartesian(
                dx, dy, load_, magnitude_, eps_star=eps_star
            )

        return self._periodic_image_sum(
            correction, load, magnitude, n_images,
            recenter_target=recenter_target,
            interior_target=self.periodic_interior_stress(load, magnitude),
        )

    def periodic_mean_stress(self, load, magnitude):
        """Exact domain-mean ``(3, 3)`` stress of the periodic array under
        strain control, by the lattice-sum Eshelby tensor -- see
        :func:`_periodic_mean_stress`. FFT- and grid-free."""
        return xp.asarray(
            _periodic_mean_stress(
                load, magnitude, self.contrast, self.matrix_lame_lambda, self.matrix_lame_mu,
                self.hole_radius, self.hole_radius,
                float(self.grid.lengths[0]), float(self.grid.lengths[1]),
            )
        )

    def periodic_antiplane_mean_stress(self, magnitude):
        """Antiplane counterpart of :meth:`periodic_mean_stress`."""
        return xp.asarray(
            _periodic_antiplane_mean_stress(
                magnitude, self.contrast, self.matrix_lame_mu,
                self.hole_radius, self.hole_radius,
                float(self.grid.lengths[0]), float(self.grid.lengths[1]),
            )
        )

    def periodic_interior_stress(self, load, magnitude):
        """Exact uniform interior ``(3, 3)`` stress of the periodic array
        under strain control -- see :func:`_periodic_interior_stress`."""
        return xp.asarray(
            _periodic_interior_stress(
                load, magnitude, self.contrast, self.matrix_lame_lambda, self.matrix_lame_mu,
                self.hole_radius, self.hole_radius,
                float(self.grid.lengths[0]), float(self.grid.lengths[1]),
            )
        )

    def periodic_antiplane_interior_stress(self, magnitude):
        """Antiplane counterpart of :meth:`periodic_interior_stress`."""
        return xp.asarray(
            _periodic_antiplane_interior_stress(
                magnitude, self.contrast, self.matrix_lame_mu,
                self.hole_radius, self.hole_radius,
                float(self.grid.lengths[0]), float(self.grid.lengths[1]),
            )
        )

    def periodic_equivalent_eigenstrain(self, load, magnitude):
        """Periodic-array equivalent eigenstrain ``(3, 3)`` of the
        inhomogeneity, calibrated with the exact lattice-sum Eshelby tensor
        (:func:`_lattice_eshelby_tensor`) rather than the isolated closed
        form: the strength each image must carry so that the image sum
        reproduces the periodic solution."""
        s = _lattice_eshelby_tensor(
            self.hole_radius, self.hole_radius,
            float(self.grid.lengths[0]), float(self.grid.lengths[1]),
            self.matrix_lame_lambda, self.matrix_lame_mu,
        )
        return xp.asarray(
            _periodic_equivalent_eigenstrain(
                magnitude, self.contrast, self.matrix_lame_lambda, self.matrix_lame_mu, load, s
            )
        )

    def periodic_antiplane_equivalent_eigenstrain(self, magnitude):
        """Antiplane counterpart of :meth:`periodic_equivalent_eigenstrain`,
        calibrated with the lattice-sum ``s1313``."""
        s = _lattice_eshelby_tensor(
            self.hole_radius, self.hole_radius,
            float(self.grid.lengths[0]), float(self.grid.lengths[1]),
            self.matrix_lame_lambda, self.matrix_lame_mu,
        )
        return xp.asarray(
            _antiplane_periodic_equivalent_eigenstrain(
                magnitude, self.contrast, self.matrix_lame_mu, s["s1313"]
            )
        )

    def periodic_gradient_inhomogeneity_stress(self, magnitude, n_images=2):
        """Periodic-array stress field for the circular inhomogeneity
        (any `contrast`, void to rigid) under the "moment" remote stress
        gradient, by image-summing the exact isolated solution -- the
        finite-contrast counterpart of :meth:`periodic_gradient_stress`
        (which is void-only), and the circular case (``a == b``) of
        :meth:`EllipticalHoleInPlateCase.periodic_gradient_inhomogeneity_stress`,
        whose docstring has the validation. Only the in-plane
        ``sigma_xx``, ``sigma_yy``, ``sigma_xy`` are populated."""
        return _periodic_gradient_inhomogeneity_stress(
            self.grid, self.center, self.hole_radius, self.hole_radius, self.contrast,
            self.matrix_lame_lambda, self.matrix_lame_mu, magnitude, n_images,
        )

    def periodic_void_pressurized_stress(self, magnitude, n_images=1):
        r"""Periodic-array stress field for a void loaded by uniform
        internal pressure `magnitude` (zero remote stress) -- the
        image-summed counterpart of :meth:`periodic_pressurized_stress`,
        built from :meth:`periodic_void_stress` instead of
        :meth:`periodic_analytic_solution` (see that method for the
        "biaxial minus uniform background" construction, identical here).

        Inherits :meth:`periodic_void_stress`'s dilute-limit bias (see its
        docstring): a smooth, ringing-free approximation of the
        concentration/decay shape, not a tight quantitative match to
        :meth:`periodic_pressurized_stress`/the numeric solver.
        """
        stress = self.periodic_void_stress("biaxial", magnitude, n_images=n_images)
        delta = xp.eye(3, dtype=stress.dtype).reshape((3, 3, 1, 1, 1))
        return stress - magnitude * delta

    def periodic_inhomogeneity_pressurized_stress(self, magnitude, n_images=1):
        r"""Periodic-array stress field for this case's actual, finite-
        `contrast` circular inhomogeneity loaded by uniform internal
        pressure `magnitude` (zero remote stress) -- the contrast-aware
        generalization of :meth:`periodic_void_pressurized_stress`, built
        from :meth:`periodic_inhomogeneity_stress` instead of
        :meth:`periodic_void_stress` (same "biaxial minus uniform
        background" construction as that method -- see
        :meth:`periodic_pressurized_stress`'s docstring for why the
        subtraction itself is valid regardless of contrast, superposing a
        spatially uniform, trivially divergence-free stress onto any
        equilibrium solution).

        Fixes the same void-vs-actual-`contrast` mismatch
        :meth:`periodic_inhomogeneity_stress` fixes for the uniform-load
        cases, left over here until now: ``hole_in_plate.py``'s own
        pressure plot was still built on the void-only
        :meth:`periodic_void_pressurized_stress` even after its
        tension/compression/biaxial/shear plots moved to the
        finite-`contrast` :meth:`periodic_inhomogeneity_stress`.

        Inherits :meth:`periodic_inhomogeneity_stress`'s own two
        recentering caveats (see that method's and
        :meth:`_periodic_image_sum`'s docstrings): recentered to
        :meth:`periodic_analytic_solution`'s domain mean, not the naive
        nominal `magnitude`, but still only a dilute-limit approximation
        of the periodic array's own self-interaction, not the exact
        periodicity :meth:`periodic_pressurized_stress` has.
        """
        stress = self.periodic_inhomogeneity_stress("biaxial", magnitude, n_images=n_images)
        delta = xp.eye(3, dtype=stress.dtype).reshape((3, 3, 1, 1, 1))
        return stress - magnitude * delta

    def remote_stress_gradient(self, load, magnitude):
        """Return the ``(3, 3, 3)`` remote stress *gradient* tensor
        (component, component, direction) for `load`.

        Parameters
        ----------
        load : {"moment"}
            ``d(sigma_11)/dx2 = magnitude`` (a pure-bending/"moment"
            background, :math:`\\sigma_{11}\\to\\mathrm{magnitude}\\cdot
            x_2`, :math:`\\sigma_{22}=\\sigma_{12}=0` far away).
        magnitude : float
        """
        if load != "moment":
            raise ValueError('load must be "moment"')
        gradient = xp.zeros((3, 3, 3))
        gradient[0, 0, 1] = magnitude
        return gradient

    def macro_strain_gradient(self, load, magnitude, solver=None):
        """Return the macroscopic strain gradient reproducing
        `remote_stress_gradient` far from the hole, via the matrix's own
        compliance -- the gradient counterpart of :meth:`macro_strain`.

        Passed to :meth:`crystallite.elastic_deformation.ElasticDeformation.solve`
        as `macro_strain_gradient`, together with `center` as
        `gradient_origin` so the background vanishes exactly at the hole
        center (where the "moment" load's bending field has its zero) --
        see that method for why this particular construction, unlike an
        arbitrary affine field, is safe to combine with the periodic FFT
        solver.
        """
        solver = solver if solver is not None else self.solver()
        sigma_gradient = self.remote_stress_gradient(load, magnitude)
        return solver.reference_green.apply_compliance(sigma_gradient)

    def periodic_gradient_stress(self, magnitude, n_images=3):
        r"""Periodic-array stress field for the hole under the "moment"
        remote stress gradient `magnitude` (see
        :meth:`remote_stress_gradient`), built by summing the exact
        traction-free-void closed form
        (:func:`crystallite.verification.elastic_deformation._gradient_void_hole_correction_cartesian`)
        over periodic images of the hole, rather than an
        equivalent-eigenstrain construction: unlike the uniform loads
        (:meth:`periodic_analytic_solution`), the disk sits exactly at the
        zero of this background field, so its *leading-order* (uniform)
        equivalent eigenstrain response is exactly zero -- capturing the
        hole's actual perturbation correctly would need a linear
        ("dipole") eigenstrain within the disk and the matching
        higher-order Eshelby tensor, which this project does not have
        implemented. Image summation sidesteps that: it reuses the
        already-validated *exact* isolated solution directly (traction-
        free at the hole boundary, in full pointwise equilibrium, and
        exactly consistent with this background far away -- all checked
        independently of this codebase, by finite differences), and each
        image's correction decays as :math:`(a/r)^3`, so with holes spaced
        ~10 radii apart on this grid a handful of images already converges
        far beyond the diffuse-boundary numerical solver's own accuracy.

        Parameters
        ----------
        magnitude : float
        n_images : int, default=3
            Sum periodic images in ``[-n_images, n_images]`` along each
            in-plane axis.

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape`` -- only the in-plane
            ``sigma_xx``, ``sigma_yy``, ``sigma_xy`` components are
            populated (as with the true-void closed forms used
            elsewhere in this module, ``sigma_33`` is left to the solver
            rather than fixed by this 2D construction).
        """
        x, y = self.grid.x[0], self.grid.x[1]
        length_x, length_y = self.grid.lengths[0], self.grid.lengths[1]

        sigma_xx = magnitude * (y - self.center[1]) * xp.ones(self.grid.shape)
        sigma_yy = xp.zeros(self.grid.shape)
        sigma_xy = xp.zeros(self.grid.shape)

        for n1 in range(-n_images, n_images + 1):
            for n2 in range(-n_images, n_images + 1):
                dx = x - (self.center[0] + n1 * length_x)
                dy = y - (self.center[1] + n2 * length_y)
                sxx, syy, sxy = _gradient_void_hole_correction_cartesian(
                    dx, dy, self.hole_radius, magnitude
                )
                sigma_xx = sigma_xx + sxx
                sigma_yy = sigma_yy + syy
                sigma_xy = sigma_xy + sxy

        stress = xp.zeros((3, 3) + self.grid.shape, dtype=sigma_xx.dtype)
        stress[0, 0] = sigma_xx
        stress[1, 1] = sigma_yy
        stress[0, 1] = stress[1, 0] = sigma_xy
        return stress

    def _gradient_offset(self):
        """``x2 - center[1]``, shape ``grid.shape`` -- the "moment"
        background's own direction and origin convention
        (:meth:`remote_stress_gradient`), reused as the dipole
        eigenstrain's linear weight everywhere below."""
        return xp.broadcast_to(self.grid.x[1] - self.center[1], self.grid.shape)

    def eigenstrain_gradient_field(self, eigenstrain_gradient_tensor):
        r"""Real-space dipole eigenstrain field
        :math:`\varepsilon^*(x)=\text{eigenstrain\_gradient\_tensor}\cdot(x_2-\text{center}_1)`
        for `radius` :math:`<` `hole_radius`, zero outside -- the
        gradient-order counterpart of :meth:`eigenstrain_field`, and this
        class's own "moment" background direction/origin convention
        (:meth:`remote_stress_gradient`)."""
        eigenstrain_gradient_tensor = xp.asarray(
            eigenstrain_gradient_tensor, dtype=self.grid.real_dtype
        )
        indicator = xp.where(self.radius < self.hole_radius, 1.0, 0.0).astype(
            self.grid.real_dtype
        )
        shape = indicator * self._gradient_offset().astype(self.grid.real_dtype)
        return eigenstrain_gradient_tensor[:, :, None, None, None] * shape[None, None, ...]

    def periodic_prescribed_gradient_eigenstrain_solution(self, eigenstrain_gradient_tensor):
        r"""Exact periodic solution for a *linear* (dipole) eigenstrain
        :math:`\varepsilon^*(x)=\text{eigenstrain\_gradient\_tensor}\cdot(x_2-\text{center}_1)`
        prescribed within the hole, zero remote/mean field -- the
        gradient-order counterpart of
        :meth:`periodic_prescribed_eigenstrain_solution`.

        Unlike that method, the region's shape enters via a direct
        numerical FFT of the real-space field
        (:meth:`eigenstrain_gradient_field`) rather than a closed-form
        analytic transform: a closed form for the "linear-weighted disk"
        transform does exist (a k-space derivative of
        :func:`_disk_fourier_transform`), but this method is only ever
        called a handful of times per case (mainly by
        :meth:`periodic_eshelby_dipole_tensor`'s probing), so a numerical
        FFT costs nothing extra while avoiding the risk of a hand-derived
        Bessel-derivative transcription error -- the same "verify
        numerically rather than trust a fragile derivation" preference
        :func:`_periodic_eshelby_tensor` already applies one order lower.

        Returns
        -------
        strain, stress : ndarray
            Shape ``(3, 3) + grid.shape``.
        """
        eigenstrain_gradient_tensor = xp.asarray(eigenstrain_gradient_tensor)
        solver = self.solver()
        green = solver.reference_green
        operator = solver.operator

        eps0 = self.eigenstrain_gradient_field(eigenstrain_gradient_tensor)
        sigma_star_dipole = _isotropic_stress_apply(
            eigenstrain_gradient_tensor, self.matrix_lame_lambda, self.matrix_lame_mu
        )
        indicator = xp.where(self.radius < self.hole_radius, 1.0, 0.0).astype(
            self.grid.real_dtype
        )
        shape = indicator * self._gradient_offset().astype(self.grid.real_dtype)
        sigma_star = sigma_star_dipole[:, :, None, None, None] * shape[None, None, ...]

        eps0_hat = self.grid.fft(eps0)
        sigma_star_hat = self.grid.fft(sigma_star)
        source_hat = -operator.div(sigma_star_hat)

        eps_hat = green.field(source_hat)

        difference = eps_hat - eps0_hat
        trace = xp.einsum("ii...->...", difference)
        delta = xp.eye(3, dtype=difference.dtype).reshape((3, 3, 1, 1, 1))
        sigma_hat = self.matrix_lame_lambda * trace[None, None, ...] * delta + (
            2.0 * self.matrix_lame_mu * difference
        )
        strain = self.grid.ifft(eps_hat)
        stress = self.grid.ifft(sigma_hat)
        return strain, stress

    def periodic_eshelby_dipole_tensor(self):
        """Numerically probed periodic *dipole* (gradient-order) Eshelby
        tensor for this case's actual `hole_radius`/domain ratio -- see
        :func:`_periodic_eshelby_dipole_tensor`."""
        return _periodic_eshelby_dipole_tensor(
            self.periodic_prescribed_gradient_eigenstrain_solution,
            self.radius < self.hole_radius,
            self._gradient_offset(),
        )

    def periodic_gradient_eigenstrain_stress(self, magnitude):
        r"""Periodic-array stress field for a *finite-contrast*
        inhomogeneity (not just a true void) under the "moment" remote
        stress gradient `magnitude` (see :meth:`remote_stress_gradient`)
        -- the Eshelby-dipole counterpart of :meth:`periodic_gradient_stress`,
        which only covers the true-void limit (see that method's own
        docstring for why: the disk sits at this background's zero, so
        capturing a real inhomogeneity's perturbation needs a linear
        ("dipole") eigenstrain and its matching gradient-order Eshelby
        tensor, calibrated here by :meth:`periodic_eshelby_dipole_tensor`).

        Built the same way as :meth:`periodic_analytic_solution` one
        order up: the equivalent-inclusion linear system
        (:func:`_periodic_equivalent_eigenstrain`) is reused completely
        unchanged -- calling it with ``load="tension"`` against the
        *dipole* tensor and this background's own
        :meth:`macro_strain_gradient` is exactly the right equation,
        since that function's algebra only encodes "match a
        remote-field/Eshelby-tensor pair," not which polynomial order
        they represent. Only the background itself needs adding back by
        hand afterward (as a spatially *linear*, not uniform, field) since
        :meth:`ElasticDeformation.reference_green`'s ``field`` only
        accepts a *uniform* background (its ``uniform_field`` argument) --
        a linear background has no representation as a single Fourier
        mode, so it is added directly in real space, matching how
        :meth:`periodic_gradient_stress` also adds its own background by
        hand rather than through `reference_green`.

        Verified directly against :meth:`periodic_gradient_stress` (whose
        image-summed true-void construction shares no code with this
        method) at a small near-void `contrast`: the two agree closely,
        confirming this more general construction reduces correctly to
        the already-validated void limit.

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape``.
        """
        solver = self.solver()
        green = solver.reference_green
        operator = solver.operator

        dipole_tensor = self.periodic_eshelby_dipole_tensor()
        eps_star = _periodic_equivalent_eigenstrain(
            magnitude, self.contrast, self.matrix_lame_lambda, self.matrix_lame_mu, "tension",
            dipole_tensor,
        )
        sigma_star_dipole = _isotropic_stress_apply(
            eps_star, self.matrix_lame_lambda, self.matrix_lame_mu
        )
        indicator = xp.where(self.radius < self.hole_radius, 1.0, 0.0).astype(
            self.grid.real_dtype
        )
        offset = self._gradient_offset()
        shape = indicator * offset.astype(self.grid.real_dtype)

        eps0 = eps_star[:, :, None, None, None] * shape[None, None, ...]
        sigma_star = sigma_star_dipole[:, :, None, None, None] * shape[None, None, ...]

        eps0_hat = self.grid.fft(eps0)
        sigma_star_hat = self.grid.fft(sigma_star)
        source_hat = -operator.div(sigma_star_hat)
        eps_correction_hat = green.field(source_hat)
        eps_correction = self.grid.ifft(eps_correction_hat)

        # macro_strain_gradient returns the full (3, 3, 3) (component,
        # component, direction) tensor -- direction 1 (x2) is the only
        # one populated for "moment" (see remote_stress_gradient), and
        # the only one this method's own x2-only `offset` weighting needs.
        eps_background_tensor = self.macro_strain_gradient(
            "moment", magnitude, solver=solver
        )[:, :, 1]
        eps_background = xp.asarray(eps_background_tensor)[:, :, None, None, None] * offset[
            None, None, ...
        ]

        difference = (eps_correction + eps_background) - eps0
        trace = xp.einsum("ii...->...", difference)
        delta = xp.eye(3, dtype=difference.dtype).reshape((3, 3, 1, 1, 1))
        stress = self.matrix_lame_lambda * trace[None, None, ...] * delta + (
            2.0 * self.matrix_lame_mu * difference
        )
        return stress


@dataclass(frozen=True)
class EllipticalHoleInPlateCase:
    r"""An elliptical hole (soft inclusion) in a periodic 2D isotropic
    plate, with the analytic Kirsch/Inglis stress field
    (:func:`_ellipse_void_cartesian_stress`) for validation -- the
    elliptical-geometry counterpart of :class:`HoleInPlateCase`.

    Two periodic reference constructions are available, the same choice
    :class:`HoleInPlateCase` offers:

    - :meth:`periodic_analytic_stress` -- an Eshelby equivalent
      eigenstrain (:func:`_ellipse_equivalent_eigenstrain`, the general
      elliptical-cylinder S-tensor) over a matrix-material ellipse,
      solved exactly in Fourier space via :func:`_ellipse_fourier_transform`.
      Handles any finite `contrast`, not just a near-void one.
    - :meth:`periodic_void_stress` -- direct periodic image summation of
      the isolated Kirsch/Inglis closed form
      (:func:`_ellipse_void_cartesian_stress`), no FFT. Void-only
      (``contrast`` ignored), like :meth:`HoleInPlateCase.periodic_void_stress`.

    An earlier version of this docstring claimed the image-sum approach
    had "a large, strongly eccentricity-dependent error (up to ~46% at
    semi_axis_b/semi_axis_a=8), confirmed against genuine FEM ground-
    truth data" and used that to justify dropping it in favor of the
    Eshelby/FFT construction exclusively. That claim did not survive a
    second look -- no FEM tool, data file, or reference backing it exists
    anywhere in this repository, and the 46% figure itself came from
    comparing against a fixed-offset sample point that badly undersamples
    the true (curvature-scaled) tip concentration at high eccentricity,
    not from a real modeling failure. See
    :meth:`periodic_void_stress`'s own docstring for the corrected
    account, including its recentering step and provenance (this
    project's original, pre-crystallite ``eshelby.cpp`` reference
    implementation). Both constructions are kept; prefer
    :meth:`periodic_void_stress` for a smooth, ringing-free reference
    when `contrast` is near-void, and :meth:`periodic_analytic_stress`
    when `contrast` is not.

    Parameters
    ----------
    grid : Grid
        A 2D grid (``grid.shape[2] == 1``).
    matrix_lame_lambda, matrix_lame_mu : float
        Lame parameters of the matrix material.
    semi_axis_a, semi_axis_b : float
        Ellipse semi-axes along the grid's first and second coordinates.
    contrast : float, default=1.0e-3
        Ratio of the hole's Lame parameters to the matrix's.
    center : tuple of float, default=(0.5, 0.5)
        Hole center in the grid's first two coordinates.
    smoothing_width : float, default=0.0
        Diffuse-boundary width in the *dimensionless* elliptical radius
        :math:`\rho=\sqrt{(x/a)^2+(y/b)^2}` (:math:`\rho=1` exactly at
        the boundary) -- unlike :class:`HoleInPlateCase`'s
        `smoothing_width`, this is a fraction of the ellipse size, not a
        length, since :math:`\rho` itself is dimensionless.
    """

    grid: Grid
    matrix_lame_lambda: float
    matrix_lame_mu: float
    semi_axis_a: float
    semi_axis_b: float
    contrast: float = 1.0e-3
    center: tuple = (0.5, 0.5)
    smoothing_width: float = 0.0
    dealias: bool = False

    def __post_init__(self):
        if self.grid.shape[2] != 1:
            raise ValueError("EllipticalHoleInPlateCase requires a 2D grid (shape[2] == 1)")

    @property
    def _relative_x(self):
        x = self.grid.x[0] - self.center[0]
        y = self.grid.x[1] - self.center[1]
        return x, y

    @property
    def elliptical_radius(self):
        r"""Dimensionless :math:`\rho=\sqrt{(x/a)^2+(y/b)^2}`, shape
        ``grid.shape`` -- exactly 1 on the ellipse boundary."""
        x, y = self._relative_x
        rho = xp.sqrt((x / self.semi_axis_a) ** 2 + (y / self.semi_axis_b) ** 2)
        return xp.broadcast_to(rho, self.grid.shape)

    def lame_fields(self):
        """Return ``(lam(x), mu(x))``: matrix constants outside the hole,
        ``contrast`` times them inside -- see :meth:`HoleInPlateCase.lame_fields`,
        identical in spirit (including its `dealias` option) but blending
        over the dimensionless `elliptical_radius` instead of a Euclidean
        radius.
        """
        rho = self.elliptical_radius
        if self.dealias:
            indicator = xp.where(rho < 1.0, 0.0, 1.0).astype(self.grid.real_dtype)
            blend = self.grid.ifft(self.grid.fft(indicator) * self.grid.lanczos_filter)
            blend = xp.clip(blend, 0.0, 1.0)
        elif self.smoothing_width > 0:
            blend = 0.5 * (1.0 + xp.tanh((rho - 1.0) / self.smoothing_width))
        else:
            blend = xp.where(rho < 1.0, 0.0, 1.0)
        lam = (self.contrast + (1.0 - self.contrast) * blend) * self.matrix_lame_lambda
        mu = (self.contrast + (1.0 - self.contrast) * blend) * self.matrix_lame_mu
        return lam.astype(self.grid.real_dtype), mu.astype(self.grid.real_dtype)

    def solver(self):
        """Return the :class:`ElasticDeformation` solver for this case."""
        lam, mu = self.lame_fields()
        return ElasticDeformation(
            self.grid,
            lame_lambda=lam,
            lame_mu=mu,
            reference_lame_lambda=self.matrix_lame_lambda,
            reference_lame_mu=self.matrix_lame_mu,
        )

    def remote_stress(self, load, magnitude):
        """Return the ``(3, 3)`` remote stress tensor for `load`.

        Parameters
        ----------
        load : {"tension", "biaxial", "shear"}
        magnitude : float
        """
        sigma = xp.zeros((3, 3))
        if load == "tension":
            sigma[0, 0] = magnitude
        elif load == "biaxial":
            sigma[0, 0] = sigma[1, 1] = magnitude
        elif load == "shear":
            sigma[0, 1] = sigma[1, 0] = magnitude
        else:
            raise ValueError('load must be "tension", "biaxial", or "shear"')
        return sigma

    def macro_strain(self, load, magnitude, solver=None):
        """Return the macroscopic strain reproducing `remote_stress` far
        from the hole, via the matrix's own compliance -- see
        :meth:`HoleInPlateCase.macro_strain`.
        """
        solver = solver if solver is not None else self.solver()
        sigma = self.remote_stress(load, magnitude)
        return solver.reference_green.apply_compliance(sigma)

    def remote_antiplane_stress(self, magnitude):
        """Return the ``(3, 3)`` remote antiplane shear stress tensor:
        :math:`\\sigma_{13}=\\sigma_{31}=` `magnitude` -- the elliptical
        counterpart of :meth:`HoleInPlateCase.remote_antiplane_stress`,
        same convention (loaded along axis 1, kept separate from
        :meth:`remote_stress` for the same reason: see that method's
        docstring)."""
        sigma = xp.zeros((3, 3))
        sigma[0, 2] = sigma[2, 0] = magnitude
        return sigma

    def macro_antiplane_strain(self, magnitude, solver=None):
        """Antiplane counterpart of :meth:`macro_strain`: the macroscopic
        strain reproducing :meth:`remote_antiplane_stress` far from the
        hole, via the matrix's own compliance."""
        solver = solver if solver is not None else self.solver()
        sigma = self.remote_antiplane_stress(magnitude)
        return solver.reference_green.apply_compliance(sigma)

    def periodic_eshelby_tensor(self):
        """Numerically probed periodic Eshelby tensor for this case's
        actual ellipse/domain ratio -- see :func:`_periodic_eshelby_tensor`
        and :meth:`HoleInPlateCase.periodic_eshelby_tensor`."""
        return _periodic_eshelby_tensor(
            self.periodic_prescribed_eigenstrain_solution, self.elliptical_radius < 1.0
        )

    def periodic_analytic_stress(self, load, magnitude):
        """Periodic-array stress field for the hole under remote `load`
        (see :meth:`remote_stress`), built from an Eshelby equivalent
        eigenstrain (:func:`_periodic_equivalent_eigenstrain`, calibrated
        against this case's own numerically probed
        :meth:`periodic_eshelby_tensor` rather than the isolated closed
        form :func:`_ellipse_equivalent_eigenstrain` -- see
        :func:`_periodic_equivalent_eigenstrain`'s docstring for why the
        isolated calibration is wrong whenever the hole is not small
        compared to the periodic cell) over a matrix-material ellipse,
        solved exactly in Fourier space via :func:`_ellipse_fourier_transform`
        -- the elliptical counterpart of
        :meth:`HoleInPlateCase.periodic_analytic_solution`; see this
        class's own docstring for why an even earlier
        periodic-image-summation approach was replaced by the (then
        isolated-calibrated) Eshelby construction this in turn refines.

        Parameters
        ----------
        load : {"tension", "biaxial", "shear"}
        magnitude : float

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape``.
        """
        solver = self.solver()
        green = solver.reference_green
        operator = solver.operator

        eps_star = _periodic_equivalent_eigenstrain(
            magnitude, self.contrast, self.matrix_lame_lambda, self.matrix_lame_mu, load,
            self.periodic_eshelby_tensor(),
        )
        sigma_star = _isotropic_stress_apply(
            eps_star, self.matrix_lame_lambda, self.matrix_lame_mu
        )
        shape_hat = _ellipse_fourier_transform(
            self.grid, self.semi_axis_a, self.semi_axis_b,
            center=(self.center[0], self.center[1], 0.0),
        )

        eps0_hat = eps_star[:, :, None, None, None] * shape_hat[None, None, ...]
        sigma_star_hat = sigma_star[:, :, None, None, None] * shape_hat[None, None, ...]
        source_hat = -operator.div(sigma_star_hat)

        eps_bar = self.macro_strain(load, magnitude, solver=solver)
        eps_hat = green.field(source_hat, uniform_field=eps_bar)

        difference = eps_hat - eps0_hat
        trace = xp.einsum("ii...->...", difference)
        delta = xp.eye(3, dtype=difference.dtype).reshape((3, 3, 1, 1, 1))
        sigma_hat = self.matrix_lame_lambda * trace[None, None, ...] * delta + (
            2.0 * self.matrix_lame_mu * difference
        )
        return self.grid.ifft(sigma_hat)

    def periodic_pressurized_stress(self, magnitude):
        r"""Periodic-array stress field for the hole loaded by uniform
        internal pressure `magnitude` (zero remote stress) -- the
        elliptical-hole counterpart of
        :meth:`HoleInPlateCase.periodic_pressurized_stress`, built the same
        way and for the same reason (a spatially uniform stress tensor is
        trivially divergence-free everywhere, regardless of the
        heterogeneous stiffness field or the hole's shape, so it may be
        freely superposed onto the "biaxial" solution to cancel its remote
        stress and leave only the boundary's traction jump).
        """
        stress = self.periodic_analytic_stress("biaxial", magnitude)
        delta = xp.eye(3, dtype=stress.dtype).reshape((3, 3, 1, 1, 1))
        return stress - magnitude * delta

    def _void_correction_cartesian(self, x, y, load, magnitude):
        r"""Local (image-frame) correction to the uniform remote
        `magnitude`/`load` background for a single copy of the
        traction-free elliptical void centered at the origin of
        ``(x, y)`` -- the elliptical counterpart of
        :meth:`HoleInPlateCase._void_correction_cartesian`, built from
        the exact isolated Kirsch/Inglis closed form
        (:func:`_ellipse_void_cartesian_stress`) rather than the circular
        one. Used by :meth:`periodic_void_stress`.
        """
        a, b = self.semi_axis_a, self.semi_axis_b
        background = self.remote_stress(load, magnitude)
        rho = xp.sqrt((x / a) ** 2 + (y / b) ** 2)
        outside = rho >= 1.0 - 1.0e-9
        sigma_xx, sigma_yy, sigma_xy = _ellipse_void_cartesian_stress(
            x, y, a, b, load, magnitude
        )
        sigma_xx = xp.real(sigma_xx) - background[0, 0]
        sigma_yy = xp.real(sigma_yy) - background[1, 1]
        sigma_xy = xp.real(sigma_xy) - background[0, 1]
        return (
            xp.where(outside, sigma_xx, -background[0, 0]),
            xp.where(outside, sigma_yy, -background[1, 1]),
            xp.where(outside, sigma_xy, -background[0, 1]),
        )

    def periodic_void_stress(self, load, magnitude, n_images=1):
        r"""Periodic-array stress field for a traction-free elliptical
        void under uniform remote `load`/`magnitude`, built by summing
        the exact isolated Kirsch/Inglis closed form
        (:func:`_ellipse_void_cartesian_stress`) over periodic images --
        the elliptical counterpart of :meth:`HoleInPlateCase.periodic_void_stress`.

        Like :meth:`HoleInPlateCase.periodic_void_stress`, the raw image
        sum's own domain mean is recentered to `magnitude` outside the
        ellipse (``field -= mean(field) - magnitude``, direct
        substitution, not a self-consistent solve) -- see that method's
        docstring for the exact recipe and its provenance
        (``eshelby.cpp``, this project's original pre-crystallite
        reference implementation, validated there against genuine
        simulation results down to the shape of the curve, not just an
        earlier claim about a single tip value).

        An earlier version of this docstring instead claimed a large
        error against this class's own numeric CG solver ("up to 46% at
        b/a=8"), attributing it to periodic self-interaction the image
        sum supposedly cannot capture. That comparison used the
        *pre-recentering* raw sum and sampled at a fixed absolute offset
        outside the boundary -- at high eccentricity that offset badly
        undersamples the actual (very sharp, curvature-scaled) tip
        concentration, and this project could find no data anywhere in
        this codebase supporting a 46% figure specifically. With
        recentering, checked directly against the numeric solver: at
        ``semi_axis_a=semi_axis_b`` (circular limit) near-tip tension
        stress agrees to a few percent; the gap grows with eccentricity,
        consistent with :meth:`HoleInPlateCase.periodic_void_stress`'s
        own residual gap against *its* numeric solver -- both expected
        from the same prescribed-stress vs. prescribed-strain boundary
        condition mismatch documented there, not a defect in the image
        sum.

        Converges essentially immediately in `n_images` (checked up to a
        13x13 block of images at ``b/a=4``: agrees with a mere ``n_images=1``
        3x3 block to 4 significant figures, since -- as with
        :meth:`HoleInPlateCase.periodic_void_stress` -- the recentering
        step above carries most of the periodicity correction, not the
        raw sum over distant images) -- ignores `contrast` entirely,
        exactly like :meth:`HoleInPlateCase.periodic_void_stress`, and
        inherits the same caveat: only valid as a reference when
        `contrast` is small enough that the numerical hole is itself a
        good void approximation.

        Regression note: an earlier version of this method summed every
        image's correction unconditionally and left interior points
        subject to the same outside-only recentering as the exterior,
        letting neighboring images' exterior field leak into the true
        ellipse's own interior once ``n_images>0`` -- exactly the bug
        :meth:`HoleInPlateCase._periodic_image_sum`'s docstring documents
        and fixes for the circular case (a true void's interior must stay
        exactly zero, not accumulate neighbor contamination). Fixed here
        the same way: the home image's own correction is evaluated once,
        separately, and used as-is inside the ellipse regardless of
        `n_images`.

        Parameters
        ----------
        load : {"tension", "biaxial", "shear"}
        magnitude : float
        n_images : int, default=1
            Sum periodic images in ``[-n_images, n_images]`` along each
            in-plane axis.

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape`` -- only the in-plane
            ``sigma_xx``, ``sigma_yy``, ``sigma_xy`` components are
            populated.
        """
        x, y = self.grid.x[0], self.grid.x[1]
        length_x, length_y = self.grid.lengths[0], self.grid.lengths[1]
        background = self.remote_stress(load, magnitude)

        sigma_xx = background[0, 0] * xp.ones(self.grid.shape)
        sigma_yy = background[1, 1] * xp.ones(self.grid.shape)
        sigma_xy = background[0, 1] * xp.ones(self.grid.shape)

        home_dx = x - self.center[0]
        home_dy = y - self.center[1]
        home_xx, home_yy, home_xy = self._void_correction_cartesian(home_dx, home_dy, load, magnitude)

        for n1 in range(-n_images, n_images + 1):
            for n2 in range(-n_images, n_images + 1):
                dx = x - (self.center[0] + n1 * length_x)
                dy = y - (self.center[1] + n2 * length_y)
                sxx, syy, sxy = self._void_correction_cartesian(dx, dy, load, magnitude)
                sigma_xx = sigma_xx + sxx
                sigma_yy = sigma_yy + syy
                sigma_xy = sigma_xy + sxy

        outside = self.elliptical_radius >= 1.0
        recentered_xx = sigma_xx - (xp.mean(sigma_xx) - background[0, 0])
        recentered_yy = sigma_yy - (xp.mean(sigma_yy) - background[1, 1])
        recentered_xy = sigma_xy - (xp.mean(sigma_xy) - background[0, 1])

        sigma_xx = xp.where(outside, recentered_xx, background[0, 0] + home_xx)
        sigma_yy = xp.where(outside, recentered_yy, background[1, 1] + home_yy)
        sigma_xy = xp.where(outside, recentered_xy, background[0, 1] + home_xy)

        stress = xp.zeros((3, 3) + self.grid.shape, dtype=sigma_xx.dtype)
        stress[0, 0] = sigma_xx
        stress[1, 1] = sigma_yy
        stress[0, 1] = stress[1, 0] = sigma_xy
        return stress

    def _antiplane_void_correction_cartesian(self, x, y, magnitude):
        r"""Local (image-frame) correction to the uniform remote
        `magnitude` antiplane background for a single copy of the
        traction-free elliptical void centered at the origin of
        ``(x, y)`` -- the elliptical counterpart of
        :meth:`HoleInPlateCase._antiplane_void_correction_cartesian`,
        built from the exact isolated closed form
        (:func:`_ellipse_void_antiplane_cartesian_stress`) rather than the
        circular one. Used by :meth:`periodic_antiplane_void_stress`.
        """
        a, b = self.semi_axis_a, self.semi_axis_b
        background = self.remote_antiplane_stress(magnitude)
        rho = xp.sqrt((x / a) ** 2 + (y / b) ** 2)
        outside = rho >= 1.0 - 1.0e-9
        sigma_13, sigma_23 = _ellipse_void_antiplane_cartesian_stress(x, y, a, b, magnitude)
        sigma_13 = sigma_13 - background[0, 2]
        sigma_23 = sigma_23 - background[1, 2]
        return (
            xp.where(outside, sigma_13, -background[0, 2]),
            xp.where(outside, sigma_23, -background[1, 2]),
        )

    def periodic_antiplane_void_stress(self, magnitude, n_images=1):
        r"""Periodic-array antiplane stress field for a traction-free
        elliptical void under uniform remote antiplane shear `magnitude`,
        built by summing the exact isolated closed form
        (:func:`_ellipse_void_antiplane_cartesian_stress`) over periodic
        images -- the elliptical counterpart of
        :meth:`HoleInPlateCase.periodic_antiplane_void_stress`, with the
        same recentering/home-image structure as this class's own
        :meth:`periodic_void_stress` (see that method's docstring), just
        for the two antiplane components instead of the three in-plane
        ones.

        Parameters
        ----------
        magnitude : float
        n_images : int, default=1

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape`` -- only ``sigma_13``/``sigma_31``
            and ``sigma_23``/``sigma_32`` are populated.
        """
        x, y = self.grid.x[0], self.grid.x[1]
        length_x, length_y = self.grid.lengths[0], self.grid.lengths[1]
        background = self.remote_antiplane_stress(magnitude)

        sigma_13 = background[0, 2] * xp.ones(self.grid.shape)
        sigma_23 = background[1, 2] * xp.ones(self.grid.shape)

        home_dx = x - self.center[0]
        home_dy = y - self.center[1]
        home_13, home_23 = self._antiplane_void_correction_cartesian(home_dx, home_dy, magnitude)

        for n1 in range(-n_images, n_images + 1):
            for n2 in range(-n_images, n_images + 1):
                dx = x - (self.center[0] + n1 * length_x)
                dy = y - (self.center[1] + n2 * length_y)
                s13, s23 = self._antiplane_void_correction_cartesian(dx, dy, magnitude)
                sigma_13 = sigma_13 + s13
                sigma_23 = sigma_23 + s23

        outside = self.elliptical_radius >= 1.0
        recentered_13 = sigma_13 - (xp.mean(sigma_13) - background[0, 2])
        recentered_23 = sigma_23 - (xp.mean(sigma_23) - background[1, 2])

        sigma_13 = xp.where(outside, recentered_13, background[0, 2] + home_13)
        sigma_23 = xp.where(outside, recentered_23, background[1, 2] + home_23)

        stress = xp.zeros((3, 3) + self.grid.shape, dtype=sigma_13.dtype)
        stress[0, 2] = stress[2, 0] = sigma_13
        stress[1, 2] = stress[2, 1] = sigma_23
        return stress

    def periodic_mean_stress(self, load, magnitude):
        """Exact domain-mean ``(3, 3)`` stress of the periodic array under
        strain control, by the lattice-sum Eshelby tensor -- see
        :func:`_periodic_mean_stress`. FFT- and grid-free."""
        return xp.asarray(
            _periodic_mean_stress(
                load, magnitude, self.contrast, self.matrix_lame_lambda, self.matrix_lame_mu,
                self.semi_axis_a, self.semi_axis_b,
                float(self.grid.lengths[0]), float(self.grid.lengths[1]),
            )
        )

    def periodic_antiplane_mean_stress(self, magnitude):
        """Antiplane counterpart of :meth:`periodic_mean_stress`."""
        return xp.asarray(
            _periodic_antiplane_mean_stress(
                magnitude, self.contrast, self.matrix_lame_mu,
                self.semi_axis_a, self.semi_axis_b,
                float(self.grid.lengths[0]), float(self.grid.lengths[1]),
            )
        )

    def periodic_interior_stress(self, load, magnitude):
        """Exact uniform interior ``(3, 3)`` stress of the periodic array
        under strain control -- see :func:`_periodic_interior_stress`."""
        return xp.asarray(
            _periodic_interior_stress(
                load, magnitude, self.contrast, self.matrix_lame_lambda, self.matrix_lame_mu,
                self.semi_axis_a, self.semi_axis_b,
                float(self.grid.lengths[0]), float(self.grid.lengths[1]),
            )
        )

    def periodic_antiplane_interior_stress(self, magnitude):
        """Antiplane counterpart of :meth:`periodic_interior_stress`."""
        return xp.asarray(
            _periodic_antiplane_interior_stress(
                magnitude, self.contrast, self.matrix_lame_mu,
                self.semi_axis_a, self.semi_axis_b,
                float(self.grid.lengths[0]), float(self.grid.lengths[1]),
            )
        )

    def periodic_equivalent_eigenstrain(self, load, magnitude):
        """Periodic-array equivalent eigenstrain ``(3, 3)`` of the
        inhomogeneity, calibrated with the exact lattice-sum Eshelby tensor
        (:func:`_lattice_eshelby_tensor`) rather than the isolated closed
        form: the strength each image must carry so that the image sum
        reproduces the periodic solution."""
        s = _lattice_eshelby_tensor(
            self.semi_axis_a, self.semi_axis_b,
            float(self.grid.lengths[0]), float(self.grid.lengths[1]),
            self.matrix_lame_lambda, self.matrix_lame_mu,
        )
        return xp.asarray(
            _periodic_equivalent_eigenstrain(
                magnitude, self.contrast, self.matrix_lame_lambda, self.matrix_lame_mu, load, s
            )
        )

    def periodic_antiplane_equivalent_eigenstrain(self, magnitude):
        """Antiplane counterpart of :meth:`periodic_equivalent_eigenstrain`,
        calibrated with the lattice-sum ``s1313``."""
        s = _lattice_eshelby_tensor(
            self.semi_axis_a, self.semi_axis_b,
            float(self.grid.lengths[0]), float(self.grid.lengths[1]),
            self.matrix_lame_lambda, self.matrix_lame_mu,
        )
        return xp.asarray(
            _antiplane_periodic_equivalent_eigenstrain(
                magnitude, self.contrast, self.matrix_lame_mu, s["s1313"]
            )
        )

    def periodic_line_force_stress(self, traction, n_images=1):
        r"""Periodic-array stress of a radial line force
        :math:`f=q\nabla\chi` (`traction` :math:`=q`, directed inward,
        :math:`\chi` the ellipse indicator) applied on the boundary of a
        region of the *same* material as the matrix -- the "pressurized
        hole" of ``cylindrical_pressurized_hole.py`` (not the pressurized
        cavity, whose exterior differs).

        A source :math:`f=-\nabla\cdot\sigma^*` is exactly the equilibrium
        source of the eigenstress :math:`\sigma^*=-q\chi I` (in plane),
        which is a uniform dilatation eigenstrain
        :math:`e=-q/(2(\lambda+\mu))` over the ellipse. The body-force
        stress is :math:`C:\varepsilon` while the eigenstrain stress is
        :math:`C:\varepsilon-\sigma^*`, so the two are identical outside
        and differ by :math:`-q\chi I` inside::

            sigma_force = sigma_eigenstrain - q * chi * I     (in plane)

        No new kernel: this is :meth:`periodic_prescribed_eigenstrain_void_stress`
        (exact closed-form image sum) plus that uniform interior shift.
        For a circle the interior is the uniform hydrostatic
        :math:`-p I`, :math:`p=q(\lambda+\mu)/(\lambda+2\mu)`, the exterior
        the isolated Lame field :math:`\sigma_{rr}=-\sigma_{\theta\theta}
        =\frac{\mu q}{\lambda+2\mu}(R/r)^2` plus the periodic-image correction,
        and the domain mean is zero (zero net force, zero macroscopic strain) up to
        the grid's sampling of the disk area (~0.1% of :math:`p` at 256^2).

        Parameters
        ----------
        traction : float
        n_images : int, default=1

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape``; only the in-plane components are
            populated.
        """
        lam, mu = self.matrix_lame_lambda, self.matrix_lame_mu
        eigenstrain = xp.zeros((3, 3))
        eigenstrain[0, 0] = eigenstrain[1, 1] = -traction / (2.0 * (lam + mu))
        stress = xp.asarray(
            self.periodic_prescribed_eigenstrain_void_stress(eigenstrain, n_images=n_images)
        )
        inside = self.elliptical_radius < 1.0
        stress[0, 0] = xp.where(inside, stress[0, 0] - traction, stress[0, 0])
        stress[1, 1] = xp.where(inside, stress[1, 1] - traction, stress[1, 1])
        return stress

    def periodic_prescribed_eigenstrain_interior_stress(self, eigenstrain_tensor):
        """Exact interior stress ``(3, 3)`` of the periodic array for a
        uniform `eigenstrain_tensor` prescribed over the ellipse in a
        homogeneous matrix (zero macroscopic strain), from the lattice-sum
        Eshelby tensor -- see :func:`_periodic_prescribed_interior_stress`.
        The classical "internal stress versus aspect ratio" Eshelby result,
        with the periodic-cell correction included."""
        return xp.asarray(
            _periodic_prescribed_interior_stress(
                eigenstrain_tensor, self.matrix_lame_lambda, self.matrix_lame_mu,
                self.semi_axis_a, self.semi_axis_b,
                float(self.grid.lengths[0]), float(self.grid.lengths[1]),
            )
        )

    def periodic_void_pressurized_stress(self, magnitude, n_images=1):
        r"""Periodic-array stress field for an elliptical void loaded by
        uniform internal pressure `magnitude` (zero remote stress) --
        the image-summed counterpart of :meth:`periodic_pressurized_stress`,
        built from :meth:`periodic_void_stress` instead of
        :meth:`periodic_analytic_stress`.
        """
        stress = self.periodic_void_stress("biaxial", magnitude, n_images=n_images)
        delta = xp.eye(3, dtype=stress.dtype).reshape((3, 3, 1, 1, 1))
        return stress - magnitude * delta

    def _prescribed_eigenstrain_correction_cartesian(self, x, y, eigenstrain_tensor):
        r"""Local (image-frame) stress for a single isolated copy of a
        uniform `eigenstrain_tensor` prescribed directly over the ellipse
        in an otherwise homogeneous matrix (``contrast=1``, no equivalent-
        inclusion step) -- exterior from
        :func:`_ellipse_inhomogeneity_exterior_stress`, interior from
        :func:`_ellipse_inhomogeneity_interior_stress`, evaluated at
        `eigenstrain_tensor` directly rather than at an
        :func:`_ellipse_equivalent_eigenstrain`-derived one. This is the
        genuinely dilute (isolated) special case of the same H/T-tensor
        machinery :meth:`_inhomogeneity_correction_cartesian` uses for a
        finite-contrast inhomogeneity -- see that method's docstring for
        the shared reasoning (no separate background subtraction needed;
        image-summed by :meth:`periodic_prescribed_eigenstrain_void_stress`
        the same way).

        Also covers the antiplane Voigt components (``eigenstrain_tensor[0, 2]``/
        ``[1, 2]``), via
        :func:`_ellipse_inhomogeneity_antiplane_exterior_stress`/
        :func:`_ellipse_inhomogeneity_antiplane_interior_stress` -- unlike
        :func:`_ellipse_inhomogeneity_exterior_stress`, which does not
        (see that function's own docstring for why).
        """
        a, b = self.semi_axis_a, self.semi_axis_b
        nu = self.matrix_lame_lambda / (2.0 * (self.matrix_lame_lambda + self.matrix_lame_mu))

        rho = xp.sqrt((x / a) ** 2 + (y / b) ** 2)
        outside = rho >= 1.0 - 1.0e-9
        rho_safe = xp.where(rho == 0, 1.0, rho)
        x_scaled = xp.where(outside, x, x / rho_safe)
        y_scaled = xp.where(outside, y, y / rho_safe)
        x_safe = xp.where(rho == 0, a, x_scaled)
        y_safe = xp.where(rho == 0, 0.0, y_scaled)

        sigma_xx_ext, sigma_yy_ext, sigma_xy_ext = _ellipse_inhomogeneity_exterior_stress(
            x_safe, y_safe, eigenstrain_tensor, a, b, self.matrix_lame_mu, nu
        )
        sigma_xx_int, sigma_yy_int, sigma_xy_int = _ellipse_inhomogeneity_interior_stress(
            eigenstrain_tensor, a, b, self.matrix_lame_mu, nu
        )
        sigma_13_ext, sigma_23_ext = _ellipse_inhomogeneity_antiplane_exterior_stress(
            x_safe, y_safe, eigenstrain_tensor, a, b, self.matrix_lame_mu
        )
        sigma_13_int, sigma_23_int = _ellipse_inhomogeneity_antiplane_interior_stress(
            eigenstrain_tensor, a, b, self.matrix_lame_mu
        )
        return (
            xp.where(outside, xp.real(sigma_xx_ext), sigma_xx_int),
            xp.where(outside, xp.real(sigma_yy_ext), sigma_yy_int),
            xp.where(outside, xp.real(sigma_xy_ext), sigma_xy_int),
            xp.where(outside, sigma_13_ext, sigma_13_int),
            xp.where(outside, sigma_23_ext, sigma_23_int),
        )

    def periodic_prescribed_eigenstrain_void_stress(self, eigenstrain_tensor, n_images=1):
        r"""Periodic-array stress field for a uniform `eigenstrain_tensor`
        prescribed directly over the ellipse in an otherwise homogeneous
        matrix, built by summing the exact isolated H/T-tensor closed form
        (:meth:`_prescribed_eigenstrain_correction_cartesian`) over
        periodic images -- the ``eshelby.cpp``-recipe (closed-form,
        image-summed) counterpart of
        :meth:`periodic_prescribed_eigenstrain_solution` (the FFT
        construction), for the cases (thin, few-grid-point ellipses:
        dislocation lines, point defects) where that FFT construction's
        own shape-function aliasing makes it unreliable -- see
        :meth:`periodic_analytic_stress`'s and ``thin_crack.py``'s own
        history for why an aliased shape-FFT washes out exactly the
        near-core concentration these cases exist to check, and
        :func:`_ellipse_inhomogeneity_exterior_stress`'s docstring for
        why the closed form used here does not share that failure mode
        (no shape-FFT step to alias).

        Recentered like :meth:`periodic_void_stress`, but to a different
        (nonzero, in general) target: not ``0``, and not ``magnitude`` --
        the true periodic problem's own domain-mean stress, for zero
        macroscopic strain (this construction's implicit boundary
        condition -- no ``uniform_field`` is added, matching
        :meth:`periodic_prescribed_eigenstrain_solution`'s own default).
        Checked directly, not assumed: expanding that FFT construction's
        own ``sigma_hat`` at the zero (mean) mode gives
        ``sigma_mean = -f*sigma_star`` exactly, ``f`` the ellipse's area
        fraction of the periodic cell and ``sigma_star`` the isotropic
        stress :func:`_isotropic_stress_apply` gives for
        `eigenstrain_tensor` directly (*not* zero, as an earlier version
        of this docstring assumed by analogy with the void/hole cases --
        an eigenstrain region's own area fraction of *strain* is clamped
        to zero macroscopically here, not its *stress*, so the reaction
        stress is generally nonzero.) -- confirmed to match
        :meth:`periodic_prescribed_eigenstrain_solution`'s own domain mean
        to floating-point precision. Interior points still take the
        home-image-only value, not the recentered one, the same reason
        :meth:`periodic_void_stress` does (no neighbor-image leakage into
        the true ellipse's own interior).

        Also covers the antiplane Voigt components now, the same way
        :meth:`_prescribed_eigenstrain_correction_cartesian` does -- e.g.
        a screw dislocation's ``eps_13``, needed by
        ``screw_dislocation.py``.

        Parameters
        ----------
        eigenstrain_tensor : array_like
            Shape ``(3, 3)``.
        n_images : int, default=1

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape``.
        """
        eigenstrain_tensor = xp.asarray(eigenstrain_tensor)
        x, y = self.grid.x[0], self.grid.x[1]
        length_x, length_y = self.grid.lengths[0], self.grid.lengths[1]

        sigma_xx = xp.zeros(self.grid.shape)
        sigma_yy = xp.zeros(self.grid.shape)
        sigma_xy = xp.zeros(self.grid.shape)
        sigma_13 = xp.zeros(self.grid.shape)
        sigma_23 = xp.zeros(self.grid.shape)

        home_dx = x - self.center[0]
        home_dy = y - self.center[1]
        home_xx, home_yy, home_xy, home_13, home_23 = self._prescribed_eigenstrain_correction_cartesian(
            home_dx, home_dy, eigenstrain_tensor
        )

        for n1 in range(-n_images, n_images + 1):
            for n2 in range(-n_images, n_images + 1):
                dx = x - (self.center[0] + n1 * length_x)
                dy = y - (self.center[1] + n2 * length_y)
                sxx, syy, sxy, s13, s23 = self._prescribed_eigenstrain_correction_cartesian(
                    dx, dy, eigenstrain_tensor
                )
                sigma_xx = sigma_xx + sxx
                sigma_yy = sigma_yy + syy
                sigma_xy = sigma_xy + sxy
                sigma_13 = sigma_13 + s13
                sigma_23 = sigma_23 + s23

        area_fraction = xp.pi * self.semi_axis_a * self.semi_axis_b / (length_x * length_y)
        sigma_star = _isotropic_stress_apply(
            eigenstrain_tensor, self.matrix_lame_lambda, self.matrix_lame_mu
        )
        target_xx = -area_fraction * sigma_star[0, 0]
        target_yy = -area_fraction * sigma_star[1, 1]
        target_xy = -area_fraction * sigma_star[0, 1]
        target_13 = -area_fraction * sigma_star[0, 2]
        target_23 = -area_fraction * sigma_star[1, 2]

        outside = self.elliptical_radius >= 1.0
        recentered_xx = sigma_xx - (xp.mean(sigma_xx) - target_xx)
        recentered_yy = sigma_yy - (xp.mean(sigma_yy) - target_yy)
        recentered_xy = sigma_xy - (xp.mean(sigma_xy) - target_xy)
        recentered_13 = sigma_13 - (xp.mean(sigma_13) - target_13)
        recentered_23 = sigma_23 - (xp.mean(sigma_23) - target_23)

        sigma_xx = xp.where(outside, recentered_xx, home_xx)
        sigma_yy = xp.where(outside, recentered_yy, home_yy)
        sigma_xy = xp.where(outside, recentered_xy, home_xy)
        sigma_13 = xp.where(outside, recentered_13, home_13)
        sigma_23 = xp.where(outside, recentered_23, home_23)

        stress = xp.zeros((3, 3) + self.grid.shape, dtype=sigma_xx.dtype)
        stress[0, 0] = sigma_xx
        stress[1, 1] = sigma_yy
        stress[0, 1] = stress[1, 0] = sigma_xy
        stress[0, 2] = stress[2, 0] = sigma_13
        stress[1, 2] = stress[2, 1] = sigma_23
        return stress

    def _inhomogeneity_correction_cartesian(self, x, y, load, magnitude, eps_star=None):
        r"""Contrast-aware analog of :meth:`_void_correction_cartesian`:
        local correction for a single isolated copy of this case's
        *actual*, finite-`contrast` elliptical inhomogeneity, not the void
        limit -- exterior from :func:`_ellipse_inhomogeneity_exterior_stress`,
        interior from :func:`_ellipse_inhomogeneity_interior_stress`, both
        evaluated at the isolated equivalent eigenstrain
        (:func:`_ellipse_equivalent_eigenstrain`, not the periodic-corrected
        :func:`_periodic_equivalent_eigenstrain` -- same reasoning
        :meth:`HoleInPlateCase._inhomogeneity_correction_cartesian` already
        uses for the circle: this feeds an *isolated*-closed-form image
        sum, not the exact-periodicity FFT construction
        :meth:`periodic_analytic_stress` uses).

        Unlike :meth:`HoleInPlateCase._inhomogeneity_correction_cartesian`,
        no separate background subtraction is needed here: both H/T-tensor
        functions already return exactly this project's own *correction*
        convention on their own (see their docstrings) -- a consequence of
        being built from an eigenstrain rather than being parameterized
        directly by the remote magnitude.

        Used by :meth:`periodic_inhomogeneity_stress` -- this project's
        original (pre-crystallite) ``eshelby.cpp`` reference's own recipe
        (its ``e()``/``f()``/``g()`` functions image-summed over a
        lattice), the elliptical counterpart of
        :meth:`HoleInPlateCase._inhomogeneity_correction_cartesian`.

        ``load="shear"`` used to inherit
        :func:`_ellipse_inhomogeneity_exterior_stress`'s shear-coupling
        bug; now fixed there (see that function's docstring) and verified
        the same way as ``"tension"``/``"biaxial"`` (and so ``"pressure"``,
        built from ``"biaxial"``).
        """
        a, b = self.semi_axis_a, self.semi_axis_b
        nu = self.matrix_lame_lambda / (2.0 * (self.matrix_lame_lambda + self.matrix_lame_mu))
        if eps_star is None:
            eps_star = _ellipse_equivalent_eigenstrain(
                magnitude, self.contrast, self.matrix_lame_lambda, self.matrix_lame_mu, load, a, b
            )
        return _inplane_eigenstrain_correction(x, y, eps_star, a, b, self.matrix_lame_mu, nu)

    def periodic_inhomogeneity_stress(self, load, magnitude, n_images=1):
        r"""Periodic-array stress field for this case's actual, finite-
        `contrast` elliptical inhomogeneity under uniform remote
        `load`/`magnitude`, built by summing the exact isolated closed
        form (:meth:`_inhomogeneity_correction_cartesian`) over periodic
        images -- the elliptical counterpart of
        :meth:`HoleInPlateCase.periodic_inhomogeneity_stress`, same
        two-pass structure (home-image-only interior overwrite,
        outside-only recentering) :meth:`periodic_void_stress` now also
        uses.

        Fixes, relative to :meth:`periodic_void_stress`, the same two
        things :meth:`HoleInPlateCase.periodic_inhomogeneity_stress` fixes
        relative to :meth:`HoleInPlateCase.periodic_void_stress`: the
        void-vs-actual-`contrast` mismatch, and the recentering target
        itself -- recentered to :meth:`periodic_analytic_stress`'s own
        domain mean (this solver's actual strain-controlled boundary
        condition), not the naive nominal `magnitude`
        (``eshelby.cpp``'s own stress-controlled target,
        :meth:`periodic_void_stress` still uses).

        Parameters
        ----------
        load : {"tension", "biaxial", "shear"}
        magnitude : float
        n_images : int, default=1

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape``.
        """
        x, y = self.grid.x[0], self.grid.x[1]
        length_x, length_y = self.grid.lengths[0], self.grid.lengths[1]
        background = self.remote_stress(load, magnitude)

        recenter_target = self.periodic_mean_stress(load, magnitude)

        sigma_xx = background[0, 0] * xp.ones(self.grid.shape)
        sigma_yy = background[1, 1] * xp.ones(self.grid.shape)
        sigma_xy = background[0, 1] * xp.ones(self.grid.shape)

        eps_star = self.periodic_equivalent_eigenstrain(load, magnitude)

        for n1 in range(-n_images, n_images + 1):
            for n2 in range(-n_images, n_images + 1):
                dx = x - (self.center[0] + n1 * length_x)
                dy = y - (self.center[1] + n2 * length_y)
                sxx, syy, sxy = self._inhomogeneity_correction_cartesian(
                    dx, dy, load, magnitude, eps_star=eps_star
                )
                sigma_xx = sigma_xx + sxx
                sigma_yy = sigma_yy + syy
                sigma_xy = sigma_xy + sxy

        outside = self.elliptical_radius >= 1.0
        interior = self.periodic_interior_stress(load, magnitude)
        recentered_xx = sigma_xx - _exterior_shift(sigma_xx, outside, interior[0, 0], recenter_target[0, 0])
        recentered_yy = sigma_yy - _exterior_shift(sigma_yy, outside, interior[1, 1], recenter_target[1, 1])
        recentered_xy = sigma_xy - _exterior_shift(sigma_xy, outside, interior[0, 1], recenter_target[0, 1])
        sigma_xx = xp.where(outside, recentered_xx, interior[0, 0])
        sigma_yy = xp.where(outside, recentered_yy, interior[1, 1])
        sigma_xy = xp.where(outside, recentered_xy, interior[0, 1])

        stress = xp.zeros((3, 3) + self.grid.shape, dtype=sigma_xx.dtype)
        stress[0, 0] = sigma_xx
        stress[1, 1] = sigma_yy
        stress[0, 1] = stress[1, 0] = sigma_xy
        return stress

    def periodic_inhomogeneity_pressurized_stress(self, magnitude, n_images=1):
        r"""Periodic-array stress field for this case's actual, finite-
        `contrast` elliptical inhomogeneity loaded by uniform internal
        pressure `magnitude` (zero remote stress) -- the contrast-aware
        generalization of :meth:`periodic_void_pressurized_stress`, built
        from :meth:`periodic_inhomogeneity_stress` instead of
        :meth:`periodic_void_stress` (same "biaxial minus uniform
        background" construction as :meth:`periodic_pressurized_stress` --
        see that method's docstring for why the subtraction itself is
        valid regardless of contrast or shape).
        """
        stress = self.periodic_inhomogeneity_stress("biaxial", magnitude, n_images=n_images)
        delta = xp.eye(3, dtype=stress.dtype).reshape((3, 3, 1, 1, 1))
        return stress - magnitude * delta

    def periodic_eshelby_antiplane_tensor(self):
        """Numerically probed periodic antiplane Eshelby tensor component
        ``s1313`` for this case's actual ellipse/domain ratio -- see
        :func:`_periodic_eshelby_antiplane_tensor` and
        :meth:`HoleInPlateCase.periodic_eshelby_antiplane_tensor`."""
        return _periodic_eshelby_antiplane_tensor(
            self.periodic_prescribed_eigenstrain_solution, self.elliptical_radius < 1.0
        )

    def periodic_antiplane_analytic_stress(self, magnitude):
        """Periodic-array antiplane stress field for the ellipse under
        remote antiplane shear `magnitude` (see
        :meth:`remote_antiplane_stress`), built from an Eshelby equivalent
        eigenstrain (:func:`_antiplane_periodic_equivalent_eigenstrain`,
        calibrated against :meth:`periodic_eshelby_antiplane_tensor`) over
        a matrix-material ellipse, solved exactly in Fourier space -- the
        antiplane counterpart of :meth:`periodic_analytic_stress`.

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape`` -- only the ``(0, 2)``/``(1, 2)``
            components are nonzero.
        """
        solver = self.solver()
        green = solver.reference_green
        operator = solver.operator

        eps_star = _antiplane_periodic_equivalent_eigenstrain(
            magnitude, self.contrast, self.matrix_lame_mu, self.periodic_eshelby_antiplane_tensor()
        )
        sigma_star = _isotropic_stress_apply(
            eps_star, self.matrix_lame_lambda, self.matrix_lame_mu
        )
        shape_hat = _ellipse_fourier_transform(
            self.grid, self.semi_axis_a, self.semi_axis_b,
            center=(self.center[0], self.center[1], 0.0),
        )

        eps0_hat = eps_star[:, :, None, None, None] * shape_hat[None, None, ...]
        sigma_star_hat = sigma_star[:, :, None, None, None] * shape_hat[None, None, ...]
        source_hat = -operator.div(sigma_star_hat)

        eps_bar = self.macro_antiplane_strain(magnitude, solver=solver)
        eps_hat = green.field(source_hat, uniform_field=eps_bar)

        difference = eps_hat - eps0_hat
        trace = xp.einsum("ii...->...", difference)
        delta = xp.eye(3, dtype=difference.dtype).reshape((3, 3, 1, 1, 1))
        sigma_hat = self.matrix_lame_lambda * trace[None, None, ...] * delta + (
            2.0 * self.matrix_lame_mu * difference
        )
        return self.grid.ifft(sigma_hat)

    def _antiplane_inhomogeneity_correction_cartesian(self, x, y, magnitude, eps_star=None):
        r"""Contrast-aware antiplane local correction for a single isolated
        copy of this case's finite-`contrast` elliptical inhomogeneity:
        the equivalent eigenstrain
        (:func:`_antiplane_periodic_equivalent_eigenstrain`, at the
        isolated elliptical-cylinder Eshelby component
        :math:`S_{1313}=b/(2(a+b))`, which follows from
        :func:`_ellipse_inhomogeneity_antiplane_interior_stress`'s uniform
        interior stress -- :math:`1/4` for a circle) evaluated through
        :meth:`_prescribed_eigenstrain_correction_cartesian`. Like
        :meth:`_inhomogeneity_correction_cartesian`, this is already a
        pure correction (no background subtraction needed).

        Returns
        -------
        sigma_13, sigma_23 : ndarray
        """
        a, b = self.semi_axis_a, self.semi_axis_b
        if eps_star is None:
            eps_star = _antiplane_periodic_equivalent_eigenstrain(
                magnitude, self.contrast, self.matrix_lame_mu, b / (2.0 * (a + b))
            )
        *_, sigma_13, sigma_23 = self._prescribed_eigenstrain_correction_cartesian(x, y, eps_star)
        return sigma_13, sigma_23

    def periodic_antiplane_inhomogeneity_stress(self, magnitude, n_images=1):
        r"""Periodic-array antiplane stress field for this case's actual,
        finite-`contrast` elliptical inhomogeneity under remote antiplane
        shear `magnitude`, built by summing the exact isolated closed form
        (:meth:`_antiplane_inhomogeneity_correction_cartesian`) over
        periodic images -- the elliptical counterpart of
        :meth:`HoleInPlateCase.periodic_antiplane_inhomogeneity_stress`
        and the antiplane counterpart of :meth:`periodic_inhomogeneity_stress`,
        with the same two-pass structure (home-image-only interior
        overwrite, outside-only recentering to
        :meth:`periodic_antiplane_analytic_stress`'s domain mean).

        Parameters
        ----------
        magnitude : float
        n_images : int, default=1

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape`` -- only the ``(0, 2)``/``(1, 2)``
            components (and their transposes) are populated.
        """
        x, y = self.grid.x[0], self.grid.x[1]
        length_x, length_y = self.grid.lengths[0], self.grid.lengths[1]
        background = self.remote_antiplane_stress(magnitude)

        mean_stress = self.periodic_antiplane_mean_stress(magnitude)
        eps_star = self.periodic_antiplane_equivalent_eigenstrain(magnitude)
        target_13 = mean_stress[0, 2]
        target_23 = mean_stress[1, 2]

        sigma_13 = background[0, 2] * xp.ones(self.grid.shape)
        sigma_23 = background[1, 2] * xp.ones(self.grid.shape)


        for n1 in range(-n_images, n_images + 1):
            for n2 in range(-n_images, n_images + 1):
                dx = x - (self.center[0] + n1 * length_x)
                dy = y - (self.center[1] + n2 * length_y)
                s13, s23 = self._antiplane_inhomogeneity_correction_cartesian(
                    dx, dy, magnitude, eps_star=eps_star
                )
                sigma_13 = sigma_13 + s13
                sigma_23 = sigma_23 + s23

        outside = self.elliptical_radius >= 1.0
        interior = self.periodic_antiplane_interior_stress(magnitude)
        recentered_13 = sigma_13 - _exterior_shift(sigma_13, outside, interior[0, 2], target_13)
        recentered_23 = sigma_23 - _exterior_shift(sigma_23, outside, interior[1, 2], target_23)
        sigma_13 = xp.where(outside, recentered_13, interior[0, 2])
        sigma_23 = xp.where(outside, recentered_23, interior[1, 2])

        stress = xp.zeros((3, 3) + self.grid.shape, dtype=sigma_13.dtype)
        stress[0, 2] = stress[2, 0] = sigma_13
        stress[1, 2] = stress[2, 1] = sigma_23
        return stress

    def eigenstrain_field(self, eigenstrain_tensor):
        """Real-space eigenstrain field for
        :meth:`periodic_prescribed_eigenstrain_solution`'s numeric
        counterpart -- see :meth:`HoleInPlateCase.eigenstrain_field`,
        identical in spirit but over the elliptical region.
        """
        eigenstrain_tensor = xp.asarray(eigenstrain_tensor, dtype=self.grid.real_dtype)
        indicator = xp.where(self.elliptical_radius < 1.0, 1.0, 0.0).astype(
            self.grid.real_dtype
        )
        return eigenstrain_tensor[:, :, None, None, None] * indicator[None, None, ...]

    def periodic_prescribed_eigenstrain_solution(self, eigenstrain_tensor):
        r"""Exact analytic periodic solution for a uniform eigenstrain
        `eigenstrain_tensor` prescribed directly within the elliptical
        region -- the elliptical counterpart of
        :meth:`HoleInPlateCase.periodic_prescribed_eigenstrain_solution`,
        identical in spirit (a homogeneous matrix with a stress-free
        transformation strain in the region, zero remote strain, no
        stiffness contrast) but using
        :func:`_ellipse_fourier_transform` in place of
        :func:`_disk_fourier_transform` for the region's shape -- since
        this is a pure eigenstrain problem, no elliptical Eshelby tensor
        is needed even here, unlike the finite-contrast elliptical hole
        cases elsewhere in this class.

        Returns
        -------
        strain, stress : ndarray
            Shape ``(3, 3) + grid.shape``.
        """
        eigenstrain_tensor = xp.asarray(eigenstrain_tensor)
        solver = self.solver()
        green = solver.reference_green
        operator = solver.operator

        sigma_star = _isotropic_stress_apply(
            eigenstrain_tensor, self.matrix_lame_lambda, self.matrix_lame_mu
        )
        shape_hat = _ellipse_fourier_transform(
            self.grid, self.semi_axis_a, self.semi_axis_b,
            center=(self.center[0], self.center[1], 0.0),
        )
        eps0_hat = eigenstrain_tensor[:, :, None, None, None] * shape_hat[None, None, ...]
        sigma_star_hat = sigma_star[:, :, None, None, None] * shape_hat[None, None, ...]
        source_hat = -operator.div(sigma_star_hat)

        eps_hat = green.field(source_hat)

        difference = eps_hat - eps0_hat
        trace = xp.einsum("ii...->...", difference)
        delta = xp.eye(3, dtype=difference.dtype).reshape((3, 3, 1, 1, 1))
        sigma_hat = self.matrix_lame_lambda * trace[None, None, ...] * delta + (
            2.0 * self.matrix_lame_mu * difference
        )
        strain = self.grid.ifft(eps_hat)
        stress = self.grid.ifft(sigma_hat)
        return strain, stress

    def remote_stress_gradient(self, load, magnitude):
        """Return the ``(3, 3, 3)`` remote stress *gradient* tensor
        (component, component, direction) for `load` -- the elliptical
        counterpart of :meth:`HoleInPlateCase.remote_stress_gradient`.

        Parameters
        ----------
        load : {"moment"}
            ``d(sigma_11)/dx2 = magnitude``, same convention as
            :meth:`HoleInPlateCase.remote_stress_gradient`.
        magnitude : float
        """
        if load != "moment":
            raise ValueError('load must be "moment"')
        gradient = xp.zeros((3, 3, 3))
        gradient[0, 0, 1] = magnitude
        return gradient

    def macro_strain_gradient(self, load, magnitude, solver=None):
        """Return the macroscopic strain gradient reproducing
        `remote_stress_gradient` far from the hole -- see
        :meth:`HoleInPlateCase.macro_strain_gradient`."""
        solver = solver if solver is not None else self.solver()
        sigma_gradient = self.remote_stress_gradient(load, magnitude)
        return solver.reference_green.apply_compliance(sigma_gradient)

    def periodic_gradient_inhomogeneity_stress(self, magnitude, n_images=2):
        r"""Periodic-array stress field for the elliptical inhomogeneity
        under the "moment" remote stress gradient `magnitude`, built by
        summing the exact isolated finite-`contrast` solution
        (:func:`_ellipse_gradient_inhomogeneity_solution`) over periodic
        images -- no FFT and no numerically probed Eshelby tensor (an
        earlier dipole-eigenstrain construction of that kind was removed:
        it was noisy in aspect ratio and disagreed with the numeric
        solver), so it carries no grid-discretization error of its own.
        The elliptical counterpart
        of :meth:`HoleInPlateCase.periodic_gradient_stress`, extended to
        finite `contrast`.

        Unlike the uniform loads (:meth:`periodic_inhomogeneity_stress`),
        this needs neither recentering nor a periodic-calibrated
        eigenstrain: a gradient load's perturbation is dipolar, so the
        periodic-environment correction is higher order in the area
        fraction. Checked against a converged CG solve (256^2, contrast
        1e-3 and 3, area fraction 0.8% to 12%): near-hole error 0.2-1.4% of
        the peak stress with no growth in area fraction (the CG solver's
        own diffuse-edge floor), domain-mean sigma_yy and sigma_xy within
        1e-7 of ``magnitude * b``, and the dominant interior slope
        ``d sigma_xx / dy`` within 1.4%.

        Parameters
        ----------
        magnitude : float
        n_images : int, default=2
            Sum periodic images in ``[-n_images, n_images]`` along each
            in-plane axis.

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape``; only the in-plane
            ``sigma_xx``, ``sigma_yy``, ``sigma_xy`` are populated.
        """
        return _periodic_gradient_inhomogeneity_stress(
            self.grid, self.center, self.semi_axis_a, self.semi_axis_b, self.contrast,
            self.matrix_lame_lambda, self.matrix_lame_mu, magnitude, n_images,
        )
