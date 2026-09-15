"""Verification case for heterogeneous elasticity: the Kirsch problem."""

from dataclasses import dataclass

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
    applied to that same (closed-form, uniform) interior stress.
    """
    nu = matrix_lame_lambda / (2.0 * (matrix_lame_lambda + matrix_lame_mu))

    def tension_eigenstrain(signed_magnitude):
        sigma_xx, sigma_yy, sigma_zz = _inhomogeneity_interior_stress(
            signed_magnitude, contrast, nu
        )
        sigma_in = xp.array(
            [[sigma_xx, 0.0, 0.0], [0.0, sigma_yy, 0.0], [0.0, 0.0, sigma_zz]]
        )
        if contrast == float("inf"):
            # A rigid inclusion has zero strain under any finite stress --
            # the eps_in/S0:sigma_in limit directly (substituting a
            # literal inf into the compliance below hits inf/inf).
            true_strain = xp.zeros((3, 3))
        else:
            true_strain = _isotropic_compliance_apply(
                sigma_in, contrast * matrix_lame_lambda, contrast * matrix_lame_mu
            )
        matrix_strain = _isotropic_compliance_apply(
            sigma_in, matrix_lame_lambda, matrix_lame_mu
        )
        return true_strain - matrix_strain

    if load == "tension":
        return tension_eigenstrain(magnitude)
    if load == "compression":
        return tension_eigenstrain(-magnitude)
    if load == "shear":
        quarter_pi = xp.pi / 4.0
        eps_plus = tension_eigenstrain(magnitude)
        eps_minus = tension_eigenstrain(-magnitude)
        return _rotate_tensor_2d(eps_plus, quarter_pi) + _rotate_tensor_2d(
            eps_minus, -quarter_pi
        )
    if load == "biaxial":
        eps_x = tension_eigenstrain(magnitude)
        return eps_x + _rotate_tensor_2d(eps_x, xp.pi / 2.0)
    raise ValueError('load must be "tension", "compression", "shear", or "biaxial"')


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
    lam1, mu1 = contrast * lam0, contrast * mu0
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
    discriminant = xp.sqrt(z**2 - 4.0 * semi_r**2 * m + 0j)
    zeta_plus = (z + discriminant) / (2.0 * semi_r)
    zeta_minus = (z - discriminant) / (2.0 * semi_r)
    zeta = xp.where(xp.abs(zeta_plus) >= xp.abs(zeta_minus), zeta_plus, zeta_minus)
    # Only exactly zero at the circular special case (m=0) exactly at the
    # ellipse center (z=0) -- always outside this function's domain of
    # validity (it describes the exterior field only), but guarded here
    # anyway to avoid a noisy divide-by-zero warning on every call.
    zeta = xp.where(zeta == 0, 1.0 + 0j, zeta)

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
        sigma_r_theta)`` at polar position ``(r, theta)`` (center-relative).

        ``load="shear"`` is obtained by superposing two rotated tension
        solutions (magnitude ``+/-magnitude`` at +/-45 degrees) -- the
        standard construction, since pure shear is biaxial tension and
        compression of equal magnitude at 45 degrees to the shear axes.
        ``load="biaxial"`` superposes two tension solutions at 0 and 90
        degrees instead (equal tension along both axes).
        """
        if load == "tension":
            return _tension_polar_stress(r, theta, self.hole_radius, magnitude)
        if load == "compression":
            return _tension_polar_stress(r, theta, self.hole_radius, -magnitude)
        if load == "shear":
            quarter_pi = xp.pi / 4.0
            rr1, tt1, rt1 = _tension_polar_stress(
                r, theta - quarter_pi, self.hole_radius, magnitude
            )
            rr2, tt2, rt2 = _tension_polar_stress(
                r, theta + quarter_pi, self.hole_radius, -magnitude
            )
            return rr1 + rr2, tt1 + tt2, rt1 + rt2
        if load == "biaxial":
            rr1, tt1, rt1 = _tension_polar_stress(r, theta, self.hole_radius, magnitude)
            rr2, tt2, rt2 = _tension_polar_stress(
                r, theta - xp.pi / 2.0, self.hole_radius, magnitude
            )
            return rr1 + rr2, tt1 + tt2, rt1 + rt2
        raise ValueError('load must be "tension", "compression", "shear", or "biaxial"')

    def hoop_stress_at_hole(self, theta, load, magnitude):
        """Analytic hoop stress ``sigma_theta_theta`` at ``r=hole_radius``."""
        _, sigma_theta_theta, _ = self.analytic_stress(
            self.hole_radius, theta, load, magnitude
        )
        return sigma_theta_theta

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
        :meth:`periodic_void_stress` to image-sum the isolated solution
        as an approximate, Gibbs-ringing-free alternative to
        :meth:`periodic_analytic_solution`'s Eshelby-eigenstrain FFT
        construction -- see that method's docstring for why it is only
        approximate, not exact, unlike the "moment" case.

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

    def periodic_void_stress(self, load, magnitude, n_images=1):
        r"""Periodic-array stress field for a traction-free void under
        uniform remote `load`/`magnitude`, built by summing the exact
        isolated closed form (:meth:`analytic_stress`) over periodic
        images -- the uniform-load counterpart of
        :meth:`periodic_gradient_stress`, and an image-sum alternative to
        :meth:`periodic_analytic_solution` for the near-void `contrast`
        this class defaults to.

        Ignores `contrast` entirely: :meth:`analytic_stress` is the exact
        ``contrast=0`` void solution regardless of what `contrast` this
        case's diffuse numerical boundary actually uses. Valid as a
        reference only when `contrast` is small enough that the numerical
        hole is itself a good void approximation (as
        ``hole_in_plate.py``'s ``1e-3`` is) -- for a genuine
        finite-contrast inhomogeneity, use
        :meth:`periodic_analytic_solution` instead (see
        ``cylindrical_inclusion.py``).

        The raw image sum's own domain mean is not exactly `magnitude`
        (each hole's presence measurably perturbs its neighbors'
        effective loading -- a real periodic self-interaction, not a
        bug), so, outside every hole, this method recenters it back to
        `magnitude` by construction: ``field -= mean(field) - magnitude``,
        applied only where :attr:`radius` :math:`\ge` `hole_radius`
        (left untouched inside, where the true stress is exactly zero).
        This is not a self-consistent local-field solve -- it is a direct
        substitution, exactly the recipe this project's own original
        (pre-crystallite) reference implementation used
        (``eshelby.cpp``'s ``sigma[i][j] -= s0_-s0xt`` after averaging
        over the whole domain, `s0xt` its remote-stress target) and
        validated against genuine finite-difference simulation results
        (the original ``hooke.tex`` presentation's own
        ``cylindrical.hole.pdf``/``elliptical.hole.pdf`` figures): dashed
        (this construction) and solid (numeric) curves there agree
        closely everywhere, including the *periodicity-elevated*
        far-field level away from a dilute limit (~1.1x the nominal
        remote stress at their hole-radius/domain ratio, not 1.0) --
        both curves track that elevation, not just the near-hole peak.

        That original comparison was against a spectral heterogeneous-
        stiffness solver driven by a prescribed remote stress. This
        project's own :class:`crystallite.elastic_deformation.ElasticDeformation`
        is driven by a prescribed macro *strain* (:meth:`macro_strain`)
        instead -- a genuinely different boundary condition for a cell
        with a real hole in it (the softer composite carries less mean
        stress for the same mean strain), not equivalent even in
        principle, so a residual few-percent-to-double-digit gap against
        *this* solver's own numeric output (as opposed to against the
        original stress-controlled reference) is expected and is not
        evidence of an error in this construction.

        Converges quickly in `n_images` (a handful of images already
        stabilizes the domain mean to 4+ significant figures at this
        module's own example geometry). `load="moment"`
        (:meth:`periodic_gradient_stress`) needs no such recentering:
        that background's leading-order (uniform) equivalent-eigenstrain
        response is exactly zero by symmetry, so its own domain mean is
        already exact.

        Parameters
        ----------
        load : {"tension", "compression", "shear", "biaxial"}
        magnitude : float
        n_images : int, default=1
            Sum periodic images in ``[-n_images, n_images]`` along each
            in-plane axis -- checked directly (against `n_images` up to
            6) to already agree with the converged answer to 4+
            significant figures at ``n_images=1``, since the recentering
            step above carries most of the periodicity correction, not
            the raw sum over distant images.

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

        sigma_xx = background[0, 0] * xp.ones(self.grid.shape)
        sigma_yy = background[1, 1] * xp.ones(self.grid.shape)
        sigma_xy = background[0, 1] * xp.ones(self.grid.shape)

        for n1 in range(-n_images, n_images + 1):
            for n2 in range(-n_images, n_images + 1):
                dx = x - (self.center[0] + n1 * length_x)
                dy = y - (self.center[1] + n2 * length_y)
                sxx, syy, sxy = self._void_correction_cartesian(dx, dy, load, magnitude)
                sigma_xx = sigma_xx + sxx
                sigma_yy = sigma_yy + syy
                sigma_xy = sigma_xy + sxy

        outside = self.radius >= self.hole_radius
        sigma_xx = xp.where(outside, sigma_xx - (xp.mean(sigma_xx) - background[0, 0]), sigma_xx)
        sigma_yy = xp.where(outside, sigma_yy - (xp.mean(sigma_yy) - background[1, 1]), sigma_yy)
        sigma_xy = xp.where(outside, sigma_xy - (xp.mean(sigma_xy) - background[0, 1]), sigma_xy)

        stress = xp.zeros((3, 3) + self.grid.shape, dtype=sigma_xx.dtype)
        stress[0, 0] = sigma_xx
        stress[1, 1] = sigma_yy
        stress[0, 1] = stress[1, 0] = sigma_xy
        return stress

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

        for n1 in range(-n_images, n_images + 1):
            for n2 in range(-n_images, n_images + 1):
                dx = x - (self.center[0] + n1 * length_x)
                dy = y - (self.center[1] + n2 * length_y)
                sxx, syy, sxy = self._void_correction_cartesian(dx, dy, load, magnitude)
                sigma_xx = sigma_xx + sxx
                sigma_yy = sigma_yy + syy
                sigma_xy = sigma_xy + sxy

        outside = self.elliptical_radius >= 1.0
        sigma_xx = xp.where(outside, sigma_xx - (xp.mean(sigma_xx) - background[0, 0]), sigma_xx)
        sigma_yy = xp.where(outside, sigma_yy - (xp.mean(sigma_yy) - background[1, 1]), sigma_yy)
        sigma_xy = xp.where(outside, sigma_xy - (xp.mean(sigma_xy) - background[0, 1]), sigma_xy)

        stress = xp.zeros((3, 3) + self.grid.shape, dtype=sigma_xx.dtype)
        stress[0, 0] = sigma_xx
        stress[1, 1] = sigma_yy
        stress[0, 1] = stress[1, 0] = sigma_xy
        return stress

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

    def _gradient_offset(self):
        """``x2 - center[1]``, shape ``grid.shape`` -- see
        :meth:`HoleInPlateCase._gradient_offset`."""
        return xp.broadcast_to(self.grid.x[1] - self.center[1], self.grid.shape)

    def eigenstrain_gradient_field(self, eigenstrain_gradient_tensor):
        """Real-space dipole eigenstrain field over the elliptical region
        -- see :meth:`HoleInPlateCase.eigenstrain_gradient_field`,
        identical in spirit but using `elliptical_radius` in place of a
        Euclidean radius."""
        eigenstrain_gradient_tensor = xp.asarray(
            eigenstrain_gradient_tensor, dtype=self.grid.real_dtype
        )
        indicator = xp.where(self.elliptical_radius < 1.0, 1.0, 0.0).astype(
            self.grid.real_dtype
        )
        shape = indicator * self._gradient_offset().astype(self.grid.real_dtype)
        return eigenstrain_gradient_tensor[:, :, None, None, None] * shape[None, None, ...]

    def periodic_prescribed_gradient_eigenstrain_solution(self, eigenstrain_gradient_tensor):
        """Exact periodic solution for a linear (dipole) eigenstrain
        prescribed within the elliptical region, zero remote/mean field --
        see :meth:`HoleInPlateCase.periodic_prescribed_gradient_eigenstrain_solution`,
        identical in spirit (including its direct numerical FFT of the
        real-space field rather than a closed-form transform) but over
        the elliptical region.

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
        indicator = xp.where(self.elliptical_radius < 1.0, 1.0, 0.0).astype(
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
        """Numerically probed periodic dipole (gradient-order) Eshelby
        tensor for this case's actual ellipse/domain ratio -- see
        :func:`_periodic_eshelby_dipole_tensor` and
        :meth:`HoleInPlateCase.periodic_eshelby_dipole_tensor`."""
        return _periodic_eshelby_dipole_tensor(
            self.periodic_prescribed_gradient_eigenstrain_solution,
            self.elliptical_radius < 1.0,
            self._gradient_offset(),
        )

    def periodic_gradient_eigenstrain_stress(self, magnitude):
        r"""Periodic-array stress field for a finite-contrast elliptical
        inhomogeneity under the "moment" remote stress gradient `magnitude`
        -- the elliptical counterpart of
        :meth:`HoleInPlateCase.periodic_gradient_eigenstrain_stress`; see
        that method's docstring for the construction (an Eshelby dipole
        eigenstrain, calibrated by :meth:`periodic_eshelby_dipole_tensor`,
        with the linear background added back by hand since
        `reference_green.field` only accepts a uniform one).

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
        indicator = xp.where(self.elliptical_radius < 1.0, 1.0, 0.0).astype(
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
