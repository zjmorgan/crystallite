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
    """
    beta = contrast
    kappa = 3.0 - 4.0 * nu
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
    ``eps_zz=0``.
    """
    beta = contrast
    kappa = 3.0 - 4.0 * nu
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
    # Small relative tolerance rather than a bare >=, for the same reason
    # as _ellipse_hole_correction_cartesian's outside check: a point
    # meant to sit exactly on the boundary can round to just inside it
    # after a center-add/center-subtract round trip.
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
    # ellipse center (z=0) -- always outside the domain of interest
    # (masked out downstream by _ellipse_hole_correction_cartesian's
    # "outside" check), but guarded here anyway to avoid a noisy
    # divide-by-zero warning on every call.
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


def _ellipse_hole_correction_cartesian(x, y, semi_axis_a, semi_axis_b, load, magnitude):
    """The pure hole-induced *correction* to the uniform remote background
    for `load` (``"tension"``: ``sigma_xx=magnitude``; ``"biaxial"``:
    ``sigma_xx=sigma_yy=magnitude``; ``"shear"``: ``sigma_xy=magnitude``),
    i.e. :func:`_ellipse_void_cartesian_stress` with that (exactly,
    pointwise coincident far-field) background subtracted back out --
    decays away from the hole, so it can be summed over periodic images of
    the hole to build the periodic solution (see
    :meth:`EllipticalHoleInPlateCase.periodic_analytic_stress`) without
    double counting the shared background, the same pattern as
    :func:`_gradient_void_hole_correction_cartesian`.
    """
    sigma_xx, sigma_yy, sigma_xy = _ellipse_void_cartesian_stress(
        x, y, semi_axis_a, semi_axis_b, load, magnitude
    )
    if load == "tension":
        sigma_xx = sigma_xx - magnitude
    elif load == "biaxial":
        sigma_xx = sigma_xx - magnitude
        sigma_yy = sigma_yy - magnitude
    elif load == "shear":
        sigma_xy = sigma_xy - magnitude
    else:
        raise ValueError('load must be "tension", "biaxial", or "shear"')
    # A small relative tolerance, not a bare >= 1.0: a point built as
    # center + a*cos(theta) (as EllipticalHoleInPlateCase.boundary_stress
    # does, to query exactly the boundary) and then re-expressed relative
    # to that same center, as every image here is, loses a couple of
    # ULPs in the round trip -- confirmed directly, e.g. rho landing at
    # 0.9999999999999993 for a point that is exactly on the boundary by
    # construction. A bare >= 1.0 misclassifies that as "inside" and
    # zeros a nonzero boundary hoop stress -- caught because it flipped
    # sign-of-error unpredictably across otherwise-symmetric aspect
    # ratios in a verification sweep, not from a theoretical worry.
    outside = (x / semi_axis_a) ** 2 + (y / semi_axis_b) ** 2 >= 1.0 - 1.0e-9
    # Zero inside the void, matching this module's other isolated closed
    # forms (_analytic_line, _gradient_void_hole_correction_cartesian): a
    # true traction-free hole carries no stress.
    return (
        xp.where(outside, sigma_xx, 0.0),
        xp.where(outside, sigma_yy, 0.0),
        xp.where(outside, sigma_xy, 0.0),
    )


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

    def periodic_analytic_solution(self, load, magnitude):
        r"""Exact analytic solution for the *periodic* array problem this
        grid actually represents -- not the isolated-hole approximation
        :meth:`analytic_stress` gives.

        Replaces the true finite-contrast inhomogeneity with an Eshelby
        equivalent eigenstrain (:func:`_equivalent_eigenstrain`) over a
        matrix-material disk, then solves the resulting periodic
        (Khachaturyan-Shatalov) eigenstrain problem exactly in Fourier
        space, reusing :meth:`ElasticDeformation.reference_green` directly
        (a homogeneous-medium body-force solve is exactly what a fixed
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

        eps_star = _equivalent_eigenstrain(
            magnitude, self.contrast, self.matrix_lame_lambda, self.matrix_lame_mu, load
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


@dataclass(frozen=True)
class EllipticalHoleInPlateCase:
    r"""An elliptical hole (soft inclusion) in a periodic 2D isotropic
    plate, with the analytic Kirsch/Inglis stress field
    (:func:`_ellipse_void_cartesian_stress`) for validation -- the
    elliptical-geometry counterpart of :class:`HoleInPlateCase`.

    Unlike :class:`HoleInPlateCase`, the periodic reference solution here
    (:meth:`periodic_analytic_stress`) is *not* built from an Eshelby
    equivalent eigenstrain: the general elliptical Eshelby tensor is a
    much larger derivation this project does not have implemented (see
    the reference material's ``stress.hole.sage.py``, worked out only for
    the general ellipsoidal-inhomogeneity *interior*-stress problem, not
    transcribed here). Instead it sums the exact isolated closed form
    over periodic images of the hole -- exact per image, approximate only
    in how many periodic neighbors are included, and converging quickly
    since each image's correction decays as the cube of the ellipse-to-
    spacing size ratio (see :func:`_ellipse_hole_correction_cartesian`).

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

    def _periodic_stress_at(self, x, y, load, magnitude, n_images):
        """(sigma_xx, sigma_yy, sigma_xy) -- background plus every
        image's own correction, summed at arbitrary points `x`, `y` (same
        shape as each other, any shape) -- no "zero inside the central
        void" masking applied (see :meth:`periodic_analytic_stress`,
        which adds that for its grid use case). Shared by that method and
        :meth:`boundary_stress`, which needs the point-wise, unmasked
        value exactly at the boundary -- interpolating the *masked* grid
        field there is unstable, landing right on that mask's
        discontinuity depending on which side neighboring grid points
        happen to fall.
        """
        length_x, length_y = self.grid.lengths[0], self.grid.lengths[1]

        background = self.remote_stress(load, magnitude)
        shape = xp.broadcast_shapes(xp.shape(x), xp.shape(y))
        sigma_xx = background[0, 0] * xp.ones(shape)
        sigma_yy = background[1, 1] * xp.ones(shape)
        sigma_xy = background[0, 1] * xp.ones(shape)

        for n1 in range(-n_images, n_images + 1):
            for n2 in range(-n_images, n_images + 1):
                dx = x - (self.center[0] + n1 * length_x)
                dy = y - (self.center[1] + n2 * length_y)
                sxx, syy, sxy = _ellipse_hole_correction_cartesian(
                    dx, dy, self.semi_axis_a, self.semi_axis_b, load, magnitude
                )
                sigma_xx = sigma_xx + sxx
                sigma_yy = sigma_yy + syy
                sigma_xy = sigma_xy + sxy
        return sigma_xx, sigma_yy, sigma_xy

    def boundary_stress(self, load, magnitude, theta, n_images=3):
        r"""Periodic-array Cartesian stress
        (``sigma_xx``, ``sigma_yy``, ``sigma_xy``) exactly at the hole
        boundary point :math:`(a\cos\theta, b\sin\theta)` (`theta` the
        ellipse's own parametric angle, not the polar angle) -- well
        defined there (no singularity: traction-free means the
        radial/shear components vanish at the boundary, not the hoop
        stress), and the natural way to query "the stress at the edge"
        pointwise, without the grid-interpolation instability
        :meth:`periodic_analytic_stress` has right at its "zero inside the
        void" mask boundary (see :meth:`_periodic_stress_at`).

        Parameters
        ----------
        load : {"tension", "biaxial", "shear"}
        magnitude : float
        theta : float or array_like
            Elliptical parametric angle(s); ``theta=pi/2`` is the tip of
            the `semi_axis_b` semi-axis, ``theta=0`` the tip of
            `semi_axis_a`.
        n_images : int, default=3

        Returns
        -------
        sigma_xx, sigma_yy, sigma_xy : float or ndarray
        """
        theta = xp.asarray(theta)
        x = self.center[0] + self.semi_axis_a * xp.cos(theta)
        y = self.center[1] + self.semi_axis_b * xp.sin(theta)
        return self._periodic_stress_at(x, y, load, magnitude, n_images)

    def periodic_analytic_stress(self, load, magnitude, n_images=3):
        """Periodic-array stress field for the hole under remote `load`
        (see :meth:`remote_stress`), built by summing
        :func:`_ellipse_hole_correction_cartesian` over periodic images of
        the hole -- see this class's own docstring for why (no elliptical
        Eshelby tensor implemented here).

        Parameters
        ----------
        load : {"tension", "biaxial", "shear"}
        magnitude : float
        n_images : int, default=3
            Sum periodic images in ``[-n_images, n_images]`` along each
            in-plane axis.

        Returns
        -------
        stress : ndarray
            Shape ``(3, 3) + grid.shape`` -- only the in-plane
            ``sigma_xx``, ``sigma_yy``, ``sigma_xy`` components are
            populated, as with :meth:`HoleInPlateCase.periodic_gradient_stress`.
        """
        x, y = self.grid.x[0], self.grid.x[1]
        sigma_xx, sigma_yy, sigma_xy = self._periodic_stress_at(x, y, load, magnitude, n_images)

        # Zero inside the *central* void: the sum above is background plus
        # every image's own (individually zeroed-inside-its-own-ellipse)
        # correction, so a point inside this hole still picks up small
        # leftover contributions from neighboring images' correction
        # fields -- well-defined but not physically meaningful for a true
        # void, and it makes the interior look like a spurious plateau
        # rather than the flat zero every other isolated closed form in
        # this module uses there.
        dx0, dy0 = x - self.center[0], y - self.center[1]
        outside_center = (
            (dx0 / self.semi_axis_a) ** 2 + (dy0 / self.semi_axis_b) ** 2 >= 1.0 - 1.0e-9
        )
        sigma_xx = xp.where(outside_center, sigma_xx, 0.0)
        sigma_yy = xp.where(outside_center, sigma_yy, 0.0)
        sigma_xy = xp.where(outside_center, sigma_xy, 0.0)

        stress = xp.zeros((3, 3) + self.grid.shape, dtype=sigma_xx.dtype)
        stress[0, 0] = sigma_xx
        stress[1, 1] = sigma_yy
        stress[0, 1] = stress[1, 0] = sigma_xy
        return stress

    def periodic_pressurized_stress(self, magnitude, n_images=3):
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
        stress = self.periodic_analytic_stress("biaxial", magnitude, n_images=n_images)
        delta = xp.eye(3, dtype=stress.dtype).reshape((3, 3, 1, 1, 1))
        return stress - magnitude * delta
