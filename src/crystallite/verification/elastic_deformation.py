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
    raise ValueError('load must be "tension", "compression", or "shear"')


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

        A sharp cutoff (`smoothing_width=0`, the default) introduces
        Gibbs-ringing in the spectral stress field right at the boundary;
        `smoothing_width` blends the two values with a tanh profile over
        that many length units, the same "diffuse rather than sharp"
        treatment used for every other interface in this project.
        """
        r = self.radius
        if self.smoothing_width > 0:
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
        load : {"tension", "compression", "shear"}
        magnitude : float
            ``sigma_inf`` (tension/compression) or ``tau_inf`` (shear).
        """
        sigma = xp.zeros((3, 3))
        if load == "tension":
            sigma[0, 0] = magnitude
        elif load == "compression":
            sigma[0, 0] = -magnitude
        elif load == "shear":
            sigma[0, 1] = sigma[1, 0] = magnitude
        else:
            raise ValueError('load must be "tension", "compression", or "shear"')
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
        raise ValueError('load must be "tension", "compression", or "shear"')

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
