"""Static heterogeneous linear elasticity (Hooke's law) via a matrix-free,
preconditioned-conjugate-gradient spectral solver."""

from dataclasses import dataclass

from crystallite.backend import xp
from crystallite.material.properties import isotropic_stiffness
from crystallite.spectral.long_range import GreenOperator
from crystallite.spectral.krylov import preconditioned_conjugate_gradient
from crystallite.spectral.short_range import DifferentialOperators


@dataclass(frozen=True)
class ElasticSolution:
    """Result of :meth:`ElasticDeformation.solve`.

    Attributes
    ----------
    displacement : ndarray
        Periodic displacement fluctuation ``u*``, shape ``(3,) + grid.shape``.
    strain : ndarray
        Total strain ``macro_strain + sym(grad(u*))``, shape
        ``(3, 3) + grid.shape``.
    stress : ndarray
        Stress ``C(x):strain``, shape ``(3, 3) + grid.shape``.
    residual_norm : float
        Final relative Krylov residual, ``||b - A(u*)|| / ||b||`` (0 if the
        applied load produces no net force, e.g. a homogeneous material).
    iterations : int
        Number of conjugate-gradient iterations taken.
    converged : bool
        Whether ``residual_norm`` reached ``tol``.
    """

    displacement: object
    strain: object
    stress: object
    residual_norm: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class ElasticDeformation:
    r"""Static mechanical equilibrium in a heterogeneous isotropic solid.

    Solves periodic equilibrium :math:`\nabla\cdot\sigma(x) + f(x) = 0` with
    :math:`\sigma(x) = C(x):\varepsilon(x)`,
    :math:`\varepsilon(x) = \bar\varepsilon + \mathrm{sym}(\nabla u^*(x))`
    for a periodic displacement fluctuation :math:`u^*` (zero mean) under a
    prescribed macroscopic strain :math:`\bar\varepsilon`, and :math:`f(x)`
    an optional applied body force (see :meth:`solve`'s `body_force`
    parameter; zero unless given). The material is isotropic but may vary
    in space -- :math:`\lambda(x)`, :math:`\mu(x)` are each a scalar or a
    ``grid.shape`` field, applied pointwise via
    :math:`\sigma_{ij} = \lambda\,\mathrm{tr}(\varepsilon)\delta_{ij} +
    2\mu\varepsilon_{ij}` -- e.g. a soft circular inclusion approximating a
    hole. This is deliberately *not* coupled to any phase-field driving
    force; it solves plain Hooke's law in isolation, optionally with a
    prescribed eigenstrain source (see :meth:`solve`'s `eigenstrain`
    parameter) but nothing that feeds back from an evolving field.

    Substituting the strain decomposition into equilibrium gives a
    matrix-free linear system for :math:`u^*`,

    .. math::

        A(u^*) = b, \qquad
        A(u) = -\nabla\cdot\!\left[C(x):\mathrm{sym}(\nabla u)\right],
        \qquad
        b = -\nabla\cdot\!\left[C(x):\bar\varepsilon\right],

    solved by preconditioned conjugate gradient. :math:`A` is symmetric
    (:class:`crystallite.spectral.short_range.DifferentialOperators`'s
    ``grad``/``div`` are exact discrete adjoints of each other by
    construction: ``grad`` inserts :math:`ik`, ``div`` contracts the same
    index with :math:`ik`) and positive semi-definite, with a spatially
    uniform displacement as its only null direction. That null direction
    never enters the iteration: every right-hand side and every
    preconditioned residual below has an identically zero k=0 Fourier
    component (a divergence's k=0 mode vanishes on any periodic grid, and
    :meth:`GreenOperator.potential` already zeroes its own k=0 output), so
    no explicit projection step is needed.

    The preconditioner is the existing
    :class:`crystallite.spectral.long_range.GreenOperator` for a homogeneous
    *reference* medium :math:`C_0` (Lame parameters `reference_lame_lambda`,
    `reference_lame_mu`, e.g. the matrix phase's own values): applying
    :meth:`GreenOperator.potential` to a residual is exactly solving the
    reference medium's own equilibrium problem for that residual as a body
    force, i.e. exactly the classic Khachaturyan-Shatalov/Moulinec-Suquet
    reference-medium correction, reused here as a linear preconditioner
    rather than iterated to convergence on its own (the "basic scheme",
    whose convergence rate degrades badly for a high-contrast/void
    inclusion -- CG on top of the same reference-medium solve does not).
    `reference_green` is public so callers can reuse it directly, e.g.
    ``solver.reference_green.apply_compliance(sigma_target)`` to convert a
    target remote stress into the macroscopic strain this solver takes as
    input.

    `preconditioner_green` (optional, distinct from `reference_green`) is
    the Green's operator CG actually preconditions with in
    :meth:`_precondition`. It defaults to `reference_green` (the original,
    single-reference-medium behavior), but a caller with an extreme,
    one-directional stiffness contrast (e.g. a small stiff/rigid inclusion
    in a much softer matrix) can pass a *different* homogeneous medium here
    -- e.g. the geometric mean of the two phases' stiffnesses, the standard
    heuristic for balancing a Lippmann-Schwinger-type scheme's convergence
    across both phases (Moulinec & Suquet 1998; Eyre & Milton 1999) --
    without disturbing `reference_lame_lambda`/`reference_lame_mu`/
    `reference_green` themselves, which other call sites rely on for exact
    results (e.g. :meth:`crystallite.verification.HoleInPlateCase.macro_strain`
    computing the macro strain that reproduces a target remote stress
    *in the true matrix material* -- that identity would break if the
    "reference" fed to it silently became some fictitious blend). CG's
    correctness (the value it converges to) never depends on which SPD
    preconditioner is used, only how fast it gets there, so this is purely
    a convergence-rate knob, safe to change independently.

    Parameters
    ----------
    grid : Grid
    lame_lambda, lame_mu : float or array_like
        Lame parameters of the (possibly heterogeneous) material. Each a
        scalar or a ``grid.shape`` real field.
    reference_lame_lambda, reference_lame_mu : float
        Lame parameters of the homogeneous reference medium used to build
        `reference_green` (see :meth:`_body_force`'s
        `macro_strain_gradient` handling and
        :meth:`crystallite.verification.HoleInPlateCase.macro_strain` for
        why this must be the *true* matrix material, not a preconditioning
        convenience).
    operator : object, optional
        Defaults to :class:`DifferentialOperators(grid)`.
    reference_green : object, optional
        Defaults to ``GreenOperator(grid, isotropic_stiffness(
        reference_lame_lambda, reference_lame_mu))``.
    preconditioner_green : object, optional
        The Green's operator :meth:`_precondition` actually uses. Defaults
        to `reference_green`.
    stiffness : array_like, optional
        A general (anisotropic) rank-4 stiffness ``C_ijkl``: shape
        ``(3, 3, 3, 3)`` (homogeneous) or ``(3, 3, 3, 3) + grid.shape``
        (heterogeneous). When given it replaces the isotropic
        `lame_lambda`/`lame_mu` law in :meth:`stress` (those two are then
        only placeholders, kept for the isotropic call signature). A
        homogeneous `stiffness` also builds the default `reference_green`,
        which is then the exact Green's function of the true medium (so a
        homogeneous problem, e.g. an eigenstrain in a uniform anisotropic
        matrix, converges in one iteration); a heterogeneous one needs an
        explicit `reference_green`. Not supported together with
        ``macro_strain_gradient`` (whose background treatment assumes an
        isotropic reference).
    """

    grid: object
    lame_lambda: object
    lame_mu: object
    reference_lame_lambda: float
    reference_lame_mu: float
    operator: object = None
    reference_green: object = None
    preconditioner_green: object = None
    stiffness: object = None

    def __post_init__(self):
        if self.operator is None:
            object.__setattr__(self, "operator", DifferentialOperators(self.grid))
        if self.stiffness is not None:
            stiffness = xp.asarray(self.stiffness, dtype=self.grid.real_dtype)
            if stiffness.shape[:4] != (3, 3, 3, 3) or stiffness.ndim not in (
                4, 4 + len(self.grid.shape)
            ):
                raise ValueError(
                    "stiffness must have shape (3, 3, 3, 3) or (3, 3, 3, 3) + "
                    f"grid.shape, got {stiffness.shape}"
                )
            object.__setattr__(self, "stiffness", stiffness)
        if self.reference_green is None:
            if self.stiffness is not None and self.stiffness.ndim == 4:
                reference_stiffness = self.stiffness
            elif self.stiffness is not None:
                raise ValueError(
                    "a heterogeneous stiffness needs an explicit reference_green"
                )
            else:
                reference_stiffness = isotropic_stiffness(
                    self.reference_lame_lambda, self.reference_lame_mu
                )
            object.__setattr__(
                self, "reference_green", GreenOperator(self.grid, reference_stiffness)
            )
        if self.preconditioner_green is None:
            object.__setattr__(self, "preconditioner_green", self.reference_green)

    def _gradient_strain_field(self, macro_strain_gradient, origin):
        r"""Return the pure (offset-free) affine strain field
        :math:`\sum_m \mathrm{macro\_strain\_gradient}[\ldots, m]\,
        (x_m - \mathrm{origin}_m)`, shape ``(3, 3) + grid.shape``.

        Built directly in real space, not via FFT -- safe here (unlike a
        genuinely non-periodic field run through `grid.fft`) only because
        callers only ever contract it against :math:`\delta C(x) =
        C(x) - C_0` (:meth:`_body_force`) or leave it as a plain pointwise
        addition to the fluctuation strain (:meth:`solve`), never FFT it
        directly -- a linearly-growing real-space array is not periodic,
        so FFTing it raw would alias.
        """
        macro_strain_gradient = xp.asarray(macro_strain_gradient, dtype=self.grid.real_dtype)
        if macro_strain_gradient.shape != (3, 3, 3):
            raise ValueError(
                "macro_strain_gradient must have shape (3, 3, 3), got "
                f"{macro_strain_gradient.shape}"
            )
        origin = (0.0, 0.0, 0.0) if origin is None else origin
        field = xp.zeros((3, 3) + self.grid.shape, dtype=self.grid.real_dtype)
        for axis in range(3):
            field = field + macro_strain_gradient[:, :, axis].reshape(
                (3, 3) + (1,) * len(self.grid.shape)
            ) * (self.grid.x[axis] - origin[axis])
        return field

    def stress(self, strain):
        """Return the pointwise isotropic stress ``C(x):strain``.

        Parameters
        ----------
        strain : array_like
            Shape ``(3, 3) + grid.shape`` (or broadcastable to it, e.g. a
            bare ``(3, 3)`` constant strain).

        Returns
        -------
        ndarray
            Shape ``(3, 3) + grid.shape``.
        """
        strain = xp.asarray(strain)
        spatial_ndim = len(self.grid.shape)
        if self.stiffness is not None:
            stiffness = self.stiffness
            if stiffness.ndim == 4:
                stiffness = stiffness.reshape(stiffness.shape + (1,) * spatial_ndim)
            sigma = xp.einsum("ijkl...,kl...->ij...", stiffness, strain)
            return xp.broadcast_to(sigma, (3, 3) + self.grid.shape)
        trace = xp.einsum("ii...->...", strain)
        delta = xp.eye(3, dtype=strain.dtype).reshape((3, 3) + (1,) * spatial_ndim)
        sigma = self.lame_lambda * trace[None, None, ...] * delta + (
            2.0 * self.lame_mu * strain
        )
        return xp.broadcast_to(sigma, (3, 3) + self.grid.shape)

    def _strain_from_displacement(self, u):
        u_hat = self.grid.fft(u)
        grad_u_hat = self.operator.grad(u_hat)
        grad_u = self.grid.ifft(grad_u_hat)
        return 0.5 * (grad_u + xp.swapaxes(grad_u, 0, 1))

    def _divergence_of_stress(self, sigma):
        sigma_hat = self.grid.fft(sigma)
        divergence_hat = self.operator.div(sigma_hat)
        return self.grid.ifft(divergence_hat)

    def _matvec(self, u):
        strain = self._strain_from_displacement(u)
        sigma = self.stress(strain)
        return -self._divergence_of_stress(sigma)

    def _precondition(self, r):
        r_hat = self.grid.fft(r)
        u_hat = self.preconditioner_green.potential(r_hat)
        return self.grid.ifft(u_hat)

    def _body_force(
        self, macro_strain, macro_strain_gradient=None, gradient_origin=None, eigenstrain=None
    ):
        macro_strain = xp.asarray(macro_strain, dtype=self.grid.real_dtype)
        if macro_strain.shape != (3, 3):
            raise ValueError(f"macro_strain must have shape (3, 3), got {macro_strain.shape}")
        spatial_ndim = len(self.grid.shape)
        strain_field = macro_strain.reshape((3, 3) + (1,) * spatial_ndim)
        sigma_bar = self.stress(strain_field)
        # A(u*) = -div(C:eps*(u*)); equilibrium gives div(C:eps_bar) +
        # div(C:eps*(u*)) = 0, so A(u*) = +div(C:eps_bar): no leading minus.
        # (A previous version had one -- invisible far from any
        # heterogeneity since u*->0 there, but wrong right around one.)
        if macro_strain_gradient is not None:
            # eps_bar is constant, so C(x):eps_bar is periodic and safe to
            # FFT as `sigma_bar` already is. A growing affine field is not
            # periodic, so C0:eps_grad(x) must never be formed/FFT'd -- but
            # it's analytically divergence-free by construction (see
            # HoleInPlateCase.remote_stress_gradient/macro_strain_gradient),
            # so only the localized deltaC(x):eps_grad(x) term contributes,
            # and it alone (deltaC -> 0 away from any heterogeneity) is safe
            # to FFT.
            if self.stiffness is not None:
                raise NotImplementedError(
                    "macro_strain_gradient is not supported with a general stiffness"
                )
            eps_grad = self._gradient_strain_field(macro_strain_gradient, gradient_origin)
            trace = xp.einsum("ii...->...", eps_grad)
            delta = xp.eye(3, dtype=eps_grad.dtype).reshape((3, 3) + (1,) * spatial_ndim)
            delta_lambda = self.lame_lambda - self.reference_lame_lambda
            delta_mu = self.lame_mu - self.reference_lame_mu
            sigma_bar = sigma_bar + (
                delta_lambda * trace[None, None, ...] * delta + 2.0 * delta_mu * eps_grad
            )
        if eigenstrain is not None:
            # sigma(x) = C(x):(eps(x) - eigenstrain(x)); substituting into
            # div(sigma)=0 adds -div(C(x):eigenstrain(x)) to A(u)'s target
            # (same sign as the "no leading minus" note above). Safe to use
            # full C(x) and FFT directly -- unlike eps_grad above,
            # eigenstrain(x) is already a localized/periodic field.
            eigenstrain = xp.asarray(eigenstrain, dtype=self.grid.real_dtype)
            sigma_bar = sigma_bar - self.stress(eigenstrain)
        return self._divergence_of_stress(sigma_bar)

    def solve(
        self,
        macro_strain,
        tol=1.0e-6,
        max_iterations=500,
        initial_displacement=None,
        macro_strain_gradient=None,
        gradient_origin=None,
        eigenstrain=None,
        body_force=None,
    ):
        """Solve for the equilibrium displacement fluctuation.

        Parameters
        ----------
        macro_strain : array_like
            Prescribed macroscopic strain, shape ``(3, 3)``.
        tol : float, default=1e-6
            Relative Krylov residual at which to stop. The default
            reflects `Grid`'s float32 arithmetic, not a limitation of the
            method.
        max_iterations : int, default=500
        initial_displacement : ndarray, optional
            Warm-start displacement, shape ``(3,) + grid.shape``. Defaults
            to zero.
        macro_strain_gradient : array_like, optional
            Constant gradient of an additional affine background strain,
            shape ``(3, 3, 3)`` (strain component, direction) -- the total
            prescribed background becomes ``macro_strain +
            macro_strain_gradient[..., m] * (x_m - gradient_origin[m])``.
            Meant for a background that is itself an exact, divergence-free
            equilibrium field of the *reference* medium (e.g. built via
            ``reference_green.apply_compliance`` on a target stress
            gradient, as :class:`crystallite.verification.HoleInPlateCase`
            does for its ``"moment"`` load) -- see :meth:`_body_force` for
            why an arbitrary affine field would not be safe here.
        gradient_origin : array_like of shape (3,), optional
            Reference point for `macro_strain_gradient`. Defaults to the
            grid's own origin (all zero).
        eigenstrain : array_like, optional
            Prescribed stress-free transformation strain
            :math:`\\varepsilon^*(x)`, shape ``(3, 3) + grid.shape``,
            entering Hooke's law as :math:`\\sigma(x) =
            C(x):(\\varepsilon(x)-\\varepsilon^*(x))`. Meant for an
            already-localized/periodic field (e.g. nonzero only inside a
            disk) -- see :meth:`_body_force` for why that matters.
        body_force : array_like, optional
            Genuinely applied external force density :math:`f(x)`, shape
            ``(3,) + grid.shape``, entering equilibrium directly as
            :math:`\\nabla\\cdot\\sigma(x) + f(x) = 0` (unlike `eigenstrain`,
            which enters through Hooke's law instead). Adds straight onto
            this method's own right-hand side, with no extra divergence
            needed -- rearranging equilibrium with `f` present shows the
            two extra terms this and `eigenstrain` contribute enter with
            the same sign, one as :math:`-\\nabla\\cdot(C(x):\\varepsilon^*)`,
            the other as :math:`+f(x)` directly.

            A spatially uniform (nonzero-mean) `body_force` has no periodic
            solution at all (a periodic operator can never balance a net
            force) -- its own mean is silently dropped (zero contribution
            at the k=0 Fourier mode, the same way a uniform `eigenstrain`
            or an unmatched `macro_strain_gradient` would be), not an
            error, so only give this a *localized*, already-near-zero-mean
            force (e.g. an excess/deficit force confined to a disk) if the
            mean being discarded is not what you intended to model.

        Returns
        -------
        ElasticSolution
        """
        macro_strain = xp.asarray(macro_strain, dtype=self.grid.real_dtype)
        b = self._body_force(macro_strain, macro_strain_gradient, gradient_origin, eigenstrain)
        if body_force is not None:
            body_force = xp.asarray(body_force, dtype=self.grid.real_dtype)
            spatial_axes = tuple(range(1, body_force.ndim))
            # A has no k=0 response, so a nonzero-mean body_force makes
            # A(u)=b inconsistent and CG diverges rather than failing
            # gracefully. Drop the mean, as GreenOperator.field's own k=0
            # branch already does for an FFT-based solve.
            b = b + (body_force - xp.mean(body_force, axis=spatial_axes, keepdims=True))

        if initial_displacement is None:
            u = xp.zeros((3,) + self.grid.shape, dtype=self.grid.real_dtype)
        else:
            u = xp.asarray(initial_displacement, dtype=self.grid.real_dtype)
            if u.shape != (3,) + self.grid.shape:
                raise ValueError(
                    "initial_displacement must have shape (3,) + grid.shape: "
                    f"expected {(3,) + self.grid.shape}, got {u.shape}"
                )

        u, residual_norm, iterations, converged = preconditioned_conjugate_gradient(
            self._matvec, self._precondition, b, u, tol=tol, max_iterations=max_iterations
        )

        strain = macro_strain.reshape((3, 3) + (1,) * len(self.grid.shape)) + (
            self._strain_from_displacement(u)
        )
        if macro_strain_gradient is not None:
            strain = strain + self._gradient_strain_field(macro_strain_gradient, gradient_origin)
        strain = xp.broadcast_to(strain, (3, 3) + self.grid.shape)
        if eigenstrain is None:
            stress = self.stress(strain)
        else:
            stress = self.stress(strain - xp.asarray(eigenstrain, dtype=self.grid.real_dtype))

        return ElasticSolution(
            displacement=u,
            strain=strain,
            stress=stress,
            residual_norm=residual_norm,
            iterations=iterations,
            converged=converged,
        )
