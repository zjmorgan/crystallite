"""Static heterogeneous linear elasticity (Hooke's law) via a matrix-free,
preconditioned-conjugate-gradient spectral solver."""

from dataclasses import dataclass

from crystallite.backend import xp
from crystallite.material.properties import isotropic_stiffness
from crystallite.spectral.long_range import GreenOperator
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

    Solves periodic equilibrium :math:`\nabla\cdot\sigma(x) = 0` with
    :math:`\sigma(x) = C(x):\varepsilon(x)`,
    :math:`\varepsilon(x) = \bar\varepsilon + \mathrm{sym}(\nabla u^*(x))`
    for a periodic displacement fluctuation :math:`u^*` (zero mean) under a
    prescribed macroscopic strain :math:`\bar\varepsilon`. The material is
    isotropic but may vary in space -- :math:`\lambda(x)`, :math:`\mu(x)`
    are each a scalar or a ``grid.shape`` field, applied pointwise via
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
    """

    grid: object
    lame_lambda: object
    lame_mu: object
    reference_lame_lambda: float
    reference_lame_mu: float
    operator: object = None
    reference_green: object = None
    preconditioner_green: object = None

    def __post_init__(self):
        if self.operator is None:
            object.__setattr__(self, "operator", DifferentialOperators(self.grid))
        if self.reference_green is None:
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
        trace = xp.einsum("ii...->...", strain)
        spatial_ndim = len(self.grid.shape)
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
        # Equilibrium: div(C:eps_bar) + div(C:eps*(u*)) = 0, and A(u) is
        # defined as -div(C:eps*(u)), so A(u*) = +div(C:eps_bar) here --
        # no leading minus (a previous version of this had one, which is
        # invisible far from any heterogeneity, since u* -> 0 there
        # regardless of its sign, but flips the correction's sign exactly
        # where it matters, right around a heterogeneity).
        if macro_strain_gradient is not None:
            # The gradient part must be handled separately from the plain
            # C(x):eps_bar path above: eps_bar is spatially constant, so
            # C(x):eps_bar is trivially periodic (its C0 part is a
            # constant array, and only the localized deltaC(x):eps_bar
            # part varies), safe to FFT as `sigma_bar` already is. A
            # spatially *growing* affine field is not periodic, so
            # C0:eps_grad(x) must never be formed and FFT'd -- but its
            # analytic divergence is exactly zero everywhere by
            # construction (`macro_strain_gradient` comes from applying
            # the reference medium's own compliance to a target stress
            # gradient, so C0:eps_grad(x) reproduces that target stress
            # gradient pointwise, which is trivially divergence-free: see
            # HoleInPlateCase.remote_stress_gradient/macro_strain_gradient
            # for the "moment" case this exists for). So only the
            # localized deltaC(x):eps_grad(x) term contributes to the
            # body force, and it alone is safe to FFT (deltaC -> 0 away
            # from any heterogeneity, keeping the product periodic even
            # though eps_grad(x) itself is not).
            eps_grad = self._gradient_strain_field(macro_strain_gradient, gradient_origin)
            trace = xp.einsum("ii...->...", eps_grad)
            delta = xp.eye(3, dtype=eps_grad.dtype).reshape((3, 3) + (1,) * spatial_ndim)
            delta_lambda = self.lame_lambda - self.reference_lame_lambda
            delta_mu = self.lame_mu - self.reference_lame_mu
            sigma_bar = sigma_bar + (
                delta_lambda * trace[None, None, ...] * delta + 2.0 * delta_mu * eps_grad
            )
        if eigenstrain is not None:
            # Hooke's law with an eigenstrain is sigma(x) = C(x):(eps(x) -
            # eigenstrain(x)); substituting into div(sigma)=0 alongside
            # the macro_strain/macro_strain_gradient background above
            # adds a further -div(C(x):eigenstrain(x)) to A(u)'s target
            # (same sign derivation as the docstring's "no leading minus"
            # note, just with an extra term). Safe to use the full C(x)
            # (not just deltaC) and FFT directly, unlike the gradient
            # background above: eigenstrain(x) is itself already a
            # prescribed, localized/periodic real-space field (e.g.
            # nonzero only inside a disk), not an unbounded affine one.
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

        Returns
        -------
        ElasticSolution
        """
        macro_strain = xp.asarray(macro_strain, dtype=self.grid.real_dtype)
        b = self._body_force(macro_strain, macro_strain_gradient, gradient_origin, eigenstrain)

        if initial_displacement is None:
            u = xp.zeros((3,) + self.grid.shape, dtype=self.grid.real_dtype)
        else:
            u = xp.asarray(initial_displacement, dtype=self.grid.real_dtype)
            if u.shape != (3,) + self.grid.shape:
                raise ValueError(
                    "initial_displacement must have shape (3,) + grid.shape: "
                    f"expected {(3,) + self.grid.shape}, got {u.shape}"
                )

        b_norm = float(xp.sqrt(xp.sum(b * b)))
        r = b - self._matvec(u)

        iterations = 0
        if b_norm < 1.0e-12:
            # No net body force (e.g. a homogeneous material, or C(x)
            # already matching the reference medium): u=0 already solves
            # A(u)=b=0 exactly, to floating-point noise -- skip the
            # relative-residual check, which would divide by ~0.
            residual_norm = 0.0
            converged = True
        else:
            z = self._precondition(r)
            p = z
            rz_old = xp.sum(r * z)
            residual_norm = float(xp.sqrt(xp.sum(r * r))) / b_norm
            converged = residual_norm < tol

            while not converged and iterations < max_iterations:
                iterations += 1
                a_p = self._matvec(p)
                alpha = rz_old / xp.sum(p * a_p)
                u = u + alpha * p
                r = r - alpha * a_p
                residual_norm = float(xp.sqrt(xp.sum(r * r))) / b_norm
                if residual_norm < tol:
                    converged = True
                    break
                z = self._precondition(r)
                rz_new = xp.sum(r * z)
                beta = rz_new / rz_old
                p = z + beta * p
                rz_old = rz_new

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
