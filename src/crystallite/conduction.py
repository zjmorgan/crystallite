"""Steady-state heterogeneous conduction (Fourier's and Ohm's laws) via a
matrix-free, preconditioned-conjugate-gradient spectral solver."""

from dataclasses import dataclass

from crystallite.backend import xp
from crystallite.spectral.krylov import preconditioned_conjugate_gradient
from crystallite.spectral.long_range import GreenOperator
from crystallite.spectral.short_range import DifferentialOperators


@dataclass(frozen=True)
class ConductionSolution:
    """Result of :meth:`SteadyConduction.solve`.

    Attributes
    ----------
    potential : ndarray
        Periodic potential fluctuation ``T*`` (zero mean), shape
        ``grid.shape``.
    driving_force : ndarray
        Total driving force ``X = X_bar - grad(T*)``, shape
        ``(3,) + grid.shape``.
    flux : ndarray
        ``q = kappa(x) . X``, shape ``(3,) + grid.shape``.
    residual_norm : float
        Final relative Krylov residual, ``||b - A(T*)|| / ||b||`` (0 if the
        load produces no net source, e.g. a homogeneous medium).
    iterations : int
        Number of conjugate-gradient iterations taken.
    converged : bool
        Whether ``residual_norm`` reached ``tol``.
    """

    potential: object
    driving_force: object
    flux: object
    residual_norm: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class SteadyConduction:
    r"""Steady conduction in a heterogeneous solid with an optional source.

    One solver for both linear-transport laws, which differ only in
    vocabulary:

    ==============  ====================  ==================
    quantity        heat (Fourier)        charge (Ohm)
    ==============  ====================  ==================
    potential       temperature :math:`T` electric :math:`\phi`
    driving force   :math:`X=-\nabla T`    :math:`E=-\nabla\phi`
    conductivity    :math:`\kappa`        :math:`\sigma`
    flux            heat flux :math:`q`   current density :math:`j`
    source          heat generation       (none: charge is conserved)
    ==============  ====================  ==================

    Solves :math:`\nabla\cdot q = \varphi` with :math:`q = \kappa(x)\,X`,
    :math:`X = \bar X - \nabla T^*` for a periodic, zero-mean potential
    fluctuation :math:`T^*` under a prescribed mean driving force
    :math:`\bar X` and source :math:`\varphi`. :math:`\kappa` may be a
    scalar, a ``grid.shape`` field, a ``(3, 3)`` tensor, or a
    ``(3, 3) + grid.shape`` tensor field.

    Substituting into conservation gives a matrix-free SPD system

    .. math::

        A(T^*) = b, \qquad A(T) = -\nabla\cdot(\kappa\nabla T), \qquad
        b = \varphi - \nabla\cdot(\kappa\bar X),

    solved by preconditioned CG. As in
    :class:`crystallite.elastic_deformation.ElasticDeformation`, the
    preconditioner is the Green's operator of a homogeneous *reference*
    medium :math:`\kappa_0`, ``G(k) = 1/(k . kappa_0 . k)``. Its k=0 output
    is zero, and so is that of every right-hand side (a divergence has no
    k=0 mode; the source's mean is dropped), so the null direction
    (a uniform potential) never enters the iteration.

    A uniform source has no periodic steady state (heat has nowhere to
    go), so `source`'s mean is dropped, as if a uniform sink balanced it.
    Give a *localized* source and read the field away from the domain
    mean accordingly.

    Parameters
    ----------
    grid : Grid
    conductivity : float or array_like
        See above. Real-valued.
    reference_conductivity : float or array_like of shape (3, 3), optional
        Homogeneous reference medium for the preconditioner. Defaults to
        `conductivity` when that is homogeneous; required otherwise.
    operator : object, optional
        Defaults to :class:`DifferentialOperators(grid)`.
    reference_green : object, optional
        Defaults to ``GreenOperator(grid, reference_conductivity)``. Public
        so callers can convert a far-field flux to a driving force with
        ``reference_green.apply_compliance(q_inf)``.
    preconditioner_green : object, optional
        The Green's operator CG actually preconditions with. Defaults to
        `reference_green`; changes only the convergence rate.
    """

    grid: object
    conductivity: object
    reference_conductivity: object = None
    operator: object = None
    reference_green: object = None
    preconditioner_green: object = None

    def __post_init__(self):
        spatial_ndim = len(self.grid.shape)
        conductivity = xp.asarray(self.conductivity, dtype=self.grid.real_dtype)
        if conductivity.ndim in (0, spatial_ndim):
            if conductivity.ndim and conductivity.shape != self.grid.shape:
                raise ValueError(
                    "an isotropic conductivity field must have shape grid.shape, "
                    f"got {conductivity.shape}"
                )
        elif conductivity.shape[:2] != (3, 3) or conductivity.ndim not in (
            2, 2 + spatial_ndim
        ):
            raise ValueError(
                "conductivity must be a scalar, grid.shape, (3, 3) or "
                f"(3, 3) + grid.shape, got {conductivity.shape}"
            )
        elif conductivity.ndim > 2 and conductivity.shape[2:] != self.grid.shape:
            raise ValueError(
                f"a conductivity tensor field must be (3, 3) + grid.shape, got {conductivity.shape}"
            )
        object.__setattr__(self, "conductivity", conductivity)

        if self.operator is None:
            object.__setattr__(self, "operator", DifferentialOperators(self.grid))

        if self.reference_green is None:
            reference = self.reference_conductivity
            if reference is None:
                if conductivity.ndim not in (0, 2):
                    raise ValueError(
                        "a heterogeneous conductivity needs an explicit "
                        "reference_conductivity or reference_green"
                    )
                reference = conductivity
            reference = xp.asarray(reference, dtype=self.grid.real_dtype)
            if reference.ndim == 0:
                reference = reference * xp.eye(3, dtype=self.grid.real_dtype)
            if reference.shape != (3, 3):
                raise ValueError(
                    f"reference_conductivity must be a scalar or (3, 3), got {reference.shape}"
                )
            object.__setattr__(self, "reference_green", GreenOperator(self.grid, reference))
        if self.preconditioner_green is None:
            object.__setattr__(self, "preconditioner_green", self.reference_green)

    def flux(self, driving_force):
        """Return the pointwise flux ``kappa(x) . driving_force``.

        Parameters
        ----------
        driving_force : array_like
            Shape ``(3,) + grid.shape`` (or broadcastable, e.g. a bare
            ``(3,)`` constant).

        Returns
        -------
        ndarray
            Shape ``(3,) + grid.shape``.
        """
        driving_force = xp.asarray(driving_force)
        spatial_ndim = len(self.grid.shape)
        if driving_force.ndim == 1:
            driving_force = driving_force.reshape((3,) + (1,) * spatial_ndim)
        kappa = self.conductivity
        if kappa.ndim in (0, spatial_ndim):
            q = kappa * driving_force
        else:
            if kappa.ndim == 2:
                kappa = kappa.reshape((3, 3) + (1,) * spatial_ndim)
            q = xp.einsum("ij...,j...->i...", kappa, driving_force)
        return xp.broadcast_to(q, (3,) + self.grid.shape)

    def _divergence(self, vector):
        return self.grid.ifft(self.operator.div(self.grid.fft(vector)))

    def _gradient(self, scalar):
        return self.grid.ifft(self.operator.grad(self.grid.fft(scalar)))

    def _matvec(self, potential):
        return -self._divergence(self.flux(self._gradient(potential)))

    def _precondition(self, residual):
        return self.grid.ifft(self.preconditioner_green.potential(self.grid.fft(residual)))

    def _right_hand_side(self, macro_field, source):
        b = -self._divergence(self.flux(macro_field))
        if source is not None:
            source = xp.asarray(source, dtype=self.grid.real_dtype)
            if source.shape != self.grid.shape:
                raise ValueError(
                    f"source must have shape grid.shape={self.grid.shape}, got {source.shape}"
                )
            # A has no k=0 response, so a nonzero-mean source makes A(T)=b
            # inconsistent and CG diverges; drop the mean.
            b = b + (source - xp.mean(source))
        return b

    def solve(
        self,
        macro_field,
        source=None,
        tol=1.0e-6,
        max_iterations=500,
        initial_potential=None,
    ):
        """Solve for the steady potential fluctuation.

        Parameters
        ----------
        macro_field : array_like
            Prescribed mean driving force :math:`\\bar X`, shape ``(3,)``
            (``X = -grad T`` averaged over the cell).
        source : array_like, optional
            Heat generation density :math:`\\varphi`, shape ``grid.shape``
            (positive generates). Its mean is dropped, see the class
            docstring.
        tol : float, default=1e-6
            Relative Krylov residual at which to stop. The default reflects
            `Grid`'s float32 arithmetic, not a limitation of the method.
        max_iterations : int, default=500
        initial_potential : ndarray, optional
            Warm start, shape ``grid.shape``. Defaults to zero.

        Returns
        -------
        ConductionSolution
        """
        macro_field = xp.asarray(macro_field, dtype=self.grid.real_dtype)
        if macro_field.shape != (3,):
            raise ValueError(f"macro_field must have shape (3,), got {macro_field.shape}")
        b = self._right_hand_side(macro_field, source)

        if initial_potential is None:
            potential = xp.zeros(self.grid.shape, dtype=self.grid.real_dtype)
        else:
            potential = xp.asarray(initial_potential, dtype=self.grid.real_dtype)
            if potential.shape != self.grid.shape:
                raise ValueError(
                    f"initial_potential must have shape grid.shape={self.grid.shape}, "
                    f"got {potential.shape}"
                )

        potential, residual_norm, iterations, converged = preconditioned_conjugate_gradient(
            self._matvec, self._precondition, b, potential, tol=tol,
            max_iterations=max_iterations,
            # No near-null sector to guard against (unlike the elastic
            # pseudo-2D out-of-plane DOF); at high contrast the residual
            # legitimately blips by >100x before recovering.
            divergence_factor=None,
        )

        driving_force = xp.broadcast_to(
            macro_field.reshape((3,) + (1,) * len(self.grid.shape)) - self._gradient(potential),
            (3,) + self.grid.shape,
        )
        return ConductionSolution(
            potential=potential,
            driving_force=driving_force,
            flux=self.flux(driving_force),
            residual_norm=residual_norm,
            iterations=iterations,
            converged=converged,
        )
