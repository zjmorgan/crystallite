"""Coherent (elastically constrained) diffusion: composition-dependent misfit
strain coupled to mechanical equilibrium."""

from dataclasses import dataclass

from crystallite.backend import xp
from crystallite.elastic_deformation import ElasticDeformation
from crystallite.spectral.long_range import GreenOperator


@dataclass(frozen=True)
class CoherentDiffusion:
    r"""Cahn-Hilliard diffusion with a coherent elastic misfit (Cahn-Larche).

    The free energy is the usual bulk-plus-gradient energy plus the elastic
    energy of a body whose stress-free (Vegard) strain follows the composition,

    .. math::

        E_\mathrm{el}[c]=\min_u\tfrac12\int(\varepsilon-\varepsilon^*):C:
        (\varepsilon-\varepsilon^*),\qquad
        \varepsilon^*_{ij}(c)=\eta_{ij}\,(c-c_\mathrm{ref}),

    with :math:`\varepsilon=\bar\varepsilon+\mathrm{sym}\nabla u` at
    mechanical equilibrium. By the envelope theorem the elastic contribution to
    the chemical potential is :math:`\mu_\mathrm{el}=\delta E_\mathrm{el}/
    \delta c=-\eta_{ij}\sigma_{ij}`, with :math:`\sigma=C:(\varepsilon-
    \varepsilon^*)` the equilibrium stress, and each step is: solve
    :class:`crystallite.elastic_deformation.ElasticDeformation` for the current
    composition, then advance
    :class:`crystallite.mass_diffusion.MassDiffusion` with
    :math:`\mu_\mathrm{el}` as its ``extra_chemical_potential``.

    For a homogeneous stiffness this reduces to Khachaturyan's :math:`E_\mathrm{el}
    =\tfrac12\sum_k B(\hat n)|\hat c_k|^2`, with :math:`B(\hat n)` the
    elastic energy per unit composition squared of a modulation along
    :math:`\hat n`, so the morphology is set by the misfit and the elastic
    anisotropy alone (the modulation lies along the elastically soft
    directions).

    Parameters
    ----------
    diffusion : MassDiffusion
    elasticity : ElasticDeformation
        On the same grid. Give it the stiffness (a ``stiffness=`` tensor or
        Lame parameters); a homogeneous one converges in about one iteration.
    misfit_strain : array_like of shape (3, 3) or (3, 3) + grid.shape
        :math:`\eta_{ij}`, the eigenstrain per unit composition: a constant
        tensor, or a field (e.g. a crystal-frame misfit rotated grain by grain
        with :meth:`crystallite.microstructure.Microstructure.property_field`).
    reference_composition : float, default=0
        Composition with zero eigenstrain. Only sets a uniform eigenstrain,
        which under a fixed macroscopic strain changes the stress by a
        constant and so drives no flux.
    macro_strain : array_like of shape (3, 3), optional
        Fixed macroscopic strain of the cell. Default zero (a clamped cell).
    dealias : bool, default=True
        Lanczos-filter the eigenstrain (band-limit it), which removes the
        Nyquist content on which the discrete elastic operator differs from
        its reference Green operator, and the potential (self-adjointly, so the
        coupling stays variational). Each mode's elastic energy is scaled by
        the square of the filter gain :math:`\prod_i\mathrm{sinc}(k_i\Delta x_i)`,
        about :math:`1-(k\Delta x)^2/3`: negligible for resolved modes.
    """

    diffusion: object
    elasticity: object
    misfit_strain: object
    reference_composition: float = 0.0
    macro_strain: object = None
    dealias: bool = True

    def __post_init__(self):
        if self.diffusion.grid is not self.elasticity.grid and (
            self.diffusion.grid.shape != self.elasticity.grid.shape
        ):
            raise ValueError("diffusion and elasticity must share a grid")
        misfit = xp.asarray(self.misfit_strain, dtype=self.diffusion.grid.real_dtype)
        spatial = tuple(self.diffusion.grid.shape)
        if misfit.shape not in ((3, 3), (3, 3) + spatial):
            raise ValueError(
                f"misfit_strain must have shape (3, 3) or (3, 3) + grid.shape, got {misfit.shape}"
            )
        if misfit.ndim == 2:
            misfit = misfit[:, :, None, None, None]
        object.__setattr__(self, "misfit_strain", misfit)
        macro = (
            xp.zeros((3, 3)) if self.macro_strain is None else xp.asarray(self.macro_strain)
        )
        object.__setattr__(self, "macro_strain", macro)

    @classmethod
    def polycrystal(
        cls, diffusion, microstructure, stiffness, misfit_strain, reference_composition=0.0,
        macro_strain=None, dealias=True, **elasticity_kwargs
    ):
        """Coherent diffusion in a polycrystal.

        The crystal-frame `stiffness` ``(3, 3, 3, 3)`` and `misfit_strain`
        ``(3, 3)`` are rotated grain by grain into the sample frame
        (:meth:`crystallite.microstructure.Microstructure.property_field`), and
        the elastic solver, now heterogeneous, is referenced to the Voigt-average
        stiffness of the grains (its Green's operator is the CG preconditioner).
        The grain structure is fixed; the phases share one stiffness, so the
        misfit alone drives the decomposition.

        Parameters
        ----------
        diffusion : MassDiffusion
        microstructure : Microstructure
            On the same grid.
        stiffness, misfit_strain : array_like
            In the crystal frame.
        elasticity_kwargs
            Passed to :class:`~crystallite.elastic_deformation.ElasticDeformation`
            (e.g. ``preconditioner_green``).
        """
        grid = diffusion.grid
        if microstructure.grid.shape != grid.shape:
            raise ValueError("microstructure and diffusion must share a grid shape")
        elasticity_kwargs.setdefault(
            "reference_green",
            GreenOperator(
                grid, xp.asarray(microstructure.average_property(stiffness), dtype=grid.real_dtype)
            ),
        )
        elasticity = ElasticDeformation(
            grid, 1.0, 1.0, 1.0, 1.0, stiffness=microstructure.property_field(stiffness),
            **elasticity_kwargs,
        )
        return cls(
            diffusion, elasticity, microstructure.property_field(misfit_strain),
            reference_composition=reference_composition, macro_strain=macro_strain, dealias=dealias,
        )

    @property
    def grid(self):
        return self.diffusion.grid

    def eigenstrain(self, composition):
        """``eps*(c) = eta (c - c_ref)``, shape ``(3, 3) + grid.shape``."""
        grid = self.grid
        composition = xp.asarray(composition, dtype=grid.real_dtype)
        field = self.misfit_strain * (composition - self.reference_composition)[None, None]
        if self.dealias:
            field = xp.real(grid.ifft(grid.fft(field) * grid.lanczos_filter))
        return field

    def solve_elastic(self, composition, displacement=None, **solve_kwargs):
        """Mechanical equilibrium at `composition`; returns the
        :class:`~crystallite.elastic_deformation.ElasticSolution`.
        `displacement` warm-starts the solve (e.g. the previous step's)."""
        return self.elasticity.solve(
            self.macro_strain,
            eigenstrain=self.eigenstrain(composition),
            initial_displacement=displacement,
            **solve_kwargs,
        )

    def elastic_potential(self, solution):
        r""":math:`\mu_\mathrm{el}=-\eta_{ij}\sigma_{ij}`, shape ``grid.shape``,
        from an equilibrium `solution` (Lanczos-filtered like the eigenstrain
        with `dealias`: the filter is self-adjoint, so this is exactly
        :math:`\delta E_\mathrm{el}/\delta c` of the filtered problem)."""
        potential = -xp.sum(self.misfit_strain * solution.stress, axis=(0, 1))
        if self.dealias:
            grid = self.grid
            potential = xp.real(grid.ifft(grid.fft(potential) * grid.lanczos_filter))
        return potential

    def elastic_energy(self, composition, solution):
        """Mean elastic energy density ``1/2 sigma:(eps - eps*)``."""
        elastic_strain = solution.strain - self.eigenstrain(composition)
        return 0.5 * xp.mean(xp.sum(solution.stress * elastic_strain, axis=(0, 1)))

    def step(self, composition, time_step, displacement=None, **solve_kwargs):
        """One coupled step. Returns ``(composition, solution)`` with the
        equilibrium solution of the *old* composition, whose ``displacement``
        warm-starts the next step."""
        composition = xp.asarray(composition, dtype=self.grid.real_dtype)
        solution = self.solve_elastic(composition, displacement, **solve_kwargs)
        new = self.diffusion.step(
            composition, time_step,
            extra_chemical_potential=self.elastic_potential(solution),
        )
        return new, solution
