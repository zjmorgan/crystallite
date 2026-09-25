Coherent diffusion driver
=========================

:class:`crystallite.chemomechanics.CoherentDiffusion` couples
:class:`crystallite.mass_diffusion.MassDiffusion` to
:class:`crystallite.elastic_deformation.ElasticDeformation` through a
composition-dependent misfit (Vegard) strain, the Cahn-Larche model of a coherent
transformation. Each step solves mechanical equilibrium for the eigenstrain
:math:`\varepsilon^*=\eta\,(c-c_\mathrm{ref})`, forms the elastic chemical
potential :math:`\mu_\mathrm{el}=-\eta_{ij}\sigma_{ij}` (the functional
derivative of the elastic energy), and advances the composition with it as
``MassDiffusion.step(..., extra_chemical_potential=...)``.

With a homogeneous cubic stiffness and a dilatational misfit, the misfit alone
selects the morphology: modulations align with the elastically soft direction,
:math:`\langle100\rangle` for a Zener ratio above 1, :math:`\langle111\rangle`
below it, and none for an isotropic crystal. The coupling is verified against
Khachaturyan's elastic energy of a composition modulation (the potential of a
mode equals :math:`B(\hat n)c_k` to the float32 precision of the solver, and is
the exact functional derivative of the elastic energy).

**Polycrystals.**
:meth:`crystallite.chemomechanics.CoherentDiffusion.polycrystal` takes a
:class:`crystallite.microstructure.Microstructure` and a crystal-frame stiffness
and misfit, rotates both grain by grain, and references the now heterogeneous
elastic solve to the Voigt-average stiffness of the grains. The decomposition
then follows each grain's own soft direction, so the modulations are aligned with
that grain's crystal axes (Zener ratio above 1) or diagonals (below 1) and change
direction across the grain boundaries. The stiffness varies between grains, so
each step is a conjugate-gradient elastic solve (warm-started), about 10 iterations
here, where the homogeneous crystal needs one. The grain structure is fixed.
:func:`crystallite.verification.chemomechanics.interface_orientation` measures
the interface orientation of a grain's interior, which the tests and
``examples/chemomechanics/verification/polycrystal_alignment.py`` compare with the
prediction from the grain's orientation alone.

See ``examples/chemomechanics/`` for the misfit-driven spinodal decomposition
(``template.py``), the polycrystal (``polycrystal.py``), the elastic stiffness of a
modulation against its direction (``verification/elastic_anisotropy.py``) and the
per-grain alignment (``verification/polycrystal_alignment.py``).
