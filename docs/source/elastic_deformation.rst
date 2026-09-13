Elastic deformation driver
==========================

The driver template at
``examples/elastic_deformation/template.py`` shows the intended high-level
workflow for a static heterogeneous-elasticity solve:

1. Build a soft circular inclusion approximating a hole
   (:class:`crystallite.verification.HoleInPlateCase`), which constructs
   the spatially varying Lame parameter fields and the
   :class:`crystallite.elastic_deformation.ElasticDeformation` solver for
   them.
2. Convert a target remote stress into the macroscopic strain the solver
   takes as input, via ``solver.reference_green.apply_compliance`` (exact
   here since the reference medium is the actual matrix material).
3. Solve for the equilibrium displacement fluctuation with a matrix-free,
   preconditioned conjugate-gradient iteration -- no eigenstrain, no
   phase-field coupling, just Hooke's law for a heterogeneous stiffness.
4. Plot the resulting stress field.

See ``examples/elastic_deformation/verification/`` for the accompanying
comparison against the closed-form Kirsch solution for a hole in an
infinite plate, under remote tension, compression, and pure shear.

.. literalinclude:: ../../examples/elastic_deformation/template.py
   :language: python
   :linenos:
