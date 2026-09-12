Mass diffusion driver
=====================

The driver template at
``examples/mass_diffusion/template.py`` shows the intended high-level
workflow for a nonlinear mass-diffusion simulation:

1. Define parent and product phases with symmetry-resolved tensor
   properties (:class:`crystallite.material.properties.Solid`).
2. Construct a grid and a small-amplitude initial composition field.
3. Mix phase properties into composition-dependent mobility and
   gradient-energy callables
   (:class:`crystallite.mobility.BinaryMobility`,
   :class:`crystallite.mass_diffusion.GradientEnergy`).
4. Advance semi-implicit :class:`crystallite.mass_diffusion.MassDiffusion`
   steps, clipping the field back to its physical composition range each
   step.
5. Plot the final microstructure.

See ``examples/mass_diffusion/verification/`` for the accompanying
verification plots comparing this solver against analytic mass-diffusion
solutions.

.. literalinclude:: ../../examples/mass_diffusion/template.py
   :language: python
   :linenos:
