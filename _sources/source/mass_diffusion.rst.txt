Mass diffusion driver
=====================

The driver template at
``examples/mass_diffusion_template.py`` shows the intended high-level
workflow for a nonlinear mass-diffusion simulation:

1. Define parent and product phases with tensor-valued properties.
2. Construct a grid and initial composition/phase-fraction state.
3. Mix phase properties into effective mobility and gradient-energy fields.
4. Advance several solver steps per reporting sweep.
5. Record energy and optionally save field snapshots.
6. Plot the final microstructure separately from the numerical run.

The template is deliberately not runnable yet. Its ``TODO`` markers identify
the missing production APIs rather than silently substituting the current
linear verification solver.

.. literalinclude:: ../../examples/mass_diffusion_template.py
   :language: python
   :linenos: