Grain orientation driver
========================

The driver template at
``examples/grain_orientation/template.py`` shows the intended high-level
workflow for a multi-order-parameter Allen-Cahn grain-growth simulation:

1. Construct a grid and seed ``n_orientations`` order-parameter fields with
   small-amplitude noise (the same idea as
   ``examples/mass_diffusion/template.py``'s spinodal decomposition).
2. Advance semi-implicit
   :class:`crystallite.grain_orientation.GrainOrientation` steps -- the
   cross-coupling term makes whichever order parameter is locally largest
   at a point grow and suppress the others there, so a polycrystalline
   structure self-organizes directly from noise.
3. Track the dominant orientation (by magnitude, since each order
   parameter's double well has minima at both ``+eta0`` and ``-eta0``) and
   the total free energy, which decreases monotonically as grains coarsen.
4. Plot the resulting grain map at several snapshots.

See ``examples/grain_orientation/verification/`` for the accompanying
verification plots comparing this solver against the analytic planar and
circular-grain equilibria.

.. literalinclude:: ../../examples/grain_orientation/template.py
   :language: python
   :linenos:
