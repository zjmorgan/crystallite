crystallite.verification namespace
==================================

The verification namespace contains small, reproducible verification cases
for each physical model. These cases define initial conditions and
observables (and, where a closed-form solution exists, the analytic
comparison); numerical time integration or equilibrium solving belongs to
the corresponding solver.

Mass diffusion cases
---------------------

.. automodule:: crystallite.verification.mass_diffusion
   :members:
   :show-inheritance:
   :undoc-members:

Grain orientation cases
-------------------------

.. automodule:: crystallite.verification.grain_orientation
   :members:
   :show-inheritance:
   :undoc-members:

Elastic deformation cases
----------------------------

.. automodule:: crystallite.verification.elastic_deformation
   :members:
   :show-inheritance:
   :undoc-members:

Conduction cases
----------------------------

The shared mathematics of steady conduction (equivalent inclusion,
depolarization tensors, image sums) that heat and charge conduction both use.

.. automodule:: crystallite.verification.conduction
   :members:
   :show-inheritance:
   :undoc-members:

Heat conduction cases
----------------------------

.. automodule:: crystallite.verification.heat_conduction
   :members:
   :show-inheritance:
   :undoc-members:

Charge conduction cases
----------------------------

.. automodule:: crystallite.verification.charge_conduction
   :members:
   :show-inheritance:
   :undoc-members:

Electrostatics cases
----------------------------

.. automodule:: crystallite.verification.electrostatics
   :members:
   :show-inheritance:
   :undoc-members:

Magnetostatics cases
----------------------------

.. automodule:: crystallite.verification.magnetostatics
   :members:
   :show-inheritance:
   :undoc-members:
