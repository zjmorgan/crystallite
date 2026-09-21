Steady conduction driver
========================

:class:`crystallite.conduction.SteadyConduction` solves steady heat
(Fourier) or charge (Ohm) conduction in a heterogeneous, possibly
anisotropic, solid, with an optional heat source. The two laws are one
problem: potential :math:`T` or :math:`\phi`, driving force
:math:`X=-\nabla T` or :math:`E=-\nabla\phi`, flux :math:`q=\kappa X` or
:math:`j=\sigma E`, and conservation :math:`\nabla\cdot q=\varphi` with
:math:`\varphi=0` for charge.

1. Build the conductivity field and the solver, e.g. with
   :class:`crystallite.verification.conduction.ConductionInclusionCase`
   (hole, soft, hard or perfectly conducting inclusion).
2. Convert a far-field flux to the mean driving force the solver takes as
   input with ``solver.reference_green.apply_compliance``.
3. Solve for the potential fluctuation by matrix-free preconditioned
   conjugate gradient, with the homogeneous reference medium's Green's
   operator as preconditioner.

Antiplane (Mode III) elasticity is this same problem with
:math:`\kappa\leftrightarrow\mu`; ``tests/test_conduction.py`` checks the two
solvers against each other.

Heat and charge conduction are verified separately, so each figure carries
its own quantities: ``examples/heat_conduction/verification/`` (heat flux
:math:`q`, thermal conductivity :math:`\\kappa`, thermal driving force
:math:`X`, and a cylinder generating heat) and
``examples/charge_conduction/verification/`` (current density :math:`j`,
electrical conductivity :math:`\\sigma`, electric field :math:`E`). Both compare
with the exact periodic lattice-sum and image-sum references.
