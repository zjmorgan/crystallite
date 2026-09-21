Electrostatics driver
=====================

:class:`crystallite.electrostatics.Electrostatics` solves Gauss's law in a
homogeneous medium from free charge and polarization,
:math:`\nabla\cdot(\varepsilon E+P)=\rho` with :math:`E=E_\infty-\nabla\phi`,
directly in Fourier space (no iteration). A polarization acts through its bound
charge :math:`-\nabla\cdot P`, which is what a ferroelectric solver needs.

Magnetostatics is the same problem: with :math:`\varepsilon=1`, no free charge and
the magnetization as `polarization`, ``electric_field`` is :math:`H` and
``displacement`` is :math:`B/\mu_0`. It is exposed with magnetic vocabulary as
:class:`crystallite.magnetostatics.Magnetostatics`, and verified separately in
``examples/magnetostatics/verification/``.

A heterogeneous dielectric is steady conduction,
:class:`crystallite.conduction.SteadyConduction` with the permittivity as the
conductivity, the charge as the source, and the flux as :math:`D`;
``tests/test_electrostatics.py`` checks the two against each other.

See ``examples/electrostatics/verification/`` for the uniformly charged and
uniformly polarized cylinders against their exact references.
