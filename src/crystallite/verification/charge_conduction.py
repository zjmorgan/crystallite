r"""Verification cases for steady charge conduction (Ohm's law): a hole,
soft, hard or perfectly conducting inclusion (circular or elliptical,
isotropic or anisotropic matrix) and a crack.

Ohm's law is :math:`j_i=\sigma_{ij}E_j` with the electric field
:math:`E=-\nabla\phi` (electrical conductivity :math:`\sigma`, current density
:math:`j`, potential :math:`\phi`) and conservation of charge
:math:`\nabla\cdot j=0` in the steady state, so there is no source (charge cannot
be generated). The mathematics, equivalent inclusion and all references are those
of :mod:`crystallite.verification.conduction`, which this module names in
charge-conduction terms (the heat-conduction counterpart is
:mod:`crystallite.verification.heat_conduction`). A static charge distribution is
electrostatics, :mod:`crystallite.verification.electrostatics`.
"""

from __future__ import annotations

from dataclasses import dataclass

from crystallite.verification.conduction import (
    ConductionInclusionCase,
    elliptical_perturbation,
)

__all__ = ["ChargeInclusionCase", "elliptical_perturbation"]


@dataclass
class ChargeInclusionCase(ConductionInclusionCase):
    """An ellipse (a circle for ``semi_axis_a == semi_axis_b``, a crack when
    thin) of electrical conductivity `contrast` times the matrix's
    (`matrix_conductivity`, the electrical conductivity :math:`\\sigma_0`). See
    :class:`~crystallite.verification.conduction.ConductionInclusionCase`;
    ``solver()`` returns the
    :class:`~crystallite.conduction.SteadyConduction` (its ``driving_force`` is
    :math:`E`, its ``flux`` the current density :math:`j`)."""

    def periodic_interior_electric_field(self, electric_field, n_modes=512):
        """Interior :math:`E` ``(3,)`` of the periodic array under the mean
        electric field."""
        return self.periodic_interior_driving_force(electric_field, n_modes)

    def periodic_interior_current_density(self, electric_field, n_modes=512):
        """Interior current density ``(3,)``; the finite limit for a perfect
        conductor."""
        return self.periodic_interior_flux(electric_field, n_modes)

    def periodic_electric_field(self, electric_field, n_images=8, n_modes=512):
        """Electric field ``(3,) + grid.shape`` of the periodic array."""
        return self.periodic_exterior_field(electric_field, n_images, n_modes)

    def periodic_current_density(self, electric_field, n_images=8, n_modes=512):
        """Current density ``(3,) + grid.shape`` of the periodic array."""
        return self.periodic_flux_field(electric_field, n_images, n_modes)
