r"""Verification cases for steady heat conduction (Fourier's law): a hole, soft,
hard or perfectly conducting inclusion (circular or elliptical, isotropic or
anisotropic matrix), a crack, and a cylinder generating heat uniformly.

Fourier's law is :math:`q_i=\kappa_{ij}X_j` with the thermal driving force
:math:`X=-\nabla T` (thermal conductivity :math:`\kappa`, heat flux :math:`q`,
temperature :math:`T`, heat generation :math:`\varphi`) and conservation
:math:`\nabla\cdot q=\varphi`. The mathematics, equivalent inclusion and all
references are those of :mod:`crystallite.verification.conduction`, which this
module names in heat-conduction terms (the charge-conduction counterpart is
:mod:`crystallite.verification.charge_conduction`).
"""

from __future__ import annotations

from dataclasses import dataclass

from crystallite.verification.conduction import (
    ConductionInclusionCase,
    disk_source_flux,
    elliptical_perturbation,
)

__all__ = ["HeatInclusionCase", "heat_generation_flux", "elliptical_perturbation"]


@dataclass
class HeatInclusionCase(ConductionInclusionCase):
    """An ellipse (a circle for ``semi_axis_a == semi_axis_b``, a crack when
    thin) of thermal conductivity `contrast` times the matrix's
    (`matrix_conductivity`, the thermal conductivity :math:`\\kappa_0`). See
    :class:`~crystallite.verification.conduction.ConductionInclusionCase`;
    ``solver()`` returns the
    :class:`~crystallite.conduction.SteadyConduction` (its ``driving_force`` is
    :math:`X`, its ``flux`` the heat flux :math:`q`)."""

    def periodic_interior_thermal_driving_force(self, thermal_driving_force, n_modes=512):
        """Interior :math:`X` ``(3,)`` of the periodic array under the mean
        thermal driving force."""
        return self.periodic_interior_driving_force(thermal_driving_force, n_modes)

    def periodic_interior_heat_flux(self, thermal_driving_force, n_modes=512):
        """Interior heat flux ``(3,)``; undefined for a perfect conductor."""
        return self.periodic_interior_flux(thermal_driving_force, n_modes)

    def periodic_thermal_driving_force(self, thermal_driving_force, n_images=8, n_modes=512):
        """Thermal driving force ``(3,) + grid.shape`` of the periodic array."""
        return self.periodic_exterior_field(thermal_driving_force, n_images, n_modes)

    def periodic_heat_flux(self, thermal_driving_force, n_images=8, n_modes=512):
        """Heat flux ``(3,) + grid.shape`` of the periodic array (NaN inside a
        perfect conductor)."""
        return self.periodic_flux_field(thermal_driving_force, n_images, n_modes)


def heat_generation_flux(grid, radius, generation, center=(0.5, 0.5), thermal_conductivity=1.0, refine=4):
    r"""Heat flux ``(3,) + grid.shape`` of a periodic array of cylinders each
    generating heat uniformly (`generation` :math:`\varphi` per unit area) in a
    homogeneous matrix, with the uniform sink that makes a periodic steady state
    exist (:math:`k=0` dropped). Isolated, the flux is radial,
    :math:`\varphi r/2` inside and :math:`\varphi a^2/(2r)` outside; for an
    isotropic matrix it is independent of the conductivity. See
    :func:`~crystallite.verification.conduction.disk_source_flux`."""
    return disk_source_flux(
        grid, radius, generation, center=center, conductivity=thermal_conductivity, refine=refine
    )
