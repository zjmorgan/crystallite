Polycrystal microstructures
===========================

:mod:`crystallite.microstructure` builds a periodic polycrystal and turns its
crystal orientations into the property fields the solvers take.

1. **Tessellate and orient.**
   :meth:`crystallite.microstructure.Microstructure.voronoi` makes a periodic
   Voronoi tessellation with a requested number of grains and gives them
   orientations: one random orientation per grain, a random pool of ``m``
   orientations assigned so touching grains differ, or an explicit pool (e.g.
   from :func:`crystallite.microstructure.rotation_from_bunge_euler`).
2. **Map properties.** ``property_field(tensor)`` rotates a crystal-frame
   property (a conductivity, a mobility, a stiffness, ...) grain by grain into
   a ``tensor.shape + grid.shape`` field for
   :class:`crystallite.conduction.SteadyConduction`,
   :class:`crystallite.elastic_deformation.ElasticDeformation` (with
   ``stiffness=`` and an explicit ``reference_green``, e.g. of the Voigt
   average) or :class:`crystallite.mass_diffusion.MassDiffusion`.
   ``average_property(tensor)`` is the volume (Voigt) average over the grains, the
   natural homogeneous reference medium, and ``grain_interior(grain, margin)``
   the periodic mask of a grain shaved by ``margin`` voxels, for measuring inside
   a grain away from its boundaries.
3. **Grain evolution.** ``order_parameters()`` gives the initial condition of
   :class:`crystallite.grain_orientation.GrainOrientation`. Order parameter
   ``i`` carries orientation ``slot_orientations[i]``; a polycrystal grown from
   noise instead gets a pool of orientations sampled once,
   :func:`crystallite.microstructure.random_rotations`. Either way
   :func:`crystallite.microstructure.orientation_weights` and
   :func:`crystallite.microstructure.mixed_property_field` map the evolving
   order parameters to property fields (arithmetic mixing of the rotated
   tensors over the diffuse boundaries; other rules replace that one function).

The tests check the chain against exact results: a random 2D polycrystal of
uniaxial-conducting grains has effective conductivity :math:`\sqrt{\kappa_a
\kappa_b}` (Dykhne), reproduced to about 0.1%, and a random cubic polycrystal's
elastic energy lies between the Reuss and Voigt bounds.
