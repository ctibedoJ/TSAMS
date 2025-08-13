"""
Visualization module for the prime_mobius package.

This module provides tools for visualizing various aspects of the Prime Indexed
Möbius Transformation State Space Theory, including Möbius transformations on the
Riemann sphere, orbits in state space, energy spectra, and more.
"""

from .riemann_sphere import (
    plot_riemann_sphere,
    plot_mobius_transformation,
    plot_orbit_on_sphere
)

from .energy_plots import (
    plot_energy_spectrum,
    plot_level_spacing_distribution,
    plot_spectral_rigidity
)

from .orbit_plots import (
    plot_orbit_2d,
    plot_orbit_density,
    plot_orbit_animation
)

from .nodal_structure_plots import (
    plot_nodal_structure,
    plot_hair_braid_nodes,
    plot_braid_operation
)

__all__ = [
    'plot_riemann_sphere',
    'plot_mobius_transformation',
    'plot_orbit_on_sphere',
    'plot_energy_spectrum',
    'plot_level_spacing_distribution',
    'plot_spectral_rigidity',
    'plot_orbit_2d',
    'plot_orbit_density',
    'plot_orbit_animation',
    'plot_nodal_structure',
    'plot_hair_braid_nodes',
    'plot_braid_operation'
]