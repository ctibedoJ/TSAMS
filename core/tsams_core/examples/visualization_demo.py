"""
Prime Indexed Möbius Transformation State Space Theory - Visualization Demo

This script demonstrates the visualization capabilities of the prime_mobius package,
showcasing the various visualization tools for Möbius transformations, orbits,
energy spectra, and nodal structures.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the parent directory to the path so that we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the core components
from prime_mobius.moebius.moebius_transformation import MoebiusTransformation
from prime_mobius.moebius.prime_indexed_moebius import PrimeIndexedMoebiusTransformation
from prime_mobius.moebius.root_420_structure import Root420Structure
from prime_mobius.state_space.state_space import StateSpace
from prime_mobius.state_space.nodal_structure_441 import NodalStructure441
from prime_mobius.hair_braid.hair_braid_nodes import HairBraidSystem
from prime_mobius.hair_braid.braid_operations import BraidOperations
from prime_mobius.hyperbolic.energy_quantization import EnergyQuantization
from prime_mobius.hyperbolic.energy_spectrum import EnergySpectrum

# Import the visualization modules
from prime_mobius.visualization.riemann_sphere import (
    plot_riemann_sphere,
    plot_mobius_transformation,
    plot_orbit_on_sphere
)
from prime_mobius.visualization.energy_plots import (
    plot_energy_spectrum,
    plot_level_spacing_distribution,
    plot_spectral_rigidity
)
from prime_mobius.visualization.orbit_plots import (
    plot_orbit_2d,
    plot_orbit_density,
    plot_orbit_animation
)
from prime_mobius.visualization.nodal_structure_plots import (
    plot_nodal_structure,
    plot_hair_braid_nodes,
    plot_braid_operation
)


def demo_riemann_sphere_visualization():
    """
    Demonstrate the Riemann sphere visualization tools.
    """
    print("\n=== Riemann Sphere Visualization Demo ===")
    
    # Create a prime indexed Möbius transformation
    transformation = PrimeIndexedMoebiusTransformation(11, 420)
    
    print(f"Visualizing the Möbius transformation: {transformation}")
    
    # Visualize the transformation on the Riemann sphere
    fig, (ax1, ax2) = plot_mobius_transformation(
        transformation, 
        grid_density=20, 
        save_path="outputs/mobius_transformation_visualization.png"
    )
    
    print("Visualization saved as 'outputs/mobius_transformation_visualization.png'")
    
    # Create a state space
    state_space = StateSpace()
    
    # Compute the orbit of a point
    initial_point = 1 + 2j
    orbit = state_space.orbit(initial_point, [transformation], max_iterations=50)
    
    print(f"Computing the orbit of {initial_point} under {transformation}")
    print(f"Orbit length: {len(orbit)}")
    
    # Visualize the orbit on the Riemann sphere
    fig, ax = plot_orbit_on_sphere(
        orbit, 
        save_path="outputs/orbit_on_sphere.png"
    )
    
    print("Orbit visualization saved as 'outputs/orbit_on_sphere.png'")


def demo_energy_spectrum_visualization():
    """
    Demonstrate the energy spectrum visualization tools.
    """
    print("\n=== Energy Spectrum Visualization Demo ===")
    
    # Create an energy spectrum analyzer
    energy_spectrum = EnergySpectrum()
    
    print("Visualizing the energy spectrum of the 420-root Möbius structure")
    
    # Plot the energy spectrum
    fig, ax = plot_energy_spectrum(
        save_path="outputs/energy_spectrum.png"
    )
    
    print("Energy spectrum visualization saved as 'outputs/energy_spectrum.png'")
    
    # Plot the level spacing distribution
    fig, ax = plot_level_spacing_distribution(
        show_wigner=True,
        show_poisson=True,
        save_path="outputs/level_spacing_distribution.png"
    )
    
    print("Level spacing distribution visualization saved as 'outputs/level_spacing_distribution.png'")
    
    # Plot the spectral rigidity curve
    fig, ax = plot_spectral_rigidity(
        max_L=20,
        num_points=20,
        show_goe=True,
        show_poisson=True,
        save_path="outputs/spectral_rigidity.png"
    )
    
    print("Spectral rigidity visualization saved as 'outputs/spectral_rigidity.png'")


def demo_orbit_visualization():
    """
    Demonstrate the orbit visualization tools.
    """
    print("\n=== Orbit Visualization Demo ===")
    
    # Create a prime indexed Möbius transformation
    transformation = PrimeIndexedMoebiusTransformation(11, 420)
    
    # Create a state space
    state_space = StateSpace()
    
    # Compute the orbit of a point
    initial_point = 1 + 2j
    orbit = state_space.orbit(initial_point, [transformation], max_iterations=100)
    
    print(f"Computing the orbit of {initial_point} under {transformation}")
    print(f"Orbit length: {len(orbit)}")
    
    # Visualize the orbit in the complex plane
    fig, ax = plot_orbit_2d(
        orbit,
        show_lines=True,
        save_path="outputs/orbit_2d.png"
    )
    
    print("Orbit visualization saved as 'outputs/orbit_2d.png'")
    
    # Visualize the orbit density
    fig, ax = plot_orbit_density(
        orbit,
        num_bins=50,
        save_path="outputs/orbit_density.png"
    )
    
    print("Orbit density visualization saved as 'outputs/orbit_density.png'")
    
    # Create an animation of the orbit
    animation = plot_orbit_animation(
        orbit,
        interval=50,
        save_path="outputs/orbit_animation.gif"
    )
    
    print("Orbit animation saved as 'outputs/orbit_animation.gif'")


def demo_nodal_structure_visualization():
    """
    Demonstrate the nodal structure visualization tools.
    """
    print("\n=== Nodal Structure Visualization Demo ===")
    
    # Visualize the 441-dimensional nodal structure
    fig, ax = plot_nodal_structure(
        show_hair_braid_nodes=True,
        save_path="outputs/nodal_structure.png"
    )
    
    print("Nodal structure visualization saved as 'outputs/nodal_structure.png'")
    
    # Visualize the hair braid nodes
    fig, ax = plot_hair_braid_nodes(
        show_connections=True,
        save_path="outputs/hair_braid_nodes.png"
    )
    
    print("Hair braid nodes visualization saved as 'outputs/hair_braid_nodes.png'")
    
    # Visualize a braid operation
    fig, ax = plot_braid_operation(
        node1_idx=0,
        node2_idx=1,
        save_path="outputs/braid_operation.png"
    )
    
    print("Braid operation visualization saved as 'outputs/braid_operation.png'")
    
    # Create a braid
    braid = [(0, 1), (1, 2), (0, 2)]
    
    # Note: We'll skip the braid invariant and Jones polynomial visualizations
    # since they're not implemented in our visualization module yet
    print("Braid operation visualization completed")
    
    # For future implementation:
    # # Visualize the braid invariant
    # fig, ax = plot_braid_invariant(
    #     braid,
    #     save_path="outputs/braid_invariant.png"
    # )
    # print("Braid invariant visualization saved as 'outputs/braid_invariant.png'")
    # 
    # # Visualize the Jones polynomial
    # fig, ax = plot_jones_polynomial(
    #     braid,
    #     save_path="outputs/jones_polynomial.png"
    # )
    # print("Jones polynomial visualization saved as 'outputs/jones_polynomial.png'")


def main():
    """
    Main function to run all visualization demos.
    """
    print("Prime Indexed Möbius Transformation State Space Theory - Visualization Demo")
    print("===========================================================================")
    
    # Create the outputs directory if it doesn't exist
    import os
    os.makedirs("outputs", exist_ok=True)
    
    # Run the demos
    demo_riemann_sphere_visualization()
    demo_energy_spectrum_visualization()
    demo_orbit_visualization()
    demo_nodal_structure_visualization()
    
    print("\nAll visualizations have been saved to the 'outputs' directory.")


if __name__ == "__main__":
    main()