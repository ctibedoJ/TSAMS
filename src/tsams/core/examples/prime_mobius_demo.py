"""
Prime Indexed Möbius Transformation State Space Theory - Demo Script

This script demonstrates the basic usage of the prime_mobius package,
showcasing the core components and their interactions.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import the core components
from prime_mobius.cyclotomic import CyclotomicField
from prime_mobius.moebius import MoebiusTransformation, PrimeIndexedMoebiusTransformation, Root420Structure
from prime_mobius.state_space import StateSpace, NodalStructure441
from prime_mobius.hair_braid import HairBraidSystem, BraidOperations
from prime_mobius.hyperbolic import EnergyQuantization, EnergySpectrum
from prime_mobius.utils import complex_analysis


def demo_cyclotomic_field():
    """
    Demonstrate the usage of the CyclotomicField class.
    """
    print("\n=== Cyclotomic Field Demo ===")
    
    # Create the cyclotomic field Q(ζ_420)
    field = CyclotomicField(420)
    
    print(f"Field: {field}")
    print(f"Dimension: {field.dimension}")
    print(f"Prime factorization of 420: {field.prime_factorization()}")
    
    # Create some elements in the field
    element1 = field.element_from_coefficients([1] + [0] * (field.dimension - 1))
    element2 = field.element_from_coefficients([0, 1] + [0] * (field.dimension - 2))
    
    # Perform operations
    sum_result = field.add(element1, element2)
    product_result = field.multiply(element1, element2)
    
    print(f"Element 1: {element1}")
    print(f"Element 2: {element2}")
    print(f"Sum: {sum_result}")
    print(f"Product: {product_result}")
    
    # Compute the Galois group structure
    galois_group = field.galois_group_structure()
    print(f"Galois group generators: {galois_group[:5]}...")
    
    # Check if this is the Dedekind cut morphic conductor
    print(f"Is Dedekind cut morphic conductor: {field.is_dedekind_cut_conductor}")
    print(f"Dedekind cut morphic conductor value: {field.dedekind_cut_morphic_conductor()}")


def demo_mobius_transformations():
    """
    Demonstrate the usage of the Möbius transformation classes.
    """
    print("\n=== Möbius Transformations Demo ===")
    
    # Create a general Möbius transformation
    mobius = MoebiusTransformation(1, 2, 3, 4)
    
    print(f"Möbius transformation: {mobius}")
    print(f"Applied to 1+2j: {mobius.apply(1+2j)}")
    print(f"Fixed points: {mobius.fixed_points()}")
    print(f"Classification: {'elliptic' if mobius.is_elliptic() else 'parabolic' if mobius.is_parabolic() else 'hyperbolic'}")
    
    # Create a prime indexed Möbius transformation
    prime_mobius = PrimeIndexedMoebiusTransformation(11, 420)
    
    print(f"\nPrime indexed Möbius transformation: {prime_mobius}")
    print(f"Prime index: {prime_mobius.prime_index}")
    print(f"Applied to 1+2j: {prime_mobius.apply(1+2j)}")
    print(f"Energy: {prime_mobius.energy()}")
    
    # Create the 420-root structure
    root_structure = Root420Structure()
    
    print(f"\n420-root structure: {root_structure}")
    print(f"Number of transformations: {len(root_structure)}")
    print(f"Primes in the structure: {root_structure.primes[:10]}...")
    
    # Get a transformation from the structure
    transformation = root_structure.get_transformation(11)
    
    print(f"Transformation for p=11: {transformation}")
    print(f"Fixed points: {root_structure.fixed_points(11)}")
    print(f"Energy: {root_structure.energy(11)}")
    
    # Compute the orbit of a point
    orbit = root_structure.orbit(1+2j, 11, max_iterations=20)
    
    print(f"Orbit of 1+2j under M_11 (first 5 points): {orbit[:5]}")


def demo_state_space():
    """
    Demonstrate the usage of the state space classes.
    """
    print("\n=== State Space Demo ===")
    
    # Create a state space
    state_space = StateSpace()
    
    print(f"State space: {state_space}")
    
    # Create a transformation
    transformation = PrimeIndexedMoebiusTransformation(11, 420)
    
    # Apply the transformation to a state
    state = 1 + 2j
    transformed_state = state_space.transform(state, transformation)
    
    print(f"State: {state}")
    print(f"Transformed state: {transformed_state}")
    
    # Compute the orbit of a state
    orbit = state_space.orbit(state, [transformation], max_iterations=20)
    
    print(f"Orbit (first 5 points): {orbit[:5]}")
    print(f"Is dense: {state_space.is_dense(orbit)}")
    
    # Create the 441-dimensional nodal structure
    nodal_structure = NodalStructure441()
    
    print(f"\n441-dimensional nodal structure: {nodal_structure}")
    
    # Get the factorized state spaces
    state_space_9, state_space_49 = nodal_structure.factorize()
    
    print(f"Factorized state spaces: {state_space_9}, {state_space_49}")
    
    # Get the hair braid nodes
    hair_braid_nodes = nodal_structure.get_hair_braid_nodes()
    
    print(f"Number of hair braid nodes: {len(hair_braid_nodes)}")
    print(f"First hair braid node: {hair_braid_nodes[0]}")
    
    # Apply a transformation to a state in the nodal structure
    state = hair_braid_nodes[0]
    transformed_state = nodal_structure.transform(state, transformation)
    
    print(f"Transformed hair braid node: {transformed_state}")


def demo_hair_braid_dynamics():
    """
    Demonstrate the usage of the hair braid dynamics classes.
    """
    print("\n=== Hair Braid Dynamics Demo ===")
    
    # Create a hair braid system
    hair_braid_system = HairBraidSystem()
    
    print(f"Hair braid system: {hair_braid_system}")
    
    # Get a hair braid node
    node = hair_braid_system.get_node(0)
    
    print(f"Hair braid node: {node}")
    
    # Apply the braid operation between two nodes
    result = hair_braid_system.braid_operation(0, 1)
    
    print(f"Braid operation result: {result}")
    
    # Create a braid
    braid = [(0, 1), (1, 2), (0, 2)]
    
    # Compute the braid invariant
    invariant = hair_braid_system.braid_invariant(braid)
    
    print(f"Braid: {braid}")
    print(f"Braid invariant: {invariant}")
    
    # Create braid operations
    braid_operations = BraidOperations()
    
    # Create a braid in the braid group B_21
    braid_group = braid_operations.braid_group
    braid = braid_group.create_braid([1, 2, 1])
    
    # Compute the Jones polynomial
    jones_polynomial = braid_operations.jones_polynomial(braid)
    
    print(f"\nBraid in B_21: {braid}")
    print(f"Jones polynomial: {jones_polynomial}")


def demo_energy_quantization():
    """
    Demonstrate the usage of the energy quantization classes.
    """
    print("\n=== Energy Quantization Demo ===")
    
    # Create an energy quantization calculator
    energy_calculator = EnergyQuantization()
    
    print(f"Energy calculator: {energy_calculator}")
    
    # Create a transformation
    transformation = PrimeIndexedMoebiusTransformation(11, 420)
    
    # Compute the energy
    energy = energy_calculator.energy(transformation)
    
    print(f"Transformation: {transformation}")
    print(f"Energy: {energy}")
    
    # Compute the energy spectrum for primes up to 100
    spectrum = energy_calculator.energy_spectrum(100)
    
    print(f"Energy spectrum (first 5 primes): {dict(list(spectrum.items())[:5])}")
    
    # Compute the nearest neighbor spacing distribution
    spacings = energy_calculator.nearest_neighbor_spacing(100)
    
    print(f"Nearest neighbor spacings (first 5): {spacings[:5]}")
    print(f"Level repulsion: {energy_calculator.level_repulsion(100)}")
    
    # Create an energy spectrum analyzer
    energy_spectrum = EnergySpectrum()
    
    # Compute the energy spectrum of the 420-root structure
    spectrum = energy_spectrum.compute_spectrum()
    
    print(f"\nEnergy spectrum of the 420-root structure (first 5 primes): {dict(list(spectrum.items())[:5])}")
    print(f"Total energy: {energy_spectrum.total_energy()}")
    print(f"Mean energy: {energy_spectrum.mean_energy()}")
    print(f"Energy variance: {energy_spectrum.energy_variance()}")


def visualize_mobius_transformation():
    """
    Visualize a Möbius transformation on the Riemann sphere.
    """
    print("\n=== Möbius Transformation Visualization ===")
    
    # Create a prime indexed Möbius transformation
    transformation = PrimeIndexedMoebiusTransformation(11, 420)
    
    # Create a grid of points in the complex plane
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Apply the transformation to each point
    W = np.zeros_like(Z, dtype=complex)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            W[i, j] = transformation.apply(Z[i, j])
    
    # Convert the points to the Riemann sphere
    X_sphere = np.zeros_like(X)
    Y_sphere = np.zeros_like(Y)
    Z_sphere = np.zeros_like(X)
    
    X_transformed = np.zeros_like(X)
    Y_transformed = np.zeros_like(Y)
    Z_transformed = np.zeros_like(X)
    
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            # Original point
            x_s, y_s, z_s = complex_analysis.stereographic_projection(Z[i, j])
            X_sphere[i, j] = x_s
            Y_sphere[i, j] = y_s
            Z_sphere[i, j] = z_s
            
            # Transformed point
            x_t, y_t, z_t = complex_analysis.stereographic_projection(W[i, j])
            X_transformed[i, j] = x_t
            Y_transformed[i, j] = y_t
            Z_transformed[i, j] = z_t
    
    # Create the figure
    fig = plt.figure(figsize=(12, 6))
    
    # Plot the original points on the Riemann sphere
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X_sphere, Y_sphere, Z_sphere, color='b', alpha=0.3)
    ax1.set_title('Original Points on Riemann Sphere')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)
    
    # Plot the transformed points on the Riemann sphere
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X_transformed, Y_transformed, Z_transformed, color='r', alpha=0.3)
    ax2.set_title('Transformed Points on Riemann Sphere')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(-1, 1)
    
    plt.tight_layout()
    plt.savefig('mobius_transformation_visualization.png')
    print("Visualization saved as 'mobius_transformation_visualization.png'")


def visualize_energy_spectrum():
    """
    Visualize the energy spectrum of the 420-root Möbius structure.
    """
    print("\n=== Energy Spectrum Visualization ===")
    
    # Create an energy spectrum analyzer
    energy_spectrum = EnergySpectrum()
    
    # Compute the energy spectrum
    spectrum = energy_spectrum.compute_spectrum()
    
    # Extract the primes and energies
    primes = sorted(spectrum.keys())
    energies = [spectrum[p] for p in primes]
    
    # Create the figure
    plt.figure(figsize=(10, 6))
    plt.scatter(primes, energies, s=20, alpha=0.7)
    plt.xlabel('Prime Index (p)')
    plt.ylabel('Energy E(M_p)')
    plt.title('Energy Spectrum of the 420-root Möbius Structure')
    plt.grid(True, alpha=0.3)
    plt.savefig('energy_spectrum_visualization.png')
    print("Visualization saved as 'energy_spectrum_visualization.png'")
    
    # Compute the level spacing distribution
    spacings, distribution = energy_spectrum.energy_spectrum.level_spacing_distribution()
    
    # Compute the Wigner surmise for comparison
    wigner_values = [energy_spectrum.energy_spectrum.energy_calculator.wigner_surmise(s) for s in spacings]
    
    # Create the figure
    plt.figure(figsize=(10, 6))
    plt.plot(spacings, distribution, 'o-', label='420-root Möbius Structure')
    plt.plot(spacings, wigner_values, 'r--', label='Wigner Surmise (GOE)')
    
    # Add a Poisson distribution for comparison
    poisson_values = [np.exp(-s) for s in spacings]
    plt.plot(spacings, poisson_values, 'g-.', label='Poisson (Uncorrelated)')
    
    plt.xlabel('Normalized Spacing (s)')
    plt.ylabel('Probability Density P(s)')
    plt.title('Level Spacing Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('level_spacing_distribution.png')
    print("Visualization saved as 'level_spacing_distribution.png'")


def main():
    """
    Main function to run all demos.
    """
    print("Prime Indexed Möbius Transformation State Space Theory - Demo")
    print("===========================================================")
    
    # Run the demos
    demo_cyclotomic_field()
    demo_mobius_transformations()
    demo_state_space()
    demo_hair_braid_dynamics()
    demo_energy_quantization()
    
    # Run the visualizations
    visualize_mobius_transformation()
    visualize_energy_spectrum()


if __name__ == "__main__":
    main()