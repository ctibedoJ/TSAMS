"""
Advanced demonstration of the TSAMS Core package.

This script demonstrates more advanced usage of the TSAMS Core package,
including the visualization of Möbius transformations, energy spectra,
and nodal structures.
"""

import numpy as np
import matplotlib.pyplot as plt
from tsams_core.moebius import Root420Structure, PrimeIndexedMoebiusTransformation
from tsams_core.state_space import NodalStructure441, StateTransformation, PrimeIndexedStateTransformation
from tsams_core.cyclotomic import CyclotomicField
from tsams_core.visualization import RiemannSphereVisualizer, EnergySpectrumPlotter, OrbitPlotter, NodalStructurePlotter
from tsams_core.utils import is_prime, prime_factors, roots_of_unity

def visualize_moebius_transformations():
    """
    Visualize Möbius transformations on the Riemann sphere.
    """
    print("\n=== Visualizing Möbius Transformations ===")
    
    # Create the 420-root structure
    root_structure = Root420Structure()
    
    # Get transformations corresponding to different prime indices
    primes = [2, 3, 5, 7, 11]
    transformations = [root_structure.get_transformation(p) for p in primes]
    
    # Create a Riemann sphere visualizer
    visualizer = RiemannSphereVisualizer()
    
    # Visualize each transformation
    for p, transformation in zip(primes, transformations):
        print(f"Visualizing transformation M_{p}")
        
        # Visualize the transformation on the Riemann sphere
        plt.figure(figsize=(10, 10))
        visualizer.plot_transformation(transformation)
        plt.title(f"Möbius Transformation M_{p} on the Riemann Sphere")
        plt.savefig(f"moebius_transformation_{p}.png")
        plt.close()
        
        # Visualize the fixed points of the transformation
        fixed_points = transformation.fixed_points()
        plt.figure(figsize=(10, 10))
        visualizer.plot_fixed_points(transformation)
        plt.title(f"Fixed Points of M_{p} on the Riemann Sphere")
        plt.savefig(f"fixed_points_{p}.png")
        plt.close()
        
        # Visualize the orbit of a point under the transformation
        initial_point = 1.0 + 0.5j
        orbit = root_structure.orbit(initial_point, p, max_iterations=20)
        plt.figure(figsize=(10, 10))
        visualizer.plot_orbit(orbit)
        plt.title(f"Orbit of {initial_point} under M_{p}")
        plt.savefig(f"orbit_{p}.png")
        plt.close()

def analyze_energy_spectrum():
    """
    Analyze the energy spectrum of the 420-root structure.
    """
    print("\n=== Analyzing Energy Spectrum ===")
    
    # Create the 420-root structure
    root_structure = Root420Structure()
    
    # Compute the energy spectrum
    spectrum = root_structure.energy_spectrum()
    print(f"Energy spectrum (first 10 values): {spectrum[:10]}")
    
    # Plot the energy spectrum
    plt.figure(figsize=(12, 6))
    EnergySpectrumPlotter.plot_energy_spectrum(spectrum)
    plt.title("Energy Spectrum of the 420-Root Möbius Structure")
    plt.savefig("energy_spectrum_full.png")
    plt.close()
    
    # Plot the level spacing distribution
    plt.figure(figsize=(12, 6))
    EnergySpectrumPlotter.plot_level_spacing_distribution(spectrum)
    plt.title("Level Spacing Distribution of the 420-Root Möbius Structure")
    plt.savefig("level_spacing_distribution.png")
    plt.close()
    
    # Plot the cumulative energy distribution
    plt.figure(figsize=(12, 6))
    EnergySpectrumPlotter.plot_cumulative_energy_distribution(spectrum)
    plt.title("Cumulative Energy Distribution of the 420-Root Möbius Structure")
    plt.savefig("cumulative_energy_distribution.png")
    plt.close()

def explore_nodal_structure():
    """
    Explore the 441-dimensional nodal structure.
    """
    print("\n=== Exploring 441-Dimensional Nodal Structure ===")
    
    # Create the 441-dimensional nodal structure
    nodal_structure = NodalStructure441()
    
    # Get the hair braid nodes
    nodes = nodal_structure.get_hair_braid_nodes()
    print(f"Number of hair braid nodes: {len(nodes)}")
    
    # Visualize the hair braid nodes
    plt.figure(figsize=(12, 12))
    NodalStructurePlotter.plot_hair_braid_nodes(nodes)
    plt.title("Hair Braid Nodes in the 441-Dimensional Structure")
    plt.savefig("hair_braid_nodes.png")
    plt.close()
    
    # Perform braid operations
    print("\nPerforming braid operations:")
    for i in range(3):
        for j in range(i+1, 3):
            result = nodal_structure.braid_operation(i, j)
            print(f"Braid operation ({i}, {j}): {result}")
    
    # Compute braid invariants
    braids = [
        [(0, 1), (1, 2), (0, 2)],
        [(0, 1), (1, 2), (2, 3), (0, 3)],
        [(0, 1), (1, 2), (2, 0), (0, 1)]
    ]
    
    print("\nComputing braid invariants:")
    for i, braid in enumerate(braids):
        invariant = nodal_structure.braid_invariant(braid)
        print(f"Braid {i+1} invariant: {invariant}")
        
        polynomial = nodal_structure.jones_polynomial(braid)
        print(f"Braid {i+1} Jones polynomial: {polynomial}")
    
    # Check the Yang-Baxter equation
    print("\nChecking the Yang-Baxter equation:")
    for i in range(3):
        for j in range(i+1, 4):
            for k in range(j+1, 5):
                result = nodal_structure.yang_baxter_equation_check(i, j, k)
                print(f"Yang-Baxter equation for ({i}, {j}, {k}): {result}")

def investigate_cyclotomic_properties():
    """
    Investigate properties of cyclotomic fields.
    """
    print("\n=== Investigating Cyclotomic Field Properties ===")
    
    # Create cyclotomic fields with different conductors
    conductors = [3, 4, 5, 7, 8, 9, 11, 12, 15, 16, 20, 24, 30, 60, 168, 420]
    fields = [CyclotomicField(n) for n in conductors]
    
    # Compare dimensions
    dimensions = [field.dimension for field in fields]
    print("Conductor\tDimension\tφ(n)")
    for n, dim in zip(conductors, dimensions):
        print(f"{n}\t\t{dim}\t\t{dim}")
    
    # Investigate the Dedekind cut morphic conductor
    field = CyclotomicField(420)
    dedekind_cut = field.dedekind_cut_morphic_conductor()
    print(f"\nDedekind cut morphic conductor: {dedekind_cut}")
    
    # Investigate the prime factorization of the conductor
    factors = field.prime_factorization()
    print(f"Prime factorization of 420: {factors}")
    
    # Investigate the Galois group structure
    galois_group = field.galois_group_structure()
    print(f"Order of the Galois group: {len(galois_group)}")
    print(f"First 10 elements of the Galois group: {galois_group[:10]}")
    
    # Create and manipulate field elements
    a = field.element_from_coefficients([1] + [0] * (field.dimension - 1))
    b = field.element_from_coefficients([0, 1] + [0] * (field.dimension - 2))
    
    # Perform field operations
    sum_ab = field.add(a, b)
    product_ab = field.multiply(a, b)
    conj_b = field.conjugate(b)
    norm_b = field.norm(b)
    
    print(f"\nElement a: {a}")
    print(f"Element b: {b}")
    print(f"a + b: {sum_ab}")
    print(f"a * b: {product_ab}")
    print(f"Conjugate of b: {conj_b}")
    print(f"Norm of b: {norm_b}")

def demonstrate_utils():
    """
    Demonstrate the utility functions.
    """
    print("\n=== Demonstrating Utility Functions ===")
    
    # Check if numbers are prime
    numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    print("Number\tIs Prime")
    for n in numbers:
        print(f"{n}\t{is_prime(n)}")
    
    # Get prime factors
    numbers = [12, 15, 20, 30, 42, 60, 168, 420]
    print("\nNumber\tPrime Factors")
    for n in numbers:
        print(f"{n}\t{prime_factors(n)}")
    
    # Get roots of unity
    n = 8
    roots = roots_of_unity(n)
    print(f"\n{n}th roots of unity:")
    for i, root in enumerate(roots):
        print(f"ζ_{n}^{i} = {root}")
    
    # Plot the roots of unity
    plt.figure(figsize=(8, 8))
    plt.scatter([root.real for root in roots], [root.imag for root in roots], s=100)
    plt.plot([root.real for root in roots] + [roots[0].real], [root.imag for root in roots] + [roots[0].imag], 'r-')
    for i, root in enumerate(roots):
        plt.text(root.real * 1.1, root.imag * 1.1, f"ζ_{n}^{i}")
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.title(f"{n}th Roots of Unity")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.axis('equal')
    plt.savefig("roots_of_unity.png")
    plt.close()

def main():
    """
    Main function to demonstrate advanced features of the TSAMS Core package.
    """
    print("TSAMS Core Package Advanced Demonstration")
    print("=======================================")
    
    # Uncomment the functions you want to run
    # visualize_moebius_transformations()
    analyze_energy_spectrum()
    explore_nodal_structure()
    investigate_cyclotomic_properties()
    demonstrate_utils()
    
    print("\nAdvanced demonstration complete!")

if __name__ == "__main__":
    main()
