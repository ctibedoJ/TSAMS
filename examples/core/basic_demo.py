"""
Basic demonstration of the TSAMS Core package.

This script demonstrates the basic usage of the TSAMS Core package,
including the creation and manipulation of Möbius transformations,
cyclotomic fields, and state spaces.
"""

import numpy as np
import matplotlib.pyplot as plt
from tsams_core.moebius import Root420Structure, MoebiusTransformation
from tsams_core.state_space import NodalStructure441
from tsams_core.cyclotomic import CyclotomicField
from tsams_core.visualization import OrbitPlotter, EnergySpectrumPlotter

def demonstrate_moebius_transformations():
    """
    Demonstrate the usage of Möbius transformations.
    """
    print("\n=== Möbius Transformations ===")
    
    # Create a simple Möbius transformation
    m = MoebiusTransformation(1, 2, 3, 4)
    print(f"Transformation: {m}")
    
    # Apply the transformation to a complex number
    z = 1 + 2j
    w = m.apply(z)
    print(f"M({z}) = {w}")
    
    # Compute the fixed points of the transformation
    fixed_points = m.fixed_points()
    print(f"Fixed points: {fixed_points}")
    
    # Check the type of the transformation
    if m.is_elliptic():
        print("The transformation is elliptic")
    elif m.is_parabolic():
        print("The transformation is parabolic")
    elif m.is_hyperbolic():
        print("The transformation is hyperbolic")
    else:
        print("The transformation is loxodromic")

def demonstrate_root420_structure():
    """
    Demonstrate the usage of the 420-root structure.
    """
    print("\n=== 420-Root Structure ===")
    
    # Create the 420-root structure
    root_structure = Root420Structure()
    print(f"Number of transformations: {len(root_structure.transformations)}")
    
    # Get a transformation corresponding to a prime index
    p = 11
    transformation = root_structure.get_transformation(p)
    print(f"Transformation M_{p}: {transformation}")
    
    # Compute the energy of the transformation
    energy = transformation.energy()
    print(f"Energy of M_{p}: {energy}")
    
    # Compute the fixed points of the transformation
    fixed_points = transformation.fixed_points()
    print(f"Fixed points of M_{p}: {fixed_points}")
    
    # Compute the orbit of a point under the transformation
    initial_point = 1.0 + 0.5j
    orbit = root_structure.orbit(initial_point, p, max_iterations=10)
    print(f"Orbit of {initial_point} under M_{p}: {orbit}")
    
    # Compute the energy spectrum of the 420-root structure
    spectrum = root_structure.energy_spectrum()
    print(f"Energy spectrum (first 5 values): {spectrum[:5]}")
    
    # Plot the energy spectrum
    plt.figure(figsize=(10, 6))
    EnergySpectrumPlotter.plot_energy_spectrum(spectrum)
    plt.title("Energy Spectrum of the 420-Root Möbius Structure")
    plt.savefig("energy_spectrum.png")
    plt.close()
    
    # Plot the orbit
    plt.figure(figsize=(8, 8))
    OrbitPlotter.plot_orbit_2d(orbit)
    plt.title(f"Orbit of {initial_point} under M_{p}")
    plt.savefig("orbit.png")
    plt.close()

def demonstrate_cyclotomic_field():
    """
    Demonstrate the usage of cyclotomic fields.
    """
    print("\n=== Cyclotomic Field ===")
    
    # Create a cyclotomic field
    field = CyclotomicField(420)
    print(f"Field: {field}")
    print(f"Dimension: {field.dimension}")
    
    # Create field elements
    a = field.element_from_coefficients([1] + [0] * (field.dimension - 1))
    b = field.element_from_coefficients([0, 1] + [0] * (field.dimension - 2))
    print(f"Element a: {a}")
    print(f"Element b: {b}")
    
    # Perform field operations
    sum_ab = field.add(a, b)
    product_ab = field.multiply(a, b)
    print(f"a + b: {sum_ab}")
    print(f"a * b: {product_ab}")
    
    # Compute the conjugate of an element
    conj_b = field.conjugate(b)
    print(f"Conjugate of b: {conj_b}")
    
    # Compute the norm of an element
    norm_b = field.norm(b)
    print(f"Norm of b: {norm_b}")
    
    # Get the prime factorization of the conductor
    factors = field.prime_factorization()
    print(f"Prime factorization of 420: {factors}")

def demonstrate_nodal_structure():
    """
    Demonstrate the usage of the 441-dimensional nodal structure.
    """
    print("\n=== 441-Dimensional Nodal Structure ===")
    
    # Create the 441-dimensional nodal structure
    nodal_structure = NodalStructure441()
    print(f"Structure: {nodal_structure}")
    
    # Get the factorized state spaces
    state_space_9, state_space_49 = nodal_structure.factorize()
    print(f"State space dimensions: {state_space_9.dimension}, {state_space_49.dimension}")
    
    # Get the hair braid nodes
    nodes = nodal_structure.get_hair_braid_nodes()
    print(f"Number of hair braid nodes: {len(nodes)}")
    print(f"First node: {nodes[0]}")
    
    # Perform a braid operation
    result = nodal_structure.braid_operation(0, 1)
    print(f"Braid operation (0, 1): {result}")
    
    # Compute a braid invariant
    braid = [(0, 1), (1, 2), (0, 2)]
    invariant = nodal_structure.braid_invariant(braid)
    print(f"Braid invariant: {invariant}")
    
    # Compute a Jones polynomial
    polynomial = nodal_structure.jones_polynomial(braid)
    print(f"Jones polynomial: {polynomial}")

def main():
    """
    Main function to demonstrate the TSAMS Core package.
    """
    print("TSAMS Core Package Demonstration")
    print("===============================")
    
    demonstrate_moebius_transformations()
    demonstrate_root420_structure()
    demonstrate_cyclotomic_field()
    demonstrate_nodal_structure()
    
    print("\nDemonstration complete!")

if __name__ == "__main__":
    main()
