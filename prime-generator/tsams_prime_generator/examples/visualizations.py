"""
Visualization examples for the TSAMS Prime Generator package.
"""

import matplotlib.pyplot as plt
import numpy as np

from prime_generator.visualization import (
    # Classical visualizations
    ulam_spiral,
    sacks_spiral,
    plot_prime_distribution,
    prime_gaps_plot,
    
    # TSAMS-specific visualizations
    cyclotomic_field_visualization,
    quantum_interference_pattern,
    e8_lattice_projection,
    modular_forms_zeros,
    l_function_visualization
)

# Create a directory for saving visualizations
import os
if not os.path.exists("visualizations"):
    os.makedirs("visualizations")

# Generate classical visualizations
print("Generating classical visualizations...")

# Ulam spiral
fig = ulam_spiral(size=50, highlight_pattern='twin_primes', cmap='viridis')
plt.savefig("visualizations/ulam_spiral.png", dpi=300)
plt.close(fig)
print("- Created Ulam spiral visualization")

# Sacks spiral
fig = sacks_spiral(limit=500, dot_size=5, cmap='plasma')
plt.savefig("visualizations/sacks_spiral.png", dpi=300)
plt.close(fig)
print("- Created Sacks spiral visualization")

# Prime distribution
fig = plot_prime_distribution(limit=1000, bin_size=50)
plt.savefig("visualizations/prime_distribution.png", dpi=300)
plt.close(fig)
print("- Created prime distribution visualization")

# Prime gaps
fig = prime_gaps_plot(limit=1000)
plt.savefig("visualizations/prime_gaps.png", dpi=300)
plt.close(fig)
print("- Created prime gaps visualization")

# Generate TSAMS-specific visualizations
print("\nGenerating TSAMS-specific visualizations...")

# Cyclotomic field visualization
fig = cyclotomic_field_visualization(conductor=8, limit=200, cmap='viridis')
plt.savefig("visualizations/cyclotomic_field.png", dpi=300)
plt.close(fig)
print("- Created cyclotomic field visualization")

# Quantum interference pattern
fig = quantum_interference_pattern(limit=100, qubits=4)
plt.savefig("visualizations/quantum_interference.png", dpi=300)
plt.close(fig)
print("- Created quantum interference pattern visualization")

# E8 lattice projection
fig = e8_lattice_projection(limit=100, dimensions=(0, 1))
plt.savefig("visualizations/e8_lattice.png", dpi=300)
plt.close(fig)
print("- Created E8 lattice projection visualization")

# Modular forms zeros
fig = modular_forms_zeros(limit=20, weight=12)
plt.savefig("visualizations/modular_forms_zeros.png", dpi=300)
plt.close(fig)
print("- Created modular forms zeros visualization")

# L-function visualization
fig = l_function_visualization(limit=100)
plt.savefig("visualizations/l_function.png", dpi=300)
plt.close(fig)
print("- Created L-function visualization")

print("\nAll visualizations saved to the 'visualizations' directory.")
print("Done!")