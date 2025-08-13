"""
TSAMS Prime Generator Module

A comprehensive toolkit for prime number generation, analysis, and visualization,
specifically designed for the Tibedo Structural Algebraic Modeling System (TSAMS) ecosystem.

This module provides advanced tools for:
- Prime number generation using cyclotomic field-based approaches
- Quantum-inspired primality testing
- E8 lattice-based prime sieving
- Modular forms and L-functions for prime analysis
- Galois theory applications to prime generation
- Visualization of prime patterns through various mathematical lenses
"""

__version__ = '0.1.0'

# Import core algorithms
from prime_generator.algorithms import (
    # Classical methods
    sieve_of_eratosthenes,
    sieve_of_atkin,
    miller_rabin_test,
    is_prime,
    next_prime,
    prev_prime,
    prime_factors,
    prime_range,
    
    # TSAMS-specific methods
    cyclotomic_sieve,
    quantum_prime_generator,
    e8_lattice_sieve,
    modular_forms_prime_test,
    l_function_prime_test,
    zeta_zeros_prime_generator,
    galois_theory_prime_generator
)

# Import visualization tools
from prime_generator.visualization import (
    # Standard visualizations
    plot_prime_distribution,
    ulam_spiral,
    sacks_spiral,
    prime_gaps_plot,
    
    # TSAMS-specific visualizations
    cyclotomic_field_visualization,
    quantum_interference_pattern,
    e8_lattice_projection,
    modular_forms_zeros,
    l_function_visualization
)

# Import utility functions
from prime_generator.utils import (
    # Standard utilities
    prime_count_estimate,
    prime_density,
    twin_primes,
    mersenne_primes,
    goldbach_partitions,
    
    # TSAMS-specific utilities
    cyclotomic_field_extension,
    quantum_fourier_transform,
    e8_root_system,
    modular_form_coefficients,
    l_function_zeros
)