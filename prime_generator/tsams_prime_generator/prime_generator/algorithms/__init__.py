"""
Prime number generation and testing algorithms.

This module provides various algorithms for generating prime numbers
and testing primality, including both classical methods and
TSAMS-specific approaches based on advanced mathematical concepts.
"""

# Classical prime generation methods
from .generation import (
    sieve_of_eratosthenes,
    sieve_of_atkin,
    segmented_sieve,
    prime_range,
    prime_generator
)

# Classical primality testing
from .testing import (
    is_prime,
    miller_rabin_test,
    fermat_test,
    lucas_lehmer_test,
    trial_division,
    aks_test
)

# Prime number operations
from .operations import (
    next_prime,
    prev_prime,
    prime_factors,
    prime_factorization,
    nth_prime,
    prime_counting_function,
    coprime_to,
    prime_gap
)

# TSAMS-specific prime algorithms
from .tsams_primes import (
    # Cyclotomic field-based methods
    cyclotomic_sieve,
    
    # Quantum-inspired methods
    quantum_prime_generator,
    
    # E8 lattice methods
    e8_lattice_sieve,
    
    # Modular forms and L-functions
    modular_forms_prime_test,
    l_function_prime_test,
    zeta_zeros_prime_generator,
    
    # Galois theory methods
    galois_theory_prime_generator
)