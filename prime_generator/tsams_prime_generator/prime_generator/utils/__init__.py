"""
Utility functions for prime number analysis.

This module provides various utility functions for analyzing prime numbers,
including statistical tools, special prime sequences, and mathematical operations
from both classical number theory and TSAMS-specific approaches.
"""

# Standard utilities
from .statistics import (
    prime_count_estimate,
    prime_density,
    prime_gap_statistics,
    prime_distribution_metrics
)

from .special_primes import (
    twin_primes,
    mersenne_primes,
    fermat_primes,
    sophie_germain_primes,
    safe_primes
)

from .number_theory import (
    goldbach_partitions,
    prime_zeta_function,
    prime_counting_function,
    legendre_symbol
)

# TSAMS-specific utilities
from .tsams_utils import (
    cyclotomic_field_extension,
    quantum_fourier_transform,
    e8_root_system,
    modular_form_coefficients,
    l_function_zeros,
    ramanujan_tau_coefficients
)