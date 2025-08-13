"""
Statistical utilities for prime number analysis.

This module provides functions for statistical analysis of prime numbers,
including distribution metrics and density calculations.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Union

from ..algorithms.generation import sieve_of_eratosthenes


def prime_count_estimate(n: int, method: str = 'pnt') -> float:
    """
    Estimate the number of primes less than or equal to n.
    
    Args:
        n: Upper bound
        method: Estimation method ('pnt', 'li', 'riemann')
        
    Returns:
        Estimated count of primes
    """
    if n < 2:
        return 0
    
    if method == 'pnt':
        # Prime Number Theorem approximation: π(n) ≈ n/ln(n)
        return n / math.log(n)
    
    elif method == 'li':
        # Logarithmic integral approximation: π(n) ≈ Li(n)
        # We use a numerical approximation of the logarithmic integral
        if n <= 10:
            return sum(1 for i in range(2, n+1) if all(i % j != 0 for j in range(2, int(i**0.5) + 1)))
        
        # Numerical integration for Li(n)
        x_values = np.linspace(2, n, 1000)
        y_values = 1 / np.log(x_values)
        return np.trapz(y_values, x_values)
    
    elif method == 'riemann':
        # Riemann's approximation: π(n) ≈ R(n)
        # This is a simplified version of Riemann's formula
        if n <= 10:
            return sum(1 for i in range(2, n+1) if all(i % j != 0 for j in range(2, int(i**0.5) + 1)))
        
        # Use Riemann's approximation
        li_n = prime_count_estimate(n, 'li')
        correction = 0
        
        # Apply corrections for small n
        if n > 10:
            correction -= 0.5 * prime_count_estimate(math.sqrt(n), 'li')
        
        return li_n + correction
    
    else:
        raise ValueError(f"Unknown estimation method: {method}")


def prime_density(n: int, window_size: Optional[int] = None) -> float:
    """
    Calculate the density of primes around n.
    
    Args:
        n: Center point
        window_size: Size of the window around n (default: sqrt(n))
        
    Returns:
        Density of primes in the specified window
    """
    if n < 2:
        return 0
    
    # Set default window size
    if window_size is None:
        window_size = int(math.sqrt(n))
    
    # Ensure window doesn't go below 2
    lower = max(2, n - window_size // 2)
    upper = n + window_size // 2
    
    # Generate primes in the window
    primes = sieve_of_eratosthenes(upper)
    primes_in_window = [p for p in primes if lower <= p <= upper]
    
    # Calculate density
    return len(primes_in_window) / (upper - lower + 1)


def prime_gap_statistics(limit: int) -> Dict[str, Union[float, Dict[int, int]]]:
    """
    Calculate statistics about gaps between consecutive primes.
    
    Args:
        limit: Upper bound for prime generation
        
    Returns:
        Dictionary with gap statistics
    """
    # Generate primes
    primes = sieve_of_eratosthenes(limit)
    
    if len(primes) <= 1:
        return {
            "mean_gap": 0,
            "max_gap": 0,
            "min_gap": 0,
            "gap_counts": {}
        }
    
    # Calculate gaps
    gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
    
    # Count occurrences of each gap size
    gap_counts = {}
    for gap in gaps:
        gap_counts[gap] = gap_counts.get(gap, 0) + 1
    
    return {
        "mean_gap": sum(gaps) / len(gaps),
        "max_gap": max(gaps),
        "min_gap": min(gaps),
        "gap_counts": gap_counts
    }


def prime_distribution_metrics(limit: int, bin_size: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate metrics about the distribution of primes.
    
    Args:
        limit: Upper bound for prime generation
        bin_size: Size of bins for distribution analysis (default: sqrt(limit))
        
    Returns:
        Dictionary with distribution metrics
    """
    # Generate primes
    primes = sieve_of_eratosthenes(limit)
    
    # Set default bin size
    if bin_size is None:
        bin_size = int(math.sqrt(limit))
    
    # Create histogram
    bins = np.arange(0, limit + bin_size, bin_size)
    hist, _ = np.histogram(primes, bins=bins)
    
    # Calculate metrics
    mean_count = np.mean(hist)
    std_dev = np.std(hist)
    cv = std_dev / mean_count if mean_count > 0 else 0  # Coefficient of variation
    
    # Calculate theoretical density based on Prime Number Theorem
    x = (bins[:-1] + bins[1:]) / 2
    theoretical_density = bin_size / np.log(x[1:])  # Skip the first bin which might contain 0
    
    # Calculate mean absolute error between actual and theoretical
    mae = np.mean(np.abs(hist[1:] - theoretical_density))
    
    return {
        "mean_primes_per_bin": mean_count,
        "std_dev": std_dev,
        "coefficient_of_variation": cv,
        "mae_from_theory": mae
    }