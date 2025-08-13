"""
Number theory utilities for prime number analysis.

This module provides functions for number theory operations related to prime numbers,
including Goldbach partitions, zeta functions, and other mathematical tools.
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Union

from ..algorithms.generation import sieve_of_eratosthenes
from ..algorithms.testing import is_prime


def goldbach_partitions(n: int) -> List[Tuple[int, int]]:
    """
    Find all ways to express an even number as a sum of two primes.
    
    This implements Goldbach's conjecture, which states that every even integer
    greater than 2 can be expressed as the sum of two primes.
    
    Args:
        n: An even integer greater than 2
        
    Returns:
        List of prime pairs (p, q) such that p + q = n
    """
    if n <= 2 or n % 2 != 0:
        return []
    
    # Generate primes up to n
    primes = set(sieve_of_eratosthenes(n))
    
    # Find all pairs of primes that sum to n
    pairs = []
    for p in primes:
        q = n - p
        if q in primes and p <= q:  # Avoid duplicates by requiring p <= q
            pairs.append((p, q))
    
    return sorted(pairs)


def prime_zeta_function(s: float, terms: int = 1000) -> float:
    """
    Calculate the prime zeta function P(s).
    
    The prime zeta function is defined as the sum of 1/p^s over all primes p.
    
    Args:
        s: Exponent (must be > 1 for convergence)
        terms: Number of prime terms to include
        
    Returns:
        Value of P(s)
    """
    if s <= 1:
        raise ValueError("s must be greater than 1 for the series to converge")
    
    # Generate primes
    primes = sieve_of_eratosthenes(terms * int(math.log(terms)))
    primes = primes[:terms]  # Limit to specified number of terms
    
    # Calculate the sum
    return sum(1 / (p**s) for p in primes)


def prime_counting_function(n: int, method: str = 'exact') -> Union[int, float]:
    """
    Calculate the prime counting function π(n).
    
    Args:
        n: Upper bound
        method: Calculation method ('exact', 'approx', 'meissel')
        
    Returns:
        Number of primes less than or equal to n
    """
    if method == 'exact':
        # Exact count using sieve
        primes = sieve_of_eratosthenes(n)
        return len(primes)
    
    elif method == 'approx':
        # Approximation using the Prime Number Theorem
        if n < 2:
            return 0
        return n / math.log(n)
    
    elif method == 'meissel':
        # Simplified Meissel-Lehmer algorithm
        if n < 2:
            return 0
        
        # For small n, use exact counting
        if n <= 1000:
            return len(sieve_of_eratosthenes(n))
        
        # For larger n, use a combination of exact counting and approximation
        sqrt_n = int(math.sqrt(n))
        
        # Count primes up to sqrt(n) exactly
        small_primes = sieve_of_eratosthenes(sqrt_n)
        count = len(small_primes)
        
        # Use approximation for the rest
        for i in range(sqrt_n + 1, n + 1):
            if all(i % p != 0 for p in small_primes):
                count += 1
        
        return count
    
    else:
        raise ValueError(f"Unknown method: {method}")


def legendre_symbol(a: int, p: int) -> int:
    """
    Calculate the Legendre symbol (a/p).
    
    Args:
        a: Integer
        p: Odd prime
        
    Returns:
        1 if a is a quadratic residue modulo p
        -1 if a is a quadratic non-residue modulo p
        0 if a is divisible by p
    """
    if p <= 1 or not is_prime(p) or p % 2 == 0:
        raise ValueError("p must be an odd prime")
    
    if a % p == 0:
        return 0
    
    # Use Euler's criterion: (a/p) ≡ a^((p-1)/2) (mod p)
    result = pow(a, (p - 1) // 2, p)
    
    # Normalize the result to {-1, 1}
    if result == p - 1:
        return -1
    return result


def jacobi_symbol(a: int, n: int) -> int:
    """
    Calculate the Jacobi symbol (a/n).
    
    Args:
        a: Integer
        n: Odd positive integer
        
    Returns:
        Jacobi symbol value (-1, 0, or 1)
    """
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be an odd positive integer")
    
    a %= n
    result = 1
    
    while a != 0:
        # Extract factors of 2 from a
        while a % 2 == 0:
            a //= 2
            # Apply quadratic reciprocity for (2/n)
            if n % 8 in (3, 5):
                result = -result
        
        # Swap a and n
        a, n = n, a
        
        # Apply quadratic reciprocity
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        
        # Reduce a modulo n
        a %= n
    
    if n == 1:
        return result
    return 0


def mobius_function(n: int) -> int:
    """
    Calculate the Möbius function μ(n).
    
    Args:
        n: Positive integer
        
    Returns:
        1 if n is a square-free integer with an even number of prime factors
        -1 if n is a square-free integer with an odd number of prime factors
        0 if n has a squared prime factor
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    
    if n == 1:
        return 1
    
    # Check for squared prime factors
    i = 2
    factors = 0
    
    while i * i <= n:
        if n % i == 0:
            n //= i
            factors += 1
            
            # Check for squared factor
            if n % i == 0:
                return 0
        else:
            i += 1
    
    # If n > 1, it's a prime factor
    if n > 1:
        factors += 1
    
    # Return 1 for even number of factors, -1 for odd
    return 1 if factors % 2 == 0 else -1


def euler_totient(n: int) -> int:
    """
    Calculate Euler's totient function φ(n).
    
    Args:
        n: Positive integer
        
    Returns:
        Number of integers k in the range 1 ≤ k ≤ n that are coprime to n
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    
    result = n  # Initialize result as n
    
    # Consider all prime factors of n and subtract their multiples
    p = 2
    while p * p <= n:
        # Check if p is a prime factor
        if n % p == 0:
            # If yes, then update n and result
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    
    # If n has a prime factor greater than sqrt(n)
    # (There can be at most one such prime factor)
    if n > 1:
        result -= result // n
    
    return result


def divisor_sum(n: int, k: int = 1) -> int:
    """
    Calculate the sum of the k-th powers of the divisors of n.
    
    Args:
        n: Positive integer
        k: Power to raise each divisor to
        
    Returns:
        Sum of d^k for all divisors d of n
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    
    # Find all divisors of n
    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:  # Avoid counting sqrt(n) twice
                divisors.append(n // i)
    
    # Calculate the sum of k-th powers
    return sum(d**k for d in divisors)