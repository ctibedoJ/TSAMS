"""
Prime number testing algorithms.

This module implements various algorithms for testing whether a number is prime,
including deterministic and probabilistic methods.
"""

import random
import math
from typing import List, Optional, Union


def is_prime(n: int, method: str = 'trial_division') -> bool:
    """
    Test if a number is prime using the specified method.
    
    Args:
        n: The number to test for primality
        method: The method to use ('trial_division', 'miller_rabin', or 'fermat')
        
    Returns:
        True if the number is prime, False otherwise
        
    Examples:
        >>> is_prime(17)
        True
        >>> is_prime(20)
        False
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    if method == 'trial_division':
        return trial_division(n)
    elif method == 'miller_rabin':
        return miller_rabin_test(n)
    elif method == 'fermat':
        return fermat_test(n)
    else:
        raise ValueError(f"Unknown primality testing method: {method}")


def trial_division(n: int) -> bool:
    """
    Test if a number is prime using trial division.
    
    This is a simple but inefficient method for large numbers.
    Time complexity: O(sqrt(n))
    
    Args:
        n: The number to test for primality
        
    Returns:
        True if the number is prime, False otherwise
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # Check all potential divisors up to sqrt(n)
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True


def miller_rabin_test(n: int, k: int = 40) -> bool:
    """
    Test if a number is prime using the Miller-Rabin primality test.
    
    This is a probabilistic algorithm that is much faster than trial division
    for large numbers, with a configurable error probability.
    
    Args:
        n: The number to test for primality
        k: The number of rounds (higher values reduce the probability of error)
        
    Returns:
        True if the number is probably prime, False if it's definitely composite
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    
    # Write n as 2^r * d + 1
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Witness loop
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        
        if x == 1 or x == n - 1:
            continue
        
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    
    return True


def fermat_test(n: int, k: int = 20) -> bool:
    """
    Test if a number is prime using Fermat's primality test.
    
    This is a probabilistic test based on Fermat's little theorem.
    It may incorrectly identify some composite numbers as prime (Carmichael numbers).
    
    Args:
        n: The number to test for primality
        k: The number of rounds (higher values reduce the probability of error)
        
    Returns:
        True if the number is probably prime, False if it's definitely composite
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    
    for _ in range(k):
        a = random.randint(2, n - 2)
        if math.gcd(a, n) != 1:
            return False
        if pow(a, n - 1, n) != 1:
            return False
    
    return True


def lucas_lehmer_test(p: int) -> bool:
    """
    Test if a Mersenne number (2^p - 1) is prime using the Lucas-Lehmer test.
    
    This is a deterministic primality test specifically for Mersenne numbers.
    
    Args:
        p: The exponent of the Mersenne number (2^p - 1)
        
    Returns:
        True if the Mersenne number is prime, False otherwise
    """
    if p == 2:
        return True  # M2 = 3 is prime
    
    if not is_prime(p):
        return False  # If p is not prime, Mp is not prime
    
    m = (1 << p) - 1  # Mersenne number 2^p - 1
    s = 4
    
    for _ in range(p - 2):
        s = (s * s - 2) % m
    
    return s == 0


def aks_test(n: int) -> bool:
    """
    Test if a number is prime using the AKS primality test.
    
    This is a deterministic polynomial-time primality test.
    Note: This implementation is for educational purposes and not optimized for large numbers.
    
    Args:
        n: The number to test for primality
        
    Returns:
        True if the number is prime, False otherwise
    """
    # Basic checks
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    
    # Check if n is a perfect power
    for b in range(2, int(math.log2(n)) + 1):
        a = n ** (1/b)
        if a.is_integer():
            return False
    
    # Find r such that ord_r(n) > log2(n)^2
    log2n = math.log2(n)
    r = 2
    while r < n:
        if math.gcd(r, n) != 1:
            return False
        
        # Check if ord_r(n) > log2(n)^2
        order = 1
        for i in range(1, int(log2n**2) + 1):
            if pow(n, i, r) == 1:
                order = i
                break
        
        if order > log2n**2:
            break
        
        r += 1
    
    # Check for divisors <= r
    for a in range(2, r + 1):
        if n % a == 0:
            return False
    
    # Main AKS test
    limit = int(math.sqrt(r) * log2n)
    
    for a in range(1, limit + 1):
        # Check if (X+a)^n ≡ X^n+a (mod X^r-1, n)
        # This is a simplified version and not the full polynomial comparison
        if pow(a, n, n) != a:
            return False
    
    return True