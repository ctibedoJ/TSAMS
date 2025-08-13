"""
Special prime number sequences and classifications.

This module provides functions for generating and working with special
types of prime numbers, such as twin primes, Mersenne primes, etc.
"""

import math
from typing import List, Optional, Tuple, Set

from ..algorithms.testing import is_prime, lucas_lehmer_test
from ..algorithms.generation import sieve_of_eratosthenes


def twin_primes(limit: int) -> List[Tuple[int, int]]:
    """
    Find all twin prime pairs up to a limit.
    
    Twin primes are pairs of primes that differ by 2.
    
    Args:
        limit: Upper bound for prime generation
        
    Returns:
        List of twin prime pairs (p, p+2)
    """
    # Generate primes up to the limit
    primes = set(sieve_of_eratosthenes(limit))
    
    # Find twin prime pairs
    pairs = []
    for p in primes:
        if p + 2 <= limit and p + 2 in primes:
            pairs.append((p, p + 2))
    
    return sorted(pairs)


def mersenne_primes(limit: int) -> List[int]:
    """
    Find Mersenne primes up to a limit.
    
    Mersenne primes are primes of the form 2^p - 1 where p is also prime.
    
    Args:
        limit: Upper bound for the exponent p
        
    Returns:
        List of Mersenne primes 2^p - 1 where p <= limit
    """
    # Generate prime exponents up to the limit
    prime_exponents = sieve_of_eratosthenes(limit)
    
    # Check each potential Mersenne prime
    mersenne = []
    for p in prime_exponents:
        m = (1 << p) - 1  # 2^p - 1
        
        # For small values, use direct primality testing
        if p < 10:
            if is_prime(m):
                mersenne.append(m)
        else:
            # For larger values, use the Lucas-Lehmer test
            if lucas_lehmer_test(p):
                mersenne.append(m)
    
    return mersenne


def fermat_primes(limit: int) -> List[int]:
    """
    Find Fermat primes up to a limit.
    
    Fermat primes are primes of the form 2^(2^n) + 1.
    
    Args:
        limit: Upper bound for n
        
    Returns:
        List of Fermat primes F_n = 2^(2^n) + 1 where n <= limit
    """
    fermat = []
    
    for n in range(limit + 1):
        # Calculate F_n = 2^(2^n) + 1
        f = (1 << (1 << n)) + 1
        
        # Check if F_n is prime
        if is_prime(f):
            fermat.append(f)
    
    return fermat


def sophie_germain_primes(limit: int) -> List[int]:
    """
    Find Sophie Germain primes up to a limit.
    
    A Sophie Germain prime p is a prime where 2p + 1 is also prime.
    
    Args:
        limit: Upper bound for prime generation
        
    Returns:
        List of Sophie Germain primes
    """
    # Generate primes up to the limit
    primes = set(sieve_of_eratosthenes(limit))
    
    # Find Sophie Germain primes
    sophie_germain = []
    for p in primes:
        if 2*p + 1 <= limit and 2*p + 1 in primes:
            sophie_germain.append(p)
    
    return sorted(sophie_germain)


def safe_primes(limit: int) -> List[int]:
    """
    Find safe primes up to a limit.
    
    A safe prime p is a prime where (p-1)/2 is also prime.
    
    Args:
        limit: Upper bound for prime generation
        
    Returns:
        List of safe primes
    """
    # Generate primes up to the limit
    primes = set(sieve_of_eratosthenes(limit))
    
    # Find safe primes
    safe = []
    for p in primes:
        if p > 2 and (p - 1) // 2 in primes:
            safe.append(p)
    
    return sorted(safe)


def chen_primes(limit: int) -> List[int]:
    """
    Find Chen primes up to a limit.
    
    A Chen prime p is a prime where p+2 is either prime or a product of two primes.
    
    Args:
        limit: Upper bound for prime generation
        
    Returns:
        List of Chen primes
    """
    # Generate primes up to the limit
    primes = set(sieve_of_eratosthenes(limit))
    
    # Find Chen primes
    chen = []
    for p in primes:
        if p + 2 > limit:
            continue
            
        if p + 2 in primes:
            # p+2 is prime
            chen.append(p)
        else:
            # Check if p+2 is a product of two primes
            is_semi_prime = False
            for i in range(2, int(math.sqrt(p + 2)) + 1):
                if (p + 2) % i == 0:
                    # i is a factor of p+2
                    j = (p + 2) // i
                    if i in primes and j in primes:
                        is_semi_prime = True
                        break
            
            if is_semi_prime:
                chen.append(p)
    
    return sorted(chen)


def circular_primes(limit: int) -> List[int]:
    """
    Find circular primes up to a limit.
    
    A circular prime is a prime number that remains prime under all rotations of its digits.
    
    Args:
        limit: Upper bound for prime generation
        
    Returns:
        List of circular primes
    """
    # Generate primes up to the limit
    primes = set(sieve_of_eratosthenes(limit))
    
    # Find circular primes
    circular = []
    for p in primes:
        if is_circular_prime(p, primes):
            circular.append(p)
    
    return sorted(circular)


def is_circular_prime(n: int, primes: Set[int]) -> bool:
    """
    Check if a number is a circular prime.
    
    Args:
        n: Number to check
        primes: Set of primes for lookup
        
    Returns:
        True if n is a circular prime, False otherwise
    """
    # Convert to string for easy rotation
    s = str(n)
    
    # Check all rotations
    for i in range(len(s)):
        rotation = int(s[i:] + s[:i])
        if rotation not in primes:
            return False
    
    return True


def sexy_primes(limit: int) -> List[Tuple[int, int]]:
    """
    Find sexy prime pairs up to a limit.
    
    Sexy primes are pairs of primes that differ by 6.
    
    Args:
        limit: Upper bound for prime generation
        
    Returns:
        List of sexy prime pairs (p, p+6)
    """
    # Generate primes up to the limit
    primes = set(sieve_of_eratosthenes(limit))
    
    # Find sexy prime pairs
    pairs = []
    for p in primes:
        if p + 6 <= limit and p + 6 in primes:
            pairs.append((p, p + 6))
    
    return sorted(pairs)