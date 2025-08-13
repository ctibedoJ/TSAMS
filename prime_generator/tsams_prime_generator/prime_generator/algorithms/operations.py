"""
Prime number operations and utilities.

This module provides various operations related to prime numbers,
such as finding the next or previous prime, factorization, and more.
"""

import math
import bisect
from typing import List, Dict, Tuple, Optional

from .testing import is_prime
from .generation import sieve_of_eratosthenes


def next_prime(n: int) -> int:
    """
    Find the smallest prime number greater than n.
    
    Args:
        n: The starting number
        
    Returns:
        The next prime number after n
        
    Examples:
        >>> next_prime(10)
        11
        >>> next_prime(20)
        23
    """
    if n < 2:
        return 2
    
    # Start with the next odd number after n
    candidate = n + 1 + (n % 2)
    
    # Check candidates until we find a prime
    while not is_prime(candidate):
        candidate += 2
    
    return candidate


def prev_prime(n: int) -> Optional[int]:
    """
    Find the largest prime number less than n.
    
    Args:
        n: The starting number
        
    Returns:
        The previous prime number before n, or None if n <= 2
        
    Examples:
        >>> prev_prime(10)
        7
        >>> prev_prime(20)
        19
    """
    if n <= 2:
        return None
    
    if n == 3:
        return 2
    
    # Start with the previous odd number before n
    candidate = n - 1 - ((n - 1) % 2)
    
    # Check candidates until we find a prime
    while candidate >= 2 and not is_prime(candidate):
        candidate -= 2
    
    return candidate if candidate >= 2 else None


def prime_factors(n: int) -> List[int]:
    """
    Find all prime factors of a number.
    
    Args:
        n: The number to factorize
        
    Returns:
        List of prime factors (with repetition for multiple occurrences)
        
    Examples:
        >>> prime_factors(60)
        [2, 2, 3, 5]
    """
    if n <= 1:
        return []
    
    factors = []
    
    # Extract factors of 2
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    
    # Extract odd prime factors
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 2
    
    # If n is a prime greater than 2
    if n > 2:
        factors.append(n)
    
    return factors


def prime_factorization(n: int) -> Dict[int, int]:
    """
    Find the prime factorization of a number as a dictionary of prime factors and their exponents.
    
    Args:
        n: The number to factorize
        
    Returns:
        Dictionary mapping prime factors to their exponents
        
    Examples:
        >>> prime_factorization(60)
        {2: 2, 3: 1, 5: 1}
    """
    factors = prime_factors(n)
    factorization = {}
    
    for factor in factors:
        if factor in factorization:
            factorization[factor] += 1
        else:
            factorization[factor] = 1
    
    return factorization


def nth_prime(n: int) -> int:
    """
    Find the nth prime number.
    
    Args:
        n: The position in the sequence of primes (1-indexed)
        
    Returns:
        The nth prime number
        
    Examples:
        >>> nth_prime(1)
        2
        >>> nth_prime(10)
        29
    """
    if n <= 0:
        raise ValueError("Position must be a positive integer")
    
    if n == 1:
        return 2
    
    # Use the prime number theorem to estimate an upper bound
    if n <= 6:
        limit = 14  # Enough for the first 6 primes
    else:
        limit = int(n * (math.log(n) + math.log(math.log(n))))
    
    # Generate primes up to the limit
    primes = sieve_of_eratosthenes(limit)
    
    # If we didn't get enough primes, increase the limit and try again
    while len(primes) < n:
        limit *= 2
        primes = sieve_of_eratosthenes(limit)
    
    return primes[n - 1]


def prime_counting_function(n: int) -> int:
    """
    Count the number of primes less than or equal to n.
    
    This is the prime counting function π(n).
    
    Args:
        n: The upper limit
        
    Returns:
        The number of primes less than or equal to n
        
    Examples:
        >>> prime_counting_function(10)
        4
        >>> prime_counting_function(100)
        25
    """
    if n < 2:
        return 0
    
    primes = sieve_of_eratosthenes(n)
    return len(primes)


def coprime_to(n: int, limit: int) -> List[int]:
    """
    Find all numbers up to a limit that are coprime to n.
    
    Args:
        n: The number to check coprimality against
        limit: The upper limit for the search
        
    Returns:
        List of numbers up to limit that are coprime to n
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    
    result = []
    for i in range(1, limit + 1):
        if math.gcd(i, n) == 1:
            result.append(i)
    
    return result


def prime_gap(n: int) -> Tuple[int, int, int]:
    """
    Find the gap between a prime and the next prime, along with the two primes.
    
    Args:
        n: The starting number
        
    Returns:
        Tuple of (gap_size, prime, next_prime)
        
    Examples:
        >>> prime_gap(7)
        (4, 7, 11)
    """
    if n < 2:
        return (0, 2, 2)
    
    # Find the first prime >= n
    if not is_prime(n):
        n = next_prime(n)
    
    # Find the next prime
    next_p = next_prime(n + 1)
    
    return (next_p - n, n, next_p)