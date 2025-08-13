"""
Prime number generation algorithms.

This module implements various algorithms for generating prime numbers,
optimized for different use cases and ranges.
"""

import numpy as np
from typing import List, Generator, Union, Optional


def sieve_of_eratosthenes(limit: int) -> List[int]:
    """
    Generate all prime numbers up to a given limit using the Sieve of Eratosthenes.
    
    This is one of the most efficient ways to generate a list of primes up to a moderate limit.
    Time complexity: O(n log log n)
    Space complexity: O(n)
    
    Args:
        limit: Upper bound for prime generation (inclusive)
        
    Returns:
        List of all prime numbers up to the limit
        
    Examples:
        >>> sieve_of_eratosthenes(20)
        [2, 3, 5, 7, 11, 13, 17, 19]
    """
    if limit < 2:
        return []
    
    # Initialize the sieve array
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0:2] = False  # 0 and 1 are not prime
    
    # Mark multiples of each prime as non-prime
    for i in range(2, int(np.sqrt(limit)) + 1):
        if sieve[i]:
            # Mark all multiples of i as non-prime
            sieve[i*i:limit+1:i] = False
    
    # Extract the primes from the sieve
    return list(np.nonzero(sieve)[0])


def sieve_of_atkin(limit: int) -> List[int]:
    """
    Generate all prime numbers up to a given limit using the Sieve of Atkin.
    
    This is a more complex but potentially more efficient algorithm than the Sieve of Eratosthenes
    for very large limits.
    
    Args:
        limit: Upper bound for prime generation (inclusive)
        
    Returns:
        List of all prime numbers up to the limit
        
    Examples:
        >>> sieve_of_atkin(20)
        [2, 3, 5, 7, 11, 13, 17, 19]
    """
    if limit < 2:
        return []
    
    # Initialize the sieve array
    sieve = np.zeros(limit + 1, dtype=bool)
    
    # Add 2 and 3 as known primes
    if limit >= 2:
        sieve[2] = True
    if limit >= 3:
        sieve[3] = True
    
    # Main part of the Sieve of Atkin
    for x in range(1, int(np.sqrt(limit)) + 1):
        for y in range(1, int(np.sqrt(limit)) + 1):
            # First quadratic: 4x² + y² = n
            n = 4 * x**2 + y**2
            if n <= limit and (n % 12 == 1 or n % 12 == 5):
                sieve[n] = not sieve[n]
            
            # Second quadratic: 3x² + y² = n
            n = 3 * x**2 + y**2
            if n <= limit and n % 12 == 7:
                sieve[n] = not sieve[n]
            
            # Third quadratic: 3x² - y² = n (when x > y)
            if x > y:
                n = 3 * x**2 - y**2
                if n <= limit and n % 12 == 11:
                    sieve[n] = not sieve[n]
    
    # Mark all multiples of squares as non-prime
    for i in range(5, int(np.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i**2, limit + 1, i**2):
                sieve[j] = False
    
    # Extract the primes from the sieve
    return list(np.nonzero(sieve)[0])


def segmented_sieve(limit: int, segment_size: int = 10**6) -> List[int]:
    """
    Generate primes up to a limit using a segmented sieve approach.
    
    This is useful for generating primes up to very large limits that might
    not fit in memory using a standard sieve.
    
    Args:
        limit: Upper bound for prime generation (inclusive)
        segment_size: Size of each segment to process
        
    Returns:
        List of all prime numbers up to the limit
    """
    if limit < 2:
        return []
    
    # Get small primes to use for sieving
    sqrt_limit = int(np.sqrt(limit))
    small_primes = sieve_of_eratosthenes(sqrt_limit)
    
    # Initialize result with small primes
    result = small_primes.copy()
    
    # Process each segment
    for low in range(sqrt_limit + 1, limit + 1, segment_size):
        high = min(low + segment_size - 1, limit)
        
        # Initialize segment sieve
        segment = np.ones(high - low + 1, dtype=bool)
        
        # Sieve the segment using small primes
        for prime in small_primes:
            # Find the first multiple of prime in the segment
            start = max(prime * prime, (low + prime - 1) // prime * prime)
            
            # Mark all multiples in this segment as non-prime
            for i in range(start, high + 1, prime):
                segment[i - low] = False
        
        # Collect primes from this segment
        for i in range(high - low + 1):
            if segment[i]:
                result.append(low + i)
    
    return result


def prime_range(start: int, end: int) -> List[int]:
    """
    Generate all prime numbers in a given range [start, end].
    
    Args:
        start: Lower bound of the range (inclusive)
        end: Upper bound of the range (inclusive)
        
    Returns:
        List of all prime numbers in the range
        
    Examples:
        >>> prime_range(10, 30)
        [11, 13, 17, 19, 23, 29]
    """
    if end < 2 or start > end:
        return []
    
    # Adjust start if needed
    start = max(2, start)
    
    # Generate primes up to end
    primes = sieve_of_eratosthenes(end)
    
    # Filter primes in the range
    return [p for p in primes if p >= start]


def prime_generator() -> Generator[int, None, None]:
    """
    Generate an infinite sequence of prime numbers.
    
    This is a generator function that yields prime numbers indefinitely.
    
    Yields:
        The next prime number in the sequence
        
    Examples:
        >>> gen = prime_generator()
        >>> [next(gen) for _ in range(5)]
        [2, 3, 5, 7, 11]
    """
    # Start with known primes
    yield 2
    yield 3
    
    # Initialize candidate and step
    candidate = 5
    step = 2
    
    # Continue indefinitely
    while True:
        if all(candidate % prime != 0 for prime in range(3, int(np.sqrt(candidate)) + 1, 2)):
            yield candidate
        
        # Move to next candidate (alternating +2, +4)
        candidate += step
        step = 6 - step  # Alternates between 2 and 4