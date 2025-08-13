"""
Number Theory Utilities.

This module provides utility functions for number theory operations that are
used throughout the prime indexed Möbius transformation framework.
"""

import numpy as np
import sympy
from typing import List, Dict, Tuple, Union, Optional, Callable, Any


def is_prime(n: int) -> bool:
    """
    Check if a number is prime.
    
    Args:
        n (int): The number to check.
            
    Returns:
        bool: True if the number is prime, False otherwise.
    """
    return sympy.isprime(n)


def prime_factors(n: int) -> Dict[int, int]:
    """
    Compute the prime factorization of a number.
    
    Args:
        n (int): The number to factorize.
            
    Returns:
        Dict[int, int]: A dictionary mapping prime factors to their exponents.
    """
    factors = {}
    
    # Trial division
    for i in range(2, int(np.sqrt(n)) + 1):
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
    
    # If n is a prime number greater than the square root
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    
    return factors


def euler_totient(n: int) -> int:
    """
    Compute Euler's totient function φ(n).
    
    The totient function φ(n) counts the positive integers up to n that are
    relatively prime to n.
    
    Args:
        n (int): The number.
            
    Returns:
        int: The value of φ(n).
    """
    if n == 1:
        return 1
    
    # Compute using the formula: φ(n) = n * ∏_{p|n} (1 - 1/p)
    factors = prime_factors(n)
    result = n
    
    for p in factors:
        result *= (1 - 1/p)
    
    return int(result)


def mobius_function(n: int) -> int:
    """
    Compute the Möbius function μ(n).
    
    The Möbius function is defined as:
    μ(n) = 0 if n has a squared prime factor
    μ(n) = 1 if n is a square-free positive integer with an even number of prime factors
    μ(n) = -1 if n is a square-free positive integer with an odd number of prime factors
    
    Args:
        n (int): The number.
            
    Returns:
        int: The value of μ(n).
    """
    if n == 1:
        return 1
    
    # Check if n has a squared prime factor
    factors = prime_factors(n)
    if any(exp > 1 for exp in factors.values()):
        return 0
    
    # n is square-free, so μ(n) = (-1)^k where k is the number of prime factors
    return (-1) ** len(factors)


def gcd(a: int, b: int) -> int:
    """
    Compute the greatest common divisor of two numbers.
    
    Args:
        a (int): The first number.
        b (int): The second number.
            
    Returns:
        int: The greatest common divisor.
    """
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """
    Compute the least common multiple of two numbers.
    
    Args:
        a (int): The first number.
        b (int): The second number.
            
    Returns:
        int: The least common multiple.
    """
    return a * b // gcd(a, b)


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Compute the extended greatest common divisor of two numbers.
    
    This function returns (gcd, x, y) such that gcd = ax + by.
    
    Args:
        a (int): The first number.
        b (int): The second number.
            
    Returns:
        Tuple[int, int, int]: The GCD and the coefficients x and y.
    """
    if a == 0:
        return (b, 0, 1)
    else:
        gcd, x, y = extended_gcd(b % a, a)
        return (gcd, y - (b // a) * x, x)


def mod_inverse(a: int, m: int) -> int:
    """
    Compute the modular multiplicative inverse of a modulo m.
    
    Args:
        a (int): The number.
        m (int): The modulus.
            
    Returns:
        int: The modular multiplicative inverse.
            
    Raises:
        ValueError: If the modular inverse does not exist.
    """
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError(f"Modular inverse does not exist (gcd({a}, {m}) = {gcd})")
    else:
        return x % m


def chinese_remainder_theorem(remainders: List[int], moduli: List[int]) -> int:
    """
    Solve a system of congruences using the Chinese remainder theorem.
    
    Args:
        remainders (List[int]): The remainders.
        moduli (List[int]): The moduli.
            
    Returns:
        int: The solution.
            
    Raises:
        ValueError: If the moduli are not pairwise coprime.
    """
    if len(remainders) != len(moduli):
        raise ValueError("The number of remainders must equal the number of moduli")
    
    # Check if the moduli are pairwise coprime
    for i in range(len(moduli)):
        for j in range(i+1, len(moduli)):
            if gcd(moduli[i], moduli[j]) != 1:
                raise ValueError(f"Moduli {moduli[i]} and {moduli[j]} are not coprime")
    
    # Compute the product of all moduli
    N = 1
    for m in moduli:
        N *= m
    
    # Compute the solution
    result = 0
    for i in range(len(moduli)):
        a_i = remainders[i]
        m_i = moduli[i]
        N_i = N // m_i
        result += a_i * N_i * mod_inverse(N_i, m_i)
    
    return result % N


def legendre_symbol(a: int, p: int) -> int:
    """
    Compute the Legendre symbol (a/p).
    
    The Legendre symbol is defined as:
    (a/p) = 0 if a is divisible by p
    (a/p) = 1 if a is a quadratic residue modulo p
    (a/p) = -1 if a is a quadratic non-residue modulo p
    
    Args:
        a (int): The number.
        p (int): The prime modulus.
            
    Returns:
        int: The value of the Legendre symbol.
            
    Raises:
        ValueError: If p is not a prime number.
    """
    if not is_prime(p):
        raise ValueError(f"{p} is not a prime number")
    
    a = a % p
    
    if a == 0:
        return 0
    
    # Use Euler's criterion: (a/p) ≡ a^((p-1)/2) (mod p)
    result = pow(a, (p - 1) // 2, p)
    
    if result == p - 1:
        return -1
    else:
        return result


def jacobi_symbol(a: int, n: int) -> int:
    """
    Compute the Jacobi symbol (a/n).
    
    The Jacobi symbol is a generalization of the Legendre symbol to composite moduli.
    
    Args:
        a (int): The number.
        n (int): The modulus (must be odd and positive).
            
    Returns:
        int: The value of the Jacobi symbol.
            
    Raises:
        ValueError: If n is not odd and positive.
    """
    if n <= 0 or n % 2 == 0:
        raise ValueError(f"{n} is not odd and positive")
    
    a = a % n
    result = 1
    
    while a != 0:
        # Extract factors of 2 from a
        t = 0
        while a % 2 == 0:
            a //= 2
            t += 1
        
        # Apply the quadratic reciprocity law for factors of 2
        if t % 2 == 1 and (n % 8 == 3 or n % 8 == 5):
            result = -result
        
        # Apply the quadratic reciprocity law
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        
        # Swap a and n
        a, n = n % a, a
    
    return result if n == 1 else 0


def cyclotomic_polynomial(n: int, x: Union[int, float, complex]) -> complex:
    """
    Evaluate the nth cyclotomic polynomial at x.
    
    The nth cyclotomic polynomial is the minimal polynomial of a primitive
    nth root of unity over the rational numbers.
    
    Args:
        n (int): The index of the cyclotomic polynomial.
        x (Union[int, float, complex]): The point at which to evaluate the polynomial.
            
    Returns:
        complex: The value of the polynomial at x.
    """
    if n == 1:
        return x - 1
    
    # Use the recursive formula: Φ_n(x) = ∏_{d|n} (x^(n/d) - 1)^μ(d)
    result = 1
    for d in range(1, n + 1):
        if n % d == 0:
            result *= (x**(n//d) - 1) ** mobius_function(d)
    
    return result


def primitive_root(n: int) -> Optional[int]:
    """
    Find a primitive root modulo n.
    
    A primitive root modulo n is an integer g such that for any integer a
    coprime to n, there exists an integer k such that g^k ≡ a (mod n).
    
    Args:
        n (int): The modulus.
            
    Returns:
        Optional[int]: A primitive root modulo n, or None if no primitive root exists.
    """
    # Check if n has a primitive root
    if n == 2 or n == 4:
        return 1
    
    if n % 2 == 0:
        n_half = n // 2
        if n_half % 2 == 0:
            return None  # No primitive root exists
        
        # Check if n_half is a power of an odd prime
        factors = prime_factors(n_half)
        if len(factors) != 1:
            return None  # No primitive root exists
    
    # Compute Euler's totient function φ(n)
    phi = euler_totient(n)
    
    # Find the prime factors of φ(n)
    phi_factors = prime_factors(phi)
    phi_prime_factors = list(phi_factors.keys())
    
    # Try each number from 2 to n-1
    for g in range(2, n):
        if gcd(g, n) != 1:
            continue
        
        # Check if g is a primitive root
        is_primitive_root = True
        for p in phi_prime_factors:
            if pow(g, phi // p, n) == 1:
                is_primitive_root = False
                break
        
        if is_primitive_root:
            return g
    
    return None  # No primitive root exists


def discrete_log(a: int, b: int, m: int) -> Optional[int]:
    """
    Compute the discrete logarithm of b to the base a modulo m.
    
    This function finds x such that a^x ≡ b (mod m).
    
    Args:
        a (int): The base.
        b (int): The number.
        m (int): The modulus.
            
    Returns:
        Optional[int]: The discrete logarithm, or None if it does not exist.
    """
    # Ensure a and m are coprime
    if gcd(a, m) != 1:
        return None
    
    # Compute the order of a modulo m
    order = 1
    a_pow = a % m
    while a_pow != 1:
        a_pow = (a_pow * a) % m
        order += 1
        if order > m:
            return None  # a does not have a finite order modulo m
    
    # Use Shanks' baby-step giant-step algorithm
    n = int(np.sqrt(order)) + 1
    
    # Precompute a^j for j = 0, 1, ..., n-1
    baby_steps = {}
    a_j = 1
    for j in range(n):
        baby_steps[a_j] = j
        a_j = (a_j * a) % m
    
    # Compute a^(-n)
    a_inv = mod_inverse(a, m)
    a_minus_n = pow(a_inv, n, m)
    
    # Compute b * (a^(-n))^i for i = 0, 1, ..., n-1
    giant_step = b % m
    for i in range(n):
        if giant_step in baby_steps:
            return (i * n + baby_steps[giant_step]) % order
        giant_step = (giant_step * a_minus_n) % m
    
    return None  # No solution exists


def is_quadratic_residue(a: int, p: int) -> bool:
    """
    Check if a number is a quadratic residue modulo p.
    
    A number a is a quadratic residue modulo p if there exists an x such that
    x^2 ≡ a (mod p).
    
    Args:
        a (int): The number.
        p (int): The prime modulus.
            
    Returns:
        bool: True if a is a quadratic residue modulo p, False otherwise.
            
    Raises:
        ValueError: If p is not a prime number.
    """
    if not is_prime(p):
        raise ValueError(f"{p} is not a prime number")
    
    # Use the Legendre symbol
    return legendre_symbol(a, p) == 1


def tonelli_shanks(a: int, p: int) -> Optional[int]:
    """
    Solve the congruence x^2 ≡ a (mod p) using the Tonelli-Shanks algorithm.
    
    Args:
        a (int): The number.
        p (int): The prime modulus.
            
    Returns:
        Optional[int]: A solution to the congruence, or None if no solution exists.
            
    Raises:
        ValueError: If p is not a prime number.
    """
    if not is_prime(p):
        raise ValueError(f"{p} is not a prime number")
    
    # Check if a is a quadratic residue modulo p
    if legendre_symbol(a, p) != 1:
        return None
    
    # Handle the special case p = 2
    if p == 2:
        return a % 2
    
    # Handle the case p ≡ 3 (mod 4)
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    
    # Find the largest power of 2 that divides p - 1
    s = 0
    q = p - 1
    while q % 2 == 0:
        q //= 2
        s += 1
    
    # Find a quadratic non-residue modulo p
    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1
    
    # Initialize variables
    m = s
    c = pow(z, q, p)
    t = pow(a, q, p)
    r = pow(a, (q + 1) // 2, p)
    
    # Main loop
    while t != 1:
        # Find the smallest i such that t^(2^i) ≡ 1 (mod p)
        i = 0
        t_i = t
        while t_i != 1:
            t_i = (t_i * t_i) % p
            i += 1
            if i == m:
                return None  # No solution exists
        
        # Compute b = c^(2^(m-i-1)) mod p
        b = pow(c, 2**(m-i-1), p)
        
        # Update variables
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p
    
    return r


def carmichael_lambda(n: int) -> int:
    """
    Compute the Carmichael function λ(n).
    
    The Carmichael function λ(n) is the smallest positive integer m such that
    a^m ≡ 1 (mod n) for all a coprime to n.
    
    Args:
        n (int): The number.
            
    Returns:
        int: The value of λ(n).
    """
    if n == 1:
        return 1
    
    # Compute the prime factorization of n
    factors = prime_factors(n)
    
    # Compute λ(n) for each prime power
    lambda_values = []
    for p, e in factors.items():
        if p == 2 and e >= 3:
            lambda_values.append(2**(e-2))
        elif p == 2:
            lambda_values.append(2**(e-1))
        else:
            lambda_values.append((p-1) * p**(e-1))
    
    # Compute the least common multiple of the λ values
    result = lambda_values[0]
    for val in lambda_values[1:]:
        result = lcm(result, val)
    
    return result


def is_carmichael_number(n: int) -> bool:
    """
    Check if a number is a Carmichael number.
    
    A Carmichael number is a composite number n such that a^(n-1) ≡ 1 (mod n)
    for all a coprime to n.
    
    Args:
        n (int): The number.
            
    Returns:
        bool: True if n is a Carmichael number, False otherwise.
    """
    if n <= 1 or is_prime(n):
        return False
    
    # Check if n is square-free
    factors = prime_factors(n)
    if any(e > 1 for e in factors.values()):
        return False
    
    # Check if p-1 divides n-1 for all prime factors p of n
    for p in factors:
        if (n - 1) % (p - 1) != 0:
            return False
    
    return True


def continued_fraction(x: float, max_terms: int = 10) -> List[int]:
    """
    Compute the continued fraction expansion of a real number.
    
    Args:
        x (float): The real number.
        max_terms (int): The maximum number of terms to compute (default: 10).
            
    Returns:
        List[int]: The continued fraction expansion.
    """
    expansion = []
    
    for _ in range(max_terms):
        a = int(x)
        expansion.append(a)
        
        # Check if we've reached a terminating expansion
        if x == a:
            break
        
        x = 1 / (x - a)
    
    return expansion


def convergents(expansion: List[int]) -> List[Tuple[int, int]]:
    """
    Compute the convergents of a continued fraction expansion.
    
    Args:
        expansion (List[int]): The continued fraction expansion.
            
    Returns:
        List[Tuple[int, int]]: The convergents as pairs (numerator, denominator).
    """
    if not expansion:
        return []
    
    # Initialize the first two convergents
    p = [expansion[0], expansion[0] * expansion[1] + 1]
    q = [1, expansion[1]]
    
    # Compute the remaining convergents
    for i in range(2, len(expansion)):
        p.append(expansion[i] * p[i-1] + p[i-2])
        q.append(expansion[i] * q[i-1] + q[i-2])
    
    return list(zip(p, q))


def diophantine_approximation(x: float, epsilon: float) -> Tuple[int, int]:
    """
    Find a rational approximation p/q of a real number x with |x - p/q| < ε/q.
    
    Args:
        x (float): The real number.
        epsilon (float): The approximation quality parameter.
            
    Returns:
        Tuple[int, int]: The rational approximation as (p, q).
    """
    # Compute the continued fraction expansion
    expansion = continued_fraction(x, max_terms=100)
    
    # Compute the convergents
    conv = convergents(expansion)
    
    # Find the first convergent that satisfies the approximation quality
    for p, q in conv:
        if abs(x - p/q) < epsilon/q:
            return (p, q)
    
    # If no convergent satisfies the condition, return the last one
    return conv[-1]