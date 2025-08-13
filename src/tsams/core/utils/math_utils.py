"""
Mathematical utilities for the TSAMS Core package.

This module provides various mathematical utilities used throughout the TSAMS Core package.
"""

import numpy as np
import sympy
from typing import List, Tuple, Union, Optional

def is_prime(n: int) -> bool:
    """
    Check if a number is prime.
    
    Args:
        n (int): The number to check.
        
    Returns:
        bool: True if the number is prime, False otherwise.
    """
    return sympy.isprime(n)

def prime_factors(n: int) -> List[int]:
    """
    Get the prime factors of a number.
    
    Args:
        n (int): The number to factorize.
        
    Returns:
        List[int]: The prime factors of the number.
    """
    return list(sympy.factorint(n).keys())

def gcd(a: int, b: int) -> int:
    """
    Compute the greatest common divisor of two numbers.
    
    Args:
        a (int): The first number.
        b (int): The second number.
        
    Returns:
        int: The greatest common divisor.
    """
    return sympy.gcd(a, b)

def lcm(a: int, b: int) -> int:
    """
    Compute the least common multiple of two numbers.
    
    Args:
        a (int): The first number.
        b (int): The second number.
        
    Returns:
        int: The least common multiple.
    """
    return sympy.lcm(a, b)

def euler_totient(n: int) -> int:
    """
    Compute Euler's totient function φ(n).
    
    Args:
        n (int): The number.
        
    Returns:
        int: The value of φ(n).
    """
    return sympy.totient(n)

def mobius_mu(n: int) -> int:
    """
    Compute the Möbius function μ(n).
    
    Args:
        n (int): The number.
        
    Returns:
        int: The value of μ(n).
    """
    return sympy.mobius(n)

def primitive_root(n: int) -> Optional[int]:
    """
    Find a primitive root modulo n if one exists.
    
    Args:
        n (int): The modulus.
        
    Returns:
        Optional[int]: A primitive root modulo n, or None if none exists.
    """
    try:
        return sympy.primitive_root(n)
    except:
        return None

def complex_modulus(z: complex) -> float:
    """
    Compute the modulus of a complex number.
    
    Args:
        z (complex): The complex number.
        
    Returns:
        float: The modulus of the complex number.
    """
    return abs(z)

def complex_argument(z: complex) -> float:
    """
    Compute the argument of a complex number.
    
    Args:
        z (complex): The complex number.
        
    Returns:
        float: The argument of the complex number in radians.
    """
    return np.angle(z)

def roots_of_unity(n: int) -> List[complex]:
    """
    Compute the nth roots of unity.
    
    Args:
        n (int): The order.
        
    Returns:
        List[complex]: The nth roots of unity.
    """
    return [np.exp(2j * np.pi * k / n) for k in range(n)]

def cyclotomic_polynomial_coefficients(n: int) -> List[int]:
    """
    Compute the coefficients of the nth cyclotomic polynomial.
    
    Args:
        n (int): The order.
        
    Returns:
        List[int]: The coefficients of the nth cyclotomic polynomial.
    """
    poly = sympy.cyclotomic_poly(n, sympy.Symbol('x'))
    return [int(c) for c in poly.all_coeffs()]
