"""
Utility functions for classical quantum computations.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the angle between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Angle in radians
    """
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    return np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))

def is_prime(n: int) -> bool:
    """
    Check if a number is prime.
    
    Args:
        n: Number to check
        
    Returns:
        True if n is prime, False otherwise
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def next_prime(n: int) -> int:
    """
    Find the next prime number greater than or equal to n.
    
    Args:
        n: Starting number
        
    Returns:
        Next prime number
    """
    if n <= 1:
        return 2
    
    prime = n
    found = False
    
    while not found:
        prime += 1 if prime % 2 == 0 else 2
        if is_prime(prime):
            found = True
    
    return prime

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean Algorithm.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Tuple of (gcd, x, y) such that gcd = ax + by
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
        a: Number to invert
        m: Modulus
        
    Returns:
        Modular multiplicative inverse
    """
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError(f"Modular inverse does not exist for {a} mod {m}")
    else:
        return x % m

def chinese_remainder_theorem(remainders: List[int], moduli: List[int]) -> int:
    """
    Solve a system of congruences using the Chinese Remainder Theorem.
    
    Args:
        remainders: List of remainders
        moduli: List of moduli
        
    Returns:
        Solution to the system of congruences
    """
    if len(remainders) != len(moduli):
        raise ValueError("Number of remainders must equal number of moduli")
    
    # Compute product of all moduli
    prod = 1
    for m in moduli:
        prod *= m
    
    # Compute partial products and their inverses
    result = 0
    for i in range(len(moduli)):
        p = prod // moduli[i]
        result += remainders[i] * p * mod_inverse(p, moduli[i])
    
    return result % prod

def matrix_to_field_elements(matrix: np.ndarray, characteristic: int) -> List[int]:
    """
    Convert a matrix to a list of field elements.
    
    Args:
        matrix: Input matrix
        characteristic: Field characteristic
        
    Returns:
        List of field elements
    """
    # Flatten matrix and convert to field elements
    return [int(x) % characteristic for x in matrix.flatten()]

def field_elements_to_matrix(elements: List[int], shape: Tuple[int, ...]) -> np.ndarray:
    """
    Convert a list of field elements to a matrix.
    
    Args:
        elements: List of field elements
        shape: Shape of the output matrix
        
    Returns:
        Matrix of field elements
    """
    # Convert to numpy array and reshape
    return np.array(elements).reshape(shape)