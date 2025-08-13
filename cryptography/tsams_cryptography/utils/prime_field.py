"""
Prime Field implementation.

This module provides an implementation of prime fields,
which are used in the ECDLP solver.
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from .finite_field import FiniteField, FieldElement


class PrimeField(FiniteField):
    """
    A class representing a prime field.
    
    A prime field is a finite field of order p, where p is a prime number.
    
    Attributes:
        order (int): The order of the field.
        is_prime (bool): Whether the order of the field is prime.
    """
    
    def __init__(self, order: int):
        """
        Initialize a prime field.
        
        Args:
            order (int): The order of the field.
        
        Raises:
            ValueError: If the order is not prime.
        """
        super().__init__(order)
        
        if not self.is_prime:
            raise ValueError(f"Order {order} is not prime")
    
    def primitive_element(self) -> FieldElement:
        """
        Find a primitive element of the field.
        
        A primitive element is an element whose powers generate all non-zero elements of the field.
        
        Returns:
            FieldElement: A primitive element of the field.
        """
        # The order of the multiplicative group is p - 1
        group_order = self.order - 1
        
        # Compute the prime factorization of the group order
        factors = self._prime_factors(group_order)
        distinct_factors = set(factors)
        
        # Compute the quotients group_order / factor for each distinct prime factor
        quotients = [group_order // factor for factor in distinct_factors]
        
        # Try each element of the field
        for i in range(2, self.order):
            element = self.element(i)
            
            # Check if the element is a primitive element
            is_primitive = True
            for quotient in quotients:
                if (element ** quotient).value == 1:
                    is_primitive = False
                    break
            
            if is_primitive:
                return element
        
        # This should never happen for a prime field
        raise RuntimeError("Failed to find a primitive element")
    
    def legendre_symbol(self, a: Union[int, FieldElement]) -> int:
        """
        Compute the Legendre symbol (a/p).
        
        The Legendre symbol is defined as:
        (a/p) = 0 if a ≡ 0 (mod p)
        (a/p) = 1 if a is a quadratic residue modulo p
        (a/p) = -1 if a is a quadratic non-residue modulo p
        
        Args:
            a (int or FieldElement): The element.
        
        Returns:
            int: The Legendre symbol.
        """
        if isinstance(a, FieldElement):
            a = a.value
        
        a = a % self.order
        
        if a == 0:
            return 0
        
        # Use Euler's criterion: a^((p-1)/2) ≡ (a/p) (mod p)
        result = pow(a, (self.order - 1) // 2, self.order)
        
        if result == self.order - 1:
            return -1
        
        return result
    
    def sqrt(self, a: Union[int, FieldElement]) -> Optional[FieldElement]:
        """
        Compute the square root of an element in the field.
        
        Args:
            a (int or FieldElement): The element.
        
        Returns:
            FieldElement or None: The square root of the element, or None if the element
                is not a quadratic residue.
        """
        if isinstance(a, FieldElement):
            a = a.value
        
        a = a % self.order
        
        # Check if the element is a quadratic residue
        if self.legendre_symbol(a) != 1:
            return None
        
        # Special case for p ≡ 3 (mod 4)
        if self.order % 4 == 3:
            # In this case, the square root is a^((p+1)/4) mod p
            exponent = (self.order + 1) // 4
            return self.element(pow(a, exponent, self.order))
        
        # General case using the Tonelli-Shanks algorithm
        # Factor out the largest power of 2 from p - 1
        q = self.order - 1
        s = 0
        while q % 2 == 0:
            q //= 2
            s += 1
        
        # Find a quadratic non-residue
        z = 2
        while self.legendre_symbol(z) != -1:
            z += 1
        
        # Initialize variables
        m = s
        c = pow(z, q, self.order)
        t = pow(a, q, self.order)
        r = pow(a, (q + 1) // 2, self.order)
        
        # Main loop
        while t != 1:
            # Find the smallest i such that t^(2^i) ≡ 1 (mod p)
            i = 0
            t_i = t
            while t_i != 1:
                t_i = (t_i * t_i) % self.order
                i += 1
                if i >= m:
                    # a is not a quadratic residue
                    return None
            
            # Compute b = c^(2^(m-i-1)) mod p
            b = pow(c, 2 ** (m - i - 1), self.order)
            
            # Update variables
            m = i
            c = (b * b) % self.order
            t = (t * c) % self.order
            r = (r * b) % self.order
        
        return self.element(r)
    
    def __str__(self) -> str:
        """
        Return a string representation of the field.
        
        Returns:
            str: A string representation of the field.
        """
        return f"PrimeField({self.order})"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the field.
        
        Returns:
            str: A string representation of the field.
        """
        return self.__str__()