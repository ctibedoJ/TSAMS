"""
Prime Indexed Möbius Transformation implementation.

This module provides a specialized implementation of Möbius transformations
that are indexed by prime numbers, forming the 420-root structure described
in Chapter 16 of the textbook.
"""

import numpy as np
import sympy
import cmath
from typing import List, Dict, Tuple, Union, Optional

from .moebius_transformation import MoebiusTransformation


class PrimeIndexedMoebiusTransformation(MoebiusTransformation):
    """
    A class representing a prime indexed Möbius transformation.
    
    Prime indexed Möbius transformations are a specialized class of transformations
    where the indexing follows prime number sequences, forming a mathematical framework
    for understanding complex structural relationships in projective space.
    
    Attributes:
        prime (int): The prime index of the transformation.
        n (int): The structural parameter (default: 420).
        angle (float): The angle parameter 2πp/n.
    """
    
    def __init__(self, p: int, n: int = 420):
        """
        Initialize a prime indexed Möbius transformation.
        
        Args:
            p (int): The prime index.
            n (int): The structural parameter (default: 420).
            
        Raises:
            ValueError: If p is not a prime number.
        """
        # Verify that p is prime
        if not sympy.isprime(p):
            raise ValueError(f"{p} is not a prime number")
        
        self.prime = p
        self.n = n
        self.angle = 2 * np.pi * p / n
        
        # Compute the coefficients based on the prime index
        a = np.cos(self.angle)
        b = np.sin(self.angle)
        c = -np.sin(self.angle)
        d = np.cos(self.angle)
        
        # Initialize the parent class
        super().__init__(a, b, c, d)
    
    @property
    def prime_index(self) -> int:
        """
        Get the prime index of this transformation.
        
        Returns:
            int: The prime index.
        """
        return self.prime
    
    def energy(self) -> float:
        """
        Compute the energy of this transformation.
        
        The energy quantization in the 420-root Möbius system follows the pattern:
        E(M_p) = (p^2/420) * (1 + sum_{k=1}^{∞} ((-1)^k/k!) * ((2πp/420)^{2k}))
        
        For small values of p/420, this can be approximated as:
        E(M_p) ≈ p^2/420
        
        Returns:
            float: The energy value.
        """
        # For small values of p/420, we can use the approximation
        if self.prime < self.n / 10:
            return self.prime**2 / self.n
        
        # For larger values, we need to use the full formula
        # We'll truncate the infinite sum after a few terms
        energy = self.prime**2 / self.n
        
        # Add correction terms
        x = 2 * np.pi * self.prime / self.n
        correction = 0
        factorial = 1
        sign = -1
        
        # Add terms until they become negligibly small
        for k in range(1, 10):
            factorial *= k
            sign *= -1
            term = sign * (x**(2*k)) / factorial
            correction += term
            
            # Stop if the term is very small
            if abs(term) < 1e-10:
                break
        
        energy *= (1 + correction)
        return energy
    
    def fixed_points(self) -> List[complex]:
        """
        Compute the fixed points of this prime indexed Möbius transformation.
        
        For a transformation in the 420-root structure, the fixed points satisfy
        the quadratic equation: c_p z^2 + (d_p - a_p)z - b_p = 0
        
        Returns:
            List[complex]: The fixed points of the transformation.
        """
        return super().fixed_points()
    
    def is_in_420_root_structure(self) -> bool:
        """
        Check if this transformation is part of the 420-root structure.
        
        A transformation is part of the 420-root structure if its prime index
        is less than 420.
        
        Returns:
            bool: True if the transformation is in the 420-root structure, False otherwise.
        """
        return self.prime < 420 and self.n == 420
    
    def diamond_operation(self, other: 'PrimeIndexedMoebiusTransformation') -> int:
        """
        Compute the diamond operation p ◇ q between two prime indices.
        
        The diamond operation is defined as:
        p ◇ q = r iff exp(2πip/420) * exp(2πiq/420) = exp(2πir/420) mod 1
        
        For the 420-root structure, this simplifies to:
        p ◇ q = (p + q) mod 420
        
        Args:
            other (PrimeIndexedMoebiusTransformation): The other transformation.
            
        Returns:
            int: The result of the diamond operation.
        """
        if self.n != other.n:
            raise ValueError("Both transformations must have the same structural parameter")
        
        return (self.prime + other.prime) % self.n
    
    def commutator(self, other: 'PrimeIndexedMoebiusTransformation') -> MoebiusTransformation:
        """
        Compute the commutator [M_p, M_q] = M_p M_q M_p^{-1} M_q^{-1}.
        
        The commutator is non-trivial if and only if p and q are not congruent modulo n.
        
        Args:
            other (PrimeIndexedMoebiusTransformation): The other transformation.
            
        Returns:
            MoebiusTransformation: The commutator of the two transformations.
        """
        # Compute M_p M_q M_p^{-1} M_q^{-1}
        return self.compose(other).compose(self.inverse()).compose(other.inverse())
    
    @classmethod
    def generate_420_root_structure(cls) -> Dict[int, 'PrimeIndexedMoebiusTransformation']:
        """
        Generate all 81 transformations in the 420-root Möbius structure.
        
        The 420-root Möbius structure contains exactly π(420) = 81 distinct transformations,
        corresponding to the primes less than 420.
        
        Returns:
            Dict[int, PrimeIndexedMoebiusTransformation]: A dictionary mapping prime indices
                to their corresponding transformations.
        """
        transformations = {}
        
        # Generate all primes less than 420
        primes = list(sympy.primerange(1, 420))
        
        # Create a transformation for each prime
        for p in primes:
            transformations[p] = cls(p, 420)
        
        return transformations
    
    def __str__(self) -> str:
        """
        Return a string representation of the prime indexed Möbius transformation.
        
        Returns:
            str: A string representation of the transformation.
        """
        return f"M_{self.prime}(z) = ({self.a}z + {self.b})/({self.c}z + {self.d})"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the prime indexed Möbius transformation.
        
        Returns:
            str: A string representation of the transformation.
        """
        return f"PrimeIndexedMoebiusTransformation({self.prime}, {self.n})"