"""
420-Root Möbius Structure implementation.

This module provides a comprehensive implementation of the 420-root Möbius structure,
which is a special case of prime indexed Möbius transformations with profound
mathematical significance, as described in Chapter 16 of the textbook.
"""

import numpy as np
import sympy
from typing import List, Dict, Tuple, Union, Optional, Set

from .moebius_transformation import MoebiusTransformation
from .prime_indexed_moebius import PrimeIndexedMoebiusTransformation


class Root420Structure:
    """
    A class representing the 420-root Möbius structure.
    
    The 420-root Möbius structure is defined as the prime indexed Möbius structure
    where the structural parameter n = 420. The coefficients of each transformation
    M_p ∈ M_{420} are given by specific trigonometric functions of the prime index.
    
    Attributes:
        transformations (Dict[int, PrimeIndexedMoebiusTransformation]): A dictionary
            mapping prime indices to their corresponding transformations.
        primes (List[int]): The list of 81 primes less than 420.
        n (int): The structural parameter (420).
    """
    
    def __init__(self):
        """
        Initialize the 420-root Möbius structure.
        
        This creates all 81 prime indexed Möbius transformations in the structure.
        """
        # Set the structural parameter
        self.n = 420
        
        # Generate all primes less than 420
        self.primes = list(sympy.primerange(1, 420))
        
        # Create a transformation for each prime
        self._transformations = {}
        for p in self.primes:
            self._transformations[p] = PrimeIndexedMoebiusTransformation(p, self.n)
    
    def get_transformation(self, p: int) -> PrimeIndexedMoebiusTransformation:
        """
        Get the transformation corresponding to a prime index.
        
        Args:
            p (int): The prime index.
            
        Returns:
            PrimeIndexedMoebiusTransformation: The corresponding transformation.
            
        Raises:
            ValueError: If p is not a prime less than 420.
        """
        if p not in self._transformations:
            raise ValueError(f"{p} is not a prime less than 420")
        
        return self._transformations[p]
    
    def compose(self, p: int, q: int) -> MoebiusTransformation:
        """
        Compose two transformations in the 420-root structure.
        
        Args:
            p (int): The first prime index.
            q (int): The second prime index.
            
        Returns:
            MoebiusTransformation: The composition of the two transformations.
        """
        return self._transformations[p].compose(self._transformations[q])
    
    def commutator(self, p: int, q: int) -> MoebiusTransformation:
        """
        Compute the commutator [M_p, M_q] = M_p M_q M_p^{-1} M_q^{-1}.
        
        Args:
            p (int): The first prime index.
            q (int): The second prime index.
            
        Returns:
            MoebiusTransformation: The commutator of the two transformations.
        """
        return self._transformations[p].commutator(self._transformations[q])
    
    def is_commuting_pair(self, p: int, q: int) -> bool:
        """
        Check if two transformations commute.
        
        Two transformations M_p and M_q commute if and only if p ≡ q (mod 420).
        
        Args:
            p (int): The first prime index.
            q (int): The second prime index.
            
        Returns:
            bool: True if the transformations commute, False otherwise.
        """
        return p % 420 == q % 420
    
    def diamond_operation(self, p: int, q: int) -> int:
        """
        Compute the diamond operation p ◇ q between two prime indices.
        
        The diamond operation is defined as:
        p ◇ q = r iff exp(2πip/420) * exp(2πiq/420) = exp(2πir/420) mod 1
        
        For the 420-root structure, this simplifies to:
        p ◇ q = (p + q) mod 420
        
        Args:
            p (int): The first prime index.
            q (int): The second prime index.
            
        Returns:
            int: The result of the diamond operation.
        """
        return (p + q) % 420
    
    def fixed_points(self, p: int) -> List[complex]:
        """
        Compute the fixed points of a transformation in the 420-root structure.
        
        Args:
            p (int): The prime index.
            
        Returns:
            List[complex]: The fixed points of the transformation.
        """
        return self._transformations[p].fixed_points()
    
    def energy(self, p: int) -> float:
        """
        Compute the energy of a transformation in the 420-root structure.
        
        Args:
            p (int): The prime index.
            
        Returns:
            float: The energy value.
        """
        return self._transformations[p].energy()
    
    def energy_spectrum(self) -> List[float]:
        """
        Compute the energy spectrum of the 420-root structure.
        
        This is the list of energy values for all 81 transformations in the structure,
        sorted in ascending order.
        
        Returns:
            List[float]: The energy spectrum.
        """
        return sorted([self.energy(p) for p in self.primes])
        
    def total_energy(self) -> float:
        """
        Compute the total energy of the 420-root system.
        
        This is the sum of energies over all 81 transformations in the structure.
        
        Returns:
            float: The total energy.
        """
        return sum(self.energy(p) for p in self.primes)
    
    def orbit(self, z: complex, p: int, max_iterations: int = 1000) -> List[complex]:
        """
        Compute the orbit of a point under repeated application of a transformation.
        
        Args:
            z (complex): The initial point.
            p (int): The prime index of the transformation.
            max_iterations (int): The maximum number of iterations (default: 1000).
            
        Returns:
            List[complex]: The orbit of the point.
        """
        orbit = [z]
        transformation = self._transformations[p]
        
        for _ in range(max_iterations):
            z = transformation.apply(z)
            if z is None:  # z = ∞
                orbit.append(None)
                break
            orbit.append(z)
            
            # Check for convergence or cycling
            if len(orbit) > 2:
                if abs(orbit[-1] - orbit[-2]) < 1e-10:
                    break
                for i in range(len(orbit) - 2):
                    if abs(orbit[-1] - orbit[i]) < 1e-10:
                        break
        
        return orbit
    
    def is_elliptic(self, p: int) -> bool:
        """
        Check if a transformation in the 420-root structure is elliptic.
        
        Args:
            p (int): The prime index.
            
        Returns:
            bool: True if the transformation is elliptic, False otherwise.
        """
        return self._transformations[p].is_elliptic()
    
    def is_parabolic(self, p: int) -> bool:
        """
        Check if a transformation in the 420-root structure is parabolic.
        
        Args:
            p (int): The prime index.
            
        Returns:
            bool: True if the transformation is parabolic, False otherwise.
        """
        return self._transformations[p].is_parabolic()
    
    def is_hyperbolic(self, p: int) -> bool:
        """
        Check if a transformation in the 420-root structure is hyperbolic.
        
        Args:
            p (int): The prime index.
            
        Returns:
            bool: True if the transformation is hyperbolic, False otherwise.
        """
        return self._transformations[p].is_hyperbolic()
    
    def classify_transformation(self, p: int) -> str:
        """
        Classify a transformation in the 420-root structure.
        
        Args:
            p (int): The prime index.
            
        Returns:
            str: The classification of the transformation ('elliptic', 'parabolic', or 'hyperbolic').
        """
        if self.is_elliptic(p):
            return 'elliptic'
        elif self.is_parabolic(p):
            return 'parabolic'
        elif self.is_hyperbolic(p):
            return 'hyperbolic'
        else:
            return 'loxodromic'
    
    def get_all_classifications(self) -> Dict[str, List[int]]:
        """
        Get the classification of all transformations in the 420-root structure.
        
        Returns:
            Dict[str, List[int]]: A dictionary mapping classifications to lists of prime indices.
        """
        classifications = {'elliptic': [], 'parabolic': [], 'hyperbolic': [], 'loxodromic': []}
        
        for p in self.primes:
            classification = self.classify_transformation(p)
            classifications[classification].append(p)
        
        return classifications
    
    def get_number_theoretic_properties(self) -> Dict[str, Union[int, List[int], Dict[int, int]]]:
        """
        Get the number-theoretic properties of the 420-root structure.
        
        Returns:
            Dict[str, Union[int, List[int], Dict[int, int]]]: A dictionary containing various
                number-theoretic properties of the structure.
        """
        properties = {}
        
        # Factorization of 420
        properties['factorization'] = {2: 2, 3: 1, 5: 1, 7: 1}
        
        # Number of divisors
        properties['num_divisors'] = 24
        
        # List of divisors
        properties['divisors'] = [1, 2, 3, 4, 5, 6, 7, 10, 12, 14, 15, 20, 21, 28, 30, 35, 42, 60, 70, 84, 105, 140, 210, 420]
        
        # Euler's totient value
        properties['euler_totient'] = 96
        
        # Number of primes less than 420
        properties['num_primes'] = len(self.primes)
        
        # List of primes less than 420
        properties['primes'] = self.primes
        
        return properties
    
    @property
    def transformations(self) -> List[PrimeIndexedMoebiusTransformation]:
        """
        Get the list of all transformations in the 420-root structure.
        
        Returns:
            List[PrimeIndexedMoebiusTransformation]: The list of transformations.
        """
        return list(self._transformations.values())
    
    def __len__(self) -> int:
        """
        Get the number of transformations in the 420-root structure.
        
        Returns:
            int: The number of transformations.
        """
        return len(self._transformations)
    
    def __str__(self) -> str:
        """
        Return a string representation of the 420-root structure.
        
        Returns:
            str: A string representation of the structure.
        """
        return f"420-Root Möbius Structure with {len(self)} transformations"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the 420-root structure.
        
        Returns:
            str: A string representation of the structure.
        """
        return "Root420Structure()"