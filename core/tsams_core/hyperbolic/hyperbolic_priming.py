"""
Hyperbolic Priming Transformations implementation.

This module provides a comprehensive implementation of hyperbolic priming transformations
described in Chapter 16 of the textbook, which provide a mechanism for energy
quantization within the mathematical framework.
"""

import numpy as np
import sympy
from typing import List, Dict, Tuple, Union, Optional, Callable, Any

from ..moebius.moebius_transformation import MoebiusTransformation
from ..moebius.prime_indexed_moebius import PrimeIndexedMoebiusTransformation


class HyperbolicPrimingTransformation:
    """
    A class representing a hyperbolic priming transformation.
    
    A hyperbolic priming transformation is a mapping that modifies each Möbius
    transformation by applying a hyperbolic scaling factor related to a prime p.
    
    Attributes:
        p (int): The prime parameter.
        hyperbolic_angle (Callable[[int], float]): A function that computes the
            hyperbolic angle parameter for a given prime q.
    """
    
    def __init__(self, p: int):
        """
        Initialize a hyperbolic priming transformation with the given prime.
        
        Args:
            p (int): The prime parameter.
                
        Raises:
            ValueError: If p is not a prime number.
        """
        # Verify that p is prime
        if not sympy.isprime(p):
            raise ValueError(f"{p} is not a prime number")
        
        self.p = p
        
        # Define the hyperbolic angle parameter function
        self.hyperbolic_angle = lambda q: (p * q) / (20 * np.log(p * q))
    
    def apply(self, transformation: MoebiusTransformation) -> MoebiusTransformation:
        """
        Apply this hyperbolic priming transformation to another transformation.
        
        Args:
            transformation (MoebiusTransformation): The transformation to modify.
                
        Returns:
            MoebiusTransformation: The modified transformation.
        """
        # If the transformation is a prime indexed Möbius transformation,
        # we can use the prime index to compute the hyperbolic angle
        if isinstance(transformation, PrimeIndexedMoebiusTransformation):
            q = transformation.prime_index
            theta = self.hyperbolic_angle(q)
            
            # Create the hyperbolic matrices
            cosh_theta = np.cosh(theta)
            sinh_theta = np.sinh(theta)
            
            left_matrix = np.array([[cosh_theta, sinh_theta],
                                    [sinh_theta, cosh_theta]])
            
            right_matrix = np.array([[cosh_theta, -sinh_theta],
                                     [-sinh_theta, cosh_theta]])
            
            # Extract the original matrix
            original_matrix = np.array([[transformation.a, transformation.b],
                                        [transformation.c, transformation.d]])
            
            # Apply the hyperbolic priming transformation
            result_matrix = np.matmul(np.matmul(left_matrix, original_matrix), right_matrix)
            
            # Create a new Möbius transformation with the modified coefficients
            return MoebiusTransformation(result_matrix[0, 0], result_matrix[0, 1],
                                         result_matrix[1, 0], result_matrix[1, 1])
        else:
            # For a general Möbius transformation, we can't use the prime index
            # Instead, we'll use a default hyperbolic angle
            theta = 0.1  # Default value
            
            # Create the hyperbolic matrices
            cosh_theta = np.cosh(theta)
            sinh_theta = np.sinh(theta)
            
            left_matrix = np.array([[cosh_theta, sinh_theta],
                                    [sinh_theta, cosh_theta]])
            
            right_matrix = np.array([[cosh_theta, -sinh_theta],
                                     [-sinh_theta, cosh_theta]])
            
            # Extract the original matrix
            original_matrix = np.array([[transformation.a, transformation.b],
                                        [transformation.c, transformation.d]])
            
            # Apply the hyperbolic priming transformation
            result_matrix = np.matmul(np.matmul(left_matrix, original_matrix), right_matrix)
            
            # Create a new Möbius transformation with the modified coefficients
            return MoebiusTransformation(result_matrix[0, 0], result_matrix[0, 1],
                                         result_matrix[1, 0], result_matrix[1, 1])
    
    def coupling_strength(self) -> float:
        """
        Compute the coupling strength of this hyperbolic priming transformation.
        
        In the 420-root structure, each root receives a 1/20 coupling strength
        increment through the hyperbolic priming transformation according to the formula:
        κ_p = (1/20) · (p/ln p) · ∏_{q|p} (1 - 1/q)
        
        Returns:
            float: The coupling strength.
        """
        # Compute the product term
        product = 1
        for q in sympy.primefactors(self.p):
            product *= (1 - 1/q)
        
        # Compute the coupling strength
        return (1/20) * (self.p / np.log(self.p)) * product
    
    def mobius_function(self) -> int:
        """
        Compute the Möbius function μ(p) for the prime p.
        
        The Möbius function is defined as:
        μ(n) = 0 if n has a squared prime factor
        μ(n) = 1 if n is a square-free positive integer with an even number of prime factors
        μ(n) = -1 if n is a square-free positive integer with an odd number of prime factors
        
        For a prime p, μ(p) = -1.
        
        Returns:
            int: The value of the Möbius function at p.
        """
        return -1  # For a prime p, μ(p) = -1
    
    def __str__(self) -> str:
        """
        Return a string representation of the hyperbolic priming transformation.
        
        Returns:
            str: A string representation of the transformation.
        """
        return f"Hyperbolic Priming Transformation H_{self.p}"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the hyperbolic priming transformation.
        
        Returns:
            str: A string representation of the transformation.
        """
        return f"HyperbolicPrimingTransformation({self.p})"


class HyperbolicPrimingSystem:
    """
    A class representing a system of hyperbolic priming transformations.
    
    This class provides methods for working with multiple hyperbolic priming
    transformations and analyzing their collective properties.
    
    Attributes:
        transformations (Dict[int, HyperbolicPrimingTransformation]): A dictionary
            mapping prime indices to hyperbolic priming transformations.
    """
    
    def __init__(self, max_prime: int = 100):
        """
        Initialize a system of hyperbolic priming transformations.
        
        Args:
            max_prime (int): The maximum prime to include in the system (default: 100).
        """
        self.transformations = {}
        
        # Create a transformation for each prime up to max_prime
        for p in sympy.primerange(2, max_prime + 1):
            self.transformations[p] = HyperbolicPrimingTransformation(p)
    
    def get_transformation(self, p: int) -> HyperbolicPrimingTransformation:
        """
        Get the hyperbolic priming transformation for a prime p.
        
        Args:
            p (int): The prime parameter.
                
        Returns:
            HyperbolicPrimingTransformation: The hyperbolic priming transformation.
                
        Raises:
            ValueError: If p is not a prime in the system.
        """
        if p not in self.transformations:
            raise ValueError(f"{p} is not a prime in the system")
        
        return self.transformations[p]
    
    def apply(self, p: int, transformation: MoebiusTransformation) -> MoebiusTransformation:
        """
        Apply the hyperbolic priming transformation for prime p to another transformation.
        
        Args:
            p (int): The prime parameter.
            transformation (MoebiusTransformation): The transformation to modify.
                
        Returns:
            MoebiusTransformation: The modified transformation.
        """
        return self.get_transformation(p).apply(transformation)
    
    def coupling_strength(self, p: int) -> float:
        """
        Compute the coupling strength of the hyperbolic priming transformation for prime p.
        
        Args:
            p (int): The prime parameter.
                
        Returns:
            float: The coupling strength.
        """
        return self.get_transformation(p).coupling_strength()
    
    def total_coupling_strength(self) -> float:
        """
        Compute the total coupling strength of all hyperbolic priming transformations.
        
        Returns:
            float: The total coupling strength.
        """
        return sum(self.coupling_strength(p) for p in self.transformations)
    
    def coupling_strength_distribution(self) -> Dict[int, float]:
        """
        Compute the distribution of coupling strengths across all primes.
        
        Returns:
            Dict[int, float]: A dictionary mapping prime indices to coupling strengths.
        """
        return {p: self.coupling_strength(p) for p in self.transformations}
    
    def __len__(self) -> int:
        """
        Get the number of hyperbolic priming transformations in the system.
        
        Returns:
            int: The number of transformations.
        """
        return len(self.transformations)
    
    def __str__(self) -> str:
        """
        Return a string representation of the hyperbolic priming system.
        
        Returns:
            str: A string representation of the system.
        """
        return f"Hyperbolic Priming System with {len(self)} transformations"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the hyperbolic priming system.
        
        Returns:
            str: A string representation of the system.
        """
        return f"HyperbolicPrimingSystem(max_prime={max(self.transformations.keys())})"