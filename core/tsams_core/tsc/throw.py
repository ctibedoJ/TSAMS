"""
Throw Phase Implementation

The Throw phase initializes the computation with parameters derived from the elliptic curve
and the problem instance, setting up the initial state for the subsequent transformations.
"""

import numpy as np
import sympy as sp


class ThrowPhase:
    """
    Implementation of the Throw Phase of the TSC Algorithm.
    
    The Throw phase initializes the computation with parameters derived from the elliptic curve
    and the problem instance, setting up the initial state for the subsequent transformations.
    """
    
    def __init__(self):
        """Initialize the ThrowPhase object."""
        self.initialization_vector = None
        self.dimension = None
        self.prime_parameter = None
        self.throw_depth = None
    
    def compute_dimension_mapping(self, bit_length):
        """
        Compute the appropriate starting dimension for the given bit length.
        
        Args:
            bit_length (int): The bit length of the ECDLP instance.
            
        Returns:
            int: The appropriate starting dimension.
        """
        # According to Definition 5.2.2, the dimension mapping function D maps the ECDLP
        # instance to an appropriate starting dimension in the spinor reduction chain:
        # D(b) = 2^⌈log₂(b)⌉
        return 2 ** int(np.ceil(np.log2(bit_length)))
    
    def select_prime_parameter(self, bit_length, order):
        """
        Select the prime parameter for the TSC algorithm.
        
        Args:
            bit_length (int): The bit length of the ECDLP instance.
            order (int): The order of the base point P.
            
        Returns:
            int: The selected prime parameter.
        """
        # According to Definition 5.2.3, the prime parameter p is selected as:
        # p = NextPrime(b × log₂(n))
        target = bit_length * np.log2(order)
        return sp.nextprime(int(target))
    
    def calculate_throw_depth(self, bit_length, order):
        """
        Calculate the throw depth for the given bit length and order.
        
        Args:
            bit_length (int): The bit length of the ECDLP instance.
            order (int): The order of the base point P.
            
        Returns:
            int: The calculated throw depth.
        """
        # According to Definition 5.2.4, the throw depth td is calculated as:
        # td = p × log₁₀(n) × b/21
        p = self.select_prime_parameter(bit_length, order)
        return int(p * np.log10(order) * bit_length / 21)
    
    def construct_initialization_vector(self, P, Q, curve_params):
        """
        Construct the initialization vector for the TSC algorithm.
        
        Args:
            P (tuple): The coordinates of the base point P (x₁, y₁).
            Q (tuple): The coordinates of the point Q (x₂, y₂).
            curve_params (dict): The parameters of the elliptic curve.
                Should contain 'a', 'b', 'p', and 'n' (order of P).
                
        Returns:
            numpy.ndarray: The initialization vector.
        """
        # According to Definition 5.2.1, the initialization vector v₀ is constructed as:
        # v₀ = [x₁, y₁, x₂, y₂, a, b, p, n]
        x1, y1 = P
        x2, y2 = Q
        a = curve_params['a']
        b = curve_params['b']
        p = curve_params['p']
        n = curve_params['n']
        
        return np.array([x1, y1, x2, y2, a, b, p, n], dtype=np.int64)
    
    def initialize(self, P, Q, curve_params):
        """
        Initialize the Throw Phase with the given parameters.
        
        Args:
            P (tuple): The coordinates of the base point P (x₁, y₁).
            Q (tuple): The coordinates of the point Q (x₂, y₂).
            curve_params (dict): The parameters of the elliptic curve.
                Should contain 'a', 'b', 'p', and 'n' (order of P).
                
        Returns:
            dict: The initial state for the Shot Phase.
        """
        bit_length = int(np.ceil(np.log2(curve_params['n'])))
        
        # Compute the dimension mapping
        self.dimension = self.compute_dimension_mapping(bit_length)
        
        # Select the prime parameter
        self.prime_parameter = self.select_prime_parameter(bit_length, curve_params['n'])
        
        # Calculate the throw depth
        self.throw_depth = self.calculate_throw_depth(bit_length, curve_params['n'])
        
        # Construct the initialization vector
        self.initialization_vector = self.construct_initialization_vector(P, Q, curve_params)
        
        # Return the initial state for the Shot Phase
        return {
            'vector': self.initialization_vector,
            'dimension': self.dimension,
            'prime_parameter': self.prime_parameter,
            'throw_depth': self.throw_depth,
            'bit_length': bit_length,
            'transformation_count': 0
        }