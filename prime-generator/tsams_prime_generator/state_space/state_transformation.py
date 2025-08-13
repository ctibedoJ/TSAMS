"""
State Transformation implementation.

This module provides a comprehensive implementation of state transformations
as described in Chapter 17 of the textbook, which are essential for understanding
the dynamics and transformations within the mathematical framework.
"""

import numpy as np
import cmath
from typing import List, Dict, Tuple, Union, Optional, Callable, Any

from ..moebius.moebius_transformation import MoebiusTransformation
from ..moebius.prime_indexed_moebius import PrimeIndexedMoebiusTransformation


class StateTransformation:
    """
    A class representing a state transformation.
    
    A state transformation is a mapping T: S → S that takes one state to another.
    In our framework, these transformations are given by the prime indexed Möbius
    transformations M_p.
    
    Attributes:
        transformation (MoebiusTransformation): The underlying Möbius transformation.
        infinitesimal_generator (Callable): The vector field corresponding to the
            infinitesimal generator of the transformation.
    """
    
    def __init__(self, transformation: MoebiusTransformation):
        """
        Initialize a state transformation with the given Möbius transformation.
        
        Args:
            transformation (MoebiusTransformation): The underlying Möbius transformation.
        """
        self.transformation = transformation
        
        # Compute the infinitesimal generator
        self.infinitesimal_generator = self._compute_infinitesimal_generator()
    
    def _compute_infinitesimal_generator(self) -> Callable[[complex], complex]:
        """
        Compute the infinitesimal generator of the transformation.
        
        The infinitesimal generator is the vector field X defined by:
        X(f) = d/dt|_{t=0} (f ∘ T_t)
        
        For a Möbius transformation, this can be computed explicitly.
        
        Returns:
            Callable[[complex], complex]: The vector field corresponding to the
                infinitesimal generator.
        """
        # For a Möbius transformation (az + b)/(cz + d), the infinitesimal generator
        # can be expressed in terms of the coefficients
        a, b, c, d = self.transformation.a, self.transformation.b, self.transformation.c, self.transformation.d
        
        # For a prime indexed Möbius transformation, the infinitesimal generator
        # takes a specific form
        if isinstance(self.transformation, PrimeIndexedMoebiusTransformation):
            p = self.transformation.prime
            n = self.transformation.n
            
            def vector_field(z: complex) -> complex:
                """
                Compute the vector field at a point z.
                
                Args:
                    z (complex): The point.
                    
                Returns:
                    complex: The vector field at z.
                """
                return (2 * np.pi * 1j * p / n) * (1 - z**2) / 2
            
            return vector_field
        else:
            # For a general Möbius transformation, the infinitesimal generator
            # is more complex
            def vector_field(z: complex) -> complex:
                """
                Compute the vector field at a point z.
                
                Args:
                    z (complex): The point.
                    
                Returns:
                    complex: The vector field at z.
                """
                return (a * d - b * c) * (z**2 * c - z * (a - d) - b) / (c * z + d)**2
            
            return vector_field
    
    def apply(self, state: complex) -> complex:
        """
        Apply the transformation to a state.
        
        Args:
            state (complex): The state to transform.
            
        Returns:
            complex: The transformed state.
        """
        return self.transformation.apply(state)
    
    def compose(self, other: 'StateTransformation') -> 'StateTransformation':
        """
        Compose this transformation with another one.
        
        Args:
            other (StateTransformation): The other transformation.
            
        Returns:
            StateTransformation: The composition of the two transformations.
        """
        composed_transformation = self.transformation.compose(other.transformation)
        return StateTransformation(composed_transformation)
    
    def inverse(self) -> 'StateTransformation':
        """
        Compute the inverse of this transformation.
        
        Returns:
            StateTransformation: The inverse transformation.
        """
        inverse_transformation = self.transformation.inverse()
        return StateTransformation(inverse_transformation)
    
    def flow(self, state: complex, t: float) -> complex:
        """
        Compute the flow of the vector field at time t starting from a state.
        
        The flow is the solution of the differential equation:
        dz/dt = X(z)
        with initial condition z(0) = state.
        
        Args:
            state (complex): The initial state.
            t (float): The time parameter.
            
        Returns:
            complex: The state at time t.
        """
        # For a prime indexed Möbius transformation, the flow can be computed explicitly
        if isinstance(self.transformation, PrimeIndexedMoebiusTransformation):
            p = self.transformation.prime
            n = self.transformation.n
            angle = 2 * np.pi * p * t / n
            
            # The flow is a rotation on the Riemann sphere
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            # Apply the rotation matrix
            numerator = cos_angle * state + sin_angle
            denominator = -sin_angle * state + cos_angle
            
            if abs(denominator) < 1e-10:
                return None  # Point at infinity
            
            return numerator / denominator
        else:
            # For a general Möbius transformation, we need to solve the differential equation
            # numerically using Euler's method
            dt = 0.01
            steps = int(abs(t) / dt)
            sign = 1 if t >= 0 else -1
            
            current_state = state
            for _ in range(steps):
                # Compute the vector field at the current state
                vector = self.infinitesimal_generator(current_state)
                
                # Update the state using Euler's method
                current_state += sign * dt * vector
            
            return current_state
    
    def commutator(self, other: 'StateTransformation') -> 'StateTransformation':
        """
        Compute the commutator [T, S] = T ∘ S ∘ T^{-1} ∘ S^{-1}.
        
        Args:
            other (StateTransformation): The other transformation.
            
        Returns:
            StateTransformation: The commutator of the two transformations.
        """
        return self.compose(other).compose(self.inverse()).compose(other.inverse())
    
    def is_identity(self) -> bool:
        """
        Check if this transformation is the identity.
        
        Returns:
            bool: True if the transformation is the identity, False otherwise.
        """
        return self.transformation.is_identity()
    
    def trace(self) -> complex:
        """
        Compute the trace of the matrix representation of this transformation.
        
        Returns:
            complex: The trace of the matrix.
        """
        return self.transformation.trace()
    
    def energy(self) -> float:
        """
        Compute the energy of this transformation.
        
        For a prime indexed Möbius transformation, the energy is given by:
        E(M_p) = p^2/420 * (1 + sum_{k=1}^{∞} ((-1)^k/k!) * ((2πp/420)^{2k}))
        
        Returns:
            float: The energy value.
        """
        if isinstance(self.transformation, PrimeIndexedMoebiusTransformation):
            return self.transformation.energy()
        else:
            # For a general Möbius transformation, we define the energy as:
            # E(M) = (1/2) * tr(A^T A) - 1
            # where A is the matrix representation of M
            a, b, c, d = self.transformation.a, self.transformation.b, self.transformation.c, self.transformation.d
            matrix = np.array([[a, b], [c, d]], dtype=complex)
            energy = 0.5 * np.trace(np.matmul(matrix.conj().T, matrix)).real - 1
            return energy
    
    def __eq__(self, other: object) -> bool:
        """
        Check if this transformation is equal to another one.
        
        Args:
            other (object): The other object to compare with.
            
        Returns:
            bool: True if the transformations are equal, False otherwise.
        """
        if not isinstance(other, StateTransformation):
            return False
        
        return self.transformation == other.transformation
    
    def __str__(self) -> str:
        """
        Return a string representation of the state transformation.
        
        Returns:
            str: A string representation of the transformation.
        """
        return f"StateTransformation({self.transformation})"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the state transformation.
        
        Returns:
            str: A string representation of the transformation.
        """
        return f"StateTransformation({repr(self.transformation)})"


class PrimeIndexedStateTransformation(StateTransformation):
    """
    A class representing a prime indexed state transformation.
    
    Prime indexed state transformations are a specialized class of transformations
    where the indexing follows prime number sequences.
    
    Attributes:
        prime (int): The prime index of the transformation.
        n (int): The structural parameter (default: 420).
    """
    
    def __init__(self, p: int, n: int = 420):
        """
        Initialize a prime indexed state transformation.
        
        Args:
            p (int): The prime index.
            n (int): The structural parameter (default: 420).
        """
        # Create the underlying prime indexed Möbius transformation
        transformation = PrimeIndexedMoebiusTransformation(p, n)
        
        # Initialize the parent class
        super().__init__(transformation)
        
        # Store the prime index and structural parameter
        self.prime = p
        self.n = n
    
    @property
    def prime_index(self) -> int:
        """
        Get the prime index of this transformation.
        
        Returns:
            int: The prime index.
        """
        return self.prime
    
    def diamond_operation(self, other: 'PrimeIndexedStateTransformation') -> int:
        """
        Compute the diamond operation p ◇ q between two prime indices.
        
        Args:
            other (PrimeIndexedStateTransformation): The other transformation.
            
        Returns:
            int: The result of the diamond operation.
        """
        if self.n != other.n:
            raise ValueError("Both transformations must have the same structural parameter")
        
        return (self.prime + other.prime) % self.n
    
    def __str__(self) -> str:
        """
        Return a string representation of the prime indexed state transformation.
        
        Returns:
            str: A string representation of the transformation.
        """
        return f"T_{self.prime}"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the prime indexed state transformation.
        
        Returns:
            str: A string representation of the transformation.
        """
        return f"PrimeIndexedStateTransformation({self.prime}, {self.n})"