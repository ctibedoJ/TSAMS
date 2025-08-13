"""
Octonion implementation.

This module provides an implementation of octonions, which are 8-dimensional
hypercomplex numbers that play a crucial role in our mathematical framework.
"""

import numpy as np
from typing import List, Dict, Tuple, Union, Optional


class Octonion:
    """
    A class representing an octonion.
    
    Octonions are 8-dimensional hypercomplex numbers that form a non-associative
    division algebra. They extend the quaternions and play a crucial role in
    our mathematical framework.
    
    Attributes:
        components (np.ndarray): The 8 components of the octonion.
    """
    
    def __init__(self, components: List[float]):
        """
        Initialize an octonion with the given components.
        
        Args:
            components (List[float]): The 8 components of the octonion.
        
        Raises:
            ValueError: If the number of components is not 8.
        """
        if len(components) != 8:
            raise ValueError("Octonions must have exactly 8 components")
        
        self.components = np.array(components, dtype=float)
    
    @classmethod
    def from_quaternions(cls, q1: List[float], q2: List[float]) -> 'Octonion':
        """
        Create an octonion from two quaternions.
        
        Args:
            q1 (List[float]): The first quaternion.
            q2 (List[float]): The second quaternion.
        
        Returns:
            Octonion: The resulting octonion.
        
        Raises:
            ValueError: If either quaternion does not have exactly 4 components.
        """
        if len(q1) != 4 or len(q2) != 4:
            raise ValueError("Quaternions must have exactly 4 components")
        
        return cls(q1 + q2)
    
    @property
    def real(self) -> float:
        """
        Get the real part of the octonion.
        
        Returns:
            float: The real part of the octonion.
        """
        return self.components[0]
    
    @property
    def imaginary(self) -> np.ndarray:
        """
        Get the imaginary parts of the octonion.
        
        Returns:
            np.ndarray: The 7 imaginary parts of the octonion.
        """
        return self.components[1:]
    
    def conjugate(self) -> 'Octonion':
        """
        Compute the conjugate of the octonion.
        
        Returns:
            Octonion: The conjugate of the octonion.
        """
        conj_components = self.components.copy()
        conj_components[1:] = -conj_components[1:]
        return Octonion(conj_components)
    
    def norm_squared(self) -> float:
        """
        Compute the squared norm of the octonion.
        
        Returns:
            float: The squared norm of the octonion.
        """
        return np.sum(self.components**2)
    
    def norm(self) -> float:
        """
        Compute the norm of the octonion.
        
        Returns:
            float: The norm of the octonion.
        """
        return np.sqrt(self.norm_squared())
    
    def inverse(self) -> 'Octonion':
        """
        Compute the inverse of the octonion.
        
        Returns:
            Octonion: The inverse of the octonion.
        
        Raises:
            ValueError: If the octonion is zero (has zero norm).
        """
        norm_sq = self.norm_squared()
        if norm_sq < 1e-10:
            raise ValueError("Cannot compute inverse of zero octonion")
        
        conj = self.conjugate()
        inv_components = conj.components / norm_sq
        return Octonion(inv_components)
    
    def __add__(self, other: 'Octonion') -> 'Octonion':
        """
        Add two octonions.
        
        Args:
            other (Octonion): The octonion to add.
        
        Returns:
            Octonion: The sum of the two octonions.
        """
        return Octonion(self.components + other.components)
    
    def __sub__(self, other: 'Octonion') -> 'Octonion':
        """
        Subtract an octonion from this one.
        
        Args:
            other (Octonion): The octonion to subtract.
        
        Returns:
            Octonion: The difference of the two octonions.
        """
        return Octonion(self.components - other.components)
    
    def __mul__(self, other: Union['Octonion', float]) -> 'Octonion':
        """
        Multiply this octonion by another octonion or a scalar.
        
        Args:
            other (Union[Octonion, float]): The octonion or scalar to multiply by.
        
        Returns:
            Octonion: The product of the two octonions or the scalar product.
        """
        if isinstance(other, (int, float)):
            return Octonion(self.components * other)
        
        # Octonion multiplication is non-associative and defined by the Fano plane
        # This is a simplified implementation
        a, b = self.components, other.components
        
        c = np.zeros(8)
        c[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3] - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7]
        c[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[4] + a[3]*b[7] - a[4]*b[2] + a[5]*b[6] - a[6]*b[5] - a[7]*b[3]
        c[2] = a[0]*b[2] - a[1]*b[4] + a[2]*b[0] + a[3]*b[5] + a[4]*b[1] - a[5]*b[3] + a[6]*b[7] - a[7]*b[6]
        c[3] = a[0]*b[3] - a[1]*b[7] - a[2]*b[5] + a[3]*b[0] + a[4]*b[6] + a[5]*b[2] - a[6]*b[4] + a[7]*b[1]
        c[4] = a[0]*b[4] + a[1]*b[2] - a[2]*b[1] - a[3]*b[6] + a[4]*b[0] + a[5]*b[7] + a[6]*b[3] - a[7]*b[5]
        c[5] = a[0]*b[5] - a[1]*b[6] + a[2]*b[3] - a[3]*b[2] - a[4]*b[7] + a[5]*b[0] + a[6]*b[1] + a[7]*b[4]
        c[6] = a[0]*b[6] + a[1]*b[5] - a[2]*b[7] + a[3]*b[4] - a[4]*b[3] - a[5]*b[1] + a[6]*b[0] + a[7]*b[2]
        c[7] = a[0]*b[7] + a[1]*b[3] + a[2]*b[6] - a[3]*b[1] + a[4]*b[5] - a[5]*b[4] - a[6]*b[2] + a[7]*b[0]
        
        return Octonion(c)
    
    def __rmul__(self, scalar: float) -> 'Octonion':
        """
        Multiply this octonion by a scalar from the right.
        
        Args:
            scalar (float): The scalar to multiply by.
        
        Returns:
            Octonion: The scalar product.
        """
        return Octonion(self.components * scalar)
    
    def __truediv__(self, other: Union['Octonion', float]) -> 'Octonion':
        """
        Divide this octonion by another octonion or a scalar.
        
        Args:
            other (Union[Octonion, float]): The octonion or scalar to divide by.
        
        Returns:
            Octonion: The quotient.
        
        Raises:
            ValueError: If dividing by zero.
        """
        if isinstance(other, (int, float)):
            if abs(other) < 1e-10:
                raise ValueError("Division by zero")
            return Octonion(self.components / other)
        
        return self * other.inverse()
    
    def to_matrix(self) -> np.ndarray:
        """
        Convert the octonion to its matrix representation.
        
        Returns:
            np.ndarray: The 8x8 matrix representation of the octonion.
        """
        # This is a simplified implementation
        # In a complete implementation, this would compute the actual matrix representation
        a = self.components
        
        matrix = np.zeros((8, 8))
        
        # Fill the matrix according to the octonion multiplication rules
        # This is a placeholder implementation
        for i in range(8):
            matrix[i, i] = a[0]
        
        return matrix
    
    def to_cyclotomic_field(self, conductor: int = 168) -> Dict[int, float]:
        """
        Convert the octonion to an element of a cyclotomic field.
        
        Args:
            conductor (int): The conductor of the cyclotomic field (default: 168).
        
        Returns:
            Dict[int, float]: The cyclotomic field element.
        """
        # This is a simplified implementation
        # In a complete implementation, this would compute the actual cyclotomic field element
        element = {}
        for i, comp in enumerate(self.components):
            if abs(comp) > 1e-10:
                element[i] = comp
        
        return element
    
    def __str__(self) -> str:
        """
        Return a string representation of the octonion.
        
        Returns:
            str: A string representation of the octonion.
        """
        terms = []
        basis = ['1', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7']
        
        for i, comp in enumerate(self.components):
            if abs(comp) > 1e-10:
                if i == 0:
                    terms.append(f"{comp}")
                else:
                    terms.append(f"{comp}{basis[i]}")
        
        if not terms:
            return "0"
        
        return " + ".join(terms)
    
    def __repr__(self) -> str:
        """
        Return a string representation of the octonion.
        
        Returns:
            str: A string representation of the octonion.
        """
        return f"Octonion({self.components.tolist()})"