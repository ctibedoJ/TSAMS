"""
Möbius Pairing Implementation

This module implements Möbius strip pairings to model the forward/backward
momentum operators in normed Euclidean space.
"""

import numpy as np
import scipy.linalg as la

class MobiusPairing:
    """
    Implementation of Möbius Pairings used in the TIBEDO Framework.
    
    Möbius pairings model the forward/backward momentum operators in normed Euclidean space.
    """
    
    def __init__(self):
        """Initialize the MobiusPairing object."""
        self.pairing_matrix = None
        self.transvector_generator = None
        
    def create_pairing(self, point_a, point_b):
        """
        Create a Möbius pairing between two points.
        
        Args:
            point_a (tuple): The first point (x1, y1).
            point_b (tuple): The second point (x2, y2).
            
        Returns:
            numpy.ndarray: The Möbius pairing matrix.
        """
        x1, y1 = point_a
        x2, y2 = point_b
        
        # Create a 2x2 Möbius transformation matrix
        # The matrix must have determinant 1 to be a valid Möbius transformation
        alpha = x1
        beta = y1
        gamma = x2
        delta = y2
        
        # Adjust to ensure determinant is 1
        determinant = alpha * delta - beta * gamma
        if determinant == 0:
            # Avoid singularity
            alpha += 1
            delta += 1
            determinant = alpha * delta - beta * gamma
            
        # Scale to get determinant 1
        scale_factor = 1.0 / np.sqrt(abs(determinant))
        alpha *= scale_factor
        beta *= scale_factor
        gamma *= scale_factor
        delta *= scale_factor
        
        # Create the pairing matrix
        self.pairing_matrix = np.array([[alpha, beta], [gamma, delta]])
        
        # Initialize the transvector generator
        self.transvector_generator = TransvectorGenerator(self.pairing_matrix)
        
        return self.pairing_matrix
        
    def apply_pairing(self, point):
        """
        Apply the Möbius pairing to a point.
        
        Args:
            point (tuple): The point (x, y) to transform.
            
        Returns:
            tuple: The transformed point.
        """
        if self.pairing_matrix is None:
            raise ValueError("Pairing matrix has not been created yet.")
            
        x, y = point
        point_vector = np.array([x, y])
        
        # Apply the Möbius transformation
        transformed_vector = self.pairing_matrix @ point_vector
        
        return tuple(transformed_vector)
        
    def generate_transvector(self, t):
        """
        Generate a transvector at parameter t.
        
        Args:
            t (float): The parameter value (0 <= t <= 1).
            
        Returns:
            numpy.ndarray: The transvector.
        """
        if self.transvector_generator is None:
            raise ValueError("Transvector generator has not been initialized.")
            
        return self.transvector_generator.generate(t)
        
    def compute_pairing_invariant(self):
        """
        Compute the invariant of the Möbius pairing.
        
        Returns:
            float: The pairing invariant.
        """
        if self.pairing_matrix is None:
            raise ValueError("Pairing matrix has not been created yet.")
            
        # The trace of the matrix is an invariant of the Möbius transformation
        return np.trace(self.pairing_matrix)
        
class TransvectorGenerator:
    """
    Generator for transvectors across paired structures.
    """
    
    def __init__(self, pairing_matrix):
        """
        Initialize the TransvectorGenerator.
        
        Args:
            pairing_matrix (numpy.ndarray): The Möbius pairing matrix.
        """
        self.pairing_matrix = pairing_matrix
        self.eigenvalues, self.eigenvectors = la.eig(pairing_matrix)
        
    def generate(self, t):
        """
        Generate a transvector at parameter t.
        
        Args:
            t (float): The parameter value (0 <= t <= 1).
            
        Returns:
            numpy.ndarray: The transvector.
        """
        # Interpolate between the eigenvectors based on t
        if len(self.eigenvalues) >= 2:
            v1 = self.eigenvectors[:, 0]
            v2 = self.eigenvectors[:, 1]
            
            # Create a linear combination of the eigenvectors
            transvector = (1 - t) * v1 + t * v2
            
            # Normalize the transvector
            norm = la.norm(transvector)
            if norm > 0:
                transvector = transvector / norm
                
            return transvector
        else:
            # If there's only one eigenvector, return it
            return self.eigenvectors[:, 0]
            
    def compute_transvector_field(self, num_points=10):
        """
        Compute a field of transvectors.
        
        Args:
            num_points (int): The number of points in the field.
            
        Returns:
            list: A list of transvectors.
        """
        return [self.generate(t / (num_points - 1)) for t in range(num_points)]