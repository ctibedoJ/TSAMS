"""
Möbius Pairing Implementation

This module implements the Möbius strip pairings that model the forward/backward
momentum operators in normed Euclidean space, enabling transvector formation
across paired structures.
"""

import numpy as np
import scipy.linalg as la
from scipy.special import gamma
import cmath
import sympy as sp
from sympy import symbols, Matrix, I, pi, exp


class MobiusPairing:
    """
    Implementation of Möbius Pairings used in the TIBEDO Framework.
    
    A Möbius pairing is a mathematical structure that connects two quantum states
    through a Möbius transformation, enabling the representation of forward/backward
    momentum operators.
    """
    
    def __init__(self, matrix=None):
        """
        Initialize the MobiusPairing object.
        
        Args:
            matrix (numpy.ndarray, optional): The 2x2 matrix representing the Möbius transformation.
                                           If None, a default identity-like matrix is used.
        """
        if matrix is None:
            # Default to identity-like matrix
            self.matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        else:
            # Ensure the matrix is 2x2
            if matrix.shape != (2, 2):
                raise ValueError("Möbius transformation matrix must be 2x2")
            
            # Normalize the matrix to have determinant 1
            det = np.linalg.det(matrix)
            if abs(det) < 1e-10:
                raise ValueError("Möbius transformation matrix must be invertible")
            
            self.matrix = matrix / np.sqrt(det)
        
        # Compute the inverse matrix
        self.inverse_matrix = self._compute_inverse()
        
        # Compute the fixed points
        self.fixed_points = self._compute_fixed_points()
        
        # Compute the invariant circle
        self.invariant_circle = self._compute_invariant_circle()
    
    def _compute_inverse(self):
        """
        Compute the inverse of the Möbius transformation matrix.
        
        Returns:
            numpy.ndarray: The inverse matrix.
        """
        a, b = self.matrix[0, 0], self.matrix[0, 1]
        c, d = self.matrix[1, 0], self.matrix[1, 1]
        
        # The inverse of [a b; c d] is [d -b; -c a] / det
        det = a * d - b * c
        
        return np.array([[d, -b], [-c, a]], dtype=complex) / det
    
    def _compute_fixed_points(self):
        """
        Compute the fixed points of the Möbius transformation.
        
        Returns:
            tuple: The fixed points (z1, z2).
        """
        a, b = self.matrix[0, 0], self.matrix[0, 1]
        c, d = self.matrix[1, 0], self.matrix[1, 1]
        
        # The fixed points satisfy az + b = z(cz + d)
        # This simplifies to cz^2 + (d-a)z - b = 0
        
        if abs(c) < 1e-10:
            # If c is approximately zero, there's one finite fixed point
            if abs(d - a) < 1e-10:
                # If d-a is also zero, there are no finite fixed points
                return (complex(np.inf), complex(np.inf))
            else:
                # One finite fixed point
                z = b / (a - d)
                return (z, complex(np.inf))
        else:
            # Two finite fixed points (quadratic formula)
            discriminant = (d - a)**2 + 4 * b * c
            z1 = ((a - d) + cmath.sqrt(discriminant)) / (2 * c)
            z2 = ((a - d) - cmath.sqrt(discriminant)) / (2 * c)
            return (z1, z2)
    
    def _compute_invariant_circle(self):
        """
        Compute the invariant circle of the Möbius transformation.
        
        Returns:
            tuple: The center and radius of the invariant circle, or None if there isn't one.
        """
        # Check if the transformation is elliptic (has an invariant circle)
        a, b = self.matrix[0, 0], self.matrix[0, 1]
        c, d = self.matrix[1, 0], self.matrix[1, 1]
        
        trace = a + d
        det = a * d - b * c
        
        # Compute the trace^2 / det
        discriminant = (trace * np.conj(trace)) / det
        
        if abs(discriminant - 4.0) < 1e-10:
            # Parabolic transformation - no invariant circle
            return None
        elif discriminant < 4.0:
            # Elliptic transformation - has an invariant circle
            
            # The center of the circle is the fixed point of the transformation
            z1, z2 = self.fixed_points
            
            if abs(z1) < abs(z2):
                center = z1
            else:
                center = z2
            
            # The radius depends on the specific transformation
            # This is a simplified calculation
            radius = abs(b / c) if abs(c) > 1e-10 else abs(b)
            
            return (center, radius)
        else:
            # Hyperbolic transformation - no invariant circle
            return None
    
    def apply(self, z):
        """
        Apply the Möbius transformation to a complex number.
        
        Args:
            z (complex): The complex number to transform.
            
        Returns:
            complex: The transformed complex number.
        """
        a, b = self.matrix[0, 0], self.matrix[0, 1]
        c, d = self.matrix[1, 0], self.matrix[1, 1]
        
        # Handle infinity
        if abs(z) > 1e10:
            return a / c if abs(c) > 1e-10 else complex(np.inf)
        
        # Apply the transformation (az + b) / (cz + d)
        denominator = c * z + d
        
        if abs(denominator) < 1e-10:
            return complex(np.inf)
        else:
            return (a * z + b) / denominator
    
    def apply_inverse(self, z):
        """
        Apply the inverse Möbius transformation to a complex number.
        
        Args:
            z (complex): The complex number to transform.
            
        Returns:
            complex: The transformed complex number.
        """
        a, b = self.inverse_matrix[0, 0], self.inverse_matrix[0, 1]
        c, d = self.inverse_matrix[1, 0], self.inverse_matrix[1, 1]
        
        # Handle infinity
        if abs(z) > 1e10:
            return a / c if abs(c) > 1e-10 else complex(np.inf)
        
        # Apply the transformation (az + b) / (cz + d)
        denominator = c * z + d
        
        if abs(denominator) < 1e-10:
            return complex(np.inf)
        else:
            return (a * z + b) / denominator
    
    def compose(self, other):
        """
        Compose this Möbius transformation with another one.
        
        Args:
            other (MobiusPairing): The other Möbius transformation.
            
        Returns:
            MobiusPairing: The composed transformation.
        """
        # Matrix multiplication gives the composition
        composed_matrix = np.matmul(self.matrix, other.matrix)
        
        return MobiusPairing(composed_matrix)
    
    def iterate(self, z, n):
        """
        Iterate the Möbius transformation n times on a complex number.
        
        Args:
            z (complex): The complex number to transform.
            n (int): The number of iterations.
            
        Returns:
            complex: The transformed complex number.
        """
        if n == 0:
            return z
        elif n < 0:
            # Use the inverse transformation for negative iterations
            return self.iterate_inverse(z, -n)
        else:
            # Apply the transformation n times
            result = z
            for _ in range(n):
                result = self.apply(result)
            return result
    
    def iterate_inverse(self, z, n):
        """
        Iterate the inverse Möbius transformation n times on a complex number.
        
        Args:
            z (complex): The complex number to transform.
            n (int): The number of iterations.
            
        Returns:
            complex: The transformed complex number.
        """
        if n == 0:
            return z
        elif n < 0:
            # Use the forward transformation for negative iterations
            return self.iterate(z, -n)
        else:
            # Apply the inverse transformation n times
            result = z
            for _ in range(n):
                result = self.apply_inverse(result)
            return result
    
    def create_edge_midpoint_edge_pattern(self):
        """
        Create an edge-midpoint-edge pattern Möbius transformation.
        
        Returns:
            MobiusPairing: The edge-midpoint-edge pattern transformation.
        """
        # The edge-midpoint-edge pattern is a specific Möbius transformation
        # that follows the sequence: edge point -> midpoint -> edge point -> original edge point
        
        # This is represented by the matrix [0 1; 1 0]^2 = [1 0; 0 1]
        pattern_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        pattern_matrix = np.matmul(pattern_matrix, pattern_matrix)  # Square it
        
        return MobiusPairing(pattern_matrix)
    
    def compute_loop_structure(self):
        """
        Compute the loop structure of the Möbius transformation.
        
        Returns:
            dict: Information about the loop structure.
        """
        # Compute the eigenvalues and eigenvectors of the matrix
        eigenvalues, eigenvectors = np.linalg.eig(self.matrix)
        
        # Compute the trace and determinant
        trace = np.trace(self.matrix)
        det = np.linalg.det(self.matrix)
        
        # Determine the type of transformation
        if abs(trace**2 - 4*det) < 1e-10:
            transformation_type = "parabolic"
        elif abs(trace**2 - 4*det) > 0:
            transformation_type = "hyperbolic"
        else:
            transformation_type = "elliptic"
        
        # Return information about the loop structure
        return {
            "type": transformation_type,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "trace": trace,
            "determinant": det,
            "fixed_points": self.fixed_points,
            "invariant_circle": self.invariant_circle
        }


class TransvectorGenerator:
    """
    Implementation of Transvector Generators used in the TIBEDO Framework.
    
    A transvector generator creates transvectors across Möbius paired structures,
    enabling the modeling of symmetry breaking and entropic decline in biological systems.
    """
    
    def __init__(self, dimension=3):
        """
        Initialize the TransvectorGenerator object.
        
        Args:
            dimension (int): The dimension of the transvector space.
        """
        self.dimension = dimension
        
        # Create a basis for the transvector space
        self.basis = self._create_basis()
        
        # Create a set of Möbius pairings
        self.pairings = self._create_pairings()
        
        # Create the transvector operators
        self.forward_operator, self.backward_operator = self._create_operators()
    
    def _create_basis(self):
        """
        Create a basis for the transvector space.
        
        Returns:
            list: The basis vectors.
        """
        # Create a set of orthonormal basis vectors
        basis = []
        
        for i in range(self.dimension):
            # Create a unit vector in the i-th direction
            vector = np.zeros(self.dimension, dtype=complex)
            vector[i] = 1.0
            basis.append(vector)
        
        return basis
    
    def _create_pairings(self):
        """
        Create a set of Möbius pairings for the transvector space.
        
        Returns:
            list: The Möbius pairings.
        """
        pairings = []
        
        # Create a pairing for each pair of basis vectors
        for i in range(self.dimension):
            for j in range(i+1, self.dimension):
                # Create a matrix that maps between these basis vectors
                matrix = np.zeros((2, 2), dtype=complex)
                matrix[0, 0] = np.exp(2j * np.pi * i / self.dimension)
                matrix[1, 1] = np.exp(2j * np.pi * j / self.dimension)
                matrix[0, 1] = 1.0
                
                # Create the pairing
                pairing = MobiusPairing(matrix)
                pairings.append(pairing)
        
        return pairings
    
    def _create_operators(self):
        """
        Create the forward and backward momentum operators.
        
        Returns:
            tuple: The forward and backward operators.
        """
        # Create the forward operator
        forward = np.zeros((self.dimension, self.dimension), dtype=complex)
        
        # Fill in the forward operator
        for i in range(self.dimension - 1):
            forward[i+1, i] = 1.0
        
        # The last element wraps around
        forward[0, self.dimension-1] = 1.0
        
        # Create the backward operator (transpose of forward)
        backward = forward.T.copy()
        
        return forward, backward
    
    def apply_forward(self, state):
        """
        Apply the forward momentum operator to a quantum state.
        
        Args:
            state (numpy.ndarray): The quantum state.
            
        Returns:
            numpy.ndarray: The transformed state.
        """
        return np.dot(self.forward_operator, state)
    
    def apply_backward(self, state):
        """
        Apply the backward momentum operator to a quantum state.
        
        Args:
            state (numpy.ndarray): The quantum state.
            
        Returns:
            numpy.ndarray: The transformed state.
        """
        return np.dot(self.backward_operator, state)
    
    def create_transvector(self, state1, state2):
        """
        Create a transvector between two quantum states.
        
        Args:
            state1 (numpy.ndarray): The first quantum state.
            state2 (numpy.ndarray): The second quantum state.
            
        Returns:
            numpy.ndarray: The transvector.
        """
        # Ensure the states have the correct dimension
        if len(state1) != self.dimension or len(state2) != self.dimension:
            raise ValueError(f"States must have dimension {self.dimension}")
        
        # Create the transvector as a tensor product
        transvector = np.outer(state1, state2)
        
        # Flatten and normalize
        transvector = transvector.flatten()
        norm = np.linalg.norm(transvector)
        
        if norm > 0:
            transvector /= norm
        
        return transvector
    
    def apply_mobius_pair(self, state, pair_index):
        """
        Apply a Möbius pairing to a quantum state.
        
        Args:
            state (numpy.ndarray): The quantum state.
            pair_index (int): The index of the pairing to apply.
            
        Returns:
            numpy.ndarray: The transformed state.
        """
        if pair_index < 0 or pair_index >= len(self.pairings):
            raise ValueError(f"Pair index must be between 0 and {len(self.pairings)-1}")
        
        # Get the pairing
        pairing = self.pairings[pair_index]
        
        # Apply the pairing to each component of the state
        result = np.zeros_like(state)
        
        for i in range(self.dimension):
            # Convert the state component to a complex number
            z = complex(state[i])
            
            # Apply the Möbius transformation
            transformed = pairing.apply(z)
            
            # Store the result
            result[i] = transformed
        
        # Normalize the result
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
        
        return result
    
    def create_5d_loop(self, state):
        """
        Create a 5D loop structure from a quantum state.
        
        Args:
            state (numpy.ndarray): The quantum state.
            
        Returns:
            list: The states forming the 5D loop.
        """
        # Create a 5D loop by applying a sequence of transformations
        loop = [state]
        
        # Apply forward momentum
        forward_state = self.apply_forward(state)
        loop.append(forward_state)
        
        # Apply a Möbius pairing
        paired_state = self.apply_mobius_pair(forward_state, 0)
        loop.append(paired_state)
        
        # Apply backward momentum
        backward_state = self.apply_backward(paired_state)
        loop.append(backward_state)
        
        # Apply another Möbius pairing to close the loop
        final_state = self.apply_mobius_pair(backward_state, 1)
        loop.append(final_state)
        
        return loop
    
    def compute_loop_invariant(self, loop):
        """
        Compute an invariant of the 5D loop.
        
        Args:
            loop (list): The states forming the 5D loop.
            
        Returns:
            complex: The loop invariant.
        """
        # Compute an invariant as the product of inner products around the loop
        invariant = 1.0
        
        for i in range(len(loop)):
            next_i = (i + 1) % len(loop)
            inner_product = np.vdot(loop[i], loop[next_i])
            invariant *= inner_product
        
        return invariant
    
    def compute_symmetry_breaking(self, state):
        """
        Compute the symmetry breaking of a quantum state.
        
        Args:
            state (numpy.ndarray): The quantum state.
            
        Returns:
            float: A measure of symmetry breaking.
        """
        # Create a 5D loop
        loop = self.create_5d_loop(state)
        
        # Compute the loop invariant
        invariant = self.compute_loop_invariant(loop)
        
        # The symmetry breaking is related to the phase of the invariant
        phase = np.angle(invariant)
        
        # Normalize to [0, 1]
        return abs(phase) / np.pi
    
    def compute_entropic_decline(self, state):
        """
        Compute the entropic decline of a quantum state.
        
        Args:
            state (numpy.ndarray): The quantum state.
            
        Returns:
            float: A measure of entropic decline.
        """
        # Compute the entropy of the state
        probabilities = np.abs(state) ** 2
        entropy = 0.0
        
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log(p)
        
        # Create a 5D loop
        loop = self.create_5d_loop(state)
        
        # Compute the entropy of the final state
        final_probabilities = np.abs(loop[-1]) ** 2
        final_entropy = 0.0
        
        for p in final_probabilities:
            if p > 0:
                final_entropy -= p * np.log(p)
        
        # The entropic decline is the difference
        return entropy - final_entropy