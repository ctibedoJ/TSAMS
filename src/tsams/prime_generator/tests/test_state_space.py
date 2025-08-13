"""
Tests for the state space implementation.
"""

import unittest
import numpy as np
import cmath
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from prime_mobius.moebius.moebius_transformation import MoebiusTransformation
from prime_mobius.moebius.prime_indexed_moebius import PrimeIndexedMoebiusTransformation
from prime_mobius.state_space.state_space import StateSpace
from prime_mobius.state_space.nodal_structure_441 import NodalStructure441


class TestStateSpace(unittest.TestCase):
    """
    Test case for the StateSpace class.
    """
    
    def test_initialization(self):
        """
        Test the initialization of a state space.
        """
        # Create a state space
        state_space = StateSpace()
        
        # Check the dimension
        self.assertEqual(state_space.dimension, 1)
        
        # Create a higher-dimensional state space
        state_space_higher = StateSpace(dimension=3)
        self.assertEqual(state_space_higher.dimension, 3)
    
    def test_transform(self):
        """
        Test the transformation of a state.
        """
        # Create a state space
        state_space = StateSpace()
        
        # Create a transformation
        transformation = MoebiusTransformation(1, 2, 3, 4)
        
        # Transform a state
        state = 1 + 2j
        transformed_state = state_space.transform(state, transformation)
        
        # Check that the transformation is correct
        expected = transformation.apply(state)
        self.assertEqual(transformed_state, expected)
    
    def test_orbit(self):
        """
        Test the computation of an orbit.
        """
        # Create a state space
        state_space = StateSpace()
        
        # Create a transformation
        transformation = MoebiusTransformation(0, -1j, 1j, 0)  # Rotation by 90 degrees
        
        # Compute an orbit
        state = 1.0
        orbit = state_space.orbit(state, [transformation], max_iterations=4)
        
        # Check that the orbit has the correct length
        self.assertEqual(len(orbit), 5)  # Initial state + 4 iterations
        
        # Check that the orbit is correct
        expected_orbit = [1.0, 1j, -1.0, -1j, 1.0]
        for actual, expected in zip(orbit, expected_orbit):
            self.assertAlmostEqual(actual, expected)
    
    def test_is_dense(self):
        """
        Test the density check for an orbit.
        """
        # Create a state space
        state_space = StateSpace()
        
        # Create a dense orbit (covering the entire complex plane)
        dense_orbit = [complex(x, y) for x in np.linspace(-5, 5, 10) for y in np.linspace(-5, 5, 10)]
        dense_orbit.append(None)  # Add the point at infinity
        
        # Check that the orbit is dense
        self.assertTrue(state_space.is_dense(dense_orbit))
        
        # Create a non-dense orbit (just a few points)
        non_dense_orbit = [1 + 2j, 3 + 4j]
        
        # Check that the orbit is not dense
        self.assertFalse(state_space.is_dense(non_dense_orbit))
    
    def test_compute_ergodic_average(self):
        """
        Test the computation of ergodic averages.
        """
        # Create a state space
        state_space = StateSpace()
        
        # Create an orbit
        orbit = [1 + 2j, 3 + 4j, 5 + 6j]
        
        # Define a function to average
        function = lambda z: z.real
        
        # Compute the ergodic average
        average = state_space.compute_ergodic_average(orbit, function)
        
        # Check that the average is correct
        expected = (1 + 3 + 5) / 3
        self.assertEqual(average, expected)
    
    def test_compute_lyapunov_exponent(self):
        """
        Test the computation of Lyapunov exponents.
        """
        # Create a state space
        state_space = StateSpace()
        
        # Create a transformation
        transformation = MoebiusTransformation(2, 0, 0, 1/2)  # Hyperbolic transformation
        
        # Create an orbit
        orbit = [1.0, 2.0, 4.0, 8.0]
        
        # Compute the Lyapunov exponent
        exponent = state_space.compute_lyapunov_exponent(orbit, transformation)
        
        # Check that the exponent is positive (indicating exponential divergence)
        self.assertGreater(exponent, 0)


class TestNodalStructure441(unittest.TestCase):
    """
    Test case for the NodalStructure441 class.
    """
    
    def test_initialization(self):
        """
        Test the initialization of the 441-dimensional nodal structure.
        """
        # Create the nodal structure
        nodal_structure = NodalStructure441()
        
        # Check that the factorized state spaces have the correct dimensions
        self.assertEqual(nodal_structure.state_space_9.dimension, 9)
        self.assertEqual(nodal_structure.state_space_49.dimension, 49)
        
        # Check that there are 21 hair braid nodes
        self.assertEqual(len(nodal_structure.hair_braid_nodes), 21)
    
    def test_factorize(self):
        """
        Test the factorization of the 441-dimensional structure.
        """
        # Create the nodal structure
        nodal_structure = NodalStructure441()
        
        # Factorize it
        state_space_9, state_space_49 = nodal_structure.factorize()
        
        # Check that the factorized state spaces have the correct dimensions
        self.assertEqual(state_space_9.dimension, 9)
        self.assertEqual(state_space_49.dimension, 49)
    
    def test_get_hair_braid_nodes(self):
        """
        Test getting the hair braid nodes.
        """
        # Create the nodal structure
        nodal_structure = NodalStructure441()
        
        # Get the hair braid nodes
        hair_braid_nodes = nodal_structure.get_hair_braid_nodes()
        
        # Check that there are 21 hair braid nodes
        self.assertEqual(len(hair_braid_nodes), 21)
        
        # Check that each node has the correct structure
        for node in hair_braid_nodes:
            self.assertIsInstance(node, tuple)
            self.assertEqual(len(node), 2)
            self.assertIsInstance(node[0], complex)
            self.assertIsInstance(node[1], complex)
    
    def test_transform(self):
        """
        Test the transformation of a state in the nodal structure.
        """
        # Create the nodal structure
        nodal_structure = NodalStructure441()
        
        # Create a transformation
        transformation = MoebiusTransformation(1, 2, 3, 4)
        
        # Get a state (a hair braid node)
        state = nodal_structure.hair_braid_nodes[0]
        
        # Transform the state
        transformed_state = nodal_structure.transform(state, transformation)
        
        # Check that the transformed state has the correct structure
        self.assertIsInstance(transformed_state, tuple)
        self.assertEqual(len(transformed_state), 2)
        self.assertIsInstance(transformed_state[0], complex)
        self.assertIsInstance(transformed_state[1], complex)
    
    def test_braid_operation(self):
        """
        Test the braid operation between two hair braid nodes.
        """
        # Create the nodal structure
        nodal_structure = NodalStructure441()
        
        # Apply the braid operation
        result = nodal_structure.braid_operation(0, 1)
        
        # Check that the result has the correct structure
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], complex)
        self.assertIsInstance(result[1], complex)
    
    def test_braid_invariant(self):
        """
        Test the computation of braid invariants.
        """
        # Create the nodal structure
        nodal_structure = NodalStructure441()
        
        # Create a braid
        braid = [(0, 1), (1, 2), (0, 2)]
        
        # Compute the braid invariant
        invariant = nodal_structure.braid_invariant(braid)
        
        # Check that the invariant is a complex number
        self.assertIsInstance(invariant, complex)
    
    def test_jones_polynomial(self):
        """
        Test the computation of Jones polynomials.
        """
        # Create the nodal structure
        nodal_structure = NodalStructure441()
        
        # Create a braid
        braid = [(0, 1), (1, 2), (0, 2)]
        
        # Compute the Jones polynomial
        jones_polynomial = nodal_structure.jones_polynomial(braid)
        
        # Check that the Jones polynomial is a list of complex numbers
        self.assertIsInstance(jones_polynomial, list)
        for coeff in jones_polynomial:
            self.assertIsInstance(coeff, complex)


if __name__ == '__main__':
    unittest.main()