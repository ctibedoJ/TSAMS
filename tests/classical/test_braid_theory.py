"""
Tests for the Braid Theory module.
"""

import unittest
import numpy as np
import sympy as sp
from tsams_classical.braid_theory import (
    BraidGroup,
    BraidWord,
    BraidInvariants,
    AlexanderPolynomial,
    JonesPolynomial,
    BraidOperations
)

class TestBraidGroup(unittest.TestCase):
    """Test cases for the BraidGroup class."""
    
    def test_initialization(self):
        """Test initialization of a braid group."""
        group = BraidGroup(3)
        self.assertEqual(group.n, 3)
        
        # Test invalid initialization
        with self.assertRaises(ValueError):
            BraidGroup(1)
    
    def test_identity(self):
        """Test the identity element of a braid group."""
        group = BraidGroup(3)
        identity = group.identity()
        self.assertEqual(identity.n, 3)
        self.assertEqual(identity.word, [])
    
    def test_generator(self):
        """Test the generators of a braid group."""
        group = BraidGroup(3)
        
        # Test the first generator
        sigma_1 = group.generator(1)
        self.assertEqual(sigma_1.n, 3)
        self.assertEqual(sigma_1.word, [1])
        
        # Test the second generator
        sigma_2 = group.generator(2)
        self.assertEqual(sigma_2.n, 3)
        self.assertEqual(sigma_2.word, [2])
        
        # Test invalid generator
        with self.assertRaises(ValueError):
            group.generator(0)
        with self.assertRaises(ValueError):
            group.generator(3)


class TestBraidWord(unittest.TestCase):
    """Test cases for the BraidWord class."""
    
    def test_initialization(self):
        """Test initialization of a braid word."""
        braid = BraidWord(3, [1, 2, 1])
        self.assertEqual(braid.n, 3)
        self.assertEqual(braid.word, [1, 2, 1])
        
        # Test empty word
        braid = BraidWord(3)
        self.assertEqual(braid.n, 3)
        self.assertEqual(braid.word, [])
        
        # Test invalid initialization
        with self.assertRaises(ValueError):
            BraidWord(1)
        with self.assertRaises(ValueError):
            BraidWord(3, [3])
    
    def test_append(self):
        """Test appending a generator to a braid word."""
        braid = BraidWord(3)
        braid.append(1)
        self.assertEqual(braid.word, [1])
        braid.append(2)
        self.assertEqual(braid.word, [1, 2])
        braid.append(-1)
        self.assertEqual(braid.word, [1, 2, -1])
        
        # Test invalid append
        with self.assertRaises(ValueError):
            braid.append(3)
        with self.assertRaises(ValueError):
            braid.append(0)
    
    def test_inverse(self):
        """Test computing the inverse of a braid word."""
        braid = BraidWord(3, [1, 2, 1])
        inverse = braid.inverse()
        self.assertEqual(inverse.n, 3)
        self.assertEqual(inverse.word, [-1, -2, -1])
    
    def test_multiply(self):
        """Test multiplying two braid words."""
        braid1 = BraidWord(3, [1, 2])
        braid2 = BraidWord(3, [2, 1])
        product = braid1.multiply(braid2)
        self.assertEqual(product.n, 3)
        self.assertEqual(product.word, [1, 2, 2, 1])
        
        # Test invalid multiplication
        braid3 = BraidWord(4, [1, 2, 3])
        with self.assertRaises(ValueError):
            braid1.multiply(braid3)
    
    def test_reduce(self):
        """Test reducing a braid word."""
        braid = BraidWord(3, [1, -1, 2, -2])
        reduced = braid.reduce()
        self.assertEqual(reduced.n, 3)
        self.assertEqual(reduced.word, [])
    
    def test_to_permutation(self):
        """Test converting a braid word to a permutation."""
        braid = BraidWord(3, [1, 2, 1])
        perm = braid.to_permutation()
        self.assertEqual(perm, [1, 2, 0])


class TestBraidInvariants(unittest.TestCase):
    """Test cases for the BraidInvariants class."""
    
    def test_signature(self):
        """Test computing the signature of a braid."""
        braid = BraidWord(3, [1, 2, 1])
        signature = BraidInvariants.signature(braid)
        self.assertIsInstance(signature, int)
    
    def test_linking_number(self):
        """Test computing the linking number of a braid closure."""
        braid = BraidWord(3, [1, 2, 1])
        linking_number = BraidInvariants.linking_number(braid)
        self.assertEqual(linking_number, 1)
        
        braid = BraidWord(3, [-1, -2, -1])
        linking_number = BraidInvariants.linking_number(braid)
        self.assertEqual(linking_number, -1)


class TestAlexanderPolynomial(unittest.TestCase):
    """Test cases for the AlexanderPolynomial class."""
    
    def test_compute(self):
        """Test computing the Alexander polynomial of a braid closure."""
        braid = BraidWord(3, [1, 2, 1])
        poly = AlexanderPolynomial.compute(braid)
        self.assertIsInstance(poly, sp.Poly)
    
    def test_evaluate(self):
        """Test evaluating the Alexander polynomial at a specific value."""
        braid = BraidWord(3, [1, 2, 1])
        poly = AlexanderPolynomial.compute(braid)
        value = AlexanderPolynomial.evaluate(poly, 1.0)
        self.assertIsInstance(value, float)


class TestJonesPolynomial(unittest.TestCase):
    """Test cases for the JonesPolynomial class."""
    
    def test_compute(self):
        """Test computing the Jones polynomial of a braid closure."""
        braid = BraidWord(3, [1, 2, 1])
        poly = JonesPolynomial.compute(braid)
        self.assertIsInstance(poly, sp.Poly)
    
    def test_evaluate(self):
        """Test evaluating the Jones polynomial at a specific value."""
        braid = BraidWord(3, [1, 2, 1])
        poly = JonesPolynomial.compute(braid)
        value = JonesPolynomial.evaluate(poly, 1.0)
        self.assertIsInstance(value, float)


class TestBraidOperations(unittest.TestCase):
    """Test cases for the BraidOperations class."""
    
    def test_closure_permutation(self):
        """Test computing the permutation cycles of a braid closure."""
        braid = BraidWord(3, [1, 2, 1])
        cycles = BraidOperations.closure_permutation(braid)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(sorted(cycles[0]), [0, 1, 2])
    
    def test_num_components(self):
        """Test computing the number of components in a braid closure."""
        braid = BraidWord(3, [1, 2, 1])
        num_components = BraidOperations.num_components(braid)
        self.assertEqual(num_components, 1)
        
        braid = BraidWord(4, [1, 3])
        num_components = BraidOperations.num_components(braid)
        self.assertEqual(num_components, 2)
    
    def test_is_knot(self):
        """Test checking if a braid closure is a knot."""
        braid = BraidWord(3, [1, 2, 1])
        is_knot = BraidOperations.is_knot(braid)
        self.assertTrue(is_knot)
        
        braid = BraidWord(4, [1, 3])
        is_knot = BraidOperations.is_knot(braid)
        self.assertFalse(is_knot)
    
    def test_stabilize(self):
        """Test stabilizing a braid."""
        braid = BraidWord(3, [1, 2, 1])
        stabilized = BraidOperations.stabilize(braid)
        self.assertEqual(stabilized.n, 4)
        self.assertEqual(stabilized.word, [1, 2, 1, 3])
        
        stabilized = BraidOperations.stabilize(braid, False)
        self.assertEqual(stabilized.n, 4)
        self.assertEqual(stabilized.word, [1, 2, 1, -3])
    
    def test_destabilize(self):
        """Test destabilizing a braid."""
        braid = BraidWord(4, [1, 2, 1, 3])
        destabilized = BraidOperations.destabilize(braid)
        self.assertEqual(destabilized.n, 3)
        self.assertEqual(destabilized.word, [1, 2, 1])
        
        braid = BraidWord(3, [1, 2, 1])
        destabilized = BraidOperations.destabilize(braid)
        self.assertIsNone(destabilized)
    
    def test_markov_move_1(self):
        """Test performing a Markov move of type 1 on a braid."""
        braid = BraidWord(3, [1, 2, 1])
        conjugated = BraidOperations.markov_move_1(braid, 1)
        self.assertEqual(conjugated.n, 3)
        self.assertEqual(conjugated.word, [1, 1, 2, 1, -1])
    
    def test_markov_move_2(self):
        """Test performing a Markov move of type 2 on a braid."""
        braid = BraidWord(3, [1, 2, 1])
        stabilized = BraidOperations.markov_move_2(braid)
        self.assertEqual(stabilized.n, 4)
        self.assertEqual(stabilized.word, [1, 2, 1, 3])
    
    def test_plat_closure(self):
        """Test computing the plat closure of a braid."""
        braid = BraidWord(4, [1, 3, 2])
        cycles = BraidOperations.plat_closure(braid)
        self.assertGreater(len(cycles), 0)
    
    def test_num_plat_components(self):
        """Test computing the number of components in a plat closure."""
        braid = BraidWord(4, [1, 3, 2])
        num_components = BraidOperations.num_plat_components(braid)
        self.assertGreater(num_components, 0)
    
    def test_is_plat_knot(self):
        """Test checking if a plat closure is a knot."""
        braid = BraidWord(4, [1, 3, 2])
        is_knot = BraidOperations.is_plat_knot(braid)
        self.assertIsInstance(is_knot, bool)
    
    def test_braid_index(self):
        """Test computing the braid index of a braid."""
        braid = BraidWord(3, [1, 2, 1])
        braid_index = BraidOperations.braid_index(braid)
        self.assertEqual(braid_index, 3)
    
    def test_to_link_diagram(self):
        """Test converting a braid to a link diagram."""
        braid = BraidWord(3, [1, 2, 1])
        diagram = BraidOperations.to_link_diagram(braid)
        self.assertEqual(diagram['components'], 1)
        self.assertEqual(diagram['crossings'], 3)


if __name__ == '__main__':
    unittest.main()