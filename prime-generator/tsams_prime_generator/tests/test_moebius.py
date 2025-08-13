"""
Unit tests for the moebius module.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from prime_mobius.moebius.moebius_transformation import MoebiusTransformation
from prime_mobius.moebius.prime_indexed_moebius import PrimeIndexedMoebiusTransformation
from prime_mobius.moebius.root_420_structure import Root420Structure


class TestMoebiusTransformation(unittest.TestCase):
    """
    Test cases for the MoebiusTransformation class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        self.identity = MoebiusTransformation(1, 0, 0, 1)
        self.translation = MoebiusTransformation(1, 1, 0, 1)
        self.scaling = MoebiusTransformation(2, 0, 0, 1)
        self.inversion = MoebiusTransformation(0, 1, 1, 0)
        self.general = MoebiusTransformation(1, 2, 3, 4)
        
    def test_init(self):
        """
        Test the initialization of a MoebiusTransformation.
        """
        m = MoebiusTransformation(1, 2, 3, 4)
        self.assertEqual(m.a, 1)
        self.assertEqual(m.b, 2)
        self.assertEqual(m.c, 3)
        self.assertEqual(m.d, 4)
        
    def test_apply(self):
        """
        Test the application of a MoebiusTransformation to a complex number.
        """
        # Identity transformation
        self.assertEqual(self.identity.apply(1+2j), 1+2j)
        
        # Translation
        self.assertEqual(self.translation.apply(1+2j), 2+2j)
        
        # Scaling
        self.assertEqual(self.scaling.apply(1+2j), 2+4j)
        
        # Inversion
        self.assertEqual(self.inversion.apply(2), 0.5)
        
        # General transformation
        z = 1+2j
        expected = (1*z + 2) / (3*z + 4)
        self.assertAlmostEqual(self.general.apply(z), expected)
        
    def test_compose(self):
        """
        Test the composition of two MoebiusTransformations.
        """
        # Identity composition
        composed = self.identity.compose(self.identity)
        self.assertEqual(composed.a, 1)
        self.assertEqual(composed.b, 0)
        self.assertEqual(composed.c, 0)
        self.assertEqual(composed.d, 1)
        
        # Translation composition
        composed = self.translation.compose(self.translation)
        self.assertEqual(composed.a, 1)
        self.assertEqual(composed.b, 2)
        self.assertEqual(composed.c, 0)
        self.assertEqual(composed.d, 1)
        
        # General composition
        composed = self.general.compose(self.translation)
        self.assertEqual(composed.a, 1)
        self.assertEqual(composed.b, 3)
        self.assertEqual(composed.c, 3)
        self.assertEqual(composed.d, 7)
        
    def test_inverse(self):
        """
        Test the inverse of a MoebiusTransformation.
        """
        # Identity inverse
        inverse = self.identity.inverse()
        self.assertEqual(inverse.a, 1)
        self.assertEqual(inverse.b, 0)
        self.assertEqual(inverse.c, 0)
        self.assertEqual(inverse.d, 1)
        
        # Translation inverse
        inverse = self.translation.inverse()
        self.assertEqual(inverse.a, 1)
        self.assertEqual(inverse.b, -1)
        self.assertEqual(inverse.c, 0)
        self.assertEqual(inverse.d, 1)
        
        # General inverse
        inverse = self.general.inverse()
        self.assertEqual(inverse.a, 4)
        self.assertEqual(inverse.b, -2)
        self.assertEqual(inverse.c, -3)
        self.assertEqual(inverse.d, 1)
        
    def test_fixed_points(self):
        """
        Test the computation of fixed points of a MoebiusTransformation.
        """
        # Identity has all points as fixed points
        with self.assertRaises(ValueError):
            self.identity.fixed_points()
        
        # Translation has one fixed point at infinity
        fps = self.translation.fixed_points()
        self.assertEqual(len(fps), 1)
        self.assertTrue(np.isinf(fps[0]))
        
        # General transformation has two fixed points
        fps = self.general.fixed_points()
        self.assertEqual(len(fps), 2)
        
        # Verify that the fixed points are actually fixed
        for fp in fps:
            if not np.isinf(fp):
                self.assertAlmostEqual(self.general.apply(fp), fp)
                
    def test_is_elliptic(self):
        """
        Test the is_elliptic method.
        """
        # Create an elliptic transformation
        elliptic = MoebiusTransformation(np.cos(0.1), -np.sin(0.1), np.sin(0.1), np.cos(0.1))
        self.assertTrue(elliptic.is_elliptic())
        
        # The general transformation is not elliptic
        self.assertFalse(self.general.is_elliptic())
        
    def test_is_parabolic(self):
        """
        Test the is_parabolic method.
        """
        # Create a parabolic transformation
        parabolic = MoebiusTransformation(1, 1, 0, 1)
        self.assertTrue(parabolic.is_parabolic())
        
        # The general transformation is not parabolic
        self.assertFalse(self.general.is_parabolic())
        
    def test_is_hyperbolic(self):
        """
        Test the is_hyperbolic method.
        """
        # Create a hyperbolic transformation
        hyperbolic = MoebiusTransformation(2, 0, 0, 0.5)
        self.assertTrue(hyperbolic.is_hyperbolic())
        
        # The general transformation is not hyperbolic
        self.assertFalse(self.general.is_hyperbolic())


class TestPrimeIndexedMoebiusTransformation(unittest.TestCase):
    """
    Test cases for the PrimeIndexedMoebiusTransformation class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        self.p11 = PrimeIndexedMoebiusTransformation(11)
        self.p13 = PrimeIndexedMoebiusTransformation(13)
        
    def test_init(self):
        """
        Test the initialization of a PrimeIndexedMoebiusTransformation.
        """
        m = PrimeIndexedMoebiusTransformation(11)
        self.assertEqual(m.prime_index, 11)
        self.assertEqual(m.n, 420)
        
    def test_prime_index(self):
        """
        Test the prime_index property.
        """
        self.assertEqual(self.p11.prime_index, 11)
        self.assertEqual(self.p13.prime_index, 13)
        
    def test_energy(self):
        """
        Test the energy method.
        """
        # Energy should be a positive real number
        self.assertGreater(self.p11.energy(), 0)
        
        # Energy should increase with the prime index
        self.assertGreater(self.p13.energy(), self.p11.energy())


class TestRoot420Structure(unittest.TestCase):
    """
    Test cases for the Root420Structure class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        self.root_structure = Root420Structure()
        
    def test_init(self):
        """
        Test the initialization of a Root420Structure.
        """
        self.assertEqual(self.root_structure.n, 420)
        
    def test_transformations(self):
        """
        Test the transformations property.
        """
        transformations = self.root_structure.transformations
        
        # There should be 81 transformations (one for each prime less than 420)
        self.assertEqual(len(transformations), 81)
        
        # Each transformation should be a PrimeIndexedMoebiusTransformation
        for t in transformations:
            self.assertIsInstance(t, PrimeIndexedMoebiusTransformation)
            
    def test_energy_spectrum(self):
        """
        Test the energy_spectrum method.
        """
        spectrum = self.root_structure.energy_spectrum()
        
        # The spectrum should have 81 energy values
        self.assertEqual(len(spectrum), 81)
        
        # Energy values should be positive
        for energy in spectrum:
            self.assertGreater(energy, 0)
            
        # Energy values should be sorted in ascending order
        self.assertEqual(spectrum, sorted(spectrum))


if __name__ == '__main__':
    unittest.main()