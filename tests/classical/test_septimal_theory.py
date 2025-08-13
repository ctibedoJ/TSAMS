"""
Tests for the Septimal Theory module.
"""

import unittest
import numpy as np
from tsams_classical.septimal import (
    SeptimalStructure,
    SeptimalLattice,
    SeptimalOperations
)

class TestSeptimalStructure(unittest.TestCase):
    """Test cases for the SeptimalStructure class."""
    
    def test_initialization(self):
        """Test initialization of a septimal structure."""
        structure = SeptimalStructure(3)
        self.assertEqual(structure.dimension, 3)
        self.assertEqual(structure.base, 7)
        
        # Test invalid initialization
        with self.assertRaises(ValueError):
            SeptimalStructure(0)
        with self.assertRaises(ValueError):
            SeptimalStructure(3, 6)
    
    def test_get_point(self):
        """Test getting a point in the lattice."""
        structure = SeptimalStructure(2)
        point = structure.get_point([3, 5])
        self.assertEqual(len(point), 2)
        
        # Test invalid coordinates
        with self.assertRaises(ValueError):
            structure.get_point([3, 5, 1])
        with self.assertRaises(ValueError):
            structure.get_point([3, 8])
    
    def test_distance(self):
        """Test computing the distance between points."""
        structure = SeptimalStructure(2)
        distance = structure.distance([1, 2], [3, 4])
        self.assertAlmostEqual(distance, np.sqrt(8))
    
    def test_neighbors(self):
        """Test finding neighbors of a point."""
        structure = SeptimalStructure(2)
        neighbors = structure.neighbors([3, 3])
        self.assertEqual(len(neighbors), 4)
        
        # Test neighbors at the boundary
        neighbors = structure.neighbors([0, 0])
        self.assertEqual(len(neighbors), 2)
    
    def test_septimal_norm(self):
        """Test computing the septimal norm of a point."""
        structure = SeptimalStructure(3)
        norm = structure.septimal_norm([1, 2, 3])
        self.assertEqual(norm, 6)
        
        norm = structure.septimal_norm([3, 5, 6])
        self.assertEqual(norm, 0)
    
    def test_septimal_product(self):
        """Test computing the septimal product of points."""
        structure = SeptimalStructure(3)
        product = structure.septimal_product([1, 2, 3], [4, 5, 6])
        self.assertEqual(product, [4, 3, 4])
        
        # Test invalid product
        with self.assertRaises(ValueError):
            structure.septimal_product([1, 2], [3, 4, 5])
    
    def test_septimal_sum(self):
        """Test computing the septimal sum of points."""
        structure = SeptimalStructure(3)
        sum_result = structure.septimal_sum([1, 2, 3], [4, 5, 6])
        self.assertEqual(sum_result, [5, 0, 2])
        
        # Test invalid sum
        with self.assertRaises(ValueError):
            structure.septimal_sum([1, 2], [3, 4, 5])


class TestSeptimalLattice(unittest.TestCase):
    """Test cases for the SeptimalLattice class."""
    
    def test_initialization(self):
        """Test initialization of a septimal lattice."""
        lattice = SeptimalLattice(2)
        self.assertEqual(lattice.dimension, 2)
        self.assertEqual(lattice.base, 7)
        self.assertEqual(len(lattice.points), 49)  # 7^2
        
        # Test invalid initialization
        with self.assertRaises(ValueError):
            SeptimalLattice(0)
        with self.assertRaises(ValueError):
            SeptimalLattice(2, 6)
    
    def test_get_point(self):
        """Test getting a point by index."""
        lattice = SeptimalLattice(2)
        point = lattice.get_point(8)
        self.assertEqual(len(point), 2)
        
        # Test invalid index
        with self.assertRaises(IndexError):
            lattice.get_point(100)
    
    def test_get_index(self):
        """Test getting the index of a point."""
        lattice = SeptimalLattice(2)
        index = lattice.get_index([1, 1])
        self.assertIsInstance(index, int)
        
        # Test invalid point
        with self.assertRaises(ValueError):
            lattice.get_index([7, 7])
    
    def test_distance_matrix(self):
        """Test computing the distance matrix."""
        lattice = SeptimalLattice(1)  # Use 1D for simplicity
        distances = lattice.distance_matrix()
        self.assertEqual(distances.shape, (7, 7))
        
        # Check that the distance from a point to itself is 0
        for i in range(7):
            self.assertEqual(distances[i, i], 0)
    
    def test_adjacency_matrix(self):
        """Test computing the adjacency matrix."""
        lattice = SeptimalLattice(1)  # Use 1D for simplicity
        adjacency = lattice.adjacency_matrix()
        self.assertEqual(adjacency.shape, (7, 7))
        
        # Check that a point is not adjacent to itself
        for i in range(7):
            self.assertEqual(adjacency[i, i], False)
    
    def test_septimal_structure(self):
        """Test getting the septimal structure."""
        lattice = SeptimalLattice(2)
        structure = lattice.septimal_structure()
        self.assertIsInstance(structure, SeptimalStructure)
        self.assertEqual(structure.dimension, 2)
        self.assertEqual(structure.base, 7)
    
    def test_len(self):
        """Test the length of the lattice."""
        lattice = SeptimalLattice(3)
        self.assertEqual(len(lattice), 343)  # 7^3
    
    def test_getitem(self):
        """Test getting a point using the [] operator."""
        lattice = SeptimalLattice(2)
        point = lattice[8]
        self.assertEqual(len(point), 2)


class TestSeptimalOperations(unittest.TestCase):
    """Test cases for the SeptimalOperations class."""
    
    def test_rotate(self):
        """Test rotating a point."""
        structure = SeptimalStructure(3)
        point = [1, 2, 3]
        rotated = SeptimalOperations.rotate(structure, point, 0, 1)
        self.assertEqual(len(rotated), 3)
        
        # Test invalid rotation
        with self.assertRaises(ValueError):
            SeptimalOperations.rotate(structure, point, 3, 1)
        with self.assertRaises(ValueError):
            SeptimalOperations.rotate(structure, point, 0, 7)
    
    def test_reflect(self):
        """Test reflecting a point."""
        structure = SeptimalStructure(3)
        point = [1, 2, 3]
        reflected = SeptimalOperations.reflect(structure, point, 0)
        self.assertEqual(reflected[0], 5)  # 7 - 1 - 1
        self.assertEqual(reflected[1:], point[1:])
        
        # Test invalid reflection
        with self.assertRaises(ValueError):
            SeptimalOperations.reflect(structure, point, 3)
    
    def test_translate(self):
        """Test translating a point."""
        structure = SeptimalStructure(3)
        point = [1, 2, 3]
        translated = SeptimalOperations.translate(structure, point, [1, 1, 1])
        self.assertEqual(translated, [2, 3, 4])
        
        # Test invalid translation
        with self.assertRaises(ValueError):
            SeptimalOperations.translate(structure, point, [1, 1])
    
    def test_scale(self):
        """Test scaling a point."""
        structure = SeptimalStructure(3)
        point = [1, 2, 3]
        scaled = SeptimalOperations.scale(structure, point, 2)
        self.assertEqual(scaled, [2, 4, 6])
    
    def test_invert(self):
        """Test inverting a point."""
        structure = SeptimalStructure(3)
        point = [1, 2, 3]
        inverted = SeptimalOperations.invert(structure, point)
        self.assertEqual(len(inverted), 3)
        
        # Check that the product of a point and its inverse is 1 (mod 7)
        for p, inv in zip(point, inverted):
            if p != 0:
                self.assertEqual((p * inv) % 7, 1)
    
    def test_orbit(self):
        """Test computing the orbit of a point."""
        structure = SeptimalStructure(2)
        point = [1, 2]
        
        # Test rotation orbit
        orbit = SeptimalOperations.orbit(structure, point, 'rotate', 0, 1)
        self.assertGreater(len(orbit), 1)
        
        # Test reflection orbit
        orbit = SeptimalOperations.orbit(structure, point, 'reflect', 0)
        self.assertGreater(len(orbit), 1)
        
        # Test translation orbit
        orbit = SeptimalOperations.orbit(structure, point, 'translate', [1, 0])
        self.assertEqual(len(orbit), 7)
        
        # Test scale orbit
        orbit = SeptimalOperations.orbit(structure, point, 'scale', 2)
        self.assertGreater(len(orbit), 1)
        
        # Test invert orbit
        orbit = SeptimalOperations.orbit(structure, point, 'invert')
        self.assertGreater(len(orbit), 1)
        
        # Test invalid operation
        with self.assertRaises(ValueError):
            SeptimalOperations.orbit(structure, point, 'unknown')
    
    def test_symmetry_group(self):
        """Test computing the symmetry group."""
        structure = SeptimalStructure(2)
        generators = SeptimalOperations.symmetry_group(structure)
        self.assertGreater(len(generators), 0)
    
    def test_apply_operation(self):
        """Test applying an operation."""
        structure = SeptimalStructure(3)
        point = [1, 2, 3]
        
        # Test rotation
        rotated = SeptimalOperations.apply_operation(structure, point, ('rotate', [0, 1]))
        self.assertEqual(len(rotated), 3)
        
        # Test reflection
        reflected = SeptimalOperations.apply_operation(structure, point, ('reflect', [0]))
        self.assertEqual(len(reflected), 3)
        
        # Test translation
        translated = SeptimalOperations.apply_operation(structure, point, ('translate', [[1, 1, 1]]))
        self.assertEqual(len(translated), 3)
        
        # Test scale
        scaled = SeptimalOperations.apply_operation(structure, point, ('scale', [2]))
        self.assertEqual(len(scaled), 3)
        
        # Test invert
        inverted = SeptimalOperations.apply_operation(structure, point, ('invert', []))
        self.assertEqual(len(inverted), 3)
        
        # Test invalid operation
        with self.assertRaises(ValueError):
            SeptimalOperations.apply_operation(structure, point, ('unknown', []))
    
    def test_septimal_transform(self):
        """Test applying a sequence of operations."""
        structure = SeptimalStructure(2)
        points = [[1, 2], [3, 4]]
        operations = [('rotate', [0, 1]), ('translate', [[1, 1]])]
        transformed = SeptimalOperations.septimal_transform(structure, points, operations)
        self.assertEqual(len(transformed), 2)
        self.assertEqual(len(transformed[0]), 2)
    
    def test_septimal_distance(self):
        """Test computing the septimal distance."""
        structure = SeptimalStructure(2)
        distance = SeptimalOperations.septimal_distance(structure, [1, 2], [3, 4])
        self.assertEqual(distance, 4)
    
    def test_septimal_centroid(self):
        """Test computing the septimal centroid."""
        structure = SeptimalStructure(2)
        points = [[1, 2], [3, 4], [5, 6]]
        centroid = SeptimalOperations.septimal_centroid(structure, points)
        self.assertEqual(len(centroid), 2)
        
        # Test empty points
        with self.assertRaises(ValueError):
            SeptimalOperations.septimal_centroid(structure, [])


if __name__ == '__main__':
    unittest.main()