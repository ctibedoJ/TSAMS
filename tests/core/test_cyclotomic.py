"""
Tests for the cyclotomic field implementation.
"""

import unittest
import numpy as np
import sympy
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from prime_mobius.cyclotomic.cyclotomic_field import CyclotomicField


class TestCyclotomicField(unittest.TestCase):
    """
    Test case for the CyclotomicField class.
    """
    
    def test_initialization(self):
        """
        Test the initialization of a cyclotomic field.
        """
        # Create a cyclotomic field
        field = CyclotomicField(420)
        
        # Check the conductor
        self.assertEqual(field.conductor, 420)
        
        # Check the dimension
        self.assertEqual(field.dimension, sympy.totient(420))
        
        # Check that the dimension is correct
        self.assertEqual(field.dimension, 96)
    
    def test_prime_factorization(self):
        """
        Test the prime factorization of the conductor.
        """
        # Create a cyclotomic field
        field = CyclotomicField(420)
        
        # Check the prime factorization
        factorization = field.prime_factorization()
        self.assertEqual(factorization, {2: 2, 3: 1, 5: 1, 7: 1})
        
        # Check that the product of prime powers equals the conductor
        product = 1
        for p, e in factorization.items():
            product *= p**e
        self.assertEqual(product, 420)
    
    def test_element_from_coefficients(self):
        """
        Test the creation of a field element from coefficients.
        """
        # Create a cyclotomic field
        field = CyclotomicField(420)
        
        # Create an element
        coefficients = [1] + [0] * (field.dimension - 1)
        element = field.element_from_coefficients(coefficients)
        
        # Check that the element has the correct structure
        self.assertIsInstance(element, dict)
        self.assertEqual(element[0], 1)
        
        # Check that creating an element with the wrong number of coefficients raises an error
        with self.assertRaises(ValueError):
            field.element_from_coefficients([1, 2, 3])
    
    def test_add(self):
        """
        Test the addition of field elements.
        """
        # Create a cyclotomic field
        field = CyclotomicField(420)
        
        # Create two elements
        element1 = field.element_from_coefficients([1] + [0] * (field.dimension - 1))
        element2 = field.element_from_coefficients([0, 1] + [0] * (field.dimension - 2))
        
        # Add them
        sum_element = field.add(element1, element2)
        
        # Check that the sum is correct
        self.assertEqual(sum_element[0], 1)
        self.assertEqual(sum_element[1], 1)
    
    def test_multiply(self):
        """
        Test the multiplication of field elements.
        """
        # Create a cyclotomic field
        field = CyclotomicField(420)
        
        # Create two elements
        element1 = field.element_from_coefficients([1] + [0] * (field.dimension - 1))
        element2 = field.element_from_coefficients([0, 1] + [0] * (field.dimension - 2))
        
        # Multiply them
        product = field.multiply(element1, element2)
        
        # Check that the product is correct
        self.assertEqual(product[1], 1)
    
    def test_conjugate(self):
        """
        Test the conjugation of field elements.
        """
        # Create a cyclotomic field
        field = CyclotomicField(420)
        
        # Create an element
        element = field.element_from_coefficients([0, 1] + [0] * (field.dimension - 2))
        
        # Compute its conjugate
        conjugate = field.conjugate(element)
        
        # Check that the conjugate is correct
        self.assertEqual(conjugate[field.dimension - 1], 1)
    
    def test_norm(self):
        """
        Test the norm of field elements.
        """
        # Create a cyclotomic field
        field = CyclotomicField(420)
        
        # Create an element
        element = field.element_from_coefficients([1] + [0] * (field.dimension - 1))
        
        # Compute its norm
        norm = field.norm(element)
        
        # Check that the norm is correct
        self.assertEqual(norm, 1)
    
    def test_cyclotomic_polynomial(self):
        """
        Test the computation of cyclotomic polynomials.
        """
        # Create a cyclotomic field
        field = CyclotomicField(420)
        
        # Compute the cyclotomic polynomial
        polynomial = field.cyclotomic_polynomial()
        
        # Check that the polynomial has the correct degree
        self.assertEqual(polynomial.degree(), field.dimension)
    
    def test_galois_group_structure(self):
        """
        Test the computation of the Galois group structure.
        """
        # Create a cyclotomic field
        field = CyclotomicField(420)
        
        # Compute the Galois group structure
        galois_group = field.galois_group_structure()
        
        # Check that the Galois group has the correct structure
        self.assertIsInstance(galois_group, list)
        self.assertEqual(len(galois_group), field.dimension)
    
    def test_dedekind_cut_morphic_conductor(self):
        """
        Test the Dedekind cut morphic conductor.
        """
        # Create a cyclotomic field
        field = CyclotomicField(420)
        
        # Check if it's a Dedekind cut morphic conductor
        self.assertTrue(field.is_dedekind_cut_conductor)
        
        # Compute the Dedekind cut morphic conductor value
        value = field.dedekind_cut_morphic_conductor()
        
        # Check that the value is positive
        self.assertGreater(value, 0)


if __name__ == '__main__':
    unittest.main()