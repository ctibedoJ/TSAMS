"""
Basic tests for the TSAMS Prime Generator package.
"""

import unittest
import numpy as np

from prime_generator import (
    sieve_of_eratosthenes,
    cyclotomic_sieve,
    is_prime,
    next_prime,
    prime_factors
)


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of the prime generator package."""
    
    def test_sieve_of_eratosthenes(self):
        """Test the classical Sieve of Eratosthenes."""
        primes = sieve_of_eratosthenes(20)
        expected = [2, 3, 5, 7, 11, 13, 17, 19]
        self.assertEqual(primes, expected)
    
    def test_cyclotomic_sieve(self):
        """Test the cyclotomic sieve."""
        primes = cyclotomic_sieve(20, conductor=8)
        expected = [2, 3, 5, 7, 11, 13, 17, 19]
        self.assertEqual(set(primes), set(expected))
    
    def test_is_prime(self):
        """Test primality testing."""
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))
        self.assertTrue(is_prime(5))
        self.assertTrue(is_prime(7))
        self.assertTrue(is_prime(11))
        
        self.assertFalse(is_prime(1))
        self.assertFalse(is_prime(4))
        self.assertFalse(is_prime(6))
        self.assertFalse(is_prime(8))
        self.assertFalse(is_prime(9))
    
    def test_next_prime(self):
        """Test next prime function."""
        self.assertEqual(next_prime(1), 2)
        self.assertEqual(next_prime(2), 3)
        self.assertEqual(next_prime(3), 5)
        self.assertEqual(next_prime(10), 11)
        self.assertEqual(next_prime(20), 23)
    
    def test_prime_factors(self):
        """Test prime factorization."""
        self.assertEqual(prime_factors(2), [2])
        self.assertEqual(prime_factors(3), [3])
        self.assertEqual(prime_factors(4), [2, 2])
        self.assertEqual(prime_factors(12), [2, 2, 3])
        self.assertEqual(prime_factors(60), [2, 2, 3, 5])


if __name__ == "__main__":
    unittest.main()