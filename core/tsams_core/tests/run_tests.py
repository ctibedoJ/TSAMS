"""
Test runner for the prime_mobius package.
"""

import unittest
import sys
import os

# Add the parent directory to the path so that the tests can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the test modules
from prime_mobius.tests.test_moebius import TestMoebiusTransformation, TestPrimeIndexedMoebiusTransformation, TestRoot420Structure
from prime_mobius.tests.test_cyclotomic import TestCyclotomicField
from prime_mobius.tests.test_state_space import TestStateSpace, TestNodalStructure441


def run_tests():
    """
    Run all the tests for the prime_mobius package.
    """
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add the test cases
    test_suite.addTest(unittest.makeSuite(TestMoebiusTransformation))
    test_suite.addTest(unittest.makeSuite(TestPrimeIndexedMoebiusTransformation))
    test_suite.addTest(unittest.makeSuite(TestRoot420Structure))
    test_suite.addTest(unittest.makeSuite(TestCyclotomicField))
    test_suite.addTest(unittest.makeSuite(TestStateSpace))
    test_suite.addTest(unittest.makeSuite(TestNodalStructure441))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return the result
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(not result.wasSuccessful())