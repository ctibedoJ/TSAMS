"""
Test module for the TIBEDO Zero-Noise Extrapolation implementation.

This module contains unit tests for the zero-noise extrapolation components,
including the ZeroNoiseExtrapolator, RichardsonExtrapolator, ExponentialExtrapolator,
PolynomialExtrapolator, and CyclotomicExtrapolator classes.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the zero-noise extrapolation module
from tibedo.quantum_information_new.zero_noise_extrapolation import (
    ZeroNoiseExtrapolator,
    RichardsonExtrapolator,
    ExponentialExtrapolator,
    PolynomialExtrapolator,
    CyclotomicExtrapolator
)

class TestZeroNoiseExtrapolator(unittest.TestCase):
    """Test cases for the ZeroNoiseExtrapolator class."""
    
    def setUp(self):
        """Set up test cases."""
        # Create a simple quantum circuit for testing
        self.circuit = QuantumCircuit(2, 2)
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.measure([0, 1], [0, 1])
        
        # Create a simulator backend
        self.simulator = Aer.get_backend('qasm_simulator')
        
        # Create a base extrapolator for testing
        self.extrapolator = RichardsonExtrapolator(
            noise_scaling_method='gate_stretching',
            scale_factors=[1.0, 2.0, 3.0],
            order=1
        )
    
    def test_initialization(self):
        """Test extrapolator initialization."""
        # Test with default parameters
        extrapolator = ZeroNoiseExtrapolator()
        self.assertEqual(extrapolator.noise_scaling_method, 'gate_stretching')
        self.assertEqual(extrapolator.scale_factors, [1.0, 2.0, 3.0])
        
        # Test with custom parameters
        extrapolator = ZeroNoiseExtrapolator(
            noise_scaling_method='parameter_scaling',
            scale_factors=[1.0, 1.5, 2.0, 2.5]
        )
        self.assertEqual(extrapolator.noise_scaling_method, 'parameter_scaling')
        self.assertEqual(extrapolator.scale_factors, [1.0, 1.5, 2.0, 2.5])
    
    def test_scale_circuit(self):
        """Test circuit scaling."""
        # Test gate stretching
        scaled_circuit = self.extrapolator.scale_circuit(self.circuit, 2.0)
        self.assertIsInstance(scaled_circuit, QuantumCircuit)
        self.assertEqual(scaled_circuit.num_qubits, self.circuit.num_qubits)
        self.assertEqual(scaled_circuit.num_clbits, self.circuit.num_clbits)
        
        # Check that the scaled circuit has more operations (excluding measurements and barriers)
        original_ops = len([op for op in self.circuit.data if op.operation.name not in ['measure', 'barrier']])
        scaled_ops = len([op for op in scaled_circuit.data if op.operation.name not in ['measure', 'barrier']])
        self.assertGreaterEqual(scaled_ops, original_ops)
    
    def test_default_observable(self):
        """Test the default observable function."""
        # Create a mock counts dictionary
        counts = {'00': 500, '01': 200, '10': 200, '11': 100}
        
        # Compute the observable
        observable_value = self.extrapolator._default_observable(counts)
        
        # The default observable is the expectation value of |0><0|, which is the probability of the '00' state
        expected_value = 500 / 1000
        self.assertEqual(observable_value, expected_value)
    
    def test_extrapolate_method_exists(self):
        """Test that the extrapolate method exists."""
        self.assertTrue(hasattr(self.extrapolator, 'extrapolate'))
        self.assertTrue(callable(getattr(self.extrapolator, 'extrapolate')))
    
    def test_visualize_extrapolation_method_exists(self):
        """Test that the visualize_extrapolation method exists."""
        self.assertTrue(hasattr(self.extrapolator, 'visualize_extrapolation'))
        self.assertTrue(callable(getattr(self.extrapolator, 'visualize_extrapolation')))


class TestRichardsonExtrapolator(unittest.TestCase):
    """Test cases for the RichardsonExtrapolator class."""
    
    def setUp(self):
        """Set up test cases."""
        # Create a Richardson extrapolator for testing
        self.extrapolator = RichardsonExtrapolator(
            noise_scaling_method='gate_stretching',
            scale_factors=[1.0, 2.0, 3.0],
            order=1
        )
    
    def test_initialization(self):
        """Test Richardson extrapolator initialization."""
        self.assertEqual(self.extrapolator.noise_scaling_method, 'gate_stretching')
        self.assertEqual(self.extrapolator.scale_factors, [1.0, 2.0, 3.0])
        self.assertEqual(self.extrapolator.order, 1)
        
        # Test with custom parameters
        extrapolator = RichardsonExtrapolator(
            noise_scaling_method='parameter_scaling',
            scale_factors=[1.0, 1.5, 2.0, 2.5],
            order=2
        )
        self.assertEqual(extrapolator.noise_scaling_method, 'parameter_scaling')
        self.assertEqual(extrapolator.scale_factors, [1.0, 1.5, 2.0, 2.5])
        self.assertEqual(extrapolator.order, 2)
    
    def test_extrapolate_to_zero(self):
        """Test Richardson extrapolation to zero."""
        # Create some test data
        scale_factors = [1.0, 2.0, 3.0]
        expectation_values = [0.9, 0.8, 0.7]
        
        # Extrapolate to zero
        extrapolated_value = self.extrapolator._extrapolate_to_zero(scale_factors, expectation_values)
        
        # For linear extrapolation, the expected value is 1.0
        self.assertAlmostEqual(extrapolated_value, 1.0, places=6)
    
    def test_richardson_extrapolation(self):
        """Test the Richardson extrapolation algorithm."""
        # Create some test data
        scale_factors = [1.0, 2.0, 3.0]
        expectation_values = [0.9, 0.8, 0.7]
        
        # Perform Richardson extrapolation
        extrapolated_value = self.extrapolator._richardson_extrapolation(scale_factors, expectation_values)
        
        # For linear extrapolation, the expected value is 1.0
        self.assertAlmostEqual(extrapolated_value, 1.0, places=6)
    
    def test_extrapolation_curve(self):
        """Test the extrapolation curve computation."""
        # Create some test data
        scale_factors = [1.0, 2.0, 3.0]
        expectation_values = [0.9, 0.8, 0.7]
        
        # Compute the extrapolation curve
        x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        y = self.extrapolator._extrapolation_curve(x, scale_factors, expectation_values)
        
        # For linear extrapolation, the expected curve is a straight line
        expected_y = 1.0 - 0.1 * x
        np.testing.assert_allclose(y, expected_y, rtol=1e-6)


class TestExponentialExtrapolator(unittest.TestCase):
    """Test cases for the ExponentialExtrapolator class."""
    
    def setUp(self):
        """Set up test cases."""
        # Create an exponential extrapolator for testing
        self.extrapolator = ExponentialExtrapolator(
            noise_scaling_method='gate_stretching',
            scale_factors=[1.0, 2.0, 3.0]
        )
    
    def test_initialization(self):
        """Test exponential extrapolator initialization."""
        self.assertEqual(self.extrapolator.noise_scaling_method, 'gate_stretching')
        self.assertEqual(self.extrapolator.scale_factors, [1.0, 2.0, 3.0])
        
        # Test with custom parameters
        extrapolator = ExponentialExtrapolator(
            noise_scaling_method='parameter_scaling',
            scale_factors=[1.0, 1.5, 2.0, 2.5]
        )
        self.assertEqual(extrapolator.noise_scaling_method, 'parameter_scaling')
        self.assertEqual(extrapolator.scale_factors, [1.0, 1.5, 2.0, 2.5])
    
    def test_extrapolate_to_zero(self):
        """Test exponential extrapolation to zero."""
        # Create some test data that follows an exponential decay
        scale_factors = [1.0, 2.0, 3.0]
        a, b, c = 0.5, -0.2, 0.5
        expectation_values = [a * np.exp(b * x) + c for x in scale_factors]
        
        # Extrapolate to zero
        extrapolated_value = self.extrapolator._extrapolate_to_zero(scale_factors, expectation_values)
        
        # The expected value is a * exp(b * 0) + c = a + c
        expected_value = a + c
        self.assertAlmostEqual(extrapolated_value, expected_value, places=6)
    
    def test_extrapolation_curve(self):
        """Test the extrapolation curve computation."""
        # Create some test data that follows an exponential decay
        scale_factors = [1.0, 2.0, 3.0]
        a, b, c = 0.5, -0.2, 0.5
        expectation_values = [a * np.exp(b * x) + c for x in scale_factors]
        
        # Compute the extrapolation curve
        x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        y = self.extrapolator._extrapolation_curve(x, scale_factors, expectation_values)
        
        # The expected curve is a * exp(b * x) + c
        expected_y = a * np.exp(b * x) + c
        np.testing.assert_allclose(y, expected_y, rtol=1e-6)


class TestPolynomialExtrapolator(unittest.TestCase):
    """Test cases for the PolynomialExtrapolator class."""
    
    def setUp(self):
        """Set up test cases."""
        # Create a polynomial extrapolator for testing
        self.extrapolator = PolynomialExtrapolator(
            noise_scaling_method='gate_stretching',
            scale_factors=[1.0, 2.0, 3.0],
            degree=2
        )
    
    def test_initialization(self):
        """Test polynomial extrapolator initialization."""
        self.assertEqual(self.extrapolator.noise_scaling_method, 'gate_stretching')
        self.assertEqual(self.extrapolator.scale_factors, [1.0, 2.0, 3.0])
        self.assertEqual(self.extrapolator.degree, 2)
        
        # Test with custom parameters
        extrapolator = PolynomialExtrapolator(
            noise_scaling_method='parameter_scaling',
            scale_factors=[1.0, 1.5, 2.0, 2.5],
            degree=3
        )
        self.assertEqual(extrapolator.noise_scaling_method, 'parameter_scaling')
        self.assertEqual(extrapolator.scale_factors, [1.0, 1.5, 2.0, 2.5])
        self.assertEqual(extrapolator.degree, 3)
    
    def test_extrapolate_to_zero(self):
        """Test polynomial extrapolation to zero."""
        # Create some test data that follows a quadratic polynomial
        scale_factors = [1.0, 2.0, 3.0]
        a, b, c = -0.1, 0.0, 1.0
        expectation_values = [a * x**2 + b * x + c for x in scale_factors]
        
        # Extrapolate to zero
        extrapolated_value = self.extrapolator._extrapolate_to_zero(scale_factors, expectation_values)
        
        # The expected value is a * 0^2 + b * 0 + c = c
        expected_value = c
        self.assertAlmostEqual(extrapolated_value, expected_value, places=6)
    
    def test_extrapolation_curve(self):
        """Test the extrapolation curve computation."""
        # Create some test data that follows a quadratic polynomial
        scale_factors = [1.0, 2.0, 3.0]
        a, b, c = -0.1, 0.0, 1.0
        expectation_values = [a * x**2 + b * x + c for x in scale_factors]
        
        # Compute the extrapolation curve
        x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        y = self.extrapolator._extrapolation_curve(x, scale_factors, expectation_values)
        
        # The expected curve is a * x^2 + b * x + c
        expected_y = a * x**2 + b * x + c
        np.testing.assert_allclose(y, expected_y, rtol=1e-6)


class TestCyclotomicExtrapolator(unittest.TestCase):
    """Test cases for the CyclotomicExtrapolator class."""
    
    def setUp(self):
        """Set up test cases."""
        # Create a cyclotomic extrapolator for testing
        self.extrapolator = CyclotomicExtrapolator(
            noise_scaling_method='gate_stretching',
            scale_factors=[1.0, 2.0, 3.0],
            cyclotomic_conductor=168,
            use_prime_indexing=True
        )
    
    def test_initialization(self):
        """Test cyclotomic extrapolator initialization."""
        self.assertEqual(self.extrapolator.noise_scaling_method, 'gate_stretching')
        self.assertEqual(self.extrapolator.scale_factors, [1.0, 2.0, 3.0])
        self.assertEqual(self.extrapolator.cyclotomic_conductor, 168)
        self.assertTrue(self.extrapolator.use_prime_indexing)
        
        # Test with custom parameters
        extrapolator = CyclotomicExtrapolator(
            noise_scaling_method='parameter_scaling',
            scale_factors=[1.0, 1.5, 2.0, 2.5],
            cyclotomic_conductor=120,
            use_prime_indexing=False
        )
        self.assertEqual(extrapolator.noise_scaling_method, 'parameter_scaling')
        self.assertEqual(extrapolator.scale_factors, [1.0, 1.5, 2.0, 2.5])
        self.assertEqual(extrapolator.cyclotomic_conductor, 120)
        self.assertFalse(extrapolator.use_prime_indexing)
    
    def test_extrapolate_to_zero(self):
        """Test cyclotomic extrapolation to zero."""
        # Create some test data
        scale_factors = [1.0, 2.0, 3.0]
        expectation_values = [0.9, 0.8, 0.7]
        
        # Extrapolate to zero
        extrapolated_value = self.extrapolator._extrapolate_to_zero(scale_factors, expectation_values)
        
        # The expected value should be between the Richardson and polynomial extrapolations
        richardson = RichardsonExtrapolator(
            noise_scaling_method='gate_stretching',
            scale_factors=scale_factors,
            order=1
        )
        richardson_value = richardson._extrapolate_to_zero(scale_factors, expectation_values)
        
        poly = PolynomialExtrapolator(
            noise_scaling_method='gate_stretching',
            scale_factors=scale_factors,
            degree=2
        )
        poly_value = poly._extrapolate_to_zero(scale_factors, expectation_values)
        
        self.assertGreaterEqual(extrapolated_value, min(richardson_value, poly_value))
        self.assertLessEqual(extrapolated_value, max(richardson_value, poly_value))
    
    def test_extrapolation_curve(self):
        """Test the extrapolation curve computation."""
        # Create some test data
        scale_factors = [1.0, 2.0, 3.0]
        expectation_values = [0.9, 0.8, 0.7]
        
        # Compute the extrapolation curve
        x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        y = self.extrapolator._extrapolation_curve(x, scale_factors, expectation_values)
        
        # The curve should be a weighted average of the Richardson and polynomial curves
        richardson = RichardsonExtrapolator(
            noise_scaling_method='gate_stretching',
            scale_factors=scale_factors,
            order=1
        )
        richardson_curve = richardson._extrapolation_curve(x, scale_factors, expectation_values)
        
        poly = PolynomialExtrapolator(
            noise_scaling_method='gate_stretching',
            scale_factors=scale_factors,
            degree=2
        )
        poly_curve = poly._extrapolation_curve(x, scale_factors, expectation_values)
        
        # Check that the curve is between the Richardson and polynomial curves
        for i in range(len(x)):
            self.assertGreaterEqual(y[i], min(richardson_curve[i], poly_curve[i]))
            self.assertLessEqual(y[i], max(richardson_curve[i], poly_curve[i]))


if __name__ == '__main__':
    unittest.main()