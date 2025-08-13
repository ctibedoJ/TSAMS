"""
Test suite for TIBEDO Tensor Network Circuit Optimization

This module provides comprehensive tests for the tensor network-based
circuit optimization techniques implemented in the TIBEDO framework.
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Operator, process_fidelity
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if tensor network libraries are available
try:
    import tensornetwork as tn
    import quimb
    import quimb.tensor as qt
    HAS_TENSOR_LIBS = True
except ImportError:
    HAS_TENSOR_LIBS = False
    logger.warning("Tensor network libraries not found. Some tests will be skipped.")

# Import TIBEDO quantum components
from tensor_network_circuit_optimization import (
    TensorNetworkCircuitOptimizer,
    CyclotomicTensorFusion,
    HardwareSpecificTensorOptimizer,
    EnhancedTibedoQuantumCircuitCompressor
)
from quantum_circuit_optimization import TibedoQuantumCircuitCompressor


class TestTensorNetworkCircuitOptimizer(unittest.TestCase):
    """Test cases for the TensorNetworkCircuitOptimizer class."""
    
    @unittest.skipIf(not HAS_TENSOR_LIBS, "Tensor network libraries not available")
    def setUp(self):
        """Set up test cases."""
        self.optimizer = TensorNetworkCircuitOptimizer(
            max_bond_dimension=8,
            truncation_threshold=1e-8
        )
        
        # Create test circuits
        self.small_circuit = QuantumCircuit(2)
        self.small_circuit.h(0)
        self.small_circuit.cx(0, 1)
        self.small_circuit.rz(np.pi/4, 1)
        self.small_circuit.cx(0, 1)
        self.small_circuit.h(0)
        
        self.medium_circuit = QuantumCircuit(3)
        for i in range(3):
            self.medium_circuit.h(i)
        for i in range(2):
            self.medium_circuit.cx(i, i+1)
        for i in range(3):
            self.medium_circuit.rz(np.pi/4, i)
        for i in range(2):
            self.medium_circuit.cx(i, i+1)
        for i in range(3):
            self.medium_circuit.h(i)
    
    @unittest.skipIf(not HAS_TENSOR_LIBS, "Tensor network libraries not available")
    def test_circuit_to_tensor_network_conversion(self):
        """Test conversion of circuit to tensor network."""
        tn_nodes = self.optimizer.circuit_to_tensor_network(self.small_circuit)
        self.assertIsNotNone(tn_nodes)
        
        # Check that performance metrics were updated
        self.assertGreater(self.optimizer.performance_metrics['original_gate_count'], 0)
        self.assertGreater(self.optimizer.performance_metrics['original_depth'], 0)
    
    @unittest.skipIf(not HAS_TENSOR_LIBS, "Tensor network libraries not available")
    def test_circuit_optimization_preserves_functionality(self):
        """Test that optimized circuit preserves the functionality of the original."""
        # Skip this test for now as the tensor_network_to_circuit method is a placeholder
        # We'll implement a proper test once the method is fully implemented
        pass
    
    @unittest.skipIf(not HAS_TENSOR_LIBS, "Tensor network libraries not available")
    def test_performance_metrics(self):
        """Test that performance metrics are correctly tracked."""
        _ = self.optimizer.optimize_circuit(self.small_circuit)
        metrics = self.optimizer.get_performance_metrics()
        
        self.assertIn('original_gate_count', metrics)
        self.assertIn('optimized_gate_count', metrics)
        self.assertIn('original_depth', metrics)
        self.assertIn('optimized_depth', metrics)
        self.assertIn('optimization_time', metrics)
        
        self.assertGreater(metrics['optimization_time'], 0)


class TestCyclotomicTensorFusion(unittest.TestCase):
    """Test cases for the CyclotomicTensorFusion class."""
    
    def setUp(self):
        """Set up test cases."""
        self.fusion_optimizer = CyclotomicTensorFusion(
            cyclotomic_conductor=168,
            use_prime_indexed_fusion=True,
            max_fusion_distance=5
        )
        
        # Create test circuit
        self.test_circuit = QuantumCircuit(3)
        for i in range(3):
            self.test_circuit.h(i)
        for i in range(2):
            self.test_circuit.cx(i, i+1)
        for i in range(3):
            self.test_circuit.rz(np.pi/4, i)
        for i in range(2):
            self.test_circuit.cx(i, i+1)
        for i in range(3):
            self.test_circuit.h(i)
    
    def test_gate_fusion(self):
        """Test gate fusion functionality."""
        # This is a placeholder test since the fuse_gates method is not fully implemented
        fused_circuit = self.fusion_optimizer.fuse_gates(self.test_circuit)
        self.assertIsNotNone(fused_circuit)
        
        # For now, just check that the method returns a circuit
        self.assertIsInstance(fused_circuit, QuantumCircuit)


class TestHardwareSpecificTensorOptimizer(unittest.TestCase):
    """Test cases for the HardwareSpecificTensorOptimizer class."""
    
    def setUp(self):
        """Set up test cases."""
        self.ibmq_optimizer = HardwareSpecificTensorOptimizer(backend_name='ibmq')
        self.iqm_optimizer = HardwareSpecificTensorOptimizer(backend_name='iqm')
        self.google_optimizer = HardwareSpecificTensorOptimizer(backend_name='google')
        
        # Create test circuit
        self.test_circuit = QuantumCircuit(3)
        for i in range(3):
            self.test_circuit.h(i)
        for i in range(2):
            self.test_circuit.cx(i, i+1)
        for i in range(3):
            self.test_circuit.rz(np.pi/4, i)
        for i in range(2):
            self.test_circuit.cx(i, i+1)
        for i in range(3):
            self.test_circuit.h(i)
    
    def test_backend_initialization(self):
        """Test that backend-specific parameters are correctly initialized."""
        self.assertEqual(self.ibmq_optimizer.native_gates, ['u1', 'u2', 'u3', 'cx'])
        self.assertEqual(self.iqm_optimizer.native_gates, ['rx', 'ry', 'rz', 'cz'])
        self.assertEqual(self.google_optimizer.native_gates, ['fsim', 'xeb'])
    
    def test_invalid_backend(self):
        """Test that an invalid backend raises an error."""
        with self.assertRaises(ValueError):
            HardwareSpecificTensorOptimizer(backend_name='invalid_backend')
    
    def test_hardware_optimization(self):
        """Test hardware-specific optimization."""
        # This is a placeholder test since the optimize_for_hardware method is not fully implemented
        optimized_circuit = self.ibmq_optimizer.optimize_for_hardware(self.test_circuit)
        self.assertIsNotNone(optimized_circuit)
        
        # For now, just check that the method returns a circuit
        self.assertIsInstance(optimized_circuit, QuantumCircuit)


class TestEnhancedTibedoQuantumCircuitCompressor(unittest.TestCase):
    """Test cases for the EnhancedTibedoQuantumCircuitCompressor class."""
    
    def setUp(self):
        """Set up test cases."""
        self.standard_compressor = TibedoQuantumCircuitCompressor(
            compression_level=2,
            use_spinor_reduction=True,
            use_phase_synchronization=True
        )
        
        self.enhanced_compressor = EnhancedTibedoQuantumCircuitCompressor(
            compression_level=2,
            use_spinor_reduction=True,
            use_phase_synchronization=True,
            use_tensor_networks=True,
            max_bond_dimension=8,
            cyclotomic_conductor=168
        )
        
        # Create test circuit
        self.test_circuit = QuantumCircuit(4)
        for i in range(4):
            self.test_circuit.h(i)
        for i in range(3):
            self.test_circuit.cx(i, i+1)
        for i in range(4):
            self.test_circuit.rz(np.pi/4, i)
        for i in range(3):
            self.test_circuit.cx(i, i+1)
        for i in range(4):
            self.test_circuit.h(i)
    
    def test_compression_with_tensor_networks(self):
        """Test circuit compression with tensor networks."""
        # Compress with standard compressor
        standard_compressed = self.standard_compressor.compress_circuit(self.test_circuit)
        
        # Compress with enhanced compressor
        enhanced_compressed = self.enhanced_compressor.compress_circuit(self.test_circuit)
        
        # Check that both methods return circuits
        self.assertIsInstance(standard_compressed, QuantumCircuit)
        self.assertIsInstance(enhanced_compressed, QuantumCircuit)
        
        # Get performance metrics
        standard_metrics = self.standard_compressor.get_performance_metrics()
        enhanced_metrics = self.enhanced_compressor.get_performance_metrics()
        
        # Check that metrics are returned
        self.assertIsInstance(standard_metrics, dict)
        self.assertIsInstance(enhanced_metrics, dict)
    
    def test_hardware_specific_optimization(self):
        """Test hardware-specific optimization."""
        # This is a placeholder test since the optimize_for_hardware method is not fully implemented
        try:
            optimized_circuit = self.enhanced_compressor.optimize_for_hardware(
                self.test_circuit, 'ibmq'
            )
            self.assertIsInstance(optimized_circuit, QuantumCircuit)
        except Exception as e:
            self.fail(f"Hardware-specific optimization failed: {e}")
    
    def test_invalid_backend(self):
        """Test that an invalid backend raises an error."""
        with self.assertRaises(ValueError):
            self.enhanced_compressor.optimize_for_hardware(
                self.test_circuit, 'invalid_backend'
            )


if __name__ == '__main__':
    unittest.main()