"""
Test module for the TIBEDO Surface Code Error Correction implementation.

This module contains unit tests for the surface code error correction components,
including the SurfaceCode, SurfaceCodeEncoder, SyndromeExtractionCircuitGenerator,
and SurfaceCodeDecoder classes.
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the surface code error correction module
from tibedo.quantum_information_new.surface_code_error_correction import (
    SurfaceCode,
    SurfaceCodeEncoder,
    SyndromeExtractionCircuitGenerator,
    SurfaceCodeDecoder,
    CyclotomicSurfaceCode
)

class TestSurfaceCode(unittest.TestCase):
    """Test cases for the SurfaceCode class."""
    
    def setUp(self):
        """Set up test cases."""
        self.distance = 3
        self.logical_qubits = 1
        self.rotated_surface_code = SurfaceCode(
            distance=self.distance,
            logical_qubits=self.logical_qubits,
            use_rotated_lattice=True
        )
        self.standard_surface_code = SurfaceCode(
            distance=self.distance,
            logical_qubits=self.logical_qubits,
            use_rotated_lattice=False
        )
    
    def test_initialization(self):
        """Test surface code initialization."""
        # Test rotated surface code
        self.assertEqual(self.rotated_surface_code.distance, self.distance)
        self.assertEqual(self.rotated_surface_code.logical_qubits, self.logical_qubits)
        self.assertEqual(self.rotated_surface_code.physical_qubits_per_logical, self.distance**2)
        self.assertEqual(self.rotated_surface_code.total_physical_qubits, 
                         self.rotated_surface_code.physical_qubits_per_logical * self.logical_qubits)
        
        # Test standard surface code
        self.assertEqual(self.standard_surface_code.distance, self.distance)
        self.assertEqual(self.standard_surface_code.logical_qubits, self.logical_qubits)
        self.assertEqual(self.standard_surface_code.physical_qubits_per_logical, (self.distance + 1)**2 - 1)
        self.assertEqual(self.standard_surface_code.total_physical_qubits, 
                         self.standard_surface_code.physical_qubits_per_logical * self.logical_qubits)
    
    def test_lattice_structure(self):
        """Test surface code lattice structure."""
        # Test rotated surface code
        self.assertEqual(self.rotated_surface_code.qubit_grid.shape, (self.distance, self.distance))
        self.assertTrue(hasattr(self.rotated_surface_code, 'x_stabilizers'))
        self.assertTrue(hasattr(self.rotated_surface_code, 'z_stabilizers'))
        self.assertTrue(hasattr(self.rotated_surface_code, 'logical_x'))
        self.assertTrue(hasattr(self.rotated_surface_code, 'logical_z'))
        
        # Test standard surface code
        self.assertEqual(self.standard_surface_code.qubit_grid.shape, (self.distance + 1, self.distance + 1))
        self.assertTrue(hasattr(self.standard_surface_code, 'x_stabilizers'))
        self.assertTrue(hasattr(self.standard_surface_code, 'z_stabilizers'))
        self.assertTrue(hasattr(self.standard_surface_code, 'logical_x'))
        self.assertTrue(hasattr(self.standard_surface_code, 'logical_z'))
    
    def test_stabilizer_circuits(self):
        """Test stabilizer circuit generation."""
        # Test rotated surface code
        stabilizer_circuits = self.rotated_surface_code.get_stabilizer_circuits()
        self.assertIn('x_stabilizers', stabilizer_circuits)
        self.assertIn('z_stabilizers', stabilizer_circuits)
        self.assertEqual(len(stabilizer_circuits['x_stabilizers']), len(self.rotated_surface_code.x_stabilizers))
        self.assertEqual(len(stabilizer_circuits['z_stabilizers']), len(self.rotated_surface_code.z_stabilizers))
        
        # Test standard surface code
        stabilizer_circuits = self.standard_surface_code.get_stabilizer_circuits()
        self.assertIn('x_stabilizers', stabilizer_circuits)
        self.assertIn('z_stabilizers', stabilizer_circuits)
        self.assertEqual(len(stabilizer_circuits['x_stabilizers']), len(self.standard_surface_code.x_stabilizers))
        self.assertEqual(len(stabilizer_circuits['z_stabilizers']), len(self.standard_surface_code.z_stabilizers))
    
    def test_logical_operator_circuits(self):
        """Test logical operator circuit generation."""
        # Test rotated surface code
        logical_circuits = self.rotated_surface_code.get_logical_operator_circuits()
        self.assertIn('logical_x', logical_circuits)
        self.assertIn('logical_z', logical_circuits)
        
        # Test standard surface code
        logical_circuits = self.standard_surface_code.get_logical_operator_circuits()
        self.assertIn('logical_x', logical_circuits)
        self.assertIn('logical_z', logical_circuits)
    
    def test_visualize_lattice(self):
        """Test lattice visualization."""
        # Test rotated surface code
        fig = self.rotated_surface_code.visualize_lattice()
        self.assertIsInstance(fig, plt.Figure)
        
        # Test standard surface code
        fig = self.standard_surface_code.visualize_lattice()
        self.assertIsInstance(fig, plt.Figure)


class TestSurfaceCodeEncoder(unittest.TestCase):
    """Test cases for the SurfaceCodeEncoder class."""
    
    def setUp(self):
        """Set up test cases."""
        self.distance = 3
        self.logical_qubits = 1
        self.surface_code = SurfaceCode(
            distance=self.distance,
            logical_qubits=self.logical_qubits,
            use_rotated_lattice=True
        )
        self.encoder = SurfaceCodeEncoder(self.surface_code)
    
    def test_create_encoding_circuit(self):
        """Test encoding circuit creation."""
        # Test encoding |0⟩ state
        circuit_0 = self.encoder.create_encoding_circuit(initial_state='0')
        self.assertIsInstance(circuit_0, QuantumCircuit)
        self.assertEqual(circuit_0.num_qubits, self.surface_code.total_physical_qubits)
        
        # Test encoding |1⟩ state
        circuit_1 = self.encoder.create_encoding_circuit(initial_state='1')
        self.assertIsInstance(circuit_1, QuantumCircuit)
        
        # Test encoding |+⟩ state
        circuit_plus = self.encoder.create_encoding_circuit(initial_state='+')
        self.assertIsInstance(circuit_plus, QuantumCircuit)
        
        # Test encoding |-⟩ state
        circuit_minus = self.encoder.create_encoding_circuit(initial_state='-')
        self.assertIsInstance(circuit_minus, QuantumCircuit)


class TestSyndromeExtractionCircuitGenerator(unittest.TestCase):
    """Test cases for the SyndromeExtractionCircuitGenerator class."""
    
    def setUp(self):
        """Set up test cases."""
        self.distance = 3
        self.logical_qubits = 1
        self.surface_code = SurfaceCode(
            distance=self.distance,
            logical_qubits=self.logical_qubits,
            use_rotated_lattice=True
        )
        self.generator = SyndromeExtractionCircuitGenerator(
            self.surface_code,
            use_flag_qubits=True,
            use_fault_tolerant_extraction=True
        )
    
    def test_generate_syndrome_extraction_circuit(self):
        """Test syndrome extraction circuit generation."""
        circuit = self.generator.generate_syndrome_extraction_circuit()
        self.assertIsInstance(circuit, QuantumCircuit)
        
        # Check that the circuit includes the correct number of qubits
        expected_qubits = (
            self.surface_code.total_physical_qubits +  # Data qubits
            len(self.surface_code.x_stabilizers) +     # X-stabilizer ancillas
            len(self.surface_code.z_stabilizers) +     # Z-stabilizer ancillas
            len(self.surface_code.x_stabilizers) +     # X-stabilizer flags
            len(self.surface_code.z_stabilizers)       # Z-stabilizer flags
        )
        self.assertEqual(circuit.num_qubits, expected_qubits)
        
        # Check that the circuit includes measurements
        self.assertTrue(any(instr.operation.name == 'measure' for instr in circuit.data))


class TestSurfaceCodeDecoder(unittest.TestCase):
    """Test cases for the SurfaceCodeDecoder class."""
    
    def setUp(self):
        """Set up test cases."""
        self.distance = 3
        self.logical_qubits = 1
        self.surface_code = SurfaceCode(
            distance=self.distance,
            logical_qubits=self.logical_qubits,
            use_rotated_lattice=True
        )
        self.decoder = SurfaceCodeDecoder(self.surface_code)
    
    def test_initialization(self):
        """Test decoder initialization."""
        self.assertIsInstance(self.decoder.x_error_graph, nx.Graph)
        self.assertIsInstance(self.decoder.z_error_graph, nx.Graph)
    
    def test_decode_syndrome(self):
        """Test syndrome decoding."""
        # Create a test syndrome
        x_syndrome = [0, 1, 0]  # Example syndrome for X-stabilizers
        z_syndrome = [1, 0, 0]  # Example syndrome for Z-stabilizers
        
        # Decode the syndrome
        errors = self.decoder.decode_syndrome(x_syndrome, z_syndrome)
        
        # Check the result structure
        self.assertIn('x_errors', errors)
        self.assertIn('z_errors', errors)
        self.assertIsInstance(errors['x_errors'], list)
        self.assertIsInstance(errors['z_errors'], list)


class TestCyclotomicSurfaceCode(unittest.TestCase):
    """Test cases for the CyclotomicSurfaceCode class."""
    
    def setUp(self):
        """Set up test cases."""
        self.distance = 3
        self.logical_qubits = 1
        self.cyclotomic_conductor = 168
        self.cyclotomic_surface_code = CyclotomicSurfaceCode(
            distance=self.distance,
            logical_qubits=self.logical_qubits,
            use_rotated_lattice=True,
            cyclotomic_conductor=self.cyclotomic_conductor,
            use_prime_indexing=True
        )
    
    def test_initialization(self):
        """Test cyclotomic surface code initialization."""
        self.assertEqual(self.cyclotomic_surface_code.distance, self.distance)
        self.assertEqual(self.cyclotomic_surface_code.logical_qubits, self.logical_qubits)
        self.assertEqual(self.cyclotomic_surface_code.cyclotomic_conductor, self.cyclotomic_conductor)
        self.assertTrue(self.cyclotomic_surface_code.use_prime_indexing)
    
    def test_optimized_stabilizer_circuits(self):
        """Test optimized stabilizer circuit generation."""
        stabilizer_circuits = self.cyclotomic_surface_code.get_optimized_stabilizer_circuits()
        self.assertIn('x_stabilizers', stabilizer_circuits)
        self.assertIn('z_stabilizers', stabilizer_circuits)
        self.assertEqual(len(stabilizer_circuits['x_stabilizers']), 
                         len(self.cyclotomic_surface_code.x_stabilizers))
        self.assertEqual(len(stabilizer_circuits['z_stabilizers']), 
                         len(self.cyclotomic_surface_code.z_stabilizers))


if __name__ == '__main__':
    unittest.main()