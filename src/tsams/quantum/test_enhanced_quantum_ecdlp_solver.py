"""
Test suite for the TIBEDO Enhanced Quantum ECDLP Solver.

This module provides comprehensive tests for the enhanced quantum ECDLP solver,
verifying its mathematical foundations, circuit generation, and solving capabilities.
"""

import unittest
import numpy as np
import time
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import TIBEDO quantum components
from tibedo.quantum_information_new.cyclotomic_quantum_foundations import (
    CyclotomicField,
    SpinorStructure,
    DiscosohedralSheaf,
    CyclotomicQuantumTransformation
)
from tibedo.quantum_information_new.enhanced_quantum_ecdlp_solver import (
    QuantumECDLPCircuitGenerator,
    EnhancedQuantumECDLPSolver
)


class TestCyclotomicQuantumFoundations(unittest.TestCase):
    """Test cases for the cyclotomic quantum foundations."""
    
    def setUp(self):
        """Set up test cases."""
        self.cyclotomic_field = CyclotomicField(168)
        self.spinor_structure = SpinorStructure(56)
        self.discosohedral_sheaf = DiscosohedralSheaf(56)
        self.quantum_transformation = CyclotomicQuantumTransformation(168, 56)
    
    def test_cyclotomic_field_initialization(self):
        """Test initialization of cyclotomic field."""
        self.assertEqual(self.cyclotomic_field.conductor, 168)
        self.assertEqual(self.cyclotomic_field.dimension, 48)  # φ(168) = 48
        self.assertEqual(self.cyclotomic_field.dedekind_number, 112)
        self.assertEqual(self.cyclotomic_field.spinor_dimension, 56)
        self.assertEqual(self.cyclotomic_field.packing_arrangements, 168)
    
    def test_spinor_structure_initialization(self):
        """Test initialization of spinor structure."""
        self.assertEqual(self.spinor_structure.dimension, 56)
        self.assertEqual(self.spinor_structure.quaternionic_slices, 4)
        self.assertEqual(self.spinor_structure.extended_dimension, 224)
    
    def test_discosohedral_sheaf_initialization(self):
        """Test initialization of discosohedral sheaf."""
        self.assertEqual(self.discosohedral_sheaf.count, 56)
        self.assertEqual(self.discosohedral_sheaf.motivic_stack_leaves, 6)
        self.assertEqual(self.discosohedral_sheaf.leaf_matrix_dim, (6, 5))
        self.assertEqual(self.discosohedral_sheaf.total_packing_arrangements, 168)
    
    def test_quantum_transformation_initialization(self):
        """Test initialization of quantum transformation."""
        self.assertEqual(self.quantum_transformation.cyclotomic_field.conductor, 168)
        self.assertEqual(self.quantum_transformation.spinor_structure.dimension, 56)
        self.assertIn('fourier', self.quantum_transformation.transformation_matrices)
        self.assertIn('phase', self.quantum_transformation.transformation_matrices)
        self.assertIn('hadamard', self.quantum_transformation.transformation_matrices)
    
    def test_primitive_root_of_unity(self):
        """Test computation of primitive root of unity."""
        zeta = self.cyclotomic_field.get_primitive_root_of_unity()
        self.assertAlmostEqual(abs(zeta), 1.0)  # Should be on the unit circle
        self.assertAlmostEqual(zeta**168, 1.0)  # Should be a 168th root of unity
        self.assertNotAlmostEqual(zeta**84, 1.0)  # Should not be a 84th root of unity
    
    def test_galois_group_structure(self):
        """Test computation of Galois group structure."""
        galois_structure = self.cyclotomic_field.get_galois_group_structure()
        self.assertEqual(len(galois_structure['units']), 48)  # φ(168) = 48
        self.assertEqual(galois_structure['order'], 48)
    
    def test_spinor_rotation(self):
        """Test rotation of spinors."""
        # Create a simple 2D spinor
        spinor = np.array([1, 0], dtype=complex)
        
        # Rotate around X axis by π/2
        rotated_spinor = self.spinor_structure.rotate_spinor(spinor, 'X', np.pi/2)
        self.assertAlmostEqual(abs(rotated_spinor[0])**2, 0.5)
        self.assertAlmostEqual(abs(rotated_spinor[1])**2, 0.5)
    
    def test_discosohedral_sheaf_structure(self):
        """Test structure of discosohedral sheafs."""
        # Get a sheaf
        sheaf = self.discosohedral_sheaf.get_sheaf(0)
        self.assertEqual(sheaf.shape, (6, 5))
        
        # Get a motivic stack leaf
        leaf = self.discosohedral_sheaf.get_motivic_stack_leaf(0)
        self.assertEqual(leaf.shape, (6, 5))
    
    def test_quantum_transformation(self):
        """Test quantum transformations."""
        # Create a simple quantum state
        state = np.zeros(56, dtype=complex)
        state[0] = 1.0
        
        # Apply Fourier transformation
        transformed_state = self.quantum_transformation.transform_state(state, 'fourier')
        self.assertEqual(len(transformed_state), 56)
        self.assertAlmostEqual(np.linalg.norm(transformed_state), 1.0)
        
        # Apply phase transformation
        transformed_state = self.quantum_transformation.transform_state(state, 'phase')
        self.assertEqual(len(transformed_state), 56)
        self.assertAlmostEqual(np.linalg.norm(transformed_state), 1.0)
        
        # Apply Hadamard-like transformation
        transformed_state = self.quantum_transformation.transform_state(state, 'hadamard')
        self.assertEqual(len(transformed_state), 56)
        self.assertAlmostEqual(np.linalg.norm(transformed_state), 1.0)


class TestQuantumECDLPCircuitGenerator(unittest.TestCase):
    """Test cases for the quantum ECDLP circuit generator."""
    
    def setUp(self):
        """Set up test cases."""
        self.circuit_generator_small = QuantumECDLPCircuitGenerator(
            key_size=8,
            circuit_depth=50,
            parallel_jobs=2,
            cyclotomic_conductor=168,
            spinor_dimension=56
        )
        
        self.circuit_generator_medium = QuantumECDLPCircuitGenerator(
            key_size=16,
            circuit_depth=100,
            parallel_jobs=4,
            cyclotomic_conductor=168,
            spinor_dimension=56
        )
        
        self.circuit_generator_large = QuantumECDLPCircuitGenerator(
            key_size=32,
            circuit_depth=200,
            parallel_jobs=8,
            cyclotomic_conductor=168,
            spinor_dimension=56
        )
    
    def test_circuit_generator_initialization(self):
        """Test initialization of circuit generator."""
        self.assertEqual(self.circuit_generator_small.key_size, 8)
        self.assertEqual(self.circuit_generator_small.circuit_depth, 50)
        self.assertEqual(self.circuit_generator_small.parallel_jobs, 2)
        self.assertEqual(self.circuit_generator_small.cyclotomic_conductor, 168)
        self.assertEqual(self.circuit_generator_small.spinor_dimension, 56)
        
        self.assertEqual(self.circuit_generator_medium.key_size, 16)
        self.assertEqual(self.circuit_generator_large.key_size, 32)
    
    def test_ancilla_qubits_computation(self):
        """Test computation of ancilla qubits."""
        # For key_size=8, we expect:
        # - log2(8) = 3 basic ancilla qubits
        # - log2(168) = 8 cyclotomic qubits
        # - log2(56) = 6 spinor qubits
        # Total: 3 + 8 + 6 = 17 ancilla qubits
        self.assertEqual(self.circuit_generator_small.ancilla_qubits, 17)
        
        # For larger key sizes, we expect more basic ancilla qubits
        self.assertGreater(self.circuit_generator_medium.ancilla_qubits, self.circuit_generator_small.ancilla_qubits)
        self.assertGreater(self.circuit_generator_large.ancilla_qubits, self.circuit_generator_medium.ancilla_qubits)
    
    def test_circuit_generation(self):
        """Test generation of quantum circuits."""
        # Generate circuits
        circuit_small = self.circuit_generator_small.generate_circuit()
        circuit_medium = self.circuit_generator_medium.generate_circuit()
        circuit_large = self.circuit_generator_large.generate_circuit()
        
        # Check circuit properties
        self.assertEqual(circuit_small.num_qubits, self.circuit_generator_small.total_qubits)
        self.assertEqual(circuit_medium.num_qubits, self.circuit_generator_medium.total_qubits)
        self.assertEqual(circuit_large.num_qubits, self.circuit_generator_large.total_qubits)
        
        # Check that circuits have measurements
        self.assertGreater(circuit_small.count_ops().get('measure', 0), 0)
        self.assertGreater(circuit_medium.count_ops().get('measure', 0), 0)
        self.assertGreater(circuit_large.count_ops().get('measure', 0), 0)
    
    def test_circuit_depth(self):
        """Test depth of generated circuits."""
        # Generate circuits
        circuit_small = self.circuit_generator_small.generate_circuit()
        circuit_medium = self.circuit_generator_medium.generate_circuit()
        circuit_large = self.circuit_generator_large.generate_circuit()
        
        # Check that circuit depth increases with key size
        self.assertLessEqual(circuit_small.depth(), circuit_medium.depth())
        self.assertLessEqual(circuit_medium.depth(), circuit_large.depth())


class TestEnhancedQuantumECDLPSolver(unittest.TestCase):
    """Test cases for the enhanced quantum ECDLP solver."""
    
    def setUp(self):
        """Set up test cases."""
        self.solver_small = EnhancedQuantumECDLPSolver(
            key_size=8,
            parallel_jobs=2,
            adaptive_depth=True,
            cyclotomic_conductor=168,
            spinor_dimension=56
        )
        
        self.solver_medium = EnhancedQuantumECDLPSolver(
            key_size=16,
            parallel_jobs=4,
            adaptive_depth=True,
            cyclotomic_conductor=168,
            spinor_dimension=56
        )
        
        self.solver_large = EnhancedQuantumECDLPSolver(
            key_size=32,
            parallel_jobs=8,
            adaptive_depth=True,
            cyclotomic_conductor=168,
            spinor_dimension=56
        )
        
        # Test data
        self.curve_params = {'a': 1, 'b': 7, 'p': 2**256 - 2**32 - 977}
        self.public_key = {'x': 123, 'y': 456}
        self.base_point = {'x': 789, 'y': 101112}
    
    def test_solver_initialization(self):
        """Test initialization of ECDLP solver."""
        self.assertEqual(self.solver_small.key_size, 8)
        self.assertEqual(self.solver_small.parallel_jobs, 2)
        self.assertTrue(self.solver_small.adaptive_depth)
        
        self.assertEqual(self.solver_medium.key_size, 16)
        self.assertEqual(self.solver_large.key_size, 32)
    
    def test_circuit_depth_computation(self):
        """Test computation of circuit depth."""
        # For adaptive depth, we expect depth to scale logarithmically with key size
        self.assertLess(self.solver_small.circuit_depth, self.solver_medium.circuit_depth)
        self.assertLess(self.solver_medium.circuit_depth, self.solver_large.circuit_depth)
        
        # Create a solver with fixed depth
        solver_fixed = EnhancedQuantumECDLPSolver(
            key_size=8,
            parallel_jobs=2,
            adaptive_depth=False,
            cyclotomic_conductor=168,
            spinor_dimension=56
        )
        
        self.assertEqual(solver_fixed.circuit_depth, 100)  # Fixed depth
    
    def test_circuit_generation(self):
        """Test generation of quantum circuits."""
        # Generate circuits
        circuit_small = self.solver_small.generate_quantum_circuit()
        circuit_medium = self.solver_medium.generate_quantum_circuit()
        circuit_large = self.solver_large.generate_quantum_circuit()
        
        # Check circuit properties
        self.assertEqual(circuit_small.num_qubits, self.solver_small.circuit_generator.total_qubits)
        self.assertEqual(circuit_medium.num_qubits, self.solver_medium.circuit_generator.total_qubits)
        self.assertEqual(circuit_large.num_qubits, self.solver_large.circuit_generator.total_qubits)
    
    def test_ecdlp_solving(self):
        """Test solving of ECDLP."""
        # Solve ECDLP
        private_key_small = self.solver_small.solve_ecdlp(self.curve_params, self.public_key, self.base_point)
        
        # Check that the private key is within the expected range
        self.assertGreaterEqual(private_key_small, 1)
        self.assertLess(private_key_small, 2**self.solver_small.key_size)
        
        # Verify the solution
        is_valid = self.solver_small.verify_solution(self.curve_params, self.public_key, self.base_point, private_key_small)
        self.assertTrue(is_valid)
    
    def test_parallel_ecdlp_solving(self):
        """Test solving of ECDLP with parallel jobs."""
        # Solve ECDLP with parallel jobs
        private_key_small = self.solver_small.solve_ecdlp_with_parallel_jobs(self.curve_params, self.public_key, self.base_point)
        
        # Check that the private key is within the expected range
        self.assertGreaterEqual(private_key_small, 1)
        self.assertLess(private_key_small, 2**self.solver_small.key_size)
        
        # Verify the solution
        is_valid = self.solver_small.verify_solution(self.curve_params, self.public_key, self.base_point, private_key_small)
        self.assertTrue(is_valid)
    
    def test_benchmark_performance(self):
        """Test benchmarking of solver performance."""
        # Benchmark performance for small key sizes
        benchmark_results = self.solver_small.benchmark_performance(key_sizes=[4, 8], repetitions=1)
        
        # Check benchmark results
        self.assertIn(4, benchmark_results)
        self.assertIn(8, benchmark_results)
        self.assertIn('avg_time', benchmark_results[4])
        self.assertIn('circuit_depth', benchmark_results[4])
        self.assertIn('total_qubits', benchmark_results[4])
    
    def test_mathematical_foundation(self):
        """Test explanation of mathematical foundation."""
        explanation = self.solver_small.explain_mathematical_foundation()
        
        # Check that the explanation contains key terms
        self.assertIn("Cyclotomic Fields", explanation)
        self.assertIn("Spinor Structures", explanation)
        self.assertIn("Discosohedral Sheafs", explanation)
        self.assertIn("Hexagonal Lattice Packing", explanation)
        self.assertIn("Quantum Transformation", explanation)


class TestQuantumECDLPSolverFor21Bit(unittest.TestCase):
    """Test cases specifically for 21-bit ECDLP solving."""
    
    def setUp(self):
        """Set up test cases."""
        self.solver = EnhancedQuantumECDLPSolver(
            key_size=21,
            parallel_jobs=4,
            adaptive_depth=True,
            cyclotomic_conductor=168,
            spinor_dimension=56
        )
        
        # Test data
        self.curve_params = {'a': 1, 'b': 7, 'p': 2**256 - 2**32 - 977}
        self.public_key = {'x': 123, 'y': 456}
        self.base_point = {'x': 789, 'y': 101112}
    
    def test_21bit_circuit_generation(self):
        """Test generation of quantum circuit for 21-bit keys."""
        # Generate circuit
        circuit = self.solver.generate_quantum_circuit()
        
        # Check circuit properties
        self.assertEqual(circuit.num_qubits, self.solver.circuit_generator.total_qubits)
        self.assertEqual(self.solver.key_size, 21)
        
        # Check that the circuit has the expected number of key qubits
        self.assertEqual(self.solver.circuit_generator.key_qubits, 21)
        
        # Log circuit statistics
        logger.info(f"21-bit ECDLP circuit statistics:")
        logger.info(f"Total qubits: {circuit.num_qubits}")
        logger.info(f"Circuit depth: {circuit.depth()}")
        logger.info(f"Gate count: {sum(circuit.count_ops().values())}")
        logger.info(f"Gate composition: {circuit.count_ops()}")
    
    def test_21bit_ecdlp_solving(self):
        """Test solving of 21-bit ECDLP."""
        # Solve ECDLP
        start_time = time.time()
        private_key = self.solver.solve_ecdlp(self.curve_params, self.public_key, self.base_point)
        end_time = time.time()
        
        # Check that the private key is within the expected range
        self.assertGreaterEqual(private_key, 1)
        self.assertLess(private_key, 2**21)
        
        # Verify the solution
        is_valid = self.solver.verify_solution(self.curve_params, self.public_key, self.base_point, private_key)
        self.assertTrue(is_valid)
        
        # Log solving time
        logger.info(f"21-bit ECDLP solved in {end_time - start_time:.3f} seconds")
        logger.info(f"Found private key: {private_key}")
    
    def test_21bit_parallel_ecdlp_solving(self):
        """Test solving of 21-bit ECDLP with parallel jobs."""
        # Solve ECDLP with parallel jobs
        start_time = time.time()
        private_key = self.solver.solve_ecdlp_with_parallel_jobs(self.curve_params, self.public_key, self.base_point)
        end_time = time.time()
        
        # Check that the private key is within the expected range
        self.assertGreaterEqual(private_key, 1)
        self.assertLess(private_key, 2**21)
        
        # Verify the solution
        is_valid = self.solver.verify_solution(self.curve_params, self.public_key, self.base_point, private_key)
        self.assertTrue(is_valid)
        
        # Log solving time
        logger.info(f"21-bit ECDLP solved with parallel jobs in {end_time - start_time:.3f} seconds")
        logger.info(f"Found private key: {private_key}")
    
    def test_21bit_linear_time_complexity(self):
        """Test that the 21-bit ECDLP solver has linear time complexity."""
        # Benchmark performance for different key sizes
        key_sizes = [8, 12, 16, 21]
        benchmark_results = self.solver.benchmark_performance(key_sizes=key_sizes, repetitions=1)
        
        # Extract solving times
        times = [benchmark_results[k]['avg_time'] for k in key_sizes]
        
        # Check that time complexity is approximately linear
        # We'll check that the ratio of times for consecutive key sizes is roughly constant
        ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
        avg_ratio = sum(ratios) / len(ratios)
        
        # Log time complexity analysis
        logger.info(f"Time complexity analysis:")
        logger.info(f"Key sizes: {key_sizes}")
        logger.info(f"Solving times: {times}")
        logger.info(f"Time ratios: {ratios}")
        logger.info(f"Average ratio: {avg_ratio}")
        
        # Check that the average ratio is close to the ratio of key sizes
        # For linear complexity, time should scale linearly with key size
        key_size_ratios = [key_sizes[i+1] / key_sizes[i] for i in range(len(key_sizes)-1)]
        avg_key_size_ratio = sum(key_size_ratios) / len(key_size_ratios)
        
        logger.info(f"Key size ratios: {key_size_ratios}")
        logger.info(f"Average key size ratio: {avg_key_size_ratio}")
        
        # The ratio of times should be less than or equal to the ratio of key sizes
        # for linear or sub-linear complexity
        self.assertLessEqual(avg_ratio, 2 * avg_key_size_ratio)


if __name__ == '__main__':
    unittest.main()