"""
Test script for TIBEDO Enhanced Quantum VQE Module

This script tests the advanced VQE capabilities including natural gradient
optimization, quantum neural networks with spinor encoding, and distributed
quantum-classical computation.
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.opflow import X, Y, Z, I, PauliSumOp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import enhanced VQE classes
from enhanced_quantum_vqe import (
    NaturalGradientVQE,
    SpinorQuantumNN,
    DistributedQuantumOptimizer
)

class TestNaturalGradientVQE(unittest.TestCase):
    """Test cases for NaturalGradientVQE"""
    
    def setUp(self):
        """Set up test environment"""
        self.backend = Aer.get_backend('statevector_simulator')
        self.vqe = NaturalGradientVQE(
            backend=self.backend,
            optimizer_method='NATURAL_GRADIENT',
            max_iterations=5,  # Small number for testing
            use_spinor_reduction=True,
            use_phase_synchronization=True,
            use_prime_indexing=True,
            natural_gradient_reg=0.01,
            qfim_approximation='diag',
            learning_rate=0.1,
            adaptive_learning_rate=True
        )
        
        # Create a simple Hamiltonian
        self.hamiltonian = 0.5 * (X ^ X) + 0.3 * (Z ^ Z)
        
        # Create a simple ansatz circuit
        self.ansatz = QuantumCircuit(2)
        self.ansatz.rx('theta', 0)
        self.ansatz.ry('phi', 1)
        self.ansatz.cx(0, 1)
    
    def test_initialization(self):
        """Test initialization of NaturalGradientVQE"""
        self.assertEqual(self.vqe.optimizer_method, 'NATURAL_GRADIENT')
        self.assertEqual(self.vqe.max_iterations, 5)
        self.assertTrue(self.vqe.use_spinor_reduction)
        self.assertTrue(self.vqe.use_phase_synchronization)
        self.assertTrue(self.vqe.use_prime_indexing)
        self.assertEqual(self.vqe.natural_gradient_reg, 0.01)
        self.assertEqual(self.vqe.qfim_approximation, 'diag')
        self.assertEqual(self.vqe.learning_rate, 0.1)
        self.assertTrue(self.vqe.adaptive_learning_rate)
    
    def test_optimize_parameters(self):
        """Test parameter optimization"""
        result = self.vqe.optimize_parameters(self.ansatz, self.hamiltonian)
        
        # Check that result contains expected keys
        self.assertIn('optimal_parameters', result)
        self.assertIn('optimal_value', result)
        self.assertIn('optimization_history', result)
        self.assertIn('num_iterations', result)
        self.assertIn('gradient_history', result)
        self.assertIn('qfim_history', result)
        self.assertIn('learning_rate_history', result)
        self.assertIn('success', result)
        
        # Check that optimal value is reasonable
        self.assertLess(result['optimal_value'], 0.0)
        
        # Check that optimization history is populated
        self.assertGreater(len(result['optimization_history']), 0)
        
        # Check that gradient history is populated
        self.assertGreater(len(result['gradient_history']), 0)
        
        # Check that QFIM history is populated
        self.assertGreater(len(result['qfim_history']), 0)
        
        # Check that learning rate history is populated
        self.assertGreater(len(result['learning_rate_history']), 0)
    
    def test_evaluate_energy(self):
        """Test energy evaluation"""
        # Set random parameters
        parameters = np.random.rand(2) * 2 * np.pi
        
        # Evaluate energy
        energy = self.vqe._evaluate_energy(parameters)
        
        # Check that energy is a float
        self.assertIsInstance(energy, float)
        
        # Check that energy is within reasonable bounds
        self.assertGreaterEqual(energy, -1.0)
        self.assertLessEqual(energy, 1.0)
    
    def test_compute_gradient(self):
        """Test gradient computation"""
        # Set random parameters
        parameters = np.random.rand(2) * 2 * np.pi
        
        # Store ansatz and hamiltonian
        self.vqe.ansatz = self.ansatz
        self.vqe.hamiltonian = self.hamiltonian
        
        # Compute gradient
        gradient = self.vqe._compute_gradient(parameters)
        
        # Check that gradient has correct shape
        self.assertEqual(gradient.shape, parameters.shape)
        
        # Check that gradient values are finite
        self.assertTrue(np.all(np.isfinite(gradient)))
    
    def test_compute_qfim(self):
        """Test QFIM computation"""
        # Set random parameters
        parameters = np.random.rand(2) * 2 * np.pi
        
        # Store ansatz and hamiltonian
        self.vqe.ansatz = self.ansatz
        self.vqe.hamiltonian = self.hamiltonian
        
        # Compute QFIM
        qfim = self.vqe._compute_qfim(parameters)
        
        # Check that QFIM has correct shape
        self.assertEqual(qfim.shape, (len(parameters), len(parameters)))
        
        # Check that QFIM is symmetric
        self.assertTrue(np.allclose(qfim, qfim.T))
        
        # Check that QFIM is positive definite
        eigvals = np.linalg.eigvals(qfim)
        self.assertTrue(np.all(eigvals > 0))
    
    def test_compute_natural_gradient(self):
        """Test natural gradient computation"""
        # Set random parameters and gradient
        parameters = np.random.rand(2) * 2 * np.pi
        gradient = np.random.rand(2)
        
        # Store ansatz and hamiltonian
        self.vqe.ansatz = self.ansatz
        self.vqe.hamiltonian = self.hamiltonian
        
        # Compute QFIM
        qfim = self.vqe._compute_qfim(parameters)
        
        # Compute natural gradient
        natural_gradient = self.vqe._compute_natural_gradient(gradient, qfim)
        
        # Check that natural gradient has correct shape
        self.assertEqual(natural_gradient.shape, gradient.shape)
        
        # Check that natural gradient values are finite
        self.assertTrue(np.all(np.isfinite(natural_gradient)))
    
    def test_analyze_convergence(self):
        """Test convergence analysis"""
        # Create mock optimization history
        optimization_history = [
            (np.array([0.1, 0.2]), -0.5),
            (np.array([0.2, 0.3]), -0.6),
            (np.array([0.3, 0.4]), -0.7),
            (np.array([0.4, 0.5]), -0.8)
        ]
        
        # Set up mock history attributes
        self.vqe.energy_history = [-0.5, -0.6, -0.7, -0.8]
        self.vqe.qfim_history = [np.eye(2) for _ in range(3)]
        self.vqe.learning_rate_history = [0.1, 0.11, 0.12]
        
        # Analyze convergence
        convergence = self.vqe.analyze_convergence(optimization_history)
        
        # Check that convergence contains expected keys
        self.assertIn('iterations', convergence)
        self.assertIn('energy_values', convergence)
        self.assertIn('energy_improvement', convergence)
        self.assertIn('parameter_changes', convergence)
        
        # Check that convergence contains natural gradient specific keys
        self.assertIn('qfim_condition_numbers', convergence)
        self.assertIn('learning_rate_history', convergence)
        
        # Check that convergence rates are calculated
        self.assertIn('convergence_rates', convergence)
        self.assertIn('mean_convergence_rate', convergence)


class TestSpinorQuantumNN(unittest.TestCase):
    """Test cases for SpinorQuantumNN"""
    
    def setUp(self):
        """Set up test environment"""
        self.backend = Aer.get_backend('statevector_simulator')
        self.qnn = SpinorQuantumNN(
            backend=self.backend,
            optimizer_method='NATURAL_GRADIENT',
            max_iterations=5,  # Small number for testing
            use_spinor_encoding=True,
            use_phase_synchronization=True,
            feature_map_type='SpinorFeatureMap',
            variational_form_type='SpinorCircuit',
            learning_rate=0.1,
            adaptive_learning_rate=True,
            use_quantum_backprop=True
        )
        
        # Create synthetic data
        np.random.seed(42)
        self.X = np.random.rand(10, 2)
        self.y = (self.X[:, 0] > 0.5).astype(float)
        
        # Create test data
        self.X_test = np.random.rand(5, 2)
        self.y_test = (self.X_test[:, 0] > 0.5).astype(float)
    
    def test_initialization(self):
        """Test initialization of SpinorQuantumNN"""
        self.assertEqual(self.qnn.optimizer_method, 'NATURAL_GRADIENT')
        self.assertEqual(self.qnn.max_iterations, 5)
        self.assertTrue(self.qnn.use_spinor_encoding)
        self.assertTrue(self.qnn.use_phase_synchronization)
        self.assertEqual(self.qnn.feature_map_type, 'SpinorFeatureMap')
        self.assertEqual(self.qnn.variational_form_type, 'SpinorCircuit')
        self.assertEqual(self.qnn.learning_rate, 0.1)
        self.assertTrue(self.qnn.adaptive_learning_rate)
        self.assertTrue(self.qnn.use_quantum_backprop)
    
    def test_encode_data_as_spinors(self):
        """Test data encoding as spinors"""
        # Encode data
        encoded_data = self.qnn.encode_data_as_spinors(self.X)
        
        # Check that encoded data has correct shape
        self.assertEqual(encoded_data.shape, self.X.shape)
        
        # Check that encoded data is complex
        self.assertEqual(encoded_data.dtype, np.complex128)
        
        # Check that magnitudes are approximately 1
        magnitudes = np.abs(encoded_data)
        self.assertTrue(np.allclose(magnitudes, 1.0, atol=1e-6))
    
    def test_generate_quantum_feature_map(self):
        """Test quantum feature map generation"""
        # Generate feature map
        feature_map = self.qnn.generate_quantum_feature_map(self.X)
        
        # Check that feature map is a QuantumCircuit
        self.assertIsInstance(feature_map, QuantumCircuit)
        
        # Check that feature map has correct number of qubits
        self.assertEqual(feature_map.num_qubits, self.X.shape[1])
        
        # Check that feature map has parameters
        self.assertGreater(len(feature_map.parameters), 0)
    
    def test_create_variational_circuit(self):
        """Test variational circuit creation"""
        # Generate feature map
        feature_map = self.qnn.generate_quantum_feature_map(self.X)
        
        # Create variational circuit
        var_circuit = self.qnn.create_variational_circuit(feature_map)
        
        # Check that variational circuit is a QuantumCircuit
        self.assertIsInstance(var_circuit, QuantumCircuit)
        
        # Check that variational circuit has correct number of qubits
        self.assertEqual(var_circuit.num_qubits, self.X.shape[1])
        
        # Check that variational circuit has parameters
        self.assertGreater(len(var_circuit.parameters), 0)
        
        # Check that variational circuit has classical register for measurement
        self.assertGreater(var_circuit.num_clbits, 0)
    
    def test_train_quantum_model(self):
        """Test quantum model training"""
        # Generate feature map
        feature_map = self.qnn.generate_quantum_feature_map(self.X)
        
        # Create variational circuit
        var_circuit = self.qnn.create_variational_circuit(feature_map)
        
        # Train model
        result = self.qnn.train_quantum_model(var_circuit, self.X, self.y)
        
        # Check that result contains expected keys
        self.assertIn('trained_parameters', result)
        self.assertIn('final_loss', result)
        self.assertIn('training_history', result)
        self.assertIn('num_iterations', result)
        self.assertIn('success', result)
        
        # Check that training history is populated
        self.assertGreater(len(result['training_history']), 0)
        
        # Check that trained parameters are available
        self.assertIsNotNone(result['trained_parameters'])
        self.assertGreater(len(result['trained_parameters']), 0)
    
    def test_predict(self):
        """Test model prediction"""
        # Generate feature map
        feature_map = self.qnn.generate_quantum_feature_map(self.X)
        
        # Create variational circuit
        var_circuit = self.qnn.create_variational_circuit(feature_map)
        
        # Store circuit
        self.qnn.circuit = var_circuit
        
        # Create random parameters
        parameters = np.random.rand(len(var_circuit.parameters))
        
        # Make predictions
        predictions = self.qnn.predict(self.X_test, parameters)
        
        # Check that predictions have correct shape
        self.assertEqual(predictions.shape, self.y_test.shape)
        
        # Check that predictions are between 0 and 1
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))
    
    def test_evaluate_model_performance(self):
        """Test model performance evaluation"""
        # Create mock training result
        training_result = {
            'trained_parameters': np.random.rand(10),
            'final_loss': 0.1,
            'training_history': [(0, 0.5), (1, 0.3), (2, 0.1)],
            'num_iterations': 3,
            'success': True
        }
        
        # Generate feature map
        feature_map = self.qnn.generate_quantum_feature_map(self.X)
        
        # Create variational circuit
        var_circuit = self.qnn.create_variational_circuit(feature_map)
        
        # Store circuit
        self.qnn.circuit = var_circuit
        
        # Mock predict method to avoid actual quantum execution
        self.qnn.predict = lambda X, params: (X[:, 0] > 0.5).astype(float)
        
        # Evaluate model performance
        performance = self.qnn.evaluate_model_performance(training_result, self.X_test, self.y_test)
        
        # Check that performance contains expected keys
        self.assertIn('mse', performance)
        self.assertIn('rmse', performance)
        self.assertIn('accuracy', performance)
        
        # Check that metrics are reasonable
        self.assertGreaterEqual(performance['mse'], 0.0)
        self.assertGreaterEqual(performance['rmse'], 0.0)
        self.assertGreaterEqual(performance['accuracy'], 0.0)
        self.assertLessEqual(performance['accuracy'], 1.0)


class TestDistributedQuantumOptimizer(unittest.TestCase):
    """Test cases for DistributedQuantumOptimizer"""
    
    def setUp(self):
        """Set up test environment"""
        # Create multiple backends
        self.backends = [Aer.get_backend('statevector_simulator') for _ in range(2)]
        
        self.optimizer = DistributedQuantumOptimizer(
            backends=self.backends,
            optimizer_method='COBYLA',
            max_iterations=5,  # Small number for testing
            use_spinor_reduction=True,
            use_phase_synchronization=True,
            use_prime_indexing=True,
            distribution_strategy='parameter_split',
            num_workers=2,
            synchronization_frequency=2
        )
        
        # Create a simple Hamiltonian
        self.hamiltonian = 0.5 * (X ^ X) + 0.3 * (Z ^ Z)
        
        # Create a simple ansatz circuit
        self.ansatz = QuantumCircuit(2)
        self.ansatz.rx('theta', 0)
        self.ansatz.ry('phi', 1)
        self.ansatz.cx(0, 1)
    
    def test_initialization(self):
        """Test initialization of DistributedQuantumOptimizer"""
        self.assertEqual(len(self.optimizer.backends), 2)
        self.assertEqual(self.optimizer.optimizer_method, 'COBYLA')
        self.assertEqual(self.optimizer.max_iterations, 5)
        self.assertTrue(self.optimizer.use_spinor_reduction)
        self.assertTrue(self.optimizer.use_phase_synchronization)
        self.assertTrue(self.optimizer.use_prime_indexing)
        self.assertEqual(self.optimizer.distribution_strategy, 'parameter_split')
        self.assertEqual(self.optimizer.num_workers, 2)
        self.assertEqual(self.optimizer.synchronization_frequency, 2)
    
    def test_optimize_vqe(self):
        """Test VQE optimization"""
        # Mock _evaluate_energy_distributed to avoid actual distributed computation
        self.optimizer._evaluate_energy_distributed = lambda params: np.sum(np.sin(params))
        
        # Optimize VQE
        result = self.optimizer.optimize_vqe(self.ansatz, self.hamiltonian)
        
        # Check that result contains expected keys
        self.assertIn('optimal_parameters', result)
        self.assertIn('optimal_value', result)
        self.assertIn('optimization_history', result)
        self.assertIn('num_iterations', result)
        self.assertIn('worker_results', result)
        self.assertIn('success', result)
        
        # Check that optimization history is populated
        self.assertGreater(len(result['optimization_history']), 0)
    
    def test_evaluate_energy_parameter_split(self):
        """Test parameter-split energy evaluation"""
        # Mock _worker_evaluate_energy_param_split to avoid actual quantum execution
        self.optimizer._worker_evaluate_energy_param_split = lambda task: {
            'backend': task['backend'].name(),
            'param_range': task['param_range'],
            'partial_energy': -0.5
        }
        
        # Store hamiltonian
        self.optimizer.hamiltonian = self.hamiltonian
        
        # Set random parameters
        parameters = np.random.rand(2) * 2 * np.pi
        
        # Evaluate energy
        energy = self.optimizer._evaluate_energy_parameter_split(parameters)
        
        # Check that energy is a float
        self.assertIsInstance(energy, float)
        
        # Check that worker results are populated
        self.assertGreater(len(self.optimizer.worker_results), 0)
    
    def test_evaluate_energy_data_split(self):
        """Test data-split energy evaluation"""
        # Mock _worker_evaluate_energy_data_split to avoid actual quantum execution
        self.optimizer._worker_evaluate_energy_data_split = lambda task: {
            'backend': task['backend'].name(),
            'shots': task['shots'],
            'partial_energy': -0.5
        }
        
        # Store hamiltonian
        self.optimizer.hamiltonian = self.hamiltonian
        
        # Set random parameters
        parameters = np.random.rand(2) * 2 * np.pi
        
        # Evaluate energy
        energy = self.optimizer._evaluate_energy_data_split(parameters)
        
        # Check that energy is a float
        self.assertIsInstance(energy, float)
        
        # Check that worker results are populated
        self.assertGreater(len(self.optimizer.worker_results), 0)
    
    def test_evaluate_energy_hybrid(self):
        """Test hybrid energy evaluation"""
        # Mock parameter-split and data-split methods
        self.optimizer._evaluate_energy_parameter_split = lambda params: -0.5
        self.optimizer._evaluate_energy_data_split = lambda params: -0.6
        
        # Set random parameters
        parameters = np.random.rand(2) * 2 * np.pi
        
        # Reset optimization history to control alternation
        self.optimizer.optimization_history = []
        
        # Evaluate energy (should use parameter-split)
        energy1 = self.optimizer._evaluate_energy_hybrid(parameters)
        
        # Add entry to optimization history
        self.optimizer.optimization_history.append((parameters, energy1))
        
        # Evaluate energy again (should use data-split)
        energy2 = self.optimizer._evaluate_energy_hybrid(parameters)
        
        # Check that different strategies were used
        self.assertNotEqual(energy1, energy2)


class TestIntegration(unittest.TestCase):
    """Integration tests for enhanced quantum VQE"""
    
    def setUp(self):
        """Set up test environment"""
        self.backend = Aer.get_backend('statevector_simulator')
        
        # Create a simple Hamiltonian
        self.hamiltonian = 0.5 * (X ^ X) + 0.3 * (Z ^ Z)
        
        # Create a simple ansatz circuit
        self.ansatz = QuantumCircuit(2)
        self.ansatz.rx('theta', 0)
        self.ansatz.ry('phi', 1)
        self.ansatz.cx(0, 1)
        
        # Create synthetic data
        np.random.seed(42)
        self.X = np.random.rand(10, 2)
        self.y = (self.X[:, 0] > 0.5).astype(float)
    
    def test_natural_gradient_vqe_workflow(self):
        """Test complete natural gradient VQE workflow"""
        # Create VQE
        vqe = NaturalGradientVQE(
            backend=self.backend,
            optimizer_method='NATURAL_GRADIENT',
            max_iterations=3,  # Small number for testing
            use_spinor_reduction=True,
            use_phase_synchronization=True,
            use_prime_indexing=True,
            natural_gradient_reg=0.01,
            qfim_approximation='diag',
            learning_rate=0.1,
            adaptive_learning_rate=True
        )
        
        # Optimize parameters
        result = vqe.optimize_parameters(self.ansatz, self.hamiltonian)
        
        # Analyze convergence
        convergence = vqe.analyze_convergence(result['optimization_history'])
        
        # Check that workflow completed successfully
        self.assertTrue(result['success'])
        self.assertGreater(len(convergence), 0)
    
    def test_spinor_quantum_nn_workflow(self):
        """Test complete spinor quantum neural network workflow"""
        # Create QNN
        qnn = SpinorQuantumNN(
            backend=self.backend,
            optimizer_method='NATURAL_GRADIENT',
            max_iterations=3,  # Small number for testing
            use_spinor_encoding=True,
            use_phase_synchronization=True,
            feature_map_type='SpinorFeatureMap',
            variational_form_type='SpinorCircuit',
            learning_rate=0.1,
            adaptive_learning_rate=True,
            use_quantum_backprop=True
        )
        
        # Generate feature map
        feature_map = qnn.generate_quantum_feature_map(self.X)
        
        # Create variational circuit
        var_circuit = qnn.create_variational_circuit(feature_map)
        
        # Mock _evaluate_loss to avoid actual quantum execution
        qnn._evaluate_loss = lambda params, X, y: np.sum(np.sin(params))
        
        # Train model
        result = qnn.train_quantum_model(var_circuit, self.X, self.y)
        
        # Check that workflow completed successfully
        self.assertTrue(result['success'])
        self.assertGreater(len(result['training_history']), 0)


if __name__ == '__main__':
    unittest.main()