"""
Test script for TIBEDO Quantum-Classical Hybrid Algorithms module.

This script demonstrates the functionality of the quantum hybrid algorithms
module, including VQE, quantum machine learning, and quantum optimization.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import logging

# Add parent directory to path to import TIBEDO modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TIBEDO quantum hybrid algorithms module
from quantum_information_new.quantum_hybrid_algorithms import (
    TibedoEnhancedVQE,
    SpinorQuantumML,
    TibedoQuantumOptimizer
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tibedo_enhanced_vqe():
    """
    Test the TibedoEnhancedVQE class.
    """
    logger.info("Testing TibedoEnhancedVQE...")
    
    # Create VQE instance
    vqe = TibedoEnhancedVQE(
        optimizer_method='COBYLA',
        max_iterations=20,  # Small number for testing
        use_spinor_reduction=True,
        use_phase_synchronization=True,
        use_prime_indexing=True
    )
    
    # Create a simple molecule data
    molecule_data = {
        'num_qubits': 4,
        'geometry': [('H', (0, 0, 0)), ('H', (0, 0, 0.735))]
    }
    
    # Prepare Hamiltonian
    hamiltonian = vqe.prepare_hamiltonian(molecule_data)
    
    logger.info(f"Prepared Hamiltonian with {hamiltonian.num_qubits} qubits")
    
    # Generate ansatz circuit
    ansatz = vqe.generate_ansatz_circuit(num_qubits=4, depth=2)
    
    logger.info(f"Generated ansatz circuit with depth {ansatz.depth()}")
    
    # Optimize parameters (with reduced iterations for testing)
    result = vqe.optimize_parameters(ansatz, hamiltonian)
    
    logger.info(f"Optimization results:")
    logger.info(f"  Optimal value: {result['optimal_value']}")
    logger.info(f"  Success: {result['success']}")
    
    # Calculate energy with optimized parameters
    optimized_circuit = ansatz.bind_parameters(result['optimal_parameters'])
    energy_result = vqe.calculate_energy(optimized_circuit, hamiltonian)
    
    logger.info(f"Energy calculation results:")
    logger.info(f"  Energy: {energy_result['energy']}")
    logger.info(f"  Number of energy components: {len(energy_result['energy_components'])}")
    
    # Analyze convergence
    convergence = vqe.analyze_convergence(result['optimization_history'])
    
    logger.info(f"Convergence analysis:")
    logger.info(f"  Initial energy: {convergence['initial_energy']}")
    logger.info(f"  Final energy: {convergence['final_energy']}")
    logger.info(f"  Energy improvement: {convergence['energy_improvement']}")
    logger.info(f"  Converged: {convergence['converged']}")
    
    # Plot optimization history
    iterations = range(len(result['optimization_history']))
    energies = [e for _, e in result['optimization_history']]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, energies, 'o-', label='Energy')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('VQE Optimization History')
    plt.grid(True)
    plt.legend()
    plt.savefig('vqe_optimization_history.png')
    logger.info("Saved optimization history plot to 'vqe_optimization_history.png'")

def test_spinor_quantum_ml():
    """
    Test the SpinorQuantumML class.
    """
    logger.info("\nTesting SpinorQuantumML...")
    
    # Create synthetic dataset
    np.random.seed(42)
    num_samples = 20
    num_features = 2
    
    # Generate random data
    X = np.random.rand(num_samples, num_features)
    y = (X[:, 0] > 0.5).astype(float)  # Simple classification task
    
    # Split into training and test sets
    train_size = int(0.7 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    logger.info(f"Created dataset with {num_samples} samples and {num_features} features")
    logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Create quantum ML instance
    qml = SpinorQuantumML(
        optimizer_method='COBYLA',
        max_iterations=20,  # Small number for testing
        use_spinor_encoding=True,
        use_phase_synchronization=True,
        feature_map_type='ZZFeatureMap'
    )
    
    # Encode data as spinors
    encoded_data = qml.encode_data_as_spinors(X_train)
    
    logger.info(f"Encoded training data as spinors with shape {encoded_data.shape}")
    
    # Generate quantum feature map
    feature_map = qml.generate_quantum_feature_map(X_train)
    
    logger.info(f"Generated quantum feature map with {feature_map.num_qubits} qubits")
    
    # Create variational circuit
    var_circuit = qml.create_variational_circuit(feature_map)
    
    logger.info(f"Created variational circuit with depth {var_circuit.depth()}")
    
    # Train quantum model (with reduced iterations for testing)
    result = qml.train_quantum_model(var_circuit, X_train, y_train)
    
    logger.info(f"Training results:")
    logger.info(f"  Final loss: {result['training_history'][-1][1]}")
    logger.info(f"  Success: {result['success']}")
    
    # Evaluate model performance
    performance = qml.evaluate_model_performance(result, X_test, y_test)
    
    logger.info(f"Model performance:")
    logger.info(f"  MSE: {performance['mse']}")
    logger.info(f"  Accuracy: {performance['accuracy']}")
    
    # Plot training history
    iterations = range(len(result['training_history']))
    losses = [loss for _, loss in result['training_history']]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses, 'o-', label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Quantum ML Training History')
    plt.grid(True)
    plt.legend()
    plt.savefig('qml_training_history.png')
    logger.info("Saved training history plot to 'qml_training_history.png'")

def test_tibedo_quantum_optimizer():
    """
    Test the TibedoQuantumOptimizer class.
    """
    logger.info("\nTesting TibedoQuantumOptimizer...")
    
    # Create quantum optimizer instance
    optimizer = TibedoQuantumOptimizer(
        optimizer_method='COBYLA',
        max_iterations=20,  # Small number for testing
        algorithm_type='QAOA',
        use_phase_synchronization=True,
        use_prime_indexing=True
    )
    
    # Test with different problem types
    test_problems = [
        {
            'type': 'ising',
            'num_qubits': 4,
            'h': {0: 1.0, 1: -0.5, 2: 0.8, 3: -0.7},
            'J': {(0, 1): -1.0, (1, 2): 0.5, (2, 3): -0.8, (0, 3): 0.7},
            'p': 1  # QAOA layers
        },
        {
            'type': 'maxcut',
            'num_qubits': 4,
            'edges': [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
            'weights': [1.0, 1.0, 1.0, 1.0, 1.0],
            'p': 1  # QAOA layers
        },
        {
            'type': 'chemistry',
            'num_qubits': 4,
            'p': 1  # QAOA layers
        }
    ]
    
    for problem in test_problems:
        logger.info(f"\nTesting {problem['type']} problem...")
        
        # Encode optimization problem
        cost_hamiltonian = optimizer.encode_optimization_problem(problem)
        
        logger.info(f"Encoded {problem['type']} problem with {problem['num_qubits']} qubits")
        
        # Generate mixer Hamiltonian
        mixer_hamiltonian = optimizer.generate_mixer_hamiltonian(problem)
        
        logger.info(f"Generated mixer Hamiltonian with TIBEDO enhancements")
        
        # Create QAOA circuit
        qaoa_circuit = optimizer.create_qaoa_circuit(problem)
        
        logger.info(f"Created QAOA circuit with depth {qaoa_circuit.depth()}")
        
        # Optimize parameters (with reduced iterations for testing)
        result = optimizer.optimize_parameters(qaoa_circuit, problem)
        
        logger.info(f"Optimization results:")
        logger.info(f"  Optimal value: {result['optimal_value']}")
        logger.info(f"  Success: {result['success']}")
        
        # Decode quantum solution
        from qiskit import Aer, execute
        backend = Aer.get_backend('qasm_simulator')
        
        # Create optimized circuit
        p = problem['p']
        optimized_params = result['optimal_parameters']
        optimized_circuit = optimizer.create_qaoa_circuit(problem, optimized_params)
        
        # Execute circuit
        job = execute(optimized_circuit, backend, shots=1024)
        counts = job.result().get_counts(optimized_circuit)
        
        # Decode solution
        solution = optimizer.decode_quantum_solution(counts)
        
        logger.info(f"Solution:")
        logger.info(f"  Bitstring: {solution['bitstring']}")
        
        if problem['type'] == 'ising':
            logger.info(f"  Energy: {solution['energy']}")
            logger.info(f"  Spins: {solution['spins']}")
        elif problem['type'] == 'maxcut':
            logger.info(f"  Cut value: {solution['cut_value']}")
            logger.info(f"  Cut edges: {solution['cut_edges']}")
            logger.info(f"  Partition 0: {solution['partition_0']}")
            logger.info(f"  Partition 1: {solution['partition_1']}")
        elif problem['type'] == 'chemistry':
            logger.info(f"  Energy: {solution['energy']}")
            logger.info(f"  Occupied orbitals: {solution['occupied_orbitals']}")
        
        # Plot optimization history
        iterations = range(len(result['optimization_history']))
        values = [v for _, v in result['optimization_history']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, values, 'o-', label='Objective Value')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title(f'QAOA Optimization History ({problem["type"]})')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'qaoa_{problem["type"]}_optimization_history.png')
        logger.info(f"Saved optimization history plot to 'qaoa_{problem['type']}_optimization_history.png'")

def test_combined_approach():
    """
    Test a combined approach using multiple quantum hybrid algorithms.
    """
    logger.info("\nTesting combined quantum hybrid approach...")
    
    # Create a chemistry problem
    problem = {
        'type': 'chemistry',
        'num_qubits': 4,
        'p': 1  # QAOA layers
    }
    
    # Step 1: Use VQE to find ground state energy
    logger.info("Step 1: Using VQE to find ground state energy")
    
    vqe = TibedoEnhancedVQE(
        optimizer_method='COBYLA',
        max_iterations=20,  # Small number for testing
        use_spinor_reduction=True,
        use_phase_synchronization=True,
        use_prime_indexing=True
    )
    
    # Prepare Hamiltonian
    hamiltonian = vqe.prepare_hamiltonian(problem)
    
    # Generate ansatz circuit
    ansatz = vqe.generate_ansatz_circuit(num_qubits=4, depth=2)
    
    # Optimize parameters
    vqe_result = vqe.optimize_parameters(ansatz, hamiltonian)
    
    logger.info(f"VQE results:")
    logger.info(f"  Ground state energy: {vqe_result['optimal_value']}")
    
    # Step 2: Use quantum optimizer to find optimal molecular configuration
    logger.info("Step 2: Using quantum optimizer to find optimal molecular configuration")
    
    optimizer = TibedoQuantumOptimizer(
        optimizer_method='COBYLA',
        max_iterations=20,  # Small number for testing
        algorithm_type='QAOA',
        use_phase_synchronization=True,
        use_prime_indexing=True
    )
    
    # Encode optimization problem
    cost_hamiltonian = optimizer.encode_optimization_problem(problem)
    
    # Generate mixer Hamiltonian
    mixer_hamiltonian = optimizer.generate_mixer_hamiltonian(problem)
    
    # Create QAOA circuit
    qaoa_circuit = optimizer.create_qaoa_circuit(problem)
    
    # Optimize parameters
    qaoa_result = optimizer.optimize_parameters(qaoa_circuit, problem)
    
    logger.info(f"QAOA results:")
    logger.info(f"  Optimal value: {qaoa_result['optimal_value']}")
    
    # Step 3: Use quantum ML to predict properties
    logger.info("Step 3: Using quantum ML to predict molecular properties")
    
    # Create synthetic dataset based on molecular configurations
    np.random.seed(42)
    num_samples = 20
    num_features = 4  # One feature per qubit
    
    # Generate random data
    X = np.random.rand(num_samples, num_features)
    y = np.sum(X, axis=1) / num_features  # Simple property prediction
    
    # Split into training and test sets
    train_size = int(0.7 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create quantum ML instance
    qml = SpinorQuantumML(
        optimizer_method='COBYLA',
        max_iterations=20,  # Small number for testing
        use_spinor_encoding=True,
        use_phase_synchronization=True,
        feature_map_type='ZZFeatureMap'
    )
    
    # Generate quantum feature map
    feature_map = qml.generate_quantum_feature_map(X_train)
    
    # Create variational circuit
    var_circuit = qml.create_variational_circuit(feature_map)
    
    # Train quantum model
    qml_result = qml.train_quantum_model(var_circuit, X_train, y_train)
    
    # Evaluate model performance
    performance = qml.evaluate_model_performance(qml_result, X_test, y_test)
    
    logger.info(f"Quantum ML results:")
    logger.info(f"  MSE: {performance['mse']}")
    
    # Combine results
    logger.info("\nCombined approach results:")
    logger.info(f"  Ground state energy (VQE): {vqe_result['optimal_value']}")
    logger.info(f"  Optimal configuration (QAOA): {qaoa_result['optimal_value']}")
    logger.info(f"  Property prediction MSE (QML): {performance['mse']}")
    
    # Plot combined results
    plt.figure(figsize=(15, 5))
    
    # VQE convergence
    plt.subplot(1, 3, 1)
    vqe_iterations = range(len(vqe_result['optimization_history']))
    vqe_energies = [e for _, e in vqe_result['optimization_history']]
    plt.plot(vqe_iterations, vqe_energies, 'o-', label='Energy')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('VQE Convergence')
    plt.grid(True)
    
    # QAOA convergence
    plt.subplot(1, 3, 2)
    qaoa_iterations = range(len(qaoa_result['optimization_history']))
    qaoa_values = [v for _, v in qaoa_result['optimization_history']]
    plt.plot(qaoa_iterations, qaoa_values, 'o-', label='Objective Value')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('QAOA Convergence')
    plt.grid(True)
    
    # QML convergence
    plt.subplot(1, 3, 3)
    qml_iterations = range(len(qml_result['training_history']))
    qml_losses = [loss for _, loss in qml_result['training_history']]
    plt.plot(qml_iterations, qml_losses, 'o-', label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('QML Convergence')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('combined_approach_results.png')
    logger.info("Saved combined approach results plot to 'combined_approach_results.png'")

def main():
    """
    Main function to run all tests.
    """
    logger.info("Starting TIBEDO Quantum Hybrid Algorithms tests...")
    
    # Test TibedoEnhancedVQE
    test_tibedo_enhanced_vqe()
    
    # Test SpinorQuantumML
    test_spinor_quantum_ml()
    
    # Test TibedoQuantumOptimizer
    test_tibedo_quantum_optimizer()
    
    # Test combined approach
    test_combined_approach()
    
    logger.info("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()