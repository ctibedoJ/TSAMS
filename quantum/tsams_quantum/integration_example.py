"""
TIBEDO Advanced Quantum Computing Integration Example

This script demonstrates how the three components of the TIBEDO Advanced Quantum Computing
Integration module work together to solve a practical quantum chemistry problem:

1. Quantum Circuit Optimization: Optimize circuits for efficient execution
2. Quantum Error Mitigation: Mitigate errors in quantum computations
3. Quantum-Classical Hybrid Algorithms: Solve quantum chemistry problems

The example solves a molecular ground state problem using VQE with optimized circuits
and error mitigation, demonstrating the full capabilities of the integration.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
import logging
import time

# Add parent directory to path to import TIBEDO modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TIBEDO quantum computing integration modules
from quantum_information_new.quantum_circuit_optimization import (
    TibedoQuantumCircuitCompressor,
    PhaseSynchronizedGateSet,
    TibedoQuantumResourceEstimator
)
from quantum_information_new.quantum_error_mitigation import (
    SpinorErrorModel,
    PhaseSynchronizedErrorCorrection,
    AdaptiveErrorMitigation
)
from quantum_information_new.quantum_hybrid_algorithms import (
    TibedoEnhancedVQE,
    SpinorQuantumML,
    TibedoQuantumOptimizer
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TibedoQuantumChemistryIntegration:
    """
    Integrated solution for quantum chemistry problems using TIBEDO's
    quantum computing integration components.
    """
    
    def __init__(self, 
                 backend_name: str = 'qasm_simulator',
                 optimization_level: int = 2,
                 error_mitigation_level: int = 2,
                 use_spinor_reduction: bool = True,
                 use_phase_synchronization: bool = True,
                 use_prime_indexing: bool = True):
        """
        Initialize the TIBEDO Quantum Chemistry Integration.
        
        Args:
            backend_name: Name of the quantum backend to use
            optimization_level: Level of circuit optimization (1-3)
            error_mitigation_level: Level of error mitigation (1-3)
            use_spinor_reduction: Whether to use spinor reduction techniques
            use_phase_synchronization: Whether to use phase synchronization
            use_prime_indexing: Whether to use prime-indexed relation techniques
        """
        self.backend_name = backend_name
        self.optimization_level = optimization_level
        self.error_mitigation_level = error_mitigation_level
        self.use_spinor_reduction = use_spinor_reduction
        self.use_phase_synchronization = use_phase_synchronization
        self.use_prime_indexing = use_prime_indexing
        
        # Set up backend
        self.backend = Aer.get_backend(backend_name)
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"Initialized TIBEDO Quantum Chemistry Integration")
        logger.info(f"  Backend: {backend_name}")
        logger.info(f"  Optimization level: {optimization_level}")
        logger.info(f"  Error mitigation level: {error_mitigation_level}")
    
    def _initialize_components(self):
        """
        Initialize all components of the integration.
        """
        # Initialize circuit optimization components
        self.circuit_compressor = TibedoQuantumCircuitCompressor(
            compression_level=self.optimization_level,
            use_spinor_reduction=self.use_spinor_reduction,
            use_phase_synchronization=self.use_phase_synchronization,
            use_prime_indexing=self.use_prime_indexing
        )
        
        self.phase_synchronizer = PhaseSynchronizedGateSet(
            optimization_level=self.optimization_level,
            cyclotomic_conductor=56
        )
        
        self.resource_estimator = TibedoQuantumResourceEstimator(
            error_rate=0.001,
            connectivity='all-to-all',
            include_error_correction=False
        )
        
        # Initialize error mitigation components
        self.error_model = SpinorErrorModel(
            error_characterization_shots=1024,
            use_spinor_reduction=self.use_spinor_reduction,
            use_phase_synchronization=self.use_phase_synchronization,
            use_prime_indexing=self.use_prime_indexing
        )
        
        self.error_correction = PhaseSynchronizedErrorCorrection(
            code_distance=3,
            use_phase_synchronization=self.use_phase_synchronization,
            use_spinor_reduction=self.use_spinor_reduction
        )
        
        self.error_mitigation = AdaptiveErrorMitigation(
            error_model=self.error_model,
            use_zero_noise_extrapolation=True,
            use_probabilistic_error_cancellation=True,
            use_measurement_mitigation=True
        )
        
        # Initialize hybrid algorithm components
        self.vqe = TibedoEnhancedVQE(
            backend=self.backend,
            optimizer_method='COBYLA',
            max_iterations=100,
            use_spinor_reduction=self.use_spinor_reduction,
            use_phase_synchronization=self.use_phase_synchronization,
            use_prime_indexing=self.use_prime_indexing
        )
        
        self.quantum_ml = SpinorQuantumML(
            backend=self.backend,
            optimizer_method='COBYLA',
            max_iterations=100,
            use_spinor_encoding=True,
            use_phase_synchronization=self.use_phase_synchronization,
            feature_map_type='ZZFeatureMap'
        )
        
        self.quantum_optimizer = TibedoQuantumOptimizer(
            backend=self.backend,
            optimizer_method='COBYLA',
            max_iterations=100,
            algorithm_type='QAOA',
            use_phase_synchronization=self.use_phase_synchronization,
            use_prime_indexing=self.use_prime_indexing
        )
    
    def solve_molecular_ground_state(self, molecule_data: dict) -> dict:
        """
        Solve molecular ground state problem using integrated approach.
        
        Args:
            molecule_data: Dictionary with molecule data
            
        Returns:
            Dictionary with solution results
        """
        logger.info(f"Solving molecular ground state problem for {molecule_data.get('name', 'molecule')}")
        
        # Start timing
        start_time = time.time()
        
        # Step 1: Prepare Hamiltonian using VQE
        logger.info("Step 1: Preparing Hamiltonian")
        hamiltonian = self.vqe.prepare_hamiltonian(molecule_data)
        
        # Step 2: Generate ansatz circuit
        logger.info("Step 2: Generating ansatz circuit")
        num_qubits = molecule_data.get('num_qubits', 4)
        depth = molecule_data.get('depth', 2)
        ansatz = self.vqe.generate_ansatz_circuit(num_qubits=num_qubits, depth=depth)
        
        # Step 3: Optimize circuit
        logger.info("Step 3: Optimizing quantum circuit")
        optimized_ansatz = self.circuit_compressor.compress_circuit(ansatz)
        
        # Step 4: Characterize error model
        logger.info("Step 4: Characterizing error model")
        error_params = self.error_model.generate_error_model(self.backend)
        
        # Step 5: Analyze circuit error profile
        logger.info("Step 5: Analyzing circuit error profile")
        error_profile = self.error_mitigation.analyze_circuit_error_profile(optimized_ansatz)
        
        # Step 6: Select error mitigation strategy
        logger.info("Step 6: Selecting error mitigation strategy")
        strategies = self.error_mitigation.select_mitigation_strategy(optimized_ansatz, error_profile)
        
        # Step 7: Optimize parameters with error mitigation
        logger.info("Step 7: Optimizing parameters with error mitigation")
        vqe_result = self.vqe.optimize_parameters(optimized_ansatz, hamiltonian)
        
        # Step 8: Apply error mitigation to final result
        logger.info("Step 8: Applying error mitigation to final result")
        optimized_circuit = optimized_ansatz.bind_parameters(vqe_result['optimal_parameters'])
        mitigation_results = self.error_mitigation.apply_mitigation_strategy(optimized_circuit, strategies, self.backend)
        
        # Step 9: Calculate final energy
        logger.info("Step 9: Calculating final energy")
        energy_result = self.vqe.calculate_energy(optimized_circuit, hamiltonian)
        
        # Step 10: Analyze convergence
        logger.info("Step 10: Analyzing convergence")
        convergence = self.vqe.analyze_convergence(vqe_result['optimization_history'])
        
        # End timing
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Prepare results
        results = {
            'molecule_data': molecule_data,
            'hamiltonian': hamiltonian,
            'original_circuit': ansatz,
            'optimized_circuit': optimized_circuit,
            'optimization_results': {
                'original_depth': ansatz.depth(),
                'optimized_depth': optimized_circuit.depth(),
                'depth_reduction': ansatz.depth() - optimized_circuit.depth(),
                'depth_reduction_percentage': (ansatz.depth() - optimized_circuit.depth()) / ansatz.depth() * 100 if ansatz.depth() > 0 else 0
            },
            'error_mitigation_results': {
                'strategies': strategies,
                'error_profile': {
                    'single_qubit_error_prob': error_profile['error_simulation']['single_qubit_error_prob'],
                    'two_qubit_error_prob': error_profile['error_simulation']['two_qubit_error_prob'],
                    'total_error_prob': error_profile['error_simulation']['total_error_prob'],
                    'fidelity': error_profile['error_simulation']['fidelity']
                }
            },
            'vqe_results': {
                'optimal_parameters': vqe_result['optimal_parameters'],
                'optimal_value': vqe_result['optimal_value'],
                'convergence': convergence
            },
            'energy_results': {
                'energy': energy_result['energy'],
                'energy_components': energy_result['energy_components']
            },
            'execution_time': execution_time
        }
        
        logger.info(f"Solution completed in {execution_time:.2f} seconds")
        logger.info(f"Final energy: {energy_result['energy']}")
        
        return results
    
    def predict_molecular_properties(self, molecule_data: dict, property_data: dict) -> dict:
        """
        Predict molecular properties using quantum machine learning.
        
        Args:
            molecule_data: Dictionary with molecule data
            property_data: Dictionary with property data for training
            
        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Predicting properties for {molecule_data.get('name', 'molecule')}")
        
        # Start timing
        start_time = time.time()
        
        # Step 1: Extract training data
        X_train = property_data.get('X_train')
        y_train = property_data.get('y_train')
        X_test = property_data.get('X_test')
        y_test = property_data.get('y_test')
        
        if X_train is None or y_train is None:
            raise ValueError("Training data not provided")
        
        # Step 2: Encode data as spinors
        logger.info("Step 2: Encoding data as spinors")
        encoded_train_data = self.quantum_ml.encode_data_as_spinors(X_train)
        
        # Step 3: Generate quantum feature map
        logger.info("Step 3: Generating quantum feature map")
        feature_map = self.quantum_ml.generate_quantum_feature_map(X_train)
        
        # Step 4: Optimize feature map circuit
        logger.info("Step 4: Optimizing feature map circuit")
        optimized_feature_map = self.circuit_compressor.compress_circuit(feature_map)
        
        # Step 5: Create variational circuit
        logger.info("Step 5: Creating variational circuit")
        var_circuit = self.quantum_ml.create_variational_circuit(optimized_feature_map)
        
        # Step 6: Analyze circuit error profile
        logger.info("Step 6: Analyzing circuit error profile")
        error_profile = self.error_mitigation.analyze_circuit_error_profile(var_circuit)
        
        # Step 7: Select error mitigation strategy
        logger.info("Step 7: Selecting error mitigation strategy")
        strategies = self.error_mitigation.select_mitigation_strategy(var_circuit, error_profile)
        
        # Step 8: Train quantum model
        logger.info("Step 8: Training quantum model")
        training_result = self.quantum_ml.train_quantum_model(var_circuit, X_train, y_train)
        
        # Step 9: Evaluate model performance
        logger.info("Step 9: Evaluating model performance")
        if X_test is not None and y_test is not None:
            performance = self.quantum_ml.evaluate_model_performance(training_result, X_test, y_test)
        else:
            performance = None
        
        # End timing
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Prepare results
        results = {
            'molecule_data': molecule_data,
            'property_data': {
                'X_train_shape': X_train.shape,
                'y_train_shape': y_train.shape if isinstance(y_train, np.ndarray) else (len(y_train),),
                'X_test_shape': X_test.shape if X_test is not None else None,
                'y_test_shape': y_test.shape if y_test is not None and isinstance(y_test, np.ndarray) else (len(y_test) if y_test is not None else None,)
            },
            'feature_map': {
                'original_depth': feature_map.depth(),
                'optimized_depth': optimized_feature_map.depth(),
                'depth_reduction': feature_map.depth() - optimized_feature_map.depth(),
                'depth_reduction_percentage': (feature_map.depth() - optimized_feature_map.depth()) / feature_map.depth() * 100 if feature_map.depth() > 0 else 0
            },
            'error_mitigation_results': {
                'strategies': strategies,
                'error_profile': {
                    'total_error_prob': error_profile['error_simulation']['total_error_prob'],
                    'fidelity': error_profile['error_simulation']['fidelity']
                }
            },
            'training_results': {
                'trained_parameters': training_result['trained_parameters'],
                'final_loss': training_result['training_history'][-1][1] if training_result['training_history'] else None,
                'success': training_result['success']
            },
            'performance': performance,
            'execution_time': execution_time
        }
        
        logger.info(f"Prediction completed in {execution_time:.2f} seconds")
        if performance:
            logger.info(f"Model performance: MSE = {performance['mse']}, Accuracy = {performance['accuracy']}")
        
        return results
    
    def optimize_molecular_configuration(self, molecule_data: dict) -> dict:
        """
        Optimize molecular configuration using quantum optimization.
        
        Args:
            molecule_data: Dictionary with molecule data
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing configuration for {molecule_data.get('name', 'molecule')}")
        
        # Start timing
        start_time = time.time()
        
        # Step 1: Create optimization problem
        logger.info("Step 1: Creating optimization problem")
        problem = {
            'type': 'chemistry',
            'num_qubits': molecule_data.get('num_qubits', 4),
            'p': molecule_data.get('p', 2)  # QAOA layers
        }
        
        # Step 2: Encode optimization problem
        logger.info("Step 2: Encoding optimization problem")
        cost_hamiltonian = self.quantum_optimizer.encode_optimization_problem(problem)
        
        # Step 3: Generate mixer Hamiltonian
        logger.info("Step 3: Generating mixer Hamiltonian")
        mixer_hamiltonian = self.quantum_optimizer.generate_mixer_hamiltonian(problem)
        
        # Step 4: Create QAOA circuit
        logger.info("Step 4: Creating QAOA circuit")
        qaoa_circuit = self.quantum_optimizer.create_qaoa_circuit(problem)
        
        # Step 5: Optimize QAOA circuit
        logger.info("Step 5: Optimizing QAOA circuit")
        optimized_qaoa_circuit = self.circuit_compressor.compress_circuit(qaoa_circuit)
        
        # Step 6: Analyze circuit error profile
        logger.info("Step 6: Analyzing circuit error profile")
        error_profile = self.error_mitigation.analyze_circuit_error_profile(optimized_qaoa_circuit)
        
        # Step 7: Select error mitigation strategy
        logger.info("Step 7: Selecting error mitigation strategy")
        strategies = self.error_mitigation.select_mitigation_strategy(optimized_qaoa_circuit, error_profile)
        
        # Step 8: Optimize parameters
        logger.info("Step 8: Optimizing parameters")
        optimization_result = self.quantum_optimizer.optimize_parameters(optimized_qaoa_circuit, problem)
        
        # Step 9: Create optimized circuit with optimal parameters
        logger.info("Step 9: Creating optimized circuit with optimal parameters")
        p = problem['p']
        optimized_params = optimization_result['optimal_parameters']
        final_circuit = self.quantum_optimizer.create_qaoa_circuit(problem, optimized_params)
        
        # Step 10: Execute optimized circuit
        logger.info("Step 10: Executing optimized circuit")
        from qiskit import execute
        job = execute(final_circuit, self.backend, shots=1024)
        counts = job.result().get_counts(final_circuit)
        
        # Step 11: Decode quantum solution
        logger.info("Step 11: Decoding quantum solution")
        solution = self.quantum_optimizer.decode_quantum_solution(counts)
        
        # End timing
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Prepare results
        results = {
            'molecule_data': molecule_data,
            'problem': problem,
            'circuit_optimization': {
                'original_depth': qaoa_circuit.depth(),
                'optimized_depth': optimized_qaoa_circuit.depth(),
                'depth_reduction': qaoa_circuit.depth() - optimized_qaoa_circuit.depth(),
                'depth_reduction_percentage': (qaoa_circuit.depth() - optimized_qaoa_circuit.depth()) / qaoa_circuit.depth() * 100 if qaoa_circuit.depth() > 0 else 0
            },
            'error_mitigation_results': {
                'strategies': strategies,
                'error_profile': {
                    'total_error_prob': error_profile['error_simulation']['total_error_prob'],
                    'fidelity': error_profile['error_simulation']['fidelity']
                }
            },
            'optimization_results': {
                'optimal_parameters': optimization_result['optimal_parameters'],
                'optimal_value': optimization_result['optimal_value'],
                'success': optimization_result.get('success', False)
            },
            'solution': solution,
            'execution_time': execution_time
        }
        
        logger.info(f"Optimization completed in {execution_time:.2f} seconds")
        logger.info(f"Optimal value: {optimization_result['optimal_value']}")
        
        return results
    
    def run_complete_workflow(self, molecule_data: dict, property_data: dict = None) -> dict:
        """
        Run complete workflow for molecular analysis.
        
        Args:
            molecule_data: Dictionary with molecule data
            property_data: Dictionary with property data for training (optional)
            
        Returns:
            Dictionary with workflow results
        """
        logger.info(f"Running complete workflow for {molecule_data.get('name', 'molecule')}")
        
        # Start timing
        start_time = time.time()
        
        # Step 1: Solve molecular ground state
        logger.info("Step 1: Solving molecular ground state")
        ground_state_results = self.solve_molecular_ground_state(molecule_data)
        
        # Step 2: Optimize molecular configuration
        logger.info("Step 2: Optimizing molecular configuration")
        configuration_results = self.optimize_molecular_configuration(molecule_data)
        
        # Step 3: Predict molecular properties (if property data provided)
        property_results = None
        if property_data is not None:
            logger.info("Step 3: Predicting molecular properties")
            property_results = self.predict_molecular_properties(molecule_data, property_data)
        
        # End timing
        end_time = time.time()
        total_execution_time = end_time - start_time
        
        # Prepare workflow results
        workflow_results = {
            'molecule_data': molecule_data,
            'ground_state_results': ground_state_results,
            'configuration_results': configuration_results,
            'property_results': property_results,
            'total_execution_time': total_execution_time
        }
        
        logger.info(f"Complete workflow completed in {total_execution_time:.2f} seconds")
        
        return workflow_results
    
    def visualize_results(self, results: dict, save_path: str = None):
        """
        Visualize workflow results.
        
        Args:
            results: Dictionary with workflow results
            save_path: Path to save visualization (if None, display only)
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Ground state convergence
        ax1 = fig.add_subplot(2, 2, 1)
        ground_state_history = results['ground_state_results']['vqe_results']['convergence']
        iterations = range(len(results['ground_state_results']['vqe_results']['optimization_history']))
        energies = [e for _, e in results['ground_state_results']['vqe_results']['optimization_history']]
        ax1.plot(iterations, energies, 'o-', label='Energy')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Energy')
        ax1.set_title('Ground State Convergence')
        ax1.grid(True)
        
        # Plot 2: Circuit optimization
        ax2 = fig.add_subplot(2, 2, 2)
        labels = ['Original', 'Optimized']
        ground_state_depths = [
            results['ground_state_results']['optimization_results']['original_depth'],
            results['ground_state_results']['optimization_results']['optimized_depth']
        ]
        configuration_depths = [
            results['configuration_results']['circuit_optimization']['original_depth'],
            results['configuration_results']['circuit_optimization']['optimized_depth']
        ]
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax2.bar(x - width/2, ground_state_depths, width, label='Ground State')
        ax2.bar(x + width/2, configuration_depths, width, label='Configuration')
        ax2.set_xlabel('Circuit Type')
        ax2.set_ylabel('Circuit Depth')
        ax2.set_title('Circuit Optimization')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Configuration optimization
        ax3 = fig.add_subplot(2, 2, 3)
        if 'optimization_history' in results['configuration_results']['optimization_results']:
            config_iterations = range(len(results['configuration_results']['optimization_results']['optimization_history']))
            config_values = [v for _, v in results['configuration_results']['optimization_results']['optimization_history']]
            ax3.plot(config_iterations, config_values, 'o-', label='Objective Value')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Objective Value')
            ax3.set_title('Configuration Optimization')
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, 'No optimization history available', 
                    horizontalalignment='center', verticalalignment='center')
            ax3.set_title('Configuration Optimization')
        
        # Plot 4: Property prediction (if available)
        ax4 = fig.add_subplot(2, 2, 4)
        if results['property_results'] is not None and 'training_results' in results['property_results']:
            if 'training_history' in results['property_results']['training_results']:
                prop_iterations = range(len(results['property_results']['training_results']['training_history']))
                prop_losses = [loss for _, loss in results['property_results']['training_results']['training_history']]
                ax4.plot(prop_iterations, prop_losses, 'o-', label='Loss')
                ax4.set_xlabel('Iteration')
                ax4.set_ylabel('Loss')
                ax4.set_title('Property Prediction Training')
                ax4.grid(True)
            else:
                ax4.text(0.5, 0.5, 'No training history available', 
                        horizontalalignment='center', verticalalignment='center')
                ax4.set_title('Property Prediction Training')
        else:
            ax4.text(0.5, 0.5, 'No property prediction results available', 
                    horizontalalignment='center', verticalalignment='center')
            ax4.set_title('Property Prediction')
        
        # Add overall title
        molecule_name = results['molecule_data'].get('name', 'Molecule')
        fig.suptitle(f'TIBEDO Quantum Chemistry Results for {molecule_name}', fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()


def main():
    """
    Main function to run the integration example.
    """
    logger.info("Starting TIBEDO Advanced Quantum Computing Integration Example")
    
    # Create integration instance
    integration = TibedoQuantumChemistryIntegration(
        backend_name='qasm_simulator',
        optimization_level=2,
        error_mitigation_level=2,
        use_spinor_reduction=True,
        use_phase_synchronization=True,
        use_prime_indexing=True
    )
    
    # Define molecule data (H2 molecule)
    molecule_data = {
        'name': 'H2',
        'num_qubits': 4,
        'depth': 2,
        'geometry': [('H', (0, 0, 0)), ('H', (0, 0, 0.735))],
        'basis': 'sto-3g',
        'p': 1  # QAOA layers
    }
    
    # Create synthetic property data
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
    
    property_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }
    
    # Run complete workflow
    results = integration.run_complete_workflow(molecule_data, property_data)
    
    # Visualize results
    integration.visualize_results(results, save_path='tibedo_quantum_chemistry_results.png')
    
    # Print summary
    logger.info("\nWorkflow Summary:")
    logger.info(f"Molecule: {molecule_data['name']}")
    logger.info(f"Ground state energy: {results['ground_state_results']['energy_results']['energy']}")
    logger.info(f"Optimal configuration value: {results['configuration_results']['optimization_results']['optimal_value']}")
    if results['property_results'] and results['property_results']['performance']:
        logger.info(f"Property prediction MSE: {results['property_results']['performance']['mse']}")
    logger.info(f"Total execution time: {results['total_execution_time']:.2f} seconds")
    
    logger.info("\nTIBEDO Advanced Quantum Computing Integration Example completed successfully!")


if __name__ == "__main__":
    main()