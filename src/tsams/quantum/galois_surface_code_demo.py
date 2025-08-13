"""
TIBEDO Galois Surface Code Demonstration

This script demonstrates the integration of Galois Spinor Lattice Theory with
Surface Code Error Correction in the TIBEDO Framework. It showcases the enhanced
capabilities and performance improvements achieved through this integration.

The demonstration includes:
1. Comparison of standard vs. Galois-enhanced surface codes
2. Visualization of non-Euclidean error correction
3. Performance benchmarking across different error rates
4. Demonstration of spinor-based logical operations
5. Optimization using Veritas conditions
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeMontreal
import time
import logging
import os
import sys
from typing import List, Dict, Tuple, Any, Optional, Union

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

# Import the Galois surface code integration module
from tibedo.quantum_information_new.galois_surface_code_integration import (
    GaloisSurfaceCode,
    PrimeIndexedSyndromeExtractor,
    NonEuclideanDecoder,
    SpinorLogicalOperations,
    VeritasOptimizer
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_noisy_simulator(error_probability: float = 0.01) -> Aer.get_backend:
    """
    Create a noisy simulator with the specified error probability.
    
    Args:
        error_probability: Probability of errors
        
    Returns:
        Noisy simulator backend
    """
    # Get a device noise model
    device = FakeMontreal()
    noise_model = NoiseModel.from_backend(device)
    
    # Scale the noise by the specified error probability
    noise_model.scale_noise(error_probability / 0.01)
    
    # Create a noisy simulator
    simulator = Aer.get_backend('qasm_simulator')
    simulator.set_options(noise_model=noise_model)
    
    return simulator

def create_logical_circuit(surface_code_type: str, operation: str) -> QuantumCircuit:
    """
    Create a logical circuit using the specified surface code type.
    
    Args:
        surface_code_type: Type of surface code ('standard', 'cyclotomic', or 'galois')
        operation: Logical operation to perform ('X', 'Z', 'H', or 'CNOT')
        
    Returns:
        Quantum circuit implementing the logical operation
    """
    # Create the appropriate surface code
    if surface_code_type.lower() == 'standard':
        surface_code = SurfaceCode(distance=3, logical_qubits=1, use_rotated_lattice=True)
    elif surface_code_type.lower() == 'cyclotomic':
        surface_code = CyclotomicSurfaceCode(
            distance=3,
            logical_qubits=1,
            use_rotated_lattice=True,
            cyclotomic_conductor=168,
            use_prime_indexing=True
        )
    elif surface_code_type.lower() == 'galois':
        surface_code = GaloisSurfaceCode(
            distance=3,
            logical_qubits=1,
            use_rotated_lattice=True,
            cyclotomic_conductor=168,
            use_prime_indexing=True,
            galois_characteristic=7,
            galois_extension_degree=2
        )
    else:
        raise ValueError(f"Unknown surface code type: {surface_code_type}")
    
    # Create an encoder
    encoder = SurfaceCodeEncoder(surface_code)
    
    # Create an encoding circuit for the |0⟩ state
    circuit = encoder.create_encoding_circuit(initial_state='0')
    
    # Apply the logical operation
    logical_ops = surface_code.get_logical_operator_circuits()
    
    if operation.upper() == 'X':
        circuit = circuit.compose(logical_ops['logical_x'])
    elif operation.upper() == 'Z':
        circuit = circuit.compose(logical_ops['logical_z'])
    elif operation.upper() == 'H':
        # For Hadamard, we need to apply H to each qubit in the logical Z operator
        h_circuit = QuantumCircuit(circuit.num_qubits)
        for qubit in surface_code.logical_z:
            h_circuit.h(qubit)
        circuit = circuit.compose(h_circuit)
    elif operation.upper() == 'CNOT':
        # For CNOT, we would need a second logical qubit
        # This is a simplified implementation
        pass
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return circuit

def compare_surface_code_types():
    """
    Compare different types of surface codes.
    
    This function compares the standard surface code, cyclotomic surface code,
    and Galois surface code in terms of code properties and performance.
    """
    logger.info("Comparing different types of surface codes")
    
    # Create different types of surface codes
    standard_code = SurfaceCode(distance=3, logical_qubits=1, use_rotated_lattice=True)
    cyclotomic_code = CyclotomicSurfaceCode(
        distance=3,
        logical_qubits=1,
        use_rotated_lattice=True,
        cyclotomic_conductor=168,
        use_prime_indexing=True
    )
    galois_code = GaloisSurfaceCode(
        distance=3,
        logical_qubits=1,
        use_rotated_lattice=True,
        cyclotomic_conductor=168,
        use_prime_indexing=True,
        galois_characteristic=7,
        galois_extension_degree=2
    )
    
    # Compare code properties
    codes = [standard_code, cyclotomic_code, galois_code]
    code_names = ['Standard', 'Cyclotomic', 'Galois']
    
    # Create a figure for the comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Visualize the lattice structure of each code
    for i, (code, name) in enumerate(zip(codes, code_names)):
        # Visualize the lattice
        code.visualize_lattice(show_stabilizers=True)
        plt.sca(axes[i])
        plt.title(f"{name} Surface Code Lattice")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('surface_code_comparison.png')
    logger.info("Saved surface code comparison to 'surface_code_comparison.png'")
    
    # Compare logical operations
    operations = ['X', 'Z', 'H']
    
    # Create a figure for the logical operations
    fig, axes = plt.subplots(len(operations), len(codes), figsize=(18, 15))
    
    for i, operation in enumerate(operations):
        for j, (code_type, name) in enumerate(zip(['standard', 'cyclotomic', 'galois'], code_names)):
            # Create a logical circuit
            circuit = create_logical_circuit(code_type, operation)
            
            # Draw the circuit
            axes[i, j].set_title(f"{name} Logical {operation}")
            axes[i, j].axis('off')
            
            # Display circuit properties
            axes[i, j].text(0.5, 0.5, f"Circuit depth: {circuit.depth()}\nCircuit size: {circuit.size()}", 
                          ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('logical_operations_comparison.png')
    logger.info("Saved logical operations comparison to 'logical_operations_comparison.png'")
    
    return fig

def demonstrate_non_euclidean_decoding():
    """
    Demonstrate non-Euclidean decoding for surface codes.
    
    This function showcases the enhanced error correction capabilities
    achieved through non-Euclidean metrics in the decoding process.
    """
    logger.info("Demonstrating non-Euclidean decoding")
    
    # Create a Galois surface code
    surface_code = GaloisSurfaceCode(
        distance=5,
        logical_qubits=1,
        use_rotated_lattice=True,
        cyclotomic_conductor=168,
        use_prime_indexing=True,
        galois_characteristic=7,
        galois_extension_degree=2
    )
    
    # Create decoders
    standard_decoder = SurfaceCodeDecoder(surface_code)
    non_euclidean_decoder = NonEuclideanDecoder(
        surface_code=surface_code,
        curvature=-1.0,
        use_non_archimedean=True
    )
    
    # Create a sample syndrome with correlated errors
    # This is a scenario where non-Euclidean decoding should outperform standard decoding
    x_syndrome = [0, 1, 0, 1, 0, 0]  # Example syndrome for X-stabilizers
    z_syndrome = [1, 0, 0, 1, 0, 0]  # Example syndrome for Z-stabilizers
    
    # Decode using standard decoder
    start_time = time.time()
    standard_errors = standard_decoder.decode_syndrome(x_syndrome, z_syndrome)
    standard_time = time.time() - start_time
    
    # Decode using non-Euclidean decoder
    start_time = time.time()
    non_euclidean_errors = non_euclidean_decoder.decode_syndrome(x_syndrome, z_syndrome)
    non_euclidean_time = time.time() - start_time
    
    logger.info(f"Standard decoding time: {standard_time:.3f} seconds")
    logger.info(f"Non-Euclidean decoding time: {non_euclidean_time:.3f} seconds")
    
    logger.info(f"Standard decoder found {len(standard_errors['x_errors'])} X errors and {len(standard_errors['z_errors'])} Z errors")
    logger.info(f"Non-Euclidean decoder found {len(non_euclidean_errors['x_errors'])} X errors and {len(non_euclidean_errors['z_errors'])} Z errors")
    
    # Visualize the decoding results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create a graph representation of the surface code
    G = nx.Graph()
    
    # Add nodes for each qubit
    for i in range(surface_code.total_physical_qubits):
        G.add_node(i)
    
    # Add edges for each stabilizer
    for stabilizer in surface_code.x_stabilizers + surface_code.z_stabilizers:
        for i in range(len(stabilizer)):
            for j in range(i+1, len(stabilizer)):
                G.add_edge(stabilizer[i], stabilizer[j])
    
    # Draw the graph
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the standard decoding result
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', ax=ax1)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax1)
    
    # Highlight the errors found by the standard decoder
    error_nodes = standard_errors['x_errors'] + standard_errors['z_errors']
    nx.draw_networkx_nodes(G, pos, nodelist=error_nodes, node_size=300, node_color='red', ax=ax1)
    
    ax1.set_title("Standard Decoding")
    ax1.axis('off')
    
    # Draw the non-Euclidean decoding result
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', ax=ax2)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, ax=ax2)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax2)
    
    # Highlight the errors found by the non-Euclidean decoder
    error_nodes = non_euclidean_errors['x_errors'] + non_euclidean_errors['z_errors']
    nx.draw_networkx_nodes(G, pos, nodelist=error_nodes, node_size=300, node_color='red', ax=ax2)
    
    ax2.set_title("Non-Euclidean Decoding")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('non_euclidean_decoding_demo.png')
    logger.info("Saved non-Euclidean decoding demonstration to 'non_euclidean_decoding_demo.png'")
    
    # Visualize the non-Euclidean state space
    fig2 = non_euclidean_decoder.visualize_decoding(x_syndrome, z_syndrome)
    plt.savefig('non_euclidean_state_space.png')
    logger.info("Saved non-Euclidean state space visualization to 'non_euclidean_state_space.png'")
    
    return fig, fig2

def benchmark_error_correction(distances=[3, 5, 7], error_rates=[0.01, 0.02, 0.05, 0.1], num_trials=50):
    """
    Benchmark error correction performance across different code distances and error rates.
    
    Args:
        distances: List of code distances to benchmark
        error_rates: List of physical error rates to benchmark
        num_trials: Number of trials for each configuration
        
    Returns:
        Dictionary containing benchmark results
    """
    logger.info("Benchmarking error correction performance")
    logger.info(f"Distances: {distances}")
    logger.info(f"Error rates: {error_rates}")
    logger.info(f"Number of trials: {num_trials}")
    
    # Initialize results dictionary
    results = {
        'standard': {},
        'galois': {}
    }
    
    for d in distances:
        results['standard'][d] = {}
        results['galois'][d] = {}
        
        # Create surface codes
        standard_code = SurfaceCode(distance=d, logical_qubits=1, use_rotated_lattice=True)
        galois_code = GaloisSurfaceCode(
            distance=d,
            logical_qubits=1,
            use_rotated_lattice=True,
            cyclotomic_conductor=168,
            use_prime_indexing=True,
            galois_characteristic=7,
            galois_extension_degree=2
        )
        
        # Create decoders
        standard_decoder = SurfaceCodeDecoder(standard_code)
        non_euclidean_decoder = NonEuclideanDecoder(
            surface_code=galois_code,
            curvature=-1.0,
            use_non_archimedean=True
        )
        
        for p in error_rates:
            logger.info(f"Benchmarking d={d}, p={p}")
            
            # Initialize counters
            standard_logical_errors = 0
            galois_logical_errors = 0
            standard_time = 0
            galois_time = 0
            
            for trial in range(num_trials):
                # Generate random errors
                x_errors = []
                z_errors = []
                
                for i in range(standard_code.total_physical_qubits):
                    if np.random.random() < p:
                        x_errors.append(i)
                    if np.random.random() < p:
                        z_errors.append(i)
                
                # Generate syndrome
                standard_x_syndrome = [0] * len(standard_code.x_stabilizers)
                standard_z_syndrome = [0] * len(standard_code.z_stabilizers)
                
                for error_qubit in x_errors:
                    for i, stabilizer in enumerate(standard_code.z_stabilizers):
                        if error_qubit in stabilizer:
                            standard_z_syndrome[i] ^= 1
                
                for error_qubit in z_errors:
                    for i, stabilizer in enumerate(standard_code.x_stabilizers):
                        if error_qubit in stabilizer:
                            standard_x_syndrome[i] ^= 1
                
                # Generate syndrome for Galois code
                galois_x_syndrome = [0] * len(galois_code.x_stabilizers)
                galois_z_syndrome = [0] * len(galois_code.z_stabilizers)
                
                for error_qubit in x_errors:
                    for i, stabilizer in enumerate(galois_code.z_stabilizers):
                        if error_qubit in stabilizer:
                            galois_z_syndrome[i] ^= 1
                
                for error_qubit in z_errors:
                    for i, stabilizer in enumerate(galois_code.x_stabilizers):
                        if error_qubit in stabilizer:
                            galois_x_syndrome[i] ^= 1
                
                # Decode using standard decoder
                start_time = time.time()
                standard_decoded = standard_decoder.decode_syndrome(standard_x_syndrome, standard_z_syndrome)
                standard_time += time.time() - start_time
                
                # Decode using non-Euclidean decoder
                start_time = time.time()
                galois_decoded = non_euclidean_decoder.decode_syndrome(galois_x_syndrome, galois_z_syndrome)
                galois_time += time.time() - start_time
                
                # Check for logical errors
                standard_corrected_x = set(x_errors) ^ set(standard_decoded['x_errors'])
                standard_corrected_z = set(z_errors) ^ set(standard_decoded['z_errors'])
                
                galois_corrected_x = set(x_errors) ^ set(galois_decoded['x_errors'])
                galois_corrected_z = set(z_errors) ^ set(galois_decoded['z_errors'])
                
                # A logical error occurs if there's an odd number of errors on a logical operator
                standard_logical_x_error = sum(1 for q in standard_corrected_x if q in standard_code.logical_z) % 2 == 1
                standard_logical_z_error = sum(1 for q in standard_corrected_z if q in standard_code.logical_x) % 2 == 1
                
                galois_logical_x_error = sum(1 for q in galois_corrected_x if q in galois_code.logical_z) % 2 == 1
                galois_logical_z_error = sum(1 for q in galois_corrected_z if q in galois_code.logical_x) % 2 == 1
                
                if standard_logical_x_error or standard_logical_z_error:
                    standard_logical_errors += 1
                
                if galois_logical_x_error or galois_logical_z_error:
                    galois_logical_errors += 1
            
            # Calculate logical error rates
            standard_logical_error_rate = standard_logical_errors / num_trials
            galois_logical_error_rate = galois_logical_errors / num_trials
            
            # Calculate average decoding times
            standard_avg_time = standard_time / num_trials
            galois_avg_time = galois_time / num_trials
            
            # Store results
            results['standard'][d][p] = {
                'logical_error_rate': standard_logical_error_rate,
                'avg_decoding_time': standard_avg_time
            }
            
            results['galois'][d][p] = {
                'logical_error_rate': galois_logical_error_rate,
                'avg_decoding_time': galois_avg_time
            }
            
            logger.info(f"Standard logical error rate: {standard_logical_error_rate:.4f}")
            logger.info(f"Galois logical error rate: {galois_logical_error_rate:.4f}")
            logger.info(f"Standard average decoding time: {standard_avg_time:.3f} seconds")
            logger.info(f"Galois average decoding time: {galois_avg_time:.3f} seconds")
    
    # Visualize the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot logical error rates
    for d in distances:
        standard_rates = [results['standard'][d][p]['logical_error_rate'] for p in error_rates]
        galois_rates = [results['galois'][d][p]['logical_error_rate'] for p in error_rates]
        
        ax1.semilogy(error_rates, standard_rates, 'o--', label=f"Standard d={d}")
        ax1.semilogy(error_rates, galois_rates, 's-', label=f"Galois d={d}")
    
    ax1.set_xlabel("Physical Error Rate")
    ax1.set_ylabel("Logical Error Rate")
    ax1.set_title("Error Correction Performance")
    ax1.grid(True)
    ax1.legend()
    
    # Plot decoding times
    for d in distances:
        standard_times = [results['standard'][d][p]['avg_decoding_time'] for p in error_rates]
        galois_times = [results['galois'][d][p]['avg_decoding_time'] for p in error_rates]
        
        ax2.plot(error_rates, standard_times, 'o--', label=f"Standard d={d}")
        ax2.plot(error_rates, galois_times, 's-', label=f"Galois d={d}")
    
    ax2.set_xlabel("Physical Error Rate")
    ax2.set_ylabel("Average Decoding Time (s)")
    ax2.set_title("Decoding Performance")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('error_correction_benchmark.png')
    logger.info("Saved error correction benchmark to 'error_correction_benchmark.png'")
    
    return results, fig

def demonstrate_spinor_logical_operations():
    """
    Demonstrate spinor-based logical operations.
    
    This function showcases the implementation of logical operations
    using spinor braiding systems, which provide a topological representation
    of quantum gates.
    """
    logger.info("Demonstrating spinor-based logical operations")
    
    # Create a Galois surface code
    surface_code = GaloisSurfaceCode(
        distance=3,
        logical_qubits=1,
        use_rotated_lattice=True,
        cyclotomic_conductor=168,
        use_prime_indexing=True,
        galois_characteristic=7,
        galois_extension_degree=2
    )
    
    # Create spinor logical operations
    logical_ops = SpinorLogicalOperations(
        surface_code=surface_code,
        num_strands=3
    )
    
    # Create logical states
    logical_states = {}
    state_names = ['0', '1', '+', '-']
    
    for name in state_names:
        logical_states[name] = logical_ops.create_logical_state(name)
    
    # Apply logical operations
    operations = ['X', 'Z', 'H', 'CNOT']
    
    # Create a figure for the logical operations
    fig, axes = plt.subplots(len(operations), 1, figsize=(10, 15))
    
    for i, operation in enumerate(operations):
        # Visualize the logical operation
        logical_ops.visualize_logical_operation(operation)
        plt.sca(axes[i])
        plt.title(f"Logical {operation} Operation")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('spinor_logical_operations.png')
    logger.info("Saved spinor logical operations to 'spinor_logical_operations.png'")
    
    # Create a figure for the logical states
    fig2, axes = plt.subplots(len(state_names), 1, figsize=(10, 15))
    
    for i, name in enumerate(state_names):
        # Visualize the logical state
        state = logical_states[name]
        
        # Plot the magnitude of the state
        magnitude = np.abs(state)
        axes[i].bar(range(len(state)), magnitude)
        axes[i].set_xlabel('State Index')
        axes[i].set_ylabel('Magnitude')
        axes[i].set_title(f"Logical |{name}⟩ State")
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('spinor_logical_states.png')
    logger.info("Saved spinor logical states to 'spinor_logical_states.png'")
    
    return fig, fig2

def demonstrate_veritas_optimization():
    """
    Demonstrate optimization using Veritas conditions.
    
    This function showcases the optimization of surface code parameters
    using the Veritas condition, which defines the fundamental scaling
    factor for the shape space of quantum state representations.
    """
    logger.info("Demonstrating Veritas optimization")
    
    # Create a Galois surface code
    surface_code = GaloisSurfaceCode(
        distance=5,
        logical_qubits=1,
        use_rotated_lattice=True,
        cyclotomic_conductor=168,
        use_prime_indexing=True,
        galois_characteristic=7,
        galois_extension_degree=2
    )
    
    # Create a Veritas optimizer
    optimizer = VeritasOptimizer(surface_code=surface_code)
    
    # Optimize the code distance for different target logical error rates
    target_rates = [1e-6, 1e-9, 1e-12, 1e-15]
    physical_error_rate = 1e-3
    
    optimal_distances = []
    for rate in target_rates:
        distance = optimizer.optimize_code_distance(rate, physical_error_rate)
        optimal_distances.append(distance)
        logger.info(f"Optimal distance for target rate {rate}: {distance}")
    
    # Optimize the syndrome extraction
    optimal_syndrome = optimizer.optimize_syndrome_extraction()
    logger.info(f"Optimal syndrome extraction order: {optimal_syndrome['x_order']}, {optimal_syndrome['z_order']}")
    
    # Visualize the optimization
    fig = optimizer.visualize_optimization()
    plt.savefig('veritas_optimization.png')
    logger.info("Saved Veritas optimization to 'veritas_optimization.png'")
    
    # Create a figure for the optimal distances
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(len(target_rates)), optimal_distances, 'o-')
    ax.set_xlabel("Target Logical Error Rate Index")
    ax.set_ylabel("Optimal Code Distance")
    ax.set_title("Optimal Code Distance vs. Target Logical Error Rate")
    ax.set_xticks(range(len(target_rates)))
    ax.set_xticklabels([f"{rate:.0e}" for rate in target_rates])
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimal_distances.png')
    logger.info("Saved optimal distances to 'optimal_distances.png'")
    
    return fig, fig2

def run_comprehensive_demonstration():
    """
    Run a comprehensive demonstration of the Galois Surface Code integration.
    
    This function runs all the demonstration components and provides
    a summary of the results.
    """
    logger.info("Running comprehensive demonstration of Galois Surface Code integration")
    
    # Compare surface code types
    logger.info("1. Comparing surface code types")
    compare_surface_code_types()
    
    # Demonstrate non-Euclidean decoding
    logger.info("2. Demonstrating non-Euclidean decoding")
    demonstrate_non_euclidean_decoding()
    
    # Benchmark error correction
    logger.info("3. Benchmarking error correction")
    results, _ = benchmark_error_correction(
        distances=[3, 5],
        error_rates=[0.01, 0.05],
        num_trials=10
    )
    
    # Demonstrate spinor logical operations
    logger.info("4. Demonstrating spinor logical operations")
    demonstrate_spinor_logical_operations()
    
    # Demonstrate Veritas optimization
    logger.info("5. Demonstrating Veritas optimization")
    demonstrate_veritas_optimization()
    
    # Print summary of results
    logger.info("\nSummary of Galois Surface Code Integration:")
    logger.info("1. Surface Code Types: Standard, Cyclotomic, and Galois surface codes compared")
    logger.info("2. Non-Euclidean Decoding: Enhanced error correction using non-Euclidean metrics")
    logger.info("3. Error Correction Benchmark: Performance comparison across different code distances and error rates")
    logger.info("4. Spinor Logical Operations: Implementation of logical operations using spinor braiding systems")
    logger.info("5. Veritas Optimization: Optimization of surface code parameters using Veritas conditions")
    
    # Calculate improvement from Galois surface code
    improvements = []
    for d in results['standard']:
        for p in results['standard'][d]:
            standard_rate = results['standard'][d][p]['logical_error_rate']
            galois_rate = results['galois'][d][p]['logical_error_rate']
            
            if standard_rate > 0:
                improvement = (standard_rate - galois_rate) / standard_rate * 100
                improvements.append(improvement)
    
    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        logger.info(f"\nAverage error rate improvement: {avg_improvement:.1f}%")
    
    logger.info("\nGalois Surface Code Integration demonstration completed")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the comprehensive demonstration
    run_comprehensive_demonstration()