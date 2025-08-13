"""
TIBEDO Surface Code Error Correction Demonstration

This script demonstrates the use of the TIBEDO surface code error correction
implementation, including code initialization, error simulation, syndrome
extraction, error correction, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import networkx as nx
import logging
import os
import sys
import time

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the surface code error correction modules
from tibedo.quantum_information_new.surface_code_error_correction import (
    SurfaceCode,
    SurfaceCodeEncoder,
    SyndromeExtractionCircuitGenerator,
    SurfaceCodeDecoder,
    CyclotomicSurfaceCode
)

from tibedo.quantum_information_new.surface_code_visualization import (
    SurfaceCodeVisualizer,
    SyndromeVisualizer,
    DecodingGraphVisualizer
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_surface_code(distance=3, use_rotated_lattice=True, use_cyclotomic=False):
    """
    Create a surface code with the specified parameters.
    
    Args:
        distance: Code distance (must be odd)
        use_rotated_lattice: Whether to use the rotated surface code lattice
        use_cyclotomic: Whether to use the cyclotomic surface code
        
    Returns:
        Surface code instance
    """
    logger.info(f"Creating surface code with distance {distance}")
    logger.info(f"Using rotated lattice: {use_rotated_lattice}")
    logger.info(f"Using cyclotomic enhancements: {use_cyclotomic}")
    
    if use_cyclotomic:
        return CyclotomicSurfaceCode(
            distance=distance,
            logical_qubits=1,
            use_rotated_lattice=use_rotated_lattice,
            cyclotomic_conductor=168,
            use_prime_indexing=True
        )
    else:
        return SurfaceCode(
            distance=distance,
            logical_qubits=1,
            use_rotated_lattice=use_rotated_lattice
        )

def simulate_random_errors(surface_code, error_rate=0.1):
    """
    Simulate random X and Z errors on the surface code.
    
    Args:
        surface_code: Surface code instance
        error_rate: Probability of each type of error on each qubit
        
    Returns:
        Lists of qubits with X and Z errors
    """
    logger.info(f"Simulating random errors with error rate {error_rate}")
    
    # Generate random errors
    x_errors = []
    z_errors = []
    
    for i in range(surface_code.total_physical_qubits):
        # X errors
        if np.random.random() < error_rate:
            x_errors.append(i)
        
        # Z errors
        if np.random.random() < error_rate:
            z_errors.append(i)
    
    logger.info(f"Generated {len(x_errors)} X errors and {len(z_errors)} Z errors")
    
    return x_errors, z_errors

def generate_syndrome(surface_code, x_errors, z_errors):
    """
    Generate syndrome measurements from error patterns.
    
    Args:
        surface_code: Surface code instance
        x_errors: List of qubits with X errors
        z_errors: List of qubits with Z errors
        
    Returns:
        Syndrome measurements for X and Z stabilizers
    """
    logger.info("Generating syndrome measurements from error patterns")
    
    # Initialize syndromes to all zeros
    x_syndrome = [0] * len(surface_code.x_stabilizers)
    z_syndrome = [0] * len(surface_code.z_stabilizers)
    
    # X errors cause Z-stabilizers to flip
    for error_qubit in x_errors:
        for i, stabilizer in enumerate(surface_code.z_stabilizers):
            if error_qubit in stabilizer:
                # Flip the syndrome bit
                z_syndrome[i] = 1 - z_syndrome[i]
    
    # Z errors cause X-stabilizers to flip
    for error_qubit in z_errors:
        for i, stabilizer in enumerate(surface_code.x_stabilizers):
            if error_qubit in stabilizer:
                # Flip the syndrome bit
                x_syndrome[i] = 1 - x_syndrome[i]
    
    logger.info(f"Generated syndrome: {sum(x_syndrome)} X-stabilizer flips, {sum(z_syndrome)} Z-stabilizer flips")
    
    return x_syndrome, z_syndrome

def correct_errors(surface_code, x_syndrome, z_syndrome):
    """
    Correct errors using the surface code decoder.
    
    Args:
        surface_code: Surface code instance
        x_syndrome: Syndrome measurements for X-stabilizers
        z_syndrome: Syndrome measurements for Z-stabilizers
        
    Returns:
        Dictionary containing lists of qubits with X and Z errors
    """
    logger.info("Correcting errors using surface code decoder")
    
    # Create a decoder
    decoder = SurfaceCodeDecoder(surface_code)
    
    # Decode the syndrome
    start_time = time.time()
    decoded_errors = decoder.decode_syndrome(x_syndrome, z_syndrome)
    end_time = time.time()
    
    logger.info(f"Decoded {len(decoded_errors['x_errors'])} X errors and {len(decoded_errors['z_errors'])} Z errors")
    logger.info(f"Decoding time: {end_time - start_time:.3f} seconds")
    
    return decoded_errors

def evaluate_correction(original_errors, decoded_errors, surface_code):
    """
    Evaluate the success of error correction.
    
    Args:
        original_errors: Dictionary with lists of qubits with original X and Z errors
        decoded_errors: Dictionary with lists of qubits with decoded X and Z errors
        surface_code: The surface code instance
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating error correction success")
    
    # Extract error lists
    original_x_errors = set(original_errors['x_errors'])
    original_z_errors = set(original_errors['z_errors'])
    decoded_x_errors = set(decoded_errors['x_errors'])
    decoded_z_errors = set(decoded_errors['z_errors'])
    
    # Calculate metrics
    x_correct = len(original_x_errors & decoded_x_errors)
    x_missed = len(original_x_errors - decoded_x_errors)
    x_extra = len(decoded_x_errors - original_x_errors)
    
    z_correct = len(original_z_errors & decoded_z_errors)
    z_missed = len(original_z_errors - decoded_z_errors)
    z_extra = len(decoded_z_errors - original_z_errors)
    
    # Calculate success rates
    x_success_rate = x_correct / len(original_x_errors) if original_x_errors else 1.0
    z_success_rate = z_correct / len(original_z_errors) if original_z_errors else 1.0
    overall_success_rate = (x_success_rate + z_success_rate) / 2
    
    # Check for logical errors
    # A logical error occurs if there's an odd number of errors on a logical operator
    logical_x_error = sum(1 for q in original_x_errors ^ decoded_x_errors if q in surface_code.logical_z) % 2 == 1
    logical_z_error = sum(1 for q in original_z_errors ^ decoded_z_errors if q in surface_code.logical_x) % 2 == 1
    
    metrics = {
        'x_correct': x_correct,
        'x_missed': x_missed,
        'x_extra': x_extra,
        'z_correct': z_correct,
        'z_missed': z_missed,
        'z_extra': z_extra,
        'x_success_rate': x_success_rate,
        'z_success_rate': z_success_rate,
        'overall_success_rate': overall_success_rate,
        'logical_x_error': logical_x_error,
        'logical_z_error': logical_z_error
    }
    
    logger.info(f"X error correction: {x_correct} correct, {x_missed} missed, {x_extra} extra")
    logger.info(f"Z error correction: {z_correct} correct, {z_missed} missed, {z_extra} extra")
    logger.info(f"Success rates: X={x_success_rate:.2f}, Z={z_success_rate:.2f}, Overall={overall_success_rate:.2f}")
    logger.info(f"Logical errors: X={logical_x_error}, Z={logical_z_error}")
    
    return metrics

def benchmark_error_correction(distances, error_rates, num_trials=100, use_cyclotomic=False):
    """
    Benchmark the performance of surface code error correction.
    
    Args:
        distances: List of code distances to benchmark
        error_rates: List of physical error rates to benchmark
        num_trials: Number of trials for each configuration
        use_cyclotomic: Whether to use the cyclotomic surface code
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking surface code error correction")
    logger.info(f"Distances: {distances}")
    logger.info(f"Error rates: {error_rates}")
    logger.info(f"Number of trials: {num_trials}")
    logger.info(f"Using cyclotomic enhancements: {use_cyclotomic}")
    
    results = {}
    
    for d in distances:
        results[d] = {}
        
        # Create a surface code with this distance
        surface_code = create_surface_code(distance=d, use_cyclotomic=use_cyclotomic)
        
        for p in error_rates:
            logger.info(f"Benchmarking d={d}, p={p}")
            
            # Initialize counters
            logical_x_errors = 0
            logical_z_errors = 0
            total_decoding_time = 0
            
            for trial in range(num_trials):
                # Simulate random errors
                x_errors, z_errors = simulate_random_errors(surface_code, error_rate=p)
                
                # Generate syndrome
                x_syndrome, z_syndrome = generate_syndrome(surface_code, x_errors, z_errors)
                
                # Correct errors
                start_time = time.time()
                decoded_errors = correct_errors(surface_code, x_syndrome, z_syndrome)
                end_time = time.time()
                total_decoding_time += end_time - start_time
                
                # Evaluate correction
                metrics = evaluate_correction(
                    {'x_errors': x_errors, 'z_errors': z_errors},
                    decoded_errors,
                    surface_code
                )
                
                # Count logical errors
                if metrics['logical_x_error']:
                    logical_x_errors += 1
                if metrics['logical_z_error']:
                    logical_z_errors += 1
            
            # Calculate logical error rates
            logical_x_error_rate = logical_x_errors / num_trials
            logical_z_error_rate = logical_z_errors / num_trials
            logical_error_rate = (logical_x_error_rate + logical_z_error_rate) / 2
            avg_decoding_time = total_decoding_time / num_trials
            
            results[d][p] = {
                'logical_x_error_rate': logical_x_error_rate,
                'logical_z_error_rate': logical_z_error_rate,
                'logical_error_rate': logical_error_rate,
                'avg_decoding_time': avg_decoding_time
            }
            
            logger.info(f"Results for d={d}, p={p}:")
            logger.info(f"  Logical X error rate: {logical_x_error_rate:.4f}")
            logger.info(f"  Logical Z error rate: {logical_z_error_rate:.4f}")
            logger.info(f"  Overall logical error rate: {logical_error_rate:.4f}")
            logger.info(f"  Average decoding time: {avg_decoding_time:.3f} seconds")
    
    return results

def visualize_benchmark_results(results, distances, error_rates):
    """
    Visualize benchmark results.
    
    Args:
        results: Dictionary with benchmark results
        distances: List of code distances
        error_rates: List of physical error rates
        
    Returns:
        Matplotlib figure showing the benchmark results
    """
    logger.info("Visualizing benchmark results")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot logical error rates vs. physical error rates
    for d in distances:
        logical_error_rates = [results[d][p]['logical_error_rate'] for p in error_rates]
        ax1.semilogy(error_rates, logical_error_rates, 'o-', label=f"d={d}")
    
    # Add threshold line if possible
    if len(error_rates) > 1 and len(distances) > 1:
        # Find the approximate threshold by looking for crossings
        threshold = None
        for i in range(len(error_rates) - 1):
            for d1_idx in range(len(distances) - 1):
                for d2_idx in range(d1_idx + 1, len(distances)):
                    d1 = distances[d1_idx]
                    d2 = distances[d2_idx]
                    if ((results[d1][error_rates[i]]['logical_error_rate'] > results[d2][error_rates[i]]['logical_error_rate'] and
                         results[d1][error_rates[i+1]]['logical_error_rate'] < results[d2][error_rates[i+1]]['logical_error_rate']) or
                        (results[d1][error_rates[i]]['logical_error_rate'] < results[d2][error_rates[i]]['logical_error_rate'] and
                         results[d1][error_rates[i+1]]['logical_error_rate'] > results[d2][error_rates[i+1]]['logical_error_rate'])):
                        # Found a crossing
                        t = (error_rates[i] + error_rates[i+1]) / 2
                        if threshold is None or abs(t - 0.01) < abs(threshold - 0.01):
                            threshold = t
        
        if threshold is not None:
            ax1.axvline(x=threshold, color='r', linestyle='--', 
                     label=f"Threshold ≈ {threshold:.3f}")
    
    # Set plot properties
    ax1.set_xlabel("Physical Error Rate")
    ax1.set_ylabel("Logical Error Rate")
    ax1.set_title("Surface Code Error Correction Performance")
    ax1.grid(True)
    ax1.legend()
    
    # Plot decoding time vs. code distance
    avg_times = [np.mean([results[d][p]['avg_decoding_time'] for p in error_rates]) for d in distances]
    ax2.plot(distances, avg_times, 'o-')
    
    # Set plot properties
    ax2.set_xlabel("Code Distance")
    ax2.set_ylabel("Average Decoding Time (s)")
    ax2.set_title("Surface Code Decoding Performance")
    ax2.grid(True)
    
    plt.tight_layout()
    
    return fig

def demonstrate_error_correction():
    """
    Demonstrate the surface code error correction process.
    """
    logger.info("Demonstrating surface code error correction")
    
    # Create a surface code
    distance = 5
    surface_code = create_surface_code(distance=distance, use_rotated_lattice=True)
    
    # Create visualizers
    surface_code_visualizer = SurfaceCodeVisualizer(surface_code)
    
    # Visualize the lattice
    lattice_fig = surface_code_visualizer.visualize_lattice()
    lattice_fig.savefig('surface_code_lattice.png')
    logger.info("Saved surface code lattice visualization to 'surface_code_lattice.png'")
    
    # Simulate specific errors for demonstration
    # Create a pattern of errors that forms a logical error
    x_errors = []
    z_errors = []
    
    # Add X errors along a vertical path (forming a logical X error)
    for i in range(distance):
        x_errors.append(surface_code.qubit_grid[i, 0])
    
    # Add Z errors along a horizontal path (forming a logical Z error)
    for j in range(distance):
        z_errors.append(surface_code.qubit_grid[0, j])
    
    # Add some random errors
    for _ in range(3):
        i = np.random.randint(0, distance)
        j = np.random.randint(0, distance)
        if np.random.random() < 0.5:
            x_errors.append(surface_code.qubit_grid[i, j])
        else:
            z_errors.append(surface_code.qubit_grid[i, j])
    
    # Visualize the errors
    errors_fig = surface_code_visualizer.visualize_errors(x_errors, z_errors)
    errors_fig.savefig('surface_code_errors.png')
    logger.info("Saved error pattern visualization to 'surface_code_errors.png'")
    
    # Generate syndrome
    x_syndrome, z_syndrome = generate_syndrome(surface_code, x_errors, z_errors)
    
    # Visualize the syndrome
    syndrome_fig = surface_code_visualizer.visualize_syndrome(x_syndrome, z_syndrome)
    syndrome_fig.savefig('surface_code_syndrome.png')
    logger.info("Saved syndrome visualization to 'surface_code_syndrome.png'")
    
    # Create a decoder
    decoder = SurfaceCodeDecoder(surface_code)
    
    # Create a decoding graph visualizer
    graph_visualizer = DecodingGraphVisualizer(decoder)
    
    # Visualize the decoding graph
    graph_fig = graph_visualizer.visualize_decoding_graph(error_type='x')
    graph_fig.savefig('surface_code_decoding_graph.png')
    logger.info("Saved decoding graph visualization to 'surface_code_decoding_graph.png'")
    
    # Visualize the matching
    matching_fig = graph_visualizer.visualize_matching(x_syndrome, error_type='x')
    matching_fig.savefig('surface_code_matching.png')
    logger.info("Saved matching visualization to 'surface_code_matching.png'")
    
    # Decode the syndrome
    decoded_errors = decoder.decode_syndrome(x_syndrome, z_syndrome)
    
    # Visualize the error correction
    correction_fig = surface_code_visualizer.visualize_error_correction(
        {'x_errors': x_errors, 'z_errors': z_errors},
        decoded_errors)
    correction_fig.savefig('surface_code_correction.png')
    logger.info("Saved error correction visualization to 'surface_code_correction.png'")
    
    # Create a syndrome visualizer
    syndrome_visualizer = SyndromeVisualizer(surface_code)
    
    # Visualize the complete error correction process
    process_figs = syndrome_visualizer.visualize_error_syndrome_correction(
        x_errors, z_errors, x_syndrome, z_syndrome, decoded_errors)
    
    for i, fig in enumerate(process_figs):
        fig.savefig(f'surface_code_process_{i+1}.png')
    logger.info("Saved error correction process visualizations")
    
    # Evaluate the correction
    metrics = evaluate_correction(
        {'x_errors': x_errors, 'z_errors': z_errors},
        decoded_errors,
        surface_code
    )
    
    # Print the evaluation metrics
    logger.info("Error correction evaluation:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    return {
        'surface_code': surface_code,
        'x_errors': x_errors,
        'z_errors': z_errors,
        'x_syndrome': x_syndrome,
        'z_syndrome': z_syndrome,
        'decoded_errors': decoded_errors,
        'metrics': metrics,
        'figures': {
            'lattice': lattice_fig,
            'errors': errors_fig,
            'syndrome': syndrome_fig,
            'graph': graph_fig,
            'matching': matching_fig,
            'correction': correction_fig,
            'process': process_figs
        }
    }

def run_benchmark():
    """
    Run a benchmark of the surface code error correction.
    """
    logger.info("Running surface code error correction benchmark")
    
    # Define benchmark parameters
    distances = [3, 5, 7]
    error_rates = [0.05, 0.1, 0.15, 0.2, 0.25]
    num_trials = 50
    
    # Run the benchmark
    results = benchmark_error_correction(distances, error_rates, num_trials)
    
    # Visualize the results
    fig = visualize_benchmark_results(results, distances, error_rates)
    fig.savefig('surface_code_benchmark.png')
    logger.info("Saved benchmark results visualization to 'surface_code_benchmark.png'")
    
    return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Demonstrate error correction
    demo_results = demonstrate_error_correction()
    
    # Run benchmark
    benchmark_results = run_benchmark()
    
    logger.info("Surface code error correction demonstration completed")