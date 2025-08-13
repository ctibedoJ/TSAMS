"""
TIBEDO Tensor Network Circuit Optimization Example

This example demonstrates how to use TIBEDO's tensor network-based circuit
optimization techniques to improve the efficiency of quantum circuits.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import os

# Import TIBEDO tensor network optimization components
from tibedo.quantum_information_new.tensor_network_circuit_optimization import (
    TensorNetworkCircuitOptimizer,
    CyclotomicTensorFusion,
    HardwareSpecificTensorOptimizer,
    EnhancedTibedoQuantumCircuitCompressor
)
from tibedo.quantum_information_new.quantum_circuit_optimization import TibedoQuantumCircuitCompressor
from tibedo.quantum_information_new.benchmark_visualization import (
    BenchmarkVisualizer,
    ComparisonPlotter
)

# Create output directories
os.makedirs('benchmark_results', exist_ok=True)
os.makedirs('comparison_results', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

def create_test_circuit(n_qubits):
    """Create a test circuit with the specified number of qubits."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    for i in range(n_qubits-1):
        qc.cx(i, i+1)
    for i in range(n_qubits):
        qc.rz(np.pi/4, i)
    for i in range(n_qubits-1):
        qc.cx(i, i+1)
    for i in range(n_qubits):
        qc.h(i)
    return qc

def print_circuit_info(circuit, name="Circuit"):
    """Print information about a quantum circuit."""
    print(f"\n{name}:")
    print(f"Number of qubits: {circuit.num_qubits}")
    print(f"Circuit depth: {circuit.depth()}")
    print(f"Gate count: {sum(circuit.count_ops().values())}")
    print(f"Gate composition: {circuit.count_ops()}")

def main():
    print("TIBEDO Tensor Network Circuit Optimization Example")
    print("=" * 50)

    # 1. Basic Circuit Optimization
    print("\n1. Basic Circuit Optimization")
    print("-" * 30)

    # Create a test circuit
    qc = create_test_circuit(5)
    print_circuit_info(qc, "Original Circuit")

    # Create a tensor network optimizer
    optimizer = TensorNetworkCircuitOptimizer(
        decomposition_method='svd',
        max_bond_dimension=16,
        truncation_threshold=1e-10,
        use_cyclotomic_optimization=True,
        use_spinor_representation=True
    )

    # Optimize the circuit
    optimized_qc = optimizer.optimize_circuit(qc)
    print_circuit_info(optimized_qc, "Tensor Network Optimized Circuit")

    # Get performance metrics
    metrics = optimizer.get_performance_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # 2. Advanced Circuit Optimization with Enhanced Compressor
    print("\n2. Advanced Circuit Optimization with Enhanced Compressor")
    print("-" * 30)

    # Create an enhanced compressor
    compressor = EnhancedTibedoQuantumCircuitCompressor(
        compression_level=3,
        use_spinor_reduction=True,
        use_phase_synchronization=True,
        use_prime_indexing=True,
        use_tensor_networks=True,
        max_bond_dimension=16,
        cyclotomic_conductor=168
    )

    # Compress the circuit
    compressed_qc = compressor.compress_circuit(qc)
    print_circuit_info(compressed_qc, "Enhanced Compressed Circuit")

    # Get performance metrics
    enhanced_metrics = compressor.get_performance_metrics()
    print("\nEnhanced Compressor Performance Metrics:")
    for key, value in enhanced_metrics.items():
        print(f"{key}: {value}")

    # 3. Hardware-Specific Optimization
    print("\n3. Hardware-Specific Optimization")
    print("-" * 30)

    # Optimize for different hardware backends
    ibmq_optimized = compressor.optimize_for_hardware(qc, 'ibmq')
    print_circuit_info(ibmq_optimized, "IBM Quantum Optimized Circuit")

    iqm_optimized = compressor.optimize_for_hardware(qc, 'iqm')
    print_circuit_info(iqm_optimized, "IQM Optimized Circuit")

    google_optimized = compressor.optimize_for_hardware(qc, 'google')
    print_circuit_info(google_optimized, "Google Quantum Optimized Circuit")

    # 4. Benchmarking Different Optimization Techniques
    print("\n4. Benchmarking Different Optimization Techniques")
    print("-" * 30)

    # Create different optimizers
    standard_compressor = TibedoQuantumCircuitCompressor(
        compression_level=2,
        use_spinor_reduction=True,
        use_phase_synchronization=True
    )

    tensor_optimizer = TensorNetworkCircuitOptimizer(
        decomposition_method='svd',
        max_bond_dimension=16,
        use_cyclotomic_optimization=False
    )

    enhanced_compressor = EnhancedTibedoQuantumCircuitCompressor(
        compression_level=3,
        use_tensor_networks=True,
        use_spinor_reduction=True,
        use_phase_synchronization=True
    )

    # Create a more complex circuit for benchmarking
    benchmark_qc = create_test_circuit(10)
    print_circuit_info(benchmark_qc, "Benchmark Circuit")

    # Benchmark standard compressor
    start_time = time.time()
    standard_compressed = standard_compressor.compress_circuit(benchmark_qc)
    standard_time = time.time() - start_time
    print_circuit_info(standard_compressed, "Standard Compressed Circuit")
    print(f"Standard compression time: {standard_time:.3f} seconds")

    # Benchmark tensor network optimizer
    start_time = time.time()
    tensor_optimized = tensor_optimizer.optimize_circuit(benchmark_qc)
    tensor_time = time.time() - start_time
    print_circuit_info(tensor_optimized, "Tensor Network Optimized Circuit")
    print(f"Tensor network optimization time: {tensor_time:.3f} seconds")

    # Benchmark enhanced compressor
    start_time = time.time()
    enhanced_compressed = enhanced_compressor.compress_circuit(benchmark_qc)
    enhanced_time = time.time() - start_time
    print_circuit_info(enhanced_compressed, "Enhanced Compressed Circuit")
    print(f"Enhanced compression time: {enhanced_time:.3f} seconds")

    # Collect benchmark results
    benchmark_results = {
        'Standard': {
            'original_gate_count': sum(benchmark_qc.count_ops().values()),
            'optimized_gate_count': sum(standard_compressed.count_ops().values()),
            'original_depth': benchmark_qc.depth(),
            'optimized_depth': standard_compressed.depth(),
            'optimization_time': standard_time
        },
        'TensorNetwork': {
            'original_gate_count': sum(benchmark_qc.count_ops().values()),
            'optimized_gate_count': sum(tensor_optimized.count_ops().values()),
            'original_depth': benchmark_qc.depth(),
            'optimized_depth': tensor_optimized.depth(),
            'optimization_time': tensor_time
        },
        'Enhanced': {
            'original_gate_count': sum(benchmark_qc.count_ops().values()),
            'optimized_gate_count': sum(enhanced_compressed.count_ops().values()),
            'original_depth': benchmark_qc.depth(),
            'optimized_depth': enhanced_compressed.depth(),
            'optimization_time': enhanced_time
        }
    }

    # Create visualizer
    visualizer = BenchmarkVisualizer(output_dir='benchmark_results')

    # Create plots
    print("\nGenerating benchmark visualizations...")
    visualizer.plot_gate_count_comparison(benchmark_results)
    visualizer.plot_circuit_depth_comparison(benchmark_results)
    visualizer.plot_optimization_time_comparison(benchmark_results)
    visualizer.plot_performance_radar(benchmark_results)

    # 5. Scaling Analysis
    print("\n5. Scaling Analysis")
    print("-" * 30)

    # Create circuits of different sizes
    circuit_sizes = [2, 4, 8, 12, 16]
    test_circuits = {size: create_test_circuit(size) for size in circuit_sizes}

    # Initialize results dictionary
    scaling_results = {
        'Standard': {},
        'TensorNetwork': {},
        'Enhanced': {}
    }

    print("\nBenchmarking circuits of different sizes...")
    # Benchmark each circuit size
    for size, circuit in test_circuits.items():
        print(f"\nBenchmarking {size}-qubit circuit:")
        
        # Standard compressor
        start_time = time.time()
        standard_compressed = standard_compressor.compress_circuit(circuit)
        standard_time = time.time() - start_time
        print(f"Standard compression time: {standard_time:.3f} seconds")
        
        # Tensor network optimizer
        start_time = time.time()
        tensor_optimized = tensor_optimizer.optimize_circuit(circuit)
        tensor_time = time.time() - start_time
        print(f"Tensor network optimization time: {tensor_time:.3f} seconds")
        
        # Enhanced compressor
        start_time = time.time()
        enhanced_compressed = enhanced_compressor.compress_circuit(circuit)
        enhanced_time = time.time() - start_time
        print(f"Enhanced compression time: {enhanced_time:.3f} seconds")
        
        # Store results
        scaling_results['Standard'][size] = {
            'optimization_time': standard_time,
            'gate_reduction': (sum(circuit.count_ops().values()) - sum(standard_compressed.count_ops().values())) / sum(circuit.count_ops().values()) * 100,
            'depth_reduction': (circuit.depth() - standard_compressed.depth()) / circuit.depth() * 100
        }
        
        scaling_results['TensorNetwork'][size] = {
            'optimization_time': tensor_time,
            'gate_reduction': (sum(circuit.count_ops().values()) - sum(tensor_optimized.count_ops().values())) / sum(circuit.count_ops().values()) * 100,
            'depth_reduction': (circuit.depth() - tensor_optimized.depth()) / circuit.depth() * 100
        }
        
        scaling_results['Enhanced'][size] = {
            'optimization_time': enhanced_time,
            'gate_reduction': (sum(circuit.count_ops().values()) - sum(enhanced_compressed.count_ops().values())) / sum(circuit.count_ops().values()) * 100,
            'depth_reduction': (circuit.depth() - enhanced_compressed.depth()) / circuit.depth() * 100
        }

    # Create comparison plotter
    comparison_plotter = ComparisonPlotter(output_dir='comparison_results')

    # Plot scaling results
    print("\nGenerating scaling visualizations...")
    comparison_plotter.plot_scaling_comparison(
        scaling_results,
        metric='optimization_time',
        x_label='Circuit Width (qubits)',
        y_label='Optimization Time (seconds)',
        title='Optimization Time Scaling'
    )

    comparison_plotter.plot_scaling_comparison(
        scaling_results,
        metric='gate_reduction',
        x_label='Circuit Width (qubits)',
        y_label='Gate Reduction (%)',
        title='Gate Reduction Scaling'
    )

    comparison_plotter.plot_scaling_comparison(
        scaling_results,
        metric='depth_reduction',
        x_label='Circuit Width (qubits)',
        y_label='Depth Reduction (%)',
        title='Depth Reduction Scaling'
    )

    print("\nExample completed successfully!")
    print("\nVisualization files have been saved to:")
    print("- benchmark_results/ (benchmark comparisons)")
    print("- comparison_results/ (scaling analysis)")


if __name__ == "__main__":
    main()