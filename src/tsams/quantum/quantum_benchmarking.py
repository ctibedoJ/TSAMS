"""
TIBEDO Quantum Benchmarking Suite

This module provides comprehensive benchmarking tools for evaluating the performance
of TIBEDO's quantum computing components, including circuit optimization, error
mitigation, and hybrid algorithms. The benchmarking suite enables systematic
comparison against industry standards and tracking of performance improvements.

Key components:
1. CircuitOptimizationBenchmark: Benchmarks quantum circuit optimization techniques
2. ErrorMitigationBenchmark: Benchmarks quantum error mitigation strategies
3. HybridAlgorithmBenchmark: Benchmarks quantum-classical hybrid algorithms
4. BenchmarkSuite: Comprehensive benchmarking suite for all quantum components
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.providers import Backend
from qiskit.opflow import PauliSumOp, PauliOp
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit.library import TwoLocal, EfficientSU2, ZZFeatureMap
from qiskit.algorithms import VQE, QAOA
from typing import List, Tuple, Dict, Any, Optional, Union
import math
import time
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
import datetime
import pandas as pd
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Import TIBEDO quantum components
from quantum_circuit_optimization import TibedoQuantumCircuitCompressor, PhaseSynchronizedGateSet
from quantum_error_mitigation import SpinorErrorModel, AdaptiveErrorMitigation
from quantum_hybrid_algorithms import TibedoEnhancedVQE, SpinorQuantumML
from advanced_error_mitigation import DynamicSpinorErrorModel, EnhancedAdaptiveErrorMitigation
from enhanced_quantum_vqe import NaturalGradientVQE, SpinorQuantumNN, DistributedQuantumOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CircuitOptimizationBenchmark:
    """
    Benchmarks quantum circuit optimization techniques.
    
    This class provides tools for benchmarking the performance of quantum circuit
    optimization techniques, including circuit compression, gate fusion, and
    hardware-specific optimizations.
    """
    
    def __init__(self, 
                 backends: List[Backend],
                 save_dir: str = './benchmark_results',
                 num_repetitions: int = 5,
                 circuit_sizes: List[int] = [5, 10, 20, 50],
                 circuit_depths: List[int] = [5, 10, 20, 50],
                 optimization_levels: List[int] = [1, 2, 3]):
        """
        Initialize the Circuit Optimization Benchmark.
        
        Args:
            backends: List of quantum backends to benchmark on
            save_dir: Directory to save benchmark results
            num_repetitions: Number of repetitions for each benchmark
            circuit_sizes: List of circuit sizes (number of qubits) to benchmark
            circuit_depths: List of circuit depths to benchmark
            optimization_levels: List of optimization levels to benchmark
        """
        self.backends = backends
        self.save_dir = save_dir
        self.num_repetitions = num_repetitions
        self.circuit_sizes = circuit_sizes
        self.circuit_depths = circuit_depths
        self.optimization_levels = optimization_levels
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'circuit_compression': [],
            'gate_fusion': [],
            'hardware_specific': []
        }
        
        logger.info(f"Initialized Circuit Optimization Benchmark")
        logger.info(f"  Number of backends: {len(backends)}")
        logger.info(f"  Number of repetitions: {num_repetitions}")
        logger.info(f"  Circuit sizes: {circuit_sizes}")
        logger.info(f"  Circuit depths: {circuit_depths}")
        logger.info(f"  Optimization levels: {optimization_levels}")
    
    def generate_random_circuit(self, num_qubits: int, depth: int) -> QuantumCircuit:
        """
        Generate a random quantum circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
            depth: Depth of the circuit
            
        Returns:
            Random quantum circuit
        """
        circuit = QuantumCircuit(num_qubits)
        
        # List of available gates
        single_qubit_gates = ['h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz']
        two_qubit_gates = ['cx', 'cz', 'swap']
        
        # Add random gates
        for _ in range(depth):
            # With 70% probability, add a single-qubit gate
            if np.random.random() < 0.7:
                qubit = np.random.randint(0, num_qubits)
                gate = np.random.choice(single_qubit_gates)
                
                if gate in ['rx', 'ry', 'rz']:
                    # Parameterized gate
                    angle = np.random.random() * 2 * np.pi
                    getattr(circuit, gate)(angle, qubit)
                else:
                    # Non-parameterized gate
                    getattr(circuit, gate)(qubit)
            else:
                # Add a two-qubit gate
                qubits = np.random.choice(num_qubits, size=2, replace=False)
                gate = np.random.choice(two_qubit_gates)
                
                getattr(circuit, gate)(qubits[0], qubits[1])
        
        return circuit
    
    def benchmark_circuit_compression(self):
        """
        Benchmark circuit compression techniques.
        """
        logger.info("Starting circuit compression benchmark")
        
        results = []
        
        # For each circuit size and depth
        for size in self.circuit_sizes:
            for depth in self.circuit_depths:
                # Skip if circuit is too large
                if size * depth > 1000:
                    continue
                
                logger.info(f"Benchmarking circuit with {size} qubits and depth {depth}")
                
                # For each optimization level
                for level in self.optimization_levels:
                    # For each repetition
                    for rep in range(self.num_repetitions):
                        # Generate random circuit
                        circuit = self.generate_random_circuit(size, depth)
                        original_depth = circuit.depth()
                        original_gate_count = sum(circuit.count_ops().values())
                        
                        # Create circuit compressor
                        compressor = TibedoQuantumCircuitCompressor(
                            compression_level=level,
                            use_spinor_reduction=True,
                            use_phase_synchronization=True,
                            use_prime_indexing=True
                        )
                        
                        # Measure compression time
                        start_time = time.time()
                        compressed_circuit = compressor.compress_circuit(circuit)
                        compression_time = time.time() - start_time
                        
                        # Measure compressed circuit properties
                        compressed_depth = compressed_circuit.depth()
                        compressed_gate_count = sum(compressed_circuit.count_ops().values())
                        
                        # Calculate metrics
                        depth_reduction = original_depth - compressed_depth
                        depth_reduction_percentage = (depth_reduction / original_depth) * 100 if original_depth > 0 else 0
                        gate_reduction = original_gate_count - compressed_gate_count
                        gate_reduction_percentage = (gate_reduction / original_gate_count) * 100 if original_gate_count > 0 else 0
                        
                        # Store results
                        result = {
                            'circuit_size': size,
                            'circuit_depth': depth,
                            'optimization_level': level,
                            'repetition': rep,
                            'original_depth': original_depth,
                            'compressed_depth': compressed_depth,
                            'depth_reduction': depth_reduction,
                            'depth_reduction_percentage': depth_reduction_percentage,
                            'original_gate_count': original_gate_count,
                            'compressed_gate_count': compressed_gate_count,
                            'gate_reduction': gate_reduction,
                            'gate_reduction_percentage': gate_reduction_percentage,
                            'compression_time': compression_time
                        }
                        
                        results.append(result)
                        
                        logger.info(f"  Level {level}, Rep {rep+1}/{self.num_repetitions}: "
                                   f"Depth reduction: {depth_reduction_percentage:.2f}%, "
                                   f"Gate reduction: {gate_reduction_percentage:.2f}%, "
                                   f"Time: {compression_time:.4f}s")
        
        # Store results
        self.results['circuit_compression'] = results
        
        # Save results to file
        self._save_results('circuit_compression')
        
        logger.info("Circuit compression benchmark completed")
        
        return results
    
    def benchmark_gate_fusion(self):
        """
        Benchmark gate fusion algorithms.
        """
        logger.info("Starting gate fusion benchmark")
        
        results = []
        
        # For each circuit size and depth
        for size in self.circuit_sizes:
            for depth in self.circuit_depths:
                # Skip if circuit is too large
                if size * depth > 1000:
                    continue
                
                logger.info(f"Benchmarking circuit with {size} qubits and depth {depth}")
                
                # For each optimization level
                for level in self.optimization_levels:
                    # For each repetition
                    for rep in range(self.num_repetitions):
                        # Generate random circuit
                        circuit = self.generate_random_circuit(size, depth)
                        original_depth = circuit.depth()
                        original_gate_count = sum(circuit.count_ops().values())
                        
                        # Create phase synchronized gate set
                        gate_set = PhaseSynchronizedGateSet(
                            optimization_level=level,
                            cyclotomic_conductor=56
                        )
                        
                        # Measure fusion time
                        start_time = time.time()
                        fused_circuit = gate_set.optimize_circuit(circuit)
                        fusion_time = time.time() - start_time
                        
                        # Measure fused circuit properties
                        fused_depth = fused_circuit.depth()
                        fused_gate_count = sum(fused_circuit.count_ops().values())
                        
                        # Calculate metrics
                        depth_reduction = original_depth - fused_depth
                        depth_reduction_percentage = (depth_reduction / original_depth) * 100 if original_depth > 0 else 0
                        gate_reduction = original_gate_count - fused_gate_count
                        gate_reduction_percentage = (gate_reduction / original_gate_count) * 100 if original_gate_count > 0 else 0
                        
                        # Store results
                        result = {
                            'circuit_size': size,
                            'circuit_depth': depth,
                            'optimization_level': level,
                            'repetition': rep,
                            'original_depth': original_depth,
                            'fused_depth': fused_depth,
                            'depth_reduction': depth_reduction,
                            'depth_reduction_percentage': depth_reduction_percentage,
                            'original_gate_count': original_gate_count,
                            'fused_gate_count': fused_gate_count,
                            'gate_reduction': gate_reduction,
                            'gate_reduction_percentage': gate_reduction_percentage,
                            'fusion_time': fusion_time
                        }
                        
                        results.append(result)
                        
                        logger.info(f"  Level {level}, Rep {rep+1}/{self.num_repetitions}: "
                                   f"Depth reduction: {depth_reduction_percentage:.2f}%, "
                                   f"Gate reduction: {gate_reduction_percentage:.2f}%, "
                                   f"Time: {fusion_time:.4f}s")
        
        # Store results
        self.results['gate_fusion'] = results
        
        # Save results to file
        self._save_results('gate_fusion')
        
        logger.info("Gate fusion benchmark completed")
        
        return results
    
    def benchmark_hardware_specific(self):
        """
        Benchmark hardware-specific optimizations.
        """
        logger.info("Starting hardware-specific optimization benchmark")
        
        results = []
        
        # For each backend
        for backend_idx, backend in enumerate(self.backends):
            logger.info(f"Benchmarking on backend: {backend.name()}")
            
            # For each circuit size and depth
            for size in self.circuit_sizes:
                # Skip if circuit is too large for the backend
                if size > backend.configuration().n_qubits:
                    continue
                
                for depth in self.circuit_depths:
                    # Skip if circuit is too large
                    if size * depth > 1000:
                        continue
                    
                    logger.info(f"Benchmarking circuit with {size} qubits and depth {depth}")
                    
                    # For each optimization level
                    for level in self.optimization_levels:
                        # For each repetition
                        for rep in range(self.num_repetitions):
                            # Generate random circuit
                            circuit = self.generate_random_circuit(size, depth)
                            original_depth = circuit.depth()
                            original_gate_count = sum(circuit.count_ops().values())
                            
                            # Create circuit compressor
                            compressor = TibedoQuantumCircuitCompressor(
                                compression_level=level,
                                use_spinor_reduction=True,
                                use_phase_synchronization=True,
                                use_prime_indexing=True
                            )
                            
                            # Measure optimization time
                            start_time = time.time()
                            optimized_circuit = compressor.optimize_for_backend(circuit, backend)
                            optimization_time = time.time() - start_time
                            
                            # Measure optimized circuit properties
                            optimized_depth = optimized_circuit.depth()
                            optimized_gate_count = sum(optimized_circuit.count_ops().values())
                            
                            # Calculate metrics
                            depth_reduction = original_depth - optimized_depth
                            depth_reduction_percentage = (depth_reduction / original_depth) * 100 if original_depth > 0 else 0
                            gate_reduction = original_gate_count - optimized_gate_count
                            gate_reduction_percentage = (gate_reduction / original_gate_count) * 100 if original_gate_count > 0 else 0
                            
                            # Store results
                            result = {
                                'backend': backend.name(),
                                'circuit_size': size,
                                'circuit_depth': depth,
                                'optimization_level': level,
                                'repetition': rep,
                                'original_depth': original_depth,
                                'optimized_depth': optimized_depth,
                                'depth_reduction': depth_reduction,
                                'depth_reduction_percentage': depth_reduction_percentage,
                                'original_gate_count': original_gate_count,
                                'optimized_gate_count': optimized_gate_count,
                                'gate_reduction': gate_reduction,
                                'gate_reduction_percentage': gate_reduction_percentage,
                                'optimization_time': optimization_time
                            }
                            
                            results.append(result)
                            
                            logger.info(f"  Level {level}, Rep {rep+1}/{self.num_repetitions}: "
                                       f"Depth reduction: {depth_reduction_percentage:.2f}%, "
                                       f"Gate reduction: {gate_reduction_percentage:.2f}%, "
                                       f"Time: {optimization_time:.4f}s")
        
        # Store results
        self.results['hardware_specific'] = results
        
        # Save results to file
        self._save_results('hardware_specific')
        
        logger.info("Hardware-specific optimization benchmark completed")
        
        return results
    
    def run_all_benchmarks(self):
        """
        Run all circuit optimization benchmarks.
        """
        logger.info("Running all circuit optimization benchmarks")
        
        # Run benchmarks
        self.benchmark_circuit_compression()
        self.benchmark_gate_fusion()
        self.benchmark_hardware_specific()
        
        # Generate summary report
        self.generate_summary_report()
        
        logger.info("All circuit optimization benchmarks completed")
    
    def generate_summary_report(self):
        """
        Generate summary report of benchmark results.
        """
        logger.info("Generating summary report")
        
        # Create summary dictionary
        summary = {
            'timestamp': datetime.datetime.now().isoformat(),
            'circuit_compression': self._summarize_results('circuit_compression'),
            'gate_fusion': self._summarize_results('gate_fusion'),
            'hardware_specific': self._summarize_results('hardware_specific')
        }
        
        # Save summary to file
        summary_path = os.path.join(self.save_dir, 'circuit_optimization_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved to {summary_path}")
        
        # Generate visualizations
        self._generate_visualizations()
        
        return summary
    
    def _summarize_results(self, benchmark_type: str) -> Dict[str, Any]:
        """
        Summarize results for a specific benchmark type.
        
        Args:
            benchmark_type: Type of benchmark to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        results = self.results.get(benchmark_type, [])
        
        if not results:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Group by circuit size, depth, and optimization level
        grouped = df.groupby(['circuit_size', 'circuit_depth', 'optimization_level'])
        
        # Calculate summary statistics
        if benchmark_type == 'circuit_compression':
            summary = grouped.agg({
                'depth_reduction_percentage': ['mean', 'std', 'min', 'max'],
                'gate_reduction_percentage': ['mean', 'std', 'min', 'max'],
                'compression_time': ['mean', 'std', 'min', 'max']
            }).reset_index()
        elif benchmark_type == 'gate_fusion':
            summary = grouped.agg({
                'depth_reduction_percentage': ['mean', 'std', 'min', 'max'],
                'gate_reduction_percentage': ['mean', 'std', 'min', 'max'],
                'fusion_time': ['mean', 'std', 'min', 'max']
            }).reset_index()
        elif benchmark_type == 'hardware_specific':
            # Include backend in grouping
            grouped = df.groupby(['backend', 'circuit_size', 'circuit_depth', 'optimization_level'])
            summary = grouped.agg({
                'depth_reduction_percentage': ['mean', 'std', 'min', 'max'],
                'gate_reduction_percentage': ['mean', 'std', 'min', 'max'],
                'optimization_time': ['mean', 'std', 'min', 'max']
            }).reset_index()
        
        # Convert to dictionary for JSON serialization
        summary_dict = summary.to_dict(orient='records')
        
        return summary_dict
    
    def _generate_visualizations(self):
        """
        Generate visualizations of benchmark results.
        """
        logger.info("Generating visualizations")
        
        # Create visualizations directory
        viz_dir = os.path.join(self.save_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate visualizations for each benchmark type
        for benchmark_type in self.results.keys():
            if not self.results[benchmark_type]:
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(self.results[benchmark_type])
            
            # Create visualizations
            self._create_depth_reduction_plot(df, benchmark_type, viz_dir)
            self._create_gate_reduction_plot(df, benchmark_type, viz_dir)
            self._create_execution_time_plot(df, benchmark_type, viz_dir)
            
            # Create optimization level comparison
            self._create_optimization_level_comparison(df, benchmark_type, viz_dir)
        
        logger.info(f"Visualizations saved to {viz_dir}")
    
    def _create_depth_reduction_plot(self, df: pd.DataFrame, benchmark_type: str, viz_dir: str):
        """
        Create depth reduction plot.
        
        Args:
            df: DataFrame with benchmark results
            benchmark_type: Type of benchmark
            viz_dir: Directory to save visualization
        """
        plt.figure(figsize=(10, 6))
        
        # Create grouped bar plot
        sns.barplot(x='circuit_size', y='depth_reduction_percentage', hue='optimization_level', data=df)
        
        plt.title(f'Circuit Depth Reduction by Circuit Size ({benchmark_type})')
        plt.xlabel('Circuit Size (qubits)')
        plt.ylabel('Depth Reduction (%)')
        plt.legend(title='Optimization Level')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(viz_dir, f'{benchmark_type}_depth_reduction.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_gate_reduction_plot(self, df: pd.DataFrame, benchmark_type: str, viz_dir: str):
        """
        Create gate reduction plot.
        
        Args:
            df: DataFrame with benchmark results
            benchmark_type: Type of benchmark
            viz_dir: Directory to save visualization
        """
        plt.figure(figsize=(10, 6))
        
        # Create grouped bar plot
        sns.barplot(x='circuit_size', y='gate_reduction_percentage', hue='optimization_level', data=df)
        
        plt.title(f'Gate Count Reduction by Circuit Size ({benchmark_type})')
        plt.xlabel('Circuit Size (qubits)')
        plt.ylabel('Gate Reduction (%)')
        plt.legend(title='Optimization Level')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(viz_dir, f'{benchmark_type}_gate_reduction.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_execution_time_plot(self, df: pd.DataFrame, benchmark_type: str, viz_dir: str):
        """
        Create execution time plot.
        
        Args:
            df: DataFrame with benchmark results
            benchmark_type: Type of benchmark
            viz_dir: Directory to save visualization
        """
        plt.figure(figsize=(10, 6))
        
        # Determine time column name
        if benchmark_type == 'circuit_compression':
            time_col = 'compression_time'
        elif benchmark_type == 'gate_fusion':
            time_col = 'fusion_time'
        else:
            time_col = 'optimization_time'
        
        # Create line plot
        sns.lineplot(x='circuit_size', y=time_col, hue='optimization_level', marker='o', data=df)
        
        plt.title(f'Execution Time by Circuit Size ({benchmark_type})')
        plt.xlabel('Circuit Size (qubits)')
        plt.ylabel('Execution Time (s)')
        plt.legend(title='Optimization Level')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(viz_dir, f'{benchmark_type}_execution_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_optimization_level_comparison(self, df: pd.DataFrame, benchmark_type: str, viz_dir: str):
        """
        Create optimization level comparison plot.
        
        Args:
            df: DataFrame with benchmark results
            benchmark_type: Type of benchmark
            viz_dir: Directory to save visualization
        """
        plt.figure(figsize=(12, 8))
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot depth reduction by optimization level
        sns.boxplot(x='optimization_level', y='depth_reduction_percentage', data=df, ax=axes[0])
        axes[0].set_title(f'Depth Reduction by Optimization Level ({benchmark_type})')
        axes[0].set_xlabel('Optimization Level')
        axes[0].set_ylabel('Depth Reduction (%)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot gate reduction by optimization level
        sns.boxplot(x='optimization_level', y='gate_reduction_percentage', data=df, ax=axes[1])
        axes[1].set_title(f'Gate Reduction by Optimization Level ({benchmark_type})')
        axes[1].set_xlabel('Optimization Level')
        axes[1].set_ylabel('Gate Reduction (%)')
        axes[1].grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(viz_dir, f'{benchmark_type}_optimization_level_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self, benchmark_type: str):
        """
        Save benchmark results to file.
        
        Args:
            benchmark_type: Type of benchmark
        """
        # Create results file path
        file_path = os.path.join(self.save_dir, f'{benchmark_type}_results.json')
        
        # Save results to file
        with open(file_path, 'w') as f:
            json.dump(self.results[benchmark_type], f, indent=2)
        
        logger.info(f"Results saved to {file_path}")


class ErrorMitigationBenchmark:
    """
    Benchmarks quantum error mitigation strategies.
    
    This class provides tools for benchmarking the performance of quantum error
    mitigation strategies, including dynamic error characterization, real-time
    error tracking, and mid-circuit measurement and feedback.
    """
    
    def __init__(self, 
                 backends: List[Backend],
                 save_dir: str = './benchmark_results',
                 num_repetitions: int = 5,
                 circuit_sizes: List[int] = [2, 3, 5, 10],
                 error_rates: List[float] = [0.001, 0.01, 0.05, 0.1],
                 mitigation_strategies: List[str] = ['measurement', 'zero_noise', 'real_time', 'mid_circuit']):
        """
        Initialize the Error Mitigation Benchmark.
        
        Args:
            backends: List of quantum backends to benchmark on
            save_dir: Directory to save benchmark results
            num_repetitions: Number of repetitions for each benchmark
            circuit_sizes: List of circuit sizes (number of qubits) to benchmark
            error_rates: List of error rates to benchmark
            mitigation_strategies: List of mitigation strategies to benchmark
        """
        self.backends = backends
        self.save_dir = save_dir
        self.num_repetitions = num_repetitions
        self.circuit_sizes = circuit_sizes
        self.error_rates = error_rates
        self.mitigation_strategies = mitigation_strategies
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'dynamic_error': [],
            'real_time_tracking': [],
            'mid_circuit_correction': [],
            'strategy_comparison': []
        }
        
        logger.info(f"Initialized Error Mitigation Benchmark")
        logger.info(f"  Number of backends: {len(backends)}")
        logger.info(f"  Number of repetitions: {num_repetitions}")
        logger.info(f"  Circuit sizes: {circuit_sizes}")
        logger.info(f"  Error rates: {error_rates}")
        logger.info(f"  Mitigation strategies: {mitigation_strategies}")
    
    def generate_ghz_circuit(self, num_qubits: int) -> QuantumCircuit:
        """
        Generate a GHZ state preparation circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
            
        Returns:
            GHZ state preparation circuit
        """
        circuit = QuantumCircuit(num_qubits)
        
        # Create GHZ state
        circuit.h(0)
        for i in range(1, num_qubits):
            circuit.cx(0, i)
        
        # Add measurement
        circuit.measure_all()
        
        return circuit
    
    def generate_qft_circuit(self, num_qubits: int) -> QuantumCircuit:
        """
        Generate a Quantum Fourier Transform circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
            
        Returns:
            QFT circuit
        """
        circuit = QuantumCircuit(num_qubits)
        
        # Initialize with superposition
        circuit.h(range(num_qubits))
        
        # Apply QFT
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                circuit.cp(np.pi / float(2**(j-i)), i, j)
            circuit.h(i)
        
        # Add measurement
        circuit.measure_all()
        
        return circuit
    
    def create_noisy_backend(self, backend: Backend, error_rate: float) -> Backend:
        """
        Create a noisy backend with specified error rate.
        
        Args:
            backend: Base backend
            error_rate: Error rate to apply
            
        Returns:
            Noisy backend
        """
        # Create noise model
        from qiskit.providers.aer.noise import NoiseModel
        from qiskit.providers.aer.noise.errors import depolarizing_error
        
        noise_model = NoiseModel()
        
        # Add single-qubit gate errors
        error = depolarizing_error(error_rate, 1)
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'h', 'x', 'y', 'z'])
        
        # Add two-qubit gate errors
        error = depolarizing_error(error_rate * 2, 2)
        noise_model.add_all_qubit_quantum_error(error, ['cx', 'cz', 'swap'])
        
        # Create noisy backend
        noisy_backend = Aer.get_backend('qasm_simulator')
        noisy_backend.set_options(noise_model=noise_model)
        
        return noisy_backend
    
    def benchmark_dynamic_error(self):
        """
        Benchmark dynamic error characterization.
        """
        logger.info("Starting dynamic error characterization benchmark")
        
        results = []
        
        # For each circuit size
        for size in self.circuit_sizes:
            logger.info(f"Benchmarking circuit with {size} qubits")
            
            # Generate test circuits
            ghz_circuit = self.generate_ghz_circuit(size)
            qft_circuit = self.generate_qft_circuit(size)
            
            # For each error rate
            for error_rate in self.error_rates:
                logger.info(f"  Error rate: {error_rate}")
                
                # Create noisy backend
                noisy_backend = self.create_noisy_backend(self.backends[0], error_rate)
                
                # For each repetition
                for rep in range(self.num_repetitions):
                    # Create standard error model
                    standard_model = SpinorErrorModel(
                        error_characterization_shots=1024,
                        use_spinor_reduction=True,
                        use_phase_synchronization=True,
                        use_prime_indexing=True
                    )
                    
                    # Create dynamic error model
                    dynamic_model = DynamicSpinorErrorModel(
                        error_characterization_shots=1024,
                        use_spinor_reduction=True,
                        use_phase_synchronization=True,
                        use_prime_indexing=True,
                        dynamic_update_frequency=5,
                        error_history_window=10,
                        use_bayesian_updating=True
                    )
                    
                    # Generate error models
                    start_time = time.time()
                    standard_params = standard_model.generate_error_model(noisy_backend)
                    standard_time = time.time() - start_time
                    
                    start_time = time.time()
                    dynamic_params = dynamic_model.generate_error_model(noisy_backend)
                    dynamic_time = time.time() - start_time
                    
                    # Execute circuits with standard error model
                    start_time = time.time()
                    standard_ghz_result = standard_model.simulate_errors(ghz_circuit)
                    standard_qft_result = standard_model.simulate_errors(qft_circuit)
                    standard_exec_time = time.time() - start_time
                    
                    # Execute circuits with dynamic error model
                    start_time = time.time()
                    dynamic_ghz_result = dynamic_model.simulate_errors(ghz_circuit)
                    
                    # Update error model based on results
                    dynamic_model.update_error_model_from_results(
                        ghz_circuit, 
                        {'counts': {'0' * size: 500, '1' * size: 500}},
                        {'counts': {'0' * size: 500, '1' * size: 500}}
                    )
                    
                    dynamic_qft_result = dynamic_model.simulate_errors(qft_circuit)
                    dynamic_exec_time = time.time() - start_time
                    
                    # Calculate metrics
                    standard_fidelity = (standard_ghz_result['fidelity'] + standard_qft_result['fidelity']) / 2
                    dynamic_fidelity = (dynamic_ghz_result['fidelity'] + dynamic_qft_result['fidelity']) / 2
                    
                    fidelity_improvement = dynamic_fidelity - standard_fidelity
                    fidelity_improvement_percentage = (fidelity_improvement / standard_fidelity) * 100 if standard_fidelity > 0 else 0
                    
                    # Store results
                    result = {
                        'circuit_size': size,
                        'error_rate': error_rate,
                        'repetition': rep,
                        'standard_fidelity': standard_fidelity,
                        'dynamic_fidelity': dynamic_fidelity,
                        'fidelity_improvement': fidelity_improvement,
                        'fidelity_improvement_percentage': fidelity_improvement_percentage,
                        'standard_characterization_time': standard_time,
                        'dynamic_characterization_time': dynamic_time,
                        'standard_execution_time': standard_exec_time,
                        'dynamic_execution_time': dynamic_exec_time
                    }
                    
                    results.append(result)
                    
                    logger.info(f"    Rep {rep+1}/{self.num_repetitions}: "
                               f"Fidelity improvement: {fidelity_improvement_percentage:.2f}%")
        
        # Store results
        self.results['dynamic_error'] = results
        
        # Save results to file
        self._save_results('dynamic_error')
        
        logger.info("Dynamic error characterization benchmark completed")
        
        return results
    
    def benchmark_real_time_tracking(self):
        """
        Benchmark real-time error tracking.
        """
        logger.info("Starting real-time error tracking benchmark")
        
        results = []
        
        # For each circuit size
        for size in self.circuit_sizes:
            logger.info(f"Benchmarking circuit with {size} qubits")
            
            # Generate test circuit (random circuit with more operations)
            circuit = QuantumCircuit(size)
            
            # Add random operations
            for _ in range(size * 3):
                # Add single-qubit gate
                qubit = np.random.randint(0, size)
                circuit.rx(np.random.random() * np.pi, qubit)
                
                # Add two-qubit gate if possible
                if size > 1:
                    control = np.random.randint(0, size)
                    target = (control + 1 + np.random.randint(0, size - 1)) % size
                    circuit.cx(control, target)
            
            # Add measurement
            circuit.measure_all()
            
            # For each error rate
            for error_rate in self.error_rates:
                logger.info(f"  Error rate: {error_rate}")
                
                # Create noisy backend
                noisy_backend = self.create_noisy_backend(self.backends[0], error_rate)
                
                # For each repetition
                for rep in range(self.num_repetitions):
                    # Create error model
                    error_model = DynamicSpinorErrorModel(
                        error_characterization_shots=1024,
                        use_spinor_reduction=True,
                        use_phase_synchronization=True,
                        use_prime_indexing=True
                    )
                    
                    # Create real-time error tracker
                    tracker = RealTimeErrorTracker(
                        error_model=error_model,
                        tracking_window_size=20,
                        compensation_threshold=0.05,
                        use_phase_compensation=True,
                        use_amplitude_compensation=True
                    )
                    
                    # Execute circuit without tracking
                    start_time = time.time()
                    job = noisy_backend.run(circuit, shots=1024)
                    untracked_result = job.result()
                    untracked_time = time.time() - start_time
                    
                    # Apply real-time tracking
                    start_time = time.time()
                    tracked_circuit = tracker.apply_compensation_to_circuit(circuit)
                    tracking_time = time.time() - start_time
                    
                    # Execute tracked circuit
                    start_time = time.time()
                    job = noisy_backend.run(tracked_circuit, shots=1024)
                    tracked_result = job.result()
                    tracked_time = time.time() - start_time
                    
                    # Calculate metrics
                    untracked_counts = untracked_result.get_counts(circuit)
                    tracked_counts = tracked_result.get_counts(tracked_circuit)
                    
                    # Calculate distribution fidelity (simplified)
                    untracked_dist = {k: v / 1024 for k, v in untracked_counts.items()}
                    tracked_dist = {k: v / 1024 for k, v in tracked_counts.items()}
                    
                    # Calculate total variation distance
                    tvd = 0.0
                    all_bitstrings = set(untracked_dist.keys()) | set(tracked_dist.keys())
                    for bitstring in all_bitstrings:
                        untracked_prob = untracked_dist.get(bitstring, 0.0)
                        tracked_prob = tracked_dist.get(bitstring, 0.0)
                        tvd += abs(untracked_prob - tracked_prob)
                    
                    tvd = tvd / 2.0
                    
                    # Calculate improvement metrics
                    num_compensation_ops = len(tracker.error_compensation_history)
                    circuit_size_increase = (len(tracked_circuit) - len(circuit)) / len(circuit) * 100
                    
                    # Store results
                    result = {
                        'circuit_size': size,
                        'error_rate': error_rate,
                        'repetition': rep,
                        'untracked_execution_time': untracked_time,
                        'tracking_time': tracking_time,
                        'tracked_execution_time': tracked_time,
                        'total_variation_distance': tvd,
                        'num_compensation_operations': num_compensation_ops,
                        'circuit_size_increase': circuit_size_increase
                    }
                    
                    results.append(result)
                    
                    logger.info(f"    Rep {rep+1}/{self.num_repetitions}: "
                               f"TVD: {tvd:.4f}, "
                               f"Compensation ops: {num_compensation_ops}")
        
        # Store results
        self.results['real_time_tracking'] = results
        
        # Save results to file
        self._save_results('real_time_tracking')
        
        logger.info("Real-time error tracking benchmark completed")
        
        return results
    
    def benchmark_mid_circuit_correction(self):
        """
        Benchmark mid-circuit error correction.
        """
        logger.info("Starting mid-circuit error correction benchmark")
        
        results = []
        
        # For each circuit size
        for size in self.circuit_sizes:
            # Skip if circuit is too small
            if size < 3:
                continue
                
            logger.info(f"Benchmarking circuit with {size} qubits")
            
            # Generate test circuit
            circuit = self.generate_ghz_circuit(size)
            
            # For each error rate
            for error_rate in self.error_rates:
                logger.info(f"  Error rate: {error_rate}")
                
                # Create noisy backend
                noisy_backend = self.create_noisy_backend(self.backends[0], error_rate)
                
                # For each repetition
                for rep in range(self.num_repetitions):
                    # Create error model
                    error_model = DynamicSpinorErrorModel(
                        error_characterization_shots=1024,
                        use_spinor_reduction=True,
                        use_phase_synchronization=True,
                        use_prime_indexing=True
                    )
                    
                    # Create mid-circuit correction
                    correction = MidCircuitErrorCorrection(
                        error_model=error_model,
                        use_parity_checks=True,
                        use_syndrome_extraction=True,
                        max_correction_rounds=3
                    )
                    
                    # Execute circuit without correction
                    start_time = time.time()
                    job = noisy_backend.run(circuit, shots=1024)
                    uncorrected_result = job.result()
                    uncorrected_time = time.time() - start_time
                    
                    # Apply mid-circuit correction
                    start_time = time.time()
                    corrected_circuit = correction.apply_error_correction(circuit)
                    correction_time = time.time() - start_time
                    
                    # Execute corrected circuit
                    start_time = time.time()
                    job = noisy_backend.run(corrected_circuit, shots=1024)
                    corrected_result = job.result()
                    corrected_time = time.time() - start_time
                    
                    # Calculate metrics
                    uncorrected_counts = uncorrected_result.get_counts(circuit)
                    corrected_counts = corrected_result.get_counts(corrected_circuit)
                    
                    # For GHZ state, we expect to see mainly |00...0⟩ and |11...1⟩
                    # Calculate success probability
                    uncorrected_success_prob = (uncorrected_counts.get('0' * size, 0) + 
                                              uncorrected_counts.get('1' * size, 0)) / 1024
                    
                    # For corrected circuit, we need to map the measurement results
                    # This is a simplification - in practice, we would need to decode the results
                    corrected_success_prob = 0.0
                    for bitstring, count in corrected_counts.items():
                        # Check if bitstring is close to |00...0⟩ or |11...1⟩
                        zeros = bitstring.count('0')
                        ones = bitstring.count('1')
                        
                        if zeros >= len(bitstring) * 0.8 or ones >= len(bitstring) * 0.8:
                            corrected_success_prob += count / 1024
                    
                    # Calculate improvement
                    success_improvement = corrected_success_prob - uncorrected_success_prob
                    success_improvement_percentage = (success_improvement / uncorrected_success_prob) * 100 if uncorrected_success_prob > 0 else 0
                    
                    # Calculate overhead
                    qubit_overhead = corrected_circuit.num_qubits / circuit.num_qubits
                    gate_overhead = len(corrected_circuit) / len(circuit)
                    
                    # Store results
                    result = {
                        'circuit_size': size,
                        'error_rate': error_rate,
                        'repetition': rep,
                        'uncorrected_success_probability': uncorrected_success_prob,
                        'corrected_success_probability': corrected_success_prob,
                        'success_improvement': success_improvement,
                        'success_improvement_percentage': success_improvement_percentage,
                        'uncorrected_execution_time': uncorrected_time,
                        'correction_time': correction_time,
                        'corrected_execution_time': corrected_time,
                        'qubit_overhead': qubit_overhead,
                        'gate_overhead': gate_overhead
                    }
                    
                    results.append(result)
                    
                    logger.info(f"    Rep {rep+1}/{self.num_repetitions}: "
                               f"Success improvement: {success_improvement_percentage:.2f}%, "
                               f"Qubit overhead: {qubit_overhead:.2f}x")
        
        # Store results
        self.results['mid_circuit_correction'] = results
        
        # Save results to file
        self._save_results('mid_circuit_correction')
        
        logger.info("Mid-circuit error correction benchmark completed")
        
        return results
    
    def benchmark_strategy_comparison(self):
        """
        Benchmark comparison of different error mitigation strategies.
        """
        logger.info("Starting error mitigation strategy comparison benchmark")
        
        results = []
        
        # For each circuit size
        for size in self.circuit_sizes:
            logger.info(f"Benchmarking circuit with {size} qubits")
            
            # Generate test circuit
            circuit = self.generate_qft_circuit(size)
            
            # For each error rate
            for error_rate in self.error_rates:
                logger.info(f"  Error rate: {error_rate}")
                
                # Create noisy backend
                noisy_backend = self.create_noisy_backend(self.backends[0], error_rate)
                
                # For each repetition
                for rep in range(self.num_repetitions):
                    # Create error model
                    error_model = DynamicSpinorErrorModel()
                    
                    # Create enhanced adaptive error mitigation
                    mitigation = EnhancedAdaptiveErrorMitigation(
                        error_model=error_model,
                        use_zero_noise_extrapolation=True,
                        use_probabilistic_error_cancellation=True,
                        use_measurement_mitigation=True,
                        use_real_time_tracking=True,
                        use_mid_circuit_correction=True
                    )
                    
                    # Execute circuit without mitigation
                    start_time = time.time()
                    job = noisy_backend.run(circuit, shots=1024)
                    unmitigated_result = job.result()
                    unmitigated_time = time.time() - start_time
                    
                    # Analyze circuit error profile
                    error_profile = mitigation.analyze_circuit_error_profile(circuit)
                    
                    # Test each mitigation strategy
                    strategy_results = {}
                    
                    for strategy in self.mitigation_strategies:
                        # Apply mitigation strategy
                        start_time = time.time()
                        mitigation_results = mitigation.apply_mitigation_strategy(
                            circuit, [strategy], noisy_backend
                        )
                        mitigation_time = time.time() - start_time
                        
                        # Execute mitigated circuit
                        start_time = time.time()
                        job = noisy_backend.run(mitigation_results['mitigated_circuit'], shots=1024)
                        mitigated_result = job.result()
                        execution_time = time.time() - start_time
                        
                        # Evaluate effectiveness
                        effectiveness = mitigation.evaluate_mitigation_effectiveness(
                            circuit, mitigation_results
                        )
                        
                        # Store strategy results
                        strategy_results[strategy] = {
                            'mitigation_time': mitigation_time,
                            'execution_time': execution_time,
                            'effectiveness': effectiveness
                        }
                    
                    # Store results
                    result = {
                        'circuit_size': size,
                        'error_rate': error_rate,
                        'repetition': rep,
                        'unmitigated_execution_time': unmitigated_time,
                        'error_profile': error_profile,
                        'strategy_results': strategy_results
                    }
                    
                    results.append(result)
                    
                    # Log results
                    logger.info(f"    Rep {rep+1}/{self.num_repetitions}:")
                    for strategy, strategy_result in strategy_results.items():
                        effectiveness = strategy_result['effectiveness']
                        if 'fidelity' in effectiveness:
                            logger.info(f"      {strategy}: Fidelity = {effectiveness['fidelity']:.4f}")
        
        # Store results
        self.results['strategy_comparison'] = results
        
        # Save results to file
        self._save_results('strategy_comparison')
        
        logger.info("Error mitigation strategy comparison benchmark completed")
        
        return results
    
    def run_all_benchmarks(self):
        """
        Run all error mitigation benchmarks.
        """
        logger.info("Running all error mitigation benchmarks")
        
        # Run benchmarks
        self.benchmark_dynamic_error()
        self.benchmark_real_time_tracking()
        self.benchmark_mid_circuit_correction()
        self.benchmark_strategy_comparison()
        
        # Generate summary report
        self.generate_summary_report()
        
        logger.info("All error mitigation benchmarks completed")
    
    def generate_summary_report(self):
        """
        Generate summary report of benchmark results.
        """
        logger.info("Generating summary report")
        
        # Create summary dictionary
        summary = {
            'timestamp': datetime.datetime.now().isoformat(),
            'dynamic_error': self._summarize_results('dynamic_error'),
            'real_time_tracking': self._summarize_results('real_time_tracking'),
            'mid_circuit_correction': self._summarize_results('mid_circuit_correction'),
            'strategy_comparison': self._summarize_strategy_comparison()
        }
        
        # Save summary to file
        summary_path = os.path.join(self.save_dir, 'error_mitigation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved to {summary_path}")
        
        # Generate visualizations
        self._generate_visualizations()
        
        return summary
    
    def _summarize_results(self, benchmark_type: str) -> Dict[str, Any]:
        """
        Summarize results for a specific benchmark type.
        
        Args:
            benchmark_type: Type of benchmark to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        results = self.results.get(benchmark_type, [])
        
        if not results:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Group by circuit size and error rate
        grouped = df.groupby(['circuit_size', 'error_rate'])
        
        # Calculate summary statistics
        if benchmark_type == 'dynamic_error':
            summary = grouped.agg({
                'fidelity_improvement_percentage': ['mean', 'std', 'min', 'max'],
                'dynamic_characterization_time': ['mean', 'std', 'min', 'max'],
                'dynamic_execution_time': ['mean', 'std', 'min', 'max']
            }).reset_index()
        elif benchmark_type == 'real_time_tracking':
            summary = grouped.agg({
                'total_variation_distance': ['mean', 'std', 'min', 'max'],
                'num_compensation_operations': ['mean', 'std', 'min', 'max'],
                'circuit_size_increase': ['mean', 'std', 'min', 'max'],
                'tracking_time': ['mean', 'std', 'min', 'max']
            }).reset_index()
        elif benchmark_type == 'mid_circuit_correction':
            summary = grouped.agg({
                'success_improvement_percentage': ['mean', 'std', 'min', 'max'],
                'qubit_overhead': ['mean', 'std', 'min', 'max'],
                'gate_overhead': ['mean', 'std', 'min', 'max'],
                'correction_time': ['mean', 'std', 'min', 'max']
            }).reset_index()
        
        # Convert to dictionary for JSON serialization
        summary_dict = summary.to_dict(orient='records')
        
        return summary_dict
    
    def _summarize_strategy_comparison(self) -> Dict[str, Any]:
        """
        Summarize strategy comparison results.
        
        Returns:
            Dictionary with summary statistics
        """
        results = self.results.get('strategy_comparison', [])
        
        if not results:
            return {}
        
        # Extract strategy effectiveness
        strategy_effectiveness = {}
        
        for result in results:
            circuit_size = result['circuit_size']
            error_rate = result['error_rate']
            
            key = (circuit_size, error_rate)
            if key not in strategy_effectiveness:
                strategy_effectiveness[key] = {strategy: [] for strategy in self.mitigation_strategies}
            
            for strategy, strategy_result in result['strategy_results'].items():
                effectiveness = strategy_result['effectiveness']
                if 'fidelity' in effectiveness:
                    strategy_effectiveness[key][strategy].append(effectiveness['fidelity'])
        
        # Calculate summary statistics
        summary = []
        
        for (circuit_size, error_rate), strategies in strategy_effectiveness.items():
            entry = {
                'circuit_size': circuit_size,
                'error_rate': error_rate
            }
            
            for strategy, fidelities in strategies.items():
                if fidelities:
                    entry[f'{strategy}_mean_fidelity'] = np.mean(fidelities)
                    entry[f'{strategy}_std_fidelity'] = np.std(fidelities)
                    entry[f'{strategy}_min_fidelity'] = np.min(fidelities)
                    entry[f'{strategy}_max_fidelity'] = np.max(fidelities)
            
            summary.append(entry)
        
        return summary
    
    def _generate_visualizations(self):
        """
        Generate visualizations of benchmark results.
        """
        logger.info("Generating visualizations")
        
        # Create visualizations directory
        viz_dir = os.path.join(self.save_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate visualizations for each benchmark type
        for benchmark_type in ['dynamic_error', 'real_time_tracking', 'mid_circuit_correction']:
            if not self.results.get(benchmark_type):
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(self.results[benchmark_type])
            
            # Create visualizations
            if benchmark_type == 'dynamic_error':
                self._create_fidelity_improvement_plot(df, benchmark_type, viz_dir)
            elif benchmark_type == 'real_time_tracking':
                self._create_tvd_plot(df, benchmark_type, viz_dir)
            elif benchmark_type == 'mid_circuit_correction':
                self._create_success_improvement_plot(df, benchmark_type, viz_dir)
        
        # Create strategy comparison visualization
        if self.results.get('strategy_comparison'):
            self._create_strategy_comparison_plot(viz_dir)
        
        logger.info(f"Visualizations saved to {viz_dir}")
    
    def _create_fidelity_improvement_plot(self, df: pd.DataFrame, benchmark_type: str, viz_dir: str):
        """
        Create fidelity improvement plot.
        
        Args:
            df: DataFrame with benchmark results
            benchmark_type: Type of benchmark
            viz_dir: Directory to save visualization
        """
        plt.figure(figsize=(10, 6))
        
        # Create grouped bar plot
        sns.barplot(x='circuit_size', y='fidelity_improvement_percentage', hue='error_rate', data=df)
        
        plt.title('Fidelity Improvement by Circuit Size and Error Rate')
        plt.xlabel('Circuit Size (qubits)')
        plt.ylabel('Fidelity Improvement (%)')
        plt.legend(title='Error Rate')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(viz_dir, f'{benchmark_type}_fidelity_improvement.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_tvd_plot(self, df: pd.DataFrame, benchmark_type: str, viz_dir: str):
        """
        Create total variation distance plot.
        
        Args:
            df: DataFrame with benchmark results
            benchmark_type: Type of benchmark
            viz_dir: Directory to save visualization
        """
        plt.figure(figsize=(10, 6))
        
        # Create grouped bar plot
        sns.barplot(x='circuit_size', y='total_variation_distance', hue='error_rate', data=df)
        
        plt.title('Total Variation Distance by Circuit Size and Error Rate')
        plt.xlabel('Circuit Size (qubits)')
        plt.ylabel('Total Variation Distance')
        plt.legend(title='Error Rate')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(viz_dir, f'{benchmark_type}_tvd.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_success_improvement_plot(self, df: pd.DataFrame, benchmark_type: str, viz_dir: str):
        """
        Create success improvement plot.
        
        Args:
            df: DataFrame with benchmark results
            benchmark_type: Type of benchmark
            viz_dir: Directory to save visualization
        """
        plt.figure(figsize=(10, 6))
        
        # Create grouped bar plot
        sns.barplot(x='circuit_size', y='success_improvement_percentage', hue='error_rate', data=df)
        
        plt.title('Success Probability Improvement by Circuit Size and Error Rate')
        plt.xlabel('Circuit Size (qubits)')
        plt.ylabel('Success Improvement (%)')
        plt.legend(title='Error Rate')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(viz_dir, f'{benchmark_type}_success_improvement.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_strategy_comparison_plot(self, viz_dir: str):
        """
        Create strategy comparison plot.
        
        Args:
            viz_dir: Directory to save visualization
        """
        # Extract strategy effectiveness
        strategy_effectiveness = {}
        
        for result in self.results['strategy_comparison']:
            circuit_size = result['circuit_size']
            error_rate = result['error_rate']
            
            key = (circuit_size, error_rate)
            if key not in strategy_effectiveness:
                strategy_effectiveness[key] = {strategy: [] for strategy in self.mitigation_strategies}
            
            for strategy, strategy_result in result['strategy_results'].items():
                effectiveness = strategy_result['effectiveness']
                if 'fidelity' in effectiveness:
                    strategy_effectiveness[key][strategy].append(effectiveness['fidelity'])
        
        # Create DataFrame for plotting
        plot_data = []
        
        for (circuit_size, error_rate), strategies in strategy_effectiveness.items():
            for strategy, fidelities in strategies.items():
                if fidelities:
                    for fidelity in fidelities:
                        plot_data.append({
                            'circuit_size': circuit_size,
                            'error_rate': error_rate,
                            'strategy': strategy,
                            'fidelity': fidelity
                        })
        
        df = pd.DataFrame(plot_data)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar plot
        g = sns.catplot(
            data=df, kind="bar",
            x="circuit_size", y="fidelity", hue="strategy",
            col="error_rate", col_wrap=2, height=4, aspect=1.5
        )
        
        g.set_axis_labels("Circuit Size (qubits)", "Fidelity")
        g.set_titles("Error Rate: {col_name}")
        g.fig.suptitle('Mitigation Strategy Comparison', y=1.02, fontsize=16)
        g.fig.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(viz_dir, 'strategy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self, benchmark_type: str):
        """
        Save benchmark results to file.
        
        Args:
            benchmark_type: Type of benchmark
        """
        # Create results file path
        file_path = os.path.join(self.save_dir, f'{benchmark_type}_results.json')
        
        # Save results to file
        with open(file_path, 'w') as f:
            json.dump(self.results[benchmark_type], f, indent=2)
        
        logger.info(f"Results saved to {file_path}")


class HybridAlgorithmBenchmark:
    """
    Benchmarks quantum-classical hybrid algorithms.
    
    This class provides tools for benchmarking the performance of quantum-classical
    hybrid algorithms, including natural gradient optimization, quantum neural
    networks, and distributed quantum-classical computation.
    """
    
    def __init__(self, 
                 backends: List[Backend],
                 save_dir: str = './benchmark_results',
                 num_repetitions: int = 5,
                 problem_sizes: List[int] = [2, 4, 6, 8],
                 optimization_methods: List[str] = ['COBYLA', 'NATURAL_GRADIENT', 'SPSA'],
                 max_iterations: int = 50):
        """
        Initialize the Hybrid Algorithm Benchmark.
        
        Args:
            backends: List of quantum backends to benchmark on
            save_dir: Directory to save benchmark results
            num_repetitions: Number of repetitions for each benchmark
            problem_sizes: List of problem sizes to benchmark
            optimization_methods: List of optimization methods to benchmark
            max_iterations: Maximum number of optimization iterations
        """
        self.backends = backends
        self.save_dir = save_dir
        self.num_repetitions = num_repetitions
        self.problem_sizes = problem_sizes
        self.optimization_methods = optimization_methods
        self.max_iterations = max_iterations
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'natural_gradient': [],
            'quantum_neural_network': [],
            'distributed_optimization': []
        }
        
        logger.info(f"Initialized Hybrid Algorithm Benchmark")
        logger.info(f"  Number of backends: {len(backends)}")
        logger.info(f"  Number of repetitions: {num_repetitions}")
        logger.info(f"  Problem sizes: {problem_sizes}")
        logger.info(f"  Optimization methods: {optimization_methods}")
        logger.info(f"  Max iterations: {max_iterations}")
    
    def generate_random_hamiltonian(self, num_qubits: int) -> PauliSumOp:
        """
        Generate a random Hamiltonian.
        
        Args:
            num_qubits: Number of qubits
            
        Returns:
            Random Hamiltonian operator
        """
        from qiskit.opflow import X, Y, Z, I
        
        # Generate random Pauli terms
        num_terms = min(10, 4**num_qubits)  # Limit number of terms for large systems
        hamiltonian = 0
        
        for _ in range(num_terms):
            # Generate random Pauli string
            pauli_string = ''
            for _ in range(num_qubits):
                pauli_string += np.random.choice(['I', 'X', 'Y', 'Z'])
            
            # Generate random coefficient
            coeff = np.random.uniform(-1, 1)
            
            # Create Pauli operator
            op = I
            for i, p in enumerate(pauli_string):
                if p == 'I':
                    new_op = I
                elif p == 'X':
                    new_op = X
                elif p == 'Y':
                    new_op = Y
                else:  # p == 'Z'
                    new_op = Z
                
                if i == 0:
                    op = new_op
                else:
                    op = op ^ new_op
            
            # Add to Hamiltonian
            hamiltonian += coeff * op
        
        return hamiltonian
    
    def generate_ansatz_circuit(self, num_qubits: int, depth: int = 2) -> QuantumCircuit:
        """
        Generate a parameterized ansatz circuit.
        
        Args:
            num_qubits: Number of qubits
            depth: Circuit depth
            
        Returns:
            Parameterized ansatz circuit
        """
        circuit = QuantumCircuit(num_qubits)
        
        # Add parameterized layers
        for d in range(depth):
            # Add single-qubit rotations
            for i in range(num_qubits):
                circuit.rx(qiskit.circuit.Parameter(f'theta_{d}_{i}_x'), i)
                circuit.ry(qiskit.circuit.Parameter(f'theta_{d}_{i}_y'), i)
                circuit.rz(qiskit.circuit.Parameter(f'theta_{d}_{i}_z'), i)
            
            # Add entanglement
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
            
            # Add final entanglement between last and first qubit
            if num_qubits > 2:
                circuit.cx(num_qubits - 1, 0)
        
        return circuit
    
    def benchmark_natural_gradient(self):
        """
        Benchmark natural gradient optimization.
        """
        logger.info("Starting natural gradient optimization benchmark")
        
        results = []
        
        # For each problem size
        for size in self.problem_sizes:
            logger.info(f"Benchmarking problem with {size} qubits")
            
            # Generate random Hamiltonian
            hamiltonian = self.generate_random_hamiltonian(size)
            
            # Generate ansatz circuit
            ansatz = self.generate_ansatz_circuit(size)
            
            # For each optimization method
            for method in self.optimization_methods:
                logger.info(f"  Optimization method: {method}")
                
                # For each repetition
                for rep in range(self.num_repetitions):
                    # Create VQE
                    if method == 'NATURAL_GRADIENT':
                        vqe = NaturalGradientVQE(
                            backend=self.backends[0],
                            optimizer_method=method,
                            max_iterations=self.max_iterations,
                            use_spinor_reduction=True,
                            use_phase_synchronization=True,
                            use_prime_indexing=True,
                            natural_gradient_reg=0.01,
                            qfim_approximation='diag',
                            learning_rate=0.1,
                            adaptive_learning_rate=True
                        )
                    else:
                        vqe = TibedoEnhancedVQE(
                            backend=self.backends[0],
                            optimizer_method=method,
                            max_iterations=self.max_iterations,
                            use_spinor_reduction=True,
                            use_phase_synchronization=True,
                            use_prime_indexing=True
                        )
                    
                    # Optimize parameters
                    start_time = time.time()
                    optimization_result = vqe.optimize_parameters(ansatz, hamiltonian)
                    optimization_time = time.time() - start_time
                    
                    # Extract results
                    optimal_value = optimization_result['optimal_value']
                    num_iterations = optimization_result['num_iterations']
                    success = optimization_result.get('success', False)
                    
                    # Calculate convergence metrics
                    if 'optimization_history' in optimization_result:
                        history = optimization_result['optimization_history']
                        energies = [h[1] for h in history]
                        
                        # Calculate convergence rate
                        if len(energies) > 2:
                            energy_diffs = np.abs(np.diff(energies))
                            convergence_rates = energy_diffs[1:] / energy_diffs[:-1]
                            mean_convergence_rate = np.mean(convergence_rates)
                        else:
                            mean_convergence_rate = None
                    else:
                        mean_convergence_rate = None
                    
                    # Store results
                    result = {
                        'problem_size': size,
                        'optimization_method': method,
                        'repetition': rep,
                        'optimal_value': float(optimal_value),  # Convert to float for JSON serialization
                        'num_iterations': num_iterations,
                        'optimization_time': optimization_time,
                        'mean_convergence_rate': float(mean_convergence_rate) if mean_convergence_rate is not None else None,
                        'success': success
                    }
                    
                    results.append(result)
                    
                    logger.info(f"    Rep {rep+1}/{self.num_repetitions}: "
                               f"Optimal value: {optimal_value:.6f}, "
                               f"Iterations: {num_iterations}, "
                               f"Time: {optimization_time:.2f}s")
        
        # Store results
        self.results['natural_gradient'] = results
        
        # Save results to file
        self._save_results('natural_gradient')
        
        logger.info("Natural gradient optimization benchmark completed")
        
        return results
    
    def benchmark_quantum_neural_network(self):
        """
        Benchmark quantum neural network.
        """
        logger.info("Starting quantum neural network benchmark")
        
        results = []
        
        # For each problem size
        for size in self.problem_sizes:
            logger.info(f"Benchmarking problem with {size} qubits")
            
            # Generate synthetic data
            np.random.seed(42)
            num_samples = 20
            X = np.random.rand(num_samples, size)
            y = (np.sum(X, axis=1) > size/2).astype(float)  # Simple classification task
            
            # Split into training and test sets
            train_size = int(0.7 * num_samples)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # For each feature map type
            for feature_map_type in ['SpinorFeatureMap', 'ZZFeatureMap']:
                logger.info(f"  Feature map type: {feature_map_type}")
                
                # For each repetition
                for rep in range(self.num_repetitions):
                    # Create quantum neural network
                    qnn = SpinorQuantumNN(
                        backend=self.backends[0],
                        optimizer_method='NATURAL_GRADIENT',
                        max_iterations=self.max_iterations,
                        use_spinor_encoding=True,
                        use_phase_synchronization=True,
                        feature_map_type=feature_map_type,
                        variational_form_type='SpinorCircuit',
                        learning_rate=0.1,
                        adaptive_learning_rate=True,
                        use_quantum_backprop=True
                    )
                    
                    # Generate feature map
                    feature_map = qnn.generate_quantum_feature_map(X_train)
                    
                    # Create variational circuit
                    var_circuit = qnn.create_variational_circuit(feature_map)
                    
                    # Train model
                    start_time = time.time()
                    training_result = qnn.train_quantum_model(var_circuit, X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Evaluate model
                    start_time = time.time()
                    performance = qnn.evaluate_model_performance(training_result, X_test, y_test)
                    evaluation_time = time.time() - start_time
                    
                    # Extract results
                    final_loss = training_result['final_loss']
                    num_iterations = training_result['num_iterations']
                    success = training_result.get('success', False)
                    
                    # Extract performance metrics
                    mse = performance.get('mse')
                    accuracy = performance.get('accuracy')
                    
                    # Store results
                    result = {
                        'problem_size': size,
                        'feature_map_type': feature_map_type,
                        'repetition': rep,
                        'final_loss': float(final_loss),
                        'num_iterations': num_iterations,
                        'training_time': training_time,
                        'evaluation_time': evaluation_time,
                        'mse': float(mse) if mse is not None else None,
                        'accuracy': float(accuracy) if accuracy is not None else None,
                        'success': success
                    }
                    
                    results.append(result)
                    
                    logger.info(f"    Rep {rep+1}/{self.num_repetitions}: "
                               f"Final loss: {final_loss:.6f}, "
                               f"Accuracy: {accuracy:.4f if accuracy is not None else None}, "
                               f"Time: {training_time:.2f}s")
        
        # Store results
        self.results['quantum_neural_network'] = results
        
        # Save results to file
        self._save_results('quantum_neural_network')
        
        logger.info("Quantum neural network benchmark completed")
        
        return results
    
    def benchmark_distributed_optimization(self):
        """
        Benchmark distributed quantum-classical optimization.
        """
        logger.info("Starting distributed optimization benchmark")
        
        results = []
        
        # For each problem size
        for size in self.problem_sizes:
            logger.info(f"Benchmarking problem with {size} qubits")
            
            # Generate random Hamiltonian
            hamiltonian = self.generate_random_hamiltonian(size)
            
            # Generate ansatz circuit
            ansatz = self.generate_ansatz_circuit(size)
            
            # For each distribution strategy
            for strategy in ['parameter_split', 'data_split', 'hybrid']:
                logger.info(f"  Distribution strategy: {strategy}")
                
                # For each number of workers
                for num_workers in [1, 2, 4]:
                    # Skip if more workers than backends
                    if num_workers > len(self.backends):
                        continue
                        
                    logger.info(f"    Number of workers: {num_workers}")
                    
                    # For each repetition
                    for rep in range(self.num_repetitions):
                        # Create distributed optimizer
                        optimizer = DistributedQuantumOptimizer(
                            backends=self.backends[:num_workers],
                            optimizer_method='COBYLA',
                            max_iterations=self.max_iterations,
                            use_spinor_reduction=True,
                            use_phase_synchronization=True,
                            use_prime_indexing=True,
                            distribution_strategy=strategy,
                            num_workers=num_workers,
                            synchronization_frequency=5
                        )
                        
                        # Optimize VQE
                        start_time = time.time()
                        optimization_result = optimizer.optimize_vqe(ansatz, hamiltonian)
                        optimization_time = time.time() - start_time
                        
                        # Extract results
                        optimal_value = optimization_result['optimal_value']
                        num_iterations = optimization_result['num_iterations']
                        success = optimization_result.get('success', False)
                        
                        # Store results
                        result = {
                            'problem_size': size,
                            'distribution_strategy': strategy,
                            'num_workers': num_workers,
                            'repetition': rep,
                            'optimal_value': float(optimal_value),
                            'num_iterations': num_iterations,
                            'optimization_time': optimization_time,
                            'success': success
                        }
                        
                        results.append(result)
                        
                        logger.info(f"      Rep {rep+1}/{self.num_repetitions}: "
                                   f"Optimal value: {optimal_value:.6f}, "
                                   f"Iterations: {num_iterations}, "
                                   f"Time: {optimization_time:.2f}s")
        
        # Store results
        self.results['distributed_optimization'] = results
        
        # Save results to file
        self._save_results('distributed_optimization')
        
        logger.info("Distributed optimization benchmark completed")
        
        return results
    
    def run_all_benchmarks(self):
        """
        Run all hybrid algorithm benchmarks.
        """
        logger.info("Running all hybrid algorithm benchmarks")
        
        # Run benchmarks
        self.benchmark_natural_gradient()
        self.benchmark_quantum_neural_network()
        self.benchmark_distributed_optimization()
        
        # Generate summary report
        self.generate_summary_report()
        
        logger.info("All hybrid algorithm benchmarks completed")
    
    def generate_summary_report(self):
        """
        Generate summary report of benchmark results.
        """
        logger.info("Generating summary report")
        
        # Create summary dictionary
        summary = {
            'timestamp': datetime.datetime.now().isoformat(),
            'natural_gradient': self._summarize_results('natural_gradient'),
            'quantum_neural_network': self._summarize_results('quantum_neural_network'),
            'distributed_optimization': self._summarize_results('distributed_optimization')
        }
        
        # Save summary to file
        summary_path = os.path.join(self.save_dir, 'hybrid_algorithm_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved to {summary_path}")
        
        # Generate visualizations
        self._generate_visualizations()
        
        return summary
    
    def _summarize_results(self, benchmark_type: str) -> Dict[str, Any]:
        """
        Summarize results for a specific benchmark type.
        
        Args:
            benchmark_type: Type of benchmark to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        results = self.results.get(benchmark_type, [])
        
        if not results:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Group by relevant columns
        if benchmark_type == 'natural_gradient':
            grouped = df.groupby(['problem_size', 'optimization_method'])
            
            summary = grouped.agg({
                'optimal_value': ['mean', 'std', 'min', 'max'],
                'num_iterations': ['mean', 'std', 'min', 'max'],
                'optimization_time': ['mean', 'std', 'min', 'max'],
                'mean_convergence_rate': ['mean', 'std', 'min', 'max'],
                'success': ['mean']
            }).reset_index()
        elif benchmark_type == 'quantum_neural_network':
            grouped = df.groupby(['problem_size', 'feature_map_type'])
            
            summary = grouped.agg({
                'final_loss': ['mean', 'std', 'min', 'max'],
                'num_iterations': ['mean', 'std', 'min', 'max'],
                'training_time': ['mean', 'std', 'min', 'max'],
                'mse': ['mean', 'std', 'min', 'max'],
                'accuracy': ['mean', 'std', 'min', 'max'],
                'success': ['mean']
            }).reset_index()
        elif benchmark_type == 'distributed_optimization':
            grouped = df.groupby(['problem_size', 'distribution_strategy', 'num_workers'])
            
            summary = grouped.agg({
                'optimal_value': ['mean', 'std', 'min', 'max'],
                'num_iterations': ['mean', 'std', 'min', 'max'],
                'optimization_time': ['mean', 'std', 'min', 'max'],
                'success': ['mean']
            }).reset_index()
        
        # Convert to dictionary for JSON serialization
        summary_dict = summary.to_dict(orient='records')
        
        return summary_dict
    
    def _generate_visualizations(self):
        """
        Generate visualizations of benchmark results.
        """
        logger.info("Generating visualizations")
        
        # Create visualizations directory
        viz_dir = os.path.join(self.save_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate visualizations for each benchmark type
        if self.results.get('natural_gradient'):
            self._create_natural_gradient_plots(viz_dir)
        
        if self.results.get('quantum_neural_network'):
            self._create_quantum_neural_network_plots(viz_dir)
        
        if self.results.get('distributed_optimization'):
            self._create_distributed_optimization_plots(viz_dir)
        
        logger.info(f"Visualizations saved to {viz_dir}")
    
    def _create_natural_gradient_plots(self, viz_dir: str):
        """
        Create natural gradient optimization plots.
        
        Args:
            viz_dir: Directory to save visualizations
        """
        df = pd.DataFrame(self.results['natural_gradient'])
        
        # Plot 1: Convergence comparison
        plt.figure(figsize=(10, 6))
        
        # Create grouped bar plot
        sns.barplot(x='problem_size', y='num_iterations', hue='optimization_method', data=df)
        
        plt.title('Convergence Comparison by Problem Size and Optimization Method')
        plt.xlabel('Problem Size (qubits)')
        plt.ylabel('Number of Iterations')
        plt.legend(title='Optimization Method')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(viz_dir, 'natural_gradient_convergence.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Optimization time comparison
        plt.figure(figsize=(10, 6))
        
        # Create grouped bar plot
        sns.barplot(x='problem_size', y='optimization_time', hue='optimization_method', data=df)
        
        plt.title('Optimization Time Comparison by Problem Size and Optimization Method')
        plt.xlabel('Problem Size (qubits)')
        plt.ylabel('Optimization Time (s)')
        plt.legend(title='Optimization Method')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(viz_dir, 'natural_gradient_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Convergence rate comparison
        plt.figure(figsize=(10, 6))
        
        # Filter out None values
        df_filtered = df[df['mean_convergence_rate'].notna()]
        
        if not df_filtered.empty:
            # Create grouped bar plot
            sns.barplot(x='problem_size', y='mean_convergence_rate', hue='optimization_method', data=df_filtered)
            
            plt.title('Convergence Rate Comparison by Problem Size and Optimization Method')
            plt.xlabel('Problem Size (qubits)')
            plt.ylabel('Mean Convergence Rate')
            plt.legend(title='Optimization Method')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig(os.path.join(viz_dir, 'natural_gradient_convergence_rate.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_quantum_neural_network_plots(self, viz_dir: str):
        """
        Create quantum neural network plots.
        
        Args:
            viz_dir: Directory to save visualizations
        """
        df = pd.DataFrame(self.results['quantum_neural_network'])
        
        # Plot 1: Accuracy comparison
        plt.figure(figsize=(10, 6))
        
        # Create grouped bar plot
        sns.barplot(x='problem_size', y='accuracy', hue='feature_map_type', data=df)
        
        plt.title('Accuracy Comparison by Problem Size and Feature Map Type')
        plt.xlabel('Problem Size (qubits)')
        plt.ylabel('Accuracy')
        plt.legend(title='Feature Map Type')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(viz_dir, 'quantum_neural_network_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Training time comparison
        plt.figure(figsize=(10, 6))
        
        # Create grouped bar plot
        sns.barplot(x='problem_size', y='training_time', hue='feature_map_type', data=df)
        
        plt.title('Training Time Comparison by Problem Size and Feature Map Type')
        plt.xlabel('Problem Size (qubits)')
        plt.ylabel('Training Time (s)')
        plt.legend(title='Feature Map Type')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(viz_dir, 'quantum_neural_network_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Loss comparison
        plt.figure(figsize=(10, 6))
        
        # Create grouped bar plot
        sns.barplot(x='problem_size', y='final_loss', hue='feature_map_type', data=df)
        
        plt.title('Final Loss Comparison by Problem Size and Feature Map Type')
        plt.xlabel('Problem Size (qubits)')
        plt.ylabel('Final Loss')
        plt.legend(title='Feature Map Type')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(viz_dir, 'quantum_neural_network_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_distributed_optimization_plots(self, viz_dir: str):
        """
        Create distributed optimization plots.
        
        Args:
            viz_dir: Directory to save visualizations
        """
        df = pd.DataFrame(self.results['distributed_optimization'])
        
        # Plot 1: Optimization time comparison
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar plot
        g = sns.catplot(
            data=df, kind="bar",
            x="problem_size", y="optimization_time", hue="num_workers",
            col="distribution_strategy", col_wrap=3, height=4, aspect=1.2
        )
        
        g.set_axis_labels("Problem Size (qubits)", "Optimization Time (s)")
        g.set_titles("Strategy: {col_name}")
        g.fig.suptitle('Optimization Time by Distribution Strategy', y=1.02, fontsize=16)
        g.fig.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(viz_dir, 'distributed_optimization_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Speedup comparison
        # Calculate speedup relative to single worker
        speedup_data = []
        
        for problem_size in df['problem_size'].unique():
            for strategy in df['distribution_strategy'].unique():
                # Get single worker time
                single_worker_times = df[(df['problem_size'] == problem_size) & 
                                       (df['distribution_strategy'] == strategy) & 
                                       (df['num_workers'] == 1)]['optimization_time']
                
                if not single_worker_times.empty:
                    single_worker_time = single_worker_times.mean()
                    
                    # Calculate speedup for each number of workers
                    for num_workers in df['num_workers'].unique():
                        if num_workers > 1:
                            worker_times = df[(df['problem_size'] == problem_size) & 
                                            (df['distribution_strategy'] == strategy) & 
                                            (df['num_workers'] == num_workers)]['optimization_time']
                            
                            if not worker_times.empty:
                                worker_time = worker_times.mean()
                                speedup = single_worker_time / worker_time
                                
                                speedup_data.append({
                                    'problem_size': problem_size,
                                    'distribution_strategy': strategy,
                                    'num_workers': num_workers,
                                    'speedup': speedup
                                })
        
        if speedup_data:
            speedup_df = pd.DataFrame(speedup_data)
            
            plt.figure(figsize=(12, 8))
            
            # Create grouped bar plot
            g = sns.catplot(
                data=speedup_df, kind="bar",
                x="problem_size", y="speedup", hue="num_workers",
                col="distribution_strategy", col_wrap=3, height=4, aspect=1.2
            )
            
            g.set_axis_labels("Problem Size (qubits)", "Speedup")
            g.set_titles("Strategy: {col_name}")
            g.fig.suptitle('Speedup by Distribution Strategy', y=1.02, fontsize=16)
            g.fig.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(viz_dir, 'distributed_optimization_speedup.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _save_results(self, benchmark_type: str):
        """
        Save benchmark results to file.
        
        Args:
            benchmark_type: Type of benchmark
        """
        # Create results file path
        file_path = os.path.join(self.save_dir, f'{benchmark_type}_results.json')
        
        # Save results to file
        with open(file_path, 'w') as f:
            json.dump(self.results[benchmark_type], f, indent=2)
        
        logger.info(f"Results saved to {file_path}")


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for all quantum components.
    
    This class provides a unified interface for benchmarking all quantum components
    of the TIBEDO Framework, including circuit optimization, error mitigation, and
    hybrid algorithms.
    """
    
    def __init__(self, 
                 backends: Optional[List[Backend]] = None,
                 save_dir: str = './benchmark_results',
                 num_repetitions: int = 5,
                 components: List[str] = ['circuit_optimization', 'error_mitigation', 'hybrid_algorithms']):
        """
        Initialize the Benchmark Suite.
        
        Args:
            backends: List of quantum backends to benchmark on (if None, use Aer simulators)
            save_dir: Directory to save benchmark results
            num_repetitions: Number of repetitions for each benchmark
            components: List of components to benchmark
        """
        # Set up backends
        if backends is None:
            backends = [Aer.get_backend('statevector_simulator'), Aer.get_backend('qasm_simulator')]
        
        self.backends = backends
        self.save_dir = save_dir
        self.num_repetitions = num_repetitions
        self.components = components
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create component benchmarks
        self.benchmarks = {}
        
        if 'circuit_optimization' in components:
            self.benchmarks['circuit_optimization'] = CircuitOptimizationBenchmark(
                backends=backends,
                save_dir=os.path.join(save_dir, 'circuit_optimization'),
                num_repetitions=num_repetitions
            )
        
        if 'error_mitigation' in components:
            self.benchmarks['error_mitigation'] = ErrorMitigationBenchmark(
                backends=backends,
                save_dir=os.path.join(save_dir, 'error_mitigation'),
                num_repetitions=num_repetitions
            )
        
        if 'hybrid_algorithms' in components:
            self.benchmarks['hybrid_algorithms'] = HybridAlgorithmBenchmark(
                backends=backends,
                save_dir=os.path.join(save_dir, 'hybrid_algorithms'),
                num_repetitions=num_repetitions
            )
        
        logger.info(f"Initialized Benchmark Suite")
        logger.info(f"  Number of backends: {len(backends)}")
        logger.info(f"  Number of repetitions: {num_repetitions}")
        logger.info(f"  Components: {components}")
    
    def run_all_benchmarks(self):
        """
        Run all benchmarks.
        """
        logger.info("Running all benchmarks")
        
        # Run benchmarks for each component
        for component, benchmark in self.benchmarks.items():
            logger.info(f"Running {component} benchmarks")
            benchmark.run_all_benchmarks()
        
        # Generate summary report
        self.generate_summary_report()
        
        logger.info("All benchmarks completed")
    
    def run_component_benchmark(self, component: str):
        """
        Run benchmarks for a specific component.
        
        Args:
            component: Component to benchmark
        """
        if component not in self.benchmarks:
            logger.error(f"Component {component} not found")
            return
        
        logger.info(f"Running {component} benchmarks")
        self.benchmarks[component].run_all_benchmarks()
        
        logger.info(f"{component} benchmarks completed")
    
    def generate_summary_report(self):
        """
        Generate summary report of all benchmark results.
        """
        logger.info("Generating summary report")
        
        # Create summary dictionary
        summary = {
            'timestamp': datetime.datetime.now().isoformat(),
            'components': self.components,
            'num_repetitions': self.num_repetitions,
            'backends': [backend.name() for backend in self.backends]
        }
        
        # Add component summaries
        for component, benchmark in self.benchmarks.items():
            if hasattr(benchmark, 'results'):
                component_summary = {}
                
                for benchmark_type, results in benchmark.results.items():
                    if results:
                        component_summary[benchmark_type] = {
                            'num_results': len(results),
                            'parameters': list(results[0].keys()) if results else []
                        }
                
                summary[component] = component_summary
        
        # Save summary to file
        summary_path = os.path.join(self.save_dir, 'benchmark_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved to {summary_path}")
        
        # Generate comparison visualizations
        self._generate_comparison_visualizations()
        
        return summary
    
    def _generate_comparison_visualizations(self):
        """
        Generate comparison visualizations across components.
        """
        logger.info("Generating comparison visualizations")
        
        # Create visualizations directory
        viz_dir = os.path.join(self.save_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate component comparison visualizations
        self._create_performance_comparison(viz_dir)
        
        logger.info(f"Comparison visualizations saved to {viz_dir}")
    
    def _create_performance_comparison(self, viz_dir: str):
        """
        Create performance comparison visualization.
        
        Args:
            viz_dir: Directory to save visualization
        """
        # Collect performance metrics
        performance_data = []
        
        # Circuit optimization performance
        if 'circuit_optimization' in self.benchmarks:
            benchmark = self.benchmarks['circuit_optimization']
            
            if 'circuit_compression' in benchmark.results:
                for result in benchmark.results['circuit_compression']:
                    performance_data.append({
                        'component': 'Circuit Optimization',
                        'metric': 'Depth Reduction (%)',
                        'value': result['depth_reduction_percentage'],
                        'circuit_size': result['circuit_size'],
                        'optimization_level': result['optimization_level']
                    })
        
        # Error mitigation performance
        if 'error_mitigation' in self.benchmarks:
            benchmark = self.benchmarks['error_mitigation']
            
            if 'dynamic_error' in benchmark.results:
                for result in benchmark.results['dynamic_error']:
                    performance_data.append({
                        'component': 'Error Mitigation',
                        'metric': 'Fidelity Improvement (%)',
                        'value': result['fidelity_improvement_percentage'],
                        'circuit_size': result['circuit_size'],
                        'error_rate': result['error_rate']
                    })
        
        # Hybrid algorithms performance
        if 'hybrid_algorithms' in self.benchmarks:
            benchmark = self.benchmarks['hybrid_algorithms']
            
            if 'natural_gradient' in benchmark.results:
                for result in benchmark.results['natural_gradient']:
                    if result['optimization_method'] == 'NATURAL_GRADIENT':
                        performance_data.append({
                            'component': 'Hybrid Algorithms',
                            'metric': 'Convergence Iterations',
                            'value': result['num_iterations'],
                            'problem_size': result['problem_size'],
                            'optimization_method': result['optimization_method']
                        })
        
        # Create DataFrame
        if performance_data:
            df = pd.DataFrame(performance_data)
            
            # Create comparison plot
            plt.figure(figsize=(15, 10))
            
            # Create subplot grid
            fig, axes = plt.subplots(3, 1, figsize=(15, 15))
            
            # Plot circuit optimization performance
            df_circuit = df[df['component'] == 'Circuit Optimization']
            if not df_circuit.empty:
                sns.barplot(x='circuit_size', y='value', hue='optimization_level', data=df_circuit, ax=axes[0])
                axes[0].set_title('Circuit Optimization: Depth Reduction by Circuit Size')
                axes[0].set_xlabel('Circuit Size (qubits)')
                axes[0].set_ylabel('Depth Reduction (%)')
                axes[0].legend(title='Optimization Level')
                axes[0].grid(True, alpha=0.3)
            
            # Plot error mitigation performance
            df_error = df[df['component'] == 'Error Mitigation']
            if not df_error.empty:
                sns.barplot(x='circuit_size', y='value', hue='error_rate', data=df_error, ax=axes[1])
                axes[1].set_title('Error Mitigation: Fidelity Improvement by Circuit Size')
                axes[1].set_xlabel('Circuit Size (qubits)')
                axes[1].set_ylabel('Fidelity Improvement (%)')
                axes[1].legend(title='Error Rate')
                axes[1].grid(True, alpha=0.3)
            
            # Plot hybrid algorithms performance
            df_hybrid = df[df['component'] == 'Hybrid Algorithms']
            if not df_hybrid.empty:
                sns.barplot(x='problem_size', y='value', hue='optimization_method', data=df_hybrid, ax=axes[2])
                axes[2].set_title('Hybrid Algorithms: Convergence Iterations by Problem Size')
                axes[2].set_xlabel('Problem Size (qubits)')
                axes[2].set_ylabel('Convergence Iterations')
                axes[2].legend(title='Optimization Method')
                axes[2].grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(viz_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()


# Example usage function
def run_benchmark_suite():
    """
    Run the benchmark suite with default settings.
    """
    # Create backends
    backends = [Aer.get_backend('statevector_simulator'), Aer.get_backend('qasm_simulator')]
    
    # Create benchmark suite
    suite = BenchmarkSuite(
        backends=backends,
        save_dir='./benchmark_results',
        num_repetitions=2,  # Use small number for demonstration
        components=['circuit_optimization', 'error_mitigation', 'hybrid_algorithms']
    )
    
    # Run all benchmarks
    suite.run_all_benchmarks()
    
    return suite


if __name__ == "__main__":
    run_benchmark_suite()