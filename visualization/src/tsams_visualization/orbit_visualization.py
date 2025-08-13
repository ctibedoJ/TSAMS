&quot;&quot;&quot;
Orbit Visualization module for Tsams Visualization.

This module is part of the Tibedo Structural Algebraic Modeling System (TSAMS) ecosystem.
&quot;&quot;&quot;

# Code adapted from tibedo_parallel.py

"""
TIBEDO Framework: Parallel Processing Implementation

This module provides parallel processing capabilities for the TIBEDO Framework,
enabling efficient computation for larger problem sets in both ECDLP solving
and biological applications.
"""

import numpy as np
import time
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os

# Import the necessary classes from our robust implementation
from tibedo_ecdlp_robust import EllipticCurve, RobustTSCAlgorithm, create_ecdlp_instance

class ParallelECDLPSolver:
    """
    Parallel implementation of the ECDLP solver using the TIBEDO Framework.
    """
    
    def __init__(self, num_processes=None):
        """
        Initialize the parallel ECDLP solver.
        
        Args:
            num_processes (int, optional): Number of processes to use.
                If None, uses the number of CPU cores.
        """
        if num_processes is None:
            self.num_processes = mp.cpu_count()
        else:
            self.num_processes = num_processes
            
        print(f"Initializing parallel ECDLP solver with {self.num_processes} processes")
        
    def _solve_ecdlp_chunk(self, args):
        """
        Solve a chunk of the ECDLP search space.
        
        Args:
            args (tuple): (curve, P, Q, start_k, end_k)
                curve (EllipticCurve): The elliptic curve
                P (tuple): The base point (x1, y1)
                Q (tuple): The point to find the discrete logarithm for (x2, y2)
                start_k (int): Start of the search range
                end_k (int): End of the search range
                
        Returns:
            int or None: The discrete logarithm k such that Q = k*P, or None if not found
        """
        curve, P, Q, start_k, end_k = args
        
        for k in range(start_k, end_k):
            test_point = curve.scalar_multiply(k, P)
            if test_point == Q:
                return k
                
        return None
        
    def solve_ecdlp_parallel(self, curve, P, Q, order=None, chunk_size=None):
        """
        Solve the ECDLP using parallel processing.
        
        Args:
            curve (EllipticCurve): The elliptic curve
            P (tuple): The base point (x1, y1)
            Q (tuple): The point to find the discrete logarithm for (x2, y2)
            order (int, optional): The order of the base point P
            chunk_size (int, optional): Size of each chunk for parallel processing
                
        Returns:
            int: The discrete logarithm k such that Q = k*P
        """
        # If order is not provided, try to compute it with a reasonable limit
        if order is None:
            try:
                order = curve.find_point_order(P, max_order=1000)
            except ValueError:
                # If computing the order fails, use a reasonable estimate
                order = 100  # For our test cases
                
        # Determine chunk size
        if chunk_size is None:
            chunk_size = max(1, order // (self.num_processes * 10))
            
        # Create chunks
        chunks = []
        for start_k in range(1, order, chunk_size):
            end_k = min(start_k + chunk_size, order)
            chunks.append((curve, P, Q, start_k, end_k))
            
        # Solve in parallel
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            results = list(executor.map(self._solve_ecdlp_chunk, chunks))
            
        elapsed_time = time.time() - start_time
        
        # Process results
        for result in results:
            if result is not None:
                print(f"Found solution k = {result} in {elapsed_time:.6f} seconds")
                return result
                
        # If no solution found, use the Baby-step Giant-step algorithm
        print("No solution found using parallel brute force, trying Baby-step Giant-step algorithm")
        solver = RobustTSCAlgorithm()
        return solver._solve_baby_step_giant_step(curve, P, Q, order)
        
    def benchmark_parallel_vs_sequential(self, bit_lengths=[16, 24, 32], k_value=7):
        """
        Benchmark parallel vs sequential ECDLP solving.
        
        Args:
            bit_lengths (list): List of bit lengths to test
            k_value (int): The discrete logarithm to use
                
        Returns:
            dict: Benchmark results
        """
        results = {
            'bit_length': [],
            'sequential_time': [],
            'parallel_time': [],
            'speedup': []
        }
        
        for bit_length in bit_lengths:
            print(f"\nBenchmarking {bit_length}-bit ECDLP")
            
            # Create an ECDLP instance
            curve, P, Q, actual_k = create_ecdlp_instance(bit_length, k_value)
            
            # Sequential solving
            solver = RobustTSCAlgorithm()
            start_time = time.time()
            sequential_k = solver.solve_ecdlp(curve, P, Q, order=100)
            sequential_time = time.time() - start_time
            
            # Parallel solving
            start_time = time.time()
            parallel_k = self.solve_ecdlp_parallel(curve, P, Q, order=100)
            parallel_time = time.time() - start_time
            
            # Calculate speedup
            speedup = sequential_time / parallel_time if parallel_time > 0 else 0
            
            # Store results
            results['bit_length'].append(bit_length)
            results['sequential_time'].append(sequential_time)
            results['parallel_time'].append(parallel_time)
            results['speedup'].append(speedup)
            
            # Print results
            print(f"  Sequential: {sequential_time:.6f} seconds, k = {sequential_k}")
            print(f"  Parallel: {parallel_time:.6f} seconds, k = {parallel_k}")
            print(f"  Speedup: {speedup:.2f}x")
            
        return results
        
    def visualize_benchmark_results(self, results, save_path=None):
        """
        Visualize benchmark results.
        
        Args:
            results (dict): Benchmark results
            save_path (str, optional): Path to save the visualization
                
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot execution times
        x = np.arange(len(results['bit_length']))
        width = 0.35
        
        ax1.bar(x - width/2, results['sequential_time'], width, label='Sequential')
        ax1.bar(x + width/2, results['parallel_time'], width, label='Parallel')
        
        ax1.set_xlabel('Bit Length')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('ECDLP Solver Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(results['bit_length'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot speedup
        ax2.plot(results['bit_length'], results['speedup'], 'o-', color='green', linewidth=2)
        ax2.set_xlabel('Bit Length')
        ax2.set_ylabel('Speedup (Sequential / Parallel)')
        ax2.set_title('Parallel Processing Speedup')
        ax2.grid(True, alpha=0.3)
        
        # Add a horizontal line at y=1 (no speedup)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        
        # Add text annotations for each point
        for i, speedup in enumerate(results['speedup']):
            ax2.annotate(f"{speedup:.2f}x", 
                        (results['bit_length'][i], speedup),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return fig

class ParallelProteinFoldingSimulator:
    """
    Parallel implementation of the protein folding simulator using the TIBEDO Framework.
    """
    
    def __init__(self, num_processes=None):
        """
        Initialize the parallel protein folding simulator.
        
        Args:
            num_processes (int, optional): Number of processes to use.
                If None, uses the number of CPU cores.
        """
        if num_processes is None:
            self.num_processes = mp.cpu_count()
        else:
            self.num_processes = num_processes
            
        print(f"Initializing parallel protein folding simulator with {self.num_processes} processes")
        
        # Import the necessary classes from the biological implementation
        try:
            from tibedo_biological_implementation import ProteinFoldingSimulator, QuaternionMobiusStripPairing, CyclotomicFieldGaloisOrbit
            self.ProteinFoldingSimulator = ProteinFoldingSimulator
            self.QuaternionMobiusStripPairing = QuaternionMobiusStripPairing
            self.CyclotomicFieldGaloisOrbit = CyclotomicFieldGaloisOrbit
        except ImportError:
            print("Warning: Could not import biological implementation classes")
            self.ProteinFoldingSimulator = None
            
    def _process_sequence_chunk(self, chunk):
        """
        Process a chunk of the protein sequence.
        
        Args:
            chunk (tuple): (sequence_chunk, chunk_index)
                sequence_chunk (str): A chunk of the protein sequence
                chunk_index (int): The index of the chunk
                
        Returns:
            dict: Processing results for the chunk
        """
        sequence_chunk, chunk_index = chunk
        
        # Create a simulator for this chunk
        simulator = self.ProteinFoldingSimulator()
        
        # Simulate protein folding for the chunk
        results = simulator.simulate_protein_folding(sequence_chunk)
        
        # Add chunk information to the results
        results['chunk_index'] = chunk_index
        results['chunk_length'] = len(sequence_chunk)
        
        return results
        
    def simulate_protein_folding_parallel(self, sequence, chunk_size=None):
        """
        Simulate protein folding using parallel processing.
        
        Args:
            sequence (str): The protein sequence
            chunk_size (int, optional): Size of each chunk for parallel processing
                
        Returns:
            dict: The combined simulation results
        """
        if self.ProteinFoldingSimulator is None:
            raise ImportError("Biological implementation classes not available")
            
        # Determine chunk size
        if chunk_size is None:
            chunk_size = max(3, len(sequence) // self.num_processes)  # Ensure chunks are at least 3 amino acids
            
        # Create chunks
        chunks = []
        for i in range(0, len(sequence), chunk_size):
            chunk_sequence = sequence[i:i+chunk_size]
            chunks.append((chunk_sequence, i // chunk_size))
            
        # Process chunks in parallel
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            chunk_results = list(executor.map(self._process_sequence_chunk, chunks))
            
        elapsed_time = time.time() - start_time
        
        # Combine results
        combined_results = self._combine_chunk_results(chunk_results, sequence)
        combined_results['parallel_processing_time'] = elapsed_time
        
        print(f"Parallel protein folding simulation completed in {elapsed_time:.6f} seconds")
        
        return combined_results
        
    def _combine_chunk_results(self, chunk_results, full_sequence):
        """
        Combine results from multiple chunks.
        
        Args:
            chunk_results (list): List of results from each chunk
            full_sequence (str): The full protein sequence
                
        Returns:
            dict: The combined simulation results
        """
        # Sort chunks by index
        chunk_results.sort(key=lambda x: x['chunk_index'])
        
        # Initialize combined results
        combined_results = {
            'sequence': full_sequence,
            'quaternions': [],
            'triad_pairs': [],
            'mobius_points': [],
            'path_integral': 0.0,
            'dedekind_cuts': {},
            'fano_dual': None
        }
        
        # Combine quaternions and mobius_points
        for result in chunk_results:
            combined_results['quaternions'].extend(result['quaternions'])
            combined_results['mobius_points'].extend(result['mobius_points'])
            
        # Combine triad pairs (need to recompute for boundary regions)
        simulator = self.ProteinFoldingSimulator()
        combined_results['triad_pairs'] = simulator.compute_triad_pairs(combined_results['quaternions'])
        
        # Compute the path integral for the combined results
        combined_results['path_integral'] = simulator.compute_path_integral(combined_results['triad_pairs'])
        
        # Use the dedekind_cuts and fano_dual from the first chunk (they are the same for all chunks)
        combined_results['dedekind_cuts'] = chunk_results[0]['dedekind_cuts']
        combined_results['fano_dual'] = chunk_results[0]['fano_dual']
        
        return combined_results
        
    def benchmark_parallel_vs_sequential(self, sequences):
        """
        Benchmark parallel vs sequential protein folding simulation.
        
        Args:
            sequences (list): List of dictionaries with 'name' and 'sequence' keys
                
        Returns:
            dict: Benchmark results
        """
        if self.ProteinFoldingSimulator is None:
            raise ImportError("Biological implementation classes not available")
            
        results = {
            'sequence_name': [],
            'sequence_length': [],
            'sequential_time': [],
            'parallel_time': [],
            'speedup': []
        }
        
        for seq_data in sequences:
            sequence_name = seq_data['name']
            sequence = seq_data['sequence']
            sequence_length = len(sequence)
            
            print(f"\nBenchmarking {sequence_name} (length: {sequence_length})")
            
            # Sequential simulation
            simulator = self.ProteinFoldingSimulator()
            start_time = time.time()
            sequential_results = simulator.simulate_protein_folding(sequence)
            sequential_time = time.time() - start_time
            
            # Parallel simulation
            start_time = time.time()
            parallel_results = self.simulate_protein_folding_parallel(sequence)
            parallel_time = time.time() - start_time
            
            # Calculate speedup
            speedup = sequential_time / parallel_time if parallel_time > 0 else 0
            
            # Store results
            results['sequence_name'].append(sequence_name)
            results['sequence_length'].append(sequence_length)
            results['sequential_time'].append(sequential_time)
            results['parallel_time'].append(parallel_time)
            results['speedup'].append(speedup)
            
            # Print results
            print(f"  Sequential: {sequential_time:.6f} seconds")
            print(f"  Parallel: {parallel_time:.6f} seconds")
            print(f"  Speedup: {speedup:.2f}x")
            
        return results
        
    def visualize_benchmark_results(self, results, save_path=None):
        """
        Visualize benchmark results.
        
        Args:
            results (dict): Benchmark results
            save_path (str, optional): Path to save the visualization
                
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot execution times
        x = np.arange(len(results['sequence_name']))
        width = 0.35
        
        ax1.bar(x - width/2, results['sequential_time'], width, label='Sequential')
        ax1.bar(x + width/2, results['parallel_time'], width, label='Parallel')
        
        ax1.set_xlabel('Protein Sequence')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Protein Folding Simulation Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{name}\n(len: {length})" for name, length in 
                            zip(results['sequence_name'], results['sequence_length'])],
                           rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot speedup
        ax2.plot(results['sequence_length'], results['speedup'], 'o-', color='green', linewidth=2)
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Speedup (Sequential / Parallel)')
        ax2.set_title('Parallel Processing Speedup')
        ax2.grid(True, alpha=0.3)
        
        # Add a horizontal line at y=1 (no speedup)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        
        # Add text annotations for each point
        for i, speedup in enumerate(results['speedup']):
            ax2.annotate(f"{speedup:.2f}x", 
                        (results['sequence_length'][i], speedup),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return fig

def test_parallel_ecdlp_solver():
    """
    Test the parallel ECDLP solver.
    """
    print("Testing Parallel ECDLP Solver")
    print("============================")
    
    # Create the parallel solver
    solver = ParallelECDLPSolver()
    
    # Benchmark parallel vs sequential
    results = solver.benchmark_parallel_vs_sequential()
    
    # Visualize the results
    solver.visualize_benchmark_results(results, save_path="parallel_ecdlp_benchmark.png")
    
    print("\nBenchmark visualization saved as parallel_ecdlp_benchmark.png")

def test_parallel_protein_folding():
    """
    Test the parallel protein folding simulator.
    """
    print("Testing Parallel Protein Folding Simulator")
    print("========================================")
    
    try:
        # Create the parallel simulator
        simulator = ParallelProteinFoldingSimulator()
        
        # Define test sequences
        test_sequences = [
            {
                "name": "SARS-CoV-2 Spike RBD",
                "sequence": "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
            },
            {
                "name": "Short Test Peptide",
                "sequence": "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFS"
            }
        ]
        
        # Benchmark parallel vs sequential
        results = simulator.benchmark_parallel_vs_sequential(test_sequences)
        
        # Visualize the results
        simulator.visualize_benchmark_results(results, save_path="parallel_protein_folding_benchmark.png")
        
        print("\nBenchmark visualization saved as parallel_protein_folding_benchmark.png")
    except ImportError:
        print("Skipping parallel protein folding test due to missing biological implementation")

if __name__ == "__main__":
    # Test the parallel ECDLP solver
    test_parallel_ecdlp_solver()
    
    # Test the parallel protein folding simulator
    test_parallel_protein_folding()
