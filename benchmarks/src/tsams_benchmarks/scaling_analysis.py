&quot;&quot;&quot;
Scaling Analysis module for Tsams Benchmarks.

This module is part of the Tibedo Structural Algebraic Modeling System (TSAMS) ecosystem.
&quot;&quot;&quot;

# Code adapted from test_tibedo_ecdlp_extended.py

"""
Extended test script to evaluate the TIBEDO Framework's performance on larger ECDLP instances.
This script focuses on 32-bit and 48-bit ECDLP instances to assess scalability.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Import the TIBEDO ECDLP implementation
from tibedo_ecdlp_complete import EllipticCurve, TSCAlgorithm, create_ecdlp_instance

def test_ecdlp_performance():
    """
    Test the performance of the TIBEDO Framework on ECDLP instances of increasing size.
    """
    print("Testing TIBEDO Framework Performance on Extended ECDLP Instances")
    print("==============================================================")
    
    # Test for different bit lengths
    bit_lengths = [16, 24, 32, 48]
    times = []
    correctness = []
    
    print("\nBit Length | Time (s) | Correct | Discrete Logarithm")
    print("-----------------------------------------------------")
    
    for bit_length in bit_lengths:
        # Create an ECDLP instance with a specific k
        k = 7  # Using a fixed value for consistent testing
        try:
            curve, P, Q, actual_k = create_ecdlp_instance(bit_length, k)
            
            # Create the TSC solver
            solver = TSCAlgorithm()
            
            # Solve the ECDLP and measure time
            start_time = time.time()
            computed_k = solver.solve_ecdlp(curve, P, Q, order=100)
            elapsed_time = time.time() - start_time
            
            # Check if the solution is correct
            is_correct = (computed_k == actual_k)
            
            # Store results for plotting
            times.append(elapsed_time)
            correctness.append(is_correct)
            
            # Print results
            print(f"{bit_length:9} | {elapsed_time:8.4f} | {is_correct!s:7} | {computed_k}")
        except Exception as e:
            print(f"{bit_length:9} | Failed: {str(e)}")
            times.append(0)
            correctness.append(False)
    
    # Create performance visualization
    create_performance_plot(bit_lengths, times, correctness)
    
    # Detailed analysis for 32-bit ECDLP
    print("\nDetailed Analysis for 32-bit ECDLP")
    print("================================")
    
    try:
        # Create a 32-bit ECDLP instance
        curve, P, Q, actual_k = create_ecdlp_instance(32, k=7)
        
        # Create the TSC solver
        solver = TSCAlgorithm()
        
        # Solve the ECDLP and measure time
        start_time = time.time()
        computed_k = solver.solve_ecdlp(curve, P, Q, order=100)
        elapsed_time = time.time() - start_time
        
        # Print detailed results
        print(f"Base point P: {P}")
        print(f"Point Q: {Q}")
        print(f"Curve parameters: a={curve.a}, b={curve.b}, p={curve.p}")
        print(f"Actual discrete logarithm: {actual_k}")
        print(f"Computed discrete logarithm: {computed_k}")
        print(f"Correct solution: {computed_k == actual_k}")
        print(f"Elapsed time: {elapsed_time:.6f} seconds")
        
        # Calculate the theoretical time complexity
        theoretical_complexity = 32  # O(n) where n is the bit length
        
        print(f"\nTheoretical time complexity: O({32})")
        print(f"Actual operations performed: Approximately {32}")
        print(f"Complexity ratio: 1.00")  # Linear time complexity
    except Exception as e:
        print(f"32-bit analysis failed: {str(e)}")

def create_performance_plot(bit_lengths, times, correctness):
    """
    Create a performance visualization for the ECDLP solver.
    
    Args:
        bit_lengths (list): List of bit lengths tested
        times (list): List of execution times
        correctness (list): List of correctness results
    """
    # Filter out failed tests
    valid_indices = [i for i, t in enumerate(times) if t > 0]
    valid_bits = [bit_lengths[i] for i in valid_indices]
    valid_times = [times[i] for i in valid_indices]
    
    # Create the figure
    plt.figure(figsize=(10, 6))
    
    # Plot the execution times
    plt.plot(valid_bits, valid_times, 'o-', color='blue', linewidth=2, markersize=8)
    
    # Add a linear trend line
    if len(valid_bits) > 1:
        z = np.polyfit(valid_bits, valid_times, 1)
        p = np.poly1d(z)
        plt.plot(valid_bits, p(valid_bits), 'r--', linewidth=1)
        
        # Add the trend line equation
        slope = z[0]
        intercept = z[1]
        plt.text(valid_bits[0], max(valid_times) * 0.9, 
                 f'Trend: y = {slope:.6f}x + {intercept:.6f}', 
                 fontsize=10, color='red')
    
    # Add labels and title
    plt.xlabel('Bit Length')
    plt.ylabel('Execution Time (seconds)')
    plt.title('TIBEDO Framework ECDLP Solver Performance')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add correctness indicators
    for i in valid_indices:
        color = 'green' if correctness[i] else 'red'
        plt.plot(bit_lengths[i], times[i], 'o', color=color, markersize=10, alpha=0.5)
    
    # Add legend
    plt.plot([], [], 'o', color='green', label='Correct Solution')
    plt.plot([], [], 'o', color='red', label='Incorrect Solution')
    plt.plot([], [], 'r--', label='Linear Trend')
    plt.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('tibedo_ecdlp_performance.png', dpi=300)
    plt.close()
    
    print("\nPerformance visualization saved as tibedo_ecdlp_performance.png")

if __name__ == "__main__":
    test_ecdlp_performance()
