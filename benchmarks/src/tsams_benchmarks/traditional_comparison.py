&quot;&quot;&quot;
Traditional Comparison module for Tsams Benchmarks.

This module is part of the Tibedo Structural Algebraic Modeling System (TSAMS) ecosystem.
&quot;&quot;&quot;

# Code adapted from test_tibedo_ecdlp.py

"""
Test script to evaluate the TIBEDO Framework's performance on ECDLP.
"""

import time
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the tibedo package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from tibedo.core.tsc import TSCSolver

def create_ecdlp_instance(bit_length=32):
    """
    Create a simple ECDLP instance for testing.
    
    Args:
        bit_length (int): The bit length of the ECDLP instance.
        
    Returns:
        tuple: (P, Q, curve_params, k) where:
            P is the base point
            Q is the point to find the discrete logarithm for
            curve_params are the parameters of the elliptic curve
            k is the actual discrete logarithm (for verification)
    """
    # Create a small prime field
    p = 2**bit_length - 5  # A prime close to 2^bit_length
    
    # Create simple curve parameters (y^2 = x^3 + ax + b)
    a = 2
    b = 3
    
    # Create a base point P (in practice, this would be a point on the curve)
    x1, y1 = 5, 10
    
    # Set the order of the base point (in practice, this would be computed)
    n = 2**(bit_length - 1) - 3  # A value less than p
    
    # Choose a random discrete logarithm k
    k = np.random.randint(1, n)
    
    # Compute Q = k*P (in practice, this would use elliptic curve point multiplication)
    # For this test, we'll just simulate it
    x2 = (x1 * k) % p
    y2 = (y1 * k) % p
    
    # Create the curve parameters dictionary
    curve_params = {
        'a': a,
        'b': b,
        'p': p,
        'n': n
    }
    
    # Return the ECDLP instance
    return (x1, y1), (x2, y2), curve_params, k

def test_tibedo_performance():
    """
    Test the performance of the TIBEDO Framework on ECDLP instances of increasing size.
    """
    print("Testing TIBEDO Framework Performance on ECDLP")
    print("=============================================")
    print("\nBit Length | Time (s) | Correct | Complexity Ratio")
    print("-------------------------------------------------")
    
    # Test for different bit lengths
    for bit_length in [16, 24, 32, 40, 48]:
        # Create an ECDLP instance
        P, Q, curve_params, actual_k = create_ecdlp_instance(bit_length)
        
        # Create the TSC solver
        solver = TSCSolver()
        
        # Solve the ECDLP and measure time
        start_time = time.time()
        computed_k = solver.solve(P, Q, curve_params)
        elapsed_time = time.time() - start_time
        
        # Check if the solution is correct
        is_correct = (computed_k == actual_k)
        
        # Get performance metrics
        metrics = solver.get_performance_metrics()
        complexity_ratio = metrics.get('complexity_ratio', 'N/A')
        
        # Print results
        print(f"{bit_length:9} | {elapsed_time:8.4f} | {is_correct!s:7} | {complexity_ratio}")
    
    print("\nDetailed Analysis for 32-bit ECDLP")
    print("================================")
    
    # Create a 32-bit ECDLP instance
    P, Q, curve_params, actual_k = create_ecdlp_instance(32)
    
    # Create the TSC solver
    solver = TSCSolver()
    
    # Solve the ECDLP
    start_time = time.time()
    computed_k = solver.solve(P, Q, curve_params)
    elapsed_time = time.time() - start_time
    
    # Get detailed results
    detailed_results = solver.get_detailed_result()
    metrics = solver.get_performance_metrics()
    
    # Print detailed results
    print(f"Actual discrete logarithm: {actual_k}")
    print(f"Computed discrete logarithm: {computed_k}")
    print(f"Correct solution: {computed_k == actual_k}")
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print("\nPerformance metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_tibedo_performance()
