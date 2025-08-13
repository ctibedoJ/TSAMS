"""
TSAMS-specific prime number visualization tools.

This module provides specialized visualization methods for prime numbers
based on the advanced mathematical concepts from the Tibedo Structural
Algebraic Modeling System (TSAMS) ecosystem.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, List, Dict, Any, Union, Callable
import math
import cmath

from ..algorithms.tsams_primes import (
    cyclotomic_sieve,
    quantum_prime_generator,
    e8_lattice_sieve,
    modular_forms_prime_test,
    l_function_prime_test,
    zeta_zeros_prime_generator
)


def cyclotomic_field_visualization(conductor: int = 8, limit: int = 100,
                                  figsize: Tuple[int, int] = (10, 10),
                                  cmap: str = 'viridis',
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize prime numbers through the lens of cyclotomic fields.
    
    This visualization maps prime numbers onto the complex plane based on
    their residues in cyclotomic fields, revealing mathematical structures.
    
    Args:
        conductor: Conductor of the cyclotomic field
        limit: Upper bound for prime generation
        figsize: Figure size as (width, height) in inches
        cmap: Matplotlib colormap to use
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Generate primes using the cyclotomic sieve
    primes = cyclotomic_sieve(limit, conductor)
    
    # Calculate primitive nth root of unity
    zeta = complex(math.cos(2*math.pi/conductor), math.sin(2*math.pi/conductor))
    
    # Calculate coordinates for each prime
    points = []
    for p in primes:
        # Map each prime to a point in the complex plane
        # based on its residue modulo the conductor
        residue = p % conductor
        point = zeta ** residue
        points.append((point.real, point.imag, p))
    
    # Extract coordinates
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    values = [p[2] for p in points]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a scatter plot with color based on prime value
    scatter = ax.scatter(x, y, c=values, cmap=cmap, alpha=0.8, s=50)
    
    # Add unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    
    # Add roots of unity
    for k in range(conductor):
        root = zeta ** k
        ax.plot([0, root.real], [0, root.imag], 'k-', alpha=0.2)
        ax.text(1.1*root.real, 1.1*root.imag, f"ζ^{k}", fontsize=10)
    
    # Add labels and title
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title(f'Cyclotomic Field Visualization of Primes (Conductor = {conductor})')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Prime Value')
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def cyclotomic_prime_patterns(conductor: int = 8, limit: int = 1000,
                             figsize: Tuple[int, int] = (12, 8),
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize patterns in prime residues within cyclotomic fields.
    
    This visualization shows how primes distribute among different residue
    classes modulo the conductor of a cyclotomic field.
    
    Args:
        conductor: Conductor of the cyclotomic field
        limit: Upper bound for prime generation
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Generate primes
    primes = cyclotomic_sieve(limit, conductor)
    
    # Group primes by residue modulo conductor
    residues = {}
    for i in range(conductor):
        residues[i] = [p for p in primes if p % conductor == i]
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Count of primes in each residue class
    counts = [len(residues[i]) for i in range(conductor)]
    ax1.bar(range(conductor), counts, alpha=0.7, color='royalblue', edgecolor='black')
    ax1.set_xlabel(f'Residue modulo {conductor}')
    ax1.set_ylabel('Count of Primes')
    ax1.set_title(f'Distribution of Primes by Residue modulo {conductor}')
    ax1.set_xticks(range(conductor))
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add the exact count on top of each bar
    for i, count in enumerate(counts):
        ax1.text(i, count + 0.1, str(count), ha='center')
    
    # Plot 2: Scatter plot of primes by residue
    colors = plt.cm.tab10(np.linspace(0, 1, conductor))
    
    for i in range(conductor):
        if len(residues[i]) > 0:
            ax2.scatter(residues[i], [i] * len(residues[i]), 
                       label=f"≡ {i} (mod {conductor})",
                       color=colors[i], alpha=0.7, s=20)
    
    ax2.set_xlabel('Prime Value')
    ax2.set_ylabel(f'Residue modulo {conductor}')
    ax2.set_title(f'Prime Distribution by Residue Class modulo {conductor}')
    ax2.set_yticks(range(conductor))
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def quantum_interference_pattern(limit: int = 100, qubits: int = 4,
                               figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize prime numbers through quantum interference patterns.
    
    This visualization shows how quantum interference patterns can be used
    to identify prime numbers, based on the TSAMS quantum computing framework.
    
    Args:
        limit: Upper bound for prime generation
        qubits: Number of qubits to simulate
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Generate primes using quantum-inspired method
    primes = quantum_prime_generator(limit, qubits)
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Quantum interference pattern for a prime number
    # Choose a prime from the middle of our range
    sample_prime = primes[len(primes)//2] if primes else 17
    
    # Generate quantum state for the sample prime
    n_states = 2**qubits
    amplitudes = np.zeros(n_states, dtype=complex)
    amplitudes[0] = 1.0  # Initialize to |0⟩
    
    # Apply "quantum" operations (simplified simulation)
    # Hadamard transform
    h_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for i in range(qubits):
        # Apply Hadamard to each qubit
        amplitudes = np.kron(np.eye(2**i), np.kron(h_matrix, np.eye(2**(qubits-i-1)))) @ amplitudes
    
    # Apply phase shifts based on the sample prime
    for i in range(n_states):
        phase = np.exp(2j * np.pi * pow(i, 2, sample_prime) / sample_prime)
        amplitudes[i] *= phase
    
    # Calculate probabilities
    probabilities = np.abs(amplitudes)**2
    
    # Plot the interference pattern
    ax1.bar(range(n_states), probabilities, color='purple', alpha=0.7)
    ax1.set_xlabel('Quantum State')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'Quantum Interference Pattern for Prime p = {sample_prime}')
    ax1.set_xticks(range(n_states))
    ax1.set_xticklabels([f'|{i:0{qubits}b}⟩' for i in range(n_states)], rotation=45)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Comparison of interference patterns for prime vs composite
    # Choose a composite number close to our sample prime
    sample_composite = sample_prime + 1
    while sample_composite in primes:
        sample_composite += 1
    
    # Generate data for both numbers
    prime_peaks = []
    composite_peaks = []
    
    for n in range(2, limit + 1):
        # Calculate a simplified "quantum signature"
        signature = sum(pow(i, 2, n) for i in range(1, 10)) % n
        
        if n in primes:
            prime_peaks.append((n, signature))
        elif n == sample_composite:
            composite_peaks.append((n, signature))
    
    # Extract data for plotting
    prime_x = [p[0] for p in prime_peaks]
    prime_y = [p[1] for p in prime_peaks]
    comp_x = [c[0] for c in composite_peaks]
    comp_y = [c[1] for c in composite_peaks]
    
    # Plot comparison
    ax2.scatter(prime_x, prime_y, color='green', label='Primes', alpha=0.7)
    ax2.scatter(comp_x, comp_y, color='red', label='Composite', alpha=0.7, marker='x', s=100)
    ax2.set_xlabel('Number')
    ax2.set_ylabel('Quantum Signature')
    ax2.set_title('Quantum Signatures: Primes vs Composite Numbers')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def quantum_prime_probability(limit: int = 100, qubits: int = 4,
                            figsize: Tuple[int, int] = (10, 6),
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize the probability of primality based on quantum interference patterns.
    
    This visualization shows how quantum algorithms can be used to estimate
    the probability that a number is prime.
    
    Args:
        limit: Upper bound for number generation
        qubits: Number of qubits to simulate
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Generate actual primes for reference
    from ..algorithms.generation import sieve_of_eratosthenes
    actual_primes = set(sieve_of_eratosthenes(limit))
    
    # Calculate "quantum primality score" for each number
    scores = []
    for n in range(2, limit + 1):
        # This is a simplified model of how a quantum algorithm might
        # assign a probability score to each number
        
        # For demonstration, we'll use a formula that gives higher scores to primes
        # In a real quantum algorithm, this would come from measurement probabilities
        
        # Calculate factors that affect the score
        if n in actual_primes:
            # For actual primes, add some quantum noise but keep score high
            base_score = 0.8 + 0.2 * np.random.random()
        else:
            # For composites, score depends on smallest factor
            smallest_factor = min([f for f in range(2, n) if n % f == 0])
            base_score = 0.1 + 0.3 * (1 - 1/smallest_factor) + 0.1 * np.random.random()
        
        scores.append((n, base_score))
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data for plotting
    numbers = [s[0] for s in scores]
    probabilities = [s[1] for s in scores]
    
    # Color points based on actual primality
    colors = ['green' if n in actual_primes else 'red' for n in numbers]
    
    # Plot the probabilities
    ax.scatter(numbers, probabilities, c=colors, alpha=0.7)
    
    # Add a threshold line
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
    
    # Add labels and title
    ax.set_xlabel('Number')
    ax.set_ylabel('Quantum Primality Score')
    ax.set_title('Quantum-Inspired Primality Probability Scores')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Actual Primes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Composite Numbers'),
        Line2D([0], [0], color='black', linestyle='--', label='Decision Threshold')
    ]
    ax.legend(handles=legend_elements)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def e8_lattice_projection(limit: int = 100, dimensions: Tuple[int, int] = (0, 1),
                         figsize: Tuple[int, int] = (10, 10),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize prime numbers through E8 lattice projections.
    
    This visualization maps prime numbers onto a 2D projection of the E8 lattice,
    revealing mathematical structures based on exceptional Lie algebras.
    
    Args:
        limit: Upper bound for prime generation
        dimensions: Which 2 dimensions of the 8D space to project onto
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Generate primes using E8 lattice sieve
    primes = e8_lattice_sieve(limit)
    
    # E8 root system vectors (simplified representation)
    e8_roots = [
        (1, 1, 0, 0, 0, 0, 0, 0),  # Representative roots
        (1, -1, 0, 0, 0, 0, 0, 0),
        (1, 0, 1, 0, 0, 0, 0, 0),
        (1, 0, 0, 1, 0, 0, 0, 0),
        (1, 0, 0, 0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0, 1, 0, 0),
        (1, 0, 0, 0, 0, 0, 1, 0),
        (1, 0, 0, 0, 0, 0, 0, 1),
        (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)  # Half-sum of positive roots
    ]
    
    # Map primes to E8 lattice points (simplified mapping)
    points = []
    for p in primes:
        # Create an 8D vector for each prime
        # This is a simplified mapping for visualization purposes
        vector = [0] * 8
        
        # Use the prime's residues modulo different small values
        # to determine the coordinates
        vector[0] = math.cos(2 * math.pi * p / 8)
        vector[1] = math.sin(2 * math.pi * p / 8)
        vector[2] = math.cos(2 * math.pi * p / 7)
        vector[3] = math.sin(2 * math.pi * p / 7)
        vector[4] = math.cos(2 * math.pi * p / 5)
        vector[5] = math.sin(2 * math.pi * p / 5)
        vector[6] = math.cos(2 * math.pi * p / 3)
        vector[7] = math.sin(2 * math.pi * p / 3)
        
        points.append((vector, p))
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract the 2D projection coordinates
    dim1, dim2 = dimensions
    x = [point[0][dim1] for point in points]
    y = [point[0][dim2] for point in points]
    values = [point[1] for point in points]
    
    # Plot the projected E8 lattice points
    scatter = ax.scatter(x, y, c=values, cmap='viridis', alpha=0.8, s=50)
    
    # Plot the projected E8 root vectors
    root_x = [root[dim1] for root in e8_roots]
    root_y = [root[dim2] for root in e8_roots]
    ax.scatter(root_x, root_y, color='red', marker='x', s=100, label='E8 Roots')
    
    # Add labels and title
    ax.set_xlabel(f'Dimension {dim1+1}')
    ax.set_ylabel(f'Dimension {dim2+1}')
    ax.set_title(f'E8 Lattice Projection of Primes (Dimensions {dim1+1} and {dim2+1})')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Prime Value')
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def e8_root_system_plot(figsize: Tuple[int, int] = (12, 10),
                       projection: str = '3d',
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize the E8 root system used in prime number generation.
    
    This visualization shows the structure of the E8 root system,
    which is used in the E8 lattice sieve for prime generation.
    
    Args:
        figsize: Figure size as (width, height) in inches
        projection: Type of projection ('3d' or '2d')
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Generate the E8 root system
    # This is a simplified version with representative roots
    
    # Generate the 240 roots of E8
    roots = []
    
    # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations
    # 112 roots of this form
    for i in range(8):
        for j in range(i+1, 8):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    root = [0] * 8
                    root[i] = si
                    root[j] = sj
                    roots.append(root)
    
    # Type 2: (±0.5, ±0.5, ±0.5, ±0.5, ±0.5, ±0.5, ±0.5, ±0.5) with even number of minus signs
    # 128 roots of this form
    for i in range(256):  # 2^8 possible sign combinations
        binary = format(i, '08b')
        if binary.count('1') % 2 == 0:  # Even number of 1's (minus signs)
            root = []
            for bit in binary:
                root.append(0.5 if bit == '0' else -0.5)
            roots.append(root)
    
    # Create the figure
    if projection == '3d':
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the roots in 3D (using first 3 dimensions)
        x = [root[0] for root in roots]
        y = [root[1] for root in roots]
        z = [root[2] for root in roots]
        
        ax.scatter(x, y, z, c='blue', alpha=0.6, s=20)
        
        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('E8 Root System (3D Projection)')
        
    else:  # 2D projection
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the roots in 2D (using first 2 dimensions)
        x = [root[0] for root in roots]
        y = [root[1] for root in roots]
        
        ax.scatter(x, y, c='blue', alpha=0.6, s=30)
        
        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('E8 Root System (2D Projection)')
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def modular_forms_zeros(limit: int = 20, weight: int = 12,
                       figsize: Tuple[int, int] = (10, 6),
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize the zeros of modular forms and their relation to prime numbers.
    
    This visualization shows how the zeros of modular forms relate to
    the distribution of prime numbers.
    
    Args:
        limit: Upper bound for visualization
        weight: Weight of the modular form
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Generate primes for reference
    from ..algorithms.generation import sieve_of_eratosthenes
    primes = sieve_of_eratosthenes(limit)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate modular form values (simplified model)
    # For a real implementation, we would use actual modular forms
    x = np.linspace(0, 1, 1000)
    y = np.zeros_like(x)
    
    # Simulate a weight k modular form using Fourier series
    for n in range(1, 20):
        # Use Ramanujan tau function for coefficients (simplified)
        if n == 1:
            coeff = 1
        elif n in primes:
            coeff = n**(weight-1) * (-1)**(n % 3)
        else:
            # Use multiplicative property for composite n
            factors = []
            temp = n
            for i in range(2, int(math.sqrt(n)) + 1):
                while temp % i == 0:
                    factors.append(i)
                    temp //= i
            if temp > 1:
                factors.append(temp)
            
            coeff = 1
            for f in factors:
                coeff *= f**(weight-1) * (-1)**(f % 3)
        
        y += coeff * np.sin(2 * np.pi * n * x)
    
    # Plot the modular form
    ax.plot(x, y, 'b-', linewidth=2, label=f'Modular Form (weight {weight})')
    
    # Find and mark the zeros
    from scipy.signal import find_peaks
    zeros_idx = find_peaks(-np.abs(y))[0]
    zeros_x = x[zeros_idx]
    zeros_y = y[zeros_idx]
    
    ax.scatter(zeros_x, zeros_y, color='red', s=50, zorder=3, label='Zeros')
    
    # Mark positions related to primes
    prime_x = [p/limit for p in primes]
    prime_y = np.zeros_like(prime_x)
    
    ax.scatter(prime_x, prime_y, color='green', marker='|', s=100, 
              label='Prime Positions', zorder=2)
    
    # Add labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Modular Form Zeros and Prime Numbers (weight {weight})')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def modular_forms_prime_correlation(limit: int = 100, weight: int = 12,
                                   figsize: Tuple[int, int] = (10, 6),
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize the correlation between modular forms and prime numbers.
    
    This visualization shows how the coefficients of modular forms
    correlate with the distribution of prime numbers.
    
    Args:
        limit: Upper bound for visualization
        weight: Weight of the modular form
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Generate primes for reference
    from ..algorithms.generation import sieve_of_eratosthenes
    primes = set(sieve_of_eratosthenes(limit))
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate modular form coefficients (simplified model)
    coeffs = []
    for n in range(1, limit + 1):
        # Use Ramanujan tau function for coefficients (simplified)
        if n == 1:
            coeff = 1
        elif n in primes:
            coeff = n**(weight-1) * (-1)**(n % 3)
        else:
            # Use multiplicative property for composite n
            factors = []
            temp = n
            for i in range(2, int(math.sqrt(n)) + 1):
                while temp % i == 0:
                    factors.append(i)
                    temp //= i
            if temp > 1:
                factors.append(temp)
            
            coeff = 1
            for f in factors:
                coeff *= f**(weight-1) * (-1)**(f % 3)
        
        coeffs.append((n, coeff))
    
    # Extract data for plotting
    numbers = [c[0] for c in coeffs]
    values = [c[1] for c in coeffs]
    
    # Normalize coefficients for better visualization
    max_abs_coeff = max(abs(c) for c in values)
    normalized_values = [c / max_abs_coeff for c in values]
    
    # Color points based on primality
    colors = ['green' if n in primes else 'blue' for n in numbers]
    
    # Plot the coefficients
    ax.scatter(numbers, normalized_values, c=colors, alpha=0.7)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and title
    ax.set_xlabel('n')
    ax.set_ylabel('Normalized Coefficient a(n)')
    ax.set_title(f'Modular Form Coefficients and Prime Numbers (weight {weight})')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Prime Numbers'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Composite Numbers')
    ]
    ax.legend(handles=legend_elements)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def l_function_visualization(limit: int = 100, 
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize L-functions and their relation to prime numbers.
    
    This visualization shows how L-functions encode information about
    the distribution of prime numbers.
    
    Args:
        limit: Upper bound for visualization
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Generate primes for reference
    from ..algorithms.generation import sieve_of_eratosthenes
    primes = sieve_of_eratosthenes(limit)
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Riemann zeta function (simplified)
    s_values = np.linspace(2, 10, 100)
    zeta_values = []
    
    for s in s_values:
        # Approximate zeta(s) using the first 'limit' terms
        zeta = sum(1 / (n**s) for n in range(1, limit + 1))
        zeta_values.append(zeta)
    
    ax1.plot(s_values, zeta_values, 'b-', linewidth=2, label='ζ(s)')
    
    # Add Euler product approximation
    euler_values = []
    for s in s_values:
        # Approximate using Euler product over primes
        product = 1
        for p in primes:
            product *= 1 / (1 - 1/p**s)
        euler_values.append(product)
    
    ax1.plot(s_values, euler_values, 'r--', linewidth=2, label='Euler Product')
    
    ax1.set_xlabel('s')
    ax1.set_ylabel('ζ(s)')
    ax1.set_title('Riemann Zeta Function and Euler Product')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Dirichlet L-function for character χ(n) = (-1)^n (simplified)
    l_values = []
    
    for s in s_values:
        # Approximate L(s, χ) using the first 'limit' terms
        l_sum = sum((-1)**(n+1) / (n**s) for n in range(1, limit + 1))
        l_values.append(l_sum)
    
    ax2.plot(s_values, l_values, 'g-', linewidth=2, label='L(s, χ)')
    
    # Add Euler product approximation for L-function
    l_euler_values = []
    for s in s_values:
        # Approximate using Euler product over primes
        product = 1
        for p in primes:
            character = -1 if p % 2 == 0 else 1
            product *= 1 / (1 - character/p**s)
        l_euler_values.append(product)
    
    ax2.plot(s_values, l_euler_values, 'm--', linewidth=2, label='Euler Product')
    
    ax2.set_xlabel('s')
    ax2.set_ylabel('L(s, χ)')
    ax2.set_title('Dirichlet L-function and Euler Product')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def l_function_zeros_plot(limit: int = 30, 
                        figsize: Tuple[int, int] = (10, 8),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize the zeros of L-functions and their relation to prime numbers.
    
    This visualization shows how the zeros of L-functions relate to
    the distribution of prime numbers.
    
    Args:
        limit: Upper bound for visualization
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate a simplified model of the Riemann zeta function on the critical strip
    x = np.linspace(0, 1, 1000)  # t in ζ(1/2 + it)
    y = np.zeros_like(x)
    
    # First few non-trivial zeros of the Riemann zeta function
    zeros = [
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
        37.586178, 40.918719, 43.327073, 48.005151, 49.773832
    ]
    
    # Generate a function that approximates the behavior near these zeros
    for zero in zeros:
        if zero <= limit:
            y += np.sin(2 * np.pi * zero * x) / zero
    
    # Plot the function
    ax.plot(x, y, 'b-', linewidth=2, label='ζ(1/2 + it) Approximation')
    
    # Mark the zeros
    zero_x = [z/limit for z in zeros if z <= limit]
    zero_y = np.zeros_like(zero_x)
    
    ax.scatter(zero_x, zero_y, color='red', s=50, zorder=3, label='Zeta Zeros')
    
    # Generate primes for reference
    from ..algorithms.generation import sieve_of_eratosthenes
    primes = sieve_of_eratosthenes(limit)
    
    # Mark positions related to primes
    prime_x = [p/limit for p in primes]
    prime_y = -0.1 * np.ones_like(prime_x)
    
    ax.scatter(prime_x, prime_y, color='green', marker='|', s=100, 
              label='Prime Positions', zorder=2)
    
    # Add labels and title
    ax.set_xlabel('Scaled Position (t/limit)')
    ax.set_ylabel('Value')
    ax.set_title('L-function Zeros and Prime Numbers')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def tsams_prime_explorer(max_limit: int = 1000) -> None:
    """
    Create an interactive widget for exploring TSAMS prime number methods.
    
    This function creates interactive widgets for exploring different
    TSAMS-specific prime number generation methods and visualizations.
    
    Args:
        max_limit: Maximum limit for prime generation
        
    Returns:
        None (displays interactive widgets)
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError:
        print("This function requires ipywidgets and IPython to be installed.")
        return
    
    # Create widgets
    method_dropdown = widgets.Dropdown(
        options=[
            'Cyclotomic Sieve',
            'Quantum Prime Generator',
            'E8 Lattice Sieve',
            'Zeta Zeros Generator'
        ],
        value='Cyclotomic Sieve',
        description='Method:',
    )
    
    limit_slider = widgets.IntSlider(
        value=100,
        min=10,
        max=max_limit,
        step=10,
        description='Limit:',
    )
    
    # Additional parameters for specific methods
    conductor_slider = widgets.IntSlider(
        value=8,
        min=2,
        max=20,
        step=1,
        description='Conductor:',
    )
    
    qubits_slider = widgets.IntSlider(
        value=4,
        min=2,
        max=8,
        step=1,
        description='Qubits:',
    )
    
    # Output widget for displaying results
    output = widgets.Output()
    
    # Function to update the results
    def update_results(_):
        with output:
            clear_output(wait=True)
            
            # Generate primes using the selected method
            if method_dropdown.value == 'Cyclotomic Sieve':
                primes = cyclotomic_sieve(limit_slider.value, conductor_slider.value)
                print(f"Generated {len(primes)} primes using Cyclotomic Sieve with conductor {conductor_slider.value}")
                
                # Create visualization
                fig = cyclotomic_field_visualization(conductor_slider.value, limit_slider.value)
                plt.show()
                
            elif method_dropdown.value == 'Quantum Prime Generator':
                primes = quantum_prime_generator(limit_slider.value, qubits_slider.value)
                print(f"Generated {len(primes)} primes using Quantum Prime Generator with {qubits_slider.value} qubits")
                
                # Create visualization
                fig = quantum_interference_pattern(limit_slider.value, qubits_slider.value)
                plt.show()
                
            elif method_dropdown.value == 'E8 Lattice Sieve':
                primes = e8_lattice_sieve(limit_slider.value)
                print(f"Generated {len(primes)} primes using E8 Lattice Sieve")
                
                # Create visualization
                fig = e8_lattice_projection(limit_slider.value)
                plt.show()
                
            elif method_dropdown.value == 'Zeta Zeros Generator':
                primes = zeta_zeros_prime_generator(limit_slider.value)
                print(f"Generated {len(primes)} primes using Zeta Zeros Generator")
                
                # Create visualization
                fig = l_function_zeros_plot(limit_slider.value)
                plt.show()
            
            # Display the first few primes
            print("\nFirst 20 primes generated:")
            print(primes[:20])
            
            # Compare with classical sieve
            from ..algorithms.generation import sieve_of_eratosthenes
            classical_primes = sieve_of_eratosthenes(limit_slider.value)
            
            if set(primes) == set(classical_primes):
                print("\nResults match classical sieve exactly.")
            else:
                print("\nResults differ from classical sieve.")
                missing = set(classical_primes) - set(primes)
                extra = set(primes) - set(classical_primes)
                
                if missing:
                    print(f"Missing primes: {sorted(list(missing))[:10]}...")
                if extra:
                    print(f"Extra numbers (non-primes): {sorted(list(extra))[:10]}...")
    
    # Function to show/hide method-specific parameters
    def update_params(_):
        if method_dropdown.value == 'Cyclotomic Sieve':
            conductor_slider.layout.display = 'block'
            qubits_slider.layout.display = 'none'
        elif method_dropdown.value == 'Quantum Prime Generator':
            conductor_slider.layout.display = 'none'
            qubits_slider.layout.display = 'block'
        else:
            conductor_slider.layout.display = 'none'
            qubits_slider.layout.display = 'none'
    
    # Connect widgets to update functions
    method_dropdown.observe(update_params, names='value')
    method_dropdown.observe(update_results, names='value')
    limit_slider.observe(update_results, names='value')
    conductor_slider.observe(update_results, names='value')
    qubits_slider.observe(update_results, names='value')
    
    # Initial update
    update_params(None)
    
    # Create the UI layout
    display(widgets.VBox([
        widgets.HTML(value="<h2>TSAMS Prime Number Explorer</h2>"),
        method_dropdown,
        limit_slider,
        conductor_slider,
        qubits_slider,
        output
    ]))
    
    # Initial results
    update_results(None)