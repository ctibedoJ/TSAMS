"""
Prime number distribution visualization tools.

This module provides functions to visualize the distribution of prime numbers,
including density plots, gap analysis, and statistical visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, Optional, List, Dict, Any, Union
import math

from ..algorithms.generation import sieve_of_eratosthenes
from ..algorithms.operations import prime_gap


def plot_prime_distribution(limit: int, bin_size: Optional[int] = None,
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the distribution of prime numbers up to a limit.
    
    This creates a histogram showing how prime numbers are distributed
    across different ranges.
    
    Args:
        limit: The upper limit for prime generation
        bin_size: Size of each histogram bin (default is sqrt(limit))
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Generate primes up to the limit
    primes = sieve_of_eratosthenes(limit)
    
    # Set default bin size if not provided
    if bin_size is None:
        bin_size = int(np.sqrt(limit))
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    bins = np.arange(0, limit + bin_size, bin_size)
    counts, edges = np.histogram(primes, bins=bins)
    
    # Plot the histogram
    ax.bar(edges[:-1], counts, width=bin_size * 0.9, align='edge', 
           alpha=0.7, color='royalblue', edgecolor='black', linewidth=0.5)
    
    # Calculate and plot the theoretical prime density based on Prime Number Theorem
    x = (edges[:-1] + edges[1:]) / 2
    theoretical_density = bin_size / np.log(x)
    ax.plot(x, theoretical_density, 'r-', linewidth=2, label='Theoretical (1/ln(n))')
    
    # Add labels and title
    ax.set_xlabel('Number Range')
    ax.set_ylabel('Count of Primes')
    ax.set_title(f'Distribution of Prime Numbers up to {limit}')
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def prime_gaps_plot(limit: int, figsize: Tuple[int, int] = (12, 8),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the gaps between consecutive prime numbers.
    
    Args:
        limit: The upper limit for prime generation
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Generate primes up to the limit
    primes = sieve_of_eratosthenes(limit)
    
    # Calculate gaps between consecutive primes
    gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Gaps vs. Prime Index
    ax1.plot(range(len(gaps)), gaps, 'o', markersize=3, alpha=0.5, color='blue')
    ax1.set_xlabel('Prime Index')
    ax1.set_ylabel('Gap Size')
    ax1.set_title(f'Gaps Between Consecutive Primes up to {limit}')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Histogram of Gap Sizes
    max_gap = max(gaps)
    ax2.hist(gaps, bins=range(1, max_gap + 2), align='left', alpha=0.7, 
             color='green', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Gap Size')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Prime Gaps')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def prime_density_heatmap(width: int, height: int, 
                         figsize: Tuple[int, int] = (12, 10),
                         cmap: str = 'viridis',
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a 2D heatmap showing the density of prime numbers.
    
    This visualizes how prime numbers are distributed in a 2D grid,
    where each cell (x,y) is colored based on whether x*width + y is prime.
    
    Args:
        width: Width of the grid
        height: Height of the grid
        figsize: Figure size as (width, height) in inches
        cmap: Matplotlib colormap to use
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Create a grid to store primality
    grid = np.zeros((height, width), dtype=bool)
    
    # Generate all primes up to width*height
    limit = width * height
    primes = set(sieve_of_eratosthenes(limit))
    
    # Fill the grid
    for y in range(height):
        for x in range(width):
            n = y * width + x + 1  # 1-indexed
            if n in primes:
                grid[y, x] = True
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the heatmap
    im = ax.imshow(grid, cmap=cmap, interpolation='nearest', aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Is Prime')
    
    # Add labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Prime Number Density Heatmap ({width}x{height})')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def prime_counting_function_plot(limit: int, figsize: Tuple[int, int] = (12, 8),
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the prime counting function π(x) and its approximations.
    
    Args:
        limit: The upper limit for the plot
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Generate primes up to the limit
    primes = sieve_of_eratosthenes(limit)
    
    # Create x values for the plot
    x = np.linspace(2, limit, 1000)
    
    # Calculate the prime counting function values
    prime_counts = []
    for val in x:
        count = sum(1 for p in primes if p <= val)
        prime_counts.append(count)
    
    # Calculate approximations
    # Li(x) - logarithmic integral
    li_x = [np.trapz(1/np.log(np.linspace(2, val, 1000)), np.linspace(2, val, 1000)) for val in x]
    
    # x/ln(x) - simple approximation from PNT
    pnt_approx = x / np.log(x)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the actual prime counting function
    ax.plot(x, prime_counts, 'b-', linewidth=2, label='π(x) - Actual')
    
    # Plot the approximations
    ax.plot(x, li_x, 'r--', linewidth=2, label='Li(x) - Logarithmic Integral')
    ax.plot(x, pnt_approx, 'g-.', linewidth=2, label='x/ln(x) - PNT Approximation')
    
    # Add labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('π(x)')
    ax.set_title('Prime Counting Function and Approximations')
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def prime_race_plot(limit: int, modulus: int = 4, remainder: int = 1,
                   figsize: Tuple[int, int] = (12, 8),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the "prime race" between different residue classes.
    
    This visualizes Chebyshev's bias in the distribution of primes in different
    residue classes modulo m.
    
    Args:
        limit: The upper limit for prime generation
        modulus: The modulus to use (e.g., 4 for comparing primes ≡ 1 or 3 (mod 4))
        remainder: The remainder to compare against others
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Generate primes up to the limit
    primes = sieve_of_eratosthenes(limit)
    
    # Skip 2 as it's the only even prime
    primes = [p for p in primes if p > 2]
    
    # Count primes in each residue class
    residue_counts = {r: [0] for r in range(modulus) if math.gcd(r, modulus) == 1}
    
    # Initialize the x-axis values (the primes themselves)
    x_values = []
    
    for p in primes:
        x_values.append(p)
        r = p % modulus
        
        # Only track residues that are coprime to the modulus
        if r in residue_counts:
            # Update all counts
            for res in residue_counts:
                if res == r:
                    residue_counts[res].append(residue_counts[res][-1] + 1)
                else:
                    residue_counts[res].append(residue_counts[res][-1])
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate the differences between the reference remainder and others
    differences = {}
    for r in residue_counts:
        if r != remainder:
            differences[r] = [residue_counts[remainder][i] - residue_counts[r][i] 
                             for i in range(len(residue_counts[r]))]
    
    # Plot the differences
    for r, diff in differences.items():
        ax.plot(x_values, diff[1:], label=f'π(x, {modulus}, {remainder}) - π(x, {modulus}, {r})')
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('Difference in Counts')
    ax.set_title(f'Prime Race: Comparing Primes ≡ {remainder} (mod {modulus}) with Other Residues')
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig