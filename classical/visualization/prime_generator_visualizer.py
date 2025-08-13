#!/usr/bin/env python3
"""
Prime Generator Formula Visualizer
Creates a visualization of the Tibedo Prime Generator Formula in action
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from sympy import isprime
import math

# Set up the figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Tibedo Prime Generator Formula Visualization', fontsize=16)

# Create custom colormap for prime vs non-prime
colors = [(0.8, 0.8, 0.8), (0.2, 0.7, 0.3)]  # Light gray to green
cmap = LinearSegmentedColormap.from_list('prime_cmap', colors, N=2)

# Parameters
max_n = 100  # Maximum number to visualize
grid_size = int(np.ceil(np.sqrt(max_n)))

# Initialize data structures
grid = np.zeros((grid_size, grid_size))
primes = []
prime_positions = []

# Function to simulate the Tibedo Prime Generator Formula
def tibedo_prime_generator(n):
    """
    Simulated version of the Tibedo Prime Generator Formula
    In a real implementation, this would use the actual formula
    """
    # For visualization purposes, we'll use sympy's isprime
    # In reality, this would be the actual formula calculation
    count = 0
    num = 2
    while count < n:
        if isprime(num):
            count += 1
            if count == n:
                return num
        num += 1
    return num

# Function to update the visualization
def update(frame):
    global grid, primes, prime_positions
    
    # Clear previous plots
    ax1.clear()
    ax2.clear()
    
    # Set titles
    ax1.set_title('Prime Number Grid')
    ax2.set_title('Prime Number Distribution')
    
    # Generate the next prime
    if frame > 0:
        next_prime = tibedo_prime_generator(frame)
        primes.append(next_prime)
        
        # Calculate grid position (row, col)
        row = (next_prime - 1) // grid_size
        col = (next_prime - 1) % grid_size
        
        if row < grid_size and col < grid_size:
            grid[row, col] = 1
            prime_positions.append((row, col))
    
    # Plot the grid
    ax1.imshow(grid, cmap=cmap, interpolation='nearest')
    
    # Add grid lines
    ax1.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax1.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    
    # Add numbers to the grid
    for i in range(grid_size):
        for j in range(grid_size):
            num = i * grid_size + j + 1
            if num <= max_n:
                ax1.text(j, i, str(num), ha='center', va='center', 
                         color='black' if grid[i, j] == 0 else 'white',
                         fontsize=8)
    
    # Highlight the latest prime
    if frame > 0 and prime_positions:
        latest_row, latest_col = prime_positions[-1]
        ax1.add_patch(plt.Rectangle((latest_col-0.5, latest_row-0.5), 1, 1, 
                                   fill=False, edgecolor='red', linewidth=2))
    
    # Plot the prime distribution
    if primes:
        x = np.arange(1, len(primes) + 1)
        ax2.plot(x, primes, 'o-', color='blue')
        ax2.set_xlabel('n-th Prime')
        ax2.set_ylabel('Prime Value')
        
        # Add the Prime Number Theorem approximation
        x_smooth = np.linspace(1, len(primes), 100)
        pnt_approx = x_smooth * np.log(x_smooth)
        ax2.plot(x_smooth, pnt_approx, '--', color='red', 
                 label='n log(n) approximation')
        
        # Add formula visualization
        if frame > 0:
            formula_text = f"P({frame}) = {primes[-1]}"
            ax2.text(0.05, 0.95, formula_text, transform=ax2.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.legend()
        ax2.grid(True)
    
    return ax1, ax2

# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(31), interval=500, blit=False)

# Save as MP4
ani.save('prime_generator_visualization.mp4', writer='ffmpeg', fps=2, dpi=150)

# Display final frame
plt.tight_layout()
plt.savefig('prime_generator_final_frame.png', dpi=150)
print("Visualization complete. Video saved as 'prime_generator_visualization.mp4'")