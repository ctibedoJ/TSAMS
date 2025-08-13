#!/usr/bin/env python3
"""
TSAMS Prime Generator Formula Visualizer
Creates a visualization of the Tibedo Prime Generator Formula in action,
specifically demonstrating the cyclotomic field and E8 lattice projection methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from sympy import isprime, mobius, totient
import math
import cmath

# Set up the figure with multiple subplots for different visualizations
fig = plt.figure(figsize=(16, 12))
fig.suptitle('TSAMS Prime Generator Formula Visualization', fontsize=20)

# Create grid for subplots
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])  # Prime number grid
ax2 = fig.add_subplot(gs[0, 1])  # Prime distribution
ax3 = fig.add_subplot(gs[1, 0])  # Cyclotomic field visualization
ax4 = fig.add_subplot(gs[1, 1], projection='3d')  # E8 lattice projection (3D view)

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

# TSAMS-specific functions for prime generation
def cyclotomic_polynomial(n, x):
    """Calculate the value of the nth cyclotomic polynomial at x"""
    if n == 1:
        return x - 1
    
    result = 1
    for d in range(1, n+1):
        if n % d == 0:
            result *= (x**(d) - 1)**(mobius(n//d))
    
    return result

def cyclotomic_sieve(limit, conductor=8):
    """Implementation of the TSAMS cyclotomic sieve algorithm"""
    primes = []
    
    # Handle small primes separately
    if limit >= 2:
        primes.append(2)
    if limit >= 3:
        primes.append(3)
    if limit >= 5:
        primes.append(5)
    if limit >= 7:
        primes.append(7)
    
    # Apply cyclotomic sieve for larger numbers
    for n in range(11, limit + 1, 2):  # Only check odd numbers
        if n % 3 == 0 or n % 5 == 0 or n % 7 == 0:
            continue
            
        # Calculate cyclotomic residue
        residue = cyclotomic_polynomial(conductor, n) % n
        
        # If residue is non-zero and passes basic primality check
        if residue != 0 and is_probable_prime(n):
            primes.append(n)
    
    return primes

def e8_projection(n):
    """Project a number onto the E8 lattice"""
    # E8 moduli based on the E8 root system
    moduli = [30, 12, 20, 15, 24, 40, 60, 35]
    
    # Calculate remainders
    remainders = [n % m for m in moduli]
    
    # Calculate 3D projection for visualization (actual E8 is 8-dimensional)
    x = (remainders[0] / moduli[0] + remainders[1] / moduli[1]) / 2
    y = (remainders[2] / moduli[2] + remainders[3] / moduli[3]) / 2
    z = (remainders[4] / moduli[4] + remainders[5] / moduli[5]) / 2
    
    # Calculate the E8 quadratic form value (simplified for visualization)
    q_value = sum((r / m)**2 for r, m in zip(remainders, moduli))
    
    return x, y, z, q_value

def is_probable_prime(n):
    """A simplified primality test for visualization purposes"""
    # In a real implementation, this would use the TSAMS-specific primality test
    # For visualization, we'll use a simple check
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True

def cyclotomic_field_visualization(n, conductor=8):
    """Create a visualization of n in the cyclotomic field"""
    # Generate points on the complex plane representing the cyclotomic field
    theta = np.linspace(0, 2*np.pi, 100)
    roots_of_unity = [complex(np.cos(2*np.pi*k/conductor), np.sin(2*np.pi*k/conductor)) 
                      for k in range(conductor)]
    
    # Calculate the cyclotomic mapping of n
    cyclotomic_value = sum(cmath.exp(2j * np.pi * k * n / conductor) 
                          for k in range(conductor) if math.gcd(k, conductor) == 1)
    
    # Normalize for visualization
    magnitude = abs(cyclotomic_value)
    normalized_value = cyclotomic_value / magnitude if magnitude > 0 else 0
    
    return roots_of_unity, normalized_value

def tibedo_prime_generator(n):
    """
    Simulated version of the Tibedo Prime Generator Formula
    In a real implementation, this would use the actual formula
    """
    # For visualization purposes, we'll use a simple prime counter
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
    ax3.clear()
    ax4.clear()
    
    # Set titles
    ax1.set_title('Prime Number Grid')
    ax2.set_title('Prime Number Distribution')
    ax3.set_title('Cyclotomic Field Representation')
    ax4.set_title('E8 Lattice Projection')
    
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
    
    # Plot the grid (Subplot 1)
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
    
    # Plot the prime distribution (Subplot 2)
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
    
    # Plot the cyclotomic field visualization (Subplot 3)
    if frame > 0:
        # Draw the unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='gray')
        ax3.add_artist(circle)
        
        # Plot the roots of unity
        conductor = 8  # Using conductor 8 for visualization
        roots_of_unity, cyclotomic_value = cyclotomic_field_visualization(primes[-1], conductor)
        
        # Plot roots of unity
        for root in roots_of_unity:
            ax3.plot(root.real, root.imag, 'o', color='blue', markersize=4)
        
        # Plot the cyclotomic mapping of the prime
        if cyclotomic_value != 0:
            ax3.arrow(0, 0, cyclotomic_value.real, cyclotomic_value.imag, 
                     head_width=0.05, head_length=0.1, fc='red', ec='red')
        
        # Add text showing the cyclotomic residue
        residue = cyclotomic_polynomial(conductor, primes[-1]) % primes[-1]
        ax3.text(0.05, 0.95, f"Cyclotomic Residue: {residue}", transform=ax3.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax3.set_xlim(-1.5, 1.5)
        ax3.set_ylim(-1.5, 1.5)
        ax3.grid(True)
        ax3.set_aspect('equal')
    
    # Plot the E8 lattice projection (Subplot 4)
    if frame > 0:
        # Plot non-primes in gray
        for i in range(10, 100):
            if i not in primes:
                x, y, z, _ = e8_projection(i)
                ax4.scatter(x, y, z, color='gray', alpha=0.2, s=10)
        
        # Plot primes in green
        for p in primes:
            x, y, z, q_value = e8_projection(p)
            ax4.scatter(x, y, z, color='green', s=30)
        
        # Highlight the latest prime
        x, y, z, q_value = e8_projection(primes[-1])
        ax4.scatter(x, y, z, color='red', s=100)
        
        # Add text showing the E8 quadratic form value
        ax4.text2D(0.05, 0.05, f"E8 Q-value: {q_value:.4f}", transform=ax4.transAxes,
                 fontsize=10, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')
        ax4.set_title('E8 Lattice Projection (3D)')
    
    return ax1, ax2, ax3, ax4

# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(31), interval=1000, blit=False)

# Save as MP4
ani.save('tsams_prime_generator_visualization.mp4', writer='ffmpeg', fps=1, dpi=150)

# Display final frame
plt.tight_layout()
plt.savefig('tsams_prime_generator_final_frame.png', dpi=150)
print("Visualization complete. Video saved as 'tsams_prime_generator_visualization.mp4'")