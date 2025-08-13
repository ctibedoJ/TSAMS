#!/usr/bin/env python3
"""
TSAMS 3D Prime Generator Visualizer
Creates an advanced 3D visualization of the Tibedo Prime Generator Formula,
showing prime-indexed Galois pairings and semi-circular lattice periods.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D
from sympy import isprime, mobius, totient
import math
import cmath

# Set up the figure with 3D plot
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
fig.suptitle('TSAMS Prime Generator: 3D Galois Pairing & Lattice Visualization', fontsize=16)

# Parameters
max_n = 200  # Maximum number to visualize
conductor = 12  # Cyclotomic field conductor
frames = 120  # Number of animation frames

# Initialize data structures
primes = []
non_primes = []
galois_pairs = []
time_sync = 0

# TSAMS-specific functions
def cyclotomic_residue(n, conductor):
    """Calculate the cyclotomic residue of n with given conductor"""
    residue = 1
    for k in range(1, conductor):
        if math.gcd(k, conductor) == 1:
            residue *= (n - np.exp(2j * np.pi * k / conductor))
    return abs(residue % n)

def galois_pairing(p, q):
    """Calculate the Galois pairing between two primes"""
    if not (isprime(p) and isprime(q)):
        return 0
    
    # Calculate pairing using cyclotomic fields
    pairing = 0
    for k in range(1, min(p, q)):
        if p % k == 0 and q % k == 0:
            continue  # Skip common factors (should only be 1 for primes)
        pairing += np.sin(2 * np.pi * k / p) * np.sin(2 * np.pi * k / q)
    
    return pairing / (np.log(p) * np.log(q))

def e8_projection(n, time_factor=0):
    """Project a number onto a time-varying E8 lattice"""
    # E8 moduli based on the E8 root system
    moduli = [30, 12, 20, 15, 24, 40, 60, 35]
    
    # Calculate remainders
    remainders = [n % m for m in moduli]
    
    # Add time variation to the projection
    t = time_factor * 2 * np.pi
    
    # Calculate 3D projection with time variation
    x = np.cos(t/7) * remainders[0]/moduli[0] + np.sin(t/11) * remainders[1]/moduli[1]
    y = np.sin(t/5) * remainders[2]/moduli[2] + np.cos(t/13) * remainders[3]/moduli[3]
    z = np.cos(t/3) * remainders[4]/moduli[4] + np.sin(t/17) * remainders[5]/moduli[5]
    
    # Scale for better visualization
    scale = 2.0
    return scale * x, scale * y, scale * z

def semi_circular_lattice(p, time_factor=0):
    """Generate semi-circular lattice coordinates for a prime"""
    if not isprime(p):
        return None
    
    # Find position in the sequence of primes
    prime_idx = len([n for n in range(2, p+1) if isprime(n)])
    
    # Calculate base angle based on prime index
    base_angle = prime_idx * (2 * np.pi / 30)  # 30 is arbitrary divisor for spacing
    
    # Add time variation
    t = time_factor * 2 * np.pi
    angle = base_angle + 0.2 * np.sin(t)
    
    # Calculate radius based on prime value with time variation
    radius = np.log(p) * (1 + 0.1 * np.sin(t/3))
    
    # Generate semi-circular coordinates
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = 0.1 * p * np.sin(t/5 + prime_idx/10)  # Height varies with time
    
    return x, y, z

def generate_primes(limit):
    """Generate all primes up to the limit"""
    primes = []
    for n in range(2, limit + 1):
        if isprime(n):
            primes.append(n)
    return primes

# Pre-compute primes
all_primes = generate_primes(max_n)

# Initialize scatter plots
prime_scatter = None
non_prime_scatter = None
galois_scatter = None
lattice_lines = []

# Function to update the visualization
def update(frame):
    global prime_scatter, non_prime_scatter, galois_scatter, lattice_lines, time_sync
    
    # Clear previous plots
    ax.clear()
    
    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    
    # Update time synchronization factor
    time_sync = frame / frames
    
    # Plot primes in the E8 lattice
    prime_points = []
    prime_colors = []
    
    for p in all_primes:
        # Get E8 projection coordinates
        x, y, z = e8_projection(p, time_sync)
        prime_points.append((x, y, z))
        
        # Color based on prime residue classes
        residue = p % 30  # 30 is product of first 3 primes
        prime_colors.append(residue / 30)
    
    # Convert to arrays for plotting
    if prime_points:
        xs, ys, zs = zip(*prime_points)
        prime_scatter = ax.scatter(xs, ys, zs, c=prime_colors, cmap='viridis', 
                                  s=50, alpha=0.8, marker='o')
    
    # Plot semi-circular lattice for selected primes
    selected_primes = [p for p in all_primes if p <= 100]
    lattice_points = []
    
    for p in selected_primes:
        lattice_coord = semi_circular_lattice(p, time_sync)
        if lattice_coord:
            lattice_points.append(lattice_coord)
    
    # Draw lattice connections
    if lattice_points:
        xs, ys, zs = zip(*lattice_points)
        
        # Plot the points
        ax.scatter(xs, ys, zs, c='red', s=100, alpha=0.7, marker='*')
        
        # Draw the semi-circular lattice structure
        for i in range(len(lattice_points)-1):
            x1, y1, z1 = lattice_points[i]
            x2, y2, z2 = lattice_points[i+1]
            
            # Create a curved line between consecutive points
            curve_points = 20
            t = np.linspace(0, 1, curve_points)
            
            # Generate a semi-circular arc
            curve_x = x1 + (x2 - x1) * t
            curve_y = y1 + (y2 - y1) * t
            # Add some curvature in z direction
            curve_z = z1 + (z2 - z1) * t + 0.5 * np.sin(np.pi * t)
            
            # Plot the curve with color gradient
            for j in range(curve_points-1):
                color = plt.cm.cool(j / curve_points)
                ax.plot(curve_x[j:j+2], curve_y[j:j+2], curve_z[j:j+2], 
                       color=color, linewidth=2, alpha=0.7)
    
    # Plot Galois pairings
    if len(selected_primes) >= 2:
        galois_lines = []
        
        # Only show pairings for some primes to avoid clutter
        for i in range(len(selected_primes)-1):
            p = selected_primes[i]
            q = selected_primes[i+1]
            
            # Get lattice coordinates
            p_coord = semi_circular_lattice(p, time_sync)
            q_coord = semi_circular_lattice(q, time_sync)
            
            if p_coord and q_coord:
                # Calculate pairing strength
                pairing = abs(galois_pairing(p, q))
                
                if pairing > 0.01:  # Only show significant pairings
                    x1, y1, z1 = p_coord
                    x2, y2, z2 = q_coord
                    
                    # Draw pairing line with thickness based on strength
                    line = ax.plot([x1, x2], [y1, y2], [z1, z2], 
                                  color='cyan', linewidth=pairing*10, alpha=0.5)
                    galois_lines.append(line)
    
    # Add time-sync indicator
    time_text = f"Time Sync: {time_sync:.2f}"
    ax.text2D(0.05, 0.95, time_text, transform=ax.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add formula visualization
    formula_text = "P(n) = 1 + Σ_{k=1}^{⌊√F(n)⌋} ⌊cos²(π·Γ(n,k)/k)⌋"
    ax.text2D(0.05, 0.90, formula_text, transform=ax.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Rotate view for 3D effect
    ax.view_init(elev=20, azim=frame % 360)
    
    return ax,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

# Save as MP4
ani.save('tsams_3d_prime_visualization.mp4', writer='ffmpeg', fps=24, dpi=150, bitrate=5000)

# Display final frame
plt.tight_layout()
plt.savefig('tsams_3d_prime_final_frame.png', dpi=150)
print("3D Visualization complete. Video saved as 'tsams_3d_prime_visualization.mp4'")