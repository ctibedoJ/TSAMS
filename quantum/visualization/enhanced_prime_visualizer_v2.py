#!/usr/bin/env python3
"""
Advanced TSAMS Prime Generator Visualizer v2
Creates a high-quality 3D visualization of the Tibedo Prime Generator Formula,
with enhanced visuals for Cyclotomic Rings and Prime Indexed Ring Pairings.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors
from mpl_toolkits.mplot3d import Axes3D
from sympy import isprime, mobius, totient
import math
import cmath
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Create custom colormaps for enhanced visual appeal
prime_cmap = LinearSegmentedColormap.from_list('prime_cmap', 
                                              ['#00008B', '#0000FF', '#1E90FF', '#00BFFF'], N=256)
pairing_cmap = LinearSegmentedColormap.from_list('pairing_cmap', 
                                               ['#006400', '#00FF00', '#7FFF00', '#ADFF2F'], N=256)
cyclotomic_cmap = LinearSegmentedColormap.from_list('cyclotomic_cmap', 
                                                  ['#8B0000', '#FF0000', '#FF4500', '#FFA500'], N=256)

# Create a custom 3D arrow class for better visualizations
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

# Set up the figure with 3D plot - higher DPI and quality settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
fig = plt.figure(figsize=(16, 9), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')
ax.xaxis.line.set_color('white')
ax.yaxis.line.set_color('white')
ax.zaxis.line.set_color('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')
ax.grid(True, linestyle='--', alpha=0.3, color='white')

fig.suptitle('TSAMS Prime Generator: Advanced Cyclotomic Ring & Prime Pairing Visualization', 
             fontsize=18, color='white', y=0.98)

# Parameters
max_n = 200  # Maximum number to visualize
conductor = 12  # Cyclotomic field conductor
frames = 60  # Number of animation frames (1 minute at 1 fps)

# Initialize data structures
primes = []
non_primes = []
galois_pairs = []
time_sync = 0

# Narration text for each frame
narration = [
    "Welcome to the enhanced TSAMS Prime Generator visualization. This shows prime numbers in cyclotomic rings and their pairings.",
    "The BLUE DOTS represent prime numbers projected into 3D space using E8 lattice coordinates.",
    "The GREEN DOTS show composite numbers, revealing the contrast between prime and non-prime structures.",
    "The YELLOW DOTS highlight twin primes - pairs of primes that differ by only 2.",
    "The RED STARS represent prime numbers projected onto a semi-circular lattice structure.",
    "The CYAN LINES represent Prime Indexed Ring Pairings - mathematical relationships between consecutive primes.",
    "The ORANGE RINGS show Cyclotomic Ring structures - algebraic constructs fundamental to the TSAMS Prime Generator.",
    "As we rotate through the visualization, notice how the Cyclotomic Rings create orbital patterns around prime clusters.",
    "The Time Sync parameter modulates the visualization, showing how prime distributions evolve dynamically.",
    "Prime numbers are not randomly distributed - they follow specific patterns revealed by cyclotomic field theory.",
    "The TSAMS Prime Generator Formula uses these patterns to efficiently identify and generate prime numbers.",
    "Notice how the Prime Indexed Ring Pairings connect primes with similar residue properties.",
    "The brightness of each pairing indicates the strength of the mathematical relationship between those primes.",
    "Cyclotomic Rings represent the algebraic structure of roots of unity in number fields.",
    "The E8 lattice projection reveals how primes cluster in specific regions of higher-dimensional space.",
    "Twin primes (YELLOW) form special patterns within the larger prime distribution.",
    "The semi-circular lattice periods correspond to important cycles in the distribution of prime numbers.",
    "Prime Indexed Ring Pairings reveal deep connections between seemingly unrelated prime numbers.",
    "The TSAMS approach combines advanced number theory with geometric visualization to reveal hidden structures.",
    "Observe how certain viewing angles reveal alignment patterns that would be invisible in traditional representations.",
    "The Cyclotomic Rings demonstrate how prime numbers relate to roots of unity in complex number fields.",
    "Prime numbers form the building blocks of all integers through the Fundamental Theorem of Arithmetic.",
    "The TSAMS Prime Generator uses these mathematical structures to efficiently identify all prime numbers.",
    "Notice how the dynamic coupling between primes changes as we move through mathematical space.",
    "Thank you for exploring the enhanced TSAMS Prime Generator visualization."
]

# Ensure we have enough narration text for all frames
while len(narration) < frames:
    narration.extend(narration[:frames-len(narration)])
narration = narration[:frames]  # Trim if we have too many

# TSAMS-specific functions
def cyclotomic_polynomial(n, x):
    """Calculate the value of the nth cyclotomic polynomial at x"""
    if n == 1:
        return x - 1
    
    result = 1
    for d in range(1, n+1):
        if n % d == 0:
            result *= (x**(d) - 1)**(mobius(n//d))
    
    return result

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

def cyclotomic_ring(n, time_factor=0, radius=1.0, points=50):
    """Generate a cyclotomic ring visualization for a number n"""
    t = time_factor * 2 * np.pi
    
    # Calculate ring properties based on number properties
    if isprime(n):
        # Prime numbers get special ring properties
        ring_radius = radius * (0.8 + 0.2 * np.sin(t/3))
        thickness = 0.08 + 0.03 * np.sin(t/5)
        center_x, center_y, center_z = e8_projection(n, time_factor)
    else:
        # Composite numbers get different properties
        ring_radius = radius * 0.6
        thickness = 0.05
        center_x, center_y, center_z = e8_projection(n, time_factor)
    
    # Generate the ring points
    theta = np.linspace(0, 2*np.pi, points)
    
    # Calculate orientation angles based on number properties
    phi1 = (n % 7) * np.pi/7 + t/5
    phi2 = (n % 11) * np.pi/11 + t/7
    
    # Generate ring coordinates with orientation
    ring_x = []
    ring_y = []
    ring_z = []
    
    for th in theta:
        # Basic ring in xy-plane
        x = ring_radius * np.cos(th)
        y = ring_radius * np.sin(th)
        z = 0
        
        # Apply rotations for 3D orientation
        # Rotation around x-axis
        y_rot = y * np.cos(phi1) - z * np.sin(phi1)
        z_rot = y * np.sin(phi1) + z * np.cos(phi1)
        y, z = y_rot, z_rot
        
        # Rotation around y-axis
        x_rot = x * np.cos(phi2) + z * np.sin(phi2)
        z_rot = -x * np.sin(phi2) + z * np.cos(phi2)
        x, z = x_rot, z_rot
        
        # Add to center position
        ring_x.append(center_x + x)
        ring_y.append(center_y + y)
        ring_z.append(center_z + z)
    
    return ring_x, ring_y, ring_z, thickness

def is_twin_prime(n):
    """Check if n is part of a twin prime pair"""
    return (isprime(n) and isprime(n+2)) or (isprime(n) and isprime(n-2))

def generate_primes(limit):
    """Generate all primes up to the limit"""
    primes = []
    for n in range(2, limit + 1):
        if isprime(n):
            primes.append(n)
    return primes

# Pre-compute primes and identify twin primes
all_primes = generate_primes(max_n)
twin_primes = [p for p in all_primes if is_twin_prime(p)]

# Function to update the visualization
def update(frame):
    global time_sync
    
    # Clear previous plots
    ax.clear()
    
    # Set labels and limits
    ax.set_xlabel('X-axis (E8 Projection)', fontsize=12, color='white')
    ax.set_ylabel('Y-axis (E8 Projection)', fontsize=12, color='white')
    ax.set_zlabel('Z-axis (E8 Projection)', fontsize=12, color='white')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.grid(True, linestyle='--', alpha=0.3, color='white')
    
    # Update time synchronization factor
    time_sync = frame / frames
    
    # Plot primes in the E8 lattice (BLUE DOTS)
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
        prime_scatter = ax.scatter(xs, ys, zs, c=prime_colors, cmap=prime_cmap, 
                                  s=80, alpha=0.8, marker='o', 
                                  label='Prime Numbers (E8 Projection)')
    
    # Plot composite numbers (GREEN DOTS)
    composite_points = []
    for n in range(4, max_n):
        if not isprime(n):
            x, y, z = e8_projection(n, time_sync)
            composite_points.append((x, y, z))
    
    if composite_points:
        xs, ys, zs = zip(*composite_points)
        ax.scatter(xs, ys, zs, color='green', s=40, alpha=0.4, marker='o',
                  label='Composite Numbers')
    
    # Plot twin primes (YELLOW DOTS)
    twin_points = []
    for p in twin_primes:
        x, y, z = e8_projection(p, time_sync)
        twin_points.append((x, y, z))
    
    if twin_points:
        xs, ys, zs = zip(*twin_points)
        ax.scatter(xs, ys, zs, color='yellow', s=100, alpha=0.8, marker='*',
                  label='Twin Primes')
    
    # Plot semi-circular lattice for selected primes (RED STARS)
    selected_primes = [p for p in all_primes if p <= 100]
    lattice_points = []
    
    for p in selected_primes:
        lattice_coord = semi_circular_lattice(p, time_sync)
        if lattice_coord:
            lattice_points.append((p, lattice_coord))
    
    # Draw lattice connections
    if lattice_points:
        # Extract coordinates for plotting
        prime_values = [p for p, _ in lattice_points]
        coords = [coord for _, coord in lattice_points]
        xs, ys, zs = zip(*coords)
        
        # Plot the points (RED STARS)
        ax.scatter(xs, ys, zs, c='red', s=120, alpha=0.9, marker='*',
                  edgecolor='white', linewidth=0.5,
                  label='Prime Lattice Points')
        
        # Add small labels for prime values
        for i, (p, (x, y, z)) in enumerate(zip(prime_values, coords)):
            if i % 5 == 0:  # Label every 5th prime to avoid clutter
                ax.text(x, y, z, f"{p}", color='white', fontsize=8, 
                       ha='center', va='bottom')
        
        # Draw the semi-circular lattice structure with enhanced visuals
        for i in range(len(lattice_points)-1):
            _, (x1, y1, z1) = lattice_points[i]
            _, (x2, y2, z2) = lattice_points[i+1]
            
            # Create a curved line between consecutive points
            curve_points = 30  # More points for smoother curves
            t = np.linspace(0, 1, curve_points)
            
            # Generate a semi-circular arc
            curve_x = x1 + (x2 - x1) * t
            curve_y = y1 + (y2 - y1) * t
            # Add some curvature in z direction
            curve_z = z1 + (z2 - z1) * t + 0.5 * np.sin(np.pi * t)
            
            # Plot the curve with enhanced color gradient
            for j in range(curve_points-1):
                color = plt.cm.cool(j / curve_points)
                ax.plot(curve_x[j:j+2], curve_y[j:j+2], curve_z[j:j+2], 
                       color=color, linewidth=2.5, alpha=0.7)
    
    # Plot Prime Indexed Ring Pairings (CYAN LINES)
    if len(selected_primes) >= 2:
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
                    # Use a gradient color based on pairing strength
                    color = plt.cm.plasma(pairing * 5)  # Scale for better color range
                    
                    # Create a fancy 3D arrow for the pairing
                    arrow = Arrow3D([x1, x2], [y1, y2], [z1, z2],
                                   mutation_scale=15, lw=2+pairing*15, 
                                   arrowstyle='->', color=color, alpha=0.7)
                    ax.add_artist(arrow)
                    
                    # Add a small label showing the pairing strength
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    mid_z = (z1 + z2) / 2
                    if i % 3 == 0:  # Label every 3rd pairing to avoid clutter
                        ax.text(mid_x, mid_y, mid_z, f"{pairing:.2f}", 
                               color='cyan', fontsize=8, ha='center', va='bottom',
                               bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
    
    # Plot Cyclotomic Rings (ORANGE RINGS)
    # Select a subset of primes for cyclotomic rings to avoid clutter
    ring_primes = [p for p in all_primes if p <= 50 and p % 4 == 1]  # Primes of form 4k+1
    
    for p in ring_primes:
        ring_x, ring_y, ring_z, thickness = cyclotomic_ring(p, time_sync, radius=0.5)
        
        # Plot the ring with a gradient color
        points = len(ring_x)
        for i in range(points-1):
            # Use a color from the cyclotomic colormap
            color_idx = (i / points) * 0.8 + 0.2  # Scale to use most of the colormap
            color = cyclotomic_cmap(color_idx)
            
            ax.plot(ring_x[i:i+2], ring_y[i:i+2], ring_z[i:i+2], 
                   color=color, linewidth=3+thickness*20, alpha=0.7)
        
        # Connect the last point to the first to close the ring
        color = cyclotomic_cmap(0.2)
        ax.plot([ring_x[-1], ring_x[0]], [ring_y[-1], ring_y[0]], [ring_z[-1], ring_z[0]], 
               color=color, linewidth=3+thickness*20, alpha=0.7)
        
        # Add a label for the cyclotomic ring
        if p % 12 == 1:  # Label only some rings to avoid clutter
            ax.text(ring_x[0], ring_y[0], ring_z[0], f"Φ({p})", 
                   color='orange', fontsize=9, ha='center', va='bottom',
                   bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
    
    # Add legend with custom colors
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                  markersize=10, label='Prime Numbers (Blue)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                  markersize=10, alpha=0.7, label='Composite Numbers (Green)'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow', 
                  markersize=12, label='Twin Primes (Yellow)'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                  markersize=12, label='Prime Lattice Points (Red)'),
        plt.Line2D([0], [0], color='cyan', lw=2, label='Prime Ring Pairings (Cyan)'),
        plt.Line2D([0], [0], color='orange', lw=2, label='Cyclotomic Rings (Orange)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
             bbox_to_anchor=(0.98, 0.98), fontsize=9, 
             facecolor='black', edgecolor='white', framealpha=0.7)
    
    # Add time-sync indicator
    time_text = f"Time Sync: {time_sync:.2f}"
    ax.text2D(0.05, 0.95, time_text, transform=ax.transAxes,
             fontsize=12, verticalalignment='top', color='white',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white'))
    
    # Add formula visualization
    formula_text = "P(n) = 1 + Σ_{k=1}^{⌊√F(n)⌋} ⌊cos²(π·Γ(n,k)/k)⌋"
    ax.text2D(0.05, 0.90, formula_text, transform=ax.transAxes,
             fontsize=10, verticalalignment='top', color='white',
             bbox=dict(boxstyle='round', facecolor='navy', alpha=0.7, edgecolor='white'))
    
    # Add narration text
    if frame < len(narration):
        narration_text = narration[frame]
        ax.text2D(0.05, 0.05, narration_text, transform=ax.transAxes,
                 fontsize=12, verticalalignment='bottom', color='white',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='white'))
    
    # Rotate view for 3D effect - smoother rotation
    elevation = 20 + 10 * np.sin(frame / frames * 2 * np.pi)
    azimuth = frame * 6 % 360  # 6 degrees per frame
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Add a subtle lighting effect
    ax.set_facecolor('black')  # Set background to black
    
    return ax,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000, blit=False)

# Save as MP4 with higher quality
ani.save('tsams_prime_generator_enhanced_v2.mp4', writer='ffmpeg', fps=1, dpi=150, bitrate=8000)

# Display final frame
plt.tight_layout()
plt.savefig('tsams_prime_generator_enhanced_v2_final_frame.png', dpi=300)
print("Enhanced 3D Visualization v2 complete. Video saved as 'tsams_prime_generator_enhanced_v2.mp4'")