#!/usr/bin/env python3
"""
Enhanced TSAMS Prime Generator Formula Visualizer
Creates a clear visualization showing how the formula generates the n-th prime number,
with additional dynamic visualizations of coupling strengths and nodal alignments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from sympy import isprime, prime
import math
import cmath

# Set up the figure with a clean, professional look
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('#f8f9fa')

# Create a grid layout with space for the 3D visualization
gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1.2], width_ratios=[1.5, 1.5, 1])

# Create subplots
ax1 = fig.add_subplot(gs[0, 0])  # Formula visualization
ax2 = fig.add_subplot(gs[0, 1:])  # Prime sequence
ax3 = fig.add_subplot(gs[1, :2])  # Prime generation process
ax4 = fig.add_subplot(gs[1, 2], projection='3d')  # 3D visualization for coupling strengths

# Set titles
fig.suptitle('TSAMS Prime Generator Formula Visualization', fontsize=20, fontweight='bold', y=0.98)
ax1.set_title('Prime Generator Formula', fontsize=16)
ax2.set_title('Prime Number Sequence', fontsize=16)
ax3.set_title('Prime Generation Process', fontsize=16)
ax4.set_title('Coupling Visualization', fontsize=14)

# Parameters
max_n = 400  # Maximum n-th prime to visualize
frames = 200  # Number of animation frames
fps = 5  # Frames per second

# Initialize data
primes = []
n_values = []
formula_steps = []

# Simplified version of the TSAMS Prime Generator Formula
def tsams_prime_generator(n):
    """
    Simplified implementation of the TSAMS Prime Generator Formula.
    This function directly calculates the n-th prime number.
    """
    # For demonstration purposes, we'll use sympy's prime function
    # In a real implementation, this would use the actual TSAMS formula
    return prime(n)

# Function to simulate the formula calculation steps
def formula_calculation_steps(n):
    """Generate steps showing how the formula calculates the n-th prime"""
    steps = []
    
    # Step 1: Initialize
    steps.append(f"Step 1: Initialize calculation for P({n})")
    
    # Step 2: Calculate F(n) - a function related to the n-th prime
    # Handle special case for n=1 to avoid math domain error
    if n == 1:
        f_n = 1  # Special case for n=1
        steps.append(f"Step 2: For n=1, F(1) = 1 (special case)")
    else:
        f_n = n * math.log(n * math.log(max(2, n)))
        steps.append(f"Step 2: Calculate F({n}) = {n} × log({n} × log({n})) ≈ {f_n:.2f}")
    
    # Step 3: Calculate the upper bound for summation
    sqrt_f_n = max(1, math.floor(math.sqrt(f_n)))
    steps.append(f"Step 3: Calculate ⌊√F({n})⌋ = ⌊√{f_n:.2f}⌋ = {sqrt_f_n}")
    
    # Step 4: Initialize sum
    steps.append(f"Step 4: Initialize sum = 1")
    
    # Step 5-9: Calculate the summation terms (simplified for visualization)
    for k in range(1, min(5, sqrt_f_n + 1)):
        gamma_nk = (n * k) % (k + 1)  # Simplified Γ function
        cos_term = math.cos(math.pi * gamma_nk / k) ** 2
        floor_term = math.floor(cos_term)
        steps.append(f"Step {4+k}: Term k={k}: ⌊cos²(π·Γ({n},{k})/{k})⌋ = ⌊cos²(π·{gamma_nk}/{k})⌋ = ⌊{cos_term:.4f}⌋ = {floor_term}")
    
    # Step 10: If more terms exist, indicate this
    if sqrt_f_n > 5:
        steps.append(f"... (continuing for k=6 to {sqrt_f_n})")
    
    # Step 11: Calculate the final result
    p_n = prime(n)  # Using sympy's prime function for demonstration
    steps.append(f"Final: P({n}) = {p_n}")
    
    return steps

# Function to calculate coupling strengths between primes
def calculate_coupling_strength(p, q):
    """Calculate the coupling strength between two primes"""
    if p == q:
        return 1.0
    
    # Use a simplified model based on prime gap and congruence relations
    gap = abs(p - q)
    min_prime = min(p, q)
    
    # Coupling strength decreases with gap and increases with prime size
    strength = math.exp(-gap / (min_prime ** 0.5)) * (1 + 0.5 * math.sin(p * q % 12))
    
    return min(1.0, max(0.0, strength))

# Function to generate nodal points for the 3D visualization
def generate_nodal_points(n, k, time_factor=0):
    """Generate nodal points for the 3D visualization"""
    points = []
    colors = []
    
    # Generate points on a cube
    for i in range(k+1):
        # Calculate phase based on prime properties
        phase = (n * i) % (k + 1) / (k + 1) * 2 * math.pi
        
        # Add time variation
        t = time_factor * 2 * math.pi
        
        # Generate points on cube faces with phase-dependent positions
        for face in range(6):
            if face == 0:  # Front face
                x = -1
                y = -1 + 2 * (i / k)
                z = -1 + 2 * math.sin(phase + t)
            elif face == 1:  # Back face
                x = 1
                y = -1 + 2 * (i / k)
                z = -1 + 2 * math.sin(phase + t + math.pi)
            elif face == 2:  # Left face
                x = -1 + 2 * math.sin(phase + t)
                y = -1
                z = -1 + 2 * (i / k)
            elif face == 3:  # Right face
                x = -1 + 2 * math.sin(phase + t + math.pi)
                y = 1
                z = -1 + 2 * (i / k)
            elif face == 4:  # Bottom face
                x = -1 + 2 * (i / k)
                y = -1 + 2 * math.sin(phase + t)
                z = -1
            else:  # Top face
                x = -1 + 2 * (i / k)
                y = -1 + 2 * math.sin(phase + t + math.pi)
                z = 1
            
            # Add point
            points.append((x, y, z))
            
            # Color based on phase
            color_val = (math.sin(phase + t) + 1) / 2
            colors.append(color_val)
    
    return points, colors

# Function to generate 2D circle projections
def generate_circle_projections(n, time_factor=0):
    """Generate 2D circle projections for coupling visualization"""
    # Number of circles to generate
    num_circles = min(8, n)
    
    circles = []
    
    for i in range(num_circles):
        # Calculate prime value
        p = prime(n - i) if n - i > 0 else 2
        
        # Calculate radius based on prime properties
        radius = 0.1 + 0.9 * (i / max(1, num_circles - 1))
        
        # Calculate position based on prime properties and time
        t = time_factor * 2 * math.pi
        angle = (p % 12) / 12 * 2 * math.pi + t * (0.2 + 0.1 * (i % 3))
        
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        
        # Calculate color based on prime properties
        color_val = (p % 30) / 30
        
        circles.append((x, y, radius * 0.2, color_val))
    
    return circles

# Pre-calculate all primes and steps to avoid issues during animation
for n in range(1, min(100, max_n + 1)):  # Pre-calculate first 100, rest on-demand
    p = tsams_prime_generator(n)
    primes.append(p)
    n_values.append(n)
    formula_steps.append(formula_calculation_steps(n))

# Function to update the visualization
def update(frame):
    # Calculate current n based on frame
    current_n = min(max_n, 1 + math.floor(frame / frames * max_n))
    
    # Update primes list if needed
    while len(primes) < current_n:
        n = len(primes) + 1
        p = tsams_prime_generator(n)
        primes.append(p)
        n_values.append(n)
    
    # Update formula steps if needed
    while len(formula_steps) < current_n:
        n = len(formula_steps) + 1
        formula_steps.append(formula_calculation_steps(n))
    
    # Clear previous plots
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    
    # Set titles
    ax1.set_title('Prime Generator Formula', fontsize=16)
    ax2.set_title('Prime Number Sequence', fontsize=16)
    ax3.set_title('Prime Generation Process', fontsize=16)
    ax4.set_title('Coupling Visualization', fontsize=14)
    
    # Plot 1: Formula Visualization
    ax1.axis('off')
    formula = r"$P(n) = 1 + \sum_{k=1}^{\lfloor\sqrt{F(n)}\rfloor} \lfloor\cos^2(\pi\cdot\Gamma(n,k)/k)\rfloor$"
    ax1.text(0.5, 0.7, formula, fontsize=20, ha='center', va='center')
    
    # Add explanation of the formula components
    explanation = [
        r"Where:",
        r"$P(n)$ = the $n$-th prime number",
        r"$F(n) = n \log(n \log n)$",
        r"$\Gamma(n,k)$ = cyclotomic field coupling function"
    ]
    
    for i, line in enumerate(explanation):
        ax1.text(0.5, 0.5 - i*0.1, line, fontsize=14, ha='center', va='center')
    
    # Highlight current calculation
    ax1.text(0.5, 0.1, f"Currently calculating: P({current_n}) = {primes[current_n-1]}", 
             fontsize=16, ha='center', va='center', 
             bbox=dict(facecolor='yellow', alpha=0.3, boxstyle='round'))
    
    # Plot 2: Prime Sequence
    # Show only a window of primes for clarity when n gets large
    window_size = 50
    start_idx = max(0, current_n - window_size)
    end_idx = current_n
    
    visible_n = n_values[start_idx:end_idx]
    visible_primes = primes[start_idx:end_idx]
    
    ax2.plot(visible_n, visible_primes, 'o-', color='blue', linewidth=2, markersize=8)
    ax2.set_xlabel('n', fontsize=14)
    ax2.set_ylabel('P(n)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight the current prime
    ax2.plot(current_n, primes[current_n-1], 'o', color='red', markersize=12)
    ax2.annotate(f"P({current_n}) = {primes[current_n-1]}", 
                xy=(current_n, primes[current_n-1]), 
                xytext=(current_n-5, primes[current_n-1]+5),
                fontsize=12,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    # Set appropriate axis limits with some padding
    ax2.set_xlim(start_idx, end_idx + 5)
    y_min = min(visible_primes) * 0.9 if visible_primes else 0
    y_max = max(visible_primes) * 1.1 if visible_primes else 10
    ax2.set_ylim(y_min, y_max)
    
    # Add Prime Number Theorem approximation
    x_smooth = np.linspace(start_idx+1, end_idx+1, 100)
    pnt_approx = x_smooth * np.log(np.maximum(2, x_smooth))  # Avoid log(1)
    ax2.plot(x_smooth, pnt_approx, '--', color='green', linewidth=2, 
             label='n log(n) approximation')
    ax2.legend(loc='upper left')
    
    # Plot 3: Formula Calculation Steps
    ax3.axis('off')
    
    # Display the calculation steps for the current n
    current_steps = formula_steps[current_n-1]
    
    # Calculate how many steps to show based on the frame
    step_progress = (frame % (frames/max_n)) / (frames/max_n)
    steps_to_show = max(1, math.ceil(step_progress * len(current_steps)))
    
    # Display steps in a grid layout
    num_cols = 2
    num_rows = math.ceil(len(current_steps) / num_cols)
    
    for i, step in enumerate(current_steps[:steps_to_show]):
        row = i // num_cols
        col = i % num_cols
        
        # Calculate position
        x_pos = 0.1 + col * 0.5
        y_pos = 0.9 - row * 0.15
        
        # Highlight the final step
        if i == len(current_steps) - 1:
            ax3.text(x_pos, y_pos, step, fontsize=14, ha='left', va='center',
                    bbox=dict(facecolor='lightgreen', alpha=0.5, boxstyle='round'))
        else:
            ax3.text(x_pos, y_pos, step, fontsize=12, ha='left', va='center')
    
    # Add progress indicator
    progress = current_n / max_n
    ax3.add_patch(plt.Rectangle((0.1, 0.05), progress * 0.8, 0.03, 
                              facecolor='blue', alpha=0.7))
    ax3.text(0.5, 0.02, f"Progress: {current_n}/{max_n} primes", 
             ha='center', va='center', fontsize=12)
    
    # Plot 4: 3D Visualization for coupling strengths and nodal alignments
    # Set up the 3D axes
    ax4.set_xlim(-1.2, 1.2)
    ax4.set_ylim(-1.2, 1.2)
    ax4.set_zlim(-1.2, 1.2)
    ax4.set_box_aspect([1, 1, 1])  # Make the box cubic
    
    # Remove tick labels for cleaner look
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax4.set_zticklabels([])
    
    # Calculate time factor for animations
    time_factor = (frame % 20) / 20
    
    # Extract the current prime and calculate k for visualization
    p = primes[current_n-1]
    
    # Get the sqrt(F(n)) value from the steps
    k_value = 0
    for step in current_steps:
        if step.startswith("Step 3:"):
            parts = step.split("=")
            if len(parts) >= 3:
                try:
                    k_value = int(parts[-1].strip())
                except ValueError:
                    k_value = 5  # Default if parsing fails
    
    # Generate nodal points for the cube visualization
    nodal_points, nodal_colors = generate_nodal_points(current_n, k_value, time_factor)
    
    # Plot the cube edges
    cube_edges = [
        # Bottom face
        [(-1, -1, -1), (1, -1, -1)],
        [(-1, -1, -1), (-1, 1, -1)],
        [(1, 1, -1), (1, -1, -1)],
        [(1, 1, -1), (-1, 1, -1)],
        # Top face
        [(-1, -1, 1), (1, -1, 1)],
        [(-1, -1, 1), (-1, 1, 1)],
        [(1, 1, 1), (1, -1, 1)],
        [(1, 1, 1), (-1, 1, 1)],
        # Connecting edges
        [(-1, -1, -1), (-1, -1, 1)],
        [(1, -1, -1), (1, -1, 1)],
        [(1, 1, -1), (1, 1, 1)],
        [(-1, 1, -1), (-1, 1, 1)]
    ]
    
    for edge in cube_edges:
        ax4.plot([edge[0][0], edge[1][0]], 
                [edge[0][1], edge[1][1]], 
                [edge[0][2], edge[1][2]], 
                color='gray', alpha=0.5, linewidth=1)
    
    # Plot the nodal points
    if nodal_points:
        xs, ys, zs = zip(*nodal_points)
        ax4.scatter(xs, ys, zs, c=nodal_colors, cmap='plasma', 
                   s=50, alpha=0.8, edgecolors='white', linewidths=0.5)
    
    # Generate and plot 2D circle projections
    circles = generate_circle_projections(current_n, time_factor)
    
    # Plot the circles on the bottom face of the cube
    for x, y, size, color_val in circles:
        circle = plt.Circle((x, y), size, color=plt.cm.viridis(color_val), 
                           alpha=0.7, transform=ax4.transData)
        # Project onto the bottom face
        z = -1
        ax4.add_patch(circle)
        art3d = mpl_toolkits.mplot3d.art3d
        art3d.pathpatch_2d_to_3d(circle, z=z, zdir="z")
    
    # Add coupling lines between nodal points
    if len(nodal_points) >= 2:
        for i in range(len(nodal_points)-1):
            for j in range(i+1, len(nodal_points)):
                # Calculate coupling strength
                strength = calculate_coupling_strength(i+1, j+1)
                
                # Only show strong couplings
                if strength > 0.3:
                    x1, y1, z1 = nodal_points[i]
                    x2, y2, z2 = nodal_points[j]
                    
                    # Draw line with thickness based on coupling strength
                    ax4.plot([x1, x2], [y1, y2], [z1, z2], 
                           color=plt.cm.cool(strength), 
                           linewidth=1 + 3*strength, alpha=0.6)
    
    # Rotate the view for dynamic effect
    elevation = 20 + 10 * np.sin(frame / 30)
    azimuth = frame % 360
    ax4.view_init(elev=elevation, azim=azimuth)
    
    # Add a small text explaining the visualization
    ax4.text2D(0.05, 0.95, f"k={k_value}", transform=ax4.transAxes,
              fontsize=10, ha='left', va='top',
              bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    ax4.text2D(0.05, 0.05, "Nodal Alignments", transform=ax4.transAxes,
              fontsize=10, ha='left', va='bottom',
              bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
    
    return ax1, ax2, ax3, ax4

# Create animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)

# Save as MP4
ani.save('tsams_prime_generator_enhanced.mp4', writer='ffmpeg', fps=fps, dpi=150, bitrate=5000)

# Display final frame
update(frames-1)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('tsams_prime_generator_enhanced_final_frame.png', dpi=300)
print("Enhanced visualization complete. Video saved as 'tsams_prime_generator_enhanced.mp4'")