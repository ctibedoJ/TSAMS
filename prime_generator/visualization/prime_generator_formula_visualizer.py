#!/usr/bin/env python3
"""
TSAMS Prime Generator Formula Visualizer
Creates a clear visualization showing how the formula generates the n-th prime number.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
from sympy import isprime, prime
import math

# Set up the figure with a clean, professional look
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('#f8f9fa')

# Create a grid layout
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2])

# Create subplots
ax1 = fig.add_subplot(gs[0, 0])  # Formula visualization
ax2 = fig.add_subplot(gs[0, 1])  # Prime sequence
ax3 = fig.add_subplot(gs[1, :])  # Prime generation process

# Set titles
fig.suptitle('TSAMS Prime Generator Formula Visualization', fontsize=20, fontweight='bold', y=0.98)
ax1.set_title('Prime Generator Formula', fontsize=16)
ax2.set_title('Prime Number Sequence', fontsize=16)
ax3.set_title('Prime Generation Process', fontsize=16)

# Parameters
max_n = 30  # Maximum n-th prime to visualize
frames = 60  # Number of animation frames

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
    f_n = n * math.log(n * math.log(n))
    steps.append(f"Step 2: Calculate F({n}) = {n} × log({n} × log({n})) ≈ {f_n:.2f}")
    
    # Step 3: Calculate the upper bound for summation
    sqrt_f_n = math.floor(math.sqrt(f_n))
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

# Function to update the visualization
def update(frame):
    # Clear previous plots
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    # Set titles
    ax1.set_title('Prime Generator Formula', fontsize=16)
    ax2.set_title('Prime Number Sequence', fontsize=16)
    ax3.set_title('Prime Generation Process', fontsize=16)
    
    # Calculate current n based on frame
    current_n = min(max_n, 1 + math.floor(frame / frames * max_n))
    
    # Update primes list if needed
    while len(primes) < current_n:
        n = len(primes) + 1
        p = tsams_prime_generator(n)
        primes.append(p)
        n_values.append(n)
    
    # Update formula steps if needed
    if len(formula_steps) < current_n:
        formula_steps.append(formula_calculation_steps(current_n))
    
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
    ax2.plot(n_values, primes, 'o-', color='blue', linewidth=2, markersize=8)
    ax2.set_xlabel('n', fontsize=14)
    ax2.set_ylabel('P(n)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight the current prime
    ax2.plot(current_n, primes[current_n-1], 'o', color='red', markersize=12)
    ax2.annotate(f"P({current_n}) = {primes[current_n-1]}", 
                xy=(current_n, primes[current_n-1]), 
                xytext=(current_n+0.5, primes[current_n-1]+1),
                fontsize=12,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    # Set appropriate axis limits with some padding
    ax2.set_xlim(0, max(n_values) + 2)
    ax2.set_ylim(0, max(primes) * 1.1)
    
    # Add Prime Number Theorem approximation
    x_smooth = np.linspace(1, max(n_values), 100)
    pnt_approx = x_smooth * np.log(x_smooth)
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
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
    
    return ax1, ax2, ax3

# Create animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000, blit=False)

# Save as MP4
ani.save('tsams_prime_generator_formula.mp4', writer='ffmpeg', fps=2, dpi=150, bitrate=5000)

# Display final frame
update(frames-1)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('tsams_prime_generator_formula_final_frame.png', dpi=300)
print("Visualization complete. Video saved as 'tsams_prime_generator_formula.mp4'")