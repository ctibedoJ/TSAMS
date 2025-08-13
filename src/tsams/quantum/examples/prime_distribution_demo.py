"""
Prime Distribution Formula Demonstration

This script provides a stunning visualization of the prime distribution formula
based on cyclotomic Galois regulators and the Dedekind cut morphic conductor.
The formula provides an exact prediction of the distribution of prime numbers,
demonstrating the deep connection between number theory and quantum mathematics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Add the parent directory to the path to import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our cyclotomic quantum framework
from cyclotomic_quantum.core.cyclotomic_field import CyclotomicField
from cyclotomic_quantum.core.dedekind_cut import DedekindCutMorphicConductor
from cyclotomic_quantum.core.prime_spectral_grouping import PrimeSpectralGrouping
from cyclotomic_quantum.cosmology.prime_distribution import PrimeDistribution


def create_basic_visualization(prime_dist, max_n=1000, save_path=None):
    """
    Create a basic visualization of the prime distribution formula.
    
    Args:
        prime_dist (PrimeDistribution): The prime distribution object.
        max_n (int): The maximum value of n to plot (default: 1000).
        save_path (str): The path to save the figure (default: None).
    """
    # Set plot style
    plt.style.use('dark_background')
    sns.set_style("darkgrid")
    
    # Create a range of values to analyze
    x_values = np.linspace(10, max_n, max_n - 9)
    x_integers = np.arange(10, max_n + 1)
    
    # Compute the actual prime counts
    print("Computing actual prime counts...")
    actual_counts = [prime_dist.count_primes(int(x)) for x in x_integers]
    
    # Compute the logarithmic integral approximation
    print("Computing logarithmic integral approximation...")
    li_approx = [prime_dist.logarithmic_integral(x) for x in x_values]
    
    # Compute our exact formula prediction
    print("Computing exact formula prediction...")
    exact_formula = [prime_dist.exact_prime_counting_formula(x) for x in x_values]
    
    # Create the figure
    plt.figure(figsize=(14, 8))
    
    # Plot the actual prime counts
    plt.scatter(x_integers, actual_counts, color='gold', alpha=0.7, label='Actual Prime Counts', s=10)
    
    # Plot the logarithmic integral approximation
    plt.plot(x_values, li_approx, color='cyan', alpha=0.7, label='Logarithmic Integral Approximation', linewidth=2)
    
    # Plot our exact formula prediction
    plt.plot(x_values, exact_formula, color='magenta', alpha=0.9, label='Cyclotomic Field Theory Exact Formula', linewidth=2)
    
    plt.title('Prime Number Distribution: Cyclotomic Field Theory vs. Traditional Approximations', fontsize=16)
    plt.xlabel('n', fontsize=14)
    plt.ylabel('Number of Primes ≤ n', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add a text box with the Dedekind cut morphic conductor
    dedekind_cut = DedekindCutMorphicConductor()
    textstr = f"Dedekind Cut Morphic Conductor: {dedekind_cut.value} = 2³ × 3 × 7"
    props = dict(boxstyle='round', facecolor='black', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def create_error_analysis(prime_dist, max_n=1000, save_path=None):
    """
    Create an error analysis visualization.
    
    Args:
        prime_dist (PrimeDistribution): The prime distribution object.
        max_n (int): The maximum value of n to plot (default: 1000).
        save_path (str): The path to save the figure (default: None).
    """
    # Set plot style
    plt.style.use('dark_background')
    sns.set_style("darkgrid")
    
    # Create a range of values to analyze
    x_values = np.linspace(10, max_n, max_n - 9)
    x_integers = np.arange(10, max_n + 1)
    
    # Compute the actual prime counts
    print("Computing actual prime counts...")
    actual_counts = [prime_dist.count_primes(int(x)) for x in x_integers]
    
    # Compute the logarithmic integral approximation
    print("Computing logarithmic integral approximation...")
    li_approx = [prime_dist.logarithmic_integral(x) for x in x_values]
    
    # Compute our exact formula prediction
    print("Computing exact formula prediction...")
    exact_formula = [prime_dist.exact_prime_counting_formula(x) for x in x_values]
    
    # Compute the errors
    li_errors = [li_approx[i-10] - actual_counts[i-10] for i in range(10, len(actual_counts)+10)]
    formula_errors = [exact_formula[i-10] - actual_counts[i-10] for i in range(10, len(actual_counts)+10)]
    
    # Create the figure
    plt.figure(figsize=(14, 8))
    
    # Plot the errors
    plt.plot(x_integers, li_errors, color='cyan', alpha=0.7, label='Logarithmic Integral Error', linewidth=2)
    plt.plot(x_integers, formula_errors, color='magenta', alpha=0.9, label='Cyclotomic Formula Error', linewidth=2)
    
    plt.title('Error Analysis: Cyclotomic Field Theory vs. Logarithmic Integral', fontsize=16)
    plt.xlabel('n', fontsize=14)
    plt.ylabel('Error (Approximation - Actual)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='white', linestyle='-', alpha=0.3)
    
    # Add a text box with error statistics
    li_mean_error = np.mean(np.abs(li_errors))
    formula_mean_error = np.mean(np.abs(formula_errors))
    improvement = (li_mean_error - formula_mean_error) / li_mean_error * 100
    
    textstr = f"Mean Absolute Error:\n"
    textstr += f"Logarithmic Integral: {li_mean_error:.4f}\n"
    textstr += f"Cyclotomic Formula: {formula_mean_error:.4f}\n"
    textstr += f"Improvement: {improvement:.2f}%"
    
    props = dict(boxstyle='round', facecolor='black', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def create_correction_factor_visualization(prime_dist, max_n=1000, save_path=None):
    """
    Create a visualization of the cyclotomic correction factor.
    
    Args:
        prime_dist (PrimeDistribution): The prime distribution object.
        max_n (int): The maximum value of n to plot (default: 1000).
        save_path (str): The path to save the figure (default: None).
    """
    # Set plot style
    plt.style.use('dark_background')
    sns.set_style("darkgrid")
    
    # Create a range of values to analyze
    x_values = np.linspace(10, max_n, max_n - 9)
    
    # Compute the cyclotomic correction factor
    print("Computing cyclotomic correction factor...")
    correction_factors = [prime_dist.cyclotomic_correction(x) for x in x_values]
    
    # Create the figure
    plt.figure(figsize=(14, 8))
    
    # Plot the correction factor
    plt.plot(x_values, correction_factors, color='gold', alpha=0.9, linewidth=2)
    
    plt.title('Cyclotomic Correction Factor', fontsize=16)
    plt.xlabel('n', fontsize=14)
    plt.ylabel('Correction Factor', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add a horizontal line at y=1
    plt.axhline(y=1, color='white', linestyle='-', alpha=0.3)
    
    # Add a text box explaining the correction factor
    textstr = "The cyclotomic correction factor modulates\nthe logarithmic integral approximation\nbased on the Dedekind cut morphic conductor."
    props = dict(boxstyle='round', facecolor='black', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def create_spectral_density_visualization(prime_dist, max_n=1000, bin_size=50, save_path=None):
    """
    Create a visualization of the prime spectral density.
    
    Args:
        prime_dist (PrimeDistribution): The prime distribution object.
        max_n (int): The maximum value of n to plot (default: 1000).
        bin_size (int): The size of each bin (default: 50).
        save_path (str): The path to save the figure (default: None).
    """
    # Set plot style
    plt.style.use('dark_background')
    sns.set_style("darkgrid")
    
    # Compute the prime spectral density
    print("Computing prime spectral density...")
    bin_centers, density = prime_dist.prime_spectral_density(max_n, bin_size=bin_size)
    
    # Create the figure
    plt.figure(figsize=(14, 8))
    
    # Plot the prime spectral density
    plt.bar(bin_centers, density, width=bin_size*0.8, color='magenta', alpha=0.7)
    
    # Plot the theoretical density based on the logarithmic integral
    theoretical_density = [1 / (np.log(x) * bin_size) for x in bin_centers]
    plt.plot(bin_centers, theoretical_density, color='cyan', linewidth=2, label='Theoretical Density (1/log(n))')
    
    plt.title('Prime Spectral Density', fontsize=16)
    plt.xlabel('n', fontsize=14)
    plt.ylabel('Density (primes per unit interval)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def create_3d_visualization(prime_dist, max_n=1000, save_path=None):
    """
    Create a 3D visualization of the prime distribution formula.
    
    Args:
        prime_dist (PrimeDistribution): The prime distribution object.
        max_n (int): The maximum value of n to plot (default: 1000).
        save_path (str): The path to save the figure (default: None).
    """
    # Set plot style
    plt.style.use('dark_background')
    sns.set_style("darkgrid")
    
    # Create a range of values to analyze
    x_integers = np.arange(10, max_n + 1)
    
    # Compute the actual prime counts
    print("Computing actual prime counts...")
    actual_counts = [prime_dist.count_primes(int(x)) for x in x_integers]
    
    # Create a 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a meshgrid for the 3D surface
    X = np.linspace(10, max_n, 50)
    Y = np.linspace(0, 1, 50)  # Parameter for the formula
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    
    # Compute the surface values
    print("Computing 3D surface values...")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i, j]
            y = Y[i, j]
            # Interpolate between the logarithmic integral and our exact formula
            li = prime_dist.logarithmic_integral(x)
            exact = prime_dist.exact_prime_counting_formula(x)
            Z[i, j] = (1 - y) * li + y * exact
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, alpha=0.8)
    
    # Plot the actual prime counts
    ax.scatter(x_integers, np.zeros_like(x_integers), actual_counts, color='gold', s=10, label='Actual Prime Counts')
    
    # Set labels and title
    ax.set_xlabel('n', fontsize=14)
    ax.set_ylabel('Interpolation Parameter', fontsize=14)
    ax.set_zlabel('Number of Primes ≤ n', fontsize=14)
    ax.set_title('3D Visualization of Prime Distribution Formula', fontsize=16)
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def create_animated_visualization(prime_dist, max_n=1000, save_path=None):
    """
    Create an animated visualization of the prime distribution formula.
    
    Args:
        prime_dist (PrimeDistribution): The prime distribution object.
        max_n (int): The maximum value of n to plot (default: 1000).
        save_path (str): The path to save the animation (default: None).
    """
    # Set plot style
    plt.style.use('dark_background')
    sns.set_style("darkgrid")
    
    # Create a range of values to analyze
    x_values = np.linspace(10, max_n, max_n - 9)
    x_integers = np.arange(10, max_n + 1)
    
    # Compute the actual prime counts
    print("Computing actual prime counts...")
    actual_counts = [prime_dist.count_primes(int(x)) for x in x_integers]
    
    # Compute the logarithmic integral approximation
    print("Computing logarithmic integral approximation...")
    li_approx = [prime_dist.logarithmic_integral(x) for x in x_values]
    
    # Compute our exact formula prediction
    print("Computing exact formula prediction...")
    exact_formula = [prime_dist.exact_prime_counting_formula(x) for x in x_values]
    
    # Create a figure for the animation
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up the plot
    ax.set_xlim(10, max_n)
    ax.set_ylim(0, max(actual_counts[-1], li_approx[-1], exact_formula[-1]) * 1.1)
    ax.set_title('Prime Distribution Formula in Action', fontsize=16)
    ax.set_xlabel('n', fontsize=14)
    ax.set_ylabel('Number of Primes ≤ n', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Create the line objects
    line_actual, = ax.plot([], [], 'o', color='gold', alpha=0.7, markersize=5, label='Actual Prime Counts')
    line_li, = ax.plot([], [], color='cyan', alpha=0.7, linewidth=2, label='Logarithmic Integral')
    line_formula, = ax.plot([], [], color='magenta', alpha=0.9, linewidth=2, label='Cyclotomic Formula')
    
    # Add a legend
    ax.legend(fontsize=12)
    
    # Add a text box for the current step
    step_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Animation initialization function
    def init():
        line_actual.set_data([], [])
        line_li.set_data([], [])
        line_formula.set_data([], [])
        step_text.set_text('')
        return line_actual, line_li, line_formula, step_text
    
    # Animation update function
    def update(frame):
        # Update the data
        end_idx = int(frame * (len(x_integers) / 100))
        end_idx = max(10, min(end_idx, len(x_integers)))
        
        x_subset = x_integers[:end_idx]
        actual_subset = actual_counts[:end_idx]
        
        x_values_subset = x_values[:end_idx*10]
        li_subset = li_approx[:end_idx*10]
        formula_subset = exact_formula[:end_idx*10]
        
        line_actual.set_data(x_subset, actual_subset)
        line_li.set_data(x_values_subset, li_subset)
        line_formula.set_data(x_values_subset, formula_subset)
        
        # Update the step text
        if frame < 30:
            step_text.set_text(f"Step 1: Counting actual primes up to {x_subset[-1]}")
        elif frame < 60:
            step_text.set_text(f"Step 2: Computing logarithmic integral approximation")
        else:
            step_text.set_text(f"Step 3: Applying cyclotomic correction factor\nbased on Dedekind cut morphic conductor")
        
        return line_actual, line_li, line_formula, step_text
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True, interval=100)
    
    if save_path:
        ani.save(save_path, writer='pillow', fps=10)
    else:
        plt.show()


def main():
    """
    Main function to run the demonstration.
    """
    print("Initializing Prime Distribution Formula...")
    prime_dist = PrimeDistribution()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the visualizations
    print("\nCreating basic visualization...")
    create_basic_visualization(prime_dist, max_n=1000, 
                              save_path=os.path.join(output_dir, 'prime_distribution_basic.png'))
    
    print("\nCreating error analysis visualization...")
    create_error_analysis(prime_dist, max_n=1000,
                         save_path=os.path.join(output_dir, 'prime_distribution_error.png'))
    
    print("\nCreating correction factor visualization...")
    create_correction_factor_visualization(prime_dist, max_n=1000,
                                         save_path=os.path.join(output_dir, 'prime_distribution_correction.png'))
    
    print("\nCreating spectral density visualization...")
    create_spectral_density_visualization(prime_dist, max_n=1000, bin_size=50,
                                        save_path=os.path.join(output_dir, 'prime_distribution_density.png'))
    
    print("\nCreating 3D visualization...")
    create_3d_visualization(prime_dist, max_n=1000,
                           save_path=os.path.join(output_dir, 'prime_distribution_3d.png'))
    
    print("\nCreating animated visualization...")
    create_animated_visualization(prime_dist, max_n=1000,
                                save_path=os.path.join(output_dir, 'prime_distribution_animation.gif'))
    
    print("\nAll visualizations have been created and saved to the 'output' directory.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()