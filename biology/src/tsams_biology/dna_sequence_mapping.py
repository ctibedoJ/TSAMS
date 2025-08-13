&quot;&quot;&quot;
Dna Sequence Mapping module for Tsams Biology.

This module is part of the Tibedo Structural Algebraic Modeling System (TSAMS) ecosystem.
&quot;&quot;&quot;

# Code adapted from basel_recursive_dna_mapping.py

"""
Basel Recursive DNA Mapping

This module implements the recursive formulaic setup for DNA mapping as described in the Basel paper.
It focuses on the finite 4-code system (G, A, U, C) and its recursive properties within the
eigenvalue log10 braid framework.

Author: Based on the work of Charles Tibedo
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sympy as sp
from sympy import zeta, I, pi, exp, Matrix, symbols, sqrt, log
from octonion_period_sequences import OctonionAlgebra, HodgeStarMoufang
from eigenvalue_ray_resolution import EigenvalueRayResolver

class BaselRecursiveDNAMapper:
    """
    Mapper for DNA sequences using the Basel recursive formulaic setup.
    """
    
    def __init__(self):
        """Initialize the Basel recursive DNA mapper."""
        self.octonion = OctonionAlgebra()
        
        # Define the DNA nucleotides (RNA version with U instead of T for generality)
        self.nucleotides = ['G', 'A', 'U', 'C']
        
        # Generate the Basel zeta values
        self.zeta_values = self._generate_zeta_values()
        
        # Generate the recursive formula components
        self.recursive_components = self._generate_recursive_components()
        
        # Generate the DNA mapping
        self.dna_mapping = self._generate_dna_mapping()
    
    def _generate_zeta_values(self):
        """Generate the Basel zeta values for even powers."""
        # Calculate ζ(2), ζ(4), ζ(6), and ζ(8) using sympy
        zeta_values = {}
        for n in range(1, 9):
            if n % 2 == 0:  # Even powers only
                # For even powers, we have the closed form: ζ(2n) = (-1)^(n+1) * B_2n * (2π)^(2n) / (2 * (2n)!)
                # where B_2n are the Bernoulli numbers
                # But we'll use sympy's zeta function for simplicity
                zeta_values[n] = float(zeta(n))
        
        return zeta_values
    
    def _generate_recursive_components(self):
        """Generate the recursive formula components."""
        # Define the recursive formula components based on the Basel paper
        
        # Component 1: Basel sum convergence rates
        # ζ(2) = π²/6, ζ(4) = π⁴/90, ζ(6) = π⁶/945, ζ(8) = π⁸/9450
        convergence_rates = {
            2: np.pi**2/6,
            4: np.pi**4/90,
            6: np.pi**6/945,
            8: np.pi**8/9450
        }
        
        # Component 2: Eigenvalue multipliers
        # These are derived from the eigenvalue ray resolution
        eigenvalue_multipliers = {
            'G': 1.0,  # Guanine - highest energy
            'A': 0.8,  # Adenine
            'U': 0.6,  # Uracil (or Thymine in DNA)
            'C': 0.4   # Cytosine - lowest energy
        }
        
        # Component 3: Phase factors
        # These introduce complex phase shifts based on nucleotide position
        phase_factors = {
            'G': np.exp(1j * np.pi/4),  # 45° phase
            'A': np.exp(1j * np.pi/2),  # 90° phase
            'U': np.exp(1j * 3*np.pi/4), # 135° phase
            'C': np.exp(1j * np.pi)      # 180° phase
        }
        
        # Component 4: Moufang loop indices
        # These map each nucleotide to a specific Moufang loop
        moufang_indices = {
            'G': 0,  # First loop
            'A': 2,  # Third loop
            'U': 4,  # Fifth loop
            'C': 6   # Seventh loop
        }
        
        return {
            'convergence_rates': convergence_rates,
            'eigenvalue_multipliers': eigenvalue_multipliers,
            'phase_factors': phase_factors,
            'moufang_indices': moufang_indices
        }
    
    def _generate_dna_mapping(self):
        """Generate the mapping for DNA nucleotides."""
        mapping = {}
        
        # For each nucleotide, create a mapping to its mathematical representation
        for nucleotide in self.nucleotides:
            # Get the components for this nucleotide
            multiplier = self.recursive_components['eigenvalue_multipliers'][nucleotide]
            phase = self.recursive_components['phase_factors'][nucleotide]
            loop_idx = self.recursive_components['moufang_indices'][nucleotide]
            
            # Create the mapping
            mapping[nucleotide] = {
                'multiplier': multiplier,
                'phase': phase,
                'loop_index': loop_idx,
                'zeta_values': self.zeta_values,
                'convergence_rates': self.recursive_components['convergence_rates']
            }
        
        return mapping
    
    def map_sequence(self, sequence):
        """
        Map a DNA/RNA sequence using the Basel recursive formula.
        
        Parameters
        ----------
        sequence : str
            The DNA/RNA sequence to map.
            
        Returns
        -------
        result : dict
            The mapping result.
        """
        # Convert sequence to uppercase and validate
        sequence = sequence.upper()
        valid_nucleotides = set(self.nucleotides)
        
        if not all(n in valid_nucleotides for n in sequence):
            raise ValueError(f"Sequence contains invalid nucleotides. Valid nucleotides are {valid_nucleotides}")
        
        # Initialize the result
        result = {
            'sequence': sequence,
            'length': len(sequence),
            'nucleotide_counts': {n: 0 for n in self.nucleotides},
            'basel_sum': 0.0,
            'eigenvalues': [],
            'phases': [],
            'loop_indices': [],
            'recursive_values': []
        }
        
        # Count nucleotides
        for n in sequence:
            result['nucleotide_counts'][n] += 1
        
        # Calculate the Basel sum for the sequence
        # This is a weighted sum of zeta values based on the sequence
        basel_sum = 0.0
        
        for i, nucleotide in enumerate(sequence):
            # Get the mapping for this nucleotide
            mapping = self.dna_mapping[nucleotide]
            
            # Calculate the position weight
            # The weight decreases with position to model the convergence of the Basel sum
            position_weight = 1.0 / (i + 1)**2
            
            # Calculate the contribution to the Basel sum
            # We use ζ(2n) where n is determined by the nucleotide and position
            n = (mapping['loop_index'] // 2 + 1) * 2  # Convert loop index to even power
            if n > 8:
                n = 8  # Cap at ζ(8)
                
            zeta_value = mapping['zeta_values'][n]
            contribution = mapping['multiplier'] * zeta_value * position_weight
            
            # Apply the phase factor
            phase_contribution = mapping['phase'] * contribution
            
            # Add to the Basel sum
            basel_sum += phase_contribution
            
            # Store the individual values
            result['eigenvalues'].append(mapping['multiplier'])
            result['phases'].append(mapping['phase'])
            result['loop_indices'].append(mapping['loop_index'])
            result['recursive_values'].append(contribution)
        
        result['basel_sum'] = basel_sum
        
        # Calculate the recursive formula result
        # This is the key innovation from the Basel paper
        recursive_result = self._apply_recursive_formula(sequence, result)
        result['recursive_result'] = recursive_result
        
        return result
    
    def _apply_recursive_formula(self, sequence, result):
        """
        Apply the recursive formula from the Basel paper.
        
        Parameters
        ----------
        sequence : str
            The DNA/RNA sequence.
        result : dict
            The mapping result so far.
            
        Returns
        -------
        recursive_result : dict
            The result of applying the recursive formula.
        """
        # Initialize the recursive result
        recursive_result = {
            'eigenvalue_product': 1.0 + 0j,
            'phase_sum': 0.0 + 0j,
            'convergence_factor': 0.0,
            'moufang_closure': 0.0,
            'final_value': 0.0 + 0j
        }
        
        # Calculate the eigenvalue product
        # This is the product of all eigenvalue multipliers in the sequence
        eigenvalue_product = 1.0
        for nucleotide in sequence:
            eigenvalue_product *= self.dna_mapping[nucleotide]['multiplier']
        
        recursive_result['eigenvalue_product'] = eigenvalue_product
        
        # Calculate the phase sum
        # This is the sum of all phase factors in the sequence
        phase_sum = 0.0 + 0j
        for i, nucleotide in enumerate(sequence):
            phase = self.dna_mapping[nucleotide]['phase']
            position_weight = 1.0 / (i + 1)
            phase_sum += phase * position_weight
        
        recursive_result['phase_sum'] = phase_sum
        
        # Calculate the convergence factor
        # This is based on the Basel sum convergence rates
        convergence_factor = 0.0
        nucleotide_counts = result['nucleotide_counts']
        total_nucleotides = sum(nucleotide_counts.values())
        
        for nucleotide, count in nucleotide_counts.items():
            if count > 0:
                # Get the loop index for this nucleotide
                loop_idx = self.dna_mapping[nucleotide]['loop_index']
                
                # Convert to even power
                n = (loop_idx // 2 + 1) * 2
                if n > 8:
                    n = 8  # Cap at ζ(8)
                
                # Get the convergence rate
                rate = self.recursive_components['convergence_rates'][n]
                
                # Add to the convergence factor
                convergence_factor += rate * (count / total_nucleotides)
        
        recursive_result['convergence_factor'] = convergence_factor
        
        # Calculate the Moufang loop closure
        # This measures how well the sequence closes the Moufang loops
        moufang_closure = 0.0
        loop_counts = [0] * 8  # 8 Moufang loops
        
        for nucleotide in sequence:
            loop_idx = self.dna_mapping[nucleotide]['loop_index']
            loop_counts[loop_idx] += 1
        
        # Calculate the entropy of the loop distribution
        from scipy.stats import entropy
        loop_probs = [count / len(sequence) for count in loop_counts]
        loop_entropy = entropy(loop_probs)
        
        # Normalize to [0, 1] where 1 is perfect closure
        max_entropy = entropy([1/8] * 8)  # Maximum entropy is uniform distribution
        moufang_closure = loop_entropy / max_entropy
        
        recursive_result['moufang_closure'] = moufang_closure
        
        # Calculate the final recursive value
        # This combines all the components according to the Basel paper formula
        final_value = (
            eigenvalue_product * 
            phase_sum * 
            convergence_factor * 
            (1 + 1j * moufang_closure)
        )
        
        recursive_result['final_value'] = final_value
        
        return recursive_result
    
    def visualize_mapping(self, result, output_dir="outputs/basel_dna"):
        """
        Visualize the Basel recursive DNA mapping.
        
        Parameters
        ----------
        result : dict
            The mapping result.
        output_dir : str, optional
            The output directory for visualizations.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        sequence = result['sequence']
        
        # Visualize the nucleotide distribution
        plt.figure(figsize=(10, 6))
        nucleotides = list(result['nucleotide_counts'].keys())
        counts = list(result['nucleotide_counts'].values())
        
        bars = plt.bar(nucleotides, counts)
        
        # Color the bars according to the nucleotide
        colors = {'G': 'green', 'A': 'red', 'U': 'blue', 'C': 'yellow'}
        for i, nucleotide in enumerate(nucleotides):
            bars[i].set_color(colors.get(nucleotide, 'gray'))
        
        plt.grid(True, axis='y')
        plt.xlabel('Nucleotide')
        plt.ylabel('Count')
        plt.title('Nucleotide Distribution')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/nucleotide_distribution.png", dpi=300)
        plt.close()
        
        # Visualize the eigenvalues
        plt.figure(figsize=(12, 6))
        plt.plot(result['eigenvalues'], 'o-', markersize=8)
        plt.grid(True)
        plt.xlabel('Sequence Position')
        plt.ylabel('Eigenvalue Multiplier')
        plt.title('Eigenvalue Multipliers Along Sequence')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/eigenvalue_multipliers.png", dpi=300)
        plt.close()
        
        # Visualize the phases
        plt.figure(figsize=(12, 6))
        
        # Extract the angles from the complex phase factors
        phases = [np.angle(p) * 180 / np.pi for p in result['phases']]
        
        plt.plot(phases, 'o-', markersize=8)
        plt.grid(True)
        plt.xlabel('Sequence Position')
        plt.ylabel('Phase Angle (degrees)')
        plt.title('Phase Angles Along Sequence')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/phase_angles.png", dpi=300)
        plt.close()
        
        # Visualize the loop indices
        plt.figure(figsize=(12, 6))
        plt.plot(result['loop_indices'], 'o-', markersize=8)
        plt.grid(True)
        plt.xlabel('Sequence Position')
        plt.ylabel('Moufang Loop Index')
        plt.title('Moufang Loop Indices Along Sequence')
        plt.yticks(range(0, 8))
        plt.tight_layout()
        plt.savefig(f"{output_dir}/loop_indices.png", dpi=300)
        plt.close()
        
        # Visualize the recursive values
        plt.figure(figsize=(12, 6))
        
        # Extract the magnitudes of the complex recursive values
        magnitudes = [abs(v) for v in result['recursive_values']]
        
        plt.plot(magnitudes, 'o-', markersize=8)
        plt.grid(True)
        plt.xlabel('Sequence Position')
        plt.ylabel('Contribution Magnitude')
        plt.title('Basel Sum Contributions Along Sequence')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/basel_contributions.png", dpi=300)
        plt.close()
        
        # Visualize the sequence in the complex plane
        plt.figure(figsize=(10, 10))
        
        # Calculate cumulative sum of contributions
        cumulative_sum = [0]
        for v in result['recursive_values']:
            cumulative_sum.append(cumulative_sum[-1] + v)
        
        # Remove the initial 0
        cumulative_sum = cumulative_sum[1:]
        
        # Extract real and imaginary parts
        real_parts = [v.real for v in cumulative_sum]
        imag_parts = [v.imag for v in cumulative_sum]
        
        # Plot the trajectory in the complex plane
        plt.scatter(real_parts, imag_parts, c=range(len(real_parts)), cmap='viridis', s=100)
        plt.plot(real_parts, imag_parts, 'k-', alpha=0.5)
        
        # Add arrows to show direction
        for i in range(len(real_parts)-1):
            plt.arrow(real_parts[i], imag_parts[i], 
                     real_parts[i+1] - real_parts[i], imag_parts[i+1] - imag_parts[i],
                     head_width=0.02, head_length=0.03, fc='k', ec='k', alpha=0.5)
        
        # Add labels for each point
        for i, (x, y) in enumerate(zip(real_parts, imag_parts)):
            plt.text(x, y, sequence[i], fontsize=12, ha='center', va='center')
        
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title('Sequence Trajectory in Complex Plane')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/complex_trajectory.png", dpi=300)
        plt.close()
        
        # Create a summary visualization
        fig = plt.figure(figsize=(15, 10))
        
        # Plot the nucleotide distribution
        ax1 = fig.add_subplot(221)
        bars = ax1.bar(nucleotides, counts)
        for i, nucleotide in enumerate(nucleotides):
            bars[i].set_color(colors.get(nucleotide, 'gray'))
        ax1.grid(True, axis='y')
        ax1.set_xlabel('Nucleotide')
        ax1.set_ylabel('Count')
        ax1.set_title('Nucleotide Distribution')
        
        # Plot the eigenvalues
        ax2 = fig.add_subplot(222)
        ax2.plot(result['eigenvalues'], 'o-', markersize=4)
        ax2.grid(True)
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Eigenvalue')
        ax2.set_title('Eigenvalue Multipliers')
        
        # Plot the phases
        ax3 = fig.add_subplot(223)
        ax3.plot(phases, 'o-', markersize=4)
        ax3.grid(True)
        ax3.set_xlabel('Position')
        ax3.set_ylabel('Phase (degrees)')
        ax3.set_title('Phase Angles')
        
        # Plot the complex trajectory
        ax4 = fig.add_subplot(224)
        scatter = ax4.scatter(real_parts, imag_parts, c=range(len(real_parts)), cmap='viridis', s=50)
        ax4.plot(real_parts, imag_parts, 'k-', alpha=0.5)
        ax4.grid(True)
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Real Part')
        ax4.set_ylabel('Imaginary Part')
        ax4.set_title('Complex Trajectory')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/mapping_summary.png", dpi=300)
        plt.close()
        
        # Create a visualization of the recursive result
        fig = plt.figure(figsize=(12, 8))
        
        # Plot the recursive components as a bar chart
        components = [
            abs(result['recursive_result']['eigenvalue_product']),
            abs(result['recursive_result']['phase_sum']),
            result['recursive_result']['convergence_factor'],
            result['recursive_result']['moufang_closure'],
            abs(result['recursive_result']['final_value'])
        ]
        
        labels = [
            'Eigenvalue Product',
            'Phase Sum',
            'Convergence Factor',
            'Moufang Closure',
            'Final Value'
        ]
        
        ax = fig.add_subplot(111)
        bars = ax.bar(labels, components)
        
        # Color the bars
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for i, bar in enumerate(bars):
            bar.set_color(colors[i])
        
        ax.grid(True, axis='y')
        ax.set_ylabel('Value')
        ax.set_title('Recursive Formula Components')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/recursive_components.png", dpi=300)
        plt.close()
        
        # Save the numerical results as a text file
        with open(f"{output_dir}/basel_mapping_results.txt", 'w') as f:
            f.write(f"Basel Recursive DNA Mapping Results for sequence: {sequence}\n\n")
            
            f.write("Nucleotide Counts:\n")
            for nucleotide, count in result['nucleotide_counts'].items():
                f.write(f"{nucleotide}: {count}\n")
            
            f.write("\nBasel Sum: {:.6f} + {:.6f}j\n".format(
                result['basel_sum'].real, result['basel_sum'].imag))
            
            f.write("\nRecursive Formula Results:\n")
            f.write("Eigenvalue Product: {:.6f}\n".format(
                abs(result['recursive_result']['eigenvalue_product'])))
            f.write("Phase Sum: {:.6f} + {:.6f}j\n".format(
                result['recursive_result']['phase_sum'].real,
                result['recursive_result']['phase_sum'].imag))
            f.write("Convergence Factor: {:.6f}\n".format(
                result['recursive_result']['convergence_factor']))
            f.write("Moufang Closure: {:.6f}\n".format(
                result['recursive_result']['moufang_closure']))
            f.write("Final Value: {:.6f} + {:.6f}j\n".format(
                result['recursive_result']['final_value'].real,
                result['recursive_result']['final_value'].imag))


def analyze_sequence(sequence, output_dir="outputs/basel_dna"):
    """
    Analyze a DNA/RNA sequence using the Basel recursive formula.
    
    Parameters
    ----------
    sequence : str
        The DNA/RNA sequence to analyze.
    output_dir : str, optional
        The output directory for visualizations.
    """
    print(f"Analyzing sequence: {sequence}")
    
    # Create the mapper
    mapper = BaselRecursiveDNAMapper()
    
    # Map the sequence
    result = mapper.map_sequence(sequence)
    
    # Print the results
    print("\nNucleotide Counts:")
    for nucleotide, count in result['nucleotide_counts'].items():
        print(f"{nucleotide}: {count}")
    
    print(f"\nBasel Sum: {result['basel_sum']}")
    
    print("\nRecursive Formula Results:")
    print(f"Eigenvalue Product: {abs(result['recursive_result']['eigenvalue_product'])}")
    print(f"Phase Sum: {result['recursive_result']['phase_sum']}")
    print(f"Convergence Factor: {result['recursive_result']['convergence_factor']}")
    print(f"Moufang Closure: {result['recursive_result']['moufang_closure']}")
    print(f"Final Value: {result['recursive_result']['final_value']}")
    
    # Visualize the results
    print("\nGenerating visualizations...")
    mapper.visualize_mapping(result, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}/")
    
    return result


def main():
    """Main function to demonstrate the Basel recursive DNA mapping."""
    print("Initializing Basel Recursive DNA Mapper...")
    
    # Example RNA sequence
    sequence = "GACUGCAUGACUGCAUGACUGCAU"
    
    # Analyze the sequence
    analyze_sequence(sequence)


if __name__ == "__main__":
    main()
