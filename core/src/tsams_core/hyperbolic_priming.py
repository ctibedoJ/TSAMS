&quot;&quot;&quot;
Hyperbolic Priming module for Tsams Core.

This module is part of the Tibedo Structural Algebraic Modeling System (TSAMS) ecosystem.
&quot;&quot;&quot;

# Code adapted from tibedo_energy_minimization.py

"""
TIBEDO Framework: Energy Minimization Algorithms

This module implements energy minimization algorithms for protein folding
using the TIBEDO Framework's mathematical principles.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

# Try to import the biological implementation classes
try:
    from tibedo_biological_implementation import ProteinFoldingSimulator, QuaternionMobiusStripPairing, CyclotomicFieldGaloisOrbit
    BIOLOGICAL_IMPLEMENTATION_AVAILABLE = True
except ImportError:
    print("Warning: Could not import biological implementation classes")
    BIOLOGICAL_IMPLEMENTATION_AVAILABLE = False

class EnergyMinimizer:
    """
    Energy minimization algorithms for protein folding using TIBEDO principles.
    """
    
    def __init__(self):
        """Initialize the energy minimizer."""
        if not BIOLOGICAL_IMPLEMENTATION_AVAILABLE:
            raise ImportError("Biological implementation classes not available")
            
        self.simulator = ProteinFoldingSimulator()
        self.mobius_pairing = QuaternionMobiusStripPairing()
        self.cyclotomic_field = CyclotomicFieldGaloisOrbit(conductor=42)
        
        # Define energy parameters
        self.energy_params = {
            'bond_strength': 1.0,
            'angle_strength': 0.5,
            'torsion_strength': 0.3,
            'non_bonded_strength': 0.8,
            'dedekind_weight': 0.4,
            'mobius_weight': 0.6
        }
        
    def compute_bond_energy(self, quaternions):
        """
        Compute the bond energy between adjacent amino acids.
        
        Args:
            quaternions (list): List of quaternions representing amino acids
            
        Returns:
            float: The bond energy
        """
        energy = 0.0
        
        for i in range(len(quaternions) - 1):
            q1 = quaternions[i]
            q2 = quaternions[i+1]
            
            # Compute the quaternion distance
            diff = q1 - q2
            distance = self.mobius_pairing.quaternion_norm(diff)
            
            # Bond energy is proportional to the square of the distance deviation from ideal
            ideal_distance = 0.5  # Ideal quaternion distance between adjacent amino acids
            energy += self.energy_params['bond_strength'] * (distance - ideal_distance) ** 2
            
        return energy
        
    def compute_angle_energy(self, quaternions):
        """
        Compute the angle energy between triplets of amino acids.
        
        Args:
            quaternions (list): List of quaternions representing amino acids
            
        Returns:
            float: The angle energy
        """
        energy = 0.0
        
        for i in range(len(quaternions) - 2):
            q1 = quaternions[i]
            q2 = quaternions[i+1]
            q3 = quaternions[i+2]
            
            # Compute vectors between consecutive amino acids
            v1 = q2 - q1
            v2 = q3 - q2
            
            # Normalize the vectors
            v1_norm = self.mobius_pairing.quaternion_norm(v1)
            v2_norm = self.mobius_pairing.quaternion_norm(v2)
            
            if v1_norm > 1e-10 and v2_norm > 1e-10:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                
                # Compute the dot product (cosine of the angle)
                dot_product = np.sum(v1 * v2)
                
                # Angle energy is proportional to the square of the cosine deviation from ideal
                ideal_cos = 0.0  # Ideal cosine is 0 (90 degrees)
                energy += self.energy_params['angle_strength'] * (dot_product - ideal_cos) ** 2
            
        return energy
        
    def compute_torsion_energy(self, quaternions):
        """
        Compute the torsion energy between quartets of amino acids.
        
        Args:
            quaternions (list): List of quaternions representing amino acids
            
        Returns:
            float: The torsion energy
        """
        energy = 0.0
        
        for i in range(len(quaternions) - 3):
            q1 = quaternions[i]
            q2 = quaternions[i+1]
            q3 = quaternions[i+2]
            q4 = quaternions[i+3]
            
            # Compute vectors between consecutive amino acids
            v1 = q2 - q1
            v2 = q3 - q2
            v3 = q4 - q3
            
            # Compute normal vectors to the planes
            n1 = np.cross(v1[1:], v2[1:])  # Cross product of the vector parts (i,j,k)
            n2 = np.cross(v2[1:], v3[1:])
            
            # Normalize the normal vectors
            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)
            
            if n1_norm > 1e-10 and n2_norm > 1e-10:
                n1 = n1 / n1_norm
                n2 = n2 / n2_norm
                
                # Compute the dot product (cosine of the dihedral angle)
                dot_product = np.sum(n1 * n2)
                
                # Torsion energy is proportional to the square of the cosine deviation from ideal
                ideal_cos = 0.0  # Ideal cosine is 0 (90 degrees)
                energy += self.energy_params['torsion_strength'] * (dot_product - ideal_cos) ** 2
            
        return energy
        
    def compute_non_bonded_energy(self, quaternions):
        """
        Compute the non-bonded energy between non-adjacent amino acids.
        
        Args:
            quaternions (list): List of quaternions representing amino acids
            
        Returns:
            float: The non-bonded energy
        """
        energy = 0.0
        
        for i in range(len(quaternions)):
            for j in range(i + 3, len(quaternions)):  # Skip adjacent and angle-related amino acids
                q1 = quaternions[i]
                q2 = quaternions[j]
                
                # Compute the quaternion distance
                diff = q1 - q2
                distance = self.mobius_pairing.quaternion_norm(diff)
                
                # Non-bonded energy uses a Lennard-Jones-like potential
                if distance > 1e-10:
                    repulsion = (1.0 / distance) ** 12
                    attraction = (1.0 / distance) ** 6
                    energy += self.energy_params['non_bonded_strength'] * (repulsion - attraction)
            
        return energy
        
    def compute_dedekind_energy(self, sequence):
        """
        Compute the energy contribution from Dedekind cut ratios.
        
        Args:
            sequence (str): The amino acid sequence
            
        Returns:
            float: The Dedekind energy
        """
        # Compute Dedekind cut ratios for key primes
        key_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
        dedekind_cuts = {}
        
        for prime in key_primes:
            try:
                dedekind_cuts[prime] = self.cyclotomic_field.compute_dedekind_cut_ratio(prime)
            except:
                # Skip primes that cause issues
                continue
                
        # Find the prime with the minimum Dedekind cut ratio
        min_prime = min(dedekind_cuts, key=dedekind_cuts.get)
        min_ratio = dedekind_cuts[min_prime]
        
        # Energy is proportional to the minimum Dedekind cut ratio
        energy = self.energy_params['dedekind_weight'] * min_ratio * len(sequence)
        
        return energy
        
    def compute_mobius_energy(self, mobius_points):
        """
        Compute the energy contribution from the Möbius strip mapping.
        
        Args:
            mobius_points (list): List of 3D points on the Möbius strip
            
        Returns:
            float: The Möbius energy
        """
        energy = 0.0
        
        # Compute the path length on the Möbius strip
        for i in range(len(mobius_points) - 1):
            p1 = mobius_points[i]
            p2 = mobius_points[i+1]
            
            # Compute the Euclidean distance
            distance = np.linalg.norm(p1 - p2)
            
            # Energy is proportional to the square of the distance
            energy += self.energy_params['mobius_weight'] * distance ** 2
            
        return energy
        
    def compute_total_energy(self, quaternions, mobius_points, sequence):
        """
        Compute the total energy of the protein folding.
        
        Args:
            quaternions (list): List of quaternions representing amino acids
            mobius_points (list): List of 3D points on the Möbius strip
            sequence (str): The amino acid sequence
            
        Returns:
            dict: The energy components and total energy
        """
        # Compute the energy components
        bond_energy = self.compute_bond_energy(quaternions)
        angle_energy = self.compute_angle_energy(quaternions)
        torsion_energy = self.compute_torsion_energy(quaternions)
        non_bonded_energy = self.compute_non_bonded_energy(quaternions)
        dedekind_energy = self.compute_dedekind_energy(sequence)
        mobius_energy = self.compute_mobius_energy(mobius_points)
        
        # Compute the total energy
        total_energy = bond_energy + angle_energy + torsion_energy + non_bonded_energy + dedekind_energy + mobius_energy
        
        # Return the energy components and total energy
        return {
            'bond_energy': bond_energy,
            'angle_energy': angle_energy,
            'torsion_energy': torsion_energy,
            'non_bonded_energy': non_bonded_energy,
            'dedekind_energy': dedekind_energy,
            'mobius_energy': mobius_energy,
            'total_energy': total_energy
        }
        
    def minimize_energy(self, sequence, max_iterations=100, learning_rate=0.01, tolerance=1e-6):
        """
        Minimize the energy of the protein folding using gradient descent.
        
        Args:
            sequence (str): The amino acid sequence
            max_iterations (int): Maximum number of iterations
            learning_rate (float): Learning rate for gradient descent
            tolerance (float): Convergence tolerance
            
        Returns:
            dict: The minimization results
        """
        # Initial simulation
        initial_results = self.simulator.simulate_protein_folding(sequence)
        
        # Extract the quaternions and Möbius points
        quaternions = np.array(initial_results['quaternions'])
        mobius_points = np.array(initial_results['mobius_points'])
        
        # Compute the initial energy
        initial_energy = self.compute_total_energy(quaternions, mobius_points, sequence)
        
        # Initialize the minimization
        current_quaternions = quaternions.copy()
        current_mobius_points = mobius_points.copy()
        current_energy = initial_energy['total_energy']
        
        # Track the energy history
        energy_history = [current_energy]
        
        # Perform gradient descent
        for iteration in range(max_iterations):
            # Compute the gradient for each quaternion
            gradient = np.zeros_like(current_quaternions)
            
            for i in range(len(current_quaternions)):
                # Perturb the quaternion slightly in each dimension
                for dim in range(4):
                    # Create a perturbed copy
                    perturbed_quaternions = current_quaternions.copy()
                    perturbed_quaternions[i, dim] += 1e-6
                    
                    # Normalize the perturbed quaternion
                    perturbed_quaternions[i] = perturbed_quaternions[i] / self.mobius_pairing.quaternion_norm(perturbed_quaternions[i])
                    
                    # Update the Möbius points
                    perturbed_mobius_points = np.array([self.mobius_pairing.map_quaternion_to_mobius(q) for q in perturbed_quaternions])
                    
                    # Compute the energy with the perturbed quaternion
                    perturbed_energy = self.compute_total_energy(perturbed_quaternions, perturbed_mobius_points, sequence)['total_energy']
                    
                    # Compute the numerical gradient
                    gradient[i, dim] = (perturbed_energy - current_energy) / 1e-6
            
            # Update the quaternions using gradient descent
            current_quaternions -= learning_rate * gradient
            
            # Normalize the quaternions
            for i in range(len(current_quaternions)):
                current_quaternions[i] = current_quaternions[i] / self.mobius_pairing.quaternion_norm(current_quaternions[i])
            
            # Update the Möbius points
            current_mobius_points = np.array([self.mobius_pairing.map_quaternion_to_mobius(q) for q in current_quaternions])
            
            # Compute the new energy
            new_energy = self.compute_total_energy(current_quaternions, current_mobius_points, sequence)['total_energy']
            
            # Track the energy history
            energy_history.append(new_energy)
            
            # Check for convergence
            energy_change = abs(new_energy - current_energy)
            if energy_change < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
                
            # Update the current energy
            current_energy = new_energy
            
            # Print progress
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Energy = {current_energy:.6f}, Change = {energy_change:.6f}")
        
        # Compute the final energy components
        final_energy = self.compute_total_energy(current_quaternions, current_mobius_points, sequence)
        
        # Return the minimization results
        return {
            'initial_quaternions': quaternions,
            'initial_mobius_points': mobius_points,
            'initial_energy': initial_energy,
            'final_quaternions': current_quaternions,
            'final_mobius_points': current_mobius_points,
            'final_energy': final_energy,
            'energy_history': energy_history,
            'iterations': len(energy_history) - 1
        }
        
    def visualize_energy_minimization(self, results, save_path=None):
        """
        Visualize the energy minimization results.
        
        Args:
            results (dict): The minimization results
            save_path (str, optional): Path to save the visualization
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Create the figure
        fig = plt.figure(figsize=(18, 10))
        
        # Plot the energy history
        ax1 = fig.add_subplot(231)
        ax1.plot(results['energy_history'], 'o-', color='blue', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Energy')
        ax1.set_title('Energy Minimization Progress')
        ax1.grid(True, alpha=0.3)
        
        # Plot the energy components
        ax2 = fig.add_subplot(232)
        energy_components = ['bond_energy', 'angle_energy', 'torsion_energy', 
                            'non_bonded_energy', 'dedekind_energy', 'mobius_energy']
        initial_values = [results['initial_energy'][comp] for comp in energy_components]
        final_values = [results['final_energy'][comp] for comp in energy_components]
        
        x = np.arange(len(energy_components))
        width = 0.35
        
        ax2.bar(x - width/2, initial_values, width, label='Initial')
        ax2.bar(x + width/2, final_values, width, label='Final')
        
        ax2.set_xlabel('Energy Component')
        ax2.set_ylabel('Energy Value')
        ax2.set_title('Energy Components Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels([comp.split('_')[0].capitalize() for comp in energy_components], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot the initial Möbius strip
        ax3 = fig.add_subplot(233, projection='3d')
        initial_points = results['initial_mobius_points']
        ax3.plot(initial_points[:, 0], initial_points[:, 1], initial_points[:, 2], 
                'o-', color='blue', linewidth=2, markersize=4)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('Initial Protein Folding')
        ax3.grid(True, alpha=0.3)
        ax3.set_box_aspect([1, 1, 1])
        
        # Plot the final Möbius strip
        ax4 = fig.add_subplot(234, projection='3d')
        final_points = results['final_mobius_points']
        ax4.plot(final_points[:, 0], final_points[:, 1], final_points[:, 2], 
                'o-', color='green', linewidth=2, markersize=4)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')
        ax4.set_title('Minimized Protein Folding')
        ax4.grid(True, alpha=0.3)
        ax4.set_box_aspect([1, 1, 1])
        
        # Plot the quaternion changes
        ax5 = fig.add_subplot(235)
        initial_q = results['initial_quaternions']
        final_q = results['final_quaternions']
        
        # Compute the quaternion differences
        q_diff = np.linalg.norm(final_q - initial_q, axis=1)
        
        ax5.plot(q_diff, 'o-', color='purple', linewidth=2)
        ax5.set_xlabel('Amino Acid Index')
        ax5.set_ylabel('Quaternion Change')
        ax5.set_title('Quaternion Modifications')
        ax5.grid(True, alpha=0.3)
        
        # Plot the total energy comparison
        ax6 = fig.add_subplot(236)
        labels = ['Initial', 'Final']
        values = [results['initial_energy']['total_energy'], results['final_energy']['total_energy']]
        
        ax6.bar(labels, values, color=['blue', 'green'])
        ax6.set_xlabel('State')
        ax6.set_ylabel('Total Energy')
        ax6.set_title('Total Energy Comparison')
        
        # Add the energy reduction percentage
        energy_reduction = (values[0] - values[1]) / values[0] * 100 if values[0] > 0 else 0
        ax6.text(0.5, 0.5, f"Energy Reduction: {energy_reduction:.2f}%", 
                transform=ax6.transAxes, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8))
        
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return fig

def test_energy_minimization():
    """
    Test the energy minimization algorithm.
    """
    if not BIOLOGICAL_IMPLEMENTATION_AVAILABLE:
        print("Skipping energy minimization test due to missing biological implementation")
        return
        
    print("Testing TIBEDO Energy Minimization")
    print("=================================")
    
    # Create the energy minimizer
    minimizer = EnergyMinimizer()
    
    # Define a test sequence (short peptide for quick testing)
    sequence = "MFVFLVLLPLVSSQCVNLTTRTQL"
    
    print(f"Minimizing energy for sequence: {sequence}")
    
    # Minimize the energy
    start_time = time.time()
    results = minimizer.minimize_energy(sequence, max_iterations=50, learning_rate=0.005)
    elapsed_time = time.time() - start_time
    
    print(f"Energy minimization completed in {elapsed_time:.6f} seconds")
    print(f"Initial energy: {results['initial_energy']['total_energy']:.6f}")
    print(f"Final energy: {results['final_energy']['total_energy']:.6f}")
    print(f"Energy reduction: {(results['initial_energy']['total_energy'] - results['final_energy']['total_energy']) / results['initial_energy']['total_energy'] * 100:.2f}%")
    
    # Visualize the results
    minimizer.visualize_energy_minimization(results, save_path="energy_minimization_results.png")
    
    print("Energy minimization visualization saved as energy_minimization_results.png")

if __name__ == "__main__":
    test_energy_minimization()
