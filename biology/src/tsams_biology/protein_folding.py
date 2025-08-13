&quot;&quot;&quot;
Protein Folding module for Tsams Biology.

This module is part of the Tibedo Structural Algebraic Modeling System (TSAMS) ecosystem.
&quot;&quot;&quot;

# Code adapted from dna_sequence_mapping.py

"""
DNA Sequence Mapping to Root Tree Eigenvalue Log10 Braids

This module implements the mapping of DNA sequence possibilities to the root tree eigenvalue log10 braids,
where each braid originates on the surface of the apple core-like topology and closes 8 Moufang loops,
with the 8th being the inverted loop where all loops tie simultaneously in the time dimension.

Author: Based on the work of Charles Tibedo
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.special import jn  # Bessel functions
import sympy as sp
from sympy import I, pi, exp, Matrix, symbols, sqrt, log
from octonion_period_sequences import OctonionAlgebra, HodgeStarMoufang, DicosohedralMatrix
from eigenvalue_ray_resolution import EigenvalueRayResolver

class DNASequenceMapper:
    """
    Mapper for DNA sequences to root tree eigenvalue log10 braids.
    """
    
    def __init__(self):
        """Initialize the DNA sequence mapper."""
        self.octonion = OctonionAlgebra()
        self.hodge = HodgeStarMoufang()
        
        # Create the eigenvalue ray resolver
        dicosohedral = DicosohedralMatrix()
        self.resolver = EigenvalueRayResolver(dicosohedral.matrix)
        
        # Resolve the eigenvalue matrix
        self.resolved_matrix, self.structures = self.resolver.resolve_eigenvalues()
        
        # Generate the log10 braids
        self.log10_braids = self._generate_log10_braids()
        
        # Define the DNA nucleotides
        self.nucleotides = ['A', 'C', 'G', 'T']
        
        # Generate the apple core topology
        self.apple_core = self._generate_apple_core_topology()
        
        # Generate the 8 Moufang loops
        self.moufang_loops = self._generate_moufang_loops()
        
        # Generate the DNA mapping
        self.dna_mapping = self._generate_dna_mapping()
    
    def _generate_log10_braids(self):
        """Generate the log10 braids."""
        # Create a matrix of log10 braid values
        log_braids = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                log_braids[i, j] = np.log10(i+1) * np.log10(j+1)
        
        return log_braids
    
    def _generate_apple_core_topology(self):
        """Generate the apple core topology."""
        # Define parameters for the apple core shape
        def apple_core(u, v):
            # Convert spherical coordinates to Cartesian
            # u: azimuthal angle (0 to 2π)
            # v: polar angle (0 to π)
            
            # Add the apple core shape modulation
            r = 1.0 + 0.3 * np.sin(3*u) * np.sin(2*v)
            
            # Add the "bite" out of the apple
            bite = 0.4 * np.exp(-5 * ((u - np.pi/2)**2 + (v - np.pi/2)**2))
            r -= bite
            
            # Convert to Cartesian coordinates
            x = r * np.sin(v) * np.cos(u)
            y = r * np.sin(v) * np.sin(u)
            z = r * np.cos(v)
            
            return x, y, z
        
        # Generate the surface
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        u, v = np.meshgrid(u, v)
        
        x, y, z = apple_core(u, v)
        
        return {
            'x': x,
            'y': y,
            'z': z,
            'u': u,
            'v': v,
            'function': apple_core
        }
    
    def _generate_moufang_loops(self):
        """Generate the 8 Moufang loops."""
        moufang_loops = []
        
        # Generate 7 standard Moufang loops
        for i in range(7):
            # Each loop is defined by a specific pattern in the octonion space
            loop = {
                'index': i,
                'origin_ray': self.resolver.rays[i],
                'braid_indices': [(i, j % 7) for j in range(7)],
                'determinant': 0.0,  # Will be calculated later
                'norm': 1.0,  # Will be calculated later
                'type': 'standard'
            }
            moufang_loops.append(loop)
        
        # Generate the 8th inverted loop where all loops tie simultaneously
        inverted_loop = {
            'index': 7,
            'origin_rays': self.resolver.rays,  # All rays
            'braid_indices': [(i, j) for i in range(7) for j in range(7)],
            'determinant': 0.0,  # Will be calculated later
            'norm': 1.0,  # Will be calculated later
            'type': 'inverted'
        }
        moufang_loops.append(inverted_loop)
        
        # Calculate determinants and norms
        for i, loop in enumerate(moufang_loops):
            if loop['type'] == 'standard':
                # Create a matrix for this loop
                matrix = np.zeros((7, 7), dtype=complex)
                for j, (r, c) in enumerate(loop['braid_indices']):
                    matrix[r, c] = self.log10_braids[r, c]
                
                # Calculate determinant and norm
                loop['determinant'] = np.linalg.det(matrix)
                loop['norm'] = np.linalg.norm(matrix, 'fro')
            else:
                # For the inverted loop, the determinant is 0 and norm is 1 by definition
                loop['determinant'] = 0.0
                loop['norm'] = 1.0
        
        return moufang_loops
    
    def _generate_dna_mapping(self):
        """Generate the mapping from DNA sequences to root tree eigenvalue log10 braids."""
        # Create a mapping from DNA triplets to braid indices
        dna_triplets = [
            a + b + c for a in self.nucleotides 
            for b in self.nucleotides 
            for c in self.nucleotides
        ]
        
        # There are 64 possible DNA triplets (codons)
        # We'll map them to specific positions in the log10 braids
        
        mapping = {}
        
        # For each DNA triplet, assign a specific braid
        for i, triplet in enumerate(dna_triplets):
            # Determine which Moufang loop this triplet belongs to
            loop_idx = i % 8
            loop = self.moufang_loops[loop_idx]
            
            # Determine the position within the loop
            if loop['type'] == 'standard':
                pos_idx = (i // 8) % len(loop['braid_indices'])
                braid_pos = loop['braid_indices'][pos_idx]
                
                # Calculate the braid value
                braid_value = self.log10_braids[braid_pos[0], braid_pos[1]]
                
                # Calculate the position on the apple core topology
                u_pos = 2 * np.pi * braid_pos[0] / 7
                v_pos = np.pi * braid_pos[1] / 7
                
                # Get the 3D position
                x, y, z = self.apple_core['function'](u_pos, v_pos)
                
                mapping[triplet] = {
                    'loop_index': loop_idx,
                    'braid_position': braid_pos,
                    'braid_value': braid_value,
                    'surface_position': (x, y, z),
                    'u_v_position': (u_pos, v_pos)
                }
            else:
                # For the inverted loop, use a special mapping
                pos_idx = (i // 8) % 49  # 7*7 = 49 positions
                r = pos_idx // 7
                c = pos_idx % 7
                braid_pos = (r, c)
                
                # Calculate the braid value
                braid_value = self.log10_braids[r, c]
                
                # For the inverted loop, the position is determined by the time dimension
                # p_n^2/logsqrtp
                p = 2.0  # pressure parameter
                n = 10.0  # log base
                
                # Calculate the time dimension factor
                log_sqrt_p = np.log(np.sqrt(p)) / np.log(n)
                time_factor = p**2 / log_sqrt_p
                
                # Calculate the position on the apple core topology
                # For the inverted loop, we use a special mapping that depends on the time factor
                u_pos = 2 * np.pi * r / 7 * np.cos(time_factor)
                v_pos = np.pi * c / 7 * np.sin(time_factor)
                
                # Get the 3D position
                x, y, z = self.apple_core['function'](u_pos, v_pos)
                
                mapping[triplet] = {
                    'loop_index': loop_idx,
                    'braid_position': braid_pos,
                    'braid_value': braid_value,
                    'surface_position': (x, y, z),
                    'u_v_position': (u_pos, v_pos),
                    'time_factor': time_factor
                }
        
        return mapping
    
    def map_dna_sequence(self, sequence):
        """
        Map a DNA sequence to root tree eigenvalue log10 braids.
        
        Parameters
        ----------
        sequence : str
            The DNA sequence to map.
            
        Returns
        -------
        mapping : list
            The mapping of the DNA sequence to braids.
        """
        # Ensure the sequence length is a multiple of 3
        if len(sequence) % 3 != 0:
            # Pad with 'A's if necessary
            sequence += 'A' * (3 - len(sequence) % 3)
        
        # Split the sequence into triplets
        triplets = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
        
        # Map each triplet to a braid
        mapping = []
        for triplet in triplets:
            if triplet in self.dna_mapping:
                mapping.append(self.dna_mapping[triplet])
            else:
                # If the triplet contains invalid nucleotides, use a default mapping
                mapping.append(self.dna_mapping['AAA'])
        
        return mapping
    
    def calculate_sequence_properties(self, mapping):
        """
        Calculate properties of a mapped DNA sequence.
        
        Parameters
        ----------
        mapping : list
            The mapping of the DNA sequence to braids.
            
        Returns
        -------
        properties : dict
            The properties of the mapped sequence.
        """
        # Calculate various properties of the mapped sequence
        
        # Extract braid values
        braid_values = [m['braid_value'] for m in mapping]
        
        # Calculate the mean braid value
        mean_braid = np.mean(braid_values)
        
        # Calculate the standard deviation of braid values
        std_braid = np.std(braid_values)
        
        # Count the number of occurrences of each Moufang loop
        loop_counts = {}
        for m in mapping:
            loop_idx = m['loop_index']
            loop_counts[loop_idx] = loop_counts.get(loop_idx, 0) + 1
        
        # Calculate the entropy of the braid values
        from scipy.stats import entropy
        # Create a histogram of braid values
        hist, _ = np.histogram(braid_values, bins=10)
        # Calculate entropy
        if np.sum(hist) > 0:
            braid_entropy = entropy(hist / np.sum(hist))
        else:
            braid_entropy = 0.0
        
        # Calculate the "closure factor" - how well the sequence closes the 8 Moufang loops
        # A perfect closure would have equal representation of all 8 loops
        ideal_count = len(mapping) / 8  # Equal distribution
        closure_deviations = [abs(loop_counts.get(i, 0) - ideal_count) for i in range(8)]
        closure_factor = 1.0 - np.sum(closure_deviations) / (len(mapping) * 2)  # Normalized to [0, 1]
        
        # Calculate the "time dimension factor" for the sequence
        # This is related to how the sequence interacts with the 8th inverted loop
        inverted_mappings = [m for m in mapping if m['loop_index'] == 7]
        if inverted_mappings:
            time_factors = [m.get('time_factor', 0.0) for m in inverted_mappings]
            mean_time_factor = np.mean(time_factors)
        else:
            mean_time_factor = 0.0
        
        return {
            'mean_braid_value': mean_braid,
            'std_braid_value': std_braid,
            'loop_counts': loop_counts,
            'braid_entropy': braid_entropy,
            'closure_factor': closure_factor,
            'mean_time_factor': mean_time_factor
        }
    
    def visualize_dna_mapping(self, mapping, output_dir="outputs/dna_mapping"):
        """
        Visualize the mapping of a DNA sequence to root tree eigenvalue log10 braids.
        
        Parameters
        ----------
        mapping : list
            The mapping of the DNA sequence to braids.
        output_dir : str, optional
            The output directory for visualizations.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract positions on the apple core topology
        positions = [m['surface_position'] for m in mapping]
        x = [p[0] for p in positions]
        y = [p[1] for p in positions]
        z = [p[2] for p in positions]
        
        # Extract loop indices for coloring
        loop_indices = [m['loop_index'] for m in mapping]
        
        # Visualize the mapping on the apple core topology
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the apple core surface
        ax.plot_surface(
            self.apple_core['x'], self.apple_core['y'], self.apple_core['z'],
            color='lightgray', alpha=0.3, rstride=1, cstride=1
        )
        
        # Plot the mapped positions
        scatter = ax.scatter(x, y, z, c=loop_indices, cmap='viridis', s=100, alpha=0.8)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Moufang Loop Index')
        
        # Connect sequential points with lines
        for i in range(len(x)-1):
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], 'k-', alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('DNA Sequence Mapping on Apple Core Topology')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dna_mapping_3d.png", dpi=300)
        plt.close()
        
        # Visualize the braid values
        braid_values = [m['braid_value'] for m in mapping]
        
        plt.figure(figsize=(12, 6))
        plt.plot(braid_values, 'o-', markersize=8)
        plt.grid(True)
        plt.xlabel('Sequence Position')
        plt.ylabel('Braid Value')
        plt.title('Braid Values Along DNA Sequence')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dna_braid_values.png", dpi=300)
        plt.close()
        
        # Visualize the loop distribution
        loop_counts = {}
        for m in mapping:
            loop_idx = m['loop_index']
            loop_counts[loop_idx] = loop_counts.get(loop_idx, 0) + 1
        
        loops = sorted(loop_counts.keys())
        counts = [loop_counts[l] for l in loops]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(loops, counts)
        
        # Color the 8th bar differently to highlight the inverted loop
        if 7 in loops:
            idx = loops.index(7)
            bars[idx].set_color('red')
        
        plt.grid(True, axis='y')
        plt.xlabel('Moufang Loop Index')
        plt.ylabel('Count')
        plt.title('Distribution of DNA Sequence Across Moufang Loops')
        plt.xticks(loops)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dna_loop_distribution.png", dpi=300)
        plt.close()
        
        # Visualize the closure of the 8 Moufang loops
        # Create a radar chart
        properties = self.calculate_sequence_properties(mapping)
        
        # Prepare the data for the radar chart
        categories = ['Loop 0', 'Loop 1', 'Loop 2', 'Loop 3', 
                     'Loop 4', 'Loop 5', 'Loop 6', 'Loop 7']
        
        # Get the counts for each loop, default to 0 if not present
        values = [properties['loop_counts'].get(i, 0) for i in range(8)]
        
        # Normalize the values
        max_count = max(values) if values else 1
        values = [v / max_count for v in values]
        
        # Create the radar chart
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Set the angles for each category
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        
        # Close the loop
        values.append(values[0])
        angles.append(angles[0])
        
        # Plot the values
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        # Set the labels
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        
        # Add a title
        ax.set_title('Moufang Loop Closure Pattern')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dna_loop_closure.png", dpi=300)
        plt.close()
        
        # Create a summary visualization
        fig = plt.figure(figsize=(12, 10))
        
        # Plot the 3D mapping
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot_surface(
            self.apple_core['x'], self.apple_core['y'], self.apple_core['z'],
            color='lightgray', alpha=0.3, rstride=1, cstride=1
        )
        ax1.scatter(x, y, z, c=loop_indices, cmap='viridis', s=50, alpha=0.8)
        ax1.set_title('DNA Mapping on Apple Core')
        
        # Plot the braid values
        ax2 = fig.add_subplot(222)
        ax2.plot(braid_values, 'o-', markersize=4)
        ax2.grid(True)
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Braid Value')
        ax2.set_title('Braid Values')
        
        # Plot the loop distribution
        ax3 = fig.add_subplot(223)
        bars = ax3.bar(loops, counts)
        if 7 in loops:
            idx = loops.index(7)
            bars[idx].set_color('red')
        ax3.grid(True, axis='y')
        ax3.set_xlabel('Loop Index')
        ax3.set_ylabel('Count')
        ax3.set_title('Loop Distribution')
        
        # Plot the radar chart
        ax4 = fig.add_subplot(224, polar=True)
        ax4.plot(angles, values, 'o-', linewidth=2)
        ax4.fill(angles, values, alpha=0.25)
        ax4.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax4.set_title('Loop Closure Pattern')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dna_mapping_summary.png", dpi=300)
        plt.close()
        
        # Create a table of the mapping
        from matplotlib.table import Table
        
        fig, ax = plt.subplots(figsize=(12, len(mapping) * 0.3 + 1))
        ax.axis('off')
        
        # Create the table data
        table_data = []
        for i, m in enumerate(mapping):
            triplet = f"Triplet {i+1}"
            loop = f"Loop {m['loop_index']}"
            braid = f"{m['braid_value']:.3f}"
            position = f"({m['surface_position'][0]:.2f}, {m['surface_position'][1]:.2f}, {m['surface_position'][2]:.2f})"
            
            table_data.append([triplet, loop, braid, position])
        
        # Create the table
        table = Table(ax, bbox=[0, 0, 1, 1])
        
        # Add header
        table.add_cell(0, 0, width=0.2, height=0.1, text='Triplet', loc='center', 
                      edgecolor='black', facecolor='lightgray')
        table.add_cell(0, 1, width=0.2, height=0.1, text='Loop', loc='center', 
                      edgecolor='black', facecolor='lightgray')
        table.add_cell(0, 2, width=0.2, height=0.1, text='Braid Value', loc='center', 
                      edgecolor='black', facecolor='lightgray')
        table.add_cell(0, 3, width=0.4, height=0.1, text='Surface Position', loc='center', 
                      edgecolor='black', facecolor='lightgray')
        
        # Add data
        for i, row in enumerate(table_data):
            for j, cell in enumerate(row):
                width = 0.4 if j == 3 else 0.2
                table.add_cell(i+1, j, width=width, height=0.1, text=cell, loc='center', 
                              edgecolor='black')
        
        # Add the table to the plot
        ax.add_table(table)
        
        plt.title('DNA Sequence Mapping Details', pad=20)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dna_mapping_table.png", dpi=300)
        plt.close()
        
        # Save the properties as a text file
        with open(f"{output_dir}/dna_sequence_properties.txt", 'w') as f:
            f.write("DNA Sequence Properties:\n")
            f.write(f"Mean Braid Value: {properties['mean_braid_value']:.4f}\n")
            f.write(f"Std Dev of Braid Values: {properties['std_braid_value']:.4f}\n")
            f.write(f"Braid Entropy: {properties['braid_entropy']:.4f}\n")
            f.write(f"Closure Factor: {properties['closure_factor']:.4f}\n")
            f.write(f"Mean Time Factor: {properties['mean_time_factor']:.4f}\n")
            f.write("\nLoop Counts:\n")
            for loop, count in properties['loop_counts'].items():
                f.write(f"Loop {loop}: {count}\n")


def analyze_dna_sequence(sequence, output_dir="outputs/dna_mapping"):
    """
    Analyze a DNA sequence using the root tree eigenvalue log10 braids.
    
    Parameters
    ----------
    sequence : str
        The DNA sequence to analyze.
    output_dir : str, optional
        The output directory for visualizations.
    """
    print(f"Analyzing DNA sequence of length {len(sequence)}...")
    
    # Create the DNA sequence mapper
    mapper = DNASequenceMapper()
    
    # Map the DNA sequence
    mapping = mapper.map_dna_sequence(sequence)
    
    # Calculate sequence properties
    properties = mapper.calculate_sequence_properties(mapping)
    
    # Print the properties
    print("\nDNA Sequence Properties:")
    print(f"Mean Braid Value: {properties['mean_braid_value']:.4f}")
    print(f"Std Dev of Braid Values: {properties['std_braid_value']:.4f}")
    print(f"Braid Entropy: {properties['braid_entropy']:.4f}")
    print(f"Closure Factor: {properties['closure_factor']:.4f}")
    print(f"Mean Time Factor: {properties['mean_time_factor']:.4f}")
    
    print("\nLoop Counts:")
    for loop, count in properties['loop_counts'].items():
        print(f"Loop {loop}: {count}")
    
    # Visualize the mapping
    print("\nGenerating visualizations...")
    mapper.visualize_dna_mapping(mapping, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}/")
    
    return mapping, properties


def main():
    """Main function to demonstrate the DNA sequence mapping."""
    print("Initializing DNA Sequence Mapping...")
    
    # Example DNA sequence (part of a gene)
    # This is a simplified example - in reality, you would use a complete gene sequence
    sequence = "ATGGCGACCACAAACGAAACCCTGGTGAACGCGCTGGCGCAGATCGGCGCGCTGGTGCTGAACGTGCAGAAAGAGCTGCAGAAAAAAGCGAAA"
    
    # Analyze the DNA sequence
    analyze_dna_sequence(sequence)


if __name__ == "__main__":
    main()
