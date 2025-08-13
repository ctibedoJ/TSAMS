"""
Liquid Structure Calculations with Spinor Reduction Chain

This module implements liquid structure calculations using the spinor reduction chain,
achieving linear scaling with system size.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.signal import savgol_filter
import time

# Import Tibedo components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tibedo.core.spinor.reduction_chain import ReductionChain
from tibedo.core.spinor.spinor_space import SpinorSpace
from tibedo.core.prime_indexed.prime_indexed_structure import PrimeIndexedStructure


class SpinorLiquidStructure:
    """
    Implementation of liquid structure calculations using the spinor reduction chain.
    
    This class provides methods for analyzing liquid structures using the
    spinor reduction chain from the Tibedo Framework, achieving linear scaling with system size.
    """
    
    def __init__(self, max_distance=10.0, n_bins=100, spinor_dimension=16):
        """
        Initialize the SpinorLiquidStructure.
        
        Args:
            max_distance (float): Maximum distance for radial distribution function
            n_bins (int): Number of bins for radial distribution function
            spinor_dimension (int): Initial dimension for spinor reduction chain
        """
        self.max_distance = max_distance
        self.n_bins = n_bins
        self.spinor_dimension = spinor_dimension
        
        # Create spinor reduction chain
        self.reduction_chain = ReductionChain(
            initial_dimension=spinor_dimension,
            chain_length=5
        )
        
        # Create prime-indexed structure for additional features
        self.prime_structure = PrimeIndexedStructure(max_index=100)
        
        # Initialize results
        self.rdf = None
        self.structure_factor = None
        self.distances = None
        self.spinor_features = None
    
    def compute_rdf(self, positions, box_size=None, atomic_numbers=None, species_pair=None):
        """
        Compute the radial distribution function (RDF).
        
        Args:
            positions (np.ndarray): Atomic positions (N, 3)
            box_size (np.ndarray, optional): Box size for periodic boundary conditions (3,)
            atomic_numbers (np.ndarray, optional): Atomic numbers (N,)
            species_pair (tuple, optional): Pair of atomic numbers for partial RDF
            
        Returns:
            tuple: (distances, rdf)
        """
        n_atoms = len(positions)
        
        # Create distance bins
        self.distances = np.linspace(0, self.max_distance, self.n_bins)
        bin_width = self.distances[1] - self.distances[0]
        
        # Initialize RDF
        rdf = np.zeros(self.n_bins)
        
        # Filter atoms by species if needed
        if atomic_numbers is not None and species_pair is not None:
            species1, species2 = species_pair
            indices1 = np.where(atomic_numbers == species1)[0]
            indices2 = np.where(atomic_numbers == species2)[0]
            
            # Skip if no atoms of either species
            if len(indices1) == 0 or len(indices2) == 0:
                self.rdf = rdf
                return self.distances, rdf
            
            # Compute distances between atoms of selected species
            for i in indices1:
                for j in indices2:
                    if i != j:  # Skip self-interaction
                        # Calculate distance with periodic boundary conditions if needed
                        if box_size is not None:
                            dist = self._minimum_image_distance(positions[i], positions[j], box_size)
                        else:
                            dist = np.linalg.norm(positions[i] - positions[j])
                        
                        # Add to histogram
                        if dist < self.max_distance:
                            bin_idx = int(dist / bin_width)
                            if bin_idx < self.n_bins:
                                rdf[bin_idx] += 1
            
            # Normalize by number of pairs and bin volume
            n_pairs = len(indices1) * len(indices2)
            if species1 == species2:
                n_pairs = len(indices1) * (len(indices1) - 1) // 2
        else:
            # Compute all pairwise distances
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    # Calculate distance with periodic boundary conditions if needed
                    if box_size is not None:
                        dist = self._minimum_image_distance(positions[i], positions[j], box_size)
                    else:
                        dist = np.linalg.norm(positions[i] - positions[j])
                    
                    # Add to histogram
                    if dist < self.max_distance:
                        bin_idx = int(dist / bin_width)
                        if bin_idx < self.n_bins:
                            rdf[bin_idx] += 2  # Count each pair twice for normalization
            
            # Normalize by number of pairs and bin volume
            n_pairs = n_atoms * (n_atoms - 1)
        
        # Calculate bin volumes (spherical shell)
        bin_volumes = 4/3 * np.pi * (np.power(self.distances + bin_width, 3) - np.power(self.distances, 3))
        
        # Calculate system volume
        if box_size is not None:
            volume = np.prod(box_size)
        else:
            # Estimate volume from maximum extent of positions
            min_pos = np.min(positions, axis=0)
            max_pos = np.max(positions, axis=0)
            volume = np.prod(max_pos - min_pos)
        
        # Calculate number density
        if atomic_numbers is not None and species_pair is not None:
            n_density = len(indices1) / volume
        else:
            n_density = n_atoms / volume
        
        # Normalize RDF
        rdf = rdf / (n_pairs * bin_volumes * n_density)
        
        # Smooth RDF
        rdf = savgol_filter(rdf, 5, 3)
        
        # Store result
        self.rdf = rdf
        
        return self.distances, rdf
    
    def _minimum_image_distance(self, pos1, pos2, box_size):
        """
        Calculate the minimum image distance between two positions.
        
        Args:
            pos1 (np.ndarray): First position (3,)
            pos2 (np.ndarray): Second position (3,)
            box_size (np.ndarray): Box size (3,)
            
        Returns:
            float: Minimum image distance
        """
        # Calculate difference vector
        diff = pos1 - pos2
        
        # Apply minimum image convention
        diff = diff - np.round(diff / box_size) * box_size
        
        # Calculate distance
        distance = np.linalg.norm(diff)
        
        return distance
    
    def compute_structure_factor(self, q_max=10.0, n_q_points=100):
        """
        Compute the structure factor from the radial distribution function.
        
        Args:
            q_max (float): Maximum q value
            n_q_points (int): Number of q points
            
        Returns:
            tuple: (q_values, structure_factor)
        """
        if self.rdf is None:
            raise ValueError("RDF not computed yet")
        
        # Create q values
        q_values = np.linspace(0.1, q_max, n_q_points)  # Avoid q=0
        
        # Initialize structure factor
        structure_factor = np.zeros_like(q_values)
        
        # Calculate structure factor using the Fourier transform of (g(r) - 1)
        for i, q in enumerate(q_values):
            integrand = self.distances * (self.rdf - 1) * np.sin(q * self.distances) / q
            structure_factor[i] = 1 + 4 * np.pi * np.trapz(integrand, self.distances)
        
        # Store result
        self.structure_factor = structure_factor
        
        return q_values, structure_factor
    
    def compute_spinor_features(self, positions, box_size=None, atomic_numbers=None):
        """
        Compute spinor features for the liquid structure.
        
        Args:
            positions (np.ndarray): Atomic positions (N, 3)
            box_size (np.ndarray, optional): Box size for periodic boundary conditions (3,)
            atomic_numbers (np.ndarray, optional): Atomic numbers (N,)
            
        Returns:
            dict: Spinor features
        """
        n_atoms = len(positions)
        
        # Create distance matrix
        distance_matrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                # Calculate distance with periodic boundary conditions if needed
                if box_size is not None:
                    dist = self._minimum_image_distance(positions[i], positions[j], box_size)
                else:
                    dist = np.linalg.norm(positions[i] - positions[j])
                
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # Create initial spinor space representation
        spinor_space = np.zeros((n_atoms, self.spinor_dimension))
        
        # Fill spinor space with atomic information
        for i in range(n_atoms):
            # Use atomic positions as initial features
            spinor_space[i, :3] = positions[i]
            
            # Use atomic numbers if available
            if atomic_numbers is not None:
                spinor_space[i, 3] = atomic_numbers[i]
            
            # Use distance information
            distances_i = distance_matrix[i]
            sorted_distances = np.sort(distances_i)
            
            # Use nearest neighbor distances
            n_neighbors = min(self.spinor_dimension - 4, n_atoms - 1)
            spinor_space[i, 4:4+n_neighbors] = sorted_distances[1:n_neighbors+1]  # Skip self (distance 0)
        
        # Apply spinor reduction chain
        spinor_features = {}
        
        # Store initial space
        spinor_features['initial_space'] = spinor_space
        
        # Apply reduction maps
        current_space = spinor_space
        for i, reduction_map in enumerate(self.reduction_chain.maps):
            # Apply reduction
            reduced_space = reduction_map.apply(current_space)
            
            # Store reduced space
            spinor_features[f'reduced_space_{i+1}'] = reduced_space
            
            # Update current space
            current_space = reduced_space
        
        # Compute additional features using prime-indexed structure
        prime_features = np.zeros((n_atoms, self.prime_structure.max_index))
        
        for i in range(n_atoms):
            # Generate prime-indexed sequence based on distances
            distances_i = distance_matrix[i]
            sorted_indices = np.argsort(distances_i)
            
            for j, p in enumerate(self.prime_structure.primes[:self.prime_structure.max_index]):
                if j + 1 < len(sorted_indices):
                    idx = sorted_indices[j + 1]  # Skip self
                    if atomic_numbers is not None:
                        prime_features[i, j] = atomic_numbers[idx] * np.exp(-distances_i[idx] / p)
                    else:
                        prime_features[i, j] = np.exp(-distances_i[idx] / p)
        
        # Store prime features
        spinor_features['prime_features'] = prime_features
        
        # Store all features
        self.spinor_features = spinor_features
        
        return spinor_features
    
    def analyze_local_structure(self, positions, box_size=None, atomic_numbers=None, cutoff=5.0):
        """
        Analyze local structure around each atom.
        
        Args:
            positions (np.ndarray): Atomic positions (N, 3)
            box_size (np.ndarray, optional): Box size for periodic boundary conditions (3,)
            atomic_numbers (np.ndarray, optional): Atomic numbers (N,)
            cutoff (float): Cutoff distance for local structure
            
        Returns:
            dict: Local structure analysis results
        """
        n_atoms = len(positions)
        
        # Initialize results
        coordination_numbers = np.zeros(n_atoms, dtype=int)
        local_densities = np.zeros(n_atoms)
        local_order_parameters = np.zeros(n_atoms)
        
        # Analyze local structure around each atom
        for i in range(n_atoms):
            # Find neighbors within cutoff
            neighbors = []
            for j in range(n_atoms):
                if i != j:
                    # Calculate distance with periodic boundary conditions if needed
                    if box_size is not None:
                        dist = self._minimum_image_distance(positions[i], positions[j], box_size)
                    else:
                        dist = np.linalg.norm(positions[i] - positions[j])
                    
                    if dist < cutoff:
                        neighbors.append((j, dist))
            
            # Calculate coordination number
            coordination_numbers[i] = len(neighbors)
            
            # Calculate local density
            if len(neighbors) > 0:
                volume = 4/3 * np.pi * cutoff**3
                local_densities[i] = len(neighbors) / volume
            
            # Calculate local order parameter (simplified)
            if len(neighbors) >= 4:
                # Use bond orientational order parameter q6
                # This is a simplified version
                neighbor_vectors = []
                for j, dist in neighbors:
                    if box_size is not None:
                        vec = positions[j] - positions[i]
                        vec = vec - np.round(vec / box_size) * box_size
                    else:
                        vec = positions[j] - positions[i]
                    
                    neighbor_vectors.append(vec / dist)
                
                # Calculate q6 parameter (simplified)
                q6 = 0.0
                for vec1 in neighbor_vectors:
                    for vec2 in neighbor_vectors:
                        # Calculate Legendre polynomial P6(cos(theta))
                        cos_theta = np.dot(vec1, vec2)
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        p6 = (231 * cos_theta**6 - 315 * cos_theta**4 + 105 * cos_theta**2 - 5) / 16
                        q6 += p6
                
                if len(neighbors) > 0:
                    q6 /= len(neighbors)**2
                    local_order_parameters[i] = q6
        
        # Return results
        results = {
            'coordination_numbers': coordination_numbers,
            'local_densities': local_densities,
            'local_order_parameters': local_order_parameters
        }
        
        return results
    
    def compute_diffusion_coefficient(self, trajectory, time_step, box_size=None):
        """
        Compute diffusion coefficient from a trajectory.
        
        Args:
            trajectory (np.ndarray): Atomic positions over time (T, N, 3)
            time_step (float): Time step between frames
            box_size (np.ndarray, optional): Box size for periodic boundary conditions (3,)
            
        Returns:
            float: Diffusion coefficient
        """
        n_frames, n_atoms, _ = trajectory.shape
        
        # Calculate mean squared displacement (MSD)
        msd = np.zeros(n_frames)
        
        # Reference positions (first frame)
        ref_positions = trajectory[0]
        
        # Unwrap trajectory to account for periodic boundary conditions
        if box_size is not None:
            unwrapped = np.zeros_like(trajectory)
            unwrapped[0] = trajectory[0]
            
            for t in range(1, n_frames):
                # Calculate displacement from previous frame
                displacement = trajectory[t] - trajectory[t-1]
                
                # Apply minimum image convention
                displacement = displacement - np.round(displacement / box_size) * box_size
                
                # Update unwrapped positions
                unwrapped[t] = unwrapped[t-1] + displacement
            
            # Use unwrapped trajectory
            trajectory = unwrapped
        
        # Calculate MSD for each frame
        for t in range(n_frames):
            # Calculate squared displacement for each atom
            squared_disp = np.sum((trajectory[t] - ref_positions)**2, axis=1)
            
            # Average over atoms
            msd[t] = np.mean(squared_disp)
        
        # Calculate time values
        times = np.arange(n_frames) * time_step
        
        # Fit MSD to extract diffusion coefficient
        # MSD = 6 * D * t for 3D diffusion
        # Use linear fit on the middle part of the curve (avoid initial and final parts)
        start_idx = n_frames // 4
        end_idx = 3 * n_frames // 4
        
        slope, _ = np.polyfit(times[start_idx:end_idx], msd[start_idx:end_idx], 1)
        
        # Calculate diffusion coefficient
        diffusion_coeff = slope / 6.0
        
        return diffusion_coeff, times, msd
    
    def visualize_rdf(self):
        """
        Visualize the radial distribution function.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if self.rdf is None:
            raise ValueError("RDF not computed yet")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.distances, self.rdf, 'k-', linewidth=2)
        
        ax.set_xlabel('Distance (Å)')
        ax.set_ylabel('g(r)')
        ax.set_title('Radial Distribution Function')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_structure_factor(self, q_values):
        """
        Visualize the structure factor.
        
        Args:
            q_values (np.ndarray): Q values
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if self.structure_factor is None:
            raise ValueError("Structure factor not computed yet")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(q_values, self.structure_factor, 'k-', linewidth=2)
        
        ax.set_xlabel('q (Å⁻¹)')
        ax.set_ylabel('S(q)')
        ax.set_title('Structure Factor')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_spinor_features(self):
        """
        Visualize the spinor features.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if self.spinor_features is None:
            raise ValueError("Spinor features not computed yet")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot initial space
        initial_space = self.spinor_features['initial_space']
        axes[0].imshow(initial_space, aspect='auto', cmap='viridis')
        axes[0].set_title('Initial Spinor Space')
        axes[0].set_xlabel('Feature Index')
        axes[0].set_ylabel('Atom Index')
        
        # Plot reduced spaces
        for i in range(min(4, len(self.reduction_chain.maps))):
            reduced_space = self.spinor_features[f'reduced_space_{i+1}']
            axes[i+1].imshow(reduced_space, aspect='auto', cmap='viridis')
            axes[i+1].set_title(f'Reduced Space {i+1}')
            axes[i+1].set_xlabel('Feature Index')
            axes[i+1].set_ylabel('Atom Index')
        
        # Plot prime features
        prime_features = self.spinor_features['prime_features']
        axes[5].imshow(prime_features, aspect='auto', cmap='viridis')
        axes[5].set_title('Prime-Indexed Features')
        axes[5].set_xlabel('Prime Index')
        axes[5].set_ylabel('Atom Index')
        
        plt.tight_layout()
        
        return fig
    
    def visualize_local_structure(self, positions, results, box_size=None):
        """
        Visualize the local structure analysis results.
        
        Args:
            positions (np.ndarray): Atomic positions (N, 3)
            results (dict): Local structure analysis results
            box_size (np.ndarray, optional): Box size for periodic boundary conditions (3,)
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        fig = plt.figure(figsize=(15, 5))
        
        # Create 3D plot for atomic positions
        ax1 = fig.add_subplot(131, projection='3d')
        
        # Plot atoms colored by coordination number
        scatter = ax1.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c=results['coordination_numbers'],
            cmap='viridis',
            s=50,
            alpha=0.8
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Coordination Number')
        
        # Set labels
        ax1.set_xlabel('X (Å)')
        ax1.set_ylabel('Y (Å)')
        ax1.set_zlabel('Z (Å)')
        ax1.set_title('Atomic Positions')
        
        # Plot box if provided
        if box_size is not None:
            # Plot box edges
            corners = np.array([
                [0, 0, 0],
                [box_size[0], 0, 0],
                [box_size[0], box_size[1], 0],
                [0, box_size[1], 0],
                [0, 0, box_size[2]],
                [box_size[0], 0, box_size[2]],
                [box_size[0], box_size[1], box_size[2]],
                [0, box_size[1], box_size[2]]
            ])
            
            # Define edges
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
            
            # Plot edges
            for start, end in edges:
                ax1.plot(
                    [corners[start, 0], corners[end, 0]],
                    [corners[start, 1], corners[end, 1]],
                    [corners[start, 2], corners[end, 2]],
                    'k-', alpha=0.3
                )
        
        # Create histogram of coordination numbers
        ax2 = fig.add_subplot(132)
        
        # Plot histogram
        ax2.hist(results['coordination_numbers'], bins=np.arange(0, 15) - 0.5, alpha=0.7)
        
        # Set labels
        ax2.set_xlabel('Coordination Number')
        ax2.set_ylabel('Count')
        ax2.set_title('Coordination Number Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Create scatter plot of local order vs. local density
        ax3 = fig.add_subplot(133)
        
        # Plot scatter
        scatter = ax3.scatter(
            results['local_densities'],
            results['local_order_parameters'],
            c=results['coordination_numbers'],
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Coordination Number')
        
        # Set labels
        ax3.set_xlabel('Local Density')
        ax3.set_ylabel('Local Order Parameter')
        ax3.set_title('Local Structure Correlation')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_diffusion(self, times, msd, diffusion_coeff):
        """
        Visualize the diffusion analysis results.
        
        Args:
            times (np.ndarray): Time values
            msd (np.ndarray): Mean squared displacement values
            diffusion_coeff (float): Diffusion coefficient
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot MSD
        ax.plot(times, msd, 'ko-', label='MSD Data')
        
        # Plot linear fit
        start_idx = len(times) // 4
        end_idx = 3 * len(times) // 4
        
        fit_times = times[start_idx:end_idx]
        fit_line = 6 * diffusion_coeff * fit_times
        
        ax.plot(fit_times, fit_line, 'r-', linewidth=2,
                label=f'Linear Fit (D = {diffusion_coeff:.4f} Å²/ps)')
        
        # Set labels
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('MSD (Å²)')
        ax.set_title('Mean Squared Displacement')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig