"""
Interface Calculations with Prime-Indexed Relations

This module implements interface calculations using prime-indexed relations,
providing insights into complex interfacial phenomena.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time

# Import Tibedo components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tibedo.core.prime_indexed.prime_indexed_structure import PrimeIndexedStructure
from tibedo.core.prime_indexed.modular_system import ModularSystem
from tibedo.core.advanced.quantum_state import ConfigurableQuantumState


class PrimeIndexedInterface:
    """
    Implementation of interface calculations using prime-indexed relations.
    
    This class provides methods for analyzing interfaces using the prime-indexed
    relations from the Tibedo Framework, providing insights into complex interfacial phenomena.
    """
    
    def __init__(self, max_prime_index=100, interface_resolution=0.1):
        """
        Initialize the PrimeIndexedInterface.
        
        Args:
            max_prime_index (int): Maximum index for prime-indexed structures
            interface_resolution (float): Resolution for interface calculations
        """
        self.max_prime_index = max_prime_index
        self.interface_resolution = interface_resolution
        
        # Create prime-indexed structure
        self.prime_structure = PrimeIndexedStructure(max_index=max_prime_index)
        
        # Create modular system for interface calculations
        self.modular_system = ModularSystem()
        
        # Create quantum state for interface representation
        self.quantum_state = ConfigurableQuantumState(dimension=7)
        
        # Initialize results
        self.interface_profile = None
        self.interface_energy = None
        self.interface_features = None
    
    def detect_interface(self, positions, atomic_numbers, direction=2, box_size=None):
        """
        Detect the interface between two phases.
        
        Args:
            positions (np.ndarray): Atomic positions (N, 3)
            atomic_numbers (np.ndarray): Atomic numbers (N,)
            direction (int): Direction perpendicular to the interface (0, 1, or 2)
            box_size (np.ndarray, optional): Box size for periodic boundary conditions (3,)
            
        Returns:
            tuple: (interface_positions, interface_width)
        """
        # Sort atoms by position in the interface direction
        sorted_indices = np.argsort(positions[:, direction])
        sorted_positions = positions[sorted_indices]
        sorted_atomic_numbers = atomic_numbers[sorted_indices]
        
        # Calculate density profile along the interface direction
        if box_size is not None:
            box_length = box_size[direction]
        else:
            box_length = np.max(positions[:, direction]) - np.min(positions[:, direction])
        
        # Create bins along the interface direction
        n_bins = int(box_length / self.interface_resolution)
        bins = np.linspace(0, box_length, n_bins + 1)
        
        # Calculate density profile
        density_profile = np.zeros(n_bins)
        atomic_type_profile = np.zeros(n_bins)
        
        for i in range(len(sorted_positions)):
            pos = sorted_positions[i, direction]
            if box_size is not None:
                pos = pos % box_size[direction]  # Apply periodic boundary conditions
            
            bin_idx = int(pos / self.interface_resolution)
            if bin_idx < n_bins:
                density_profile[bin_idx] += 1
                atomic_type_profile[bin_idx] += sorted_atomic_numbers[i]
        
        # Normalize atomic type profile
        with np.errstate(divide='ignore', invalid='ignore'):
            atomic_type_profile = np.divide(atomic_type_profile, density_profile)
            atomic_type_profile = np.nan_to_num(atomic_type_profile)
        
        # Smooth profiles
        density_profile = self._smooth_profile(density_profile)
        atomic_type_profile = self._smooth_profile(atomic_type_profile)
        
        # Detect interface using gradient of density or atomic type profile
        gradient = np.gradient(atomic_type_profile)
        
        # Find peaks in gradient (interface positions)
        interface_positions = []
        for i in range(1, len(gradient) - 1):
            if (gradient[i] > gradient[i-1] and gradient[i] > gradient[i+1] and gradient[i] > 0.1 * np.max(gradient)) or \
               (gradient[i] < gradient[i-1] and gradient[i] < gradient[i+1] and gradient[i] < 0.1 * np.min(gradient)):
                interface_positions.append(bins[i])
        
        # If no interface detected, use middle of the box
        if not interface_positions:
            interface_positions = [box_length / 2]
        
        # Estimate interface width using the width of the gradient peaks
        interface_width = 0
        for pos in interface_positions:
            bin_idx = int(pos / self.interface_resolution)
            
            # Find width of gradient peak
            left_idx = bin_idx
            while left_idx > 0 and abs(gradient[left_idx]) > 0.1 * abs(gradient[bin_idx]):
                left_idx -= 1
            
            right_idx = bin_idx
            while right_idx < len(gradient) - 1 and abs(gradient[right_idx]) > 0.1 * abs(gradient[bin_idx]):
                right_idx += 1
            
            width = (right_idx - left_idx) * self.interface_resolution
            interface_width += width
        
        interface_width /= len(interface_positions)
        
        # Store interface profile
        self.interface_profile = {
            'bins': bins,
            'density_profile': density_profile,
            'atomic_type_profile': atomic_type_profile,
            'gradient': gradient,
            'interface_positions': interface_positions,
            'interface_width': interface_width
        }
        
        return interface_positions, interface_width
    
    def _smooth_profile(self, profile, window_size=5):
        """
        Smooth a profile using a moving average.
        
        Args:
            profile (np.ndarray): Profile to smooth
            window_size (int): Window size for moving average
            
        Returns:
            np.ndarray: Smoothed profile
        """
        smoothed = np.zeros_like(profile)
        
        for i in range(len(profile)):
            start = max(0, i - window_size // 2)
            end = min(len(profile), i + window_size // 2 + 1)
            smoothed[i] = np.mean(profile[start:end])
        
        return smoothed
    
    def compute_interface_energy(self, positions, atomic_numbers, potential_func, direction=2, box_size=None):
        """
        Compute the interface energy.
        
        Args:
            positions (np.ndarray): Atomic positions (N, 3)
            atomic_numbers (np.ndarray): Atomic numbers (N,)
            potential_func (callable): Function to compute potential energy
            direction (int): Direction perpendicular to the interface (0, 1, or 2)
            box_size (np.ndarray, optional): Box size for periodic boundary conditions (3,)
            
        Returns:
            float: Interface energy
        """
        # Detect interface
        interface_positions, interface_width = self.detect_interface(
            positions, atomic_numbers, direction, box_size
        )
        
        # Calculate total energy
        total_energy = potential_func(positions, atomic_numbers)
        
        # Calculate bulk energies
        bulk_energies = []
        
        for interface_pos in interface_positions:
            # Define bulk regions (away from interfaces)
            bulk_mask = np.abs(positions[:, direction] - interface_pos) > 2 * interface_width
            
            if np.sum(bulk_mask) > 0:
                # Calculate energy of bulk atoms
                bulk_positions = positions[bulk_mask]
                bulk_atomic_numbers = atomic_numbers[bulk_mask]
                
                bulk_energy = potential_func(bulk_positions, bulk_atomic_numbers)
                bulk_energies.append(bulk_energy * len(positions) / len(bulk_positions))
        
        # If no bulk regions found, use average energy
        if not bulk_energies:
            bulk_energy = total_energy
        else:
            bulk_energy = np.mean(bulk_energies)
        
        # Calculate interface energy
        interface_energy = total_energy - bulk_energy
        
        # Normalize by interface area
        if box_size is not None:
            if direction == 0:
                interface_area = box_size[1] * box_size[2]
            elif direction == 1:
                interface_area = box_size[0] * box_size[2]
            else:  # direction == 2
                interface_area = box_size[0] * box_size[1]
        else:
            # Estimate area from maximum extent of positions
            min_pos = np.min(positions, axis=0)
            max_pos = np.max(positions, axis=0)
            
            if direction == 0:
                interface_area = (max_pos[1] - min_pos[1]) * (max_pos[2] - min_pos[2])
            elif direction == 1:
                interface_area = (max_pos[0] - min_pos[0]) * (max_pos[2] - min_pos[2])
            else:  # direction == 2
                interface_area = (max_pos[0] - min_pos[0]) * (max_pos[1] - min_pos[1])
        
        # Normalize by number of interfaces
        interface_energy /= (len(interface_positions) * interface_area)
        
        # Store interface energy
        self.interface_energy = interface_energy
        
        return interface_energy
    
    def compute_interface_features(self, positions, atomic_numbers, direction=2, box_size=None):
        """
        Compute prime-indexed features for the interface.
        
        Args:
            positions (np.ndarray): Atomic positions (N, 3)
            atomic_numbers (np.ndarray): Atomic numbers (N,)
            direction (int): Direction perpendicular to the interface (0, 1, or 2)
            box_size (np.ndarray, optional): Box size for periodic boundary conditions (3,)
            
        Returns:
            dict: Interface features
        """
        # Detect interface
        interface_positions, interface_width = self.detect_interface(
            positions, atomic_numbers, direction, box_size
        )
        
        # Initialize features
        features = {}
        
        # Calculate distance to nearest interface for each atom
        distances_to_interface = np.zeros(len(positions))
        
        for i in range(len(positions)):
            # Calculate distances to all interfaces
            distances = []
            for interface_pos in interface_positions:
                # Calculate distance with periodic boundary conditions if needed
                if box_size is not None:
                    dist = positions[i, direction] - interface_pos
                    dist = dist - np.round(dist / box_size[direction]) * box_size[direction]
                    distances.append(abs(dist))
                else:
                    distances.append(abs(positions[i, direction] - interface_pos))
            
            # Use minimum distance
            distances_to_interface[i] = min(distances)
        
        # Identify interface atoms
        interface_mask = distances_to_interface < interface_width
        interface_atoms = np.where(interface_mask)[0]
        
        # Calculate prime-indexed features for interface atoms
        prime_features = np.zeros((len(interface_atoms), self.max_prime_index))
        
        for i, atom_idx in enumerate(interface_atoms):
            # Calculate distances to other atoms
            for j, p in enumerate(self.prime_structure.primes[:self.max_prime_index]):
                # Find atoms within distance p
                neighbors = []
                for k in range(len(positions)):
                    if k != atom_idx:
                        # Calculate distance with periodic boundary conditions if needed
                        if box_size is not None:
                            diff = positions[k] - positions[atom_idx]
                            diff = diff - np.round(diff / box_size) * box_size
                            dist = np.linalg.norm(diff)
                        else:
                            dist = np.linalg.norm(positions[k] - positions[atom_idx])
                        
                        if dist < p:
                            neighbors.append((k, dist))
                
                # Calculate feature based on neighbors
                if neighbors:
                    # Weight by atomic number and distance
                    weighted_sum = 0
                    for k, dist in neighbors:
                        weighted_sum += atomic_numbers[k] * np.exp(-dist / p)
                    
                    prime_features[i, j] = weighted_sum / len(neighbors)
        
        # Store prime features
        features['prime_features'] = prime_features
        
        # Calculate modular system features
        modular_features = np.zeros((len(interface_atoms), 10))
        
        for i, atom_idx in enumerate(interface_atoms):
            # Use modular system to calculate features
            modular_features[i] = self.modular_system.compute_modular_features(
                positions[atom_idx],
                atomic_numbers[atom_idx],
                distances_to_interface[atom_idx] / interface_width
            )
        
        # Store modular features
        features['modular_features'] = modular_features
        
        # Calculate quantum state features
        quantum_features = np.zeros((len(interface_atoms), 7))
        
        for i, atom_idx in enumerate(interface_atoms):
            # Create parameters for quantum state
            parameters = {
                'phase_factors': np.ones(7),
                'amplitude_factors': np.ones(7) / np.sqrt(7),
                'entanglement_pattern': 'interface',
                'cyclotomic_parameters': {'n': 7, 'k': 1},
                'symmetry_breaking': distances_to_interface[atom_idx] / interface_width,
                'entropic_decline': 0.0
            }
            
            # Configure quantum state
            self.quantum_state.configure(parameters)
            
            # Compute quantum features
            quantum_features[i] = self.quantum_state.compute_features(
                positions[atom_idx],
                atomic_numbers[atom_idx]
            )
        
        # Store quantum features
        features['quantum_features'] = quantum_features
        
        # Store all features
        self.interface_features = features
        
        return features
    
    def predict_interface_properties(self, features, property_type='energy'):
        """
        Predict interface properties using prime-indexed features.
        
        Args:
            features (dict): Interface features
            property_type (str): Type of property to predict ('energy', 'tension', or 'stability')
            
        Returns:
            float: Predicted property value
        """
        # Simple linear model for prediction
        if property_type == 'energy':
            # Use prime features for energy prediction
            prime_features = features['prime_features']
            
            # Calculate mean of each feature
            mean_features = np.mean(prime_features, axis=0)
            
            # Simple weighted sum model
            weights = np.linspace(1.0, 0.1, len(mean_features))
            predicted_value = np.sum(mean_features * weights) / np.sum(weights)
            
        elif property_type == 'tension':
            # Use modular features for tension prediction
            modular_features = features['modular_features']
            
            # Calculate mean of each feature
            mean_features = np.mean(modular_features, axis=0)
            
            # Simple weighted sum model
            weights = np.linspace(1.0, 0.1, len(mean_features))
            predicted_value = np.sum(mean_features * weights) / np.sum(weights)
            
        elif property_type == 'stability':
            # Use quantum features for stability prediction
            quantum_features = features['quantum_features']
            
            # Calculate mean of each feature
            mean_features = np.mean(quantum_features, axis=0)
            
            # Simple weighted sum model
            weights = np.linspace(1.0, 0.1, len(mean_features))
            predicted_value = np.sum(mean_features * weights) / np.sum(weights)
            
        else:
            raise ValueError(f"Unknown property type: {property_type}")
        
        return predicted_value
    
    def optimize_interface_structure(self, positions, atomic_numbers, potential_func, direction=2, box_size=None):
        """
        Optimize the interface structure to minimize energy.
        
        Args:
            positions (np.ndarray): Initial atomic positions (N, 3)
            atomic_numbers (np.ndarray): Atomic numbers (N,)
            potential_func (callable): Function to compute potential energy
            direction (int): Direction perpendicular to the interface (0, 1, or 2)
            box_size (np.ndarray, optional): Box size for periodic boundary conditions (3,)
            
        Returns:
            np.ndarray: Optimized atomic positions
        """
        # Detect interface
        interface_positions, interface_width = self.detect_interface(
            positions, atomic_numbers, direction, box_size
        )
        
        # Identify interface atoms
        distances_to_interface = np.zeros(len(positions))
        
        for i in range(len(positions)):
            # Calculate distances to all interfaces
            distances = []
            for interface_pos in interface_positions:
                # Calculate distance with periodic boundary conditions if needed
                if box_size is not None:
                    dist = positions[i, direction] - interface_pos
                    dist = dist - np.round(dist / box_size[direction]) * box_size[direction]
                    distances.append(abs(dist))
                else:
                    distances.append(abs(positions[i, direction] - interface_pos))
            
            # Use minimum distance
            distances_to_interface[i] = min(distances)
        
        # Identify interface atoms
        interface_mask = distances_to_interface < interface_width
        interface_atoms = np.where(interface_mask)[0]
        
        # Define objective function for optimization
        def objective(x):
            # Reshape flat array to positions
            new_positions = positions.copy()
            
            # Update positions of interface atoms
            for i, atom_idx in enumerate(interface_atoms):
                new_positions[atom_idx] = x.reshape(-1, 3)[i]
                
                # Apply periodic boundary conditions if needed
                if box_size is not None:
                    new_positions[atom_idx] = new_positions[atom_idx] % box_size
            
            # Calculate energy
            energy = potential_func(new_positions, atomic_numbers)
            
            return energy
        
        # Initial guess (current positions of interface atoms)
        x0 = positions[interface_atoms].flatten()
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 100, 'disp': False}
        )
        
        # Update positions with optimized values
        optimized_positions = positions.copy()
        
        # Update positions of interface atoms
        for i, atom_idx in enumerate(interface_atoms):
            optimized_positions[atom_idx] = result.x.reshape(-1, 3)[i]
            
            # Apply periodic boundary conditions if needed
            if box_size is not None:
                optimized_positions[atom_idx] = optimized_positions[atom_idx] % box_size
        
        return optimized_positions
    
    def visualize_interface_profile(self):
        """
        Visualize the interface profile.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if self.interface_profile is None:
            raise ValueError("Interface profile not computed yet")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot density profile
        ax1.plot(self.interface_profile['bins'][:-1], self.interface_profile['density_profile'], 'k-', linewidth=2)
        
        # Mark interface positions
        for pos in self.interface_profile['interface_positions']:
            ax1.axvline(x=pos, color='r', linestyle='--', alpha=0.7)
        
        ax1.set_ylabel('Density')
        ax1.set_title('Interface Profile')
        ax1.grid(True, alpha=0.3)
        
        # Plot atomic type profile
        ax2.plot(self.interface_profile['bins'][:-1], self.interface_profile['atomic_type_profile'], 'b-', linewidth=2)
        
        # Mark interface positions
        for pos in self.interface_profile['interface_positions']:
            ax2.axvline(x=pos, color='r', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Atomic Type')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_interface_features(self):
        """
        Visualize the interface features.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if self.interface_features is None:
            raise ValueError("Interface features not computed yet")
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot prime features
        prime_features = self.interface_features['prime_features']
        im1 = ax1.imshow(prime_features, aspect='auto', cmap='viridis')
        ax1.set_xlabel('Prime Index')
        ax1.set_ylabel('Atom Index')
        ax1.set_title('Prime-Indexed Features')
        plt.colorbar(im1, ax=ax1)
        
        # Plot modular features
        modular_features = self.interface_features['modular_features']
        im2 = ax2.imshow(modular_features, aspect='auto', cmap='plasma')
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Atom Index')
        ax2.set_title('Modular System Features')
        plt.colorbar(im2, ax=ax2)
        
        # Plot quantum features
        quantum_features = self.interface_features['quantum_features']
        im3 = ax3.imshow(quantum_features, aspect='auto', cmap='inferno')
        ax3.set_xlabel('Feature Index')
        ax3.set_ylabel('Atom Index')
        ax3.set_title('Quantum State Features')
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_interface_structure(self, positions, atomic_numbers, direction=2, box_size=None):
        """
        Visualize the interface structure.
        
        Args:
            positions (np.ndarray): Atomic positions (N, 3)
            atomic_numbers (np.ndarray): Atomic numbers (N,)
            direction (int): Direction perpendicular to the interface (0, 1, or 2)
            box_size (np.ndarray, optional): Box size for periodic boundary conditions (3,)
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        # Detect interface
        interface_positions, interface_width = self.detect_interface(
            positions, atomic_numbers, direction, box_size
        )
        
        # Calculate distance to nearest interface for each atom
        distances_to_interface = np.zeros(len(positions))
        
        for i in range(len(positions)):
            # Calculate distances to all interfaces
            distances = []
            for interface_pos in interface_positions:
                # Calculate distance with periodic boundary conditions if needed
                if box_size is not None:
                    dist = positions[i, direction] - interface_pos
                    dist = dist - np.round(dist / box_size[direction]) * box_size[direction]
                    distances.append(abs(dist))
                else:
                    distances.append(abs(positions[i, direction] - interface_pos))
            
            # Use minimum distance
            distances_to_interface[i] = min(distances)
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        
        # Create 3D plot
        ax = fig.add_subplot(111, projection='3d')
        
        # Define colors for different atomic numbers
        colors = {
            1: 'white',   # H
            6: 'gray',    # C
            7: 'blue',    # N
            8: 'red',     # O
            16: 'yellow'  # S
        }
        
        # Define sizes for different atomic numbers
        sizes = {
            1: 50,    # H
            6: 100,   # C
            7: 100,   # N
            8: 100,   # O
            16: 150   # S
        }
        
        # Plot atoms
        for i in range(len(positions)):
            color = colors.get(atomic_numbers[i], 'green')
            size = sizes.get(atomic_numbers[i], 100)
            
            # Adjust transparency based on distance to interface
            alpha = 1.0 if distances_to_interface[i] < interface_width else 0.3
            
            ax.scatter(
                positions[i, 0],
                positions[i, 1],
                positions[i, 2],
                c=color,
                s=size,
                alpha=alpha,
                edgecolors='black' if distances_to_interface[i] < interface_width else None
            )
        
        # Plot interface planes
        if box_size is not None:
            for pos in interface_positions:
                if direction == 0:
                    # YZ plane
                    xx, yy = np.meshgrid([pos, pos], [0, box_size[1]], [0, box_size[2]])
                    ax.plot_surface(xx, yy, zz, alpha=0.2, color='red')
                elif direction == 1:
                    # XZ plane
                    xx, zz = np.meshgrid([0, box_size[0]], [pos, pos], [0, box_size[2]])
                    ax.plot_surface(xx, yy, zz, alpha=0.2, color='red')
                else:  # direction == 2
                    # XY plane
                    xx, yy = np.meshgrid([0, box_size[0]], [0, box_size[1]])
                    zz = np.ones_like(xx) * pos
                    ax.plot_surface(xx, yy, zz, alpha=0.2, color='red')
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Interface Structure')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        return fig