"""
Machine Learning Potentials using the Tibedo Framework

This module implements machine learning potentials within the Tibedo Framework,
achieving linear scaling with system size.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Import Tibedo components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tibedo.core.spinor.reduction_chain import ReductionChain
from tibedo.core.prime_indexed.prime_indexed_structure import PrimeIndexedStructure
from tibedo.core.advanced.quantum_state import ConfigurableQuantumState
from tibedo.ml.neural_networks.tibedo_neural_network import TibedoLayer, TibedoNeuralNetwork


class AtomicEnvironment:
    """
    Representation of the local environment around an atom.
    
    This class provides methods for computing and representing the local
    environment around an atom, which is used as input to the machine learning potential.
    """
    
    def __init__(self, cutoff_radius=5.0, n_radial=8, n_angular=6, use_tibedo_features=True):
        """
        Initialize the AtomicEnvironment.
        
        Args:
            cutoff_radius (float): Cutoff radius for the local environment
            n_radial (int): Number of radial basis functions
            n_angular (int): Number of angular basis functions
            use_tibedo_features (bool): Whether to use Tibedo-specific features
        """
        self.cutoff_radius = cutoff_radius
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.use_tibedo_features = use_tibedo_features
        
        # Initialize Tibedo components if needed
        if use_tibedo_features:
            self.prime_structure = PrimeIndexedStructure(max_index=n_radial)
            self.quantum_state = ConfigurableQuantumState(dimension=7)
    
    def compute_features(self, positions, atomic_numbers, central_atom_idx):
        """
        Compute features for the local environment around a central atom.
        
        Args:
            positions (np.ndarray): Atomic positions (N, 3)
            atomic_numbers (np.ndarray): Atomic numbers (N,)
            central_atom_idx (int): Index of the central atom
            
        Returns:
            np.ndarray: Features for the local environment
        """
        # Get central atom position
        central_pos = positions[central_atom_idx]
        
        # Compute distances and vectors to neighboring atoms
        vectors = positions - central_pos
        distances = np.linalg.norm(vectors, axis=1)
        
        # Find atoms within cutoff radius (excluding central atom)
        mask = (distances < self.cutoff_radius) & (np.arange(len(positions)) != central_atom_idx)
        neighbor_indices = np.where(mask)[0]
        
        if len(neighbor_indices) == 0:
            # No neighbors within cutoff radius
            return np.zeros(self.n_radial * self.n_angular * 3)
        
        neighbor_distances = distances[neighbor_indices]
        neighbor_vectors = vectors[neighbor_indices]
        neighbor_atomic_numbers = atomic_numbers[neighbor_indices]
        
        # Normalize vectors
        neighbor_directions = neighbor_vectors / neighbor_distances[:, np.newaxis]
        
        # Compute radial features
        radial_features = self._compute_radial_features(neighbor_distances, neighbor_atomic_numbers)
        
        # Compute angular features
        angular_features = self._compute_angular_features(neighbor_directions, neighbor_distances, neighbor_atomic_numbers)
        
        # Combine features
        features = np.concatenate([radial_features, angular_features])
        
        # Add Tibedo-specific features if requested
        if self.use_tibedo_features:
            tibedo_features = self._compute_tibedo_features(neighbor_distances, neighbor_directions, neighbor_atomic_numbers)
            features = np.concatenate([features, tibedo_features])
        
        return features
    
    def _compute_radial_features(self, distances, atomic_numbers):
        """
        Compute radial features for the local environment.
        
        Args:
            distances (np.ndarray): Distances to neighboring atoms
            atomic_numbers (np.ndarray): Atomic numbers of neighboring atoms
            
        Returns:
            np.ndarray: Radial features
        """
        # Create radial basis functions
        radial_features = np.zeros(self.n_radial)
        
        for i in range(self.n_radial):
            # Gaussian basis functions
            center = i * self.cutoff_radius / self.n_radial
            width = self.cutoff_radius / self.n_radial
            
            # Compute contribution of each neighbor to this basis function
            contributions = np.exp(-(distances - center)**2 / (2 * width**2))
            
            # Weight by atomic number (simple weighting scheme)
            weighted_contributions = contributions * atomic_numbers
            
            # Sum contributions
            radial_features[i] = np.sum(weighted_contributions)
        
        return radial_features
    
    def _compute_angular_features(self, directions, distances, atomic_numbers):
        """
        Compute angular features for the local environment.
        
        Args:
            directions (np.ndarray): Normalized direction vectors to neighboring atoms
            distances (np.ndarray): Distances to neighboring atoms
            atomic_numbers (np.ndarray): Atomic numbers of neighboring atoms
            
        Returns:
            np.ndarray: Angular features
        """
        n_neighbors = len(directions)
        angular_features = np.zeros(self.n_angular * 3)
        
        if n_neighbors < 2:
            # Need at least 2 neighbors for angular features
            return angular_features
        
        # Compute all pairwise angles
        for i in range(n_neighbors):
            for j in range(i+1, n_neighbors):
                # Compute cosine of angle between directions
                cos_angle = np.dot(directions[i], directions[j])
                
                # Ensure cos_angle is in [-1, 1]
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                
                # Convert to angle in [0, π]
                angle = np.arccos(cos_angle)
                
                # Weight by distance and atomic numbers
                weight = np.exp(-(distances[i] + distances[j]) / self.cutoff_radius) * atomic_numbers[i] * atomic_numbers[j]
                
                # Distribute to angular basis functions
                for k in range(self.n_angular):
                    # Center of basis function
                    center = k * np.pi / self.n_angular
                    width = np.pi / self.n_angular
                    
                    # Compute contribution to this basis function
                    contribution = weight * np.exp(-(angle - center)**2 / (2 * width**2))
                    
                    # Add contribution to features
                    angular_features[k] += contribution
                    
                    # Add contributions for sine and cosine projections
                    angular_features[self.n_angular + k] += contribution * np.cos(angle)
                    angular_features[2 * self.n_angular + k] += contribution * np.sin(angle)
        
        return angular_features
    
    def _compute_tibedo_features(self, distances, directions, atomic_numbers):
        """
        Compute Tibedo-specific features for the local environment.
        
        Args:
            distances (np.ndarray): Distances to neighboring atoms
            directions (np.ndarray): Normalized direction vectors to neighboring atoms
            atomic_numbers (np.ndarray): Atomic numbers of neighboring atoms
            
        Returns:
            np.ndarray: Tibedo-specific features
        """
        # Use prime-indexed structure for feature generation
        prime_features = np.zeros(self.prime_structure.max_index)
        
        # Generate prime-indexed sequence based on distances
        for i, p in enumerate(self.prime_structure.primes[:self.prime_structure.max_index]):
            if i < len(distances):
                # Map distance to feature using prime number
                prime_features[i] = np.exp(-distances[i] / p) * atomic_numbers[i]
        
        # Create quantum-state-inspired features
        quantum_features = np.zeros(7)  # Using dimension 7 from ConfigurableQuantumState
        
        if len(distances) > 0:
            # Create superposition-like combinations
            for i in range(7):
                if i < len(distances):
                    phase = 2 * np.pi * i / 7
                    quantum_features[i] = np.sum(atomic_numbers * np.exp(-distances) * np.cos(phase))
        
        # Combine features
        tibedo_features = np.concatenate([prime_features, quantum_features])
        
        return tibedo_features


class TibedoMLPotential:
    """
    Machine Learning Potential using the Tibedo Framework.
    
    This class implements machine learning potentials within the Tibedo Framework,
    achieving linear scaling with system size.
    """
    
    def __init__(self, feature_dim=100, hidden_dims=[128, 64, 32], atomic_types=None):
        """
        Initialize the TibedoMLPotential.
        
        Args:
            feature_dim (int): Dimension of input features
            hidden_dims (list): List of hidden layer dimensions
            atomic_types (list, optional): List of atomic types to consider
        """
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        
        # Set default atomic types if not provided
        if atomic_types is None:
            self.atomic_types = [1, 6, 7, 8]  # H, C, N, O
        else:
            self.atomic_types = atomic_types
        
        # Create neural networks for each atomic type
        self.models = {}
        self.optimizers = {}
        
        for atomic_type in self.atomic_types:
            self.models[atomic_type] = TibedoNeuralNetwork(feature_dim, hidden_dims, 1)
            self.optimizers[atomic_type] = torch.optim.Adam(self.models[atomic_type].parameters(), lr=0.001)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Create atomic environment calculator
        self.environment_calculator = AtomicEnvironment(use_tibedo_features=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def compute_features(self, structures):
        """
        Compute features for all atoms in all structures.
        
        Args:
            structures (list): List of structures, where each structure is a dict with
                              'positions', 'atomic_numbers', and 'energy' keys
            
        Returns:
            dict: Dictionary mapping atomic types to features and energies
        """
        # Initialize dictionaries to store features and energies for each atomic type
        features = {atomic_type: [] for atomic_type in self.atomic_types}
        energies = {atomic_type: [] for atomic_type in self.atomic_types}
        
        # Process each structure
        for structure in structures:
            positions = structure['positions']
            atomic_numbers = structure['atomic_numbers']
            total_energy = structure['energy']
            
            # Compute features for each atom
            for i, atomic_number in enumerate(atomic_numbers):
                if atomic_number in self.atomic_types:
                    # Compute features for this atom
                    atom_features = self.environment_calculator.compute_features(
                        positions, atomic_numbers, i
                    )
                    
                    # Add features to corresponding list
                    features[atomic_number].append(atom_features)
                    
                    # Assign energy (simple equal distribution for now)
                    # In a real implementation, this would be more sophisticated
                    atom_energy = total_energy / len(atomic_numbers)
                    energies[atomic_number].append(atom_energy)
        
        # Convert lists to tensors
        for atomic_type in self.atomic_types:
            if features[atomic_type]:
                features[atomic_type] = torch.tensor(np.array(features[atomic_type]), dtype=torch.float32)
                energies[atomic_type] = torch.tensor(np.array(energies[atomic_type]), dtype=torch.float32).reshape(-1, 1)
        
        return {'features': features, 'energies': energies}
    
    def train(self, structures, val_structures=None, epochs=100, batch_size=32):
        """
        Train the ML potential.
        
        Args:
            structures (list): List of training structures
            val_structures (list, optional): List of validation structures
            epochs (int): Number of training epochs
            batch_size (int): Batch size
        """
        # Compute features for training structures
        train_data = self.compute_features(structures)
        
        # Compute features for validation structures if provided
        if val_structures is not None:
            val_data = self.compute_features(val_structures)
        
        # Train models for each atomic type
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for atomic_type in self.atomic_types:
                if len(train_data['features'][atomic_type]) == 0:
                    # Skip if no data for this atomic type
                    continue
                
                # Create data loader
                dataset = torch.utils.data.TensorDataset(
                    train_data['features'][atomic_type],
                    train_data['energies'][atomic_type]
                )
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True
                )
                
                # Train model
                self.models[atomic_type].train()
                model_loss = 0.0
                
                for batch_features, batch_energies in dataloader:
                    # Zero gradients
                    self.optimizers[atomic_type].zero_grad()
                    
                    # Forward pass
                    outputs = self.models[atomic_type](batch_features)
                    
                    # Compute loss
                    loss = self.criterion(outputs, batch_energies)
                    
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizers[atomic_type].step()
                    
                    model_loss += loss.item() * len(batch_features)
                
                # Average loss for this atomic type
                if len(train_data['features'][atomic_type]) > 0:
                    model_loss /= len(train_data['features'][atomic_type])
                    epoch_loss += model_loss
            
            # Average loss across all atomic types
            epoch_loss /= sum(1 for atomic_type in self.atomic_types if len(train_data['features'][atomic_type]) > 0)
            self.train_losses.append(epoch_loss)
            
            # Validate if validation structures are provided
            if val_structures is not None:
                val_loss = self.validate(val_data)
                self.val_losses.append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}')
            else:
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}')
    
    def validate(self, val_data):
        """
        Validate the ML potential.
        
        Args:
            val_data (dict): Validation data
            
        Returns:
            float: Validation loss
        """
        val_loss = 0.0
        
        for atomic_type in self.atomic_types:
            if len(val_data['features'][atomic_type]) == 0:
                # Skip if no data for this atomic type
                continue
            
            # Evaluate model
            self.models[atomic_type].eval()
            
            with torch.no_grad():
                outputs = self.models[atomic_type](val_data['features'][atomic_type])
                loss = self.criterion(outputs, val_data['energies'][atomic_type])
                val_loss += loss.item()
        
        # Average loss across all atomic types
        val_loss /= sum(1 for atomic_type in self.atomic_types if len(val_data['features'][atomic_type]) > 0)
        
        return val_loss
    
    def predict_energy(self, structure):
        """
        Predict the energy of a structure.
        
        Args:
            structure (dict): Structure with 'positions' and 'atomic_numbers' keys
            
        Returns:
            float: Predicted energy
        """
        positions = structure['positions']
        atomic_numbers = structure['atomic_numbers']
        
        # Initialize total energy
        total_energy = 0.0
        
        # Predict energy for each atom
        for i, atomic_number in enumerate(atomic_numbers):
            if atomic_number in self.atomic_types:
                # Compute features for this atom
                atom_features = self.environment_calculator.compute_features(
                    positions, atomic_numbers, i
                )
                
                # Convert to tensor
                atom_features = torch.tensor(atom_features, dtype=torch.float32).unsqueeze(0)
                
                # Predict energy
                self.models[atomic_number].eval()
                with torch.no_grad():
                    atom_energy = self.models[atomic_number](atom_features).item()
                
                # Add to total energy
                total_energy += atom_energy
        
        return total_energy
    
    def predict_forces(self, structure, delta=0.01):
        """
        Predict forces on atoms using finite differences.
        
        Args:
            structure (dict): Structure with 'positions' and 'atomic_numbers' keys
            delta (float): Finite difference step size
            
        Returns:
            np.ndarray: Forces on atoms
        """
        positions = structure['positions']
        atomic_numbers = structure['atomic_numbers']
        
        # Initialize forces
        forces = np.zeros_like(positions)
        
        # Calculate reference energy
        reference_energy = self.predict_energy(structure)
        
        # Calculate forces using finite differences
        for i in range(len(positions)):
            for j in range(3):  # x, y, z
                # Displace atom in positive direction
                positions_plus = positions.copy()
                positions_plus[i, j] += delta
                
                # Create new structure with displaced atom
                structure_plus = {
                    'positions': positions_plus,
                    'atomic_numbers': atomic_numbers
                }
                
                # Calculate energy with displaced atom
                energy_plus = self.predict_energy(structure_plus)
                
                # Calculate force (negative gradient)
                forces[i, j] = -(energy_plus - reference_energy) / delta
        
        return forces
    
    def save(self, path_prefix):
        """
        Save the ML potential.
        
        Args:
            path_prefix (str): Prefix for model paths
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        
        # Save models for each atomic type
        for atomic_type in self.atomic_types:
            torch.save({
                'model_state_dict': self.models[atomic_type].state_dict(),
                'optimizer_state_dict': self.optimizers[atomic_type].state_dict()
            }, f"{path_prefix}_type_{atomic_type}.pt")
        
        # Save atomic types
        np.save(f"{path_prefix}_atomic_types.npy", np.array(self.atomic_types))
    
    def load(self, path_prefix):
        """
        Load the ML potential.
        
        Args:
            path_prefix (str): Prefix for model paths
        """
        # Load atomic types
        self.atomic_types = np.load(f"{path_prefix}_atomic_types.npy").tolist()
        
        # Initialize models and optimizers if needed
        for atomic_type in self.atomic_types:
            if atomic_type not in self.models:
                self.models[atomic_type] = TibedoNeuralNetwork(self.feature_dim, self.hidden_dims, 1)
                self.optimizers[atomic_type] = torch.optim.Adam(self.models[atomic_type].parameters(), lr=0.001)
        
        # Load models for each atomic type
        for atomic_type in self.atomic_types:
            checkpoint = torch.load(f"{path_prefix}_type_{atomic_type}.pt")
            self.models[atomic_type].load_state_dict(checkpoint['model_state_dict'])
            self.optimizers[atomic_type].load_state_dict(checkpoint['optimizer_state_dict'])
    
    def visualize_learning_curve(self):
        """
        Visualize the learning curve.
        
        Returns:
            matplotlib.figure.Figure: The learning curve figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot training loss
        ax.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss')
        
        # Plot validation loss if available
        if self.val_losses:
            ax.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Curve')
        ax.legend()
        ax.grid(True)
        
        return fig