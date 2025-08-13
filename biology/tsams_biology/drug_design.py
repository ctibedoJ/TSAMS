"""
Drug Design with Accurate Binding Energy Calculations

This module implements drug design tools using the Tibedo Framework,
with accurate binding energy calculations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time
import random

# Import Tibedo components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tibedo.core.spinor.reduction_chain import ReductionChain
from tibedo.core.prime_indexed.prime_indexed_structure import PrimeIndexedStructure
from tibedo.core.advanced.quantum_state import ConfigurableQuantumState
from tibedo.ml.neural_networks.tibedo_neural_network import TibedoNeuralNetwork


class MoleculeRepresentation:
    """
    Representation of a molecule for drug design.
    
    This class provides methods for representing molecules in a format
    suitable for drug design using the Tibedo Framework.
    """
    
    def __init__(self):
        """
        Initialize the MoleculeRepresentation.
        """
        # Define atom properties
        self.atom_types = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']
        
        # Atomic radii (Å)
        self.atomic_radii = {
            'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'P': 1.8,
            'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98, 'H': 1.2
        }
        
        # Electronegativity (Pauling scale)
        self.electronegativity = {
            'C': 2.55, 'N': 3.04, 'O': 3.44, 'S': 2.58, 'P': 2.19,
            'F': 3.98, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66, 'H': 2.2
        }
        
        # Atomic mass
        self.atomic_mass = {
            'C': 12.01, 'N': 14.01, 'O': 16.0, 'S': 32.07, 'P': 30.97,
            'F': 19.0, 'Cl': 35.45, 'Br': 79.9, 'I': 126.9, 'H': 1.01
        }
        
        # Create atom type to index mapping
        self.atom_to_idx = {atom: i for i, atom in enumerate(self.atom_types)}
        
        # Create spinor reduction chain
        self.reduction_chain = ReductionChain(
            initial_dimension=16,
            chain_length=5
        )
        
        # Create prime-indexed structure
        self.prime_structure = PrimeIndexedStructure(max_index=100)
    
    def encode_molecule(self, atoms, positions, bonds=None):
        """
        Encode a molecule.
        
        Args:
            atoms (list): List of atom types
            positions (np.ndarray): Atomic positions (N, 3)
            bonds (list, optional): List of bonds as (i, j, type) tuples
            
        Returns:
            dict: Encoded molecule
        """
        n_atoms = len(atoms)
        
        # One-hot encode atom types
        atom_features = np.zeros((n_atoms, len(self.atom_types)))
        
        for i, atom in enumerate(atoms):
            if atom in self.atom_to_idx:
                atom_features[i, self.atom_to_idx[atom]] = 1
        
        # Add atomic properties
        atomic_properties = np.zeros((n_atoms, 3))
        
        for i, atom in enumerate(atoms):
            if atom in self.atomic_radii:
                atomic_properties[i, 0] = self.atomic_radii[atom]
                atomic_properties[i, 1] = self.electronegativity[atom]
                atomic_properties[i, 2] = self.atomic_mass[atom]
        
        # Calculate pairwise distances
        distances = cdist(positions, positions)
        
        # Create bond features if bonds are provided
        if bonds is not None:
            bond_features = np.zeros((n_atoms, n_atoms, 4))  # 4 bond types: none, single, double, triple
            
            # Initialize all to 'no bond'
            bond_features[:, :, 0] = 1
            
            for i, j, bond_type in bonds:
                # Set bond type (1: single, 2: double, 3: triple)
                bond_features[i, j, 0] = 0
                bond_features[i, j, bond_type] = 1
                
                # Bonds are symmetric
                bond_features[j, i, 0] = 0
                bond_features[j, i, bond_type] = 1
        else:
            # Estimate bonds based on distances
            bond_features = np.zeros((n_atoms, n_atoms, 4))
            
            # Initialize all to 'no bond'
            bond_features[:, :, 0] = 1
            
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    # Get atomic radii
                    r_i = self.atomic_radii.get(atoms[i], 1.5)
                    r_j = self.atomic_radii.get(atoms[j], 1.5)
                    
                    # Calculate bond threshold
                    threshold = (r_i + r_j) * 1.3  # 30% tolerance
                    
                    # Check if atoms are bonded
                    if distances[i, j] < threshold:
                        # Assume single bond
                        bond_features[i, j, 0] = 0
                        bond_features[i, j, 1] = 1
                        
                        # Bonds are symmetric
                        bond_features[j, i, 0] = 0
                        bond_features[j, i, 1] = 1
        
        # Apply spinor reduction to atomic features
        spinor_features = self._apply_spinor_reduction(atom_features, positions)
        
        # Apply prime-indexed structure to distances
        prime_features = self._apply_prime_indexed_structure(distances, atoms)
        
        # Combine features
        encoded_molecule = {
            'atoms': atoms,
            'positions': positions,
            'atom_features': atom_features,
            'atomic_properties': atomic_properties,
            'distances': distances,
            'bond_features': bond_features,
            'spinor_features': spinor_features,
            'prime_features': prime_features
        }
        
        return encoded_molecule
    
    def _apply_spinor_reduction(self, atom_features, positions):
        """
        Apply spinor reduction to atomic features.
        
        Args:
            atom_features (np.ndarray): Atomic features (N, F)
            positions (np.ndarray): Atomic positions (N, 3)
            
        Returns:
            dict: Spinor features
        """
        n_atoms = len(positions)
        
        # Create initial spinor space
        spinor_dim = self.reduction_chain.initial_dimension
        spinor_space = np.zeros((n_atoms, spinor_dim))
        
        # Fill spinor space with atomic features and positions
        for i in range(n_atoms):
            # Use atomic features
            feature_dim = min(len(atom_features[i]), spinor_dim - 3)
            spinor_space[i, :feature_dim] = atom_features[i, :feature_dim]
            
            # Use positions
            spinor_space[i, -3:] = positions[i]
        
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
        
        return spinor_features
    
    def _apply_prime_indexed_structure(self, distances, atoms):
        """
        Apply prime-indexed structure to distances.
        
        Args:
            distances (np.ndarray): Pairwise distances (N, N)
            atoms (list): List of atom types
            
        Returns:
            np.ndarray: Prime-indexed features
        """
        n_atoms = len(atoms)
        
        # Create prime-indexed features
        prime_features = np.zeros((n_atoms, self.prime_structure.max_index))
        
        for i in range(n_atoms):
            # Sort atoms by distance
            sorted_indices = np.argsort(distances[i])
            
            for j, p in enumerate(self.prime_structure.primes[:self.prime_structure.max_index]):
                if j + 1 < len(sorted_indices):
                    idx = sorted_indices[j + 1]  # Skip self
                    
                    # Get atom properties
                    atom = atoms[idx]
                    
                    if atom in self.electronegativity:
                        # Weight by electronegativity and distance
                        prime_features[i, j] = self.electronegativity[atom] * np.exp(-distances[i, idx] / p)
        
        return prime_features
    
    def calculate_molecular_descriptors(self, encoded_molecule):
        """
        Calculate molecular descriptors.
        
        Args:
            encoded_molecule (dict): Encoded molecule
            
        Returns:
            dict: Molecular descriptors
        """
        # Extract data
        atoms = encoded_molecule['atoms']
        positions = encoded_molecule['positions']
        distances = encoded_molecule['distances']
        bond_features = encoded_molecule['bond_features']
        
        # Calculate basic descriptors
        n_atoms = len(atoms)
        
        # Count atom types
        atom_counts = {}
        for atom in atoms:
            atom_counts[atom] = atom_counts.get(atom, 0) + 1
        
        # Calculate molecular weight
        molecular_weight = sum(self.atomic_mass.get(atom, 0) for atom in atoms)
        
        # Count bonds
        n_bonds = np.sum(bond_features[:, :, 1:]) / 2  # Divide by 2 because bonds are counted twice
        
        # Calculate bond types
        n_single_bonds = np.sum(bond_features[:, :, 1]) / 2
        n_double_bonds = np.sum(bond_features[:, :, 2]) / 2
        n_triple_bonds = np.sum(bond_features[:, :, 3]) / 2
        
        # Calculate topological descriptors
        # Adjacency matrix
        adjacency = np.sum(bond_features[:, :, 1:], axis=2) > 0
        
        # Degree (number of bonds per atom)
        degree = np.sum(adjacency, axis=1)
        
        # Calculate geometric descriptors
        # Center of mass
        com = np.zeros(3)
        total_mass = 0
        
        for i, atom in enumerate(atoms):
            mass = self.atomic_mass.get(atom, 0)
            com += mass * positions[i]
            total_mass += mass
        
        if total_mass > 0:
            com /= total_mass
        
        # Radius of gyration
        rg = 0
        if total_mass > 0:
            for i, atom in enumerate(atoms):
                mass = self.atomic_mass.get(atom, 0)
                rg += mass * np.sum((positions[i] - com)**2)
            
            rg = np.sqrt(rg / total_mass)
        
        # Calculate electronic descriptors
        # Total electronegativity
        total_en = sum(self.electronegativity.get(atom, 0) for atom in atoms)
        
        # Average electronegativity
        avg_en = total_en / n_atoms if n_atoms > 0 else 0
        
        # Combine descriptors
        descriptors = {
            'n_atoms': n_atoms,
            'atom_counts': atom_counts,
            'molecular_weight': molecular_weight,
            'n_bonds': n_bonds,
            'n_single_bonds': n_single_bonds,
            'n_double_bonds': n_double_bonds,
            'n_triple_bonds': n_triple_bonds,
            'degree': degree,
            'center_of_mass': com,
            'radius_of_gyration': rg,
            'total_electronegativity': total_en,
            'average_electronegativity': avg_en
        }
        
        return descriptors


class BindingAffinityCalculator:
    """
    Calculator for binding affinity between molecules.
    
    This class provides methods for calculating binding affinity between
    molecules using the Tibedo Framework.
    """
    
    def __init__(self):
        """
        Initialize the BindingAffinityCalculator.
        """
        # Create molecule representation
        self.molecule_representation = MoleculeRepresentation()
        
        # Create quantum state
        self.quantum_state = ConfigurableQuantumState(dimension=7)
        
        # Create model
        self.model = None
    
    def build_model(self, input_dim=256, hidden_dims=[128, 64, 32], output_dim=1):
        """
        Build a model for binding affinity prediction.
        
        Args:
            input_dim (int): Input dimension
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Output dimension
            
        Returns:
            TibedoNeuralNetwork: The created model
        """
        self.model = TibedoNeuralNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        return self.model
    
    def calculate_binding_features(self, ligand_molecule, target_molecule):
        """
        Calculate features for binding affinity prediction.
        
        Args:
            ligand_molecule (dict): Encoded ligand molecule
            target_molecule (dict): Encoded target molecule
            
        Returns:
            np.ndarray: Binding features
        """
        # Extract data
        ligand_atoms = ligand_molecule['atoms']
        ligand_positions = ligand_molecule['positions']
        ligand_features = ligand_molecule['atom_features']
        
        target_atoms = target_molecule['atoms']
        target_positions = target_molecule['positions']
        target_features = target_molecule['atom_features']
        
        # Calculate intermolecular distances
        intermolecular_distances = cdist(ligand_positions, target_positions)
        
        # Calculate basic binding features
        min_distance = np.min(intermolecular_distances)
        mean_distance = np.mean(intermolecular_distances)
        
        # Calculate contact features
        contact_threshold = 4.0  # Å
        n_contacts = np.sum(intermolecular_distances < contact_threshold)
        
        # Calculate interaction features
        interaction_features = []
        
        # For each ligand atom, find the closest target atom
        for i in range(len(ligand_atoms)):
            closest_idx = np.argmin(intermolecular_distances[i])
            
            # Get atom types
            ligand_atom = ligand_atoms[i]
            target_atom = target_atoms[closest_idx]
            
            # Get distance
            distance = intermolecular_distances[i, closest_idx]
            
            # Calculate interaction energy (simple model)
            if ligand_atom in self.molecule_representation.electronegativity and \
               target_atom in self.molecule_representation.electronegativity:
                # Electronegativity difference
                en_diff = abs(
                    self.molecule_representation.electronegativity[ligand_atom] -
                    self.molecule_representation.electronegativity[target_atom]
                )
                
                # Simple interaction energy model
                energy = en_diff * np.exp(-distance / 2.0)
                
                interaction_features.append(energy)
        
        # Calculate quantum features
        quantum_features = self._calculate_quantum_features(
            ligand_molecule, target_molecule, intermolecular_distances
        )
        
        # Combine features
        binding_features = np.concatenate([
            np.array([min_distance, mean_distance, n_contacts]),
            np.array(interaction_features),
            quantum_features.flatten()
        ])
        
        return binding_features
    
    def _calculate_quantum_features(self, ligand_molecule, target_molecule, intermolecular_distances):
        """
        Calculate quantum features for binding affinity prediction.
        
        Args:
            ligand_molecule (dict): Encoded ligand molecule
            target_molecule (dict): Encoded target molecule
            intermolecular_distances (np.ndarray): Intermolecular distances
            
        Returns:
            np.ndarray: Quantum features
        """
        # Extract spinor features
        ligand_spinor = ligand_molecule['spinor_features']['reduced_space_1']
        target_spinor = target_molecule['spinor_features']['reduced_space_1']
        
        # Calculate quantum features
        quantum_features = np.zeros((7, 7))
        
        # Configure quantum state
        parameters = {
            'phase_factors': np.ones(7),
            'amplitude_factors': np.ones(7) / np.sqrt(7),
            'entanglement_pattern': 'binding',
            'cyclotomic_parameters': {'n': 7, 'k': 1},
            'symmetry_breaking': np.min(intermolecular_distances) / 10.0,
            'entropic_decline': 0.0
        }
        
        self.quantum_state.configure(parameters)
        
        # Calculate quantum features
        for i in range(min(7, len(ligand_spinor))):
            for j in range(min(7, len(target_spinor))):
                # Use quantum state to calculate interaction
                quantum_features[i, j] = self.quantum_state.calculate_interaction(
                    ligand_spinor[i],
                    target_spinor[j]
                )
        
        return quantum_features
    
    def calculate_binding_affinity(self, ligand_molecule, target_molecule):
        """
        Calculate binding affinity between two molecules.
        
        Args:
            ligand_molecule (dict): Encoded ligand molecule
            target_molecule (dict): Encoded target molecule
            
        Returns:
            float: Binding affinity
        """
        if self.model is None:
            raise ValueError("Model not built yet")
        
        # Calculate binding features
        binding_features = self.calculate_binding_features(ligand_molecule, target_molecule)
        
        # Convert to tensor
        binding_features = torch.tensor(binding_features, dtype=torch.float32).unsqueeze(0)
        
        # Predict binding affinity
        self.model.eval()
        with torch.no_grad():
            affinity = self.model(binding_features).item()
        
        return affinity
    
    def train(self, train_loader, val_loader=None, epochs=100):
        """
        Train the binding affinity model.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader, optional): Validation data loader
            epochs (int): Number of training epochs
        """
        if self.model is None:
            raise ValueError("Model not built yet")
        
        # Training history
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            epoch_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                
                # Compute loss
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(batch_x)
            
            # Average loss
            epoch_loss /= len(train_loader.dataset)
            train_losses.append(epoch_loss)
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                val_losses.append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}')
            else:
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}')
        
        return train_losses, val_losses
    
    def validate(self, val_loader):
        """
        Validate the binding affinity model.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            float: Validation loss
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                # Forward pass
                outputs = self.model(batch_x)
                
                # Compute loss
                loss = self.criterion(outputs, batch_y)
                
                val_loss += loss.item() * len(batch_x)
        
        # Average loss
        val_loss /= len(val_loader.dataset)
        
        return val_loss
    
    def save_model(self, path):
        """
        Save the binding affinity model.
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built yet")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path):
        """
        Load the binding affinity model.
        
        Args:
            path (str): Path to load the model from
        """
        if self.model is None:
            raise ValueError("Model not built yet")
        
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class DrugDesigner:
    """
    Designer for drug molecules.
    
    This class provides methods for designing drug molecules using the Tibedo Framework.
    """
    
    def __init__(self):
        """
        Initialize the DrugDesigner.
        """
        # Create molecule representation
        self.molecule_representation = MoleculeRepresentation()
        
        # Create binding affinity calculator
        self.binding_affinity_calculator = BindingAffinityCalculator()
        
        # Initialize molecule library
        self.molecule_library = []
        
        # Initialize target molecule
        self.target_molecule = None
    
    def add_molecule_to_library(self, atoms, positions, bonds=None):
        """
        Add a molecule to the library.
        
        Args:
            atoms (list): List of atom types
            positions (np.ndarray): Atomic positions (N, 3)
            bonds (list, optional): List of bonds as (i, j, type) tuples
            
        Returns:
            dict: Encoded molecule
        """
        # Encode molecule
        encoded_molecule = self.molecule_representation.encode_molecule(atoms, positions, bonds)
        
        # Add to library
        self.molecule_library.append(encoded_molecule)
        
        return encoded_molecule
    
    def set_target_molecule(self, atoms, positions, bonds=None):
        """
        Set the target molecule.
        
        Args:
            atoms (list): List of atom types
            positions (np.ndarray): Atomic positions (N, 3)
            bonds (list, optional): List of bonds as (i, j, type) tuples
            
        Returns:
            dict: Encoded target molecule
        """
        # Encode target molecule
        self.target_molecule = self.molecule_representation.encode_molecule(atoms, positions, bonds)
        
        return self.target_molecule
    
    def screen_library(self):
        """
        Screen the molecule library against the target.
        
        Returns:
            list: Sorted list of (molecule_idx, affinity) tuples
        """
        if self.target_molecule is None:
            raise ValueError("Target molecule not set")
        
        if not self.molecule_library:
            raise ValueError("Molecule library is empty")
        
        # Calculate binding affinity for each molecule
        affinities = []
        
        for i, molecule in enumerate(self.molecule_library):
            affinity = self.binding_affinity_calculator.calculate_binding_affinity(
                molecule, self.target_molecule
            )
            
            affinities.append((i, affinity))
        
        # Sort by affinity (higher is better)
        affinities.sort(key=lambda x: x[1], reverse=True)
        
        return affinities
    
    def optimize_molecule(self, molecule_idx, n_iterations=100):
        """
        Optimize a molecule for binding to the target.
        
        Args:
            molecule_idx (int): Index of the molecule to optimize
            n_iterations (int): Number of optimization iterations
            
        Returns:
            dict: Optimized molecule
        """
        if self.target_molecule is None:
            raise ValueError("Target molecule not set")
        
        if molecule_idx >= len(self.molecule_library):
            raise ValueError(f"Invalid molecule index: {molecule_idx}")
        
        # Get molecule
        molecule = self.molecule_library[molecule_idx]
        
        # Extract data
        atoms = molecule['atoms']
        positions = molecule['positions'].copy()
        
        # Define objective function (negative binding affinity for minimization)
        def objective(x):
            # Reshape flat array to positions
            pos = x.reshape(-1, 3)
            
            # Encode molecule
            encoded_molecule = self.molecule_representation.encode_molecule(atoms, pos)
            
            # Calculate binding affinity
            affinity = self.binding_affinity_calculator.calculate_binding_affinity(
                encoded_molecule, self.target_molecule
            )
            
            # Return negative affinity (for minimization)
            return -affinity
        
        # Optimize
        result = minimize(
            objective,
            positions.flatten(),
            method='L-BFGS-B',
            options={'maxiter': n_iterations}
        )
        
        # Get optimized positions
        optimized_positions = result.x.reshape(-1, 3)
        
        # Encode optimized molecule
        optimized_molecule = self.molecule_representation.encode_molecule(atoms, optimized_positions)
        
        # Calculate binding affinity
        affinity = self.binding_affinity_calculator.calculate_binding_affinity(
            optimized_molecule, self.target_molecule
        )
        
        # Add to library
        self.molecule_library.append(optimized_molecule)
        
        return optimized_molecule, affinity
    
    def generate_molecule(self, n_atoms=10, atom_types=None):
        """
        Generate a random molecule.
        
        Args:
            n_atoms (int): Number of atoms
            atom_types (list, optional): List of atom types to use
            
        Returns:
            dict: Generated molecule
        """
        if atom_types is None:
            atom_types = ['C', 'N', 'O', 'H']
        
        # Generate atoms
        atoms = [random.choice(atom_types) for _ in range(n_atoms)]
        
        # Generate positions
        positions = np.random.uniform(-5, 5, (n_atoms, 3))
        
        # Encode molecule
        molecule = self.molecule_representation.encode_molecule(atoms, positions)
        
        # Add to library
        self.molecule_library.append(molecule)
        
        return molecule
    
    def mutate_molecule(self, molecule_idx, mutation_rate=0.1):
        """
        Mutate a molecule.
        
        Args:
            molecule_idx (int): Index of the molecule to mutate
            mutation_rate (float): Mutation rate
            
        Returns:
            dict: Mutated molecule
        """
        if molecule_idx >= len(self.molecule_library):
            raise ValueError(f"Invalid molecule index: {molecule_idx}")
        
        # Get molecule
        molecule = self.molecule_library[molecule_idx]
        
        # Extract data
        atoms = molecule['atoms'].copy()
        positions = molecule['positions'].copy()
        
        # Mutate atoms
        for i in range(len(atoms)):
            if random.random() < mutation_rate:
                # Change atom type
                atoms[i] = random.choice(self.molecule_representation.atom_types)
        
        # Mutate positions
        for i in range(len(positions)):
            if random.random() < mutation_rate:
                # Perturb position
                positions[i] += np.random.normal(0, 1, 3)
        
        # Encode mutated molecule
        mutated_molecule = self.molecule_representation.encode_molecule(atoms, positions)
        
        # Add to library
        self.molecule_library.append(mutated_molecule)
        
        return mutated_molecule
    
    def crossover_molecules(self, molecule_idx1, molecule_idx2):
        """
        Perform crossover between two molecules.
        
        Args:
            molecule_idx1 (int): Index of the first molecule
            molecule_idx2 (int): Index of the second molecule
            
        Returns:
            dict: Crossover molecule
        """
        if molecule_idx1 >= len(self.molecule_library) or molecule_idx2 >= len(self.molecule_library):
            raise ValueError(f"Invalid molecule indices: {molecule_idx1}, {molecule_idx2}")
        
        # Get molecules
        molecule1 = self.molecule_library[molecule_idx1]
        molecule2 = self.molecule_library[molecule_idx2]
        
        # Extract data
        atoms1 = molecule1['atoms']
        positions1 = molecule1['positions']
        
        atoms2 = molecule2['atoms']
        positions2 = molecule2['positions']
        
        # Determine crossover point
        crossover_point = random.randint(1, min(len(atoms1), len(atoms2)) - 1)
        
        # Create crossover molecule
        atoms = atoms1[:crossover_point] + atoms2[crossover_point:]
        positions = np.vstack([positions1[:crossover_point], positions2[crossover_point:]])
        
        # Encode crossover molecule
        crossover_molecule = self.molecule_representation.encode_molecule(atoms, positions)
        
        # Add to library
        self.molecule_library.append(crossover_molecule)
        
        return crossover_molecule
    
    def evolve_molecules(self, n_generations=10, population_size=10, mutation_rate=0.1):
        """
        Evolve molecules for binding to the target.
        
        Args:
            n_generations (int): Number of generations
            population_size (int): Population size
            mutation_rate (float): Mutation rate
            
        Returns:
            dict: Best molecule
        """
        if self.target_molecule is None:
            raise ValueError("Target molecule not set")
        
        # Initialize population
        if len(self.molecule_library) < population_size:
            # Generate random molecules
            for _ in range(population_size - len(self.molecule_library)):
                self.generate_molecule()
        
        # Evolution loop
        for generation in range(n_generations):
            # Screen library
            affinities = self.screen_library()
            
            # Select top molecules
            top_indices = [idx for idx, _ in affinities[:population_size//2]]
            
            # Create new generation
            new_molecules = []
            
            # Elitism: keep top molecule
            new_molecules.append(self.molecule_library[affinities[0][0]])
            
            # Crossover and mutation
            while len(new_molecules) < population_size:
                # Select parents
                parent1_idx = random.choice(top_indices)
                parent2_idx = random.choice(top_indices)
                
                # Crossover
                child = self.crossover_molecules(parent1_idx, parent2_idx)
                
                # Mutation
                if random.random() < mutation_rate:
                    child = self.mutate_molecule(len(self.molecule_library) - 1)
                
                new_molecules.append(child)
            
            # Replace population
            self.molecule_library = new_molecules
            
            # Print progress
            best_affinity = affinities[0][1]
            print(f"Generation {generation+1}/{n_generations}, Best Affinity: {best_affinity:.6f}")
        
        # Final screening
        affinities = self.screen_library()
        best_idx = affinities[0][0]
        
        return self.molecule_library[best_idx], affinities[0][1]
    
    def visualize_molecule(self, molecule_idx):
        """
        Visualize a molecule.
        
        Args:
            molecule_idx (int): Index of the molecule to visualize
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if molecule_idx >= len(self.molecule_library):
            raise ValueError(f"Invalid molecule index: {molecule_idx}")
        
        # Get molecule
        molecule = self.molecule_library[molecule_idx]
        
        # Extract data
        atoms = molecule['atoms']
        positions = molecule['positions']
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define colors for different atom types
        colors = {
            'C': 'gray',
            'N': 'blue',
            'O': 'red',
            'S': 'yellow',
            'P': 'orange',
            'F': 'green',
            'Cl': 'green',
            'Br': 'brown',
            'I': 'purple',
            'H': 'white'
        }
        
        # Define sizes for different atom types
        sizes = {
            'C': 100,
            'N': 100,
            'O': 100,
            'S': 150,
            'P': 150,
            'F': 80,
            'Cl': 120,
            'Br': 140,
            'I': 160,
            'H': 50
        }
        
        # Plot atoms
        for i, (atom, pos) in enumerate(zip(atoms, positions)):
            color = colors.get(atom, 'gray')
            size = sizes.get(atom, 100)
            
            ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, edgecolors='black')
            
            # Add atom label
            ax.text(pos[0], pos[1], pos[2], atom, fontsize=8)
        
        # Plot bonds
        bond_features = molecule['bond_features']
        
        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                # Check if atoms are bonded
                if np.any(bond_features[i, j, 1:] > 0):
                    # Plot bond
                    ax.plot(
                        [positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        [positions[i, 2], positions[j, 2]],
                        'k-', alpha=0.7
                    )
        
        # Set labels
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'Molecule {molecule_idx}')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        return fig
    
    def visualize_binding(self, molecule_idx):
        """
        Visualize binding between a molecule and the target.
        
        Args:
            molecule_idx (int): Index of the molecule to visualize
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if self.target_molecule is None:
            raise ValueError("Target molecule not set")
        
        if molecule_idx >= len(self.molecule_library):
            raise ValueError(f"Invalid molecule index: {molecule_idx}")
        
        # Get molecule
        molecule = self.molecule_library[molecule_idx]
        
        # Extract data
        ligand_atoms = molecule['atoms']
        ligand_positions = molecule['positions']
        
        target_atoms = self.target_molecule['atoms']
        target_positions = self.target_molecule['positions']
        
        # Calculate intermolecular distances
        intermolecular_distances = cdist(ligand_positions, target_positions)
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define colors for different atom types
        colors = {
            'C': 'gray',
            'N': 'blue',
            'O': 'red',
            'S': 'yellow',
            'P': 'orange',
            'F': 'green',
            'Cl': 'green',
            'Br': 'brown',
            'I': 'purple',
            'H': 'white'
        }
        
        # Define sizes for different atom types
        sizes = {
            'C': 100,
            'N': 100,
            'O': 100,
            'S': 150,
            'P': 150,
            'F': 80,
            'Cl': 120,
            'Br': 140,
            'I': 160,
            'H': 50
        }
        
        # Plot ligand atoms
        for i, (atom, pos) in enumerate(zip(ligand_atoms, ligand_positions)):
            color = colors.get(atom, 'gray')
            size = sizes.get(atom, 100)
            
            ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, edgecolors='black')
        
        # Plot target atoms
        for i, (atom, pos) in enumerate(zip(target_atoms, target_positions)):
            color = colors.get(atom, 'gray')
            size = sizes.get(atom, 100)
            
            ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, edgecolors='black', alpha=0.5)
        
        # Plot ligand bonds
        ligand_bond_features = molecule['bond_features']
        
        for i in range(len(ligand_atoms)):
            for j in range(i+1, len(ligand_atoms)):
                # Check if atoms are bonded
                if np.any(ligand_bond_features[i, j, 1:] > 0):
                    # Plot bond
                    ax.plot(
                        [ligand_positions[i, 0], ligand_positions[j, 0]],
                        [ligand_positions[i, 1], ligand_positions[j, 1]],
                        [ligand_positions[i, 2], ligand_positions[j, 2]],
                        'k-', alpha=0.7
                    )
        
        # Plot target bonds
        target_bond_features = self.target_molecule['bond_features']
        
        for i in range(len(target_atoms)):
            for j in range(i+1, len(target_atoms)):
                # Check if atoms are bonded
                if np.any(target_bond_features[i, j, 1:] > 0):
                    # Plot bond
                    ax.plot(
                        [target_positions[i, 0], target_positions[j, 0]],
                        [target_positions[i, 1], target_positions[j, 1]],
                        [target_positions[i, 2], target_positions[j, 2]],
                        'k-', alpha=0.3
                    )
        
        # Plot intermolecular interactions
        contact_threshold = 4.0  # Å
        
        for i in range(len(ligand_atoms)):
            for j in range(len(target_atoms)):
                # Check if atoms are in contact
                if intermolecular_distances[i, j] < contact_threshold:
                    # Plot interaction
                    ax.plot(
                        [ligand_positions[i, 0], target_positions[j, 0]],
                        [ligand_positions[i, 1], target_positions[j, 1]],
                        [ligand_positions[i, 2], target_positions[j, 2]],
                        'r--', alpha=0.5
                    )
        
        # Calculate binding affinity
        affinity = self.binding_affinity_calculator.calculate_binding_affinity(
            molecule, self.target_molecule
        )
        
        # Set labels
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'Binding: Molecule {molecule_idx} to Target (Affinity: {affinity:.6f})')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        return fig