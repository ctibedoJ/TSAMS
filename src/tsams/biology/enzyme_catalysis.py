"""
Enzyme Catalysis Mechanisms

This module implements tools for studying enzyme catalysis mechanisms
using the Tibedo Framework.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import time

# Import Tibedo components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tibedo.core.spinor.reduction_chain import ReductionChain
from tibedo.core.prime_indexed.prime_indexed_structure import PrimeIndexedStructure
from tibedo.core.advanced.quantum_state import ConfigurableQuantumState
from tibedo.ml.neural_networks.tibedo_neural_network import TibedoNeuralNetwork


class ReactionCoordinate:
    """
    Representation of a reaction coordinate.
    
    This class provides methods for defining and manipulating reaction coordinates
    for enzyme catalysis simulations.
    """
    
    def __init__(self, name, description, min_value=0.0, max_value=1.0):
        """
        Initialize the ReactionCoordinate.
        
        Args:
            name (str): Name of the reaction coordinate
            description (str): Description of the reaction coordinate
            min_value (float): Minimum value of the reaction coordinate
            max_value (float): Maximum value of the reaction coordinate
        """
        self.name = name
        self.description = description
        self.min_value = min_value
        self.max_value = max_value
        
        # Initialize value
        self.value = min_value
    
    def set_value(self, value):
        """
        Set the value of the reaction coordinate.
        
        Args:
            value (float): Value to set
            
        Returns:
            float: Clamped value
        """
        # Clamp value to range
        self.value = max(self.min_value, min(self.max_value, value))
        
        return self.value
    
    def get_normalized_value(self):
        """
        Get the normalized value of the reaction coordinate.
        
        Returns:
            float: Normalized value in [0, 1]
        """
        return (self.value - self.min_value) / (self.max_value - self.min_value)


class EnzymeCatalysisSimulator:
    """
    Simulator for enzyme catalysis mechanisms.
    
    This class implements tools for simulating enzyme catalysis mechanisms
    using the Tibedo Framework.
    """
    
    def __init__(self):
        """
        Initialize the EnzymeCatalysisSimulator.
        """
        # Create spinor reduction chain
        self.reduction_chain = ReductionChain(
            initial_dimension=16,
            chain_length=5
        )
        
        # Create prime-indexed structure
        self.prime_structure = PrimeIndexedStructure(max_index=100)
        
        # Create quantum state
        self.quantum_state = ConfigurableQuantumState(dimension=7)
        
        # Initialize reaction coordinates
        self.reaction_coordinates = []
        
        # Initialize energy profile
        self.energy_profile = None
    
    def add_reaction_coordinate(self, name, description, min_value=0.0, max_value=1.0):
        """
        Add a reaction coordinate.
        
        Args:
            name (str): Name of the reaction coordinate
            description (str): Description of the reaction coordinate
            min_value (float): Minimum value of the reaction coordinate
            max_value (float): Maximum value of the reaction coordinate
            
        Returns:
            ReactionCoordinate: The created reaction coordinate
        """
        coordinate = ReactionCoordinate(name, description, min_value, max_value)
        self.reaction_coordinates.append(coordinate)
        
        return coordinate
    
    def set_reaction_coordinates(self, values):
        """
        Set the values of all reaction coordinates.
        
        Args:
            values (list): Values to set
            
        Returns:
            list: Clamped values
        """
        if len(values) != len(self.reaction_coordinates):
            raise ValueError("Number of values must match number of reaction coordinates")
        
        # Set values
        clamped_values = []
        for i, value in enumerate(values):
            clamped_value = self.reaction_coordinates[i].set_value(value)
            clamped_values.append(clamped_value)
        
        return clamped_values
    
    def get_reaction_coordinate_values(self):
        """
        Get the values of all reaction coordinates.
        
        Returns:
            list: Values of all reaction coordinates
        """
        return [coord.value for coord in self.reaction_coordinates]
    
    def get_normalized_reaction_coordinate_values(self):
        """
        Get the normalized values of all reaction coordinates.
        
        Returns:
            list: Normalized values of all reaction coordinates
        """
        return [coord.get_normalized_value() for coord in self.reaction_coordinates]
    
    def calculate_energy(self, coordinates=None):
        """
        Calculate the energy for a given set of reaction coordinates.
        
        Args:
            coordinates (list, optional): Reaction coordinate values
            
        Returns:
            float: Energy
        """
        if coordinates is not None:
            self.set_reaction_coordinates(coordinates)
        
        # Get normalized coordinates
        norm_coords = self.get_normalized_reaction_coordinate_values()
        
        # Calculate energy using a simple model
        # This is a placeholder and should be replaced with a real energy function
        energy = 0.0
        
        # Add contribution from each coordinate
        for i, coord in enumerate(norm_coords):
            # Simple double-well potential
            energy += 10 * (coord - 0.3)**2 * (coord - 0.7)**2
        
        # Add coupling terms
        for i in range(len(norm_coords) - 1):
            energy += 2 * norm_coords[i] * norm_coords[i+1]
        
        return energy
    
    def calculate_energy_profile(self, n_points=100):
        """
        Calculate the energy profile along a reaction path.
        
        Args:
            n_points (int): Number of points along the path
            
        Returns:
            tuple: (coordinates, energies)
        """
        if len(self.reaction_coordinates) == 0:
            raise ValueError("No reaction coordinates defined")
        
        # Create reaction path
        if len(self.reaction_coordinates) == 1:
            # 1D path
            coord_values = np.linspace(
                self.reaction_coordinates[0].min_value,
                self.reaction_coordinates[0].max_value,
                n_points
            )
            
            # Calculate energies
            energies = []
            for value in coord_values:
                energies.append(self.calculate_energy([value]))
            
            # Store energy profile
            self.energy_profile = {
                'coordinates': coord_values.reshape(-1, 1),
                'energies': np.array(energies)
            }
            
            return coord_values.reshape(-1, 1), np.array(energies)
            
        elif len(self.reaction_coordinates) == 2:
            # 2D path
            coord1_values = np.linspace(
                self.reaction_coordinates[0].min_value,
                self.reaction_coordinates[0].max_value,
                n_points
            )
            
            coord2_values = np.linspace(
                self.reaction_coordinates[1].min_value,
                self.reaction_coordinates[1].max_value,
                n_points
            )
            
            # Create grid
            coord1_grid, coord2_grid = np.meshgrid(coord1_values, coord2_values)
            
            # Calculate energies
            energies = np.zeros((n_points, n_points))
            
            for i in range(n_points):
                for j in range(n_points):
                    energies[i, j] = self.calculate_energy([coord1_grid[i, j], coord2_grid[i, j]])
            
            # Store energy profile
            self.energy_profile = {
                'coord1_grid': coord1_grid,
                'coord2_grid': coord2_grid,
                'energies': energies
            }
            
            return (coord1_grid, coord2_grid), energies
            
        else:
            # Higher-dimensional path
            # For simplicity, we'll create a linear path from min to max values
            
            # Create path
            path = np.zeros((n_points, len(self.reaction_coordinates)))
            
            for i in range(n_points):
                t = i / (n_points - 1)
                
                for j, coord in enumerate(self.reaction_coordinates):
                    path[i, j] = coord.min_value + t * (coord.max_value - coord.min_value)
            
            # Calculate energies
            energies = []
            for i in range(n_points):
                energies.append(self.calculate_energy(path[i]))
            
            # Store energy profile
            self.energy_profile = {
                'coordinates': path,
                'energies': np.array(energies)
            }
            
            return path, np.array(energies)
    
    def find_transition_state(self, initial_guess=None):
        """
        Find the transition state along the reaction path.
        
        Args:
            initial_guess (list, optional): Initial guess for the transition state
            
        Returns:
            tuple: (transition_state_coordinates, transition_state_energy)
        """
        if len(self.reaction_coordinates) == 0:
            raise ValueError("No reaction coordinates defined")
        
        # Set initial guess
        if initial_guess is None:
            initial_guess = []
            for coord in self.reaction_coordinates:
                initial_guess.append((coord.min_value + coord.max_value) / 2)
        
        # Define objective function (negative energy for maximization)
        def objective(x):
            return -self.calculate_energy(x)
        
        # Define bounds
        bounds = [(coord.min_value, coord.max_value) for coord in self.reaction_coordinates]
        
        # Find transition state (maximum energy along path)
        result = minimize(
            objective,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # Get transition state coordinates and energy
        ts_coordinates = result.x
        ts_energy = self.calculate_energy(ts_coordinates)
        
        return ts_coordinates, ts_energy
    
    def calculate_activation_energy(self, reactant_coordinates, product_coordinates):
        """
        Calculate the activation energy for a reaction.
        
        Args:
            reactant_coordinates (list): Reaction coordinate values for reactant
            product_coordinates (list): Reaction coordinate values for product
            
        Returns:
            tuple: (forward_activation_energy, reverse_activation_energy)
        """
        # Calculate reactant and product energies
        reactant_energy = self.calculate_energy(reactant_coordinates)
        product_energy = self.calculate_energy(product_coordinates)
        
        # Find transition state
        ts_coordinates, ts_energy = self.find_transition_state()
        
        # Calculate activation energies
        forward_activation_energy = ts_energy - reactant_energy
        reverse_activation_energy = ts_energy - product_energy
        
        return forward_activation_energy, reverse_activation_energy
    
    def visualize_energy_profile(self):
        """
        Visualize the energy profile.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if self.energy_profile is None:
            raise ValueError("Energy profile not calculated yet")
        
        if len(self.reaction_coordinates) == 1:
            # 1D profile
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot energy profile
            ax.plot(self.energy_profile['coordinates'], self.energy_profile['energies'], 'k-', linewidth=2)
            
            # Set labels
            ax.set_xlabel(self.reaction_coordinates[0].name)
            ax.set_ylabel('Energy')
            ax.set_title('Energy Profile')
            ax.grid(True, alpha=0.3)
            
            return fig
            
        elif len(self.reaction_coordinates) == 2:
            # 2D profile
            fig = plt.figure(figsize=(12, 10))
            
            # Create 3D plot
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot energy surface
            surf = ax.plot_surface(
                self.energy_profile['coord1_grid'],
                self.energy_profile['coord2_grid'],
                self.energy_profile['energies'],
                cmap='viridis',
                alpha=0.8
            )
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            # Set labels
            ax.set_xlabel(self.reaction_coordinates[0].name)
            ax.set_ylabel(self.reaction_coordinates[1].name)
            ax.set_zlabel('Energy')
            ax.set_title('Energy Surface')
            
            return fig
            
        else:
            # Higher-dimensional profile
            # Plot energy along the linear path
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot energy profile
            ax.plot(range(len(self.energy_profile['energies'])), self.energy_profile['energies'], 'k-', linewidth=2)
            
            # Set labels
            ax.set_xlabel('Path Progress')
            ax.set_ylabel('Energy')
            ax.set_title('Energy Profile')
            ax.grid(True, alpha=0.3)
            
            return fig


class EnzymeCatalysisModel:
    """
    Model for enzyme catalysis mechanisms.
    
    This class implements a machine learning model for predicting enzyme catalysis
    mechanisms using the Tibedo Framework.
    """
    
    def __init__(self, input_dim=10, hidden_dims=[128, 64, 32], output_dim=1):
        """
        Initialize the EnzymeCatalysisModel.
        
        Args:
            input_dim (int): Input dimension
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Output dimension
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Create model
        self.model = TibedoNeuralNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_loader, val_loader=None, epochs=100):
        """
        Train the enzyme catalysis model.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader, optional): Validation data loader
            epochs (int): Number of training epochs
        """
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
            self.train_losses.append(epoch_loss)
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}')
            else:
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}')
    
    def validate(self, val_loader):
        """
        Validate the enzyme catalysis model.
        
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
    
    def predict(self, x):
        """
        Make predictions with the enzyme catalysis model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Predicted values
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(x)
    
    def save_model(self, path):
        """
        Save the enzyme catalysis model.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim
        }, path)
    
    def load_model(self, path):
        """
        Load the enzyme catalysis model.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path)
        
        # Update parameters
        self.input_dim = checkpoint['input_dim']
        self.hidden_dims = checkpoint['hidden_dims']
        self.output_dim = checkpoint['output_dim']
        
        # Recreate model
        self.model = TibedoNeuralNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim
        )
        
        # Load state dictionaries
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def visualize_learning_curve(self):
        """
        Visualize the learning curve.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
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


class EnzymeCatalysisAnalyzer:
    """
    Analyzer for enzyme catalysis mechanisms.
    
    This class provides tools for analyzing enzyme catalysis mechanisms
    using the Tibedo Framework.
    """
    
    def __init__(self):
        """
        Initialize the EnzymeCatalysisAnalyzer.
        """
        # Create simulator
        self.simulator = EnzymeCatalysisSimulator()
        
        # Create model
        self.model = None
        
        # Initialize results
        self.results = {}
    
    def setup_reaction_coordinates(self, coordinates):
        """
        Set up reaction coordinates.
        
        Args:
            coordinates (list): List of (name, description, min_value, max_value) tuples
            
        Returns:
            list: Created reaction coordinates
        """
        # Clear existing coordinates
        self.simulator.reaction_coordinates = []
        
        # Add coordinates
        created_coordinates = []
        for name, description, min_value, max_value in coordinates:
            coord = self.simulator.add_reaction_coordinate(name, description, min_value, max_value)
            created_coordinates.append(coord)
        
        return created_coordinates
    
    def analyze_reaction_path(self, n_points=100):
        """
        Analyze the reaction path.
        
        Args:
            n_points (int): Number of points along the path
            
        Returns:
            dict: Analysis results
        """
        # Calculate energy profile
        coordinates, energies = self.simulator.calculate_energy_profile(n_points)
        
        # Find transition state
        ts_coordinates, ts_energy = self.simulator.find_transition_state()
        
        # Calculate activation energy
        if len(self.simulator.reaction_coordinates) == 1:
            # 1D case
            reactant_coordinates = [self.simulator.reaction_coordinates[0].min_value]
            product_coordinates = [self.simulator.reaction_coordinates[0].max_value]
        else:
            # Multi-dimensional case
            reactant_coordinates = [coord.min_value for coord in self.simulator.reaction_coordinates]
            product_coordinates = [coord.max_value for coord in self.simulator.reaction_coordinates]
        
        forward_activation_energy, reverse_activation_energy = self.simulator.calculate_activation_energy(
            reactant_coordinates, product_coordinates
        )
        
        # Store results
        self.results = {
            'coordinates': coordinates,
            'energies': energies,
            'ts_coordinates': ts_coordinates,
            'ts_energy': ts_energy,
            'reactant_coordinates': reactant_coordinates,
            'product_coordinates': product_coordinates,
            'forward_activation_energy': forward_activation_energy,
            'reverse_activation_energy': reverse_activation_energy
        }
        
        return self.results
    
    def visualize_results(self):
        """
        Visualize the analysis results.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if not self.results:
            raise ValueError("No analysis results available")
        
        if len(self.simulator.reaction_coordinates) == 1:
            # 1D case
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot energy profile
            ax.plot(self.results['coordinates'], self.results['energies'], 'k-', linewidth=2)
            
            # Mark transition state
            ax.plot(self.results['ts_coordinates'][0], self.results['ts_energy'], 'ro', markersize=8)
            
            # Mark reactant and product
            reactant_energy = self.simulator.calculate_energy(self.results['reactant_coordinates'])
            product_energy = self.simulator.calculate_energy(self.results['product_coordinates'])
            
            ax.plot(self.results['reactant_coordinates'][0], reactant_energy, 'bo', markersize=8)
            ax.plot(self.results['product_coordinates'][0], product_energy, 'go', markersize=8)
            
            # Add annotations
            ax.annotate(
                f'TS\nE = {self.results["ts_energy"]:.2f}',
                (self.results['ts_coordinates'][0], self.results['ts_energy']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )
            
            ax.annotate(
                f'Reactant\nE = {reactant_energy:.2f}',
                (self.results['reactant_coordinates'][0], reactant_energy),
                textcoords="offset points",
                xytext=(0, -20),
                ha='center'
            )
            
            ax.annotate(
                f'Product\nE = {product_energy:.2f}',
                (self.results['product_coordinates'][0], product_energy),
                textcoords="offset points",
                xytext=(0, -20),
                ha='center'
            )
            
            # Add activation energy annotations
            ax.annotate(
                f'Forward Activation Energy = {self.results["forward_activation_energy"]:.2f}',
                (0.5, 0.95),
                xycoords='axes fraction',
                ha='center'
            )
            
            ax.annotate(
                f'Reverse Activation Energy = {self.results["reverse_activation_energy"]:.2f}',
                (0.5, 0.9),
                xycoords='axes fraction',
                ha='center'
            )
            
            # Set labels
            ax.set_xlabel(self.simulator.reaction_coordinates[0].name)
            ax.set_ylabel('Energy')
            ax.set_title('Reaction Energy Profile')
            ax.grid(True, alpha=0.3)
            
            return fig
            
        elif len(self.simulator.reaction_coordinates) == 2:
            # 2D case
            fig = plt.figure(figsize=(12, 10))
            
            # Create 3D plot
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot energy surface
            if isinstance(self.results['coordinates'], tuple):
                coord1_grid, coord2_grid = self.results['coordinates']
                surf = ax.plot_surface(
                    coord1_grid,
                    coord2_grid,
                    self.results['energies'],
                    cmap='viridis',
                    alpha=0.8
                )
            else:
                # Reshape coordinates and energies into a grid
                n_points = int(np.sqrt(len(self.results['coordinates'])))
                coord1_grid = self.results['coordinates'][:, 0].reshape(n_points, n_points)
                coord2_grid = self.results['coordinates'][:, 1].reshape(n_points, n_points)
                energies_grid = self.results['energies'].reshape(n_points, n_points)
                
                surf = ax.plot_surface(
                    coord1_grid,
                    coord2_grid,
                    energies_grid,
                    cmap='viridis',
                    alpha=0.8
                )
            
            # Mark transition state
            ax.scatter(
                self.results['ts_coordinates'][0],
                self.results['ts_coordinates'][1],
                self.results['ts_energy'],
                color='red',
                s=100,
                label='Transition State'
            )
            
            # Mark reactant and product
            reactant_energy = self.simulator.calculate_energy(self.results['reactant_coordinates'])
            product_energy = self.simulator.calculate_energy(self.results['product_coordinates'])
            
            ax.scatter(
                self.results['reactant_coordinates'][0],
                self.results['reactant_coordinates'][1],
                reactant_energy,
                color='blue',
                s=100,
                label='Reactant'
            )
            
            ax.scatter(
                self.results['product_coordinates'][0],
                self.results['product_coordinates'][1],
                product_energy,
                color='green',
                s=100,
                label='Product'
            )
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            # Set labels
            ax.set_xlabel(self.simulator.reaction_coordinates[0].name)
            ax.set_ylabel(self.simulator.reaction_coordinates[1].name)
            ax.set_zlabel('Energy')
            ax.set_title('Reaction Energy Surface')
            ax.legend()
            
            return fig
            
        else:
            # Higher-dimensional case
            # Plot energy along the linear path
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot energy profile
            ax.plot(range(len(self.results['energies'])), self.results['energies'], 'k-', linewidth=2)
            
            # Mark transition state
            # Find closest point on path to transition state
            path = self.results['coordinates']
            ts_coords = self.results['ts_coordinates']
            
            distances = np.sum((path - ts_coords)**2, axis=1)
            closest_idx = np.argmin(distances)
            
            ax.plot(closest_idx, self.results['energies'][closest_idx], 'ro', markersize=8)
            
            # Mark reactant and product
            ax.plot(0, self.results['energies'][0], 'bo', markersize=8)
            ax.plot(len(self.results['energies']) - 1, self.results['energies'][-1], 'go', markersize=8)
            
            # Add annotations
            ax.annotate(
                f'TS\nE = {self.results["ts_energy"]:.2f}',
                (closest_idx, self.results['energies'][closest_idx]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )
            
            ax.annotate(
                f'Reactant\nE = {self.results["energies"][0]:.2f}',
                (0, self.results['energies'][0]),
                textcoords="offset points",
                xytext=(0, -20),
                ha='center'
            )
            
            ax.annotate(
                f'Product\nE = {self.results["energies"][-1]:.2f}',
                (len(self.results['energies']) - 1, self.results['energies'][-1]),
                textcoords="offset points",
                xytext=(0, -20),
                ha='center'
            )
            
            # Add activation energy annotations
            ax.annotate(
                f'Forward Activation Energy = {self.results["forward_activation_energy"]:.2f}',
                (0.5, 0.95),
                xycoords='axes fraction',
                ha='center'
            )
            
            ax.annotate(
                f'Reverse Activation Energy = {self.results["reverse_activation_energy"]:.2f}',
                (0.5, 0.9),
                xycoords='axes fraction',
                ha='center'
            )
            
            # Set labels
            ax.set_xlabel('Path Progress')
            ax.set_ylabel('Energy')
            ax.set_title('Reaction Energy Profile')
            ax.grid(True, alpha=0.3)
            
            return fig
    
    def build_model(self, input_dim=10, hidden_dims=[128, 64, 32], output_dim=1):
        """
        Build a model for enzyme catalysis prediction.
        
        Args:
            input_dim (int): Input dimension
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Output dimension
            
        Returns:
            EnzymeCatalysisModel: The created model
        """
        self.model = EnzymeCatalysisModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
        return self.model
    
    def generate_training_data(self, n_samples=1000):
        """
        Generate training data for enzyme catalysis prediction.
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            tuple: (features, targets)
        """
        if len(self.simulator.reaction_coordinates) == 0:
            raise ValueError("No reaction coordinates defined")
        
        # Generate random coordinates
        features = np.zeros((n_samples, len(self.simulator.reaction_coordinates)))
        
        for i in range(n_samples):
            for j, coord in enumerate(self.simulator.reaction_coordinates):
                features[i, j] = np.random.uniform(coord.min_value, coord.max_value)
        
        # Calculate energies
        targets = np.zeros(n_samples)
        
        for i in range(n_samples):
            targets[i] = self.simulator.calculate_energy(features[i])
        
        return features, targets.reshape(-1, 1)
    
    def save_results(self, path):
        """
        Save analysis results.
        
        Args:
            path (str): Path to save the results
        """
        import pickle
        
        with open(path, 'wb') as f:
            pickle.dump(self.results, f)
    
    def load_results(self, path):
        """
        Load analysis results.
        
        Args:
            path (str): Path to load the results from
            
        Returns:
            dict: Analysis results
        """
        import pickle
        
        with open(path, 'rb') as f:
            self.results = pickle.load(f)
        
        return self.results