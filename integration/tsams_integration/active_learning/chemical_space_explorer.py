"""
Chemical Space Explorer with Active Learning

This module implements tools for exploring chemical space efficiently using
active learning strategies within the Tibedo Framework.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
from typing import List, Dict, Tuple, Union, Optional, Callable

# Import Tibedo components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tibedo.core.spinor.reduction_chain import ReductionChain
from tibedo.core.prime_indexed.prime_indexed_structure import PrimeIndexedStructure
from tibedo.ml.neural_networks.tibedo_neural_network import TibedoNeuralNetwork
from tibedo.ml.potentials.tibedo_potential import TibedoMLPotential


class ChemicalSpaceDataset(Dataset):
    """
    Dataset for chemical space exploration.
    
    This class provides a PyTorch dataset for chemical space exploration,
    mapping molecular representations to properties.
    """
    
    def __init__(self, features: np.ndarray, properties: np.ndarray, 
                 uncertainties: Optional[np.ndarray] = None):
        """
        Initialize the ChemicalSpaceDataset.
        
        Args:
            features (np.ndarray): Molecular features (N, feature_dim)
            properties (np.ndarray): Molecular properties (N, property_dim)
            uncertainties (np.ndarray, optional): Prediction uncertainties (N, property_dim)
        """
        self.features = features
        self.properties = properties
        
        if uncertainties is None:
            self.uncertainties = np.zeros_like(properties)
        else:
            self.uncertainties = uncertainties
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (features, properties, uncertainties)
        """
        return (torch.tensor(self.features[idx], dtype=torch.float32),
                torch.tensor(self.properties[idx], dtype=torch.float32),
                torch.tensor(self.uncertainties[idx], dtype=torch.float32))
    
    def add_samples(self, features: np.ndarray, properties: np.ndarray, 
                    uncertainties: Optional[np.ndarray] = None) -> None:
        """
        Add new samples to the dataset.
        
        Args:
            features (np.ndarray): Molecular features (N, feature_dim)
            properties (np.ndarray): Molecular properties (N, property_dim)
            uncertainties (np.ndarray, optional): Prediction uncertainties (N, property_dim)
        """
        self.features = np.vstack([self.features, features])
        self.properties = np.vstack([self.properties, properties])
        
        if uncertainties is None:
            new_uncertainties = np.zeros_like(properties)
        else:
            new_uncertainties = uncertainties
        
        self.uncertainties = np.vstack([self.uncertainties, new_uncertainties])


class AcquisitionFunction:
    """
    Acquisition function for active learning.
    
    This class provides acquisition functions for selecting the next points
    to evaluate in active learning.
    """
    
    @staticmethod
    def uncertainty_sampling(uncertainties: np.ndarray) -> np.ndarray:
        """
        Uncertainty sampling acquisition function.
        
        Args:
            uncertainties (np.ndarray): Prediction uncertainties (N, property_dim)
            
        Returns:
            np.ndarray: Acquisition scores (N,)
        """
        # For multi-dimensional properties, take the maximum uncertainty
        if uncertainties.ndim > 1 and uncertainties.shape[1] > 1:
            return np.max(uncertainties, axis=1)
        else:
            return uncertainties.flatten()
    
    @staticmethod
    def expected_improvement(mean: np.ndarray, std: np.ndarray, best_value: float, 
                             xi: float = 0.01) -> np.ndarray:
        """
        Expected improvement acquisition function.
        
        Args:
            mean (np.ndarray): Predicted mean values (N, property_dim)
            std (np.ndarray): Predicted standard deviations (N, property_dim)
            best_value (float): Current best value
            xi (float): Exploration parameter
            
        Returns:
            np.ndarray: Acquisition scores (N,)
        """
        # Handle multi-dimensional properties
        if mean.ndim > 1 and mean.shape[1] > 1:
            # Take the first property dimension for simplicity
            # Could be extended to handle multi-objective optimization
            mean = mean[:, 0]
            std = std[:, 0]
        else:
            mean = mean.flatten()
            std = std.flatten()
        
        # Avoid division by zero
        std = np.maximum(std, 1e-6)
        
        # Calculate improvement
        z = (mean - best_value - xi) / std
        
        # Calculate expected improvement
        ei = (mean - best_value - xi) * norm.cdf(z) + std * norm.pdf(z)
        
        # Set negative values to zero
        ei = np.maximum(ei, 0)
        
        return ei
    
    @staticmethod
    def upper_confidence_bound(mean: np.ndarray, std: np.ndarray, 
                               beta: float = 2.0) -> np.ndarray:
        """
        Upper confidence bound acquisition function.
        
        Args:
            mean (np.ndarray): Predicted mean values (N, property_dim)
            std (np.ndarray): Predicted standard deviations (N, property_dim)
            beta (float): Exploration parameter
            
        Returns:
            np.ndarray: Acquisition scores (N,)
        """
        # Handle multi-dimensional properties
        if mean.ndim > 1 and mean.shape[1] > 1:
            # Take the first property dimension for simplicity
            mean = mean[:, 0]
            std = std[:, 0]
        else:
            mean = mean.flatten()
            std = std.flatten()
        
        # Calculate upper confidence bound
        ucb = mean + beta * std
        
        return ucb
    
    @staticmethod
    def thompson_sampling(mean: np.ndarray, std: np.ndarray, 
                          n_samples: int = 10) -> np.ndarray:
        """
        Thompson sampling acquisition function.
        
        Args:
            mean (np.ndarray): Predicted mean values (N, property_dim)
            std (np.ndarray): Predicted standard deviations (N, property_dim)
            n_samples (int): Number of samples to draw
            
        Returns:
            np.ndarray: Acquisition scores (N,)
        """
        # Handle multi-dimensional properties
        if mean.ndim > 1 and mean.shape[1] > 1:
            # Take the first property dimension for simplicity
            mean = mean[:, 0]
            std = std[:, 0]
        else:
            mean = mean.flatten()
            std = std.flatten()
        
        # Draw samples from the posterior
        samples = np.random.normal(mean[:, np.newaxis], 
                                  std[:, np.newaxis], 
                                  size=(mean.shape[0], n_samples))
        
        # Take the maximum value across samples
        return np.mean(samples, axis=1)


class ChemicalSpaceExplorer:
    """
    Explorer for chemical space using active learning.
    
    This class implements tools for exploring chemical space efficiently using
    active learning strategies within the Tibedo Framework.
    """
    
    def __init__(self, feature_dim: int, property_dim: int, 
                 hidden_dims: List[int] = [128, 64, 32],
                 acquisition_function: str = 'uncertainty'):
        """
        Initialize the ChemicalSpaceExplorer.
        
        Args:
            feature_dim (int): Dimension of molecular features
            property_dim (int): Dimension of molecular properties
            hidden_dims (list): List of hidden layer dimensions
            acquisition_function (str): Acquisition function type
        """
        self.feature_dim = feature_dim
        self.property_dim = property_dim
        self.hidden_dims = hidden_dims
        
        # Create ML potential
        self.ml_potential = TibedoMLPotential(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            output_dim=property_dim * 2  # Mean and uncertainty
        )
        
        # Set acquisition function
        self.acquisition_function_type = acquisition_function
        
        # Initialize dataset
        self.dataset = None
        
        # Create spinor reduction chain for feature transformation
        self.reduction_chain = ReductionChain(
            initial_dimension=max(16, feature_dim),
            chain_length=5
        )
        
        # Create prime-indexed structure for feature transformation
        self.prime_structure = PrimeIndexedStructure(
            max_index=min(100, feature_dim)
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        # Exploration history
        self.exploration_history = {
            'iterations': [],
            'sampled_indices': [],
            'acquisition_scores': [],
            'best_properties': []
        }
    
    def initialize_dataset(self, features: np.ndarray, properties: np.ndarray, 
                          uncertainties: Optional[np.ndarray] = None) -> None:
        """
        Initialize the dataset with initial samples.
        
        Args:
            features (np.ndarray): Molecular features (N, feature_dim)
            properties (np.ndarray): Molecular properties (N, property_dim)
            uncertainties (np.ndarray, optional): Prediction uncertainties (N, property_dim)
        """
        self.dataset = ChemicalSpaceDataset(features, properties, uncertainties)
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        Transform molecular features using Tibedo structures.
        
        Args:
            features (np.ndarray): Molecular features (N, feature_dim)
            
        Returns:
            np.ndarray: Transformed features (N, transformed_dim)
        """
        # Apply spinor reduction if feature dimension is large enough
        if features.shape[1] >= 16:
            # Apply first reduction map
            reduced_features = self.reduction_chain.maps[0].apply(features)
            
            # Extract key statistics
            stats = np.array([
                np.mean(reduced_features, axis=1),
                np.std(reduced_features, axis=1),
                np.max(reduced_features, axis=1),
                np.min(reduced_features, axis=1)
            ]).T
            
            # Combine with original features
            transformed = np.hstack([features, stats])
        else:
            transformed = features.copy()
        
        # Apply prime-indexed transformation
        prime_features = np.zeros((features.shape[0], min(100, features.shape[1])))
        
        for i in range(min(features.shape[1], prime_features.shape[1])):
            prime_idx = i % features.shape[1]
            prime_features[:, i] = features[:, prime_idx]
        
        # Combine all features
        transformed = np.hstack([transformed, prime_features])
        
        return transformed
    
    def train_model(self, epochs: int = 100, batch_size: int = 32, 
                   validation_split: float = 0.2) -> None:
        """
        Train the ML potential on the current dataset.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Fraction of data to use for validation
        """
        if self.dataset is None:
            raise ValueError("Dataset not initialized")
        
        # Split dataset into training and validation
        n_samples = len(self.dataset)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [n_train, n_val]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Train model
        self.ml_potential.train()
        
        for epoch in range(epochs):
            # Train
            epoch_loss = 0.0
            
            for batch_x, batch_y, _ in train_loader:
                # Transform features
                batch_x_transformed = torch.tensor(
                    self.transform_features(batch_x.numpy()),
                    dtype=torch.float32
                )
                
                # Zero gradients
                self.ml_potential.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.ml_potential.model(batch_x_transformed)
                
                # Split outputs into mean and log variance
                mean, log_var = torch.split(outputs, self.property_dim, dim=1)
                
                # Calculate negative log likelihood loss
                var = torch.exp(log_var)
                loss = 0.5 * torch.mean(
                    log_var + (batch_y - mean)**2 / var
                )
                
                # Backward pass and optimize
                loss.backward()
                self.ml_potential.optimizer.step()
                
                epoch_loss += loss.item() * len(batch_x)
            
            # Average loss
            epoch_loss /= n_train
            self.train_losses.append(epoch_loss)
            
            # Validate
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y, _ in val_loader:
                    # Transform features
                    batch_x_transformed = torch.tensor(
                        self.transform_features(batch_x.numpy()),
                        dtype=torch.float32
                    )
                    
                    # Forward pass
                    outputs = self.ml_potential.model(batch_x_transformed)
                    
                    # Split outputs into mean and log variance
                    mean, log_var = torch.split(outputs, self.property_dim, dim=1)
                    
                    # Calculate negative log likelihood loss
                    var = torch.exp(log_var)
                    loss = 0.5 * torch.mean(
                        log_var + (batch_y - mean)**2 / var
                    )
                    
                    val_loss += loss.item() * len(batch_x)
            
            # Average loss
            val_loss /= n_val
            self.val_losses.append(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict properties and uncertainties for molecular features.
        
        Args:
            features (np.ndarray): Molecular features (N, feature_dim)
            
        Returns:
            tuple: (predicted_properties, predicted_uncertainties)
        """
        # Transform features
        transformed_features = self.transform_features(features)
        
        # Convert to tensor
        features_tensor = torch.tensor(transformed_features, dtype=torch.float32)
        
        # Predict
        self.ml_potential.model.eval()
        with torch.no_grad():
            outputs = self.ml_potential.model(features_tensor)
        
        # Split outputs into mean and log variance
        mean, log_var = torch.split(outputs, self.property_dim, dim=1)
        
        # Convert log variance to standard deviation
        std = torch.exp(0.5 * log_var)
        
        return mean.numpy(), std.numpy()
    
    def select_next_batch(self, candidate_features: np.ndarray, 
                          batch_size: int = 1) -> np.ndarray:
        """
        Select the next batch of points to evaluate.
        
        Args:
            candidate_features (np.ndarray): Candidate molecular features (N, feature_dim)
            batch_size (int): Number of points to select
            
        Returns:
            np.ndarray: Indices of selected points
        """
        # Predict properties and uncertainties
        mean, std = self.predict(candidate_features)
        
        # Calculate acquisition scores
        if self.acquisition_function_type == 'uncertainty':
            scores = AcquisitionFunction.uncertainty_sampling(std)
        elif self.acquisition_function_type == 'ei':
            # Get current best value
            best_value = np.min(self.dataset.properties[:, 0])
            scores = AcquisitionFunction.expected_improvement(mean, std, best_value)
        elif self.acquisition_function_type == 'ucb':
            scores = AcquisitionFunction.upper_confidence_bound(mean, std)
        elif self.acquisition_function_type == 'thompson':
            scores = AcquisitionFunction.thompson_sampling(mean, std)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function_type}")
        
        # Select top batch_size points
        selected_indices = np.argsort(scores)[-batch_size:]
        
        # Store acquisition scores
        self.exploration_history['acquisition_scores'].append(scores)
        
        return selected_indices
    
    def update_dataset(self, features: np.ndarray, properties: np.ndarray, 
                      uncertainties: Optional[np.ndarray] = None) -> None:
        """
        Update the dataset with new samples.
        
        Args:
            features (np.ndarray): Molecular features (N, feature_dim)
            properties (np.ndarray): Molecular properties (N, property_dim)
            uncertainties (np.ndarray, optional): Prediction uncertainties (N, property_dim)
        """
        self.dataset.add_samples(features, properties, uncertainties)
    
    def explore(self, candidate_features: np.ndarray, 
               property_evaluator: Callable[[np.ndarray], np.ndarray],
               n_iterations: int = 10, batch_size: int = 1, 
               retrain_interval: int = 1) -> Dict:
        """
        Explore chemical space using active learning.
        
        Args:
            candidate_features (np.ndarray): Candidate molecular features (N, feature_dim)
            property_evaluator (callable): Function to evaluate properties for selected features
            n_iterations (int): Number of exploration iterations
            batch_size (int): Number of points to select in each iteration
            retrain_interval (int): Interval for retraining the model
            
        Returns:
            dict: Exploration results
        """
        for iteration in range(n_iterations):
            print(f"Exploration iteration {iteration+1}/{n_iterations}")
            
            # Select next batch
            selected_indices = self.select_next_batch(
                candidate_features, batch_size
            )
            
            # Evaluate properties for selected features
            selected_features = candidate_features[selected_indices]
            selected_properties = property_evaluator(selected_features)
            
            # Update dataset
            self.update_dataset(selected_features, selected_properties)
            
            # Update exploration history
            self.exploration_history['iterations'].append(iteration)
            self.exploration_history['sampled_indices'].append(selected_indices)
            
            # Get current best property value
            best_property = np.min(self.dataset.properties[:, 0])
            self.exploration_history['best_properties'].append(best_property)
            
            # Retrain model if needed
            if (iteration + 1) % retrain_interval == 0:
                print("Retraining model...")
                self.train_model()
        
        return self.exploration_history
    
    def visualize_exploration(self) -> plt.Figure:
        """
        Visualize the exploration process.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if not self.exploration_history['iterations']:
            raise ValueError("No exploration history available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot best property value over iterations
        iterations = self.exploration_history['iterations']
        best_properties = self.exploration_history['best_properties']
        
        ax1.plot(iterations, best_properties, 'o-')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Best Property Value')
        ax1.set_title('Exploration Progress')
        ax1.grid(True, alpha=0.3)
        
        # Plot acquisition scores for the last iteration
        scores = self.exploration_history['acquisition_scores'][-1]
        
        ax2.hist(scores, bins=30)
        ax2.set_xlabel('Acquisition Score')
        ax2.set_ylabel('Count')
        ax2.set_title('Acquisition Score Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_chemical_space(self, features: np.ndarray, 
                               properties: Optional[np.ndarray] = None,
                               method: str = 'pca') -> plt.Figure:
        """
        Visualize the chemical space.
        
        Args:
            features (np.ndarray): Molecular features (N, feature_dim)
            properties (np.ndarray, optional): Molecular properties (N, property_dim)
            method (str): Dimensionality reduction method ('pca' or 'tsne')
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'tsne':
            reducer = TSNE(n_components=2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        reduced = reducer.fit_transform(features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot points
        if properties is not None:
            scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=properties[:, 0], 
                               cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Property Value')
        else:
            ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
        
        # Plot sampled points if available
        if self.exploration_history['sampled_indices']:
            sampled_indices = np.concatenate(self.exploration_history['sampled_indices'])
            sampled_reduced = reducer.transform(features[sampled_indices])
            ax.scatter(sampled_reduced[:, 0], sampled_reduced[:, 1], 
                      c='red', marker='x', s=100, label='Sampled Points')
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title('Chemical Space Visualization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def save_model(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path (str): Path to save the model
        """
        self.ml_potential.save(path)
    
    def load_model(self, path: str) -> None:
        """
        Load the model.
        
        Args:
            path (str): Path to load the model from
        """
        self.ml_potential.load(path)