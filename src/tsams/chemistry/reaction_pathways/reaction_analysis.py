"""
Chemical Space Explorer Module

This module implements chemical space exploration using the classical quantum formalism.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import os
import sys
import time

# Add the parent directory to the path to import the classical_quantum modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical_quantum.cyclotomic_field import CyclotomicField
from classical_quantum.spinor_structure import SpinorStructure
from classical_quantum.discosohedral_mapping import DiscosohedralMapping
from classical_quantum.phase_synchronization import PhaseSynchronization
import classical_quantum.utils as utils

class ChemicalSpaceDataset:
    """
    Dataset for chemical space exploration.
    """
    
    def __init__(self, feature_dim: int = 56):
        """
        Initialize the chemical space dataset.
        
        Args:
            feature_dim: The dimension of the molecular features
        """
        self.feature_dim = feature_dim
        self.features = []
        self.properties = []
        self.metadata = []
    
    def add_sample(self, features: np.ndarray, properties: np.ndarray, metadata: Dict[str, Any] = None) -> None:
        """
        Add a sample to the dataset.
        
        Args:
            features: The molecular features
            properties: The molecular properties
            metadata: Additional metadata about the molecule
        """
        # Ensure the features have the right dimension
        if len(features) != self.feature_dim:
            raise ValueError(f"Expected features of dimension {self.feature_dim}, got {len(features)}")
        
        # Add the sample
        self.features.append(features)
        self.properties.append(properties)
        self.metadata.append(metadata if metadata is not None else {})
    
    def add_samples(self, features: np.ndarray, properties: np.ndarray, metadata: List[Dict[str, Any]] = None) -> None:
        """
        Add multiple samples to the dataset.
        
        Args:
            features: The molecular features (batch_size, feature_dim)
            properties: The molecular properties (batch_size, property_dim)
            metadata: Additional metadata about the molecules
        """
        # Ensure the features have the right dimension
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected features of dimension {self.feature_dim}, got {features.shape[1]}")
        
        # Ensure the number of features and properties match
        if len(features) != len(properties):
            raise ValueError(f"Number of features ({len(features)}) and properties ({len(properties)}) must match")
        
        # Add the samples
        for i in range(len(features)):
            meta = metadata[i] if metadata is not None and i < len(metadata) else {}
            self.add_sample(features[i], properties[i], meta)
    
    def get_features(self) -> np.ndarray:
        """
        Get the features as a numpy array.
        
        Returns:
            The features
        """
        return np.array(self.features)
    
    def get_properties(self) -> np.ndarray:
        """
        Get the properties as a numpy array.
        
        Returns:
            The properties
        """
        return np.array(self.properties)
    
    def get_metadata(self) -> List[Dict[str, Any]]:
        """
        Get the metadata.
        
        Returns:
            The metadata
        """
        return self.metadata
    
    def get_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Get a sample from the dataset.
        
        Args:
            index: The index of the sample
            
        Returns:
            The features, properties, and metadata of the sample
        """
        if index < 0 or index >= len(self.features):
            raise ValueError(f"Index {index} out of range for dataset of size {len(self.features)}")
        
        return self.features[index], self.properties[index], self.metadata[index]
    
    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Get a batch of samples from the dataset.
        
        Args:
            indices: The indices of the samples
            
        Returns:
            The features, properties, and metadata of the samples
        """
        # Ensure the indices are valid
        for index in indices:
            if index < 0 or index >= len(self.features):
                raise ValueError(f"Index {index} out of range for dataset of size {len(self.features)}")
        
        # Get the samples
        features = np.array([self.features[i] for i in indices])
        properties = np.array([self.properties[i] for i in indices])
        metadata = [self.metadata[i] for i in indices]
        
        return features, properties, metadata
    
    def size(self) -> int:
        """
        Get the size of the dataset.
        
        Returns:
            The size of the dataset
        """
        return len(self.features)
    
    def save(self, filename: str) -> None:
        """
        Save the dataset to a file.
        
        Args:
            filename: The filename
        """
        # Create a dictionary with the dataset
        dataset = {
            'feature_dim': self.feature_dim,
            'features': self.features,
            'properties': self.properties,
            'metadata': self.metadata
        }
        
        # Save the dataset
        np.save(filename, dataset)
    
    @classmethod
    def load(cls, filename: str) -> 'ChemicalSpaceDataset':
        """
        Load a dataset from a file.
        
        Args:
            filename: The filename
            
        Returns:
            The loaded dataset
        """
        # Load the dataset
        dataset = np.load(filename, allow_pickle=True).item()
        
        # Create a new dataset
        new_dataset = cls(feature_dim=dataset['feature_dim'])
        
        # Add the samples
        for i in range(len(dataset['features'])):
            new_dataset.add_sample(
                dataset['features'][i],
                dataset['properties'][i],
                dataset['metadata'][i]
            )
        
        return new_dataset

class ChemicalSpaceExplorer:
    """
    Explorer for chemical space using the classical quantum formalism.
    """
    
    def __init__(self, feature_dim: int, property_dim: int, hidden_dims: List[int] = None, acquisition_function: str = 'uncertainty'):
        """
        Initialize the chemical space explorer.
        
        Args:
            feature_dim: The dimension of the molecular features
            property_dim: The dimension of the molecular properties
            hidden_dims: The dimensions of the hidden layers in the surrogate model
            acquisition_function: The acquisition function to use ('uncertainty', 'expected_improvement', or 'upper_confidence_bound')
        """
        self.feature_dim = feature_dim
        self.property_dim = property_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]
        self.acquisition_function = acquisition_function
        
        # Initialize the dataset
        self.dataset = ChemicalSpaceDataset(feature_dim)
        
        # Initialize the surrogate model
        self.surrogate_model = self._initialize_surrogate_model()
        
        # Initialize the classical quantum components
        self.cyclotomic_field = CyclotomicField(168)
        self.spinor_structure = SpinorStructure(56)
        self.discosohedral_mapping = DiscosohedralMapping(56)
        self.phase_synchronization = PhaseSynchronization(56)
        
        # Initialize the exploration history
        self.exploration_history = {
            'sampled_indices': [],
            'acquisition_values': [],
            'predicted_properties': [],
            'true_properties': []
        }
    
    def _initialize_surrogate_model(self) -> Dict[str, Any]:
        """
        Initialize the surrogate model.
        
        Returns:
            The surrogate model
        """
        # Create a simple neural network model
        model = {
            'layers': [],
            'training_history': {
                'loss': []
            }
        }
        
        # Input dimension is the feature dimension
        input_dim = self.feature_dim
        
        # Add hidden layers
        for hidden_dim in self.hidden_dims:
            # Create a layer
            layer = {
                'weights': np.random.randn(hidden_dim, input_dim) / np.sqrt(input_dim),
                'bias': np.zeros(hidden_dim)
            }
            model['layers'].append(layer)
            
            # Update the input dimension for the next layer
            input_dim = hidden_dim
        
        # Add the output layer
        output_layer = {
            'weights': np.random.randn(self.property_dim * 2, input_dim) / np.sqrt(input_dim),
            'bias': np.zeros(self.property_dim * 2)
        }
        model['layers'].append(output_layer)
        
        return model
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the ReLU activation function.
        
        Args:
            x: The input
            
        Returns:
            The output
        """
        return np.maximum(0, x)
    
    def _forward(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the surrogate model.
        
        Args:
            features: The molecular features
            
        Returns:
            The predicted mean and variance of the properties
        """
        # Forward pass through the hidden layers
        x = features
        for i, layer in enumerate(self.surrogate_model['layers'][:-1]):
            x = self._relu(layer['weights'] @ x + layer['bias'])
        
        # Forward pass through the output layer
        output_layer = self.surrogate_model['layers'][-1]
        y = output_layer['weights'] @ x + output_layer['bias']
        
        # Split the output into mean and variance
        mean = y[:self.property_dim]
        variance = np.exp(y[self.property_dim:])  # Ensure positive variance
        
        return mean, variance
    
    def initialize_dataset(self, features: np.ndarray, properties: np.ndarray, metadata: List[Dict[str, Any]] = None) -> None:
        """
        Initialize the dataset with samples.
        
        Args:
            features: The molecular features (batch_size, feature_dim)
            properties: The molecular properties (batch_size, property_dim)
            metadata: Additional metadata about the molecules
        """
        self.dataset.add_samples(features, properties, metadata)
    
    def train_model(self, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.01) -> Dict[str, List[float]]:
        """
        Train the surrogate model on the dataset.
        
        Args:
            epochs: The number of epochs
            batch_size: The batch size
            learning_rate: The learning rate
            
        Returns:
            A dictionary of training metrics
        """
        # Ensure the dataset is not empty
        if self.dataset.size() == 0:
            raise ValueError("Dataset is empty. Initialize it with samples first.")
        
        # Get the features and properties
        features = self.dataset.get_features()
        properties = self.dataset.get_properties()
        
        # Initialize the training metrics
        metrics = {
            'loss': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(len(features))
            
            # Initialize the epoch loss
            epoch_loss = 0.0
            
            # Process the data in batches
            for i in range(0, len(indices), batch_size):
                # Get the batch indices
                batch_indices = indices[i:i+batch_size]
                
                # Get the batch data
                batch_features = features[batch_indices]
                batch_properties = properties[batch_indices]
                
                # Forward pass
                batch_loss = 0.0
                batch_gradients = None
                
                for j in range(len(batch_features)):
                    # Forward pass
                    mean, variance = self._forward(batch_features[j])
                    
                    # Compute the negative log likelihood loss
                    diff = mean - batch_properties[j]
                    loss = 0.5 * np.sum(diff**2 / variance + np.log(variance))
                    batch_loss += loss
                    
                    # Compute the gradients
                    gradients = self._compute_gradients(batch_features[j], batch_properties[j], mean, variance)
                    
                    # Accumulate the gradients
                    if batch_gradients is None:
                        batch_gradients = gradients
                    else:
                        for k in range(len(batch_gradients)):
                            batch_gradients[k]['weights'] += gradients[k]['weights']
                            batch_gradients[k]['bias'] += gradients[k]['bias']
                
                # Average the loss and gradients
                batch_loss /= len(batch_features)
                for k in range(len(batch_gradients)):
                    batch_gradients[k]['weights'] /= len(batch_features)
                    batch_gradients[k]['bias'] /= len(batch_features)
                
                # Update the parameters
                self._update_parameters(batch_gradients, learning_rate)
                
                # Accumulate the epoch loss
                epoch_loss += batch_loss * len(batch_features)
            
            # Average the epoch loss
            epoch_loss /= len(features)
            metrics['loss'].append(epoch_loss)
            
            # Print the progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        # Store the training history
        self.surrogate_model['training_history'] = metrics
        
        return metrics
    
    def _compute_gradients(self, features: np.ndarray, properties: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        Compute the gradients of the negative log likelihood loss with respect to the parameters.
        
        Args:
            features: The molecular features
            properties: The molecular properties
            mean: The predicted mean
            variance: The predicted variance
            
        Returns:
            The gradients
        """
        # Initialize the gradients
        gradients = []
        for layer in self.surrogate_model['layers']:
            gradients.append({
                'weights': np.zeros_like(layer['weights']),
                'bias': np.zeros_like(layer['bias'])
            })
        
        # Forward pass
        activations = [features]
        for i, layer in enumerate(self.surrogate_model['layers'][:-1]):
            z = layer['weights'] @ activations[-1] + layer['bias']
            a = self._relu(z)
            activations.append(a)
        
        # Output layer
        output_layer = self.surrogate_model['layers'][-1]
        z = output_layer['weights'] @ activations[-1] + output_layer['bias']
        
        # Split the output into mean and variance
        mean_output = z[:self.property_dim]
        variance_output = z[self.property_dim:]
        
        # Compute the gradients of the loss with respect to the outputs
        diff = mean - properties
        d_mean = diff / variance
        d_variance = 0.5 * (1.0 / variance - diff**2 / variance**2)
        
        # Combine the gradients
        d_output = np.zeros_like(z)
        d_output[:self.property_dim] = d_mean
        d_output[self.property_dim:] = d_variance * np.exp(variance_output)  # Chain rule for exp
        
        # Backpropagate the error
        delta = d_output
        gradients[-1]['weights'] = np.outer(delta, activations[-1])
        gradients[-1]['bias'] = delta
        
        for i in range(len(self.surrogate_model['layers']) - 2, -1, -1):
            delta = self.surrogate_model['layers'][i+1]['weights'].T @ delta
            delta = delta * (activations[i+1] > 0)  # ReLU derivative
            gradients[i]['weights'] = np.outer(delta, activations[i])
            gradients[i]['bias'] = delta
        
        return gradients
    
    def _update_parameters(self, gradients: List[Dict[str, np.ndarray]], learning_rate: float) -> None:
        """
        Update the parameters using the gradients.
        
        Args:
            gradients: The gradients
            learning_rate: The learning rate
        """
        for i in range(len(self.surrogate_model['layers'])):
            self.surrogate_model['layers'][i]['weights'] -= learning_rate * gradients[i]['weights']
            self.surrogate_model['layers'][i]['bias'] -= learning_rate * gradients[i]['bias']
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the properties and uncertainties for molecular features.
        
        Args:
            features: The molecular features (batch_size, feature_dim)
            
        Returns:
            The predicted properties and uncertainties
        """
        # Ensure the features have the right dimension
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected features of dimension {self.feature_dim}, got {features.shape[1]}")
        
        # Initialize the predictions
        batch_size = features.shape[0]
        predictions = np.zeros((batch_size, self.property_dim))
        uncertainties = np.zeros((batch_size, self.property_dim))
        
        # Predict for each feature vector
        for i in range(batch_size):
            mean, variance = self._forward(features[i])
            predictions[i] = mean
            uncertainties[i] = np.sqrt(variance)  # Standard deviation as uncertainty
        
        return predictions, uncertainties
    
    def _compute_acquisition_function(self, mean: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
        """
        Compute the acquisition function value.
        
        Args:
            mean: The predicted mean
            uncertainty: The predicted uncertainty
            
        Returns:
            The acquisition function value
        """
        if self.acquisition_function == 'uncertainty':
            # Pure exploration: maximize uncertainty
            return uncertainty
        elif self.acquisition_function == 'expected_improvement':
            # Expected improvement: balance exploration and exploitation
            # Assume we want to minimize the property
            best_value = np.min(self.dataset.get_properties(), axis=0)
            improvement = best_value - mean
            z = improvement / (uncertainty + 1e-6)
            return improvement * (0.5 * (1 + np.tanh(z / np.sqrt(2))))
        elif self.acquisition_function == 'upper_confidence_bound':
            # Upper confidence bound: balance exploration and exploitation
            # Assume we want to minimize the property
            return -mean + 2.0 * uncertainty
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
    
    def select_next_samples(self, candidate_features: np.ndarray, batch_size: int = 1) -> List[int]:
        """
        Select the next samples to evaluate.
        
        Args:
            candidate_features: The candidate molecular features (n_candidates, feature_dim)
            batch_size: The number of samples to select
            
        Returns:
            The indices of the selected samples
        """
        # Ensure the features have the right dimension
        if candidate_features.ndim == 1:
            candidate_features = candidate_features.reshape(1, -1)
        
        if candidate_features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected features of dimension {self.feature_dim}, got {candidate_features.shape[1]}")
        
        # Predict the properties and uncertainties
        predictions, uncertainties = self.predict(candidate_features)
        
        # Compute the acquisition function values
        acquisition_values = self._compute_acquisition_function(predictions, uncertainties)
        
        # Compute the total acquisition value for each candidate
        total_acquisition_values = np.sum(acquisition_values, axis=1)
        
        # Select the samples with the highest acquisition values
        selected_indices = np.argsort(total_acquisition_values)[-batch_size:]
        
        return selected_indices.tolist()
    
    def explore(self, candidate_features: np.ndarray, property_evaluator: Callable[[np.ndarray], np.ndarray], n_iterations: int = 10, batch_size: int = 1, retrain_interval: int = 1) -> Dict[str, List[Any]]:
        """
        Explore the chemical space.
        
        Args:
            candidate_features: The candidate molecular features (n_candidates, feature_dim)
            property_evaluator: A function that evaluates the properties of molecular features
            n_iterations: The number of exploration iterations
            batch_size: The number of samples to select in each iteration
            retrain_interval: The interval for retraining the surrogate model
            
        Returns:
            A dictionary of exploration results
        """
        # Ensure the features have the right dimension
        if candidate_features.ndim == 1:
            candidate_features = candidate_features.reshape(1, -1)
        
        if candidate_features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected features of dimension {self.feature_dim}, got {candidate_features.shape[1]}")
        
        # Initialize the exploration history
        exploration_history = {
            'sampled_indices': [],
            'acquisition_values': [],
            'predicted_properties': [],
            'true_properties': []
        }
        
        # Exploration loop
        for iteration in range(n_iterations):
            print(f"Exploration iteration {iteration + 1}/{n_iterations}")
            
            # Select the next samples
            selected_indices = self.select_next_samples(candidate_features, batch_size)
            exploration_history['sampled_indices'].append(selected_indices)
            
            # Get the selected features
            selected_features = candidate_features[selected_indices]
            
            # Predict the properties and uncertainties
            predictions, uncertainties = self.predict(selected_features)
            exploration_history['predicted_properties'].append(predictions)
            
            # Compute the acquisition function values
            acquisition_values = self._compute_acquisition_function(predictions, uncertainties)
            exploration_history['acquisition_values'].append(acquisition_values)
            
            # Evaluate the properties
            true_properties = property_evaluator(selected_features)
            exploration_history['true_properties'].append(true_properties)
            
            # Add the samples to the dataset
            self.dataset.add_samples(selected_features, true_properties)
            
            # Retrain the surrogate model if needed
            if (iteration + 1) % retrain_interval == 0:
                print(f"Retraining surrogate model...")
                self.train_model()
        
        # Store the exploration history
        self.exploration_history = exploration_history
        
        return exploration_history
    
    def visualize_exploration(self) -> plt.Figure:
        """
        Visualize the exploration progress.
        
        Returns:
            The matplotlib figure
        """
        # Ensure there is exploration history
        if not self.exploration_history['sampled_indices']:
            raise ValueError("No exploration history available. Run explore() first.")
        
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot the predicted vs. true properties
        predicted_properties = np.concatenate(self.exploration_history['predicted_properties'])
        true_properties = np.concatenate(self.exploration_history['true_properties'])
        
        # Flatten the properties if they are multi-dimensional
        if predicted_properties.ndim > 1:
            predicted_properties = predicted_properties.flatten()
            true_properties = true_properties.flatten()
        
        ax1.scatter(true_properties, predicted_properties, alpha=0.7)
        ax1.plot([np.min(true_properties), np.max(true_properties)], [np.min(true_properties), np.max(true_properties)], 'r--')
        ax1.set_xlabel('True Properties')
        ax1.set_ylabel('Predicted Properties')
        ax1.set_title('Predicted vs. True Properties')
        
        # Plot the acquisition function values over iterations
        acquisition_values = [np.mean(values) for values in self.exploration_history['acquisition_values']]
        ax2.plot(range(1, len(acquisition_values) + 1), acquisition_values, 'b-o')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Mean Acquisition Value')
        ax2.set_title('Acquisition Function Values')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def visualize_chemical_space(self, features: np.ndarray, properties: np.ndarray, selected_indices: List[int] = None, title: str = "Chemical Space") -> plt.Figure:
        """
        Visualize the chemical space using PCA.
        
        Args:
            features: The molecular features (n_molecules, feature_dim)
            properties: The molecular properties (n_molecules, property_dim)
            selected_indices: The indices of selected molecules
            title: The title of the plot
            
        Returns:
            The matplotlib figure
        """
        # Ensure the features have the right dimension
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected features of dimension {self.feature_dim}, got {features.shape[1]}")
        
        # Use PCA to reduce dimensionality for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all data points with property as color
        if properties.ndim > 1 and properties.shape[1] > 1:
            # If there are multiple properties, use the first one for coloring
            property_values = properties[:, 0]
        else:
            property_values = properties.flatten()
        
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=property_values, cmap='viridis', alpha=0.7)
        
        # Plot selected points if provided
        if selected_indices is not None:
            selected_features_2d = features_2d[selected_indices]
            ax.scatter(selected_features_2d[:, 0], selected_features_2d[:, 1], c='red', marker='x', s=100, label='Selected Points')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Property Value')
        
        # Set labels
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title(title)
        
        # Add legend if selected points are plotted
        if selected_indices is not None:
            ax.legend()
        
        return fig
    
    def apply_quantum_like_transformations(self, features: np.ndarray) -> np.ndarray:
        """
        Apply quantum-like transformations to molecular features.
        
        Args:
            features: The molecular features (batch_size, feature_dim)
            
        Returns:
            The transformed features
        """
        # Ensure the features have the right dimension
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected features of dimension {self.feature_dim}, got {features.shape[1]}")
        
        # Initialize the transformed features
        batch_size = features.shape[0]
        transformed_features = np.zeros_like(features)
        
        # Transform each feature vector
        for i in range(batch_size):
            # Convert to complex state
            state = features[i].astype(complex)
            
            # Normalize the state
            norm = np.sqrt(np.sum(np.abs(state)**2))
            if norm > 0:
                state /= norm
            
            # Apply spinor transformation
            spinor_state = self.spinor_structure.apply_quaternionic_transformation(
                state[:56], (1.0, 0.1, 0.2, 0.3))
            
            # Apply discosohedral transformation
            discosohedral_state = self.discosohedral_mapping.apply_sheaf_transformation(
                spinor_state, 0)
            
            # Apply phase synchronization
            synchronized_state = self.phase_synchronization.apply_phase_synchronization(
                discosohedral_state)
            
            # Convert back to real features
            transformed_features[i, :56] = np.real(synchronized_state)
            if features.shape[1] > 56:
                transformed_features[i, 56:] = features[i, 56:]
        
        return transformed_features
    
    def save(self, filename: str) -> None:
        """
        Save the chemical space explorer to a file.
        
        Args:
            filename: The filename
        """
        # Create a dictionary with the explorer state
        explorer_state = {
            'feature_dim': self.feature_dim,
            'property_dim': self.property_dim,
            'hidden_dims': self.hidden_dims,
            'acquisition_function': self.acquisition_function,
            'surrogate_model': self.surrogate_model,
            'exploration_history': self.exploration_history
        }
        
        # Save the explorer state
        np.save(filename, explorer_state)
        
        # Save the dataset separately
        dataset_filename = filename.replace('.npy', '_dataset.npy')
        self.dataset.save(dataset_filename)
    
    @classmethod
    def load(cls, filename: str) -> 'ChemicalSpaceExplorer':
        """
        Load a chemical space explorer from a file.
        
        Args:
            filename: The filename
            
        Returns:
            The loaded chemical space explorer
        """
        # Load the explorer state
        explorer_state = np.load(filename, allow_pickle=True).item()
        
        # Create a new explorer
        explorer = cls(
            feature_dim=explorer_state['feature_dim'],
            property_dim=explorer_state['property_dim'],
            hidden_dims=explorer_state['hidden_dims'],
            acquisition_function=explorer_state['acquisition_function']
        )
        
        # Set the explorer state
        explorer.surrogate_model = explorer_state['surrogate_model']
        explorer.exploration_history = explorer_state['exploration_history']
        
        # Load the dataset
        dataset_filename = filename.replace('.npy', '_dataset.npy')
        if os.path.exists(dataset_filename):
            explorer.dataset = ChemicalSpaceDataset.load(dataset_filename)
        
        return explorer