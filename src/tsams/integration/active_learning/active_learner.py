"""
Active Learning for Chemical Space Exploration

This module implements active learning strategies within the Tibedo Framework
to efficiently explore chemical space.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Import Tibedo components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tibedo.ml.neural_networks.tibedo_neural_network import TibedoNeuralNetwork
from tibedo.core.prime_indexed.prime_indexed_structure import PrimeIndexedStructure
from tibedo.core.advanced.quantum_state import ConfigurableQuantumState


class UncertaintyEstimator:
    """
    Estimates uncertainty in predictions for active learning.
    
    This class implements various methods for estimating uncertainty in
    predictions, which is used to guide the active learning process.
    """
    
    def __init__(self, method='ensemble', n_estimators=5):
        """
        Initialize the UncertaintyEstimator.
        
        Args:
            method (str): Method for uncertainty estimation ('ensemble', 'dropout', or 'evidential')
            n_estimators (int): Number of estimators for ensemble method
        """
        self.method = method
        self.n_estimators = n_estimators
        self.models = []
        
        if method == 'ensemble':
            # Create ensemble of models
            for _ in range(n_estimators):
                self.models.append(None)  # Will be set later
        elif method == 'dropout':
            self.dropout_rate = 0.2
        elif method == 'evidential':
            # Evidential regression parameters
            self.evidential_params = None  # Will be set later
        else:
            raise ValueError(f"Unknown uncertainty estimation method: {method}")
    
    def set_models(self, models):
        """
        Set the models for ensemble uncertainty estimation.
        
        Args:
            models (list): List of models
        """
        if self.method == 'ensemble':
            self.models = models
    
    def estimate(self, x, models=None):
        """
        Estimate uncertainty for given inputs.
        
        Args:
            x (torch.Tensor): Input tensor
            models (list, optional): List of models for ensemble method
            
        Returns:
            torch.Tensor: Uncertainty estimates
        """
        if self.method == 'ensemble':
            # Use provided models or stored models
            if models is None:
                models = self.models
            
            # Get predictions from all models
            predictions = []
            for model in models:
                model.eval()
                with torch.no_grad():
                    predictions.append(model(x))
            
            # Calculate variance across predictions
            predictions = torch.stack(predictions)
            uncertainty = torch.var(predictions, dim=0)
            
            return uncertainty
        
        elif self.method == 'dropout':
            # Enable dropout during inference
            model = self.models[0]
            model.train()  # Set to train mode to enable dropout
            
            # Get multiple predictions with dropout
            predictions = []
            for _ in range(self.n_estimators):
                with torch.no_grad():
                    predictions.append(model(x))
            
            # Calculate variance across predictions
            predictions = torch.stack(predictions)
            uncertainty = torch.var(predictions, dim=0)
            
            return uncertainty
        
        elif self.method == 'evidential':
            # Evidential regression provides uncertainty directly
            model = self.models[0]
            model.eval()
            with torch.no_grad():
                _, uncertainty = model(x)
            
            return uncertainty


class TibedoActiveLearner:
    """
    Active Learning for Chemical Space Exploration using the Tibedo Framework.
    
    This class implements active learning strategies within the Tibedo Framework
    to efficiently explore chemical space.
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=1, 
                 uncertainty_method='ensemble', n_estimators=5):
        """
        Initialize the TibedoActiveLearner.
        
        Args:
            input_dim (int): Input dimension
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Output dimension
            uncertainty_method (str): Method for uncertainty estimation
            n_estimators (int): Number of estimators for ensemble method
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Create uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            method=uncertainty_method,
            n_estimators=n_estimators
        )
        
        # Create models
        self.models = []
        for _ in range(n_estimators):
            model = TibedoNeuralNetwork(input_dim, hidden_dims, output_dim)
            self.models.append(model)
        
        # Set models in uncertainty estimator
        self.uncertainty_estimator.set_models(self.models)
        
        # Create optimizers
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) for model in self.models]
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.selected_indices = []
        
        # Prime-indexed structure for chemical space representation
        self.prime_structure = PrimeIndexedStructure(max_index=100)
        
        # Quantum state for uncertainty representation
        self.quantum_state = ConfigurableQuantumState(dimension=7)
    
    def train_models(self, train_loader, val_loader=None, epochs=100):
        """
        Train all models in the ensemble.
        
        Args:
            train_loader (DataLoader): DataLoader for training data
            val_loader (DataLoader, optional): DataLoader for validation data
            epochs (int): Number of training epochs
        """
        for epoch in range(epochs):
            # Train all models
            epoch_train_loss = 0.0
            for model_idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                model.train()
                model_loss = 0.0
                
                for batch_x, batch_y in train_loader:
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(batch_x)
                    
                    # Compute loss
                    loss = self.criterion(outputs, batch_y)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    model_loss += loss.item()
                
                avg_model_loss = model_loss / len(train_loader)
                epoch_train_loss += avg_model_loss
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Model {model_idx+1}, Train Loss: {avg_model_loss:.6f}')
            
            # Average train loss across all models
            avg_epoch_train_loss = epoch_train_loss / len(self.models)
            self.train_losses.append(avg_epoch_train_loss)
            
            # Validate if validation loader is provided
            if val_loader is not None:
                epoch_val_loss = 0.0
                for model_idx, model in enumerate(self.models):
                    model.eval()
                    model_val_loss = 0.0
                    
                    with torch.no_grad():
                        for batch_x, batch_y in val_loader:
                            outputs = model(batch_x)
                            loss = self.criterion(outputs, batch_y)
                            model_val_loss += loss.item()
                    
                    avg_model_val_loss = model_val_loss / len(val_loader)
                    epoch_val_loss += avg_model_val_loss
                
                # Average validation loss across all models
                avg_epoch_val_loss = epoch_val_loss / len(self.models)
                self.val_losses.append(avg_epoch_val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {avg_epoch_val_loss:.6f}')
    
    def select_samples(self, pool_loader, n_samples=10, strategy='uncertainty'):
        """
        Select samples from the pool for labeling.
        
        Args:
            pool_loader (DataLoader): DataLoader for the pool of unlabeled data
            n_samples (int): Number of samples to select
            strategy (str): Selection strategy ('uncertainty', 'diversity', or 'hybrid')
            
        Returns:
            list: Indices of selected samples
        """
        # Get all pool data
        pool_data = []
        pool_indices = []
        for batch_x, batch_indices in pool_loader:
            pool_data.append(batch_x)
            pool_indices.append(batch_indices)
        
        pool_data = torch.cat(pool_data)
        pool_indices = torch.cat(pool_indices)
        
        if strategy == 'uncertainty':
            # Select samples with highest uncertainty
            uncertainties = self.uncertainty_estimator.estimate(pool_data)
            uncertainties = uncertainties.mean(dim=1)  # Average across output dimensions
            
            # Get indices of samples with highest uncertainty
            _, selected_idx = torch.topk(uncertainties, n_samples)
            selected_indices = pool_indices[selected_idx].tolist()
            
        elif strategy == 'diversity':
            # Select diverse samples using k-means clustering
            pool_data_np = pool_data.numpy()
            kmeans = KMeans(n_clusters=n_samples, random_state=0).fit(pool_data_np)
            
            # Find closest samples to cluster centers
            centers = kmeans.cluster_centers_
            distances = cdist(pool_data_np, centers)
            closest_points = np.argmin(distances, axis=0)
            
            selected_indices = pool_indices[closest_points].tolist()
            
        elif strategy == 'hybrid':
            # Hybrid approach: combine uncertainty and diversity
            
            # Get uncertainties
            uncertainties = self.uncertainty_estimator.estimate(pool_data)
            uncertainties = uncertainties.mean(dim=1).numpy()  # Average across output dimensions
            
            # Select top 2*n_samples uncertain samples
            top_uncertain_idx = np.argsort(-uncertainties)[:2*n_samples]
            top_uncertain_data = pool_data[top_uncertain_idx].numpy()
            
            # Cluster the uncertain samples to ensure diversity
            kmeans = KMeans(n_clusters=n_samples, random_state=0).fit(top_uncertain_data)
            
            # Find closest samples to cluster centers
            centers = kmeans.cluster_centers_
            distances = cdist(top_uncertain_data, centers)
            closest_points = np.argmin(distances, axis=0)
            
            # Map back to original indices
            selected_indices = pool_indices[top_uncertain_idx[closest_points]].tolist()
            
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
        
        # Store selected indices
        self.selected_indices.extend(selected_indices)
        
        return selected_indices
    
    def predict(self, x):
        """
        Make predictions with uncertainty estimates.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            tuple: (predictions, uncertainties)
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                predictions.append(model(x))
        
        # Calculate mean and variance across predictions
        predictions = torch.stack(predictions)
        mean_predictions = torch.mean(predictions, dim=0)
        uncertainties = torch.var(predictions, dim=0)
        
        return mean_predictions, uncertainties
    
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
    
    def visualize_selected_samples(self, all_data, selected_indices=None):
        """
        Visualize the selected samples.
        
        Args:
            all_data (torch.Tensor): All available data
            selected_indices (list, optional): Indices of selected samples
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if selected_indices is None:
            selected_indices = self.selected_indices
        
        # Use PCA to reduce dimensionality for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        all_data_np = all_data.numpy()
        all_data_2d = pca.fit_transform(all_data_np)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all data points
        ax.scatter(all_data_2d[:, 0], all_data_2d[:, 1], c='lightgray', alpha=0.5, label='Pool')
        
        # Plot selected data points
        selected_data_2d = all_data_2d[selected_indices]
        ax.scatter(selected_data_2d[:, 0], selected_data_2d[:, 1], c='red', s=100, marker='x', label='Selected')
        
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('Selected Samples in Chemical Space')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def save_models(self, path_prefix):
        """
        Save all models.
        
        Args:
            path_prefix (str): Prefix for model paths
        """
        for i, model in enumerate(self.models):
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizers[i].state_dict()
            }, f"{path_prefix}_model_{i}.pt")
    
    def load_models(self, path_prefix, n_models=None):
        """
        Load all models.
        
        Args:
            path_prefix (str): Prefix for model paths
            n_models (int, optional): Number of models to load
        """
        if n_models is None:
            n_models = len(self.models)
        
        for i in range(n_models):
            checkpoint = torch.load(f"{path_prefix}_model_{i}.pt")
            self.models[i].load_state_dict(checkpoint['model_state_dict'])
            self.optimizers[i].load_state_dict(checkpoint['optimizer_state_dict'])


class ChemicalSpaceExplorer:
    """
    Explores chemical space using active learning and the Tibedo Framework.
    
    This class implements a workflow for exploring chemical space using
    active learning strategies within the Tibedo Framework.
    """
    
    def __init__(self, feature_extractor=None, active_learner=None):
        """
        Initialize the ChemicalSpaceExplorer.
        
        Args:
            feature_extractor (callable, optional): Function to extract features from molecules
            active_learner (TibedoActiveLearner, optional): Active learner instance
        """
        self.feature_extractor = feature_extractor
        self.active_learner = active_learner
        
        # If active learner is not provided, create a default one
        if active_learner is None:
            self.active_learner = TibedoActiveLearner(
                input_dim=100,  # Default dimension
                hidden_dims=[128, 64, 32],
                output_dim=1
            )
        
        # History of explored regions
        self.explored_regions = []
        self.property_values = []
    
    def set_feature_extractor(self, feature_extractor):
        """
        Set the feature extractor.
        
        Args:
            feature_extractor (callable): Function to extract features from molecules
        """
        self.feature_extractor = feature_extractor
    
    def extract_features(self, molecules):
        """
        Extract features from molecules.
        
        Args:
            molecules (list): List of molecules
            
        Returns:
            torch.Tensor: Features
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not set")
        
        features = []
        for molecule in molecules:
            feature = self.feature_extractor(molecule)
            features.append(feature)
        
        return torch.stack(features)
    
    def explore(self, molecule_pool, property_calculator, n_iterations=10, 
                n_samples_per_iteration=5, initial_samples=None):
        """
        Explore chemical space using active learning.
        
        Args:
            molecule_pool (list): Pool of molecules to explore
            property_calculator (callable): Function to calculate properties for molecules
            n_iterations (int): Number of active learning iterations
            n_samples_per_iteration (int): Number of samples to select per iteration
            initial_samples (list, optional): Initial samples to start with
            
        Returns:
            dict: Results of exploration
        """
        # Extract features for all molecules in the pool
        all_features = self.extract_features(molecule_pool)
        
        # Create dataset from pool
        pool_indices = list(range(len(molecule_pool)))
        
        # Initialize with some samples if provided
        if initial_samples is not None:
            labeled_indices = initial_samples
        else:
            # Randomly select initial samples
            labeled_indices = np.random.choice(pool_indices, size=n_samples_per_iteration, replace=False).tolist()
        
        # Calculate properties for initial samples
        labeled_features = all_features[labeled_indices]
        labeled_properties = []
        for idx in labeled_indices:
            property_value = property_calculator(molecule_pool[idx])
            labeled_properties.append(property_value)
        
        labeled_properties = torch.tensor(labeled_properties).float().reshape(-1, 1)
        
        # Active learning loop
        for iteration in range(n_iterations):
            print(f"Active Learning Iteration {iteration+1}/{n_iterations}")
            
            # Update pool (remove labeled samples)
            pool_indices = list(set(range(len(molecule_pool))) - set(labeled_indices))
            
            # Create data loaders
            from torch.utils.data import TensorDataset, DataLoader
            
            # Labeled dataset
            labeled_dataset = TensorDataset(labeled_features, labeled_properties)
            labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)
            
            # Pool dataset
            pool_features = all_features[pool_indices]
            pool_dataset = TensorDataset(pool_features, torch.tensor(pool_indices))
            pool_loader = DataLoader(pool_dataset, batch_size=32, shuffle=False)
            
            # Train models on labeled data
            self.active_learner.train_models(labeled_loader, epochs=50)
            
            # Select new samples from pool
            new_indices = self.active_learner.select_samples(pool_loader, n_samples=n_samples_per_iteration)
            
            # Calculate properties for new samples
            new_features = all_features[new_indices]
            new_properties = []
            for idx in new_indices:
                property_value = property_calculator(molecule_pool[idx])
                new_properties.append(property_value)
            
            new_properties = torch.tensor(new_properties).float().reshape(-1, 1)
            
            # Add new samples to labeled set
            labeled_indices.extend(new_indices)
            labeled_features = torch.cat([labeled_features, new_features])
            labeled_properties = torch.cat([labeled_properties, new_properties])
            
            # Store explored region
            self.explored_regions.append(new_indices)
            self.property_values.append(new_properties.numpy())
            
            # Print progress
            print(f"  Selected {len(new_indices)} new samples")
            print(f"  Total labeled samples: {len(labeled_indices)}")
            print(f"  Average property value: {labeled_properties.mean().item():.4f}")
            
            # Visualize current state
            fig = self.active_learner.visualize_selected_samples(all_features, labeled_indices)
            plt.close(fig)  # Close figure to avoid display in notebook
        
        # Final training on all labeled data
        labeled_dataset = TensorDataset(labeled_features, labeled_properties)
        labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)
        self.active_learner.train_models(labeled_loader, epochs=100)
        
        # Make predictions for all molecules
        all_predictions, all_uncertainties = self.active_learner.predict(all_features)
        
        # Return results
        results = {
            'labeled_indices': labeled_indices,
            'labeled_properties': labeled_properties.numpy(),
            'all_predictions': all_predictions.numpy(),
            'all_uncertainties': all_uncertainties.numpy(),
            'explored_regions': self.explored_regions,
            'property_values': self.property_values
        }
        
        return results
    
    def visualize_exploration(self, all_features, results):
        """
        Visualize the exploration process.
        
        Args:
            all_features (torch.Tensor): Features for all molecules
            results (dict): Results from exploration
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        # Use PCA to reduce dimensionality for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        all_features_np = all_features.numpy()
        all_features_2d = pca.fit_transform(all_features_np)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot all data points with predicted property as color
        predictions = results['all_predictions'].flatten()
        scatter = ax.scatter(all_features_2d[:, 0], all_features_2d[:, 1], 
                             c=predictions, cmap='viridis', alpha=0.5)
        
        # Plot labeled points
        labeled_indices = results['labeled_indices']
        labeled_features_2d = all_features_2d[labeled_indices]
        labeled_properties = results['labeled_properties'].flatten()
        
        ax.scatter(labeled_features_2d[:, 0], labeled_features_2d[:, 1], 
                   c=labeled_properties, cmap='viridis', 
                   s=100, edgecolors='black', linewidths=1.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Property Value')
        
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('Chemical Space Exploration')
        ax.grid(True)
        
        return fig