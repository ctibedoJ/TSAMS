"""
TIBEDO API Module

This module provides the main API for TIBEDO, integrating all components
into a unified interface.
"""

import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

# Import TIBEDO components
from .factory import QMLFactory, TensorNetworkFactory, GeometryFactory
from .configuration import Configuration, QMLConfig, TensorNetworkConfig, GeometryConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantumAPI:
    """
    API for Quantum Machine Learning components.
    
    This class provides a unified interface for working with QML components.
    """
    
    def __init__(self, config: Optional[QMLConfig] = None):
        """
        Initialize the Quantum API.
        
        Args:
            config: Configuration for QML components
        """
        self.config = config or QMLConfig()
        self.factory = QMLFactory()
        
        logger.info("Initialized Quantum API")
    
    def create_quantum_neural_network(self, **kwargs):
        """
        Create a quantum neural network.
        
        Args:
            **kwargs: Configuration overrides
            
        Returns:
            Quantum neural network
        """
        # Merge configuration with overrides
        n_qubits = kwargs.get("n_qubits", self.config.n_qubits)
        n_layers = kwargs.get("n_layers", self.config.n_layers)
        use_spinor = kwargs.get("use_spinor", self.config.use_spinor)
        
        return self.factory.create_quantum_neural_network(
            n_qubits=n_qubits,
            n_layers=n_layers,
            use_spinor=use_spinor,
            **kwargs
        )
    
    def create_feature_map(self, **kwargs):
        """
        Create a quantum feature map.
        
        Args:
            **kwargs: Configuration overrides
            
        Returns:
            Quantum feature map
        """
        # Merge configuration with overrides
        n_qubits = kwargs.get("n_qubits", self.config.n_qubits)
        n_features = kwargs.get("n_features", self.config.n_features)
        map_type = kwargs.get("feature_map_type", self.config.feature_map_type)
        
        return self.factory.create_feature_map(
            n_qubits=n_qubits,
            n_features=n_features,
            map_type=map_type,
            **kwargs
        )
    
    def train_quantum_model(self, model, X_train, y_train, **kwargs):
        """
        Train a quantum model.
        
        Args:
            model: Quantum model to train
            X_train: Training data
            y_train: Training labels
            **kwargs: Training parameters
            
        Returns:
            Training history
        """
        # Get training parameters
        n_epochs = kwargs.get("n_epochs", 100)
        batch_size = kwargs.get("batch_size", 10)
        learning_rate = kwargs.get("learning_rate", self.config.learning_rate)
        
        # Create optimizer
        optimizer_type = kwargs.get("optimizer_type", self.config.optimizer_type)
        if optimizer_type == "quantum_gradient_descent":
            optimizer = self.factory.create("quantum_gradient_descent", learning_rate=learning_rate)
        elif optimizer_type == "parameter_shift":
            optimizer = self.factory.create("parameter_shift")
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Define loss function
        def loss_fn(y_true, y_pred):
            return np.mean((y_true - y_pred)**2)
        
        # Train the model
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            optimizer=optimizer,
            loss_fn=loss_fn,
            n_epochs=n_epochs,
            batch_size=batch_size
        )
        
        return history
    
    def evaluate_quantum_model(self, model, X_test, y_test):
        """
        Evaluate a quantum model.
        
        Args:
            model: Quantum model to evaluate
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        # Make predictions
        y_pred = np.array([model.forward(x) for x in X_test])
        
        # Compute metrics
        mse = np.mean((y_test - y_pred)**2)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # For classification
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            # Multi-class classification
            y_pred_class = np.argmax(y_pred, axis=1)
            y_true_class = np.argmax(y_test, axis=1)
        else:
            # Binary classification
            y_pred_class = (y_pred > 0.5).astype(int)
            y_true_class = y_test
        
        accuracy = np.mean(y_pred_class == y_true_class)
        
        return {
            "mse": mse,
            "mae": mae,
            "accuracy": accuracy
        }


class TensorNetworkAPI:
    """
    API for Tensor Network components.
    
    This class provides a unified interface for working with tensor network components.
    """
    
    def __init__(self, config: Optional[TensorNetworkConfig] = None):
        """
        Initialize the Tensor Network API.
        
        Args:
            config: Configuration for tensor network components
        """
        self.config = config or TensorNetworkConfig()
        self.factory = TensorNetworkFactory()
        
        logger.info("Initialized Tensor Network API")
    
    def create_mps(self, **kwargs):
        """
        Create a Matrix Product State.
        
        Args:
            **kwargs: Configuration overrides
            
        Returns:
            Matrix Product State
        """
        # Merge configuration with overrides
        n_sites = kwargs.get("n_sites", self.config.mps_n_sites)
        local_dim = kwargs.get("local_dim", self.config.mps_local_dim)
        bond_dim = kwargs.get("bond_dim", self.config.mps_bond_dim)
        
        return self.factory.create_mps(
            n_sites=n_sites,
            local_dim=local_dim,
            bond_dim=bond_dim,
            **kwargs
        )
    
    def create_peps(self, **kwargs):
        """
        Create a Projected Entangled Pair State.
        
        Args:
            **kwargs: Configuration overrides
            
        Returns:
            Projected Entangled Pair State
        """
        # Merge configuration with overrides
        width = kwargs.get("width", self.config.peps_width)
        height = kwargs.get("height", self.config.peps_height)
        local_dim = kwargs.get("local_dim", self.config.peps_local_dim)
        bond_dim = kwargs.get("bond_dim", self.config.peps_bond_dim)
        
        return self.factory.create_peps(
            width=width,
            height=height,
            local_dim=local_dim,
            bond_dim=bond_dim,
            **kwargs
        )
    
    def create_mera(self, **kwargs):
        """
        Create a Multi-scale Entanglement Renormalization Ansatz.
        
        Args:
            **kwargs: Configuration overrides
            
        Returns:
            Multi-scale Entanglement Renormalization Ansatz
        """
        # Merge configuration with overrides
        n_sites = kwargs.get("n_sites", self.config.mera_n_sites)
        local_dim = kwargs.get("local_dim", self.config.mera_local_dim)
        n_layers = kwargs.get("n_layers", self.config.mera_n_layers)
        
        return self.factory.create_mera(
            n_sites=n_sites,
            local_dim=local_dim,
            n_layers=n_layers,
            **kwargs
        )
    
    def simulate_quantum_circuit(self, circuit, **kwargs):
        """
        Simulate a quantum circuit using tensor networks.
        
        Args:
            circuit: Quantum circuit to simulate
            **kwargs: Simulation parameters
            
        Returns:
            Simulation results
        """
        # Get simulation parameters
        framework = kwargs.get("framework", self.config.circuit_framework)
        shots = kwargs.get("shots", None)
        use_gpu = kwargs.get("use_gpu", self.config.use_gpu)
        
        # Create simulator
        simulator = self.factory.create_simulator(use_gpu=use_gpu)
        
        # Simulate the circuit
        results = simulator.simulate(circuit, framework=framework, shots=shots)
        
        return results
    
    def get_state_vector(self, circuit, **kwargs):
        """
        Get the state vector of a quantum circuit.
        
        Args:
            circuit: Quantum circuit
            **kwargs: Simulation parameters
            
        Returns:
            State vector
        """
        # Get simulation parameters
        framework = kwargs.get("framework", self.config.circuit_framework)
        use_gpu = kwargs.get("use_gpu", self.config.use_gpu)
        
        # Create simulator
        simulator = self.factory.create_simulator(use_gpu=use_gpu)
        
        # Get the state vector
        state_vector = simulator.get_state_vector(circuit, framework=framework)
        
        return state_vector
    
    def compute_entanglement_entropy(self, mps, bond):
        """
        Compute the entanglement entropy across a bond in an MPS.
        
        Args:
            mps: Matrix Product State
            bond: Bond index
            
        Returns:
            Entanglement entropy
        """
        return mps.compute_entanglement_entropy(bond)


class GeometryAPI:
    """
    API for Non-Euclidean Geometry components.
    
    This class provides a unified interface for working with geometry components.
    """
    
    def __init__(self, config: Optional[GeometryConfig] = None):
        """
        Initialize the Geometry API.
        
        Args:
            config: Configuration for geometry components
        """
        self.config = config or GeometryConfig()
        self.factory = GeometryFactory()
        
        logger.info("Initialized Geometry API")
    
    def create_manifold(self, **kwargs):
        """
        Create a Riemannian manifold.
        
        Args:
            **kwargs: Configuration overrides
            
        Returns:
            Riemannian manifold
        """
        # Merge configuration with overrides
        dimension = kwargs.get("dimension", self.config.dimension)
        manifold_type = kwargs.get("manifold_type", self.config.manifold_type)
        
        return self.factory.create_manifold(
            dimension=dimension,
            manifold_type=manifold_type,
            **kwargs
        )
    
    def create_hyperbolic_layer(self, **kwargs):
        """
        Create a hyperbolic neural network layer.
        
        Args:
            **kwargs: Configuration overrides
            
        Returns:
            Hyperbolic layer
        """
        # Merge configuration with overrides
        in_features = kwargs.get("in_features", self.config.in_features)
        out_features = kwargs.get("out_features", self.config.out_features)
        
        return self.factory.create_hyperbolic_layer(
            in_features=in_features,
            out_features=out_features,
            **kwargs
        )
    
    def create_spherical_cnn(self, **kwargs):
        """
        Create a spherical CNN.
        
        Args:
            **kwargs: Configuration overrides
            
        Returns:
            Spherical CNN
        """
        # Merge configuration with overrides
        in_channels = kwargs.get("in_channels", self.config.in_channels)
        out_channels = kwargs.get("out_channels", self.config.out_channels)
        hidden_channels = kwargs.get("hidden_channels", self.config.hidden_channels)
        n_layers = kwargs.get("n_layers", self.config.n_layers)
        max_degree = kwargs.get("max_degree", self.config.max_degree)
        
        return self.factory.create_spherical_cnn(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            max_degree=max_degree,
            **kwargs
        )
    
    def create_curvature_aware_optimizer(self, params, **kwargs):
        """
        Create a curvature-aware optimizer.
        
        Args:
            params: Parameters to optimize
            **kwargs: Configuration overrides
            
        Returns:
            Curvature-aware optimizer
        """
        # Merge configuration with overrides
        lr = kwargs.get("learning_rate", self.config.learning_rate)
        
        return self.factory.create_curvature_aware_optimizer(
            params=params,
            lr=lr,
            **kwargs
        )
    
    def compute_geodesic(self, manifold, x0, v0, t_span, n_points=100):
        """
        Compute a geodesic curve on a manifold.
        
        Args:
            manifold: Riemannian manifold
            x0: Initial position
            v0: Initial velocity
            t_span: Time span for integration
            n_points: Number of points to return
            
        Returns:
            Tuple of (times, positions)
        """
        return manifold.geodesic(x0, v0, t_span, n_points)
    
    def estimate_ricci_curvature(self, X, n_neighbors=None, alpha=None):
        """
        Estimate the Ricci curvature of a data manifold.
        
        Args:
            X: Data points
            n_neighbors: Number of neighbors to consider
            alpha: Parameter for the Ollivier-Ricci curvature
            
        Returns:
            Ricci curvature estimates
        """
        # Merge configuration with overrides
        n_neighbors = n_neighbors or self.config.n_neighbors
        alpha = alpha or self.config.alpha
        
        # Create Ricci curvature estimator
        ricci_estimator = self.factory.create("ricci_curvature", n_neighbors=n_neighbors, alpha=alpha)
        
        # Estimate the Ricci curvature
        curvature = ricci_estimator.fit_transform(X)
        
        return curvature


class TIBEDO:
    """
    Main API for TIBEDO.
    
    This class provides a unified interface for all TIBEDO components.
    """
    
    def __init__(self, config: Optional[Configuration] = None):
        """
        Initialize the TIBEDO API.
        
        Args:
            config: Configuration for TIBEDO
        """
        self.config = config or Configuration()
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        # Set random seed
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        
        # Initialize component APIs
        self.quantum = QuantumAPI(self.config.qml)
        self.tensor_network = TensorNetworkAPI(self.config.tensor_network)
        self.geometry = GeometryAPI(self.config.geometry)
        
        logger.info("Initialized TIBEDO API")
    
    def load_configuration(self, file_path: str) -> None:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file
        """
        from .configuration import create_configuration_from_file
        
        self.config = create_configuration_from_file(file_path)
        
        # Update component APIs
        self.quantum.config = self.config.qml
        self.tensor_network.config = self.config.tensor_network
        self.geometry.config = self.config.geometry
        
        # Update logging level
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        # Update random seed
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        
        logger.info(f"Loaded configuration from {file_path}")
    
    def save_configuration(self, file_path: str) -> None:
        """
        Save configuration to a file.
        
        Args:
            file_path: Path to the configuration file
        """
        if file_path.endswith('.json'):
            self.config.save_json(file_path)
        elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
            self.config.save_yaml(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def update_configuration(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with values from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        self.config.update(config_dict)
        
        # Update component APIs
        self.quantum.config = self.config.qml
        self.tensor_network.config = self.config.tensor_network
        self.geometry.config = self.config.geometry
        
        # Update logging level
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        # Update random seed
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        
        logger.info("Updated configuration")
    
    def __repr__(self) -> str:
        """String representation of the TIBEDO API."""
        return f"TIBEDO(config={self.config})"


def create_tibedo(config_file: Optional[str] = None) -> TIBEDO:
    """
    Create a TIBEDO API instance.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        TIBEDO API instance
    """
    if config_file is None:
        return TIBEDO()
    else:
        tibedo = TIBEDO()
        tibedo.load_configuration(config_file)
        return tibedo
"""