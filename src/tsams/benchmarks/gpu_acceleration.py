"""
GPU Acceleration Module for TIBEDO Framework

This module provides GPU acceleration capabilities for the TIBEDO Framework,
enabling efficient computation for large-scale quantum chemistry problems.
"""

import numpy as np
import time
import os
import threading
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check if CuPy is available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

class GPUManager:
    """
    Manager for GPU resources.
    
    This class provides tools for managing GPU resources, including device selection,
    memory management, and fallback to CPU when necessary.
    """
    
    def __init__(self, device_id: Optional[int] = None, memory_fraction: float = 0.8):
        """
        Initialize the GPUManager.
        
        Args:
            device_id (int, optional): GPU device ID. If None, uses the first available GPU.
            memory_fraction (float): Fraction of GPU memory to use (0.0 to 1.0)
        """
        self.device_id = device_id
        self.memory_fraction = memory_fraction
        
        # Check if GPU is available
        self.gpu_available = self._check_gpu_available()
        
        # Set device if GPU is available
        if self.gpu_available:
            self._set_device()
            
        # Initialize statistics
        self.stats = {
            'gpu_available': self.gpu_available,
            'device_id': self.device_id,
            'memory_used': 0,
            'memory_total': 0,
            'operations_gpu': 0,
            'operations_cpu': 0
        }
        
    def _check_gpu_available(self) -> bool:
        """
        Check if GPU is available.
        
        Returns:
            bool: True if GPU is available, False otherwise
        """
        if TORCH_AVAILABLE:
            return torch.cuda.is_available()
        elif CUPY_AVAILABLE:
            return cp.cuda.is_available()
        else:
            return False
            
    def _set_device(self) -> None:
        """
        Set the GPU device.
        """
        if not self.gpu_available:
            return
            
        # Set device ID if not specified
        if self.device_id is None:
            self.device_id = 0
            
        # Set device
        if TORCH_AVAILABLE:
            torch.cuda.set_device(self.device_id)
            
            # Limit memory usage
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction, self.device_id)
        elif CUPY_AVAILABLE:
            cp.cuda.Device(self.device_id).use()
            
        # Update statistics
        self._update_memory_stats()
        
    def _update_memory_stats(self) -> None:
        """
        Update memory statistics.
        """
        if not self.gpu_available:
            return
            
        if TORCH_AVAILABLE:
            self.stats['memory_used'] = torch.cuda.memory_allocated(self.device_id)
            self.stats['memory_total'] = torch.cuda.get_device_properties(self.device_id).total_memory
        elif CUPY_AVAILABLE:
            device = cp.cuda.Device(self.device_id)
            self.stats['memory_used'] = device.mem_info[0] - device.mem_info[1]
            self.stats['memory_total'] = device.mem_info[0]
            
    def to_gpu(self, data: Union[np.ndarray, List, Tuple]) -> Any:
        """
        Transfer data to GPU.
        
        Args:
            data: Data to transfer
            
        Returns:
            Data on GPU
        """
        if not self.gpu_available:
            # Update statistics
            self.stats['operations_cpu'] += 1
            return data
            
        try:
            # Transfer data to GPU
            if TORCH_AVAILABLE:
                if isinstance(data, np.ndarray):
                    gpu_data = torch.from_numpy(data).cuda(self.device_id)
                elif isinstance(data, (list, tuple)):
                    gpu_data = torch.tensor(data).cuda(self.device_id)
                else:
                    gpu_data = data
            elif CUPY_AVAILABLE:
                if isinstance(data, np.ndarray):
                    gpu_data = cp.array(data)
                elif isinstance(data, (list, tuple)):
                    gpu_data = cp.array(data)
                else:
                    gpu_data = data
            else:
                gpu_data = data
                
            # Update statistics
            self.stats['operations_gpu'] += 1
            self._update_memory_stats()
            
            return gpu_data
        except Exception as e:
            print(f"Error transferring data to GPU: {e}")
            
            # Update statistics
            self.stats['operations_cpu'] += 1
            
            return data
            
    def to_cpu(self, data: Any) -> np.ndarray:
        """
        Transfer data to CPU.
        
        Args:
            data: Data to transfer
            
        Returns:
            np.ndarray: Data on CPU
        """
        if not self.gpu_available:
            return data
            
        try:
            # Transfer data to CPU
            if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
                cpu_data = data.cpu().numpy()
            elif CUPY_AVAILABLE and isinstance(data, cp.ndarray):
                cpu_data = cp.asnumpy(data)
            else:
                cpu_data = data
                
            # Update statistics
            self._update_memory_stats()
            
            return cpu_data
        except Exception as e:
            print(f"Error transferring data to CPU: {e}")
            return data
            
    def clear_cache(self) -> None:
        """
        Clear GPU memory cache.
        """
        if not self.gpu_available:
            return
            
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()
        elif CUPY_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
            
        # Update statistics
        self._update_memory_stats()
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get GPU statistics.
        
        Returns:
            Dict[str, Any]: GPU statistics
        """
        # Update memory statistics
        self._update_memory_stats()
        
        return self.stats.copy()


class GPUAccelerator:
    """
    GPU accelerator for the TIBEDO Framework.
    
    This class provides GPU-accelerated implementations of common operations used in
    the TIBEDO Framework, such as matrix operations, tensor operations, and neural
    network operations.
    """
    
    def __init__(self, device_id: Optional[int] = None, memory_fraction: float = 0.8):
        """
        Initialize the GPUAccelerator.
        
        Args:
            device_id (int, optional): GPU device ID. If None, uses the first available GPU.
            memory_fraction (float): Fraction of GPU memory to use (0.0 to 1.0)
        """
        # Create GPU manager
        self.gpu_manager = GPUManager(device_id=device_id, memory_fraction=memory_fraction)
        
        # Check if GPU is available
        self.gpu_available = self.gpu_manager.gpu_available
        
        # Print GPU status
        if self.gpu_available:
            print(f"GPU acceleration enabled (Device ID: {self.gpu_manager.device_id})")
        else:
            print("GPU acceleration not available, falling back to CPU")
            
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Perform matrix multiplication on GPU.
        
        Args:
            A (np.ndarray): First matrix
            B (np.ndarray): Second matrix
            
        Returns:
            np.ndarray: Result of matrix multiplication
        """
        if not self.gpu_available:
            return np.matmul(A, B)
            
        try:
            # Transfer matrices to GPU
            A_gpu = self.gpu_manager.to_gpu(A)
            B_gpu = self.gpu_manager.to_gpu(B)
            
            # Perform matrix multiplication
            if TORCH_AVAILABLE and isinstance(A_gpu, torch.Tensor):
                C_gpu = torch.matmul(A_gpu, B_gpu)
            elif CUPY_AVAILABLE and isinstance(A_gpu, cp.ndarray):
                C_gpu = cp.matmul(A_gpu, B_gpu)
            else:
                # Fallback to CPU
                return np.matmul(A, B)
                
            # Transfer result to CPU
            C = self.gpu_manager.to_cpu(C_gpu)
            
            return C
        except Exception as e:
            print(f"Error in GPU matrix multiplication: {e}")
            
            # Fallback to CPU
            return np.matmul(A, B)
            
    def eigendecomposition(self, matrix: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform eigenvalue decomposition on GPU.
        
        Args:
            matrix (np.ndarray): Matrix to decompose
            k (int, optional): Number of eigenvalues/eigenvectors to compute
                If None, computes all eigenvalues/eigenvectors.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Eigenvalues and eigenvectors
        """
        if not self.gpu_available:
            # Use numpy's eigenvalue solver for CPU
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            
            # Sort eigenvalues and eigenvectors
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Return only k eigenvalues/eigenvectors if specified
            if k is not None:
                return eigenvalues[:k], eigenvectors[:, :k]
            else:
                return eigenvalues, eigenvectors
                
        try:
            # Transfer matrix to GPU
            matrix_gpu = self.gpu_manager.to_gpu(matrix)
            
            if TORCH_AVAILABLE and isinstance(matrix_gpu, torch.Tensor):
                # Use PyTorch's eigenvalue solver
                eigenvalues, eigenvectors = torch.linalg.eigh(matrix_gpu)
                
                # Sort eigenvalues and eigenvectors
                idx = eigenvalues.argsort(descending=True)
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                # Transfer results to CPU
                eigenvalues = self.gpu_manager.to_cpu(eigenvalues)
                eigenvectors = self.gpu_manager.to_cpu(eigenvectors)
                
                # Return only k eigenvalues/eigenvectors if specified
                if k is not None:
                    return eigenvalues[:k], eigenvectors[:, :k]
                else:
                    return eigenvalues, eigenvectors
            elif CUPY_AVAILABLE and isinstance(matrix_gpu, cp.ndarray):
                # Use CuPy's eigenvalue solver
                eigenvalues, eigenvectors = cp.linalg.eigh(matrix_gpu)
                
                # Sort eigenvalues and eigenvectors
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                # Transfer results to CPU
                eigenvalues = self.gpu_manager.to_cpu(eigenvalues)
                eigenvectors = self.gpu_manager.to_cpu(eigenvectors)
                
                # Return only k eigenvalues/eigenvectors if specified
                if k is not None:
                    return eigenvalues[:k], eigenvectors[:, :k]
                else:
                    return eigenvalues, eigenvectors
            else:
                # Fallback to CPU
                return self.eigendecomposition(matrix, k)
        except Exception as e:
            print(f"Error in GPU eigendecomposition: {e}")
            
            # Fallback to CPU
            return self.eigendecomposition(matrix, k)
            
    def svd(self, matrix: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform singular value decomposition on GPU.
        
        Args:
            matrix (np.ndarray): Matrix to decompose
            k (int, optional): Number of singular values/vectors to compute
                If None, computes all singular values/vectors.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: U, S, V matrices
        """
        if not self.gpu_available:
            # Use numpy's SVD for CPU
            U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
            
            # Return only k singular values/vectors if specified
            if k is not None:
                return U[:, :k], S[:k], Vt[:k, :].T
            else:
                return U, S, Vt.T
                
        try:
            # Transfer matrix to GPU
            matrix_gpu = self.gpu_manager.to_gpu(matrix)
            
            if TORCH_AVAILABLE and isinstance(matrix_gpu, torch.Tensor):
                # Use PyTorch's SVD
                U, S, V = torch.linalg.svd(matrix_gpu, full_matrices=False)
                
                # Transfer results to CPU
                U = self.gpu_manager.to_cpu(U)
                S = self.gpu_manager.to_cpu(S)
                V = self.gpu_manager.to_cpu(V)
                
                # Return only k singular values/vectors if specified
                if k is not None:
                    return U[:, :k], S[:k], V[:k, :].T
                else:
                    return U, S, V.T
            elif CUPY_AVAILABLE and isinstance(matrix_gpu, cp.ndarray):
                # Use CuPy's SVD
                U, S, Vt = cp.linalg.svd(matrix_gpu, full_matrices=False)
                
                # Transfer results to CPU
                U = self.gpu_manager.to_cpu(U)
                S = self.gpu_manager.to_cpu(S)
                Vt = self.gpu_manager.to_cpu(Vt)
                
                # Return only k singular values/vectors if specified
                if k is not None:
                    return U[:, :k], S[:k], Vt[:k, :].T
                else:
                    return U, S, Vt.T
            else:
                # Fallback to CPU
                return self.svd(matrix, k)
        except Exception as e:
            print(f"Error in GPU SVD: {e}")
            
            # Fallback to CPU
            return self.svd(matrix, k)
            
    def tensor_operations(self, operation: str, *tensors, **kwargs) -> np.ndarray:
        """
        Perform tensor operations on GPU.
        
        Args:
            operation (str): Operation to perform ('add', 'subtract', 'multiply', 'divide', 'dot', 'outer')
            *tensors: Tensors to operate on
            **kwargs: Additional arguments for the operation
            
        Returns:
            np.ndarray: Result of the operation
        """
        if not self.gpu_available:
            # Perform operation on CPU
            return self._tensor_operations_cpu(operation, *tensors, **kwargs)
            
        try:
            # Transfer tensors to GPU
            gpu_tensors = [self.gpu_manager.to_gpu(tensor) for tensor in tensors]
            
            if TORCH_AVAILABLE and all(isinstance(tensor, torch.Tensor) for tensor in gpu_tensors):
                # Perform operation using PyTorch
                result_gpu = self._tensor_operations_torch(operation, *gpu_tensors, **kwargs)
            elif CUPY_AVAILABLE and all(isinstance(tensor, cp.ndarray) for tensor in gpu_tensors):
                # Perform operation using CuPy
                result_gpu = self._tensor_operations_cupy(operation, *gpu_tensors, **kwargs)
            else:
                # Fallback to CPU
                return self._tensor_operations_cpu(operation, *tensors, **kwargs)
                
            # Transfer result to CPU
            result = self.gpu_manager.to_cpu(result_gpu)
            
            return result
        except Exception as e:
            print(f"Error in GPU tensor operation: {e}")
            
            # Fallback to CPU
            return self._tensor_operations_cpu(operation, *tensors, **kwargs)
            
    def _tensor_operations_cpu(self, operation: str, *tensors, **kwargs) -> np.ndarray:
        """
        Perform tensor operations on CPU.
        
        Args:
            operation (str): Operation to perform
            *tensors: Tensors to operate on
            **kwargs: Additional arguments for the operation
            
        Returns:
            np.ndarray: Result of the operation
        """
        if operation == 'add':
            return tensors[0] + tensors[1]
        elif operation == 'subtract':
            return tensors[0] - tensors[1]
        elif operation == 'multiply':
            return tensors[0] * tensors[1]
        elif operation == 'divide':
            return tensors[0] / tensors[1]
        elif operation == 'dot':
            return np.dot(tensors[0], tensors[1])
        elif operation == 'outer':
            return np.outer(tensors[0], tensors[1])
        else:
            raise ValueError(f"Unsupported operation: {operation}")
            
    def _tensor_operations_torch(self, operation: str, *tensors, **kwargs) -> torch.Tensor:
        """
        Perform tensor operations using PyTorch.
        
        Args:
            operation (str): Operation to perform
            *tensors: Tensors to operate on
            **kwargs: Additional arguments for the operation
            
        Returns:
            torch.Tensor: Result of the operation
        """
        if operation == 'add':
            return tensors[0] + tensors[1]
        elif operation == 'subtract':
            return tensors[0] - tensors[1]
        elif operation == 'multiply':
            return tensors[0] * tensors[1]
        elif operation == 'divide':
            return tensors[0] / tensors[1]
        elif operation == 'dot':
            return torch.matmul(tensors[0], tensors[1])
        elif operation == 'outer':
            return torch.outer(tensors[0], tensors[1])
        else:
            raise ValueError(f"Unsupported operation: {operation}")
            
    def _tensor_operations_cupy(self, operation: str, *tensors, **kwargs) -> cp.ndarray:
        """
        Perform tensor operations using CuPy.
        
        Args:
            operation (str): Operation to perform
            *tensors: Tensors to operate on
            **kwargs: Additional arguments for the operation
            
        Returns:
            cp.ndarray: Result of the operation
        """
        if operation == 'add':
            return tensors[0] + tensors[1]
        elif operation == 'subtract':
            return tensors[0] - tensors[1]
        elif operation == 'multiply':
            return tensors[0] * tensors[1]
        elif operation == 'divide':
            return tensors[0] / tensors[1]
        elif operation == 'dot':
            return cp.dot(tensors[0], tensors[1])
        elif operation == 'outer':
            return cp.outer(tensors[0], tensors[1])
        else:
            raise ValueError(f"Unsupported operation: {operation}")
            
    def neural_network_forward(self, inputs: np.ndarray, weights: List[np.ndarray], 
                              biases: List[np.ndarray], activation: str = 'relu') -> np.ndarray:
        """
        Perform neural network forward pass on GPU.
        
        Args:
            inputs (np.ndarray): Input data
            weights (List[np.ndarray]): List of weight matrices
            biases (List[np.ndarray]): List of bias vectors
            activation (str): Activation function ('relu', 'sigmoid', 'tanh')
            
        Returns:
            np.ndarray: Output of the neural network
        """
        if not self.gpu_available:
            # Perform forward pass on CPU
            return self._neural_network_forward_cpu(inputs, weights, biases, activation)
            
        try:
            # Transfer inputs, weights, and biases to GPU
            inputs_gpu = self.gpu_manager.to_gpu(inputs)
            weights_gpu = [self.gpu_manager.to_gpu(w) for w in weights]
            biases_gpu = [self.gpu_manager.to_gpu(b) for b in biases]
            
            if TORCH_AVAILABLE and isinstance(inputs_gpu, torch.Tensor):
                # Perform forward pass using PyTorch
                outputs_gpu = self._neural_network_forward_torch(inputs_gpu, weights_gpu, biases_gpu, activation)
            elif CUPY_AVAILABLE and isinstance(inputs_gpu, cp.ndarray):
                # Perform forward pass using CuPy
                outputs_gpu = self._neural_network_forward_cupy(inputs_gpu, weights_gpu, biases_gpu, activation)
            else:
                # Fallback to CPU
                return self._neural_network_forward_cpu(inputs, weights, biases, activation)
                
            # Transfer outputs to CPU
            outputs = self.gpu_manager.to_cpu(outputs_gpu)
            
            return outputs
        except Exception as e:
            print(f"Error in GPU neural network forward pass: {e}")
            
            # Fallback to CPU
            return self._neural_network_forward_cpu(inputs, weights, biases, activation)
            
    def _neural_network_forward_cpu(self, inputs: np.ndarray, weights: List[np.ndarray], 
                                   biases: List[np.ndarray], activation: str) -> np.ndarray:
        """
        Perform neural network forward pass on CPU.
        
        Args:
            inputs (np.ndarray): Input data
            weights (List[np.ndarray]): List of weight matrices
            biases (List[np.ndarray]): List of bias vectors
            activation (str): Activation function
            
        Returns:
            np.ndarray: Output of the neural network
        """
        # Initialize activations with inputs
        activations = inputs
        
        # Forward pass through each layer
        for i in range(len(weights)):
            # Linear transformation
            z = np.matmul(activations, weights[i]) + biases[i]
            
            # Apply activation function
            if i < len(weights) - 1:  # Apply activation to all but the last layer
                if activation == 'relu':
                    activations = np.maximum(0, z)
                elif activation == 'sigmoid':
                    activations = 1 / (1 + np.exp(-z))
                elif activation == 'tanh':
                    activations = np.tanh(z)
                else:
                    raise ValueError(f"Unsupported activation function: {activation}")
            else:
                # No activation for the output layer
                activations = z
                
        return activations
        
    def _neural_network_forward_torch(self, inputs: torch.Tensor, weights: List[torch.Tensor], 
                                     biases: List[torch.Tensor], activation: str) -> torch.Tensor:
        """
        Perform neural network forward pass using PyTorch.
        
        Args:
            inputs (torch.Tensor): Input data
            weights (List[torch.Tensor]): List of weight matrices
            biases (List[torch.Tensor]): List of bias vectors
            activation (str): Activation function
            
        Returns:
            torch.Tensor: Output of the neural network
        """
        # Initialize activations with inputs
        activations = inputs
        
        # Forward pass through each layer
        for i in range(len(weights)):
            # Linear transformation
            z = torch.matmul(activations, weights[i]) + biases[i]
            
            # Apply activation function
            if i < len(weights) - 1:  # Apply activation to all but the last layer
                if activation == 'relu':
                    activations = torch.relu(z)
                elif activation == 'sigmoid':
                    activations = torch.sigmoid(z)
                elif activation == 'tanh':
                    activations = torch.tanh(z)
                else:
                    raise ValueError(f"Unsupported activation function: {activation}")
            else:
                # No activation for the output layer
                activations = z
                
        return activations
        
    def _neural_network_forward_cupy(self, inputs: cp.ndarray, weights: List[cp.ndarray], 
                                    biases: List[cp.ndarray], activation: str) -> cp.ndarray:
        """
        Perform neural network forward pass using CuPy.
        
        Args:
            inputs (cp.ndarray): Input data
            weights (List[cp.ndarray]): List of weight matrices
            biases (List[cp.ndarray]): List of bias vectors
            activation (str): Activation function
            
        Returns:
            cp.ndarray: Output of the neural network
        """
        # Initialize activations with inputs
        activations = inputs
        
        # Forward pass through each layer
        for i in range(len(weights)):
            # Linear transformation
            z = cp.matmul(activations, weights[i]) + biases[i]
            
            # Apply activation function
            if i < len(weights) - 1:  # Apply activation to all but the last layer
                if activation == 'relu':
                    activations = cp.maximum(0, z)
                elif activation == 'sigmoid':
                    activations = 1 / (1 + cp.exp(-z))
                elif activation == 'tanh':
                    activations = cp.tanh(z)
                else:
                    raise ValueError(f"Unsupported activation function: {activation}")
            else:
                # No activation for the output layer
                activations = z
                
        return activations
        
    def clear_cache(self) -> None:
        """
        Clear GPU memory cache.
        """
        self.gpu_manager.clear_cache()
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get GPU statistics.
        
        Returns:
            Dict[str, Any]: GPU statistics
        """
        return self.gpu_manager.get_stats()


class GPUTensorOperations:
    """
    GPU-accelerated tensor operations for quantum chemistry.
    
    This class provides GPU-accelerated implementations of tensor operations
    commonly used in quantum chemistry calculations, such as tensor contractions,
    tensor decompositions, and tensor transformations.
    """
    
    def __init__(self, device_id: Optional[int] = None, memory_fraction: float = 0.8):
        """
        Initialize the GPUTensorOperations.
        
        Args:
            device_id (int, optional): GPU device ID. If None, uses the first available GPU.
            memory_fraction (float): Fraction of GPU memory to use (0.0 to 1.0)
        """
        # Create GPU accelerator
        self.gpu_accelerator = GPUAccelerator(device_id=device_id, memory_fraction=memory_fraction)
        
        # Check if GPU is available
        self.gpu_available = self.gpu_accelerator.gpu_available
        
    def tensor_contraction(self, tensor1: np.ndarray, tensor2: np.ndarray, 
                          axes1: Union[int, Tuple[int, ...]], 
                          axes2: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """
        Perform tensor contraction on GPU.
        
        Args:
            tensor1 (np.ndarray): First tensor
            tensor2 (np.ndarray): Second tensor
            axes1 (int or tuple): Axes of the first tensor to contract
            axes2 (int or tuple): Axes of the second tensor to contract
            
        Returns:
            np.ndarray: Result of tensor contraction
        """
        if not self.gpu_available:
            # Use numpy's tensordot for CPU
            return np.tensordot(tensor1, tensor2, axes=(axes1, axes2))
            
        try:
            # Transfer tensors to GPU
            tensor1_gpu = self.gpu_accelerator.gpu_manager.to_gpu(tensor1)
            tensor2_gpu = self.gpu_accelerator.gpu_manager.to_gpu(tensor2)
            
            if TORCH_AVAILABLE and isinstance(tensor1_gpu, torch.Tensor):
                # Use PyTorch's tensordot
                result_gpu = torch.tensordot(tensor1_gpu, tensor2_gpu, dims=([axes1], [axes2]))
            elif CUPY_AVAILABLE and isinstance(tensor1_gpu, cp.ndarray):
                # Use CuPy's tensordot
                result_gpu = cp.tensordot(tensor1_gpu, tensor2_gpu, axes=(axes1, axes2))
            else:
                # Fallback to CPU
                return np.tensordot(tensor1, tensor2, axes=(axes1, axes2))
                
            # Transfer result to CPU
            result = self.gpu_accelerator.gpu_manager.to_cpu(result_gpu)
            
            return result
        except Exception as e:
            print(f"Error in GPU tensor contraction: {e}")
            
            # Fallback to CPU
            return np.tensordot(tensor1, tensor2, axes=(axes1, axes2))
            
    def tensor_decomposition(self, tensor: np.ndarray, rank: int, method: str = 'cp') -> List[np.ndarray]:
        """
        Perform tensor decomposition on GPU.
        
        Args:
            tensor (np.ndarray): Tensor to decompose
            rank (int): Decomposition rank
            method (str): Decomposition method ('cp', 'tucker')
            
        Returns:
            List[np.ndarray]: Decomposed tensors
        """
        if not self.gpu_available:
            # Use CPU implementation
            return self._tensor_decomposition_cpu(tensor, rank, method)
            
        try:
            # Transfer tensor to GPU
            tensor_gpu = self.gpu_accelerator.gpu_manager.to_gpu(tensor)
            
            if TORCH_AVAILABLE and isinstance(tensor_gpu, torch.Tensor):
                # Use PyTorch for tensor decomposition
                result_gpu = self._tensor_decomposition_torch(tensor_gpu, rank, method)
            elif CUPY_AVAILABLE and isinstance(tensor_gpu, cp.ndarray):
                # Use CuPy for tensor decomposition
                result_gpu = self._tensor_decomposition_cupy(tensor_gpu, rank, method)
            else:
                # Fallback to CPU
                return self._tensor_decomposition_cpu(tensor, rank, method)
                
            # Transfer results to CPU
            result = [self.gpu_accelerator.gpu_manager.to_cpu(r) for r in result_gpu]
            
            return result
        except Exception as e:
            print(f"Error in GPU tensor decomposition: {e}")
            
            # Fallback to CPU
            return self._tensor_decomposition_cpu(tensor, rank, method)
            
    def _tensor_decomposition_cpu(self, tensor: np.ndarray, rank: int, method: str) -> List[np.ndarray]:
        """
        Perform tensor decomposition on CPU.
        
        Args:
            tensor (np.ndarray): Tensor to decompose
            rank (int): Decomposition rank
            method (str): Decomposition method
            
        Returns:
            List[np.ndarray]: Decomposed tensors
        """
        try:
            import tensorly as tl
            from tensorly.decomposition import parafac, tucker
            
            # Set backend to NumPy
            tl.set_backend('numpy')
            
            if method == 'cp':
                # CP decomposition (PARAFAC)
                factors = parafac(tensor, rank=rank, n_iter_max=100, tol=1e-6)
                return factors
            elif method == 'tucker':
                # Tucker decomposition
                core, factors = tucker(tensor, rank=rank, n_iter_max=100, tol=1e-6)
                return [core] + factors
            else:
                raise ValueError(f"Unsupported decomposition method: {method}")
        except ImportError:
            print("TensorLy not available, using simple SVD-based decomposition")
            
            # Simple SVD-based decomposition
            if method == 'cp':
                # Reshape tensor to matrix
                tensor_shape = tensor.shape
                tensor_matrix = tensor.reshape(tensor_shape[0], -1)
                
                # Perform SVD
                U, S, Vt = np.linalg.svd(tensor_matrix, full_matrices=False)
                
                # Truncate to rank
                U = U[:, :rank]
                S = S[:rank]
                Vt = Vt[:rank, :]
                
                # Reshape V back to tensor
                V = Vt.reshape(rank, *tensor_shape[1:])
                
                return [U, np.diag(S), V]
            else:
                raise ValueError(f"Unsupported decomposition method without TensorLy: {method}")
                
    def _tensor_decomposition_torch(self, tensor: torch.Tensor, rank: int, method: str) -> List[torch.Tensor]:
        """
        Perform tensor decomposition using PyTorch.
        
        Args:
            tensor (torch.Tensor): Tensor to decompose
            rank (int): Decomposition rank
            method (str): Decomposition method
            
        Returns:
            List[torch.Tensor]: Decomposed tensors
        """
        try:
            import tensorly as tl
            from tensorly.decomposition import parafac, tucker
            
            # Convert tensor to NumPy
            tensor_np = self.gpu_accelerator.gpu_manager.to_cpu(tensor)
            
            # Set backend to NumPy
            tl.set_backend('numpy')
            
            if method == 'cp':
                # CP decomposition (PARAFAC)
                factors = parafac(tensor_np, rank=rank, n_iter_max=100, tol=1e-6)
                
                # Convert factors to PyTorch tensors
                factors_torch = [self.gpu_accelerator.gpu_manager.to_gpu(f) for f in factors]
                
                return factors_torch
            elif method == 'tucker':
                # Tucker decomposition
                core, factors = tucker(tensor_np, rank=rank, n_iter_max=100, tol=1e-6)
                
                # Convert core and factors to PyTorch tensors
                core_torch = self.gpu_accelerator.gpu_manager.to_gpu(core)
                factors_torch = [self.gpu_accelerator.gpu_manager.to_gpu(f) for f in factors]
                
                return [core_torch] + factors_torch
            else:
                raise ValueError(f"Unsupported decomposition method: {method}")
        except ImportError:
            print("TensorLy not available, using simple SVD-based decomposition")
            
            # Simple SVD-based decomposition
            if method == 'cp':
                # Reshape tensor to matrix
                tensor_shape = tensor.shape
                tensor_matrix = tensor.reshape(tensor_shape[0], -1)
                
                # Perform SVD
                U, S, V = torch.svd(tensor_matrix)
                
                # Truncate to rank
                U = U[:, :rank]
                S = S[:rank]
                V = V[:, :rank]
                
                # Reshape V back to tensor
                V = V.reshape(rank, *tensor_shape[1:])
                
                return [U, torch.diag(S), V]
            else:
                raise ValueError(f"Unsupported decomposition method without TensorLy: {method}")
                
    def _tensor_decomposition_cupy(self, tensor: cp.ndarray, rank: int, method: str) -> List[cp.ndarray]:
        """
        Perform tensor decomposition using CuPy.
        
        Args:
            tensor (cp.ndarray): Tensor to decompose
            rank (int): Decomposition rank
            method (str): Decomposition method
            
        Returns:
            List[cp.ndarray]: Decomposed tensors
        """
        # Convert tensor to NumPy for decomposition
        tensor_np = self.gpu_accelerator.gpu_manager.to_cpu(tensor)
        
        # Perform decomposition on CPU
        result = self._tensor_decomposition_cpu(tensor_np, rank, method)
        
        # Convert result back to CuPy
        result_cp = [self.gpu_accelerator.gpu_manager.to_gpu(r) for r in result]
        
        return result_cp
        
    def tensor_transformation(self, tensor: np.ndarray, transformation: str, **kwargs) -> np.ndarray:
        """
        Perform tensor transformation on GPU.
        
        Args:
            tensor (np.ndarray): Tensor to transform
            transformation (str): Transformation to apply ('transpose', 'reshape', 'permute')
            **kwargs: Additional arguments for the transformation
            
        Returns:
            np.ndarray: Transformed tensor
        """
        if not self.gpu_available:
            # Use CPU implementation
            return self._tensor_transformation_cpu(tensor, transformation, **kwargs)
            
        try:
            # Transfer tensor to GPU
            tensor_gpu = self.gpu_accelerator.gpu_manager.to_gpu(tensor)
            
            if TORCH_AVAILABLE and isinstance(tensor_gpu, torch.Tensor):
                # Use PyTorch for tensor transformation
                result_gpu = self._tensor_transformation_torch(tensor_gpu, transformation, **kwargs)
            elif CUPY_AVAILABLE and isinstance(tensor_gpu, cp.ndarray):
                # Use CuPy for tensor transformation
                result_gpu = self._tensor_transformation_cupy(tensor_gpu, transformation, **kwargs)
            else:
                # Fallback to CPU
                return self._tensor_transformation_cpu(tensor, transformation, **kwargs)
                
            # Transfer result to CPU
            result = self.gpu_accelerator.gpu_manager.to_cpu(result_gpu)
            
            return result
        except Exception as e:
            print(f"Error in GPU tensor transformation: {e}")
            
            # Fallback to CPU
            return self._tensor_transformation_cpu(tensor, transformation, **kwargs)
            
    def _tensor_transformation_cpu(self, tensor: np.ndarray, transformation: str, **kwargs) -> np.ndarray:
        """
        Perform tensor transformation on CPU.
        
        Args:
            tensor (np.ndarray): Tensor to transform
            transformation (str): Transformation to apply
            **kwargs: Additional arguments for the transformation
            
        Returns:
            np.ndarray: Transformed tensor
        """
        if transformation == 'transpose':
            axes = kwargs.get('axes', None)
            if axes is None:
                return tensor.T
            else:
                return np.transpose(tensor, axes=axes)
        elif transformation == 'reshape':
            shape = kwargs.get('shape')
            if shape is None:
                raise ValueError("Shape must be provided for reshape transformation")
            return tensor.reshape(shape)
        elif transformation == 'permute':
            axes = kwargs.get('axes')
            if axes is None:
                raise ValueError("Axes must be provided for permute transformation")
            return np.transpose(tensor, axes=axes)
        else:
            raise ValueError(f"Unsupported transformation: {transformation}")
            
    def _tensor_transformation_torch(self, tensor: torch.Tensor, transformation: str, **kwargs) -> torch.Tensor:
        """
        Perform tensor transformation using PyTorch.
        
        Args:
            tensor (torch.Tensor): Tensor to transform
            transformation (str): Transformation to apply
            **kwargs: Additional arguments for the transformation
            
        Returns:
            torch.Tensor: Transformed tensor
        """
        if transformation == 'transpose':
            axes = kwargs.get('axes', None)
            if axes is None:
                return tensor.t()
            else:
                return tensor.permute(*axes)
        elif transformation == 'reshape':
            shape = kwargs.get('shape')
            if shape is None:
                raise ValueError("Shape must be provided for reshape transformation")
            return tensor.reshape(*shape)
        elif transformation == 'permute':
            axes = kwargs.get('axes')
            if axes is None:
                raise ValueError("Axes must be provided for permute transformation")
            return tensor.permute(*axes)
        else:
            raise ValueError(f"Unsupported transformation: {transformation}")
            
    def _tensor_transformation_cupy(self, tensor: cp.ndarray, transformation: str, **kwargs) -> cp.ndarray:
        """
        Perform tensor transformation using CuPy.
        
        Args:
            tensor (cp.ndarray): Tensor to transform
            transformation (str): Transformation to apply
            **kwargs: Additional arguments for the transformation
            
        Returns:
            cp.ndarray: Transformed tensor
        """
        if transformation == 'transpose':
            axes = kwargs.get('axes', None)
            if axes is None:
                return tensor.T
            else:
                return cp.transpose(tensor, axes=axes)
        elif transformation == 'reshape':
            shape = kwargs.get('shape')
            if shape is None:
                raise ValueError("Shape must be provided for reshape transformation")
            return tensor.reshape(shape)
        elif transformation == 'permute':
            axes = kwargs.get('axes')
            if axes is None:
                raise ValueError("Axes must be provided for permute transformation")
            return cp.transpose(tensor, axes=axes)
        else:
            raise ValueError(f"Unsupported transformation: {transformation}")
            
    def clear_cache(self) -> None:
        """
        Clear GPU memory cache.
        """
        self.gpu_accelerator.clear_cache()
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get GPU statistics.
        
        Returns:
            Dict[str, Any]: GPU statistics
        """
        return self.gpu_accelerator.get_stats()