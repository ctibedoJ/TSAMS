"""
Parallel Processing Module for TIBEDO Framework

This module provides comprehensive parallel processing capabilities for the TIBEDO Framework,
enabling efficient computation for large-scale quantum chemistry problems.
"""

import numpy as np
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import time
import threading
import queue
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, TypeVar, Generic, Iterable

# Type variables for generic functions
T = TypeVar('T')
U = TypeVar('U')

class TaskExecutor:
    """
    General-purpose parallel task executor for the TIBEDO Framework.
    
    This class provides a flexible interface for executing tasks in parallel,
    with support for both process-based and thread-based parallelism.
    """
    
    def __init__(self, 
                 num_workers: Optional[int] = None, 
                 executor_type: str = 'process',
                 max_queue_size: int = 1000):
        """
        Initialize the TaskExecutor.
        
        Args:
            num_workers (int, optional): Number of worker processes/threads.
                If None, uses the number of CPU cores.
            executor_type (str): Type of executor to use ('process' or 'thread').
            max_queue_size (int): Maximum size of the task queue.
        """
        # Set number of workers
        if num_workers is None:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = num_workers
            
        # Set executor type
        self.executor_type = executor_type
        
        # Set maximum queue size
        self.max_queue_size = max_queue_size
        
        # Create task queue
        self.task_queue = queue.Queue(maxsize=max_queue_size)
        
        # Create result dictionary
        self.results = {}
        
        # Create lock for thread safety
        self.lock = threading.Lock()
        
        # Create executor
        self.executor = None
        
        # Create stop event
        self.stop_event = threading.Event()
        
        # Create worker threads/processes
        self.workers = []
        
        # Initialize statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0
        }
        
    def start(self) -> None:
        """
        Start the task executor.
        """
        # Create executor based on type
        if self.executor_type == 'process':
            self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        elif self.executor_type == 'thread':
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        else:
            raise ValueError(f"Invalid executor type: {self.executor_type}")
            
        # Reset stop event
        self.stop_event.clear()
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        print(f"Started {self.executor_type} executor with {self.num_workers} workers")
        
    def _process_queue(self) -> None:
        """
        Process tasks from the queue.
        """
        while not self.stop_event.is_set():
            try:
                # Get task from queue with timeout
                task_id, func, args, kwargs = self.task_queue.get(timeout=0.1)
                
                # Submit task to executor
                future = self.executor.submit(func, *args, **kwargs)
                
                # Add callback to handle result
                future.add_done_callback(lambda f, tid=task_id: self._handle_result(tid, f))
                
            except queue.Empty:
                # Queue is empty, continue
                continue
            except Exception as e:
                # Log error
                print(f"Error processing task: {e}")
                
    def _handle_result(self, task_id: str, future) -> None:
        """
        Handle the result of a task.
        
        Args:
            task_id (str): Task ID
            future: Future object
        """
        with self.lock:
            try:
                # Get result from future
                result = future.result()
                
                # Store result
                self.results[task_id] = {
                    'status': 'completed',
                    'result': result,
                    'error': None
                }
                
                # Update statistics
                self.stats['tasks_completed'] += 1
                
            except Exception as e:
                # Store error
                self.results[task_id] = {
                    'status': 'failed',
                    'result': None,
                    'error': str(e)
                }
                
                # Update statistics
                self.stats['tasks_failed'] += 1
                
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """
        Submit a task for execution.
        
        Args:
            func (Callable): Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            str: Task ID
        """
        # Generate task ID
        task_id = f"task_{time.time()}_{self.stats['tasks_submitted']}"
        
        # Add task to queue
        self.task_queue.put((task_id, func, args, kwargs))
        
        # Initialize result entry
        with self.lock:
            self.results[task_id] = {
                'status': 'pending',
                'result': None,
                'error': None
            }
            
            # Update statistics
            self.stats['tasks_submitted'] += 1
            
        return task_id
        
    def map(self, func: Callable[[T], U], items: Iterable[T]) -> List[U]:
        """
        Apply a function to each item in parallel.
        
        Args:
            func (Callable): Function to apply
            items (Iterable): Items to process
            
        Returns:
            List: Results
        """
        # Submit tasks
        task_ids = [self.submit_task(func, item) for item in items]
        
        # Wait for results
        results = [self.get_result(task_id) for task_id in task_ids]
        
        return results
        
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get the result of a task.
        
        Args:
            task_id (str): Task ID
            timeout (float, optional): Timeout in seconds
            
        Returns:
            Any: Task result
            
        Raises:
            TimeoutError: If the task does not complete within the timeout
            RuntimeError: If the task fails
        """
        start_time = time.time()
        
        while True:
            # Check if timeout has been exceeded
            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
                
            # Get result status
            with self.lock:
                if task_id not in self.results:
                    raise ValueError(f"Task {task_id} not found")
                    
                status = self.results[task_id]['status']
                
                if status == 'completed':
                    return self.results[task_id]['result']
                elif status == 'failed':
                    raise RuntimeError(f"Task {task_id} failed: {self.results[task_id]['error']}")
                    
            # Wait before checking again
            time.sleep(0.01)
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dict[str, Any]: Execution statistics
        """
        with self.lock:
            return self.stats.copy()
            
    def stop(self) -> None:
        """
        Stop the task executor.
        """
        # Set stop event
        self.stop_event.set()
        
        # Wait for worker thread to complete
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
            
        # Shutdown executor
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            
        print(f"Stopped {self.executor_type} executor")


class ParallelMatrixOperations:
    """
    Parallel implementation of common matrix operations used in quantum chemistry.
    
    This class provides parallel implementations of matrix operations that are
    commonly used in quantum chemistry calculations, such as matrix multiplication,
    eigenvalue decomposition, and singular value decomposition.
    """
    
    def __init__(self, num_workers: Optional[int] = None):
        """
        Initialize the ParallelMatrixOperations.
        
        Args:
            num_workers (int, optional): Number of worker processes.
                If None, uses the number of CPU cores.
        """
        # Set number of workers
        if num_workers is None:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = num_workers
            
        # Create task executor
        self.executor = TaskExecutor(num_workers=self.num_workers, executor_type='process')
        self.executor.start()
        
    def _split_matrix(self, matrix: np.ndarray, axis: int = 0) -> List[np.ndarray]:
        """
        Split a matrix into chunks for parallel processing.
        
        Args:
            matrix (np.ndarray): Matrix to split
            axis (int): Axis along which to split the matrix
            
        Returns:
            List[np.ndarray]: List of matrix chunks
        """
        # Calculate chunk size
        chunk_size = max(1, matrix.shape[axis] // self.num_workers)
        
        # Split matrix
        chunks = []
        for i in range(0, matrix.shape[axis], chunk_size):
            end = min(i + chunk_size, matrix.shape[axis])
            if axis == 0:
                chunks.append(matrix[i:end, :])
            else:
                chunks.append(matrix[:, i:end])
                
        return chunks
        
    def _process_matrix_chunk(self, func: Callable, chunk: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Process a matrix chunk.
        
        Args:
            func (Callable): Function to apply to the chunk
            chunk (np.ndarray): Matrix chunk
            *args: Additional positional arguments for the function
            **kwargs: Additional keyword arguments for the function
            
        Returns:
            np.ndarray: Processed chunk
        """
        return func(chunk, *args, **kwargs)
        
    def apply(self, func: Callable, matrix: np.ndarray, axis: int = 0, *args, **kwargs) -> np.ndarray:
        """
        Apply a function to a matrix in parallel.
        
        Args:
            func (Callable): Function to apply
            matrix (np.ndarray): Matrix to process
            axis (int): Axis along which to split the matrix
            *args: Additional positional arguments for the function
            **kwargs: Additional keyword arguments for the function
            
        Returns:
            np.ndarray: Processed matrix
        """
        # Split matrix into chunks
        chunks = self._split_matrix(matrix, axis=axis)
        
        # Process chunks in parallel
        processed_chunks = self.executor.map(
            lambda chunk: self._process_matrix_chunk(func, chunk, *args, **kwargs),
            chunks
        )
        
        # Combine processed chunks
        if axis == 0:
            return np.vstack(processed_chunks)
        else:
            return np.hstack(processed_chunks)
            
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Perform matrix multiplication in parallel.
        
        Args:
            A (np.ndarray): First matrix
            B (np.ndarray): Second matrix
            
        Returns:
            np.ndarray: Result of matrix multiplication
        """
        # Check if matrices can be multiplied
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Matrix shapes incompatible for multiplication: {A.shape} and {B.shape}")
            
        # Split first matrix into chunks
        A_chunks = self._split_matrix(A, axis=0)
        
        # Multiply chunks in parallel
        result_chunks = self.executor.map(
            lambda chunk: np.matmul(chunk, B),
            A_chunks
        )
        
        # Combine result chunks
        return np.vstack(result_chunks)
        
    def eigendecomposition(self, matrix: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform eigenvalue decomposition in parallel.
        
        Args:
            matrix (np.ndarray): Matrix to decompose
            k (int, optional): Number of eigenvalues/eigenvectors to compute
                If None, computes all eigenvalues/eigenvectors.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Eigenvalues and eigenvectors
        """
        # Check if matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matrix must be square for eigendecomposition: {matrix.shape}")
            
        # If k is None, compute all eigenvalues/eigenvectors
        if k is None:
            k = matrix.shape[0]
            
        # Use scipy's sparse eigenvalue solver for large matrices
        if matrix.shape[0] > 1000:
            from scipy.sparse.linalg import eigsh
            return eigsh(matrix, k=k)
        else:
            # Use numpy's eigenvalue solver for small matrices
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            
            # Sort eigenvalues and eigenvectors
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Return only k eigenvalues/eigenvectors
            return eigenvalues[:k], eigenvectors[:, :k]
            
    def svd(self, matrix: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform singular value decomposition in parallel.
        
        Args:
            matrix (np.ndarray): Matrix to decompose
            k (int, optional): Number of singular values/vectors to compute
                If None, computes all singular values/vectors.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: U, S, V matrices
        """
        # If k is None, compute all singular values/vectors
        if k is None:
            k = min(matrix.shape)
            
        # Use scipy's sparse SVD for large matrices
        if matrix.shape[0] > 1000 or matrix.shape[1] > 1000:
            from scipy.sparse.linalg import svds
            U, S, Vt = svds(matrix, k=k)
            
            # Sort singular values and vectors
            idx = S.argsort()[::-1]
            S = S[idx]
            U = U[:, idx]
            Vt = Vt[idx, :]
            
            return U, S, Vt.T
        else:
            # Use numpy's SVD for small matrices
            U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
            
            # Return only k singular values/vectors
            return U[:, :k], S[:k], Vt[:k, :].T
            
    def stop(self) -> None:
        """
        Stop the parallel matrix operations.
        """
        self.executor.stop()


class ParallelAlgorithms:
    """
    Parallel implementations of core TIBEDO algorithms.
    
    This class provides parallel implementations of the core algorithms used in
    the TIBEDO Framework, such as the TSC algorithm, spinor reduction chain, and
    prime-indexed congruential relations.
    """
    
    def __init__(self, num_workers: Optional[int] = None):
        """
        Initialize the ParallelAlgorithms.
        
        Args:
            num_workers (int, optional): Number of worker processes.
                If None, uses the number of CPU cores.
        """
        # Set number of workers
        if num_workers is None:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = num_workers
            
        # Create task executor
        self.executor = TaskExecutor(num_workers=self.num_workers, executor_type='process')
        self.executor.start()
        
        # Create matrix operations
        self.matrix_ops = ParallelMatrixOperations(num_workers=self.num_workers)
        
    def parallel_tsc(self, throw_params: Dict[str, Any], shot_params: Dict[str, Any], 
                    catch_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parallel implementation of the TSC algorithm.
        
        Args:
            throw_params (Dict[str, Any]): Parameters for the throw phase
            shot_params (Dict[str, Any]): Parameters for the shot phase
            catch_params (Dict[str, Any]): Parameters for the catch phase
            
        Returns:
            Dict[str, Any]: Results of the TSC algorithm
        """
        # Import TSC components
        try:
            from tibedo.core.tsc.throw import ThrowPhase
            from tibedo.core.tsc.shot import ShotPhase
            from tibedo.core.tsc.catch import CatchPhase
        except ImportError:
            raise ImportError("TSC components not found. Make sure the TIBEDO core module is installed.")
            
        # Create phases
        throw_phase = ThrowPhase()
        shot_phase = ShotPhase()
        catch_phase = CatchPhase()
        
        # Execute throw phase
        throw_result = throw_phase.execute(**throw_params)
        
        # Split shot phase work
        shot_chunks = self._split_shot_work(throw_result, shot_params)
        
        # Execute shot phase in parallel
        shot_results = self.executor.map(
            lambda chunk: shot_phase.execute(**chunk),
            shot_chunks
        )
        
        # Combine shot results
        combined_shot_result = self._combine_shot_results(shot_results)
        
        # Execute catch phase
        catch_result = catch_phase.execute(shot_result=combined_shot_result, **catch_params)
        
        return {
            'throw_result': throw_result,
            'shot_result': combined_shot_result,
            'catch_result': catch_result
        }
        
    def _split_shot_work(self, throw_result: Dict[str, Any], shot_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split the shot phase work into chunks for parallel processing.
        
        Args:
            throw_result (Dict[str, Any]): Result of the throw phase
            shot_params (Dict[str, Any]): Parameters for the shot phase
            
        Returns:
            List[Dict[str, Any]]: List of shot phase parameter chunks
        """
        # Get the spinor space from throw result
        spinor_space = throw_result.get('spinor_space')
        
        if spinor_space is None:
            raise ValueError("Throw result does not contain spinor_space")
            
        # Split spinor space into chunks
        if isinstance(spinor_space, np.ndarray):
            spinor_chunks = np.array_split(spinor_space, self.num_workers)
        elif isinstance(spinor_space, list):
            chunk_size = max(1, len(spinor_space) // self.num_workers)
            spinor_chunks = [spinor_space[i:i+chunk_size] for i in range(0, len(spinor_space), chunk_size)]
        else:
            raise ValueError(f"Unsupported spinor_space type: {type(spinor_space)}")
            
        # Create parameter chunks
        chunks = []
        for i, spinor_chunk in enumerate(spinor_chunks):
            # Create a copy of shot parameters
            chunk_params = shot_params.copy()
            
            # Replace spinor space with chunk
            chunk_params['spinor_space'] = spinor_chunk
            
            # Add chunk index
            chunk_params['chunk_index'] = i
            
            # Add throw result (excluding spinor space to save memory)
            throw_result_copy = throw_result.copy()
            throw_result_copy.pop('spinor_space', None)
            chunk_params['throw_result'] = throw_result_copy
            
            chunks.append(chunk_params)
            
        return chunks
        
    def _combine_shot_results(self, shot_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine the results of parallel shot phase execution.
        
        Args:
            shot_results (List[Dict[str, Any]]): Results of parallel shot phase execution
            
        Returns:
            Dict[str, Any]: Combined shot phase result
        """
        # Initialize combined result
        combined_result = {}
        
        # Combine results based on type
        for key in shot_results[0].keys():
            if key == 'chunk_index':
                continue
                
            values = [result[key] for result in shot_results]
            
            if isinstance(values[0], np.ndarray):
                # Combine arrays
                combined_result[key] = np.concatenate(values)
            elif isinstance(values[0], list):
                # Combine lists
                combined_result[key] = [item for sublist in values for item in sublist]
            elif isinstance(values[0], dict):
                # Combine dictionaries
                combined_result[key] = {}
                for value_dict in values:
                    combined_result[key].update(value_dict)
            else:
                # Use the first value for scalars
                combined_result[key] = values[0]
                
        return combined_result
        
    def parallel_spinor_reduction(self, spinor_space: np.ndarray, reduction_maps: List[np.ndarray]) -> np.ndarray:
        """
        Parallel implementation of spinor reduction chain.
        
        Args:
            spinor_space (np.ndarray): Spinor space to reduce
            reduction_maps (List[np.ndarray]): List of reduction maps
            
        Returns:
            np.ndarray: Reduced spinor space
        """
        # Split spinor space into chunks
        spinor_chunks = np.array_split(spinor_space, self.num_workers)
        
        # Apply reduction maps to each chunk in parallel
        reduced_chunks = self.executor.map(
            lambda chunk: self._apply_reduction_maps(chunk, reduction_maps),
            spinor_chunks
        )
        
        # Combine reduced chunks
        return np.concatenate(reduced_chunks)
        
    def _apply_reduction_maps(self, spinor_chunk: np.ndarray, reduction_maps: List[np.ndarray]) -> np.ndarray:
        """
        Apply a sequence of reduction maps to a spinor chunk.
        
        Args:
            spinor_chunk (np.ndarray): Spinor chunk to reduce
            reduction_maps (List[np.ndarray]): List of reduction maps
            
        Returns:
            np.ndarray: Reduced spinor chunk
        """
        reduced_spinor = spinor_chunk
        
        for reduction_map in reduction_maps:
            reduced_spinor = np.matmul(reduced_spinor, reduction_map)
            
        return reduced_spinor
        
    def parallel_prime_indexed(self, values: np.ndarray, prime_indices: List[int]) -> Dict[int, np.ndarray]:
        """
        Parallel implementation of prime-indexed congruential relations.
        
        Args:
            values (np.ndarray): Values to process
            prime_indices (List[int]): List of prime indices
            
        Returns:
            Dict[int, np.ndarray]: Dictionary mapping prime indices to values
        """
        # Split prime indices into chunks
        prime_chunks = np.array_split(prime_indices, self.num_workers)
        
        # Process each chunk in parallel
        result_chunks = self.executor.map(
            lambda chunk: self._process_prime_chunk(values, chunk),
            prime_chunks
        )
        
        # Combine results
        combined_result = {}
        for chunk_result in result_chunks:
            combined_result.update(chunk_result)
            
        return combined_result
        
    def _process_prime_chunk(self, values: np.ndarray, prime_indices: List[int]) -> Dict[int, np.ndarray]:
        """
        Process a chunk of prime indices.
        
        Args:
            values (np.ndarray): Values to process
            prime_indices (List[int]): Chunk of prime indices
            
        Returns:
            Dict[int, np.ndarray]: Dictionary mapping prime indices to values
        """
        result = {}
        
        for prime in prime_indices:
            # Calculate congruential values
            congruential_values = values % prime
            
            # Store result
            result[prime] = congruential_values
            
        return result
        
    def stop(self) -> None:
        """
        Stop the parallel algorithms.
        """
        self.executor.stop()
        self.matrix_ops.stop()


class LoadBalancer:
    """
    Load balancer for distributing work across multiple processes.
    
    This class provides strategies for distributing work across multiple processes,
    taking into account the computational complexity of different tasks and the
    available resources.
    """
    
    def __init__(self, num_workers: Optional[int] = None):
        """
        Initialize the LoadBalancer.
        
        Args:
            num_workers (int, optional): Number of worker processes.
                If None, uses the number of CPU cores.
        """
        # Set number of workers
        if num_workers is None:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = num_workers
            
        # Initialize worker loads
        self.worker_loads = [0] * self.num_workers
        
        # Create lock for thread safety
        self.lock = threading.Lock()
        
    def get_worker_for_task(self, task_complexity: float = 1.0) -> int:
        """
        Get the worker with the lowest load for a task.
        
        Args:
            task_complexity (float): Complexity of the task
            
        Returns:
            int: Worker index
        """
        with self.lock:
            # Find worker with lowest load
            min_load = float('inf')
            min_worker = 0
            
            for i, load in enumerate(self.worker_loads):
                if load < min_load:
                    min_load = load
                    min_worker = i
                    
            # Update worker load
            self.worker_loads[min_worker] += task_complexity
            
            return min_worker
            
    def task_completed(self, worker_index: int, task_complexity: float = 1.0) -> None:
        """
        Notify that a task has been completed.
        
        Args:
            worker_index (int): Worker index
            task_complexity (float): Complexity of the task
        """
        with self.lock:
            # Update worker load
            self.worker_loads[worker_index] -= task_complexity
            
            # Ensure load is not negative
            self.worker_loads[worker_index] = max(0, self.worker_loads[worker_index])
            
    def get_worker_loads(self) -> List[float]:
        """
        Get the current worker loads.
        
        Returns:
            List[float]: Worker loads
        """
        with self.lock:
            return self.worker_loads.copy()
            
    def reset(self) -> None:
        """
        Reset the load balancer.
        """
        with self.lock:
            self.worker_loads = [0] * self.num_workers