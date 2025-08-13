"""
Parallel Processing implementation.

This module provides implementations of parallel processing techniques for
computationally intensive calculations in the cyclotomic field theory framework.
"""

import numpy as np
import multiprocessing as mp
from typing import List, Dict, Tuple, Union, Optional, Callable, Any
from functools import partial
import time
import os
from ..core.cyclotomic_field import CyclotomicField
from ..core.prime_spectral_grouping import PrimeSpectralGrouping
from ..cosmology.prime_distribution import PrimeDistribution


class ParallelPrimeDistribution:
    """
    A class for parallel computation of prime distribution calculations.
    
    This class provides methods to accelerate the computation of prime distribution
    formulas using parallel processing techniques.
    
    Attributes:
        cyclotomic_field (CyclotomicField): The cyclotomic field.
        prime_spectral_grouping (PrimeSpectralGrouping): The prime spectral grouping.
        prime_distribution (PrimeDistribution): The prime distribution.
        num_processes (int): The number of processes to use for parallel computation.
        chunk_size (int): The chunk size for parallel computation.
    """
    
    def __init__(self, cyclotomic_field: CyclotomicField, num_processes: Optional[int] = None, chunk_size: int = 1000):
        """
        Initialize a parallel prime distribution calculator.
        
        Args:
            cyclotomic_field (CyclotomicField): The cyclotomic field.
            num_processes (Optional[int]): The number of processes to use (default: number of CPU cores).
            chunk_size (int): The chunk size for parallel computation (default: 1000).
        
        Raises:
            ValueError: If the chunk size is not positive.
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        self.cyclotomic_field = cyclotomic_field
        self.prime_spectral_grouping = PrimeSpectralGrouping()
        self.prime_distribution = PrimeDistribution(cyclotomic_field)
        self.num_processes = num_processes if num_processes is not None else mp.cpu_count()
        self.chunk_size = chunk_size
    
    def set_num_processes(self, num_processes: int):
        """
        Set the number of processes to use for parallel computation.
        
        Args:
            num_processes (int): The number of processes.
        
        Raises:
            ValueError: If the number of processes is not positive.
        """
        if num_processes <= 0:
            raise ValueError("Number of processes must be positive")
        
        self.num_processes = num_processes
    
    def set_chunk_size(self, chunk_size: int):
        """
        Set the chunk size for parallel computation.
        
        Args:
            chunk_size (int): The chunk size.
        
        Raises:
            ValueError: If the chunk size is not positive.
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        self.chunk_size = chunk_size
    
    def compute_prime_counts_parallel(self, x_values: np.ndarray) -> np.ndarray:
        """
        Compute the prime counting function for multiple values in parallel.
        
        Args:
            x_values (np.ndarray): The values at which to compute the prime counting function.
        
        Returns:
            np.ndarray: The prime counts.
        """
        # Split the values into chunks
        chunks = [x_values[i:i+self.chunk_size] for i in range(0, len(x_values), self.chunk_size)]
        
        # Create a pool of processes
        with mp.Pool(processes=self.num_processes) as pool:
            # Compute the prime counts for each chunk in parallel
            results = pool.map(self._compute_prime_counts_chunk, chunks)
        
        # Combine the results
        prime_counts = np.concatenate(results)
        
        return prime_counts
    
    def _compute_prime_counts_chunk(self, x_chunk: np.ndarray) -> np.ndarray:
        """
        Compute the prime counting function for a chunk of values.
        
        Args:
            x_chunk (np.ndarray): The chunk of values.
        
        Returns:
            np.ndarray: The prime counts.
        """
        return np.array([self.prime_distribution.prime_counting_function(x) for x in x_chunk])
    
    def compute_prime_distribution_formula_parallel(self, x_values: np.ndarray) -> np.ndarray:
        """
        Compute the prime distribution formula for multiple values in parallel.
        
        Args:
            x_values (np.ndarray): The values at which to compute the formula.
        
        Returns:
            np.ndarray: The formula values.
        """
        # Split the values into chunks
        chunks = [x_values[i:i+self.chunk_size] for i in range(0, len(x_values), self.chunk_size)]
        
        # Create a pool of processes
        with mp.Pool(processes=self.num_processes) as pool:
            # Compute the formula values for each chunk in parallel
            results = pool.map(self._compute_prime_distribution_formula_chunk, chunks)
        
        # Combine the results
        formula_values = np.concatenate(results)
        
        return formula_values
    
    def _compute_prime_distribution_formula_chunk(self, x_chunk: np.ndarray) -> np.ndarray:
        """
        Compute the prime distribution formula for a chunk of values.
        
        Args:
            x_chunk (np.ndarray): The chunk of values.
        
        Returns:
            np.ndarray: The formula values.
        """
        return np.array([self.prime_distribution.prime_distribution_formula(x) for x in x_chunk])
    
    def compute_prime_spectral_groupings_parallel(self, prime_groups: List[List[int]]) -> Dict[Tuple[int, ...], float]:
        """
        Compute multiple prime spectral groupings in parallel.
        
        Args:
            prime_groups (List[List[int]]): The groups of primes.
        
        Returns:
            Dict[Tuple[int, ...], float]: The spectral groupings.
        """
        # Create a pool of processes
        with mp.Pool(processes=self.num_processes) as pool:
            # Compute the spectral groupings in parallel
            results = pool.map(self._compute_prime_spectral_grouping, prime_groups)
        
        # Combine the results
        groupings = {tuple(group): value for group, value in zip(prime_groups, results)}
        
        return groupings
    
    def _compute_prime_spectral_grouping(self, prime_group: List[int]) -> float:
        """
        Compute a prime spectral grouping.
        
        Args:
            prime_group (List[int]): The group of primes.
        
        Returns:
            float: The spectral grouping value.
        """
        return self.prime_spectral_grouping.get_group(prime_group)
    
    def benchmark_parallel_vs_serial(self, x_values: np.ndarray, num_runs: int = 3) -> Dict[str, float]:
        """
        Benchmark parallel computation against serial computation.
        
        Args:
            x_values (np.ndarray): The values at which to compute the prime counting function.
            num_runs (int): The number of runs for each computation.
        
        Returns:
            Dict[str, float]: The benchmark results.
        """
        # Benchmark serial computation
        serial_times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = np.array([self.prime_distribution.prime_counting_function(x) for x in x_values])
            end_time = time.time()
            serial_times.append(end_time - start_time)
        
        serial_time = np.mean(serial_times)
        
        # Benchmark parallel computation
        parallel_times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.compute_prime_counts_parallel(x_values)
            end_time = time.time()
            parallel_times.append(end_time - start_time)
        
        parallel_time = np.mean(parallel_times)
        
        # Compute the speedup
        speedup = serial_time / parallel_time if parallel_time > 0 else float('inf')
        
        # Compute the efficiency
        efficiency = speedup / self.num_processes
        
        return {
            "serial_time": serial_time,
            "parallel_time": parallel_time,
            "speedup": speedup,
            "efficiency": efficiency,
            "num_processes": self.num_processes
        }
    
    def optimize_chunk_size(self, x_values: np.ndarray, chunk_sizes: List[int], num_runs: int = 3) -> Dict[int, float]:
        """
        Optimize the chunk size for parallel computation.
        
        Args:
            x_values (np.ndarray): The values at which to compute the prime counting function.
            chunk_sizes (List[int]): The chunk sizes to test.
            num_runs (int): The number of runs for each chunk size.
        
        Returns:
            Dict[int, float]: The execution times for each chunk size.
        """
        results = {}
        
        for chunk_size in chunk_sizes:
            self.set_chunk_size(chunk_size)
            
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.compute_prime_counts_parallel(x_values)
                end_time = time.time()
                times.append(end_time - start_time)
            
            results[chunk_size] = np.mean(times)
        
        return results
    
    def optimize_num_processes(self, x_values: np.ndarray, max_processes: int, num_runs: int = 3) -> Dict[int, float]:
        """
        Optimize the number of processes for parallel computation.
        
        Args:
            x_values (np.ndarray): The values at which to compute the prime counting function.
            max_processes (int): The maximum number of processes to test.
            num_runs (int): The number of runs for each number of processes.
        
        Returns:
            Dict[int, float]: The execution times for each number of processes.
        """
        results = {}
        
        for num_processes in range(1, max_processes + 1):
            self.set_num_processes(num_processes)
            
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.compute_prime_counts_parallel(x_values)
                end_time = time.time()
                times.append(end_time - start_time)
            
            results[num_processes] = np.mean(times)
        
        return results
    
    def __str__(self) -> str:
        """
        Return a string representation of the parallel prime distribution calculator.
        
        Returns:
            str: A string representation of the parallel prime distribution calculator.
        """
        return f"Parallel Prime Distribution Calculator with {self.num_processes} processes and chunk size {self.chunk_size}"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the parallel prime distribution calculator.
        
        Returns:
            str: A string representation of the parallel prime distribution calculator.
        """
        return f"ParallelPrimeDistribution(CyclotomicField({self.cyclotomic_field.conductor}), {self.num_processes}, {self.chunk_size})"


class ParallelProcessingManager:
    """
    A class for managing parallel processing in the cyclotomic field theory framework.
    
    This class provides methods to parallelize various computations in the framework.
    
    Attributes:
        num_processes (int): The number of processes to use for parallel computation.
        process_pool (Optional[mp.Pool]): The process pool.
    """
    
    def __init__(self, num_processes: Optional[int] = None):
        """
        Initialize a parallel processing manager.
        
        Args:
            num_processes (Optional[int]): The number of processes to use (default: number of CPU cores).
        """
        self.num_processes = num_processes if num_processes is not None else mp.cpu_count()
        self.process_pool = None
    
    def start_pool(self):
        """
        Start the process pool.
        """
        if self.process_pool is None:
            self.process_pool = mp.Pool(processes=self.num_processes)
    
    def stop_pool(self):
        """
        Stop the process pool.
        """
        if self.process_pool is not None:
            self.process_pool.close()
            self.process_pool.join()
            self.process_pool = None
    
    def map(self, func: Callable, iterable: List[Any]) -> List[Any]:
        """
        Apply a function to each element in an iterable in parallel.
        
        Args:
            func (Callable): The function to apply.
            iterable (List[Any]): The iterable.
        
        Returns:
            List[Any]: The results.
        """
        if self.process_pool is None:
            self.start_pool()
        
        return self.process_pool.map(func, iterable)
    
    def starmap(self, func: Callable, iterable: List[Tuple]) -> List[Any]:
        """
        Apply a function to each tuple of arguments in an iterable in parallel.
        
        Args:
            func (Callable): The function to apply.
            iterable (List[Tuple]): The iterable of argument tuples.
        
        Returns:
            List[Any]: The results.
        """
        if self.process_pool is None:
            self.start_pool()
        
        return self.process_pool.starmap(func, iterable)
    
    def apply_async(self, func: Callable, args: Tuple = (), kwds: Dict = {}) -> mp.pool.AsyncResult:
        """
        Apply a function to arguments asynchronously.
        
        Args:
            func (Callable): The function to apply.
            args (Tuple): The positional arguments.
            kwds (Dict): The keyword arguments.
        
        Returns:
            mp.pool.AsyncResult: The async result.
        """
        if self.process_pool is None:
            self.start_pool()
        
        return self.process_pool.apply_async(func, args, kwds)
    
    def parallelize_function(self, func: Callable, args_list: List[Tuple]) -> List[Any]:
        """
        Parallelize a function over a list of argument tuples.
        
        Args:
            func (Callable): The function to parallelize.
            args_list (List[Tuple]): The list of argument tuples.
        
        Returns:
            List[Any]: The results.
        """
        return self.starmap(func, args_list)
    
    def parallelize_method(self, obj: Any, method_name: str, args_list: List[Tuple]) -> List[Any]:
        """
        Parallelize a method of an object over a list of argument tuples.
        
        Args:
            obj (Any): The object.
            method_name (str): The name of the method.
            args_list (List[Tuple]): The list of argument tuples.
        
        Returns:
            List[Any]: The results.
        """
        method = getattr(obj, method_name)
        return self.starmap(method, args_list)
    
    def chunk_data(self, data: List[Any], num_chunks: Optional[int] = None) -> List[List[Any]]:
        """
        Split data into chunks for parallel processing.
        
        Args:
            data (List[Any]): The data to split.
            num_chunks (Optional[int]): The number of chunks (default: number of processes).
        
        Returns:
            List[List[Any]]: The chunks.
        """
        if num_chunks is None:
            num_chunks = self.num_processes
        
        chunk_size = len(data) // num_chunks
        if chunk_size == 0:
            chunk_size = 1
        
        return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
    def __enter__(self):
        """
        Enter the context manager.
        
        Returns:
            ParallelProcessingManager: The parallel processing manager.
        """
        self.start_pool()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.
        
        Args:
            exc_type: The exception type.
            exc_val: The exception value.
            exc_tb: The exception traceback.
        """
        self.stop_pool()
    
    def __str__(self) -> str:
        """
        Return a string representation of the parallel processing manager.
        
        Returns:
            str: A string representation of the parallel processing manager.
        """
        return f"Parallel Processing Manager with {self.num_processes} processes"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the parallel processing manager.
        
        Returns:
            str: A string representation of the parallel processing manager.
        """
        return f"ParallelProcessingManager({self.num_processes})"