"""
Benchmarking Module for TIBEDO Framework

This module provides tools for benchmarking the performance of the TIBEDO Framework,
enabling comparison between different implementations and optimization strategies.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import os
import json
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import pandas as pd
from datetime import datetime

class PerformanceBenchmark:
    """
    Performance benchmarking for the TIBEDO Framework.
    
    This class provides tools for benchmarking the performance of different
    implementations and optimization strategies in the TIBEDO Framework.
    """
    
    def __init__(self, name: str, save_dir: str = 'benchmark_results'):
        """
        Initialize the PerformanceBenchmark.
        
        Args:
            name (str): Name of the benchmark
            save_dir (str): Directory to save benchmark results
        """
        self.name = name
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize results
        self.results = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'implementations': {},
            'parameters': {},
            'metrics': {}
        }
        
    def register_implementation(self, name: str, func: Callable, description: str = '') -> None:
        """
        Register an implementation for benchmarking.
        
        Args:
            name (str): Name of the implementation
            func (Callable): Implementation function
            description (str): Description of the implementation
        """
        self.results['implementations'][name] = {
            'description': description,
            'results': {}
        }
        
        # Store function reference (not in results dictionary)
        setattr(self, f"_func_{name}", func)
        
    def set_parameters(self, **kwargs) -> None:
        """
        Set benchmark parameters.
        
        Args:
            **kwargs: Benchmark parameters
        """
        self.results['parameters'].update(kwargs)
        
    def run_benchmark(self, implementation_name: str, repeat: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Run a benchmark for a specific implementation.
        
        Args:
            implementation_name (str): Name of the implementation
            repeat (int): Number of times to repeat the benchmark
            **kwargs: Arguments for the implementation function
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        if implementation_name not in self.results['implementations']:
            raise ValueError(f"Implementation '{implementation_name}' not registered")
            
        # Get implementation function
        func = getattr(self, f"_func_{implementation_name}")
        
        # Run benchmark
        execution_times = []
        results = None
        
        for i in range(repeat):
            # Measure execution time
            start_time = time.time()
            results = func(**kwargs)
            execution_time = time.time() - start_time
            
            execution_times.append(execution_time)
            
        # Calculate statistics
        mean_time = np.mean(execution_times)
        std_time = np.std(execution_times)
        min_time = np.min(execution_times)
        max_time = np.max(execution_times)
        
        # Store results
        benchmark_results = {
            'mean_time': mean_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'execution_times': execution_times,
            'repeat': repeat,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store parameters
        param_key = self._get_param_key(kwargs)
        self.results['implementations'][implementation_name]['results'][param_key] = benchmark_results
        
        return benchmark_results
        
    def _get_param_key(self, params: Dict[str, Any]) -> str:
        """
        Get a key for parameters.
        
        Args:
            params (Dict[str, Any]): Parameters
            
        Returns:
            str: Parameter key
        """
        # Sort keys for consistent ordering
        sorted_keys = sorted(params.keys())
        
        # Create key
        key_parts = []
        for k in sorted_keys:
            v = params[k]
            if isinstance(v, (list, tuple, np.ndarray)):
                # Use length for arrays
                key_parts.append(f"{k}={len(v)}")
            else:
                # Use value for scalars
                key_parts.append(f"{k}={v}")
                
        return ','.join(key_parts)
        
    def compare_implementations(self, param_key: Optional[str] = None) -> pd.DataFrame:
        """
        Compare different implementations.
        
        Args:
            param_key (str, optional): Parameter key to compare.
                If None, compares all parameter combinations.
            
        Returns:
            pd.DataFrame: Comparison results
        """
        # Initialize data for DataFrame
        data = []
        
        # Iterate over implementations
        for impl_name, impl_data in self.results['implementations'].items():
            # Get results for implementation
            results = impl_data['results']
            
            # Filter by parameter key if specified
            if param_key is not None:
                if param_key in results:
                    result = results[param_key]
                    data.append({
                        'Implementation': impl_name,
                        'Parameters': param_key,
                        'Mean Time (s)': result['mean_time'],
                        'Std Time (s)': result['std_time'],
                        'Min Time (s)': result['min_time'],
                        'Max Time (s)': result['max_time']
                    })
            else:
                # Include all parameter combinations
                for param_key, result in results.items():
                    data.append({
                        'Implementation': impl_name,
                        'Parameters': param_key,
                        'Mean Time (s)': result['mean_time'],
                        'Std Time (s)': result['std_time'],
                        'Min Time (s)': result['min_time'],
                        'Max Time (s)': result['max_time']
                    })
                    
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Sort by mean time
        df = df.sort_values('Mean Time (s)')
        
        return df
        
    def plot_comparison(self, param_key: Optional[str] = None, 
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of different implementations.
        
        Args:
            param_key (str, optional): Parameter key to compare.
                If None, compares all parameter combinations.
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The figure object
        """
        # Get comparison data
        df = self.compare_implementations(param_key)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bars
        bars = ax.bar(df['Implementation'], df['Mean Time (s)'])
        
        # Add error bars
        ax.errorbar(df['Implementation'], df['Mean Time (s)'], 
                   yerr=df['Std Time (s)'], fmt='none', ecolor='black', capsize=5)
        
        # Add labels and title
        ax.set_xlabel('Implementation')
        ax.set_ylabel('Execution Time (seconds)')
        
        if param_key is not None:
            ax.set_title(f'Performance Comparison ({param_key})')
        else:
            ax.set_title('Performance Comparison')
            
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.4f}s', ha='center', va='bottom')
                   
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return fig
        
    def plot_scaling(self, implementation_name: str, param_name: str, 
                    param_values: List[Any], save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot scaling of an implementation with respect to a parameter.
        
        Args:
            implementation_name (str): Name of the implementation
            param_name (str): Name of the parameter to vary
            param_values (List[Any]): List of parameter values
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The figure object
        """
        if implementation_name not in self.results['implementations']:
            raise ValueError(f"Implementation '{implementation_name}' not registered")
            
        # Get implementation results
        impl_results = self.results['implementations'][implementation_name]['results']
        
        # Extract execution times for each parameter value
        x_values = []
        y_values = []
        y_errors = []
        
        for param_value in param_values:
            # Create parameter key
            param_key = f"{param_name}={param_value}"
            
            # Find matching parameter key
            matching_key = None
            for key in impl_results.keys():
                if param_key in key:
                    matching_key = key
                    break
                    
            if matching_key is None:
                continue
                
            # Get results
            result = impl_results[matching_key]
            
            # Store values
            x_values.append(param_value)
            y_values.append(result['mean_time'])
            y_errors.append(result['std_time'])
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot line
        ax.errorbar(x_values, y_values, yerr=y_errors, fmt='o-', capsize=5)
        
        # Add labels and title
        ax.set_xlabel(param_name)
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'Scaling of {implementation_name} with {param_name}')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set log scale if values span multiple orders of magnitude
        if max(x_values) / min(x_values) > 100:
            ax.set_xscale('log')
            
        if max(y_values) / min(y_values) > 100:
            ax.set_yscale('log')
            
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return fig
        
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save benchmark results to a file.
        
        Args:
            filename (str, optional): Filename to save results.
                If None, uses the benchmark name and timestamp.
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            # Create filename from benchmark name and timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.name}_{timestamp}.json"
            
        # Create full path
        filepath = os.path.join(self.save_dir, filename)
        
        # Save results
        with open(filepath, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_copy = self._prepare_for_json(self.results)
            json.dump(results_copy, f, indent=2)
            
        return filepath
        
    def _prepare_for_json(self, obj):
        """
        Prepare an object for JSON serialization.
        
        Args:
            obj: Object to prepare
            
        Returns:
            Object prepared for JSON serialization
        """
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
            
    def load_results(self, filepath: str) -> None:
        """
        Load benchmark results from a file.
        
        Args:
            filepath (str): Path to the results file
        """
        with open(filepath, 'r') as f:
            self.results = json.load(f)
            
        # Update benchmark name
        self.name = self.results['name']
        
    def add_metric(self, name: str, value: Any, description: str = '') -> None:
        """
        Add a custom metric to the benchmark results.
        
        Args:
            name (str): Name of the metric
            value (Any): Value of the metric
            description (str): Description of the metric
        """
        self.results['metrics'][name] = {
            'value': value,
            'description': description
        }
        
    def get_metric(self, name: str) -> Any:
        """
        Get a custom metric from the benchmark results.
        
        Args:
            name (str): Name of the metric
            
        Returns:
            Any: Value of the metric
        """
        if name not in self.results['metrics']:
            raise ValueError(f"Metric '{name}' not found")
            
        return self.results['metrics'][name]['value']


class MemoryBenchmark:
    """
    Memory usage benchmarking for the TIBEDO Framework.
    
    This class provides tools for benchmarking the memory usage of different
    implementations and optimization strategies in the TIBEDO Framework.
    """
    
    def __init__(self, name: str, save_dir: str = 'benchmark_results'):
        """
        Initialize the MemoryBenchmark.
        
        Args:
            name (str): Name of the benchmark
            save_dir (str): Directory to save benchmark results
        """
        self.name = name
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize results
        self.results = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'implementations': {},
            'parameters': {},
            'metrics': {}
        }
        
        # Import memory profiler if available
        try:
            import memory_profiler
            self.memory_profiler = memory_profiler
            self.has_memory_profiler = True
        except ImportError:
            print("Warning: memory_profiler not available, using psutil instead")
            self.has_memory_profiler = False
            
            # Import psutil
            import psutil
            self.psutil = psutil
            
    def register_implementation(self, name: str, func: Callable, description: str = '') -> None:
        """
        Register an implementation for benchmarking.
        
        Args:
            name (str): Name of the implementation
            func (Callable): Implementation function
            description (str): Description of the implementation
        """
        self.results['implementations'][name] = {
            'description': description,
            'results': {}
        }
        
        # Store function reference (not in results dictionary)
        setattr(self, f"_func_{name}", func)
        
    def set_parameters(self, **kwargs) -> None:
        """
        Set benchmark parameters.
        
        Args:
            **kwargs: Benchmark parameters
        """
        self.results['parameters'].update(kwargs)
        
    def run_benchmark(self, implementation_name: str, **kwargs) -> Dict[str, Any]:
        """
        Run a memory benchmark for a specific implementation.
        
        Args:
            implementation_name (str): Name of the implementation
            **kwargs: Arguments for the implementation function
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        if implementation_name not in self.results['implementations']:
            raise ValueError(f"Implementation '{implementation_name}' not registered")
            
        # Get implementation function
        func = getattr(self, f"_func_{implementation_name}")
        
        # Run benchmark
        if self.has_memory_profiler:
            # Use memory_profiler
            memory_usage = self.memory_profiler.memory_usage(
                (func, [], kwargs),
                interval=0.1,
                timeout=None,
                max_iterations=1
            )
            
            # Calculate statistics
            baseline = memory_usage[0]
            peak = max(memory_usage)
            mean = np.mean(memory_usage)
            
            # Store results
            benchmark_results = {
                'baseline_memory': baseline,
                'peak_memory': peak,
                'mean_memory': mean,
                'memory_usage': memory_usage,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Use psutil
            process = self.psutil.Process()
            
            # Get baseline memory usage
            baseline = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Run function
            result = func(**kwargs)
            
            # Get peak memory usage
            peak = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Store results
            benchmark_results = {
                'baseline_memory': baseline,
                'peak_memory': peak,
                'memory_increase': peak - baseline,
                'timestamp': datetime.now().isoformat()
            }
            
        # Store parameters
        param_key = self._get_param_key(kwargs)
        self.results['implementations'][implementation_name]['results'][param_key] = benchmark_results
        
        return benchmark_results
        
    def _get_param_key(self, params: Dict[str, Any]) -> str:
        """
        Get a key for parameters.
        
        Args:
            params (Dict[str, Any]): Parameters
            
        Returns:
            str: Parameter key
        """
        # Sort keys for consistent ordering
        sorted_keys = sorted(params.keys())
        
        # Create key
        key_parts = []
        for k in sorted_keys:
            v = params[k]
            if isinstance(v, (list, tuple, np.ndarray)):
                # Use length for arrays
                key_parts.append(f"{k}={len(v)}")
            else:
                # Use value for scalars
                key_parts.append(f"{k}={v}")
                
        return ','.join(key_parts)
        
    def compare_implementations(self, param_key: Optional[str] = None) -> pd.DataFrame:
        """
        Compare different implementations.
        
        Args:
            param_key (str, optional): Parameter key to compare.
                If None, compares all parameter combinations.
            
        Returns:
            pd.DataFrame: Comparison results
        """
        # Initialize data for DataFrame
        data = []
        
        # Iterate over implementations
        for impl_name, impl_data in self.results['implementations'].items():
            # Get results for implementation
            results = impl_data['results']
            
            # Filter by parameter key if specified
            if param_key is not None:
                if param_key in results:
                    result = results[param_key]
                    data.append({
                        'Implementation': impl_name,
                        'Parameters': param_key,
                        'Baseline Memory (MB)': result.get('baseline_memory', 0),
                        'Peak Memory (MB)': result.get('peak_memory', 0),
                        'Memory Increase (MB)': result.get('memory_increase', 
                                                        result.get('peak_memory', 0) - result.get('baseline_memory', 0))
                    })
            else:
                # Include all parameter combinations
                for param_key, result in results.items():
                    data.append({
                        'Implementation': impl_name,
                        'Parameters': param_key,
                        'Baseline Memory (MB)': result.get('baseline_memory', 0),
                        'Peak Memory (MB)': result.get('peak_memory', 0),
                        'Memory Increase (MB)': result.get('memory_increase', 
                                                        result.get('peak_memory', 0) - result.get('baseline_memory', 0))
                    })
                    
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Sort by memory increase
        df = df.sort_values('Memory Increase (MB)')
        
        return df
        
    def plot_comparison(self, param_key: Optional[str] = None, 
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of different implementations.
        
        Args:
            param_key (str, optional): Parameter key to compare.
                If None, compares all parameter combinations.
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The figure object
        """
        # Get comparison data
        df = self.compare_implementations(param_key)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bars
        bars = ax.bar(df['Implementation'], df['Memory Increase (MB)'])
        
        # Add labels and title
        ax.set_xlabel('Implementation')
        ax.set_ylabel('Memory Increase (MB)')
        
        if param_key is not None:
            ax.set_title(f'Memory Usage Comparison ({param_key})')
        else:
            ax.set_title('Memory Usage Comparison')
            
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f} MB', ha='center', va='bottom')
                   
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return fig
        
    def plot_memory_profile(self, implementation_name: str, param_key: str, 
                           save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Plot memory profile of an implementation.
        
        Args:
            implementation_name (str): Name of the implementation
            param_key (str): Parameter key
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The figure object, or None if memory profile not available
        """
        if not self.has_memory_profiler:
            print("Memory profiler not available, cannot plot memory profile")
            return None
            
        if implementation_name not in self.results['implementations']:
            raise ValueError(f"Implementation '{implementation_name}' not registered")
            
        # Get implementation results
        impl_results = self.results['implementations'][implementation_name]['results']
        
        if param_key not in impl_results:
            raise ValueError(f"Parameter key '{param_key}' not found for implementation '{implementation_name}'")
            
        # Get memory usage
        result = impl_results[param_key]
        memory_usage = result.get('memory_usage')
        
        if memory_usage is None:
            print("Memory usage profile not available")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot memory usage
        time_points = np.arange(len(memory_usage)) * 0.1  # 0.1s interval
        ax.plot(time_points, memory_usage)
        
        # Add labels and title
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title(f'Memory Profile of {implementation_name} ({param_key})')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line for baseline memory
        ax.axhline(y=result['baseline_memory'], color='r', linestyle='--', 
                  label=f'Baseline: {result["baseline_memory"]:.2f} MB')
                  
        # Add horizontal line for peak memory
        ax.axhline(y=result['peak_memory'], color='g', linestyle='--', 
                  label=f'Peak: {result["peak_memory"]:.2f} MB')
                  
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return fig
        
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save benchmark results to a file.
        
        Args:
            filename (str, optional): Filename to save results.
                If None, uses the benchmark name and timestamp.
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            # Create filename from benchmark name and timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.name}_memory_{timestamp}.json"
            
        # Create full path
        filepath = os.path.join(self.save_dir, filename)
        
        # Save results
        with open(filepath, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_copy = self._prepare_for_json(self.results)
            json.dump(results_copy, f, indent=2)
            
        return filepath
        
    def _prepare_for_json(self, obj):
        """
        Prepare an object for JSON serialization.
        
        Args:
            obj: Object to prepare
            
        Returns:
            Object prepared for JSON serialization
        """
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
            
    def load_results(self, filepath: str) -> None:
        """
        Load benchmark results from a file.
        
        Args:
            filepath (str): Path to the results file
        """
        with open(filepath, 'r') as f:
            self.results = json.load(f)
            
        # Update benchmark name
        self.name = self.results['name']
        
    def add_metric(self, name: str, value: Any, description: str = '') -> None:
        """
        Add a custom metric to the benchmark results.
        
        Args:
            name (str): Name of the metric
            value (Any): Value of the metric
            description (str): Description of the metric
        """
        self.results['metrics'][name] = {
            'value': value,
            'description': description
        }
        
    def get_metric(self, name: str) -> Any:
        """
        Get a custom metric from the benchmark results.
        
        Args:
            name (str): Name of the metric
            
        Returns:
            Any: Value of the metric
        """
        if name not in self.results['metrics']:
            raise ValueError(f"Metric '{name}' not found")
            
        return self.results['metrics'][name]['value']


def benchmark_matrix_operations(sizes: List[int], repeat: int = 3) -> Dict[str, Any]:
    """
    Benchmark matrix operations with different implementations.
    
    Args:
        sizes (List[int]): List of matrix sizes to benchmark
        repeat (int): Number of times to repeat each benchmark
        
    Returns:
        Dict[str, Any]: Benchmark results
    """
    # Import implementations
    from tibedo.performance.parallel_processing import ParallelMatrixOperations
    from tibedo.performance.gpu_acceleration import GPUAccelerator
    
    # Create benchmark
    benchmark = PerformanceBenchmark('matrix_operations')
    
    # Register implementations
    benchmark.register_implementation(
        'numpy',
        lambda A, B: np.matmul(A, B),
        'NumPy implementation'
    )
    
    benchmark.register_implementation(
        'parallel',
        lambda A, B: ParallelMatrixOperations().matmul(A, B),
        'Parallel implementation'
    )
    
    benchmark.register_implementation(
        'gpu',
        lambda A, B: GPUAccelerator().matmul(A, B),
        'GPU implementation'
    )
    
    # Set parameters
    benchmark.set_parameters(sizes=sizes, repeat=repeat)
    
    # Run benchmarks
    for size in sizes:
        # Create random matrices
        A = np.random.random((size, size))
        B = np.random.random((size, size))
        
        # Run benchmarks for each implementation
        for impl_name in ['numpy', 'parallel', 'gpu']:
            try:
                benchmark.run_benchmark(impl_name, A=A, B=B)
            except Exception as e:
                print(f"Error benchmarking {impl_name} with size {size}: {e}")
                
    # Save results
    benchmark.save_results()
    
    # Plot comparison for largest size
    largest_size = max(sizes)
    benchmark.plot_comparison(f"A={largest_size},B={largest_size}", 
                             save_path=f"matrix_operations_{largest_size}.png")
                             
    # Plot scaling
    benchmark.plot_scaling('numpy', 'A', sizes, 
                          save_path="matrix_operations_scaling_numpy.png")
    benchmark.plot_scaling('parallel', 'A', sizes, 
                          save_path="matrix_operations_scaling_parallel.png")
    benchmark.plot_scaling('gpu', 'A', sizes, 
                          save_path="matrix_operations_scaling_gpu.png")
                          
    return benchmark.results


def benchmark_tensor_operations(sizes: List[int], repeat: int = 3) -> Dict[str, Any]:
    """
    Benchmark tensor operations with different implementations.
    
    Args:
        sizes (List[int]): List of tensor sizes to benchmark
        repeat (int): Number of times to repeat each benchmark
        
    Returns:
        Dict[str, Any]: Benchmark results
    """
    # Import implementations
    from tibedo.performance.gpu_acceleration import GPUTensorOperations
    
    # Create benchmark
    benchmark = PerformanceBenchmark('tensor_operations')
    
    # Register implementations
    benchmark.register_implementation(
        'numpy',
        lambda tensor1, tensor2, axes1, axes2: np.tensordot(tensor1, tensor2, axes=(axes1, axes2)),
        'NumPy implementation'
    )
    
    benchmark.register_implementation(
        'gpu',
        lambda tensor1, tensor2, axes1, axes2: GPUTensorOperations().tensor_contraction(tensor1, tensor2, axes1, axes2),
        'GPU implementation'
    )
    
    # Set parameters
    benchmark.set_parameters(sizes=sizes, repeat=repeat)
    
    # Run benchmarks
    for size in sizes:
        # Create random tensors
        tensor1 = np.random.random((size, size, size))
        tensor2 = np.random.random((size, size, size))
        
        # Run benchmarks for each implementation
        for impl_name in ['numpy', 'gpu']:
            try:
                benchmark.run_benchmark(impl_name, tensor1=tensor1, tensor2=tensor2, axes1=1, axes2=0)
            except Exception as e:
                print(f"Error benchmarking {impl_name} with size {size}: {e}")
                
    # Save results
    benchmark.save_results()
    
    # Plot comparison for largest size
    largest_size = max(sizes)
    benchmark.plot_comparison(f"tensor1={largest_size},tensor2={largest_size},axes1=1,axes2=0", 
                             save_path=f"tensor_operations_{largest_size}.png")
                             
    # Plot scaling
    benchmark.plot_scaling('numpy', 'tensor1', sizes, 
                          save_path="tensor_operations_scaling_numpy.png")
    benchmark.plot_scaling('gpu', 'tensor1', sizes, 
                          save_path="tensor_operations_scaling_gpu.png")
                          
    return benchmark.results


def benchmark_memory_usage(sizes: List[int]) -> Dict[str, Any]:
    """
    Benchmark memory usage with different implementations.
    
    Args:
        sizes (List[int]): List of matrix sizes to benchmark
        
    Returns:
        Dict[str, Any]: Benchmark results
    """
    # Import implementations
    from tibedo.performance.memory_optimization import MemoryEfficientArray
    
    # Create benchmark
    benchmark = MemoryBenchmark('memory_usage')
    
    # Register implementations
    benchmark.register_implementation(
        'numpy',
        lambda size: np.random.random((size, size)),
        'NumPy implementation'
    )
    
    benchmark.register_implementation(
        'memory_efficient',
        lambda size: MemoryEfficientArray((size, size), dtype=np.float32, mode='mmap'),
        'Memory-efficient implementation'
    )
    
    # Set parameters
    benchmark.set_parameters(sizes=sizes)
    
    # Run benchmarks
    for size in sizes:
        # Run benchmarks for each implementation
        for impl_name in ['numpy', 'memory_efficient']:
            try:
                benchmark.run_benchmark(impl_name, size=size)
            except Exception as e:
                print(f"Error benchmarking {impl_name} with size {size}: {e}")
                
    # Save results
    benchmark.save_results()
    
    # Plot comparison for largest size
    largest_size = max(sizes)
    benchmark.plot_comparison(f"size={largest_size}", 
                             save_path=f"memory_usage_{largest_size}.png")
                             
    return benchmark.results