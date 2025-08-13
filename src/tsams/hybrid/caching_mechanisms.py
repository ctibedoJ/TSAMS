"""
Caching Mechanisms implementation.

This module provides implementations of caching mechanisms for expensive computations
in the cyclotomic field theory framework.
"""

import numpy as np
import os
import pickle
import hashlib
import time
from typing import List, Dict, Tuple, Union, Optional, Callable, Any, TypeVar, Generic
from functools import lru_cache, wraps
import json
from pathlib import Path
from ..core.cyclotomic_field import CyclotomicField
from ..core.prime_spectral_grouping import PrimeSpectralGrouping
from ..cosmology.prime_distribution import PrimeDistribution


T = TypeVar('T')  # Type variable for generic caching


class ComputationCache(Generic[T]):
    """
    A generic cache for expensive computations.
    
    This class provides methods to cache the results of expensive computations,
    with support for both in-memory and disk-based caching.
    
    Attributes:
        cache_dir (str): The directory for disk-based caching.
        memory_cache (Dict[str, T]): The in-memory cache.
        max_memory_entries (int): The maximum number of entries in the memory cache.
        disk_cache_enabled (bool): Whether disk-based caching is enabled.
        cache_stats (Dict[str, int]): Statistics about cache hits and misses.
    """
    
    def __init__(self, cache_dir: str = ".cache", max_memory_entries: int = 1000, disk_cache_enabled: bool = True):
        """
        Initialize a computation cache.
        
        Args:
            cache_dir (str): The directory for disk-based caching.
            max_memory_entries (int): The maximum number of entries in the memory cache.
            disk_cache_enabled (bool): Whether disk-based caching is enabled.
        """
        self.cache_dir = cache_dir
        self.memory_cache = {}
        self.max_memory_entries = max_memory_entries
        self.disk_cache_enabled = disk_cache_enabled
        self.cache_stats = {"hits": 0, "misses": 0, "memory_hits": 0, "disk_hits": 0}
        
        # Create the cache directory if it doesn't exist
        if self.disk_cache_enabled:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _compute_key(self, *args, **kwargs) -> str:
        """
        Compute a cache key from the arguments.
        
        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        
        Returns:
            str: The cache key.
        """
        # Convert the arguments to a string representation
        args_str = str(args) + str(sorted(kwargs.items()))
        
        # Compute a hash of the string
        key = hashlib.md5(args_str.encode()).hexdigest()
        
        return key
    
    def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.
        
        Args:
            key (str): The cache key.
        
        Returns:
            Optional[T]: The cached value, or None if not found.
        """
        # Check the memory cache
        if key in self.memory_cache:
            self.cache_stats["hits"] += 1
            self.cache_stats["memory_hits"] += 1
            return self.memory_cache[key]
        
        # Check the disk cache
        if self.disk_cache_enabled:
            cache_file = os.path.join(self.cache_dir, key + ".pickle")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        value = pickle.load(f)
                    
                    # Add to the memory cache
                    self._add_to_memory_cache(key, value)
                    
                    self.cache_stats["hits"] += 1
                    self.cache_stats["disk_hits"] += 1
                    return value
                except (pickle.PickleError, EOFError, AttributeError):
                    # If there's an error loading the cache, ignore it
                    pass
        
        self.cache_stats["misses"] += 1
        return None
    
    def set(self, key: str, value: T):
        """
        Set a value in the cache.
        
        Args:
            key (str): The cache key.
            value (T): The value to cache.
        """
        # Add to the memory cache
        self._add_to_memory_cache(key, value)
        
        # Add to the disk cache
        if self.disk_cache_enabled:
            cache_file = os.path.join(self.cache_dir, key + ".pickle")
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(value, f)
            except (pickle.PickleError, TypeError, AttributeError):
                # If there's an error saving the cache, ignore it
                pass
    
    def _add_to_memory_cache(self, key: str, value: T):
        """
        Add a value to the memory cache.
        
        Args:
            key (str): The cache key.
            value (T): The value to cache.
        """
        # If the cache is full, remove the oldest entry
        if len(self.memory_cache) >= self.max_memory_entries:
            # Get the first key (oldest entry)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        # Add the new entry
        self.memory_cache[key] = value
    
    def clear(self):
        """
        Clear the cache.
        """
        # Clear the memory cache
        self.memory_cache.clear()
        
        # Clear the disk cache
        if self.disk_cache_enabled:
            for file in os.listdir(self.cache_dir):
                if file.endswith(".pickle"):
                    os.remove(os.path.join(self.cache_dir, file))
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, int]: The cache statistics.
        """
        return self.cache_stats.copy()
    
    def cached(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator for caching function results.
        
        Args:
            func (Callable[..., T]): The function to cache.
        
        Returns:
            Callable[..., T]: The cached function.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Compute the cache key
            key = self._compute_key(func.__name__, *args, **kwargs)
            
            # Check the cache
            cached_value = self.get(key)
            if cached_value is not None:
                return cached_value
            
            # Compute the value
            value = func(*args, **kwargs)
            
            # Cache the value
            self.set(key, value)
            
            return value
        
        return wrapper


class PrimeDistributionCache:
    """
    A cache for prime distribution calculations.
    
    This class provides methods to cache the results of prime distribution calculations,
    which are computationally expensive.
    
    Attributes:
        cache (ComputationCache): The underlying computation cache.
        prime_distribution (PrimeDistribution): The prime distribution calculator.
    """
    
    def __init__(self, cyclotomic_field: CyclotomicField, cache_dir: str = ".cache/prime_distribution"):
        """
        Initialize a prime distribution cache.
        
        Args:
            cyclotomic_field (CyclotomicField): The cyclotomic field.
            cache_dir (str): The directory for disk-based caching.
        """
        self.cache = ComputationCache[np.ndarray](cache_dir=cache_dir)
        self.prime_distribution = PrimeDistribution(cyclotomic_field)
    
    def prime_counting_function(self, x: float) -> int:
        """
        Compute the prime counting function with caching.
        
        Args:
            x (float): The value at which to compute the function.
        
        Returns:
            int: The number of primes less than or equal to x.
        """
        # Compute the cache key
        key = self.cache._compute_key("prime_counting_function", x)
        
        # Check the cache
        cached_value = self.cache.get(key)
        if cached_value is not None:
            return cached_value
        
        # Compute the value
        value = self.prime_distribution.prime_counting_function(x)
        
        # Cache the value
        self.cache.set(key, value)
        
        return value
    
    def prime_distribution_formula(self, x: float) -> float:
        """
        Compute the prime distribution formula with caching.
        
        Args:
            x (float): The value at which to compute the formula.
        
        Returns:
            float: The value of the formula.
        """
        # Compute the cache key
        key = self.cache._compute_key("prime_distribution_formula", x)
        
        # Check the cache
        cached_value = self.cache.get(key)
        if cached_value is not None:
            return cached_value
        
        # Compute the value
        value = self.prime_distribution.prime_distribution_formula(x)
        
        # Cache the value
        self.cache.set(key, value)
        
        return value
    
    def compute_prime_counts(self, x_values: np.ndarray) -> np.ndarray:
        """
        Compute the prime counting function for multiple values with caching.
        
        Args:
            x_values (np.ndarray): The values at which to compute the function.
        
        Returns:
            np.ndarray: The prime counts.
        """
        # Compute the cache key
        key = self.cache._compute_key("compute_prime_counts", x_values.tobytes())
        
        # Check the cache
        cached_value = self.cache.get(key)
        if cached_value is not None:
            return cached_value
        
        # Compute the values
        values = np.array([self.prime_counting_function(x) for x in x_values])
        
        # Cache the values
        self.cache.set(key, values)
        
        return values
    
    def compute_formula_values(self, x_values: np.ndarray) -> np.ndarray:
        """
        Compute the prime distribution formula for multiple values with caching.
        
        Args:
            x_values (np.ndarray): The values at which to compute the formula.
        
        Returns:
            np.ndarray: The formula values.
        """
        # Compute the cache key
        key = self.cache._compute_key("compute_formula_values", x_values.tobytes())
        
        # Check the cache
        cached_value = self.cache.get(key)
        if cached_value is not None:
            return cached_value
        
        # Compute the values
        values = np.array([self.prime_distribution_formula(x) for x in x_values])
        
        # Cache the values
        self.cache.set(key, values)
        
        return values
    
    def clear_cache(self):
        """
        Clear the cache.
        """
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, int]: The cache statistics.
        """
        return self.cache.get_stats()


class CyclotomicFieldCache:
    """
    A cache for cyclotomic field operations.
    
    This class provides methods to cache the results of cyclotomic field operations,
    which can be computationally expensive.
    
    Attributes:
        cache (ComputationCache): The underlying computation cache.
        cyclotomic_field (CyclotomicField): The cyclotomic field.
    """
    
    def __init__(self, cyclotomic_field: CyclotomicField, cache_dir: str = ".cache/cyclotomic_field"):
        """
        Initialize a cyclotomic field cache.
        
        Args:
            cyclotomic_field (CyclotomicField): The cyclotomic field.
            cache_dir (str): The directory for disk-based caching.
        """
        self.cache = ComputationCache[Dict[int, float]](cache_dir=cache_dir)
        self.cyclotomic_field = cyclotomic_field
    
    def add(self, a: Dict[int, float], b: Dict[int, float]) -> Dict[int, float]:
        """
        Add two elements of the cyclotomic field with caching.
        
        Args:
            a (Dict[int, float]): The first element.
            b (Dict[int, float]): The second element.
        
        Returns:
            Dict[int, float]: The sum of the two elements.
        """
        # Convert the elements to a hashable representation
        a_tuple = tuple(sorted((k, v) for k, v in a.items()))
        b_tuple = tuple(sorted((k, v) for k, v in b.items()))
        
        # Compute the cache key
        key = self.cache._compute_key("add", a_tuple, b_tuple)
        
        # Check the cache
        cached_value = self.cache.get(key)
        if cached_value is not None:
            return cached_value
        
        # Compute the value
        value = self.cyclotomic_field.add(a, b)
        
        # Cache the value
        self.cache.set(key, value)
        
        return value
    
    def multiply(self, a: Dict[int, float], b: Dict[int, float]) -> Dict[int, float]:
        """
        Multiply two elements of the cyclotomic field with caching.
        
        Args:
            a (Dict[int, float]): The first element.
            b (Dict[int, float]): The second element.
        
        Returns:
            Dict[int, float]: The product of the two elements.
        """
        # Convert the elements to a hashable representation
        a_tuple = tuple(sorted((k, v) for k, v in a.items()))
        b_tuple = tuple(sorted((k, v) for k, v in b.items()))
        
        # Compute the cache key
        key = self.cache._compute_key("multiply", a_tuple, b_tuple)
        
        # Check the cache
        cached_value = self.cache.get(key)
        if cached_value is not None:
            return cached_value
        
        # Compute the value
        value = self.cyclotomic_field.multiply(a, b)
        
        # Cache the value
        self.cache.set(key, value)
        
        return value
    
    def conjugate(self, a: Dict[int, float]) -> Dict[int, float]:
        """
        Compute the complex conjugate of a field element with caching.
        
        Args:
            a (Dict[int, float]): The field element.
        
        Returns:
            Dict[int, float]: The complex conjugate of the element.
        """
        # Convert the element to a hashable representation
        a_tuple = tuple(sorted((k, v) for k, v in a.items()))
        
        # Compute the cache key
        key = self.cache._compute_key("conjugate", a_tuple)
        
        # Check the cache
        cached_value = self.cache.get(key)
        if cached_value is not None:
            return cached_value
        
        # Compute the value
        value = self.cyclotomic_field.conjugate(a)
        
        # Cache the value
        self.cache.set(key, value)
        
        return value
    
    def norm(self, a: Dict[int, float]) -> float:
        """
        Compute the norm of a field element with caching.
        
        Args:
            a (Dict[int, float]): The field element.
        
        Returns:
            float: The norm of the element.
        """
        # Convert the element to a hashable representation
        a_tuple = tuple(sorted((k, v) for k, v in a.items()))
        
        # Compute the cache key
        key = self.cache._compute_key("norm", a_tuple)
        
        # Check the cache
        cached_value = self.cache.get(key)
        if cached_value is not None:
            return cached_value
        
        # Compute the value
        value = self.cyclotomic_field.norm(a)
        
        # Cache the value
        self.cache.set(key, value)
        
        return value
    
    def clear_cache(self):
        """
        Clear the cache.
        """
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, int]: The cache statistics.
        """
        return self.cache.get_stats()


class PersistentCache:
    """
    A persistent cache for expensive computations.
    
    This class provides methods to cache the results of expensive computations
    in a persistent storage, with support for JSON serialization.
    
    Attributes:
        cache_dir (Path): The directory for persistent caching.
        memory_cache (Dict[str, Any]): The in-memory cache.
        max_memory_entries (int): The maximum number of entries in the memory cache.
        cache_stats (Dict[str, int]): Statistics about cache hits and misses.
    """
    
    def __init__(self, cache_dir: str = ".cache/persistent", max_memory_entries: int = 1000):
        """
        Initialize a persistent cache.
        
        Args:
            cache_dir (str): The directory for persistent caching.
            max_memory_entries (int): The maximum number of entries in the memory cache.
        """
        self.cache_dir = Path(cache_dir)
        self.memory_cache = {}
        self.max_memory_entries = max_memory_entries
        self.cache_stats = {"hits": 0, "misses": 0, "memory_hits": 0, "disk_hits": 0}
        
        # Create the cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_key(self, *args, **kwargs) -> str:
        """
        Compute a cache key from the arguments.
        
        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        
        Returns:
            str: The cache key.
        """
        # Convert the arguments to a string representation
        args_str = str(args) + str(sorted(kwargs.items()))
        
        # Compute a hash of the string
        key = hashlib.md5(args_str.encode()).hexdigest()
        
        return key
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key (str): The cache key.
        
        Returns:
            Optional[Any]: The cached value, or None if not found.
        """
        # Check the memory cache
        if key in self.memory_cache:
            self.cache_stats["hits"] += 1
            self.cache_stats["memory_hits"] += 1
            return self.memory_cache[key]
        
        # Check the persistent cache
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    value = json.load(f)
                
                # Add to the memory cache
                self._add_to_memory_cache(key, value)
                
                self.cache_stats["hits"] += 1
                self.cache_stats["disk_hits"] += 1
                return value
            except (json.JSONDecodeError, EOFError):
                # If there's an error loading the cache, ignore it
                pass
        
        self.cache_stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any):
        """
        Set a value in the cache.
        
        Args:
            key (str): The cache key.
            value (Any): The value to cache.
        """
        # Add to the memory cache
        self._add_to_memory_cache(key, value)
        
        # Add to the persistent cache
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(value, f)
        except (TypeError, AttributeError):
            # If there's an error saving the cache, ignore it
            pass
    
    def _add_to_memory_cache(self, key: str, value: Any):
        """
        Add a value to the memory cache.
        
        Args:
            key (str): The cache key.
            value (Any): The value to cache.
        """
        # If the cache is full, remove the oldest entry
        if len(self.memory_cache) >= self.max_memory_entries:
            # Get the first key (oldest entry)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        # Add the new entry
        self.memory_cache[key] = value
    
    def clear(self):
        """
        Clear the cache.
        """
        # Clear the memory cache
        self.memory_cache.clear()
        
        # Clear the persistent cache
        for file in self.cache_dir.glob("*.json"):
            file.unlink()
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, int]: The cache statistics.
        """
        return self.cache_stats.copy()
    
    def cached(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator for caching function results.
        
        Args:
            func (Callable[..., Any]): The function to cache.
        
        Returns:
            Callable[..., Any]: The cached function.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Compute the cache key
            key = self._compute_key(func.__name__, *args, **kwargs)
            
            # Check the cache
            cached_value = self.get(key)
            if cached_value is not None:
                return cached_value
            
            # Compute the value
            value = func(*args, **kwargs)
            
            # Cache the value
            self.set(key, value)
            
            return value
        
        return wrapper


def memoize(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for memoizing function results.
    
    Args:
        func (Callable[..., Any]): The function to memoize.
    
    Returns:
        Callable[..., Any]: The memoized function.
    """
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a key from the function arguments
        key = str(args) + str(sorted(kwargs.items()))
        
        # Check if the result is already in the cache
        if key in cache:
            return cache[key]
        
        # Compute the result
        result = func(*args, **kwargs)
        
        # Cache the result
        cache[key] = result
        
        return result
    
    # Add a method to clear the cache
    def clear_cache():
        cache.clear()
    
    wrapper.clear_cache = clear_cache
    
    return wrapper


def timed_lru_cache(seconds: int, maxsize: int = 128):
    """
    Decorator for LRU caching with a time limit.
    
    Args:
        seconds (int): The time limit in seconds.
        maxsize (int): The maximum cache size.
    
    Returns:
        Callable: The decorator.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Create a cache for the results
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a key from the function arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            # Get the current time
            current_time = time.time()
            
            # Check if the result is already in the cache and not expired
            if key in cache and current_time - cache[key][1] < seconds:
                return cache[key][0]
            
            # Compute the result
            result = func(*args, **kwargs)
            
            # Cache the result with the current time
            cache[key] = (result, current_time)
            
            # If the cache is too large, remove the oldest entries
            if len(cache) > maxsize:
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
            
            return result
        
        # Add a method to clear the cache
        def clear_cache():
            cache.clear()
        
        wrapper.clear_cache = clear_cache
        
        return wrapper
    
    return decorator


def disk_cache(cache_dir: str = ".cache"):
    """
    Decorator for disk-based caching.
    
    Args:
        cache_dir (str): The directory for disk-based caching.
    
    Returns:
        Callable: The decorator.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Create the cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a key from the function arguments
            args_str = str(args) + str(sorted(kwargs.items()))
            key = hashlib.md5(args_str.encode()).hexdigest()
            
            # Check if the result is already in the cache
            cache_file = os.path.join(cache_dir, f"{func.__name__}_{key}.pickle")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
                except (pickle.PickleError, EOFError, AttributeError):
                    # If there's an error loading the cache, ignore it
                    pass
            
            # Compute the result
            result = func(*args, **kwargs)
            
            # Cache the result
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
            except (pickle.PickleError, TypeError, AttributeError):
                # If there's an error saving the cache, ignore it
                pass
            
            return result
        
        # Add a method to clear the cache
        def clear_cache():
            for file in os.listdir(cache_dir):
                if file.startswith(f"{func.__name__}_") and file.endswith(".pickle"):
                    os.remove(os.path.join(cache_dir, file))
        
        wrapper.clear_cache = clear_cache
        
        return wrapper
    
    return decorator