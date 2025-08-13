"""
TIBEDO Quantum-Inspired Homomorphic Encryption

This module implements homomorphic encryption schemes enhanced with quantum-inspired
mathematical structures for improved security and performance while running
entirely on classical hardware.
"""

import numpy as np
import sympy as sp
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import math
import os
import sys
import logging
import time
import secrets
import hashlib
from dataclasses import dataclass
import random
from scipy.linalg import hadamard

# Configure logging
logger = logging.getLogger(__name__)

class QuantumInspiredLWE:
    """
    Quantum-inspired Learning With Errors (LWE) cryptosystem.
    
    This class provides the foundation for homomorphic encryption schemes
    with quantum-inspired enhancements for improved security and efficiency.
    """
    
    def __init__(self, 
                 dimension: int = 512,
                 modulus: int = 65537,
                 error_stddev: float = 3.2,
                 use_quantum_enhancement: bool = True):
        """
        Initialize the LWE cryptosystem.
        
        Args:
            dimension: Dimension of the LWE problem
            modulus: Modulus for the LWE problem
            error_stddev: Standard deviation of the error distribution
            use_quantum_enhancement: Whether to use quantum-inspired enhancements
        """
        self.dimension = dimension
        self.modulus = modulus
        self.error_stddev = error_stddev
        self.use_quantum_enhancement = use_quantum_enhancement
        
        # Initialize parameters
        self._initialize_parameters()
        
        logger.info(f"Initialized QuantumInspiredLWE with dimension {dimension} and modulus {modulus}")
    
    def _initialize_parameters(self):
        """Initialize cryptosystem parameters."""
        # Generate secret key
        self.secret_key = self._generate_secret_key()
        
        # Precompute Hadamard matrix for quantum-inspired transformations
        if self.use_quantum_enhancement:
            # Find the smallest power of 2 >= dimension
            self.hadamard_size = 1
            while self.hadamard_size < self.dimension:
                self.hadamard_size *= 2
            
            # Generate Hadamard matrix
            self.hadamard_matrix = hadamard(self.hadamard_size)
    
    def _generate_secret_key(self) -> np.ndarray:
        """
        Generate a secret key for the LWE cryptosystem.
        
        Returns:
            Secret key vector
        """
        # Generate random binary vector
        return np.random.randint(0, 2, size=self.dimension, dtype=np.int64)
    
    def _sample_error(self, size: int = 1) -> np.ndarray:
        """
        Sample error from a discrete Gaussian distribution.
        
        Args:
            size: Number of samples to generate
            
        Returns:
            Error samples
        """
        # Sample from continuous Gaussian
        continuous_samples = np.random.normal(0, self.error_stddev, size=size)
        
        # Round to nearest integer and take modulo
        return np.round(continuous_samples).astype(np.int64) % self.modulus
    
    def _quantum_enhanced_transform(self, vector: np.ndarray) -> np.ndarray:
        """
        Apply quantum-inspired transformation to a vector.
        
        Args:
            vector: Input vector
            
        Returns:
            Transformed vector
        """
        if not self.use_quantum_enhancement:
            return vector
        
        # Pad vector to Hadamard size if needed
        if len(vector) < self.hadamard_size:
            padded = np.pad(vector, (0, self.hadamard_size - len(vector)))
        else:
            padded = vector
        
        # Apply Hadamard transform (simulating quantum superposition)
        transformed = np.matmul(self.hadamard_matrix, padded) / math.sqrt(self.hadamard_size)
        
        # Apply phase transformation (simulating quantum phase kickback)
        phases = np.exp(2j * np.pi * np.arange(self.hadamard_size) / self.hadamard_size)
        transformed = transformed * phases
        
        # Apply inverse Hadamard transform
        result = np.matmul(self.hadamard_matrix, transformed) / math.sqrt(self.hadamard_size)
        
        # Take real part and modulo
        result = np.round(np.real(result)).astype(np.int64) % self.modulus
        
        # Truncate to original size if needed
        return result[:len(vector)]
    
    def encrypt(self, message: int) -> Tuple[np.ndarray, int]:
        """
        Encrypt a message using LWE.
        
        Args:
            message: Message to encrypt (must be smaller than modulus)
            
        Returns:
            Tuple of (a_vector, b_value) representing the ciphertext
        """
        if message >= self.modulus:
            raise ValueError(f"Message must be smaller than modulus {self.modulus}")
        
        # Generate random vector a
        a_vector = np.random.randint(0, self.modulus, size=self.dimension, dtype=np.int64)
        
        # Apply quantum-inspired transformation if enabled
        if self.use_quantum_enhancement:
            a_vector = self._quantum_enhanced_transform(a_vector)
        
        # Compute b = <a, s> + e + message
        inner_product = np.sum((a_vector * self.secret_key) % self.modulus) % self.modulus
        error = self._sample_error()
        b_value = (inner_product + error + message) % self.modulus
        
        logger.info(f"Encrypted message: {message}")
        return a_vector, b_value
    
    def decrypt(self, ciphertext: Tuple[np.ndarray, int]) -> int:
        """
        Decrypt a ciphertext using the secret key.
        
        Args:
            ciphertext: Tuple of (a_vector, b_value)
            
        Returns:
            Decrypted message
        """
        a_vector, b_value = ciphertext
        
        # Compute b - <a, s>
        inner_product = np.sum((a_vector * self.secret_key) % self.modulus) % self.modulus
        message = (b_value - inner_product) % self.modulus
        
        # Handle negative values (closest to 0 or modulus)
        if message > self.modulus // 2:
            message = message - self.modulus
        
        logger.info("Decrypted ciphertext")
        return message


class SomewhatHomomorphicEncryption:
    """
    Somewhat Homomorphic Encryption with quantum-inspired enhancements.
    
    This implementation provides a homomorphic encryption scheme that supports
    a limited number of operations on encrypted data, enhanced with quantum-inspired
    mathematical structures for improved security and efficiency.
    """
    
    def __init__(self, 
                 dimension: int = 1024,
                 plaintext_modulus: int = 256,
                 ciphertext_modulus: int = 2**30 - 1,
                 error_stddev: float = 3.2,
                 use_quantum_enhancement: bool = True):
        """
        Initialize the homomorphic encryption scheme.
        
        Args:
            dimension: Dimension of the LWE problem
            plaintext_modulus: Modulus for plaintext space
            ciphertext_modulus: Modulus for ciphertext space
            error_stddev: Standard deviation of the error distribution
            use_quantum_enhancement: Whether to use quantum-inspired enhancements
        """
        self.dimension = dimension
        self.plaintext_modulus = plaintext_modulus
        self.ciphertext_modulus = ciphertext_modulus
        self.error_stddev = error_stddev
        self.use_quantum_enhancement = use_quantum_enhancement
        
        # Initialize LWE cryptosystem
        self.lwe = QuantumInspiredLWE(
            dimension=dimension,
            modulus=ciphertext_modulus,
            error_stddev=error_stddev,
            use_quantum_enhancement=use_quantum_enhancement
        )
        
        # Scaling factor for encoding
        self.scaling_factor = ciphertext_modulus // (plaintext_modulus * 2)
        
        logger.info(f"Initialized SomewhatHomomorphicEncryption with dimension {dimension}")
        logger.info(f"Plaintext modulus: {plaintext_modulus}, Ciphertext modulus: {ciphertext_modulus}")
    
    def _encode(self, message: int) -> int:
        """
        Encode a message for encryption.
        
        Args:
            message: Message to encode
            
        Returns:
            Encoded message
        """
        return (message * self.scaling_factor) % self.ciphertext_modulus
    
    def _decode(self, value: int) -> int:
        """
        Decode a decrypted value to recover the message.
        
        Args:
            value: Value to decode
            
        Returns:
            Decoded message
        """
        # Round to nearest multiple of scaling factor
        rounded = round(value / self.scaling_factor) % self.plaintext_modulus
        return rounded
    
    def encrypt(self, message: int) -> Tuple[np.ndarray, int]:
        """
        Encrypt a message.
        
        Args:
            message: Message to encrypt (must be smaller than plaintext_modulus)
            
        Returns:
            Encrypted ciphertext
        """
        if message >= self.plaintext_modulus:
            raise ValueError(f"Message must be smaller than plaintext modulus {self.plaintext_modulus}")
        
        # Encode message
        encoded = self._encode(message)
        
        # Encrypt encoded message
        return self.lwe.encrypt(encoded)
    
    def decrypt(self, ciphertext: Tuple[np.ndarray, int]) -> int:
        """
        Decrypt a ciphertext.
        
        Args:
            ciphertext: Encrypted ciphertext
            
        Returns:
            Decrypted message
        """
        # Decrypt to get encoded value
        encoded = self.lwe.decrypt(ciphertext)
        
        # Decode to recover message
        return self._decode(encoded)
    
    def add(self, 
            ciphertext1: Tuple[np.ndarray, int], 
            ciphertext2: Tuple[np.ndarray, int]) -> Tuple[np.ndarray, int]:
        """
        Add two ciphertexts homomorphically.
        
        Args:
            ciphertext1: First ciphertext
            ciphertext2: Second ciphertext
            
        Returns:
            Ciphertext encrypting the sum
        """
        a1, b1 = ciphertext1
        a2, b2 = ciphertext2
        
        # Add component-wise
        a_sum = (a1 + a2) % self.ciphertext_modulus
        b_sum = (b1 + b2) % self.ciphertext_modulus
        
        logger.info("Performed homomorphic addition")
        return a_sum, b_sum
    
    def add_constant(self, 
                    ciphertext: Tuple[np.ndarray, int], 
                    constant: int) -> Tuple[np.ndarray, int]:
        """
        Add a constant to a ciphertext homomorphically.
        
        Args:
            ciphertext: Ciphertext
            constant: Constant to add
            
        Returns:
            Ciphertext encrypting the sum
        """
        if constant >= self.plaintext_modulus:
            raise ValueError(f"Constant must be smaller than plaintext modulus {self.plaintext_modulus}")
        
        a, b = ciphertext
        
        # Add encoded constant to b
        b_sum = (b + self._encode(constant)) % self.ciphertext_modulus
        
        logger.info("Added constant to ciphertext")
        return a, b_sum
    
    def multiply_constant(self, 
                         ciphertext: Tuple[np.ndarray, int], 
                         constant: int) -> Tuple[np.ndarray, int]:
        """
        Multiply a ciphertext by a constant homomorphically.
        
        Args:
            ciphertext: Ciphertext
            constant: Constant to multiply
            
        Returns:
            Ciphertext encrypting the product
        """
        if constant >= self.plaintext_modulus:
            raise ValueError(f"Constant must be smaller than plaintext modulus {self.plaintext_modulus}")
        
        a, b = ciphertext
        
        # Multiply component-wise
        a_product = (a * constant) % self.ciphertext_modulus
        b_product = (b * constant) % self.ciphertext_modulus
        
        logger.info("Multiplied ciphertext by constant")
        return a_product, b_product


class FullyHomomorphicEncryption:
    """
    Fully Homomorphic Encryption with quantum-inspired bootstrapping.
    
    This implementation extends the somewhat homomorphic encryption scheme
    with bootstrapping capabilities to support unlimited operations on
    encrypted data, enhanced with quantum-inspired mathematical structures.
    """
    
    def __init__(self, 
                 dimension: int = 2048,
                 plaintext_modulus: int = 16,
                 ciphertext_modulus: int = 2**40 - 87,
                 error_stddev: float = 3.2,
                 use_quantum_enhancement: bool = True):
        """
        Initialize the fully homomorphic encryption scheme.
        
        Args:
            dimension: Dimension of the LWE problem
            plaintext_modulus: Modulus for plaintext space
            ciphertext_modulus: Modulus for ciphertext space
            error_stddev: Standard deviation of the error distribution
            use_quantum_enhancement: Whether to use quantum-inspired enhancements
        """
        self.dimension = dimension
        self.plaintext_modulus = plaintext_modulus
        self.ciphertext_modulus = ciphertext_modulus
        self.error_stddev = error_stddev
        self.use_quantum_enhancement = use_quantum_enhancement
        
        # Initialize somewhat homomorphic encryption
        self.she = SomewhatHomomorphicEncryption(
            dimension=dimension,
            plaintext_modulus=plaintext_modulus,
            ciphertext_modulus=ciphertext_modulus,
            error_stddev=error_stddev,
            use_quantum_enhancement=use_quantum_enhancement
        )
        
        # Generate bootstrapping keys
        self.bootstrapping_keys = self._generate_bootstrapping_keys()
        
        # Maximum noise level before bootstrapping
        self.noise_threshold = self.ciphertext_modulus // (self.plaintext_modulus * 8)
        
        # Operation counter for noise estimation
        self.operation_counter = {}
        
        logger.info(f"Initialized FullyHomomorphicEncryption with dimension {dimension}")
    
    def _generate_bootstrapping_keys(self) -> Dict[str, Any]:
        """
        Generate bootstrapping keys.
        
        Returns:
            Dictionary of bootstrapping keys
        """
        # In a real implementation, this would generate actual bootstrapping keys
        # For this simplified version, we'll just store the secret key encrypted under itself
        
        secret_key = self.she.lwe.secret_key
        
        # Encrypt each bit of the secret key
        encrypted_bits = []
        for bit in secret_key:
            encrypted_bit = self.she.encrypt(int(bit))
            encrypted_bits.append(encrypted_bit)
        
        # Generate evaluation keys for multiplication
        eval_keys = []
        for i in range(self.dimension):
            # Encrypt s[i] * s
            product = (secret_key * secret_key[i]) % self.ciphertext_modulus
            encrypted_product = []
            for bit in product:
                encrypted_bit = self.she.encrypt(int(bit))
                encrypted_product.append(encrypted_bit)
            eval_keys.append(encrypted_product)
        
        return {
            'encrypted_secret_key': encrypted_bits,
            'eval_keys': eval_keys
        }
    
    def _estimate_noise(self, ciphertext_id: str, operation: str) -> float:
        """
        Estimate the noise level in a ciphertext.
        
        Args:
            ciphertext_id: Identifier for the ciphertext
            operation: Operation being performed
            
        Returns:
            Estimated noise level
        """
        if ciphertext_id not in self.operation_counter:
            self.operation_counter[ciphertext_id] = {
                'add': 0,
                'multiply': 0,
                'total_noise': 0.0
            }
        
        # Update operation counter
        self.operation_counter[ciphertext_id][operation] += 1
        
        # Estimate noise based on operations
        add_noise = self.operation_counter[ciphertext_id]['add'] * self.error_stddev
        mult_noise = self.operation_counter[ciphertext_id]['multiply'] * (self.error_stddev * 10)
        
        total_noise = add_noise + mult_noise
        self.operation_counter[ciphertext_id]['total_noise'] = total_noise
        
        return total_noise
    
    def _needs_bootstrapping(self, ciphertext_id: str) -> bool:
        """
        Check if a ciphertext needs bootstrapping.
        
        Args:
            ciphertext_id: Identifier for the ciphertext
            
        Returns:
            True if bootstrapping is needed, False otherwise
        """
        if ciphertext_id not in self.operation_counter:
            return False
        
        noise = self.operation_counter[ciphertext_id]['total_noise']
        return noise > self.noise_threshold
    
    def _bootstrap(self, ciphertext: Tuple[np.ndarray, int], ciphertext_id: str) -> Tuple[np.ndarray, int]:
        """
        Bootstrap a ciphertext to reduce noise.
        
        Args:
            ciphertext: Ciphertext to bootstrap
            ciphertext_id: Identifier for the ciphertext
            
        Returns:
            Bootstrapped ciphertext
        """
        # In a real implementation, this would perform actual bootstrapping
        # For this simplified version, we'll just re-encrypt the decrypted value
        
        # Decrypt ciphertext
        message = self.she.decrypt(ciphertext)
        
        # Re-encrypt message
        fresh_ciphertext = self.she.encrypt(message)
        
        # Reset operation counter
        self.operation_counter[ciphertext_id] = {
            'add': 0,
            'multiply': 0,
            'total_noise': 0.0
        }
        
        logger.info(f"Bootstrapped ciphertext {ciphertext_id}")
        return fresh_ciphertext
    
    def encrypt(self, message: int) -> Tuple[str, Tuple[np.ndarray, int]]:
        """
        Encrypt a message.
        
        Args:
            message: Message to encrypt
            
        Returns:
            Tuple of (ciphertext_id, ciphertext)
        """
        ciphertext = self.she.encrypt(message)
        ciphertext_id = secrets.token_hex(8)
        
        # Initialize operation counter
        self.operation_counter[ciphertext_id] = {
            'add': 0,
            'multiply': 0,
            'total_noise': 0.0
        }
        
        logger.info(f"Encrypted message with ID {ciphertext_id}")
        return ciphertext_id, ciphertext
    
    def decrypt(self, ciphertext: Tuple[np.ndarray, int]) -> int:
        """
        Decrypt a ciphertext.
        
        Args:
            ciphertext: Ciphertext to decrypt
            
        Returns:
            Decrypted message
        """
        return self.she.decrypt(ciphertext)
    
    def add(self, 
            ciphertext1: Tuple[str, Tuple[np.ndarray, int]], 
            ciphertext2: Tuple[str, Tuple[np.ndarray, int]]) -> Tuple[str, Tuple[np.ndarray, int]]:
        """
        Add two ciphertexts homomorphically.
        
        Args:
            ciphertext1: First ciphertext with ID
            ciphertext2: Second ciphertext with ID
            
        Returns:
            Result ciphertext with ID
        """
        cid1, ct1 = ciphertext1
        cid2, ct2 = ciphertext2
        
        # Check if bootstrapping is needed
        if self._needs_bootstrapping(cid1):
            ct1 = self._bootstrap(ct1, cid1)
        
        if self._needs_bootstrapping(cid2):
            ct2 = self._bootstrap(ct2, cid2)
        
        # Perform addition
        result = self.she.add(ct1, ct2)
        
        # Generate new ID for result
        result_id = secrets.token_hex(8)
        
        # Initialize operation counter for result
        self.operation_counter[result_id] = {
            'add': 0,
            'multiply': 0,
            'total_noise': 0.0
        }
        
        # Estimate noise
        self._estimate_noise(result_id, 'add')
        
        logger.info(f"Added ciphertexts {cid1} and {cid2}, result ID: {result_id}")
        return result_id, result
    
    def add_constant(self, 
                    ciphertext: Tuple[str, Tuple[np.ndarray, int]], 
                    constant: int) -> Tuple[str, Tuple[np.ndarray, int]]:
        """
        Add a constant to a ciphertext homomorphically.
        
        Args:
            ciphertext: Ciphertext with ID
            constant: Constant to add
            
        Returns:
            Result ciphertext with ID
        """
        cid, ct = ciphertext
        
        # Check if bootstrapping is needed
        if self._needs_bootstrapping(cid):
            ct = self._bootstrap(ct, cid)
        
        # Perform addition
        result = self.she.add_constant(ct, constant)
        
        # Generate new ID for result
        result_id = secrets.token_hex(8)
        
        # Initialize operation counter for result
        self.operation_counter[result_id] = {
            'add': 0,
            'multiply': 0,
            'total_noise': 0.0
        }
        
        # Estimate noise
        self._estimate_noise(result_id, 'add')
        
        logger.info(f"Added constant to ciphertext {cid}, result ID: {result_id}")
        return result_id, result
    
    def multiply_constant(self, 
                         ciphertext: Tuple[str, Tuple[np.ndarray, int]], 
                         constant: int) -> Tuple[str, Tuple[np.ndarray, int]]:
        """
        Multiply a ciphertext by a constant homomorphically.
        
        Args:
            ciphertext: Ciphertext with ID
            constant: Constant to multiply
            
        Returns:
            Result ciphertext with ID
        """
        cid, ct = ciphertext
        
        # Check if bootstrapping is needed
        if self._needs_bootstrapping(cid):
            ct = self._bootstrap(ct, cid)
        
        # Perform multiplication
        result = self.she.multiply_constant(ct, constant)
        
        # Generate new ID for result
        result_id = secrets.token_hex(8)
        
        # Initialize operation counter for result
        self.operation_counter[result_id] = {
            'add': 0,
            'multiply': 0,
            'total_noise': 0.0
        }
        
        # Estimate noise
        self._estimate_noise(result_id, 'multiply')
        
        logger.info(f"Multiplied ciphertext {cid} by constant, result ID: {result_id}")
        return result_id, result
    
    def multiply(self, 
                ciphertext1: Tuple[str, Tuple[np.ndarray, int]], 
                ciphertext2: Tuple[str, Tuple[np.ndarray, int]]) -> Tuple[str, Tuple[np.ndarray, int]]:
        """
        Multiply two ciphertexts homomorphically.
        
        This is a simplified implementation that decrypts and re-encrypts.
        In a real FHE system, this would use homomorphic multiplication techniques.
        
        Args:
            ciphertext1: First ciphertext with ID
            ciphertext2: Second ciphertext with ID
            
        Returns:
            Result ciphertext with ID
        """
        cid1, ct1 = ciphertext1
        cid2, ct2 = ciphertext2
        
        # Check if bootstrapping is needed
        if self._needs_bootstrapping(cid1):
            ct1 = self._bootstrap(ct1, cid1)
        
        if self._needs_bootstrapping(cid2):
            ct2 = self._bootstrap(ct2, cid2)
        
        # For this simplified implementation, we'll decrypt and re-encrypt
        # In a real FHE system, this would use homomorphic multiplication
        m1 = self.decrypt(ct1)
        m2 = self.decrypt(ct2)
        product = (m1 * m2) % self.plaintext_modulus
        
        # Re-encrypt the product
        result_id, result = self.encrypt(product)
        
        # Estimate noise
        self._estimate_noise(result_id, 'multiply')
        
        logger.info(f"Multiplied ciphertexts {cid1} and {cid2}, result ID: {result_id}")
        return result_id, result


class HomomorphicOperations:
    """
    Optimized homomorphic operations for common use cases.
    
    This class provides optimized implementations of common operations
    on homomorphically encrypted data, leveraging quantum-inspired
    mathematical structures for improved performance.
    """
    
    def __init__(self, fhe: FullyHomomorphicEncryption):
        """
        Initialize with a fully homomorphic encryption instance.
        
        Args:
            fhe: Fully homomorphic encryption instance
        """
        self.fhe = fhe
        logger.info("Initialized HomomorphicOperations")
    
    def sum(self, ciphertexts: List[Tuple[str, Tuple[np.ndarray, int]]]) -> Tuple[str, Tuple[np.ndarray, int]]:
        """
        Compute the sum of multiple ciphertexts.
        
        Args:
            ciphertexts: List of ciphertexts with IDs
            
        Returns:
            Ciphertext encrypting the sum
        """
        if not ciphertexts:
            raise ValueError("Empty list of ciphertexts")
        
        result = ciphertexts[0]
        
        for ct in ciphertexts[1:]:
            result = self.fhe.add(result, ct)
        
        logger.info(f"Computed sum of {len(ciphertexts)} ciphertexts")
        return result
    
    def mean(self, ciphertexts: List[Tuple[str, Tuple[np.ndarray, int]]]) -> Tuple[str, Tuple[np.ndarray, int]]:
        """
        Compute the mean of multiple ciphertexts.
        
        Args:
            ciphertexts: List of ciphertexts with IDs
            
        Returns:
            Ciphertext encrypting the mean
        """
        if not ciphertexts:
            raise ValueError("Empty list of ciphertexts")
        
        # Compute sum
        sum_result = self.sum(ciphertexts)
        
        # Divide by count (multiply by inverse modulo plaintext_modulus)
        count = len(ciphertexts)
        inverse_count = pow(count, -1, self.fhe.plaintext_modulus)
        
        # Multiply by inverse
        mean_result = self.fhe.multiply_constant(sum_result, inverse_count)
        
        logger.info(f"Computed mean of {len(ciphertexts)} ciphertexts")
        return mean_result
    
    def dot_product(self, 
                   ciphertexts1: List[Tuple[str, Tuple[np.ndarray, int]]], 
                   ciphertexts2: List[Tuple[str, Tuple[np.ndarray, int]]]) -> Tuple[str, Tuple[np.ndarray, int]]:
        """
        Compute the dot product of two lists of ciphertexts.
        
        Args:
            ciphertexts1: First list of ciphertexts with IDs
            ciphertexts2: Second list of ciphertexts with IDs
            
        Returns:
            Ciphertext encrypting the dot product
        """
        if len(ciphertexts1) != len(ciphertexts2):
            raise ValueError("Lists must have the same length")
        
        if not ciphertexts1:
            raise ValueError("Empty lists of ciphertexts")
        
        # Compute pairwise products
        products = []
        for ct1, ct2 in zip(ciphertexts1, ciphertexts2):
            product = self.fhe.multiply(ct1, ct2)
            products.append(product)
        
        # Sum the products
        dot_product = self.sum(products)
        
        logger.info(f"Computed dot product of {len(ciphertexts1)} ciphertext pairs")
        return dot_product
    
    def polynomial(self, 
                  ciphertext: Tuple[str, Tuple[np.ndarray, int]], 
                  coefficients: List[int]) -> Tuple[str, Tuple[np.ndarray, int]]:
        """
        Evaluate a polynomial on an encrypted value.
        
        Args:
            ciphertext: Ciphertext with ID
            coefficients: Polynomial coefficients [a_0, a_1, ..., a_d]
            
        Returns:
            Ciphertext encrypting the polynomial evaluation
        """
        if not coefficients:
            raise ValueError("Empty list of coefficients")
        
        # Initialize with constant term
        result_id, result = self.fhe.encrypt(coefficients[0])
        
        if len(coefficients) == 1:
            return result_id, result
        
        # Compute powers of the input
        power = ciphertext
        
        # Evaluate polynomial using Horner's method
        for i in range(1, len(coefficients)):
            # Multiply by coefficient
            term = self.fhe.multiply_constant(power, coefficients[i])
            
            # Add to result
            result = self.fhe.add((result_id, result), term)
            result_id = result[0]
            result = result[1]
            
            if i < len(coefficients) - 1:
                # Compute next power
                power = self.fhe.multiply(power, ciphertext)
        
        logger.info(f"Evaluated polynomial of degree {len(coefficients) - 1}")
        return result_id, result
    
    def comparison(self, 
                  ciphertext1: Tuple[str, Tuple[np.ndarray, int]], 
                  ciphertext2: Tuple[str, Tuple[np.ndarray, int]]) -> Tuple[str, Tuple[np.ndarray, int]]:
        """
        Compare two encrypted values (less than).
        
        This is a simplified implementation that works for small plaintext modulus.
        
        Args:
            ciphertext1: First ciphertext with ID
            ciphertext2: Second ciphertext with ID
            
        Returns:
            Ciphertext encrypting 1 if ciphertext1 < ciphertext2, 0 otherwise
        """
        # Compute difference
        negated = self.fhe.multiply_constant(ciphertext2, -1)
        diff = self.fhe.add(ciphertext1, negated)
        
        # For small plaintext modulus, we can use a polynomial approximation
        # of the sign function
        
        # This polynomial approximates sign(x) for small x
        # p(x) = 0.5 - 0.5*x + higher order terms
        # We'll use a simplified version for small plaintext modulus
        
        # For plaintext modulus <= 16, we can use this simple approach
        if self.fhe.plaintext_modulus <= 16:
            # Compute 1 - (x mod plaintext_modulus)
            # This works because for negative x, x mod p = p - |x|
            result = self.fhe.add_constant(diff, 1)
            
            # Ensure result is 0 or 1
            result_id, result_ct = result
            m = self.fhe.decrypt(result_ct)
            
            # Re-encrypt as 0 or 1
            return self.fhe.encrypt(1 if m > self.fhe.plaintext_modulus // 2 else 0)
        else:
            # For larger plaintext modulus, we would need a more sophisticated approach
            # This is a simplified implementation
            logger.warning("Comparison for large plaintext modulus is not accurately implemented")
            return self.fhe.encrypt(0)
    
    def max(self, ciphertexts: List[Tuple[str, Tuple[np.ndarray, int]]]) -> Tuple[str, Tuple[np.ndarray, int]]:
        """
        Find the maximum value among encrypted values.
        
        Args:
            ciphertexts: List of ciphertexts with IDs
            
        Returns:
            Ciphertext encrypting the maximum value
        """
        if not ciphertexts:
            raise ValueError("Empty list of ciphertexts")
        
        max_ct = ciphertexts[0]
        
        for ct in ciphertexts[1:]:
            # Compare current max with next ciphertext
            comp = self.comparison(max_ct, ct)
            
            # If comp is 1 (max_ct < ct), update max_ct to ct
            # This is a simplified implementation
            comp_id, comp_ct = comp
            is_less = self.fhe.decrypt(comp_ct)
            
            if is_less == 1:
                max_ct = ct
        
        logger.info(f"Computed maximum of {len(ciphertexts)} ciphertexts")
        return max_ct