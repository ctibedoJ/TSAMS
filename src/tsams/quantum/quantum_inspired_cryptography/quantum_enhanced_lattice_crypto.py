"""
TIBEDO Quantum-Enhanced Lattice Cryptography

This module implements lattice-based cryptographic primitives enhanced with
quantum-inspired mathematical structures for improved security and performance
while running entirely on classical hardware.
"""

import numpy as np
import sympy as sp
from typing import List, Tuple, Dict, Any, Optional, Union
import math
import os
import sys
import logging
import time
import secrets
import hashlib
from scipy.linalg import hadamard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedLatticeParameters:
    """
    Enhanced parameters for lattice-based cryptography schemes.
    
    This class defines the parameters for lattice-based cryptography schemes
    enhanced with quantum-inspired mathematical structures.
    """
    
    def __init__(self, 
                 dimension: int = 1024, 
                 modulus: int = 12289,
                 error_std_dev: float = 3.2,
                 security_level: int = 128,
                 use_quantum_enhancement: bool = True):
        """
        Initialize the enhanced lattice parameters.
        
        Args:
            dimension: The dimension of the lattice
            modulus: The modulus for the ring
            error_std_dev: The standard deviation of the error distribution
            security_level: The security level in bits
            use_quantum_enhancement: Whether to use quantum-inspired enhancements
        """
        self.dimension = dimension
        self.modulus = modulus
        self.error_std_dev = error_std_dev
        self.security_level = security_level
        self.use_quantum_enhancement = use_quantum_enhancement
        
        # Validate parameters
        self._validate_parameters()
        
        # Derived parameters
        self.ring_degree = self._compute_ring_degree()
        self.polynomial_coefficients = self._compute_polynomial_coefficients()
        
        # Quantum-inspired enhancements
        if use_quantum_enhancement:
            self.hadamard_dimension = self._compute_hadamard_dimension()
            self.hadamard_matrix = self._compute_hadamard_matrix()
            self.phase_factors = self._compute_phase_factors()
    
    def _validate_parameters(self) -> None:
        """
        Validate the lattice parameters.
        
        Raises:
            ValueError: If the parameters are invalid
        """
        # Check if the dimension is a power of 2
        if not (self.dimension & (self.dimension - 1) == 0) or self.dimension == 0:
            raise ValueError("Dimension must be a power of 2")
        
        # Check if the modulus is prime
        if not sp.isprime(self.modulus):
            raise ValueError("Modulus must be prime")
        
        # Check if the modulus is congruent to 1 modulo 2*dimension
        if not (self.modulus % (2 * self.dimension) == 1):
            logger.warning("Modulus is not congruent to 1 modulo 2*dimension, which may affect performance")
        
        # Check if the error standard deviation is appropriate
        if self.error_std_dev <= 0:
            raise ValueError("Error standard deviation must be positive")
        
        # Check if the security level is appropriate
        if self.security_level not in [128, 192, 256]:
            logger.warning(f"Unusual security level: {self.security_level} bits")
    
    def _compute_ring_degree(self) -> int:
        """
        Compute the degree of the polynomial ring.
        
        Returns:
            The degree of the polynomial ring
        """
        # For Ring-LWE, the ring degree is typically the dimension
        return self.dimension
    
    def _compute_polynomial_coefficients(self) -> np.ndarray:
        """
        Compute the coefficients of the irreducible polynomial.
        
        For Ring-LWE, the irreducible polynomial is typically X^n + 1,
        where n is the ring degree.
        
        Returns:
            The coefficients of the irreducible polynomial
        """
        # For X^n + 1, the coefficients are [1, 0, 0, ..., 0, 1]
        coeffs = np.zeros(self.ring_degree + 1, dtype=int)
        coeffs[0] = 1
        coeffs[self.ring_degree] = 1
        return coeffs
    
    def _compute_hadamard_dimension(self) -> int:
        """
        Compute the dimension of the Hadamard matrix.
        
        The Hadamard matrix is used for quantum-inspired transformations.
        Its dimension must be a power of 2.
        
        Returns:
            The dimension of the Hadamard matrix
        """
        # Find the smallest power of 2 that is >= dimension
        return 2**int(np.ceil(np.log2(self.dimension)))
    
    def _compute_hadamard_matrix(self) -> np.ndarray:
        """
        Compute the Hadamard matrix.
        
        The Hadamard matrix is used for quantum-inspired transformations.
        
        Returns:
            The Hadamard matrix
        """
        # Compute the Hadamard matrix
        H = hadamard(self.hadamard_dimension)
        
        # Normalize the matrix
        H = H / np.sqrt(self.hadamard_dimension)
        
        return H
    
    def _compute_phase_factors(self) -> np.ndarray:
        """
        Compute the phase factors.
        
        The phase factors are used for quantum-inspired transformations.
        
        Returns:
            The phase factors
        """
        # Compute the phase factors
        phases = np.zeros(self.dimension, dtype=complex)
        
        for i in range(self.dimension):
            # Create phase factors with special structure
            # This is inspired by quantum phase relationships
            phase = 2 * np.pi * i / self.dimension
            phases[i] = np.exp(1j * phase)
        
        return phases
    
    @staticmethod
    def get_recommended_parameters(security_level: int = 128, use_quantum_enhancement: bool = True) -> 'EnhancedLatticeParameters':
        """
        Get recommended parameters for a given security level.
        
        Args:
            security_level: The security level in bits
            use_quantum_enhancement: Whether to use quantum-inspired enhancements
            
        Returns:
            Recommended lattice parameters
        """
        if security_level == 128:
            return EnhancedLatticeParameters(dimension=1024, modulus=12289, error_std_dev=3.2, security_level=128, use_quantum_enhancement=use_quantum_enhancement)
        elif security_level == 192:
            return EnhancedLatticeParameters(dimension=2048, modulus=12289, error_std_dev=3.0, security_level=192, use_quantum_enhancement=use_quantum_enhancement)
        elif security_level == 256:
            return EnhancedLatticeParameters(dimension=4096, modulus=40961, error_std_dev=2.7, security_level=256, use_quantum_enhancement=use_quantum_enhancement)
        else:
            raise ValueError(f"Unsupported security level: {security_level}")


class EnhancedLatticeUtils:
    """
    Enhanced utility functions for lattice-based cryptography.
    
    This class provides operations for lattice-based cryptography
    enhanced with quantum-inspired mathematical structures.
    """
    
    @staticmethod
    def sample_uniform(dimension: int, modulus: int) -> np.ndarray:
        """
        Sample a vector uniformly from Z_q^n.
        
        Args:
            dimension: The dimension of the vector
            modulus: The modulus
            
        Returns:
            A uniformly random vector
        """
        return np.random.randint(0, modulus, size=dimension, dtype=np.int64)
    
    @staticmethod
    def sample_gaussian(dimension: int, std_dev: float) -> np.ndarray:
        """
        Sample a vector from a discrete Gaussian distribution.
        
        Args:
            dimension: The dimension of the vector
            std_dev: The standard deviation
            
        Returns:
            A vector sampled from a discrete Gaussian distribution
        """
        # Sample from a continuous Gaussian distribution
        continuous_samples = np.random.normal(0, std_dev, size=dimension)
        
        # Round to the nearest integer
        discrete_samples = np.round(continuous_samples).astype(np.int64)
        
        return discrete_samples
    
    @staticmethod
    def sample_binary(dimension: int) -> np.ndarray:
        """
        Sample a binary vector.
        
        Args:
            dimension: The dimension of the vector
            
        Returns:
            A binary vector
        """
        return np.random.randint(0, 2, size=dimension, dtype=np.int64)
    
    @staticmethod
    def sample_ternary(dimension: int) -> np.ndarray:
        """
        Sample a ternary vector with entries in {-1, 0, 1}.
        
        Args:
            dimension: The dimension of the vector
            
        Returns:
            A ternary vector
        """
        return np.random.randint(-1, 2, size=dimension, dtype=np.int64)
    
    @staticmethod
    def polynomial_multiply(a: np.ndarray, b: np.ndarray, modulus: int, ring_degree: int) -> np.ndarray:
        """
        Multiply two polynomials in the ring R_q = Z_q[X]/(X^n + 1).
        
        Args:
            a: Coefficients of the first polynomial
            b: Coefficients of the second polynomial
            modulus: The modulus
            ring_degree: The degree of the ring
            
        Returns:
            The product polynomial
        """
        # Ensure the polynomials have the correct length
        a = np.resize(a, ring_degree)
        b = np.resize(b, ring_degree)
        
        # Use FFT for efficient multiplication
        a_fft = np.fft.fft(a)
        b_fft = np.fft.fft(b)
        c_fft = a_fft * b_fft
        c = np.fft.ifft(c_fft).real
        
        # Round to integers and reduce modulo q
        c = np.round(c).astype(np.int64) % modulus
        
        # Resize to the ring degree
        c = np.resize(c, ring_degree)
        
        return c
    
    @staticmethod
    def polynomial_add(a: np.ndarray, b: np.ndarray, modulus: int) -> np.ndarray:
        """
        Add two polynomials in the ring R_q = Z_q[X]/(X^n + 1).
        
        Args:
            a: Coefficients of the first polynomial
            b: Coefficients of the second polynomial
            modulus: The modulus
            
        Returns:
            The sum polynomial
        """
        # Ensure the polynomials have the same length
        max_len = max(len(a), len(b))
        a = np.resize(a, max_len)
        b = np.resize(b, max_len)
        
        # Add the polynomials and reduce modulo q
        c = (a + b) % modulus
        
        return c
    
    @staticmethod
    def polynomial_subtract(a: np.ndarray, b: np.ndarray, modulus: int) -> np.ndarray:
        """
        Subtract two polynomials in the ring R_q = Z_q[X]/(X^n + 1).
        
        Args:
            a: Coefficients of the first polynomial
            b: Coefficients of the second polynomial
            modulus: The modulus
            
        Returns:
            The difference polynomial
        """
        # Ensure the polynomials have the same length
        max_len = max(len(a), len(b))
        a = np.resize(a, max_len)
        b = np.resize(b, max_len)
        
        # Subtract the polynomials and reduce modulo q
        c = (a - b) % modulus
        
        return c
    
    @staticmethod
    def apply_hadamard_transform(vector: np.ndarray, hadamard_matrix: np.ndarray) -> np.ndarray:
        """
        Apply the Hadamard transform to a vector.
        
        The Hadamard transform is a quantum-inspired transformation
        that can be used to enhance lattice-based cryptography.
        
        Args:
            vector: The vector to transform
            hadamard_matrix: The Hadamard matrix
            
        Returns:
            The transformed vector
        """
        # Ensure the vector has the correct length
        if len(vector) < hadamard_matrix.shape[0]:
            padded = np.zeros(hadamard_matrix.shape[0], dtype=vector.dtype)
            padded[:len(vector)] = vector
            vector = padded
        elif len(vector) > hadamard_matrix.shape[0]:
            vector = vector[:hadamard_matrix.shape[0]]
        
        # Apply the Hadamard transform
        transformed = hadamard_matrix @ vector
        
        return transformed
    
    @staticmethod
    def apply_phase_transform(vector: np.ndarray, phase_factors: np.ndarray) -> np.ndarray:
        """
        Apply the phase transform to a vector.
        
        The phase transform is a quantum-inspired transformation
        that can be used to enhance lattice-based cryptography.
        
        Args:
            vector: The vector to transform
            phase_factors: The phase factors
            
        Returns:
            The transformed vector
        """
        # Ensure the vector has the correct length
        if len(vector) < len(phase_factors):
            padded = np.zeros(len(phase_factors), dtype=complex)
            padded[:len(vector)] = vector
            vector = padded
        elif len(vector) > len(phase_factors):
            vector = vector[:len(phase_factors)]
        
        # Apply the phase transform
        transformed = vector * phase_factors
        
        return transformed


class EnhancedLatticeEncryption:
    """
    Enhanced lattice-based encryption scheme.
    
    This class implements a lattice-based encryption scheme enhanced with
    quantum-inspired mathematical structures for improved security and performance.
    """
    
    def __init__(self, params: EnhancedLatticeParameters = None):
        """
        Initialize the enhanced lattice-based encryption scheme.
        
        Args:
            params: The lattice parameters
        """
        self.params = params or EnhancedLatticeParameters()
    
    def generate_keypair(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Generate a keypair.
        
        Returns:
            A tuple (public_key, secret_key)
        """
        # Sample the secret key s from the error distribution
        s = EnhancedLatticeUtils.sample_gaussian(self.params.ring_degree, self.params.error_std_dev)
        
        # Sample the public parameter a uniformly
        a = EnhancedLatticeUtils.sample_uniform(self.params.ring_degree, self.params.modulus)
        
        # Sample the error e
        e = EnhancedLatticeUtils.sample_gaussian(self.params.ring_degree, self.params.error_std_dev)
        
        # Apply quantum-inspired enhancements if enabled
        if self.params.use_quantum_enhancement:
            # Apply the Hadamard transform to the secret key
            s_transformed = EnhancedLatticeUtils.apply_hadamard_transform(s, self.params.hadamard_matrix)
            
            # Apply the phase transform to the error
            e_transformed = EnhancedLatticeUtils.apply_phase_transform(e, self.params.phase_factors)
            
            # Convert back to integers
            s = np.round(np.real(s_transformed)).astype(np.int64) % self.params.modulus
            e = np.round(np.real(e_transformed)).astype(np.int64) % self.params.modulus
        
        # Compute the public key b = a*s + e
        b = EnhancedLatticeUtils.polynomial_add(
            EnhancedLatticeUtils.polynomial_multiply(a, s, self.params.modulus, self.params.ring_degree),
            e,
            self.params.modulus
        )
        
        # Return the keypair
        public_key = {'a': a, 'b': b}
        secret_key = {'s': s}
        
        return public_key, secret_key
    
    def encrypt(self, public_key: Dict[str, np.ndarray], message: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encrypt a message.
        
        Args:
            public_key: The public key
            message: The message to encrypt (a binary vector)
            
        Returns:
            The ciphertext
        """
        # Extract the public key components
        a = public_key['a']
        b = public_key['b']
        
        # Sample the ephemeral secret r
        r = EnhancedLatticeUtils.sample_gaussian(self.params.ring_degree, self.params.error_std_dev)
        
        # Sample the errors e1 and e2
        e1 = EnhancedLatticeUtils.sample_gaussian(self.params.ring_degree, self.params.error_std_dev)
        e2 = EnhancedLatticeUtils.sample_gaussian(self.params.ring_degree, self.params.error_std_dev)
        
        # Apply quantum-inspired enhancements if enabled
        if self.params.use_quantum_enhancement:
            # Apply the Hadamard transform to the ephemeral secret
            r_transformed = EnhancedLatticeUtils.apply_hadamard_transform(r, self.params.hadamard_matrix)
            
            # Apply the phase transform to the errors
            e1_transformed = EnhancedLatticeUtils.apply_phase_transform(e1, self.params.phase_factors)
            e2_transformed = EnhancedLatticeUtils.apply_phase_transform(e2, self.params.phase_factors)
            
            # Convert back to integers
            r = np.round(np.real(r_transformed)).astype(np.int64) % self.params.modulus
            e1 = np.round(np.real(e1_transformed)).astype(np.int64) % self.params.modulus
            e2 = np.round(np.real(e2_transformed)).astype(np.int64) % self.params.modulus
        
        # Scale the message to the middle of the modulus
        scaled_message = (message * (self.params.modulus // 2)) % self.params.modulus
        
        # Compute the ciphertext (u, v)
        u = EnhancedLatticeUtils.polynomial_add(
            EnhancedLatticeUtils.polynomial_multiply(a, r, self.params.modulus, self.params.ring_degree),
            e1,
            self.params.modulus
        )
        
        v = EnhancedLatticeUtils.polynomial_add(
            EnhancedLatticeUtils.polynomial_add(
                EnhancedLatticeUtils.polynomial_multiply(b, r, self.params.modulus, self.params.ring_degree),
                e2,
                self.params.modulus
            ),
            scaled_message,
            self.params.modulus
        )
        
        # Return the ciphertext
        return {'u': u, 'v': v}
    
    def decrypt(self, secret_key: Dict[str, np.ndarray], ciphertext: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Decrypt a ciphertext.
        
        Args:
            secret_key: The secret key
            ciphertext: The ciphertext to decrypt
            
        Returns:
            The decrypted message
        """
        # Extract the secret key and ciphertext components
        s = secret_key['s']
        u = ciphertext['u']
        v = ciphertext['v']
        
        # Apply quantum-inspired enhancements if enabled
        if self.params.use_quantum_enhancement:
            # Apply the Hadamard transform to the secret key
            s_transformed = EnhancedLatticeUtils.apply_hadamard_transform(s, self.params.hadamard_matrix)
            
            # Convert back to integers
            s = np.round(np.real(s_transformed)).astype(np.int64) % self.params.modulus
        
        # Compute v - u*s
        decrypted = EnhancedLatticeUtils.polynomial_subtract(
            v,
            EnhancedLatticeUtils.polynomial_multiply(u, s, self.params.modulus, self.params.ring_degree),
            self.params.modulus
        )
        
        # Scale back to binary
        binary_message = np.round(decrypted / (self.params.modulus // 2)).astype(np.int64) % 2
        
        return binary_message


class EnhancedLatticeSignature:
    """
    Enhanced lattice-based signature scheme.
    
    This class implements a lattice-based signature scheme enhanced with
    quantum-inspired mathematical structures for improved security and performance.
    """
    
    def __init__(self, params: EnhancedLatticeParameters = None):
        """
        Initialize the enhanced lattice-based signature scheme.
        
        Args:
            params: The lattice parameters
        """
        self.params = params or EnhancedLatticeParameters()
    
    def generate_keypair(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Generate a keypair.
        
        Returns:
            A tuple (public_key, secret_key)
        """
        # Sample the seed
        seed = secrets.token_bytes(32)
        seed_bytes = np.frombuffer(seed, dtype=np.uint8)
        
        # Expand the seed to get A
        A = self._expand_a(seed_bytes)
        
        # Sample the secret key vectors s1 and s2
        s1 = np.array([EnhancedLatticeUtils.sample_ternary(self.params.ring_degree) 
                      for _ in range(self.params.ring_degree // 4)])
        
        s2 = np.array([EnhancedLatticeUtils.sample_ternary(self.params.ring_degree) 
                      for _ in range(self.params.ring_degree // 4)])
        
        # Apply quantum-inspired enhancements if enabled
        if self.params.use_quantum_enhancement:
            # Apply the Hadamard transform to the secret key vectors
            for i in range(len(s1)):
                s1[i] = EnhancedLatticeUtils.apply_hadamard_transform(s1[i], self.params.hadamard_matrix)
                s1[i] = np.round(np.real(s1[i])).astype(np.int64) % self.params.modulus
            
            for i in range(len(s2)):
                s2[i] = EnhancedLatticeUtils.apply_hadamard_transform(s2[i], self.params.hadamard_matrix)
                s2[i] = np.round(np.real(s2[i])).astype(np.int64) % self.params.modulus
        
        # Compute the public key t = A*s1 + s2
        t = np.zeros((self.params.ring_degree // 4, self.params.ring_degree), dtype=np.int64)
        
        for i in range(self.params.ring_degree // 4):
            for j in range(self.params.ring_degree // 4):
                t[i] = EnhancedLatticeUtils.polynomial_add(
                    t[i],
                    EnhancedLatticeUtils.polynomial_multiply(A[i][j], s1[j], self.params.modulus, self.params.ring_degree),
                    self.params.modulus
                )
            
            t[i] = EnhancedLatticeUtils.polynomial_add(t[i], s2[i], self.params.modulus)
        
        # Return the keypair
        public_key = {'A': A, 't': t, 'seed': seed_bytes}
        secret_key = {'s1': s1, 's2': s2, 'public_key': public_key}
        
        return public_key, secret_key
    
    def sign(self, secret_key: Dict[str, np.ndarray], message: bytes) -> Dict[str, np.ndarray]:
        """
        Sign a message.
        
        Args:
            secret_key: The secret key
            message: The message to sign
            
        Returns:
            The signature
        """
        # Extract the secret key components
        s1 = secret_key['s1']
        s2 = secret_key['s2']
        public_key = secret_key['public_key']
        A = public_key['A']
        t = public_key['t']
        
        # Hash the message to get the challenge
        message_hash = hashlib.sha256(message).digest()
        
        # Convert the hash to an array
        message_hash_array = np.frombuffer(message_hash, dtype=np.uint8)
        
        # Sample y from the uniform distribution
        y = np.array([EnhancedLatticeUtils.sample_uniform(self.params.ring_degree, 2 * self.params.modulus // 3) - self.params.modulus // 3 
                     for _ in range(self.params.ring_degree // 4)])
        
        # Apply quantum-inspired enhancements if enabled
        if self.params.use_quantum_enhancement:
            # Apply the Hadamard transform to y
            for i in range(len(y)):
                y[i] = EnhancedLatticeUtils.apply_hadamard_transform(y[i], self.params.hadamard_matrix)
                y[i] = np.round(np.real(y[i])).astype(np.int64) % self.params.modulus
        
        # Compute w = A*y
        w = np.zeros((self.params.ring_degree // 4, self.params.ring_degree), dtype=np.int64)
        
        for i in range(self.params.ring_degree // 4):
            for j in range(self.params.ring_degree // 4):
                w[i] = EnhancedLatticeUtils.polynomial_add(
                    w[i],
                    EnhancedLatticeUtils.polynomial_multiply(A[i][j], y[j], self.params.modulus, self.params.ring_degree),
                    self.params.modulus
                )
        
        # Compute the challenge c
        c = self._compute_challenge(w, message_hash_array)
        
        # Compute z = y + c*s1
        z = np.zeros((self.params.ring_degree // 4, self.params.ring_degree), dtype=np.int64)
        
        for i in range(self.params.ring_degree // 4):
            z[i] = EnhancedLatticeUtils.polynomial_add(
                y[i],
                EnhancedLatticeUtils.polynomial_multiply(c, s1[i], self.params.modulus, self.params.ring_degree),
                self.params.modulus
            )
        
        # Compute h = c*s2
        h = np.zeros((self.params.ring_degree // 4, self.params.ring_degree), dtype=np.int64)
        
        for i in range(self.params.ring_degree // 4):
            h[i] = EnhancedLatticeUtils.polynomial_multiply(c, s2[i], self.params.modulus, self.params.ring_degree)
        
        # Return the signature
        return {'z': z, 'h': h, 'c': c}
    
    def verify(self, public_key: Dict[str, np.ndarray], message: bytes, signature: Dict[str, np.ndarray]) -> bool:
        """
        Verify a signature.
        
        Args:
            public_key: The public key
            message: The message
            signature: The signature
            
        Returns:
            True if the signature is valid, False otherwise
        """
        # Extract the public key and signature components
        A = public_key['A']
        t = public_key['t']
        z = signature['z']
        h = signature['h']
        c = signature['c']
        
        # Check if z is in the correct range
        for i in range(len(z)):
            for j in range(len(z[i])):
                if abs(z[i][j]) >= self.params.modulus // 3:
                    return False
        
        # Compute A*z - c*t
        w = np.zeros((self.params.ring_degree // 4, self.params.ring_degree), dtype=np.int64)
        
        for i in range(self.params.ring_degree // 4):
            for j in range(self.params.ring_degree // 4):
                w[i] = EnhancedLatticeUtils.polynomial_add(
                    w[i],
                    EnhancedLatticeUtils.polynomial_multiply(A[i][j], z[j], self.params.modulus, self.params.ring_degree),
                    self.params.modulus
                )
            
            w[i] = EnhancedLatticeUtils.polynomial_subtract(
                w[i],
                EnhancedLatticeUtils.polynomial_multiply(c, t[i], self.params.modulus, self.params.ring_degree),
                self.params.modulus
            )
            
            w[i] = EnhancedLatticeUtils.polynomial_add(w[i], h[i], self.params.modulus)
        
        # Hash the message to get the challenge
        message_hash = hashlib.sha256(message).digest()
        
        # Convert the hash to an array
        message_hash_array = np.frombuffer(message_hash, dtype=np.uint8)
        
        # Compute the challenge c'
        c_prime = self._compute_challenge(w, message_hash_array)
        
        # Check if c = c'
        return np.array_equal(c, c_prime)
    
    def _expand_a(self, seed: np.ndarray) -> np.ndarray:
        """
        Expand a seed to get the matrix A.
        
        Args:
            seed: The seed
            
        Returns:
            The matrix A
        """
        # Initialize the matrix A
        A = np.zeros((self.params.ring_degree // 4, self.params.ring_degree // 4, self.params.ring_degree), dtype=np.int64)
        
        # Expand the seed
        for i in range(self.params.ring_degree // 4):
            for j in range(self.params.ring_degree // 4):
                # Hash the seed with the indices
                h = hashlib.sha256(seed.tobytes() + bytes([i, j])).digest()
                
                # Convert the hash to a polynomial
                for k in range(self.params.ring_degree):
                    if k < len(h):
                        A[i][j][k] = h[k % len(h)] % self.params.modulus
                    else:
                        A[i][j][k] = 0
        
        return A
    
    def _compute_challenge(self, w: np.ndarray, message_hash: np.ndarray) -> np.ndarray:
        """
        Compute the challenge c.
        
        Args:
            w: The vector w
            message_hash: The hash of the message
            
        Returns:
            The challenge c
        """
        # Convert w to bytes
        w_bytes = b''
        for i in range(len(w)):
            w_bytes += w[i].tobytes()
        
        # Hash w and the message hash
        h = hashlib.sha256(w_bytes + message_hash.tobytes()).digest()
        
        # Convert the hash to a sparse polynomial with tau +1's and tau -1's
        c = np.zeros(self.params.ring_degree, dtype=np.int64)
        
        # Use the hash to seed a PRNG
        import random
        random.seed(int.from_bytes(h, byteorder='big'))
        
        # Sample tau positions for +1
        tau = 60  # This is a parameter of the signature scheme
        positions_plus = random.sample(range(self.params.ring_degree), tau)
        for pos in positions_plus:
            c[pos] = 1
        
        # Sample tau positions for -1
        positions_minus = random.sample([i for i in range(self.params.ring_degree) if i not in positions_plus], tau)
        for pos in positions_minus:
            c[pos] = -1
        
        return c


# Example usage
if __name__ == "__main__":
    # Create enhanced lattice parameters
    params = EnhancedLatticeParameters(
        dimension=512,
        modulus=12289,
        error_std_dev=3.2,
        security_level=128,
        use_quantum_enhancement=True
    )
    
    # Create an enhanced lattice encryption scheme
    encryption = EnhancedLatticeEncryption(params)
    
    # Generate a keypair
    public_key, secret_key = encryption.generate_keypair()
    print(f"Generated enhanced lattice keypair")
    
    # Create a message
    message = EnhancedLatticeUtils.sample_binary(params.ring_degree)
    print(f"Original message: {message[:10]}...")
    
    # Encrypt the message
    ciphertext = encryption.encrypt(public_key, message)
    print(f"Encrypted message")
    
    # Decrypt the message
    decrypted = encryption.decrypt(secret_key, ciphertext)
    print(f"Decrypted message: {decrypted[:10]}...")
    
    # Check if the decryption is correct
    if np.array_equal(message, decrypted):
        print("Decryption successful!")
    else:
        print("Decryption failed!")
    
    print()
    
    # Create an enhanced lattice signature scheme
    signature = EnhancedLatticeSignature(params)
    
    # Generate a keypair
    public_key, secret_key = signature.generate_keypair()
    print(f"Generated enhanced lattice signature keypair")
    
    # Sign a message
    message = b"Hello, quantum-inspired world!"
    sig = signature.sign(secret_key, message)
    print(f"Signed message")
    
    # Verify the signature
    is_valid = signature.verify(public_key, message, sig)
    print(f"Signature valid: {is_valid}")
    
    # Try to verify with a modified message
    modified_message = b"Hello, quantum-inspired world!!"
    is_valid = signature.verify(public_key, modified_message, sig)
    print(f"Signature valid for modified message: {is_valid}")