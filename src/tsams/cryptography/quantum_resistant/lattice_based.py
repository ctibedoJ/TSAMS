"""
Lattice-Based Encryption implementation.

This module provides an implementation of lattice-based encryption,
which is resistant to quantum computing attacks.
"""

import numpy as np
import os
from typing import Tuple, List, Dict, Optional, Union, Callable, Any
import time
import pickle
import hashlib
import hmac


class LatticeBasedEncryption:
    """
    A class for lattice-based encryption.
    
    This class provides methods for generating key pairs, encrypting, and decrypting
    messages using lattice-based cryptography, which is resistant to quantum computing attacks.
    
    Attributes:
        n (int): The dimension of the lattice.
        q (int): The modulus for the lattice.
        sigma (float): The standard deviation for the error distribution.
        m (int): The number of samples for the public key.
    """
    
    def __init__(self, n: int = 512, q: int = 12289, sigma: float = 3.0, m: int = 1024):
        """
        Initialize the lattice-based encryption scheme.
        
        Args:
            n (int, optional): The dimension of the lattice. Defaults to 512.
            q (int, optional): The modulus for the lattice. Defaults to 12289.
            sigma (float, optional): The standard deviation for the error distribution.
                Defaults to 3.0.
            m (int, optional): The number of samples for the public key. Defaults to 1024.
        """
        self.n = n
        self.q = q
        self.sigma = sigma
        self.m = m
    
    def generate_key_pair(self) -> Tuple[bytes, bytes]:
        """
        Generate a key pair for the encryption scheme.
        
        Returns:
            Tuple[bytes, bytes]: The public key and private key as byte strings.
        """
        # Generate a random private key
        s = np.random.randint(0, self.q, size=self.n)
        
        # Generate a random matrix A
        A = np.random.randint(0, self.q, size=(self.m, self.n))
        
        # Generate a small error vector
        e = np.random.normal(0, self.sigma, size=self.m).astype(int) % self.q
        
        # Compute the public key
        b = (A @ s + e) % self.q
        
        # Pack the keys
        public_key = {
            'A': A,
            'b': b,
            'n': self.n,
            'q': self.q,
            'm': self.m
        }
        
        private_key = {
            's': s,
            'n': self.n,
            'q': self.q
        }
        
        # Serialize the keys
        public_key_bytes = pickle.dumps(public_key)
        private_key_bytes = pickle.dumps(private_key)
        
        return public_key_bytes, private_key_bytes
    
    def encrypt(self, message: bytes, public_key_bytes: bytes) -> bytes:
        """
        Encrypt a message using the public key.
        
        Args:
            message (bytes): The message to encrypt.
            public_key_bytes (bytes): The public key as a byte string.
        
        Returns:
            bytes: The encrypted message as a byte string.
        """
        # Deserialize the public key
        public_key = pickle.loads(public_key_bytes)
        A = public_key['A']
        b = public_key['b']
        n = public_key['n']
        q = public_key['q']
        m = public_key['m']
        
        # Convert the message to a bit array
        message_bits = np.unpackbits(np.frombuffer(message, dtype=np.uint8))
        
        # Pad the message bits to a multiple of n
        padding_length = (n - (len(message_bits) % n)) % n
        message_bits = np.pad(message_bits, (0, padding_length))
        
        # Split the message bits into blocks of size n
        message_blocks = message_bits.reshape(-1, n)
        
        # Encrypt each block
        ciphertext_blocks = []
        for block in message_blocks:
            # Generate a random vector
            r = np.random.randint(0, 2, size=m)
            
            # Compute the ciphertext
            u = (r @ A) % q
            v = (r @ b + (q // 2) * block) % q
            
            ciphertext_blocks.append((u, v))
        
        # Pack the ciphertext
        ciphertext = {
            'blocks': ciphertext_blocks,
            'n': n,
            'q': q,
            'padding_length': padding_length
        }
        
        # Serialize the ciphertext
        ciphertext_bytes = pickle.dumps(ciphertext)
        
        return ciphertext_bytes
    
    def decrypt(self, ciphertext_bytes: bytes, private_key_bytes: bytes) -> bytes:
        """
        Decrypt a message using the private key.
        
        Args:
            ciphertext_bytes (bytes): The encrypted message as a byte string.
            private_key_bytes (bytes): The private key as a byte string.
        
        Returns:
            bytes: The decrypted message as a byte string.
        """
        # Deserialize the ciphertext and private key
        ciphertext = pickle.loads(ciphertext_bytes)
        private_key = pickle.loads(private_key_bytes)
        
        s = private_key['s']
        n = private_key['n']
        q = private_key['q']
        
        blocks = ciphertext['blocks']
        padding_length = ciphertext['padding_length']
        
        # Decrypt each block
        message_bits = []
        for u, v in blocks:
            # Compute the decryption
            w = (v - u @ s) % q
            
            # Determine the message bit
            # If w is closer to 0 than to q/2, the bit is 0
            # If w is closer to q/2 than to 0, the bit is 1
            block = np.zeros(n, dtype=np.uint8)
            for i in range(n):
                if w[i] > q // 4 and w[i] < 3 * q // 4:
                    block[i] = 1
            
            message_bits.append(block)
        
        # Combine the message bits
        message_bits = np.concatenate(message_bits)
        
        # Remove the padding
        if padding_length > 0:
            message_bits = message_bits[:-padding_length]
        
        # Convert the bit array back to bytes
        message = np.packbits(message_bits).tobytes()
        
        return message


class RingLWEEncryption(LatticeBasedEncryption):
    """
    A class for Ring-LWE based encryption.
    
    This class provides methods for generating key pairs, encrypting, and decrypting
    messages using Ring-LWE based cryptography, which is a more efficient variant of
    lattice-based cryptography.
    
    Attributes:
        n (int): The dimension of the lattice.
        q (int): The modulus for the lattice.
        sigma (float): The standard deviation for the error distribution.
    """
    
    def __init__(self, n: int = 1024, q: int = 12289, sigma: float = 3.0):
        """
        Initialize the Ring-LWE based encryption scheme.
        
        Args:
            n (int, optional): The dimension of the lattice. Defaults to 1024.
            q (int, optional): The modulus for the lattice. Defaults to 12289.
            sigma (float, optional): The standard deviation for the error distribution.
                Defaults to 3.0.
        """
        super().__init__(n, q, sigma, n)
    
    def _ntt(self, a: np.ndarray) -> np.ndarray:
        """
        Compute the Number Theoretic Transform (NTT) of a polynomial.
        
        Args:
            a (np.ndarray): The polynomial coefficients.
        
        Returns:
            np.ndarray: The NTT of the polynomial.
        """
        # This is a simplified implementation of the NTT
        # In a real implementation, this would use a more efficient algorithm
        
        n = len(a)
        result = np.zeros(n, dtype=np.int64)
        
        for k in range(n):
            for j in range(n):
                result[k] = (result[k] + a[j] * pow(3, j * k, self.q)) % self.q
        
        return result
    
    def _intt(self, a: np.ndarray) -> np.ndarray:
        """
        Compute the Inverse Number Theoretic Transform (INTT) of a polynomial.
        
        Args:
            a (np.ndarray): The NTT coefficients.
        
        Returns:
            np.ndarray: The polynomial coefficients.
        """
        # This is a simplified implementation of the INTT
        # In a real implementation, this would use a more efficient algorithm
        
        n = len(a)
        result = np.zeros(n, dtype=np.int64)
        
        for k in range(n):
            for j in range(n):
                result[k] = (result[k] + a[j] * pow(3, -j * k, self.q)) % self.q
        
        # Multiply by n^(-1) mod q
        n_inv = pow(n, -1, self.q)
        result = (result * n_inv) % self.q
        
        return result
    
    def _poly_mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Multiply two polynomials in the ring R_q = Z_q[x]/(x^n + 1).
        
        Args:
            a (np.ndarray): The first polynomial.
            b (np.ndarray): The second polynomial.
        
        Returns:
            np.ndarray: The product of the polynomials.
        """
        # Convert to NTT domain
        a_ntt = self._ntt(a)
        b_ntt = self._ntt(b)
        
        # Multiply in the NTT domain
        c_ntt = (a_ntt * b_ntt) % self.q
        
        # Convert back to the polynomial domain
        c = self._intt(c_ntt)
        
        return c
    
    def generate_key_pair(self) -> Tuple[bytes, bytes]:
        """
        Generate a key pair for the encryption scheme.
        
        Returns:
            Tuple[bytes, bytes]: The public key and private key as byte strings.
        """
        # Generate a random private key
        s = np.random.randint(0, self.q, size=self.n)
        
        # Generate a random polynomial a
        a = np.random.randint(0, self.q, size=self.n)
        
        # Generate a small error polynomial
        e = np.random.normal(0, self.sigma, size=self.n).astype(int) % self.q
        
        # Compute the public key
        b = (self._poly_mul(a, s) + e) % self.q
        
        # Pack the keys
        public_key = {
            'a': a,
            'b': b,
            'n': self.n,
            'q': self.q
        }
        
        private_key = {
            's': s,
            'n': self.n,
            'q': self.q
        }
        
        # Serialize the keys
        public_key_bytes = pickle.dumps(public_key)
        private_key_bytes = pickle.dumps(private_key)
        
        return public_key_bytes, private_key_bytes
    
    def encrypt(self, message: bytes, public_key_bytes: bytes) -> bytes:
        """
        Encrypt a message using the public key.
        
        Args:
            message (bytes): The message to encrypt.
            public_key_bytes (bytes): The public key as a byte string.
        
        Returns:
            bytes: The encrypted message as a byte string.
        """
        # Deserialize the public key
        public_key = pickle.loads(public_key_bytes)
        a = public_key['a']
        b = public_key['b']
        n = public_key['n']
        q = public_key['q']
        
        # Convert the message to a bit array
        message_bits = np.unpackbits(np.frombuffer(message, dtype=np.uint8))
        
        # Pad the message bits to a multiple of n
        padding_length = (n - (len(message_bits) % n)) % n
        message_bits = np.pad(message_bits, (0, padding_length))
        
        # Split the message bits into blocks of size n
        message_blocks = message_bits.reshape(-1, n)
        
        # Encrypt each block
        ciphertext_blocks = []
        for block in message_blocks:
            # Generate a random polynomial
            r = np.random.randint(0, 2, size=n)
            
            # Generate small error polynomials
            e1 = np.random.normal(0, self.sigma, size=n).astype(int) % q
            e2 = np.random.normal(0, self.sigma, size=n).astype(int) % q
            
            # Compute the ciphertext
            u = (self._poly_mul(a, r) + e1) % q
            v = (self._poly_mul(b, r) + e2 + (q // 2) * block) % q
            
            ciphertext_blocks.append((u, v))
        
        # Pack the ciphertext
        ciphertext = {
            'blocks': ciphertext_blocks,
            'n': n,
            'q': q,
            'padding_length': padding_length
        }
        
        # Serialize the ciphertext
        ciphertext_bytes = pickle.dumps(ciphertext)
        
        return ciphertext_bytes
    
    def decrypt(self, ciphertext_bytes: bytes, private_key_bytes: bytes) -> bytes:
        """
        Decrypt a message using the private key.
        
        Args:
            ciphertext_bytes (bytes): The encrypted message as a byte string.
            private_key_bytes (bytes): The private key as a byte string.
        
        Returns:
            bytes: The decrypted message as a byte string.
        """
        # Deserialize the ciphertext and private key
        ciphertext = pickle.loads(ciphertext_bytes)
        private_key = pickle.loads(private_key_bytes)
        
        s = private_key['s']
        n = private_key['n']
        q = private_key['q']
        
        blocks = ciphertext['blocks']
        padding_length = ciphertext['padding_length']
        
        # Decrypt each block
        message_bits = []
        for u, v in blocks:
            # Compute the decryption
            w = (v - self._poly_mul(u, s)) % q
            
            # Determine the message bit
            # If w is closer to 0 than to q/2, the bit is 0
            # If w is closer to q/2 than to 0, the bit is 1
            block = np.zeros(n, dtype=np.uint8)
            for i in range(n):
                if w[i] > q // 4 and w[i] < 3 * q // 4:
                    block[i] = 1
            
            message_bits.append(block)
        
        # Combine the message bits
        message_bits = np.concatenate(message_bits)
        
        # Remove the padding
        if padding_length > 0:
            message_bits = message_bits[:-padding_length]
        
        # Convert the bit array back to bytes
        message = np.packbits(message_bits).tobytes()
        
        return message