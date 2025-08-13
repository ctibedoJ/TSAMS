"""
Hash-Based Signature implementation.

This module provides an implementation of hash-based digital signatures,
which are resistant to quantum computing attacks.
"""

import numpy as np
import os
from typing import Tuple, List, Dict, Optional, Union, Callable, Any
import time
import pickle
import hashlib
import hmac


class HashBasedSignature:
    """
    A class for hash-based digital signatures.
    
    This class provides methods for generating key pairs, signing, and verifying
    messages using hash-based cryptography, which is resistant to quantum computing attacks.
    
    Attributes:
        hash_function (Callable): The hash function to use.
        hash_length (int): The length of the hash output in bytes.
        tree_height (int): The height of the Merkle tree.
    """
    
    def __init__(self, hash_function: Callable = hashlib.sha256, hash_length: int = 32, tree_height: int = 10):
        """
        Initialize the hash-based signature scheme.
        
        Args:
            hash_function (Callable, optional): The hash function to use. Defaults to hashlib.sha256.
            hash_length (int, optional): The length of the hash output in bytes. Defaults to 32.
            tree_height (int, optional): The height of the Merkle tree. Defaults to 10.
        """
        self.hash_function = hash_function
        self.hash_length = hash_length
        self.tree_height = tree_height
    
    def _hash(self, data: bytes) -> bytes:
        """
        Compute the hash of data.
        
        Args:
            data (bytes): The data to hash.
        
        Returns:
            bytes: The hash of the data.
        """
        h = self.hash_function()
        h.update(data)
        return h.digest()
    
    def _generate_wots_key_pair(self, seed: bytes) -> Tuple[List[List[bytes]], List[bytes]]:
        """
        Generate a Winternitz One-Time Signature (WOTS) key pair.
        
        Args:
            seed (bytes): The seed for the key pair.
        
        Returns:
            Tuple[List[List[bytes]], List[bytes]]: The private key and public key.
        """
        # Parameters for WOTS
        w = 16  # Winternitz parameter
        n = self.hash_length
        
        # Compute the number of elements in the signature
        len_1 = (8 * n) // np.log2(w)
        len_2 = np.floor(np.log2(len_1 * (w - 1)) / np.log2(w)) + 1
        len_total = int(len_1 + len_2)
        
        # Generate the private key
        private_key = []
        for i in range(len_total):
            # Generate a chain of hash values
            chain = []
            current = self._hash(seed + i.to_bytes(4, byteorder='big'))
            chain.append(current)
            
            for j in range(w - 1):
                current = self._hash(current)
                chain.append(current)
            
            private_key.append(chain)
        
        # Generate the public key
        public_key = [chain[-1] for chain in private_key]
        
        return private_key, public_key
    
    def _wots_sign(self, message: bytes, private_key: List[List[bytes]]) -> List[bytes]:
        """
        Sign a message using the Winternitz One-Time Signature (WOTS) scheme.
        
        Args:
            message (bytes): The message to sign.
            private_key (List[List[bytes]]): The private key.
        
        Returns:
            List[bytes]: The signature.
        """
        # Parameters for WOTS
        w = 16  # Winternitz parameter
        n = self.hash_length
        
        # Compute the number of elements in the signature
        len_1 = (8 * n) // np.log2(w)
        len_2 = np.floor(np.log2(len_1 * (w - 1)) / np.log2(w)) + 1
        len_total = int(len_1 + len_2)
        
        # Compute the message digest
        digest = self._hash(message)
        
        # Convert the digest to base w
        base_w_digest = []
        for i in range(len_1):
            if i * np.log2(w) // 8 < len(digest):
                byte_index = int(i * np.log2(w) // 8)
                bit_index = int(i * np.log2(w) % 8)
                
                # Extract the value from the digest
                if bit_index + np.log2(w) <= 8:
                    # The value is contained in a single byte
                    value = (digest[byte_index] >> bit_index) & (w - 1)
                else:
                    # The value spans two bytes
                    value = ((digest[byte_index] >> bit_index) | (digest[byte_index + 1] << (8 - bit_index))) & (w - 1)
                
                base_w_digest.append(value)
            else:
                base_w_digest.append(0)
        
        # Compute the checksum
        checksum = sum((w - 1 - value) for value in base_w_digest)
        
        # Convert the checksum to base w
        base_w_checksum = []
        for i in range(len_2):
            base_w_checksum.append((checksum >> (i * np.log2(w))) & (w - 1))
        
        # Combine the base w digest and checksum
        base_w = base_w_digest + base_w_checksum
        
        # Generate the signature
        signature = []
        for i, value in enumerate(base_w):
            signature.append(private_key[i][value])
        
        return signature
    
    def _wots_verify(self, message: bytes, signature: List[bytes], public_key: List[bytes]) -> bool:
        """
        Verify a signature using the Winternitz One-Time Signature (WOTS) scheme.
        
        Args:
            message (bytes): The message that was signed.
            signature (List[bytes]): The signature.
            public_key (List[bytes]): The public key.
        
        Returns:
            bool: True if the signature is valid, False otherwise.
        """
        # Parameters for WOTS
        w = 16  # Winternitz parameter
        n = self.hash_length
        
        # Compute the number of elements in the signature
        len_1 = (8 * n) // np.log2(w)
        len_2 = np.floor(np.log2(len_1 * (w - 1)) / np.log2(w)) + 1
        len_total = int(len_1 + len_2)
        
        # Compute the message digest
        digest = self._hash(message)
        
        # Convert the digest to base w
        base_w_digest = []
        for i in range(len_1):
            if i * np.log2(w) // 8 < len(digest):
                byte_index = int(i * np.log2(w) // 8)
                bit_index = int(i * np.log2(w) % 8)
                
                # Extract the value from the digest
                if bit_index + np.log2(w) <= 8:
                    # The value is contained in a single byte
                    value = (digest[byte_index] >> bit_index) & (w - 1)
                else:
                    # The value spans two bytes
                    value = ((digest[byte_index] >> bit_index) | (digest[byte_index + 1] << (8 - bit_index))) & (w - 1)
                
                base_w_digest.append(value)
            else:
                base_w_digest.append(0)
        
        # Compute the checksum
        checksum = sum((w - 1 - value) for value in base_w_digest)
        
        # Convert the checksum to base w
        base_w_checksum = []
        for i in range(len_2):
            base_w_checksum.append((checksum >> (i * np.log2(w))) & (w - 1))
        
        # Combine the base w digest and checksum
        base_w = base_w_digest + base_w_checksum
        
        # Verify the signature
        for i, value in enumerate(base_w):
            # Hash the signature element (w - 1 - value) times
            current = signature[i]
            for j in range(w - 1 - value):
                current = self._hash(current)
            
            # Check if the result matches the public key
            if current != public_key[i]:
                return False
        
        return True
    
    def _build_merkle_tree(self, seeds: List[bytes]) -> List[List[bytes]]:
        """
        Build a Merkle tree from a list of seeds.
        
        Args:
            seeds (List[bytes]): The seeds for the WOTS key pairs.
        
        Returns:
            List[List[bytes]]: The Merkle tree.
        """
        # Generate the WOTS key pairs
        wots_public_keys = []
        for seed in seeds:
            _, public_key = self._generate_wots_key_pair(seed)
            wots_public_keys.append(public_key)
        
        # Build the Merkle tree
        tree = [wots_public_keys]
        
        # Build the tree from the leaves to the root
        for i in range(self.tree_height):
            level = []
            for j in range(0, len(tree[i]), 2):
                if j + 1 < len(tree[i]):
                    # Hash the concatenation of the two child nodes
                    left = self._hash(b''.join(tree[i][j]))
                    right = self._hash(b''.join(tree[i][j + 1]))
                    level.append(self._hash(left + right))
                else:
                    # If there's an odd number of nodes, just copy the last one
                    level.append(self._hash(b''.join(tree[i][j])))
            tree.append(level)
        
        return tree
    
    def generate_key_pair(self) -> Tuple[bytes, bytes]:
        """
        Generate a key pair for the signature scheme.
        
        Returns:
            Tuple[bytes, bytes]: The private key and public key as byte strings.
        """
        # Generate a master seed
        master_seed = os.urandom(self.hash_length)
        
        # Generate seeds for the WOTS key pairs
        seeds = []
        for i in range(2 ** self.tree_height):
            seed = self._hash(master_seed + i.to_bytes(4, byteorder='big'))
            seeds.append(seed)
        
        # Build the Merkle tree
        tree = self._build_merkle_tree(seeds)
        
        # The public key is the root of the Merkle tree
        public_key = tree[-1][0]
        
        # The private key is the master seed and the current index
        private_key = {
            'master_seed': master_seed,
            'index': 0,
            'tree_height': self.tree_height
        }
        
        # Serialize the keys
        private_key_bytes = pickle.dumps(private_key)
        
        return private_key_bytes, public_key
    
    def sign(self, message: bytes, private_key_bytes: bytes) -> bytes:
        """
        Sign a message using the private key.
        
        Args:
            message (bytes): The message to sign.
            private_key_bytes (bytes): The private key as a byte string.
        
        Returns:
            bytes: The signature as a byte string.
        
        Raises:
            ValueError: If the private key has been used too many times.
        """
        # Deserialize the private key
        private_key = pickle.loads(private_key_bytes)
        master_seed = private_key['master_seed']
        index = private_key['index']
        tree_height = private_key['tree_height']
        
        # Check if the private key has been used too many times
        if index >= 2 ** tree_height:
            raise ValueError("Private key has been used too many times")
        
        # Generate the WOTS key pair for the current index
        seed = self._hash(master_seed + index.to_bytes(4, byteorder='big'))
        wots_private_key, wots_public_key = self._generate_wots_key_pair(seed)
        
        # Sign the message using WOTS
        wots_signature = self._wots_sign(message, wots_private_key)
        
        # Compute the authentication path
        auth_path = []
        tree_index = index
        for i in range(tree_height):
            # Determine if the current node is a left or right child
            if tree_index % 2 == 0:
                # Left child, include the right sibling in the auth path
                if tree_index + 1 < 2 ** (tree_height - i):
                    sibling_seed = self._hash(master_seed + (tree_index + 1).to_bytes(4, byteorder='big'))
                    _, sibling_public_key = self._generate_wots_key_pair(sibling_seed)
                    auth_path.append(self._hash(b''.join(sibling_public_key)))
                else:
                    # No right sibling, use a dummy node
                    auth_path.append(b'\x00' * self.hash_length)
            else:
                # Right child, include the left sibling in the auth path
                sibling_seed = self._hash(master_seed + (tree_index - 1).to_bytes(4, byteorder='big'))
                _, sibling_public_key = self._generate_wots_key_pair(sibling_seed)
                auth_path.append(self._hash(b''.join(sibling_public_key)))
            
            # Move up the tree
            tree_index //= 2
        
        # Increment the index for the next signature
        private_key['index'] += 1
        
        # Pack the signature
        signature = {
            'index': index,
            'wots_signature': wots_signature,
            'auth_path': auth_path
        }
        
        # Serialize the signature
        signature_bytes = pickle.dumps(signature)
        
        # Return the updated private key and the signature
        return pickle.dumps(private_key), signature_bytes
    
    def verify(self, message: bytes, signature_bytes: bytes, public_key: bytes) -> bool:
        """
        Verify a signature using the public key.
        
        Args:
            message (bytes): The message that was signed.
            signature_bytes (bytes): The signature as a byte string.
            public_key (bytes): The public key as a byte string.
        
        Returns:
            bool: True if the signature is valid, False otherwise.
        """
        # Deserialize the signature
        signature = pickle.loads(signature_bytes)
        index = signature['index']
        wots_signature = signature['wots_signature']
        auth_path = signature['auth_path']
        
        # Compute the WOTS public key from the signature
        wots_public_key = []
        for i, sig_element in enumerate(wots_signature):
            # Hash the signature element the appropriate number of times
            current = sig_element
            for j in range(16 - 1 - i % 16):
                current = self._hash(current)
            wots_public_key.append(current)
        
        # Compute the leaf node
        leaf = self._hash(b''.join(wots_public_key))
        
        # Compute the root of the Merkle tree
        root = leaf
        tree_index = index
        for i, auth_node in enumerate(auth_path):
            # Determine if the current node is a left or right child
            if tree_index % 2 == 0:
                # Left child, combine with the right sibling
                root = self._hash(root + auth_node)
            else:
                # Right child, combine with the left sibling
                root = self._hash(auth_node + root)
            
            # Move up the tree
            tree_index //= 2
        
        # Verify that the computed root matches the public key
        return root == public_key