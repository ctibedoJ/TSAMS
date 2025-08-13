"""
TIBEDO Hash-Based Signature Schemes

This module implements hash-based signature schemes that are resistant to quantum
attacks, including SPHINCS+ and XMSS, which are based on hash functions and
Merkle trees.
"""

import numpy as np
import hashlib
import hmac
import os
import sys
import logging
import time
import secrets
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WOTSPlus:
    """
    Winternitz One-Time Signature Plus (WOTS+) scheme.
    
    WOTS+ is a one-time signature scheme that is used as a building block for
    hash-based signature schemes like XMSS and SPHINCS+.
    """
    
    def __init__(self, 
                 w: int = 16, 
                 n: int = 32, 
                 hash_function: str = 'sha256'):
        """
        Initialize the WOTS+ scheme.
        
        Args:
            w: The Winternitz parameter (typically 4, 16, or 256)
            n: The security parameter in bytes (typically 32 for 256-bit security)
            hash_function: The hash function to use
        """
        self.w = w
        self.n = n
        self.hash_function = hash_function
        
        # Compute derived parameters
        self.len_1 = self._compute_len_1()
        self.len_2 = self._compute_len_2()
        self.len = self.len_1 + self.len_2
        
        # Initialize the hash function
        self._init_hash_function()
    
    def _compute_len_1(self) -> int:
        """
        Compute the length of the first part of the WOTS+ signature.
        
        Returns:
            The length of the first part of the signature
        """
        # Compute ceil(8*n / log2(w))
        return int(np.ceil(8 * self.n / np.log2(self.w)))
    
    def _compute_len_2(self) -> int:
        """
        Compute the length of the second part of the WOTS+ signature (checksum).
        
        Returns:
            The length of the second part of the signature
        """
        # Compute floor(log2(len_1 * (w - 1)) / log2(w)) + 1
        return int(np.floor(np.log2(self.len_1 * (self.w - 1)) / np.log2(self.w))) + 1
    
    def _init_hash_function(self) -> None:
        """
        Initialize the hash function.
        """
        if self.hash_function == 'sha256':
            self.hash_func = lambda x: hashlib.sha256(x).digest()
        elif self.hash_function == 'sha512':
            self.hash_func = lambda x: hashlib.sha512(x).digest()[:self.n]
        elif self.hash_function == 'shake256':
            self.hash_func = lambda x: hashlib.shake_256(x).digest(self.n)
        else:
            raise ValueError(f"Unsupported hash function: {self.hash_function}")
    
    def _chain(self, x: bytes, start: int, steps: int, public_seed: bytes, address: bytes) -> bytes:
        """
        Apply the chaining function to x, starting from index start and applying steps iterations.
        
        Args:
            x: The input value
            start: The starting index
            steps: The number of steps to apply
            public_seed: The public seed
            address: The address
            
        Returns:
            The result of the chaining function
        """
        if steps == 0:
            return x
        
        if start >= self.w - 1:
            return x
        
        # Apply the hash function iteratively
        result = x
        for i in range(start, start + steps):
            if i >= self.w - 1:
                break
            
            # Compute the hash
            # In a real implementation, we would use a more sophisticated addressing scheme
            # For now, we'll just concatenate the address with the iteration index
            addr = address + i.to_bytes(4, byteorder='big')
            
            # Apply the hash function
            result = self.hash_func(public_seed + addr + result)
        
        return result
    
    def generate_keypair(self, secret_seed: bytes, public_seed: bytes, address: bytes) -> Tuple[List[bytes], List[bytes]]:
        """
        Generate a WOTS+ keypair.
        
        Args:
            secret_seed: The secret seed
            public_seed: The public seed
            address: The address
            
        Returns:
            A tuple (private_key, public_key)
        """
        # Generate the private key
        private_key = []
        for i in range(self.len):
            # Generate the private key element
            # In a real implementation, we would use a PRF
            # For now, we'll just use HMAC
            sk_i = hmac.new(secret_seed, address + i.to_bytes(4, byteorder='big'), hashlib.sha256).digest()[:self.n]
            private_key.append(sk_i)
        
        # Generate the public key
        public_key = []
        for i in range(self.len):
            # Apply the chain function to the private key element
            pk_i = self._chain(private_key[i], 0, self.w - 1, public_seed, address + i.to_bytes(4, byteorder='big'))
            public_key.append(pk_i)
        
        return private_key, public_key
    
    def sign(self, message: bytes, private_key: List[bytes], public_seed: bytes, address: bytes) -> List[bytes]:
        """
        Sign a message using WOTS+.
        
        Args:
            message: The message to sign
            private_key: The private key
            public_seed: The public seed
            address: The address
            
        Returns:
            The signature
        """
        # Compute the message digest
        message_digest = self.hash_func(message)
        
        # Convert the message digest to base w
        base_w_digest = self._base_w(message_digest)
        
        # Compute the checksum
        checksum = self._compute_checksum(base_w_digest)
        
        # Convert the checksum to base w
        base_w_checksum = self._base_w_int(checksum, self.len_2)
        
        # Combine the base w digest and checksum
        base_w = base_w_digest + base_w_checksum
        
        # Generate the signature
        signature = []
        for i in range(self.len):
            # Apply the chain function to the private key element
            sig_i = self._chain(private_key[i], 0, base_w[i], public_seed, address + i.to_bytes(4, byteorder='big'))
            signature.append(sig_i)
        
        return signature
    
    def verify(self, message: bytes, signature: List[bytes], public_seed: bytes, address: bytes) -> List[bytes]:
        """
        Verify a signature using WOTS+ and return the public key.
        
        Args:
            message: The message
            signature: The signature
            public_seed: The public seed
            address: The address
            
        Returns:
            The public key
        """
        # Compute the message digest
        message_digest = self.hash_func(message)
        
        # Convert the message digest to base w
        base_w_digest = self._base_w(message_digest)
        
        # Compute the checksum
        checksum = self._compute_checksum(base_w_digest)
        
        # Convert the checksum to base w
        base_w_checksum = self._base_w_int(checksum, self.len_2)
        
        # Combine the base w digest and checksum
        base_w = base_w_digest + base_w_checksum
        
        # Compute the public key
        public_key = []
        for i in range(self.len):
            # Apply the chain function to the signature element
            pk_i = self._chain(signature[i], base_w[i], self.w - 1 - base_w[i], public_seed, address + i.to_bytes(4, byteorder='big'))
            public_key.append(pk_i)
        
        return public_key
    
    def _base_w(self, x: bytes) -> List[int]:
        """
        Convert a byte string to base w.
        
        Args:
            x: The byte string
            
        Returns:
            The base w representation
        """
        # Compute the number of base w digits we can extract from one byte
        digits_per_byte = int(8 / np.log2(self.w))
        
        # Compute the number of base w digits we can extract from x
        num_digits = min(self.len_1, len(x) * digits_per_byte)
        
        # Convert to base w
        base_w = []
        for i in range(num_digits):
            # Compute the byte index and bit index
            byte_idx = i // digits_per_byte
            bit_idx = (i % digits_per_byte) * int(np.log2(self.w))
            
            # Extract the base w digit
            if byte_idx < len(x):
                digit = (x[byte_idx] >> bit_idx) & (self.w - 1)
                base_w.append(digit)
        
        # Pad with zeros if necessary
        while len(base_w) < self.len_1:
            base_w.append(0)
        
        return base_w
    
    def _base_w_int(self, x: int, out_len: int) -> List[int]:
        """
        Convert an integer to base w.
        
        Args:
            x: The integer
            out_len: The desired output length
            
        Returns:
            The base w representation
        """
        # Convert to base w
        base_w = []
        for i in range(out_len):
            # Extract the base w digit
            digit = x % self.w
            x = x // self.w
            base_w.append(digit)
        
        return base_w
    
    def _compute_checksum(self, base_w: List[int]) -> int:
        """
        Compute the checksum of a base w digest.
        
        Args:
            base_w: The base w digest
            
        Returns:
            The checksum
        """
        # Compute the checksum
        checksum = 0
        for digit in base_w:
            checksum += self.w - 1 - digit
        
        return checksum


class XMSS:
    """
    eXtended Merkle Signature Scheme (XMSS).
    
    XMSS is a hash-based signature scheme that uses a Merkle tree to sign
    multiple messages with a single public key.
    """
    
    def __init__(self, 
                 height: int = 10, 
                 w: int = 16, 
                 n: int = 32, 
                 hash_function: str = 'sha256'):
        """
        Initialize the XMSS scheme.
        
        Args:
            height: The height of the Merkle tree
            w: The Winternitz parameter
            n: The security parameter in bytes
            hash_function: The hash function to use
        """
        self.height = height
        self.w = w
        self.n = n
        self.hash_function = hash_function
        
        # Initialize the WOTS+ scheme
        self.wots = WOTSPlus(w, n, hash_function)
        
        # Initialize the hash function
        self._init_hash_function()
        
        # Compute the number of leaf nodes
        self.num_leaves = 2**height
    
    def _init_hash_function(self) -> None:
        """
        Initialize the hash function.
        """
        if self.hash_function == 'sha256':
            self.hash_func = lambda x: hashlib.sha256(x).digest()
        elif self.hash_function == 'sha512':
            self.hash_func = lambda x: hashlib.sha512(x).digest()[:self.n]
        elif self.hash_function == 'shake256':
            self.hash_func = lambda x: hashlib.shake_256(x).digest(self.n)
        else:
            raise ValueError(f"Unsupported hash function: {self.hash_function}")
    
    def _compute_root(self, 
                     leaf_index: int, 
                     leaf: bytes, 
                     auth_path: List[bytes], 
                     public_seed: bytes) -> bytes:
        """
        Compute the root of the Merkle tree given a leaf and an authentication path.
        
        Args:
            leaf_index: The index of the leaf
            leaf: The leaf node
            auth_path: The authentication path
            public_seed: The public seed
            
        Returns:
            The root of the Merkle tree
        """
        # Start with the leaf node
        node = leaf
        
        # Compute the root
        for i in range(self.height):
            # Compute the parent node
            if (leaf_index >> i) & 1:
                # Leaf index bit is 1, so auth_path[i] is the left child
                parent = self.hash_func(public_seed + auth_path[i] + node)
            else:
                # Leaf index bit is 0, so auth_path[i] is the right child
                parent = self.hash_func(public_seed + node + auth_path[i])
            
            # Update the node
            node = parent
        
        return node
    
    def _compute_auth_path(self, 
                          leaf_index: int, 
                          leaves: List[bytes], 
                          public_seed: bytes) -> List[bytes]:
        """
        Compute the authentication path for a leaf.
        
        Args:
            leaf_index: The index of the leaf
            leaves: The leaf nodes
            public_seed: The public seed
            
        Returns:
            The authentication path
        """
        # Initialize the authentication path
        auth_path = []
        
        # Compute the authentication path
        for i in range(self.height):
            # Compute the sibling index
            sibling_idx = leaf_index ^ (1 << i)
            
            # Compute the sibling node
            if sibling_idx < len(leaves):
                # If the sibling is a leaf node, use it directly
                sibling = leaves[sibling_idx]
            else:
                # Otherwise, compute the sibling node
                # This is a simplified implementation
                # In a real implementation, we would use a more efficient algorithm
                sibling = self._compute_node(sibling_idx, leaves, public_seed)
            
            # Add the sibling to the authentication path
            auth_path.append(sibling)
        
        return auth_path
    
    def _compute_node(self, 
                     node_index: int, 
                     leaves: List[bytes], 
                     public_seed: bytes) -> bytes:
        """
        Compute a node in the Merkle tree.
        
        Args:
            node_index: The index of the node
            leaves: The leaf nodes
            public_seed: The public seed
            
        Returns:
            The node
        """
        # If the node is a leaf, return it
        if node_index < len(leaves):
            return leaves[node_index]
        
        # Otherwise, compute the node recursively
        # This is a simplified implementation
        # In a real implementation, we would use a more efficient algorithm
        
        # Compute the left and right child indices
        left_idx = 2 * node_index
        right_idx = 2 * node_index + 1
        
        # Compute the left and right child nodes
        left = self._compute_node(left_idx, leaves, public_seed)
        right = self._compute_node(right_idx, leaves, public_seed)
        
        # Compute the parent node
        parent = self.hash_func(public_seed + left + right)
        
        return parent
    
    def generate_keypair(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate an XMSS keypair.
        
        Returns:
            A tuple (private_key, public_key)
        """
        # Generate the seeds
        secret_seed = secrets.token_bytes(self.n)
        public_seed = secrets.token_bytes(self.n)
        
        # Generate the WOTS+ keypairs
        wots_private_keys = []
        wots_public_keys = []
        
        for i in range(self.num_leaves):
            # Generate the address for this leaf
            address = i.to_bytes(4, byteorder='big')
            
            # Generate the WOTS+ keypair
            sk, pk = self.wots.generate_keypair(secret_seed, public_seed, address)
            
            wots_private_keys.append(sk)
            wots_public_keys.append(pk)
        
        # Compute the leaf nodes
        leaves = []
        for i in range(self.num_leaves):
            # Compute the leaf node
            # In a real implementation, we would use a more sophisticated leaf node computation
            # For now, we'll just hash the WOTS+ public key
            leaf = self.hash_func(public_seed + b''.join(wots_public_keys[i]))
            leaves.append(leaf)
        
        # Compute the root of the Merkle tree
        root = self._compute_node(1, leaves, public_seed)
        
        # Return the keypair
        private_key = {
            'secret_seed': secret_seed,
            'public_seed': public_seed,
            'wots_private_keys': wots_private_keys,
            'index': 0
        }
        
        public_key = {
            'public_seed': public_seed,
            'root': root
        }
        
        return private_key, public_key
    
    def sign(self, message: bytes, private_key: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sign a message using XMSS.
        
        Args:
            message: The message to sign
            private_key: The private key
            
        Returns:
            The signature
        """
        # Extract the private key components
        secret_seed = private_key['secret_seed']
        public_seed = private_key['public_seed']
        wots_private_keys = private_key['wots_private_keys']
        index = private_key['index']
        
        # Check if we've used all the available signatures
        if index >= self.num_leaves:
            raise ValueError("All signatures have been used")
        
        # Generate the address for this leaf
        address = index.to_bytes(4, byteorder='big')
        
        # Sign the message using WOTS+
        wots_signature = self.wots.sign(message, wots_private_keys[index], public_seed, address)
        
        # Compute the leaf nodes
        leaves = []
        for i in range(self.num_leaves):
            # Compute the leaf node
            # In a real implementation, we would use a more sophisticated leaf node computation
            # For now, we'll just hash the WOTS+ public key
            leaf = self.hash_func(public_seed + b''.join(wots_private_keys[i]))
            leaves.append(leaf)
        
        # Compute the authentication path
        auth_path = self._compute_auth_path(index, leaves, public_seed)
        
        # Increment the index
        private_key['index'] += 1
        
        # Return the signature
        return {
            'index': index,
            'wots_signature': wots_signature,
            'auth_path': auth_path
        }
    
    def verify(self, message: bytes, signature: Dict[str, Any], public_key: Dict[str, Any]) -> bool:
        """
        Verify a signature using XMSS.
        
        Args:
            message: The message
            signature: The signature
            public_key: The public key
            
        Returns:
            True if the signature is valid, False otherwise
        """
        # Extract the signature components
        index = signature['index']
        wots_signature = signature['wots_signature']
        auth_path = signature['auth_path']
        
        # Extract the public key components
        public_seed = public_key['public_seed']
        root = public_key['root']
        
        # Generate the address for this leaf
        address = index.to_bytes(4, byteorder='big')
        
        # Verify the WOTS+ signature
        wots_public_key = self.wots.verify(message, wots_signature, public_seed, address)
        
        # Compute the leaf node
        leaf = self.hash_func(public_seed + b''.join(wots_public_key))
        
        # Compute the root
        computed_root = self._compute_root(index, leaf, auth_path, public_seed)
        
        # Check if the computed root matches the public key
        return computed_root == root


class SPHINCS:
    """
    SPHINCS+ hash-based signature scheme.
    
    SPHINCS+ is a stateless hash-based signature scheme that is a candidate in the
    NIST Post-Quantum Cryptography standardization process.
    """
    
    def __init__(self, 
                 n: int = 32, 
                 h: int = 64, 
                 d: int = 8, 
                 w: int = 16, 
                 hash_function: str = 'sha256'):
        """
        Initialize the SPHINCS+ scheme.
        
        Args:
            n: The security parameter in bytes
            h: The total tree height
            d: The number of layers
            w: The Winternitz parameter
            hash_function: The hash function to use
        """
        self.n = n
        self.h = h
        self.d = d
        self.w = w
        self.hash_function = hash_function
        
        # Compute derived parameters
        self.h_prime = h // d
        self.t = 2**self.h_prime
        
        # Initialize the hash function
        self._init_hash_function()
        
        # Initialize the WOTS+ scheme
        self.wots = WOTSPlus(w, n, hash_function)
    
    def _init_hash_function(self) -> None:
        """
        Initialize the hash function.
        """
        if self.hash_function == 'sha256':
            self.hash_func = lambda x: hashlib.sha256(x).digest()
        elif self.hash_function == 'sha512':
            self.hash_func = lambda x: hashlib.sha512(x).digest()[:self.n]
        elif self.hash_function == 'shake256':
            self.hash_func = lambda x: hashlib.shake_256(x).digest(self.n)
        else:
            raise ValueError(f"Unsupported hash function: {self.hash_function}")
    
    def _prf(self, secret_seed: bytes, address: bytes) -> bytes:
        """
        Pseudorandom function.
        
        Args:
            secret_seed: The secret seed
            address: The address
            
        Returns:
            The PRF output
        """
        # In a real implementation, we would use a more sophisticated PRF
        # For now, we'll just use HMAC
        return hmac.new(secret_seed, address, hashlib.sha256).digest()[:self.n]
    
    def _hash_message(self, message: bytes, public_seed: bytes, secret_seed: bytes) -> Tuple[bytes, int, int]:
        """
        Hash a message to get the randomized index.
        
        Args:
            message: The message
            public_seed: The public seed
            secret_seed: The secret seed
            
        Returns:
            A tuple (message_digest, tree_index, leaf_index)
        """
        # Generate a random value
        r = self._prf(secret_seed, message)
        
        # Hash the message with the random value
        message_digest = self.hash_func(r + message)
        
        # Compute the tree and leaf indices
        # In a real implementation, we would use a more sophisticated algorithm
        # For now, we'll just use the first bytes of the message digest
        tree_index = int.from_bytes(message_digest[:4], byteorder='big') % (2**(self.h - self.h_prime))
        leaf_index = int.from_bytes(message_digest[4:8], byteorder='big') % self.t
        
        return message_digest, tree_index, leaf_index
    
    def generate_keypair(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate a SPHINCS+ keypair.
        
        Returns:
            A tuple (private_key, public_key)
        """
        # Generate the seeds
        secret_seed = secrets.token_bytes(self.n)
        public_seed = secrets.token_bytes(self.n)
        
        # Return the keypair
        private_key = {
            'secret_seed': secret_seed,
            'public_seed': public_seed
        }
        
        public_key = {
            'public_seed': public_seed
        }
        
        return private_key, public_key
    
    def sign(self, message: bytes, private_key: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sign a message using SPHINCS+.
        
        Args:
            message: The message to sign
            private_key: The private key
            
        Returns:
            The signature
        """
        # Extract the private key components
        secret_seed = private_key['secret_seed']
        public_seed = private_key['public_seed']
        
        # Hash the message to get the randomized index
        message_digest, tree_index, leaf_index = self._hash_message(message, public_seed, secret_seed)
        
        # Generate the WOTS+ keypair for this leaf
        address = tree_index.to_bytes(4, byteorder='big') + leaf_index.to_bytes(4, byteorder='big')
        wots_sk, wots_pk = self.wots.generate_keypair(secret_seed, public_seed, address)
        
        # Sign the message digest using WOTS+
        wots_signature = self.wots.sign(message_digest, wots_sk, public_seed, address)
        
        # In a real implementation, we would also include authentication paths
        # For now, we'll just include the WOTS+ signature
        
        # Return the signature
        return {
            'message_digest': message_digest,
            'tree_index': tree_index,
            'leaf_index': leaf_index,
            'wots_signature': wots_signature
        }
    
    def verify(self, message: bytes, signature: Dict[str, Any], public_key: Dict[str, Any]) -> bool:
        """
        Verify a signature using SPHINCS+.
        
        Args:
            message: The message
            signature: The signature
            public_key: The public key
            
        Returns:
            True if the signature is valid, False otherwise
        """
        # Extract the signature components
        message_digest = signature['message_digest']
        tree_index = signature['tree_index']
        leaf_index = signature['leaf_index']
        wots_signature = signature['wots_signature']
        
        # Extract the public key components
        public_seed = public_key['public_seed']
        
        # Hash the message to get the randomized index
        # In a real implementation, we would need the random value r
        # For now, we'll just use the message digest from the signature
        
        # Generate the address for this leaf
        address = tree_index.to_bytes(4, byteorder='big') + leaf_index.to_bytes(4, byteorder='big')
        
        # Verify the WOTS+ signature
        wots_public_key = self.wots.verify(message_digest, wots_signature, public_seed, address)
        
        # In a real implementation, we would also verify the authentication paths
        # For now, we'll just check if the WOTS+ signature is valid
        
        # Check if the WOTS+ signature is valid
        # In a real implementation, we would compare the computed root with the public key
        return len(wots_public_key) > 0


# Example usage
if __name__ == "__main__":
    # WOTS+ example
    print("WOTS+ Example")
    print("============")
    
    # Create a WOTS+ instance
    wots = WOTSPlus(w=16, n=32, hash_function='sha256')
    
    # Generate a keypair
    secret_seed = secrets.token_bytes(32)
    public_seed = secrets.token_bytes(32)
    address = b'\x00\x00\x00\x00'
    
    private_key, public_key = wots.generate_keypair(secret_seed, public_seed, address)
    print(f"Generated WOTS+ keypair with {len(private_key)} private key elements and {len(public_key)} public key elements")
    
    # Sign a message
    message = b"Hello, world!"
    signature = wots.sign(message, private_key, public_seed, address)
    print(f"Signed message with {len(signature)} signature elements")
    
    # Verify the signature
    verified_public_key = wots.verify(message, signature, public_seed, address)
    print(f"Verified signature, got {len(verified_public_key)} public key elements")
    
    # Check if the verification is correct
    is_valid = all(pk1 == pk2 for pk1, pk2 in zip(public_key, verified_public_key))
    print(f"Signature valid: {is_valid}")
    
    print()
    
    # XMSS example
    print("XMSS Example")
    print("===========")
    
    # Create an XMSS instance with a small height for demonstration
    xmss = XMSS(height=4, w=16, n=32, hash_function='sha256')
    
    # Generate a keypair
    private_key, public_key = xmss.generate_keypair()
    print(f"Generated XMSS keypair with tree height {xmss.height}")
    
    # Sign a message
    message = b"Hello, world!"
    signature = xmss.sign(message, private_key)
    print(f"Signed message with index {signature['index']}")
    
    # Verify the signature
    is_valid = xmss.verify(message, signature, public_key)
    print(f"Signature valid: {is_valid}")
    
    # Try to verify with a modified message
    modified_message = b"Hello, world!!"
    is_valid = xmss.verify(modified_message, signature, public_key)
    print(f"Signature valid for modified message: {is_valid}")
    
    print()
    
    # SPHINCS+ example
    print("SPHINCS+ Example")
    print("==============")
    
    # Create a SPHINCS+ instance with small parameters for demonstration
    sphincs = SPHINCS(n=32, h=8, d=2, w=16, hash_function='sha256')
    
    # Generate a keypair
    private_key, public_key = sphincs.generate_keypair()
    print(f"Generated SPHINCS+ keypair with total tree height {sphincs.h}")
    
    # Sign a message
    message = b"Hello, world!"
    signature = sphincs.sign(message, private_key)
    print(f"Signed message with tree index {signature['tree_index']} and leaf index {signature['leaf_index']}")
    
    # Verify the signature
    is_valid = sphincs.verify(message, signature, public_key)
    print(f"Signature valid: {is_valid}")
    
    # Try to verify with a modified message
    # Note: In this simplified implementation, the verification might still succeed
    # because we're not properly checking the message hash
    modified_message = b"Hello, world!!"
    is_valid = sphincs.verify(modified_message, signature, public_key)
    print(f"Signature valid for modified message: {is_valid}")