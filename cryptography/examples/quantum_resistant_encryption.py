#!/usr/bin/env python3
"""
Example script demonstrating the usage of quantum-resistant encryption.
"""

import argparse
import time
import os
from tsams_cryptography.quantum_resistant import LatticeBasedEncryption


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Demonstrate quantum-resistant encryption.')
    parser.add_argument('--mode', type=str, choices=['encrypt', 'decrypt'], required=True,
                        help='The mode of operation (encrypt or decrypt)')
    parser.add_argument('--input', type=str, required=True,
                        help='The input file (plaintext for encryption, ciphertext for decryption)')
    parser.add_argument('--output', type=str, required=True,
                        help='The output file (ciphertext for encryption, plaintext for decryption)')
    parser.add_argument('--public-key', type=str, default=None,
                        help='The public key file (required for encryption)')
    parser.add_argument('--private-key', type=str, default=None,
                        help='The private key file (required for decryption)')
    parser.add_argument('--generate-keys', action='store_true',
                        help='Generate a new key pair')
    parser.add_argument('--public-key-output', type=str, default='public_key.bin',
                        help='The output file for the public key (default: public_key.bin)')
    parser.add_argument('--private-key-output', type=str, default='private_key.bin',
                        help='The output file for the private key (default: private_key.bin)')
    args = parser.parse_args()
    
    # Initialize the encryption scheme
    encryption = LatticeBasedEncryption()
    
    # Generate a new key pair if requested
    if args.generate_keys:
        print("Generating a new key pair...")
        start_time = time.time()
        public_key, private_key = encryption.generate_key_pair()
        elapsed_time = time.time() - start_time
        print(f"Key pair generated in {elapsed_time:.2f} seconds")
        
        # Save the keys to files
        with open(args.public_key_output, 'wb') as f:
            f.write(public_key)
        print(f"Public key saved to {args.public_key_output}")
        
        with open(args.private_key_output, 'wb') as f:
            f.write(private_key)
        print(f"Private key saved to {args.private_key_output}")
    
    # Encrypt or decrypt the input file
    if args.mode == 'encrypt':
        # Check if the public key is provided
        if args.public_key is None and not args.generate_keys:
            print("Error: Public key is required for encryption")
            return 1
        
        # Load the public key
        if args.generate_keys:
            # Use the newly generated public key
            with open(args.public_key_output, 'rb') as f:
                public_key = f.read()
        else:
            # Load the public key from the specified file
            with open(args.public_key, 'rb') as f:
                public_key = f.read()
        
        # Load the plaintext
        with open(args.input, 'rb') as f:
            plaintext = f.read()
        
        # Encrypt the plaintext
        print(f"Encrypting {len(plaintext)} bytes...")
        start_time = time.time()
        ciphertext = encryption.encrypt(plaintext, public_key)
        elapsed_time = time.time() - start_time
        print(f"Encryption completed in {elapsed_time:.2f} seconds")
        
        # Save the ciphertext
        with open(args.output, 'wb') as f:
            f.write(ciphertext)
        print(f"Ciphertext saved to {args.output}")
    
    elif args.mode == 'decrypt':
        # Check if the private key is provided
        if args.private_key is None and not args.generate_keys:
            print("Error: Private key is required for decryption")
            return 1
        
        # Load the private key
        if args.generate_keys:
            # Use the newly generated private key
            with open(args.private_key_output, 'rb') as f:
                private_key = f.read()
        else:
            # Load the private key from the specified file
            with open(args.private_key, 'rb') as f:
                private_key = f.read()
        
        # Load the ciphertext
        with open(args.input, 'rb') as f:
            ciphertext = f.read()
        
        # Decrypt the ciphertext
        print(f"Decrypting {len(ciphertext)} bytes...")
        start_time = time.time()
        plaintext = encryption.decrypt(ciphertext, private_key)
        elapsed_time = time.time() - start_time
        print(f"Decryption completed in {elapsed_time:.2f} seconds")
        
        # Save the plaintext
        with open(args.output, 'wb') as f:
            f.write(plaintext)
        print(f"Plaintext saved to {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())