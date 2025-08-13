# TSAMS Cryptography

Cryptographic applications of the Tibedo Structural Algebraic Modeling System (TSAMS), including the groundbreaking ECDLP solver.

## Overview

TSAMS Cryptography provides implementations of cryptographic applications based on the mathematical framework of TSAMS. The centerpiece is the revolutionary ECDLP (Elliptic Curve Discrete Logarithm Problem) solver, which demonstrates significant advancements over traditional approaches.

Key components include:

- **Classical 256-bit ECDLP Solver**: A breakthrough implementation that can solve the ECDLP for 256-bit curves using novel mathematical techniques derived from the TSAMS framework
- **Quantum-Resistant Primitives**: Cryptographic primitives designed to be resistant to quantum computing attacks
- **Hybrid Cryptosystems**: Systems that combine classical and quantum-resistant approaches for maximum security

## Installation

```bash
pip install tsams-cryptography
```

## Usage

### Classical 256-bit ECDLP Solver

```python
from tsams_cryptography.ecdlp import ClassicalECDLPSolver

# Initialize the solver
solver = ClassicalECDLPSolver(curve='secp256k1')

# Define the public key point (the result of k*G where k is the private key)
public_key = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
              0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)

# Solve for the private key
private_key = solver.solve(public_key)
print(f"Private key: {private_key}")
```

### Quantum-Resistant Primitives

```python
from tsams_cryptography.quantum_resistant import LatticeBasedEncryption

# Initialize the encryption scheme
encryption = LatticeBasedEncryption()

# Generate key pair
public_key, private_key = encryption.generate_key_pair()

# Encrypt a message
message = b"This is a secret message"
ciphertext = encryption.encrypt(message, public_key)

# Decrypt the message
decrypted = encryption.decrypt(ciphertext, private_key)
print(f"Decrypted message: {decrypted.decode()}")
```

## Features

### Classical 256-bit ECDLP Solver

- **Revolutionary Algorithm**: Based on the TSAMS mathematical framework
- **High Performance**: Optimized implementation for maximum efficiency
- **Scalable**: Can be extended to larger bit sizes
- **Parallelizable**: Can leverage multi-core and distributed computing

### Quantum-Resistant Primitives

- **Lattice-Based Cryptography**: Implementations of lattice-based encryption schemes
- **Hash-Based Signatures**: Post-quantum digital signature schemes
- **Code-Based Cryptography**: Error-correcting code based encryption

### Hybrid Cryptosystems

- **Combined Security**: Leverages both classical and quantum-resistant approaches
- **Flexible Integration**: Can be integrated with existing cryptographic infrastructure
- **Forward Security**: Designed to maintain security even if one component is compromised

## Documentation

For detailed documentation, see the [TSAMS Documentation](https://github.com/ctibedoJ/tsams-docs).

## Dependencies

- numpy
- sympy
- tsams-core
- tsams-classical

## License

This project is licensed under the MIT License - see the LICENSE file for details.