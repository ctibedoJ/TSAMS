"""
TIBEDO Quantum-Inspired Classical Cryptography Module

This module implements quantum-inspired classical cryptographic algorithms that leverage
quantum mathematical structures and principles to enhance security and performance
while running entirely on classical hardware.
"""

from .quantum_inspired_ecdlp_solver import QuantumInspiredECDLPSolver
from .quantum_enhanced_lattice_crypto import EnhancedLatticeEncryption, EnhancedLatticeSignature
from .quantum_resistant_hash_signatures import EnhancedMerkleSignature, AdaptiveHashSignature
from .quantum_inspired_isogeny_crypto import SupersingularIsogenyEncryption
from .quantum_inspired_randomness import EnhancedRandomnessGenerator, EntropyAmplifier

__all__ = [
    # Quantum-Inspired ECDLP Solver
    'QuantumInspiredECDLPSolver',
    
    # Quantum-Enhanced Lattice Cryptography
    'EnhancedLatticeEncryption',
    'EnhancedLatticeSignature',
    
    # Quantum-Resistant Hash Signatures
    'EnhancedMerkleSignature',
    'AdaptiveHashSignature',
    
    # Quantum-Inspired Isogeny Cryptography
    'SupersingularIsogenyEncryption',
    
    # Quantum-Inspired Randomness
    'EnhancedRandomnessGenerator',
    'EntropyAmplifier'
]