"""
TIBEDO Quantum Cryptography Module

This module integrates all quantum cryptography components of the TIBEDO Framework,
including quantum ECDLP solver, post-quantum cryptographic primitives, and quantum
random number generation.
"""

from .extended_quantum_ecdlp_solver import ExtendedQuantumECDLPSolver, ExtendedQuantumECDLPCircuitGenerator
from .lattice_based_cryptography import (
    LatticeParameters, LatticeUtils, RingLWE, ModuleLWE, Kyber, Dilithium
)
from .hash_based_signatures import WOTSPlus, XMSS, SPHINCS
from .isogeny_based_cryptography import (
    FiniteField, QuadraticExtensionField, EllipticCurve, MontgomeryCurve, SIKE
)
from .quantum_random_number_generator import (
    QuantumRandomnessSource, HardwareQuantumRandomnessSource, SoftwareQuantumRandomnessSource,
    QuantumRandomnessTest, QuantumRandomNumberGenerator, QuantumRandomNumberService
)

__all__ = [
    # Extended Quantum ECDLP Solver
    'ExtendedQuantumECDLPSolver',
    'ExtendedQuantumECDLPCircuitGenerator',
    
    # Lattice-Based Cryptography
    'LatticeParameters',
    'LatticeUtils',
    'RingLWE',
    'ModuleLWE',
    'Kyber',
    'Dilithium',
    
    # Hash-Based Signatures
    'WOTSPlus',
    'XMSS',
    'SPHINCS',
    
    # Isogeny-Based Cryptography
    'FiniteField',
    'QuadraticExtensionField',
    'EllipticCurve',
    'MontgomeryCurve',
    'SIKE',
    
    # Quantum Random Number Generator
    'QuantumRandomnessSource',
    'HardwareQuantumRandomnessSource',
    'SoftwareQuantumRandomnessSource',
    'QuantumRandomnessTest',
    'QuantumRandomNumberGenerator',
    'QuantumRandomNumberService'
]