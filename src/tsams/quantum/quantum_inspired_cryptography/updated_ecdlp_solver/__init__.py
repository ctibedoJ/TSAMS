"""
TIBEDO Quantum-Inspired ECDLP Solver

This module implements a quantum-inspired algorithm for solving the Elliptic Curve
Discrete Logarithm Problem (ECDLP) using advanced mathematical structures inspired
by quantum computing principles, but running entirely on classical hardware.
"""

from .core_solver import QuantumInspiredECDLPSolver
from .optimizers import CyclotomicFieldOptimizer, SpinorOptimizer, DiscosohedralOptimizer
from .elliptic_curve import EllipticCurve, ECPoint

__all__ = [
    'QuantumInspiredECDLPSolver',
    'CyclotomicFieldOptimizer',
    'SpinorOptimizer',
    'DiscosohedralOptimizer',
    'EllipticCurve',
    'ECPoint'
]