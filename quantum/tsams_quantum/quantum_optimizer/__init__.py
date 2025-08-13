"""
TIBEDO Quantum Optimizer Module

This module implements classical algorithms that achieve quantum-like optimization
performance without requiring quantum hardware. It leverages the mathematical
foundations of the TIBEDO Framework to provide efficient solutions to optimization
problems traditionally targeted by quantum computing approaches.
"""

from .core_optimizer import QuantumInspiredOptimizer
from .annealing_simulator import QuantumAnnealingSimulator
from .tensor_optimizer import TensorNetworkOptimizer
from .variational_optimizer import ClassicalVQEOptimizer

__all__ = [
    'QuantumInspiredOptimizer',
    'QuantumAnnealingSimulator',
    'TensorNetworkOptimizer',
    'ClassicalVQEOptimizer'
]