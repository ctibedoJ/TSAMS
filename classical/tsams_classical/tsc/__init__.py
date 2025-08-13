"""
Throw-Shot-Catch (TSC) Algorithm Implementation

This module implements the Throw-Shot-Catch algorithm, a three-phase computational process
for solving the Elliptic Curve Discrete Logarithm Problem (ECDLP) in linear time.

The algorithm consists of three phases:
1. Throw Phase: Initialize computation with parameters from the elliptic curve
2. Shot Phase: Apply transformations based on the spinor reduction chain
3. Catch Phase: Extract the discrete logarithm from the transformed state
"""

from .throw import ThrowPhase
from .shot import ShotPhase
from .catch import CatchPhase
from .tsc_solver import TSCSolver

__all__ = ['ThrowPhase', 'ShotPhase', 'CatchPhase', 'TSCSolver']