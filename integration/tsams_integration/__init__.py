"""
Quantum-Classical Bridge Components.

This package provides components for bridging between quantum and classical domains,
which is essential for understanding the emergence of classical behavior from
quantum mechanics and the quantum-classical transition.
"""

from .quantum_classical_correspondence import QuantumClassicalCorrespondence
from .decoherence_boundary import DecoherenceBoundary
from .measurement_projection import MeasurementProjection
from .classical_limit import ClassicalLimit

__all__ = [
    'QuantumClassicalCorrespondence',
    'DecoherenceBoundary',
    'MeasurementProjection',
    'ClassicalLimit'
]