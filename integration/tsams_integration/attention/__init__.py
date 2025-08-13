"""
Attention Mechanisms for TIBEDO Framework

This module provides attention mechanisms and related components for neural networks
in the TIBEDO Framework, with a focus on quantum chemistry applications.
"""

from .spinor_positional_encoding import (
    SpinorPositionalEncoding,
    PhaseSynchronizedAttention,
    QuantumChemistryTransformer,
    TransformerLayer,
    QuantumChemistryEnergyPredictor,
    MolecularPropertyPredictor
)