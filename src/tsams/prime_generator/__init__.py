"""
Prime-Indexed Congruential Relations Implementation

This module implements the Prime-Indexed Congruential Relations component of the TIBEDO Framework,
which provides computational shortcuts that contribute significantly to its linear time complexity.
"""

from .prime_indexed_structure import PrimeIndexedStructure
from .modular_system import ModularSystem
from .congruential_accelerator import CongruentialAccelerator

__all__ = ['PrimeIndexedStructure', 'ModularSystem', 'CongruentialAccelerator']