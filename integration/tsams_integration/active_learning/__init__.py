"""
Active Learning Module for TIBEDO Framework

This module provides active learning tools for efficient exploration of chemical space
and other high-dimensional parameter spaces within the TIBEDO Framework.
"""

from .active_learner import ActiveLearner
from .chemical_space_explorer import ChemicalSpaceExplorer, ChemicalSpaceDataset, AcquisitionFunction