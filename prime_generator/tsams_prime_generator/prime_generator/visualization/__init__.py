"""
Prime number visualization tools.

This module provides various visualization methods for prime numbers,
including classical representations and TSAMS-specific visualizations
based on advanced mathematical structures.
"""

# Classical prime visualizations
from .spirals import (
    ulam_spiral,
    sacks_spiral,
    hexagonal_prime_spiral
)

from .distribution import (
    plot_prime_distribution,
    prime_gaps_plot,
    prime_density_heatmap,
    prime_counting_function_plot,
    prime_race_plot
)

from .interactive import (
    interactive_prime_explorer,
    animated_sieve,
    prime_pattern_viewer
)

# TSAMS-specific visualizations
from .tsams_visualizations import (
    # Cyclotomic field visualizations
    cyclotomic_field_visualization,
    cyclotomic_prime_patterns,
    
    # Quantum-inspired visualizations
    quantum_interference_pattern,
    quantum_prime_probability,
    
    # E8 lattice visualizations
    e8_lattice_projection,
    e8_root_system_plot,
    
    # Modular forms visualizations
    modular_forms_zeros,
    modular_forms_prime_correlation,
    
    # L-function visualizations
    l_function_visualization,
    l_function_zeros_plot,
    
    # Advanced interactive tools
    tsams_prime_explorer
)