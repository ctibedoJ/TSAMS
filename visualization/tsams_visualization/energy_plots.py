"""
Energy Spectrum Visualization module.

This module provides tools for visualizing the energy spectrum of the 420-root
Möbius structure, including level spacing distributions, spectral rigidity, and more.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional, Callable, Any

from ..moebius.root_420_structure import Root420Structure
from ..hyperbolic.energy_quantization import EnergyQuantization
from ..hyperbolic.energy_spectrum import EnergySpectrum


def plot_energy_spectrum(max_prime: Optional[int] = None,
                        fig_size: Tuple[int, int] = (10, 6),
                        marker_size: int = 20,
                        color: str = 'blue',
                        alpha: float = 0.7,
                        show_grid: bool = True,
                        save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the energy spectrum of the 420-root Möbius structure.
    
    Args:
        max_prime (int, optional): The maximum prime to include. If None, all primes in the
            420-root structure are included.
        fig_size (Tuple[int, int]): The size of the figure (default: (10, 6)).
        marker_size (int): The size of the markers (default: 20).
        color (str): The color of the markers (default: 'blue').
        alpha (float): The transparency of the markers (default: 0.7).
        show_grid (bool): Whether to show the grid (default: True).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes with the visualization.
    """
    # Create an energy spectrum analyzer
    energy_spectrum = EnergySpectrum()
    
    # Compute the energy spectrum
    spectrum = energy_spectrum.compute_spectrum()
    
    # Filter by max_prime if specified
    if max_prime is not None:
        spectrum = {p: e for p, e in spectrum.items() if p <= max_prime}
    
    # Extract the primes and energies
    primes = sorted(spectrum.keys())
    energies = [spectrum[p] for p in primes]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot the spectrum
    ax.scatter(primes, energies, s=marker_size, alpha=alpha, color=color)
    
    # Set the labels and title
    ax.set_xlabel('Prime Index (p)')
    ax.set_ylabel('Energy E(M_p)')
    ax.set_title('Energy Spectrum of the 420-root Möbius Structure')
    
    # Show the grid if requested
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, ax


def plot_level_spacing_distribution(max_prime: Optional[int] = None,
                                   fig_size: Tuple[int, int] = (10, 6),
                                   num_bins: int = 20,
                                   show_wigner: bool = True,
                                   show_poisson: bool = True,
                                   save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the level spacing distribution of the energy spectrum.
    
    Args:
        max_prime (int, optional): The maximum prime to include. If None, all primes in the
            420-root structure are included.
        fig_size (Tuple[int, int]): The size of the figure (default: (10, 6)).
        num_bins (int): The number of bins in the histogram (default: 20).
        show_wigner (bool): Whether to show the Wigner surmise (default: True).
        show_poisson (bool): Whether to show the Poisson distribution (default: True).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes with the visualization.
    """
    # Create an energy spectrum analyzer
    energy_spectrum = EnergySpectrum()
    
    # Compute the nearest neighbor spacing distribution
    spacings, frequencies = energy_spectrum.nearest_neighbor_spacing_distribution()
    
    # Create the figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot the distribution
    ax.plot(spacings, frequencies, 'o-', label='420-root Möbius Structure')
    
    # Show the Wigner surmise if requested
    if show_wigner:
        wigner_values = [energy_spectrum.energy_calculator.wigner_surmise(s) for s in spacings]
        ax.plot(spacings, wigner_values, 'r--', label='Wigner Surmise (GOE)')
    
    # Show the Poisson distribution if requested
    if show_poisson:
        poisson_values = [np.exp(-s) for s in spacings]
        ax.plot(spacings, poisson_values, 'g-.', label='Poisson (Uncorrelated)')
    
    # Set the labels and title
    ax.set_xlabel('Normalized Spacing (s)')
    ax.set_ylabel('Probability Density P(s)')
    ax.set_title('Level Spacing Distribution')
    
    # Add a legend
    ax.legend()
    
    # Show the grid
    ax.grid(True, alpha=0.3)
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, ax


def plot_spectral_rigidity(max_prime: Optional[int] = None,
                          max_L: float = 20,
                          num_points: int = 20,
                          fig_size: Tuple[int, int] = (10, 6),
                          show_goe: bool = True,
                          show_poisson: bool = True,
                          save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the spectral rigidity curve.
    
    Args:
        max_prime (int, optional): The maximum prime to include. If None, all primes in the
            420-root structure are included.
        max_L (float): The maximum interval length (default: 20).
        num_points (int): The number of points in the curve (default: 20).
        fig_size (Tuple[int, int]): The size of the figure (default: (10, 6)).
        show_goe (bool): Whether to show the GOE prediction (default: True).
        show_poisson (bool): Whether to show the Poisson prediction (default: True).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes with the visualization.
    """
    # Create an energy spectrum analyzer
    energy_spectrum = EnergySpectrum()
    
    # Compute the spectral rigidity curve
    L_values, rigidity_values = energy_spectrum.spectral_rigidity_curve(max_L, num_points)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot the spectral rigidity curve
    ax.plot(L_values, rigidity_values, 'o-', label='420-root Möbius Structure')
    
    # Show the GOE prediction if requested
    if show_goe:
        # The GOE prediction for the spectral rigidity is approximately (1/π²) * log(L)
        goe_values = [(1 / np.pi**2) * np.log(L) for L in L_values]
        ax.plot(L_values, goe_values, 'r--', label='GOE Prediction')
    
    # Show the Poisson prediction if requested
    if show_poisson:
        # The Poisson prediction for the spectral rigidity is approximately L/15
        poisson_values = [L / 15 for L in L_values]
        ax.plot(L_values, poisson_values, 'g-.', label='Poisson Prediction')
    
    # Set the labels and title
    ax.set_xlabel('Interval Length (L)')
    ax.set_ylabel('Spectral Rigidity Δ₃(L)')
    ax.set_title('Spectral Rigidity Curve')
    
    # Add a legend
    ax.legend()
    
    # Show the grid
    ax.grid(True, alpha=0.3)
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, ax


def plot_energy_histogram(max_prime: Optional[int] = None,
                         num_bins: int = 20,
                         fig_size: Tuple[int, int] = (10, 6),
                         color: str = 'blue',
                         alpha: float = 0.7,
                         save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a histogram of the energy spectrum.
    
    Args:
        max_prime (int, optional): The maximum prime to include. If None, all primes in the
            420-root structure are included.
        num_bins (int): The number of bins in the histogram (default: 20).
        fig_size (Tuple[int, int]): The size of the figure (default: (10, 6)).
        color (str): The color of the histogram (default: 'blue').
        alpha (float): The transparency of the histogram (default: 0.7).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes with the visualization.
    """
    # Create an energy spectrum analyzer
    energy_spectrum = EnergySpectrum()
    
    # Compute the energy histogram
    bin_edges, counts = energy_spectrum.energy_histogram(num_bins)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot the histogram
    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), color=color, alpha=alpha)
    
    # Set the labels and title
    ax.set_xlabel('Energy E(M_p)')
    ax.set_ylabel('Count')
    ax.set_title('Energy Spectrum Histogram')
    
    # Show the grid
    ax.grid(True, alpha=0.3)
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, ax


def plot_energy_variance(structural_parameters: List[int] = [210, 420, 840],
                        fig_size: Tuple[int, int] = (10, 6),
                        marker_size: int = 20,
                        save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the energy variance for different structural parameters.
    
    Args:
        structural_parameters (List[int]): The structural parameters to compare (default: [210, 420, 840]).
        fig_size (Tuple[int, int]): The size of the figure (default: (10, 6)).
        marker_size (int): The size of the markers (default: 20).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes with the visualization.
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Compute the energy variance for each structural parameter
    variances = []
    
    for n in structural_parameters:
        # Create an energy calculator with the given structural parameter
        energy_calculator = EnergyQuantization(n)
        
        # Compute the energy spectrum for primes up to 100
        spectrum = energy_calculator.energy_spectrum(100)
        
        # Compute the variance
        variance = np.var(list(spectrum.values()))
        variances.append(variance)
    
    # Plot the variances
    ax.scatter(structural_parameters, variances, s=marker_size)
    ax.plot(structural_parameters, variances, 'b-')
    
    # Set the labels and title
    ax.set_xlabel('Structural Parameter (n)')
    ax.set_ylabel('Energy Variance')
    ax.set_title('Energy Variance vs. Structural Parameter')
    
    # Show the grid
    ax.grid(True, alpha=0.3)
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, ax