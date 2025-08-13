"""
Energy Spectrum implementation.

This module provides a comprehensive implementation of energy spectrum analysis
described in Chapter 16 and 17 of the textbook, which reveals connections between
the 420-root Möbius structure and quantum chaotic systems.
"""

import numpy as np
import sympy
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional, Callable, Any

from ..moebius.root_420_structure import Root420Structure
from .energy_quantization import EnergyQuantization, EnergySpectrum


class EnergySpectrumAnalysis:
    """
    A class for advanced analysis of the energy spectrum of the 420-root Möbius structure.
    
    This class extends the basic EnergySpectrum class with more sophisticated
    analysis methods and visualization capabilities.
    
    Attributes:
        energy_spectrum (EnergySpectrum): The energy spectrum analyzer.
        spectrum_data (Dict[int, float]): The cached energy spectrum data.
    """
    
    def __init__(self):
        """
        Initialize an energy spectrum analysis tool.
        """
        self.energy_spectrum = EnergySpectrum()
        self.spectrum_data = self.energy_spectrum.compute_spectrum()
    
    def level_density(self, energy_range: Tuple[float, float], num_bins: int = 50) -> Tuple[List[float], List[float]]:
        """
        Compute the level density of the energy spectrum.
        
        Args:
            energy_range (Tuple[float, float]): The energy range to consider.
            num_bins (int): The number of bins (default: 50).
                
        Returns:
            Tuple[List[float], List[float]]: The energy values and level densities.
        """
        # Extract the energy values
        energies = list(self.spectrum_data.values())
        
        # Filter energies within the specified range
        filtered_energies = [e for e in energies if energy_range[0] <= e <= energy_range[1]]
        
        # Compute the histogram
        counts, bin_edges = np.histogram(filtered_energies, bins=num_bins)
        
        # Compute the bin centers
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
        
        # Normalize the counts to get the level density
        level_density = [count / (bin_edges[i+1] - bin_edges[i]) / len(filtered_energies)
                        for i, count in enumerate(counts)]
        
        return bin_centers, level_density
    
    def level_spacing_distribution(self, num_bins: int = 50) -> Tuple[List[float], List[float]]:
        """
        Compute the level spacing distribution of the energy spectrum.
        
        Args:
            num_bins (int): The number of bins (default: 50).
                
        Returns:
            Tuple[List[float], List[float]]: The spacing values and their distribution.
        """
        # Extract the energy values and sort them
        energies = sorted(self.spectrum_data.values())
        
        # Compute the spacings
        spacings = [energies[i+1] - energies[i] for i in range(len(energies)-1)]
        
        # Normalize the spacings
        mean_spacing = np.mean(spacings)
        normalized_spacings = [s / mean_spacing for s in spacings]
        
        # Compute the histogram
        counts, bin_edges = np.histogram(normalized_spacings, bins=num_bins, density=True)
        
        # Compute the bin centers
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
        
        return bin_centers, counts.tolist()
    
    def number_variance(self, max_L: float = 20, num_points: int = 20) -> Tuple[List[float], List[float]]:
        """
        Compute the number variance of the energy spectrum.
        
        The number variance Σ²(L) measures the variance of the number of levels
        in an interval of length L.
        
        Args:
            max_L (float): The maximum interval length (default: 20).
            num_points (int): The number of points in the curve (default: 20).
                
        Returns:
            Tuple[List[float], List[float]]: The interval lengths and variance values.
        """
        # Extract the energy values and sort them
        energies = sorted(self.spectrum_data.values())
        
        # Normalize the energies to have mean spacing 1
        mean_spacing = (energies[-1] - energies[0]) / (len(energies) - 1)
        normalized_energies = [e / mean_spacing for e in energies]
        
        # Compute the number variance for different interval lengths
        L_values = np.linspace(1, max_L, num_points)
        variance_values = []
        
        for L in L_values:
            # Count the number of levels in intervals of length L
            counts = []
            
            for i in range(len(normalized_energies) - int(L)):
                # Count levels in the interval [normalized_energies[i], normalized_energies[i] + L]
                count = sum(1 for e in normalized_energies
                           if normalized_energies[i] <= e < normalized_energies[i] + L)
                counts.append(count)
            
            # Compute the variance of the counts
            variance = np.var(counts)
            variance_values.append(variance)
        
        return L_values.tolist(), variance_values
    
    def form_factor(self, max_tau: float = 10, num_points: int = 100) -> Tuple[List[float], List[float]]:
        """
        Compute the form factor of the energy spectrum.
        
        The form factor K(τ) is the Fourier transform of the two-point correlation function.
        
        Args:
            max_tau (float): The maximum time parameter (default: 10).
            num_points (int): The number of points in the curve (default: 100).
                
        Returns:
            Tuple[List[float], List[float]]: The time parameters and form factor values.
        """
        # Extract the energy values
        energies = list(self.spectrum_data.values())
        
        # Normalize the energies to have mean spacing 1
        mean_spacing = np.mean([energies[i+1] - energies[i] for i in range(len(energies)-1)])
        normalized_energies = [e / mean_spacing for e in energies]
        
        # Compute the form factor for different time parameters
        tau_values = np.linspace(0.1, max_tau, num_points)
        form_factor_values = []
        
        for tau in tau_values:
            # Compute the form factor using the definition
            K = 0
            for i in range(len(normalized_energies)):
                for j in range(len(normalized_energies)):
                    if i != j:
                        K += np.exp(2j * np.pi * tau * (normalized_energies[i] - normalized_energies[j]))
            
            K = abs(K) / len(normalized_energies)**2
            form_factor_values.append(K)
        
        return tau_values.tolist(), form_factor_values
    
    def level_clustering(self) -> float:
        """
        Compute the level clustering parameter of the energy spectrum.
        
        The level clustering parameter measures the tendency of energy levels
        to cluster together.
        
        Returns:
            float: The level clustering parameter.
        """
        # Extract the energy values and sort them
        energies = sorted(self.spectrum_data.values())
        
        # Compute the spacings
        spacings = [energies[i+1] - energies[i] for i in range(len(energies)-1)]
        
        # Normalize the spacings
        mean_spacing = np.mean(spacings)
        normalized_spacings = [s / mean_spacing for s in spacings]
        
        # Compute the level clustering parameter
        # This is defined as the variance of the nearest neighbor spacings
        return np.var(normalized_spacings)
    
    def spectral_compressibility(self) -> float:
        """
        Compute the spectral compressibility of the energy spectrum.
        
        The spectral compressibility χ is related to the asymptotic behavior
        of the number variance: Σ²(L) ~ χL for large L.
        
        Returns:
            float: The spectral compressibility.
        """
        # Compute the number variance for large L
        L_values, variance_values = self.number_variance(max_L=50, num_points=10)
        
        # Fit a line to the large-L part of the curve
        x = np.array(L_values[-5:])
        y = np.array(variance_values[-5:])
        
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # The slope m is the spectral compressibility
        return m
    
    def level_statistics_comparison(self) -> Dict[str, Any]:
        """
        Compare the level statistics of the energy spectrum with theoretical predictions.
        
        Returns:
            Dict[str, Any]: A dictionary containing various comparison results.
        """
        results = {}
        
        # Compare with the Wigner surmise
        spacings, distribution = self.level_spacing_distribution()
        wigner_values = [self.energy_spectrum.energy_calculator.wigner_surmise(s) for s in spacings]
        
        results['spacings'] = spacings
        results['distribution'] = distribution
        results['wigner_values'] = wigner_values
        
        # Compute the mean ratio of consecutive spacings
        energies = sorted(self.spectrum_data.values())
        spacings = [energies[i+1] - energies[i] for i in range(len(energies)-1)]
        ratios = [min(spacings[i], spacings[i+1]) / max(spacings[i], spacings[i+1])
                 for i in range(len(spacings)-1)]
        
        results['mean_ratio'] = np.mean(ratios)
        results['goe_prediction'] = 0.536  # GOE prediction
        results['poisson_prediction'] = 0.386  # Poisson prediction
        
        # Compute the level clustering parameter
        results['level_clustering'] = self.level_clustering()
        results['goe_clustering'] = 0.178  # GOE prediction
        results['poisson_clustering'] = 1.0  # Poisson prediction
        
        # Compute the spectral compressibility
        results['spectral_compressibility'] = self.spectral_compressibility()
        results['goe_compressibility'] = 0.0  # GOE prediction
        results['poisson_compressibility'] = 1.0  # Poisson prediction
        
        return results
    
    def energy_level_dynamics(self, parameter_range: Tuple[float, float], num_points: int = 100) -> Dict[int, List[float]]:
        """
        Compute the energy level dynamics as a function of a parameter.
        
        Args:
            parameter_range (Tuple[float, float]): The parameter range to consider.
            num_points (int): The number of points in the parameter range (default: 100).
                
        Returns:
            Dict[int, List[float]]: A dictionary mapping prime indices to lists of energy values.
        """
        # Create a range of parameter values
        parameter_values = np.linspace(parameter_range[0], parameter_range[1], num_points)
        
        # Compute the energy for each prime and parameter value
        dynamics = {}
        
        for p in self.spectrum_data.keys():
            energy_values = []
            
            for param in parameter_values:
                # Compute the energy with the modified parameter
                energy = (p**2 / 420) * (1 + param * np.sin(2 * np.pi * p / 420))
                energy_values.append(energy)
            
            dynamics[p] = energy_values
        
        return dynamics
    
    def spectral_correlations(self) -> np.ndarray:
        """
        Compute the spectral correlations between energy levels.
        
        Returns:
            np.ndarray: The correlation matrix.
        """
        # Extract the energy values
        primes = sorted(self.spectrum_data.keys())
        energies = [self.spectrum_data[p] for p in primes]
        
        # Compute the correlation matrix
        return np.corrcoef(energies)
    
    def __str__(self) -> str:
        """
        Return a string representation of the energy spectrum analysis tool.
        
        Returns:
            str: A string representation of the analysis tool.
        """
        return "Energy Spectrum Analysis Tool for the 420-root Möbius Structure"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the energy spectrum analysis tool.
        
        Returns:
            str: A string representation of the analysis tool.
        """
        return "EnergySpectrumAnalysis()"


class EnergySpectrumVisualization:
    """
    A class for visualizing the energy spectrum of the 420-root Möbius structure.
    
    This class provides methods for creating various plots and visualizations
    of the energy spectrum and its statistical properties.
    
    Attributes:
        analysis (EnergySpectrumAnalysis): The energy spectrum analysis tool.
    """
    
    def __init__(self):
        """
        Initialize an energy spectrum visualization tool.
        """
        self.analysis = EnergySpectrumAnalysis()
    
    def plot_energy_spectrum(self, save_path: Optional[str] = None) -> None:
        """
        Plot the energy spectrum of the 420-root Möbius structure.
        
        Args:
            save_path (Optional[str]): The path to save the plot (default: None).
        """
        # Extract the data
        primes = sorted(self.analysis.spectrum_data.keys())
        energies = [self.analysis.spectrum_data[p] for p in primes]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.scatter(primes, energies, s=20, alpha=0.7)
        plt.xlabel('Prime Index (p)')
        plt.ylabel('Energy E(M_p)')
        plt.title('Energy Spectrum of the 420-root Möbius Structure')
        plt.grid(True, alpha=0.3)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_level_spacing_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Plot the level spacing distribution of the energy spectrum.
        
        Args:
            save_path (Optional[str]): The path to save the plot (default: None).
        """
        # Compute the level spacing distribution
        spacings, distribution = self.analysis.level_spacing_distribution()
        
        # Compute the Wigner surmise for comparison
        wigner_values = [self.analysis.energy_spectrum.energy_calculator.wigner_surmise(s) for s in spacings]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(spacings, distribution, 'o-', label='420-root Möbius Structure')
        plt.plot(spacings, wigner_values, 'r--', label='Wigner Surmise (GOE)')
        
        # Add a Poisson distribution for comparison
        poisson_values = [np.exp(-s) for s in spacings]
        plt.plot(spacings, poisson_values, 'g-.', label='Poisson (Uncorrelated)')
        
        plt.xlabel('Normalized Spacing (s)')
        plt.ylabel('Probability Density P(s)')
        plt.title('Level Spacing Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_number_variance(self, save_path: Optional[str] = None) -> None:
        """
        Plot the number variance of the energy spectrum.
        
        Args:
            save_path (Optional[str]): The path to save the plot (default: None).
        """
        # Compute the number variance
        L_values, variance_values = self.analysis.number_variance()
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(L_values, variance_values, 'o-', label='420-root Möbius Structure')
        
        # Add theoretical predictions for comparison
        goe_values = [np.log(2 * np.pi * L) / np.pi**2 + 0.5 - 0.25 * np.pi**2 / 8 for L in L_values]
        poisson_values = [L for L in L_values]
        
        plt.plot(L_values, goe_values, 'r--', label='GOE Prediction')
        plt.plot(L_values, poisson_values, 'g-.', label='Poisson Prediction')
        
        plt.xlabel('Interval Length (L)')
        plt.ylabel('Number Variance Σ²(L)')
        plt.title('Number Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_form_factor(self, save_path: Optional[str] = None) -> None:
        """
        Plot the form factor of the energy spectrum.
        
        Args:
            save_path (Optional[str]): The path to save the plot (default: None).
        """
        # Compute the form factor
        tau_values, form_factor_values = self.analysis.form_factor()
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(tau_values, form_factor_values, 'o-', label='420-root Möbius Structure')
        
        # Add theoretical predictions for comparison
        goe_values = []
        for tau in tau_values:
            if tau < 1:
                goe_values.append(2 * tau - tau * np.log(1 + 2 * tau))
            else:
                goe_values.append(2 - tau * np.log((2 * tau + 1) / (2 * tau - 1)))
        
        poisson_values = [1 for _ in tau_values]
        
        plt.plot(tau_values, goe_values, 'r--', label='GOE Prediction')
        plt.plot(tau_values, poisson_values, 'g-.', label='Poisson Prediction')
        
        plt.xlabel('Time (τ)')
        plt.ylabel('Form Factor K(τ)')
        plt.title('Spectral Form Factor')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_energy_level_dynamics(self, save_path: Optional[str] = None) -> None:
        """
        Plot the energy level dynamics as a function of a parameter.
        
        Args:
            save_path (Optional[str]): The path to save the plot (default: None).
        """
        # Compute the energy level dynamics
        dynamics = self.analysis.energy_level_dynamics((-0.5, 0.5))
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot the first 20 energy levels
        primes = sorted(dynamics.keys())[:20]
        parameter_values = np.linspace(-0.5, 0.5, len(next(iter(dynamics.values()))))
        
        for p in primes:
            plt.plot(parameter_values, dynamics[p], label=f'p = {p}')
        
        plt.xlabel('Parameter Value')
        plt.ylabel('Energy E(M_p)')
        plt.title('Energy Level Dynamics')
        plt.grid(True, alpha=0.3)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_spectral_correlations(self, save_path: Optional[str] = None) -> None:
        """
        Plot the spectral correlations between energy levels.
        
        Args:
            save_path (Optional[str]): The path to save the plot (default: None).
        """
        # Compute the spectral correlations
        correlation_matrix = self.analysis.spectral_correlations()
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        plt.imshow(correlation_matrix, cmap='viridis', origin='lower')
        plt.colorbar(label='Correlation')
        plt.xlabel('Energy Level Index')
        plt.ylabel('Energy Level Index')
        plt.title('Spectral Correlations')
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def create_summary_plots(self, save_dir: Optional[str] = None) -> None:
        """
        Create a set of summary plots for the energy spectrum.
        
        Args:
            save_dir (Optional[str]): The directory to save the plots (default: None).
        """
        # Create the individual plots
        if save_dir:
            self.plot_energy_spectrum(f"{save_dir}/energy_spectrum.png")
            self.plot_level_spacing_distribution(f"{save_dir}/level_spacing.png")
            self.plot_number_variance(f"{save_dir}/number_variance.png")
            self.plot_form_factor(f"{save_dir}/form_factor.png")
            self.plot_energy_level_dynamics(f"{save_dir}/energy_dynamics.png")
            self.plot_spectral_correlations(f"{save_dir}/spectral_correlations.png")
        else:
            self.plot_energy_spectrum()
            self.plot_level_spacing_distribution()
            self.plot_number_variance()
            self.plot_form_factor()
            self.plot_energy_level_dynamics()
            self.plot_spectral_correlations()
    
    def __str__(self) -> str:
        """
        Return a string representation of the energy spectrum visualization tool.
        
        Returns:
            str: A string representation of the visualization tool.
        """
        return "Energy Spectrum Visualization Tool for the 420-root Möbius Structure"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the energy spectrum visualization tool.
        
        Returns:
            str: A string representation of the visualization tool.
        """
        return "EnergySpectrumVisualization()"