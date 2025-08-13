"""
Energy Quantization implementation.

This module provides a comprehensive implementation of energy quantization functions
described in Chapter 16 and 17 of the textbook, which reveal deep connections to
spectral theory and quantum mechanics.
"""

import numpy as np
import sympy
from typing import List, Dict, Tuple, Union, Optional, Callable, Any

from ..moebius.moebius_transformation import MoebiusTransformation
from ..moebius.prime_indexed_moebius import PrimeIndexedMoebiusTransformation
from ..moebius.root_420_structure import Root420Structure


class EnergyQuantization:
    """
    A class for computing the energy of Möbius transformations.
    
    The energy quantization in the 420-root Möbius system follows the pattern:
    E(M_p) = (p^2/420) * (1 + sum_{k=1}^{∞} ((-1)^k/k!) * ((2πp/420)^{2k}))
    
    For small values of p/420, this can be approximated as:
    E(M_p) ≈ p^2/420
    
    Attributes:
        structural_parameter (int): The structural parameter (default: 420).
    """
    
    def __init__(self, structural_parameter: int = 420):
        """
        Initialize an energy quantization calculator.
        
        Args:
            structural_parameter (int): The structural parameter (default: 420).
        """
        self.structural_parameter = structural_parameter
    
    def energy(self, transformation: MoebiusTransformation) -> float:
        """
        Compute the energy of a Möbius transformation.
        
        Args:
            transformation (MoebiusTransformation): The transformation.
                
        Returns:
            float: The energy value.
        """
        if isinstance(transformation, PrimeIndexedMoebiusTransformation):
            # For a prime indexed Möbius transformation, use the specific formula
            return self._energy_prime_indexed(transformation)
        else:
            # For a general Möbius transformation, use the trace formula
            return self._energy_general(transformation)
    
    def _energy_prime_indexed(self, transformation: PrimeIndexedMoebiusTransformation) -> float:
        """
        Compute the energy of a prime indexed Möbius transformation.
        
        Args:
            transformation (PrimeIndexedMoebiusTransformation): The transformation.
                
        Returns:
            float: The energy value.
        """
        p = transformation.prime_index
        n = self.structural_parameter
        
        # For small values of p/n, use the approximation
        if p < n / 10:
            return p**2 / n
        
        # For larger values, use the full formula
        # We'll truncate the infinite sum after a few terms
        energy = p**2 / n
        
        # Add correction terms
        x = 2 * np.pi * p / n
        correction = 0
        factorial = 1
        sign = -1
        
        # Add terms until they become negligibly small
        for k in range(1, 10):
            factorial *= k
            sign *= -1
            term = sign * (x**(2*k)) / factorial
            correction += term
            
            # Stop if the term is very small
            if abs(term) < 1e-10:
                break
        
        energy *= (1 + correction)
        return energy
    
    def _energy_general(self, transformation: MoebiusTransformation) -> float:
        """
        Compute the energy of a general Möbius transformation.
        
        Args:
            transformation (MoebiusTransformation): The transformation.
                
        Returns:
            float: The energy value.
        """
        # For a general Möbius transformation, the energy is defined as:
        # E(M) = (1/2) * tr(A^T A) - 1
        # where A is the matrix representation of M
        a, b, c, d = transformation.a, transformation.b, transformation.c, transformation.d
        matrix = np.array([[a, b], [c, d]], dtype=complex)
        energy = 0.5 * np.trace(np.matmul(matrix.conj().T, matrix)).real - 1
        return energy
    
    def energy_spectrum(self, max_prime: int) -> Dict[int, float]:
        """
        Compute the energy spectrum for primes up to a maximum value.
        
        Args:
            max_prime (int): The maximum prime to include.
                
        Returns:
            Dict[int, float]: A dictionary mapping prime indices to energy values.
        """
        spectrum = {}
        
        for p in sympy.primerange(2, max_prime + 1):
            transformation = PrimeIndexedMoebiusTransformation(p, self.structural_parameter)
            spectrum[p] = self.energy(transformation)
        
        return spectrum
    
    def nearest_neighbor_spacing(self, max_prime: int) -> List[float]:
        """
        Compute the nearest neighbor spacing distribution of the energy spectrum.
        
        Args:
            max_prime (int): The maximum prime to include.
                
        Returns:
            List[float]: The nearest neighbor spacings.
        """
        # Compute the energy spectrum
        spectrum = self.energy_spectrum(max_prime)
        
        # Extract the energy values and sort them
        energies = sorted(spectrum.values())
        
        # Compute the spacings
        spacings = [energies[i+1] - energies[i] for i in range(len(energies)-1)]
        
        # Normalize the spacings
        mean_spacing = np.mean(spacings)
        normalized_spacings = [s / mean_spacing for s in spacings]
        
        return normalized_spacings
    
    def wigner_surmise(self, s: float) -> float:
        """
        Compute the Wigner surmise for a given spacing.
        
        The Wigner surmise is a theoretical prediction for the nearest neighbor
        spacing distribution of eigenvalues in random matrix theory:
        P(s) = (π*s/2) * exp(-π*s^2/4)
        
        Args:
            s (float): The spacing.
                
        Returns:
            float: The value of the Wigner surmise.
        """
        return (np.pi * s / 2) * np.exp(-np.pi * s**2 / 4)
    
    def spectral_rigidity(self, max_prime: int, L: float) -> float:
        """
        Compute the spectral rigidity of the energy spectrum.
        
        The spectral rigidity Δ₃(L) measures the deviation of the energy levels
        from equally spaced levels over an interval of length L.
        
        Args:
            max_prime (int): The maximum prime to include.
            L (float): The interval length.
                
        Returns:
            float: The spectral rigidity.
        """
        # Compute the energy spectrum
        spectrum = self.energy_spectrum(max_prime)
        
        # Extract the energy values and sort them
        energies = sorted(spectrum.values())
        
        # Normalize the energies to have mean spacing 1
        mean_spacing = (energies[-1] - energies[0]) / (len(energies) - 1)
        normalized_energies = [e / mean_spacing for e in energies]
        
        # Compute the spectral rigidity
        rigidity = 0
        
        for i in range(len(normalized_energies) - int(L)):
            # Extract a segment of length L
            segment = normalized_energies[i:i+int(L)]
            
            # Fit a straight line to the segment
            x = np.arange(len(segment))
            y = np.array(segment)
            
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # Compute the mean square deviation from the line
            deviation = np.sum((y - (m * x + c))**2) / len(segment)
            
            rigidity += deviation
        
        # Average over all segments
        rigidity /= (len(normalized_energies) - int(L))
        
        return rigidity
    
    def level_repulsion(self, max_prime: int) -> bool:
        """
        Check if the energy spectrum exhibits level repulsion.
        
        Level repulsion is a characteristic of quantum chaotic systems, where
        energy levels tend to avoid each other.
        
        Args:
            max_prime (int): The maximum prime to include.
                
        Returns:
            bool: True if the spectrum exhibits level repulsion, False otherwise.
        """
        # Compute the nearest neighbor spacing distribution
        spacings = self.nearest_neighbor_spacing(max_prime)
        
        # Count the number of small spacings
        small_spacings = sum(1 for s in spacings if s < 0.1)
        
        # If there are few small spacings, the spectrum exhibits level repulsion
        return small_spacings < 0.05 * len(spacings)
    
    def __str__(self) -> str:
        """
        Return a string representation of the energy quantization calculator.
        
        Returns:
            str: A string representation of the calculator.
        """
        return f"Energy Quantization Calculator (n={self.structural_parameter})"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the energy quantization calculator.
        
        Returns:
            str: A string representation of the calculator.
        """
        return f"EnergyQuantization(structural_parameter={self.structural_parameter})"


class EnergySpectrum:
    """
    A class for analyzing the energy spectrum of the 420-root Möbius structure.
    
    This class provides methods for computing and analyzing the energy spectrum,
    including statistical properties and connections to random matrix theory.
    
    Attributes:
        root_structure (Root420Structure): The 420-root Möbius structure.
        energy_calculator (EnergyQuantization): The energy quantization calculator.
    """
    
    def __init__(self):
        """
        Initialize an energy spectrum analyzer.
        """
        self.root_structure = Root420Structure()
        self.energy_calculator = EnergyQuantization()
    
    def compute_spectrum(self) -> Dict[int, float]:
        """
        Compute the energy spectrum of the 420-root Möbius structure.
        
        Returns:
            Dict[int, float]: A dictionary mapping prime indices to energy values.
        """
        spectrum = {}
        
        for p in self.root_structure.primes:
            transformation = self.root_structure.get_transformation(p)
            spectrum[p] = self.energy_calculator.energy(transformation)
        
        return spectrum
    
    def total_energy(self) -> float:
        """
        Compute the total energy of the 420-root Möbius structure.
        
        Returns:
            float: The total energy.
        """
        spectrum = self.compute_spectrum()
        return sum(spectrum.values())
    
    def mean_energy(self) -> float:
        """
        Compute the mean energy of the 420-root Möbius structure.
        
        Returns:
            float: The mean energy.
        """
        spectrum = self.compute_spectrum()
        return np.mean(list(spectrum.values()))
    
    def energy_variance(self) -> float:
        """
        Compute the variance of the energy spectrum.
        
        Returns:
            float: The energy variance.
        """
        spectrum = self.compute_spectrum()
        return np.var(list(spectrum.values()))
    
    def energy_histogram(self, num_bins: int = 20) -> Tuple[List[float], List[float]]:
        """
        Compute a histogram of the energy spectrum.
        
        Args:
            num_bins (int): The number of bins (default: 20).
                
        Returns:
            Tuple[List[float], List[float]]: The bin edges and counts.
        """
        spectrum = self.compute_spectrum()
        energies = list(spectrum.values())
        
        # Compute the histogram
        counts, bin_edges = np.histogram(energies, bins=num_bins)
        
        return bin_edges.tolist(), counts.tolist()
    
    def nearest_neighbor_spacing_distribution(self) -> Tuple[List[float], List[float]]:
        """
        Compute the nearest neighbor spacing distribution of the energy spectrum.
        
        Returns:
            Tuple[List[float], List[float]]: The spacings and their frequencies.
        """
        # Compute the nearest neighbor spacings
        spacings = self.energy_calculator.nearest_neighbor_spacing(max(self.root_structure.primes))
        
        # Compute a histogram of the spacings
        counts, bin_edges = np.histogram(spacings, bins=20, density=True)
        
        # Compute the bin centers
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
        
        return bin_centers, counts.tolist()
    
    def wigner_surmise_comparison(self) -> Tuple[List[float], List[float], List[float]]:
        """
        Compare the nearest neighbor spacing distribution with the Wigner surmise.
        
        Returns:
            Tuple[List[float], List[float], List[float]]: The spacings, their frequencies,
                and the Wigner surmise values.
        """
        # Compute the nearest neighbor spacing distribution
        spacings, frequencies = self.nearest_neighbor_spacing_distribution()
        
        # Compute the Wigner surmise for each spacing
        wigner_values = [self.energy_calculator.wigner_surmise(s) for s in spacings]
        
        return spacings, frequencies, wigner_values
    
    def spectral_rigidity_curve(self, max_L: float = 20, num_points: int = 20) -> Tuple[List[float], List[float]]:
        """
        Compute the spectral rigidity curve.
        
        Args:
            max_L (float): The maximum interval length (default: 20).
            num_points (int): The number of points in the curve (default: 20).
                
        Returns:
            Tuple[List[float], List[float]]: The interval lengths and rigidity values.
        """
        # Compute the spectral rigidity for different interval lengths
        L_values = np.linspace(1, max_L, num_points)
        rigidity_values = [self.energy_calculator.spectral_rigidity(max(self.root_structure.primes), L)
                          for L in L_values]
        
        return L_values.tolist(), rigidity_values
    
    def random_matrix_theory_comparison(self) -> Dict[str, Any]:
        """
        Compare the energy spectrum with predictions from random matrix theory.
        
        Returns:
            Dict[str, Any]: A dictionary containing various comparison results.
        """
        results = {}
        
        # Check for level repulsion
        results['level_repulsion'] = self.energy_calculator.level_repulsion(max(self.root_structure.primes))
        
        # Compare the nearest neighbor spacing distribution with the Wigner surmise
        spacings, frequencies, wigner_values = self.wigner_surmise_comparison()
        results['spacings'] = spacings
        results['frequencies'] = frequencies
        results['wigner_values'] = wigner_values
        
        # Compute the mean ratio of consecutive spacings
        spectrum = self.compute_spectrum()
        energies = sorted(spectrum.values())
        spacings = [energies[i+1] - energies[i] for i in range(len(energies)-1)]
        ratios = [min(spacings[i], spacings[i+1]) / max(spacings[i], spacings[i+1])
                 for i in range(len(spacings)-1)]
        results['mean_ratio'] = np.mean(ratios)
        
        # The GOE prediction for the mean ratio is approximately 0.536
        results['goe_prediction'] = 0.536
        
        return results
    
    def __str__(self) -> str:
        """
        Return a string representation of the energy spectrum analyzer.
        
        Returns:
            str: A string representation of the analyzer.
        """
        return "Energy Spectrum Analyzer for the 420-root Möbius Structure"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the energy spectrum analyzer.
        
        Returns:
            str: A string representation of the analyzer.
        """
        return "EnergySpectrum()"