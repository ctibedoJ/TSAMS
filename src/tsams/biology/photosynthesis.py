"""
Quantum Effects in Photosynthesis

This module implements tools for studying quantum effects in photosynthesis
using the phase synchronization mechanism from the Tibedo Framework.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import scipy.linalg as la
from scipy.integrate import solve_ivp

# Import Tibedo components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tibedo.core.spinor.reduction_chain import ReductionChain
from tibedo.core.prime_indexed.prime_indexed_structure import PrimeIndexedStructure
from tibedo.core.advanced.quantum_state import ConfigurableQuantumState


class ExcitonState:
    """
    Representation of an exciton state in photosynthetic systems.
    
    This class provides methods for representing and manipulating exciton states
    in photosynthetic systems using the Tibedo Framework.
    """
    
    def __init__(self, n_sites=7, initial_state=None):
        """
        Initialize the ExcitonState.
        
        Args:
            n_sites (int): Number of chromophore sites
            initial_state (np.ndarray, optional): Initial state vector
        """
        self.n_sites = n_sites
        
        # Initialize state vector
        if initial_state is None:
            # Start with excitation at site 0
            self.state = np.zeros(n_sites, dtype=complex)
            self.state[0] = 1.0
        else:
            if len(initial_state) != n_sites:
                raise ValueError(f"Initial state must have length {n_sites}")
            
            self.state = np.array(initial_state, dtype=complex)
            
            # Normalize state
            norm = np.sqrt(np.sum(np.abs(self.state)**2))
            if norm > 0:
                self.state /= norm
        
        # Create quantum state
        self.quantum_state = ConfigurableQuantumState(dimension=n_sites)
        
        # Configure quantum state
        parameters = {
            'phase_factors': np.ones(n_sites),
            'amplitude_factors': np.ones(n_sites) / np.sqrt(n_sites),
            'entanglement_pattern': 'photosynthesis',
            'cyclotomic_parameters': {'n': n_sites, 'k': 1},
            'symmetry_breaking': 0.0,
            'entropic_decline': 0.0
        }
        
        self.quantum_state.configure(parameters)
    
    def set_state(self, state):
        """
        Set the exciton state.
        
        Args:
            state (np.ndarray): State vector
            
        Returns:
            np.ndarray: Normalized state vector
        """
        if len(state) != self.n_sites:
            raise ValueError(f"State must have length {self.n_sites}")
        
        self.state = np.array(state, dtype=complex)
        
        # Normalize state
        norm = np.sqrt(np.sum(np.abs(self.state)**2))
        if norm > 0:
            self.state /= norm
        
        return self.state
    
    def get_site_populations(self):
        """
        Get the population of each site.
        
        Returns:
            np.ndarray: Site populations
        """
        return np.abs(self.state)**2
    
    def get_coherence(self, site1, site2):
        """
        Get the coherence between two sites.
        
        Args:
            site1 (int): First site index
            site2 (int): Second site index
            
        Returns:
            complex: Coherence
        """
        return self.state[site1] * np.conj(self.state[site2])
    
    def get_density_matrix(self):
        """
        Get the density matrix representation of the state.
        
        Returns:
            np.ndarray: Density matrix
        """
        return np.outer(self.state, np.conj(self.state))
    
    def apply_hamiltonian(self, hamiltonian, dt):
        """
        Apply a Hamiltonian to evolve the state.
        
        Args:
            hamiltonian (np.ndarray): Hamiltonian matrix
            dt (float): Time step
            
        Returns:
            np.ndarray: Evolved state
        """
        # Calculate evolution operator
        U = la.expm(-1j * hamiltonian * dt)
        
        # Apply evolution operator
        self.state = np.dot(U, self.state)
        
        # Normalize state
        norm = np.sqrt(np.sum(np.abs(self.state)**2))
        if norm > 0:
            self.state /= norm
        
        return self.state
    
    def apply_phase_synchronization(self, coupling_strength=0.1):
        """
        Apply phase synchronization mechanism.
        
        Args:
            coupling_strength (float): Strength of phase synchronization
            
        Returns:
            np.ndarray: Synchronized state
        """
        # Use quantum state to apply phase synchronization
        synchronized_state = self.quantum_state.apply_phase_synchronization(
            self.state, coupling_strength
        )
        
        # Update state
        self.state = synchronized_state
        
        return self.state


class PhotosynthesisSimulator:
    """
    Simulator for quantum effects in photosynthesis.
    
    This class implements tools for simulating quantum effects in photosynthesis
    using the phase synchronization mechanism from the Tibedo Framework.
    """
    
    def __init__(self, n_sites=7):
        """
        Initialize the PhotosynthesisSimulator.
        
        Args:
            n_sites (int): Number of chromophore sites
        """
        self.n_sites = n_sites
        
        # Create exciton state
        self.exciton_state = ExcitonState(n_sites)
        
        # Initialize Hamiltonian
        self.hamiltonian = np.zeros((n_sites, n_sites), dtype=complex)
        
        # Initialize site positions
        self.site_positions = np.zeros((n_sites, 3))
        
        # Initialize results
        self.results = {}
    
    def set_fmo_hamiltonian(self):
        """
        Set the Hamiltonian for the Fenna-Matthews-Olson (FMO) complex.
        
        Returns:
            np.ndarray: FMO Hamiltonian
        """
        # FMO Hamiltonian from literature (in cm^-1)
        fmo_hamiltonian = np.array([
            [12410, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
            [-87.7, 12530, 30.8, 8.2, 0.7, 11.8, 4.3],
            [5.5, 30.8, 12210, -53.5, -2.2, -9.6, 6.0],
            [-5.9, 8.2, -53.5, 12320, -70.7, -17.0, -63.3],
            [6.7, 0.7, -2.2, -70.7, 12480, 81.1, -1.3],
            [-13.7, 11.8, -9.6, -17.0, 81.1, 12630, 39.7],
            [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 12440]
        ])
        
        # Convert to atomic units
        fmo_hamiltonian = fmo_hamiltonian * 4.556e-6  # cm^-1 to Hartree
        
        # Set Hamiltonian
        self.hamiltonian = fmo_hamiltonian
        
        return self.hamiltonian
    
    def set_site_positions(self, positions):
        """
        Set the positions of chromophore sites.
        
        Args:
            positions (np.ndarray): Site positions (n_sites, 3)
            
        Returns:
            np.ndarray: Site positions
        """
        if positions.shape != (self.n_sites, 3):
            raise ValueError(f"Positions must have shape ({self.n_sites}, 3)")
        
        self.site_positions = positions
        
        return self.site_positions
    
    def set_fmo_site_positions(self):
        """
        Set the positions for the Fenna-Matthews-Olson (FMO) complex.
        
        Returns:
            np.ndarray: FMO site positions
        """
        # Approximate FMO site positions (in Angstroms)
        fmo_positions = np.array([
            [0.0, 0.0, 0.0],
            [3.8, -4.9, -8.7],
            [-7.5, 0.0, 0.0],
            [-3.8, 7.3, 0.0],
            [7.5, 0.0, 0.0],
            [11.3, -4.9, 8.7],
            [3.8, 7.3, 8.7]
        ])
        
        # Set site positions
        self.site_positions = fmo_positions
        
        return self.site_positions
    
    def simulate_exciton_dynamics(self, initial_site=0, total_time=1.0, n_steps=1000, 
                                  use_phase_sync=True, coupling_strength=0.1):
        """
        Simulate exciton dynamics.
        
        Args:
            initial_site (int): Initial site for excitation
            total_time (float): Total simulation time (in ps)
            n_steps (int): Number of time steps
            use_phase_sync (bool): Whether to use phase synchronization
            coupling_strength (float): Strength of phase synchronization
            
        Returns:
            dict: Simulation results
        """
        # Set initial state
        initial_state = np.zeros(self.n_sites, dtype=complex)
        initial_state[initial_site] = 1.0
        
        self.exciton_state.set_state(initial_state)
        
        # Create time array
        times = np.linspace(0, total_time, n_steps)
        dt = times[1] - times[0]
        
        # Initialize results
        populations = np.zeros((n_steps, self.n_sites))
        coherences = np.zeros((n_steps, self.n_sites, self.n_sites), dtype=complex)
        
        # Record initial state
        populations[0] = self.exciton_state.get_site_populations()
        coherences[0] = self.exciton_state.get_density_matrix()
        
        # Simulate dynamics
        for i in range(1, n_steps):
            # Apply Hamiltonian
            self.exciton_state.apply_hamiltonian(self.hamiltonian, dt)
            
            # Apply phase synchronization if enabled
            if use_phase_sync:
                self.exciton_state.apply_phase_synchronization(coupling_strength)
            
            # Record state
            populations[i] = self.exciton_state.get_site_populations()
            coherences[i] = self.exciton_state.get_density_matrix()
        
        # Store results
        self.results = {
            'times': times,
            'populations': populations,
            'coherences': coherences,
            'initial_site': initial_site,
            'use_phase_sync': use_phase_sync,
            'coupling_strength': coupling_strength
        }
        
        return self.results
    
    def calculate_energy_transfer_efficiency(self, target_site=3):
        """
        Calculate energy transfer efficiency to a target site.
        
        Args:
            target_site (int): Target site index
            
        Returns:
            float: Energy transfer efficiency
        """
        if not self.results:
            raise ValueError("No simulation results available")
        
        # Get final population at target site
        final_population = self.results['populations'][-1, target_site]
        
        return final_population
    
    def calculate_coherence_lifetime(self, site1=0, site2=1, threshold=0.01):
        """
        Calculate coherence lifetime between two sites.
        
        Args:
            site1 (int): First site index
            site2 (int): Second site index
            threshold (float): Threshold for coherence
            
        Returns:
            float: Coherence lifetime
        """
        if not self.results:
            raise ValueError("No simulation results available")
        
        # Get coherence magnitude over time
        coherence_mag = np.abs(self.results['coherences'][:, site1, site2])
        
        # Find time when coherence drops below threshold
        times = self.results['times']
        
        for i in range(len(times)):
            if coherence_mag[i] < threshold:
                return times[i]
        
        # If coherence never drops below threshold
        return times[-1]
    
    def visualize_populations(self):
        """
        Visualize site populations over time.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if not self.results:
            raise ValueError("No simulation results available")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot populations
        times = self.results['times']
        populations = self.results['populations']
        
        for i in range(self.n_sites):
            ax.plot(times, populations[:, i], label=f'Site {i+1}')
        
        # Set labels
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Population')
        ax.set_title('Exciton Populations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_coherences(self, site1=0, site2=1):
        """
        Visualize coherence between two sites over time.
        
        Args:
            site1 (int): First site index
            site2 (int): Second site index
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if not self.results:
            raise ValueError("No simulation results available")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot coherence magnitude
        times = self.results['times']
        coherences = self.results['coherences']
        
        coherence_mag = np.abs(coherences[:, site1, site2])
        
        ax.plot(times, coherence_mag, 'k-', linewidth=2)
        
        # Set labels
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Coherence Magnitude')
        ax.set_title(f'Coherence between Sites {site1+1} and {site2+1}')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_complex(self):
        """
        Visualize the chromophore complex.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if np.all(self.site_positions == 0):
            raise ValueError("Site positions not set")
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot sites
        for i in range(self.n_sites):
            ax.scatter(
                self.site_positions[i, 0],
                self.site_positions[i, 1],
                self.site_positions[i, 2],
                s=100,
                label=f'Site {i+1}'
            )
        
        # Plot connections
        for i in range(self.n_sites):
            for j in range(i+1, self.n_sites):
                # Check if sites are coupled
                if abs(self.hamiltonian[i, j]) > 1e-6:
                    # Plot connection
                    ax.plot(
                        [self.site_positions[i, 0], self.site_positions[j, 0]],
                        [self.site_positions[i, 1], self.site_positions[j, 1]],
                        [self.site_positions[i, 2], self.site_positions[j, 2]],
                        'k-', alpha=0.5
                    )
        
        # Set labels
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('Chromophore Complex')
        ax.legend()
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        return fig
    
    def visualize_energy_landscape(self):
        """
        Visualize the energy landscape of the complex.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if np.all(self.hamiltonian == 0):
            raise ValueError("Hamiltonian not set")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get site energies (diagonal elements of Hamiltonian)
        site_energies = np.real(np.diag(self.hamiltonian))
        
        # Convert to cm^-1 for better readability
        site_energies = site_energies / 4.556e-6
        
        # Plot site energies
        ax.bar(range(1, self.n_sites + 1), site_energies)
        
        # Set labels
        ax.set_xlabel('Site')
        ax.set_ylabel('Energy (cm$^{-1}$)')
        ax.set_title('Site Energies')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_comparison(self, with_sync_results, without_sync_results):
        """
        Visualize comparison between simulations with and without phase synchronization.
        
        Args:
            with_sync_results (dict): Results with phase synchronization
            without_sync_results (dict): Results without phase synchronization
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot populations with phase synchronization
        times = with_sync_results['times']
        populations = with_sync_results['populations']
        
        for i in range(self.n_sites):
            ax1.plot(times, populations[:, i], label=f'Site {i+1}')
        
        # Set labels
        ax1.set_xlabel('Time (ps)')
        ax1.set_ylabel('Population')
        ax1.set_title('With Phase Synchronization')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot populations without phase synchronization
        times = without_sync_results['times']
        populations = without_sync_results['populations']
        
        for i in range(self.n_sites):
            ax2.plot(times, populations[:, i], label=f'Site {i+1}')
        
        # Set labels
        ax2.set_xlabel('Time (ps)')
        ax2.set_ylabel('Population')
        ax2.set_title('Without Phase Synchronization')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig


class EnvironmentInteraction:
    """
    Representation of environment interactions in photosynthetic systems.
    
    This class provides methods for modeling environment interactions in
    photosynthetic systems using the Tibedo Framework.
    """
    
    def __init__(self, n_sites=7):
        """
        Initialize the EnvironmentInteraction.
        
        Args:
            n_sites (int): Number of chromophore sites
        """
        self.n_sites = n_sites
        
        # Initialize reorganization energies
        self.reorganization_energies = np.ones(n_sites) * 0.01  # in Hartree
        
        # Initialize bath correlation times
        self.correlation_times = np.ones(n_sites) * 0.1  # in ps
        
        # Initialize temperature
        self.temperature = 300.0  # in K
        
        # Create spinor reduction chain
        self.reduction_chain = ReductionChain(
            initial_dimension=16,
            chain_length=5
        )
        
        # Create prime-indexed structure
        self.prime_structure = PrimeIndexedStructure(max_index=100)
    
    def set_reorganization_energies(self, energies):
        """
        Set the reorganization energies.
        
        Args:
            energies (np.ndarray): Reorganization energies
            
        Returns:
            np.ndarray: Reorganization energies
        """
        if len(energies) != self.n_sites:
            raise ValueError(f"Energies must have length {self.n_sites}")
        
        self.reorganization_energies = np.array(energies)
        
        return self.reorganization_energies
    
    def set_correlation_times(self, times):
        """
        Set the bath correlation times.
        
        Args:
            times (np.ndarray): Correlation times
            
        Returns:
            np.ndarray: Correlation times
        """
        if len(times) != self.n_sites:
            raise ValueError(f"Times must have length {self.n_sites}")
        
        self.correlation_times = np.array(times)
        
        return self.correlation_times
    
    def set_temperature(self, temperature):
        """
        Set the temperature.
        
        Args:
            temperature (float): Temperature in K
            
        Returns:
            float: Temperature
        """
        self.temperature = temperature
        
        return self.temperature
    
    def calculate_spectral_density(self, omega, site_idx):
        """
        Calculate the spectral density for a site.
        
        Args:
            omega (float): Frequency
            site_idx (int): Site index
            
        Returns:
            float: Spectral density
        """
        # Drude-Lorentz spectral density
        lambda_j = self.reorganization_energies[site_idx]
        gamma_j = 1.0 / self.correlation_times[site_idx]
        
        J = 2 * lambda_j * omega * gamma_j / (omega**2 + gamma_j**2)
        
        return J
    
    def calculate_bath_correlation_function(self, t, site_idx):
        """
        Calculate the bath correlation function for a site.
        
        Args:
            t (float): Time
            site_idx (int): Site index
            
        Returns:
            complex: Bath correlation function
        """
        # Parameters
        lambda_j = self.reorganization_energies[site_idx]
        gamma_j = 1.0 / self.correlation_times[site_idx]
        
        # Boltzmann constant in atomic units
        k_B = 3.166811e-6  # Hartree/K
        
        # Temperature
        T = self.temperature
        
        # Calculate correlation function
        if T > 0:
            # Finite temperature
            beta = 1.0 / (k_B * T)
            
            # Real part
            C_real = lambda_j * gamma_j * (1.0 / np.tan(beta * gamma_j / 2.0) * np.cos(gamma_j * t) - 1j * np.sin(gamma_j * t))
            
            # Imaginary part
            C_imag = -1j * lambda_j * gamma_j * np.exp(-gamma_j * t)
            
            return C_real + C_imag
        else:
            # Zero temperature
            return -1j * lambda_j * gamma_j * np.exp(-gamma_j * t)
    
    def calculate_lineshape_function(self, t, site_idx):
        """
        Calculate the lineshape function for a site.
        
        Args:
            t (float): Time
            site_idx (int): Site index
            
        Returns:
            complex: Lineshape function
        """
        # Parameters
        lambda_j = self.reorganization_energies[site_idx]
        gamma_j = 1.0 / self.correlation_times[site_idx]
        
        # Boltzmann constant in atomic units
        k_B = 3.166811e-6  # Hartree/K
        
        # Temperature
        T = self.temperature
        
        # Calculate lineshape function
        if T > 0:
            # Finite temperature
            beta = 1.0 / (k_B * T)
            
            # Real part
            g_real = lambda_j * (1.0 - np.cos(gamma_j * t)) / (np.tanh(beta * gamma_j / 2.0) * gamma_j**2)
            
            # Imaginary part
            g_imag = lambda_j * np.sin(gamma_j * t) / gamma_j**2
            
            return g_real + 1j * g_imag
        else:
            # Zero temperature
            return lambda_j * (1.0 - np.cos(gamma_j * t) + 1j * np.sin(gamma_j * t)) / gamma_j**2
    
    def apply_environment_effects(self, density_matrix, dt):
        """
        Apply environment effects to a density matrix.
        
        Args:
            density_matrix (np.ndarray): Density matrix
            dt (float): Time step
            
        Returns:
            np.ndarray: Updated density matrix
        """
        # Create Lindblad operators
        lindblad_ops = []
        
        for i in range(self.n_sites):
            # Create operator for site i
            op = np.zeros((self.n_sites, self.n_sites), dtype=complex)
            op[i, i] = 1.0
            
            # Add to list
            lindblad_ops.append(op)
        
        # Apply Lindblad equation
        rho = density_matrix.copy()
        
        for i in range(self.n_sites):
            # Get operator
            L = lindblad_ops[i]
            
            # Get rate
            gamma = 1.0 / self.correlation_times[i]
            
            # Calculate terms
            term1 = np.dot(L, np.dot(rho, L.conj().T))
            term2 = 0.5 * np.dot(np.dot(L.conj().T, L), rho)
            term3 = 0.5 * np.dot(rho, np.dot(L.conj().T, L))
            
            # Update density matrix
            rho += dt * gamma * (term1 - term2 - term3)
        
        return rho
    
    def apply_phase_synchronization(self, density_matrix, coupling_strength=0.1):
        """
        Apply phase synchronization to a density matrix.
        
        Args:
            density_matrix (np.ndarray): Density matrix
            coupling_strength (float): Strength of phase synchronization
            
        Returns:
            np.ndarray: Synchronized density matrix
        """
        # Extract state vector (assume pure state)
        eigenvalues, eigenvectors = la.eigh(density_matrix)
        state = eigenvectors[:, -1]
        
        # Create exciton state
        exciton = ExcitonState(self.n_sites)
        exciton.set_state(state)
        
        # Apply phase synchronization
        synchronized_state = exciton.apply_phase_synchronization(coupling_strength)
        
        # Create new density matrix
        synchronized_dm = np.outer(synchronized_state, np.conj(synchronized_state))
        
        return synchronized_dm


class PhotosynthesisAnalyzer:
    """
    Analyzer for quantum effects in photosynthesis.
    
    This class provides tools for analyzing quantum effects in photosynthesis
    using the phase synchronization mechanism from the Tibedo Framework.
    """
    
    def __init__(self):
        """
        Initialize the PhotosynthesisAnalyzer.
        """
        # Create simulator
        self.simulator = PhotosynthesisSimulator()
        
        # Create environment interaction
        self.environment = EnvironmentInteraction()
        
        # Initialize results
        self.results = {}
    
    def setup_fmo_complex(self):
        """
        Set up the Fenna-Matthews-Olson (FMO) complex.
        
        Returns:
            tuple: (hamiltonian, site_positions)
        """
        # Set FMO Hamiltonian
        hamiltonian = self.simulator.set_fmo_hamiltonian()
        
        # Set FMO site positions
        site_positions = self.simulator.set_fmo_site_positions()
        
        return hamiltonian, site_positions
    
    def analyze_energy_transfer(self, initial_site=0, target_site=3, total_time=1.0, n_steps=1000):
        """
        Analyze energy transfer with and without phase synchronization.
        
        Args:
            initial_site (int): Initial site for excitation
            target_site (int): Target site for energy transfer
            total_time (float): Total simulation time (in ps)
            n_steps (int): Number of time steps
            
        Returns:
            dict: Analysis results
        """
        # Simulate with phase synchronization
        with_sync_results = self.simulator.simulate_exciton_dynamics(
            initial_site=initial_site,
            total_time=total_time,
            n_steps=n_steps,
            use_phase_sync=True,
            coupling_strength=0.1
        )
        
        # Calculate efficiency with phase synchronization
        with_sync_efficiency = self.simulator.calculate_energy_transfer_efficiency(target_site)
        
        # Simulate without phase synchronization
        without_sync_results = self.simulator.simulate_exciton_dynamics(
            initial_site=initial_site,
            total_time=total_time,
            n_steps=n_steps,
            use_phase_sync=False
        )
        
        # Calculate efficiency without phase synchronization
        without_sync_efficiency = self.simulator.calculate_energy_transfer_efficiency(target_site)
        
        # Store results
        self.results = {
            'with_sync_results': with_sync_results,
            'without_sync_results': without_sync_results,
            'with_sync_efficiency': with_sync_efficiency,
            'without_sync_efficiency': without_sync_efficiency,
            'initial_site': initial_site,
            'target_site': target_site
        }
        
        return self.results
    
    def analyze_coherence(self, site1=0, site2=1, total_time=1.0, n_steps=1000):
        """
        Analyze coherence with and without phase synchronization.
        
        Args:
            site1 (int): First site index
            site2 (int): Second site index
            total_time (float): Total simulation time (in ps)
            n_steps (int): Number of time steps
            
        Returns:
            dict: Analysis results
        """
        # Simulate with phase synchronization
        with_sync_results = self.simulator.simulate_exciton_dynamics(
            initial_site=site1,
            total_time=total_time,
            n_steps=n_steps,
            use_phase_sync=True,
            coupling_strength=0.1
        )
        
        # Calculate coherence lifetime with phase synchronization
        with_sync_lifetime = self.simulator.calculate_coherence_lifetime(site1, site2)
        
        # Simulate without phase synchronization
        without_sync_results = self.simulator.simulate_exciton_dynamics(
            initial_site=site1,
            total_time=total_time,
            n_steps=n_steps,
            use_phase_sync=False
        )
        
        # Calculate coherence lifetime without phase synchronization
        without_sync_lifetime = self.simulator.calculate_coherence_lifetime(site1, site2)
        
        # Store results
        self.results = {
            'with_sync_results': with_sync_results,
            'without_sync_results': without_sync_results,
            'with_sync_lifetime': with_sync_lifetime,
            'without_sync_lifetime': without_sync_lifetime,
            'site1': site1,
            'site2': site2
        }
        
        return self.results
    
    def analyze_environment_effects(self, temperature=300.0, reorganization_energy=0.01, correlation_time=0.1):
        """
        Analyze environment effects on energy transfer.
        
        Args:
            temperature (float): Temperature in K
            reorganization_energy (float): Reorganization energy in Hartree
            correlation_time (float): Bath correlation time in ps
            
        Returns:
            dict: Analysis results
        """
        # Set environment parameters
        self.environment.set_temperature(temperature)
        self.environment.set_reorganization_energies(np.ones(self.simulator.n_sites) * reorganization_energy)
        self.environment.set_correlation_times(np.ones(self.simulator.n_sites) * correlation_time)
        
        # Simulate with phase synchronization
        with_sync_results = self.simulator.simulate_exciton_dynamics(
            initial_site=0,
            total_time=1.0,
            n_steps=1000,
            use_phase_sync=True,
            coupling_strength=0.1
        )
        
        # Apply environment effects
        with_env_populations = np.zeros_like(with_sync_results['populations'])
        with_env_coherences = np.zeros_like(with_sync_results['coherences'])
        
        # Copy initial state
        with_env_populations[0] = with_sync_results['populations'][0]
        with_env_coherences[0] = with_sync_results['coherences'][0]
        
        # Apply environment effects to each time step
        dt = with_sync_results['times'][1] - with_sync_results['times'][0]
        
        for i in range(1, len(with_sync_results['times'])):
            # Get density matrix from previous step
            rho = with_env_coherences[i-1]
            
            # Apply environment effects
            rho = self.environment.apply_environment_effects(rho, dt)
            
            # Apply phase synchronization
            rho = self.environment.apply_phase_synchronization(rho, 0.1)
            
            # Store results
            with_env_coherences[i] = rho
            with_env_populations[i] = np.real(np.diag(rho))
        
        # Store results
        self.results = {
            'with_sync_results': with_sync_results,
            'with_env_populations': with_env_populations,
            'with_env_coherences': with_env_coherences,
            'temperature': temperature,
            'reorganization_energy': reorganization_energy,
            'correlation_time': correlation_time
        }
        
        return self.results
    
    def visualize_energy_transfer_comparison(self):
        """
        Visualize comparison of energy transfer with and without phase synchronization.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if 'with_sync_efficiency' not in self.results:
            raise ValueError("Energy transfer analysis not performed")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot populations with phase synchronization
        times = self.results['with_sync_results']['times']
        populations = self.results['with_sync_results']['populations']
        target_site = self.results['target_site']
        
        for i in range(self.simulator.n_sites):
            if i == target_site:
                ax1.plot(times, populations[:, i], 'r-', linewidth=2, label=f'Site {i+1} (Target)')
            else:
                ax1.plot(times, populations[:, i], label=f'Site {i+1}')
        
        # Set labels
        ax1.set_xlabel('Time (ps)')
        ax1.set_ylabel('Population')
        ax1.set_title(f'With Phase Synchronization (Efficiency: {self.results["with_sync_efficiency"]:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot populations without phase synchronization
        times = self.results['without_sync_results']['times']
        populations = self.results['without_sync_results']['populations']
        
        for i in range(self.simulator.n_sites):
            if i == target_site:
                ax2.plot(times, populations[:, i], 'r-', linewidth=2, label=f'Site {i+1} (Target)')
            else:
                ax2.plot(times, populations[:, i], label=f'Site {i+1}')
        
        # Set labels
        ax2.set_xlabel('Time (ps)')
        ax2.set_ylabel('Population')
        ax2.set_title(f'Without Phase Synchronization (Efficiency: {self.results["without_sync_efficiency"]:.4f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_coherence_comparison(self):
        """
        Visualize comparison of coherence with and without phase synchronization.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if 'with_sync_lifetime' not in self.results:
            raise ValueError("Coherence analysis not performed")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot coherence magnitude with phase synchronization
        times = self.results['with_sync_results']['times']
        coherences = self.results['with_sync_results']['coherences']
        site1 = self.results['site1']
        site2 = self.results['site2']
        
        coherence_mag_with = np.abs(coherences[:, site1, site2])
        
        ax.plot(times, coherence_mag_with, 'r-', linewidth=2, 
                label=f'With Phase Sync (Lifetime: {self.results["with_sync_lifetime"]:.4f} ps)')
        
        # Plot coherence magnitude without phase synchronization
        times = self.results['without_sync_results']['times']
        coherences = self.results['without_sync_results']['coherences']
        
        coherence_mag_without = np.abs(coherences[:, site1, site2])
        
        ax.plot(times, coherence_mag_without, 'b-', linewidth=2, 
                label=f'Without Phase Sync (Lifetime: {self.results["without_sync_lifetime"]:.4f} ps)')
        
        # Set labels
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Coherence Magnitude')
        ax.set_title(f'Coherence between Sites {site1+1} and {site2+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_environment_effects(self):
        """
        Visualize environment effects on energy transfer.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if 'with_env_populations' not in self.results:
            raise ValueError("Environment effects analysis not performed")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot populations with phase synchronization (no environment)
        times = self.results['with_sync_results']['times']
        populations = self.results['with_sync_results']['populations']
        
        for i in range(self.simulator.n_sites):
            ax1.plot(times, populations[:, i], label=f'Site {i+1}')
        
        # Set labels
        ax1.set_xlabel('Time (ps)')
        ax1.set_ylabel('Population')
        ax1.set_title('With Phase Sync (No Environment)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot populations with phase synchronization and environment
        times = self.results['with_sync_results']['times']
        populations = self.results['with_env_populations']
        
        for i in range(self.simulator.n_sites):
            ax2.plot(times, populations[:, i], label=f'Site {i+1}')
        
        # Set labels
        ax2.set_xlabel('Time (ps)')
        ax2.set_ylabel('Population')
        ax2.set_title(f'With Phase Sync and Environment (T = {self.results["temperature"]} K)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def save_results(self, path):
        """
        Save analysis results.
        
        Args:
            path (str): Path to save the results
        """
        import pickle
        
        with open(path, 'wb') as f:
            pickle.dump(self.results, f)
    
    def load_results(self, path):
        """
        Load analysis results.
        
        Args:
            path (str): Path to load the results from
            
        Returns:
            dict: Analysis results
        """
        import pickle
        
        with open(path, 'rb') as f:
            self.results = pickle.load(f)
        
        return self.results