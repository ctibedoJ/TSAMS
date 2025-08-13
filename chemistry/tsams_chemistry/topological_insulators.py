"""
Quantum Effects in Topological Insulators

This module implements tools for studying quantum effects in topological insulators
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
from scipy.optimize import minimize

# Import Tibedo components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tibedo.core.spinor.reduction_chain import ReductionChain
from tibedo.core.prime_indexed.prime_indexed_structure import PrimeIndexedStructure
from tibedo.core.advanced.quantum_state import ConfigurableQuantumState


class TopologicalState:
    """
    Representation of a topological state in topological insulators.
    
    This class provides methods for representing and manipulating topological states
    in topological insulators using the Tibedo Framework.
    """
    
    def __init__(self, lattice_size=10):
        """
        Initialize the TopologicalState.
        
        Args:
            lattice_size (int): Size of the lattice
        """
        self.lattice_size = lattice_size
        
        # Initialize state
        self.state = self._create_initial_state()
        
        # Initialize parameters
        self.mass = 0.5  # Mass parameter
        self.spin_orbit_coupling = 1.0  # Spin-orbit coupling
        
        # Create quantum state
        self.quantum_state = ConfigurableQuantumState(dimension=7)
        
        # Configure quantum state
        parameters = {
            'phase_factors': np.ones(7),
            'amplitude_factors': np.ones(7) / np.sqrt(7),
            'entanglement_pattern': 'topological',
            'cyclotomic_parameters': {'n': 7, 'k': 1},
            'symmetry_breaking': 0.0,
            'entropic_decline': 0.0
        }
        
        self.quantum_state.configure(parameters)
    
    def _create_initial_state(self):
        """
        Create the initial state of the topological insulator.
        
        Returns:
            np.ndarray: Initial state vector
        """
        # Dimension of the Hilbert space
        # 2 (spin) × lattice_size^2 (position)
        dim = 2 * self.lattice_size**2
        
        # Create edge state (simplified)
        state = np.zeros(dim, dtype=complex)
        
        # Set amplitude on the edge
        for i in range(self.lattice_size):
            # Top edge
            idx1 = 2 * (0 * self.lattice_size + i)
            idx2 = 2 * (0 * self.lattice_size + i) + 1
            
            state[idx1] = 1.0 / np.sqrt(4 * self.lattice_size)
            state[idx2] = 1j / np.sqrt(4 * self.lattice_size)
            
            # Bottom edge
            idx1 = 2 * ((self.lattice_size - 1) * self.lattice_size + i)
            idx2 = 2 * ((self.lattice_size - 1) * self.lattice_size + i) + 1
            
            state[idx1] = 1.0 / np.sqrt(4 * self.lattice_size)
            state[idx2] = -1j / np.sqrt(4 * self.lattice_size)
        
        return state
    
    def set_mass(self, mass):
        """
        Set the mass parameter.
        
        Args:
            mass (float): Mass parameter
            
        Returns:
            float: Mass parameter
        """
        self.mass = mass
        
        return self.mass
    
    def set_spin_orbit_coupling(self, coupling):
        """
        Set the spin-orbit coupling.
        
        Args:
            coupling (float): Spin-orbit coupling
            
        Returns:
            float: Spin-orbit coupling
        """
        self.spin_orbit_coupling = coupling
        
        return self.spin_orbit_coupling
    
    def calculate_hamiltonian(self):
        """
        Calculate the Hamiltonian of the topological insulator.
        
        Returns:
            np.ndarray: Hamiltonian matrix
        """
        # Dimension of the Hilbert space
        dim = 2 * self.lattice_size**2
        
        # Initialize Hamiltonian
        H = np.zeros((dim, dim), dtype=complex)
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Build Hamiltonian
        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                # Current site index
                site_idx = i * self.lattice_size + j
                
                # Mass term
                H[2*site_idx:2*site_idx+2, 2*site_idx:2*site_idx+2] += self.mass * sigma_z
                
                # Spin-orbit coupling terms
                # Right neighbor
                if j < self.lattice_size - 1:
                    neighbor_idx = i * self.lattice_size + (j + 1)
                    H[2*site_idx:2*site_idx+2, 2*neighbor_idx:2*neighbor_idx+2] += 0.5j * self.spin_orbit_coupling * sigma_x
                    H[2*neighbor_idx:2*neighbor_idx+2, 2*site_idx:2*site_idx+2] += -0.5j * self.spin_orbit_coupling * sigma_x
                
                # Left neighbor
                if j > 0:
                    neighbor_idx = i * self.lattice_size + (j - 1)
                    H[2*site_idx:2*site_idx+2, 2*neighbor_idx:2*neighbor_idx+2] += -0.5j * self.spin_orbit_coupling * sigma_x
                    H[2*neighbor_idx:2*neighbor_idx+2, 2*site_idx:2*site_idx+2] += 0.5j * self.spin_orbit_coupling * sigma_x
                
                # Down neighbor
                if i < self.lattice_size - 1:
                    neighbor_idx = (i + 1) * self.lattice_size + j
                    H[2*site_idx:2*site_idx+2, 2*neighbor_idx:2*neighbor_idx+2] += 0.5j * self.spin_orbit_coupling * sigma_y
                    H[2*neighbor_idx:2*neighbor_idx+2, 2*site_idx:2*site_idx+2] += -0.5j * self.spin_orbit_coupling * sigma_y
                
                # Up neighbor
                if i > 0:
                    neighbor_idx = (i - 1) * self.lattice_size + j
                    H[2*site_idx:2*site_idx+2, 2*neighbor_idx:2*neighbor_idx+2] += -0.5j * self.spin_orbit_coupling * sigma_y
                    H[2*neighbor_idx:2*neighbor_idx+2, 2*site_idx:2*site_idx+2] += 0.5j * self.spin_orbit_coupling * sigma_y
        
        return H
    
    def evolve_state(self, dt):
        """
        Evolve the state of the topological insulator.
        
        Args:
            dt (float): Time step in fs
            
        Returns:
            np.ndarray: Evolved state
        """
        # Calculate Hamiltonian
        H = self.calculate_hamiltonian()
        
        # Calculate evolution operator
        U = la.expm(-1j * H * dt)
        
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
        # Reshape state to match quantum state dimension
        if len(self.state) > self.quantum_state.dimension:
            # Truncate
            state_vector = self.state[:self.quantum_state.dimension]
        else:
            # Pad with zeros
            padded = np.zeros(self.quantum_state.dimension, dtype=complex)
            padded[:len(self.state)] = self.state
            state_vector = padded
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(state_vector)**2))
        if norm > 0:
            state_vector /= norm
        
        # Apply phase synchronization
        synchronized_vector = self.quantum_state.apply_phase_synchronization(
            state_vector, coupling_strength
        )
        
        # Truncate or pad back to original size
        if len(self.state) > self.quantum_state.dimension:
            # Pad with zeros
            padded = np.zeros(len(self.state), dtype=complex)
            padded[:self.quantum_state.dimension] = synchronized_vector
            synchronized_vector = padded
        else:
            # Truncate
            synchronized_vector = synchronized_vector[:len(self.state)]
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(synchronized_vector)**2))
        if norm > 0:
            synchronized_vector /= norm
        
        # Update state
        self.state = synchronized_vector
        
        return self.state
    
    def calculate_edge_localization(self):
        """
        Calculate the edge localization of the state.
        
        Returns:
            float: Edge localization
        """
        # Calculate probability density on the edges
        edge_density = 0.0
        
        for i in range(self.lattice_size):
            # Top edge
            idx1 = 2 * (0 * self.lattice_size + i)
            idx2 = 2 * (0 * self.lattice_size + i) + 1
            
            edge_density += np.abs(self.state[idx1])**2 + np.abs(self.state[idx2])**2
            
            # Bottom edge
            idx1 = 2 * ((self.lattice_size - 1) * self.lattice_size + i)
            idx2 = 2 * ((self.lattice_size - 1) * self.lattice_size + i) + 1
            
            edge_density += np.abs(self.state[idx1])**2 + np.abs(self.state[idx2])**2
            
            # Left edge
            idx1 = 2 * (i * self.lattice_size + 0)
            idx2 = 2 * (i * self.lattice_size + 0) + 1
            
            edge_density += np.abs(self.state[idx1])**2 + np.abs(self.state[idx2])**2
            
            # Right edge
            idx1 = 2 * (i * self.lattice_size + (self.lattice_size - 1))
            idx2 = 2 * (i * self.lattice_size + (self.lattice_size - 1)) + 1
            
            edge_density += np.abs(self.state[idx1])**2 + np.abs(self.state[idx2])**2
        
        # Normalize by total edge sites
        edge_density /= 4 * self.lattice_size
        
        return edge_density
    
    def calculate_spin_polarization(self):
        """
        Calculate the spin polarization of the state.
        
        Returns:
            np.ndarray: Spin polarization vector
        """
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Initialize spin polarization
        spin_polarization = np.zeros(3, dtype=complex)
        
        # Calculate expectation values
        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                # Current site index
                site_idx = i * self.lattice_size + j
                
                # Extract spin state
                spin_state = self.state[2*site_idx:2*site_idx+2]
                
                # Calculate expectation values
                spin_polarization[0] += np.vdot(spin_state, np.dot(sigma_x, spin_state))
                spin_polarization[1] += np.vdot(spin_state, np.dot(sigma_y, spin_state))
                spin_polarization[2] += np.vdot(spin_state, np.dot(sigma_z, spin_state))
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(spin_polarization)**2))
        if norm > 0:
            spin_polarization /= norm
        
        return spin_polarization
    
    def calculate_chern_number(self):
        """
        Calculate the Chern number of the system.
        
        Returns:
            float: Chern number
        """
        # This is a simplified calculation of the Chern number
        # In a real implementation, this would involve integrating the Berry curvature
        
        # For the Bernevig-Hughes-Zhang model, the Chern number is determined by the sign of the mass
        if self.mass < 0:
            return 1.0
        else:
            return 0.0


class TopologicalInsulatorSimulator:
    """
    Simulator for quantum effects in topological insulators.
    
    This class implements tools for simulating quantum effects in topological insulators
    using the phase synchronization mechanism from the Tibedo Framework.
    """
    
    def __init__(self, lattice_size=10):
        """
        Initialize the TopologicalInsulatorSimulator.
        
        Args:
            lattice_size (int): Size of the lattice
        """
        self.lattice_size = lattice_size
        
        # Create topological state
        self.topological_state = TopologicalState(lattice_size)
        
        # Initialize results
        self.results = {}
    
    def set_mass(self, mass):
        """
        Set the mass parameter.
        
        Args:
            mass (float): Mass parameter
            
        Returns:
            float: Mass parameter
        """
        return self.topological_state.set_mass(mass)
    
    def set_spin_orbit_coupling(self, coupling):
        """
        Set the spin-orbit coupling.
        
        Args:
            coupling (float): Spin-orbit coupling
            
        Returns:
            float: Spin-orbit coupling
        """
        return self.topological_state.set_spin_orbit_coupling(coupling)
    
    def simulate_edge_state_dynamics(self, total_time=1000.0, n_steps=1000, 
                                    use_phase_sync=True, coupling_strength=0.1):
        """
        Simulate edge state dynamics.
        
        Args:
            total_time (float): Total simulation time in fs
            n_steps (int): Number of time steps
            use_phase_sync (bool): Whether to use phase synchronization
            coupling_strength (float): Strength of phase synchronization
            
        Returns:
            dict: Simulation results
        """
        # Create time array
        times = np.linspace(0, total_time, n_steps)
        dt = times[1] - times[0]
        
        # Initialize results
        edge_localizations = np.zeros(n_steps)
        spin_polarizations = np.zeros((n_steps, 3), dtype=complex)
        
        # Reset topological state
        self.topological_state = TopologicalState(self.lattice_size)
        
        # Record initial state
        edge_localizations[0] = self.topological_state.calculate_edge_localization()
        spin_polarizations[0] = self.topological_state.calculate_spin_polarization()
        
        # Simulate dynamics
        for i in range(1, n_steps):
            # Evolve state
            self.topological_state.evolve_state(dt)
            
            # Apply phase synchronization if enabled
            if use_phase_sync:
                self.topological_state.apply_phase_synchronization(coupling_strength)
            
            # Record state
            edge_localizations[i] = self.topological_state.calculate_edge_localization()
            spin_polarizations[i] = self.topological_state.calculate_spin_polarization()
        
        # Store results
        self.results = {
            'times': times,
            'edge_localizations': edge_localizations,
            'spin_polarizations': spin_polarizations,
            'mass': self.topological_state.mass,
            'spin_orbit_coupling': self.topological_state.spin_orbit_coupling,
            'chern_number': self.topological_state.calculate_chern_number(),
            'use_phase_sync': use_phase_sync,
            'coupling_strength': coupling_strength
        }
        
        return self.results
    
    def calculate_mass_dependence(self, masses=np.linspace(-2.0, 2.0, 21), 
                                 total_time=1000.0, n_steps=1000, use_phase_sync=True):
        """
        Calculate the dependence of topological properties on the mass parameter.
        
        Args:
            masses (np.ndarray): Mass parameters
            total_time (float): Total simulation time in fs
            n_steps (int): Number of time steps
            use_phase_sync (bool): Whether to use phase synchronization
            
        Returns:
            dict: Mass dependence results
        """
        # Initialize results
        edge_localizations = np.zeros(len(masses))
        chern_numbers = np.zeros(len(masses))
        
        # Calculate properties for each mass
        for i, mass in enumerate(masses):
            # Set mass
            self.set_mass(mass)
            
            # Simulate dynamics
            self.simulate_edge_state_dynamics(
                total_time=total_time,
                n_steps=n_steps,
                use_phase_sync=use_phase_sync
            )
            
            # Record final values
            edge_localizations[i] = self.results['edge_localizations'][-1]
            chern_numbers[i] = self.results['chern_number']
        
        # Store results
        mass_results = {
            'masses': masses,
            'edge_localizations': edge_localizations,
            'chern_numbers': chern_numbers,
            'spin_orbit_coupling': self.topological_state.spin_orbit_coupling,
            'use_phase_sync': use_phase_sync
        }
        
        return mass_results
    
    def calculate_phase_diagram(self, masses=np.linspace(-2.0, 2.0, 11), 
                              couplings=np.linspace(0.0, 2.0, 11), 
                              total_time=1000.0, n_steps=1000):
        """
        Calculate the phase diagram of the topological insulator.
        
        Args:
            masses (np.ndarray): Mass parameters
            couplings (np.ndarray): Spin-orbit couplings
            total_time (float): Total simulation time in fs
            n_steps (int): Number of time steps
            
        Returns:
            dict: Phase diagram results
        """
        # Initialize results
        edge_localizations = np.zeros((len(masses), len(couplings)))
        chern_numbers = np.zeros((len(masses), len(couplings)))
        
        # Calculate properties for each mass and coupling
        for i, mass in enumerate(masses):
            # Set mass
            self.set_mass(mass)
            
            for j, coupling in enumerate(couplings):
                # Set coupling
                self.set_spin_orbit_coupling(coupling)
                
                # Simulate dynamics
                self.simulate_edge_state_dynamics(
                    total_time=total_time,
                    n_steps=n_steps,
                    use_phase_sync=True
                )
                
                # Record final values
                edge_localizations[i, j] = self.results['edge_localizations'][-1]
                chern_numbers[i, j] = self.results['chern_number']
        
        # Store results
        phase_diagram = {
            'masses': masses,
            'couplings': couplings,
            'edge_localizations': edge_localizations,
            'chern_numbers': chern_numbers
        }
        
        return phase_diagram
    
    def visualize_edge_state_dynamics(self):
        """
        Visualize edge state dynamics.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if not self.results:
            raise ValueError("No simulation results available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot edge localization
        times = self.results['times']
        edge_localizations = self.results['edge_localizations']
        
        ax1.plot(times, edge_localizations, 'b-', linewidth=2)
        
        # Set labels
        ax1.set_xlabel('Time (fs)')
        ax1.set_ylabel('Edge Localization')
        ax1.set_title('Edge State Localization')
        ax1.grid(True, alpha=0.3)
        
        # Plot spin polarization
        spin_polarizations = self.results['spin_polarizations']
        
        ax2.plot(times, np.real(spin_polarizations[:, 0]), 'r-', linewidth=2, label='x')
        ax2.plot(times, np.real(spin_polarizations[:, 1]), 'g-', linewidth=2, label='y')
        ax2.plot(times, np.real(spin_polarizations[:, 2]), 'b-', linewidth=2, label='z')
        
        # Set labels
        ax2.set_xlabel('Time (fs)')
        ax2.set_ylabel('Spin Polarization')
        ax2.set_title('Spin Polarization Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_mass_dependence(self, mass_results):
        """
        Visualize the dependence of topological properties on the mass parameter.
        
        Args:
            mass_results (dict): Mass dependence results
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot edge localization vs. mass
        masses = mass_results['masses']
        edge_localizations = mass_results['edge_localizations']
        
        ax1.plot(masses, edge_localizations, 'b-', linewidth=2)
        
        # Mark phase transition
        ax1.axvline(x=0.0, color='r', linestyle='--')
        
        # Set labels
        ax1.set_xlabel('Mass Parameter')
        ax1.set_ylabel('Edge Localization')
        ax1.set_title('Edge Localization vs. Mass')
        ax1.grid(True, alpha=0.3)
        
        # Plot Chern number vs. mass
        chern_numbers = mass_results['chern_numbers']
        
        ax2.plot(masses, chern_numbers, 'g-', linewidth=2)
        
        # Mark phase transition
        ax2.axvline(x=0.0, color='r', linestyle='--')
        
        # Set labels
        ax2.set_xlabel('Mass Parameter')
        ax2.set_ylabel('Chern Number')
        ax2.set_title('Chern Number vs. Mass')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_phase_diagram(self, phase_diagram):
        """
        Visualize the phase diagram of the topological insulator.
        
        Args:
            phase_diagram (dict): Phase diagram results
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot edge localization as a 2D color map
        masses = phase_diagram['masses']
        couplings = phase_diagram['couplings']
        edge_localizations = phase_diagram['edge_localizations']
        
        X, Y = np.meshgrid(couplings, masses)
        
        contour1 = ax1.contourf(X, Y, edge_localizations, cmap='viridis')
        
        # Add colorbar
        cbar1 = plt.colorbar(contour1, ax=ax1)
        cbar1.set_label('Edge Localization')
        
        # Mark phase transition
        ax1.axhline(y=0.0, color='r', linestyle='--')
        
        # Set labels
        ax1.set_xlabel('Spin-Orbit Coupling')
        ax1.set_ylabel('Mass Parameter')
        ax1.set_title('Edge Localization')
        
        # Plot Chern number as a 2D color map
        chern_numbers = phase_diagram['chern_numbers']
        
        contour2 = ax2.contourf(X, Y, chern_numbers, cmap='coolwarm', levels=[-0.5, 0.5, 1.5])
        
        # Add colorbar
        cbar2 = plt.colorbar(contour2, ax=ax2)
        cbar2.set_label('Chern Number')
        
        # Mark phase transition
        ax2.axhline(y=0.0, color='r', linestyle='--')
        
        # Set labels
        ax2.set_xlabel('Spin-Orbit Coupling')
        ax2.set_ylabel('Mass Parameter')
        ax2.set_title('Topological Phase Diagram')
        
        plt.tight_layout()
        
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
        
        # Plot edge localization with phase synchronization
        times = with_sync_results['times']
        edge_localizations = with_sync_results['edge_localizations']
        
        ax1.plot(times, edge_localizations, 'b-', linewidth=2)
        
        # Set labels
        ax1.set_xlabel('Time (fs)')
        ax1.set_ylabel('Edge Localization')
        ax1.set_title('With Phase Synchronization')
        ax1.grid(True, alpha=0.3)
        
        # Plot edge localization without phase synchronization
        times = without_sync_results['times']
        edge_localizations = without_sync_results['edge_localizations']
        
        ax2.plot(times, edge_localizations, 'r-', linewidth=2)
        
        # Set labels
        ax2.set_xlabel('Time (fs)')
        ax2.set_ylabel('Edge Localization')
        ax2.set_title('Without Phase Synchronization')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig


class TopologicalInsulatorAnalyzer:
    """
    Analyzer for quantum effects in topological insulators.
    
    This class provides tools for analyzing quantum effects in topological insulators
    using the phase synchronization mechanism from the Tibedo Framework.
    """
    
    def __init__(self):
        """
        Initialize the TopologicalInsulatorAnalyzer.
        """
        # Create simulator
        self.simulator = TopologicalInsulatorSimulator()
        
        # Initialize results
        self.results = {}
    
    def setup_bhz_model(self):
        """
        Set up parameters for the Bernevig-Hughes-Zhang (BHZ) model.
        
        Returns:
            tuple: (mass, spin_orbit_coupling)
        """
        # Set mass parameter (negative for topological phase)
        mass = -0.5
        self.simulator.set_mass(mass)
        
        # Set spin-orbit coupling
        spin_orbit_coupling = 1.0
        self.simulator.set_spin_orbit_coupling(spin_orbit_coupling)
        
        return mass, spin_orbit_coupling
    
    def analyze_mass_effects(self, masses=np.linspace(-2.0, 2.0, 21), 
                            total_time=1000.0, n_steps=1000):
        """
        Analyze the effects of the mass parameter on topological properties.
        
        Args:
            masses (np.ndarray): Mass parameters
            total_time (float): Total simulation time in fs
            n_steps (int): Number of time steps
            
        Returns:
            dict: Analysis results
        """
        # Calculate mass dependence with phase synchronization
        with_sync_results = self.simulator.calculate_mass_dependence(
            masses=masses,
            total_time=total_time,
            n_steps=n_steps,
            use_phase_sync=True
        )
        
        # Calculate mass dependence without phase synchronization
        without_sync_results = self.simulator.calculate_mass_dependence(
            masses=masses,
            total_time=total_time,
            n_steps=n_steps,
            use_phase_sync=False
        )
        
        # Store results
        self.results = {
            'with_sync_results': with_sync_results,
            'without_sync_results': without_sync_results
        }
        
        return self.results
    
    def analyze_phase_synchronization_effects(self, coupling_strengths=np.linspace(0, 0.5, 11), 
                                            total_time=1000.0, n_steps=1000):
        """
        Analyze the effects of phase synchronization on topological properties.
        
        Args:
            coupling_strengths (np.ndarray): Phase synchronization coupling strengths
            total_time (float): Total simulation time in fs
            n_steps (int): Number of time steps
            
        Returns:
            dict: Analysis results
        """
        # Initialize results
        edge_localizations = np.zeros(len(coupling_strengths))
        
        # Analyze each coupling strength
        for i, coupling_strength in enumerate(coupling_strengths):
            # Simulate dynamics
            self.simulator.simulate_edge_state_dynamics(
                total_time=total_time,
                n_steps=n_steps,
                use_phase_sync=True,
                coupling_strength=coupling_strength
            )
            
            # Record final values
            edge_localizations[i] = self.simulator.results['edge_localizations'][-1]
        
        # Store results
        self.results = {
            'coupling_strengths': coupling_strengths,
            'edge_localizations': edge_localizations
        }
        
        return self.results
    
    def analyze_phase_diagram(self, masses=np.linspace(-2.0, 2.0, 11), 
                             couplings=np.linspace(0.0, 2.0, 11), 
                             total_time=1000.0, n_steps=1000):
        """
        Analyze the phase diagram of the topological insulator.
        
        Args:
            masses (np.ndarray): Mass parameters
            couplings (np.ndarray): Spin-orbit couplings
            total_time (float): Total simulation time in fs
            n_steps (int): Number of time steps
            
        Returns:
            dict: Analysis results
        """
        # Calculate phase diagram
        phase_diagram = self.simulator.calculate_phase_diagram(
            masses=masses,
            couplings=couplings,
            total_time=total_time,
            n_steps=n_steps
        )
        
        # Store results
        self.results = {
            'phase_diagram': phase_diagram
        }
        
        return self.results
    
    def visualize_mass_effects(self):
        """
        Visualize the effects of the mass parameter on topological properties.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if 'with_sync_results' not in self.results:
            raise ValueError("Mass effects not analyzed")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot edge localization vs. mass
        masses = self.results['with_sync_results']['masses']
        with_sync_edge_localizations = self.results['with_sync_results']['edge_localizations']
        without_sync_edge_localizations = self.results['without_sync_results']['edge_localizations']
        
        ax1.plot(masses, with_sync_edge_localizations, 'b-', linewidth=2, label='With Phase Sync')
        ax1.plot(masses, without_sync_edge_localizations, 'r-', linewidth=2, label='Without Phase Sync')
        
        # Mark phase transition
        ax1.axvline(x=0.0, color='k', linestyle='--')
        
        # Set labels
        ax1.set_xlabel('Mass Parameter')
        ax1.set_ylabel('Edge Localization')
        ax1.set_title('Edge Localization vs. Mass')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot Chern number vs. mass
        chern_numbers = self.results['with_sync_results']['chern_numbers']
        
        ax2.plot(masses, chern_numbers, 'g-', linewidth=2)
        
        # Mark phase transition
        ax2.axvline(x=0.0, color='k', linestyle='--')
        
        # Set labels
        ax2.set_xlabel('Mass Parameter')
        ax2.set_ylabel('Chern Number')
        ax2.set_title('Chern Number vs. Mass')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_phase_synchronization_effects(self):
        """
        Visualize the effects of phase synchronization on topological properties.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if 'coupling_strengths' not in self.results:
            raise ValueError("Phase synchronization effects not analyzed")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot edge localization vs. coupling strength
        coupling_strengths = self.results['coupling_strengths']
        edge_localizations = self.results['edge_localizations']
        
        ax.plot(coupling_strengths, edge_localizations, 'b-', linewidth=2)
        
        # Set labels
        ax.set_xlabel('Coupling Strength')
        ax.set_ylabel('Edge Localization')
        ax.set_title('Edge Localization vs. Coupling Strength')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_phase_diagram(self):
        """
        Visualize the phase diagram of the topological insulator.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if 'phase_diagram' not in self.results:
            raise ValueError("Phase diagram not analyzed")
        
        # Visualize phase diagram
        fig = self.simulator.visualize_phase_diagram(self.results['phase_diagram'])
        
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