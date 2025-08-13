"""
Quantum Effects in Superconductors

This module implements tools for studying quantum effects in superconductors
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


class CooperPair:
    """
    Representation of a Cooper pair in superconductors.
    
    This class provides methods for representing and manipulating Cooper pairs
    in superconductors using the Tibedo Framework.
    """
    
    def __init__(self, lattice_size=10):
        """
        Initialize the CooperPair.
        
        Args:
            lattice_size (int): Size of the lattice
        """
        self.lattice_size = lattice_size
        
        # Initialize state
        self.state = self._create_initial_state()
        
        # Initialize parameters
        self.temperature = 0.0  # in K
        self.critical_temperature = 10.0  # in K
        self.gap = 1.0  # in meV
        
        # Create quantum state
        self.quantum_state = ConfigurableQuantumState(dimension=7)
        
        # Configure quantum state
        parameters = {
            'phase_factors': np.ones(7),
            'amplitude_factors': np.ones(7) / np.sqrt(7),
            'entanglement_pattern': 'superconductor',
            'cyclotomic_parameters': {'n': 7, 'k': 1},
            'symmetry_breaking': 0.0,
            'entropic_decline': 0.0
        }
        
        self.quantum_state.configure(parameters)
    
    def _create_initial_state(self):
        """
        Create the initial state of the Cooper pair.
        
        Returns:
            np.ndarray: Initial state vector
        """
        # Dimension of the Hilbert space
        dim = self.lattice_size**2
        
        # Create BCS ground state (simplified)
        state = np.zeros(dim, dtype=complex)
        
        # Set uniform amplitude
        state[:] = 1.0 / np.sqrt(dim)
        
        return state
    
    def set_temperature(self, temperature):
        """
        Set the temperature.
        
        Args:
            temperature (float): Temperature in K
            
        Returns:
            float: Temperature
        """
        self.temperature = temperature
        
        # Update gap based on temperature
        if temperature >= self.critical_temperature:
            self.gap = 0.0
        else:
            # BCS temperature dependence
            self.gap = 1.0 * np.sqrt(1.0 - temperature / self.critical_temperature)
        
        return self.temperature
    
    def set_critical_temperature(self, critical_temperature):
        """
        Set the critical temperature.
        
        Args:
            critical_temperature (float): Critical temperature in K
            
        Returns:
            float: Critical temperature
        """
        self.critical_temperature = critical_temperature
        
        # Update gap based on temperature
        if self.temperature >= critical_temperature:
            self.gap = 0.0
        else:
            # BCS temperature dependence
            self.gap = 1.0 * np.sqrt(1.0 - self.temperature / critical_temperature)
        
        return self.critical_temperature
    
    def calculate_hamiltonian(self):
        """
        Calculate the Hamiltonian of the Cooper pair.
        
        Returns:
            np.ndarray: Hamiltonian matrix
        """
        # Dimension of the Hilbert space
        dim = self.lattice_size**2
        
        # Initialize Hamiltonian
        H = np.zeros((dim, dim), dtype=complex)
        
        # Kinetic energy term
        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                # Current site index
                site_idx = i * self.lattice_size + j
                
                # Neighbor sites
                neighbors = []
                
                # Right neighbor
                if j < self.lattice_size - 1:
                    neighbors.append(i * self.lattice_size + j + 1)
                
                # Left neighbor
                if j > 0:
                    neighbors.append(i * self.lattice_size + j - 1)
                
                # Down neighbor
                if i < self.lattice_size - 1:
                    neighbors.append((i + 1) * self.lattice_size + j)
                
                # Up neighbor
                if i > 0:
                    neighbors.append((i - 1) * self.lattice_size + j)
                
                # Add hopping terms
                for neighbor_idx in neighbors:
                    H[site_idx, neighbor_idx] = -1.0  # Hopping parameter
        
        # Pairing term
        for i in range(dim):
            H[i, i] = self.gap
        
        return H
    
    def evolve_state(self, dt):
        """
        Evolve the state of the Cooper pair.
        
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
    
    def calculate_order_parameter(self):
        """
        Calculate the superconducting order parameter.
        
        Returns:
            complex: Order parameter
        """
        # Calculate order parameter as average of state
        order_parameter = np.mean(self.state)
        
        return order_parameter
    
    def calculate_condensation_energy(self):
        """
        Calculate the condensation energy.
        
        Returns:
            float: Condensation energy
        """
        # Calculate Hamiltonian
        H = self.calculate_hamiltonian()
        
        # Calculate energy
        energy = np.real(np.vdot(self.state, np.dot(H, self.state)))
        
        return energy
    
    def calculate_coherence_length(self):
        """
        Calculate the coherence length.
        
        Returns:
            float: Coherence length
        """
        # Calculate coherence length based on BCS theory
        if self.gap > 0:
            coherence_length = 1.0 / self.gap
        else:
            coherence_length = float('inf')
        
        return coherence_length


class SuperconductorSimulator:
    """
    Simulator for quantum effects in superconductors.
    
    This class implements tools for simulating quantum effects in superconductors
    using the phase synchronization mechanism from the Tibedo Framework.
    """
    
    def __init__(self, lattice_size=10):
        """
        Initialize the SuperconductorSimulator.
        
        Args:
            lattice_size (int): Size of the lattice
        """
        self.lattice_size = lattice_size
        
        # Create Cooper pair
        self.cooper_pair = CooperPair(lattice_size)
        
        # Initialize results
        self.results = {}
    
    def set_temperature(self, temperature):
        """
        Set the temperature.
        
        Args:
            temperature (float): Temperature in K
            
        Returns:
            float: Temperature
        """
        return self.cooper_pair.set_temperature(temperature)
    
    def set_critical_temperature(self, critical_temperature):
        """
        Set the critical temperature.
        
        Args:
            critical_temperature (float): Critical temperature in K
            
        Returns:
            float: Critical temperature
        """
        return self.cooper_pair.set_critical_temperature(critical_temperature)
    
    def simulate_cooper_pair_dynamics(self, total_time=1000.0, n_steps=1000, 
                                     use_phase_sync=True, coupling_strength=0.1):
        """
        Simulate Cooper pair dynamics.
        
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
        order_parameters = np.zeros(n_steps, dtype=complex)
        energies = np.zeros(n_steps)
        coherence_lengths = np.zeros(n_steps)
        
        # Reset Cooper pair state
        self.cooper_pair = CooperPair(self.lattice_size)
        
        # Record initial state
        order_parameters[0] = self.cooper_pair.calculate_order_parameter()
        energies[0] = self.cooper_pair.calculate_condensation_energy()
        coherence_lengths[0] = self.cooper_pair.calculate_coherence_length()
        
        # Simulate dynamics
        for i in range(1, n_steps):
            # Evolve state
            self.cooper_pair.evolve_state(dt)
            
            # Apply phase synchronization if enabled
            if use_phase_sync:
                self.cooper_pair.apply_phase_synchronization(coupling_strength)
            
            # Record state
            order_parameters[i] = self.cooper_pair.calculate_order_parameter()
            energies[i] = self.cooper_pair.calculate_condensation_energy()
            coherence_lengths[i] = self.cooper_pair.calculate_coherence_length()
        
        # Store results
        self.results = {
            'times': times,
            'order_parameters': order_parameters,
            'energies': energies,
            'coherence_lengths': coherence_lengths,
            'temperature': self.cooper_pair.temperature,
            'critical_temperature': self.cooper_pair.critical_temperature,
            'gap': self.cooper_pair.gap,
            'use_phase_sync': use_phase_sync,
            'coupling_strength': coupling_strength
        }
        
        return self.results
    
    def calculate_temperature_dependence(self, temperatures=np.linspace(0, 20, 21), 
                                        total_time=1000.0, n_steps=1000, use_phase_sync=True):
        """
        Calculate the temperature dependence of superconducting properties.
        
        Args:
            temperatures (np.ndarray): Temperatures in K
            total_time (float): Total simulation time in fs
            n_steps (int): Number of time steps
            use_phase_sync (bool): Whether to use phase synchronization
            
        Returns:
            dict: Temperature dependence results
        """
        # Initialize results
        order_parameters = np.zeros(len(temperatures), dtype=complex)
        energies = np.zeros(len(temperatures))
        coherence_lengths = np.zeros(len(temperatures))
        
        # Calculate properties for each temperature
        for i, temperature in enumerate(temperatures):
            # Set temperature
            self.set_temperature(temperature)
            
            # Simulate dynamics
            self.simulate_cooper_pair_dynamics(
                total_time=total_time,
                n_steps=n_steps,
                use_phase_sync=use_phase_sync
            )
            
            # Record final values
            order_parameters[i] = self.results['order_parameters'][-1]
            energies[i] = self.results['energies'][-1]
            coherence_lengths[i] = self.results['coherence_lengths'][-1]
        
        # Store results
        temperature_results = {
            'temperatures': temperatures,
            'order_parameters': order_parameters,
            'energies': energies,
            'coherence_lengths': coherence_lengths,
            'critical_temperature': self.cooper_pair.critical_temperature,
            'use_phase_sync': use_phase_sync
        }
        
        return temperature_results
    
    def calculate_phase_diagram(self, temperatures=np.linspace(0, 20, 11), 
                              coupling_strengths=np.linspace(0, 0.5, 11), 
                              total_time=1000.0, n_steps=1000):
        """
        Calculate the phase diagram of the superconductor.
        
        Args:
            temperatures (np.ndarray): Temperatures in K
            coupling_strengths (np.ndarray): Phase synchronization coupling strengths
            total_time (float): Total simulation time in fs
            n_steps (int): Number of time steps
            
        Returns:
            dict: Phase diagram results
        """
        # Initialize results
        order_parameters = np.zeros((len(temperatures), len(coupling_strengths)), dtype=complex)
        
        # Calculate order parameter for each temperature and coupling strength
        for i, temperature in enumerate(temperatures):
            # Set temperature
            self.set_temperature(temperature)
            
            for j, coupling_strength in enumerate(coupling_strengths):
                # Simulate dynamics
                self.simulate_cooper_pair_dynamics(
                    total_time=total_time,
                    n_steps=n_steps,
                    use_phase_sync=True,
                    coupling_strength=coupling_strength
                )
                
                # Record final order parameter
                order_parameters[i, j] = self.results['order_parameters'][-1]
        
        # Store results
        phase_diagram = {
            'temperatures': temperatures,
            'coupling_strengths': coupling_strengths,
            'order_parameters': order_parameters,
            'critical_temperature': self.cooper_pair.critical_temperature
        }
        
        return phase_diagram
    
    def visualize_cooper_pair_dynamics(self):
        """
        Visualize Cooper pair dynamics.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if not self.results:
            raise ValueError("No simulation results available")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot order parameter magnitude
        times = self.results['times']
        order_parameters = self.results['order_parameters']
        
        ax1.plot(times, np.abs(order_parameters), 'b-', linewidth=2)
        
        # Set labels
        ax1.set_xlabel('Time (fs)')
        ax1.set_ylabel('|Δ|')
        ax1.set_title('Order Parameter Magnitude')
        ax1.grid(True, alpha=0.3)
        
        # Plot order parameter phase
        ax2.plot(times, np.angle(order_parameters), 'r-', linewidth=2)
        
        # Set labels
        ax2.set_xlabel('Time (fs)')
        ax2.set_ylabel('arg(Δ)')
        ax2.set_title('Order Parameter Phase')
        ax2.grid(True, alpha=0.3)
        
        # Plot condensation energy
        energies = self.results['energies']
        
        ax3.plot(times, energies, 'g-', linewidth=2)
        
        # Set labels
        ax3.set_xlabel('Time (fs)')
        ax3.set_ylabel('Energy')
        ax3.set_title('Condensation Energy')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_temperature_dependence(self, temperature_results):
        """
        Visualize temperature dependence of superconducting properties.
        
        Args:
            temperature_results (dict): Temperature dependence results
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot order parameter magnitude vs. temperature
        temperatures = temperature_results['temperatures']
        order_parameters = temperature_results['order_parameters']
        
        ax1.plot(temperatures, np.abs(order_parameters), 'b-', linewidth=2)
        
        # Mark critical temperature
        critical_temperature = temperature_results['critical_temperature']
        ax1.axvline(x=critical_temperature, color='r', linestyle='--')
        
        # Set labels
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('|Δ|')
        ax1.set_title('Order Parameter vs. Temperature')
        ax1.grid(True, alpha=0.3)
        
        # Plot coherence length vs. temperature
        coherence_lengths = temperature_results['coherence_lengths']
        
        ax2.plot(temperatures, coherence_lengths, 'g-', linewidth=2)
        
        # Mark critical temperature
        ax2.axvline(x=critical_temperature, color='r', linestyle='--')
        
        # Set labels
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Coherence Length')
        ax2.set_title('Coherence Length vs. Temperature')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_phase_diagram(self, phase_diagram):
        """
        Visualize the phase diagram of the superconductor.
        
        Args:
            phase_diagram (dict): Phase diagram results
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot order parameter magnitude as a 2D color map
        temperatures = phase_diagram['temperatures']
        coupling_strengths = phase_diagram['coupling_strengths']
        order_parameters = phase_diagram['order_parameters']
        
        X, Y = np.meshgrid(coupling_strengths, temperatures)
        
        contour = ax.contourf(X, Y, np.abs(order_parameters), cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('|Δ|')
        
        # Mark critical temperature
        critical_temperature = phase_diagram['critical_temperature']
        ax.axhline(y=critical_temperature, color='r', linestyle='--')
        
        # Set labels
        ax.set_xlabel('Coupling Strength')
        ax.set_ylabel('Temperature (K)')
        ax.set_title('Phase Diagram')
        
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
        
        # Plot order parameter magnitude with phase synchronization
        times = with_sync_results['times']
        order_parameters = with_sync_results['order_parameters']
        
        ax1.plot(times, np.abs(order_parameters), 'b-', linewidth=2)
        
        # Set labels
        ax1.set_xlabel('Time (fs)')
        ax1.set_ylabel('|Δ|')
        ax1.set_title('With Phase Synchronization')
        ax1.grid(True, alpha=0.3)
        
        # Plot order parameter magnitude without phase synchronization
        times = without_sync_results['times']
        order_parameters = without_sync_results['order_parameters']
        
        ax2.plot(times, np.abs(order_parameters), 'r-', linewidth=2)
        
        # Set labels
        ax2.set_xlabel('Time (fs)')
        ax2.set_ylabel('|Δ|')
        ax2.set_title('Without Phase Synchronization')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig


class SuperconductorAnalyzer:
    """
    Analyzer for quantum effects in superconductors.
    
    This class provides tools for analyzing quantum effects in superconductors
    using the phase synchronization mechanism from the Tibedo Framework.
    """
    
    def __init__(self):
        """
        Initialize the SuperconductorAnalyzer.
        """
        # Create simulator
        self.simulator = SuperconductorSimulator()
        
        # Initialize results
        self.results = {}
    
    def setup_high_tc_superconductor(self):
        """
        Set up parameters for a high-temperature superconductor.
        
        Returns:
            float: Critical temperature
        """
        # Set critical temperature (e.g., YBCO ~ 90 K)
        critical_temperature = 90.0
        self.simulator.set_critical_temperature(critical_temperature)
        
        # Set initial temperature
        self.simulator.set_temperature(0.0)
        
        return critical_temperature
    
    def analyze_temperature_effects(self, temperatures=np.linspace(0, 100, 11), 
                                   total_time=1000.0, n_steps=1000):
        """
        Analyze the effects of temperature on superconducting properties.
        
        Args:
            temperatures (np.ndarray): Temperatures in K
            total_time (float): Total simulation time in fs
            n_steps (int): Number of time steps
            
        Returns:
            dict: Analysis results
        """
        # Calculate temperature dependence with phase synchronization
        with_sync_results = self.simulator.calculate_temperature_dependence(
            temperatures=temperatures,
            total_time=total_time,
            n_steps=n_steps,
            use_phase_sync=True
        )
        
        # Calculate temperature dependence without phase synchronization
        without_sync_results = self.simulator.calculate_temperature_dependence(
            temperatures=temperatures,
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
        Analyze the effects of phase synchronization on superconducting properties.
        
        Args:
            coupling_strengths (np.ndarray): Phase synchronization coupling strengths
            total_time (float): Total simulation time in fs
            n_steps (int): Number of time steps
            
        Returns:
            dict: Analysis results
        """
        # Initialize results
        order_parameters = np.zeros(len(coupling_strengths), dtype=complex)
        energies = np.zeros(len(coupling_strengths))
        coherence_lengths = np.zeros(len(coupling_strengths))
        
        # Analyze each coupling strength
        for i, coupling_strength in enumerate(coupling_strengths):
            # Simulate dynamics
            self.simulator.simulate_cooper_pair_dynamics(
                total_time=total_time,
                n_steps=n_steps,
                use_phase_sync=True,
                coupling_strength=coupling_strength
            )
            
            # Record final values
            order_parameters[i] = self.simulator.results['order_parameters'][-1]
            energies[i] = self.simulator.results['energies'][-1]
            coherence_lengths[i] = self.simulator.results['coherence_lengths'][-1]
        
        # Store results
        self.results = {
            'coupling_strengths': coupling_strengths,
            'order_parameters': order_parameters,
            'energies': energies,
            'coherence_lengths': coherence_lengths
        }
        
        return self.results
    
    def analyze_phase_diagram(self, temperatures=np.linspace(0, 100, 11), 
                             coupling_strengths=np.linspace(0, 0.5, 11), 
                             total_time=1000.0, n_steps=1000):
        """
        Analyze the phase diagram of the superconductor.
        
        Args:
            temperatures (np.ndarray): Temperatures in K
            coupling_strengths (np.ndarray): Phase synchronization coupling strengths
            total_time (float): Total simulation time in fs
            n_steps (int): Number of time steps
            
        Returns:
            dict: Analysis results
        """
        # Calculate phase diagram
        phase_diagram = self.simulator.calculate_phase_diagram(
            temperatures=temperatures,
            coupling_strengths=coupling_strengths,
            total_time=total_time,
            n_steps=n_steps
        )
        
        # Store results
        self.results = {
            'phase_diagram': phase_diagram
        }
        
        return self.results
    
    def visualize_temperature_effects(self):
        """
        Visualize the effects of temperature on superconducting properties.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if 'with_sync_results' not in self.results:
            raise ValueError("Temperature effects not analyzed")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot order parameter magnitude vs. temperature
        temperatures = self.results['with_sync_results']['temperatures']
        with_sync_order_parameters = self.results['with_sync_results']['order_parameters']
        without_sync_order_parameters = self.results['without_sync_results']['order_parameters']
        
        ax1.plot(temperatures, np.abs(with_sync_order_parameters), 'b-', linewidth=2, label='With Phase Sync')
        ax1.plot(temperatures, np.abs(without_sync_order_parameters), 'r-', linewidth=2, label='Without Phase Sync')
        
        # Mark critical temperature
        critical_temperature = self.results['with_sync_results']['critical_temperature']
        ax1.axvline(x=critical_temperature, color='k', linestyle='--')
        
        # Set labels
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('|Δ|')
        ax1.set_title('Order Parameter vs. Temperature')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot coherence length vs. temperature
        with_sync_coherence_lengths = self.results['with_sync_results']['coherence_lengths']
        without_sync_coherence_lengths = self.results['without_sync_results']['coherence_lengths']
        
        ax2.plot(temperatures, with_sync_coherence_lengths, 'b-', linewidth=2, label='With Phase Sync')
        ax2.plot(temperatures, without_sync_coherence_lengths, 'r-', linewidth=2, label='Without Phase Sync')
        
        # Mark critical temperature
        ax2.axvline(x=critical_temperature, color='k', linestyle='--')
        
        # Set labels
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Coherence Length')
        ax2.set_title('Coherence Length vs. Temperature')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_phase_synchronization_effects(self):
        """
        Visualize the effects of phase synchronization on superconducting properties.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if 'coupling_strengths' not in self.results:
            raise ValueError("Phase synchronization effects not analyzed")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot order parameter magnitude vs. coupling strength
        coupling_strengths = self.results['coupling_strengths']
        order_parameters = self.results['order_parameters']
        
        ax1.plot(coupling_strengths, np.abs(order_parameters), 'b-', linewidth=2)
        
        # Set labels
        ax1.set_xlabel('Coupling Strength')
        ax1.set_ylabel('|Δ|')
        ax1.set_title('Order Parameter vs. Coupling Strength')
        ax1.grid(True, alpha=0.3)
        
        # Plot condensation energy vs. coupling strength
        energies = self.results['energies']
        
        ax2.plot(coupling_strengths, energies, 'g-', linewidth=2)
        
        # Set labels
        ax2.set_xlabel('Coupling Strength')
        ax2.set_ylabel('Energy')
        ax2.set_title('Condensation Energy vs. Coupling Strength')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_phase_diagram(self):
        """
        Visualize the phase diagram of the superconductor.
        
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