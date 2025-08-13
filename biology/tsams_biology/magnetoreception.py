"""
Quantum Effects in Magnetoreception

This module implements tools for studying quantum effects in magnetoreception
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


class RadicalPair:
    """
    Representation of a radical pair for magnetoreception.
    
    This class provides methods for representing and manipulating radical pairs
    in magnetoreception using the Tibedo Framework.
    """
    
    def __init__(self, n_nuclear_spins=3):
        """
        Initialize the RadicalPair.
        
        Args:
            n_nuclear_spins (int): Number of nuclear spins
        """
        self.n_nuclear_spins = n_nuclear_spins
        
        # Pauli matrices
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Initialize state
        self.state = self._create_initial_state()
        
        # Initialize parameters
        self.g_factor = 2.0023  # g-factor for free electron
        self.hyperfine_couplings = np.ones(n_nuclear_spins) * 1.0  # mT
        
        # Initialize magnetic field
        self.magnetic_field = np.array([0.0, 0.0, 50.0])  # Earth's magnetic field in μT
        
        # Create quantum state
        self.quantum_state = ConfigurableQuantumState(dimension=7)
        
        # Configure quantum state
        parameters = {
            'phase_factors': np.ones(7),
            'amplitude_factors': np.ones(7) / np.sqrt(7),
            'entanglement_pattern': 'magnetoreception',
            'cyclotomic_parameters': {'n': 7, 'k': 1},
            'symmetry_breaking': 0.0,
            'entropic_decline': 0.0
        }
        
        self.quantum_state.configure(parameters)
    
    def _create_initial_state(self):
        """
        Create the initial state of the radical pair.
        
        Returns:
            np.ndarray: Initial state vector
        """
        # Dimension of the Hilbert space
        # 2 (electron 1) × 2 (electron 2) × 2^n_nuclear_spins
        dim = 4 * 2**self.n_nuclear_spins
        
        # Create singlet state for electrons
        singlet = np.zeros((4, 1), dtype=complex)
        singlet[0, 0] = 0.0  # |↑↑⟩
        singlet[1, 0] = 1.0 / np.sqrt(2.0)  # |↑↓⟩
        singlet[2, 0] = -1.0 / np.sqrt(2.0)  # |↓↑⟩
        singlet[3, 0] = 0.0  # |↓↓⟩
        
        # Create state for nuclear spins (all in |↑⟩ state)
        nuclear = np.zeros((2**self.n_nuclear_spins, 1), dtype=complex)
        nuclear[0, 0] = 1.0
        
        # Combine electron and nuclear states
        state = np.kron(singlet, nuclear)
        
        return state
    
    def set_hyperfine_couplings(self, couplings):
        """
        Set the hyperfine couplings.
        
        Args:
            couplings (np.ndarray): Hyperfine couplings in mT
            
        Returns:
            np.ndarray: Hyperfine couplings
        """
        if len(couplings) != self.n_nuclear_spins:
            raise ValueError(f"Couplings must have length {self.n_nuclear_spins}")
        
        self.hyperfine_couplings = np.array(couplings)
        
        return self.hyperfine_couplings
    
    def set_magnetic_field(self, field):
        """
        Set the magnetic field.
        
        Args:
            field (np.ndarray): Magnetic field vector in μT
            
        Returns:
            np.ndarray: Magnetic field
        """
        if len(field) != 3:
            raise ValueError("Field must have length 3")
        
        self.magnetic_field = np.array(field)
        
        return self.magnetic_field
    
    def calculate_hamiltonian(self):
        """
        Calculate the Hamiltonian of the radical pair.
        
        Returns:
            np.ndarray: Hamiltonian matrix
        """
        # Dimension of the Hilbert space
        dim = 4 * 2**self.n_nuclear_spins
        
        # Initialize Hamiltonian
        H = np.zeros((dim, dim), dtype=complex)
        
        # Zeeman interaction
        H += self._calculate_zeeman_hamiltonian()
        
        # Hyperfine interaction
        H += self._calculate_hyperfine_hamiltonian()
        
        return H
    
    def _calculate_zeeman_hamiltonian(self):
        """
        Calculate the Zeeman Hamiltonian.
        
        Returns:
            np.ndarray: Zeeman Hamiltonian
        """
        # Dimension of the Hilbert space
        dim = 4 * 2**self.n_nuclear_spins
        
        # Initialize Hamiltonian
        H_zeeman = np.zeros((dim, dim), dtype=complex)
        
        # Constants
        mu_B = 5.788e-5  # Bohr magneton in eV/T
        
        # Magnetic field in T
        B = self.magnetic_field * 1e-6
        
        # Zeeman interaction for electron 1
        for i, B_i in enumerate(B):
            if i == 0:
                sigma = np.kron(self.sigma_x, np.eye(2))
            elif i == 1:
                sigma = np.kron(self.sigma_y, np.eye(2))
            else:  # i == 2
                sigma = np.kron(self.sigma_z, np.eye(2))
            
            # Extend to full Hilbert space
            sigma_full = np.kron(sigma, np.eye(2**self.n_nuclear_spins))
            
            # Add to Hamiltonian
            H_zeeman += mu_B * self.g_factor * B_i * sigma_full
        
        # Zeeman interaction for electron 2
        for i, B_i in enumerate(B):
            if i == 0:
                sigma = np.kron(np.eye(2), self.sigma_x)
            elif i == 1:
                sigma = np.kron(np.eye(2), self.sigma_y)
            else:  # i == 2
                sigma = np.kron(np.eye(2), self.sigma_z)
            
            # Extend to full Hilbert space
            sigma_full = np.kron(sigma, np.eye(2**self.n_nuclear_spins))
            
            # Add to Hamiltonian
            H_zeeman += mu_B * self.g_factor * B_i * sigma_full
        
        return H_zeeman
    
    def _calculate_hyperfine_hamiltonian(self):
        """
        Calculate the hyperfine Hamiltonian.
        
        Returns:
            np.ndarray: Hyperfine Hamiltonian
        """
        # Dimension of the Hilbert space
        dim = 4 * 2**self.n_nuclear_spins
        
        # Initialize Hamiltonian
        H_hf = np.zeros((dim, dim), dtype=complex)
        
        # Constants
        mu_B = 5.788e-5  # Bohr magneton in eV/T
        
        # Hyperfine interaction for each nuclear spin
        for n in range(self.n_nuclear_spins):
            # Hyperfine coupling in T
            A = self.hyperfine_couplings[n] * 1e-3
            
            # Create nuclear spin operators
            I_x = self._create_nuclear_spin_operator(n, 0)
            I_y = self._create_nuclear_spin_operator(n, 1)
            I_z = self._create_nuclear_spin_operator(n, 2)
            
            # Electron spin operators (only for electron 1)
            S_x = np.kron(np.kron(self.sigma_x, np.eye(2)), np.eye(2**self.n_nuclear_spins))
            S_y = np.kron(np.kron(self.sigma_y, np.eye(2)), np.eye(2**self.n_nuclear_spins))
            S_z = np.kron(np.kron(self.sigma_z, np.eye(2)), np.eye(2**self.n_nuclear_spins))
            
            # Add isotropic hyperfine interaction
            H_hf += A * (S_x @ I_x + S_y @ I_y + S_z @ I_z)
        
        return H_hf
    
    def _create_nuclear_spin_operator(self, n, axis):
        """
        Create a nuclear spin operator.
        
        Args:
            n (int): Nuclear spin index
            axis (int): Axis (0 for x, 1 for y, 2 for z)
            
        Returns:
            np.ndarray: Nuclear spin operator
        """
        # Select Pauli matrix
        if axis == 0:
            sigma = self.sigma_x
        elif axis == 1:
            sigma = self.sigma_y
        else:  # axis == 2
            sigma = self.sigma_z
        
        # Create operator for nuclear spin n
        op = np.eye(1)
        
        # Electron spins
        op = np.kron(np.eye(4), op)
        
        # Nuclear spins before n
        op = np.kron(op, np.eye(2**n))
        
        # Nuclear spin n
        op = np.kron(op, sigma)
        
        # Nuclear spins after n
        if n < self.n_nuclear_spins - 1:
            op = np.kron(op, np.eye(2**(self.n_nuclear_spins - n - 1)))
        
        return op
    
    def evolve_state(self, dt):
        """
        Evolve the state of the radical pair.
        
        Args:
            dt (float): Time step in ns
            
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
        # Extract electron spin state (trace out nuclear spins)
        electron_state = self._extract_electron_state()
        
        # Apply phase synchronization to electron state
        synchronized_electron_state = self._apply_phase_sync_to_electron(electron_state, coupling_strength)
        
        # Update state
        self._update_state_with_electron(synchronized_electron_state)
        
        return self.state
    
    def _extract_electron_state(self):
        """
        Extract the electron spin state by tracing out nuclear spins.
        
        Returns:
            np.ndarray: Electron spin state (density matrix)
        """
        # Reshape state to separate electron and nuclear parts
        state_reshaped = self.state.reshape(4, 2**self.n_nuclear_spins)
        
        # Calculate reduced density matrix
        rho_electron = np.zeros((4, 4), dtype=complex)
        
        for i in range(4):
            for j in range(4):
                for k in range(2**self.n_nuclear_spins):
                    rho_electron[i, j] += state_reshaped[i, k] * np.conj(state_reshaped[j, k])
        
        return rho_electron
    
    def _apply_phase_sync_to_electron(self, electron_state, coupling_strength):
        """
        Apply phase synchronization to the electron spin state.
        
        Args:
            electron_state (np.ndarray): Electron spin state (density matrix)
            coupling_strength (float): Strength of phase synchronization
            
        Returns:
            np.ndarray: Synchronized electron spin state (density matrix)
        """
        # Extract state vector (assume pure state)
        eigenvalues, eigenvectors = la.eigh(electron_state)
        state_vector = eigenvectors[:, -1]
        
        # Reshape to match quantum state dimension
        if len(state_vector) > self.quantum_state.dimension:
            # Truncate
            state_vector = state_vector[:self.quantum_state.dimension]
        else:
            # Pad with zeros
            padded = np.zeros(self.quantum_state.dimension, dtype=complex)
            padded[:len(state_vector)] = state_vector
            state_vector = padded
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(state_vector)**2))
        if norm > 0:
            state_vector /= norm
        
        # Apply phase synchronization
        synchronized_vector = self.quantum_state.apply_phase_synchronization(
            state_vector, coupling_strength
        )
        
        # Truncate back to original size
        synchronized_vector = synchronized_vector[:4]
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(synchronized_vector)**2))
        if norm > 0:
            synchronized_vector /= norm
        
        # Create density matrix
        synchronized_state = np.outer(synchronized_vector, np.conj(synchronized_vector))
        
        return synchronized_state
    
    def _update_state_with_electron(self, electron_state):
        """
        Update the state with a new electron spin state.
        
        Args:
            electron_state (np.ndarray): Electron spin state (density matrix)
            
        Returns:
            np.ndarray: Updated state
        """
        # Extract state vector (assume pure state)
        eigenvalues, eigenvectors = la.eigh(electron_state)
        electron_vector = eigenvectors[:, -1]
        
        # Reshape state to separate electron and nuclear parts
        state_reshaped = self.state.reshape(4, 2**self.n_nuclear_spins)
        
        # Calculate nuclear state
        nuclear_state = np.zeros(2**self.n_nuclear_spins, dtype=complex)
        
        for i in range(4):
            nuclear_state += np.conj(electron_vector[i]) * state_reshaped[i, :]
        
        # Normalize nuclear state
        norm = np.sqrt(np.sum(np.abs(nuclear_state)**2))
        if norm > 0:
            nuclear_state /= norm
        
        # Create new state
        new_state = np.zeros((4, 2**self.n_nuclear_spins), dtype=complex)
        
        for i in range(4):
            new_state[i, :] = electron_vector[i] * nuclear_state
        
        # Reshape back
        self.state = new_state.reshape(-1, 1)
        
        return self.state
    
    def calculate_singlet_probability(self):
        """
        Calculate the singlet probability.
        
        Returns:
            float: Singlet probability
        """
        # Extract electron spin state
        electron_state = self._extract_electron_state()
        
        # Singlet projector
        P_S = np.zeros((4, 4), dtype=complex)
        P_S[1, 1] = 0.5
        P_S[1, 2] = -0.5
        P_S[2, 1] = -0.5
        P_S[2, 2] = 0.5
        
        # Calculate singlet probability
        singlet_prob = np.real(np.trace(np.dot(P_S, electron_state)))
        
        return singlet_prob
    
    def calculate_triplet_probability(self):
        """
        Calculate the triplet probability.
        
        Returns:
            float: Triplet probability
        """
        # Extract electron spin state
        electron_state = self._extract_electron_state()
        
        # Triplet projector
        P_T = np.zeros((4, 4), dtype=complex)
        P_T[0, 0] = 1.0
        P_T[1, 1] = 0.5
        P_T[1, 2] = 0.5
        P_T[2, 1] = 0.5
        P_T[2, 2] = 0.5
        P_T[3, 3] = 1.0
        
        # Calculate triplet probability
        triplet_prob = np.real(np.trace(np.dot(P_T, electron_state)))
        
        return triplet_prob


class MagnetoreceptionSimulator:
    """
    Simulator for quantum effects in magnetoreception.
    
    This class implements tools for simulating quantum effects in magnetoreception
    using the phase synchronization mechanism from the Tibedo Framework.
    """
    
    def __init__(self, n_nuclear_spins=3):
        """
        Initialize the MagnetoreceptionSimulator.
        
        Args:
            n_nuclear_spins (int): Number of nuclear spins
        """
        self.n_nuclear_spins = n_nuclear_spins
        
        # Create radical pair
        self.radical_pair = RadicalPair(n_nuclear_spins)
        
        # Initialize results
        self.results = {}
    
    def set_magnetic_field(self, field):
        """
        Set the magnetic field.
        
        Args:
            field (np.ndarray): Magnetic field vector in μT
            
        Returns:
            np.ndarray: Magnetic field
        """
        return self.radical_pair.set_magnetic_field(field)
    
    def set_hyperfine_couplings(self, couplings):
        """
        Set the hyperfine couplings.
        
        Args:
            couplings (np.ndarray): Hyperfine couplings in mT
            
        Returns:
            np.ndarray: Hyperfine couplings
        """
        return self.radical_pair.set_hyperfine_couplings(couplings)
    
    def simulate_radical_pair_dynamics(self, total_time=1000.0, n_steps=1000, 
                                      use_phase_sync=True, coupling_strength=0.1):
        """
        Simulate radical pair dynamics.
        
        Args:
            total_time (float): Total simulation time in ns
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
        singlet_probs = np.zeros(n_steps)
        triplet_probs = np.zeros(n_steps)
        
        # Reset radical pair state
        self.radical_pair = RadicalPair(self.n_nuclear_spins)
        
        # Record initial state
        singlet_probs[0] = self.radical_pair.calculate_singlet_probability()
        triplet_probs[0] = self.radical_pair.calculate_triplet_probability()
        
        # Simulate dynamics
        for i in range(1, n_steps):
            # Evolve state
            self.radical_pair.evolve_state(dt)
            
            # Apply phase synchronization if enabled
            if use_phase_sync:
                self.radical_pair.apply_phase_synchronization(coupling_strength)
            
            # Record state
            singlet_probs[i] = self.radical_pair.calculate_singlet_probability()
            triplet_probs[i] = self.radical_pair.calculate_triplet_probability()
        
        # Store results
        self.results = {
            'times': times,
            'singlet_probs': singlet_probs,
            'triplet_probs': triplet_probs,
            'magnetic_field': self.radical_pair.magnetic_field,
            'hyperfine_couplings': self.radical_pair.hyperfine_couplings,
            'use_phase_sync': use_phase_sync,
            'coupling_strength': coupling_strength
        }
        
        return self.results
    
    def calculate_yield_anisotropy(self, inclinations=np.linspace(0, np.pi, 19), 
                                  total_time=1000.0, n_steps=1000, use_phase_sync=True):
        """
        Calculate the yield anisotropy for different magnetic field inclinations.
        
        Args:
            inclinations (np.ndarray): Magnetic field inclinations in radians
            total_time (float): Total simulation time in ns
            n_steps (int): Number of time steps
            use_phase_sync (bool): Whether to use phase synchronization
            
        Returns:
            dict: Yield anisotropy results
        """
        # Initialize results
        singlet_yields = np.zeros(len(inclinations))
        triplet_yields = np.zeros(len(inclinations))
        
        # Magnetic field strength
        B_strength = np.linalg.norm(self.radical_pair.magnetic_field)
        
        # Calculate yields for each inclination
        for i, theta in enumerate(inclinations):
            # Set magnetic field
            B = B_strength * np.array([np.sin(theta), 0, np.cos(theta)])
            self.set_magnetic_field(B)
            
            # Simulate dynamics
            self.simulate_radical_pair_dynamics(
                total_time=total_time,
                n_steps=n_steps,
                use_phase_sync=use_phase_sync
            )
            
            # Calculate yields (final probabilities)
            singlet_yields[i] = self.results['singlet_probs'][-1]
            triplet_yields[i] = self.results['triplet_probs'][-1]
        
        # Calculate anisotropy
        max_yield = np.max(singlet_yields)
        min_yield = np.min(singlet_yields)
        anisotropy = (max_yield - min_yield) / max_yield
        
        # Store results
        anisotropy_results = {
            'inclinations': inclinations,
            'singlet_yields': singlet_yields,
            'triplet_yields': triplet_yields,
            'anisotropy': anisotropy,
            'use_phase_sync': use_phase_sync
        }
        
        return anisotropy_results
    
    def calculate_compass_precision(self, inclinations=np.linspace(0, np.pi, 19), 
                                   total_time=1000.0, n_steps=1000, use_phase_sync=True):
        """
        Calculate the compass precision for different magnetic field inclinations.
        
        Args:
            inclinations (np.ndarray): Magnetic field inclinations in radians
            total_time (float): Total simulation time in ns
            n_steps (int): Number of time steps
            use_phase_sync (bool): Whether to use phase synchronization
            
        Returns:
            dict: Compass precision results
        """
        # Calculate yield anisotropy
        anisotropy_results = self.calculate_yield_anisotropy(
            inclinations=inclinations,
            total_time=total_time,
            n_steps=n_steps,
            use_phase_sync=use_phase_sync
        )
        
        # Calculate derivative of singlet yield
        singlet_yields = anisotropy_results['singlet_yields']
        derivative = np.gradient(singlet_yields, inclinations)
        
        # Find maximum derivative
        max_derivative_idx = np.argmax(np.abs(derivative))
        max_derivative = derivative[max_derivative_idx]
        max_derivative_angle = inclinations[max_derivative_idx]
        
        # Calculate precision (inverse of width at half maximum)
        half_max = max_derivative / 2.0
        
        # Find width at half maximum
        width = 0.0
        for i in range(len(derivative)):
            if derivative[i] >= half_max:
                # Find angle where derivative equals half maximum
                if i > 0:
                    # Interpolate
                    angle1 = inclinations[i-1]
                    angle2 = inclinations[i]
                    deriv1 = derivative[i-1]
                    deriv2 = derivative[i]
                    
                    angle = angle1 + (half_max - deriv1) * (angle2 - angle1) / (deriv2 - deriv1)
                    
                    # Calculate width
                    width = abs(angle - max_derivative_angle)
                    break
        
        # Calculate precision
        precision = 1.0 / width if width > 0 else 0.0
        
        # Store results
        precision_results = {
            'inclinations': inclinations,
            'derivative': derivative,
            'max_derivative': max_derivative,
            'max_derivative_angle': max_derivative_angle,
            'width': width,
            'precision': precision,
            'use_phase_sync': use_phase_sync
        }
        
        return precision_results
    
    def visualize_radical_pair_dynamics(self):
        """
        Visualize radical pair dynamics.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if not self.results:
            raise ValueError("No simulation results available")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot singlet and triplet probabilities
        times = self.results['times']
        singlet_probs = self.results['singlet_probs']
        triplet_probs = self.results['triplet_probs']
        
        ax.plot(times, singlet_probs, 'b-', linewidth=2, label='Singlet')
        ax.plot(times, triplet_probs, 'r-', linewidth=2, label='Triplet')
        
        # Set labels
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Probability')
        ax.set_title('Radical Pair Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_yield_anisotropy(self, anisotropy_results):
        """
        Visualize yield anisotropy.
        
        Args:
            anisotropy_results (dict): Yield anisotropy results
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot singlet yield vs. inclination
        inclinations = anisotropy_results['inclinations']
        singlet_yields = anisotropy_results['singlet_yields']
        
        ax.plot(inclinations * 180 / np.pi, singlet_yields, 'b-', linewidth=2)
        
        # Set labels
        ax.set_xlabel('Inclination (degrees)')
        ax.set_ylabel('Singlet Yield')
        ax.set_title(f'Yield Anisotropy (Anisotropy = {anisotropy_results["anisotropy"]:.4f})')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_compass_precision(self, precision_results):
        """
        Visualize compass precision.
        
        Args:
            precision_results (dict): Compass precision results
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot derivative of singlet yield
        inclinations = precision_results['inclinations']
        derivative = precision_results['derivative']
        
        ax.plot(inclinations * 180 / np.pi, derivative, 'b-', linewidth=2)
        
        # Mark maximum derivative
        max_derivative_angle = precision_results['max_derivative_angle']
        max_derivative = precision_results['max_derivative']
        
        ax.plot(max_derivative_angle * 180 / np.pi, max_derivative, 'ro', markersize=8)
        
        # Set labels
        ax.set_xlabel('Inclination (degrees)')
        ax.set_ylabel('dY/dθ')
        ax.set_title(f'Compass Precision (Precision = {precision_results["precision"]:.4f})')
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
        
        # Plot singlet probability with phase synchronization
        times = with_sync_results['times']
        singlet_probs = with_sync_results['singlet_probs']
        
        ax1.plot(times, singlet_probs, 'b-', linewidth=2)
        
        # Set labels
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Singlet Probability')
        ax1.set_title('With Phase Synchronization')
        ax1.grid(True, alpha=0.3)
        
        # Plot singlet probability without phase synchronization
        times = without_sync_results['times']
        singlet_probs = without_sync_results['singlet_probs']
        
        ax2.plot(times, singlet_probs, 'r-', linewidth=2)
        
        # Set labels
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel('Singlet Probability')
        ax2.set_title('Without Phase Synchronization')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig


class MagnetoreceptionAnalyzer:
    """
    Analyzer for quantum effects in magnetoreception.
    
    This class provides tools for analyzing quantum effects in magnetoreception
    using the phase synchronization mechanism from the Tibedo Framework.
    """
    
    def __init__(self):
        """
        Initialize the MagnetoreceptionAnalyzer.
        """
        # Create simulator
        self.simulator = MagnetoreceptionSimulator()
        
        # Initialize results
        self.results = {}
    
    def setup_cryptochrome(self):
        """
        Set up parameters for cryptochrome radical pair.
        
        Returns:
            tuple: (magnetic_field, hyperfine_couplings)
        """
        # Set magnetic field (Earth's magnetic field, ~50 μT)
        magnetic_field = np.array([0.0, 0.0, 50.0])
        self.simulator.set_magnetic_field(magnetic_field)
        
        # Set hyperfine couplings for cryptochrome
        # These are approximate values based on literature
        hyperfine_couplings = np.array([0.5, 0.3, 0.1])
        self.simulator.set_hyperfine_couplings(hyperfine_couplings)
        
        return magnetic_field, hyperfine_couplings
    
    def analyze_magnetic_field_effects(self, field_strengths=np.linspace(0, 100, 11), 
                                      total_time=1000.0, n_steps=1000):
        """
        Analyze the effects of magnetic field strength.
        
        Args:
            field_strengths (np.ndarray): Magnetic field strengths in μT
            total_time (float): Total simulation time in ns
            n_steps (int): Number of time steps
            
        Returns:
            dict: Analysis results
        """
        # Initialize results
        with_sync_yields = np.zeros(len(field_strengths))
        without_sync_yields = np.zeros(len(field_strengths))
        
        # Analyze each field strength
        for i, strength in enumerate(field_strengths):
            # Set magnetic field
            field = np.array([0.0, 0.0, strength])
            self.simulator.set_magnetic_field(field)
            
            # Simulate with phase synchronization
            with_sync_results = self.simulator.simulate_radical_pair_dynamics(
                total_time=total_time,
                n_steps=n_steps,
                use_phase_sync=True
            )
            
            # Simulate without phase synchronization
            without_sync_results = self.simulator.simulate_radical_pair_dynamics(
                total_time=total_time,
                n_steps=n_steps,
                use_phase_sync=False
            )
            
            # Record yields
            with_sync_yields[i] = with_sync_results['singlet_probs'][-1]
            without_sync_yields[i] = without_sync_results['singlet_probs'][-1]
        
        # Store results
        self.results = {
            'field_strengths': field_strengths,
            'with_sync_yields': with_sync_yields,
            'without_sync_yields': without_sync_yields
        }
        
        return self.results
    
    def analyze_hyperfine_coupling_effects(self, coupling_strengths=np.linspace(0, 2, 11), 
                                         total_time=1000.0, n_steps=1000):
        """
        Analyze the effects of hyperfine coupling strength.
        
        Args:
            coupling_strengths (np.ndarray): Hyperfine coupling strengths in mT
            total_time (float): Total simulation time in ns
            n_steps (int): Number of time steps
            
        Returns:
            dict: Analysis results
        """
        # Initialize results
        with_sync_yields = np.zeros(len(coupling_strengths))
        without_sync_yields = np.zeros(len(coupling_strengths))
        
        # Analyze each coupling strength
        for i, strength in enumerate(coupling_strengths):
            # Set hyperfine couplings
            couplings = np.ones(self.simulator.n_nuclear_spins) * strength
            self.simulator.set_hyperfine_couplings(couplings)
            
            # Simulate with phase synchronization
            with_sync_results = self.simulator.simulate_radical_pair_dynamics(
                total_time=total_time,
                n_steps=n_steps,
                use_phase_sync=True
            )
            
            # Simulate without phase synchronization
            without_sync_results = self.simulator.simulate_radical_pair_dynamics(
                total_time=total_time,
                n_steps=n_steps,
                use_phase_sync=False
            )
            
            # Record yields
            with_sync_yields[i] = with_sync_results['singlet_probs'][-1]
            without_sync_yields[i] = without_sync_results['singlet_probs'][-1]
        
        # Store results
        self.results = {
            'coupling_strengths': coupling_strengths,
            'with_sync_yields': with_sync_yields,
            'without_sync_yields': without_sync_yields
        }
        
        return self.results
    
    def analyze_compass_sensitivity(self, inclinations=np.linspace(0, np.pi, 19), 
                                  total_time=1000.0, n_steps=1000):
        """
        Analyze the compass sensitivity with and without phase synchronization.
        
        Args:
            inclinations (np.ndarray): Magnetic field inclinations in radians
            total_time (float): Total simulation time in ns
            n_steps (int): Number of time steps
            
        Returns:
            dict: Analysis results
        """
        # Calculate yield anisotropy with phase synchronization
        with_sync_anisotropy = self.simulator.calculate_yield_anisotropy(
            inclinations=inclinations,
            total_time=total_time,
            n_steps=n_steps,
            use_phase_sync=True
        )
        
        # Calculate yield anisotropy without phase synchronization
        without_sync_anisotropy = self.simulator.calculate_yield_anisotropy(
            inclinations=inclinations,
            total_time=total_time,
            n_steps=n_steps,
            use_phase_sync=False
        )
        
        # Calculate compass precision with phase synchronization
        with_sync_precision = self.simulator.calculate_compass_precision(
            inclinations=inclinations,
            total_time=total_time,
            n_steps=n_steps,
            use_phase_sync=True
        )
        
        # Calculate compass precision without phase synchronization
        without_sync_precision = self.simulator.calculate_compass_precision(
            inclinations=inclinations,
            total_time=total_time,
            n_steps=n_steps,
            use_phase_sync=False
        )
        
        # Store results
        self.results = {
            'with_sync_anisotropy': with_sync_anisotropy,
            'without_sync_anisotropy': without_sync_anisotropy,
            'with_sync_precision': with_sync_precision,
            'without_sync_precision': without_sync_precision
        }
        
        return self.results
    
    def visualize_magnetic_field_effects(self):
        """
        Visualize the effects of magnetic field strength.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if 'field_strengths' not in self.results:
            raise ValueError("Magnetic field effects not analyzed")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot yields vs. field strength
        field_strengths = self.results['field_strengths']
        with_sync_yields = self.results['with_sync_yields']
        without_sync_yields = self.results['without_sync_yields']
        
        ax.plot(field_strengths, with_sync_yields, 'b-', linewidth=2, label='With Phase Sync')
        ax.plot(field_strengths, without_sync_yields, 'r-', linewidth=2, label='Without Phase Sync')
        
        # Set labels
        ax.set_xlabel('Magnetic Field Strength (μT)')
        ax.set_ylabel('Singlet Yield')
        ax.set_title('Effect of Magnetic Field Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_hyperfine_coupling_effects(self):
        """
        Visualize the effects of hyperfine coupling strength.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if 'coupling_strengths' not in self.results:
            raise ValueError("Hyperfine coupling effects not analyzed")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot yields vs. coupling strength
        coupling_strengths = self.results['coupling_strengths']
        with_sync_yields = self.results['with_sync_yields']
        without_sync_yields = self.results['without_sync_yields']
        
        ax.plot(coupling_strengths, with_sync_yields, 'b-', linewidth=2, label='With Phase Sync')
        ax.plot(coupling_strengths, without_sync_yields, 'r-', linewidth=2, label='Without Phase Sync')
        
        # Set labels
        ax.set_xlabel('Hyperfine Coupling Strength (mT)')
        ax.set_ylabel('Singlet Yield')
        ax.set_title('Effect of Hyperfine Coupling Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_compass_sensitivity(self):
        """
        Visualize the compass sensitivity with and without phase synchronization.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if 'with_sync_anisotropy' not in self.results:
            raise ValueError("Compass sensitivity not analyzed")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot yield anisotropy
        inclinations = self.results['with_sync_anisotropy']['inclinations']
        with_sync_yields = self.results['with_sync_anisotropy']['singlet_yields']
        without_sync_yields = self.results['without_sync_anisotropy']['singlet_yields']
        
        ax1.plot(inclinations * 180 / np.pi, with_sync_yields, 'b-', linewidth=2, 
                label=f'With Phase Sync (Anisotropy = {self.results["with_sync_anisotropy"]["anisotropy"]:.4f})')
        ax1.plot(inclinations * 180 / np.pi, without_sync_yields, 'r-', linewidth=2, 
                label=f'Without Phase Sync (Anisotropy = {self.results["without_sync_anisotropy"]["anisotropy"]:.4f})')
        
        # Set labels
        ax1.set_xlabel('Inclination (degrees)')
        ax1.set_ylabel('Singlet Yield')
        ax1.set_title('Yield Anisotropy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot compass precision
        with_sync_derivative = self.results['with_sync_precision']['derivative']
        without_sync_derivative = self.results['without_sync_precision']['derivative']
        
        ax2.plot(inclinations * 180 / np.pi, with_sync_derivative, 'b-', linewidth=2, 
                label=f'With Phase Sync (Precision = {self.results["with_sync_precision"]["precision"]:.4f})')
        ax2.plot(inclinations * 180 / np.pi, without_sync_derivative, 'r-', linewidth=2, 
                label=f'Without Phase Sync (Precision = {self.results["without_sync_precision"]["precision"]:.4f})')
        
        # Set labels
        ax2.set_xlabel('Inclination (degrees)')
        ax2.set_ylabel('dY/dθ')
        ax2.set_title('Compass Precision')
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