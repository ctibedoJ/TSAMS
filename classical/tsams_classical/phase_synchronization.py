"""
Phase Synchronization implementation for classical quantum formalism.
"""

import numpy as np

class PhaseSynchronization:
    """
    Implementation of Phase Synchronization for classical quantum computations.
    """
    
    def __init__(self, num_oscillators: int = 10, coupling_strength: float = 0.1):
        """
        Initialize a Phase Synchronization system.
        
        Args:
            num_oscillators: Number of oscillators in the system
            coupling_strength: Strength of coupling between oscillators
        """
        self.num_oscillators = num_oscillators
        self.coupling_strength = coupling_strength
        self.natural_frequencies = np.random.normal(1.0, 0.1, num_oscillators)
        self.phases = np.random.uniform(0, 2*np.pi, num_oscillators)
    
    def update(self, dt: float = 0.1, steps: int = 1):
        """
        Update the phases of the oscillators.
        
        Args:
            dt: Time step
            steps: Number of steps to simulate
            
        Returns:
            Updated phases
        """
        for _ in range(steps):
            # Compute the mean field
            mean_field = np.mean(np.exp(1j * self.phases))
            mean_phase = np.angle(mean_field)
            mean_amplitude = np.abs(mean_field)
            
            # Update each oscillator's phase
            for i in range(self.num_oscillators):
                self.phases[i] += dt * (
                    self.natural_frequencies[i] + 
                    self.coupling_strength * mean_amplitude * np.sin(mean_phase - self.phases[i])
                )
            
            # Wrap phases to [0, 2π]
            self.phases = self.phases % (2 * np.pi)
        
        return self.phases
    
    def order_parameter(self) -> float:
        """
        Compute the Kuramoto order parameter.
        
        Returns:
            Order parameter (0 to 1, where 1 is perfect synchronization)
        """
        mean_field = np.mean(np.exp(1j * self.phases))
        return np.abs(mean_field)
    
    def reset(self, random_seed: int = None):
        """
        Reset the oscillator phases.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.phases = np.random.uniform(0, 2*np.pi, self.num_oscillators)