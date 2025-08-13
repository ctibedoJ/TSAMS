"""
TIBEDO Zero-Noise Extrapolation Demonstration

This script demonstrates the use of the TIBEDO zero-noise extrapolation
implementation, including different extrapolation methods, noise scaling
techniques, and visualization of results.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeMontreal
import os
import sys
import time
import logging

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the zero-noise extrapolation module
from tibedo.quantum_information_new.zero_noise_extrapolation import (
    RichardsonExtrapolator,
    ExponentialExtrapolator,
    PolynomialExtrapolator,
    CyclotomicExtrapolator
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_ghz_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Create a GHZ state preparation circuit.
    
    Args:
        num_qubits: Number of qubits in the GHZ state
        
    Returns:
        Quantum circuit for GHZ state preparation
    """
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    # Apply Hadamard to the first qubit
    circuit.h(0)
    
    # Apply CNOT gates to entangle all qubits
    for i in range(num_qubits - 1):
        circuit.cx(i, i + 1)
    
    # Measure all qubits
    circuit.measure(range(num_qubits), range(num_qubits))
    
    return circuit

def create_quantum_fourier_transform_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Create a Quantum Fourier Transform (QFT) circuit.
    
    Args:
        num_qubits: Number of qubits in the QFT
        
    Returns:
        Quantum circuit for QFT
    """
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    # Initialize all qubits to |1>
    circuit.x(range(num_qubits))
    
    # Apply QFT
    for i in range(num_qubits):
        circuit.h(i)
        for j in range(i + 1, num_qubits):
            circuit.cp(2 * np.pi / 2**(j - i + 1), i, j)
    
    # Swap qubits (optional, for standard QFT ordering)
    for i in range(num_qubits // 2):
        circuit.swap(i, num_qubits - i - 1)
    
    # Measure all qubits
    circuit.measure(range(num_qubits), range(num_qubits))
    
    return circuit

def create_bernstein_vazirani_circuit(num_qubits: int, secret_string: str) -> QuantumCircuit:
    """
    Create a Bernstein-Vazirani circuit for finding a secret string.
    
    Args:
        num_qubits: Number of qubits (excluding the ancilla qubit)
        secret_string: Binary string to encode (e.g., '101')
        
    Returns:
        Quantum circuit for the Bernstein-Vazirani algorithm
    """
    # Ensure the secret string has the correct length
    if len(secret_string) != num_qubits:
        secret_string = secret_string.zfill(num_qubits)[:num_qubits]
    
    # Create a circuit with num_qubits + 1 qubits (including the ancilla)
    circuit = QuantumCircuit(num_qubits + 1, num_qubits)
    
    # Initialize the ancilla qubit to |1>
    circuit.x(num_qubits)
    
    # Apply Hadamard gates to all qubits
    circuit.h(range(num_qubits + 1))
    
    # Apply the oracle (controlled-Z gates for each '1' in the secret string)
    for i in range(num_qubits):
        if secret_string[i] == '1':
            circuit.cz(i, num_qubits)
    
    # Apply Hadamard gates to the data qubits
    circuit.h(range(num_qubits))
    
    # Measure the data qubits
    circuit.measure(range(num_qubits), range(num_qubits))
    
    return circuit

def get_noisy_simulator(error_probability: float = 0.01) -> Aer.get_backend:
    """
    Create a noisy simulator with the specified error probability.
    
    Args:
        error_probability: Probability of errors
        
    Returns:
        Noisy simulator backend
    """
    # Get a device noise model
    device = FakeMontreal()
    noise_model = NoiseModel.from_backend(device)
    
    # Scale the noise by the specified error probability
    noise_model.scale_noise(error_probability / 0.01)
    
    # Create a noisy simulator
    simulator = Aer.get_backend('qasm_simulator')
    simulator.set_options(noise_model=noise_model)
    
    return simulator

def ghz_state_observable(counts: dict) -> float:
    """
    Compute the GHZ state observable: |00...0><00...0| + |11...1><11...1|
    
    Args:
        counts: Counts dictionary from circuit execution
        
    Returns:
        Expectation value of the GHZ state observable
    """
    total_shots = sum(counts.values())
    
    # Count the number of all-zero and all-one bitstrings
    num_qubits = len(next(iter(counts.keys())))
    zero_state = '0' * num_qubits
    one_state = '1' * num_qubits
    
    zero_count = counts.get(zero_state, 0)
    one_count = counts.get(one_state, 0)
    
    return (zero_count + one_count) / total_shots

def demonstrate_extrapolation_methods():
    """
    Demonstrate different extrapolation methods on a GHZ state preparation circuit.
    """
    logger.info("Demonstrating different extrapolation methods")
    
    # Create a GHZ state preparation circuit
    num_qubits = 3
    circuit = create_ghz_circuit(num_qubits)
    
    # Create a noisy simulator
    simulator = get_noisy_simulator(error_probability=0.02)
    
    # Define the extrapolators to test
    extrapolators = [
        RichardsonExtrapolator(scale_factors=[1.0, 2.0, 3.0], order=1),
        ExponentialExtrapolator(scale_factors=[1.0, 2.0, 3.0]),
        PolynomialExtrapolator(scale_factors=[1.0, 2.0, 3.0], degree=2),
        CyclotomicExtrapolator(scale_factors=[1.0, 2.0, 3.0])
    ]
    
    # Create a figure for the results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Run each extrapolator
    for i, extrapolator in enumerate(extrapolators):
        logger.info(f"Running {extrapolator.__class__.__name__}")
        
        # Extrapolate to the zero-noise limit
        start_time = time.time()
        results = extrapolator.extrapolate(
            circuit, 
            simulator, 
            shots=8192, 
            observable=ghz_state_observable
        )
        end_time = time.time()
        
        logger.info(f"Extrapolation time: {end_time - start_time:.3f} seconds")
        logger.info(f"Scale factors: {results['scale_factors']}")
        logger.info(f"Expectation values: {results['expectation_values']}")
        logger.info(f"Extrapolated value: {results['extrapolated_value']}")
        
        # Plot the results
        ax = axes[i]
        ax.plot(results['scale_factors'], results['expectation_values'], 'o', label='Measured values')
        ax.plot(0, results['extrapolated_value'], 'ro', label='Extrapolated value')
        
        # Plot the extrapolation curve
        x = np.linspace(0, max(results['scale_factors']), 100)
        y = extrapolator._extrapolation_curve(x, results['scale_factors'], results['expectation_values'])
        ax.plot(x, y, '--', label='Extrapolation curve')
        
        # Set plot properties
        ax.set_xlabel('Noise scale factor')
        ax.set_ylabel('GHZ state fidelity')
        ax.set_title(f"{extrapolator.__class__.__name__}\nExtrapolated value: {results['extrapolated_value']:.4f}")
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('extrapolation_methods_comparison.png')
    logger.info("Saved extrapolation methods comparison to 'extrapolation_methods_comparison.png'")
    
    return fig

def demonstrate_noise_scaling_methods():
    """
    Demonstrate different noise scaling methods on a GHZ state preparation circuit.
    """
    logger.info("Demonstrating different noise scaling methods")
    
    # Create a GHZ state preparation circuit
    num_qubits = 3
    circuit = create_ghz_circuit(num_qubits)
    
    # Create a noisy simulator
    simulator = get_noisy_simulator(error_probability=0.02)
    
    # Define the noise scaling methods to test
    noise_scaling_methods = ['gate_stretching', 'pulse_stretching', 'parameter_scaling']
    
    # Create a figure for the results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Run each noise scaling method
    for i, method in enumerate(noise_scaling_methods):
        logger.info(f"Running {method}")
        
        # Create an extrapolator with the specified noise scaling method
        extrapolator = RichardsonExtrapolator(
            noise_scaling_method=method,
            scale_factors=[1.0, 2.0, 3.0],
            order=1
        )
        
        # Extrapolate to the zero-noise limit
        start_time = time.time()
        results = extrapolator.extrapolate(
            circuit, 
            simulator, 
            shots=8192, 
            observable=ghz_state_observable
        )
        end_time = time.time()
        
        logger.info(f"Extrapolation time: {end_time - start_time:.3f} seconds")
        logger.info(f"Scale factors: {results['scale_factors']}")
        logger.info(f"Expectation values: {results['expectation_values']}")
        logger.info(f"Extrapolated value: {results['extrapolated_value']}")
        
        # Plot the results
        ax = axes[i]
        ax.plot(results['scale_factors'], results['expectation_values'], 'o', label='Measured values')
        ax.plot(0, results['extrapolated_value'], 'ro', label='Extrapolated value')
        
        # Plot the extrapolation curve
        x = np.linspace(0, max(results['scale_factors']), 100)
        y = extrapolator._extrapolation_curve(x, results['scale_factors'], results['expectation_values'])
        ax.plot(x, y, '--', label='Extrapolation curve')
        
        # Set plot properties
        ax.set_xlabel('Noise scale factor')
        ax.set_ylabel('GHZ state fidelity')
        ax.set_title(f"{method}\nExtrapolated value: {results['extrapolated_value']:.4f}")
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('noise_scaling_methods_comparison.png')
    logger.info("Saved noise scaling methods comparison to 'noise_scaling_methods_comparison.png'")
    
    return fig

def demonstrate_circuit_complexity():
    """
    Demonstrate the effect of circuit complexity on extrapolation performance.
    """
    logger.info("Demonstrating the effect of circuit complexity on extrapolation performance")
    
    # Create circuits with different complexity
    circuits = [
        create_ghz_circuit(3),
        create_quantum_fourier_transform_circuit(3),
        create_bernstein_vazirani_circuit(3, '101')
    ]
    circuit_names = ['GHZ', 'QFT', 'Bernstein-Vazirani']
    
    # Create a noisy simulator
    simulator = get_noisy_simulator(error_probability=0.02)
    
    # Create an extrapolator
    extrapolator = RichardsonExtrapolator(
        scale_factors=[1.0, 2.0, 3.0],
        order=1
    )
    
    # Create a figure for the results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Run each circuit
    for i, (circuit, name) in enumerate(zip(circuits, circuit_names)):
        logger.info(f"Running {name} circuit")
        
        # Extrapolate to the zero-noise limit
        start_time = time.time()
        results = extrapolator.extrapolate(
            circuit, 
            simulator, 
            shots=8192
        )
        end_time = time.time()
        
        logger.info(f"Extrapolation time: {end_time - start_time:.3f} seconds")
        logger.info(f"Scale factors: {results['scale_factors']}")
        logger.info(f"Expectation values: {results['expectation_values']}")
        logger.info(f"Extrapolated value: {results['extrapolated_value']}")
        
        # Plot the results
        ax = axes[i]
        ax.plot(results['scale_factors'], results['expectation_values'], 'o', label='Measured values')
        ax.plot(0, results['extrapolated_value'], 'ro', label='Extrapolated value')
        
        # Plot the extrapolation curve
        x = np.linspace(0, max(results['scale_factors']), 100)
        y = extrapolator._extrapolation_curve(x, results['scale_factors'], results['expectation_values'])
        ax.plot(x, y, '--', label='Extrapolation curve')
        
        # Set plot properties
        ax.set_xlabel('Noise scale factor')
        ax.set_ylabel('Expectation value')
        ax.set_title(f"{name}\nExtrapolated value: {results['extrapolated_value']:.4f}")
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('circuit_complexity_comparison.png')
    logger.info("Saved circuit complexity comparison to 'circuit_complexity_comparison.png'")
    
    return fig

def demonstrate_noise_levels():
    """
    Demonstrate the effect of noise levels on extrapolation performance.
    """
    logger.info("Demonstrating the effect of noise levels on extrapolation performance")
    
    # Create a GHZ state preparation circuit
    num_qubits = 3
    circuit = create_ghz_circuit(num_qubits)
    
    # Define the noise levels to test
    noise_levels = [0.01, 0.02, 0.05]
    
    # Create a figure for the results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Run each noise level
    for i, noise_level in enumerate(noise_levels):
        logger.info(f"Running noise level {noise_level}")
        
        # Create a noisy simulator
        simulator = get_noisy_simulator(error_probability=noise_level)
        
        # Create an extrapolator
        extrapolator = RichardsonExtrapolator(
            scale_factors=[1.0, 2.0, 3.0],
            order=1
        )
        
        # Extrapolate to the zero-noise limit
        start_time = time.time()
        results = extrapolator.extrapolate(
            circuit, 
            simulator, 
            shots=8192, 
            observable=ghz_state_observable
        )
        end_time = time.time()
        
        logger.info(f"Extrapolation time: {end_time - start_time:.3f} seconds")
        logger.info(f"Scale factors: {results['scale_factors']}")
        logger.info(f"Expectation values: {results['expectation_values']}")
        logger.info(f"Extrapolated value: {results['extrapolated_value']}")
        
        # Plot the results
        ax = axes[i]
        ax.plot(results['scale_factors'], results['expectation_values'], 'o', label='Measured values')
        ax.plot(0, results['extrapolated_value'], 'ro', label='Extrapolated value')
        
        # Plot the extrapolation curve
        x = np.linspace(0, max(results['scale_factors']), 100)
        y = extrapolator._extrapolation_curve(x, results['scale_factors'], results['expectation_values'])
        ax.plot(x, y, '--', label='Extrapolation curve')
        
        # Set plot properties
        ax.set_xlabel('Noise scale factor')
        ax.set_ylabel('GHZ state fidelity')
        ax.set_title(f"Noise level: {noise_level}\nExtrapolated value: {results['extrapolated_value']:.4f}")
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('noise_levels_comparison.png')
    logger.info("Saved noise levels comparison to 'noise_levels_comparison.png'")
    
    return fig

def run_comprehensive_benchmark():
    """
    Run a comprehensive benchmark of the zero-noise extrapolation methods.
    """
    logger.info("Running comprehensive benchmark")
    
    # Define the parameters to test
    num_qubits_list = [2, 3, 4]
    noise_levels = [0.01, 0.02, 0.05]
    extrapolators = [
        RichardsonExtrapolator(scale_factors=[1.0, 2.0, 3.0], order=1),
        ExponentialExtrapolator(scale_factors=[1.0, 2.0, 3.0]),
        PolynomialExtrapolator(scale_factors=[1.0, 2.0, 3.0], degree=2),
        CyclotomicExtrapolator(scale_factors=[1.0, 2.0, 3.0])
    ]
    extrapolator_names = [
        'Richardson',
        'Exponential',
        'Polynomial',
        'Cyclotomic'
    ]
    
    # Create a figure for the results
    fig, axes = plt.subplots(len(extrapolators), len(noise_levels), figsize=(15, 15))
    
    # Run the benchmark
    for i, extrapolator in enumerate(extrapolators):
        for j, noise_level in enumerate(noise_levels):
            logger.info(f"Running {extrapolator.__class__.__name__} with noise level {noise_level}")
            
            # Create a noisy simulator
            simulator = get_noisy_simulator(error_probability=noise_level)
            
            # Initialize lists to store results
            extrapolated_values = []
            true_values = []
            
            # Run for each number of qubits
            for num_qubits in num_qubits_list:
                # Create a GHZ state preparation circuit
                circuit = create_ghz_circuit(num_qubits)
                
                # Extrapolate to the zero-noise limit
                results = extrapolator.extrapolate(
                    circuit, 
                    simulator, 
                    shots=8192, 
                    observable=ghz_state_observable
                )
                
                # Store the results
                extrapolated_values.append(results['extrapolated_value'])
                true_values.append(1.0)  # The ideal GHZ state fidelity is 1.0
            
            # Plot the results
            ax = axes[i, j]
            ax.plot(num_qubits_list, extrapolated_values, 'o-', label='Extrapolated')
            ax.plot(num_qubits_list, true_values, 's-', label='Ideal')
            
            # Set plot properties
            ax.set_xlabel('Number of qubits')
            ax.set_ylabel('GHZ state fidelity')
            ax.set_title(f"{extrapolator_names[i]}, Noise: {noise_level}")
            ax.grid(True)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('comprehensive_benchmark.png')
    logger.info("Saved comprehensive benchmark to 'comprehensive_benchmark.png'")
    
    return fig

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Demonstrate different extrapolation methods
    extrapolation_fig = demonstrate_extrapolation_methods()
    
    # Demonstrate different noise scaling methods
    noise_scaling_fig = demonstrate_noise_scaling_methods()
    
    # Demonstrate the effect of circuit complexity
    complexity_fig = demonstrate_circuit_complexity()
    
    # Demonstrate the effect of noise levels
    noise_levels_fig = demonstrate_noise_levels()
    
    # Run comprehensive benchmark
    benchmark_fig = run_comprehensive_benchmark()
    
    logger.info("Zero-noise extrapolation demonstration completed")