"""
TIBEDO Zero-Noise Extrapolation Module

This module implements zero-noise extrapolation techniques for quantum error mitigation
in the TIBEDO Framework. Zero-noise extrapolation works by executing quantum circuits
at different noise levels and extrapolating to the zero-noise limit.

Key components:
1. ZeroNoiseExtrapolator: Base class for zero-noise extrapolation
2. RichardsonExtrapolator: Implements Richardson extrapolation
3. ExponentialExtrapolator: Implements exponential fitting extrapolation
4. PolynomialExtrapolator: Implements polynomial fitting extrapolation
5. CyclotomicExtrapolator: Implements TIBEDO-enhanced extrapolation using cyclotomic fields

The implementation leverages TIBEDO's mathematical structures, particularly cyclotomic
fields, to enhance the accuracy of extrapolation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from scipy.optimize import curve_fit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.providers import Backend
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ZeroNoiseExtrapolator:
    """
    Base class for zero-noise extrapolation.
    
    Zero-noise extrapolation works by executing quantum circuits at different
    noise levels and extrapolating to the zero-noise limit.
    """
    
    def __init__(self, 
                 noise_scaling_method: str = 'gate_stretching',
                 scale_factors: List[float] = None):
        """
        Initialize the zero-noise extrapolator.
        
        Args:
            noise_scaling_method: Method for scaling noise ('gate_stretching', 'pulse_stretching', or 'parameter_scaling')
            scale_factors: List of scale factors to use for extrapolation
        """
        self.noise_scaling_method = noise_scaling_method
        
        # Default scale factors if none provided
        if scale_factors is None:
            self.scale_factors = [1.0, 2.0, 3.0]
        else:
            self.scale_factors = scale_factors
            
        logger.info(f"Initialized zero-noise extrapolator with {noise_scaling_method} method")
        logger.info(f"Scale factors: {self.scale_factors}")
    
    def scale_circuit(self, circuit: QuantumCircuit, scale_factor: float) -> QuantumCircuit:
        """
        Scale the noise in a quantum circuit by the given factor.
        
        Args:
            circuit: Quantum circuit to scale
            scale_factor: Factor by which to scale the noise
            
        Returns:
            Scaled quantum circuit
        """
        if self.noise_scaling_method == 'gate_stretching':
            return self._scale_circuit_gate_stretching(circuit, scale_factor)
        elif self.noise_scaling_method == 'pulse_stretching':
            return self._scale_circuit_pulse_stretching(circuit, scale_factor)
        elif self.noise_scaling_method == 'parameter_scaling':
            return self._scale_circuit_parameter_scaling(circuit, scale_factor)
        else:
            raise ValueError(f"Unknown noise scaling method: {self.noise_scaling_method}")
    
    def _scale_circuit_gate_stretching(self, circuit: QuantumCircuit, scale_factor: float) -> QuantumCircuit:
        """
        Scale the noise in a quantum circuit by repeating gates.
        
        Args:
            circuit: Quantum circuit to scale
            scale_factor: Factor by which to scale the noise
            
        Returns:
            Scaled quantum circuit
        """
        # Create a new circuit with the same qubits and classical bits
        scaled_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        
        # Copy the circuit metadata
        scaled_circuit.name = circuit.name
        
        # Integer part of the scale factor
        integer_factor = int(scale_factor)
        
        # Fractional part of the scale factor
        fractional_factor = scale_factor - integer_factor
        
        # Process each instruction in the original circuit
        for instruction in circuit.data:
            operation = instruction.operation
            qubits = instruction.qubits
            clbits = instruction.clbits
            
            # Skip measurement operations (they should only appear once)
            if operation.name == 'measure':
                scaled_circuit.append(operation, qubits, clbits)
                continue
            
            # Skip barrier operations (they don't introduce noise)
            if operation.name == 'barrier':
                scaled_circuit.append(operation, qubits, clbits)
                continue
            
            # Repeat the gate integer_factor times
            for _ in range(integer_factor):
                scaled_circuit.append(operation, qubits, clbits)
            
            # Apply the gate one more time with probability equal to the fractional part
            if np.random.random() < fractional_factor:
                scaled_circuit.append(operation, qubits, clbits)
        
        return scaled_circuit
    
    def _scale_circuit_pulse_stretching(self, circuit: QuantumCircuit, scale_factor: float) -> QuantumCircuit:
        """
        Scale the noise in a quantum circuit by stretching pulses.
        
        Note: This is a placeholder implementation. Actual pulse stretching
        requires pulse-level control, which is not implemented here.
        
        Args:
            circuit: Quantum circuit to scale
            scale_factor: Factor by which to scale the noise
            
        Returns:
            Scaled quantum circuit
        """
        logger.warning("Pulse stretching is not fully implemented. Using gate stretching instead.")
        return self._scale_circuit_gate_stretching(circuit, scale_factor)
    
    def _scale_circuit_parameter_scaling(self, circuit: QuantumCircuit, scale_factor: float) -> QuantumCircuit:
        """
        Scale the noise in a quantum circuit by scaling noise parameters.
        
        Note: This is a placeholder implementation. Actual parameter scaling
        requires access to the noise model parameters, which is not implemented here.
        
        Args:
            circuit: Quantum circuit to scale
            scale_factor: Factor by which to scale the noise
            
        Returns:
            Scaled quantum circuit
        """
        logger.warning("Parameter scaling is not fully implemented. Using gate stretching instead.")
        return self._scale_circuit_gate_stretching(circuit, scale_factor)
    
    def extrapolate(self, 
                   circuit: QuantumCircuit, 
                   backend: Backend, 
                   shots: int = 1024,
                   observable: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute the circuit at different noise levels and extrapolate to the zero-noise limit.
        
        Args:
            circuit: Quantum circuit to execute
            backend: Backend to execute the circuit on
            shots: Number of shots for each circuit execution
            observable: Function to compute the observable from the counts
            
        Returns:
            Dictionary containing the extrapolation results
        """
        # If no observable function is provided, use the default (expectation value of |0><0|)
        if observable is None:
            observable = self._default_observable
        
        # Execute the circuit at different noise levels
        expectation_values = []
        for scale_factor in self.scale_factors:
            # Scale the circuit
            scaled_circuit = self.scale_circuit(circuit, scale_factor)
            
            # Execute the circuit
            transpiled_circuit = transpile(scaled_circuit, backend)
            job = backend.run(transpiled_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()
            
            # Compute the observable
            expectation_value = observable(counts)
            expectation_values.append(expectation_value)
            
            logger.info(f"Scale factor {scale_factor}: Expectation value = {expectation_value}")
        
        # Extrapolate to the zero-noise limit
        extrapolated_value = self._extrapolate_to_zero(self.scale_factors, expectation_values)
        
        return {
            'scale_factors': self.scale_factors,
            'expectation_values': expectation_values,
            'extrapolated_value': extrapolated_value
        }
    
    def _default_observable(self, counts: Dict[str, int]) -> float:
        """
        Default observable function: expectation value of |0><0|.
        
        Args:
            counts: Counts dictionary from circuit execution
            
        Returns:
            Expectation value of |0><0|
        """
        total_shots = sum(counts.values())
        
        # Count the number of all-zero bitstrings
        zero_state = '0' * len(next(iter(counts.keys())))
        zero_count = counts.get(zero_state, 0)
        
        return zero_count / total_shots
    
    def _extrapolate_to_zero(self, scale_factors: List[float], expectation_values: List[float]) -> float:
        """
        Extrapolate to the zero-noise limit.
        
        This is a placeholder method that should be overridden by subclasses.
        
        Args:
            scale_factors: List of scale factors
            expectation_values: List of expectation values
            
        Returns:
            Extrapolated value at zero noise
        """
        raise NotImplementedError("Extrapolation method not implemented in base class")
    
    def visualize_extrapolation(self, 
                               scale_factors: List[float], 
                               expectation_values: List[float],
                               extrapolated_value: float) -> plt.Figure:
        """
        Visualize the extrapolation.
        
        Args:
            scale_factors: List of scale factors
            expectation_values: List of expectation values
            extrapolated_value: Extrapolated value at zero noise
            
        Returns:
            Matplotlib figure showing the extrapolation
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the data points
        ax.plot(scale_factors, expectation_values, 'o', label='Measured values')
        
        # Plot the extrapolated value
        ax.plot(0, extrapolated_value, 'ro', label='Extrapolated value')
        
        # Plot the extrapolation curve
        x = np.linspace(0, max(scale_factors), 100)
        y = self._extrapolation_curve(x, scale_factors, expectation_values)
        ax.plot(x, y, '--', label='Extrapolation curve')
        
        # Set plot properties
        ax.set_xlabel('Noise scale factor')
        ax.set_ylabel('Expectation value')
        ax.set_title('Zero-Noise Extrapolation')
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def _extrapolation_curve(self, 
                            x: np.ndarray, 
                            scale_factors: List[float], 
                            expectation_values: List[float]) -> np.ndarray:
        """
        Compute the extrapolation curve.
        
        This is a placeholder method that should be overridden by subclasses.
        
        Args:
            x: Points at which to evaluate the curve
            scale_factors: List of scale factors
            expectation_values: List of expectation values
            
        Returns:
            Values of the extrapolation curve at the given points
        """
        raise NotImplementedError("Extrapolation curve not implemented in base class")


class RichardsonExtrapolator(ZeroNoiseExtrapolator):
    """
    Implements Richardson extrapolation for zero-noise extrapolation.
    
    Richardson extrapolation is a technique for improving the accuracy of
    a numerical method by combining results at different step sizes.
    """
    
    def __init__(self, 
                 noise_scaling_method: str = 'gate_stretching',
                 scale_factors: List[float] = None,
                 order: int = 1):
        """
        Initialize the Richardson extrapolator.
        
        Args:
            noise_scaling_method: Method for scaling noise ('gate_stretching', 'pulse_stretching', or 'parameter_scaling')
            scale_factors: List of scale factors to use for extrapolation
            order: Order of the Richardson extrapolation
        """
        super().__init__(noise_scaling_method, scale_factors)
        self.order = order
        
        # Ensure we have enough scale factors for the requested order
        if len(self.scale_factors) < self.order + 1:
            raise ValueError(f"Need at least {self.order + 1} scale factors for order-{self.order} Richardson extrapolation")
        
        logger.info(f"Initialized Richardson extrapolator with order {order}")
    
    def _extrapolate_to_zero(self, scale_factors: List[float], expectation_values: List[float]) -> float:
        """
        Extrapolate to the zero-noise limit using Richardson extrapolation.
        
        Args:
            scale_factors: List of scale factors
            expectation_values: List of expectation values
            
        Returns:
            Extrapolated value at zero noise
        """
        # Ensure we have enough data points
        if len(scale_factors) < self.order + 1:
            raise ValueError(f"Need at least {self.order + 1} data points for order-{self.order} Richardson extrapolation")
        
        # Sort the data points by scale factor
        sorted_data = sorted(zip(scale_factors, expectation_values))
        sorted_scale_factors = [x[0] for x in sorted_data]
        sorted_expectation_values = [x[1] for x in sorted_data]
        
        # Perform Richardson extrapolation
        extrapolated_value = self._richardson_extrapolation(sorted_scale_factors[:self.order + 1], 
                                                           sorted_expectation_values[:self.order + 1])
        
        logger.info(f"Richardson extrapolation result: {extrapolated_value}")
        
        return extrapolated_value
    
    def _richardson_extrapolation(self, scale_factors: List[float], expectation_values: List[float]) -> float:
        """
        Perform Richardson extrapolation.
        
        Args:
            scale_factors: List of scale factors
            expectation_values: List of expectation values
            
        Returns:
            Extrapolated value at zero noise
        """
        # Initialize the tableau for Richardson extrapolation
        tableau = np.zeros((self.order + 1, self.order + 1))
        
        # Fill the first column with the expectation values
        for i in range(self.order + 1):
            tableau[i, 0] = expectation_values[i]
        
        # Fill the rest of the tableau
        for j in range(1, self.order + 1):
            for i in range(self.order + 1 - j):
                # Richardson extrapolation formula
                factor = scale_factors[i + j] / scale_factors[i]
                tableau[i, j] = tableau[i + 1, j - 1] + (tableau[i + 1, j - 1] - tableau[i, j - 1]) / (factor - 1)
        
        # The extrapolated value is in the top-right corner of the tableau
        return tableau[0, self.order]
    
    def _extrapolation_curve(self, 
                            x: np.ndarray, 
                            scale_factors: List[float], 
                            expectation_values: List[float]) -> np.ndarray:
        """
        Compute the extrapolation curve for Richardson extrapolation.
        
        Args:
            x: Points at which to evaluate the curve
            scale_factors: List of scale factors
            expectation_values: List of expectation values
            
        Returns:
            Values of the extrapolation curve at the given points
        """
        # Sort the data points by scale factor
        sorted_data = sorted(zip(scale_factors, expectation_values))
        sorted_scale_factors = [x[0] for x in sorted_data]
        sorted_expectation_values = [x[1] for x in sorted_data]
        
        # Fit a polynomial of degree equal to the order
        coeffs = np.polyfit(sorted_scale_factors, sorted_expectation_values, self.order)
        
        # Evaluate the polynomial at the given points
        return np.polyval(coeffs, x)


class ExponentialExtrapolator(ZeroNoiseExtrapolator):
    """
    Implements exponential fitting extrapolation for zero-noise extrapolation.
    
    Exponential extrapolation fits an exponential function to the data points
    and extrapolates to the zero-noise limit.
    """
    
    def __init__(self, 
                 noise_scaling_method: str = 'gate_stretching',
                 scale_factors: List[float] = None):
        """
        Initialize the exponential extrapolator.
        
        Args:
            noise_scaling_method: Method for scaling noise ('gate_stretching', 'pulse_stretching', or 'parameter_scaling')
            scale_factors: List of scale factors to use for extrapolation
        """
        super().__init__(noise_scaling_method, scale_factors)
        
        # Ensure we have enough scale factors
        if len(self.scale_factors) < 2:
            raise ValueError("Need at least 2 scale factors for exponential extrapolation")
        
        logger.info("Initialized exponential extrapolator")
    
    def _extrapolate_to_zero(self, scale_factors: List[float], expectation_values: List[float]) -> float:
        """
        Extrapolate to the zero-noise limit using exponential fitting.
        
        Args:
            scale_factors: List of scale factors
            expectation_values: List of expectation values
            
        Returns:
            Extrapolated value at zero noise
        """
        # Ensure we have enough data points
        if len(scale_factors) < 2:
            raise ValueError("Need at least 2 data points for exponential extrapolation")
        
        # Define the exponential function to fit
        def exp_func(x, a, b, c):
            return a * np.exp(b * x) + c
        
        # Initial parameter guess
        p0 = [1.0, -1.0, 0.0]
        
        try:
            # Fit the exponential function to the data
            popt, _ = curve_fit(exp_func, scale_factors, expectation_values, p0=p0)
            
            # Extrapolate to zero noise
            extrapolated_value = exp_func(0, *popt)
            
            logger.info(f"Exponential extrapolation result: {extrapolated_value}")
            logger.info(f"Fitted parameters: a={popt[0]}, b={popt[1]}, c={popt[2]}")
            
            return extrapolated_value
        
        except RuntimeError as e:
            logger.error(f"Exponential fitting failed: {e}")
            logger.warning("Falling back to linear extrapolation")
            
            # Fallback to linear extrapolation
            coeffs = np.polyfit(scale_factors, expectation_values, 1)
            extrapolated_value = np.polyval(coeffs, 0)
            
            logger.info(f"Linear extrapolation result: {extrapolated_value}")
            
            return extrapolated_value
    
    def _extrapolation_curve(self, 
                            x: np.ndarray, 
                            scale_factors: List[float], 
                            expectation_values: List[float]) -> np.ndarray:
        """
        Compute the extrapolation curve for exponential extrapolation.
        
        Args:
            x: Points at which to evaluate the curve
            scale_factors: List of scale factors
            expectation_values: List of expectation values
            
        Returns:
            Values of the extrapolation curve at the given points
        """
        # Define the exponential function to fit
        def exp_func(x, a, b, c):
            return a * np.exp(b * x) + c
        
        # Initial parameter guess
        p0 = [1.0, -1.0, 0.0]
        
        try:
            # Fit the exponential function to the data
            popt, _ = curve_fit(exp_func, scale_factors, expectation_values, p0=p0)
            
            # Evaluate the function at the given points
            return exp_func(x, *popt)
        
        except RuntimeError:
            # Fallback to linear extrapolation
            coeffs = np.polyfit(scale_factors, expectation_values, 1)
            return np.polyval(coeffs, x)


class PolynomialExtrapolator(ZeroNoiseExtrapolator):
    """
    Implements polynomial fitting extrapolation for zero-noise extrapolation.
    
    Polynomial extrapolation fits a polynomial function to the data points
    and extrapolates to the zero-noise limit.
    """
    
    def __init__(self, 
                 noise_scaling_method: str = 'gate_stretching',
                 scale_factors: List[float] = None,
                 degree: int = 2):
        """
        Initialize the polynomial extrapolator.
        
        Args:
            noise_scaling_method: Method for scaling noise ('gate_stretching', 'pulse_stretching', or 'parameter_scaling')
            scale_factors: List of scale factors to use for extrapolation
            degree: Degree of the polynomial to fit
        """
        super().__init__(noise_scaling_method, scale_factors)
        self.degree = degree
        
        # Ensure we have enough scale factors for the requested degree
        if len(self.scale_factors) < self.degree + 1:
            raise ValueError(f"Need at least {self.degree + 1} scale factors for degree-{self.degree} polynomial extrapolation")
        
        logger.info(f"Initialized polynomial extrapolator with degree {degree}")
    
    def _extrapolate_to_zero(self, scale_factors: List[float], expectation_values: List[float]) -> float:
        """
        Extrapolate to the zero-noise limit using polynomial fitting.
        
        Args:
            scale_factors: List of scale factors
            expectation_values: List of expectation values
            
        Returns:
            Extrapolated value at zero noise
        """
        # Ensure we have enough data points
        if len(scale_factors) < self.degree + 1:
            raise ValueError(f"Need at least {self.degree + 1} data points for degree-{self.degree} polynomial extrapolation")
        
        # Fit a polynomial of the specified degree
        coeffs = np.polyfit(scale_factors, expectation_values, self.degree)
        
        # Extrapolate to zero noise
        extrapolated_value = np.polyval(coeffs, 0)
        
        logger.info(f"Polynomial extrapolation result: {extrapolated_value}")
        logger.info(f"Fitted coefficients: {coeffs}")
        
        return extrapolated_value
    
    def _extrapolation_curve(self, 
                            x: np.ndarray, 
                            scale_factors: List[float], 
                            expectation_values: List[float]) -> np.ndarray:
        """
        Compute the extrapolation curve for polynomial extrapolation.
        
        Args:
            x: Points at which to evaluate the curve
            scale_factors: List of scale factors
            expectation_values: List of expectation values
            
        Returns:
            Values of the extrapolation curve at the given points
        """
        # Fit a polynomial of the specified degree
        coeffs = np.polyfit(scale_factors, expectation_values, self.degree)
        
        # Evaluate the polynomial at the given points
        return np.polyval(coeffs, x)


class CyclotomicExtrapolator(ZeroNoiseExtrapolator):
    """
    Implements TIBEDO-enhanced extrapolation using cyclotomic fields.
    
    This extrapolator leverages TIBEDO's cyclotomic field theory to enhance
    the accuracy of extrapolation, particularly for quantum circuits with
    complex error patterns.
    """
    
    def __init__(self, 
                 noise_scaling_method: str = 'gate_stretching',
                 scale_factors: List[float] = None,
                 cyclotomic_conductor: int = 168,
                 use_prime_indexing: bool = True):
        """
        Initialize the cyclotomic extrapolator.
        
        Args:
            noise_scaling_method: Method for scaling noise ('gate_stretching', 'pulse_stretching', or 'parameter_scaling')
            scale_factors: List of scale factors to use for extrapolation
            cyclotomic_conductor: Conductor for the cyclotomic field
            use_prime_indexing: Whether to use prime-indexed optimization
        """
        super().__init__(noise_scaling_method, scale_factors)
        self.cyclotomic_conductor = cyclotomic_conductor
        self.use_prime_indexing = use_prime_indexing
        
        # Ensure we have enough scale factors
        if len(self.scale_factors) < 3:
            raise ValueError("Need at least 3 scale factors for cyclotomic extrapolation")
        
        # Initialize cyclotomic field structures
        self._initialize_cyclotomic_structures()
        
        logger.info(f"Initialized cyclotomic extrapolator with conductor {cyclotomic_conductor}")
        logger.info(f"Using prime-indexed optimization: {use_prime_indexing}")
    
    def _initialize_cyclotomic_structures(self):
        """Initialize cyclotomic field structures for enhanced extrapolation."""
        # TODO: Implement cyclotomic field structures
        # This is a placeholder for future implementation
        pass
    
    def _extrapolate_to_zero(self, scale_factors: List[float], expectation_values: List[float]) -> float:
        """
        Extrapolate to the zero-noise limit using cyclotomic field theory.
        
        Args:
            scale_factors: List of scale factors
            expectation_values: List of expectation values
            
        Returns:
            Extrapolated value at zero noise
        """
        # Ensure we have enough data points
        if len(scale_factors) < 3:
            raise ValueError("Need at least 3 data points for cyclotomic extrapolation")
        
        # TODO: Implement cyclotomic field-based extrapolation
        # This is a placeholder implementation that uses a combination of
        # Richardson extrapolation and polynomial fitting
        
        # Use Richardson extrapolation for the first estimate
        richardson = RichardsonExtrapolator(self.noise_scaling_method, scale_factors, order=1)
        richardson_value = richardson._extrapolate_to_zero(scale_factors, expectation_values)
        
        # Use polynomial fitting for the second estimate
        poly = PolynomialExtrapolator(self.noise_scaling_method, scale_factors, degree=2)
        poly_value = poly._extrapolate_to_zero(scale_factors, expectation_values)
        
        # Combine the estimates using a weighted average
        # In a real implementation, the weights would be determined by cyclotomic field theory
        weight_richardson = 0.6
        weight_poly = 0.4
        extrapolated_value = weight_richardson * richardson_value + weight_poly * poly_value
        
        logger.info(f"Cyclotomic extrapolation result: {extrapolated_value}")
        logger.info(f"Richardson value: {richardson_value}, Polynomial value: {poly_value}")
        
        return extrapolated_value
    
    def _extrapolation_curve(self, 
                            x: np.ndarray, 
                            scale_factors: List[float], 
                            expectation_values: List[float]) -> np.ndarray:
        """
        Compute the extrapolation curve for cyclotomic extrapolation.
        
        Args:
            x: Points at which to evaluate the curve
            scale_factors: List of scale factors
            expectation_values: List of expectation values
            
        Returns:
            Values of the extrapolation curve at the given points
        """
        # TODO: Implement cyclotomic field-based extrapolation curve
        # This is a placeholder implementation that uses a combination of
        # Richardson extrapolation and polynomial fitting
        
        # Use Richardson extrapolation for the first curve
        richardson = RichardsonExtrapolator(self.noise_scaling_method, scale_factors, order=1)
        richardson_curve = richardson._extrapolation_curve(x, scale_factors, expectation_values)
        
        # Use polynomial fitting for the second curve
        poly = PolynomialExtrapolator(self.noise_scaling_method, scale_factors, degree=2)
        poly_curve = poly._extrapolation_curve(x, scale_factors, expectation_values)
        
        # Combine the curves using a weighted average
        # In a real implementation, the weights would be determined by cyclotomic field theory
        weight_richardson = 0.6
        weight_poly = 0.4
        return weight_richardson * richardson_curve + weight_poly * poly_curve


# Example usage
if __name__ == "__main__":
    # Create a simple quantum circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    
    # Create a simulator backend with noise
    simulator = Aer.get_backend('qasm_simulator')
    
    # Create a Richardson extrapolator
    extrapolator = RichardsonExtrapolator(
        noise_scaling_method='gate_stretching',
        scale_factors=[1.0, 2.0, 3.0],
        order=1
    )
    
    # Extrapolate to the zero-noise limit
    results = extrapolator.extrapolate(circuit, simulator, shots=1024)
    
    # Print the results
    print(f"Scale factors: {results['scale_factors']}")
    print(f"Expectation values: {results['expectation_values']}")
    print(f"Extrapolated value: {results['extrapolated_value']}")
    
    # Visualize the extrapolation
    fig = extrapolator.visualize_extrapolation(
        results['scale_factors'],
        results['expectation_values'],
        results['extrapolated_value']
    )
    plt.savefig('zero_noise_extrapolation.png')
    
    # Try other extrapolators
    extrapolators = [
        ExponentialExtrapolator(scale_factors=[1.0, 2.0, 3.0]),
        PolynomialExtrapolator(scale_factors=[1.0, 2.0, 3.0], degree=2),
        CyclotomicExtrapolator(scale_factors=[1.0, 2.0, 3.0])
    ]
    
    for i, extrapolator in enumerate(extrapolators):
        results = extrapolator.extrapolate(circuit, simulator, shots=1024)
        print(f"\nExtrapolator {i+1}: {extrapolator.__class__.__name__}")
        print(f"Extrapolated value: {results['extrapolated_value']}")