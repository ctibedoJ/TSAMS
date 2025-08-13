"""
TIBEDO Quantum Random Number Generator

This module implements quantum random number generation services using both
hardware-based and software-based approaches, with statistical tests for
verifying the quality of the generated random numbers.
"""

import numpy as np
import sympy as sp
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import math
import os
import sys
import logging
import time
import secrets
import hashlib
import random
import matplotlib.pyplot as plt
from scipy import stats
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import plot_histogram

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuantumRandomnessSource:
    """
    Abstract base class for quantum randomness sources.
    
    This class defines the interface for quantum randomness sources, which can be
    hardware-based or software-based.
    """
    
    def __init__(self, name: str):
        """
        Initialize the quantum randomness source.
        
        Args:
            name: The name of the randomness source
        """
        self.name = name
    
    def generate_random_bits(self, num_bits: int) -> np.ndarray:
        """
        Generate random bits using the quantum randomness source.
        
        Args:
            num_bits: The number of random bits to generate
            
        Returns:
            An array of random bits (0 or 1)
        """
        raise NotImplementedError("Subclasses must implement generate_random_bits")
    
    def generate_random_bytes(self, num_bytes: int) -> bytes:
        """
        Generate random bytes using the quantum randomness source.
        
        Args:
            num_bytes: The number of random bytes to generate
            
        Returns:
            Random bytes
        """
        # Generate random bits
        bits = self.generate_random_bits(num_bytes * 8)
        
        # Convert bits to bytes
        bytes_list = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte |= (bits[i + j] << (7 - j))
            bytes_list.append(byte)
        
        return bytes(bytes_list)
    
    def generate_random_int(self, min_value: int, max_value: int) -> int:
        """
        Generate a random integer in the range [min_value, max_value].
        
        Args:
            min_value: The minimum value (inclusive)
            max_value: The maximum value (inclusive)
            
        Returns:
            A random integer
        """
        # Compute the number of bits needed
        range_size = max_value - min_value + 1
        num_bits = (range_size - 1).bit_length()
        
        # Generate random bits
        while True:
            bits = self.generate_random_bits(num_bits)
            
            # Convert bits to an integer
            value = 0
            for i, bit in enumerate(bits):
                value |= (bit << i)
            
            # Check if the value is in range
            if value < range_size:
                return min_value + value
    
    def generate_random_float(self) -> float:
        """
        Generate a random float in the range [0, 1).
        
        Returns:
            A random float
        """
        # Generate 53 random bits (the precision of a double)
        bits = self.generate_random_bits(53)
        
        # Convert bits to a float
        value = 0.0
        for i, bit in enumerate(bits):
            value += bit * (2 ** -(i + 1))
        
        return value


class HardwareQuantumRandomnessSource(QuantumRandomnessSource):
    """
    Hardware-based quantum randomness source.
    
    This class implements a quantum randomness source that uses quantum hardware
    to generate random numbers.
    """
    
    def __init__(self, device_type: str = 'simulator'):
        """
        Initialize the hardware-based quantum randomness source.
        
        Args:
            device_type: The type of quantum device to use ('simulator' or 'real')
        """
        super().__init__(f"Hardware-{device_type}")
        self.device_type = device_type
        
        # Initialize the quantum device
        if device_type == 'simulator':
            self.backend = Aer.get_backend('qasm_simulator')
        elif device_type == 'real':
            # In a real implementation, we would connect to a real quantum device
            # For now, we'll just use the simulator
            logger.warning("Real quantum hardware not available, using simulator instead")
            self.backend = Aer.get_backend('qasm_simulator')
        else:
            raise ValueError(f"Unsupported device type: {device_type}")
    
    def generate_random_bits(self, num_bits: int) -> np.ndarray:
        """
        Generate random bits using the quantum hardware.
        
        Args:
            num_bits: The number of random bits to generate
            
        Returns:
            An array of random bits (0 or 1)
        """
        # Create a quantum circuit with num_bits qubits
        qr = QuantumRegister(num_bits, 'q')
        cr = ClassicalRegister(num_bits, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        # Apply Hadamard gates to all qubits
        for i in range(num_bits):
            circuit.h(qr[i])
        
        # Measure all qubits
        circuit.measure(qr, cr)
        
        # Execute the circuit
        job = execute(circuit, self.backend, shots=1)
        result = job.result()
        
        # Get the measurement result
        counts = result.get_counts(circuit)
        
        # There should be only one result
        if len(counts) != 1:
            raise RuntimeError(f"Expected 1 result, got {len(counts)}")
        
        # Convert the result to an array of bits
        bitstring = list(counts.keys())[0]
        bits = np.array([int(bit) for bit in bitstring], dtype=np.int8)
        
        return bits


class SoftwareQuantumRandomnessSource(QuantumRandomnessSource):
    """
    Software-based quantum randomness source.
    
    This class implements a quantum randomness source that uses quantum-inspired
    algorithms to generate random numbers.
    """
    
    def __init__(self, algorithm: str = 'chaotic'):
        """
        Initialize the software-based quantum randomness source.
        
        Args:
            algorithm: The algorithm to use ('chaotic', 'quantum_walk', or 'hybrid')
        """
        super().__init__(f"Software-{algorithm}")
        self.algorithm = algorithm
        
        # Initialize the algorithm
        if algorithm == 'chaotic':
            self.generate_func = self._generate_chaotic
        elif algorithm == 'quantum_walk':
            self.generate_func = self._generate_quantum_walk
        elif algorithm == 'hybrid':
            self.generate_func = self._generate_hybrid
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Initialize the state
        self.state = np.random.randint(0, 2**32)
    
    def generate_random_bits(self, num_bits: int) -> np.ndarray:
        """
        Generate random bits using the quantum-inspired algorithm.
        
        Args:
            num_bits: The number of random bits to generate
            
        Returns:
            An array of random bits (0 or 1)
        """
        return self.generate_func(num_bits)
    
    def _generate_chaotic(self, num_bits: int) -> np.ndarray:
        """
        Generate random bits using a chaotic map.
        
        Args:
            num_bits: The number of random bits to generate
            
        Returns:
            An array of random bits (0 or 1)
        """
        # Use the logistic map: x_{n+1} = r * x_n * (1 - x_n)
        r = 3.99  # Chaotic regime
        
        # Initialize the state
        x = self.state / 2**32
        
        # Generate random bits
        bits = np.zeros(num_bits, dtype=np.int8)
        
        for i in range(num_bits):
            # Iterate the map
            x = r * x * (1 - x)
            
            # Extract a bit
            bits[i] = 1 if x > 0.5 else 0
        
        # Update the state
        self.state = int(x * 2**32)
        
        return bits
    
    def _generate_quantum_walk(self, num_bits: int) -> np.ndarray:
        """
        Generate random bits using a quantum walk.
        
        Args:
            num_bits: The number of random bits to generate
            
        Returns:
            An array of random bits (0 or 1)
        """
        # Use a quantum walk on a line
        n = 100  # Size of the line
        
        # Initialize the state
        position = self.state % n
        
        # Generate random bits
        bits = np.zeros(num_bits, dtype=np.int8)
        
        for i in range(num_bits):
            # Perform a quantum walk step
            # In a real quantum walk, we would use quantum superposition
            # For now, we'll just use a classical random walk with quantum-inspired probabilities
            
            # Compute the probabilities
            p_left = np.sin(position / n * np.pi) ** 2
            p_right = np.cos(position / n * np.pi) ** 2
            
            # Normalize the probabilities
            p_sum = p_left + p_right
            p_left /= p_sum
            p_right /= p_sum
            
            # Choose a direction
            if np.random.random() < p_left:
                position = (position - 1) % n
            else:
                position = (position + 1) % n
            
            # Extract a bit
            bits[i] = position % 2
        
        # Update the state
        self.state = position
        
        return bits
    
    def _generate_hybrid(self, num_bits: int) -> np.ndarray:
        """
        Generate random bits using a hybrid approach.
        
        Args:
            num_bits: The number of random bits to generate
            
        Returns:
            An array of random bits (0 or 1)
        """
        # Use a combination of chaotic map and quantum walk
        
        # Generate half the bits using the chaotic map
        chaotic_bits = self._generate_chaotic(num_bits // 2)
        
        # Generate the other half using the quantum walk
        quantum_walk_bits = self._generate_quantum_walk(num_bits - len(chaotic_bits))
        
        # Combine the bits
        bits = np.concatenate([chaotic_bits, quantum_walk_bits])
        
        # Shuffle the bits
        np.random.shuffle(bits)
        
        return bits


class QuantumRandomnessTest:
    """
    Statistical tests for quantum randomness.
    
    This class implements statistical tests for verifying the quality of
    quantum random numbers.
    """
    
    def __init__(self):
        """
        Initialize the quantum randomness test.
        """
        pass
    
    def run_tests(self, bits: np.ndarray) -> Dict[str, Any]:
        """
        Run statistical tests on the random bits.
        
        Args:
            bits: The random bits to test
            
        Returns:
            A dictionary with the test results
        """
        results = {}
        
        # Run the tests
        results['frequency'] = self.frequency_test(bits)
        results['runs'] = self.runs_test(bits)
        results['serial'] = self.serial_test(bits)
        results['entropy'] = self.entropy_test(bits)
        
        return results
    
    def frequency_test(self, bits: np.ndarray) -> Dict[str, Any]:
        """
        Frequency (monobit) test.
        
        This test checks if the number of 1's and 0's in the sequence are approximately equal.
        
        Args:
            bits: The random bits to test
            
        Returns:
            A dictionary with the test results
        """
        # Count the number of 1's
        count_ones = np.sum(bits)
        
        # Count the number of 0's
        count_zeros = len(bits) - count_ones
        
        # Compute the test statistic
        s_obs = abs(count_ones - count_zeros) / np.sqrt(len(bits))
        
        # Compute the p-value
        p_value = math.erfc(s_obs / np.sqrt(2))
        
        # Check if the test passes
        alpha = 0.01  # Significance level
        passed = p_value >= alpha
        
        return {
            'name': 'Frequency Test',
            'count_ones': count_ones,
            'count_zeros': count_zeros,
            'statistic': s_obs,
            'p_value': p_value,
            'passed': passed
        }
    
    def runs_test(self, bits: np.ndarray) -> Dict[str, Any]:
        """
        Runs test.
        
        This test checks if the number of runs (sequences of consecutive 0's or 1's)
        is as expected for a random sequence.
        
        Args:
            bits: The random bits to test
            
        Returns:
            A dictionary with the test results
        """
        # Count the number of runs
        runs = 1
        for i in range(1, len(bits)):
            if bits[i] != bits[i-1]:
                runs += 1
        
        # Count the number of 1's
        count_ones = np.sum(bits)
        
        # Count the number of 0's
        count_zeros = len(bits) - count_ones
        
        # Compute the expected number of runs
        expected_runs = 1 + 2 * count_ones * count_zeros / len(bits)
        
        # Compute the variance
        variance = 2 * count_ones * count_zeros * (2 * count_ones * count_zeros - len(bits)) / (len(bits)**2 * (len(bits) - 1))
        
        # Compute the test statistic
        z = (runs - expected_runs) / np.sqrt(variance)
        
        # Compute the p-value
        p_value = math.erfc(abs(z) / np.sqrt(2))
        
        # Check if the test passes
        alpha = 0.01  # Significance level
        passed = p_value >= alpha
        
        return {
            'name': 'Runs Test',
            'runs': runs,
            'expected_runs': expected_runs,
            'statistic': z,
            'p_value': p_value,
            'passed': passed
        }
    
    def serial_test(self, bits: np.ndarray) -> Dict[str, Any]:
        """
        Serial test.
        
        This test checks if the frequency of all possible overlapping m-bit patterns
        is approximately the same.
        
        Args:
            bits: The random bits to test
            
        Returns:
            A dictionary with the test results
        """
        # Use m = 2 for the serial test
        m = 2
        
        # Count the frequency of each m-bit pattern
        counts = np.zeros(2**m, dtype=int)
        
        for i in range(len(bits) - m + 1):
            pattern = 0
            for j in range(m):
                pattern |= (bits[i + j] << (m - j - 1))
            counts[pattern] += 1
        
        # Compute the test statistic
        expected = (len(bits) - m + 1) / 2**m
        chi_squared = np.sum((counts - expected)**2) / expected
        
        # Compute the p-value
        p_value = 1 - stats.chi2.cdf(chi_squared, 2**m - 1)
        
        # Check if the test passes
        alpha = 0.01  # Significance level
        passed = p_value >= alpha
        
        return {
            'name': 'Serial Test',
            'counts': counts,
            'expected': expected,
            'statistic': chi_squared,
            'p_value': p_value,
            'passed': passed
        }
    
    def entropy_test(self, bits: np.ndarray) -> Dict[str, Any]:
        """
        Entropy test.
        
        This test checks if the entropy of the sequence is close to the maximum entropy
        for a random sequence.
        
        Args:
            bits: The random bits to test
            
        Returns:
            A dictionary with the test results
        """
        # Use blocks of size 8
        block_size = 8
        
        # Count the frequency of each block
        counts = np.zeros(2**block_size, dtype=int)
        
        for i in range(0, len(bits) - block_size + 1, block_size):
            block = 0
            for j in range(block_size):
                block |= (bits[i + j] << (block_size - j - 1))
            counts[block] += 1
        
        # Compute the entropy
        probabilities = counts / np.sum(counts)
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # Compute the maximum entropy
        max_entropy = block_size
        
        # Compute the entropy ratio
        entropy_ratio = entropy / max_entropy
        
        # Check if the test passes
        passed = entropy_ratio > 0.99
        
        return {
            'name': 'Entropy Test',
            'entropy': entropy,
            'max_entropy': max_entropy,
            'entropy_ratio': entropy_ratio,
            'passed': passed
        }


class QuantumRandomNumberGenerator:
    """
    Quantum random number generator.
    
    This class implements a quantum random number generator that can use
    different quantum randomness sources and perform statistical tests.
    """
    
    def __init__(self, source_type: str = 'software', source_params: Dict[str, Any] = None):
        """
        Initialize the quantum random number generator.
        
        Args:
            source_type: The type of randomness source ('hardware' or 'software')
            source_params: Parameters for the randomness source
        """
        self.source_type = source_type
        self.source_params = source_params or {}
        
        # Initialize the randomness source
        if source_type == 'hardware':
            device_type = self.source_params.get('device_type', 'simulator')
            self.source = HardwareQuantumRandomnessSource(device_type=device_type)
        elif source_type == 'software':
            algorithm = self.source_params.get('algorithm', 'chaotic')
            self.source = SoftwareQuantumRandomnessSource(algorithm=algorithm)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        # Initialize the randomness test
        self.test = QuantumRandomnessTest()
    
    def generate_random_bits(self, num_bits: int) -> np.ndarray:
        """
        Generate random bits.
        
        Args:
            num_bits: The number of random bits to generate
            
        Returns:
            An array of random bits (0 or 1)
        """
        return self.source.generate_random_bits(num_bits)
    
    def generate_random_bytes(self, num_bytes: int) -> bytes:
        """
        Generate random bytes.
        
        Args:
            num_bytes: The number of random bytes to generate
            
        Returns:
            Random bytes
        """
        return self.source.generate_random_bytes(num_bytes)
    
    def generate_random_int(self, min_value: int, max_value: int) -> int:
        """
        Generate a random integer in the range [min_value, max_value].
        
        Args:
            min_value: The minimum value (inclusive)
            max_value: The maximum value (inclusive)
            
        Returns:
            A random integer
        """
        return self.source.generate_random_int(min_value, max_value)
    
    def generate_random_float(self) -> float:
        """
        Generate a random float in the range [0, 1).
        
        Returns:
            A random float
        """
        return self.source.generate_random_float()
    
    def test_randomness(self, num_bits: int = 10000) -> Dict[str, Any]:
        """
        Test the randomness of the generated bits.
        
        Args:
            num_bits: The number of random bits to generate for testing
            
        Returns:
            A dictionary with the test results
        """
        # Generate random bits
        bits = self.generate_random_bits(num_bits)
        
        # Run the tests
        results = self.test.run_tests(bits)
        
        return results
    
    def visualize_randomness(self, num_bits: int = 10000) -> None:
        """
        Visualize the randomness of the generated bits.
        
        Args:
            num_bits: The number of random bits to generate for visualization
        """
        # Generate random bits
        bits = self.generate_random_bits(num_bits)
        
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot the bits
        axs[0, 0].plot(bits)
        axs[0, 0].set_title('Random Bits')
        axs[0, 0].set_xlabel('Index')
        axs[0, 0].set_ylabel('Bit Value')
        
        # Plot the histogram
        axs[0, 1].hist(bits, bins=2, rwidth=0.8)
        axs[0, 1].set_title('Histogram')
        axs[0, 1].set_xlabel('Bit Value')
        axs[0, 1].set_ylabel('Frequency')
        
        # Plot the autocorrelation
        autocorr = np.correlate(bits - np.mean(bits), bits - np.mean(bits), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0]
        axs[1, 0].plot(autocorr)
        axs[1, 0].set_title('Autocorrelation')
        axs[1, 0].set_xlabel('Lag')
        axs[1, 0].set_ylabel('Autocorrelation')
        
        # Plot the runs
        runs = np.zeros_like(bits)
        run_length = 1
        for i in range(1, len(bits)):
            if bits[i] == bits[i-1]:
                run_length += 1
            else:
                runs[i-run_length:i] = run_length
                run_length = 1
        runs[len(bits)-run_length:] = run_length
        axs[1, 1].plot(runs)
        axs[1, 1].set_title('Run Lengths')
        axs[1, 1].set_xlabel('Index')
        axs[1, 1].set_ylabel('Run Length')
        
        # Adjust the layout
        plt.tight_layout()
        
        # Show the figure
        plt.show()


class QuantumRandomNumberService:
    """
    Quantum random number service.
    
    This class implements a service that provides quantum random numbers
    for various applications.
    """
    
    def __init__(self):
        """
        Initialize the quantum random number service.
        """
        # Initialize the generators
        self.generators = {
            'hardware-simulator': QuantumRandomNumberGenerator(
                source_type='hardware',
                source_params={'device_type': 'simulator'}
            ),
            'software-chaotic': QuantumRandomNumberGenerator(
                source_type='software',
                source_params={'algorithm': 'chaotic'}
            ),
            'software-quantum_walk': QuantumRandomNumberGenerator(
                source_type='software',
                source_params={'algorithm': 'quantum_walk'}
            ),
            'software-hybrid': QuantumRandomNumberGenerator(
                source_type='software',
                source_params={'algorithm': 'hybrid'}
            )
        }
        
        # Set the default generator
        self.default_generator = 'software-hybrid'
    
    def get_generator(self, generator_name: str = None) -> QuantumRandomNumberGenerator:
        """
        Get a quantum random number generator.
        
        Args:
            generator_name: The name of the generator to get
            
        Returns:
            A quantum random number generator
        """
        if generator_name is None:
            generator_name = self.default_generator
        
        if generator_name not in self.generators:
            raise ValueError(f"Unknown generator: {generator_name}")
        
        return self.generators[generator_name]
    
    def generate_random_bits(self, num_bits: int, generator_name: str = None) -> np.ndarray:
        """
        Generate random bits.
        
        Args:
            num_bits: The number of random bits to generate
            generator_name: The name of the generator to use
            
        Returns:
            An array of random bits (0 or 1)
        """
        generator = self.get_generator(generator_name)
        return generator.generate_random_bits(num_bits)
    
    def generate_random_bytes(self, num_bytes: int, generator_name: str = None) -> bytes:
        """
        Generate random bytes.
        
        Args:
            num_bytes: The number of random bytes to generate
            generator_name: The name of the generator to use
            
        Returns:
            Random bytes
        """
        generator = self.get_generator(generator_name)
        return generator.generate_random_bytes(num_bytes)
    
    def generate_random_int(self, min_value: int, max_value: int, generator_name: str = None) -> int:
        """
        Generate a random integer in the range [min_value, max_value].
        
        Args:
            min_value: The minimum value (inclusive)
            max_value: The maximum value (inclusive)
            generator_name: The name of the generator to use
            
        Returns:
            A random integer
        """
        generator = self.get_generator(generator_name)
        return generator.generate_random_int(min_value, max_value)
    
    def generate_random_float(self, generator_name: str = None) -> float:
        """
        Generate a random float in the range [0, 1).
        
        Args:
            generator_name: The name of the generator to use
            
        Returns:
            A random float
        """
        generator = self.get_generator(generator_name)
        return generator.generate_random_float()
    
    def generate_random_password(self, length: int = 16, generator_name: str = None) -> str:
        """
        Generate a random password.
        
        Args:
            length: The length of the password
            generator_name: The name of the generator to use
            
        Returns:
            A random password
        """
        # Define the character sets
        lowercase = 'abcdefghijklmnopqrstuvwxyz'
        uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        digits = '0123456789'
        special = '!@#$%^&*()-_=+[]{}|;:,.<>?'
        
        # Combine the character sets
        all_chars = lowercase + uppercase + digits + special
        
        # Generate random indices
        generator = self.get_generator(generator_name)
        indices = [generator.generate_random_int(0, len(all_chars) - 1) for _ in range(length)]
        
        # Generate the password
        password = ''.join(all_chars[i] for i in indices)
        
        return password
    
    def generate_random_uuid(self, generator_name: str = None) -> str:
        """
        Generate a random UUID.
        
        Args:
            generator_name: The name of the generator to use
            
        Returns:
            A random UUID
        """
        # Generate 16 random bytes
        random_bytes = self.generate_random_bytes(16, generator_name)
        
        # Set the version (4) and variant (2) bits
        random_bytes = bytearray(random_bytes)
        random_bytes[6] = (random_bytes[6] & 0x0F) | 0x40  # Version 4
        random_bytes[8] = (random_bytes[8] & 0x3F) | 0x80  # Variant 2
        
        # Convert to a UUID string
        uuid = '-'.join([
            random_bytes[:4].hex(),
            random_bytes[4:6].hex(),
            random_bytes[6:8].hex(),
            random_bytes[8:10].hex(),
            random_bytes[10:].hex()
        ])
        
        return uuid
    
    def test_generator(self, generator_name: str = None, num_bits: int = 10000) -> Dict[str, Any]:
        """
        Test a quantum random number generator.
        
        Args:
            generator_name: The name of the generator to test
            num_bits: The number of random bits to generate for testing
            
        Returns:
            A dictionary with the test results
        """
        generator = self.get_generator(generator_name)
        return generator.test_randomness(num_bits)
    
    def visualize_generator(self, generator_name: str = None, num_bits: int = 10000) -> None:
        """
        Visualize the randomness of a quantum random number generator.
        
        Args:
            generator_name: The name of the generator to visualize
            num_bits: The number of random bits to generate for visualization
        """
        generator = self.get_generator(generator_name)
        generator.visualize_randomness(num_bits)


# Example usage
if __name__ == "__main__":
    # Create a quantum random number service
    qrng_service = QuantumRandomNumberService()
    
    # Generate random bits
    bits = qrng_service.generate_random_bits(100)
    print(f"Random bits: {bits[:10]}...")
    
    # Generate random bytes
    bytes_data = qrng_service.generate_random_bytes(10)
    print(f"Random bytes: {bytes_data.hex()}")
    
    # Generate a random integer
    random_int = qrng_service.generate_random_int(1, 100)
    print(f"Random integer: {random_int}")
    
    # Generate a random float
    random_float = qrng_service.generate_random_float()
    print(f"Random float: {random_float}")
    
    # Generate a random password
    random_password = qrng_service.generate_random_password()
    print(f"Random password: {random_password}")
    
    # Generate a random UUID
    random_uuid = qrng_service.generate_random_uuid()
    print(f"Random UUID: {random_uuid}")
    
    # Test the randomness of the default generator
    print("\nTesting the randomness of the default generator:")
    results = qrng_service.test_generator(num_bits=1000)
    
    for test_name, test_result in results.items():
        print(f"\n{test_result['name']}:")
        print(f"  Passed: {test_result['passed']}")
        print(f"  P-value: {test_result.get('p_value', 'N/A')}")
        
        if test_name == 'frequency':
            print(f"  Count of 1's: {test_result['count_ones']}")
            print(f"  Count of 0's: {test_result['count_zeros']}")
        elif test_name == 'runs':
            print(f"  Runs: {test_result['runs']}")
            print(f"  Expected runs: {test_result['expected_runs']}")
        elif test_name == 'entropy':
            print(f"  Entropy: {test_result['entropy']}")
            print(f"  Max entropy: {test_result['max_entropy']}")
            print(f"  Entropy ratio: {test_result['entropy_ratio']}")
    
    # Visualize the randomness of the default generator
    # Uncomment to show the visualization
    # qrng_service.visualize_generator(num_bits=1000)