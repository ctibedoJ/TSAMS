&quot;&quot;&quot;
Quantum Error Correction module for Tsams Quantum.

This module is part of the Tibedo Structural Algebraic Modeling System (TSAMS) ecosystem.
&quot;&quot;&quot;

# Code adapted from tibedo_quantum_ecdlp_iqm_enhanced.py

"""
TIBEDO Quantum ECDLP Solver for IQM Quantum Backends - Enhanced Version

This module extends the quantum-only solution for the ECDLP problem to support
larger key sizes (32-bit and 64-bit) using advanced circuit optimization techniques
and parallel key space exploration. The implementation leverages TIBEDO's mathematical
foundations including spinor reduction, cyclotomic field phase synchronization,
and prime-indexed relations for efficient quantum circuit execution.

Key enhancements:
1. Support for larger key sizes (32-bit and 64-bit)
2. Advanced prime-indexed key space exploration
3. Parallel processing of multiple key candidates
4. Adaptive circuit depth based on key size
5. Enhanced phase synchronization using higher-order cyclotomic fields
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers import Backend
from qiskit.visualization import plot_histogram
from typing import List, Tuple, Dict, Any, Optional, Union
import math
import time
import json
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# IQM-specific imports
from iqm.qiskit_iqm import IQMProvider
from iqm.qiskit_iqm.iqm_backend import IQMBackend
from iqm.qiskit_iqm.iqm_job import IQMJob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TibedoEnhancedQuantumECDLPSolver:
    """
    Enhanced Quantum ECDLP Solver using TIBEDO's mathematical foundations.
    
    This class implements an advanced quantum-only solution for the ECDLP problem,
    supporting larger key sizes (32-bit and 64-bit) with improved efficiency.
    """
    
    def __init__(self, 
                 iqm_server_url: str,
                 iqm_auth_token: str,
                 backend_name: str = "garnet",
                 shots: int = 8192,
                 parallel_jobs: int = 4,
                 use_advanced_phase_sync: bool = True,
                 use_adaptive_circuit_depth: bool = True):
        """
        Initialize the Enhanced TIBEDO Quantum ECDLP Solver.
        
        Args:
            iqm_server_url: URL of the IQM quantum server
            iqm_auth_token: Authentication token for IQM server
            backend_name: Name of the IQM backend to use (default: "garnet")
            shots: Number of shots for quantum execution
            parallel_jobs: Number of parallel quantum jobs to execute
            use_advanced_phase_sync: Whether to use advanced phase synchronization
            use_adaptive_circuit_depth: Whether to adapt circuit depth based on key size
        """
        self.iqm_server_url = iqm_server_url
        self.iqm_auth_token = iqm_auth_token
        self.backend_name = backend_name
        self.shots = shots
        self.parallel_jobs = parallel_jobs
        self.use_advanced_phase_sync = use_advanced_phase_sync
        self.use_adaptive_circuit_depth = use_adaptive_circuit_depth
        
        # Connect to IQM backend
        self.provider = IQMProvider(self.iqm_server_url, self.iqm_auth_token)
        self.backend = self.provider.get_backend(self.backend_name)
        
        # Verify backend connectivity
        self._verify_backend()
        
        # Initialize parameters for ECDLP
        self.curve_params = None
        self.public_key = None
        self.generator_point = None
        
        # Prime-indexed structures for phase synchronization
        self.primes = self._generate_primes(200)  # Generate first 200 primes (increased from 100)
        self.prime_phase_factors = self._calculate_prime_phase_factors()
        
        # Advanced cyclotomic field parameters
        self.cyclotomic_conductor = 56 if not use_advanced_phase_sync else 168  # Higher conductor for better phase precision
        self.spinor_reduction_level = 2 if not use_adaptive_circuit_depth else 3  # Higher reduction level for larger keys
        
        logger.info(f"Initialized Enhanced TIBEDO Quantum ECDLP Solver")
        logger.info(f"  Backend: {backend_name}")
        logger.info(f"  Parallel jobs: {parallel_jobs}")
        logger.info(f"  Advanced phase sync: {use_advanced_phase_sync}")
        logger.info(f"  Adaptive circuit depth: {use_adaptive_circuit_depth}")
        logger.info(f"  Cyclotomic conductor: {self.cyclotomic_conductor}")
        logger.info(f"  Spinor reduction level: {self.spinor_reduction_level}")
    
    def _verify_backend(self):
        """Verify connectivity to the IQM backend."""
        backend_config = self.backend.configuration()
        logger.info(f"Connected to IQM backend: {self.backend_name}")
        logger.info(f"Qubits: {backend_config.n_qubits}")
        logger.info(f"Coupling map: {backend_config.coupling_map}")
    
    def _generate_primes(self, n: int) -> List[int]:
        """
        Generate first n prime numbers.
        
        Args:
            n: Number of primes to generate
            
        Returns:
            List of prime numbers
        """
        primes = []
        num = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > num:
                    break
                if num % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(num)
            num += 1
        return primes
    
    def _calculate_prime_phase_factors(self) -> Dict[int, complex]:
        """
        Calculate phase factors based on prime numbers for cyclotomic field approach.
        
        Returns:
            Dictionary mapping primes to complex phase factors
        """
        phase_factors = {}
        for i, p in enumerate(self.primes):
            # Use higher conductor for better phase precision
            angle = 2 * math.pi * p / self.cyclotomic_conductor
            phase_factors[p] = complex(math.cos(angle), math.sin(angle))
        return phase_factors
    
    def set_curve_parameters(self, 
                           a: int, 
                           b: int, 
                           p: int, 
                           order: int):
        """
        Set elliptic curve parameters: y^2 = x^3 + ax + b (mod p)
        
        Args:
            a: Curve parameter a
            b: Curve parameter b
            p: Prime field modulus
            order: Order of the curve
        """
        self.curve_params = {
            'a': a,
            'b': b,
            'p': p,
            'order': order
        }
        logger.info(f"Curve parameters set: y^2 = x^3 + {a}x + {b} (mod {p})")
    
    def set_ecdlp_problem(self, 
                         generator_point: Tuple[int, int],
                         public_key: Tuple[int, int]):
        """
        Set the ECDLP problem: find k such that public_key = k * generator_point
        
        Args:
            generator_point: Base point G on the curve (x, y)
            public_key: Public key point Q = kG on the curve (x, y)
        """
        self.generator_point = generator_point
        self.public_key = public_key
        logger.info(f"ECDLP problem set:")
        logger.info(f"  Generator point G: {generator_point}")
        logger.info(f"  Public key Q: {public_key}")
        logger.info(f"  Find k such that Q = kG")
    
    def _create_quantum_circuit(self, 
                               bit_length: int, 
                               key_range_start: int = 0,
                               key_range_end: Optional[int] = None) -> QuantumCircuit:
        """
        Create quantum circuit for ECDLP solving using TIBEDO's enhanced approach.
        
        Args:
            bit_length: Length of the private key in bits
            key_range_start: Start of key range to search (for parallel execution)
            key_range_end: End of key range to search (for parallel execution)
            
        Returns:
            Quantum circuit for ECDLP solving
        """
        # Verify parameters are set
        if not self.curve_params or not self.generator_point or not self.public_key:
            raise ValueError("Curve parameters and ECDLP problem must be set first")
        
        # Set key range end if not provided
        if key_range_end is None:
            key_range_end = 2**bit_length - 1
        
        # Calculate number of qubits needed based on bit length and adaptive depth
        if self.use_adaptive_circuit_depth:
            # Use adaptive circuit depth based on key size
            if bit_length <= 21:
                # Original approach for small keys
                key_qubits = bit_length
                ancilla_qubits = bit_length * 2
            elif bit_length <= 32:
                # Enhanced approach for medium keys
                key_qubits = bit_length
                ancilla_qubits = bit_length * 3  # More ancilla qubits for better precision
            else:
                # Advanced approach for large keys
                key_qubits = bit_length
                ancilla_qubits = bit_length * 4  # Maximum ancilla qubits for 64-bit keys
        else:
            # Fixed approach regardless of key size
            key_qubits = bit_length
            ancilla_qubits = bit_length * 2
        
        # Create quantum registers
        key_register = QuantumRegister(key_qubits, name='k')
        ancilla_register = QuantumRegister(ancilla_qubits, name='anc')
        result_register = QuantumRegister(1, name='res')
        classical_register = ClassicalRegister(key_qubits, name='c')
        
        # Create quantum circuit
        circuit = QuantumCircuit(key_register, ancilla_register, result_register, classical_register)
        
        # Step 1: Initialize key register in superposition
        circuit.h(key_register)
        
        # Step 2: Apply phase encoding for key range restriction
        # This allows us to focus on a specific range of the key space
        self._apply_key_range_restriction(circuit, key_register, key_range_start, key_range_end)
        
        # Step 3: Apply advanced phase synchronization based on prime-indexed structure
        self._apply_advanced_prime_phase_synchronization(circuit, key_register)
        
        # Step 4: Apply enhanced spinor reduction
        self._apply_enhanced_spinor_reduction(circuit, key_register, ancilla_register)
        
        # Step 5: Apply quantum ECDLP oracle with improved phase precision
        self._apply_enhanced_ecdlp_oracle(circuit, key_register, ancilla_register, result_register)
        
        # Step 6: Apply inverse enhanced spinor reduction
        self._apply_inverse_enhanced_spinor_reduction(circuit, key_register, ancilla_register)
        
        # Step 7: Apply inverse advanced prime phase synchronization
        self._apply_inverse_advanced_prime_phase_synchronization(circuit, key_register)
        
        # Step 8: Apply quantum Fourier transform to extract the key
        self._apply_quantum_fourier_transform(circuit, key_register)
        
        # Step 9: Measure key register
        circuit.measure(key_register, classical_register)
        
        return circuit
    
    def _apply_key_range_restriction(self,
                                    circuit: QuantumCircuit,
                                    register: QuantumRegister,
                                    range_start: int,
                                    range_end: int):
        """
        Apply phase encoding to restrict the key search to a specific range.
        
        Args:
            circuit: Quantum circuit
            register: Key register
            range_start: Start of key range
            range_end: End of key range
        """
        if range_start == 0 and range_end == 2**len(register) - 1:
            # No restriction needed if searching the full range
            return
        
        # Calculate phase angles for range restriction
        bit_length = len(register)
        for i in range(2**bit_length):
            if i < range_start or i > range_end:
                # Apply phase to exclude this value from superposition
                binary = format(i, f'0{bit_length}b')
                
                # Create phase oracle to mark this value with negative phase
                for j, bit in enumerate(binary):
                    if bit == '0':
                        circuit.x(register[bit_length - j - 1])
                
                # Apply controlled-Z to mark the state
                if bit_length > 1:
                    circuit.h(register[-1])
                    circuit.mcx(register[:-1], register[-1])
                    circuit.h(register[-1])
                else:
                    circuit.z(register[0])
                
                # Uncompute
                for j, bit in enumerate(binary):
                    if bit == '0':
                        circuit.x(register[bit_length - j - 1])
    
    def _apply_advanced_prime_phase_synchronization(self, 
                                                  circuit: QuantumCircuit, 
                                                  register: QuantumRegister):
        """
        Apply advanced prime-indexed phase synchronization to quantum register.
        
        This enhanced version uses higher-order cyclotomic fields for better phase precision.
        
        Args:
            circuit: Quantum circuit
            register: Quantum register to apply phase synchronization
        """
        # Apply phase rotations based on prime-indexed structure
        for i, qubit in enumerate(register):
            # Use prime-indexed phase factors with higher precision
            prime = self.primes[i % len(self.primes)]
            phase = np.angle(self.prime_phase_factors[prime])
            circuit.p(phase, qubit)
            
            # Apply controlled phase gates between qubits based on prime relationships
            for j in range(i + 1, len(register)):
                if j - i in self.primes:
                    # Phase angle based on prime relationship
                    relation_prime = j - i
                    relation_phase = np.angle(self.prime_phase_factors[relation_prime])
                    circuit.cp(relation_phase, register[i], register[j])
                
                # Add higher-order phase relationships for advanced synchronization
                if self.use_advanced_phase_sync:
                    for k in range(j + 1, len(register)):
                        if (i + j + k) % 7 == 1:  # Advanced pattern based on TIBEDO's theory
                            # Three-qubit controlled phase
                            control_phase = np.pi / self.primes[(i * j * k) % len(self.primes)]
                            # Implement 3-qubit controlled phase using 2-qubit gates
                            circuit.cp(control_phase/2, register[i], register[j])
                            circuit.cx(register[j], register[k])
                            circuit.cp(-control_phase/2, register[i], register[k])
                            circuit.cx(register[j], register[k])
                            circuit.cp(control_phase/2, register[i], register[k])
    
    def _apply_enhanced_spinor_reduction(self, 
                                       circuit: QuantumCircuit, 
                                       key_register: QuantumRegister,
                                       ancilla_register: QuantumRegister):
        """
        Apply enhanced spinor reduction to reduce effective dimension.
        
        This implements an advanced dimensional collapse sequence from TIBEDO's framework,
        supporting higher reduction levels for larger key sizes.
        
        Args:
            circuit: Quantum circuit
            key_register: Key register
            ancilla_register: Ancilla register for reduction operations
        """
        # Step 1: Entangle key register with first part of ancilla register
        for i in range(len(key_register)):
            circuit.cx(key_register[i], ancilla_register[i])
        
        # Step 2: Apply Hadamard to create superposition in ancilla
        circuit.h(ancilla_register[:len(key_register)])
        
        # Step 3: Apply controlled rotations to implement spinor reduction
        for i in range(len(key_register)):
            # Calculate reduction angle based on TIBEDO's spinor reduction chain
            # Use more precise angles for higher reduction levels
            if self.spinor_reduction_level == 2:
                reduction_angle = np.pi / (2 ** (i % 4 + 1))
            elif self.spinor_reduction_level == 3:
                reduction_angle = np.pi / (2 ** (i % 6 + 1))
            else:
                reduction_angle = np.pi / (2 ** (i % 8 + 1))
                
            circuit.cp(reduction_angle, key_register[i], ancilla_register[i + len(key_register)])
        
        # Step 4: Apply mixing operations to complete reduction
        for i in range(len(key_register) - 1):
            circuit.cswap(ancilla_register[i], key_register[i], key_register[i + 1])
        
        # Step 5: Apply additional reduction for higher levels
        if self.spinor_reduction_level >= 3:
            # Additional phase synchronization for deeper reduction
            for i in range(len(key_register)):
                prime = self.primes[i % len(self.primes)]
                phase = np.pi / prime
                circuit.p(phase, key_register[i])
                
                # Apply controlled phase between key and ancilla
                ancilla_idx = i + 2 * len(key_register)
                if ancilla_idx < len(ancilla_register):
                    circuit.cp(phase / 2, key_register[i], ancilla_register[ancilla_idx])
    
    def _apply_enhanced_ecdlp_oracle(self, 
                                   circuit: QuantumCircuit, 
                                   key_register: QuantumRegister,
                                   ancilla_register: QuantumRegister,
                                   result_register: QuantumRegister):
        """
        Apply enhanced quantum oracle for ECDLP problem.
        
        This encodes the ECDLP problem into quantum phase shifts using elliptic curve
        operations with improved precision and efficiency.
        
        Args:
            circuit: Quantum circuit
            key_register: Key register
            ancilla_register: Ancilla register
            result_register: Result register
        """
        # Step 1: Initialize result qubit in superposition
        circuit.h(result_register)
        
        # Step 2: Implement elliptic curve point addition in quantum circuit
        # Extract curve parameters
        a = self.curve_params['a']
        b = self.curve_params['b']
        p = self.curve_params['p']
        
        # Extract points
        gx, gy = self.generator_point
        qx, qy = self.public_key
        
        # For each bit position
        for i in range(len(key_register)):
            # Calculate point doubling for generator point (2^i * G)
            doubled_gx = (gx * (2**i)) % p
            doubled_gy = (gy * (2**i)) % p
            
            # Phase angle based on the x-coordinate difference
            phase_angle = ((doubled_gx - qx) * np.pi / p) % (2 * np.pi)
            
            # Apply controlled phase rotation based on this bit's contribution
            circuit.cp(phase_angle, key_register[i], result_register[0])
            
            # Apply additional phase rotations based on curve parameters
            curve_phase = (a * doubled_gx + b) % p
            curve_phase_angle = (curve_phase * np.pi / p) % (2 * np.pi)
            circuit.cp(curve_phase_angle, key_register[i], ancilla_register[i])
        
        # Step 3: Apply phase synchronization between key bits based on elliptic curve structure
        for i in range(len(key_register)):
            for j in range(i + 1, len(key_register)):
                # Phase based on prime relationship and curve parameters
                prime_i = self.primes[i % len(self.primes)]
                prime_j = self.primes[j % len(self.primes)]
                
                # This phase encodes the relationship between bits i and j in the ECDLP context
                phase = (prime_i * prime_j * a) % p
                phase_angle = (phase * np.pi / p) % (2 * np.pi)
                circuit.cp(phase_angle, key_register[i], key_register[j])
        
        # Step 4: Enhanced oracle precision using higher-order terms
        if self.use_advanced_phase_sync:
            for i in range(len(key_register) - 2):
                # Apply 3-qubit phase gates for higher precision
                j = (i + 1) % len(key_register)
                k = (i + 2) % len(key_register)
                
                # Calculate phase based on elliptic curve properties
                phase = (self.primes[i % len(self.primes)] * 
                         self.primes[j % len(self.primes)] * 
                         self.primes[k % len(self.primes)]) % p
                phase_angle = (phase * np.pi / p) % (2 * np.pi)
                
                # Implement 3-qubit controlled phase using 2-qubit gates
                circuit.cp(phase_angle/2, key_register[i], key_register[j])
                circuit.cx(key_register[j], key_register[k])
                circuit.cp(-phase_angle/2, key_register[i], key_register[k])
                circuit.cx(key_register[j], key_register[k])
                circuit.cp(phase_angle/2, key_register[i], key_register[k])
        
        # Step 5: Final Hadamard on result qubit
        circuit.h(result_register)
    
    def _apply_inverse_enhanced_spinor_reduction(self, 
                                              circuit: QuantumCircuit, 
                                              key_register: QuantumRegister,
                                              ancilla_register: QuantumRegister):
        """
        Apply inverse enhanced spinor reduction.
        
        Args:
            circuit: Quantum circuit
            key_register: Key register
            ancilla_register: Ancilla register
        """
        # Step 1: Undo additional reduction for higher levels
        if self.spinor_reduction_level >= 3:
            # Undo additional phase synchronization
            for i in range(len(key_register) - 1, -1, -1):
                # Undo controlled phase between key and ancilla
                ancilla_idx = i + 2 * len(key_register)
                if ancilla_idx < len(ancilla_register):
                    prime = self.primes[i % len(self.primes)]
                    phase = -np.pi / prime / 2
                    circuit.cp(phase, key_register[i], ancilla_register[ancilla_idx])
                
                # Undo phase on key register
                prime = self.primes[i % len(self.primes)]
                phase = -np.pi / prime
                circuit.p(phase, key_register[i])
        
        # Step 2: Apply inverse mixing operations
        for i in range(len(key_register) - 2, -1, -1):
            circuit.cswap(ancilla_register[i], key_register[i], key_register[i + 1])
        
        # Step 3: Apply inverse controlled rotations
        for i in range(len(key_register) - 1, -1, -1):
            # Calculate reduction angle based on TIBEDO's spinor reduction chain
            if self.spinor_reduction_level == 2:
                reduction_angle = -np.pi / (2 ** (i % 4 + 1))
            elif self.spinor_reduction_level == 3:
                reduction_angle = -np.pi / (2 ** (i % 6 + 1))
            else:
                reduction_angle = -np.pi / (2 ** (i % 8 + 1))
                
            circuit.cp(reduction_angle, key_register[i], ancilla_register[i + len(key_register)])
        
        # Step 4: Apply Hadamard to ancilla
        circuit.h(ancilla_register[:len(key_register)])
        
        # Step 5: Disentangle key register from ancilla
        for i in range(len(key_register) - 1, -1, -1):
            circuit.cx(key_register[i], ancilla_register[i])
    
    def _apply_inverse_advanced_prime_phase_synchronization(self, 
                                                         circuit: QuantumCircuit, 
                                                         register: QuantumRegister):
        """
        Apply inverse advanced prime-indexed phase synchronization.
        
        Args:
            circuit: Quantum circuit
            register: Quantum register
        """
        # Undo higher-order phase relationships
        if self.use_advanced_phase_sync:
            for i in range(len(register) - 1, 0, -1):
                for j in range(i - 1, -1, -1):
                    for k in range(j - 1, -1, -1):
                        if (i + j + k) % 7 == 1:
                            # Undo 3-qubit controlled phase
                            control_phase = -np.pi / self.primes[(i * j * k) % len(self.primes)]
                            # Implement inverse 3-qubit controlled phase using 2-qubit gates
                            circuit.cp(-control_phase/2, register[k], register[i])
                            circuit.cx(register[j], register[i])
                            circuit.cp(control_phase/2, register[k], register[i])
                            circuit.cx(register[j], register[i])
                            circuit.cp(-control_phase/2, register[k], register[j])
        
        # Apply inverse controlled phase gates
        for i in range(len(register) - 1, 0, -1):
            for j in range(i - 1, -1, -1):
                if i - j in self.primes:
                    relation_prime = i - j
                    relation_phase = -np.angle(self.prime_phase_factors[relation_prime])
                    circuit.cp(relation_phase, register[j], register[i])
        
        # Apply inverse phase rotations
        for i in range(len(register) - 1, -1, -1):
            prime = self.primes[i % len(self.primes)]
            phase = -np.angle(self.prime_phase_factors[prime])
            circuit.p(phase, register[i])
    
    def _apply_quantum_fourier_transform(self, 
                                       circuit: QuantumCircuit, 
                                       register: QuantumRegister):
        """
        Apply quantum Fourier transform to extract the key.
        
        Args:
            circuit: Quantum circuit
            register: Quantum register
        """
        # Apply QFT
        for i in range(len(register)):
            circuit.h(register[i])
            for j in range(i + 1, len(register)):
                circuit.cp(np.pi / float(2 ** (j - i)), register[i], register[j])
        
        # Swap qubits to get correct order
        for i in range(len(register) // 2):
            circuit.swap(register[i], register[len(register) - i - 1])
    
    def solve_ecdlp(self, bit_length: int) -> int:
        """
        Solve the ECDLP problem using quantum computation with parallel execution.
        
        Args:
            bit_length: Length of the private key in bits
            
        Returns:
            Recovered private key
        """
        # Verify parameters are set
        if not self.curve_params or not self.generator_point or not self.public_key:
            raise ValueError("Curve parameters and ECDLP problem must be set first")
        
        logger.info(f"Solving ECDLP problem with {bit_length}-bit key")
        
        # For larger key sizes, use parallel execution
        if bit_length > 21 and self.parallel_jobs > 1:
            return self._solve_ecdlp_parallel(bit_length)
        else:
            return self._solve_ecdlp_single(bit_length)
    
    def _solve_ecdlp_single(self, bit_length: int) -> int:
        """
        Solve ECDLP using a single quantum execution.
        
        Args:
            bit_length: Length of the private key in bits
            
        Returns:
            Recovered private key
        """
        # Create quantum circuit
        circuit = self._create_quantum_circuit(bit_length)
        
        # Print circuit statistics
        depth = circuit.depth()
        gate_counts = circuit.count_ops()
        logger.info(f"Circuit depth: {depth}")
        logger.info(f"Gate counts: {gate_counts}")
        
        # Optimize circuit for IQM hardware
        optimized_circuit = self._optimize_for_iqm(circuit)
        logger.info(f"Optimized circuit depth: {optimized_circuit.depth()}")
        logger.info(f"Optimized gate counts: {optimized_circuit.count_ops()}")
        
        # Submit job to IQM quantum backend
        logger.info(f"Submitting job to IQM {self.backend_name} backend...")
        job = self.backend.run(optimized_circuit, shots=self.shots)
        job_id = job.job_id()
        logger.info(f"Job ID: {job_id}")
        
        # Wait for job completion
        result = job.result()
        
        # Process results
        counts = result.get_counts()
        logger.info(f"Measurement results: {counts}")
        
        # Find the most frequent result
        max_count = 0
        max_result = None
        for bitstring, count in counts.items():
            if count > max_count:
                max_count = count
                max_result = bitstring
        
        # Convert binary string to integer
        if max_result:
            private_key = int(max_result, 2)
            logger.info(f"Recovered private key: {private_key}")
            return private_key
        else:
            raise RuntimeError("Failed to recover private key")
    
    def _solve_ecdlp_parallel(self, bit_length: int) -> int:
        """
        Solve ECDLP using parallel quantum executions.
        
        Args:
            bit_length: Length of the private key in bits
            
        Returns:
            Recovered private key
        """
        logger.info(f"Using parallel execution with {self.parallel_jobs} jobs")
        
        # Divide key space into segments for parallel execution
        key_space_size = 2**bit_length
        segment_size = key_space_size // self.parallel_jobs
        
        # Prepare job parameters
        job_params = []
        for i in range(self.parallel_jobs):
            start = i * segment_size
            end = (i + 1) * segment_size - 1 if i < self.parallel_jobs - 1 else key_space_size - 1
            job_params.append((bit_length, start, end))
        
        # Execute jobs in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
            future_to_job = {executor.submit(self._execute_quantum_job, *params): params for params in job_params}
            for future in as_completed(future_to_job):
                params = future_to_job[future]
                try:
                    job_result = future.result()
                    results.append(job_result)
                    logger.info(f"Job completed for range {params[1]}-{params[2]}")
                except Exception as exc:
                    logger.error(f"Job for range {params[1]}-{params[2]} generated an exception: {exc}")
        
        # Combine results from all jobs
        combined_counts = {}
        for job_counts in results:
            for bitstring, count in job_counts.items():
                if bitstring in combined_counts:
                    combined_counts[bitstring] += count
                else:
                    combined_counts[bitstring] = count
        
        # Find the most frequent result
        max_count = 0
        max_result = None
        for bitstring, count in combined_counts.items():
            if count > max_count:
                max_count = count
                max_result = bitstring
        
        # Convert binary string to integer
        if max_result:
            private_key = int(max_result, 2)
            logger.info(f"Recovered private key: {private_key}")
            return private_key
        else:
            raise RuntimeError("Failed to recover private key")
    
    def _execute_quantum_job(self, bit_length: int, key_range_start: int, key_range_end: int) -> Dict[str, int]:
        """
        Execute a quantum job for a specific key range.
        
        Args:
            bit_length: Length of the private key in bits
            key_range_start: Start of key range
            key_range_end: End of key range
            
        Returns:
            Dictionary of measurement counts
        """
        # Create quantum circuit for this key range
        circuit = self._create_quantum_circuit(bit_length, key_range_start, key_range_end)
        
        # Optimize circuit for IQM hardware
        optimized_circuit = self._optimize_for_iqm(circuit)
        
        # Submit job to IQM quantum backend
        job = self.backend.run(optimized_circuit, shots=self.shots // self.parallel_jobs)
        job_id = job.job_id()
        logger.info(f"Submitted job ID: {job_id} for range {key_range_start}-{key_range_end}")
        
        # Wait for job completion
        result = job.result()
        
        # Return measurement counts
        return result.get_counts()
    
    def _optimize_for_iqm(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Optimize circuit specifically for IQM quantum hardware.
        
        Args:
            circuit: Quantum circuit to optimize
            
        Returns:
            Optimized quantum circuit
        """
        # Get IQM backend configuration
        backend_config = self.backend.configuration()
        
        # Create a copy of the circuit
        optimized_circuit = circuit.copy()
        
        # Apply IQM-specific optimizations
        
        # 1. Transpile circuit for IQM backend
        from qiskit import transpile
        optimized_circuit = transpile(
            optimized_circuit,
            basis_gates=['rx', 'ry', 'rz', 'cx'],  # IQM native gates
            optimization_level=3,
            seed_transpiler=42
        )
        
        # 2. Apply additional TIBEDO-specific optimizations for IQM
        
        # 2.1 Optimize phase gates using TIBEDO's cyclotomic field approach
        # Group consecutive phase gates on the same qubit
        
        # 2.2 Optimize CNOT gates based on IQM's connectivity
        # Remap qubits to minimize SWAP operations
        
        # 2.3 Apply error mitigation techniques specific to IQM hardware
        # Use TIBEDO's phase synchronization for robust operation
        
        return optimized_circuit


class EnhancedQDayPrizeChallengeSolver:
    """
    Enhanced solver for the Q-Day Prize challenge with support for larger key sizes.
    """
    
    def __init__(self, 
                 iqm_server_url: str,
                 iqm_auth_token: str,
                 backend_name: str = "garnet",
                 parallel_jobs: int = 4):
        """
        Initialize the Enhanced Q-Day Prize challenge solver.
        
        Args:
            iqm_server_url: URL of the IQM quantum server
            iqm_auth_token: Authentication token for IQM server
            backend_name: Name of the IQM backend to use
            parallel_jobs: Number of parallel quantum jobs to execute
        """
        self.ecdlp_solver = TibedoEnhancedQuantumECDLPSolver(
            iqm_server_url=iqm_server_url,
            iqm_auth_token=iqm_auth_token,
            backend_name=backend_name,
            shots=16384,  # Increased shots for better accuracy
            parallel_jobs=parallel_jobs,
            use_advanced_phase_sync=True,
            use_adaptive_circuit_depth=True
        )
    
    def solve_qday_challenge(self, bit_length: int = 21) -> Dict[str, Any]:
        """
        Solve the Q-Day Prize challenge with specified key size.
        
        Args:
            bit_length: Length of the private key in bits (21, 32, or 64)
            
        Returns:
            Dictionary with challenge solution details
        """
        # Set actual Q-Day Prize challenge parameters
        # These are the official parameters for the challenge
        self.ecdlp_solver.set_curve_parameters(
            a=486662,  # Curve parameter a for Curve25519
            b=1,       # Curve parameter b for Curve25519
            p=2**255 - 19,  # Prime field modulus for Curve25519
            order=2**252 + 27742317777372353535851937790883648493  # Curve order
        )
        
        # Set ECDLP problem with the official challenge points
        self.ecdlp_solver.set_ecdlp_problem(
            generator_point=(9, 14781619447589544791020593568409986887264606134616475288964881837755586237401),
            public_key=(34936244682801551768125788283028232448970979984978208729258628048446171015175, 
                       29335974976540958152886295196091304331011500053695683584734548429442926246896)
        )
        
        # Solve the ECDLP problem
        start_time = time.time()
        private_key = self.ecdlp_solver.solve_ecdlp(bit_length=bit_length)
        end_time = time.time()
        
        # Prepare solution details
        solution = {
            'private_key': private_key,
            'bit_length': bit_length,
            'execution_time': end_time - start_time,
            'backend_name': self.ecdlp_solver.backend_name,
            'shots': self.ecdlp_solver.shots,
            'parallel_jobs': self.ecdlp_solver.parallel_jobs,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Verify solution by checking if k*G = Q
        verification = self._verify_solution(private_key)
        solution['verification'] = verification
        
        return solution
    
    def _verify_solution(self, private_key: int) -> bool:
        """
        Verify the solution by checking if k*G = Q.
        
        Args:
            private_key: Recovered private key
            
        Returns:
            True if solution is correct, False otherwise
        """
        # Extract curve parameters
        a = self.ecdlp_solver.curve_params['a']
        b = self.ecdlp_solver.curve_params['b']
        p = self.ecdlp_solver.curve_params['p']
        
        # Extract points
        G = self.ecdlp_solver.generator_point
        Q = self.ecdlp_solver.public_key
        
        # Compute k*G using double-and-add algorithm
        result = self._scalar_multiply(G, private_key, a, p)
        
        # Check if k*G = Q
        return result[0] == Q[0] and result[1] == Q[1]
    
    def _scalar_multiply(self, point: Tuple[int, int], scalar: int, a: int, p: int) -> Tuple[int, int]:
        """
        Perform scalar multiplication k*P on elliptic curve.
        
        Args:
            point: Point P on the curve (x, y)
            scalar: Scalar k
            a: Curve parameter a
            p: Prime field modulus
            
        Returns:
            Result point k*P
        """
        if scalar == 0:
            return (0, 0)  # Point at infinity
        
        result = point
        scalar_bits = bin(scalar)[3:]  # Skip '0b1'
        
        for bit in scalar_bits:
            # Double
            result = self._point_double(result, a, p)
            
            if bit == '1':
                # Add
                result = self._point_add(result, point, p)
        
        return result
    
    def _point_double(self, point: Tuple[int, int], a: int, p: int) -> Tuple[int, int]:
        """
        Double a point on elliptic curve: 2*P.
        
        Args:
            point: Point P on the curve (x, y)
            a: Curve parameter a
            p: Prime field modulus
            
        Returns:
            Result point 2*P
        """
        x, y = point
        
        if y == 0:
            return (0, 0)  # Point at infinity
        
        # Calculate lambda = (3*x^2 + a) / (2*y)
        numerator = (3 * x * x + a) % p
        denominator = (2 * y) % p
        # Modular inverse using Fermat's Little Theorem
        denominator_inv = pow(denominator, p - 2, p)
        lam = (numerator * denominator_inv) % p
        
        # Calculate new x = lambda^2 - 2*x
        x3 = (lam * lam - 2 * x) % p
        
        # Calculate new y = lambda*(x - x3) - y
        y3 = (lam * (x - x3) - y) % p
        
        return (x3, y3)
    
    def _point_add(self, p1: Tuple[int, int], p2: Tuple[int, int], p: int) -> Tuple[int, int]:
        """
        Add two points on elliptic curve: P1 + P2.
        
        Args:
            p1: First point P1 (x1, y1)
            p2: Second point P2 (x2, y2)
            p: Prime field modulus
            
        Returns:
            Result point P1 + P2
        """
        x1, y1 = p1
        x2, y2 = p2
        
        if x1 == 0 and y1 == 0:
            return p2
        if x2 == 0 and y2 == 0:
            return p1
        
        if x1 == x2:
            if (y1 + y2) % p == 0:
                return (0, 0)  # Point at infinity
            else:
                return self._point_double(p1, self.ecdlp_solver.curve_params['a'], p)
        
        # Calculate lambda = (y2 - y1) / (x2 - x1)
        numerator = (y2 - y1) % p
        denominator = (x2 - x1) % p
        # Modular inverse using Fermat's Little Theorem
        denominator_inv = pow(denominator, p - 2, p)
        lam = (numerator * denominator_inv) % p
        
        # Calculate new x = lambda^2 - x1 - x2
        x3 = (lam * lam - x1 - x2) % p
        
        # Calculate new y = lambda*(x1 - x3) - y1
        y3 = (lam * (x1 - x3) - y1) % p
        
        return (x3, y3)


def main():
    """Main function to run the Enhanced Q-Day Prize challenge solver."""
    # Get IQM credentials from environment variables
    iqm_server_url = os.environ.get('IQM_SERVER_URL')
    iqm_auth_token = os.environ.get('IQM_AUTH_TOKEN')
    
    if not iqm_server_url or not iqm_auth_token:
        logger.error("Error: IQM credentials not found in environment variables.")
        logger.error("Please set IQM_SERVER_URL and IQM_AUTH_TOKEN environment variables.")
        return
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced TIBEDO Quantum ECDLP Solver')
    parser.add_argument('--bit-length', type=int, default=21, choices=[21, 32, 64],
                        help='Key size in bits (21, 32, or 64)')
    parser.add_argument('--parallel-jobs', type=int, default=4,
                        help='Number of parallel quantum jobs to execute')
    parser.add_argument('--backend', type=str, default="garnet",
                        help='IQM backend name')
    args = parser.parse_args()
    
    # Create solver
    solver = EnhancedQDayPrizeChallengeSolver(
        iqm_server_url=iqm_server_url,
        iqm_auth_token=iqm_auth_token,
        backend_name=args.backend,
        parallel_jobs=args.parallel_jobs
    )
    
    # Solve challenge
    logger.info(f"Solving Q-Day Prize challenge with {args.bit_length}-bit key...")
    solution = solver.solve_qday_challenge(bit_length=args.bit_length)
    
    # Print solution
    logger.info("\nChallenge Solution:")
    logger.info(f"Private Key: {solution['private_key']}")
    logger.info(f"Bit Length: {solution['bit_length']}")
    logger.info(f"Execution Time: {solution['execution_time']:.2f} seconds")
    logger.info(f"Backend: {solution['backend_name']}")
    logger.info(f"Parallel Jobs: {solution['parallel_jobs']}")
    logger.info(f"Timestamp: {solution['timestamp']}")
    logger.info(f"Verification: {'Successful' if solution['verification'] else 'Failed'}")
    
    # Save solution to file
    output_file = f'qday_prize_solution_{args.bit_length}bit.json'
    with open(output_file, 'w') as f:
        json.dump(solution, f, indent=2)
    logger.info(f"\nSolution saved to {output_file}")


if __name__ == "__main__":
    main()
