"""
Example script demonstrating TIBEDO Quantum Circuit Optimization with IQM quantum backend.

This script shows how to use the TIBEDO Quantum Circuit Optimization module
to optimize quantum circuits for the IQM quantum backend, building on the
existing quantum ECDLP solver for IQM quantum backends.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import logging
import time
import json

# Add parent directory to path to import TIBEDO modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TIBEDO quantum circuit optimization module
from quantum_information_new.quantum_circuit_optimization import (
    TibedoQuantumCircuitCompressor,
    PhaseSynchronizedGateSet,
    TibedoQuantumResourceEstimator
)

# Import TIBEDO quantum ECDLP solver for IQM
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from tibedo_quantum_ecdlp_iqm import TibedoQuantumECDLPSolver
    IQM_SOLVER_AVAILABLE = True
except ImportError:
    print("TIBEDO Quantum ECDLP Solver for IQM not found. Some functionality will be limited.")
    IQM_SOLVER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TibedoIQMOptimizer:
    """
    Optimizer for quantum circuits targeting IQM quantum backends.
    
    This class combines TIBEDO's quantum circuit optimization techniques
    with IQM-specific optimizations to create highly efficient circuits
    for IQM quantum hardware.
    """
    
    def __init__(self, 
                 iqm_server_url=None,
                 iqm_auth_token=None,
                 backend_name="garnet",
                 compression_level=2,
                 use_error_mitigation=True):
        """
        Initialize the TIBEDO IQM Optimizer.
        
        Args:
            iqm_server_url: URL of the IQM quantum server (optional)
            iqm_auth_token: Authentication token for IQM server (optional)
            backend_name: Name of the IQM backend to use (default: "garnet")
            compression_level: Level of circuit compression (1-3)
            use_error_mitigation: Whether to use error mitigation techniques
        """
        self.iqm_server_url = iqm_server_url
        self.iqm_auth_token = iqm_auth_token
        self.backend_name = backend_name
        self.compression_level = compression_level
        self.use_error_mitigation = use_error_mitigation
        
        # Initialize quantum circuit compressor
        self.compressor = TibedoQuantumCircuitCompressor(
            compression_level=compression_level,
            preserve_measurement=True,
            use_spinor_reduction=True,
            use_phase_synchronization=True,
            use_prime_indexing=True
        )
        
        # Initialize phase synchronizer
        self.phase_sync = PhaseSynchronizedGateSet(
            optimization_level=compression_level,
            cyclotomic_conductor=56
        )
        
        # Initialize resource estimator
        self.resource_estimator = TibedoQuantumResourceEstimator(
            error_rate=0.001,  # Typical error rate for IQM hardware
            connectivity='grid',  # IQM hardware typically has grid connectivity
            include_error_correction=False,  # We'll use error mitigation instead
            error_correction_overhead=1.0
        )
        
        # Initialize IQM solver if credentials are provided
        self.iqm_solver = None
        if IQM_SOLVER_AVAILABLE and iqm_server_url and iqm_auth_token:
            try:
                self.iqm_solver = TibedoQuantumECDLPSolver(
                    iqm_server_url=iqm_server_url,
                    iqm_auth_token=iqm_auth_token,
                    backend_name=backend_name,
                    shots=4096
                )
                logger.info(f"Connected to IQM backend: {backend_name}")
            except Exception as e:
                logger.warning(f"Failed to connect to IQM backend: {e}")
                self.iqm_solver = None
        
        logger.info(f"Initialized TIBEDO IQM Optimizer (compression level: {compression_level})")
    
    def optimize_circuit(self, circuit):
        """
        Optimize a quantum circuit for IQM quantum hardware.
        
        Args:
            circuit: Quantum circuit to optimize
            
        Returns:
            Optimized quantum circuit
        """
        logger.info(f"Optimizing circuit for IQM hardware (original depth: {circuit.depth()}, gates: {sum(circuit.count_ops().values())})")
        
        # Step 1: Apply phase synchronization
        phase_optimized = self.phase_sync.optimize_phase_relations(circuit)
        logger.info(f"After phase optimization: depth={phase_optimized.depth()}, gates={sum(phase_optimized.count_ops().values())}")
        
        # Step 2: Apply circuit compression
        compressed = self.compressor.compress_circuit(phase_optimized)
        logger.info(f"After compression: depth={compressed.depth()}, gates={sum(compressed.count_ops().values())}")
        
        # Step 3: Apply IQM-specific optimizations
        iqm_optimized = self._apply_iqm_specific_optimizations(compressed)
        logger.info(f"After IQM-specific optimizations: depth={iqm_optimized.depth()}, gates={sum(iqm_optimized.count_ops().values())}")
        
        # Calculate overall improvement
        depth_reduction = (circuit.depth() - iqm_optimized.depth()) / circuit.depth() * 100
        gate_reduction = (sum(circuit.count_ops().values()) - sum(iqm_optimized.count_ops().values())) / sum(circuit.count_ops().values()) * 100
        
        logger.info(f"Overall improvement:")
        logger.info(f"  Depth reduction: {depth_reduction:.2f}%")
        logger.info(f"  Gate reduction: {gate_reduction:.2f}%")
        
        # Verify equivalence
        is_equivalent = self.compressor.verify_circuit_equivalence(circuit, iqm_optimized)
        logger.info(f"Circuits are equivalent: {is_equivalent}")
        
        return iqm_optimized
    
    def _apply_iqm_specific_optimizations(self, circuit):
        """
        Apply IQM-specific optimizations to a quantum circuit.
        
        Args:
            circuit: Quantum circuit to optimize
            
        Returns:
            Optimized quantum circuit
        """
        # This is a placeholder for IQM-specific optimizations
        # In a real implementation, we would apply optimizations specific to IQM hardware
        # For now, we'll just return the original circuit
        
        # Make a copy of the circuit
        optimized = circuit.copy()
        
        # Apply standard Qiskit optimizations
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import Unroller, Optimize1qGates, CXCancellation
        
        # Create pass manager with standard optimization passes
        pass_manager = PassManager()
        pass_manager.append(Unroller(['u', 'cx']))
        pass_manager.append(Optimize1qGates())
        pass_manager.append(CXCancellation())
        
        # Apply optimization passes
        optimized = pass_manager.run(optimized)
        
        logger.info("Applied IQM-specific optimizations (placeholder implementation)")
        
        return optimized
    
    def estimate_resources(self, circuit):
        """
        Estimate resources required to run a circuit on IQM hardware.
        
        Args:
            circuit: Quantum circuit to analyze
            
        Returns:
            Resource report
        """
        # Generate resource report
        report = self.resource_estimator.generate_resource_report(circuit)
        
        # Add IQM-specific information
        report['iqm_backend'] = self.backend_name
        report['iqm_specific'] = {
            'estimated_runtime': self._estimate_runtime(circuit),
            'estimated_success_probability': self._estimate_success_probability(circuit),
            'recommended_shots': self._recommend_shots(circuit)
        }
        
        return report
    
    def _estimate_runtime(self, circuit):
        """
        Estimate runtime for a circuit on IQM hardware.
        
        Args:
            circuit: Quantum circuit to analyze
            
        Returns:
            Estimated runtime in seconds
        """
        # This is a simplified estimate based on circuit characteristics
        # In a real implementation, we would use more sophisticated models
        
        # Get gate counts
        gate_counts = circuit.count_ops()
        
        # Calculate total gates
        total_gates = sum(gate_counts.values())
        
        # Estimate runtime based on gate count and depth
        # Typical gate time on IQM hardware is around 100-200 ns
        gate_time_ns = 150  # nanoseconds
        
        # Runtime is approximately proportional to circuit depth
        depth = circuit.depth()
        
        # Estimate runtime in seconds
        runtime = depth * gate_time_ns * 1e-9
        
        # Add overhead for measurement and classical processing
        runtime += 0.1  # seconds
        
        return runtime
    
    def _estimate_success_probability(self, circuit):
        """
        Estimate success probability for a circuit on IQM hardware.
        
        Args:
            circuit: Quantum circuit to analyze
            
        Returns:
            Estimated success probability
        """
        # This is a simplified estimate based on circuit characteristics
        # In a real implementation, we would use more sophisticated models
        
        # Get gate counts
        gate_counts = circuit.count_ops()
        
        # Calculate total gates
        total_gates = sum(gate_counts.values())
        
        # Estimate error probability based on gate count
        # Typical error rate on IQM hardware is around 0.1-1%
        error_rate = 0.005  # 0.5%
        
        # Error probability increases with gate count
        error_prob = 1 - (1 - error_rate) ** total_gates
        
        # Success probability is complement of error probability
        success_prob = 1 - error_prob
        
        # If using error mitigation, success probability is higher
        if self.use_error_mitigation:
            success_prob = 1 - error_prob * 0.5  # Assume 50% error reduction
        
        return success_prob
    
    def _recommend_shots(self, circuit):
        """
        Recommend number of shots for a circuit on IQM hardware.
        
        Args:
            circuit: Quantum circuit to analyze
            
        Returns:
            Recommended number of shots
        """
        # This is a simplified recommendation based on circuit characteristics
        # In a real implementation, we would use more sophisticated models
        
        # Get number of qubits
        num_qubits = circuit.num_qubits
        
        # Get circuit depth
        depth = circuit.depth()
        
        # Base recommendation on circuit complexity
        if depth > 50 or num_qubits > 10:
            # Complex circuits need more shots
            return 8192
        elif depth > 20 or num_qubits > 5:
            # Moderately complex circuits
            return 4096
        else:
            # Simple circuits
            return 2048
    
    def run_on_iqm(self, circuit, shots=None):
        """
        Run a circuit on IQM quantum hardware.
        
        Args:
            circuit: Quantum circuit to run
            shots: Number of shots (if None, use recommended shots)
            
        Returns:
            Results from IQM hardware
        """
        if not self.iqm_solver:
            raise ValueError("IQM solver not initialized. Please provide IQM credentials.")
        
        # Optimize circuit for IQM hardware
        optimized_circuit = self.optimize_circuit(circuit)
        
        # Determine number of shots
        if shots is None:
            shots = self._recommend_shots(optimized_circuit)
        
        logger.info(f"Running circuit on IQM {self.backend_name} backend with {shots} shots...")
        
        # This is a placeholder for running on IQM hardware
        # In a real implementation, we would use the IQM SDK to run the circuit
        
        # For now, we'll just simulate the results
        from qiskit import Aer, execute
        
        # Use Qiskit's simulator
        simulator = Aer.get_backend('qasm_simulator')
        
        # Execute the circuit
        job = execute(optimized_circuit, simulator, shots=shots)
        
        # Get the results
        result = job.result()
        counts = result.get_counts()
        
        logger.info(f"Circuit execution completed. Results: {counts}")
        
        return {
            'counts': counts,
            'shots': shots,
            'backend': self.backend_name,
            'optimized_circuit': optimized_circuit,
            'execution_time': self._estimate_runtime(optimized_circuit),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }


def create_ecdlp_test_circuit(bit_length=8):
    """
    Create a test circuit for ECDLP solving.
    
    This is a simplified version of the circuit used in the TIBEDO Quantum ECDLP Solver.
    
    Args:
        bit_length: Length of the private key in bits
        
    Returns:
        Test quantum circuit for ECDLP
    """
    # Create quantum registers
    key_register = QuantumRegister(bit_length, name='k')
    ancilla_register = QuantumRegister(bit_length * 2, name='anc')
    result_register = QuantumRegister(1, name='res')
    classical_register = ClassicalRegister(bit_length, name='c')
    
    # Create quantum circuit
    circuit = QuantumCircuit(key_register, ancilla_register, result_register, classical_register, name="ECDLP_Test")
    
    # Step 1: Initialize key register in superposition
    circuit.h(key_register)
    
    # Step 2: Apply phase rotations
    for i, qubit in enumerate(key_register):
        phase = np.pi / (2 ** (i % 4 + 1))
        circuit.p(phase, qubit)
    
    # Step 3: Apply controlled operations between key and ancilla
    for i in range(bit_length):
        circuit.cx(key_register[i], ancilla_register[i])
    
    # Step 4: Apply Hadamard to ancilla
    circuit.h(ancilla_register[:bit_length])
    
    # Step 5: Apply controlled rotations
    for i in range(bit_length):
        circuit.cp(np.pi / (2 ** (i % 4 + 1)), key_register[i], ancilla_register[i + bit_length])
    
    # Step 6: Apply mixing operations
    for i in range(bit_length - 1):
        circuit.cswap(ancilla_register[i], key_register[i], key_register[i + 1])
    
    # Step 7: Initialize result qubit in superposition
    circuit.h(result_register)
    
    # Step 8: Apply controlled phase rotations
    for i in range(bit_length):
        g_phase = (i * np.pi) / bit_length
        circuit.cp(g_phase, key_register[i], result_register[0])
    
    # Step 9: Apply inverse operations
    for i in range(bit_length - 1, 0, -1):
        circuit.cswap(ancilla_register[i - 1], key_register[i - 1], key_register[i])
    
    for i in range(bit_length - 1, -1, -1):
        circuit.cp(-np.pi / (2 ** (i % 4 + 1)), key_register[i], ancilla_register[i + bit_length])
    
    circuit.h(ancilla_register[:bit_length])
    
    for i in range(bit_length - 1, -1, -1):
        circuit.cx(key_register[i], ancilla_register[i])
    
    # Step 10: Apply quantum Fourier transform
    for i in range(bit_length):
        circuit.h(key_register[i])
        for j in range(i + 1, bit_length):
            circuit.cp(np.pi / float(2 ** (j - i)), key_register[i], key_register[j])
    
    # Step 11: Swap qubits
    for i in range(bit_length // 2):
        circuit.swap(key_register[i], key_register[bit_length - i - 1])
    
    # Step 12: Measure key register
    circuit.measure(key_register, classical_register)
    
    return circuit

def example_ecdlp_optimization():
    """
    Example of optimizing an ECDLP circuit for IQM quantum hardware.
    """
    logger.info("Example: ECDLP Circuit Optimization for IQM Hardware")
    
    # Create ECDLP test circuit
    bit_length = 8  # Use 8 bits for testing
    circuit = create_ecdlp_test_circuit(bit_length)
    
    logger.info(f"Created ECDLP test circuit with {bit_length} bits")
    logger.info(f"Original circuit: depth={circuit.depth()}, gates={sum(circuit.count_ops().values())}")
    
    # Create IQM optimizer
    # Note: We're not providing actual IQM credentials here
    optimizer = TibedoIQMOptimizer(
        compression_level=2,
        use_error_mitigation=True
    )
    
    # Optimize circuit
    optimized_circuit = optimizer.optimize_circuit(circuit)
    
    # Estimate resources
    resources = optimizer.estimate_resources(optimized_circuit)
    
    # Print resource summary
    logger.info("\nResource estimates for optimized circuit:")
    logger.info(f"  Qubits: {resources['qubit_requirements']['logical_qubits']}")
    logger.info(f"  Circuit depth: {resources['circuit_depth']['original_depth']}")
    logger.info(f"  Total gates: {resources['gate_counts']['total_gates']}")
    logger.info(f"  Estimated runtime: {resources['iqm_specific']['estimated_runtime']:.6f} seconds")
    logger.info(f"  Estimated success probability: {resources['iqm_specific']['estimated_success_probability']:.6f}")
    logger.info(f"  Recommended shots: {resources['iqm_specific']['recommended_shots']}")
    
    # Simulate execution on IQM hardware
    try:
        results = optimizer.run_on_iqm(circuit)
        logger.info("\nSimulated execution results:")
        logger.info(f"  Most frequent result: {max(results['counts'].items(), key=lambda x: x[1])[0]}")
        logger.info(f"  Execution time: {results['execution_time']:.6f} seconds")
    except ValueError as e:
        logger.info(f"\nSkipping IQM execution: {e}")
        logger.info("  To run on actual IQM hardware, provide valid IQM credentials.")

def main():
    """
    Main function to run the example.
    """
    logger.info("Starting TIBEDO Quantum Circuit Optimization for IQM example...")
    
    # Run ECDLP optimization example
    example_ecdlp_optimization()
    
    logger.info("\nExample completed successfully!")

if __name__ == "__main__":
    main()