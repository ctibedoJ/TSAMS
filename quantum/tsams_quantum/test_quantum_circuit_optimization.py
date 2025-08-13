"""
Test script for TIBEDO Quantum Circuit Optimization module.

This script demonstrates the functionality of the quantum circuit optimization
module, including circuit compression, phase synchronization, and resource estimation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import logging

# Add parent directory to path to import TIBEDO modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TIBEDO quantum circuit optimization module
from quantum_information_new.quantum_circuit_optimization import (
    TibedoQuantumCircuitCompressor,
    PhaseSynchronizedGateSet,
    TibedoQuantumResourceEstimator
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_circuit(num_qubits=5, depth=5):
    """
    Create a test quantum circuit with specified number of qubits and depth.
    
    Args:
        num_qubits: Number of qubits in the circuit
        depth: Approximate circuit depth
        
    Returns:
        Test quantum circuit
    """
    # Create quantum circuit
    qr = QuantumRegister(num_qubits, 'q')
    cr = ClassicalRegister(num_qubits, 'c')
    circuit = QuantumCircuit(qr, cr)
    
    # Add gates to create a non-trivial circuit
    for d in range(depth):
        # Add single-qubit gates
        for i in range(num_qubits):
            circuit.h(qr[i])
            circuit.t(qr[i])
            circuit.rz(np.pi/4, qr[i])
        
        # Add two-qubit gates
        for i in range(num_qubits-1):
            circuit.cx(qr[i], qr[i+1])
        
        # Add phase gates
        for i in range(num_qubits):
            circuit.s(qr[i])
            circuit.rz(np.pi/8, qr[i])
    
    # Add measurements
    circuit.measure(qr, cr)
    
    return circuit

def create_qft_circuit(num_qubits=5):
    """
    Create a Quantum Fourier Transform circuit.
    
    Args:
        num_qubits: Number of qubits in the circuit
        
    Returns:
        QFT circuit
    """
    # Create quantum circuit
    qr = QuantumRegister(num_qubits, 'q')
    cr = ClassicalRegister(num_qubits, 'c')
    circuit = QuantumCircuit(qr, cr, name="QFT")
    
    # Apply QFT
    for i in range(num_qubits):
        circuit.h(qr[i])
        for j in range(i+1, num_qubits):
            circuit.cp(np.pi / float(2**(j-i)), qr[i], qr[j])
    
    # Swap qubits
    for i in range(num_qubits//2):
        circuit.swap(qr[i], qr[num_qubits-i-1])
    
    # Add measurements
    circuit.measure(qr, cr)
    
    return circuit

def create_grover_circuit(num_qubits=3):
    """
    Create a simple Grover's algorithm circuit.
    
    Args:
        num_qubits: Number of qubits in the circuit
        
    Returns:
        Grover circuit
    """
    # Create quantum circuit
    qr = QuantumRegister(num_qubits, 'q')
    cr = ClassicalRegister(num_qubits, 'c')
    circuit = QuantumCircuit(qr, cr, name="Grover")
    
    # Initialize in superposition
    circuit.h(qr)
    
    # Oracle (mark state |111...1>)
    circuit.x(qr[0])
    circuit.h(qr[-1])
    circuit.mcx(qr[:-1], qr[-1])
    circuit.h(qr[-1])
    circuit.x(qr[0])
    
    # Diffusion operator
    circuit.h(qr)
    circuit.x(qr)
    circuit.h(qr[-1])
    circuit.mcx(qr[:-1], qr[-1])
    circuit.h(qr[-1])
    circuit.x(qr)
    circuit.h(qr)
    
    # Add measurements
    circuit.measure(qr, cr)
    
    return circuit

def test_circuit_compression():
    """
    Test the TibedoQuantumCircuitCompressor class.
    """
    logger.info("Testing TibedoQuantumCircuitCompressor...")
    
    # Create test circuits
    test_circuit = create_test_circuit(num_qubits=5, depth=5)
    qft_circuit = create_qft_circuit(num_qubits=5)
    grover_circuit = create_grover_circuit(num_qubits=3)
    
    # Create circuit compressor
    compressor = TibedoQuantumCircuitCompressor(
        compression_level=2,
        preserve_measurement=True,
        use_spinor_reduction=True,
        use_phase_synchronization=True,
        use_prime_indexing=True
    )
    
    # Test compression on different circuits
    for circuit, name in [(test_circuit, "Test Circuit"), 
                         (qft_circuit, "QFT Circuit"), 
                         (grover_circuit, "Grover Circuit")]:
        logger.info(f"\nCompressing {name}...")
        
        # Analyze compression potential
        potential = compressor.analyze_compression_potential(circuit)
        logger.info(f"Compression potential analysis:")
        logger.info(f"  Original depth: {potential['original_depth']}")
        logger.info(f"  Original gates: {potential['original_gates']}")
        logger.info(f"  Potential depth reduction: {potential['potential_depth_reduction']:.2f}")
        logger.info(f"  Potential gate reduction: {potential['potential_gate_reduction']:.2f}")
        logger.info(f"  Estimated depth after: {potential['estimated_depth_after']}")
        logger.info(f"  Estimated gates after: {potential['estimated_gates_after']}")
        
        # Identify compression patterns
        patterns = compressor.identify_compression_patterns(circuit)
        logger.info(f"Compression patterns identified:")
        for pattern_type, pattern_list in patterns.items():
            logger.info(f"  {pattern_type}: {len(pattern_list)} instances")
        
        # Compress circuit
        compressed_circuit = compressor.compress_circuit(circuit)
        
        # Verify equivalence
        is_equivalent = compressor.verify_circuit_equivalence(circuit, compressed_circuit)
        logger.info(f"Circuits are equivalent: {is_equivalent}")
        
        # Print statistics
        original_depth = circuit.depth()
        original_gates = sum(circuit.count_ops().values())
        compressed_depth = compressed_circuit.depth()
        compressed_gates = sum(compressed_circuit.count_ops().values())
        
        logger.info(f"Compression results:")
        logger.info(f"  Original depth: {original_depth}")
        logger.info(f"  Compressed depth: {compressed_depth}")
        logger.info(f"  Depth reduction: {original_depth - compressed_depth} ({(original_depth - compressed_depth) / original_depth * 100:.2f}%)")
        logger.info(f"  Original gates: {original_gates}")
        logger.info(f"  Compressed gates: {compressed_gates}")
        logger.info(f"  Gate reduction: {original_gates - compressed_gates} ({(original_gates - compressed_gates) / original_gates * 100:.2f}%)")

def test_phase_synchronization():
    """
    Test the PhaseSynchronizedGateSet class.
    """
    logger.info("\nTesting PhaseSynchronizedGateSet...")
    
    # Create test circuits
    test_circuit = create_test_circuit(num_qubits=5, depth=5)
    qft_circuit = create_qft_circuit(num_qubits=5)
    
    # Create phase synchronizer
    phase_sync = PhaseSynchronizedGateSet(
        optimization_level=2,
        cyclotomic_conductor=56
    )
    
    # Test phase synchronization on different circuits
    for circuit, name in [(test_circuit, "Test Circuit"), 
                         (qft_circuit, "QFT Circuit")]:
        logger.info(f"\nOptimizing phase relations in {name}...")
        
        # Calculate phase efficiency
        efficiency = phase_sync.calculate_phase_efficiency(circuit)
        logger.info(f"Phase efficiency analysis:")
        logger.info(f"  Phase gate count: {efficiency['phase_gate_count']}")
        logger.info(f"  Rotation gate count: {efficiency['rotation_gate_count']}")
        logger.info(f"  Phase density: {efficiency['phase_density']:.2f}")
        logger.info(f"  Rotation density: {efficiency['rotation_density']:.2f}")
        logger.info(f"  Sequential phase count: {efficiency['sequential_phase_count']}")
        logger.info(f"  Commuting phase count: {efficiency['commuting_phase_count']}")
        logger.info(f"  Phase block count: {efficiency['phase_block_count']}")
        logger.info(f"  Rotation sequence count: {efficiency['rotation_sequence_count']}")
        logger.info(f"  Optimization potential: {efficiency['optimization_potential']:.2f}")
        
        # Identify phase patterns
        patterns = phase_sync.identify_phase_patterns(circuit)
        logger.info(f"Phase patterns identified:")
        for pattern_type, pattern_list in patterns.items():
            logger.info(f"  {pattern_type}: {len(pattern_list)} instances")
        
        # Optimize phase relations
        optimized_circuit = phase_sync.optimize_phase_relations(circuit)
        
        # Print statistics
        original_depth = circuit.depth()
        original_gates = sum(circuit.count_ops().values())
        optimized_depth = optimized_circuit.depth()
        optimized_gates = sum(optimized_circuit.count_ops().values())
        
        logger.info(f"Phase optimization results:")
        logger.info(f"  Original depth: {original_depth}")
        logger.info(f"  Optimized depth: {optimized_depth}")
        logger.info(f"  Depth reduction: {original_depth - optimized_depth} ({(original_depth - optimized_depth) / original_depth * 100:.2f}%)")
        logger.info(f"  Original gates: {original_gates}")
        logger.info(f"  Optimized gates: {optimized_gates}")
        logger.info(f"  Gate reduction: {original_gates - optimized_gates} ({(original_gates - optimized_gates) / original_gates * 100:.2f}%)")

def test_resource_estimation():
    """
    Test the TibedoQuantumResourceEstimator class.
    """
    logger.info("\nTesting TibedoQuantumResourceEstimator...")
    
    # Create test circuits
    test_circuit = create_test_circuit(num_qubits=5, depth=5)
    qft_circuit = create_qft_circuit(num_qubits=5)
    grover_circuit = create_grover_circuit(num_qubits=3)
    
    # Create resource estimator
    estimator = TibedoQuantumResourceEstimator(
        error_rate=0.001,
        connectivity='all-to-all',
        include_error_correction=True,
        error_correction_overhead=15.0
    )
    
    # Test resource estimation on different circuits
    for circuit, name in [(test_circuit, "Test Circuit"), 
                         (qft_circuit, "QFT Circuit"), 
                         (grover_circuit, "Grover Circuit")]:
        logger.info(f"\nEstimating resources for {name}...")
        
        # Generate resource report
        report = estimator.generate_resource_report(circuit)
        
        # Print report summary
        logger.info(f"Resource report summary for {report['circuit_name']}:")
        logger.info(f"  Qubit requirements:")
        logger.info(f"    Logical qubits: {report['qubit_requirements']['logical_qubits']}")
        logger.info(f"    Physical qubits: {report['qubit_requirements']['physical_qubits']}")
        logger.info(f"    Ancilla qubits: {report['qubit_requirements']['ancilla_qubits']}")
        logger.info(f"    Total qubits: {report['qubit_requirements']['total_qubits']}")
        
        logger.info(f"  Gate counts:")
        logger.info(f"    Single-qubit gates: {report['gate_counts']['single_qubit_gates']:.1f}")
        logger.info(f"    Two-qubit gates: {report['gate_counts']['two_qubit_gates']:.1f}")
        logger.info(f"    Multi-qubit gates: {report['gate_counts']['multi_qubit_gates']:.1f}")
        logger.info(f"    Additional gates for connectivity: {report['gate_counts']['additional_gates_for_connectivity']:.1f}")
        logger.info(f"    Total gates with overhead: {report['gate_counts']['total_gates_with_overhead']:.1f}")
        
        logger.info(f"  Circuit depth:")
        logger.info(f"    Original depth: {report['circuit_depth']['original_depth']}")
        logger.info(f"    Depth with connectivity: {report['circuit_depth']['depth_with_connectivity']}")
        logger.info(f"    Depth with error correction: {report['circuit_depth']['depth_with_error_correction']}")
        logger.info(f"    Critical path length: {report['circuit_depth']['critical_path_length']}")
        
        logger.info(f"  Error sensitivity:")
        logger.info(f"    Single-qubit error probability: {report['error_sensitivity']['single_qubit_error_prob']:.6f}")
        logger.info(f"    Two-qubit error probability: {report['error_sensitivity']['two_qubit_error_prob']:.6f}")
        logger.info(f"    Total error probability: {report['error_sensitivity']['total_error_prob']:.6f}")
        logger.info(f"    Success probability with correction: {report['error_sensitivity']['success_prob_with_correction']:.6f}")
        
        # Visualize report (uncomment to save visualization)
        # estimator.visualize_resource_report(report, save_path=f"{name.replace(' ', '_').lower()}_resources.png")

def test_combined_optimization():
    """
    Test combined optimization using both circuit compression and phase synchronization.
    """
    logger.info("\nTesting combined optimization...")
    
    # Create test circuit
    circuit = create_qft_circuit(num_qubits=5)
    logger.info(f"Original circuit: depth={circuit.depth()}, gates={sum(circuit.count_ops().values())}")
    
    # Create optimizers
    compressor = TibedoQuantumCircuitCompressor(compression_level=2)
    phase_sync = PhaseSynchronizedGateSet(optimization_level=2)
    
    # Apply phase synchronization first
    phase_optimized = phase_sync.optimize_phase_relations(circuit)
    logger.info(f"After phase optimization: depth={phase_optimized.depth()}, gates={sum(phase_optimized.count_ops().values())}")
    
    # Then apply circuit compression
    compressed = compressor.compress_circuit(phase_optimized)
    logger.info(f"After compression: depth={compressed.depth()}, gates={sum(compressed.count_ops().values())}")
    
    # Calculate overall improvement
    depth_reduction = (circuit.depth() - compressed.depth()) / circuit.depth() * 100
    gate_reduction = (sum(circuit.count_ops().values()) - sum(compressed.count_ops().values())) / sum(circuit.count_ops().values()) * 100
    
    logger.info(f"Overall improvement:")
    logger.info(f"  Depth reduction: {depth_reduction:.2f}%")
    logger.info(f"  Gate reduction: {gate_reduction:.2f}%")
    
    # Verify equivalence
    is_equivalent = compressor.verify_circuit_equivalence(circuit, compressed)
    logger.info(f"Circuits are equivalent: {is_equivalent}")

def main():
    """
    Main function to run all tests.
    """
    logger.info("Starting TIBEDO Quantum Circuit Optimization tests...")
    
    # Test circuit compression
    test_circuit_compression()
    
    # Test phase synchronization
    test_phase_synchronization()
    
    # Test resource estimation
    test_resource_estimation()
    
    # Test combined optimization
    test_combined_optimization()
    
    logger.info("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()