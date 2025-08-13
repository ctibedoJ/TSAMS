"""
Test script for TIBEDO Quantum Error Mitigation module.

This script demonstrates the functionality of the quantum error mitigation
module, including error modeling, error correction, and adaptive error mitigation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import logging

# Add parent directory to path to import TIBEDO modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TIBEDO quantum error mitigation module
from quantum_information_new.quantum_error_mitigation import (
    SpinorErrorModel,
    PhaseSynchronizedErrorCorrection,
    AdaptiveErrorMitigation
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

def create_ghz_circuit(num_qubits=5):
    """
    Create a GHZ state preparation circuit.
    
    Args:
        num_qubits: Number of qubits in the circuit
        
    Returns:
        GHZ circuit
    """
    # Create quantum circuit
    qr = QuantumRegister(num_qubits, 'q')
    cr = ClassicalRegister(num_qubits, 'c')
    circuit = QuantumCircuit(qr, cr, name="GHZ")
    
    # Prepare GHZ state
    circuit.h(qr[0])
    for i in range(1, num_qubits):
        circuit.cx(qr[0], qr[i])
    
    # Add measurements
    circuit.measure(qr, cr)
    
    return circuit

def create_bell_state_circuit():
    """
    Create a Bell state preparation circuit.
    
    Returns:
        Bell state circuit
    """
    # Create quantum circuit
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(2, 'c')
    circuit = QuantumCircuit(qr, cr, name="Bell")
    
    # Prepare Bell state
    circuit.h(qr[0])
    circuit.cx(qr[0], qr[1])
    
    # Add measurements
    circuit.measure(qr, cr)
    
    return circuit

def test_spinor_error_model():
    """
    Test the SpinorErrorModel class.
    """
    logger.info("Testing SpinorErrorModel...")
    
    # Create test circuits
    test_circuit = create_test_circuit(num_qubits=5, depth=5)
    ghz_circuit = create_ghz_circuit(num_qubits=5)
    bell_circuit = create_bell_state_circuit()
    
    # Create error model
    error_model = SpinorErrorModel(
        error_characterization_shots=1024,
        use_spinor_reduction=True,
        use_phase_synchronization=True,
        use_prime_indexing=True
    )
    
    # Generate error model
    from qiskit import Aer
    backend = Aer.get_backend('qasm_simulator')
    error_params = error_model.generate_error_model(backend)
    
    logger.info(f"Generated error model for backend: {error_params['backend_name']}")
    logger.info(f"Number of qubits: {error_params['num_qubits']}")
    
    # Test error simulation on different circuits
    for circuit, name in [(test_circuit, "Test Circuit"), 
                         (ghz_circuit, "GHZ Circuit"), 
                         (bell_circuit, "Bell State Circuit")]:
        logger.info(f"\nSimulating errors for {name}...")
        
        # Simulate errors
        error_simulation = error_model.simulate_errors(circuit)
        
        # Print simulation results
        logger.info(f"Error simulation results:")
        logger.info(f"  Single-qubit error probability: {error_simulation['single_qubit_error_prob']:.6f}")
        logger.info(f"  Two-qubit error probability: {error_simulation['two_qubit_error_prob']:.6f}")
        logger.info(f"  Measurement error probability: {error_simulation['measurement_error_prob']:.6f}")
        logger.info(f"  Total error probability: {error_simulation['total_error_prob']:.6f}")
        logger.info(f"  Fidelity: {error_simulation['fidelity']:.6f}")
        
        # Analyze error propagation
        error_propagation = error_model.analyze_error_propagation(circuit)
        
        logger.info(f"Error propagation analysis:")
        logger.info(f"  Critical qubits: {error_propagation['critical_qubits']}")
        logger.info(f"  Vulnerable qubits: {error_propagation['vulnerable_qubits']}")
        
        # Identify error-sensitive components
        error_sensitive = error_model.identify_error_sensitive_components(circuit)
        
        logger.info(f"Error-sensitive components:")
        logger.info(f"  Sensitive qubits: {error_sensitive['sensitive_qubits']}")
        if error_sensitive['sensitive_instructions']:
            logger.info(f"  Most sensitive instruction: {error_sensitive['sensitive_instructions'][0]['gate']} on qubits {error_sensitive['sensitive_instructions'][0]['qubits']}")

def test_phase_synchronized_error_correction():
    """
    Test the PhaseSynchronizedErrorCorrection class.
    """
    logger.info("\nTesting PhaseSynchronizedErrorCorrection...")
    
    # Create error model
    error_model = SpinorErrorModel()
    
    # Generate error model
    from qiskit import Aer
    backend = Aer.get_backend('qasm_simulator')
    error_params = error_model.generate_error_model(backend)
    
    # Create error correction
    error_correction = PhaseSynchronizedErrorCorrection(
        code_distance=3,
        use_phase_synchronization=True,
        use_spinor_reduction=True,
        cyclotomic_conductor=56
    )
    
    # Generate error correction code
    code = error_correction.generate_error_correction_code(error_params)
    
    logger.info(f"Generated error correction code:")
    logger.info(f"  Type: {code['type']}")
    logger.info(f"  Distance: {code['distance']}")
    logger.info(f"  Physical qubits: {code['physical_qubits']}")
    logger.info(f"  Logical qubits: {code['logical_qubits']}")
    
    # Calculate code efficiency
    efficiency = error_correction.calculate_code_efficiency(code)
    
    logger.info(f"Code efficiency metrics:")
    logger.info(f"  Encoding rate: {efficiency['encoding_rate']:.6f}")
    logger.info(f"  Error correction capability: {efficiency['error_correction_capability']}")
    logger.info(f"  Overhead: {efficiency['overhead']:.6f}")
    logger.info(f"  Error threshold: {efficiency['error_threshold']:.6f}")
    
    # Test encoding and error correction
    # Create a simple state to encode
    state_circuit = QuantumCircuit(1)
    state_circuit.h(0)  # Create superposition state
    
    # Encode state
    encoded_circuit = error_correction.encode_quantum_state(state_circuit)
    
    logger.info(f"Encoded circuit: {encoded_circuit.num_qubits} qubits, {encoded_circuit.depth()} depth")
    
    # Detect errors
    error_detection = error_correction.detect_errors(encoded_circuit, backend)
    
    logger.info(f"Error detection results:")
    logger.info(f"  Most likely syndrome: {error_detection['most_likely_syndrome']}")
    logger.info(f"  Errors detected: {error_detection['errors_detected']}")
    
    # Correct errors
    corrected_circuit = error_correction.correct_errors(encoded_circuit, error_detection['most_likely_syndrome'])
    
    logger.info(f"Corrected circuit: {corrected_circuit.num_qubits} qubits, {corrected_circuit.depth()} depth")

def test_adaptive_error_mitigation():
    """
    Test the AdaptiveErrorMitigation class.
    """
    logger.info("\nTesting AdaptiveErrorMitigation...")
    
    # Create test circuits
    test_circuit = create_test_circuit(num_qubits=3, depth=3)  # Smaller circuit for faster testing
    bell_circuit = create_bell_state_circuit()
    
    # Create error model
    error_model = SpinorErrorModel()
    
    # Generate error model
    from qiskit import Aer
    backend = Aer.get_backend('qasm_simulator')
    error_params = error_model.generate_error_model(backend)
    
    # Create error mitigation
    error_mitigation = AdaptiveErrorMitigation(
        error_model=error_model,
        use_zero_noise_extrapolation=True,
        use_probabilistic_error_cancellation=True,
        use_measurement_mitigation=True
    )
    
    # Test error mitigation on different circuits
    for circuit, name in [(test_circuit, "Test Circuit"), 
                         (bell_circuit, "Bell State Circuit")]:
        logger.info(f"\nApplying error mitigation to {name}...")
        
        # Analyze circuit error profile
        error_profile = error_mitigation.analyze_circuit_error_profile(circuit)
        
        logger.info(f"Error profile analysis:")
        logger.info(f"  Total error probability: {error_profile['error_simulation']['total_error_prob']:.6f}")
        logger.info(f"  Fidelity: {error_profile['error_simulation']['fidelity']:.6f}")
        logger.info(f"  Critical qubits: {error_profile['error_propagation']['critical_qubits']}")
        logger.info(f"  Sensitive qubits: {error_profile['error_sensitive_components']['sensitive_qubits']}")
        
        # Select mitigation strategy
        strategies = error_mitigation.select_mitigation_strategy(circuit, error_profile)
        
        logger.info(f"Selected mitigation strategies: {strategies}")
        
        # Apply mitigation strategy
        mitigation_results = error_mitigation.apply_mitigation_strategy(circuit, strategies, backend)
        
        logger.info(f"Applied mitigation strategies:")
        for strategy in strategies:
            logger.info(f"  {strategy}: {strategy in mitigation_results['mitigated_results']}")
        
        # Evaluate mitigation effectiveness
        effectiveness = error_mitigation.evaluate_mitigation_effectiveness(circuit, mitigation_results)
        
        logger.info(f"Mitigation effectiveness:")
        logger.info(f"  Fidelity: {effectiveness['fidelity']:.6f}")
        logger.info(f"  Unmitigated fidelity: {effectiveness['unmitigated_fidelity']:.6f}")
        logger.info(f"  Improvement: {effectiveness['improvement']:.6f}")
        
        # Test adaptive strategy
        adaptive_results = error_mitigation.adapt_strategy_during_execution(circuit, backend)
        
        logger.info(f"Adaptive mitigation results:")
        logger.info(f"  Initial strategies: {adaptive_results['initial_strategies']}")
        logger.info(f"  Final strategies: {adaptive_results['final_strategies']}")
        logger.info(f"  Adaptation successful: {adaptive_results['adaptation_successful']}")
        logger.info(f"  Initial fidelity: {adaptive_results['initial_effectiveness']['fidelity']:.6f}")
        if 'alternative_effectiveness' in adaptive_results:
            logger.info(f"  Alternative fidelity: {adaptive_results['alternative_effectiveness']['fidelity']:.6f}")

def test_combined_error_mitigation():
    """
    Test combined error mitigation approach.
    """
    logger.info("\nTesting combined error mitigation approach...")
    
    # Create a circuit with error-prone characteristics
    circuit = create_test_circuit(num_qubits=3, depth=5)
    
    # Create error model
    error_model = SpinorErrorModel()
    
    # Generate error model
    from qiskit import Aer
    backend = Aer.get_backend('qasm_simulator')
    error_params = error_model.generate_error_model(backend)
    
    # Create error correction
    error_correction = PhaseSynchronizedErrorCorrection(
        code_distance=3,
        use_phase_synchronization=True,
        use_spinor_reduction=True
    )
    
    # Generate error correction code
    code = error_correction.generate_error_correction_code(error_params)
    
    # Create error mitigation
    error_mitigation = AdaptiveErrorMitigation(
        error_model=error_model,
        use_zero_noise_extrapolation=True,
        use_probabilistic_error_cancellation=True,
        use_measurement_mitigation=True
    )
    
    # Approach 1: Apply error correction first, then error mitigation
    logger.info("Approach 1: Error correction followed by error mitigation")
    
    # Create a simple state to encode
    state_circuit = QuantumCircuit(1)
    state_circuit.h(0)  # Create superposition state
    
    # Encode state
    encoded_circuit = error_correction.encode_quantum_state(state_circuit)
    
    # Apply error mitigation to encoded circuit
    error_profile = error_mitigation.analyze_circuit_error_profile(encoded_circuit)
    strategies = error_mitigation.select_mitigation_strategy(encoded_circuit, error_profile)
    mitigation_results = error_mitigation.apply_mitigation_strategy(encoded_circuit, strategies, backend)
    
    # Evaluate effectiveness
    effectiveness1 = error_mitigation.evaluate_mitigation_effectiveness(encoded_circuit, mitigation_results)
    
    logger.info(f"Approach 1 effectiveness:")
    logger.info(f"  Fidelity: {effectiveness1['fidelity']:.6f}")
    logger.info(f"  Unmitigated fidelity: {effectiveness1['unmitigated_fidelity']:.6f}")
    logger.info(f"  Improvement: {effectiveness1['improvement']:.6f}")
    
    # Approach 2: Apply error mitigation directly to the circuit
    logger.info("\nApproach 2: Direct error mitigation")
    
    # Apply error mitigation to original circuit
    error_profile = error_mitigation.analyze_circuit_error_profile(circuit)
    strategies = error_mitigation.select_mitigation_strategy(circuit, error_profile)
    mitigation_results = error_mitigation.apply_mitigation_strategy(circuit, strategies, backend)
    
    # Evaluate effectiveness
    effectiveness2 = error_mitigation.evaluate_mitigation_effectiveness(circuit, mitigation_results)
    
    logger.info(f"Approach 2 effectiveness:")
    logger.info(f"  Fidelity: {effectiveness2['fidelity']:.6f}")
    logger.info(f"  Unmitigated fidelity: {effectiveness2['unmitigated_fidelity']:.6f}")
    logger.info(f"  Improvement: {effectiveness2['improvement']:.6f}")
    
    # Compare approaches
    logger.info("\nComparison of approaches:")
    logger.info(f"  Approach 1 (correction + mitigation) fidelity: {effectiveness1['fidelity']:.6f}")
    logger.info(f"  Approach 2 (mitigation only) fidelity: {effectiveness2['fidelity']:.6f}")
    
    if effectiveness1['fidelity'] > effectiveness2['fidelity']:
        logger.info("  Conclusion: Approach 1 (correction + mitigation) is more effective")
    else:
        logger.info("  Conclusion: Approach 2 (mitigation only) is more effective")

def main():
    """
    Main function to run all tests.
    """
    logger.info("Starting TIBEDO Quantum Error Mitigation tests...")
    
    # Test spinor error model
    test_spinor_error_model()
    
    # Test phase synchronized error correction
    test_phase_synchronized_error_correction()
    
    # Test adaptive error mitigation
    test_adaptive_error_mitigation()
    
    # Test combined error mitigation
    test_combined_error_mitigation()
    
    logger.info("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()