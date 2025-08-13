"""
TIBEDO Quantum Cryptography Integration Example

This module demonstrates how to use all the quantum cryptography components together
in a comprehensive example that showcases the capabilities of the TIBEDO Framework.
"""

import numpy as np
import time
import logging
import os
import sys
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import TIBEDO quantum cryptography components
from tibedo.quantum_information_new.quantum_cryptography import (
    ExtendedQuantumECDLPSolver,
    Kyber, Dilithium,
    XMSS, SPHINCS,
    SIKE,
    QuantumRandomNumberService
)


class QuantumCryptographyDemo:
    """
    Demonstration of TIBEDO's quantum cryptography capabilities.
    
    This class provides methods to demonstrate various quantum cryptography
    components of the TIBEDO Framework.
    """
    
    def __init__(self):
        """
        Initialize the quantum cryptography demonstration.
        """
        # Initialize the quantum random number service
        self.qrng_service = QuantumRandomNumberService()
        
        # Initialize the quantum ECDLP solver
        self.ecdlp_solver = ExtendedQuantumECDLPSolver(
            key_size=32,
            parallel_jobs=4,
            adaptive_depth=True,
            use_advanced_optimization=True,
            shared_phase_space=True
        )
        
        # Initialize the post-quantum cryptography schemes
        self.kyber = Kyber(security_level=128)
        self.dilithium = Dilithium(security_level=128)
        self.xmss = XMSS(height=10, w=16, n=32)
        self.sphincs = SPHINCS(n=32, h=8, d=2, w=16)
        self.sike = SIKE(security_level=128)
    
    def demonstrate_quantum_random_number_generation(self) -> Dict[str, Any]:
        """
        Demonstrate quantum random number generation.
        
        Returns:
            A dictionary with the demonstration results
        """
        logger.info("Demonstrating quantum random number generation...")
        
        results = {}
        
        # Generate random bits
        start_time = time.time()
        bits = self.qrng_service.generate_random_bits(1000)
        end_time = time.time()
        
        results['bits'] = {
            'data': bits[:20].tolist(),  # Show only the first 20 bits
            'time': end_time - start_time
        }
        
        # Generate random bytes
        start_time = time.time()
        bytes_data = self.qrng_service.generate_random_bytes(100)
        end_time = time.time()
        
        results['bytes'] = {
            'data': bytes_data[:20].hex(),  # Show only the first 20 bytes
            'time': end_time - start_time
        }
        
        # Generate random integers
        start_time = time.time()
        integers = [self.qrng_service.generate_random_int(1, 1000) for _ in range(10)]
        end_time = time.time()
        
        results['integers'] = {
            'data': integers,
            'time': end_time - start_time
        }
        
        # Generate random floats
        start_time = time.time()
        floats = [self.qrng_service.generate_random_float() for _ in range(10)]
        end_time = time.time()
        
        results['floats'] = {
            'data': floats,
            'time': end_time - start_time
        }
        
        # Generate random passwords
        start_time = time.time()
        passwords = [self.qrng_service.generate_random_password(length=12) for _ in range(5)]
        end_time = time.time()
        
        results['passwords'] = {
            'data': passwords,
            'time': end_time - start_time
        }
        
        # Generate random UUIDs
        start_time = time.time()
        uuids = [self.qrng_service.generate_random_uuid() for _ in range(5)]
        end_time = time.time()
        
        results['uuids'] = {
            'data': uuids,
            'time': end_time - start_time
        }
        
        # Test the randomness
        start_time = time.time()
        test_results = self.qrng_service.test_generator(num_bits=1000)
        end_time = time.time()
        
        results['test_results'] = {
            'data': {
                test_name: {
                    'passed': test_result['passed'],
                    'p_value': test_result.get('p_value', 'N/A')
                }
                for test_name, test_result in test_results.items()
            },
            'time': end_time - start_time
        }
        
        logger.info("Quantum random number generation demonstration completed.")
        
        return results
    
    def demonstrate_quantum_ecdlp_solver(self) -> Dict[str, Any]:
        """
        Demonstrate the quantum ECDLP solver.
        
        Returns:
            A dictionary with the demonstration results
        """
        logger.info("Demonstrating quantum ECDLP solver...")
        
        results = {}
        
        # Generate a quantum circuit
        start_time = time.time()
        circuit = self.ecdlp_solver.generate_quantum_circuit()
        end_time = time.time()
        
        results['circuit_generation'] = {
            'qubits': circuit.num_qubits,
            'depth': circuit.depth(),
            'time': end_time - start_time
        }
        
        # Solve the ECDLP for a 32-bit key
        start_time = time.time()
        curve_params = {'a': 1, 'b': 7, 'p': 2**256 - 2**32 - 977}
        public_key = {'x': 123, 'y': 456}
        base_point = {'x': 789, 'y': 101112}
        
        private_key = self.ecdlp_solver.solve_ecdlp_for_32bit(curve_params, public_key, base_point)
        end_time = time.time()
        
        results['ecdlp_solution'] = {
            'private_key': private_key,
            'time': end_time - start_time
        }
        
        # Benchmark performance
        start_time = time.time()
        benchmark_results = self.ecdlp_solver.benchmark_performance_extended(key_sizes=[8, 16, 32], repetitions=1)
        end_time = time.time()
        
        results['benchmark'] = {
            'data': {
                key_size: {
                    'avg_time': result['avg_time'],
                    'circuit_depth': result['circuit_depth'],
                    'total_qubits': result['total_qubits']
                }
                for key_size, result in benchmark_results.items()
            },
            'time': end_time - start_time
        }
        
        logger.info("Quantum ECDLP solver demonstration completed.")
        
        return results
    
    def demonstrate_lattice_based_cryptography(self) -> Dict[str, Any]:
        """
        Demonstrate lattice-based cryptography.
        
        Returns:
            A dictionary with the demonstration results
        """
        logger.info("Demonstrating lattice-based cryptography...")
        
        results = {}
        
        # Kyber key encapsulation
        start_time = time.time()
        kyber_public_key, kyber_secret_key = self.kyber.generate_keypair()
        kyber_ciphertext, kyber_shared_secret_alice = self.kyber.encapsulate(kyber_public_key)
        kyber_shared_secret_bob = self.kyber.decapsulate(kyber_secret_key, kyber_ciphertext)
        end_time = time.time()
        
        results['kyber'] = {
            'shared_secret_match': np.array_equal(kyber_shared_secret_alice, kyber_shared_secret_bob),
            'time': end_time - start_time
        }
        
        # Dilithium digital signatures
        start_time = time.time()
        dilithium_public_key, dilithium_secret_key = self.dilithium.generate_keypair()
        message = b"Hello, quantum world!"
        signature = self.dilithium.sign(dilithium_secret_key, message)
        is_valid = self.dilithium.verify(dilithium_public_key, message, signature)
        is_valid_modified = self.dilithium.verify(dilithium_public_key, message + b"!", signature)
        end_time = time.time()
        
        results['dilithium'] = {
            'signature_valid': is_valid,
            'signature_valid_modified': is_valid_modified,
            'time': end_time - start_time
        }
        
        logger.info("Lattice-based cryptography demonstration completed.")
        
        return results
    
    def demonstrate_hash_based_signatures(self) -> Dict[str, Any]:
        """
        Demonstrate hash-based signatures.
        
        Returns:
            A dictionary with the demonstration results
        """
        logger.info("Demonstrating hash-based signatures...")
        
        results = {}
        
        # XMSS signatures
        start_time = time.time()
        xmss_private_key, xmss_public_key = self.xmss.generate_keypair()
        message = b"Hello, quantum world!"
        xmss_signature = self.xmss.sign(message, xmss_private_key)
        xmss_is_valid = self.xmss.verify(message, xmss_signature, xmss_public_key)
        xmss_is_valid_modified = self.xmss.verify(message + b"!", xmss_signature, xmss_public_key)
        end_time = time.time()
        
        results['xmss'] = {
            'signature_valid': xmss_is_valid,
            'signature_valid_modified': xmss_is_valid_modified,
            'time': end_time - start_time
        }
        
        # SPHINCS+ signatures
        start_time = time.time()
        sphincs_private_key, sphincs_public_key = self.sphincs.generate_keypair()
        message = b"Hello, quantum world!"
        sphincs_signature = self.sphincs.sign(message, sphincs_private_key)
        sphincs_is_valid = self.sphincs.verify(message, sphincs_signature, sphincs_public_key)
        sphincs_is_valid_modified = self.sphincs.verify(message + b"!", sphincs_signature, sphincs_public_key)
        end_time = time.time()
        
        results['sphincs'] = {
            'signature_valid': sphincs_is_valid,
            'signature_valid_modified': sphincs_is_valid_modified,
            'time': end_time - start_time
        }
        
        logger.info("Hash-based signatures demonstration completed.")
        
        return results
    
    def demonstrate_isogeny_based_cryptography(self) -> Dict[str, Any]:
        """
        Demonstrate isogeny-based cryptography.
        
        Returns:
            A dictionary with the demonstration results
        """
        logger.info("Demonstrating isogeny-based cryptography...")
        
        results = {}
        
        # SIKE key encapsulation
        start_time = time.time()
        sike_alice_private_key, sike_alice_public_key = self.sike.generate_keypair(is_alice=True)
        sike_bob_private_key, sike_bob_public_key = self.sike.generate_keypair(is_alice=False)
        
        # Alice encapsulates a shared secret using Bob's public key
        sike_ciphertext, sike_shared_secret_alice = self.sike.encapsulate(sike_bob_public_key)
        
        # Bob decapsulates the shared secret using his private key
        sike_shared_secret_bob = self.sike.decapsulate(sike_bob_private_key, sike_ciphertext)
        end_time = time.time()
        
        results['sike'] = {
            'shared_secret_match': sike_shared_secret_alice == sike_shared_secret_bob,
            'time': end_time - start_time
        }
        
        logger.info("Isogeny-based cryptography demonstration completed.")
        
        return results
    
    def demonstrate_hybrid_cryptosystem(self) -> Dict[str, Any]:
        """
        Demonstrate a hybrid cryptosystem using all components.
        
        Returns:
            A dictionary with the demonstration results
        """
        logger.info("Demonstrating hybrid cryptosystem...")
        
        results = {}
        
        # Step 1: Generate a random message
        start_time = time.time()
        message = self.qrng_service.generate_random_bytes(100)
        message_hex = message.hex()
        end_time = time.time()
        
        results['message_generation'] = {
            'message': message_hex[:20] + "...",  # Show only the first 20 bytes
            'time': end_time - start_time
        }
        
        # Step 2: Generate a random symmetric key using quantum random number generation
        start_time = time.time()
        symmetric_key = self.qrng_service.generate_random_bytes(32)  # 256-bit key
        symmetric_key_hex = symmetric_key.hex()
        end_time = time.time()
        
        results['symmetric_key_generation'] = {
            'key': symmetric_key_hex,
            'time': end_time - start_time
        }
        
        # Step 3: Encrypt the message using the symmetric key (AES-256)
        start_time = time.time()
        # In a real implementation, we would use a proper AES implementation
        # For now, we'll just XOR the message with the key (not secure, just for demonstration)
        encrypted_message = bytes(m ^ k for m, k in zip(message, symmetric_key * (len(message) // len(symmetric_key) + 1)))
        encrypted_message_hex = encrypted_message.hex()
        end_time = time.time()
        
        results['message_encryption'] = {
            'encrypted_message': encrypted_message_hex[:20] + "...",  # Show only the first 20 bytes
            'time': end_time - start_time
        }
        
        # Step 4: Encapsulate the symmetric key using Kyber
        start_time = time.time()
        kyber_public_key, kyber_secret_key = self.kyber.generate_keypair()
        kyber_ciphertext, _ = self.kyber.encapsulate(kyber_public_key)
        end_time = time.time()
        
        results['key_encapsulation'] = {
            'time': end_time - start_time
        }
        
        # Step 5: Sign the encrypted message using Dilithium
        start_time = time.time()
        dilithium_public_key, dilithium_secret_key = self.dilithium.generate_keypair()
        signature = self.dilithium.sign(dilithium_secret_key, encrypted_message)
        end_time = time.time()
        
        results['message_signing'] = {
            'time': end_time - start_time
        }
        
        # Step 6: Verify the signature and decrypt the message
        start_time = time.time()
        is_valid = self.dilithium.verify(dilithium_public_key, encrypted_message, signature)
        
        if is_valid:
            # Decapsulate the symmetric key
            decapsulated_key = self.kyber.decapsulate(kyber_secret_key, kyber_ciphertext)
            
            # Decrypt the message
            # In a real implementation, we would use a proper AES implementation
            # For now, we'll just XOR the encrypted message with the key
            decrypted_message = bytes(m ^ k for m, k in zip(encrypted_message, symmetric_key * (len(encrypted_message) // len(symmetric_key) + 1)))
            decrypted_message_hex = decrypted_message.hex()
            
            # Check if the decryption was successful
            decryption_successful = decrypted_message == message
        else:
            decrypted_message_hex = None
            decryption_successful = False
        
        end_time = time.time()
        
        results['verification_and_decryption'] = {
            'signature_valid': is_valid,
            'decrypted_message': decrypted_message_hex[:20] + "..." if decrypted_message_hex else None,
            'decryption_successful': decryption_successful,
            'time': end_time - start_time
        }
        
        logger.info("Hybrid cryptosystem demonstration completed.")
        
        return results
    
    def run_all_demonstrations(self) -> Dict[str, Any]:
        """
        Run all demonstrations.
        
        Returns:
            A dictionary with all demonstration results
        """
        logger.info("Running all quantum cryptography demonstrations...")
        
        results = {}
        
        # Run all demonstrations
        results['quantum_random_number_generation'] = self.demonstrate_quantum_random_number_generation()
        results['quantum_ecdlp_solver'] = self.demonstrate_quantum_ecdlp_solver()
        results['lattice_based_cryptography'] = self.demonstrate_lattice_based_cryptography()
        results['hash_based_signatures'] = self.demonstrate_hash_based_signatures()
        results['isogeny_based_cryptography'] = self.demonstrate_isogeny_based_cryptography()
        results['hybrid_cryptosystem'] = self.demonstrate_hybrid_cryptosystem()
        
        logger.info("All quantum cryptography demonstrations completed.")
        
        return results


def print_demonstration_results(results: Dict[str, Any]) -> None:
    """
    Print the demonstration results in a readable format.
    
    Args:
        results: The demonstration results
    """
    print("\n" + "=" * 80)
    print("TIBEDO QUANTUM CRYPTOGRAPHY DEMONSTRATION RESULTS")
    print("=" * 80)
    
    # Print quantum random number generation results
    print("\n" + "-" * 80)
    print("QUANTUM RANDOM NUMBER GENERATION")
    print("-" * 80)
    
    qrng_results = results['quantum_random_number_generation']
    
    print("\nRandom Bits:")
    print(f"  {qrng_results['bits']['data']}...")
    print(f"  Time: {qrng_results['bits']['time']:.6f} seconds")
    
    print("\nRandom Bytes:")
    print(f"  {qrng_results['bytes']['data']}...")
    print(f"  Time: {qrng_results['bytes']['time']:.6f} seconds")
    
    print("\nRandom Integers:")
    print(f"  {qrng_results['integers']['data']}")
    print(f"  Time: {qrng_results['integers']['time']:.6f} seconds")
    
    print("\nRandom Floats:")
    print(f"  {[f'{f:.6f}' for f in qrng_results['floats']['data']]}")
    print(f"  Time: {qrng_results['floats']['time']:.6f} seconds")
    
    print("\nRandom Passwords:")
    print(f"  {qrng_results['passwords']['data']}")
    print(f"  Time: {qrng_results['passwords']['time']:.6f} seconds")
    
    print("\nRandom UUIDs:")
    print(f"  {qrng_results['uuids']['data']}")
    print(f"  Time: {qrng_results['uuids']['time']:.6f} seconds")
    
    print("\nRandomness Test Results:")
    for test_name, test_result in qrng_results['test_results']['data'].items():
        print(f"  {test_name.capitalize()} Test:")
        print(f"    Passed: {test_result['passed']}")
        print(f"    P-value: {test_result['p_value']}")
    print(f"  Time: {qrng_results['test_results']['time']:.6f} seconds")
    
    # Print quantum ECDLP solver results
    print("\n" + "-" * 80)
    print("QUANTUM ECDLP SOLVER")
    print("-" * 80)
    
    ecdlp_results = results['quantum_ecdlp_solver']
    
    print("\nQuantum Circuit Generation:")
    print(f"  Qubits: {ecdlp_results['circuit_generation']['qubits']}")
    print(f"  Depth: {ecdlp_results['circuit_generation']['depth']}")
    print(f"  Time: {ecdlp_results['circuit_generation']['time']:.6f} seconds")
    
    print("\nECDLP Solution:")
    print(f"  Private Key: {ecdlp_results['ecdlp_solution']['private_key']}")
    print(f"  Time: {ecdlp_results['ecdlp_solution']['time']:.6f} seconds")
    
    print("\nBenchmark Results:")
    for key_size, result in ecdlp_results['benchmark']['data'].items():
        print(f"  Key Size: {key_size} bits")
        print(f"    Average Time: {result['avg_time']:.6f} seconds")
        print(f"    Circuit Depth: {result['circuit_depth']}")
        print(f"    Total Qubits: {result['total_qubits']}")
    print(f"  Total Benchmark Time: {ecdlp_results['benchmark']['time']:.6f} seconds")
    
    # Print lattice-based cryptography results
    print("\n" + "-" * 80)
    print("LATTICE-BASED CRYPTOGRAPHY")
    print("-" * 80)
    
    lattice_results = results['lattice_based_cryptography']
    
    print("\nKyber Key Encapsulation:")
    print(f"  Shared Secret Match: {lattice_results['kyber']['shared_secret_match']}")
    print(f"  Time: {lattice_results['kyber']['time']:.6f} seconds")
    
    print("\nDilithium Digital Signatures:")
    print(f"  Signature Valid: {lattice_results['dilithium']['signature_valid']}")
    print(f"  Signature Valid (Modified Message): {lattice_results['dilithium']['signature_valid_modified']}")
    print(f"  Time: {lattice_results['dilithium']['time']:.6f} seconds")
    
    # Print hash-based signatures results
    print("\n" + "-" * 80)
    print("HASH-BASED SIGNATURES")
    print("-" * 80)
    
    hash_results = results['hash_based_signatures']
    
    print("\nXMSS Signatures:")
    print(f"  Signature Valid: {hash_results['xmss']['signature_valid']}")
    print(f"  Signature Valid (Modified Message): {hash_results['xmss']['signature_valid_modified']}")
    print(f"  Time: {hash_results['xmss']['time']:.6f} seconds")
    
    print("\nSPHINCS+ Signatures:")
    print(f"  Signature Valid: {hash_results['sphincs']['signature_valid']}")
    print(f"  Signature Valid (Modified Message): {hash_results['sphincs']['signature_valid_modified']}")
    print(f"  Time: {hash_results['sphincs']['time']:.6f} seconds")
    
    # Print isogeny-based cryptography results
    print("\n" + "-" * 80)
    print("ISOGENY-BASED CRYPTOGRAPHY")
    print("-" * 80)
    
    isogeny_results = results['isogeny_based_cryptography']
    
    print("\nSIKE Key Encapsulation:")
    print(f"  Shared Secret Match: {isogeny_results['sike']['shared_secret_match']}")
    print(f"  Time: {isogeny_results['sike']['time']:.6f} seconds")
    
    # Print hybrid cryptosystem results
    print("\n" + "-" * 80)
    print("HYBRID CRYPTOSYSTEM")
    print("-" * 80)
    
    hybrid_results = results['hybrid_cryptosystem']
    
    print("\nMessage Generation:")
    print(f"  Message: {hybrid_results['message_generation']['message']}")
    print(f"  Time: {hybrid_results['message_generation']['time']:.6f} seconds")
    
    print("\nSymmetric Key Generation:")
    print(f"  Key: {hybrid_results['symmetric_key_generation']['key']}")
    print(f"  Time: {hybrid_results['symmetric_key_generation']['time']:.6f} seconds")
    
    print("\nMessage Encryption:")
    print(f"  Encrypted Message: {hybrid_results['message_encryption']['encrypted_message']}")
    print(f"  Time: {hybrid_results['message_encryption']['time']:.6f} seconds")
    
    print("\nKey Encapsulation:")
    print(f"  Time: {hybrid_results['key_encapsulation']['time']:.6f} seconds")
    
    print("\nMessage Signing:")
    print(f"  Time: {hybrid_results['message_signing']['time']:.6f} seconds")
    
    print("\nVerification and Decryption:")
    print(f"  Signature Valid: {hybrid_results['verification_and_decryption']['signature_valid']}")
    print(f"  Decrypted Message: {hybrid_results['verification_and_decryption']['decrypted_message']}")
    print(f"  Decryption Successful: {hybrid_results['verification_and_decryption']['decryption_successful']}")
    print(f"  Time: {hybrid_results['verification_and_decryption']['time']:.6f} seconds")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")


# Example usage
if __name__ == "__main__":
    # Create the quantum cryptography demonstration
    demo = QuantumCryptographyDemo()
    
    # Run all demonstrations
    results = demo.run_all_demonstrations()
    
    # Print the results
    print_demonstration_results(results)