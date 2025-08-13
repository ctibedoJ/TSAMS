"""
Basic usage examples for the TSAMS Prime Generator package.
"""

import matplotlib.pyplot as plt
import numpy as np

from prime_generator import (
    sieve_of_eratosthenes,
    cyclotomic_sieve,
    quantum_prime_generator,
    e8_lattice_sieve,
    modular_forms_prime_test,
    l_function_prime_test
)

# Generate primes using different methods
limit = 100
print(f"Generating primes up to {limit}...")

# Classical method
classical_primes = sieve_of_eratosthenes(limit)
print(f"Classical sieve found {len(classical_primes)} primes: {classical_primes}")

# TSAMS methods
cyclotomic_primes = cyclotomic_sieve(limit, conductor=8)
print(f"Cyclotomic sieve found {len(cyclotomic_primes)} primes: {cyclotomic_primes}")

quantum_primes = quantum_prime_generator(limit, qubits=4)
print(f"Quantum generator found {len(quantum_primes)} primes: {quantum_primes}")

e8_primes = e8_lattice_sieve(limit)
print(f"E8 lattice sieve found {len(e8_primes)} primes: {e8_primes}")

# Compare the results
print("\nComparing results:")
print(f"Classical and Cyclotomic match: {set(classical_primes) == set(cyclotomic_primes)}")
print(f"Classical and Quantum match: {set(classical_primes) == set(quantum_primes)}")
print(f"Classical and E8 match: {set(classical_primes) == set(e8_primes)}")

# Test primality of some numbers
test_numbers = [17, 23, 91, 97]
print("\nTesting primality:")
for n in test_numbers:
    modular_result = modular_forms_prime_test(n)
    l_function_result = l_function_prime_test(n)
    actual_prime = n in classical_primes
    
    print(f"{n}: Modular forms: {modular_result}, L-function: {l_function_result}, Actual: {actual_prime}")

# Plot comparison of prime generation methods
methods = ["Classical", "Cyclotomic", "Quantum", "E8"]
counts = [
    len(classical_primes),
    len(cyclotomic_primes),
    len(quantum_primes),
    len(e8_primes)
]

plt.figure(figsize=(10, 6))
plt.bar(methods, counts, color=['blue', 'green', 'red', 'purple'])
plt.title(f"Prime Count Comparison (up to {limit})")
plt.ylabel("Number of Primes Found")
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, count in enumerate(counts):
    plt.text(i, count + 0.1, str(count), ha='center')

plt.tight_layout()
plt.savefig("prime_count_comparison.png", dpi=300)
print("\nSaved comparison plot to 'prime_count_comparison.png'")

print("\nDone!")