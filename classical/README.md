# TSAMS Classical

Classical mathematical implementations of the Tibedo Structural Algebraic Modeling System (TSAMS).

## Overview

TSAMS Classical provides classical mathematical implementations that build upon the core structures of TSAMS. It implements:

- **Dedekind Cut Theory**: Implementation of Dedekind cut morphic conductors and automorphic structures
- **Prime Theory**: Implementation of prime indexed structures and prime distribution formulas
- **Braid Theory**: Implementation of braid groups, braid invariants, and braid operations
- **Septimal Theory**: Implementation of septimal structures and operations

## Installation

```bash
pip install tsams-classical
```

## Usage

### Dedekind Cut Theory

```python
from tsams_classical.dedekind_cut import DedekindCutMorphicConductor

# Create a Dedekind cut morphic conductor
conductor = DedekindCutMorphicConductor(168)
print(f"Conductor: {conductor}")

# Compute the automorphic structure
automorphic = conductor.automorphic_structure()
print(f"Automorphic structure: {automorphic}")

# Compute the j-invariant
tau = complex(0.5, 1.0)
j = conductor.j_invariant(tau)
print(f"j-invariant at τ = {tau}: {j}")
```

### Prime Theory

```python
from tsams_classical.prime_theory import PrimeDistribution, PrimeIndexedStructures

# Create a prime distribution
distribution = PrimeDistribution(1000)
print(f"Number of primes up to 1000: {distribution.count()}")

# Compute the prime counting function
pi_100 = distribution.prime_counting_function(100)
print(f"π(100) = {pi_100}")

# Compute the prime number theorem estimate
pnt_100 = distribution.prime_number_theorem_estimate(100)
print(f"PNT estimate for π(100) ≈ {pnt_100:.2f}")
```

### Braid Theory

```python
from tsams_classical.braid_theory import BraidGroup, BraidWord, BraidInvariants, BraidOperations

# Create a braid group
group = BraidGroup(3)

# Create a braid word representing the trefoil knot
trefoil = BraidWord(3, [1, 1, 1])

# Compute the Alexander polynomial
from tsams_classical.braid_theory import AlexanderPolynomial
alex_poly = AlexanderPolynomial.compute(trefoil)
print(f"Alexander polynomial: {alex_poly}")

# Check if the braid closure is a knot
is_knot = BraidOperations.is_knot(trefoil)
print(f"Is the closure a knot? {is_knot}")

# Compute the number of components in the closure
num_components = BraidOperations.num_components(trefoil)
print(f"Number of components: {num_components}")
```

### Septimal Theory

```python
from tsams_classical.septimal import SeptimalStructure, SeptimalLattice, SeptimalOperations

# Create a septimal structure
structure = SeptimalStructure(3)

# Create a septimal lattice
lattice = SeptimalLattice(2)

# Perform operations on points in the septimal structure
point = [1, 2, 3]
rotated = SeptimalOperations.rotate(structure, point, 0, 1)
reflected = SeptimalOperations.reflect(structure, point, 0)
translated = SeptimalOperations.translate(structure, point, [1, 1, 1])

print(f"Original point: {point}")
print(f"Rotated point: {rotated}")
print(f"Reflected point: {reflected}")
print(f"Translated point: {translated}")
```

## Documentation

For detailed documentation, see the [TSAMS Documentation](https://github.com/ctibedoJ/tsams-docs).

## Dependencies

- numpy
- sympy

## License

This project is licensed under the MIT License - see the LICENSE file for details.