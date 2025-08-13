# TSAMS Chemistry

Chemical applications of the Tibedo Structural Algebraic Modeling System (TSAMS).

## Overview

TSAMS Chemistry provides implementations of chemical applications based on the mathematical framework of TSAMS. It implements:

- **Molecular Structure Module**: Implementation of molecular structures using the TSAMS framework
- **Quantum Chemistry Module**: Implementation of quantum chemistry calculations using TSAMS
- **Reaction Dynamics Module**: Implementation of chemical reaction dynamics using TSAMS

## Installation

```bash
pip install tsams-chemistry
```

## Usage

### Molecular Structure Module

```python
from tsams_chemistry.molecular_structure import MolecularStructure, Molecule

# Create a molecule
molecule = Molecule.from_smiles("CCO")  # Ethanol

# Analyze the molecular structure
structure = MolecularStructure(molecule)
print(f"Number of atoms: {structure.num_atoms}")
print(f"Number of bonds: {structure.num_bonds}")

# Compute properties
energy = structure.compute_energy()
print(f"Energy: {energy} kcal/mol")

# Optimize the structure
optimized = structure.optimize()
print(f"Optimized energy: {optimized.compute_energy()} kcal/mol")
```

### Quantum Chemistry Module

```python
from tsams_chemistry.quantum_chemistry import QuantumCalculation

# Create a quantum calculation
calculation = QuantumCalculation(method="dft", basis_set="6-31g")

# Load a molecule
calculation.load_molecule("CCO")  # Ethanol

# Run the calculation
result = calculation.run()

# Analyze the results
print(f"Electronic energy: {result.electronic_energy} Hartree")
print(f"HOMO-LUMO gap: {result.homo_lumo_gap} eV")

# Visualize the molecular orbitals
result.visualize_orbital("HOMO")
result.visualize_orbital("LUMO")
```

### Reaction Dynamics Module

```python
from tsams_chemistry.reaction_dynamics import ReactionSimulation

# Create a reaction simulation
simulation = ReactionSimulation(temperature=298.15, pressure=1.0)

# Define the reaction
simulation.add_reactant("CCO")  # Ethanol
simulation.add_reactant("O=O")  # Oxygen
simulation.add_product("CC=O")  # Acetaldehyde
simulation.add_product("O")     # Water

# Run the simulation
result = simulation.run(time=1000, timestep=0.1)

# Analyze the results
print(f"Reaction rate: {result.rate} mol/L/s")
print(f"Activation energy: {result.activation_energy} kcal/mol")

# Plot the concentration profiles
result.plot_concentrations()
```

## Documentation

For detailed documentation, see the [TSAMS Documentation](https://github.com/ctibedoJ/tsams-docs).

## Dependencies

- numpy
- scipy
- sympy
- matplotlib
- rdkit
- tsams-core
- tsams-classical

## License

This project is licensed under the MIT License - see the LICENSE file for details.