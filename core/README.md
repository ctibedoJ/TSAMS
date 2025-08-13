# TSAMS Core

Core mathematical structures for the Tibedo Structural Algebraic Modeling System (TSAMS).

## Overview

TSAMS Core provides the fundamental mathematical structures that underpin the entire TSAMS ecosystem. It implements:

- **Cyclotomic Fields**: Implementation of cyclotomic fields Q(ζ_n) and their properties
- **Möbius Transformations**: Tools for working with Möbius transformations and the 420-root structure
- **State Space Theory**: Implementation of state spaces and the 441-dimensional nodal structure
- **Hair Braid Dynamics**: Tools for working with hair braid nodes and braid invariants
- **Hyperbolic Priming**: Implementation of hyperbolic priming transformations and energy quantization

## Installation

```bash
pip install tsams-core
```

## Usage

```python
import numpy as np
from tsams_core.moebius import Root420Structure
from tsams_core.state_space import NodalStructure441

# Create the 420-root Möbius structure
root_structure = Root420Structure()

# Get a transformation corresponding to a prime index
transformation = root_structure.get_transformation(11)

# Compute the energy of the transformation
energy = transformation.energy()
print(f"Energy of M_11: {energy}")

# Create the 441-dimensional nodal structure
nodal_structure = NodalStructure441()

# Get the hair braid nodes
nodes = nodal_structure.get_hair_braid_nodes()
print(f"Number of hair braid nodes: {len(nodes)}")
```

## Documentation

For detailed documentation, see the [TSAMS Documentation](https://github.com/ctibedoJ/tsams-docs).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
