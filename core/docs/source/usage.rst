Usage
=====

TSAMS Core provides a comprehensive set of tools for working with Prime Indexed Möbius Transformation State Space Theory.

Basic Usage
----------

Here's a simple example of using TSAMS Core:

.. code-block:: python

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

Möbius Transformations
--------------------

TSAMS Core provides a comprehensive implementation of Möbius transformations:

.. code-block:: python

   from tsams_core.moebius import MoebiusTransformation

   # Create a Möbius transformation
   m = MoebiusTransformation(1, 2, 3, 4)

   # Apply the transformation to a complex number
   z = 1 + 2j
   w = m.apply(z)
   print(f"M({z}) = {w}")

   # Compute the fixed points of the transformation
   fixed_points = m.fixed_points()
   print(f"Fixed points: {fixed_points}")

   # Check the type of the transformation
   if m.is_elliptic():
       print("The transformation is elliptic")
   elif m.is_parabolic():
       print("The transformation is parabolic")
   elif m.is_hyperbolic():
       print("The transformation is hyperbolic")
   else:
       print("The transformation is loxodromic")

Cyclotomic Fields
---------------

TSAMS Core provides a comprehensive implementation of cyclotomic fields:

.. code-block:: python

   from tsams_core.cyclotomic import CyclotomicField

   # Create a cyclotomic field
   field = CyclotomicField(420)
   print(f"Field: {field}")
   print(f"Dimension: {field.dimension}")

   # Create field elements
   a = field.element_from_coefficients([1] + [0] * (field.dimension - 1))
   b = field.element_from_coefficients([0, 1] + [0] * (field.dimension - 2))
   print(f"Element a: {a}")
   print(f"Element b: {b}")

   # Perform field operations
   sum_ab = field.add(a, b)
   product_ab = field.multiply(a, b)
   print(f"a + b: {sum_ab}")
   print(f"a * b: {product_ab}")

State Space Theory
---------------

TSAMS Core provides a comprehensive implementation of state space theory:

.. code-block:: python

   from tsams_core.state_space import StateSpace, NodalStructure441

   # Create a state space
   state_space = StateSpace()

   # Create the 441-dimensional nodal structure
   nodal_structure = NodalStructure441()

   # Get the hair braid nodes
   nodes = nodal_structure.get_hair_braid_nodes()
   print(f"Number of hair braid nodes: {len(nodes)}")

   # Perform a braid operation
   result = nodal_structure.braid_operation(0, 1)
   print(f"Braid operation (0, 1): {result}")

   # Compute a braid invariant
   braid = [(0, 1), (1, 2), (0, 2)]
   invariant = nodal_structure.braid_invariant(braid)
   print(f"Braid invariant: {invariant}")

Visualization
-----------

TSAMS Core provides a comprehensive set of visualization tools:

.. code-block:: python

   import matplotlib.pyplot as plt
   from tsams_core.moebius import Root420Structure
   from tsams_core.visualization import OrbitPlotter, EnergySpectrumPlotter

   # Create the 420-root structure
   root_structure = Root420Structure()

   # Compute the energy spectrum
   spectrum = root_structure.energy_spectrum()

   # Plot the energy spectrum
   plt.figure(figsize=(10, 6))
   EnergySpectrumPlotter.plot_energy_spectrum(spectrum)
   plt.title("Energy Spectrum of the 420-Root Möbius Structure")
   plt.show()

   # Compute the orbit of a point under a transformation
   initial_point = 1.0 + 0.5j
   orbit = root_structure.orbit(initial_point, 11, max_iterations=100)

   # Plot the orbit
   plt.figure(figsize=(8, 8))
   OrbitPlotter.plot_orbit_2d(orbit)
   plt.title(f"Orbit of {initial_point} under M_11")
   plt.show()
