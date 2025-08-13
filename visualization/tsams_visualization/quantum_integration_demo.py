"""
Quantum Integration Demonstration

This script demonstrates the integration between cyclotomic field theory
and quantum computing, showing how our framework can be used to create
quantum circuits based on cyclotomic fields and braid structures.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Add the parent directory to the path to import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our cyclotomic quantum framework
from cyclotomic_quantum.core.cyclotomic_field import CyclotomicField
from cyclotomic_quantum.core.dedekind_cut import DedekindCutMorphicConductor
from cyclotomic_quantum.core.braid_theory import BraidStructure
from cyclotomic_quantum.quantum.quantum_circuit import QuantumCircuitRepresentation
from cyclotomic_quantum.quantum.cnot_operations import CNOTOperations
from cyclotomic_quantum.quantum.moebius_braiding import MoebiusBraiding
from cyclotomic_quantum.quantum.infinite_time_looping import InfiniteTimeLooping
from cyclotomic_quantum.quantum.qiskit_integration import QiskitIntegration


def create_braid_visualization(braid, save_path=None):
    """
    Create a visualization of a braid structure.
    
    Args:
        braid (BraidStructure): The braid structure to visualize.
        save_path (str): The path to save the figure (default: None).
    """
    # Set plot style
    plt.style.use('dark_background')
    sns.set_style("darkgrid")
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Get the number of strands and crossings
    num_strands = braid.num_strands
    crossings = braid.operations
    
    # Plot the strands
    for i in range(num_strands):
        plt.plot([0, 1], [i, i], 'w-', alpha=0.5)
    
    # Plot the crossings
    for i, j, positive in crossings:
        # Compute the x-coordinate based on the crossing index
        x = (crossings.index((i, j, positive)) + 1) / (len(crossings) + 1)
        
        # Plot the crossing
        if positive:
            plt.plot([x, x], [i, j], 'r-', linewidth=2)
        else:
            plt.plot([x, x], [i, j], 'b-', linewidth=2)
    
    plt.title('Braid Structure Visualization', fontsize=16)
    plt.xlabel('Position', fontsize=14)
    plt.ylabel('Strand', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add a text box with information about the braid
    textstr = f"Number of strands: {num_strands}\n"
    textstr += f"Number of crossings: {len(crossings)}\n"
    textstr += f"Jones polynomial: {braid.compute_jones_polynomial()}"
    
    props = dict(boxstyle='round', facecolor='black', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def create_moebius_braiding_visualization(moebius_braiding, save_path=None):
    """
    Create a visualization of a Möbius braiding sequence.
    
    Args:
        moebius_braiding (MoebiusBraiding): The Möbius braiding sequence to visualize.
        save_path (str): The path to save the figure (default: None).
    """
    # Set plot style
    plt.style.use('dark_background')
    sns.set_style("darkgrid")
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Visualize the Möbius braiding sequence
    im = ax1.imshow(moebius_braiding.sequence, cmap='viridis')
    fig.colorbar(im, ax=ax1)
    ax1.set_title('Möbius Braiding Sequence', fontsize=16)
    ax1.set_xlabel('Strand Index', fontsize=14)
    ax1.set_ylabel('Strand Index', fontsize=14)
    
    # Generate and visualize the braid structure
    braid = moebius_braiding.generate_braid()
    
    # Get the number of strands and crossings
    num_strands = braid.num_strands
    crossings = braid.operations
    
    # Plot the strands
    for i in range(num_strands):
        ax2.plot([0, 1], [i, i], 'w-', alpha=0.5)
    
    # Plot the crossings
    for i, j, positive in crossings:
        # Compute the x-coordinate based on the crossing index
        x = (crossings.index((i, j, positive)) + 1) / (len(crossings) + 1)
        
        # Plot the crossing
        if positive:
            ax2.plot([x, x], [i, j], 'r-', linewidth=2)
        else:
            ax2.plot([x, x], [i, j], 'b-', linewidth=2)
    
    ax2.set_title('Generated Braid Structure', fontsize=16)
    ax2.set_xlabel('Position', fontsize=14)
    ax2.set_ylabel('Strand', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add a text box with information about the braid
    textstr = f"Number of strands: {num_strands}\n"
    textstr += f"Number of crossings: {len(crossings)}\n"
    textstr += f"Jones polynomial: {braid.compute_jones_polynomial()}"
    
    props = dict(boxstyle='round', facecolor='black', alpha=0.5)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def create_infinite_time_looping_visualization(infinite_time_looping, save_path=None):
    """
    Create a visualization of an infinite time looping structure.
    
    Args:
        infinite_time_looping (InfiniteTimeLooping): The infinite time looping structure to visualize.
        save_path (str): The path to save the figure (default: None).
    """
    # Set plot style
    plt.style.use('dark_background')
    sns.set_style("darkgrid")
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Visualize the infinite time looping structure as a heatmap
    im = ax1.imshow(infinite_time_looping.sequence.T, aspect='auto', cmap='viridis',
                  extent=[0, infinite_time_looping.cycles, 0, infinite_time_looping.num_dimensions])
    fig.colorbar(im, ax=ax1)
    ax1.set_title('Infinite Time Looping Structure', fontsize=16)
    ax1.set_xlabel('Cycle', fontsize=14)
    ax1.set_ylabel('Dimension', fontsize=14)
    
    # Plot the evolution of a few dimensions
    for i in range(min(5, infinite_time_looping.num_dimensions)):
        evolution = infinite_time_looping.get_dimension(i)
        ax2.plot(range(infinite_time_looping.cycles), evolution, label=f'Dimension {i}')
    
    ax2.set_title('Dimension Evolution', fontsize=16)
    ax2.set_xlabel('Cycle', fontsize=14)
    ax2.set_ylabel('Value', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add a text box with information about the structure
    textstr = f"Number of cycles: {infinite_time_looping.cycles}\n"
    textstr += f"Number of dimensions: {infinite_time_looping.num_dimensions}\n"
    textstr += f"Average periodicity: {infinite_time_looping.compute_average_periodicity():.2f}"
    
    props = dict(boxstyle='round', facecolor='black', alpha=0.5)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def create_qiskit_circuit_visualization(qiskit_integration, braid, save_path=None):
    """
    Create a visualization of a Qiskit quantum circuit generated from a braid structure.
    
    Args:
        qiskit_integration (QiskitIntegration): The Qiskit integration object.
        braid (BraidStructure): The braid structure.
        save_path (str): The path to save the figure (default: None).
    """
    try:
        # Convert the braid to a Qiskit circuit
        circuit = qiskit_integration.braid_to_qiskit_circuit(braid)
        
        # Visualize the circuit
        fig = qiskit_integration.visualize_circuit(circuit)
        
        if save_path:
            fig.savefig(save_path)
        else:
            plt.show()
    except ImportError:
        print("Qiskit is required for this visualization. Please install it with 'pip install qiskit'.")


def main():
    """
    Main function to run the demonstration.
    """
    print("=" * 80)
    print("Quantum Integration Demonstration")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize our objects
    print("\nInitializing objects...")
    cyclotomic_field = CyclotomicField(168)
    dedekind_cut = DedekindCutMorphicConductor()
    
    # Create a braid structure
    print("\nCreating a braid structure...")
    braid = BraidStructure(5)  # Using 5 strands for clarity
    braid.add_crossing(0, 1, True)
    braid.add_crossing(1, 2, True)
    braid.add_crossing(2, 3, True)
    braid.add_crossing(3, 4, True)
    braid.add_crossing(0, 2, False)
    braid.add_crossing(1, 3, False)
    braid.add_crossing(2, 4, False)
    
    # Visualize the braid structure
    print("\nVisualizing the braid structure...")
    create_braid_visualization(braid, save_path=os.path.join(output_dir, 'braid_structure.png'))
    
    # Create a Möbius braiding sequence
    print("\nCreating a Möbius braiding sequence...")
    moebius_braiding = MoebiusBraiding(10)  # Using 10 strands for clarity
    
    # Visualize the Möbius braiding sequence
    print("\nVisualizing the Möbius braiding sequence...")
    create_moebius_braiding_visualization(moebius_braiding, save_path=os.path.join(output_dir, 'moebius_braiding.png'))
    
    # Create an infinite time looping structure
    print("\nCreating an infinite time looping structure...")
    infinite_time_looping = InfiniteTimeLooping(100, 10)  # 100 cycles, 10 dimensions
    
    # Visualize the infinite time looping structure
    print("\nVisualizing the infinite time looping structure...")
    create_infinite_time_looping_visualization(infinite_time_looping, save_path=os.path.join(output_dir, 'infinite_time_looping.png'))
    
    # Create a quantum circuit representation
    print("\nCreating a quantum circuit representation...")
    quantum_circuit = QuantumCircuitRepresentation(5)  # Using 5 qubits for clarity
    
    # Create CNOT operations
    print("\nCreating CNOT operations...")
    cnot_operations = CNOTOperations(5)  # Using 5 qubits for clarity
    
    # Try to create a Qiskit integration
    try:
        print("\nCreating a Qiskit integration...")
        qiskit_integration = QiskitIntegration(5)  # Using 5 qubits for clarity
        
        # Visualize the Qiskit circuit
        print("\nVisualizing the Qiskit circuit...")
        create_qiskit_circuit_visualization(qiskit_integration, braid, save_path=os.path.join(output_dir, 'qiskit_circuit.png'))
        
        # Run a simulation
        print("\nRunning a simulation...")
        circuit = qiskit_integration.braid_to_qiskit_circuit(braid)
        counts = qiskit_integration.run_simulation(circuit)
        print(f"Simulation results: {counts}")
    except ImportError:
        print("\nQiskit is not installed. Skipping Qiskit integration demonstration.")
    
    print("\nDemonstration complete! All visualizations have been saved to the 'output' directory.")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()