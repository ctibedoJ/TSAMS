"""
Nodal Structure Visualization module.

This module provides tools for visualizing the 441-dimensional nodal structure,
hair braid nodes, and braid operations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Union, Optional, Callable, Any

from ..state_space.nodal_structure_441 import NodalStructure441
from ..hair_braid.hair_braid_nodes import HairBraidNode, HairBraidSystem
from ..hair_braid.braid_operations import BraidOperations


def plot_nodal_structure(fig_size: Tuple[int, int] = (12, 10),
                        show_hair_braid_nodes: bool = True,
                        save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize the 441-dimensional nodal structure.
    
    Args:
        fig_size (Tuple[int, int]): The size of the figure (default: (12, 10)).
        show_hair_braid_nodes (bool): Whether to highlight the hair braid nodes (default: True).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes with the visualization.
    """
    # Create the nodal structure
    nodal_structure = NodalStructure441()
    
    # Create the figure
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Since we can't visualize 441 dimensions directly, we'll create a 3D projection
    # We'll use the factorization 441 = 9 × 49 to help with the visualization
    
    # Create a grid of points representing the structure
    x = np.linspace(-1, 1, 9)
    y = np.linspace(-1, 1, 7)
    z = np.linspace(-1, 1, 7)
    
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Plot the grid points
    ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), s=10, alpha=0.3, color='blue')
    
    # If requested, highlight the hair braid nodes
    if show_hair_braid_nodes:
        # Get the hair braid nodes
        hair_braid_nodes = nodal_structure.get_hair_braid_nodes()
        
        # Create a 3D projection of the hair braid nodes
        node_x = []
        node_y = []
        node_z = []
        
        for i, node in enumerate(hair_braid_nodes):
            # Use the node's components to create a 3D projection
            node_9, node_49 = node
            
            # Map the complex numbers to 3D coordinates
            x_coord = node_9.real
            y_coord = node_9.imag
            z_coord = node_49.real
            
            node_x.append(x_coord)
            node_y.append(y_coord)
            node_z.append(z_coord)
        
        # Plot the hair braid nodes
        ax.scatter(node_x, node_y, node_z, s=100, color='red', marker='*', label='Hair Braid Nodes')
    
    # Set the labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Projection of the 441-Dimensional Nodal Structure')
    
    # Add a legend if hair braid nodes are shown
    if show_hair_braid_nodes:
        ax.legend()
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, ax


def plot_hair_braid_nodes(fig_size: Tuple[int, int] = (12, 10),
                         show_connections: bool = True,
                         save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize the 21 hair braid nodes.
    
    Args:
        fig_size (Tuple[int, int]): The size of the figure (default: (12, 10)).
        show_connections (bool): Whether to show connections between nodes (default: True).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes with the visualization.
    """
    # Create the hair braid system
    hair_braid_system = HairBraidSystem()
    
    # Create the figure
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract the coordinates of the hair braid nodes
    node_x = []
    node_y = []
    node_z = []
    
    for i in range(21):
        node = hair_braid_system.get_node(i)
        
        # Extract the coordinates from the node
        node_9, node_49 = node.coordinates
        
        # Map the complex numbers to 3D coordinates
        x_coord = node_9.real
        y_coord = node_9.imag
        z_coord = node_49.real
        
        node_x.append(x_coord)
        node_y.append(y_coord)
        node_z.append(z_coord)
    
    # Plot the hair braid nodes
    ax.scatter(node_x, node_y, node_z, s=100, color='red', marker='*')
    
    # Add labels to the nodes
    for i in range(21):
        ax.text(node_x[i], node_y[i], node_z[i], f'{i}', fontsize=10)
    
    # If requested, show connections between nodes
    if show_connections:
        # Connect nodes based on the factorization 21 = 3 × 7
        for i in range(3):
            for j in range(7):
                idx1 = i * 7 + j
                
                # Connect to the next node in the same row
                idx2 = i * 7 + ((j + 1) % 7)
                ax.plot([node_x[idx1], node_x[idx2]], [node_y[idx1], node_y[idx2]], [node_z[idx1], node_z[idx2]], 'b-', alpha=0.3)
                
                # Connect to the corresponding node in the next row
                idx3 = ((i + 1) % 3) * 7 + j
                ax.plot([node_x[idx1], node_x[idx3]], [node_y[idx1], node_y[idx3]], [node_z[idx1], node_z[idx3]], 'g-', alpha=0.3)
    
    # Set the labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hair Braid Nodes (21 = 3 × 7)')
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, ax


def plot_braid_operation(node1_idx: int, node2_idx: int,
                        fig_size: Tuple[int, int] = (12, 10),
                        save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize a braid operation between two hair braid nodes.
    
    Args:
        node1_idx (int): The index of the first hair braid node (0-20).
        node2_idx (int): The index of the second hair braid node (0-20).
        fig_size (Tuple[int, int]): The size of the figure (default: (12, 10)).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes with the visualization.
    """
    # Create the hair braid system
    hair_braid_system = HairBraidSystem()
    
    # Get the nodes
    node1 = hair_braid_system.get_node(node1_idx)
    node2 = hair_braid_system.get_node(node2_idx)
    
    # Apply the braid operation
    result = hair_braid_system.braid_operation(node1_idx, node2_idx)
    
    # Create the figure
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract the coordinates of the nodes
    nodes = [node1, node2, result]
    colors = ['red', 'blue', 'green']
    labels = [f'Node {node1_idx}', f'Node {node2_idx}', f'Result']
    
    for i, (node, color, label) in enumerate(zip(nodes, colors, labels)):
        # Extract the coordinates from the node
        node_9, node_49 = node.coordinates
        
        # Map the complex numbers to 3D coordinates
        x_coord = node_9.real
        y_coord = node_9.imag
        z_coord = node_49.real
        
        # Plot the node
        ax.scatter([x_coord], [y_coord], [z_coord], s=100, color=color, marker='*', label=label)
        
        # Add a label
        ax.text(x_coord, y_coord, z_coord, label, fontsize=10)
    
    # Connect the nodes with arrows to show the operation
    node1_coords = (node1.coordinates[0].real, node1.coordinates[0].imag, node1.coordinates[1].real)
    node2_coords = (node2.coordinates[0].real, node2.coordinates[0].imag, node2.coordinates[1].real)
    result_coords = (result.coordinates[0].real, result.coordinates[0].imag, result.coordinates[1].real)
    
    # Draw arrows from the input nodes to the result
    ax.quiver(node1_coords[0], node1_coords[1], node1_coords[2],
             result_coords[0] - node1_coords[0],
             result_coords[1] - node1_coords[1],
             result_coords[2] - node1_coords[2],
             color='red', arrow_length_ratio=0.1)
    
    ax.quiver(node2_coords[0], node2_coords[1], node2_coords[2],
             result_coords[0] - node2_coords[0],
             result_coords[1] - node2_coords[1],
             result_coords[2] - node2_coords[2],
             color='blue', arrow_length_ratio=0.1)
    
    # Set the labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Braid Operation: Node {node1_idx} ⊗ Node {node2_idx} = Node {result.index}')
    
    # Add a legend
    ax.legend()
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, ax


def plot_braid_invariant(braid: List[Tuple[int, int]],
                        fig_size: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize the braid invariant of a sequence of braid operations.
    
    Args:
        braid (List[Tuple[int, int]]): A sequence of pairs of node indices representing braid operations.
        fig_size (Tuple[int, int]): The size of the figure (default: (10, 6)).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes with the visualization.
    """
    # Create the hair braid system
    hair_braid_system = HairBraidSystem()
    
    # Compute the braid invariant
    invariant = hair_braid_system.braid_invariant(braid)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot the braid as a sequence of operations
    x = np.arange(len(braid) + 1)
    y = [0]  # Start with 0
    
    # Compute the cumulative invariant at each step
    for i in range(len(braid)):
        partial_braid = braid[:i+1]
        partial_invariant = hair_braid_system.braid_invariant(partial_braid)
        y.append(abs(partial_invariant))
    
    # Plot the cumulative invariant
    ax.plot(x, y, 'o-', linewidth=2)
    
    # Set the labels and title
    ax.set_xlabel('Operation Index')
    ax.set_ylabel('Absolute Braid Invariant')
    ax.set_title(f'Braid Invariant Evolution (Final: {abs(invariant):.4f})')
    
    # Show the grid
    ax.grid(True, alpha=0.3)
    
    # Set the x-ticks to integers
    ax.set_xticks(x)
    
    # Add labels for each operation
    for i, (node1, node2) in enumerate(braid):
        ax.annotate(f'({node1},{node2})', (i+1, y[i+1]), textcoords="offset points", xytext=(0,10), ha='center')
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, ax


def plot_jones_polynomial(braid: List[Tuple[int, int]],
                         fig_size: Tuple[int, int] = (10, 6),
                         save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize the Jones polynomial of a braid.
    
    Args:
        braid (List[Tuple[int, int]]): A sequence of pairs of node indices representing braid operations.
        fig_size (Tuple[int, int]): The size of the figure (default: (10, 6)).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes with the visualization.
    """
    # Create the hair braid system
    hair_braid_system = HairBraidSystem()
    
    # Create braid operations
    braid_operations = BraidOperations()
    
    # Create a braid in the braid group
    braid_group = braid_operations.braid_group
    braid_obj = braid_group.create_braid([node1 + 1 for node1, _ in braid])
    
    # Compute the Jones polynomial
    jones_polynomial = braid_operations.jones_polynomial(braid_obj)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot the Jones polynomial
    degrees = list(range(len(jones_polynomial)))
    coefficients = [abs(coeff) for coeff in jones_polynomial]
    
    ax.bar(degrees, coefficients, alpha=0.7)
    
    # Set the labels and title
    ax.set_xlabel('Degree')
    ax.set_ylabel('Coefficient Magnitude')
    ax.set_title('Jones Polynomial Coefficients')
    
    # Show the grid
    ax.grid(True, alpha=0.3)
    
    # Set the x-ticks to integers
    ax.set_xticks(degrees)
    
    # Add labels for the coefficients
    for i, coeff in enumerate(jones_polynomial):
        ax.annotate(f'{coeff:.2f}', (i, abs(coeff)), textcoords="offset points", xytext=(0,5), ha='center')
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, ax