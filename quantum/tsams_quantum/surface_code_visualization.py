"""
TIBEDO Surface Code Visualization Module

This module provides visualization tools for surface code error correction,
including lattice visualization, error pattern visualization, and syndrome
visualization.

Key components:
1. SurfaceCodeVisualizer: Visualizes surface code lattices and error patterns
2. SyndromeVisualizer: Visualizes syndrome measurements and error correction
3. DecodingGraphVisualizer: Visualizes the decoding graph and matching
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from qiskit_aer import Aer
from typing import List, Tuple, Dict, Any, Optional, Union
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import surface code components
from tibedo.quantum_information_new.surface_code_error_correction import (
    SurfaceCode,
    SurfaceCodeDecoder
)

class SurfaceCodeVisualizer:
    """
    Visualizes surface code lattices and error patterns.
    
    This class provides methods for visualizing the surface code lattice,
    including data qubits, stabilizers, and logical operators, as well as
    error patterns and their effects.
    """
    
    def __init__(self, surface_code: SurfaceCode):
        """
        Initialize the surface code visualizer.
        
        Args:
            surface_code: The surface code to visualize
        """
        self.surface_code = surface_code
        
        # Define colors for visualization
        self.colors = {
            'data_qubit': 'black',
            'x_stabilizer': 'red',
            'z_stabilizer': 'blue',
            'logical_x': 'orange',
            'logical_z': 'green',
            'error_x': 'magenta',
            'error_z': 'cyan',
            'error_y': 'yellow',
            'syndrome': 'purple',
            'correction': 'lime'
        }
    
    def visualize_lattice(self, show_stabilizers: bool = True, 
                         show_logical_operators: bool = True,
                         title: str = None) -> plt.Figure:
        """
        Visualize the surface code lattice.
        
        Args:
            show_stabilizers: Whether to show the stabilizers
            show_logical_operators: Whether to show the logical operators
            title: Title for the plot
            
        Returns:
            Matplotlib figure showing the lattice
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot the physical qubits
        if self.surface_code.use_rotated_lattice:
            for i in range(self.surface_code.distance):
                for j in range(self.surface_code.distance):
                    qubit_index = self.surface_code.qubit_grid[i, j]
                    ax.plot(j, -i, 'o', color=self.colors['data_qubit'], markersize=10)
                    ax.text(j, -i, str(qubit_index), color='white', ha='center', va='center')
        else:
            for i in range(self.surface_code.distance + 1):
                for j in range(self.surface_code.distance + 1):
                    if i == self.surface_code.distance and j == self.surface_code.distance:
                        continue  # Skip the bottom-right corner
                    qubit_index = self.surface_code.qubit_grid[i, j]
                    ax.plot(j, -i, 'o', color=self.colors['data_qubit'], markersize=10)
                    ax.text(j, -i, str(qubit_index), color='white', ha='center', va='center')
        
        # Plot the stabilizers
        if show_stabilizers:
            # Plot X-stabilizers
            for stabilizer in self.surface_code.x_stabilizers:
                x_coords = []
                y_coords = []
                for qubit in stabilizer:
                    i, j = np.where(self.surface_code.qubit_grid == qubit)
                    if len(i) > 0 and len(j) > 0:
                        y_coords.append(-i[0])
                        x_coords.append(j[0])
                if x_coords and y_coords:
                    center_x = sum(x_coords) / len(x_coords)
                    center_y = sum(y_coords) / len(y_coords)
                    ax.plot(center_x, center_y, 's', color=self.colors['x_stabilizer'], markersize=15, alpha=0.5)
                    ax.text(center_x, center_y, 'X', color='white', ha='center', va='center')
            
            # Plot Z-stabilizers
            for stabilizer in self.surface_code.z_stabilizers:
                x_coords = []
                y_coords = []
                for qubit in stabilizer:
                    i, j = np.where(self.surface_code.qubit_grid == qubit)
                    if len(i) > 0 and len(j) > 0:
                        y_coords.append(-i[0])
                        x_coords.append(j[0])
                if x_coords and y_coords:
                    center_x = sum(x_coords) / len(x_coords)
                    center_y = sum(y_coords) / len(y_coords)
                    ax.plot(center_x, center_y, 's', color=self.colors['z_stabilizer'], markersize=15, alpha=0.5)
                    ax.text(center_x, center_y, 'Z', color='white', ha='center', va='center')
        
        # Plot the logical operators
        if show_logical_operators:
            x_coords = []
            y_coords = []
            for qubit in self.surface_code.logical_x:
                i, j = np.where(self.surface_code.qubit_grid == qubit)
                if len(i) > 0 and len(j) > 0:
                    y_coords.append(-i[0])
                    x_coords.append(j[0])
            ax.plot(x_coords, y_coords, '-', color=self.colors['logical_x'], linewidth=3, label='Logical X')
            
            x_coords = []
            y_coords = []
            for qubit in self.surface_code.logical_z:
                i, j = np.where(self.surface_code.qubit_grid == qubit)
                if len(i) > 0 and len(j) > 0:
                    y_coords.append(-i[0])
                    x_coords.append(j[0])
            ax.plot(x_coords, y_coords, '-', color=self.colors['logical_z'], linewidth=3, label='Logical Z')
        
        # Set plot properties
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Surface Code (d={self.surface_code.distance})")
        
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        
        return fig
    
    def visualize_errors(self, x_errors: List[int], z_errors: List[int], 
                        title: str = None) -> plt.Figure:
        """
        Visualize error patterns on the surface code lattice.
        
        Args:
            x_errors: List of qubits with X errors
            z_errors: List of qubits with Z errors
            title: Title for the plot
            
        Returns:
            Matplotlib figure showing the error pattern
        """
        # Start with the basic lattice visualization
        fig = self.visualize_lattice(show_stabilizers=True, show_logical_operators=False)
        ax = fig.axes[0]
        
        # Identify Y errors (qubits with both X and Z errors)
        y_errors = list(set(x_errors) & set(z_errors))
        
        # Remove Y errors from X and Z error lists
        x_errors = list(set(x_errors) - set(y_errors))
        z_errors = list(set(z_errors) - set(y_errors))
        
        # Plot X errors
        for qubit in x_errors:
            i, j = np.where(self.surface_code.qubit_grid == qubit)
            if len(i) > 0 and len(j) > 0:
                ax.add_patch(patches.Circle((j[0], -i[0]), 0.3, 
                                          color=self.colors['error_x'], alpha=0.7))
        
        # Plot Z errors
        for qubit in z_errors:
            i, j = np.where(self.surface_code.qubit_grid == qubit)
            if len(i) > 0 and len(j) > 0:
                ax.add_patch(patches.Circle((j[0], -i[0]), 0.3, 
                                          color=self.colors['error_z'], alpha=0.7))
        
        # Plot Y errors
        for qubit in y_errors:
            i, j = np.where(self.surface_code.qubit_grid == qubit)
            if len(i) > 0 and len(j) > 0:
                ax.add_patch(patches.Circle((j[0], -i[0]), 0.3, 
                                          color=self.colors['error_y'], alpha=0.7))
        
        # Add legend for errors
        ax.add_patch(patches.Circle((0, 0), 0.3, color=self.colors['error_x'], alpha=0.7, label='X Error'))
        ax.add_patch(patches.Circle((0, 0), 0.3, color=self.colors['error_z'], alpha=0.7, label='Z Error'))
        ax.add_patch(patches.Circle((0, 0), 0.3, color=self.colors['error_y'], alpha=0.7, label='Y Error'))
        
        # Update legend
        ax.legend()
        
        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Surface Code (d={self.surface_code.distance}) with Errors")
        
        return fig
    
    def visualize_syndrome(self, x_syndrome: List[int], z_syndrome: List[int],
                          title: str = None) -> plt.Figure:
        """
        Visualize syndrome measurements on the surface code lattice.
        
        Args:
            x_syndrome: Syndrome measurements for X-stabilizers
            z_syndrome: Syndrome measurements for Z-stabilizers
            title: Title for the plot
            
        Returns:
            Matplotlib figure showing the syndrome pattern
        """
        # Start with the basic lattice visualization
        fig = self.visualize_lattice(show_stabilizers=True, show_logical_operators=False)
        ax = fig.axes[0]
        
        # Plot X-stabilizer syndromes
        for i, syndrome_bit in enumerate(x_syndrome):
            if syndrome_bit == 1:
                # Find the center of the X-stabilizer
                stabilizer = self.surface_code.x_stabilizers[i]
                x_coords = []
                y_coords = []
                for qubit in stabilizer:
                    i_coords, j_coords = np.where(self.surface_code.qubit_grid == qubit)
                    if len(i_coords) > 0 and len(j_coords) > 0:
                        y_coords.append(-i_coords[0])
                        x_coords.append(j_coords[0])
                if x_coords and y_coords:
                    center_x = sum(x_coords) / len(x_coords)
                    center_y = sum(y_coords) / len(y_coords)
                    ax.add_patch(patches.Circle((center_x, center_y), 0.3, 
                                              color=self.colors['syndrome'], alpha=0.7))
        
        # Plot Z-stabilizer syndromes
        for i, syndrome_bit in enumerate(z_syndrome):
            if syndrome_bit == 1:
                # Find the center of the Z-stabilizer
                stabilizer = self.surface_code.z_stabilizers[i]
                x_coords = []
                y_coords = []
                for qubit in stabilizer:
                    i_coords, j_coords = np.where(self.surface_code.qubit_grid == qubit)
                    if len(i_coords) > 0 and len(j_coords) > 0:
                        y_coords.append(-i_coords[0])
                        x_coords.append(j_coords[0])
                if x_coords and y_coords:
                    center_x = sum(x_coords) / len(x_coords)
                    center_y = sum(y_coords) / len(y_coords)
                    ax.add_patch(patches.Circle((center_x, center_y), 0.3, 
                                              color=self.colors['syndrome'], alpha=0.7))
        
        # Add legend for syndrome
        ax.add_patch(patches.Circle((0, 0), 0.3, color=self.colors['syndrome'], alpha=0.7, label='Syndrome'))
        
        # Update legend
        ax.legend()
        
        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Surface Code (d={self.surface_code.distance}) Syndrome")
        
        return fig
    
    def visualize_error_correction(self, original_errors: Dict[str, List[int]],
                                  decoded_errors: Dict[str, List[int]],
                                  title: str = None) -> plt.Figure:
        """
        Visualize error correction on the surface code lattice.
        
        Args:
            original_errors: Dictionary with lists of qubits with original X and Z errors
            decoded_errors: Dictionary with lists of qubits with decoded X and Z errors
            title: Title for the plot
            
        Returns:
            Matplotlib figure showing the error correction
        """
        # Start with the basic lattice visualization
        fig = self.visualize_lattice(show_stabilizers=True, show_logical_operators=False)
        ax = fig.axes[0]
        
        # Extract error lists
        original_x_errors = original_errors.get('x_errors', [])
        original_z_errors = original_errors.get('z_errors', [])
        decoded_x_errors = decoded_errors.get('x_errors', [])
        decoded_z_errors = decoded_errors.get('z_errors', [])
        
        # Identify Y errors (qubits with both X and Z errors)
        original_y_errors = list(set(original_x_errors) & set(original_z_errors))
        
        # Remove Y errors from X and Z error lists
        original_x_errors = list(set(original_x_errors) - set(original_y_errors))
        original_z_errors = list(set(original_z_errors) - set(original_y_errors))
        
        # Plot original errors
        for qubit in original_x_errors:
            i, j = np.where(self.surface_code.qubit_grid == qubit)
            if len(i) > 0 and len(j) > 0:
                ax.add_patch(patches.Circle((j[0], -i[0]), 0.3, 
                                          color=self.colors['error_x'], alpha=0.7))
        
        for qubit in original_z_errors:
            i, j = np.where(self.surface_code.qubit_grid == qubit)
            if len(i) > 0 and len(j) > 0:
                ax.add_patch(patches.Circle((j[0], -i[0]), 0.3, 
                                          color=self.colors['error_z'], alpha=0.7))
        
        for qubit in original_y_errors:
            i, j = np.where(self.surface_code.qubit_grid == qubit)
            if len(i) > 0 and len(j) > 0:
                ax.add_patch(patches.Circle((j[0], -i[0]), 0.3, 
                                          color=self.colors['error_y'], alpha=0.7))
        
        # Plot decoded errors (corrections)
        for qubit in decoded_x_errors:
            i, j = np.where(self.surface_code.qubit_grid == qubit)
            if len(i) > 0 and len(j) > 0:
                ax.add_patch(patches.Rectangle((j[0] - 0.3, -i[0] - 0.3), 0.6, 0.6, 
                                             color=self.colors['correction'], alpha=0.7, fill=False, linewidth=2))
        
        for qubit in decoded_z_errors:
            i, j = np.where(self.surface_code.qubit_grid == qubit)
            if len(i) > 0 and len(j) > 0:
                ax.add_patch(patches.Rectangle((j[0] - 0.3, -i[0] - 0.3), 0.6, 0.6, 
                                             color=self.colors['correction'], alpha=0.7, fill=False, linewidth=2))
        
        # Add legend
        ax.add_patch(patches.Circle((0, 0), 0.3, color=self.colors['error_x'], alpha=0.7, label='X Error'))
        ax.add_patch(patches.Circle((0, 0), 0.3, color=self.colors['error_z'], alpha=0.7, label='Z Error'))
        ax.add_patch(patches.Circle((0, 0), 0.3, color=self.colors['error_y'], alpha=0.7, label='Y Error'))
        ax.add_patch(patches.Rectangle((0, 0), 0.6, 0.6, color=self.colors['correction'], 
                                     alpha=0.7, fill=False, linewidth=2, label='Correction'))
        
        # Update legend
        ax.legend()
        
        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Surface Code (d={self.surface_code.distance}) Error Correction")
        
        return fig


class SyndromeVisualizer:
    """
    Visualizes syndrome measurements and error correction.
    
    This class provides methods for visualizing syndrome measurements,
    error patterns, and the error correction process.
    """
    
    def __init__(self, surface_code: SurfaceCode):
        """
        Initialize the syndrome visualizer.
        
        Args:
            surface_code: The surface code to visualize
        """
        self.surface_code = surface_code
        self.surface_code_visualizer = SurfaceCodeVisualizer(surface_code)
    
    def visualize_error_syndrome_correction(self, x_errors: List[int], z_errors: List[int],
                                          x_syndrome: List[int], z_syndrome: List[int],
                                          decoded_errors: Dict[str, List[int]]) -> List[plt.Figure]:
        """
        Visualize the complete error correction process.
        
        Args:
            x_errors: List of qubits with X errors
            z_errors: List of qubits with Z errors
            x_syndrome: Syndrome measurements for X-stabilizers
            z_syndrome: Syndrome measurements for Z-stabilizers
            decoded_errors: Dictionary with lists of qubits with decoded X and Z errors
            
        Returns:
            List of matplotlib figures showing the error correction process
        """
        figures = []
        
        # 1. Visualize the original error pattern
        original_errors_fig = self.surface_code_visualizer.visualize_errors(
            x_errors, z_errors, title="Original Error Pattern")
        figures.append(original_errors_fig)
        
        # 2. Visualize the syndrome measurements
        syndrome_fig = self.surface_code_visualizer.visualize_syndrome(
            x_syndrome, z_syndrome, title="Syndrome Measurements")
        figures.append(syndrome_fig)
        
        # 3. Visualize the error correction
        correction_fig = self.surface_code_visualizer.visualize_error_correction(
            {'x_errors': x_errors, 'z_errors': z_errors},
            decoded_errors,
            title="Error Correction")
        figures.append(correction_fig)
        
        return figures
    
    def visualize_error_rates(self, physical_error_rates: List[float],
                             logical_error_rates: List[float],
                             distances: List[int]) -> plt.Figure:
        """
        Visualize the relationship between physical and logical error rates.
        
        Args:
            physical_error_rates: List of physical error rates
            logical_error_rates: List of logical error rates
            distances: List of code distances
            
        Returns:
            Matplotlib figure showing the error rate relationship
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the error rates
        for i, d in enumerate(distances):
            ax.semilogy(physical_error_rates, logical_error_rates[i], 'o-', label=f"d={d}")
        
        # Add threshold line if possible
        if len(physical_error_rates) > 1 and len(logical_error_rates) > 1 and len(distances) > 1:
            # Find the approximate threshold by looking for crossings
            threshold = None
            for i in range(len(physical_error_rates) - 1):
                for d1 in range(len(distances) - 1):
                    for d2 in range(d1 + 1, len(distances)):
                        if ((logical_error_rates[d1][i] > logical_error_rates[d2][i] and
                             logical_error_rates[d1][i+1] < logical_error_rates[d2][i+1]) or
                            (logical_error_rates[d1][i] < logical_error_rates[d2][i] and
                             logical_error_rates[d1][i+1] > logical_error_rates[d2][i+1])):
                            # Found a crossing
                            t = (physical_error_rates[i] + physical_error_rates[i+1]) / 2
                            if threshold is None or abs(t - 0.01) < abs(threshold - 0.01):
                                threshold = t
            
            if threshold is not None:
                ax.axvline(x=threshold, color='r', linestyle='--', 
                         label=f"Threshold ≈ {threshold:.3f}")
        
        # Set plot properties
        ax.set_xlabel("Physical Error Rate")
        ax.set_ylabel("Logical Error Rate")
        ax.set_title("Surface Code Error Correction Performance")
        ax.grid(True)
        ax.legend()
        
        return fig


class DecodingGraphVisualizer:
    """
    Visualizes the decoding graph and matching.
    
    This class provides methods for visualizing the decoding graph used in
    the minimum-weight perfect matching algorithm, as well as the matching
    solution.
    """
    
    def __init__(self, decoder: SurfaceCodeDecoder):
        """
        Initialize the decoding graph visualizer.
        
        Args:
            decoder: The surface code decoder to visualize
        """
        self.decoder = decoder
        self.surface_code = decoder.surface_code
    
    def visualize_decoding_graph(self, error_type: str = 'x') -> plt.Figure:
        """
        Visualize the decoding graph.
        
        Args:
            error_type: Type of error to visualize ('x' or 'z')
            
        Returns:
            Matplotlib figure showing the decoding graph
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Select the appropriate graph based on error type
        if error_type.lower() == 'x':
            graph = self.decoder.x_error_graph
            stabilizers = self.surface_code.z_stabilizers  # X errors are detected by Z stabilizers
            color = 'red'
        else:
            graph = self.decoder.z_error_graph
            stabilizers = self.surface_code.x_stabilizers  # Z errors are detected by X stabilizers
            color = 'blue'
        
        # Create a position dictionary for the graph nodes
        pos = {}
        for i, stabilizer in enumerate(stabilizers):
            x_coords = []
            y_coords = []
            for qubit in stabilizer:
                i_coords, j_coords = np.where(self.surface_code.qubit_grid == qubit)
                if len(i_coords) > 0 and len(j_coords) > 0:
                    y_coords.append(-i_coords[0])
                    x_coords.append(j_coords[0])
            if x_coords and y_coords:
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                pos[i] = (center_x, center_y)
        
        # Draw the graph
        nx.draw_networkx_nodes(graph, pos, node_color=color, node_size=300, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(graph, pos, font_size=10, font_color='white')
        
        # Draw edge weights
        edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
        
        # Set plot properties
        ax.set_title(f"Decoding Graph for {error_type.upper()} Errors")
        ax.set_axis_off()
        
        return fig
    
    def visualize_matching(self, syndrome: List[int], error_type: str = 'x') -> plt.Figure:
        """
        Visualize the minimum-weight perfect matching solution.
        
        Args:
            syndrome: Syndrome measurements
            error_type: Type of error to visualize ('x' or 'z')
            
        Returns:
            Matplotlib figure showing the matching solution
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Select the appropriate graph based on error type
        if error_type.lower() == 'x':
            graph = self.decoder.x_error_graph
            stabilizers = self.surface_code.z_stabilizers  # X errors are detected by Z stabilizers
            color = 'red'
        else:
            graph = self.decoder.z_error_graph
            stabilizers = self.surface_code.x_stabilizers  # Z errors are detected by X stabilizers
            color = 'blue'
        
        # Create a position dictionary for the graph nodes
        pos = {}
        for i, stabilizer in enumerate(stabilizers):
            x_coords = []
            y_coords = []
            for qubit in stabilizer:
                i_coords, j_coords = np.where(self.surface_code.qubit_grid == qubit)
                if len(i_coords) > 0 and len(j_coords) > 0:
                    y_coords.append(-i_coords[0])
                    x_coords.append(j_coords[0])
            if x_coords and y_coords:
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                pos[i] = (center_x, center_y)
        
        # Draw the graph
        nx.draw_networkx_nodes(graph, pos, node_color='gray', node_size=300, alpha=0.3)
        nx.draw_networkx_edges(graph, pos, width=0.5, alpha=0.2)
        
        # Find flipped stabilizers
        flipped = [i for i, s in enumerate(syndrome) if s == 1]
        
        # Draw flipped stabilizers
        nx.draw_networkx_nodes(graph, pos, nodelist=flipped, node_color=color, node_size=500, alpha=0.8)
        
        # Create a complete graph of flipped stabilizers
        matching_graph = nx.Graph()
        for i in range(len(flipped)):
            for j in range(i + 1, len(flipped)):
                s1 = flipped[i]
                s2 = flipped[j]
                
                # Find the shortest path between the stabilizers in the decoding graph
                try:
                    path = nx.shortest_path(graph, s1, s2, weight='weight')
                    weight = nx.shortest_path_length(graph, s1, s2, weight='weight')
                    matching_graph.add_edge(i, j, weight=weight, path=path)
                except nx.NetworkXNoPath:
                    # If there's no path, use a large weight
                    matching_graph.add_edge(i, j, weight=1000, path=[])
        
        # Find the minimum-weight perfect matching
        if len(flipped) % 2 == 1:
            # Add a virtual node to make the number of nodes even
            for i in range(len(flipped)):
                matching_graph.add_edge(i, len(flipped), weight=1000, path=[])
        
        # Use NetworkX's maximum_weight_matching with negative weights
        # to find the minimum-weight matching
        for u, v in matching_graph.edges():
            matching_graph[u][v]['weight'] = -matching_graph[u][v]['weight']
        
        matching = nx.algorithms.matching.max_weight_matching(matching_graph)
        
        # Draw the matching
        for u, v in matching:
            if u < len(flipped) and v < len(flipped):
                s1 = flipped[u]
                s2 = flipped[v]
                path = nx.shortest_path(graph, s1, s2, weight='weight')
                
                # Draw the path
                edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                nx.draw_networkx_edges(graph, pos, edgelist=edges, width=3.0, alpha=0.8, edge_color=color)
        
        # Set plot properties
        ax.set_title(f"Minimum-Weight Perfect Matching for {error_type.upper()} Errors")
        ax.set_axis_off()
        
        return fig


# Example usage
if __name__ == "__main__":
    # Create a surface code
    surface_code = SurfaceCode(distance=3, logical_qubits=1, use_rotated_lattice=True)
    
    # Create a surface code visualizer
    visualizer = SurfaceCodeVisualizer(surface_code)
    
    # Visualize the lattice
    lattice_fig = visualizer.visualize_lattice()
    plt.savefig('surface_code_lattice_visualization.png')
    
    # Create some example errors
    x_errors = [0, 4]
    z_errors = [2, 4]
    
    # Visualize the errors
    errors_fig = visualizer.visualize_errors(x_errors, z_errors)
    plt.savefig('surface_code_errors_visualization.png')
    
    # Create a decoder
    decoder = SurfaceCodeDecoder(surface_code)
    
    # Create a syndrome from the errors
    x_syndrome = [0, 1, 0]  # Example syndrome for X-stabilizers
    z_syndrome = [1, 0, 0]  # Example syndrome for Z-stabilizers
    
    # Visualize the syndrome
    syndrome_fig = visualizer.visualize_syndrome(x_syndrome, z_syndrome)
    plt.savefig('surface_code_syndrome_visualization.png')
    
    # Decode the syndrome
    decoded_errors = decoder.decode_syndrome(x_syndrome, z_syndrome)
    
    # Visualize the error correction
    correction_fig = visualizer.visualize_error_correction(
        {'x_errors': x_errors, 'z_errors': z_errors},
        decoded_errors)
    plt.savefig('surface_code_correction_visualization.png')
    
    # Create a syndrome visualizer
    syndrome_visualizer = SyndromeVisualizer(surface_code)
    
    # Visualize the complete error correction process
    process_figs = syndrome_visualizer.visualize_error_syndrome_correction(
        x_errors, z_errors, x_syndrome, z_syndrome, decoded_errors)
    
    # Create a decoding graph visualizer
    graph_visualizer = DecodingGraphVisualizer(decoder)
    
    # Visualize the decoding graph
    graph_fig = graph_visualizer.visualize_decoding_graph(error_type='x')
    plt.savefig('surface_code_decoding_graph_visualization.png')
    
    # Visualize the matching
    matching_fig = graph_visualizer.visualize_matching(x_syndrome, error_type='x')
    plt.savefig('surface_code_matching_visualization.png')