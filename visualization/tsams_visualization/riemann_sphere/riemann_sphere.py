"""
Möbius Transformation Visualizer

This module provides tools for visualizing molecular structures using Möbius transformations,
which enable intuitive representation of complex biomolecular structures and their dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import torch
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import time

# Import core TIBEDO components if available
try:
    from tibedo.core.advanced.mobius_pairing import MobiusPairing
    from tibedo.core.spinor.reduction_chain import SpinorReductionChain
    from tibedo.core.prime_indexed.prime_indexed_structure import PrimeIndexedStructure
    TIBEDO_CORE_AVAILABLE = True
except ImportError:
    TIBEDO_CORE_AVAILABLE = False
    print("Warning: TIBEDO core components not available. Using standalone implementation.")

# Import performance optimization components if available
try:
    from tibedo.performance.gpu_acceleration import GPUAccelerator
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class MobiusTransformationVisualizer:
    """
    A class for visualizing molecular structures using Möbius transformations.
    
    This class provides tools for mapping molecular structures onto Möbius strips
    and other topological surfaces, enabling intuitive visualization of complex
    biomolecular structures and their dynamics.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the MobiusTransformationVisualizer.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available.
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            self.gpu_accelerator = GPUAccelerator()
        
        # Initialize Möbius pairing if TIBEDO core is available
        if TIBEDO_CORE_AVAILABLE:
            self.mobius_pairing = MobiusPairing()
        
        # Set default visualization parameters
        self.default_params = {
            'strip_width': 1.0,
            'strip_resolution': 100,
            'color_map': 'viridis',
            'alpha': 0.7,
            'point_size': 50,
            'line_width': 2,
            'show_grid': True,
            'show_axes': True,
            'show_labels': True,
            'show_title': True,
            'show_legend': True,
            'show_colorbar': True,
            'show_path_integral': True,
            'show_energy': True,
            'show_binding_sites': True,
            'show_interactions': True,
            'show_hydrogen_bonds': True,
            'show_hydrophobic_interactions': True,
            'show_ionic_interactions': True,
            'show_pi_stacking': True,
            'show_cation_pi': True,
            'show_halogen_bonds': True,
            'show_metal_coordination': True,
            'show_water_bridges': True,
            'show_salt_bridges': True,
            'show_disulfide_bonds': True,
            'show_aromatic_interactions': True,
            'show_vdw_interactions': True,
        }
    
    def create_mobius_strip(self, 
                           width: float = 1.0, 
                           resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a Möbius strip for visualization.
        
        Args:
            width: Width of the Möbius strip.
            resolution: Resolution of the Möbius strip mesh.
            
        Returns:
            Tuple of X, Y, Z coordinates of the Möbius strip mesh.
        """
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(-width / 2, width / 2, resolution)
        u_grid, v_grid = np.meshgrid(u, v)
        
        # Möbius strip parametric equations
        x = (1 + v_grid * np.cos(u_grid / 2)) * np.cos(u_grid)
        y = (1 + v_grid * np.cos(u_grid / 2)) * np.sin(u_grid)
        z = v_grid * np.sin(u_grid / 2)
        
        return x, y, z
    
    def map_sequence_to_mobius(self, 
                              sequence: List[np.ndarray], 
                              sequence_properties: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Map a sequence of points (e.g., amino acids in a protein) to a Möbius strip.
        
        Args:
            sequence: List of 3D coordinates representing the sequence.
            sequence_properties: Optional list of properties for coloring.
            
        Returns:
            Tuple of X, Y, Z coordinates and properties mapped to the Möbius strip.
        """
        n_points = len(sequence)
        
        if sequence_properties is None:
            sequence_properties = np.arange(n_points)
        
        # Map sequence to Möbius strip parameters
        u = np.linspace(0, 2 * np.pi, n_points)
        v = np.zeros(n_points)
        
        # Use TIBEDO's MobiusPairing if available for more accurate mapping
        if TIBEDO_CORE_AVAILABLE:
            try:
                # Convert sequence to tensor format expected by MobiusPairing
                sequence_tensor = torch.tensor(np.array(sequence), dtype=torch.float32)
                
                # Use MobiusPairing to compute optimal mapping parameters
                u, v = self.mobius_pairing.compute_optimal_mapping(sequence_tensor)
                u = u.numpy()
                v = v.numpy()
            except Exception as e:
                print(f"Warning: Error using MobiusPairing: {e}. Falling back to simple mapping.")
        
        # Compute Möbius strip coordinates for the sequence
        x = (1 + v * np.cos(u / 2)) * np.cos(u)
        y = (1 + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        
        return x, y, z, np.array(sequence_properties)
    
    def calculate_path_integral(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """
        Calculate the path integral along the sequence mapped to the Möbius strip.
        
        Args:
            x, y, z: Coordinates of the sequence mapped to the Möbius strip.
            
        Returns:
            Path integral value.
        """
        # Calculate distances between consecutive points
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        
        # Calculate segment lengths
        segment_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Sum segment lengths to get path integral
        path_integral = np.sum(segment_lengths)
        
        return path_integral
    
    def visualize_sequence_on_mobius(self, 
                                    sequence: List[np.ndarray], 
                                    sequence_properties: Optional[List[float]] = None,
                                    title: str = "Protein Folding Path on Möbius Strip",
                                    subtitle: str = "",
                                    save_path: Optional[str] = None,
                                    show: bool = True,
                                    **kwargs) -> plt.Figure:
        """
        Visualize a sequence (e.g., protein structure) on a Möbius strip.
        
        Args:
            sequence: List of 3D coordinates representing the sequence.
            sequence_properties: Optional list of properties for coloring.
            title: Title of the plot.
            subtitle: Subtitle of the plot.
            save_path: Path to save the figure.
            show: Whether to show the figure.
            **kwargs: Additional visualization parameters.
            
        Returns:
            Matplotlib figure object.
        """
        # Update visualization parameters with provided kwargs
        params = self.default_params.copy()
        params.update(kwargs)
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create Möbius strip
        strip_x, strip_y, strip_z = self.create_mobius_strip(
            width=params['strip_width'],
            resolution=params['strip_resolution']
        )
        
        # Plot Möbius strip
        surf = ax.plot_surface(strip_x, strip_y, strip_z, alpha=params['alpha'], 
                              color='lightblue', edgecolor='none')
        
        # Map sequence to Möbius strip
        seq_x, seq_y, seq_z, seq_props = self.map_sequence_to_mobius(
            sequence, sequence_properties
        )
        
        # Calculate path integral
        path_integral = self.calculate_path_integral(seq_x, seq_y, seq_z)
        
        # Plot sequence on Möbius strip
        scatter = ax.scatter(seq_x, seq_y, seq_z, c=seq_props, 
                           s=params['point_size'], 
                           cmap=params['color_map'], 
                           edgecolor='black')
        
        # Plot line connecting sequence points
        ax.plot(seq_x, seq_y, seq_z, color='red', linewidth=params['line_width'])
        
        # Set plot properties
        if params['show_title']:
            full_title = f"{title}\n{subtitle}\nPath Integral: {path_integral:.4f}"
            ax.set_title(full_title, fontsize=14)
        
        if params['show_axes']:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        if params['show_grid']:
            ax.grid(True)
        
        if params['show_colorbar']:
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Sequence Position')
        
        # Add description text
        if params['show_legend']:
            description = (
                "Visualization of protein folding using quaternion-based Möbius strip dual pairing.\n"
                "The colored path represents the protein's amino acid sequence mapped to the Möbius strip.\n"
                "Colors indicate the position in the sequence from start (blue) to end (red)."
            )
            fig.text(0.02, 0.02, description, wrap=True, fontsize=10)
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show figure if requested
        if show:
            plt.tight_layout()
            plt.show()
        
        return fig
    
    def create_interactive_visualization(self, 
                                        sequence: List[np.ndarray], 
                                        sequence_properties: Optional[List[float]] = None,
                                        title: str = "Interactive Protein Visualization",
                                        save_path: Optional[str] = None,
                                        **kwargs) -> FuncAnimation:
        """
        Create an interactive visualization of a sequence on a Möbius strip.
        
        Args:
            sequence: List of 3D coordinates representing the sequence.
            sequence_properties: Optional list of properties for coloring.
            title: Title of the plot.
            save_path: Path to save the animation.
            **kwargs: Additional visualization parameters.
            
        Returns:
            Matplotlib animation object.
        """
        # Update visualization parameters with provided kwargs
        params = self.default_params.copy()
        params.update(kwargs)
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create Möbius strip
        strip_x, strip_y, strip_z = self.create_mobius_strip(
            width=params['strip_width'],
            resolution=params['strip_resolution']
        )
        
        # Map sequence to Möbius strip
        seq_x, seq_y, seq_z, seq_props = self.map_sequence_to_mobius(
            sequence, sequence_properties
        )
        
        # Calculate path integral
        path_integral = self.calculate_path_integral(seq_x, seq_y, seq_z)
        
        # Plot Möbius strip
        surf = ax.plot_surface(strip_x, strip_y, strip_z, alpha=params['alpha'], 
                              color='lightblue', edgecolor='none')
        
        # Initialize empty scatter and line plots
        scatter = ax.scatter([], [], [], s=params['point_size'], 
                           cmap=params['color_map'], edgecolor='black')
        line, = ax.plot([], [], [], color='red', linewidth=params['line_width'])
        
        # Set plot properties
        if params['show_title']:
            full_title = f"{title}\nPath Integral: {path_integral:.4f}"
            ax.set_title(full_title, fontsize=14)
        
        if params['show_axes']:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        if params['show_grid']:
            ax.grid(True)
        
        # Set axis limits
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-0.5, 0.5])
        
        # Animation update function
        def update(frame):
            # Update view angle
            ax.view_init(elev=30, azim=frame)
            
            # Update scatter and line data
            scatter._offsets3d = (seq_x, seq_y, seq_z)
            scatter.set_array(seq_props)
            
            line.set_data(seq_x[:frame+1], seq_y[:frame+1])
            line.set_3d_properties(seq_z[:frame+1])
            
            return scatter, line
        
        # Create animation
        n_frames = 360  # Full rotation
        animation = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
        
        # Save animation if path is provided
        if save_path:
            animation.save(save_path, writer='pillow', fps=30, dpi=100)
        
        return animation
    
    def visualize_energy_landscape(self,
                                  sequence: List[np.ndarray],
                                  energies: List[float],
                                  title: str = "Energy Landscape on Möbius Strip",
                                  save_path: Optional[str] = None,
                                  show: bool = True,
                                  **kwargs) -> plt.Figure:
        """
        Visualize the energy landscape of a sequence on a Möbius strip.
        
        Args:
            sequence: List of 3D coordinates representing the sequence.
            energies: List of energy values for each point in the sequence.
            title: Title of the plot.
            save_path: Path to save the figure.
            show: Whether to show the figure.
            **kwargs: Additional visualization parameters.
            
        Returns:
            Matplotlib figure object.
        """
        # Update visualization parameters with provided kwargs
        params = self.default_params.copy()
        params.update(kwargs)
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create Möbius strip
        strip_x, strip_y, strip_z = self.create_mobius_strip(
            width=params['strip_width'],
            resolution=params['strip_resolution']
        )
        
        # Map sequence to Möbius strip
        seq_x, seq_y, seq_z, _ = self.map_sequence_to_mobius(sequence)
        
        # Normalize energies for coloring
        energies_array = np.array(energies)
        normalized_energies = (energies_array - np.min(energies_array)) / (np.max(energies_array) - np.min(energies_array))
        
        # Plot Möbius strip
        surf = ax.plot_surface(strip_x, strip_y, strip_z, alpha=params['alpha'], 
                              color='lightblue', edgecolor='none')
        
        # Plot sequence on Möbius strip with energy coloring
        scatter = ax.scatter(seq_x, seq_y, seq_z, c=energies, 
                           s=params['point_size'], 
                           cmap='coolwarm', 
                           edgecolor='black')
        
        # Plot line connecting sequence points
        points = np.array([seq_x, seq_y, seq_z]).T
        segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
        
        # Create line collection with energy-based coloring
        from matplotlib.collections import LineCollection
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        
        # Create line segments
        line_segments = []
        for i in range(len(seq_x) - 1):
            line_segments.append([(seq_x[i], seq_y[i], seq_z[i]), 
                                 (seq_x[i+1], seq_y[i+1], seq_z[i+1])])
        
        # Create line collection
        energy_colors = plt.cm.coolwarm(normalized_energies)
        line_collection = Line3DCollection(line_segments, colors=energy_colors, linewidths=params['line_width'])
        ax.add_collection(line_collection)
        
        # Set plot properties
        if params['show_title']:
            ax.set_title(title, fontsize=14)
        
        if params['show_axes']:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        if params['show_grid']:
            ax.grid(True)
        
        if params['show_colorbar']:
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Energy (kcal/mol)')
        
        # Add description text
        if params['show_legend']:
            description = (
                "Visualization of energy landscape using quaternion-based Möbius strip mapping.\n"
                "The colored path represents the energy values along the protein's conformational space.\n"
                "Colors indicate energy values from low (blue) to high (red)."
            )
            fig.text(0.02, 0.02, description, wrap=True, fontsize=10)
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show figure if requested
        if show:
            plt.tight_layout()
            plt.show()
        
        return fig