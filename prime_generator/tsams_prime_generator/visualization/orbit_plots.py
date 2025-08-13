"""
Orbit Visualization module.

This module provides tools for visualizing orbits of points under Möbius transformations,
including 2D plots, density plots, and animations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Tuple, Union, Optional, Callable, Any

from ..moebius.moebius_transformation import MoebiusTransformation
from ..state_space.state_space import StateSpace


def plot_orbit_2d(orbit: List[complex],
                 fig_size: Tuple[int, int] = (10, 10),
                 marker_size: int = 20,
                 color_map: str = 'viridis',
                 show_lines: bool = True,
                 show_grid: bool = True,
                 save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot an orbit in the complex plane.
    
    Args:
        orbit (List[complex]): The orbit to plot.
        fig_size (Tuple[int, int]): The size of the figure (default: (10, 10)).
        marker_size (int): The size of the markers (default: 20).
        color_map (str): The color map to use (default: 'viridis').
        show_lines (bool): Whether to connect the points with lines (default: True).
        show_grid (bool): Whether to show the grid (default: True).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes with the visualization.
    """
    # Filter out None values (points at infinity)
    finite_orbit = [p for p in orbit if p is not None]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Extract the real and imaginary parts
    real_parts = [p.real for p in finite_orbit]
    imag_parts = [p.imag for p in finite_orbit]
    
    # Plot the orbit
    colors = np.linspace(0, 1, len(finite_orbit))
    scatter = ax.scatter(real_parts, imag_parts, c=colors, cmap=color_map, s=marker_size)
    
    # Connect the points with lines if requested
    if show_lines:
        ax.plot(real_parts, imag_parts, 'k-', alpha=0.3)
    
    # Add a colorbar to show the time evolution
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Iteration')
    
    # Set the labels and title
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title(f'Orbit in Complex Plane ({len(finite_orbit)} points)')
    
    # Make the plot square
    ax.set_aspect('equal')
    
    # Show the grid if requested
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, ax


def plot_orbit_density(orbit: List[complex],
                      num_bins: int = 50,
                      fig_size: Tuple[int, int] = (10, 10),
                      color_map: str = 'viridis',
                      show_grid: bool = True,
                      save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the density of an orbit in the complex plane.
    
    Args:
        orbit (List[complex]): The orbit to plot.
        num_bins (int): The number of bins in each dimension (default: 50).
        fig_size (Tuple[int, int]): The size of the figure (default: (10, 10)).
        color_map (str): The color map to use (default: 'viridis').
        show_grid (bool): Whether to show the grid (default: True).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes with the visualization.
    """
    # Filter out None values (points at infinity)
    finite_orbit = [p for p in orbit if p is not None]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Extract the real and imaginary parts
    real_parts = [p.real for p in finite_orbit]
    imag_parts = [p.imag for p in finite_orbit]
    
    # Compute the 2D histogram
    hist, xedges, yedges = np.histogram2d(real_parts, imag_parts, bins=num_bins)
    
    # Plot the density
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(hist.T, extent=extent, origin='lower', aspect='auto', cmap=color_map)
    
    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count')
    
    # Set the labels and title
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title(f'Orbit Density in Complex Plane ({len(finite_orbit)} points)')
    
    # Show the grid if requested
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, ax


def plot_orbit_animation(orbit: List[complex],
                        fig_size: Tuple[int, int] = (10, 10),
                        marker_size: int = 20,
                        color: str = 'blue',
                        interval: int = 100,
                        save_path: Optional[str] = None) -> FuncAnimation:
    """
    Create an animation of an orbit in the complex plane.
    
    Args:
        orbit (List[complex]): The orbit to animate.
        fig_size (Tuple[int, int]): The size of the figure (default: (10, 10)).
        marker_size (int): The size of the markers (default: 20).
        color (str): The color of the markers (default: 'blue').
        interval (int): The interval between frames in milliseconds (default: 100).
        save_path (str, optional): The path to save the animation. If None, the animation is not saved.
        
    Returns:
        FuncAnimation: The animation object.
    """
    # Filter out None values (points at infinity)
    finite_orbit = [p for p in orbit if p is not None]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Set the limits based on the orbit
    real_parts = [p.real for p in finite_orbit]
    imag_parts = [p.imag for p in finite_orbit]
    
    min_real, max_real = min(real_parts), max(real_parts)
    min_imag, max_imag = min(imag_parts), max(imag_parts)
    
    # Add some padding
    padding = 0.1 * max(max_real - min_real, max_imag - min_imag)
    ax.set_xlim(min_real - padding, max_real + padding)
    ax.set_ylim(min_imag - padding, max_imag + padding)
    
    # Set the labels and title
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title('Orbit Animation in Complex Plane')
    
    # Make the plot square
    ax.set_aspect('equal')
    
    # Show the grid
    ax.grid(True, alpha=0.3)
    
    # Initialize the scatter plot
    scatter = ax.scatter([], [], s=marker_size, color=color)
    line, = ax.plot([], [], 'k-', alpha=0.3)
    
    # Define the update function for the animation
    def update(frame):
        scatter.set_offsets(np.column_stack((real_parts[:frame+1], imag_parts[:frame+1])))
        line.set_data(real_parts[:frame+1], imag_parts[:frame+1])
        ax.set_title(f'Orbit Animation in Complex Plane (Frame {frame+1}/{len(finite_orbit)})')
        return scatter, line
    
    # Create the animation
    animation = FuncAnimation(fig, update, frames=len(finite_orbit), interval=interval, blit=True)
    
    # Save the animation if requested
    if save_path is not None:
        animation.save(save_path, writer='pillow')
    
    return animation


def plot_orbit_comparison(orbit1: List[complex], orbit2: List[complex],
                         labels: Tuple[str, str] = ('Orbit 1', 'Orbit 2'),
                         fig_size: Tuple[int, int] = (12, 6),
                         marker_size: int = 20,
                         colors: Tuple[str, str] = ('blue', 'red'),
                         show_lines: bool = True,
                         show_grid: bool = True,
                         save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compare two orbits in the complex plane.
    
    Args:
        orbit1 (List[complex]): The first orbit to plot.
        orbit2 (List[complex]): The second orbit to plot.
        labels (Tuple[str, str]): The labels for the orbits (default: ('Orbit 1', 'Orbit 2')).
        fig_size (Tuple[int, int]): The size of the figure (default: (12, 6)).
        marker_size (int): The size of the markers (default: 20).
        colors (Tuple[str, str]): The colors of the markers (default: ('blue', 'red')).
        show_lines (bool): Whether to connect the points with lines (default: True).
        show_grid (bool): Whether to show the grid (default: True).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes with the visualization.
    """
    # Filter out None values (points at infinity)
    finite_orbit1 = [p for p in orbit1 if p is not None]
    finite_orbit2 = [p for p in orbit2 if p is not None]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Extract the real and imaginary parts
    real_parts1 = [p.real for p in finite_orbit1]
    imag_parts1 = [p.imag for p in finite_orbit1]
    
    real_parts2 = [p.real for p in finite_orbit2]
    imag_parts2 = [p.imag for p in finite_orbit2]
    
    # Plot the first orbit
    ax.scatter(real_parts1, imag_parts1, s=marker_size, color=colors[0], label=labels[0])
    if show_lines:
        ax.plot(real_parts1, imag_parts1, color=colors[0], alpha=0.3)
    
    # Plot the second orbit
    ax.scatter(real_parts2, imag_parts2, s=marker_size, color=colors[1], label=labels[1])
    if show_lines:
        ax.plot(real_parts2, imag_parts2, color=colors[1], alpha=0.3)
    
    # Set the labels and title
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title('Orbit Comparison in Complex Plane')
    
    # Add a legend
    ax.legend()
    
    # Make the plot square
    ax.set_aspect('equal')
    
    # Show the grid if requested
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, ax


def plot_orbit_statistics(orbit: List[complex],
                         fig_size: Tuple[int, int] = (12, 8),
                         save_path: Optional[str] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot various statistics of an orbit.
    
    Args:
        orbit (List[complex]): The orbit to analyze.
        fig_size (Tuple[int, int]): The size of the figure (default: (12, 8)).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, List[plt.Axes]]: The figure and axes with the visualizations.
    """
    # Filter out None values (points at infinity)
    finite_orbit = [p for p in orbit if p is not None]
    
    # Create the figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=fig_size)
    
    # Extract the real and imaginary parts
    real_parts = [p.real for p in finite_orbit]
    imag_parts = [p.imag for p in finite_orbit]
    
    # Plot 1: The orbit in the complex plane
    axs[0, 0].scatter(real_parts, imag_parts, s=10)
    axs[0, 0].set_xlabel('Re(z)')
    axs[0, 0].set_ylabel('Im(z)')
    axs[0, 0].set_title('Orbit in Complex Plane')
    axs[0, 0].set_aspect('equal')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram of real parts
    axs[0, 1].hist(real_parts, bins=30, alpha=0.7)
    axs[0, 1].set_xlabel('Re(z)')
    axs[0, 1].set_ylabel('Count')
    axs[0, 1].set_title('Distribution of Real Parts')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Histogram of imaginary parts
    axs[1, 0].hist(imag_parts, bins=30, alpha=0.7)
    axs[1, 0].set_xlabel('Im(z)')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].set_title('Distribution of Imaginary Parts')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot of consecutive points
    if len(finite_orbit) > 1:
        axs[1, 1].scatter([p.real for p in finite_orbit[:-1]], [p.real for p in finite_orbit[1:]], s=10)
        axs[1, 1].set_xlabel('Re(z_n)')
        axs[1, 1].set_ylabel('Re(z_{n+1})')
        axs[1, 1].set_title('Return Map of Real Parts')
        axs[1, 1].set_aspect('equal')
        axs[1, 1].grid(True, alpha=0.3)
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, axs