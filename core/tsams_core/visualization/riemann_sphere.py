"""
Riemann Sphere Visualization module.

This module provides tools for visualizing Möbius transformations on the Riemann sphere,
which is a powerful way to understand the geometric action of these transformations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Union, Optional, Callable, Any

from ..moebius.moebius_transformation import MoebiusTransformation
from ..utils.complex_analysis import stereographic_projection, inverse_stereographic_projection


def plot_riemann_sphere(ax: Optional[plt.Axes] = None, grid_density: int = 20, 
                        alpha: float = 0.3, color: str = 'b', 
                        show_equator: bool = True) -> plt.Axes:
    """
    Plot the Riemann sphere.
    
    Args:
        ax (plt.Axes, optional): The matplotlib axes to plot on. If None, a new figure is created.
        grid_density (int): The density of the grid on the sphere (default: 20).
        alpha (float): The transparency of the sphere (default: 0.3).
        color (str): The color of the sphere (default: 'b').
        show_equator (bool): Whether to show the equator of the sphere (default: True).
        
    Returns:
        plt.Axes: The matplotlib axes with the Riemann sphere plotted.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Create a grid of points on the sphere
    u = np.linspace(0, 2 * np.pi, grid_density)
    v = np.linspace(0, np.pi, grid_density)
    
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot the sphere
    ax.plot_surface(x, y, z, color=color, alpha=alpha)
    
    # Show the equator if requested
    if show_equator:
        theta = np.linspace(0, 2 * np.pi, 100)
        x_eq = np.cos(theta)
        y_eq = np.sin(theta)
        z_eq = np.zeros_like(theta)
        ax.plot(x_eq, y_eq, z_eq, 'k-', linewidth=2)
    
    # Set the axes limits and labels
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return ax


def plot_mobius_transformation(transformation: MoebiusTransformation, 
                              grid_density: int = 20, 
                              fig_size: Tuple[int, int] = (12, 6),
                              save_path: Optional[str] = None) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Visualize a Möbius transformation on the Riemann sphere.
    
    Args:
        transformation (MoebiusTransformation): The Möbius transformation to visualize.
        grid_density (int): The density of the grid on the sphere (default: 20).
        fig_size (Tuple[int, int]): The size of the figure (default: (12, 6)).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]: The figure and axes with the visualization.
    """
    # Create a grid of points in the complex plane
    x = np.linspace(-2, 2, grid_density)
    y = np.linspace(-2, 2, grid_density)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Apply the transformation to each point
    W = np.zeros_like(Z, dtype=complex)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            W[i, j] = transformation.apply(Z[i, j])
    
    # Convert the points to the Riemann sphere
    X_sphere = np.zeros_like(X)
    Y_sphere = np.zeros_like(Y)
    Z_sphere = np.zeros_like(X)
    
    X_transformed = np.zeros_like(X)
    Y_transformed = np.zeros_like(Y)
    Z_transformed = np.zeros_like(X)
    
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            # Original point
            x_s, y_s, z_s = stereographic_projection(Z[i, j])
            X_sphere[i, j] = x_s
            Y_sphere[i, j] = y_s
            Z_sphere[i, j] = z_s
            
            # Transformed point
            x_t, y_t, z_t = stereographic_projection(W[i, j])
            X_transformed[i, j] = x_t
            Y_transformed[i, j] = y_t
            Z_transformed[i, j] = z_t
    
    # Create the figure
    fig = plt.figure(figsize=fig_size)
    
    # Plot the original points on the Riemann sphere
    ax1 = fig.add_subplot(121, projection='3d')
    ax1 = plot_riemann_sphere(ax1, grid_density=grid_density)
    ax1.plot_surface(X_sphere, Y_sphere, Z_sphere, color='b', alpha=0.3)
    ax1.set_title('Original Points on Riemann Sphere')
    
    # Plot the transformed points on the Riemann sphere
    ax2 = fig.add_subplot(122, projection='3d')
    ax2 = plot_riemann_sphere(ax2, grid_density=grid_density, color='r')
    ax2.plot_surface(X_transformed, Y_transformed, Z_transformed, color='r', alpha=0.3)
    ax2.set_title('Transformed Points on Riemann Sphere')
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, (ax1, ax2)


def plot_orbit_on_sphere(orbit: List[complex], 
                         fig_size: Tuple[int, int] = (10, 10),
                         marker_size: int = 20,
                         color_map: str = 'viridis',
                         save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot an orbit on the Riemann sphere.
    
    Args:
        orbit (List[complex]): The orbit to plot.
        fig_size (Tuple[int, int]): The size of the figure (default: (10, 10)).
        marker_size (int): The size of the markers (default: 20).
        color_map (str): The color map to use (default: 'viridis').
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes with the visualization.
    """
    # Create the figure
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the Riemann sphere
    ax = plot_riemann_sphere(ax)
    
    # Convert the orbit points to the Riemann sphere
    x_points = []
    y_points = []
    z_points = []
    
    for point in orbit:
        if point is None:  # Point at infinity
            x, y, z = 0, 0, 1  # North pole
        else:
            x, y, z = stereographic_projection(point)
        
        x_points.append(x)
        y_points.append(y)
        z_points.append(z)
    
    # Plot the orbit
    colors = np.linspace(0, 1, len(x_points))
    scatter = ax.scatter(x_points, y_points, z_points, c=colors, cmap=color_map, s=marker_size)
    
    # Add a colorbar to show the time evolution
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Iteration')
    
    # Connect the points with lines
    ax.plot(x_points, y_points, z_points, 'k-', alpha=0.3)
    
    # Set the title
    ax.set_title(f'Orbit on Riemann Sphere ({len(orbit)} points)')
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, ax


def plot_fixed_points(transformation: MoebiusTransformation, 
                     ax: Optional[plt.Axes] = None,
                     marker_size: int = 100,
                     save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the fixed points of a Möbius transformation on the Riemann sphere.
    
    Args:
        transformation (MoebiusTransformation): The Möbius transformation.
        ax (plt.Axes, optional): The matplotlib axes to plot on. If None, a new figure is created.
        marker_size (int): The size of the markers (default: 100).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes with the visualization.
    """
    # Create the figure if needed
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax = plot_riemann_sphere(ax)
    else:
        fig = ax.figure
    
    # Get the fixed points
    fixed_points = transformation.fixed_points()
    
    # Plot each fixed point
    for i, point in enumerate(fixed_points):
        if point is None:  # Point at infinity
            x, y, z = 0, 0, 1  # North pole
        else:
            x, y, z = stereographic_projection(point)
        
        ax.scatter([x], [y], [z], color='r', s=marker_size, marker='*', label=f'Fixed Point {i+1}')
    
    # Add a legend
    ax.legend()
    
    # Set the title
    ax.set_title(f'Fixed Points of {transformation}')
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, ax


def plot_transformation_classification(transformation: MoebiusTransformation,
                                      fig_size: Tuple[int, int] = (10, 10),
                                      save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize the classification of a Möbius transformation on the Riemann sphere.
    
    Args:
        transformation (MoebiusTransformation): The Möbius transformation.
        fig_size (Tuple[int, int]): The size of the figure (default: (10, 10)).
        save_path (str, optional): The path to save the figure. If None, the figure is not saved.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes with the visualization.
    """
    # Create the figure
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the Riemann sphere
    ax = plot_riemann_sphere(ax)
    
    # Plot the fixed points
    fig, ax = plot_fixed_points(transformation, ax)
    
    # Determine the classification
    if transformation.is_elliptic():
        classification = "Elliptic"
        color = "blue"
    elif transformation.is_parabolic():
        classification = "Parabolic"
        color = "green"
    elif transformation.is_hyperbolic():
        classification = "Hyperbolic"
        color = "red"
    else:
        classification = "Loxodromic"
        color = "purple"
    
    # Update the title
    ax.set_title(f'{classification} Transformation: {transformation}')
    
    # Add a text box with the classification
    props = dict(boxstyle='round', facecolor=color, alpha=0.5)
    ax.text2D(0.05, 0.95, classification, transform=ax.transAxes, fontsize=14,
             verticalalignment='top', bbox=props)
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig, ax