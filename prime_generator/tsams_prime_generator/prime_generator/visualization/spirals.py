"""
Prime number spiral visualizations.

This module provides functions to create various prime number spiral visualizations,
including the Ulam spiral, Sacks spiral, and hexagonal prime spiral.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, Optional, List, Dict, Any

from ..algorithms.testing import is_prime


def ulam_spiral(size: int, highlight_pattern: Optional[str] = None, 
                cmap: str = 'viridis', figsize: Tuple[int, int] = (10, 10),
                save_path: Optional[str] = None) -> plt.Figure:
    """
    Generate an Ulam spiral visualization of prime numbers.
    
    The Ulam spiral is created by arranging the positive integers in a spiral
    and highlighting the prime numbers, revealing interesting patterns.
    
    Args:
        size: The size of the spiral (size x size grid)
        highlight_pattern: Optional pattern to highlight ('primes', 'twin_primes', 'quadratic')
        cmap: Matplotlib colormap to use
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Create a grid for the spiral
    grid = np.zeros((size, size), dtype=int)
    
    # Fill the grid with numbers in a spiral pattern
    x, y = size // 2, size // 2
    num = 1
    grid[y, x] = num
    
    # Direction vectors: right, up, left, down
    dx = [1, 0, -1, 0]
    dy = [0, -1, 0, 1]
    
    direction = 0
    steps = 1
    
    while num < size * size:
        # Take 'steps' steps in the current direction
        for _ in range(2):
            for _ in range(steps):
                x += dx[direction]
                y += dy[direction]
                num += 1
                
                if 0 <= x < size and 0 <= y < size:
                    grid[y, x] = num
                    
                if num >= size * size:
                    break
            
            if num >= size * size:
                break
                
            direction = (direction + 1) % 4
            
        steps += 1
    
    # Create a mask for prime numbers
    prime_mask = np.zeros_like(grid, dtype=bool)
    for i in range(size):
        for j in range(size):
            if grid[i, j] > 1:
                prime_mask[i, j] = is_prime(grid[i, j])
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Apply different highlighting based on the pattern
    if highlight_pattern == 'twin_primes':
        # Highlight twin primes (primes that differ by 2)
        twin_mask = np.zeros_like(grid, dtype=bool)
        for i in range(size):
            for j in range(size):
                if prime_mask[i, j]:
                    n = grid[i, j]
                    if is_prime(n + 2) or is_prime(n - 2):
                        twin_mask[i, j] = True
        
        # Plot regular primes in one color and twin primes in another
        ax.imshow(~prime_mask, cmap='binary', interpolation='nearest')
        ax.imshow(twin_mask, cmap='hot', alpha=0.7, interpolation='nearest')
        title = "Ulam Spiral with Twin Primes Highlighted"
        
    elif highlight_pattern == 'quadratic':
        # Highlight primes from quadratic formulas like n² + n + 41
        quadratic_mask = np.zeros_like(grid, dtype=bool)
        for i in range(size):
            for j in range(size):
                n = grid[i, j]
                if n > 0:
                    # Check if n is of form k² + k + 41 for some k
                    k = int(np.sqrt(n - 41))
                    if k >= 0 and k**2 + k + 41 == n:
                        quadratic_mask[i, j] = True
        
        # Plot regular primes and quadratic formula primes
        ax.imshow(prime_mask, cmap='Blues', interpolation='nearest')
        ax.imshow(quadratic_mask, cmap='Reds', alpha=0.7, interpolation='nearest')
        title = "Ulam Spiral with Quadratic Formula Primes Highlighted"
        
    else:
        # Default: just highlight all primes
        ax.imshow(~prime_mask, cmap='binary', interpolation='nearest')
        ax.imshow(prime_mask, cmap=cmap, alpha=0.7, interpolation='nearest')
        title = "Ulam Spiral of Prime Numbers"
    
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def sacks_spiral(limit: int, dot_size: float = 0.5, 
                 cmap: str = 'viridis', figsize: Tuple[int, int] = (10, 10),
                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Generate a Sacks spiral visualization of prime numbers.
    
    The Sacks spiral arranges integers along an Archimedean spiral,
    with prime numbers highlighted, revealing different patterns than the Ulam spiral.
    
    Args:
        limit: The upper limit for numbers in the spiral
        dot_size: Size of the dots representing primes
        cmap: Matplotlib colormap to use
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Generate prime numbers up to the limit
    primes = []
    for n in range(2, limit + 1):
        if is_prime(n):
            primes.append(n)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate coordinates for each number along the spiral
    theta = np.sqrt(np.arange(1, limit + 1)) * 2 * np.pi
    x = np.cos(theta)
    y = np.sin(theta)
    
    # Plot all numbers as small dots
    ax.scatter(x, y, s=0.1, color='gray', alpha=0.3)
    
    # Extract coordinates for prime numbers
    prime_x = [x[p-1] for p in primes]
    prime_y = [y[p-1] for p in primes]
    
    # Create a colormap based on prime values
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(primes)))
    
    # Plot prime numbers as colored dots
    ax.scatter(prime_x, prime_y, s=dot_size, c=colors, alpha=0.8)
    
    ax.set_title("Sacks Spiral of Prime Numbers")
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def hexagonal_prime_spiral(size: int, figsize: Tuple[int, int] = (10, 10),
                          cmap: str = 'viridis', save_path: Optional[str] = None) -> plt.Figure:
    """
    Generate a hexagonal spiral visualization of prime numbers.
    
    This creates a spiral on a hexagonal grid, offering yet another perspective
    on prime number patterns.
    
    Args:
        size: The number of layers in the hexagonal spiral
        figsize: Figure size as (width, height) in inches
        cmap: Matplotlib colormap to use
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Direction vectors for hexagonal grid (six directions)
    # These represent the six neighbors in a hexagonal grid
    directions = [
        (1, 0),    # right
        (0, 1),    # up-right
        (-1, 1),   # up-left
        (-1, 0),   # left
        (0, -1),   # down-left
        (1, -1)    # down-right
    ]
    
    # Generate the hexagonal spiral coordinates
    coords = {}
    x, y = 0, 0
    coords[(x, y)] = 1  # Center
    
    num = 2
    for layer in range(1, size + 1):
        # Move to the start of this layer
        x += 1
        y -= 1
        coords[(x, y)] = num
        num += 1
        
        # Follow each of the six sides of the hexagon
        for direction in range(6):
            # The number of steps in this direction is equal to the layer number
            for _ in range(layer):
                dx, dy = directions[direction]
                x += dx
                y += dy
                coords[(x, y)] = num
                num += 1
    
    # Convert coordinates to arrays for plotting
    points = np.array(list(coords.keys()))
    values = np.array(list(coords.values()))
    
    # Identify prime numbers
    prime_mask = np.array([is_prime(v) for v in values])
    prime_points = points[prime_mask]
    prime_values = values[prime_mask]
    
    # Convert hexagonal coordinates to Cartesian for plotting
    # In a hexagonal grid, the x-coordinate is shifted based on the y-coordinate
    def hex_to_cartesian(coords):
        cart_x = coords[:, 0] + 0.5 * coords[:, 1]
        cart_y = coords[:, 1] * np.sqrt(3) / 2
        return cart_x, cart_y
    
    cart_x, cart_y = hex_to_cartesian(points)
    prime_cart_x, prime_cart_y = hex_to_cartesian(prime_points)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all numbers as small gray dots
    ax.scatter(cart_x, cart_y, s=1, color='gray', alpha=0.3)
    
    # Plot prime numbers with a colormap
    scatter = ax.scatter(prime_cart_x, prime_cart_y, s=10, 
                        c=prime_values, cmap=cmap, alpha=0.8)
    
    ax.set_title("Hexagonal Prime Spiral")
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.colorbar(scatter, label="Prime Numbers")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig