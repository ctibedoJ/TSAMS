"""
Interactive prime number visualization tools.

This module provides interactive visualizations for exploring prime numbers,
including animated visualizations and interactive explorers.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from typing import Tuple, Optional, List, Dict, Any, Union, Callable
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import io
import base64

from ..algorithms.generation import sieve_of_eratosthenes
from ..algorithms.testing import is_prime
from .spirals import ulam_spiral, sacks_spiral


def animated_sieve(limit: int = 100, interval: int = 200,
                  figsize: Tuple[int, int] = (8, 8),
                  save_path: Optional[str] = None) -> FuncAnimation:
    """
    Create an animation of the Sieve of Eratosthenes algorithm.
    
    Args:
        limit: The upper limit for the sieve
        interval: Time interval between frames in milliseconds
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the animation as a GIF
        
    Returns:
        Matplotlib animation object
    """
    # Create a grid size that's approximately square
    size = int(np.ceil(np.sqrt(limit)))
    grid_size = size * size
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Initialize the grid
    grid = np.arange(1, grid_size + 1)
    grid = grid.reshape(size, size)
    
    # Create a mask for numbers beyond our limit
    beyond_limit = grid > limit
    
    # Create a boolean array to track which numbers have been marked as composite
    sieve = np.zeros_like(grid, dtype=bool)
    sieve[0, 0] = True  # Mark 1 as not prime
    
    # Create the initial plot
    cmap = plt.cm.get_cmap('viridis', 3)
    colors = [cmap(0), cmap(1), cmap(2)]  # For unmarked, current, marked
    
    # Custom colormap: unmarked (white), current (red), marked (gray)
    custom_cmap = mcolors.ListedColormap(['white', 'red', 'lightgray'])
    
    # Initial state: all unmarked except 1
    state = np.zeros_like(grid, dtype=int)
    state[0, 0] = 2  # Mark 1 as not prime
    state[beyond_limit] = 2  # Mark numbers beyond limit
    
    # Create the initial plot
    im = ax.imshow(state, cmap=custom_cmap, interpolation='nearest')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    
    # Add text annotations for each cell
    texts = []
    for i in range(size):
        for j in range(size):
            if grid[i, j] <= limit:
                texts.append(ax.text(j, i, str(grid[i, j]), 
                                    ha='center', va='center', fontsize=8))
    
    # Add title
    title = ax.set_title('Sieve of Eratosthenes: Initializing')
    
    # Function to update the animation
    def update(frame):
        nonlocal sieve
        
        if frame == 0:
            # Reset for animation restart
            sieve = np.zeros_like(grid, dtype=bool)
            sieve[0, 0] = True  # Mark 1 as not prime
            state.fill(0)
            state[0, 0] = 2  # Mark 1 as not prime
            state[beyond_limit] = 2  # Mark numbers beyond limit
            title.set_text('Sieve of Eratosthenes: Initializing')
            return im,
        
        # Find the next unmarked number (which is prime)
        flat_indices = np.where((~sieve) & (~beyond_limit))[0]
        if len(flat_indices) == 0:
            title.set_text('Sieve of Eratosthenes: Complete')
            return im,
        
        # Get the smallest unmarked number
        min_idx = flat_indices[0]
        i, j = min_idx // size, min_idx % size
        current_prime = grid[i, j]
        
        # Reset previous current marker
        state[state == 1] = 0
        
        # Mark the current prime
        state[i, j] = 1
        
        # Mark all multiples of the current prime
        for k in range(current_prime * current_prime, grid_size + 1, current_prime):
            if k <= limit:
                # Convert k to grid coordinates
                ki, kj = (k - 1) // size, (k - 1) % size
                sieve[ki, kj] = True
                state[ki, kj] = 2
        
        title.set_text(f'Sieve of Eratosthenes: Marking multiples of {current_prime}')
        
        # Update the image
        im.set_array(state)
        
        return im,
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=limit, interval=interval, blit=True)
    
    # Save the animation if a path is provided
    if save_path:
        anim.save(save_path, writer='pillow', fps=1000//interval)
    
    return anim


def interactive_prime_explorer(max_limit: int = 1000) -> None:
    """
    Create an interactive widget for exploring prime numbers.
    
    This function creates interactive widgets for exploring different
    prime number visualizations and properties.
    
    Args:
        max_limit: Maximum limit for prime generation
        
    Returns:
        None (displays interactive widgets)
    """
    # Create widgets
    visualization_type = widgets.Dropdown(
        options=['Ulam Spiral', 'Sacks Spiral', 'Prime Distribution', 'Prime Gaps'],
        value='Ulam Spiral',
        description='Visualization:',
    )
    
    limit_slider = widgets.IntSlider(
        value=100,
        min=10,
        max=max_limit,
        step=10,
        description='Limit:',
    )
    
    # Additional options for specific visualizations
    ulam_options = widgets.Dropdown(
        options=['primes', 'twin_primes', 'quadratic'],
        value='primes',
        description='Highlight:',
    )
    
    # Output widget for displaying the visualization
    output = widgets.Output()
    
    # Function to update the visualization
    def update_visualization(_):
        with output:
            clear_output(wait=True)
            
            # Create figure based on selected visualization type
            if visualization_type.value == 'Ulam Spiral':
                size = int(np.ceil(np.sqrt(limit_slider.value)))
                highlight = None if ulam_options.value == 'primes' else ulam_options.value
                fig = ulam_spiral(size, highlight_pattern=highlight)
                plt.show()
                
            elif visualization_type.value == 'Sacks Spiral':
                fig = sacks_spiral(limit_slider.value)
                plt.show()
                
            elif visualization_type.value == 'Prime Distribution':
                # Generate primes
                primes = sieve_of_eratosthenes(limit_slider.value)
                
                # Create histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                bin_size = max(1, limit_slider.value // 20)
                bins = np.arange(0, limit_slider.value + bin_size, bin_size)
                ax.hist(primes, bins=bins, alpha=0.7, color='royalblue', edgecolor='black')
                ax.set_xlabel('Number Range')
                ax.set_ylabel('Count of Primes')
                ax.set_title(f'Distribution of Prime Numbers up to {limit_slider.value}')
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.show()
                
            elif visualization_type.value == 'Prime Gaps':
                # Generate primes
                primes = sieve_of_eratosthenes(limit_slider.value)
                
                # Calculate gaps
                gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(primes[:-1], gaps, 'o-', markersize=4, alpha=0.7)
                ax.set_xlabel('Prime Number')
                ax.set_ylabel('Gap to Next Prime')
                ax.set_title(f'Gaps Between Consecutive Primes up to {limit_slider.value}')
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.show()
    
    # Connect the widgets to the update function
    visualization_type.observe(update_visualization, names='value')
    limit_slider.observe(update_visualization, names='value')
    ulam_options.observe(update_visualization, names='value')
    
    # Create the UI layout
    options_box = widgets.VBox([visualization_type, limit_slider])
    conditional_options = widgets.VBox([ulam_options])
    
    # Function to show/hide conditional options based on visualization type
    def update_options(_):
        if visualization_type.value == 'Ulam Spiral':
            conditional_options.layout.display = 'block'
        else:
            conditional_options.layout.display = 'none'
    
    visualization_type.observe(update_options, names='value')
    update_options(None)  # Initial update
    
    # Display the widgets
    display(widgets.VBox([options_box, conditional_options, output]))
    
    # Initial visualization
    update_visualization(None)


def prime_pattern_viewer(pattern_type: str = 'polynomial', 
                        parameters: Dict[str, Any] = None) -> None:
    """
    Interactive viewer for exploring patterns in prime numbers.
    
    Args:
        pattern_type: Type of pattern to explore ('polynomial', 'modular', 'sequence')
        parameters: Dictionary of parameters specific to the pattern type
        
    Returns:
        None (displays interactive widgets)
    """
    if parameters is None:
        parameters = {}
    
    # Default parameters
    if pattern_type == 'polynomial':
        a = parameters.get('a', 1)
        b = parameters.get('b', 1)
        c = parameters.get('c', 41)  # Default to n² + n + 41
        
        # Create widgets
        a_slider = widgets.IntSlider(value=a, min=-10, max=10, description='a:')
        b_slider = widgets.IntSlider(value=b, min=-20, max=20, description='b:')
        c_slider = widgets.IntSlider(value=c, min=-50, max=50, description='c:')
        limit_slider = widgets.IntSlider(value=100, min=10, max=1000, description='Limit:')
        
        # Output widget
        output = widgets.Output()
        
        # Update function
        def update(_):
            with output:
                clear_output(wait=True)
                
                # Generate values from the polynomial
                n_values = np.arange(limit_slider.value)
                polynomial_values = a_slider.value * n_values**2 + b_slider.value * n_values + c_slider.value
                
                # Check which values are prime
                is_prime_array = np.array([is_prime(abs(val)) if val > 1 else False for val in polynomial_values])
                
                # Count primes
                prime_count = np.sum(is_prime_array)
                
                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(n_values, polynomial_values, c=is_prime_array, cmap='coolwarm', 
                          alpha=0.7, s=30)
                
                # Add labels and title
                formula = f"{a_slider.value}n² + {b_slider.value}n + {c_slider.value}"
                ax.set_xlabel('n')
                ax.set_ylabel(f'P(n) = {formula}')
                ax.set_title(f'Prime Values of {formula}\n'
                            f'Found {prime_count} primes out of {limit_slider.value} values ({prime_count/limit_slider.value:.1%})')
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7)
                
                plt.show()
                
                # Display the first few prime values
                prime_indices = np.where(is_prime_array)[0][:10]
                if len(prime_indices) > 0:
                    print("First few prime values:")
                    for idx in prime_indices:
                        print(f"n = {idx}, P({idx}) = {polynomial_values[idx]}")
        
        # Connect widgets to update function
        a_slider.observe(update, names='value')
        b_slider.observe(update, names='value')
        c_slider.observe(update, names='value')
        limit_slider.observe(update, names='value')
        
        # Display widgets
        display(widgets.VBox([
            widgets.HTML(value="<h3>Quadratic Polynomial Prime Generator</h3>"
                        "<p>Explore the pattern: an² + bn + c</p>"),
            widgets.HBox([a_slider, b_slider, c_slider]),
            limit_slider,
            output
        ]))
        
        # Initial update
        update(None)
        
    elif pattern_type == 'modular':
        modulus = parameters.get('modulus', 4)
        
        # Create widgets
        modulus_slider = widgets.IntSlider(value=modulus, min=2, max=20, description='Modulus:')
        limit_slider = widgets.IntSlider(value=1000, min=100, max=10000, description='Limit:')
        
        # Output widget
        output = widgets.Output()
        
        # Update function
        def update(_):
            with output:
                clear_output(wait=True)
                
                # Generate primes
                primes = sieve_of_eratosthenes(limit_slider.value)
                
                # Skip 2 as it's the only even prime
                primes = [p for p in primes if p > 2]
                
                # Group primes by remainder
                remainders = {}
                for i in range(modulus_slider.value):
                    remainders[i] = [p for p in primes if p % modulus_slider.value == i]
                
                # Create the plot
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Plot histogram of remainders
                counts = [len(remainders[i]) for i in range(modulus_slider.value)]
                ax.bar(range(modulus_slider.value), counts, alpha=0.7, 
                      color='skyblue', edgecolor='black')
                
                # Add labels and title
                ax.set_xlabel(f'Remainder when divided by {modulus_slider.value}')
                ax.set_ylabel('Count of Primes')
                ax.set_title(f'Distribution of Primes by Remainder mod {modulus_slider.value}\n'
                            f'(Primes between 3 and {limit_slider.value})')
                
                # Add the exact count on top of each bar
                for i, count in enumerate(counts):
                    ax.text(i, count + 0.1, str(count), ha='center')
                
                # Set x-ticks to show all remainders
                ax.set_xticks(range(modulus_slider.value))
                
                # Add grid
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                
                plt.show()
                
                # Display some statistics
                print(f"Total primes between 3 and {limit_slider.value}: {len(primes)}")
                for i in range(modulus_slider.value):
                    if math.gcd(i, modulus_slider.value) == 1:
                        print(f"Remainder {i}: {len(remainders[i])} primes ({len(remainders[i])/len(primes):.1%})")
        
        # Connect widgets to update function
        modulus_slider.observe(update, names='value')
        limit_slider.observe(update, names='value')
        
        # Display widgets
        display(widgets.VBox([
            widgets.HTML(value="<h3>Modular Distribution of Primes</h3>"
                        "<p>Explore how primes are distributed among different remainders</p>"),
            widgets.HBox([modulus_slider, limit_slider]),
            output
        ]))
        
        # Initial update
        update(None)
    
    else:
        print(f"Pattern type '{pattern_type}' not implemented")