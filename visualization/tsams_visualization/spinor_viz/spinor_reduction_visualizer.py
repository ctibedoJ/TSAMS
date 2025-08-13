"""
Spinor Reduction Visualization

This module provides visualization tools for the Spinor Reduction Chain,
helping to illustrate the dimensional reduction process that is central
to the TIBEDO Framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add the parent directory to the path so we can import the tibedo package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from tibedo.core.spinor import SpinorSpace, ReductionMap, ReductionChain


class SpinorReductionVisualizer:
    """
    Visualization tools for the Spinor Reduction Chain.
    """
    
    def __init__(self, initial_dimension=16, chain_length=5):
        """
        Initialize the SpinorReductionVisualizer.
        
        Args:
            initial_dimension (int): The initial dimension of the chain.
            chain_length (int): The number of spaces in the chain.
        """
        self.chain = ReductionChain(initial_dimension, chain_length)
    
    def visualize_dimension_sequence(self, save_path=None):
        """
        Visualize the dimension sequence of the reduction chain.
        
        Args:
            save_path (str): Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the dimension sequence
        dimensions = self.chain.dimensions
        x = np.arange(len(dimensions))
        ax.plot(x, dimensions, 'o-', linewidth=2, markersize=10)
        
        # Add labels and title
        ax.set_xlabel('Reduction Step', fontsize=12)
        ax.set_ylabel('Dimension', fontsize=12)
        ax.set_title('Spinor Reduction Sequence', fontsize=14)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add dimension labels
        for i, dim in enumerate(dimensions):
            ax.annotate(f"{dim}", (i, dim), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=10)
        
        # Set y-axis to log scale if the range is large
        if max(dimensions) / min(dimensions) > 10:
            ax.set_yscale('log')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_reduction_step(self, step_index, save_path=None):
        """
        Visualize a specific reduction step.
        
        Args:
            step_index (int): The index of the reduction step to visualize.
            save_path (str): Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        if step_index < 0 or step_index >= len(self.chain.maps):
            raise ValueError(f"Step index {step_index} out of range [0, {len(self.chain.maps) - 1}]")
        
        # Get the reduction map for this step
        reduction_map = self.chain.maps[step_index]
        
        # Create a random spinor in the source space
        source_spinor = np.random.randn(reduction_map.source_space.vector_space_dimension) + \
                       1j * np.random.randn(reduction_map.source_space.vector_space_dimension)
        source_spinor = source_spinor / np.linalg.norm(source_spinor)
        
        # Apply the reduction map
        target_spinor = reduction_map.apply(source_spinor)
        
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot the source spinor
        self._plot_spinor(ax1, source_spinor, 
                         f"Source Spinor (Dimension {reduction_map.source_dimension})")
        
        # Plot the target spinor
        self._plot_spinor(ax2, target_spinor, 
                         f"Target Spinor (Dimension {reduction_map.target_dimension})")
        
        # Add a title
        fig.suptitle(f"Spinor Reduction Step {step_index}: "
                    f"{reduction_map.source_dimension} → {reduction_map.target_dimension}", 
                    fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_spinor(self, ax, spinor, title):
        """
        Plot a spinor as a bar chart of its components.
        
        Args:
            ax (matplotlib.axes.Axes): The axes to plot on.
            spinor (numpy.ndarray): The spinor to plot.
            title (str): The title for the plot.
        """
        # Get the real and imaginary parts
        real_parts = np.real(spinor)
        imag_parts = np.imag(spinor)
        
        # Create x positions
        x = np.arange(len(spinor))
        width = 0.35
        
        # Plot the real and imaginary parts
        ax.bar(x - width/2, real_parts, width, label='Real Part')
        ax.bar(x + width/2, imag_parts, width, label='Imaginary Part')
        
        # Add labels and title
        ax.set_xlabel('Component Index', fontsize=10)
        ax.set_ylabel('Component Value', fontsize=10)
        ax.set_title(title, fontsize=12)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend()
    
    def visualize_reduction_sequence(self, save_path=None):
        """
        Visualize the entire reduction sequence.
        
        Args:
            save_path (str): Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Create a random initial spinor
        initial_spinor = np.random.randn(self.chain.spaces[0].vector_space_dimension) + \
                        1j * np.random.randn(self.chain.spaces[0].vector_space_dimension)
        initial_spinor = initial_spinor / np.linalg.norm(initial_spinor)
        
        # Apply the reduction sequence
        states = self.chain.apply_reduction_sequence(initial_spinor)
        
        # Create the figure
        fig, axes = plt.subplots(len(states), 1, figsize=(10, 3 * len(states)))
        
        # If there's only one state, wrap axes in a list
        if len(states) == 1:
            axes = [axes]
        
        # Plot each state
        for i, (state, ax) in enumerate(zip(states, axes)):
            self._plot_spinor(ax, state, 
                             f"State {i} (Dimension {self.chain.dimensions[i]})")
        
        # Add a title
        fig.suptitle("Spinor Reduction Sequence", fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def animate_reduction_sequence(self, save_path=None, fps=5):
        """
        Create an animation of the reduction sequence.
        
        Args:
            save_path (str): Path to save the animation. If None, the animation is displayed.
            fps (int): Frames per second for the animation.
            
        Returns:
            matplotlib.animation.FuncAnimation: The animation object.
        """
        # Create a random initial spinor
        initial_spinor = np.random.randn(self.chain.spaces[0].vector_space_dimension) + \
                        1j * np.random.randn(self.chain.spaces[0].vector_space_dimension)
        initial_spinor = initial_spinor / np.linalg.norm(initial_spinor)
        
        # Apply the reduction sequence
        states = self.chain.apply_reduction_sequence(initial_spinor)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Function to update the plot for each frame
        def update(frame):
            ax.clear()
            self._plot_spinor(ax, states[frame], 
                             f"State {frame} (Dimension {self.chain.dimensions[frame]})")
            ax.set_title(f"Spinor Reduction Sequence - Step {frame}", fontsize=14)
            return ax,
        
        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(states), blit=True)
        
        # Save or display the animation
        if save_path:
            ani.save(save_path, fps=fps, extra_args=['-vcodec', 'libx264'])
        
        return ani
    
    def visualize_complexity_reduction(self, problem_complexity_exponent=0.5, save_path=None):
        """
        Visualize the complexity reduction achieved by the reduction chain.
        
        Args:
            problem_complexity_exponent (float): The exponent d in the original
                                               complexity O(2^(dn)).
            save_path (str): Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate the complexity at each step
        dimensions = self.chain.dimensions
        complexities = [2**(problem_complexity_exponent * dim) for dim in dimensions]
        
        # Plot the complexities
        x = np.arange(len(dimensions))
        ax.semilogy(x, complexities, 'o-', linewidth=2, markersize=10)
        
        # Add labels and title
        ax.set_xlabel('Reduction Step', fontsize=12)
        ax.set_ylabel('Computational Complexity', fontsize=12)
        ax.set_title(f'Complexity Reduction (Original Exponent = {problem_complexity_exponent})', 
                    fontsize=14)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add complexity labels
        for i, complexity in enumerate(complexities):
            if complexity >= 1e6:
                label = f"{complexity:.1e}"
            else:
                label = f"{complexity:.1f}"
            ax.annotate(label, (i, complexity), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_3d_reduction(self, save_path=None):
        """
        Create a 3D visualization of the reduction process.
        
        Args:
            save_path (str): Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Create the figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a random initial spinor
        initial_spinor = np.random.randn(self.chain.spaces[0].vector_space_dimension) + \
                        1j * np.random.randn(self.chain.spaces[0].vector_space_dimension)
        initial_spinor = initial_spinor / np.linalg.norm(initial_spinor)
        
        # Apply the reduction sequence
        states = self.chain.apply_reduction_sequence(initial_spinor)
        
        # Create a colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(states)))
        
        # Plot each state as a 3D scatter plot
        for i, state in enumerate(states):
            # Use the first three components (or fewer if the state has fewer components)
            components = min(len(state), 3)
            if components == 1:
                x = [np.real(state[0])]
                y = [np.imag(state[0])]
                z = [0]
            elif components == 2:
                x = [np.real(state[0])]
                y = [np.imag(state[0])]
                z = [np.real(state[1])]
            else:
                x = [np.real(state[0])]
                y = [np.imag(state[0])]
                z = [np.real(state[1])]
            
            # Plot the state
            ax.scatter(x, y, z, color=colors[i], s=100, label=f"State {i} (Dim {self.chain.dimensions[i]})")
            
            # If this is not the last state, draw a line to the next state
            if i < len(states) - 1:
                next_state = states[i + 1]
                next_components = min(len(next_state), 3)
                if next_components == 1:
                    next_x = [np.real(next_state[0])]
                    next_y = [np.imag(next_state[0])]
                    next_z = [0]
                elif next_components == 2:
                    next_x = [np.real(next_state[0])]
                    next_y = [np.imag(next_state[0])]
                    next_z = [np.real(next_state[1])]
                else:
                    next_x = [np.real(next_state[0])]
                    next_y = [np.imag(next_state[0])]
                    next_z = [np.real(next_state[1])]
                
                ax.plot([x[0], next_x[0]], [y[0], next_y[0]], [z[0], next_z[0]], 
                       color='gray', linestyle='--', alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel('Re(Component 0)', fontsize=10)
        ax.set_ylabel('Im(Component 0)', fontsize=10)
        ax.set_zlabel('Re(Component 1)', fontsize=10)
        ax.set_title('3D Visualization of Spinor Reduction', fontsize=14)
        
        # Add legend
        ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def main():
    """
    Main function to demonstrate the spinor reduction visualization.
    """
    print("Spinor Reduction Visualization Demo")
    print("==================================")
    
    # Create the visualizer
    visualizer = SpinorReductionVisualizer(initial_dimension=16, chain_length=5)
    
    # Visualize the dimension sequence
    print("\nVisualizing dimension sequence...")
    fig1 = visualizer.visualize_dimension_sequence("spinor_dimension_sequence.png")
    
    # Visualize a reduction step
    print("Visualizing reduction step...")
    fig2 = visualizer.visualize_reduction_step(0, "spinor_reduction_step.png")
    
    # Visualize the reduction sequence
    print("Visualizing reduction sequence...")
    fig3 = visualizer.visualize_reduction_sequence("spinor_reduction_sequence.png")
    
    # Visualize the complexity reduction
    print("Visualizing complexity reduction...")
    fig4 = visualizer.visualize_complexity_reduction(0.5, "spinor_complexity_reduction.png")
    
    # Visualize the 3D reduction
    print("Visualizing 3D reduction...")
    fig5 = visualizer.visualize_3d_reduction("spinor_3d_reduction.png")
    
    print("\nVisualization complete. Images saved to current directory.")
    
    # Show the plots
    plt.show()


if __name__ == "__main__":
    main()