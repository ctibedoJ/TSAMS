"""
Prime-Indexed Congruential Relations Visualization

This module provides visualization tools for the Prime-Indexed Congruential Relations,
helping to illustrate the computational shortcuts that contribute to the linear time
complexity of the TIBEDO Framework.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from matplotlib.animation import FuncAnimation
import sys
import os

# Add the parent directory to the path so we can import the tibedo package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from tibedo.core.prime_indexed import PrimeIndexedStructure, ModularSystem, CongruentialAccelerator


class PrimeIndexedVisualizer:
    """
    Visualization tools for Prime-Indexed Congruential Relations.
    """
    
    def __init__(self, max_index=100, modulus=56):
        """
        Initialize the PrimeIndexedVisualizer.
        
        Args:
            max_index (int): The maximum index for prime generation.
            modulus (int): The modulus for the congruential system.
        """
        self.prime_structure = PrimeIndexedStructure(max_index)
        self.modular_system = ModularSystem(modulus)
        self.accelerator = CongruentialAccelerator(modulus)
    
    def visualize_prime_sequence(self, formula=None, save_path=None):
        """
        Visualize a prime-indexed sequence.
        
        Args:
            formula (callable): A function that takes a prime number as input
                               and returns the corresponding value in the sequence.
                               If None, the standard sequence (log p)/√p is used.
            save_path (str): Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Generate the sequence
        if formula:
            sequence = self.prime_structure.generate_sequence(formula)
        else:
            sequence = self.prime_structure.generate_standard_sequence()
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the sequence
        primes = self.prime_structure.primes
        ax.plot(primes, sequence, 'o-', linewidth=2, markersize=6)
        
        # Add labels and title
        ax.set_xlabel('Prime Number', fontsize=12)
        ax.set_ylabel('Sequence Value', fontsize=12)
        if formula:
            ax.set_title('Prime-Indexed Sequence', fontsize=14)
        else:
            ax.set_title('Standard Prime-Indexed Sequence: (log p)/√p', fontsize=14)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-axis to log scale
        ax.set_xscale('log')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_prime_matrix(self, formula=None, save_path=None):
        """
        Visualize a prime-indexed matrix.
        
        Args:
            formula (callable): A function that takes two prime numbers as input
                               and returns the corresponding matrix entry.
                               If None, a standard formula is used.
            save_path (str): Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Generate the matrix
        if formula:
            matrix = self.prime_structure.generate_matrix(formula)
        else:
            matrix = self.prime_structure.generate_standard_matrix()
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the matrix as a heatmap
        im = ax.imshow(matrix, cmap='viridis')
        
        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Matrix Value', fontsize=10)
        
        # Add labels and title
        ax.set_xlabel('Prime Index j', fontsize=12)
        ax.set_ylabel('Prime Index i', fontsize=12)
        ax.set_title('Prime-Indexed Matrix', fontsize=14)
        
        # Add prime number labels to the axes
        primes = self.prime_structure.primes
        num_primes = min(10, len(primes))  # Limit the number of labels to avoid overcrowding
        step = len(primes) // num_primes
        indices = list(range(0, len(primes), step))
        ax.set_xticks(indices)
        ax.set_yticks(indices)
        ax.set_xticklabels([str(primes[i]) for i in indices], rotation=45)
        ax.set_yticklabels([str(primes[i]) for i in indices])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_modular_system(self, save_path=None):
        """
        Visualize the modular system.
        
        Args:
            save_path (str): Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Visualize the divisors of the modulus
        divisors = self.modular_system.divisors
        ax1.bar(range(len(divisors)), divisors, color='skyblue')
        ax1.set_xlabel('Index', fontsize=12)
        ax1.set_ylabel('Divisor Value', fontsize=12)
        ax1.set_title(f'Divisors of {self.modular_system.modulus}', fontsize=14)
        for i, div in enumerate(divisors):
            ax1.text(i, div, str(div), ha='center', va='bottom', fontsize=10)
        
        # Visualize the coprime classes
        coprime_classes = self.modular_system.coprime_classes
        ax2.bar(range(len(coprime_classes)), coprime_classes, color='lightgreen')
        ax2.set_xlabel('Index', fontsize=12)
        ax2.set_ylabel('Congruence Class', fontsize=12)
        ax2.set_title(f'Congruence Classes Coprime to {self.modular_system.modulus}', fontsize=14)
        for i, cls in enumerate(coprime_classes):
            ax2.text(i, cls, str(cls), ha='center', va='bottom', fontsize=10)
        
        # Add a suptitle
        fig.suptitle(f'{self.modular_system.modulus}-Modular System', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_compatible_prime_set(self, size=7, save_path=None):
        """
        Visualize a compatible prime set for the modular system.
        
        Args:
            size (int): The size of the prime set to visualize.
            save_path (str): Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Find a compatible prime set
        prime_set = self.modular_system.find_compatible_prime_set(size)
        
        if not prime_set:
            raise ValueError(f"No compatible prime set of size {size} found.")
        
        # Calculate the sequence values for these primes
        values = [np.log(p) / np.sqrt(p) for p in prime_set]
        
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot the primes
        ax1.bar(range(len(prime_set)), prime_set, color='coral')
        ax1.set_xlabel('Index', fontsize=12)
        ax1.set_ylabel('Prime Value', fontsize=12)
        ax1.set_title(f'Compatible Prime Set of Size {size}', fontsize=14)
        for i, p in enumerate(prime_set):
            ax1.text(i, p, str(p), ha='center', va='bottom', fontsize=10)
        
        # Plot the sequence values
        ax2.bar(range(len(values)), values, color='purple')
        ax2.set_xlabel('Index', fontsize=12)
        ax2.set_ylabel('Sequence Value (log p)/√p', fontsize=12)
        ax2.set_title('Sequence Values for Compatible Prime Set', fontsize=14)
        for i, v in enumerate(values):
            ax2.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=10)
        
        # Add a suptitle
        sum_value = sum(values)
        mod_sum = sum_value % self.modular_system.modulus
        fig.suptitle(f'Compatible Prime Set for {self.modular_system.modulus}-Modular System\n'
                    f'Sum of Values: {sum_value:.4f}, Modulo {self.modular_system.modulus}: {mod_sum:.4f}',
                    fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_congruential_acceleration(self, save_path=None):
        """
        Visualize the congruential acceleration process.
        
        Args:
            save_path (str): Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Precompute prime sets
        self.accelerator.precompute_prime_sets(max_size=7)
        
        # Create a simple computation function for demonstration
        def computation_function(x):
            return np.sin(x)
        
        # Create input values
        x_values = np.linspace(0, 2*np.pi, 100)
        
        # Compute direct results
        direct_results = [computation_function(x) for x in x_values]
        
        # Compute accelerated results
        accelerated_results = [self.accelerator.accelerate_computation(computation_function, x) 
                              for x in x_values]
        
        # Create the figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot the direct computation
        ax1.plot(x_values, direct_results, 'b-', linewidth=2)
        ax1.set_xlabel('Input Value', fontsize=12)
        ax1.set_ylabel('Output Value', fontsize=12)
        ax1.set_title('Direct Computation', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot the accelerated computation
        ax2.plot(x_values, accelerated_results, 'r-', linewidth=2)
        ax2.set_xlabel('Input Value', fontsize=12)
        ax2.set_ylabel('Output Value', fontsize=12)
        ax2.set_title('Accelerated Computation', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot the difference
        differences = [direct - accel for direct, accel in zip(direct_results, accelerated_results)]
        ax3.plot(x_values, differences, 'g-', linewidth=2)
        ax3.set_xlabel('Input Value', fontsize=12)
        ax3.set_ylabel('Difference', fontsize=12)
        ax3.set_title('Difference (Direct - Accelerated)', fontsize=14)
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Add a suptitle
        fig.suptitle('Congruential Acceleration Demonstration', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_prime_network(self, size=20, save_path=None):
        """
        Visualize the network of prime relationships in the congruential system.
        
        Args:
            size (int): The number of primes to include in the network.
            save_path (str): Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Get the first 'size' primes
        primes = self.prime_structure.primes[:size]
        
        # Create a graph
        G = nx.Graph()
        
        # Add nodes for each prime
        for i, p in enumerate(primes):
            G.add_node(i, prime=p, value=np.log(p)/np.sqrt(p))
        
        # Add edges between primes whose sum of values is close to a multiple of the modulus
        modulus = self.modular_system.modulus
        for i in range(len(primes)):
            for j in range(i+1, len(primes)):
                pi = primes[i]
                pj = primes[j]
                vi = np.log(pi)/np.sqrt(pi)
                vj = np.log(pj)/np.sqrt(pj)
                sum_mod = (vi + vj) % modulus
                
                # If the sum is close to 0 or close to the modulus, add an edge
                if sum_mod < 0.1 * modulus or sum_mod > 0.9 * modulus:
                    G.add_edge(i, j, weight=1.0 - min(sum_mod, modulus - sum_mod) / modulus)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create a layout for the graph
        pos = nx.spring_layout(G, seed=42)
        
        # Draw the nodes
        node_sizes = [300 * np.log(G.nodes[i]['prime']) for i in G.nodes]
        node_colors = [G.nodes[i]['value'] for i in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color=node_colors, cmap='viridis', 
                              alpha=0.8, ax=ax)
        
        # Draw the edges
        edge_weights = [G[u][v]['weight'] for u, v in G.edges]
        nx.draw_networkx_edges(G, pos, width=edge_weights, 
                              edge_color=edge_weights, edge_cmap=plt.cm.Blues, 
                              alpha=0.6, ax=ax)
        
        # Draw the labels
        labels = {i: str(G.nodes[i]['prime']) for i in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, 
                               font_color='black', ax=ax)
        
        # Add a title
        ax.set_title(f'Prime Relationship Network in {modulus}-Modular System', fontsize=16)
        
        # Remove axis
        ax.axis('off')
        
        # Add a colorbar for the node colors
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(
            vmin=min(node_colors), vmax=max(node_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Sequence Value (log p)/√p', fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def animate_congruential_reduction(self, save_path=None, fps=5):
        """
        Create an animation of the congruential reduction process.
        
        Args:
            save_path (str): Path to save the animation. If None, the animation is displayed.
            fps (int): Frames per second for the animation.
            
        Returns:
            matplotlib.animation.FuncAnimation: The animation object.
        """
        # Create a simple computation function for demonstration
        def computation_function(x):
            return np.sin(x)
        
        # Create a reduction chain
        chain_length = 5
        chain_results = self.accelerator.create_reduction_chain(
            computation_function, np.pi/4, chain_length)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Function to update the plot for each frame
        def update(frame):
            ax.clear()
            
            # Plot the result at this step
            x_values = np.linspace(0, 2*np.pi, 100)
            y_values = [computation_function(x) for x in x_values]
            ax.plot(x_values, y_values, 'b-', linewidth=2, alpha=0.3, label='Original Function')
            
            # Highlight the result at this step
            result = chain_results[frame]
            ax.plot([np.pi/4], [result], 'ro', markersize=10, label=f'Step {frame} Result')
            
            # Add labels and title
            ax.set_xlabel('Input Value', fontsize=12)
            ax.set_ylabel('Output Value', fontsize=12)
            ax.set_title(f'Congruential Reduction Chain - Step {frame}', fontsize=14)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend
            ax.legend()
            
            return ax,
        
        # Create the animation
        ani = FuncAnimation(fig, update, frames=chain_length, blit=True)
        
        # Save or display the animation
        if save_path:
            ani.save(save_path, fps=fps, extra_args=['-vcodec', 'libx264'])
        
        return ani


def main():
    """
    Main function to demonstrate the prime-indexed visualization.
    """
    print("Prime-Indexed Congruential Relations Visualization Demo")
    print("=====================================================")
    
    # Create the visualizer
    visualizer = PrimeIndexedVisualizer(max_index=50, modulus=56)
    
    # Visualize the prime sequence
    print("\nVisualizing prime sequence...")
    fig1 = visualizer.visualize_prime_sequence(save_path="prime_sequence.png")
    
    # Visualize the prime matrix
    print("Visualizing prime matrix...")
    fig2 = visualizer.visualize_prime_matrix(save_path="prime_matrix.png")
    
    # Visualize the modular system
    print("Visualizing modular system...")
    fig3 = visualizer.visualize_modular_system(save_path="modular_system.png")
    
    # Visualize a compatible prime set
    print("Visualizing compatible prime set...")
    fig4 = visualizer.visualize_compatible_prime_set(size=7, save_path="compatible_prime_set.png")
    
    # Visualize congruential acceleration
    print("Visualizing congruential acceleration...")
    fig5 = visualizer.visualize_congruential_acceleration(save_path="congruential_acceleration.png")
    
    # Visualize the prime network
    print("Visualizing prime network...")
    fig6 = visualizer.visualize_prime_network(size=20, save_path="prime_network.png")
    
    print("\nVisualization complete. Images saved to current directory.")
    
    # Show the plots
    plt.show()


if __name__ == "__main__":
    main()