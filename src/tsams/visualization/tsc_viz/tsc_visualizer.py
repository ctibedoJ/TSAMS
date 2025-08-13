"""
Throw-Shot-Catch Algorithm Visualization

This module provides visualization tools for the Throw-Shot-Catch Algorithm,
helping to illustrate the three-phase computational process for solving the ECDLP.
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

from tibedo.core.tsc import TSCSolver, ThrowPhase, ShotPhase, CatchPhase


class TSCVisualizer:
    """
    Visualization tools for the Throw-Shot-Catch Algorithm.
    """
    
    def __init__(self):
        """
        Initialize the TSCVisualizer.
        """
        self.solver = TSCSolver()
        self.throw_phase = ThrowPhase()
        self.shot_phase = ShotPhase()
        self.catch_phase = CatchPhase()
        self.result = None
    
    def create_demo_ecdlp_instance(self, bit_length=32):
        """
        Create a simple ECDLP instance for demonstration purposes.
        
        Args:
            bit_length (int): The bit length of the ECDLP instance.
            
        Returns:
            tuple: (P, Q, curve_params, k) where:
                P is the base point
                Q is the point to find the discrete logarithm for
                curve_params are the parameters of the elliptic curve
                k is the actual discrete logarithm (for verification)
        """
        # For simplicity, we'll create a very basic instance
        # In practice, you would use a proper elliptic curve library
        
        # Create a small prime field
        p = 2**bit_length - 5  # A prime close to 2^bit_length
        
        # Create simple curve parameters (y^2 = x^3 + ax + b)
        a = 2
        b = 3
        
        # Create a base point P (in practice, this would be a point on the curve)
        x1, y1 = 5, 10
        
        # Set the order of the base point (in practice, this would be computed)
        n = 2**(bit_length - 1) - 3  # A value less than p
        
        # Choose a random discrete logarithm k
        k = np.random.randint(1, n)
        
        # Compute Q = k*P (in practice, this would use elliptic curve point multiplication)
        # For this example, we'll just simulate it
        x2 = (x1 * k) % p
        y2 = (y1 * k) % p
        
        # Create the curve parameters dictionary
        curve_params = {
            'a': a,
            'b': b,
            'p': p,
            'n': n
        }
        
        # Return the ECDLP instance
        return (x1, y1), (x2, y2), curve_params, k
    
    def solve_and_visualize(self, bit_length=32):
        """
        Solve an ECDLP instance and collect data for visualization.
        
        Args:
            bit_length (int): The bit length of the ECDLP instance.
            
        Returns:
            dict: The visualization data.
        """
        # Create an ECDLP instance
        P, Q, curve_params, actual_k = self.create_demo_ecdlp_instance(bit_length)
        
        # Execute the Throw Phase
        throw_state = self.throw_phase.initialize(P, Q, curve_params)
        
        # Execute the Shot Phase
        shot_state = self.shot_phase.execute(throw_state)
        
        # Execute the Catch Phase
        self.result = self.catch_phase.execute(shot_state, curve_params)
        
        # Collect visualization data
        viz_data = {
            'P': P,
            'Q': Q,
            'curve_params': curve_params,
            'actual_k': actual_k,
            'computed_k': self.result['discrete_logarithm'],
            'throw_state': throw_state,
            'shot_state': shot_state,
            'catch_state': self.result,
            'is_verified': self.result['is_verified']
        }
        
        return viz_data
    
    def visualize_throw_phase(self, viz_data, save_path=None):
        """
        Visualize the Throw Phase of the TSC Algorithm.
        
        Args:
            viz_data (dict): The visualization data from solve_and_visualize.
            save_path (str): Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Extract data
        throw_state = viz_data['throw_state']
        P = viz_data['P']
        Q = viz_data['Q']
        curve_params = viz_data['curve_params']
        
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Visualize the initialization vector
        vector = throw_state['vector']
        labels = ['x₁', 'y₁', 'x₂', 'y₂', 'a', 'b', 'p', 'n']
        ax1.bar(labels, vector, color='skyblue')
        ax1.set_xlabel('Component', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title('Initialization Vector', fontsize=14)
        for i, v in enumerate(vector):
            if v > 1000:
                label = f"{v:.1e}"
            else:
                label = f"{v}"
            ax1.text(i, v, label, ha='center', va='bottom', fontsize=10)
        
        # Visualize the elliptic curve and points
        # This is a simplified visualization - in practice, you would plot the actual curve
        x_range = np.linspace(0, 20, 100)
        y_values = np.sqrt(x_range**3 + curve_params['a']*x_range + curve_params['b'])
        ax2.plot(x_range, y_values, 'b-', label='y = √(x³ + ax + b)')
        ax2.plot(x_range, -y_values, 'b-')
        
        # Plot the points P and Q
        ax2.plot(P[0], P[1], 'ro', markersize=10, label=f'P = ({P[0]}, {P[1]})')
        ax2.plot(Q[0], Q[1], 'go', markersize=10, label=f'Q = ({Q[0]}, {Q[1]})')
        
        # Add labels and title
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('y', fontsize=12)
        ax2.set_title('Elliptic Curve and Points', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Add a suptitle
        fig.suptitle('Throw Phase Visualization', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_shot_phase(self, viz_data, save_path=None):
        """
        Visualize the Shot Phase of the TSC Algorithm.
        
        Args:
            viz_data (dict): The visualization data from solve_and_visualize.
            save_path (str): Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Extract data
        shot_state = viz_data['shot_state']
        
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Visualize the transformation sequence
        if 'transformation_sequence' in shot_state:
            transformations = shot_state['transformation_sequence']
            dimensions = [t['dimension'] for t in transformations]
            
            # Plot the dimensions
            ax1.plot(range(len(dimensions)), dimensions, 'o-', linewidth=2, markersize=10)
            ax1.set_xlabel('Transformation Step', fontsize=12)
            ax1.set_ylabel('Dimension', fontsize=12)
            ax1.set_title('Dimensional Reduction Sequence', fontsize=14)
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Add dimension labels
            for i, dim in enumerate(dimensions):
                ax1.annotate(f"{dim}", (i, dim), textcoords="offset points", 
                           xytext=(0, 10), ha='center', fontsize=10)
        else:
            ax1.text(0.5, 0.5, "No transformation sequence data available", 
                    ha='center', va='center', fontsize=12)
        
        # Visualize the final state vector
        vector = shot_state['vector']
        ax2.bar(range(len(vector)), vector, color='lightgreen')
        ax2.set_xlabel('Component Index', fontsize=12)
        ax2.set_ylabel('Value', fontsize=12)
        ax2.set_title('Final State Vector', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add a suptitle
        fig.suptitle('Shot Phase Visualization', fontsize=16)
        
        # Add shot phase parameters as text
        shot_depth = shot_state.get('shot_depth', 'N/A')
        shot_value = shot_state.get('shot_value', 'N/A')
        transformation_count = shot_state.get('transformation_count', 'N/A')
        
        param_text = f"Shot Depth: {shot_depth}\n" \
                    f"Shot Value: {shot_value}\n" \
                    f"Transformation Count: {transformation_count}"
        
        fig.text(0.02, 0.02, param_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_catch_phase(self, viz_data, save_path=None):
        """
        Visualize the Catch Phase of the TSC Algorithm.
        
        Args:
            viz_data (dict): The visualization data from solve_and_visualize.
            save_path (str): Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Extract data
        catch_state = viz_data['catch_state']
        actual_k = viz_data['actual_k']
        computed_k = viz_data['computed_k']
        is_verified = viz_data['is_verified']
        
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Visualize the catch window
        catch_window = catch_state.get('catch_window', 0)
        shot_value = viz_data['shot_state'].get('shot_value', 0)
        modulus = viz_data['curve_params']['p']
        
        # Create a circular visualization of the modular space
        theta = np.linspace(0, 2*np.pi, 100)
        r = 1
        ax1.plot(r*np.cos(theta), r*np.sin(theta), 'k-')
        
        # Mark the catch window position
        angle = 2*np.pi * catch_window / modulus
        ax1.plot([0, r*np.cos(angle)], [0, r*np.sin(angle)], 'r-', linewidth=2)
        ax1.plot(r*np.cos(angle), r*np.sin(angle), 'ro', markersize=10)
        
        # Add labels
        ax1.text(1.1*r*np.cos(angle), 1.1*r*np.sin(angle), f"Catch Window: {catch_window}", 
                fontsize=10, ha='center', va='center')
        
        # Remove axis
        ax1.axis('equal')
        ax1.axis('off')
        ax1.set_title('Catch Window in Modular Space', fontsize=14)
        
        # Visualize the discrete logarithm
        values = [actual_k, computed_k]
        labels = ['Actual k', 'Computed k']
        colors = ['blue', 'green' if is_verified else 'red']
        
        ax2.bar(labels, values, color=colors)
        ax2.set_xlabel('Value Type', fontsize=12)
        ax2.set_ylabel('Discrete Logarithm Value', fontsize=12)
        ax2.set_title('Discrete Logarithm Comparison', fontsize=14)
        
        # Add value labels
        for i, v in enumerate(values):
            ax2.text(i, v, str(v), ha='center', va='bottom', fontsize=10)
        
        # Add verification status
        status_text = "Verification: " + ("Successful" if is_verified else "Failed")
        ax2.text(0.5, 0.9, status_text, transform=ax2.transAxes, 
                ha='center', va='center', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8, 
                         edgecolor='green' if is_verified else 'red'))
        
        # Add a suptitle
        fig.suptitle('Catch Phase Visualization', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_complete_algorithm(self, viz_data, save_path=None):
        """
        Visualize the complete TSC Algorithm.
        
        Args:
            viz_data (dict): The visualization data from solve_and_visualize.
            save_path (str): Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Create the figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Visualize the Throw Phase
        self._visualize_throw_phase_subplot(axes[0], viz_data)
        
        # Visualize the Shot Phase
        self._visualize_shot_phase_subplot(axes[1], viz_data)
        
        # Visualize the Catch Phase
        self._visualize_catch_phase_subplot(axes[2], viz_data)
        
        # Add a suptitle
        fig.suptitle('Throw-Shot-Catch Algorithm Visualization', fontsize=18)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _visualize_throw_phase_subplot(self, ax, viz_data):
        """
        Visualize the Throw Phase on a subplot.
        
        Args:
            ax (matplotlib.axes.Axes): The axes to plot on.
            viz_data (dict): The visualization data.
        """
        # Extract data
        throw_state = viz_data['throw_state']
        vector = throw_state['vector']
        labels = ['x₁', 'y₁', 'x₂', 'y₂', 'a', 'b', 'p', 'n']
        
        # Plot the initialization vector
        ax.bar(labels, vector, color='skyblue')
        ax.set_xlabel('Component', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Throw Phase: Initialization Vector', fontsize=14)
        
        # Add parameter values as text
        dimension = throw_state.get('dimension', 'N/A')
        prime_parameter = throw_state.get('prime_parameter', 'N/A')
        throw_depth = throw_state.get('throw_depth', 'N/A')
        
        param_text = f"Dimension: {dimension}\n" \
                    f"Prime Parameter: {prime_parameter}\n" \
                    f"Throw Depth: {throw_depth}"
        
        ax.text(0.02, 0.95, param_text, transform=ax.transAxes, fontsize=10, 
               va='top', bbox=dict(facecolor='white', alpha=0.8))
    
    def _visualize_shot_phase_subplot(self, ax, viz_data):
        """
        Visualize the Shot Phase on a subplot.
        
        Args:
            ax (matplotlib.axes.Axes): The axes to plot on.
            viz_data (dict): The visualization data.
        """
        # Extract data
        shot_state = viz_data['shot_state']
        
        # Visualize the transformation sequence
        if 'transformation_sequence' in shot_state:
            transformations = shot_state['transformation_sequence']
            dimensions = [t['dimension'] for t in transformations]
            
            # Plot the dimensions
            ax.plot(range(len(dimensions)), dimensions, 'o-', linewidth=2, markersize=10)
            ax.set_xlabel('Transformation Step', fontsize=12)
            ax.set_ylabel('Dimension', fontsize=12)
            ax.set_title('Shot Phase: Dimensional Reduction Sequence', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add dimension labels
            for i, dim in enumerate(dimensions):
                ax.annotate(f"{dim}", (i, dim), textcoords="offset points", 
                           xytext=(0, 10), ha='center', fontsize=10)
        else:
            ax.text(0.5, 0.5, "No transformation sequence data available", 
                   ha='center', va='center', fontsize=12)
        
        # Add shot phase parameters as text
        shot_depth = shot_state.get('shot_depth', 'N/A')
        shot_value = shot_state.get('shot_value', 'N/A')
        transformation_count = shot_state.get('transformation_count', 'N/A')
        
        param_text = f"Shot Depth: {shot_depth}\n" \
                    f"Shot Value: {shot_value}\n" \
                    f"Transformation Count: {transformation_count}"
        
        ax.text(0.02, 0.95, param_text, transform=ax.transAxes, fontsize=10, 
               va='top', bbox=dict(facecolor='white', alpha=0.8))
    
    def _visualize_catch_phase_subplot(self, ax, viz_data):
        """
        Visualize the Catch Phase on a subplot.
        
        Args:
            ax (matplotlib.axes.Axes): The axes to plot on.
            viz_data (dict): The visualization data.
        """
        # Extract data
        catch_state = viz_data['catch_state']
        actual_k = viz_data['actual_k']
        computed_k = viz_data['computed_k']
        is_verified = viz_data['is_verified']
        
        # Visualize the discrete logarithm
        values = [actual_k, computed_k]
        labels = ['Actual k', 'Computed k']
        colors = ['blue', 'green' if is_verified else 'red']
        
        ax.bar(labels, values, color=colors)
        ax.set_xlabel('Value Type', fontsize=12)
        ax.set_ylabel('Discrete Logarithm Value', fontsize=12)
        ax.set_title('Catch Phase: Discrete Logarithm Extraction', fontsize=14)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v, str(v), ha='center', va='bottom', fontsize=10)
        
        # Add verification status
        status_text = "Verification: " + ("Successful" if is_verified else "Failed")
        ax.text(0.5, 0.9, status_text, transform=ax.transAxes, 
               ha='center', va='center', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8, 
                        edgecolor='green' if is_verified else 'red'))
        
        # Add catch phase parameters as text
        catch_window = catch_state.get('catch_window', 'N/A')
        
        param_text = f"Catch Window: {catch_window}"
        
        ax.text(0.02, 0.95, param_text, transform=ax.transAxes, fontsize=10, 
               va='top', bbox=dict(facecolor='white', alpha=0.8))
    
    def animate_tsc_algorithm(self, viz_data, save_path=None, fps=1):
        """
        Create an animation of the TSC Algorithm.
        
        Args:
            viz_data (dict): The visualization data from solve_and_visualize.
            save_path (str): Path to save the animation. If None, the animation is displayed.
            fps (int): Frames per second for the animation.
            
        Returns:
            matplotlib.animation.FuncAnimation: The animation object.
        """
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define the phases
        phases = ['Throw', 'Shot', 'Catch']
        
        # Function to update the plot for each frame
        def update(frame):
            ax.clear()
            
            if frame == 0:
                # Throw Phase
                self._visualize_throw_phase_subplot(ax, viz_data)
            elif frame == 1:
                # Shot Phase
                self._visualize_shot_phase_subplot(ax, viz_data)
            else:
                # Catch Phase
                self._visualize_catch_phase_subplot(ax, viz_data)
            
            # Add a title
            ax.set_title(f'{phases[frame]} Phase', fontsize=16)
            
            return ax,
        
        # Create the animation
        ani = FuncAnimation(fig, update, frames=3, blit=True)
        
        # Save or display the animation
        if save_path:
            ani.save(save_path, fps=fps, extra_args=['-vcodec', 'libx264'])
        
        return ani
    
    def visualize_performance_metrics(self, bit_lengths=None, save_path=None):
        """
        Visualize the performance metrics of the TSC Algorithm for different bit lengths.
        
        Args:
            bit_lengths (list): List of bit lengths to test. If None, defaults to [8, 16, 24, 32].
            save_path (str): Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        if bit_lengths is None:
            bit_lengths = [8, 16, 24, 32]
        
        # Collect performance metrics for each bit length
        metrics = []
        for bit_length in bit_lengths:
            viz_data = self.solve_and_visualize(bit_length)
            performance = self.solver.get_performance_metrics()
            metrics.append(performance)
        
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot the theoretical complexity
        theoretical = [m['theoretical_complexity'] for m in metrics]
        actual = [m['actual_operations'] for m in metrics]
        
        ax1.plot(bit_lengths, theoretical, 'bo-', linewidth=2, label='Theoretical Complexity')
        ax1.plot(bit_lengths, actual, 'ro-', linewidth=2, label='Actual Operations')
        ax1.set_xlabel('Bit Length', fontsize=12)
        ax1.set_ylabel('Complexity / Operations', fontsize=12)
        ax1.set_title('TSC Algorithm Complexity', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Plot the complexity ratio
        ratios = [m['complexity_ratio'] for m in metrics]
        ax2.plot(bit_lengths, ratios, 'go-', linewidth=2)
        ax2.set_xlabel('Bit Length', fontsize=12)
        ax2.set_ylabel('Complexity Ratio (Actual / Theoretical)', fontsize=12)
        ax2.set_title('TSC Algorithm Efficiency', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add a horizontal line at y=1
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.7)
        
        # Add a suptitle
        fig.suptitle('TSC Algorithm Performance Metrics', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def main():
    """
    Main function to demonstrate the TSC algorithm visualization.
    """
    print("Throw-Shot-Catch Algorithm Visualization Demo")
    print("============================================")
    
    # Create the visualizer
    visualizer = TSCVisualizer()
    
    # Solve an ECDLP instance and collect visualization data
    print("\nSolving ECDLP instance...")
    viz_data = visualizer.solve_and_visualize(bit_length=32)
    
    # Visualize the Throw Phase
    print("Visualizing Throw Phase...")
    fig1 = visualizer.visualize_throw_phase(viz_data, "tsc_throw_phase.png")
    
    # Visualize the Shot Phase
    print("Visualizing Shot Phase...")
    fig2 = visualizer.visualize_shot_phase(viz_data, "tsc_shot_phase.png")
    
    # Visualize the Catch Phase
    print("Visualizing Catch Phase...")
    fig3 = visualizer.visualize_catch_phase(viz_data, "tsc_catch_phase.png")
    
    # Visualize the complete algorithm
    print("Visualizing complete algorithm...")
    fig4 = visualizer.visualize_complete_algorithm(viz_data, "tsc_complete_algorithm.png")
    
    # Visualize performance metrics
    print("Visualizing performance metrics...")
    fig5 = visualizer.visualize_performance_metrics([8, 16, 24, 32], "tsc_performance_metrics.png")
    
    print("\nVisualization complete. Images saved to current directory.")
    
    # Show the plots
    plt.show()


if __name__ == "__main__":
    main()