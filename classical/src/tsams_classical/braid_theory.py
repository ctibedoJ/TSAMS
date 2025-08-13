&quot;&quot;&quot;
Braid Theory module for Tsams Classical.

This module is part of the Tibedo Structural Algebraic Modeling System (TSAMS) ecosystem.
&quot;&quot;&quot;

# Code adapted from tibedo_enhanced_visualization.py

"""
Enhanced Visualization for the TIBEDO Framework Protein Folding Simulation

This script creates improved visualizations of protein folding paths on a Möbius strip,
highlighting the quaternion-based Möbius strip dual pairing approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

def create_mobius_strip(radius=2.0, width=1.0, n_points=100):
    """
    Create a Möbius strip for visualization.
    
    Args:
        radius (float): The radius of the Möbius strip
        width (float): The width of the Möbius strip
        n_points (int): Number of points for the mesh
        
    Returns:
        tuple: (u_grid, v_grid, x, y, z) for plotting
    """
    # Parameter ranges
    u = np.linspace(0, 2*np.pi, n_points)
    v = np.linspace(-width/2, width/2, n_points)
    
    # Create meshgrid
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Calculate the points on the Möbius strip
    x = (radius + v_grid * np.cos(u_grid/2)) * np.cos(u_grid)
    y = (radius + v_grid * np.cos(u_grid/2)) * np.sin(u_grid)
    z = v_grid * np.sin(u_grid/2)
    
    return u_grid, v_grid, x, y, z

def visualize_mobius_strip_enhanced(sequence_name, sequence_length, path_integral, mobius_points):
    """
    Create an enhanced visualization of the protein folding path on a Möbius strip.
    
    Args:
        sequence_name (str): Name of the protein sequence
        sequence_length (int): Length of the protein sequence
        path_integral (float): The computed path integral value
        mobius_points (list): List of 3D points representing the protein folding path
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create the figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the Möbius strip surface
    u_grid, v_grid, x, y, z = create_mobius_strip(radius=2.0, width=1.0, n_points=100)
    
    # Create a custom colormap for the Möbius strip
    colors = [(0.8, 0.8, 1.0, 0.3), (0.6, 0.6, 0.9, 0.3)]  # Light blue to slightly darker blue, transparent
    cmap = LinearSegmentedColormap.from_list('mobius_cmap', colors, N=100)
    
    # Plot the Möbius strip surface
    surf = ax.plot_surface(x, y, z, cmap=cmap, alpha=0.6, linewidth=0, antialiased=True)
    
    # Convert mobius_points to a numpy array
    points = np.array(mobius_points)
    
    # Create a colormap for the protein folding path
    path_colors = cm.rainbow(np.linspace(0, 1, len(points)))
    
    # Plot the protein folding path with gradient colors
    for i in range(len(points)-1):
        ax.plot(points[i:i+2, 0], points[i:i+2, 1], points[i:i+2, 2], 
                color=path_colors[i], linewidth=2.5)
    
    # Plot the amino acid positions with gradient colors
    for i, point in enumerate(points):
        ax.scatter(point[0], point[1], point[2], 
                  color=path_colors[i], s=50, edgecolor='black', alpha=0.8)
    
    # Add a colorbar to show the sequence progression
    sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=plt.Normalize(0, len(points)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.5)
    cbar.set_label('Sequence Position')
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'Protein Folding Path on Möbius Strip\n{sequence_name} (Length: {sequence_length})\nPath Integral: {path_integral:.4f}', 
                fontsize=14)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set the viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Add grid lines
    ax.grid(True, alpha=0.3)
    
    # Add a text annotation explaining the visualization
    fig.text(0.02, 0.02, 
             "Visualization of protein folding using quaternion-based Möbius strip dual pairing.\n"
             "The colored path represents the protein's amino acid sequence mapped to the Möbius strip.\n"
             "Colors indicate the position in the sequence from start (blue) to end (red).",
             fontsize=10, wrap=True)
    
    return fig

def visualize_triad_pairs(sequence_name, triad_pairs):
    """
    Visualize the triad pairs in quaternion space.
    
    Args:
        sequence_name (str): Name of the protein sequence
        triad_pairs (list): List of triad pairs (triad_center, sq_root_conj)
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create the figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract the triad centers and square root conjugates
    centers = np.array([pair[0] for pair in triad_pairs])
    conjugates = np.array([pair[1] for pair in triad_pairs])
    
    # Create a colormap for the triad pairs
    colors = cm.rainbow(np.linspace(0, 1, len(triad_pairs)))
    
    # Plot the triad centers
    ax.scatter(centers[:, 1], centers[:, 2], centers[:, 3], 
              color=colors, s=80, label='Triad Centers', alpha=0.8, edgecolor='black')
    
    # Plot the square root conjugates
    ax.scatter(conjugates[:, 1], conjugates[:, 2], conjugates[:, 3], 
              color=colors, s=50, marker='x', label='Square Root Conjugates', alpha=0.8)
    
    # Connect each triad center to its square root conjugate
    for i in range(len(triad_pairs)):
        center = centers[i, 1:4]  # Extract i, j, k components
        conj = conjugates[i, 1:4]  # Extract i, j, k components
        ax.plot([center[0], conj[0]], [center[1], conj[1]], [center[2], conj[2]], 
                color=colors[i], linestyle='--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('i component', fontsize=12)
    ax.set_ylabel('j component', fontsize=12)
    ax.set_zlabel('k component', fontsize=12)
    ax.set_title(f'Triad Pairs in Quaternion Space\n{sequence_name}', fontsize=14)
    
    # Add a legend
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set the viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Add grid lines
    ax.grid(True, alpha=0.3)
    
    # Add a text annotation explaining the visualization
    fig.text(0.02, 0.02, 
             "Visualization of triad pairs in quaternion space.\n"
             "Each triad center (dot) is connected to its square root conjugate (x).\n"
             "Colors indicate the position in the sequence from start (blue) to end (red).",
             fontsize=10, wrap=True)
    
    return fig

def visualize_dedekind_cuts(sequence_name, dedekind_cuts):
    """
    Visualize the Dedekind cut ratios.
    
    Args:
        sequence_name (str): Name of the protein sequence
        dedekind_cuts (dict): Dictionary mapping primes to their Dedekind cut ratios
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract the primes and ratios
    primes = list(dedekind_cuts.keys())
    ratios = list(dedekind_cuts.values())
    
    # Create a bar chart
    bars = ax.bar(primes, ratios, color='skyblue', edgecolor='navy')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{height:.6f}',
                ha='center', va='bottom', rotation=90, fontsize=8)
    
    # Set labels and title
    ax.set_xlabel('Prime Number', fontsize=12)
    ax.set_ylabel('Dedekind Cut Ratio', fontsize=12)
    ax.set_title(f'Dedekind Cut Ratios for {sequence_name}', fontsize=14)
    
    # Set y-axis to logarithmic scale
    ax.set_yscale('log')
    
    # Add grid lines
    ax.grid(True, alpha=0.3, which='both')
    
    # Add a text annotation explaining the visualization
    fig.text(0.02, 0.02, 
             "Visualization of Dedekind cut ratios for different primes.\n"
             "The Dedekind cut ratio is related to the inertia degree of the prime in the cyclotomic field.\n"
             "Lower values indicate stronger structural significance in the protein folding process.",
             fontsize=10, wrap=True)
    
    return fig

def create_enhanced_visualizations(sequence_name, results):
    """
    Create enhanced visualizations for the protein folding simulation results.
    
    Args:
        sequence_name (str): Name of the protein sequence
        results (dict): The simulation results
        
    Returns:
        list: List of figure objects
    """
    figures = []
    
    # Extract the relevant data
    sequence = results['sequence']
    quaternions = results['quaternions']
    triad_pairs = results['triad_pairs']
    path_integral = results['path_integral']
    mobius_points = results['mobius_points']
    dedekind_cuts = results['dedekind_cuts']
    
    # Create the enhanced Möbius strip visualization
    fig1 = visualize_mobius_strip_enhanced(sequence_name, len(sequence), path_integral, mobius_points)
    figures.append(fig1)
    
    # Create the triad pairs visualization
    fig2 = visualize_triad_pairs(sequence_name, triad_pairs)
    figures.append(fig2)
    
    # Create the Dedekind cuts visualization
    fig3 = visualize_dedekind_cuts(sequence_name, dedekind_cuts)
    figures.append(fig3)
    
    return figures

# Import the necessary functions from the biological implementation
from tibedo_biological_implementation import ProteinFoldingSimulator

def main():
    """
    Main function to create enhanced visualizations for the protein folding simulations.
    """
    print("Creating Enhanced Visualizations for TIBEDO Framework Protein Folding Simulation")
    print("===========================================================================")
    
    # Create the simulator
    simulator = ProteinFoldingSimulator()
    
    # Define the test protein sequences
    test_sequences = [
        {
            "name": "SARS-CoV-2 Spike Protein (Full)",
            "sequence": "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
        },
        {
            "name": "SARS-CoV-2 Spike Protein (RBD)",
            "sequence": "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
        }
    ]
    
    for seq_data in test_sequences:
        print(f"\nProcessing {seq_data['name']} (length: {len(seq_data['sequence'])})")
        
        # Simulate protein folding
        results = simulator.simulate_protein_folding(seq_data['sequence'])
        
        # Create enhanced visualizations
        figures = create_enhanced_visualizations(seq_data['name'], results)
        
        # Save the figures
        for i, fig in enumerate(figures):
            filename = f"{seq_data['name'].replace(' ', '_')}_{i+1}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved visualization as {filename}")

if __name__ == "__main__":
    main()
