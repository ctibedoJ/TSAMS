&quot;&quot;&quot;
Biosynthesis Modeling module for Tsams Biology.

This module is part of the Tibedo Structural Algebraic Modeling System (TSAMS) ecosystem.
&quot;&quot;&quot;

# Code adapted from run_octonion_simulation.py

#!/usr/bin/env python3
"""
Run script for the Octonion Period Sequences Framework simulation.

This script executes the octonion-based period sequences simulation,
generates visualizations, and saves the results.

Author: Based on the work of Charles Tibedo
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, animation
from octonion_period_sequences import (
    OctonionAlgebra, 
    HodgeStarMoufang, 
    DicosohedralSurface, 
    RightKernel,
    PeriodSequenceGenerator, 
    DicosohedralMatrix
)

def create_output_directory():
    """Create output directory for simulation results."""
    output_dir = "outputs/octonion_simulation"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def run_simulation():
    """Run the octonion period sequences simulation."""
    print("Starting Octonion Period Sequences Simulation...")
    start_time = time.time()
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Initialize components
    print("Initializing mathematical structures...")
    octonion = OctonionAlgebra()
    hodge = HodgeStarMoufang()
    right_kernel = RightKernel(pressure_param=2.0, log_base=10)
    generator = PeriodSequenceGenerator()
    
    # Generate period sequences
    print("Generating period sequences...")
    sequence = generator.generate_period_sequence(length=28)
    outer_codes, spectral_sheafs = generator.generate_all_period_sequences()
    
    # Create dicosohedral surfaces
    print("Creating dicosohedral surfaces...")
    surfaces = [DicosohedralSurface(i) for i in range(24)]
    
    # Create dicosohedral matrix
    print("Creating dicosohedral matrix...")
    matrix = DicosohedralMatrix()
    
    # Evolve the matrix
    print("Evolving matrix over time...")
    evolution = matrix.evolve(steps=28)
    
    # Analyze the evolution
    print("Analyzing matrix evolution...")
    analysis = matrix.analyze_evolution(evolution)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Visualize period sequence
    plt.figure(figsize=(12, 6))
    plt.plot(sequence, 'o-', markersize=8, color='blue')
    plt.title('Period Sequence (28-Period Moufang Structure)', fontsize=16)
    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/period_sequence.png", dpi=300)
    plt.close()
    
    # Visualize spectral matrix
    plt.figure(figsize=(14, 6))
    
    # Plot the real part
    plt.subplot(121)
    im1 = plt.imshow(np.real(matrix.matrix), cmap='viridis')
    plt.colorbar(im1)
    plt.title('Real Part', fontsize=14)
    
    # Plot the imaginary part
    plt.subplot(122)
    im2 = plt.imshow(np.imag(matrix.matrix), cmap='plasma')
    plt.colorbar(im2)
    plt.title('Imaginary Part', fontsize=14)
    
    plt.suptitle('Spectral Matrix (8-pulsed 7-ray Dicosohedral Structure)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spectral_matrix.png", dpi=300)
    plt.close()
    
    # Visualize eigenvalue evolution
    plt.figure(figsize=(14, 10))
    
    # Plot the real parts
    plt.subplot(211)
    for i in range(len(analysis['eigenvalues'][0])):
        real_parts = [evals[i].real for evals in analysis['eigenvalues']]
        plt.plot(real_parts, label=f'λ{i+1}', linewidth=2)
    plt.title('Real Parts of Eigenvalues', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Real Part', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    # Plot the imaginary parts
    plt.subplot(212)
    for i in range(len(analysis['eigenvalues'][0])):
        imag_parts = [evals[i].imag for evals in analysis['eigenvalues']]
        plt.plot(imag_parts, label=f'λ{i+1}', linewidth=2)
    plt.title('Imaginary Parts of Eigenvalues', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Imaginary Part', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    plt.suptitle('Eigenvalue Evolution Over 28-Period', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eigenvalue_evolution.png", dpi=300)
    plt.close()
    
    # Visualize spectral radius and determinant
    plt.figure(figsize=(14, 6))
    
    # Plot spectral radius
    plt.subplot(121)
    plt.plot(analysis['spectral_radii'], 'o-', color='red', linewidth=2)
    plt.title('Spectral Radius Evolution', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Spectral Radius', fontsize=12)
    plt.grid(True)
    
    # Plot determinant
    plt.subplot(122)
    det_real = [d.real for d in analysis['determinants']]
    plt.plot(det_real, 'o-', color='green', linewidth=2)
    plt.title('Determinant Evolution (Real Part)', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Determinant', fontsize=12)
    plt.grid(True)
    
    plt.suptitle('Matrix Properties Evolution', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/matrix_properties.png", dpi=300)
    plt.close()
    
    # Visualize dicosohedral surfaces
    for i in range(min(6, len(surfaces))):  # Visualize first 6 surfaces
        surface = surfaces[i]
        points = surface.generate_surface_points(resolution=30)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        scatter = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=points[:, 2],
            cmap='viridis',
            s=30,
            alpha=0.8
        )
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Z Coordinate')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(f'Dicosohedral Surface {i+1}/24', fontsize=14)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        plt.savefig(f"{output_dir}/dicosohedral_surface_{i+1}.png", dpi=300)
        plt.close()
    
    # Create animation of matrix evolution
    print("Creating animation of matrix evolution...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Initialize plots
    im1 = ax1.imshow(np.real(evolution[0]), cmap='viridis', animated=True)
    im2 = ax2.imshow(np.imag(evolution[0]), cmap='plasma', animated=True)
    
    ax1.set_title('Real Part')
    ax2.set_title('Imaginary Part')
    
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    
    fig.suptitle('Dicosohedral Matrix Evolution', fontsize=16)
    
    # Animation function
    def update_frame(frame):
        im1.set_array(np.real(evolution[frame]))
        im2.set_array(np.imag(evolution[frame]))
        fig.suptitle(f'Dicosohedral Matrix Evolution - Step {frame+1}/28', fontsize=16)
        return [im1, im2]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update_frame, frames=len(evolution),
        interval=500, blit=True
    )
    
    # Try to find an available writer for MP4
    try:
        # Check available writers
        writers = animation.writers.list()
        print(f"Available animation writers: {writers}")
        
        if 'ffmpeg' in writers:
            writer = animation.writers['ffmpeg'](fps=4, bitrate=1800)
            anim.save(f"{output_dir}/matrix_evolution.mp4", writer=writer)
        elif 'imagemagick' in writers:
            writer = animation.writers['imagemagick'](fps=4)
            anim.save(f"{output_dir}/matrix_evolution.mp4", writer=writer)
        else:
            # Fall back to GIF if no suitable writer is found
            print("No suitable video writer found, saving as GIF instead")
            anim.save(f"{output_dir}/matrix_evolution.gif", writer='pillow', fps=4, dpi=200)
    except Exception as e:
        print(f"Error saving animation: {e}")
        # Fall back to GIF if there's an error
        try:
            anim.save(f"{output_dir}/matrix_evolution.gif", writer='pillow', fps=4, dpi=200)
        except Exception as e2:
            print(f"Could not save animation as GIF either: {e2}")
    
    plt.close()
    
    # Create 3D visualization of eigenvalues
    print("Creating 3D visualization of eigenvalues...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract eigenvalues
    for i in range(len(analysis['eigenvalues'][0])):
        x = [t for t in range(len(analysis['eigenvalues']))]
        y = [evals[i].real for evals in analysis['eigenvalues']]
        z = [evals[i].imag for evals in analysis['eigenvalues']]
        
        ax.plot(x, y, z, label=f'λ{i+1}', linewidth=2)
        
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Real Part', fontsize=12)
    ax.set_zlabel('Imaginary Part', fontsize=12)
    ax.set_title('3D Eigenvalue Trajectories', fontsize=16)
    ax.legend()
    
    plt.savefig(f"{output_dir}/eigenvalue_3d_trajectories.png", dpi=300)
    plt.close()
    
    # Create visualization of period sequences
    print("Creating visualization of period sequences...")
    plt.figure(figsize=(14, 8))
    
    # Plot a sample of outer codes
    plt.subplot(211)
    for i in range(min(5, len(outer_codes))):
        plt.plot(outer_codes[i], label=f'Code {i+1}', linewidth=2)
    plt.title('Sample of Outer Codes (56 total)', fontsize=14)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True)
    plt.legend()
    
    # Plot all spectral sheafs
    plt.subplot(212)
    for i in range(len(spectral_sheafs)):
        plt.plot(spectral_sheafs[i], label=f'Sheaf {i+1}', linewidth=2)
    plt.title('All Spectral Sheafs (8 heights)', fontsize=14)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True)
    plt.legend()
    
    plt.suptitle('Period Sequences Structure', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/period_sequences.png", dpi=300)
    plt.close()
    
    # Save numerical results
    print("Saving numerical results...")
    
    # Save period sequence
    np.savetxt(f"{output_dir}/period_sequence.csv", sequence, delimiter=',', fmt='%d')
    
    # Save spectral matrix
    np.savetxt(f"{output_dir}/spectral_matrix_real.csv", np.real(matrix.matrix), delimiter=',')
    np.savetxt(f"{output_dir}/spectral_matrix_imag.csv", np.imag(matrix.matrix), delimiter=',')
    
    # Save eigenvalues
    with open(f"{output_dir}/eigenvalues.csv", 'w') as f:
        f.write("time_step,eigenvalue_index,real_part,imag_part\n")
        for t in range(len(analysis['eigenvalues'])):
            for i, eigenvalue in enumerate(analysis['eigenvalues'][t]):
                f.write(f"{t},{i},{eigenvalue.real},{eigenvalue.imag}\n")
    
    # Save spectral radii and determinants
    with open(f"{output_dir}/matrix_properties.csv", 'w') as f:
        f.write("time_step,spectral_radius,determinant_real,determinant_imag\n")
        for t in range(len(analysis['spectral_radii'])):
            f.write(f"{t},{analysis['spectral_radii'][t]},{analysis['determinants'][t].real},{analysis['determinants'][t].imag}\n")
    
    elapsed_time = time.time() - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds")
    print(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    run_simulation()
