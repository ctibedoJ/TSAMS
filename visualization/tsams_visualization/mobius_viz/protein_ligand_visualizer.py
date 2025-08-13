"""
Protein-Ligand Interaction Visualizer

This module provides tools for visualizing protein-ligand interactions using Möbius transformations,
enabling intuitive representation of binding sites and interaction networks.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import torch
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import time
from scipy.spatial.distance import cdist

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

# Import MobiusTransformationVisualizer
from .mobius_transformation_visualizer import MobiusTransformationVisualizer


class ProteinLigandVisualizer:
    """
    A class for visualizing protein-ligand interactions using Möbius transformations.
    
    This class provides tools for visualizing binding sites, interaction networks,
    and binding energetics between proteins and ligands.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the ProteinLigandVisualizer.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available.
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            self.gpu_accelerator = GPUAccelerator()
        
        # Initialize Möbius transformation visualizer
        self.mobius_visualizer = MobiusTransformationVisualizer(use_gpu=use_gpu)
        
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
            'interaction_cutoff': 4.0,  # Angstroms
            'hydrogen_bond_cutoff': 3.5,  # Angstroms
            'hydrophobic_cutoff': 4.0,  # Angstroms
            'ionic_cutoff': 4.0,  # Angstroms
            'pi_stacking_cutoff': 5.5,  # Angstroms
            'cation_pi_cutoff': 6.0,  # Angstroms
            'halogen_bond_cutoff': 4.0,  # Angstroms
        }
        
        # Define amino acid properties
        self.aa_properties = {
            'ALA': {'hydrophobicity': 1.8, 'charge': 0, 'aromatic': False, 'polar': False},
            'ARG': {'hydrophobicity': -4.5, 'charge': 1, 'aromatic': False, 'polar': True},
            'ASN': {'hydrophobicity': -3.5, 'charge': 0, 'aromatic': False, 'polar': True},
            'ASP': {'hydrophobicity': -3.5, 'charge': -1, 'aromatic': False, 'polar': True},
            'CYS': {'hydrophobicity': 2.5, 'charge': 0, 'aromatic': False, 'polar': True},
            'GLN': {'hydrophobicity': -3.5, 'charge': 0, 'aromatic': False, 'polar': True},
            'GLU': {'hydrophobicity': -3.5, 'charge': -1, 'aromatic': False, 'polar': True},
            'GLY': {'hydrophobicity': -0.4, 'charge': 0, 'aromatic': False, 'polar': False},
            'HIS': {'hydrophobicity': -3.2, 'charge': 0.5, 'aromatic': True, 'polar': True},
            'ILE': {'hydrophobicity': 4.5, 'charge': 0, 'aromatic': False, 'polar': False},
            'LEU': {'hydrophobicity': 3.8, 'charge': 0, 'aromatic': False, 'polar': False},
            'LYS': {'hydrophobicity': -3.9, 'charge': 1, 'aromatic': False, 'polar': True},
            'MET': {'hydrophobicity': 1.9, 'charge': 0, 'aromatic': False, 'polar': False},
            'PHE': {'hydrophobicity': 2.8, 'charge': 0, 'aromatic': True, 'polar': False},
            'PRO': {'hydrophobicity': -1.6, 'charge': 0, 'aromatic': False, 'polar': False},
            'SER': {'hydrophobicity': -0.8, 'charge': 0, 'aromatic': False, 'polar': True},
            'THR': {'hydrophobicity': -0.7, 'charge': 0, 'aromatic': False, 'polar': True},
            'TRP': {'hydrophobicity': -0.9, 'charge': 0, 'aromatic': True, 'polar': True},
            'TYR': {'hydrophobicity': -1.3, 'charge': 0, 'aromatic': True, 'polar': True},
            'VAL': {'hydrophobicity': 4.2, 'charge': 0, 'aromatic': False, 'polar': False},
        }
    
    def identify_binding_site(self, 
                             protein_coords: np.ndarray, 
                             protein_residues: List[str],
                             ligand_coords: np.ndarray,
                             cutoff: float = 4.0) -> Tuple[np.ndarray, List[str], List[int]]:
        """
        Identify protein residues that form the binding site with a ligand.
        
        Args:
            protein_coords: Array of protein atom coordinates.
            protein_residues: List of residue names corresponding to protein_coords.
            ligand_coords: Array of ligand atom coordinates.
            cutoff: Distance cutoff for binding site identification.
            
        Returns:
            Tuple of binding site coordinates, residue names, and residue indices.
        """
        # Calculate distances between protein and ligand atoms
        distances = cdist(protein_coords, ligand_coords)
        
        # Identify protein atoms within cutoff distance of any ligand atom
        binding_site_mask = np.any(distances <= cutoff, axis=1)
        binding_site_indices = np.where(binding_site_mask)[0]
        
        # Extract binding site coordinates and residues
        binding_site_coords = protein_coords[binding_site_indices]
        binding_site_residues = [protein_residues[i] for i in binding_site_indices]
        
        return binding_site_coords, binding_site_residues, binding_site_indices.tolist()
    
    def identify_interactions(self,
                             protein_coords: np.ndarray,
                             protein_residues: List[str],
                             protein_atoms: List[str],
                             ligand_coords: np.ndarray,
                             ligand_atoms: List[str],
                             **kwargs) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        Identify interactions between protein and ligand.
        
        Args:
            protein_coords: Array of protein atom coordinates.
            protein_residues: List of residue names corresponding to protein_coords.
            protein_atoms: List of atom names corresponding to protein_coords.
            ligand_coords: Array of ligand atom coordinates.
            ligand_atoms: List of atom names corresponding to ligand_coords.
            **kwargs: Additional parameters for interaction identification.
            
        Returns:
            Dictionary of interaction types and their details.
        """
        # Update parameters with provided kwargs
        params = self.default_params.copy()
        params.update(kwargs)
        
        # Calculate distances between protein and ligand atoms
        distances = cdist(protein_coords, ligand_coords)
        
        # Initialize interaction dictionary
        interactions = {
            'hydrogen_bonds': [],
            'hydrophobic': [],
            'ionic': [],
            'pi_stacking': [],
            'cation_pi': [],
            'halogen_bonds': [],
            'metal_coordination': [],
            'water_bridges': [],
            'salt_bridges': [],
            'all': []
        }
        
        # Identify all interactions within cutoff
        for p_idx in range(len(protein_coords)):
            for l_idx in range(len(ligand_coords)):
                dist = distances[p_idx, l_idx]
                
                # Skip if distance is greater than general cutoff
                if dist > params['interaction_cutoff']:
                    continue
                
                # Add to all interactions
                interactions['all'].append((p_idx, l_idx, dist))
                
                # Identify specific interaction types based on atom types and distances
                p_res = protein_residues[p_idx]
                p_atom = protein_atoms[p_idx]
                l_atom = ligand_atoms[l_idx]
                
                # Hydrogen bonds (simplified check)
                if (dist <= params['hydrogen_bond_cutoff'] and
                    (p_atom.startswith('N') or p_atom.startswith('O') or 
                     p_atom.startswith('S')) and
                    (l_atom.startswith('N') or l_atom.startswith('O') or 
                     l_atom.startswith('S'))):
                    interactions['hydrogen_bonds'].append((p_idx, l_idx, dist))
                
                # Hydrophobic interactions
                if (dist <= params['hydrophobic_cutoff'] and
                    p_atom.startswith('C') and l_atom.startswith('C') and
                    p_res in ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'TYR']):
                    interactions['hydrophobic'].append((p_idx, l_idx, dist))
                
                # Ionic interactions
                if (dist <= params['ionic_cutoff'] and
                    ((p_res in ['ARG', 'LYS', 'HIS'] and l_atom.startswith('O')) or
                     (p_res in ['ASP', 'GLU'] and l_atom.startswith('N')))):
                    interactions['ionic'].append((p_idx, l_idx, dist))
                
                # Other interaction types would require more sophisticated checks
                # and are simplified here
        
        return interactions
    
    def calculate_interaction_energies(self,
                                      protein_coords: np.ndarray,
                                      protein_residues: List[str],
                                      protein_atoms: List[str],
                                      ligand_coords: np.ndarray,
                                      ligand_atoms: List[str],
                                      **kwargs) -> Dict[str, float]:
        """
        Calculate interaction energies between protein and ligand.
        
        Args:
            protein_coords: Array of protein atom coordinates.
            protein_residues: List of residue names corresponding to protein_coords.
            protein_atoms: List of atom names corresponding to protein_coords.
            ligand_coords: Array of ligand atom coordinates.
            ligand_atoms: List of atom names corresponding to ligand_coords.
            **kwargs: Additional parameters for energy calculation.
            
        Returns:
            Dictionary of energy components and total energy.
        """
        # Identify interactions
        interactions = self.identify_interactions(
            protein_coords, protein_residues, protein_atoms,
            ligand_coords, ligand_atoms, **kwargs
        )
        
        # Initialize energy components
        energies = {
            'electrostatic': 0.0,
            'van_der_waals': 0.0,
            'hydrogen_bond': 0.0,
            'hydrophobic': 0.0,
            'total': 0.0
        }
        
        # Calculate simplified energy components
        # Note: This is a simplified model for demonstration purposes
        
        # Electrostatic energy (simplified)
        for p_idx, l_idx, dist in interactions['ionic']:
            # Simple Coulomb-like term
            energies['electrostatic'] += -1.0 / dist
        
        # Van der Waals energy (simplified Lennard-Jones)
        for p_idx, l_idx, dist in interactions['all']:
            # Simplified Lennard-Jones potential
            energies['van_der_waals'] += 0.1 * ((1/dist)**12 - 2*(1/dist)**6)
        
        # Hydrogen bond energy
        for p_idx, l_idx, dist in interactions['hydrogen_bonds']:
            # Simple distance-dependent term
            energies['hydrogen_bond'] += -2.0 / dist
        
        # Hydrophobic energy
        for p_idx, l_idx, dist in interactions['hydrophobic']:
            # Simple hydrophobic contact term
            p_res = protein_residues[p_idx]
            hydrophobicity = self.aa_properties.get(p_res, {}).get('hydrophobicity', 0)
            energies['hydrophobic'] += -0.5 * hydrophobicity / dist
        
        # Calculate total energy
        energies['total'] = sum(v for k, v in energies.items() if k != 'total')
        
        return energies
    
    def visualize_binding_site(self,
                              protein_coords: np.ndarray,
                              protein_residues: List[str],
                              protein_atoms: List[str],
                              ligand_coords: np.ndarray,
                              ligand_atoms: List[str],
                              title: str = "Protein-Ligand Binding Site",
                              save_path: Optional[str] = None,
                              show: bool = True,
                              **kwargs) -> plt.Figure:
        """
        Visualize the binding site between protein and ligand.
        
        Args:
            protein_coords: Array of protein atom coordinates.
            protein_residues: List of residue names corresponding to protein_coords.
            protein_atoms: List of atom names corresponding to protein_coords.
            ligand_coords: Array of ligand atom coordinates.
            ligand_atoms: List of atom names corresponding to ligand_coords.
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
        
        # Identify binding site
        binding_site_coords, binding_site_residues, binding_site_indices = self.identify_binding_site(
            protein_coords, protein_residues, ligand_coords, cutoff=params['interaction_cutoff']
        )
        
        # Identify interactions
        interactions = self.identify_interactions(
            protein_coords, protein_residues, protein_atoms,
            ligand_coords, ligand_atoms, **kwargs
        )
        
        # Calculate interaction energies
        energies = self.calculate_interaction_energies(
            protein_coords, protein_residues, protein_atoms,
            ligand_coords, ligand_atoms, **kwargs
        )
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot binding site residues
        ax.scatter(binding_site_coords[:, 0], 
                  binding_site_coords[:, 1], 
                  binding_site_coords[:, 2],
                  c='blue', s=params['point_size'], alpha=0.7, label='Binding Site')
        
        # Plot ligand atoms
        ax.scatter(ligand_coords[:, 0], 
                  ligand_coords[:, 1], 
                  ligand_coords[:, 2],
                  c='red', s=params['point_size']*1.5, alpha=0.9, label='Ligand')
        
        # Plot interactions
        if params['show_interactions']:
            # Plot hydrogen bonds
            if params['show_hydrogen_bonds'] and interactions['hydrogen_bonds']:
                for p_idx, l_idx, dist in interactions['hydrogen_bonds']:
                    ax.plot([protein_coords[p_idx, 0], ligand_coords[l_idx, 0]],
                           [protein_coords[p_idx, 1], ligand_coords[l_idx, 1]],
                           [protein_coords[p_idx, 2], ligand_coords[l_idx, 2]],
                           'g-', linewidth=1.5, alpha=0.7)
            
            # Plot hydrophobic interactions
            if params['show_hydrophobic_interactions'] and interactions['hydrophobic']:
                for p_idx, l_idx, dist in interactions['hydrophobic']:
                    ax.plot([protein_coords[p_idx, 0], ligand_coords[l_idx, 0]],
                           [protein_coords[p_idx, 1], ligand_coords[l_idx, 1]],
                           [protein_coords[p_idx, 2], ligand_coords[l_idx, 2]],
                           'y-', linewidth=1.5, alpha=0.7)
            
            # Plot ionic interactions
            if params['show_ionic_interactions'] and interactions['ionic']:
                for p_idx, l_idx, dist in interactions['ionic']:
                    ax.plot([protein_coords[p_idx, 0], ligand_coords[l_idx, 0]],
                           [protein_coords[p_idx, 1], ligand_coords[l_idx, 1]],
                           [protein_coords[p_idx, 2], ligand_coords[l_idx, 2]],
                           'm-', linewidth=1.5, alpha=0.7)
        
        # Set plot properties
        if params['show_title']:
            energy_text = f"Total Binding Energy: {energies['total']:.2f} kcal/mol"
            full_title = f"{title}\n{energy_text}"
            ax.set_title(full_title, fontsize=14)
        
        if params['show_axes']:
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
        
        if params['show_grid']:
            ax.grid(True)
        
        if params['show_legend']:
            ax.legend()
        
        # Add energy breakdown text
        if params['show_energy']:
            energy_breakdown = (
                f"Energy Components (kcal/mol):\n"
                f"Electrostatic: {energies['electrostatic']:.2f}\n"
                f"Van der Waals: {energies['van_der_waals']:.2f}\n"
                f"Hydrogen Bond: {energies['hydrogen_bond']:.2f}\n"
                f"Hydrophobic: {energies['hydrophobic']:.2f}\n"
                f"Total: {energies['total']:.2f}"
            )
            fig.text(0.02, 0.02, energy_breakdown, fontsize=10)
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show figure if requested
        if show:
            plt.tight_layout()
            plt.show()
        
        return fig
    
    def visualize_binding_site_on_mobius(self,
                                        protein_coords: np.ndarray,
                                        protein_residues: List[str],
                                        protein_atoms: List[str],
                                        ligand_coords: np.ndarray,
                                        ligand_atoms: List[str],
                                        title: str = "Protein-Ligand Binding Site on Möbius Strip",
                                        save_path: Optional[str] = None,
                                        show: bool = True,
                                        **kwargs) -> plt.Figure:
        """
        Visualize the binding site between protein and ligand on a Möbius strip.
        
        Args:
            protein_coords: Array of protein atom coordinates.
            protein_residues: List of residue names corresponding to protein_coords.
            protein_atoms: List of atom names corresponding to protein_coords.
            ligand_coords: Array of ligand atom coordinates.
            ligand_atoms: List of atom names corresponding to ligand_coords.
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
        
        # Identify binding site
        binding_site_coords, binding_site_residues, binding_site_indices = self.identify_binding_site(
            protein_coords, protein_residues, ligand_coords, cutoff=params['interaction_cutoff']
        )
        
        # Calculate interaction energies
        energies = self.calculate_interaction_energies(
            protein_coords, protein_residues, protein_atoms,
            ligand_coords, ligand_atoms, **kwargs
        )
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create Möbius strip
        strip_x, strip_y, strip_z = self.mobius_visualizer.create_mobius_strip(
            width=params['strip_width'],
            resolution=params['strip_resolution']
        )
        
        # Plot Möbius strip
        surf = ax.plot_surface(strip_x, strip_y, strip_z, alpha=params['alpha'], 
                              color='lightblue', edgecolor='none')
        
        # Map binding site to Möbius strip
        if len(binding_site_coords) > 0:
            bs_x, bs_y, bs_z, _ = self.mobius_visualizer.map_sequence_to_mobius(
                binding_site_coords.tolist()
            )
            
            # Map ligand to Möbius strip (centered around binding site)
            ligand_x, ligand_y, ligand_z, _ = self.mobius_visualizer.map_sequence_to_mobius(
                ligand_coords.tolist()
            )
            
            # Plot binding site on Möbius strip
            ax.scatter(bs_x, bs_y, bs_z, c='blue', 
                      s=params['point_size'], alpha=0.7, label='Binding Site')
            
            # Plot ligand on Möbius strip
            ax.scatter(ligand_x, ligand_y, ligand_z, c='red', 
                      s=params['point_size']*1.5, alpha=0.9, label='Ligand')
            
            # Plot connections between binding site and ligand
            for i in range(min(len(bs_x), len(ligand_x))):
                ax.plot([bs_x[i % len(bs_x)], ligand_x[i % len(ligand_x)]],
                       [bs_y[i % len(bs_y)], ligand_y[i % len(ligand_y)]],
                       [bs_z[i % len(bs_z)], ligand_z[i % len(ligand_z)]],
                       'g-', linewidth=1.0, alpha=0.5)
        
        # Set plot properties
        if params['show_title']:
            energy_text = f"Total Binding Energy: {energies['total']:.2f} kcal/mol"
            full_title = f"{title}\n{energy_text}"
            ax.set_title(full_title, fontsize=14)
        
        if params['show_axes']:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        if params['show_grid']:
            ax.grid(True)
        
        if params['show_legend']:
            ax.legend()
        
        # Add description text
        description = (
            "Visualization of protein-ligand binding using quaternion-based Möbius strip mapping.\n"
            "Blue points represent protein binding site residues, red points represent ligand atoms.\n"
            "Green lines indicate potential interactions between protein and ligand."
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
    
    def visualize_interaction_network(self,
                                     protein_coords: np.ndarray,
                                     protein_residues: List[str],
                                     protein_atoms: List[str],
                                     ligand_coords: np.ndarray,
                                     ligand_atoms: List[str],
                                     title: str = "Protein-Ligand Interaction Network",
                                     save_path: Optional[str] = None,
                                     show: bool = True,
                                     **kwargs) -> plt.Figure:
        """
        Visualize the interaction network between protein and ligand.
        
        Args:
            protein_coords: Array of protein atom coordinates.
            protein_residues: List of residue names corresponding to protein_coords.
            protein_atoms: List of atom names corresponding to protein_coords.
            ligand_coords: Array of ligand atom coordinates.
            ligand_atoms: List of atom names corresponding to ligand_coords.
            title: Title of the plot.
            save_path: Path to save the figure.
            show: Whether to show the figure.
            **kwargs: Additional visualization parameters.
            
        Returns:
            Matplotlib figure object.
        """
        # Try to import networkx for network visualization
        try:
            import networkx as nx
        except ImportError:
            print("Warning: networkx not available. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "networkx"])
            import networkx as nx
        
        # Update visualization parameters with provided kwargs
        params = self.default_params.copy()
        params.update(kwargs)
        
        # Identify binding site
        binding_site_coords, binding_site_residues, binding_site_indices = self.identify_binding_site(
            protein_coords, protein_residues, ligand_coords, cutoff=params['interaction_cutoff']
        )
        
        # Identify interactions
        interactions = self.identify_interactions(
            protein_coords, protein_residues, protein_atoms,
            ligand_coords, ligand_atoms, **kwargs
        )
        
        # Create graph
        G = nx.Graph()
        
        # Add protein nodes
        for i, (res, atom) in enumerate(zip(protein_residues, protein_atoms)):
            if i in binding_site_indices:
                G.add_node(f"P{i}", type="protein", residue=res, atom=atom, 
                          color="blue", size=params['point_size'])
        
        # Add ligand nodes
        for i, atom in enumerate(ligand_atoms):
            G.add_node(f"L{i}", type="ligand", atom=atom, 
                      color="red", size=params['point_size']*1.5)
        
        # Add interaction edges
        for interaction_type, interaction_list in interactions.items():
            if interaction_type == 'all':
                continue
                
            for p_idx, l_idx, dist in interaction_list:
                if p_idx in binding_site_indices:
                    G.add_edge(f"P{p_idx}", f"L{l_idx}", 
                              type=interaction_type, distance=dist, weight=1.0/dist)
        
        # Create figure
        fig = plt.figure(figsize=(14, 12))
        
        # Get node positions using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Get node colors and sizes
        node_colors = [data['color'] for _, data in G.nodes(data=True)]
        node_sizes = [data['size'] for _, data in G.nodes(data=True)]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        
        # Draw edges with different colors based on interaction type
        edge_colors = {
            'hydrogen_bonds': 'green',
            'hydrophobic': 'yellow',
            'ionic': 'magenta',
            'pi_stacking': 'cyan',
            'cation_pi': 'orange',
            'halogen_bonds': 'purple',
        }
        
        for interaction_type, color in edge_colors.items():
            edges = [(u, v) for u, v, data in G.edges(data=True) if data['type'] == interaction_type]
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, width=2, alpha=0.7)
        
        # Draw labels
        labels = {}
        for node, data in G.nodes(data=True):
            if data['type'] == 'protein':
                labels[node] = f"{data['residue']}"
            else:
                labels[node] = f"{data['atom']}"
        
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', marker='o', linestyle='', markersize=10, label='Protein'),
            plt.Line2D([0], [0], color='red', marker='o', linestyle='', markersize=10, label='Ligand'),
            plt.Line2D([0], [0], color='green', linestyle='-', linewidth=2, label='Hydrogen Bond'),
            plt.Line2D([0], [0], color='yellow', linestyle='-', linewidth=2, label='Hydrophobic'),
            plt.Line2D([0], [0], color='magenta', linestyle='-', linewidth=2, label='Ionic'),
            plt.Line2D([0], [0], color='cyan', linestyle='-', linewidth=2, label='Pi-Stacking'),
            plt.Line2D([0], [0], color='orange', linestyle='-', linewidth=2, label='Cation-Pi'),
            plt.Line2D([0], [0], color='purple', linestyle='-', linewidth=2, label='Halogen Bond'),
        ]
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Set title
        plt.title(title, fontsize=14)
        
        # Remove axis
        plt.axis('off')
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show figure if requested
        if show:
            plt.tight_layout()
            plt.show()
        
        return fig