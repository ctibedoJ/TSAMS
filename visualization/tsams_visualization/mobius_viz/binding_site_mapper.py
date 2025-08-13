"""
Binding Site Mapper

This module provides tools for mapping protein binding sites to Möbius transformations,
enabling advanced analysis and visualization of protein-ligand interactions.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

# Import core TIBEDO components if available
try:
    from tibedo.core.advanced.mobius_pairing import MobiusPairing
    from tibedo.core.spinor.reduction_chain import SpinorReductionChain
    from tibedo.core.prime_indexed.prime_indexed_structure import PrimeIndexedStructure
    from tibedo.core.advanced.cyclotomic_braid import CyclotomicBraid
    TIBEDO_CORE_AVAILABLE = True
except ImportError:
    TIBEDO_CORE_AVAILABLE = False
    print("Warning: TIBEDO core components not available. Using standalone implementation.")

# Import performance optimization components if available
try:
    from tibedo.performance.gpu_acceleration import GPUAccelerator
    from tibedo.performance.parallel_processing import ParallelMatrixOperations
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class BindingSiteMapper:
    """
    A class for mapping protein binding sites to Möbius transformations.
    
    This class provides tools for analyzing and visualizing protein binding sites
    using advanced mathematical transformations from the TIBEDO framework.
    """
    
    def __init__(self, use_gpu: bool = True, use_parallel: bool = True):
        """
        Initialize the BindingSiteMapper.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available.
            use_parallel: Whether to use parallel processing.
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.use_parallel = use_parallel
        
        if self.use_gpu:
            self.gpu_accelerator = GPUAccelerator()
        
        if self.use_parallel:
            self.parallel_ops = ParallelMatrixOperations()
        
        # Initialize TIBEDO components if available
        if TIBEDO_CORE_AVAILABLE:
            self.mobius_pairing = MobiusPairing()
            self.spinor_reduction = SpinorReductionChain()
            self.prime_indexed = PrimeIndexedStructure()
            self.cyclotomic_braid = CyclotomicBraid()
        
        # Define amino acid properties
        self.aa_properties = {
            'ALA': {'hydrophobicity': 1.8, 'charge': 0, 'aromatic': False, 'polar': False, 'size': 'small'},
            'ARG': {'hydrophobicity': -4.5, 'charge': 1, 'aromatic': False, 'polar': True, 'size': 'large'},
            'ASN': {'hydrophobicity': -3.5, 'charge': 0, 'aromatic': False, 'polar': True, 'size': 'medium'},
            'ASP': {'hydrophobicity': -3.5, 'charge': -1, 'aromatic': False, 'polar': True, 'size': 'medium'},
            'CYS': {'hydrophobicity': 2.5, 'charge': 0, 'aromatic': False, 'polar': True, 'size': 'small'},
            'GLN': {'hydrophobicity': -3.5, 'charge': 0, 'aromatic': False, 'polar': True, 'size': 'medium'},
            'GLU': {'hydrophobicity': -3.5, 'charge': -1, 'aromatic': False, 'polar': True, 'size': 'medium'},
            'GLY': {'hydrophobicity': -0.4, 'charge': 0, 'aromatic': False, 'polar': False, 'size': 'small'},
            'HIS': {'hydrophobicity': -3.2, 'charge': 0.5, 'aromatic': True, 'polar': True, 'size': 'medium'},
            'ILE': {'hydrophobicity': 4.5, 'charge': 0, 'aromatic': False, 'polar': False, 'size': 'large'},
            'LEU': {'hydrophobicity': 3.8, 'charge': 0, 'aromatic': False, 'polar': False, 'size': 'large'},
            'LYS': {'hydrophobicity': -3.9, 'charge': 1, 'aromatic': False, 'polar': True, 'size': 'large'},
            'MET': {'hydrophobicity': 1.9, 'charge': 0, 'aromatic': False, 'polar': False, 'size': 'medium'},
            'PHE': {'hydrophobicity': 2.8, 'charge': 0, 'aromatic': True, 'polar': False, 'size': 'large'},
            'PRO': {'hydrophobicity': -1.6, 'charge': 0, 'aromatic': False, 'polar': False, 'size': 'medium'},
            'SER': {'hydrophobicity': -0.8, 'charge': 0, 'aromatic': False, 'polar': True, 'size': 'small'},
            'THR': {'hydrophobicity': -0.7, 'charge': 0, 'aromatic': False, 'polar': True, 'size': 'medium'},
            'TRP': {'hydrophobicity': -0.9, 'charge': 0, 'aromatic': True, 'polar': True, 'size': 'large'},
            'TYR': {'hydrophobicity': -1.3, 'charge': 0, 'aromatic': True, 'polar': True, 'size': 'large'},
            'VAL': {'hydrophobicity': 4.2, 'charge': 0, 'aromatic': False, 'polar': False, 'size': 'medium'},
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
        if self.use_parallel:
            # Use parallel implementation for large matrices
            distances = self.parallel_ops.pairwise_distances(protein_coords, ligand_coords)
        else:
            distances = cdist(protein_coords, ligand_coords)
        
        # Identify protein atoms within cutoff distance of any ligand atom
        binding_site_mask = np.any(distances <= cutoff, axis=1)
        binding_site_indices = np.where(binding_site_mask)[0]
        
        # Extract binding site coordinates and residues
        binding_site_coords = protein_coords[binding_site_indices]
        binding_site_residues = [protein_residues[i] for i in binding_site_indices]
        
        return binding_site_coords, binding_site_residues, binding_site_indices.tolist()
    
    def calculate_binding_site_properties(self,
                                         binding_site_residues: List[str]) -> Dict[str, float]:
        """
        Calculate properties of the binding site based on residue composition.
        
        Args:
            binding_site_residues: List of residue names in the binding site.
            
        Returns:
            Dictionary of binding site properties.
        """
        # Initialize properties
        properties = {
            'hydrophobicity': 0.0,
            'charge': 0.0,
            'aromatic_fraction': 0.0,
            'polar_fraction': 0.0,
            'size_small': 0.0,
            'size_medium': 0.0,
            'size_large': 0.0,
        }
        
        # Count residue types
        n_residues = len(binding_site_residues)
        if n_residues == 0:
            return properties
        
        # Calculate properties
        for res in binding_site_residues:
            if res in self.aa_properties:
                properties['hydrophobicity'] += self.aa_properties[res]['hydrophobicity']
                properties['charge'] += self.aa_properties[res]['charge']
                properties['aromatic_fraction'] += 1 if self.aa_properties[res]['aromatic'] else 0
                properties['polar_fraction'] += 1 if self.aa_properties[res]['polar'] else 0
                
                if self.aa_properties[res]['size'] == 'small':
                    properties['size_small'] += 1
                elif self.aa_properties[res]['size'] == 'medium':
                    properties['size_medium'] += 1
                elif self.aa_properties[res]['size'] == 'large':
                    properties['size_large'] += 1
        
        # Normalize properties
        properties['hydrophobicity'] /= n_residues
        properties['aromatic_fraction'] /= n_residues
        properties['polar_fraction'] /= n_residues
        properties['size_small'] /= n_residues
        properties['size_medium'] /= n_residues
        properties['size_large'] /= n_residues
        
        return properties
    
    def map_binding_site_to_mobius(self,
                                  binding_site_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Map binding site coordinates to a Möbius strip.
        
        Args:
            binding_site_coords: Array of binding site coordinates.
            
        Returns:
            Tuple of X, Y, Z coordinates mapped to the Möbius strip.
        """
        n_points = len(binding_site_coords)
        
        if n_points == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Map binding site to Möbius strip parameters
        u = np.linspace(0, 2 * np.pi, n_points)
        v = np.zeros(n_points)
        
        # Use TIBEDO's MobiusPairing if available for more accurate mapping
        if TIBEDO_CORE_AVAILABLE:
            try:
                # Convert sequence to tensor format expected by MobiusPairing
                sequence_tensor = torch.tensor(binding_site_coords, dtype=torch.float32)
                
                # Use MobiusPairing to compute optimal mapping parameters
                u, v = self.mobius_pairing.compute_optimal_mapping(sequence_tensor)
                u = u.numpy()
                v = v.numpy()
            except Exception as e:
                print(f"Warning: Error using MobiusPairing: {e}. Falling back to simple mapping.")
        
        # Compute Möbius strip coordinates for the binding site
        x = (1 + v * np.cos(u / 2)) * np.cos(u)
        y = (1 + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        
        return x, y, z
    
    def calculate_binding_site_curvature(self,
                                        binding_site_coords: np.ndarray) -> float:
        """
        Calculate the curvature of the binding site.
        
        Args:
            binding_site_coords: Array of binding site coordinates.
            
        Returns:
            Curvature value.
        """
        n_points = len(binding_site_coords)
        
        if n_points < 3:
            return 0.0
        
        # Calculate centroid
        centroid = np.mean(binding_site_coords, axis=0)
        
        # Calculate distances from centroid
        distances = np.linalg.norm(binding_site_coords - centroid, axis=1)
        
        # Calculate standard deviation of distances (measure of curvature)
        curvature = np.std(distances) / np.mean(distances)
        
        return curvature
    
    def calculate_binding_site_planarity(self,
                                        binding_site_coords: np.ndarray) -> float:
        """
        Calculate the planarity of the binding site.
        
        Args:
            binding_site_coords: Array of binding site coordinates.
            
        Returns:
            Planarity value (0 = perfectly planar, higher values = less planar).
        """
        n_points = len(binding_site_coords)
        
        if n_points < 3:
            return 0.0
        
        # Calculate centroid
        centroid = np.mean(binding_site_coords, axis=0)
        
        # Center coordinates
        centered_coords = binding_site_coords - centroid
        
        # Perform SVD to find principal components
        U, S, Vt = np.linalg.svd(centered_coords, full_matrices=False)
        
        # Planarity is the ratio of the smallest singular value to the sum
        planarity = S[2] / np.sum(S)
        
        return planarity
    
    def calculate_binding_site_depth(self,
                                    binding_site_coords: np.ndarray,
                                    protein_coords: np.ndarray) -> float:
        """
        Calculate the depth of the binding site within the protein.
        
        Args:
            binding_site_coords: Array of binding site coordinates.
            protein_coords: Array of all protein coordinates.
            
        Returns:
            Depth value.
        """
        n_points = len(binding_site_coords)
        
        if n_points == 0:
            return 0.0
        
        # Calculate centroid of binding site
        binding_site_centroid = np.mean(binding_site_coords, axis=0)
        
        # Calculate centroid of protein
        protein_centroid = np.mean(protein_coords, axis=0)
        
        # Calculate distance from binding site centroid to protein surface
        # (approximated as distance to furthest protein atom from protein centroid)
        protein_distances = np.linalg.norm(protein_coords - protein_centroid, axis=1)
        protein_radius = np.max(protein_distances)
        
        # Calculate distance from binding site centroid to protein centroid
        binding_site_distance = np.linalg.norm(binding_site_centroid - protein_centroid)
        
        # Calculate depth as 1 - (distance / radius)
        # 0 = on surface, 1 = at center
        depth = 1.0 - (binding_site_distance / protein_radius)
        
        return depth
    
    def calculate_binding_site_volume(self,
                                     binding_site_coords: np.ndarray) -> float:
        """
        Calculate the volume of the binding site.
        
        Args:
            binding_site_coords: Array of binding site coordinates.
            
        Returns:
            Volume value.
        """
        n_points = len(binding_site_coords)
        
        if n_points < 4:
            return 0.0
        
        # Calculate convex hull volume
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(binding_site_coords)
            volume = hull.volume
        except Exception as e:
            print(f"Warning: Error calculating convex hull: {e}. Using approximate volume.")
            # Approximate volume as bounding box
            min_coords = np.min(binding_site_coords, axis=0)
            max_coords = np.max(binding_site_coords, axis=0)
            volume = np.prod(max_coords - min_coords)
        
        return volume
    
    def calculate_binding_site_mobius_invariants(self,
                                               binding_site_coords: np.ndarray) -> Dict[str, float]:
        """
        Calculate Möbius invariants of the binding site.
        
        Args:
            binding_site_coords: Array of binding site coordinates.
            
        Returns:
            Dictionary of Möbius invariants.
        """
        n_points = len(binding_site_coords)
        
        if n_points < 4:
            return {'cross_ratio': 0.0, 'mobius_invariant': 0.0}
        
        # Calculate cross-ratio for first four points
        # Cross-ratio is a Möbius invariant
        p1, p2, p3, p4 = binding_site_coords[:4]
        
        # Calculate distances
        d12 = np.linalg.norm(p1 - p2)
        d34 = np.linalg.norm(p3 - p4)
        d13 = np.linalg.norm(p1 - p3)
        d24 = np.linalg.norm(p2 - p4)
        d14 = np.linalg.norm(p1 - p4)
        d23 = np.linalg.norm(p2 - p3)
        
        # Calculate cross-ratio
        cross_ratio = (d12 * d34) / (d13 * d24)
        
        # Calculate another Möbius invariant
        mobius_invariant = (d12 * d34 * d14 * d23) / (d13 * d24 * d12 * d34)
        
        return {'cross_ratio': cross_ratio, 'mobius_invariant': mobius_invariant}
    
    def analyze_binding_site(self,
                            protein_coords: np.ndarray,
                            protein_residues: List[str],
                            ligand_coords: np.ndarray,
                            cutoff: float = 4.0) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a binding site.
        
        Args:
            protein_coords: Array of protein atom coordinates.
            protein_residues: List of residue names corresponding to protein_coords.
            ligand_coords: Array of ligand atom coordinates.
            cutoff: Distance cutoff for binding site identification.
            
        Returns:
            Dictionary of binding site analysis results.
        """
        # Identify binding site
        binding_site_coords, binding_site_residues, binding_site_indices = self.identify_binding_site(
            protein_coords, protein_residues, ligand_coords, cutoff=cutoff
        )
        
        # Calculate binding site properties
        properties = self.calculate_binding_site_properties(binding_site_residues)
        
        # Calculate binding site geometry
        curvature = self.calculate_binding_site_curvature(binding_site_coords)
        planarity = self.calculate_binding_site_planarity(binding_site_coords)
        depth = self.calculate_binding_site_depth(binding_site_coords, protein_coords)
        volume = self.calculate_binding_site_volume(binding_site_coords)
        
        # Calculate Möbius invariants
        mobius_invariants = self.calculate_binding_site_mobius_invariants(binding_site_coords)
        
        # Map binding site to Möbius strip
        mobius_x, mobius_y, mobius_z = self.map_binding_site_to_mobius(binding_site_coords)
        
        # Compile results
        results = {
            'binding_site_coords': binding_site_coords,
            'binding_site_residues': binding_site_residues,
            'binding_site_indices': binding_site_indices,
            'properties': properties,
            'geometry': {
                'curvature': curvature,
                'planarity': planarity,
                'depth': depth,
                'volume': volume,
            },
            'mobius_invariants': mobius_invariants,
            'mobius_mapping': {
                'x': mobius_x,
                'y': mobius_y,
                'z': mobius_z,
            }
        }
        
        return results
    
    def visualize_binding_site_properties(self,
                                         analysis_results: Dict[str, Any],
                                         title: str = "Binding Site Properties",
                                         save_path: Optional[str] = None,
                                         show: bool = True) -> plt.Figure:
        """
        Visualize binding site properties from analysis results.
        
        Args:
            analysis_results: Results from analyze_binding_site.
            title: Title of the plot.
            save_path: Path to save the figure.
            show: Whether to show the figure.
            
        Returns:
            Matplotlib figure object.
        """
        # Extract properties
        properties = analysis_results['properties']
        geometry = analysis_results['geometry']
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot residue composition
        residue_counts = {}
        for res in analysis_results['binding_site_residues']:
            residue_counts[res] = residue_counts.get(res, 0) + 1
        
        sorted_residues = sorted(residue_counts.items(), key=lambda x: x[1], reverse=True)
        residues = [r[0] for r in sorted_residues]
        counts = [r[1] for r in sorted_residues]
        
        axs[0, 0].bar(residues, counts)
        axs[0, 0].set_title('Residue Composition')
        axs[0, 0].set_xlabel('Residue')
        axs[0, 0].set_ylabel('Count')
        axs[0, 0].tick_params(axis='x', rotation=90)
        
        # Plot property distribution
        property_names = ['hydrophobicity', 'charge', 'aromatic_fraction', 'polar_fraction']
        property_values = [properties[p] for p in property_names]
        
        axs[0, 1].bar(property_names, property_values)
        axs[0, 1].set_title('Property Distribution')
        axs[0, 1].set_xlabel('Property')
        axs[0, 1].set_ylabel('Value')
        axs[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot size distribution
        size_names = ['size_small', 'size_medium', 'size_large']
        size_values = [properties[s] for s in size_names]
        
        axs[1, 0].pie(size_values, labels=['Small', 'Medium', 'Large'], autopct='%1.1f%%')
        axs[1, 0].set_title('Size Distribution')
        
        # Plot geometry properties
        geometry_names = ['curvature', 'planarity', 'depth', 'volume']
        geometry_values = [geometry[g] for g in geometry_names]
        
        # Normalize volume for better visualization
        geometry_values[3] = geometry_values[3] / 100 if geometry_values[3] > 100 else geometry_values[3]
        
        axs[1, 1].bar(geometry_names, geometry_values)
        axs[1, 1].set_title('Geometry Properties')
        axs[1, 1].set_xlabel('Property')
        axs[1, 1].set_ylabel('Value')
        axs[1, 1].tick_params(axis='x', rotation=45)
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show figure if requested
        if show:
            plt.show()
        
        return fig
    
    def visualize_binding_site_on_mobius(self,
                                        analysis_results: Dict[str, Any],
                                        title: str = "Binding Site on Möbius Strip",
                                        save_path: Optional[str] = None,
                                        show: bool = True) -> plt.Figure:
        """
        Visualize binding site on a Möbius strip from analysis results.
        
        Args:
            analysis_results: Results from analyze_binding_site.
            title: Title of the plot.
            save_path: Path to save the figure.
            show: Whether to show the figure.
            
        Returns:
            Matplotlib figure object.
        """
        # Extract Möbius mapping
        mobius_mapping = analysis_results['mobius_mapping']
        mobius_x = mobius_mapping['x']
        mobius_y = mobius_mapping['y']
        mobius_z = mobius_mapping['z']
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create Möbius strip
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(-0.5, 0.5, 100)
        u_grid, v_grid = np.meshgrid(u, v)
        
        # Möbius strip parametric equations
        strip_x = (1 + v_grid * np.cos(u_grid / 2)) * np.cos(u_grid)
        strip_y = (1 + v_grid * np.cos(u_grid / 2)) * np.sin(u_grid)
        strip_z = v_grid * np.sin(u_grid / 2)
        
        # Plot Möbius strip
        surf = ax.plot_surface(strip_x, strip_y, strip_z, alpha=0.7, 
                              color='lightblue', edgecolor='none')
        
        # Plot binding site on Möbius strip
        if len(mobius_x) > 0:
            # Color points by residue properties
            properties = analysis_results['properties']
            residues = analysis_results['binding_site_residues']
            
            # Get hydrophobicity values for coloring
            hydrophobicity = []
            for res in residues:
                if res in self.aa_properties:
                    hydrophobicity.append(self.aa_properties[res]['hydrophobicity'])
                else:
                    hydrophobicity.append(0)
            
            # Normalize hydrophobicity for coloring
            if len(hydrophobicity) > 0:
                min_h = min(hydrophobicity)
                max_h = max(hydrophobicity)
                if max_h > min_h:
                    hydrophobicity = [(h - min_h) / (max_h - min_h) for h in hydrophobicity]
            
            # Plot binding site points
            scatter = ax.scatter(mobius_x, mobius_y, mobius_z, 
                               c=hydrophobicity, cmap='coolwarm', 
                               s=50, edgecolor='black')
            
            # Plot line connecting points
            ax.plot(mobius_x, mobius_y, mobius_z, 'k-', linewidth=1)
            
            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Hydrophobicity')
        
        # Set plot properties
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True)
        
        # Add description text
        description = (
            "Visualization of binding site using quaternion-based Möbius strip mapping.\n"
            "Points represent binding site residues colored by hydrophobicity (blue=hydrophilic, red=hydrophobic).\n"
            f"Binding site properties: Volume={analysis_results['geometry']['volume']:.2f}, "
            f"Curvature={analysis_results['geometry']['curvature']:.2f}, "
            f"Depth={analysis_results['geometry']['depth']:.2f}"
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