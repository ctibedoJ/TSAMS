"""
SARS-CoV-2 Spike Protein Analysis Module

This module provides tools for analyzing the SARS-CoV-2 Spike protein structure,
binding interactions, and dynamics using the TIBEDO Framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import time
from scipy.spatial.distance import cdist
import os
import sys

# Import TIBEDO components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tibedo.core.spinor.reduction_chain import ReductionChain
from tibedo.core.prime_indexed.prime_indexed_structure import PrimeIndexedStructure
from tibedo.core.advanced.protein_simulator import ProteinSimulator
from tibedo.ml.neural_networks.tibedo_neural_network import TibedoNeuralNetwork
from tibedo.visualization.mobius_viz.mobius_transformation_visualizer import MobiusTransformationVisualizer
from tibedo.visualization.mobius_viz.protein_ligand_visualizer import ProteinLigandVisualizer
from tibedo.visualization.mobius_viz.binding_site_mapper import BindingSiteMapper

# Import performance optimization components
try:
    from tibedo.performance.gpu_acceleration import GPUAccelerator
    from tibedo.performance.parallel_processing import ParallelMatrixOperations
    from tibedo.performance.memory_optimization import MemoryEfficientArray
    PERFORMANCE_MODULES_AVAILABLE = True
except ImportError:
    PERFORMANCE_MODULES_AVAILABLE = False
    print("Warning: Performance optimization modules not available.")


class SpikeProteinAnalyzer:
    """
    A class for analyzing the SARS-CoV-2 Spike protein structure and interactions.
    
    This class provides tools for analyzing the Spike protein's binding to the ACE2 receptor,
    structural dynamics, and potential therapeutic targets using the TIBEDO Framework.
    """
    
    def __init__(self, use_gpu: bool = True, use_parallel: bool = True):
        """
        Initialize the SpikeProteinAnalyzer.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available.
            use_parallel: Whether to use parallel processing.
        """
        self.use_gpu = use_gpu and PERFORMANCE_MODULES_AVAILABLE
        self.use_parallel = use_parallel and PERFORMANCE_MODULES_AVAILABLE
        
        if self.use_gpu:
            self.gpu_accelerator = GPUAccelerator()
        
        if self.use_parallel:
            self.parallel_ops = ParallelMatrixOperations()
        
        # Initialize visualization tools
        self.mobius_visualizer = MobiusTransformationVisualizer(use_gpu=use_gpu)
        self.protein_ligand_visualizer = ProteinLigandVisualizer(use_gpu=use_gpu)
        self.binding_site_mapper = BindingSiteMapper(use_gpu=use_gpu, use_parallel=use_parallel)
        
        # Initialize protein simulator
        self.protein_simulator = ProteinSimulator()
        
        # Define spike protein domains
        self.spike_domains = {
            'NTD': (1, 305),    # N-terminal domain
            'RBD': (306, 541),  # Receptor binding domain
            'RBM': (437, 508),  # Receptor binding motif
            'SD1': (542, 591),  # Subdomain 1
            'SD2': (592, 686),  # Subdomain 2
            'S1/S2': (682, 685), # S1/S2 cleavage site
            'FP': (816, 837),   # Fusion peptide
            'HR1': (912, 984),  # Heptad repeat 1
            'HR2': (1163, 1213), # Heptad repeat 2
            'TM': (1214, 1234), # Transmembrane domain
            'CT': (1235, 1273)  # Cytoplasmic tail
        }
        
        # Define ACE2 binding residues on spike RBD
        self.ace2_binding_residues = [
            417, 439, 446, 449, 453, 455, 456, 475, 
            476, 484, 486, 487, 489, 493, 496, 498, 
            500, 501, 502, 505
        ]
        
        # Define key mutations in variants of concern
        self.variant_mutations = {
            'Alpha': ['N501Y', 'A570D', 'D614G', 'P681H'],
            'Beta': ['K417N', 'E484K', 'N501Y', 'D614G'],
            'Gamma': ['K417T', 'E484K', 'N501Y', 'D614G'],
            'Delta': ['L452R', 'T478K', 'D614G', 'P681R'],
            'Omicron': ['G339D', 'S371L', 'S373P', 'S375F', 'K417N', 
                       'N440K', 'G446S', 'S477N', 'T478K', 'E484A', 
                       'Q493R', 'G496S', 'Q498R', 'N501Y', 'Y505H']
        }
    
    def load_spike_protein_structure(self, 
                                    structure_file: str) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load the SARS-CoV-2 Spike protein structure from a file.
        
        Args:
            structure_file: Path to the structure file (PDB format).
            
        Returns:
            Tuple of coordinates, residue names, and atom names.
        """
        try:
            # Check if BioPython is available
            import Bio.PDB
            parser = Bio.PDB.PDBParser(QUIET=True)
            structure = parser.get_structure("spike", structure_file)
            
            # Extract coordinates and residue information
            coords = []
            residues = []
            atoms = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            coords.append(atom.get_coord())
                            residues.append(residue.get_resname())
                            atoms.append(atom.get_name())
            
            return np.array(coords), residues, atoms
        
        except ImportError:
            print("BioPython not available. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "biopython"])
            
            # Retry after installation
            import Bio.PDB
            parser = Bio.PDB.PDBParser(QUIET=True)
            structure = parser.get_structure("spike", structure_file)
            
            # Extract coordinates and residue information
            coords = []
            residues = []
            atoms = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            coords.append(atom.get_coord())
                            residues.append(residue.get_resname())
                            atoms.append(atom.get_name())
            
            return np.array(coords), residues, atoms
        
        except FileNotFoundError:
            print(f"Structure file {structure_file} not found.")
            # Generate synthetic data for demonstration
            print("Generating synthetic data for demonstration...")
            
            # Generate synthetic coordinates for spike protein
            n_atoms = 5000
            coords = np.random.randn(n_atoms, 3) * 10
            
            # Generate synthetic residue names
            residue_types = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                           'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                           'THR', 'TRP', 'TYR', 'VAL']
            residues = [residue_types[i % len(residue_types)] for i in range(n_atoms)]
            
            # Generate synthetic atom names
            atom_types = ['CA', 'C', 'N', 'O', 'CB']
            atoms = [atom_types[i % len(atom_types)] for i in range(n_atoms)]
            
            return coords, residues, atoms
    
    def load_ace2_structure(self, 
                           structure_file: str) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load the ACE2 receptor structure from a file.
        
        Args:
            structure_file: Path to the structure file (PDB format).
            
        Returns:
            Tuple of coordinates, residue names, and atom names.
        """
        try:
            # Check if BioPython is available
            import Bio.PDB
            parser = Bio.PDB.PDBParser(QUIET=True)
            structure = parser.get_structure("ace2", structure_file)
            
            # Extract coordinates and residue information
            coords = []
            residues = []
            atoms = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            coords.append(atom.get_coord())
                            residues.append(residue.get_resname())
                            atoms.append(atom.get_name())
            
            return np.array(coords), residues, atoms
        
        except (ImportError, FileNotFoundError):
            print(f"Error loading ACE2 structure. Generating synthetic data for demonstration...")
            
            # Generate synthetic coordinates for ACE2
            n_atoms = 2000
            coords = np.random.randn(n_atoms, 3) * 5 + np.array([20, 0, 0])
            
            # Generate synthetic residue names
            residue_types = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                           'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                           'THR', 'TRP', 'TYR', 'VAL']
            residues = [residue_types[i % len(residue_types)] for i in range(n_atoms)]
            
            # Generate synthetic atom names
            atom_types = ['CA', 'C', 'N', 'O', 'CB']
            atoms = [atom_types[i % len(atom_types)] for i in range(n_atoms)]
            
            return coords, residues, atoms
    
    def extract_domain_coordinates(self,
                                  coords: np.ndarray,
                                  residues: List[str],
                                  atoms: List[str],
                                  domain: str) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract coordinates for a specific domain of the Spike protein.
        
        Args:
            coords: Array of protein coordinates.
            residues: List of residue names.
            atoms: List of atom names.
            domain: Domain name to extract.
            
        Returns:
            Tuple of domain coordinates, residue names, and atom names.
        """
        if domain not in self.spike_domains:
            raise ValueError(f"Unknown domain: {domain}. Available domains: {list(self.spike_domains.keys())}")
        
        # Get domain range
        start, end = self.spike_domains[domain]
        
        # Generate residue numbers (simplified approach for synthetic data)
        n_atoms = len(coords)
        n_residues = n_atoms // 5  # Assuming ~5 atoms per residue
        residue_numbers = []
        for i in range(n_residues):
            residue_numbers.extend([i+1] * 5)
        residue_numbers.extend([n_residues+1] * (n_atoms - len(residue_numbers)))
        
        # Extract domain atoms
        domain_mask = [(start <= rn <= end) for rn in residue_numbers]
        domain_coords = coords[domain_mask]
        domain_residues = [r for r, m in zip(residues, domain_mask) if m]
        domain_atoms = [a for a, m in zip(atoms, domain_mask) if m]
        
        return domain_coords, domain_residues, domain_atoms
    
    def analyze_spike_ace2_binding(self,
                                  spike_coords: np.ndarray,
                                  spike_residues: List[str],
                                  spike_atoms: List[str],
                                  ace2_coords: np.ndarray,
                                  ace2_residues: List[str],
                                  ace2_atoms: List[str]) -> Dict[str, Any]:
        """
        Analyze the binding between Spike protein and ACE2 receptor.
        
        Args:
            spike_coords: Array of Spike protein coordinates.
            spike_residues: List of Spike protein residue names.
            spike_atoms: List of Spike protein atom names.
            ace2_coords: Array of ACE2 receptor coordinates.
            ace2_residues: List of ACE2 receptor residue names.
            ace2_atoms: List of ACE2 receptor atom names.
            
        Returns:
            Dictionary of binding analysis results.
        """
        # Extract RBD domain
        try:
            rbd_coords, rbd_residues, rbd_atoms = self.extract_domain_coordinates(
                spike_coords, spike_residues, spike_atoms, 'RBD'
            )
        except ValueError:
            # Use full spike if domain extraction fails
            rbd_coords, rbd_residues, rbd_atoms = spike_coords, spike_residues, spike_atoms
        
        # Identify binding site
        binding_site_analysis = self.binding_site_mapper.analyze_binding_site(
            rbd_coords, rbd_residues, ace2_coords, cutoff=5.0
        )
        
        # Calculate binding energetics
        binding_energies = self.protein_ligand_visualizer.calculate_interaction_energies(
            rbd_coords, rbd_residues, rbd_atoms,
            ace2_coords, ace2_residues, ace2_atoms
        )
        
        # Identify interactions
        interactions = self.protein_ligand_visualizer.identify_interactions(
            rbd_coords, rbd_residues, rbd_atoms,
            ace2_coords, ace2_residues, ace2_atoms
        )
        
        # Compile results
        results = {
            'binding_site_analysis': binding_site_analysis,
            'binding_energies': binding_energies,
            'interactions': interactions,
            'rbd_coords': rbd_coords,
            'rbd_residues': rbd_residues,
            'rbd_atoms': rbd_atoms
        }
        
        return results
    
    def analyze_spike_variants(self,
                              variant: str,
                              spike_coords: np.ndarray,
                              spike_residues: List[str],
                              spike_atoms: List[str],
                              ace2_coords: np.ndarray,
                              ace2_residues: List[str],
                              ace2_atoms: List[str]) -> Dict[str, Any]:
        """
        Analyze the impact of mutations in a Spike protein variant.
        
        Args:
            variant: Variant name (e.g., 'Alpha', 'Beta', 'Delta').
            spike_coords: Array of Spike protein coordinates.
            spike_residues: List of Spike protein residue names.
            spike_atoms: List of Spike protein atom names.
            ace2_coords: Array of ACE2 receptor coordinates.
            ace2_residues: List of ACE2 receptor residue names.
            ace2_atoms: List of ACE2 receptor atom names.
            
        Returns:
            Dictionary of variant analysis results.
        """
        if variant not in self.variant_mutations:
            raise ValueError(f"Unknown variant: {variant}. Available variants: {list(self.variant_mutations.keys())}")
        
        # Get variant mutations
        mutations = self.variant_mutations[variant]
        
        # Extract mutation positions
        mutation_positions = [int(''.join(filter(str.isdigit, mut))) for mut in mutations]
        
        # Generate residue numbers (simplified approach for synthetic data)
        n_atoms = len(spike_coords)
        n_residues = n_atoms // 5  # Assuming ~5 atoms per residue
        residue_numbers = []
        for i in range(n_residues):
            residue_numbers.extend([i+1] * 5)
        residue_numbers.extend([n_residues+1] * (n_atoms - len(residue_numbers)))
        
        # Identify atoms at mutation sites
        mutation_masks = {}
        for pos in mutation_positions:
            mutation_masks[pos] = [(rn == pos) for rn in residue_numbers]
        
        # Extract coordinates at mutation sites
        mutation_coords = {}
        for pos, mask in mutation_masks.items():
            mutation_coords[pos] = spike_coords[mask]
        
        # Analyze binding with ACE2
        binding_analysis = self.analyze_spike_ace2_binding(
            spike_coords, spike_residues, spike_atoms,
            ace2_coords, ace2_residues, ace2_atoms
        )
        
        # Compile results
        results = {
            'variant': variant,
            'mutations': mutations,
            'mutation_positions': mutation_positions,
            'mutation_coords': mutation_coords,
            'binding_analysis': binding_analysis
        }
        
        return results
    
    def visualize_spike_structure(self,
                                 spike_coords: np.ndarray,
                                 spike_residues: List[str],
                                 spike_atoms: List[str],
                                 domain: Optional[str] = None,
                                 title: str = "SARS-CoV-2 Spike Protein Structure",
                                 save_path: Optional[str] = None,
                                 show: bool = True) -> plt.Figure:
        """
        Visualize the SARS-CoV-2 Spike protein structure.
        
        Args:
            spike_coords: Array of Spike protein coordinates.
            spike_residues: List of Spike protein residue names.
            spike_atoms: List of Spike protein atom names.
            domain: Optional domain to highlight.
            title: Title of the plot.
            save_path: Path to save the figure.
            show: Whether to show the figure.
            
        Returns:
            Matplotlib figure object.
        """
        # Create figure and 3D axis
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all atoms
        ax.scatter(spike_coords[:, 0], spike_coords[:, 1], spike_coords[:, 2],
                  c='lightgray', s=10, alpha=0.5)
        
        # Highlight domain if specified
        if domain:
            try:
                domain_coords, domain_residues, domain_atoms = self.extract_domain_coordinates(
                    spike_coords, spike_residues, spike_atoms, domain
                )
                
                ax.scatter(domain_coords[:, 0], domain_coords[:, 1], domain_coords[:, 2],
                          c='red', s=20, alpha=0.8, label=domain)
                
                # Update title
                title = f"{title} - {domain} Domain Highlighted"
            except ValueError as e:
                print(f"Warning: {e}")
        
        # Set plot properties
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.grid(True)
        
        if domain:
            ax.legend()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show figure if requested
        if show:
            plt.tight_layout()
            plt.show()
        
        return fig
    
    def visualize_spike_on_mobius(self,
                                 spike_coords: np.ndarray,
                                 domain: Optional[str] = None,
                                 title: str = "SARS-CoV-2 Spike Protein on Möbius Strip",
                                 save_path: Optional[str] = None,
                                 show: bool = True) -> plt.Figure:
        """
        Visualize the SARS-CoV-2 Spike protein on a Möbius strip.
        
        Args:
            spike_coords: Array of Spike protein coordinates.
            domain: Optional domain to visualize.
            title: Title of the plot.
            save_path: Path to save the figure.
            show: Whether to show the figure.
            
        Returns:
            Matplotlib figure object.
        """
        # Extract domain coordinates if specified
        if domain:
            try:
                domain_coords, domain_residues, domain_atoms = self.extract_domain_coordinates(
                    spike_coords, ['ALA'] * len(spike_coords), ['CA'] * len(spike_coords), domain
                )
                coords = domain_coords
                subtitle = f"{domain} Domain (Length: {len(domain_coords)})"
            except ValueError as e:
                print(f"Warning: {e}")
                coords = spike_coords
                subtitle = f"Full Structure (Length: {len(spike_coords)})"
        else:
            coords = spike_coords
            subtitle = f"Full Structure (Length: {len(spike_coords)})"
        
        # Subsample coordinates if too many
        max_points = 1000
        if len(coords) > max_points:
            indices = np.linspace(0, len(coords)-1, max_points, dtype=int)
            coords = coords[indices]
            subtitle += f" (Subsampled to {max_points} points)"
        
        # Visualize on Möbius strip
        fig = self.mobius_visualizer.visualize_sequence_on_mobius(
            coords.tolist(),
            sequence_properties=np.arange(len(coords)),
            title=title,
            subtitle=subtitle,
            save_path=save_path,
            show=show
        )
        
        return fig
    
    def visualize_spike_ace2_binding(self,
                                    binding_analysis: Dict[str, Any],
                                    title: str = "SARS-CoV-2 Spike-ACE2 Binding",
                                    save_path: Optional[str] = None,
                                    show: bool = True) -> plt.Figure:
        """
        Visualize the binding between Spike protein and ACE2 receptor.
        
        Args:
            binding_analysis: Results from analyze_spike_ace2_binding.
            title: Title of the plot.
            save_path: Path to save the figure.
            show: Whether to show the figure.
            
        Returns:
            Matplotlib figure object.
        """
        # Extract data from binding analysis
        rbd_coords = binding_analysis['rbd_coords']
        rbd_residues = binding_analysis['rbd_residues']
        rbd_atoms = binding_analysis['rbd_atoms']
        binding_site = binding_analysis['binding_site_analysis']['binding_site_coords']
        binding_energies = binding_analysis['binding_energies']
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot RBD atoms
        ax.scatter(rbd_coords[:, 0], rbd_coords[:, 1], rbd_coords[:, 2],
                  c='lightgray', s=10, alpha=0.3, label='RBD')
        
        # Plot binding site atoms
        if len(binding_site) > 0:
            ax.scatter(binding_site[:, 0], binding_site[:, 1], binding_site[:, 2],
                      c='red', s=30, alpha=0.8, label='Binding Site')
        
        # Set plot properties
        energy_text = f"Total Binding Energy: {binding_energies['total']:.2f} kcal/mol"
        full_title = f"{title}\n{energy_text}"
        ax.set_title(full_title, fontsize=14)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.grid(True)
        ax.legend()
        
        # Add energy breakdown text
        energy_breakdown = (
            f"Energy Components (kcal/mol):\n"
            f"Electrostatic: {binding_energies['electrostatic']:.2f}\n"
            f"Van der Waals: {binding_energies['van_der_waals']:.2f}\n"
            f"Hydrogen Bond: {binding_energies['hydrogen_bond']:.2f}\n"
            f"Hydrophobic: {binding_energies['hydrophobic']:.2f}\n"
            f"Total: {binding_energies['total']:.2f}"
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
    
    def visualize_spike_ace2_binding_on_mobius(self,
                                             binding_analysis: Dict[str, Any],
                                             title: str = "SARS-CoV-2 Spike-ACE2 Binding on Möbius Strip",
                                             save_path: Optional[str] = None,
                                             show: bool = True) -> plt.Figure:
        """
        Visualize the binding between Spike protein and ACE2 receptor on a Möbius strip.
        
        Args:
            binding_analysis: Results from analyze_spike_ace2_binding.
            title: Title of the plot.
            save_path: Path to save the figure.
            show: Whether to show the figure.
            
        Returns:
            Matplotlib figure object.
        """
        # Extract binding site analysis
        binding_site_analysis = binding_analysis['binding_site_analysis']
        
        # Visualize binding site on Möbius strip
        fig = self.binding_site_mapper.visualize_binding_site_on_mobius(
            binding_site_analysis,
            title=title,
            save_path=save_path,
            show=show
        )
        
        return fig
    
    def visualize_spike_variant_comparison(self,
                                         variant_analyses: Dict[str, Dict[str, Any]],
                                         title: str = "SARS-CoV-2 Spike Variant Comparison",
                                         save_path: Optional[str] = None,
                                         show: bool = True) -> plt.Figure:
        """
        Visualize a comparison of different Spike protein variants.
        
        Args:
            variant_analyses: Dictionary of variant analysis results.
            title: Title of the plot.
            save_path: Path to save the figure.
            show: Whether to show the figure.
            
        Returns:
            Matplotlib figure object.
        """
        # Extract binding energies for each variant
        variants = list(variant_analyses.keys())
        binding_energies = {v: variant_analyses[v]['binding_analysis']['binding_energies']['total'] 
                          for v in variants}
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot binding energies
        bars = ax.bar(variants, [binding_energies[v] for v in variants])
        
        # Color bars based on binding energy (more negative = stronger binding)
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(variants)))
        for i, bar in enumerate(bars):
            bar.set_color(colors[i])
        
        # Set plot properties
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Variant')
        ax.set_ylabel('Binding Energy (kcal/mol)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add mutation information
        for i, variant in enumerate(variants):
            mutations = variant_analyses[variant]['mutations']
            mutation_text = ', '.join(mutations)
            ax.annotate(f"{mutation_text}", 
                       xy=(i, binding_energies[variant]),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       va='bottom',
                       fontsize=8,
                       rotation=90)
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show figure if requested
        if show:
            plt.tight_layout()
            plt.show()
        
        return fig
    
    def run_spike_protein_case_study(self,
                                    spike_structure_file: Optional[str] = None,
                                    ace2_structure_file: Optional[str] = None,
                                    output_dir: str = "spike_case_study",
                                    variants: List[str] = ['Alpha', 'Beta', 'Delta', 'Omicron']) -> Dict[str, Any]:
        """
        Run a comprehensive case study of the SARS-CoV-2 Spike protein.
        
        Args:
            spike_structure_file: Path to the Spike protein structure file.
            ace2_structure_file: Path to the ACE2 receptor structure file.
            output_dir: Directory to save output files.
            variants: List of variants to analyze.
            
        Returns:
            Dictionary of case study results.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load structures
        print("Loading Spike protein structure...")
        spike_coords, spike_residues, spike_atoms = self.load_spike_protein_structure(
            spike_structure_file if spike_structure_file else "spike.pdb"
        )
        
        print("Loading ACE2 receptor structure...")
        ace2_coords, ace2_residues, ace2_atoms = self.load_ace2_structure(
            ace2_structure_file if ace2_structure_file else "ace2.pdb"
        )
        
        # Analyze wild-type Spike-ACE2 binding
        print("Analyzing wild-type Spike-ACE2 binding...")
        wt_binding_analysis = self.analyze_spike_ace2_binding(
            spike_coords, spike_residues, spike_atoms,
            ace2_coords, ace2_residues, ace2_atoms
        )
        
        # Analyze variants
        print("Analyzing Spike protein variants...")
        variant_analyses = {}
        for variant in variants:
            print(f"  Analyzing {variant} variant...")
            try:
                variant_analysis = self.analyze_spike_variants(
                    variant,
                    spike_coords, spike_residues, spike_atoms,
                    ace2_coords, ace2_residues, ace2_atoms
                )
                variant_analyses[variant] = variant_analysis
            except ValueError as e:
                print(f"  Error analyzing {variant}: {e}")
        
        # Generate visualizations
        print("Generating visualizations...")
        
        # Visualize Spike protein structure
        print("  Visualizing Spike protein structure...")
        spike_structure_fig = self.visualize_spike_structure(
            spike_coords, spike_residues, spike_atoms,
            domain='RBD',
            save_path=os.path.join(output_dir, "spike_structure.png")
        )
        
        # Visualize Spike protein on Möbius strip
        print("  Visualizing Spike protein on Möbius strip...")
        spike_mobius_fig = self.visualize_spike_on_mobius(
            spike_coords,
            domain='RBD',
            save_path=os.path.join(output_dir, "spike_mobius.png")
        )
        
        # Visualize Spike-ACE2 binding
        print("  Visualizing Spike-ACE2 binding...")
        binding_fig = self.visualize_spike_ace2_binding(
            wt_binding_analysis,
            save_path=os.path.join(output_dir, "spike_ace2_binding.png")
        )
        
        # Visualize Spike-ACE2 binding on Möbius strip
        print("  Visualizing Spike-ACE2 binding on Möbius strip...")
        binding_mobius_fig = self.visualize_spike_ace2_binding_on_mobius(
            wt_binding_analysis,
            save_path=os.path.join(output_dir, "spike_ace2_binding_mobius.png")
        )
        
        # Visualize variant comparison
        if variant_analyses:
            print("  Visualizing variant comparison...")
            variant_comparison_fig = self.visualize_spike_variant_comparison(
                variant_analyses,
                save_path=os.path.join(output_dir, "spike_variant_comparison.png")
            )
        
        # Compile results
        results = {
            'spike_coords': spike_coords,
            'spike_residues': spike_residues,
            'spike_atoms': spike_atoms,
            'ace2_coords': ace2_coords,
            'ace2_residues': ace2_residues,
            'ace2_atoms': ace2_atoms,
            'wt_binding_analysis': wt_binding_analysis,
            'variant_analyses': variant_analyses,
            'output_dir': output_dir
        }
        
        print(f"Case study completed. Results saved to {output_dir}")
        
        return results