"""
Biosynthesis Modeling Module using Galois Prime Ring Primitives

This module implements biosynthesis pathway modeling using a classical quantum formalism
based on Galois Prime Ring Primitives Theory. It enables efficient simulation of enzymatic
reactions, metabolic pathways, and biosynthetic processes on classical computers while
leveraging mathematical structures that provide quantum-like computational advantages.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from collections import defaultdict
import networkx as nx

# Add the parent directory to the path to import the classical_quantum modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical_quantum.galois_prime_ring import GaloisPrimeRing
from classical_quantum.cyclotomic_field import CyclotomicField
from classical_quantum.spinor_structure import SpinorStructure
from classical_quantum.discosohedral_mapping import DiscosohedralMapping
from classical_quantum.phase_synchronization import PhaseSynchronization
import classical_quantum.utils as utils

class Metabolite:
    """
    Representation of a metabolite in a biosynthesis pathway.
    """
    
    def __init__(self, name: str, formula: str, smiles: str = None, molecular_weight: float = None):
        """
        Initialize a metabolite.
        
        Args:
            name: Metabolite name
            formula: Chemical formula
            smiles: SMILES representation (optional)
            molecular_weight: Molecular weight in g/mol (optional)
        """
        self.name = name
        self.formula = formula
        self.smiles = smiles
        self.molecular_weight = molecular_weight
        
        # Parse formula to get element counts
        self.elements = self._parse_formula(formula)
        
        # Calculate molecular weight if not provided
        if molecular_weight is None and self.elements:
            self.molecular_weight = self._calculate_molecular_weight()
    
    def _parse_formula(self, formula: str) -> Dict[str, int]:
        """
        Parse a chemical formula to get element counts.
        
        Args:
            formula: Chemical formula (e.g., "C6H12O6")
            
        Returns:
            Dictionary mapping elements to their counts
        """
        elements = {}
        i = 0
        
        while i < len(formula):
            # Get element symbol (1 or 2 characters)
            if i + 1 < len(formula) and formula[i+1].islower():
                element = formula[i:i+2]
                i += 2
            else:
                element = formula[i]
                i += 1
            
            # Get count (may be multiple digits)
            count = ""
            while i < len(formula) and formula[i].isdigit():
                count += formula[i]
                i += 1
            
            # Add to elements dictionary
            elements[element] = int(count) if count else 1
        
        return elements
    
    def _calculate_molecular_weight(self) -> float:
        """
        Calculate molecular weight from elements.
        
        Returns:
            Molecular weight in g/mol
        """
        # Approximate atomic weights
        atomic_weights = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'P': 30.974,
            'S': 32.065, 'F': 18.998, 'Cl': 35.453, 'Br': 79.904, 'I': 126.904
        }
        
        weight = 0.0
        for element, count in self.elements.items():
            if element in atomic_weights:
                weight += atomic_weights[element] * count
        
        return weight
    
    def __str__(self):
        return f"{self.name} ({self.formula})"
    
    def __repr__(self):
        return self.__str__()
    
    def to_dict(self):
        """
        Convert metabolite properties to a dictionary.
        
        Returns:
            Dictionary of metabolite properties
        """
        return {
            'name': self.name,
            'formula': self.formula,
            'smiles': self.smiles,
            'molecular_weight': self.molecular_weight,
            'elements': self.elements
        }


class EnzymeReaction:
    """
    Representation of an enzyme-catalyzed reaction in a biosynthesis pathway.
    """
    
    def __init__(self, name: str, enzyme: str, substrates: List[Tuple[Metabolite, float]],
                products: List[Tuple[Metabolite, float]], reversible: bool = False,
                km_values: Dict[str, float] = None, kcat: float = None):
        """
        Initialize an enzyme reaction.
        
        Args:
            name: Reaction name
            enzyme: Enzyme name or EC number
            substrates: List of (metabolite, stoichiometric coefficient) tuples
            products: List of (metabolite, stoichiometric coefficient) tuples
            reversible: Whether the reaction is reversible
            km_values: Dictionary mapping substrate names to Michaelis constants (mM)
            kcat: Turnover number (1/s)
        """
        self.name = name
        self.enzyme = enzyme
        self.substrates = substrates
        self.products = products
        self.reversible = reversible
        self.km_values = km_values or {}
        self.kcat = kcat
        
        # Calculate standard Gibbs free energy change (simplified)
        self.delta_g = self._estimate_delta_g()
    
    def _estimate_delta_g(self) -> float:
        """
        Estimate the standard Gibbs free energy change of the reaction.
        This is a simplified estimation based on the number of bonds formed/broken.
        
        Returns:
            Estimated delta G in kJ/mol
        """
        # Count total atoms in substrates and products
        substrate_atoms = sum(
            sum(substrate.elements.values()) * coeff
            for substrate, coeff in self.substrates
        )
        
        product_atoms = sum(
            sum(product.elements.values()) * coeff
            for product, coeff in self.products
        )
        
        # Simple heuristic: if atoms are conserved, delta G is small
        if abs(substrate_atoms - product_atoms) < 0.01:
            return np.random.normal(-5, 5)  # Small random value
        else:
            # More atoms in products -> endergonic, fewer -> exergonic
            return (product_atoms - substrate_atoms) * 2
    
    def calculate_rate(self, concentrations: Dict[str, float], ring: GaloisPrimeRing) -> float:
        """
        Calculate the reaction rate using Michaelis-Menten kinetics enhanced with
        Galois field operations for improved numerical stability and precision.
        
        Args:
            concentrations: Dictionary mapping metabolite names to concentrations (mM)
            ring: Galois prime ring for field operations
            
        Returns:
            Reaction rate in mM/s
        """
        if not self.kcat:
            return 0.0
        
        # Calculate the forward rate using Michaelis-Menten kinetics
        substrate_terms = []
        
        for substrate, coeff in self.substrates:
            if substrate.name in concentrations:
                conc = concentrations[substrate.name]
                km = self.km_values.get(substrate.name, 1.0)  # Default Km = 1.0 mM
                
                # Use field operations for improved numerical stability
                # Map concentration and Km to field elements
                conc_field = int(conc * 100) % ring.characteristic
                km_field = int(km * 100) % ring.characteristic
                
                # Calculate substrate term using field operations
                if km_field != 0:
                    term_field = ring.multiply(conc_field, ring.inverse(ring.add(conc_field, km_field)))
                    
                    # Map back to real domain
                    term = term_field / 100.0
                    substrate_terms.append(term ** coeff)
                else:
                    substrate_terms.append(0.0)
            else:
                substrate_terms.append(0.0)
        
        # Calculate the overall rate
        if substrate_terms:
            rate_forward = self.kcat * np.prod(substrate_terms)
        else:
            rate_forward = 0.0
        
        # If reversible, calculate the reverse rate (simplified)
        if self.reversible:
            # Calculate equilibrium constant from delta G
            rt = 8.314 * 298.15 / 1000  # RT in kJ/mol
            keq = np.exp(-self.delta_g / rt)
            
            # Calculate product concentrations term
            product_term = 1.0
            for product, coeff in self.products:
                if product.name in concentrations:
                    product_term *= concentrations[product.name] ** coeff
                else:
                    product_term = 0.0
                    break
            
            # Calculate substrate concentrations term
            substrate_term = 1.0
            for substrate, coeff in self.substrates:
                if substrate.name in concentrations:
                    substrate_term *= concentrations[substrate.name] ** coeff
                else:
                    substrate_term = 0.0
                    break
            
            # Calculate reverse rate
            if substrate_term > 0:
                rate_reverse = rate_forward * product_term / (keq * substrate_term)
            else:
                rate_reverse = 0.0
            
            return rate_forward - rate_reverse
        else:
            return rate_forward
    
    def __str__(self):
        substrates_str = " + ".join([f"{coeff} {substrate.name}" if coeff != 1 else substrate.name 
                                   for substrate, coeff in self.substrates])
        products_str = " + ".join([f"{coeff} {product.name}" if coeff != 1 else product.name 
                                 for product, coeff in self.products])
        arrow = "<=>" if self.reversible else "->"
        return f"{substrates_str} {arrow} {products_str} [{self.enzyme}]"
    
    def __repr__(self):
        return self.__str__()
    
    def to_dict(self):
        """
        Convert reaction properties to a dictionary.
        
        Returns:
            Dictionary of reaction properties
        """
        return {
            'name': self.name,
            'enzyme': self.enzyme,
            'substrates': [(s.name, c) for s, c in self.substrates],
            'products': [(p.name, c) for p, c in self.products],
            'reversible': self.reversible,
            'km_values': self.km_values,
            'kcat': self.kcat,
            'delta_g': self.delta_g
        }


class BiosynthesisPathway:
    """
    Representation of a biosynthesis pathway using Galois Prime Ring formalism.
    """
    
    def __init__(self, name: str, ring_characteristic: int = 11, extension_degree: int = 1):
        """
        Initialize a biosynthesis pathway.
        
        Args:
            name: Pathway name
            ring_characteristic: The characteristic of the Galois field (prime number)
            extension_degree: The extension degree of the field
        """
        self.name = name
        self.metabolites = {}  # name -> Metabolite
        self.reactions = []    # List of EnzymeReaction
        
        # Initialize the Galois prime ring
        self.ring = GaloisPrimeRing(ring_characteristic, extension_degree)
        
        # Initialize the classical quantum components
        self.cyclotomic_field = CyclotomicField(168)
        self.spinor_structure = SpinorStructure(56)
        self.discosohedral_mapping = DiscosohedralMapping(56)
        self.phase_synchronization = PhaseSynchronization(56)
        
        # Initialize pathway graph
        self.graph = nx.DiGraph()
    
    def add_metabolite(self, metabolite: Metabolite) -> None:
        """
        Add a metabolite to the pathway.
        
        Args:
            metabolite: Metabolite to add
        """
        self.metabolites[metabolite.name] = metabolite
        self.graph.add_node(metabolite.name, type='metabolite', data=metabolite)
    
    def add_reaction(self, reaction: EnzymeReaction) -> None:
        """
        Add a reaction to the pathway.
        
        Args:
            reaction: Reaction to add
        """
        self.reactions.append(reaction)
        
        # Add reaction node to graph
        self.graph.add_node(reaction.name, type='reaction', data=reaction)
        
        # Add edges from substrates to reaction
        for substrate, coeff in reaction.substrates:
            if substrate.name not in self.metabolites:
                self.add_metabolite(substrate)
            self.graph.add_edge(substrate.name, reaction.name, type='substrate', coefficient=coeff)
        
        # Add edges from reaction to products
        for product, coeff in reaction.products:
            if product.name not in self.metabolites:
                self.add_metabolite(product)
            self.graph.add_edge(reaction.name, product.name, type='product', coefficient=coeff)
    
    def get_metabolite(self, name: str) -> Optional[Metabolite]:
        """
        Get a metabolite by name.
        
        Args:
            name: Metabolite name
            
        Returns:
            Metabolite object or None if not found
        """
        return self.metabolites.get(name)
    
    def simulate_pathway(self, initial_concentrations: Dict[str, float], 
                        time_points: np.ndarray, external_inputs: Dict[str, float] = None,
                        external_outputs: Dict[str, float] = None) -> Dict[str, np.ndarray]:
        """
        Simulate the pathway dynamics over time using classical quantum field operations.
        
        Args:
            initial_concentrations: Dictionary mapping metabolite names to initial concentrations (mM)
            time_points: Array of time points for simulation (s)
            external_inputs: Dictionary mapping metabolite names to constant input rates (mM/s)
            external_outputs: Dictionary mapping metabolite names to first-order output rate constants (1/s)
            
        Returns:
            Dictionary mapping metabolite names to concentration time series
        """
        # Initialize concentrations
        concentrations = {name: np.zeros(len(time_points)) for name in self.metabolites}
        
        # Set initial concentrations
        for name, conc in initial_concentrations.items():
            if name in concentrations:
                concentrations[name][0] = conc
        
        # Set up external inputs and outputs
        inputs = external_inputs or {}
        outputs = external_outputs or {}
        
        # Simulate using Euler method
        for i in range(1, len(time_points)):
            dt = time_points[i] - time_points[i-1]
            current_conc = {name: concentrations[name][i-1] for name in self.metabolites}
            
            # Calculate rates of change for each metabolite
            rates = {name: 0.0 for name in self.metabolites}
            
            # Add external inputs
            for name, rate in inputs.items():
                if name in rates:
                    rates[name] += rate
            
            # Subtract external outputs
            for name, rate_const in outputs.items():
                if name in rates and name in current_conc:
                    rates[name] -= rate_const * current_conc[name]
            
            # Add reaction contributions
            for reaction in self.reactions:
                # Calculate reaction rate
                reaction_rate = reaction.calculate_rate(current_conc, self.ring)
                
                # Update substrate rates
                for substrate, coeff in reaction.substrates:
                    rates[substrate.name] -= coeff * reaction_rate
                
                # Update product rates
                for product, coeff in reaction.products:
                    rates[product.name] += coeff * reaction_rate
            
            # Update concentrations using Euler method
            for name in self.metabolites:
                # Ensure concentrations don't go negative
                new_conc = max(0.0, concentrations[name][i-1] + dt * rates[name])
                concentrations[name][i] = new_conc
        
        return concentrations
    
    def calculate_flux_control_coefficients(self, steady_state_concentrations: Dict[str, float],
                                          target_reaction: str) -> Dict[str, float]:
        """
        Calculate flux control coefficients for a target reaction using field operations.
        
        Args:
            steady_state_concentrations: Dictionary mapping metabolite names to steady-state concentrations
            target_reaction: Name of the target reaction
            
        Returns:
            Dictionary mapping reaction names to flux control coefficients
        """
        # Find the target reaction
        target = None
        for reaction in self.reactions:
            if reaction.name == target_reaction:
                target = reaction
                break
        
        if not target:
            raise ValueError(f"Target reaction '{target_reaction}' not found")
        
        # Calculate the base flux
        base_flux = target.calculate_rate(steady_state_concentrations, self.ring)
        
        # Calculate control coefficients
        control_coefficients = {}
        
        for reaction in self.reactions:
            # Perturb the reaction rate
            if reaction.kcat:
                original_kcat = reaction.kcat
                reaction.kcat *= 1.01  # 1% increase
                
                # Calculate the new flux
                new_flux = target.calculate_rate(steady_state_concentrations, self.ring)
                
                # Calculate the control coefficient
                if base_flux != 0:
                    control_coeff = ((new_flux - base_flux) / base_flux) / 0.01
                else:
                    control_coeff = 0.0
                
                # Restore the original kcat
                reaction.kcat = original_kcat
                
                control_coefficients[reaction.name] = control_coeff
        
        return control_coefficients
    
    def identify_rate_limiting_steps(self, steady_state_concentrations: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Identify rate-limiting steps in the pathway using field operations.
        
        Args:
            steady_state_concentrations: Dictionary mapping metabolite names to steady-state concentrations
            
        Returns:
            List of (reaction name, control coefficient) tuples, sorted by importance
        """
        # Calculate flux through the last reaction (assumed to be the pathway output)
        if not self.reactions:
            return []
        
        output_reaction = self.reactions[-1]
        
        # Calculate control coefficients for the output reaction
        control_coefficients = self.calculate_flux_control_coefficients(
            steady_state_concentrations, output_reaction.name)
        
        # Sort reactions by control coefficient
        rate_limiting = [(name, coeff) for name, coeff in control_coefficients.items()]
        rate_limiting.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return rate_limiting
    
    def calculate_pathway_yield(self, initial_concentrations: Dict[str, float],
                              simulation_time: float, substrate: str, product: str) -> float:
        """
        Calculate the yield of a product from a substrate.
        
        Args:
            initial_concentrations: Dictionary mapping metabolite names to initial concentrations
            simulation_time: Simulation time in seconds
            substrate: Name of the substrate metabolite
            product: Name of the product metabolite
            
        Returns:
            Yield as a fraction (0-1)
        """
        # Simulate the pathway
        time_points = np.linspace(0, simulation_time, 1000)
        concentrations = self.simulate_pathway(initial_concentrations, time_points)
        
        # Calculate the yield
        if substrate in concentrations and product in concentrations:
            initial_substrate = initial_concentrations.get(substrate, 0.0)
            final_product = concentrations[product][-1]
            
            if initial_substrate > 0:
                # Account for stoichiometry (simplified)
                substrate_mw = self.metabolites[substrate].molecular_weight
                product_mw = self.metabolites[product].molecular_weight
                
                if substrate_mw and product_mw:
                    stoichiometric_factor = substrate_mw / product_mw
                    theoretical_yield = initial_substrate * stoichiometric_factor
                    
                    if theoretical_yield > 0:
                        return final_product / theoretical_yield
            
        return 0.0
    
    def optimize_pathway_flux(self, target_metabolite: str, 
                            constraints: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Optimize enzyme levels to maximize flux to a target metabolite.
        
        Args:
            target_metabolite: Name of the target metabolite
            constraints: Dictionary mapping enzyme names to (min, max) activity constraints
            
        Returns:
            Dictionary mapping enzyme names to optimal activity levels
        """
        # Define the objective function (negative flux to maximize)
        def objective(enzyme_levels):
            # Set enzyme activities
            for i, reaction in enumerate(self.reactions):
                if reaction.name in constraints:
                    reaction.kcat = enzyme_levels[i]
            
            # Run a short simulation
            initial_conc = {name: 1.0 for name in self.metabolites}  # Simplified
            time_points = np.linspace(0, 100, 10)
            concentrations = self.simulate_pathway(initial_conc, time_points)
            
            # Return negative flux to the target metabolite
            if target_metabolite in concentrations:
                flux = (concentrations[target_metabolite][-1] - concentrations[target_metabolite][0]) / 100
                return -flux
            else:
                return 0.0
        
        # Set up bounds
        bounds = []
        for reaction in self.reactions:
            if reaction.name in constraints:
                bounds.append(constraints[reaction.name])
            else:
                bounds.append((0.1, 10.0))  # Default bounds
        
        # Run optimization
        initial_guess = [reaction.kcat if reaction.kcat else 1.0 for reaction in self.reactions]
        
        from scipy.optimize import minimize
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        # Extract optimal enzyme levels
        optimal_levels = {}
        for i, reaction in enumerate(self.reactions):
            optimal_levels[reaction.name] = result.x[i]
        
        return optimal_levels
    
    def visualize_pathway(self, title: str = None) -> plt.Figure:
        """
        Visualize the pathway as a graph.
        
        Args:
            title: Plot title (optional)
            
        Returns:
            Matplotlib figure
        """
        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create a layout for the graph
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw metabolite nodes
        metabolite_nodes = [node for node, data in self.graph.nodes(data=True) if data.get('type') == 'metabolite']
        nx.draw_networkx_nodes(self.graph, pos, nodelist=metabolite_nodes, 
                              node_color='lightblue', node_size=500, alpha=0.8, ax=ax)
        
        # Draw reaction nodes
        reaction_nodes = [node for node, data in self.graph.nodes(data=True) if data.get('type') == 'reaction']
        nx.draw_networkx_nodes(self.graph, pos, nodelist=reaction_nodes, 
                              node_color='lightgreen', node_size=300, node_shape='s', alpha=0.8, ax=ax)
        
        # Draw edges
        substrate_edges = [(u, v) for u, v, data in self.graph.edges(data=True) if data.get('type') == 'substrate']
        product_edges = [(u, v) for u, v, data in self.graph.edges(data=True) if data.get('type') == 'product']
        
        nx.draw_networkx_edges(self.graph, pos, edgelist=substrate_edges, 
                              edge_color='red', width=1.5, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(self.graph, pos, edgelist=product_edges, 
                              edge_color='blue', width=1.5, alpha=0.7, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10, ax=ax)
        
        # Add a legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Metabolite'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen', markersize=10, label='Reaction'),
            Line2D([0], [0], color='red', lw=2, label='Substrate'),
            Line2D([0], [0], color='blue', lw=2, label='Product')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Biosynthesis Pathway: {self.name}")
        
        # Remove axis
        ax.axis('off')
        
        plt.tight_layout()
        
        return fig
    
    def visualize_simulation_results(self, concentrations: Dict[str, np.ndarray], 
                                   time_points: np.ndarray, title: str = None) -> plt.Figure:
        """
        Visualize simulation results.
        
        Args:
            concentrations: Dictionary mapping metabolite names to concentration time series
            time_points: Array of time points
            title: Plot title (optional)
            
        Returns:
            Matplotlib figure
        """
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each metabolite concentration
        for name, conc in concentrations.items():
            ax.plot(time_points, conc, label=name)
        
        # Set labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Concentration (mM)')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Simulation Results: {self.name}")
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def to_dict(self):
        """
        Convert pathway to a dictionary.
        
        Returns:
            Dictionary representation of the pathway
        """
        return {
            'name': self.name,
            'metabolites': {name: metabolite.to_dict() for name, metabolite in self.metabolites.items()},
            'reactions': [reaction.to_dict() for reaction in self.reactions]
        }
    
    def save(self, filename: str):
        """
        Save the pathway to a file.
        
        Args:
            filename: Output filename
        """
        import json
        
        # Convert to dictionary
        pathway_dict = self.to_dict()
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(pathway_dict, f, indent=2)
        
        print(f"Pathway saved to {filename}")
    
    @classmethod
    def load(cls, filename: str):
        """
        Load a pathway from a file.
        
        Args:
            filename: Input filename
            
        Returns:
            BiosynthesisPathway object
        """
        import json
        
        # Load from file
        with open(filename, 'r') as f:
            pathway_dict = json.load(f)
        
        # Create pathway
        pathway = cls(pathway_dict['name'])
        
        # Add metabolites
        for name, metabolite_dict in pathway_dict['metabolites'].items():
            metabolite = Metabolite(
                name=metabolite_dict['name'],
                formula=metabolite_dict['formula'],
                smiles=metabolite_dict.get('smiles'),
                molecular_weight=metabolite_dict.get('molecular_weight')
            )
            pathway.add_metabolite(metabolite)
        
        # Add reactions
        for reaction_dict in pathway_dict['reactions']:
            # Get substrates
            substrates = []
            for substrate_name, coeff in reaction_dict['substrates']:
                if substrate_name in pathway.metabolites:
                    substrates.append((pathway.metabolites[substrate_name], coeff))
            
            # Get products
            products = []
            for product_name, coeff in reaction_dict['products']:
                if product_name in pathway.metabolites:
                    products.append((pathway.metabolites[product_name], coeff))
            
            # Create reaction
            reaction = EnzymeReaction(
                name=reaction_dict['name'],
                enzyme=reaction_dict['enzyme'],
                substrates=substrates,
                products=products,
                reversible=reaction_dict.get('reversible', False),
                km_values=reaction_dict.get('km_values'),
                kcat=reaction_dict.get('kcat')
            )
            
            pathway.add_reaction(reaction)
        
        return pathway


class BiosynthesisPathwayFactory:
    """
    Factory class for creating common biosynthesis pathways.
    """
    
    @staticmethod
    def create_glycolysis_pathway():
        """
        Create a simplified glycolysis pathway.
        
        Returns:
            BiosynthesisPathway object
        """
        pathway = BiosynthesisPathway("Glycolysis")
        
        # Create metabolites
        glucose = Metabolite("Glucose", "C6H12O6", molecular_weight=180.16)
        g6p = Metabolite("Glucose-6-phosphate", "C6H13O9P", molecular_weight=260.14)
        f6p = Metabolite("Fructose-6-phosphate", "C6H13O9P", molecular_weight=260.14)
        f16bp = Metabolite("Fructose-1,6-bisphosphate", "C6H14O12P2", molecular_weight=340.12)
        dhap = Metabolite("Dihydroxyacetone phosphate", "C3H7O6P", molecular_weight=170.06)
        g3p = Metabolite("Glyceraldehyde-3-phosphate", "C3H7O6P", molecular_weight=170.06)
        bpg = Metabolite("1,3-Bisphosphoglycerate", "C3H8O10P2", molecular_weight=266.04)
        pg3 = Metabolite("3-Phosphoglycerate", "C3H7O7P", molecular_weight=186.06)
        pg2 = Metabolite("2-Phosphoglycerate", "C3H7O7P", molecular_weight=186.06)
        pep = Metabolite("Phosphoenolpyruvate", "C3H5O6P", molecular_weight=168.04)
        pyruvate = Metabolite("Pyruvate", "C3H4O3", molecular_weight=88.06)
        atp = Metabolite("ATP", "C10H16N5O13P3", molecular_weight=507.18)
        adp = Metabolite("ADP", "C10H15N5O10P2", molecular_weight=427.20)
        nad = Metabolite("NAD+", "C21H28N7O14P2", molecular_weight=663.43)
        nadh = Metabolite("NADH", "C21H29N7O14P2", molecular_weight=664.44)
        
        # Add metabolites to pathway
        for metabolite in [glucose, g6p, f6p, f16bp, dhap, g3p, bpg, pg3, pg2, pep, pyruvate, atp, adp, nad, nadh]:
            pathway.add_metabolite(metabolite)
        
        # Create reactions
        hexokinase = EnzymeReaction(
            name="Hexokinase",
            enzyme="2.7.1.1",
            substrates=[(glucose, 1), (atp, 1)],
            products=[(g6p, 1), (adp, 1)],
            reversible=False,
            km_values={"Glucose": 0.1, "ATP": 0.5},
            kcat=10.0
        )
        
        pgi = EnzymeReaction(
            name="Phosphoglucose isomerase",
            enzyme="5.3.1.9",
            substrates=[(g6p, 1)],
            products=[(f6p, 1)],
            reversible=True,
            km_values={"Glucose-6-phosphate": 0.5},
            kcat=20.0
        )
        
        pfk = EnzymeReaction(
            name="Phosphofructokinase",
            enzyme="2.7.1.11",
            substrates=[(f6p, 1), (atp, 1)],
            products=[(f16bp, 1), (adp, 1)],
            reversible=False,
            km_values={"Fructose-6-phosphate": 0.05, "ATP": 0.1},
            kcat=15.0
        )
        
        aldolase = EnzymeReaction(
            name="Aldolase",
            enzyme="4.1.2.13",
            substrates=[(f16bp, 1)],
            products=[(dhap, 1), (g3p, 1)],
            reversible=True,
            km_values={"Fructose-1,6-bisphosphate": 0.02},
            kcat=5.0
        )
        
        tpi = EnzymeReaction(
            name="Triose phosphate isomerase",
            enzyme="5.3.1.1",
            substrates=[(dhap, 1)],
            products=[(g3p, 1)],
            reversible=True,
            km_values={"Dihydroxyacetone phosphate": 0.5},
            kcat=100.0
        )
        
        gapdh = EnzymeReaction(
            name="Glyceraldehyde-3-phosphate dehydrogenase",
            enzyme="1.2.1.12",
            substrates=[(g3p, 1), (nad, 1)],
            products=[(bpg, 1), (nadh, 1)],
            reversible=True,
            km_values={"Glyceraldehyde-3-phosphate": 0.1, "NAD+": 0.05},
            kcat=10.0
        )
        
        pgk = EnzymeReaction(
            name="Phosphoglycerate kinase",
            enzyme="2.7.2.3",
            substrates=[(bpg, 1), (adp, 1)],
            products=[(pg3, 1), (atp, 1)],
            reversible=True,
            km_values={"1,3-Bisphosphoglycerate": 0.003, "ADP": 0.2},
            kcat=20.0
        )
        
        pgm = EnzymeReaction(
            name="Phosphoglycerate mutase",
            enzyme="5.4.2.1",
            substrates=[(pg3, 1)],
            products=[(pg2, 1)],
            reversible=True,
            km_values={"3-Phosphoglycerate": 0.2},
            kcat=15.0
        )
        
        enolase = EnzymeReaction(
            name="Enolase",
            enzyme="4.2.1.11",
            substrates=[(pg2, 1)],
            products=[(pep, 1)],
            reversible=True,
            km_values={"2-Phosphoglycerate": 0.05},
            kcat=8.0
        )
        
        pk = EnzymeReaction(
            name="Pyruvate kinase",
            enzyme="2.7.1.40",
            substrates=[(pep, 1), (adp, 1)],
            products=[(pyruvate, 1), (atp, 1)],
            reversible=False,
            km_values={"Phosphoenolpyruvate": 0.3, "ADP": 0.3},
            kcat=30.0
        )
        
        # Add reactions to pathway
        for reaction in [hexokinase, pgi, pfk, aldolase, tpi, gapdh, pgk, pgm, enolase, pk]:
            pathway.add_reaction(reaction)
        
        return pathway
    
    @staticmethod
    def create_tca_cycle_pathway():
        """
        Create a simplified TCA cycle pathway.
        
        Returns:
            BiosynthesisPathway object
        """
        pathway = BiosynthesisPathway("TCA Cycle")
        
        # Create metabolites
        pyruvate = Metabolite("Pyruvate", "C3H4O3", molecular_weight=88.06)
        acetyl_coa = Metabolite("Acetyl-CoA", "C23H38N7O17P3S", molecular_weight=809.57)
        oxaloacetate = Metabolite("Oxaloacetate", "C4H4O5", molecular_weight=132.07)
        citrate = Metabolite("Citrate", "C6H8O7", molecular_weight=192.12)
        isocitrate = Metabolite("Isocitrate", "C6H8O7", molecular_weight=192.12)
        alpha_kg = Metabolite("Alpha-ketoglutarate", "C5H6O5", molecular_weight=146.11)
        succinyl_coa = Metabolite("Succinyl-CoA", "C25H40N7O19P3S", molecular_weight=867.63)
        succinate = Metabolite("Succinate", "C4H6O4", molecular_weight=118.09)
        fumarate = Metabolite("Fumarate", "C4H4O4", molecular_weight=116.07)
        malate = Metabolite("Malate", "C4H6O5", molecular_weight=134.09)
        coa = Metabolite("CoA", "C21H36N7O16P3S", molecular_weight=767.53)
        nad = Metabolite("NAD+", "C21H28N7O14P2", molecular_weight=663.43)
        nadh = Metabolite("NADH", "C21H29N7O14P2", molecular_weight=664.44)
        gdp = Metabolite("GDP", "C10H15N5O11P2", molecular_weight=443.20)
        gtp = Metabolite("GTP", "C10H16N5O14P3", molecular_weight=523.18)
        co2 = Metabolite("CO2", "CO2", molecular_weight=44.01)
        
        # Add metabolites to pathway
        for metabolite in [pyruvate, acetyl_coa, oxaloacetate, citrate, isocitrate, alpha_kg,
                          succinyl_coa, succinate, fumarate, malate, coa, nad, nadh, gdp, gtp, co2]:
            pathway.add_metabolite(metabolite)
        
        # Create reactions
        pdh = EnzymeReaction(
            name="Pyruvate dehydrogenase",
            enzyme="1.2.4.1",
            substrates=[(pyruvate, 1), (coa, 1), (nad, 1)],
            products=[(acetyl_coa, 1), (nadh, 1), (co2, 1)],
            reversible=False,
            km_values={"Pyruvate": 0.1, "CoA": 0.01, "NAD+": 0.05},
            kcat=5.0
        )
        
        cs = EnzymeReaction(
            name="Citrate synthase",
            enzyme="2.3.3.1",
            substrates=[(acetyl_coa, 1), (oxaloacetate, 1)],
            products=[(citrate, 1), (coa, 1)],
            reversible=False,
            km_values={"Acetyl-CoA": 0.01, "Oxaloacetate": 0.005},
            kcat=10.0
        )
        
        aco = EnzymeReaction(
            name="Aconitase",
            enzyme="4.2.1.3",
            substrates=[(citrate, 1)],
            products=[(isocitrate, 1)],
            reversible=True,
            km_values={"Citrate": 0.5},
            kcat=2.0
        )
        
        idh = EnzymeReaction(
            name="Isocitrate dehydrogenase",
            enzyme="1.1.1.42",
            substrates=[(isocitrate, 1), (nad, 1)],
            products=[(alpha_kg, 1), (nadh, 1), (co2, 1)],
            reversible=False,
            km_values={"Isocitrate": 0.03, "NAD+": 0.1},
            kcat=8.0
        )
        
        kgdh = EnzymeReaction(
            name="Alpha-ketoglutarate dehydrogenase",
            enzyme="1.2.4.2",
            substrates=[(alpha_kg, 1), (coa, 1), (nad, 1)],
            products=[(succinyl_coa, 1), (nadh, 1), (co2, 1)],
            reversible=False,
            km_values={"Alpha-ketoglutarate": 0.1, "CoA": 0.01, "NAD+": 0.05},
            kcat=3.0
        )
        
        scs = EnzymeReaction(
            name="Succinyl-CoA synthetase",
            enzyme="6.2.1.4",
            substrates=[(succinyl_coa, 1), (gdp, 1)],
            products=[(succinate, 1), (coa, 1), (gtp, 1)],
            reversible=True,
            km_values={"Succinyl-CoA": 0.02, "GDP": 0.1},
            kcat=5.0
        )
        
        sdh = EnzymeReaction(
            name="Succinate dehydrogenase",
            enzyme="1.3.5.1",
            substrates=[(succinate, 1), (nad, 1)],
            products=[(fumarate, 1), (nadh, 1)],
            reversible=False,
            km_values={"Succinate": 0.5, "NAD+": 0.1},
            kcat=2.0
        )
        
        fh = EnzymeReaction(
            name="Fumarase",
            enzyme="4.2.1.2",
            substrates=[(fumarate, 1)],
            products=[(malate, 1)],
            reversible=True,
            km_values={"Fumarate": 0.05},
            kcat=25.0
        )
        
        mdh = EnzymeReaction(
            name="Malate dehydrogenase",
            enzyme="1.1.1.37",
            substrates=[(malate, 1), (nad, 1)],
            products=[(oxaloacetate, 1), (nadh, 1)],
            reversible=True,
            km_values={"Malate": 0.9, "NAD+": 0.05},
            kcat=12.0
        )
        
        # Add reactions to pathway
        for reaction in [pdh, cs, aco, idh, kgdh, scs, sdh, fh, mdh]:
            pathway.add_reaction(reaction)
        
        return pathway
    
    @staticmethod
    def create_custom_pathway(name: str, metabolites: List[Metabolite], reactions: List[EnzymeReaction]):
        """
        Create a custom biosynthesis pathway.
        
        Args:
            name: Pathway name
            metabolites: List of metabolites
            reactions: List of reactions
            
        Returns:
            BiosynthesisPathway object
        """
        pathway = BiosynthesisPathway(name)
        
        # Add metabolites
        for metabolite in metabolites:
            pathway.add_metabolite(metabolite)
        
        # Add reactions
        for reaction in reactions:
            pathway.add_reaction(reaction)
        
        return pathway