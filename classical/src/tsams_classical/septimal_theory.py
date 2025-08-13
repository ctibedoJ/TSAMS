&quot;&quot;&quot;
Septimal Theory module for Tsams Classical.

This module is part of the Tibedo Structural Algebraic Modeling System (TSAMS) ecosystem.
&quot;&quot;&quot;

# Code adapted from tibedo_biological_implementation.py

"""
TIBEDO Framework: Biological Implementation with Quaternion-Based Möbius Strip Dual Pairing

This module implements the TIBEDO Framework for biological applications,
particularly protein folding and DNA/mRNA processes, using quaternion-based
Möbius strip dual pairing and cyclotomic field theory.
"""

import numpy as np
import sympy as sp
import scipy.linalg as la
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class QuaternionMobiusStripPairing:
    """
    Implementation of Quaternion-Based Möbius Strip Dual Pairing.
    
    This class models the [0,1] qubit superimposition base state using
    quaternion mathematics and Möbius strip topology.
    """
    
    def __init__(self):
        """Initialize the QuaternionMobiusStripPairing object."""
        # Define the quaternion basis (1, i, j, k)
        self.basis = np.array([
            [1, 0, 0, 0],  # 1
            [0, 1, 0, 0],  # i
            [0, 0, 1, 0],  # j
            [0, 0, 0, 1]   # k
        ])
        
        # Initialize the Möbius strip parameters
        self.strip_width = 1.0
        self.strip_radius = 2.0
        
    def quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions.
        
        Args:
            q1 (numpy.ndarray): First quaternion [a, b, c, d] = a + bi + cj + dk
            q2 (numpy.ndarray): Second quaternion [e, f, g, h] = e + fi + gj + hk
            
        Returns:
            numpy.ndarray: The product quaternion
        """
        a, b, c, d = q1
        e, f, g, h = q2
        
        return np.array([
            a*e - b*f - c*g - d*h,  # Real part
            a*f + b*e + c*h - d*g,  # i component
            a*g - b*h + c*e + d*f,  # j component
            a*h + b*g - c*f + d*e   # k component
        ])
        
    def quaternion_conjugate(self, q):
        """
        Compute the conjugate of a quaternion.
        
        Args:
            q (numpy.ndarray): Quaternion [a, b, c, d] = a + bi + cj + dk
            
        Returns:
            numpy.ndarray: The conjugate quaternion [a, -b, -c, -d] = a - bi - cj - dk
        """
        return np.array([q[0], -q[1], -q[2], -q[3]])
        
    def quaternion_norm(self, q):
        """
        Compute the norm of a quaternion.
        
        Args:
            q (numpy.ndarray): Quaternion [a, b, c, d] = a + bi + cj + dk
            
        Returns:
            float: The norm of the quaternion
        """
        return np.sqrt(np.sum(q * q))
        
    def quaternion_inverse(self, q):
        """
        Compute the inverse of a quaternion.
        
        Args:
            q (numpy.ndarray): Quaternion [a, b, c, d] = a + bi + cj + dk
            
        Returns:
            numpy.ndarray: The inverse quaternion
        """
        norm_squared = np.sum(q * q)
        if norm_squared < 1e-10:
            raise ValueError("Cannot invert a quaternion with zero norm")
        
        return self.quaternion_conjugate(q) / norm_squared
        
    def create_mobius_strip_points(self, num_points=100):
        """
        Create points on a Möbius strip.
        
        Args:
            num_points (int): Number of points to generate
            
        Returns:
            numpy.ndarray: Array of 3D points on the Möbius strip
        """
        # Parameter ranges
        u_range = np.linspace(0, 2*np.pi, num_points)
        v_range = np.linspace(-self.strip_width/2, self.strip_width/2, num_points)
        
        # Create meshgrid
        u, v = np.meshgrid(u_range, v_range)
        
        # Flatten the meshgrid
        u = u.flatten()
        v = v.flatten()
        
        # Calculate the points on the Möbius strip
        x = (self.strip_radius + v * np.cos(u/2)) * np.cos(u)
        y = (self.strip_radius + v * np.cos(u/2)) * np.sin(u)
        z = v * np.sin(u/2)
        
        return np.column_stack((x, y, z))
        
    def map_quaternion_to_mobius(self, q):
        """
        Map a quaternion to a point on the Möbius strip.
        
        Args:
            q (numpy.ndarray): Quaternion [a, b, c, d] = a + bi + cj + dk
            
        Returns:
            numpy.ndarray: 3D point on the Möbius strip
        """
        # Normalize the quaternion
        q_norm = q / self.quaternion_norm(q)
        
        # Map the quaternion to parameters u and v
        u = 2 * np.pi * (q_norm[0] + 1) / 2  # Map real part to [0, 2π]
        v = self.strip_width * (q_norm[1] / 2)  # Map i component to strip width
        
        # Calculate the point on the Möbius strip
        x = (self.strip_radius + v * np.cos(u/2)) * np.cos(u)
        y = (self.strip_radius + v * np.cos(u/2)) * np.sin(u)
        z = v * np.sin(u/2)
        
        return np.array([x, y, z])
        
    def create_dual_pairing(self, q1, q2):
        """
        Create a dual pairing between two quaternions.
        
        Args:
            q1 (numpy.ndarray): First quaternion
            q2 (numpy.ndarray): Second quaternion
            
        Returns:
            tuple: (pairing_matrix, pairing_invariant)
        """
        # Compute the dual pairing matrix
        q1_conj = self.quaternion_conjugate(q1)
        q2_conj = self.quaternion_conjugate(q2)
        
        # The pairing is defined as a 4x4 matrix
        pairing_matrix = np.zeros((4, 4))
        
        for i in range(4):
            for j in range(4):
                # Compute the pairing between basis elements
                e_i = np.zeros(4)
                e_i[i] = 1
                
                e_j = np.zeros(4)
                e_j[j] = 1
                
                # The pairing is defined as <q1*e_i*q2_conj, e_j>
                q1_e_i = self.quaternion_multiply(q1, e_i)
                q1_e_i_q2_conj = self.quaternion_multiply(q1_e_i, q2_conj)
                
                # The inner product is the dot product
                pairing_matrix[i, j] = np.dot(q1_e_i_q2_conj, e_j)
        
        # Compute the pairing invariant (trace of the matrix)
        pairing_invariant = np.trace(pairing_matrix)
        
        return pairing_matrix, pairing_invariant
        
    def compute_sq_root_conjugate(self, q):
        """
        Compute the square root conjugate of a quaternion.
        
        Args:
            q (numpy.ndarray): Quaternion [a, b, c, d] = a + bi + cj + dk
            
        Returns:
            numpy.ndarray: The square root conjugate quaternion
        """
        # Compute the square root of the norm
        norm = self.quaternion_norm(q)
        sqrt_norm = np.sqrt(norm)
        
        # Compute the square root of the quaternion
        if norm < 1e-10:
            return np.zeros(4)
            
        # For a unit quaternion q = cos(θ) + u*sin(θ) where u is a unit pure quaternion,
        # the square root is q_sqrt = cos(θ/2) + u*sin(θ/2)
        
        # Extract the scalar and vector parts
        scalar = q[0]
        vector = q[1:]
        
        # Compute the angle θ
        theta = np.arccos(scalar / norm)
        
        # Compute the unit vector u
        vector_norm = np.linalg.norm(vector)
        if vector_norm < 1e-10:
            # If the vector part is zero, the quaternion is real
            return np.array([np.sqrt(scalar), 0, 0, 0])
            
        u = vector / vector_norm
        
        # Compute the square root quaternion
        q_sqrt = np.zeros(4)
        q_sqrt[0] = sqrt_norm * np.cos(theta / 2)
        q_sqrt[1:] = sqrt_norm * u * np.sin(theta / 2)
        
        # Compute the conjugate of the square root
        q_sqrt_conj = self.quaternion_conjugate(q_sqrt)
        
        return q_sqrt_conj

class CyclotomicFieldGaloisOrbit:
    """
    Implementation of Galois Prime Ideal Cyclotomic Field Ring structures.
    
    This class models the cyclotomic field structures and Galois orbits
    used in the TIBEDO Framework for biological applications.
    """
    
    def __init__(self, conductor=42):
        """
        Initialize the CyclotomicFieldGaloisOrbit object.
        
        Args:
            conductor (int): The conductor of the cyclotomic field (default: 42)
        """
        self.conductor = conductor
        self.phi_n = self._euler_totient(conductor)
        self.galois_group = self._compute_galois_group()
        self.prime_ideals = self._compute_prime_ideals()
        
    def _euler_totient(self, n):
        """
        Compute Euler's totient function φ(n).
        
        Args:
            n (int): The input number
            
        Returns:
            int: The value of φ(n)
        """
        result = n  # Initialize result as n
        
        # Consider all prime factors of n and subtract their
        # multiples from result
        p = 2
        while p * p <= n:
            # Check if p is a prime factor
            if n % p == 0:
                # If yes, then update n and result
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
            
        # If n has a prime factor greater than sqrt(n)
        # (There can be at most one such prime factor)
        if n > 1:
            result -= result // n
            
        return result
        
    def _compute_galois_group(self):
        """
        Compute the Galois group of the cyclotomic field.
        
        Returns:
            list: The elements of the Galois group
        """
        # The Galois group of Q(ζ_n) is isomorphic to (Z/nZ)*
        # We represent it as the set of integers a such that gcd(a, n) = 1
        galois_group = []
        for a in range(1, self.conductor):
            if np.gcd(a, self.conductor) == 1:
                galois_group.append(a)
                
        return galois_group
        
    def _compute_prime_ideals(self):
        """
        Compute the prime ideals in the cyclotomic field.
        
        Returns:
            dict: A dictionary mapping primes to their factorization pattern
        """
        # For a prime p that doesn't divide the conductor n,
        # the factorization of p in Z[ζ_n] is determined by the order of p modulo n
        
        # We'll compute the factorization pattern for the first few primes
        prime_ideals = {}
        
        for p in range(2, 100):  # Consider primes up to 100
            if self._is_prime(p) and self.conductor % p != 0:
                # Compute the order of p modulo n
                order = self._multiplicative_order(p, self.conductor)
                
                # The number of prime ideals is φ(n) / order
                num_ideals = self.phi_n // order
                
                # The ramification index is 1 since p doesn't divide n
                ramification = 1
                
                prime_ideals[p] = {
                    'order': order,
                    'num_ideals': num_ideals,
                    'ramification': ramification
                }
                
        return prime_ideals
        
    def _is_prime(self, n):
        """
        Check if a number is prime.
        
        Args:
            n (int): The number to check
            
        Returns:
            bool: True if n is prime, False otherwise
        """
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
        
    def _multiplicative_order(self, a, n):
        """
        Compute the multiplicative order of a modulo n.
        
        Args:
            a (int): The element
            n (int): The modulus
            
        Returns:
            int: The smallest positive integer k such that a^k ≡ 1 (mod n)
        """
        if np.gcd(a, n) != 1:
            raise ValueError(f"Element {a} is not coprime to modulus {n}")
            
        a = a % n
        if a == 1:
            return 1
            
        order = 1
        value = a
        
        while value != 1:
            value = (value * a) % n
            order += 1
            
            # Safety check to avoid infinite loops
            if order > n:
                return -1  # Error condition
                
        return order
        
    def compute_dedekind_cut_ratio(self, p):
        """
        Compute the Dedekind cut ratio for a prime p.
        
        Args:
            p (int): A prime number
            
        Returns:
            float: The Dedekind cut ratio
        """
        if p not in self.prime_ideals:
            if not self._is_prime(p):
                raise ValueError(f"{p} is not a prime number")
            if self.conductor % p == 0:
                raise ValueError(f"Prime {p} divides the conductor {self.conductor}")
            
            # Compute the factorization pattern for this prime
            order = self._multiplicative_order(p, self.conductor)
            num_ideals = self.phi_n // order
            ramification = 1
            
            self.prime_ideals[p] = {
                'order': order,
                'num_ideals': num_ideals,
                'ramification': ramification
            }
            
        # The Dedekind cut ratio is related to the inertia degree
        inertia_degree = self.prime_ideals[p]['order']
        
        # The cut ratio is defined as the logarithmic derivative of the local zeta function
        cut_ratio = np.log(p) / (p**inertia_degree - 1)
        
        return cut_ratio
        
    def compute_fano_plane_dual(self):
        """
        Compute the 42-plane Fano pairs as dual of prime 41.
        
        Returns:
            dict: The Fano plane dual structure
        """
        # The 42-plane Fano pairs are related to the prime 41
        # We'll compute the Galois orbits of prime 41 in the cyclotomic field
        
        if 41 not in self.prime_ideals:
            # Compute the factorization pattern for prime 41
            order = self._multiplicative_order(41, self.conductor)
            num_ideals = self.phi_n // order
            ramification = 1
            
            self.prime_ideals[41] = {
                'order': order,
                'num_ideals': num_ideals,
                'ramification': ramification
            }
            
        # Compute the Galois orbits of prime 41
        orbits = []
        for sigma in self.galois_group:
            orbit = []
            for i in range(1, 42):
                if np.gcd(i, 42) == 1:
                    orbit.append((sigma * i) % 42)
            orbits.append(orbit)
            
        # Create the Fano plane dual structure
        fano_dual = {
            'prime': 41,
            'conductor': 42,
            'galois_orbits': orbits,
            'num_planes': len(orbits)
        }
        
        return fano_dual

class ProteinFoldingSimulator:
    """
    Implementation of the Protein Folding Simulator using the TIBEDO Framework.
    
    This class models protein folding dynamics using quaternion-based Möbius strip
    dual pairing and cyclotomic field theory.
    """
    
    def __init__(self):
        """Initialize the ProteinFoldingSimulator object."""
        self.mobius_pairing = QuaternionMobiusStripPairing()
        self.cyclotomic_field = CyclotomicFieldGaloisOrbit(conductor=42)
        self.amino_acids = self._initialize_amino_acids()
        
    def _initialize_amino_acids(self):
        """
        Initialize the amino acid properties.
        
        Returns:
            dict: A dictionary mapping amino acid codes to their properties
        """
        amino_acids = {}
        
        # Define the 20 standard amino acids with their properties
        amino_acids['A'] = {'name': 'Alanine', 'hydrophobicity': 1.8, 'charge': 0, 'size': 0.88}
        amino_acids['C'] = {'name': 'Cysteine', 'hydrophobicity': 2.5, 'charge': 0, 'size': 1.08}
        amino_acids['D'] = {'name': 'Aspartic Acid', 'hydrophobicity': -3.5, 'charge': -1, 'size': 1.12}
        amino_acids['E'] = {'name': 'Glutamic Acid', 'hydrophobicity': -3.5, 'charge': -1, 'size': 1.38}
        amino_acids['F'] = {'name': 'Phenylalanine', 'hydrophobicity': 2.8, 'charge': 0, 'size': 1.90}
        amino_acids['G'] = {'name': 'Glycine', 'hydrophobicity': -0.4, 'charge': 0, 'size': 0.60}
        amino_acids['H'] = {'name': 'Histidine', 'hydrophobicity': -3.2, 'charge': 0.1, 'size': 1.53}
        amino_acids['I'] = {'name': 'Isoleucine', 'hydrophobicity': 4.5, 'charge': 0, 'size': 1.69}
        amino_acids['K'] = {'name': 'Lysine', 'hydrophobicity': -3.9, 'charge': 1, 'size': 1.71}
        amino_acids['L'] = {'name': 'Leucine', 'hydrophobicity': 3.8, 'charge': 0, 'size': 1.69}
        amino_acids['M'] = {'name': 'Methionine', 'hydrophobicity': 1.9, 'charge': 0, 'size': 1.70}
        amino_acids['N'] = {'name': 'Asparagine', 'hydrophobicity': -3.5, 'charge': 0, 'size': 1.28}
        amino_acids['P'] = {'name': 'Proline', 'hydrophobicity': -1.6, 'charge': 0, 'size': 1.23}
        amino_acids['Q'] = {'name': 'Glutamine', 'hydrophobicity': -3.5, 'charge': 0, 'size': 1.53}
        amino_acids['R'] = {'name': 'Arginine', 'hydrophobicity': -4.5, 'charge': 1, 'size': 1.91}
        amino_acids['S'] = {'name': 'Serine', 'hydrophobicity': -0.8, 'charge': 0, 'size': 0.93}
        amino_acids['T'] = {'name': 'Threonine', 'hydrophobicity': -0.7, 'charge': 0, 'size': 1.22}
        amino_acids['V'] = {'name': 'Valine', 'hydrophobicity': 4.2, 'charge': 0, 'size': 1.40}
        amino_acids['W'] = {'name': 'Tryptophan', 'hydrophobicity': -0.9, 'charge': 0, 'size': 2.28}
        amino_acids['Y'] = {'name': 'Tyrosine', 'hydrophobicity': -1.3, 'charge': 0, 'size': 2.03}
        
        return amino_acids
        
    def map_sequence_to_quaternions(self, sequence):
        """
        Map an amino acid sequence to quaternions.
        
        Args:
            sequence (str): A string of amino acid codes
            
        Returns:
            list: A list of quaternions representing the amino acids
        """
        quaternions = []
        
        for aa in sequence:
            if aa in self.amino_acids:
                # Map amino acid properties to quaternion components
                hydrophobicity = self.amino_acids[aa]['hydrophobicity'] / 5.0  # Normalize to [-1, 1]
                charge = self.amino_acids[aa]['charge']
                size = self.amino_acids[aa]['size'] / 2.5  # Normalize to [0, 1]
                
                # Create a quaternion [w, x, y, z]
                q = np.array([
                    1.0,                # Real part (always 1 for simplicity)
                    hydrophobicity,     # i component (hydrophobicity)
                    charge,             # j component (charge)
                    size                # k component (size)
                ])
                
                # Normalize the quaternion
                q = q / self.mobius_pairing.quaternion_norm(q)
                
                quaternions.append(q)
            else:
                # Skip unknown amino acids
                continue
                
        return quaternions
        
    def compute_triad_pairs(self, quaternions):
        """
        Compute triad pairs from quaternions.
        
        Args:
            quaternions (list): A list of quaternions
            
        Returns:
            list: A list of triad pairs
        """
        triad_pairs = []
        
        # Process quaternions in groups of 3 (triads)
        for i in range(0, len(quaternions) - 2, 3):
            triad = quaternions[i:i+3]
            
            # Compute the triad center as the average of the three quaternions
            triad_center = np.mean(triad, axis=0)
            triad_center = triad_center / self.mobius_pairing.quaternion_norm(triad_center)
            
            # Compute the square root conjugate of the triad center
            sq_root_conj = self.mobius_pairing.compute_sq_root_conjugate(triad_center)
            
            # Create a triad pair (triad_center, sq_root_conj)
            triad_pair = (triad_center, sq_root_conj)
            triad_pairs.append(triad_pair)
            
        return triad_pairs
        
    def compute_path_integral(self, triad_pairs):
        """
        Compute the arithmetic/metric scalar depth ratio path integral.
        
        Args:
            triad_pairs (list): A list of triad pairs
            
        Returns:
            float: The path integral value
        """
        path_integral = 0.0
        
        for i in range(len(triad_pairs) - 1):
            # Extract the current and next triad pairs
            current_pair = triad_pairs[i]
            next_pair = triad_pairs[i+1]
            
            # Compute the dual pairing between the current and next triad centers
            current_center = current_pair[0]
            next_center = next_pair[0]
            
            # Compute the pairing matrix and invariant
            _, pairing_invariant = self.mobius_pairing.create_dual_pairing(current_center, next_center)
            
            # Compute the scalar depth ratio
            current_sq_conj = current_pair[1]
            next_sq_conj = next_pair[1]
            
            # Compute the quaternion distance between the square root conjugates
            diff = next_sq_conj - current_sq_conj
            depth_ratio = self.mobius_pairing.quaternion_norm(diff) / pairing_invariant
            
            # Add to the path integral
            path_integral += depth_ratio
            
        return path_integral
        
    def simulate_protein_folding(self, sequence):
        """
        Simulate protein folding for an amino acid sequence.
        
        Args:
            sequence (str): A string of amino acid codes
            
        Returns:
            dict: The simulation results
        """
        # Step 1: Map the sequence to quaternions
        quaternions = self.map_sequence_to_quaternions(sequence)
        
        # Step 2: Compute triad pairs
        triad_pairs = self.compute_triad_pairs(quaternions)
        
        # Step 3: Compute the path integral
        path_integral = self.compute_path_integral(triad_pairs)
        
        # Step 4: Map quaternions to the Möbius strip
        mobius_points = [self.mobius_pairing.map_quaternion_to_mobius(q) for q in quaternions]
        
        # Step 5: Compute the Dedekind cut ratios for relevant primes
        dedekind_cuts = {}
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]:
            try:
                dedekind_cuts[p] = self.cyclotomic_field.compute_dedekind_cut_ratio(p)
            except ValueError:
                # Skip primes that divide the conductor
                continue
                
        # Step 6: Compute the Fano plane dual structure
        fano_dual = self.cyclotomic_field.compute_fano_plane_dual()
        
        # Return the simulation results
        return {
            'sequence': sequence,
            'quaternions': quaternions,
            'triad_pairs': triad_pairs,
            'path_integral': path_integral,
            'mobius_points': mobius_points,
            'dedekind_cuts': dedekind_cuts,
            'fano_dual': fano_dual
        }
        
    def visualize_mobius_strip(self, mobius_points):
        """
        Visualize points on the Möbius strip.
        
        Args:
            mobius_points (list): A list of 3D points on the Möbius strip
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Create the figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create the Möbius strip surface
        strip_points = self.mobius_pairing.create_mobius_strip_points(num_points=20)
        
        # Plot the Möbius strip surface as a scatter plot with low alpha
        ax.scatter(strip_points[:, 0], strip_points[:, 1], strip_points[:, 2], 
                  color='lightgray', alpha=0.1, s=10)
        
        # Convert mobius_points to a numpy array
        points = np.array(mobius_points)
        
        # Plot the protein folding path
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', linewidth=2)
        
        # Plot the amino acid positions
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  color='blue', s=50, edgecolor='black')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Protein Folding Path on Möbius Strip')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        return fig

def test_protein_folding_simulation():
    """
    Test the protein folding simulation using the TIBEDO Framework.
    """
    print("Testing TIBEDO Framework for Protein Folding Simulation")
    print("=====================================================")
    
    # Create the simulator
    simulator = ProteinFoldingSimulator()
    
    # Define some test protein sequences
    test_sequences = [
        "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT",
        "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
    ]
    
    for i, sequence in enumerate(test_sequences):
        print(f"\nSimulating protein folding for sequence {i+1} (length: {len(sequence)})")
        
        # Measure the simulation time
        start_time = time.time()
        results = simulator.simulate_protein_folding(sequence)
        elapsed_time = time.time() - start_time
        
        # Print the results
        print(f"Path integral: {results['path_integral']:.6f}")
        print(f"Number of triad pairs: {len(results['triad_pairs'])}")
        print(f"Simulation time: {elapsed_time:.6f} seconds")
        
        # Print the Dedekind cut ratios
        print("\nDedekind cut ratios:")
        for p, ratio in results['dedekind_cuts'].items():
            print(f"  Prime {p}: {ratio:.6f}")
            
        # Print information about the Fano plane dual
        fano_dual = results['fano_dual']
        print(f"\nFano plane dual of prime {fano_dual['prime']}:")
        print(f"  Conductor: {fano_dual['conductor']}")
        print(f"  Number of planes: {fano_dual['num_planes']}")
        
        # Create a visualization of the protein folding path
        fig = simulator.visualize_mobius_strip(results['mobius_points'])
        
        # Save the figure
        fig.savefig(f"protein_folding_sequence_{i+1}.png")
        plt.close(fig)
        
        print(f"Visualization saved as protein_folding_sequence_{i+1}.png")

if __name__ == "__main__":
    test_protein_folding_simulation()
