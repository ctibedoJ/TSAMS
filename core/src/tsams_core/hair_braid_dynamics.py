&quot;&quot;&quot;
Hair Braid Dynamics module for Tsams Core.

This module is part of the Tibedo Structural Algebraic Modeling System (TSAMS) ecosystem.
&quot;&quot;&quot;

# Code adapted from tibedo_braid_neural_mapping.py

"""
TIBEDO Braid Neural Mapping Tool

This module implements a braid neural mapping tool that takes Prime Coordinates
and produces linear coordinates in the X, Y, Z plane through right triangle involutions
around a central left-turn facing piston-like strip with convex to concave ratios
among paired primes along Mersenne prime cycles using Dedekind cuts.

The implementation follows the quaternionic-octonionic foundations established in
Chapters 30-33, where the right triangle and its self-adjoint left side triangle
create a dual system. The left spinor creates forward entropy (time entropic dimension),
whereas X, Y, Z relatives enable backward state evolution of quantum fluctuations.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import prime, isprime
import math
from typing import List, Tuple, Dict, Any, Optional, Union


class DedekindCut:
    """
    Implementation of Dedekind cuts for rational and irrational numbers.
    
    A Dedekind cut is a partition of the rational numbers into two non-empty sets A and B,
    such that every element of A is less than every element of B, and A contains no greatest element.
    """
    
    def __init__(self, value: float):
        """
        Initialize a Dedekind cut for the given value.
        
        Args:
            value: The value to create a Dedekind cut for
        """
        self.value = value
        self.precision = 1e-10
        
    def contains(self, rational: float) -> bool:
        """
        Check if a rational number is in the lower set of the Dedekind cut.
        
        Args:
            rational: The rational number to check
            
        Returns:
            bool: True if the rational is in the lower set, False otherwise
        """
        return rational < self.value
    
    def get_approximation(self, depth: int = 10) -> List[float]:
        """
        Get a sequence of rational approximations to the Dedekind cut.
        
        Args:
            depth: The number of approximations to generate
            
        Returns:
            List[float]: A sequence of rational approximations
        """
        approximations = []
        step = 1.0
        current = 0.0
        
        for _ in range(depth):
            while current + step < self.value:
                current += step
            step /= 10
            approximations.append(current)
            
        return approximations
    
    def get_rational_bounds(self) -> Tuple[float, float]:
        """
        Get rational bounds for the Dedekind cut.
        
        Returns:
            Tuple[float, float]: Lower and upper rational bounds
        """
        lower = math.floor(self.value * 1e10) / 1e10
        upper = math.ceil(self.value * 1e10) / 1e10
        return lower, upper


class MersennePrimeSequence:
    """
    Generator for Mersenne primes and related sequences.
    
    A Mersenne prime is a prime number of the form 2^n - 1.
    """
    
    def __init__(self, max_exponent: int = 100):
        """
        Initialize the Mersenne prime sequence generator.
        
        Args:
            max_exponent: The maximum exponent to check for Mersenne primes
        """
        self.max_exponent = max_exponent
        self.mersenne_primes = self._generate_mersenne_primes()
        
    def _generate_mersenne_primes(self) -> List[int]:
        """
        Generate Mersenne primes up to the maximum exponent.
        
        Returns:
            List[int]: The list of Mersenne primes
        """
        mersenne_primes = []
        
        for n in range(2, self.max_exponent + 1):
            if not sp.isprime(n):
                continue
                
            mersenne = 2**n - 1
            if sp.isprime(mersenne):
                mersenne_primes.append(mersenne)
                
        return mersenne_primes
    
    def get_mersenne_primes(self) -> List[int]:
        """
        Get the list of Mersenne primes.
        
        Returns:
            List[int]: The list of Mersenne primes
        """
        return self.mersenne_primes
    
    def get_mersenne_exponents(self) -> List[int]:
        """
        Get the exponents of the Mersenne primes.
        
        Returns:
            List[int]: The list of exponents
        """
        exponents = []
        
        for mp in self.mersenne_primes:
            # Find n such that 2^n - 1 = mp
            n = int(math.log2(mp + 1))
            exponents.append(n)
            
        return exponents
    
    def get_paired_primes(self) -> List[Tuple[int, int]]:
        """
        Get pairs of primes related to Mersenne primes.
        
        Returns:
            List[Tuple[int, int]]: List of prime pairs
        """
        pairs = []
        
        for mp in self.mersenne_primes:
            # Find primes p such that p divides mp+1 or mp-1
            p_plus = self._find_prime_factors(mp + 1)
            p_minus = self._find_prime_factors(mp - 1)
            
            for p in p_plus:
                for q in p_minus:
                    pairs.append((p, q))
                    
        return pairs
    
    def _find_prime_factors(self, n: int) -> List[int]:
        """
        Find the prime factors of a number.
        
        Args:
            n: The number to factorize
            
        Returns:
            List[int]: The list of prime factors
        """
        factors = []
        
        # Simple trial division for prime factorization
        i = 2
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                if sp.isprime(i):
                    factors.append(i)
        
        if n > 1 and sp.isprime(n):
            factors.append(n)
            
        return factors


class RightTriangleInvolution:
    """
    Implementation of right triangle involutions for coordinate mapping.
    
    Right triangle involutions are transformations that map points through
    reflections across the hypotenuse of right triangles.
    """
    
    def __init__(self, origin: Tuple[float, float, float] = (0, 0, 0)):
        """
        Initialize the right triangle involution.
        
        Args:
            origin: The origin point for the involutions
        """
        self.origin = origin
        
    def reflect_point(self, point: Tuple[float, float, float], 
                     normal: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Reflect a point across a plane defined by a normal vector.
        
        Args:
            point: The point to reflect
            normal: The normal vector of the reflection plane
            
        Returns:
            Tuple[float, float, float]: The reflected point
        """
        # Normalize the normal vector
        norm = math.sqrt(sum(n*n for n in normal))
        unit_normal = tuple(n/norm for n in normal)
        
        # Calculate the dot product
        dot_product = sum(p*n for p, n in zip(point, unit_normal))
        
        # Calculate the reflection
        reflection = tuple(p - 2 * dot_product * n for p, n in zip(point, unit_normal))
        
        return reflection
    
    def create_right_triangle(self, a: float, b: float) -> Tuple[
        Tuple[float, float, float], 
        Tuple[float, float, float], 
        Tuple[float, float, float]
    ]:
        """
        Create a right triangle in 3D space.
        
        Args:
            a: The length of the first leg
            b: The length of the second leg
            
        Returns:
            Tuple[Tuple[float, float, float], ...]: The three vertices of the triangle
        """
        # Create a right triangle in the XY plane
        p1 = self.origin
        p2 = (p1[0] + a, p1[1], p1[2])
        p3 = (p1[0], p1[1] + b, p1[2])
        
        return (p1, p2, p3)
    
    def get_hypotenuse_normal(self, triangle: Tuple[
        Tuple[float, float, float], 
        Tuple[float, float, float], 
        Tuple[float, float, float]
    ]) -> Tuple[float, float, float]:
        """
        Get the normal vector to the hypotenuse of a right triangle.
        
        Args:
            triangle: The right triangle vertices
            
        Returns:
            Tuple[float, float, float]: The normal vector
        """
        p1, p2, p3 = triangle
        
        # Calculate vectors along the legs
        v1 = tuple(p2[i] - p1[i] for i in range(3))
        v2 = tuple(p3[i] - p1[i] for i in range(3))
        
        # Calculate the cross product to get the normal
        normal = (
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        )
        
        return normal
    
    def apply_involution(self, point: Tuple[float, float, float], 
                        a: float, b: float) -> Tuple[float, float, float]:
        """
        Apply a right triangle involution to a point.
        
        Args:
            point: The point to transform
            a: The length of the first leg
            b: The length of the second leg
            
        Returns:
            Tuple[float, float, float]: The transformed point
        """
        # Create the right triangle
        triangle = self.create_right_triangle(a, b)
        
        # Get the normal to the hypotenuse
        normal = self.get_hypotenuse_normal(triangle)
        
        # Reflect the point
        reflected = self.reflect_point(point, normal)
        
        return reflected


class PistonStripMapping:
    """
    Implementation of the central left-turn facing piston-like strip mapping.
    
    This class implements the mapping of points through a piston-like strip
    with convex to concave ratios among paired primes.
    """
    
    def __init__(self, num_segments: int = 10):
        """
        Initialize the piston strip mapping.
        
        Args:
            num_segments: The number of segments in the piston strip
        """
        self.num_segments = num_segments
        self.mersenne_sequence = MersennePrimeSequence()
        self.paired_primes = self.mersenne_sequence.get_paired_primes()
        
    def calculate_convex_concave_ratio(self, prime_pair: Tuple[int, int]) -> float:
        """
        Calculate the convex to concave ratio for a pair of primes.
        
        Args:
            prime_pair: A pair of primes
            
        Returns:
            float: The convex to concave ratio
        """
        p, q = prime_pair
        
        # Calculate the ratio based on the logarithmic relationship
        ratio = math.log(p) / math.log(q)
        
        return ratio
    
    def generate_piston_strip(self) -> List[Tuple[float, float, float]]:
        """
        Generate the central piston-like strip.
        
        Returns:
            List[Tuple[float, float, float]]: Points defining the piston strip
        """
        strip_points = []
        
        # Use paired primes to define the strip geometry
        for i, prime_pair in enumerate(self.paired_primes[:self.num_segments]):
            ratio = self.calculate_convex_concave_ratio(prime_pair)
            
            # Calculate the position along the strip
            t = i / (self.num_segments - 1)
            
            # Create a left-turning spiral pattern
            x = math.cos(2 * math.pi * t) * ratio
            y = math.sin(2 * math.pi * t) * ratio
            z = t * 2 - 1  # Map to [-1, 1]
            
            strip_points.append((x, y, z))
            
        return strip_points
    
    def map_point_to_strip(self, point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Map a point to the piston strip.
        
        Args:
            point: The point to map
            
        Returns:
            Tuple[float, float, float]: The mapped point
        """
        strip_points = self.generate_piston_strip()
        
        # Find the closest point on the strip
        min_dist = float('inf')
        closest_point = None
        
        for strip_point in strip_points:
            dist = sum((p - s)**2 for p, s in zip(point, strip_point))
            if dist < min_dist:
                min_dist = dist
                closest_point = strip_point
                
        return closest_point
    
    def calculate_strip_parameter(self, point: Tuple[float, float, float]) -> float:
        """
        Calculate the parameter along the strip for a point.
        
        Args:
            point: The point to parameterize
            
        Returns:
            float: The parameter value in [0, 1]
        """
        strip_points = self.generate_piston_strip()
        
        # Find the closest point on the strip
        min_dist = float('inf')
        closest_idx = 0
        
        for i, strip_point in enumerate(strip_points):
            dist = sum((p - s)**2 for p, s in zip(point, strip_point))
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        # Calculate the parameter
        t = closest_idx / (len(strip_points) - 1)
        
        return t


class BraidNeuralMapping:
    """
    Implementation of the braid neural mapping tool.
    
    This class implements the mapping of prime coordinates to linear coordinates
    in the X, Y, Z plane through right triangle involutions around a central
    left-turn facing piston-like strip.
    """
    
    def __init__(self, max_prime_index: int = 100):
        """
        Initialize the braid neural mapping tool.
        
        Args:
            max_prime_index: The maximum index for prime generation
        """
        self.max_prime_index = max_prime_index
        self.primes = [prime(i) for i in range(1, max_prime_index + 1)]
        self.mersenne_sequence = MersennePrimeSequence()
        self.triangle_involution = RightTriangleInvolution()
        self.piston_strip = PistonStripMapping()
        
    def prime_to_coordinate(self, p: int) -> Tuple[float, float, float]:
        """
        Map a prime number to a 3D coordinate.
        
        Args:
            p: The prime number
            
        Returns:
            Tuple[float, float, float]: The 3D coordinate
        """
        if not isprime(p):
            raise ValueError(f"{p} is not a prime number")
            
        # Create a Dedekind cut for the prime
        cut = DedekindCut(p)
        lower, upper = cut.get_rational_bounds()
        
        # Calculate the coordinate based on the prime's properties
        log_p = math.log(p)
        sqrt_p = math.sqrt(p)
        
        x = log_p / sqrt_p
        y = (upper - lower) * log_p
        z = math.sin(p) * math.cos(log_p)
        
        return (x, y, z)
    
    def apply_dual_triangle_mapping(self, coord: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Apply the dual triangle mapping (right triangle and self-adjoint left triangle).
        
        Args:
            coord: The coordinate to transform
            
        Returns:
            Tuple[float, float, float]: The transformed coordinate
        """
        # Apply right triangle involution
        a = math.log(coord[0] + 2)  # Ensure positive value
        b = math.log(coord[1] + 2)  # Ensure positive value
        
        right_transformed = self.triangle_involution.apply_involution(coord, a, b)
        
        # Apply self-adjoint left triangle involution
        # For the left triangle, we use the conjugate values
        a_left = b
        b_left = a
        
        # Create a new involution with origin at the transformed point
        left_involution = RightTriangleInvolution(origin=right_transformed)
        
        # Apply the left triangle involution
        final_coord = left_involution.apply_involution(right_transformed, a_left, b_left)
        
        return final_coord
    
    def apply_piston_strip_mapping(self, coord: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Apply the piston strip mapping to a coordinate.
        
        Args:
            coord: The coordinate to transform
            
        Returns:
            Tuple[float, float, float]: The transformed coordinate
        """
        # Map the point to the piston strip
        strip_point = self.piston_strip.map_point_to_strip(coord)
        
        # Calculate the parameter along the strip
        t = self.piston_strip.calculate_strip_parameter(coord)
        
        # Find paired primes based on the parameter
        paired_primes = self.mersenne_sequence.get_paired_primes()
        idx = min(int(t * len(paired_primes)), len(paired_primes) - 1)
        prime_pair = paired_primes[idx]
        
        # Calculate the convex to concave ratio
        ratio = self.piston_strip.calculate_convex_concave_ratio(prime_pair)
        
        # Apply the transformation based on the ratio
        transformed = (
            coord[0] * ratio,
            coord[1] * ratio,
            coord[2] * ratio
        )
        
        return transformed
    
    def map_prime_coordinates(self, prime_coords: List[int]) -> List[Tuple[float, float, float]]:
        """
        Map a list of prime coordinates to 3D coordinates.
        
        Args:
            prime_coords: The list of prime coordinates
            
        Returns:
            List[Tuple[float, float, float]]: The mapped 3D coordinates
        """
        mapped_coords = []
        
        for p in prime_coords:
            # Convert prime to initial coordinate
            initial_coord = self.prime_to_coordinate(p)
            
            # Apply dual triangle mapping
            triangle_mapped = self.apply_dual_triangle_mapping(initial_coord)
            
            # Apply piston strip mapping
            final_coord = self.apply_piston_strip_mapping(triangle_mapped)
            
            mapped_coords.append(final_coord)
            
        return mapped_coords
    
    def create_braid_structure(self, num_primes: int = 10) -> List[Tuple[float, float, float]]:
        """
        Create a braid structure using the first num_primes prime numbers.
        
        Args:
            num_primes: The number of primes to use
            
        Returns:
            List[Tuple[float, float, float]]: The points in the braid structure
        """
        # Use the first num_primes prime numbers
        prime_coords = self.primes[:num_primes]
        
        # Map the prime coordinates
        mapped_coords = self.map_prime_coordinates(prime_coords)
        
        return mapped_coords
    
    def visualize_braid_structure(self, num_primes: int = 10, save_path: Optional[str] = None):
        """
        Visualize the braid structure.
        
        Args:
            num_primes: The number of primes to use
            save_path: Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Create the braid structure
        braid_points = self.create_braid_structure(num_primes)
        
        # Extract x, y, z coordinates
        x_coords = [p[0] for p in braid_points]
        y_coords = [p[1] for p in braid_points]
        z_coords = [p[2] for p in braid_points]
        
        # Create the figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the braid structure
        ax.scatter(x_coords, y_coords, z_coords, c=range(len(braid_points)), 
                  cmap='viridis', s=100, alpha=0.8)
        
        # Connect the points with lines
        ax.plot(x_coords, y_coords, z_coords, 'gray', alpha=0.5)
        
        # Add labels for the prime numbers
        for i, (x, y, z) in enumerate(braid_points):
            ax.text(x, y, z, f"p{i+1}={self.primes[i]}", fontsize=10)
        
        # Plot the piston strip
        strip_points = self.piston_strip.generate_piston_strip()
        strip_x = [p[0] for p in strip_points]
        strip_y = [p[1] for p in strip_points]
        strip_z = [p[2] for p in strip_points]
        
        ax.plot(strip_x, strip_y, strip_z, 'r-', linewidth=3, alpha=0.7, label='Piston Strip')
        
        # Add labels and title
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_zlabel('Z Coordinate', fontsize=12)
        ax.set_title('TIBEDO Braid Neural Mapping Structure', fontsize=14)
        
        # Add a legend
        ax.legend()
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_prime_mapping(self, p: int, save_path: Optional[str] = None):
        """
        Visualize the mapping process for a specific prime.
        
        Args:
            p: The prime number to visualize
            save_path: Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if not isprime(p):
            raise ValueError(f"{p} is not a prime number")
        
        # Calculate the mapping steps
        initial_coord = self.prime_to_coordinate(p)
        triangle_mapped = self.apply_dual_triangle_mapping(initial_coord)
        final_coord = self.apply_piston_strip_mapping(triangle_mapped)
        
        # Create the figure
        fig = plt.figure(figsize=(15, 5))
        
        # Plot the initial coordinate
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter([initial_coord[0]], [initial_coord[1]], [initial_coord[2]], 
                   c='blue', s=100, alpha=0.8)
        ax1.set_xlabel('X Coordinate', fontsize=10)
        ax1.set_ylabel('Y Coordinate', fontsize=10)
        ax1.set_zlabel('Z Coordinate', fontsize=10)
        ax1.set_title(f'Initial Prime Coordinate (p={p})', fontsize=12)
        
        # Plot the triangle mapped coordinate
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter([triangle_mapped[0]], [triangle_mapped[1]], [triangle_mapped[2]], 
                   c='green', s=100, alpha=0.8)
        ax2.set_xlabel('X Coordinate', fontsize=10)
        ax2.set_ylabel('Y Coordinate', fontsize=10)
        ax2.set_zlabel('Z Coordinate', fontsize=10)
        ax2.set_title('After Dual Triangle Mapping', fontsize=12)
        
        # Plot the final coordinate
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter([final_coord[0]], [final_coord[1]], [final_coord[2]], 
                   c='red', s=100, alpha=0.8)
        ax3.set_xlabel('X Coordinate', fontsize=10)
        ax3.set_ylabel('Y Coordinate', fontsize=10)
        ax3.set_zlabel('Z Coordinate', fontsize=10)
        ax3.set_title('After Piston Strip Mapping', fontsize=12)
        
        # Add a suptitle
        fig.suptitle(f'TIBEDO Braid Neural Mapping Process for Prime p={p}', fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_linear_coordinates(self, prime_coords: List[int]) -> List[Tuple[float, float, float]]:
        """
        Generate linear coordinates in the X, Y, Z plane from prime coordinates.
        
        Args:
            prime_coords: The list of prime coordinates
            
        Returns:
            List[Tuple[float, float, float]]: The linear coordinates
        """
        # Map the prime coordinates to 3D space
        mapped_coords = self.map_prime_coordinates(prime_coords)
        
        # Apply a linear transformation to ensure the coordinates are in a linear space
        linear_coords = []
        
        for coord in mapped_coords:
            # Apply a linear transformation
            linear_coord = (
                coord[0] * 2,  # Scale X
                coord[1] * 2,  # Scale Y
                coord[2] * 2   # Scale Z
            )
            
            linear_coords.append(linear_coord)
            
        return linear_coords
    
    def calculate_forward_entropy(self, prime_coords: List[int]) -> float:
        """
        Calculate the forward entropy (time entropic dimension) using left spinor.
        
        Args:
            prime_coords: The list of prime coordinates
            
        Returns:
            float: The forward entropy value
        """
        # Map the prime coordinates
        mapped_coords = self.map_prime_coordinates(prime_coords)
        
        # Calculate the entropy based on the distribution of coordinates
        entropy = 0.0
        
        # Calculate the centroid
        centroid = [sum(coord[i] for coord in mapped_coords) / len(mapped_coords) 
                   for i in range(3)]
        
        # Calculate the entropy based on distances from centroid
        for coord in mapped_coords:
            dist = math.sqrt(sum((c - cent)**2 for c, cent in zip(coord, centroid)))
            if dist > 0:
                entropy -= dist * math.log(dist)
                
        return entropy
    
    def calculate_backward_state_evolution(self, prime_coords: List[int], steps: int = 5) -> List[List[Tuple[float, float, float]]]:
        """
        Calculate the backward state evolution of quantum fluctuations using X, Y, Z relatives.
        
        Args:
            prime_coords: The list of prime coordinates
            steps: The number of backward steps
            
        Returns:
            List[List[Tuple[float, float, float]]]: The backward evolution states
        """
        # Map the prime coordinates
        mapped_coords = self.map_prime_coordinates(prime_coords)
        
        # Calculate backward evolution
        backward_states = [mapped_coords]
        
        for step in range(steps):
            # Calculate the previous state
            prev_state = []
            
            for coord in backward_states[-1]:
                # Apply inverse transformation
                factor = 1.0 / (step + 2)  # Decreasing factor for each step back
                
                prev_coord = (
                    coord[0] * factor,
                    coord[1] * factor,
                    coord[2] * factor
                )
                
                prev_state.append(prev_coord)
                
            backward_states.append(prev_state)
            
        return backward_states
    
    def visualize_backward_evolution(self, prime_coords: List[int], steps: int = 5, save_path: Optional[str] = None):
        """
        Visualize the backward state evolution.
        
        Args:
            prime_coords: The list of prime coordinates
            steps: The number of backward steps
            save_path: Path to save the visualization. If None, the plot is displayed.
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Calculate backward evolution
        backward_states = self.calculate_backward_state_evolution(prime_coords, steps)
        
        # Create the figure
        fig = plt.figure(figsize=(15, 10))
        
        # Create a grid of subplots
        cols = min(3, steps + 1)
        rows = (steps + 1 + cols - 1) // cols
        
        for i, state in enumerate(backward_states):
            # Create a subplot
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            
            # Extract coordinates
            x_coords = [coord[0] for coord in state]
            y_coords = [coord[1] for coord in state]
            z_coords = [coord[2] for coord in state]
            
            # Plot the state
            ax.scatter(x_coords, y_coords, z_coords, c=range(len(state)), 
                      cmap='viridis', s=50, alpha=0.8)
            
            # Connect the points with lines
            ax.plot(x_coords, y_coords, z_coords, 'gray', alpha=0.5)
            
            # Set labels and title
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.set_zlabel('Z', fontsize=8)
            
            if i == 0:
                ax.set_title('Current State', fontsize=10)
            else:
                ax.set_title(f'Backward Step {i}', fontsize=10)
                
        # Add a suptitle
        fig.suptitle('Backward State Evolution of Quantum Fluctuations', fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def main():
    """
    Main function to demonstrate the braid neural mapping tool.
    """
    print("TIBEDO Braid Neural Mapping Tool Demo")
    print("=====================================")
    
    # Create the braid neural mapping tool
    mapper = BraidNeuralMapping(max_prime_index=50)
    
    # Create a braid structure using the first 10 primes
    print("\nCreating braid structure...")
    braid_points = mapper.create_braid_structure(num_primes=10)
    
    # Print the mapped coordinates
    print("\nMapped coordinates:")
    for i, point in enumerate(braid_points):
        print(f"Prime {mapper.primes[i]} -> ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})")
    
    # Visualize the braid structure
    print("\nVisualizing braid structure...")
    fig1 = mapper.visualize_braid_structure(num_primes=10, save_path="braid_structure.png")
    
    # Visualize the mapping process for a specific prime
    print("\nVisualizing mapping process for prime 17...")
    fig2 = mapper.visualize_prime_mapping(17, save_path="prime_mapping_process.png")
    
    # Generate linear coordinates
    print("\nGenerating linear coordinates...")
    linear_coords = mapper.generate_linear_coordinates([2, 3, 5, 7, 11])
    
    # Print the linear coordinates
    print("\nLinear coordinates:")
    for i, coord in enumerate(linear_coords):
        print(f"Prime {[2, 3, 5, 7, 11][i]} -> ({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f})")
    
    # Calculate forward entropy
    print("\nCalculating forward entropy...")
    entropy = mapper.calculate_forward_entropy([2, 3, 5, 7, 11])
    print(f"Forward entropy: {entropy:.4f}")
    
    # Calculate backward state evolution
    print("\nCalculating backward state evolution...")
    backward_states = mapper.calculate_backward_state_evolution([2, 3, 5, 7, 11], steps=3)
    
    # Visualize backward evolution
    print("\nVisualizing backward evolution...")
    fig3 = mapper.visualize_backward_evolution([2, 3, 5, 7, 11], steps=3, save_path="backward_evolution.png")
    
    print("\nDemo complete. Images saved to current directory.")
    
    # Show the plots
    plt.show()


if __name__ == "__main__":
    main()
