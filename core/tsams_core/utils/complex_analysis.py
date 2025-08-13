"""
Complex Analysis Utilities.

This module provides utility functions for complex analysis operations that are
used throughout the prime indexed Möbius transformation framework.
"""

import numpy as np
import cmath
from typing import List, Dict, Tuple, Union, Optional, Callable, Any


def mobius_transformation(z: complex, a: complex, b: complex, c: complex, d: complex) -> Optional[complex]:
    """
    Apply a Möbius transformation to a complex number.
    
    Args:
        z (complex): The complex number to transform.
        a (complex): The coefficient a.
        b (complex): The coefficient b.
        c (complex): The coefficient c.
        d (complex): The coefficient d.
            
    Returns:
        Optional[complex]: The transformed complex number, or None if it maps to infinity.
    """
    if z is None:  # z = ∞
        if abs(c) < 1e-10:
            return None  # ∞ maps to ∞
        else:
            return a / c  # ∞ maps to a/c
    
    numerator = a * z + b
    denominator = c * z + d
    
    if abs(denominator) < 1e-10:
        return None  # Maps to ∞
    
    return numerator / denominator


def fixed_points(a: complex, b: complex, c: complex, d: complex) -> List[complex]:
    """
    Compute the fixed points of a Möbius transformation.
    
    Args:
        a (complex): The coefficient a.
        b (complex): The coefficient b.
        c (complex): The coefficient c.
        d (complex): The coefficient d.
            
    Returns:
        List[complex]: The fixed points of the transformation.
    """
    # A fixed point z satisfies (az + b)/(cz + d) = z
    # This gives the quadratic equation: cz^2 + (d - a)z - b = 0
    
    if abs(c) < 1e-10:  # c = 0
        if abs(a - d) < 1e-10:  # a = d
            # Every point is fixed (identity transformation)
            return [complex('inf')]  # Representing the entire complex plane
        else:
            # One fixed point: b/(a - d)
            return [b / (a - d)]
    
    # Solve the quadratic equation
    discriminant = (d - a)**2 + 4 * b * c
    
    if abs(discriminant) < 1e-10:  # Discriminant = 0
        # One fixed point of multiplicity 2
        return [(a - d) / (2 * c)]
    
    # Two distinct fixed points
    sqrt_discriminant = cmath.sqrt(discriminant)
    z1 = ((a - d) + sqrt_discriminant) / (2 * c)
    z2 = ((a - d) - sqrt_discriminant) / (2 * c)
    
    return [z1, z2]


def classify_mobius_transformation(a: complex, b: complex, c: complex, d: complex) -> str:
    """
    Classify a Möbius transformation based on its fixed points.
    
    Args:
        a (complex): The coefficient a.
        b (complex): The coefficient b.
        c (complex): The coefficient c.
        d (complex): The coefficient d.
            
    Returns:
        str: The classification of the transformation ('elliptic', 'parabolic', 'hyperbolic', or 'loxodromic').
    """
    # Compute the trace of the matrix
    trace = a + d
    
    # Normalize the matrix to have determinant 1
    det = a * d - b * c
    if abs(det - 1.0) > 1e-10:
        factor = 1.0 / cmath.sqrt(det)
        a *= factor
        b *= factor
        c *= factor
        d *= factor
        trace = a + d
    
    # Classify based on the trace
    trace_squared_norm = (trace * trace.conjugate()).real
    
    if abs(trace_squared_norm - 4) < 1e-10:
        return 'parabolic'
    elif trace.imag == 0 and 0 <= trace_squared_norm < 4:
        return 'elliptic'
    elif trace.imag == 0 and trace_squared_norm > 4:
        return 'hyperbolic'
    else:
        return 'loxodromic'


def cross_ratio(z1: complex, z2: complex, z3: complex, z4: complex) -> complex:
    """
    Compute the cross-ratio of four complex numbers.
    
    Args:
        z1 (complex): The first complex number.
        z2 (complex): The second complex number.
        z3 (complex): The third complex number.
        z4 (complex): The fourth complex number.
            
    Returns:
        complex: The cross-ratio (z1 - z3)(z2 - z4)/((z1 - z4)(z2 - z3)).
    """
    return ((z1 - z3) * (z2 - z4)) / ((z1 - z4) * (z2 - z3))


def mobius_from_points(z1: complex, z2: complex, z3: complex, w1: complex, w2: complex, w3: complex) -> Tuple[complex, complex, complex, complex]:
    """
    Find the Möbius transformation that maps three points to three other points.
    
    Args:
        z1 (complex): The first source point.
        z2 (complex): The second source point.
        z3 (complex): The third source point.
        w1 (complex): The first target point.
        w2 (complex): The second target point.
        w3 (complex): The third target point.
            
    Returns:
        Tuple[complex, complex, complex, complex]: The coefficients (a, b, c, d) of the Möbius transformation.
    """
    # Ensure the points are distinct
    if len(set([z1, z2, z3])) < 3 or len(set([w1, w2, w3])) < 3:
        raise ValueError("The points must be distinct")
    
    # First, find the Möbius transformation that maps (z1, z2, z3) to (0, 1, ∞)
    def to_standard(z: complex, z1: complex, z2: complex, z3: complex) -> complex:
        return cross_ratio(z, z3, z1, z2)
    
    # Then, find the Möbius transformation that maps (0, 1, ∞) to (w1, w2, w3)
    def from_standard(z: complex, w1: complex, w2: complex, w3: complex) -> complex:
        return (w1 * (w3 - w2) * z + w2 * w3 * (1 - z)) / (w3 * (1 - z) + w2 * z)
    
    # Compose the two transformations
    def transformation(z: complex) -> complex:
        z_std = to_standard(z, z1, z2, z3)
        return from_standard(z_std, w1, w2, w3)
    
    # Compute the coefficients of the composed transformation
    # We can do this by applying the transformation to three points and solving for the coefficients
    test_points = [0, 1, complex('inf')]
    images = [transformation(z) for z in test_points]
    
    # Handle the case where one of the images is infinity
    if None in images:
        idx = images.index(None)
        z = test_points[idx]
        
        # Find two other points
        other_points = [p for i, p in enumerate(test_points) if i != idx]
        z1, z2 = other_points
        w1, w2 = [transformation(z) for z in [z1, z2]]
        
        # The matrix is of the form [[a, b], [c, 0]] where c != 0
        # We have w1 = (a*z1 + b)/(c*z1) and w2 = (a*z2 + b)/(c*z2)
        # Solving for a, b, c:
        c = 1.0  # We can set c = 1 since the matrix is defined up to a scalar
        a = (w1 * z1 - w2 * z2) / (z1 - z2)
        b = w1 * z1 - a * z1
        d = 0.0
        
        return (a, b, c, d)
    
    # Handle the case where one of the test points is infinity
    if complex('inf') in test_points:
        idx = test_points.index(complex('inf'))
        w = images[idx]
        
        # Find two other points
        other_indices = [i for i in range(len(test_points)) if i != idx]
        z1_idx, z2_idx = other_indices
        z1, z2 = test_points[z1_idx], test_points[z2_idx]
        w1, w2 = images[z1_idx], images[z2_idx]
        
        # The matrix is of the form [[0, b], [c, d]] where c != 0
        # We have w1 = b/(c*z1 + d) and w2 = b/(c*z2 + d)
        # Solving for b, c, d:
        b = 1.0  # We can set b = 1 since the matrix is defined up to a scalar
        c = (w2 - w1) / (w1 * w2 * (z1 - z2))
        d = -c * z1 - 1 / w1
        a = 0.0
        
        return (a, b, c, d)
    
    # General case: three finite points mapping to three finite points
    z1, z2, z3 = test_points
    w1, w2, w3 = images
    
    # We have w_i = (a*z_i + b)/(c*z_i + d) for i = 1, 2, 3
    # This gives us three equations in four unknowns (a, b, c, d)
    # We can set d = 1 since the matrix is defined up to a scalar
    
    # Rearranging, we get c*z_i*w_i + d*w_i = a*z_i + b
    # or c*z_i*w_i + w_i = a*z_i + b (setting d = 1)
    # This gives us three equations in three unknowns (a, b, c)
    
    A = np.array([
        [z1, 1, -z1*w1],
        [z2, 1, -z2*w2],
        [z3, 1, -z3*w3]
    ])
    
    b_vec = np.array([w1, w2, w3])
    
    try:
        a, b, c = np.linalg.solve(A, b_vec)
        return (a, b, c, 1.0)
    except np.linalg.LinAlgError:
        # If the system is singular, try a different approach
        # This can happen if the points are not in general position
        
        # Try setting a = 1 instead
        A = np.array([
            [1, -z1*w1, -w1],
            [1, -z2*w2, -w2],
            [1, -z3*w3, -w3]
        ])
        
        b_vec = np.array([-z1, -z2, -z3])
        
        try:
            b, c, d = np.linalg.solve(A, b_vec)
            return (1.0, b, c, d)
        except np.linalg.LinAlgError:
            # If still singular, try setting b = 1
            A = np.array([
                [z1, -z1*w1, -w1],
                [z2, -z2*w2, -w2],
                [z3, -z3*w3, -w3]
            ])
            
            b_vec = np.array([1, 1, 1])
            
            try:
                a, c, d = np.linalg.solve(A, b_vec)
                return (a, 1.0, c, d)
            except np.linalg.LinAlgError:
                # If still singular, try setting c = 1
                A = np.array([
                    [z1, 1, -w1],
                    [z2, 1, -w2],
                    [z3, 1, -w3]
                ])
                
                b_vec = np.array([z1*w1, z2*w2, z3*w3])
                
                a, b, d = np.linalg.solve(A, b_vec)
                return (a, b, 1.0, d)


def riemann_sphere_to_cartesian(z: complex) -> Tuple[float, float, float]:
    """
    Convert a point on the Riemann sphere to Cartesian coordinates.
    
    Args:
        z (complex): The complex number, or None for infinity.
            
    Returns:
        Tuple[float, float, float]: The Cartesian coordinates (x, y, z).
    """
    if z is None:  # z = ∞
        return (0.0, 0.0, 1.0)
    
    x = z.real
    y = z.imag
    denom = 1.0 + x*x + y*y
    
    return (2*x / denom, 2*y / denom, (denom - 2.0) / denom)


def cartesian_to_riemann_sphere(x: float, y: float, z: float) -> Optional[complex]:
    """
    Convert Cartesian coordinates to a point on the Riemann sphere.
    
    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        z (float): The z-coordinate.
            
    Returns:
        Optional[complex]: The complex number, or None for infinity.
    """
    if abs(1.0 - z) < 1e-10:
        return None  # North pole = ∞
    
    return complex(x / (1.0 - z), y / (1.0 - z))


def stereographic_projection(z: complex) -> Tuple[float, float, float]:
    """
    Compute the stereographic projection of a complex number onto the unit sphere.
    
    Args:
        z (complex): The complex number, or None for infinity.
            
    Returns:
        Tuple[float, float, float]: The Cartesian coordinates (x, y, z) on the unit sphere.
    """
    return riemann_sphere_to_cartesian(z)


def inverse_stereographic_projection(x: float, y: float, z: float) -> Optional[complex]:
    """
    Compute the inverse stereographic projection of a point on the unit sphere.
    
    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        z (float): The z-coordinate.
            
    Returns:
        Optional[complex]: The complex number, or None for infinity.
    """
    return cartesian_to_riemann_sphere(x, y, z)


def mobius_to_isometry(a: complex, b: complex, c: complex, d: complex) -> np.ndarray:
    """
    Convert a Möbius transformation to an isometry of the unit sphere.
    
    Args:
        a (complex): The coefficient a.
        b (complex): The coefficient b.
        c (complex): The coefficient c.
        d (complex): The coefficient d.
            
    Returns:
        np.ndarray: The 3x3 matrix representing the isometry.
    """
    # Normalize the matrix to have determinant 1
    det = a * d - b * c
    if abs(det - 1.0) > 1e-10:
        factor = 1.0 / cmath.sqrt(det)
        a *= factor
        b *= factor
        c *= factor
        d *= factor
    
    # Construct the isometry matrix
    matrix = np.zeros((3, 3), dtype=complex)
    
    matrix[0, 0] = a * d.conjugate() + b * c.conjugate()
    matrix[0, 1] = 1j * (a * d.conjugate() - b * c.conjugate())
    matrix[0, 2] = 2 * b * d.conjugate()
    
    matrix[1, 0] = -1j * (a * d.conjugate() - b * c.conjugate())
    matrix[1, 1] = a * d.conjugate() + b * c.conjugate()
    matrix[1, 2] = -2j * b * d.conjugate()
    
    matrix[2, 0] = 2 * a * c.conjugate()
    matrix[2, 1] = 2j * a * c.conjugate()
    matrix[2, 2] = a * a.conjugate() - b * b.conjugate() - c * c.conjugate() + d * d.conjugate()
    
    # Convert to real matrix
    return np.real(matrix)


def isometry_to_mobius(matrix: np.ndarray) -> Tuple[complex, complex, complex, complex]:
    """
    Convert an isometry of the unit sphere to a Möbius transformation.
    
    Args:
        matrix (np.ndarray): The 3x3 matrix representing the isometry.
            
    Returns:
        Tuple[complex, complex, complex, complex]: The coefficients (a, b, c, d) of the Möbius transformation.
    """
    # Extract the components of the isometry matrix
    a00 = matrix[0, 0]
    a01 = matrix[0, 1]
    a02 = matrix[0, 2]
    a10 = matrix[1, 0]
    a11 = matrix[1, 1]
    a12 = matrix[1, 2]
    a20 = matrix[2, 0]
    a21 = matrix[2, 1]
    a22 = matrix[2, 2]
    
    # Compute the coefficients of the Möbius transformation
    a = complex(a00 + a11 + a22 + 1j * (a01 - a10)) / 2
    b = complex(a02 - 1j * a12) / 2
    c = complex(a20 - 1j * a21) / 2
    d = complex(a00 + a11 - a22 + 1j * (a01 - a10)) / 2
    
    # Normalize the coefficients to have determinant 1
    det = a * d - b * c
    if abs(det - 1.0) > 1e-10:
        factor = 1.0 / cmath.sqrt(det)
        a *= factor
        b *= factor
        c *= factor
        d *= factor
    
    return (a, b, c, d)


def hyperbolic_distance(z1: complex, z2: complex) -> float:
    """
    Compute the hyperbolic distance between two points in the upper half-plane.
    
    Args:
        z1 (complex): The first point (must have positive imaginary part).
        z2 (complex): The second point (must have positive imaginary part).
            
    Returns:
        float: The hyperbolic distance.
            
    Raises:
        ValueError: If either point is not in the upper half-plane.
    """
    if z1.imag <= 0 or z2.imag <= 0:
        raise ValueError("Points must be in the upper half-plane")
    
    # Compute the hyperbolic distance using the formula
    # d(z1, z2) = arcosh(1 + |z1 - z2|^2 / (2 * Im(z1) * Im(z2)))
    numerator = abs(z1 - z2)**2
    denominator = 2 * z1.imag * z2.imag
    
    return np.arccosh(1 + numerator / denominator)


def poincare_disk_to_upper_half_plane(z: complex) -> complex:
    """
    Convert a point in the Poincaré disk to the upper half-plane.
    
    Args:
        z (complex): The point in the Poincaré disk (|z| < 1).
            
    Returns:
        complex: The corresponding point in the upper half-plane.
            
    Raises:
        ValueError: If the point is not in the Poincaré disk.
    """
    if abs(z) >= 1:
        raise ValueError("Point must be in the Poincaré disk (|z| < 1)")
    
    # The transformation is T(z) = i * (1 + z) / (1 - z)
    return 1j * (1 + z) / (1 - z)


def upper_half_plane_to_poincare_disk(z: complex) -> complex:
    """
    Convert a point in the upper half-plane to the Poincaré disk.
    
    Args:
        z (complex): The point in the upper half-plane (Im(z) > 0).
            
    Returns:
        complex: The corresponding point in the Poincaré disk.
            
    Raises:
        ValueError: If the point is not in the upper half-plane.
    """
    if z.imag <= 0:
        raise ValueError("Point must be in the upper half-plane (Im(z) > 0)")
    
    # The transformation is T^(-1)(z) = (z - i) / (z + i)
    return (z - 1j) / (z + 1j)


def klein_disk_to_poincare_disk(z: complex) -> complex:
    """
    Convert a point in the Klein disk to the Poincaré disk.
    
    Args:
        z (complex): The point in the Klein disk (|z| < 1).
            
    Returns:
        complex: The corresponding point in the Poincaré disk.
            
    Raises:
        ValueError: If the point is not in the Klein disk.
    """
    if abs(z) >= 1:
        raise ValueError("Point must be in the Klein disk (|z| < 1)")
    
    # The transformation is T(z) = z / (1 + sqrt(1 - |z|^2))
    denominator = 1 + np.sqrt(1 - abs(z)**2)
    return z / denominator


def poincare_disk_to_klein_disk(z: complex) -> complex:
    """
    Convert a point in the Poincaré disk to the Klein disk.
    
    Args:
        z (complex): The point in the Poincaré disk (|z| < 1).
            
    Returns:
        complex: The corresponding point in the Klein disk.
            
    Raises:
        ValueError: If the point is not in the Poincaré disk.
    """
    if abs(z) >= 1:
        raise ValueError("Point must be in the Poincaré disk (|z| < 1)")
    
    # The transformation is T^(-1)(z) = 2z / (1 + |z|^2)
    denominator = 1 + abs(z)**2
    return 2 * z / denominator


def mobius_invariant(z1: complex, z2: complex, z3: complex, z4: complex) -> complex:
    """
    Compute the Möbius invariant of four points.
    
    The Möbius invariant is the cross-ratio [z1, z2, z3, z4], which is invariant
    under Möbius transformations.
    
    Args:
        z1 (complex): The first point.
        z2 (complex): The second point.
        z3 (complex): The third point.
        z4 (complex): The fourth point.
            
    Returns:
        complex: The Möbius invariant.
    """
    return cross_ratio(z1, z2, z3, z4)


def schwarzian_derivative(f: Callable[[complex], complex], z: complex, h: float = 1e-6) -> complex:
    """
    Compute the Schwarzian derivative of a function at a point.
    
    The Schwarzian derivative is defined as:
    S(f)(z) = f'''(z)/f'(z) - (3/2) * (f''(z)/f'(z))^2
    
    Args:
        f (Callable[[complex], complex]): The function.
        z (complex): The point.
        h (float): The step size for numerical differentiation (default: 1e-6).
            
    Returns:
        complex: The Schwarzian derivative.
    """
    # Compute the derivatives using finite differences
    f_z = f(z)
    f_z_plus_h = f(z + h)
    f_z_minus_h = f(z - h)
    f_z_plus_2h = f(z + 2*h)
    f_z_minus_2h = f(z - 2*h)
    
    # First derivative: f'(z) ≈ (f(z+h) - f(z-h)) / (2h)
    f_prime = (f_z_plus_h - f_z_minus_h) / (2*h)
    
    # Second derivative: f''(z) ≈ (f(z+h) - 2f(z) + f(z-h)) / h^2
    f_double_prime = (f_z_plus_h - 2*f_z + f_z_minus_h) / (h*h)
    
    # Third derivative: f'''(z) ≈ (f(z+2h) - 2f(z+h) + 2f(z-h) - f(z-2h)) / (2h^3)
    f_triple_prime = (f_z_plus_2h - 2*f_z_plus_h + 2*f_z_minus_h - f_z_minus_2h) / (2*h*h*h)
    
    # Compute the Schwarzian derivative
    term1 = f_triple_prime / f_prime
    term2 = (3/2) * (f_double_prime / f_prime)**2
    
    return term1 - term2


def is_mobius_transformation(f: Callable[[complex], complex], points: List[complex], tolerance: float = 1e-6) -> bool:
    """
    Check if a function is a Möbius transformation.
    
    A function is a Möbius transformation if and only if its Schwarzian derivative
    is identically zero.
    
    Args:
        f (Callable[[complex], complex]): The function.
        points (List[complex]): Points to check the Schwarzian derivative at.
        tolerance (float): The tolerance for numerical comparisons (default: 1e-6).
            
    Returns:
        bool: True if the function is a Möbius transformation, False otherwise.
    """
    for z in points:
        schwarzian = schwarzian_derivative(f, z)
        if abs(schwarzian) > tolerance:
            return False
    
    return True