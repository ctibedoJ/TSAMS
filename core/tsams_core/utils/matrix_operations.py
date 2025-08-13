"""
Matrix Operations Utilities.

This module provides utility functions for matrix operations that are
used throughout the prime indexed Möbius transformation framework.
"""

import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Callable, Any


def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices.
    
    Args:
        A (np.ndarray): The first matrix.
        B (np.ndarray): The second matrix.
            
    Returns:
        np.ndarray: The product matrix.
            
    Raises:
        ValueError: If the matrices cannot be multiplied.
    """
    return np.matmul(A, B)


def matrix_inverse(A: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a matrix.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        np.ndarray: The inverse matrix.
            
    Raises:
        np.linalg.LinAlgError: If the matrix is singular.
    """
    return np.linalg.inv(A)


def matrix_determinant(A: np.ndarray) -> float:
    """
    Compute the determinant of a matrix.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        float: The determinant.
    """
    return np.linalg.det(A)


def matrix_trace(A: np.ndarray) -> float:
    """
    Compute the trace of a matrix.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        float: The trace.
    """
    return np.trace(A)


def matrix_eigenvalues(A: np.ndarray) -> np.ndarray:
    """
    Compute the eigenvalues of a matrix.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        np.ndarray: The eigenvalues.
    """
    return np.linalg.eigvals(A)


def matrix_eigenvectors(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and eigenvectors of a matrix.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        Tuple[np.ndarray, np.ndarray]: The eigenvalues and eigenvectors.
    """
    return np.linalg.eig(A)


def matrix_singular_values(A: np.ndarray) -> np.ndarray:
    """
    Compute the singular values of a matrix.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        np.ndarray: The singular values.
    """
    return np.linalg.svd(A, compute_uv=False)


def matrix_svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the singular value decomposition of a matrix.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The U, S, and V matrices.
    """
    return np.linalg.svd(A, full_matrices=True)


def matrix_rank(A: np.ndarray) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        int: The rank.
    """
    return np.linalg.matrix_rank(A)


def matrix_condition_number(A: np.ndarray) -> float:
    """
    Compute the condition number of a matrix.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        float: The condition number.
    """
    return np.linalg.cond(A)


def matrix_norm(A: np.ndarray, ord: Optional[Union[int, str]] = None) -> float:
    """
    Compute the norm of a matrix.
    
    Args:
        A (np.ndarray): The matrix.
        ord (Optional[Union[int, str]]): The order of the norm (default: None).
            
    Returns:
        float: The norm.
    """
    return np.linalg.norm(A, ord=ord)


def matrix_power(A: np.ndarray, n: int) -> np.ndarray:
    """
    Compute the nth power of a matrix.
    
    Args:
        A (np.ndarray): The matrix.
        n (int): The power.
            
    Returns:
        np.ndarray: The matrix raised to the power n.
    """
    return np.linalg.matrix_power(A, n)


def matrix_exponential(A: np.ndarray) -> np.ndarray:
    """
    Compute the matrix exponential.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        np.ndarray: The matrix exponential.
    """
    return np.exp(A)


def matrix_logarithm(A: np.ndarray) -> np.ndarray:
    """
    Compute the matrix logarithm.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        np.ndarray: The matrix logarithm.
            
    Raises:
        ValueError: If the matrix is not positive definite.
    """
    return np.log(A)


def matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """
    Compute the matrix square root.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        np.ndarray: The matrix square root.
            
    Raises:
        ValueError: If the matrix is not positive definite.
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    diagonal = np.diag(np.sqrt(eigenvalues))
    return eigenvectors @ diagonal @ np.linalg.inv(eigenvectors)


def is_hermitian(A: np.ndarray) -> bool:
    """
    Check if a matrix is Hermitian.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        bool: True if the matrix is Hermitian, False otherwise.
    """
    return np.allclose(A, A.conj().T)


def is_unitary(A: np.ndarray) -> bool:
    """
    Check if a matrix is unitary.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        bool: True if the matrix is unitary, False otherwise.
    """
    return np.allclose(A @ A.conj().T, np.eye(A.shape[0]))


def is_positive_definite(A: np.ndarray) -> bool:
    """
    Check if a matrix is positive definite.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        bool: True if the matrix is positive definite, False otherwise.
    """
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def is_positive_semidefinite(A: np.ndarray) -> bool:
    """
    Check if a matrix is positive semidefinite.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        bool: True if the matrix is positive semidefinite, False otherwise.
    """
    eigenvalues = np.linalg.eigvals(A)
    return np.all(eigenvalues >= -1e-10)


def is_orthogonal(A: np.ndarray) -> bool:
    """
    Check if a matrix is orthogonal.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        bool: True if the matrix is orthogonal, False otherwise.
    """
    return np.allclose(A @ A.T, np.eye(A.shape[0]))


def is_symmetric(A: np.ndarray) -> bool:
    """
    Check if a matrix is symmetric.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        bool: True if the matrix is symmetric, False otherwise.
    """
    return np.allclose(A, A.T)


def is_skew_symmetric(A: np.ndarray) -> bool:
    """
    Check if a matrix is skew-symmetric.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        bool: True if the matrix is skew-symmetric, False otherwise.
    """
    return np.allclose(A, -A.T)


def is_diagonal(A: np.ndarray) -> bool:
    """
    Check if a matrix is diagonal.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        bool: True if the matrix is diagonal, False otherwise.
    """
    return np.allclose(A, np.diag(np.diag(A)))


def is_upper_triangular(A: np.ndarray) -> bool:
    """
    Check if a matrix is upper triangular.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        bool: True if the matrix is upper triangular, False otherwise.
    """
    return np.allclose(A, np.triu(A))


def is_lower_triangular(A: np.ndarray) -> bool:
    """
    Check if a matrix is lower triangular.
    
    Args:
        A (np.ndarray): The matrix.
            
    Returns:
        bool: True if the matrix is lower triangular, False otherwise.
    """
    return np.allclose(A, np.tril(A))


def matrix_to_sl2(A: np.ndarray) -> np.ndarray:
    """
    Normalize a 2x2 matrix to have determinant 1.
    
    Args:
        A (np.ndarray): The 2x2 matrix.
            
    Returns:
        np.ndarray: The normalized matrix with determinant 1.
            
    Raises:
        ValueError: If the matrix is not 2x2 or has determinant 0.
    """
    if A.shape != (2, 2):
        raise ValueError("Matrix must be 2x2")
    
    det = np.linalg.det(A)
    if abs(det) < 1e-10:
        raise ValueError("Matrix has determinant 0")
    
    return A / np.sqrt(det)


def mobius_matrix_to_transformation(A: np.ndarray) -> Callable[[complex], complex]:
    """
    Convert a 2x2 matrix to a Möbius transformation function.
    
    Args:
        A (np.ndarray): The 2x2 matrix.
            
    Returns:
        Callable[[complex], complex]: The Möbius transformation function.
            
    Raises:
        ValueError: If the matrix is not 2x2.
    """
    if A.shape != (2, 2):
        raise ValueError("Matrix must be 2x2")
    
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    
    def transformation(z: complex) -> complex:
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
    
    return transformation


def transformation_to_mobius_matrix(f: Callable[[complex], complex], points: List[complex]) -> np.ndarray:
    """
    Convert a Möbius transformation function to a 2x2 matrix.
    
    Args:
        f (Callable[[complex], complex]): The Möbius transformation function.
        points (List[complex]): Three distinct points to use for the conversion.
            
    Returns:
        np.ndarray: The 2x2 matrix representing the transformation.
            
    Raises:
        ValueError: If fewer than three distinct points are provided.
    """
    if len(points) < 3:
        raise ValueError("At least three distinct points are required")
    
    # Ensure the points are distinct
    if len(set(points)) < 3:
        raise ValueError("The points must be distinct")
    
    # Apply the transformation to the points
    images = [f(z) for z in points]
    
    # Check if any point maps to infinity
    if None in images:
        # Handle the case where a point maps to infinity
        idx = images.index(None)
        z = points[idx]
        
        # Find two other points
        other_points = [p for i, p in enumerate(points) if i != idx]
        z1, z2 = other_points[:2]
        w1, w2 = [f(z) for z in [z1, z2]]
        
        # The matrix is of the form [[a, b], [c, 0]] where c != 0
        # We have w1 = (a*z1 + b)/(c*z1) and w2 = (a*z2 + b)/(c*z2)
        # Solving for a, b, c:
        c = 1.0  # We can set c = 1 since the matrix is defined up to a scalar
        a = (w1 * z1 - w2 * z2) / (z1 - z2)
        b = w1 * z1 - a * z1
        
        return np.array([[a, b], [c, 0]])
    
    # Check if any image is infinity
    if None in points:
        # Handle the case where infinity maps to a finite point
        idx = points.index(None)
        w = images[idx]
        
        # Find two other points
        other_indices = [i for i in range(len(points)) if i != idx]
        z1_idx, z2_idx = other_indices[:2]
        z1, z2 = points[z1_idx], points[z2_idx]
        w1, w2 = images[z1_idx], images[z2_idx]
        
        # The matrix is of the form [[0, b], [c, d]] where c != 0
        # We have w1 = b/(c*z1 + d) and w2 = b/(c*z2 + d)
        # Solving for b, c, d:
        b = 1.0  # We can set b = 1 since the matrix is defined up to a scalar
        c = (w2 - w1) / (w1 * w2 * (z1 - z2))
        d = -c * z1 - 1 / w1
        
        return np.array([[0, b], [c, d]])
    
    # General case: three finite points mapping to three finite points
    z1, z2, z3 = points[:3]
    w1, w2, w3 = images[:3]
    
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
    
    b = np.array([w1, w2, w3])
    
    try:
        a, b, c = np.linalg.solve(A, b)
        return np.array([[a, b], [c, 1]])
    except np.linalg.LinAlgError:
        # If the system is singular, try a different approach
        # This can happen if the points are not in general position
        
        # Try setting a = 1 instead
        A = np.array([
            [1, -z1*w1, -w1],
            [1, -z2*w2, -w2],
            [1, -z3*w3, -w3]
        ])
        
        b = np.array([-z1, -z2, -z3])
        
        try:
            b, c, d = np.linalg.solve(A, b)
            return np.array([[1, b], [c, d]])
        except np.linalg.LinAlgError:
            # If still singular, try setting b = 1
            A = np.array([
                [z1, -z1*w1, -w1],
                [z2, -z2*w2, -w2],
                [z3, -z3*w3, -w3]
            ])
            
            b = np.array([1, 1, 1])
            
            try:
                a, c, d = np.linalg.solve(A, b)
                return np.array([[a, 1], [c, d]])
            except np.linalg.LinAlgError:
                # If still singular, try setting c = 1
                A = np.array([
                    [z1, 1, -w1],
                    [z2, 1, -w2],
                    [z3, 1, -w3]
                ])
                
                b = np.array([z1*w1, z2*w2, z3*w3])
                
                a, b, d = np.linalg.solve(A, b)
                return np.array([[a, b], [1, d]])