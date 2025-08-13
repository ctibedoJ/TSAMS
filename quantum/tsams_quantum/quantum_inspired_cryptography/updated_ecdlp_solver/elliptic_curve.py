"""
Elliptic Curve Implementation for the TIBEDO Quantum-Inspired ECDLP Solver

This module provides the elliptic curve implementation used by the
quantum-inspired ECDLP solver.
"""

import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, List, Union

class ECPoint:
    """
    Elliptic curve point implementation.
    
    This class represents a point on an elliptic curve, with support for
    point addition, doubling, and scalar multiplication.
    """
    
    def __init__(self, x: Optional[int] = None, y: Optional[int] = None):
        """
        Initialize an elliptic curve point.
        
        Args:
            x: x-coordinate of the point, or None for point at infinity
            y: y-coordinate of the point, or None for point at infinity
        """
        self.x = x
        self.y = y
        self.infinity = (x is None and y is None)
    
    def __eq__(self, other: 'ECPoint') -> bool:
        """Check if two points are equal."""
        if self.infinity and other.infinity:
            return True
        if self.infinity or other.infinity:
            return False
        return self.x == other.x and self.y == other.y
    
    def __str__(self) -> str:
        """String representation of the point."""
        if self.infinity:
            return "Point at infinity"
        return f"({hex(self.x)}, {hex(self.y)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the point."""
        if self.infinity:
            return "ECPoint(infinity)"
        return f"ECPoint(x={hex(self.x)}, y={hex(self.y)})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert point to dictionary representation."""
        if self.infinity:
            return {'infinity': True}
        return {'x': self.x, 'y': self.y, 'infinity': False}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ECPoint':
        """Create point from dictionary representation."""
        if data.get('infinity', False):
            return cls()
        return cls(data['x'], data['y'])


class EllipticCurve:
    """
    Elliptic curve implementation.
    
    This class represents an elliptic curve in short Weierstrass form:
    y^2 = x^3 + ax + b (mod p)
    """
    
    def __init__(self, p: int, a: int, b: int):
        """
        Initialize an elliptic curve with the given parameters.
        
        Args:
            p: Field prime
            a: Coefficient a in the curve equation
            b: Coefficient b in the curve equation
        """
        self.p = p
        self.a = a
        self.b = b
    
    def __str__(self) -> str:
        """String representation of the curve."""
        return f"EllipticCurve(y^2 = x^3 + {hex(self.a)}x + {hex(self.b)} mod {hex(self.p)})"
    
    def contains_point(self, point: ECPoint) -> bool:
        """
        Check if a point is on the curve.
        
        Args:
            point: Point to check
            
        Returns:
            True if the point is on the curve, False otherwise
        """
        if point.infinity:
            return True
        
        left = (point.y * point.y) % self.p
        right = (point.x * point.x * point.x + self.a * point.x + self.b) % self.p
        return left == right
    
    def add_points(self, p1: ECPoint, p2: ECPoint) -> ECPoint:
        """
        Add two points on the curve.
        
        Args:
            p1: First point
            p2: Second point
            
        Returns:
            Sum of the two points
        """
        if p1.infinity:
            return p2
        if p2.infinity:
            return p1
        
        if p1.x == p2.x and p1.y != p2.y:
            return ECPoint()  # Point at infinity
        
        if p1.x == p2.x:
            # Point doubling
            lam = (3 * p1.x * p1.x + self.a) * pow(2 * p1.y, self.p - 2, self.p) % self.p
        else:
            # Point addition
            lam = (p2.y - p1.y) * pow(p2.x - p1.x, self.p - 2, self.p) % self.p
        
        x3 = (lam * lam - p1.x - p2.x) % self.p
        y3 = (lam * (p1.x - x3) - p1.y) % self.p
        
        return ECPoint(x3, y3)
    
    def negate_point(self, point: ECPoint) -> ECPoint:
        """
        Negate a point on the curve.
        
        Args:
            point: Point to negate
            
        Returns:
            Negated point
        """
        if point.infinity:
            return point
        return ECPoint(point.x, (-point.y) % self.p)
    
    def scalar_multiply(self, k: int, point: ECPoint) -> ECPoint:
        """
        Multiply a point by a scalar using double-and-add algorithm.
        
        Args:
            k: Scalar multiplier
            point: Point to multiply
            
        Returns:
            Resulting point k*point
        """
        if k == 0 or point.infinity:
            return ECPoint()  # Point at infinity
        
        if k < 0:
            return self.scalar_multiply(-k, self.negate_point(point))
        
        result = ECPoint()  # Point at infinity
        addend = point
        
        while k > 0:
            if k & 1:
                result = self.add_points(result, addend)
            addend = self.add_points(addend, addend)
            k >>= 1
        
        return result
    
    def scalar_multiply_windowed(self, k: int, point: ECPoint, window_size: int = 4) -> ECPoint:
        """
        Multiply a point by a scalar using windowed method for better performance.
        
        Args:
            k: Scalar multiplier
            point: Point to multiply
            window_size: Size of the window (typically 4 or 5)
            
        Returns:
            Resulting point k*point
        """
        if k == 0 or point.infinity:
            return ECPoint()  # Point at infinity
        
        if k < 0:
            return self.scalar_multiply_windowed(-k, self.negate_point(point), window_size)
        
        # Precompute small multiples of the point
        precomp = [ECPoint()]  # 0*P = infinity
        precomp.append(point)  # 1*P = P
        
        for i in range(2, 2**window_size):
            precomp.append(self.add_points(precomp[i-1], point))
        
        # Process scalar in windows
        result = ECPoint()  # Point at infinity
        
        # Start from the most significant bit
        bits = k.bit_length()
        for i in range(bits-1, -1, -window_size):
            # Double 'window_size' times
            for _ in range(window_size):
                result = self.add_points(result, result)
            
            # Extract window value
            if i >= window_size - 1:
                window = (k >> (i - window_size + 1)) & ((1 << window_size) - 1)
            else:
                window = k & ((1 << (i + 1)) - 1)
            
            # Add precomputed value
            if window > 0:
                result = self.add_points(result, precomp[window])
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert curve to dictionary representation."""
        return {'p': self.p, 'a': self.a, 'b': self.b}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EllipticCurve':
        """Create curve from dictionary representation."""
        return cls(data['p'], data['a'], data['b'])


# Standard elliptic curves
STANDARD_CURVES = {
    'P-256': {
        'p': 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF,
        'a': 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC,
        'b': 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B,
        'gx': 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296,
        'gy': 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5,
        'n': 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
    },
    'P-224': {
        'p': 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF000000000000000000000001,
        'a': 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFE,
        'b': 0xB4050A850C04B3ABF54132565044B0B7D7BFD8BA270B39432355FFB4,
        'gx': 0xB70E0CBD6BB4BF7F321390B94A03C1D356C21122343280D6115C1D21,
        'gy': 0xBD376388B5F723FB4C22DFE6CD4375A05A07476444D5819985007E34,
        'n': 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFF16A2E0B8F03E13DD29455C5C2A3D
    },
    'P-192': {
        'p': 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFF,
        'a': 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFC,
        'b': 0x64210519E59C80E70FA7E9AB72243049FEB8DEECC146B9B1,
        'gx': 0x188DA80EB03090F67CBF20EB43A18800F4FF0AFD82FF1012,
        'gy': 0x07192B95FFC8DA78631011ED6B24CDD573F977A11E794811,
        'n': 0xFFFFFFFFFFFFFFFFFFFFFFFF99DEF836146BC9B1B4D22831
    }
}

def get_standard_curve(name: str) -> Tuple[EllipticCurve, ECPoint]:
    """
    Get a standard elliptic curve and its generator point.
    
    Args:
        name: Name of the standard curve ('P-256', 'P-224', or 'P-192')
        
    Returns:
        Tuple of (curve, generator_point)
    """
    if name not in STANDARD_CURVES:
        raise ValueError(f"Unknown curve: {name}")
    
    params = STANDARD_CURVES[name]
    curve = EllipticCurve(params['p'], params['a'], params['b'])
    generator = ECPoint(params['gx'], params['gy'])
    
    return curve, generator