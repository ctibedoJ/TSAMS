"""
Elliptic Curve implementation.

This module provides an implementation of elliptic curves over finite fields,
which are used in the ECDLP solver.
"""

from typing import Tuple, List, Dict, Optional, Union
import numpy as np


class Point:
    """
    A class representing a point on an elliptic curve.
    
    Attributes:
        x (int): The x-coordinate of the point.
        y (int): The y-coordinate of the point.
        infinity (bool): Whether the point is the point at infinity.
    """
    
    def __init__(self, x: int = None, y: int = None, infinity: bool = False):
        """
        Initialize a point on an elliptic curve.
        
        Args:
            x (int, optional): The x-coordinate of the point. Defaults to None.
            y (int, optional): The y-coordinate of the point. Defaults to None.
            infinity (bool, optional): Whether the point is the point at infinity.
                Defaults to False.
        """
        self.x = x
        self.y = y
        self.infinity = infinity
    
    def __eq__(self, other: 'Point') -> bool:
        """
        Check if two points are equal.
        
        Args:
            other (Point): The other point.
        
        Returns:
            bool: True if the points are equal, False otherwise.
        """
        if not isinstance(other, Point):
            return False
        
        if self.infinity and other.infinity:
            return True
        
        if self.infinity or other.infinity:
            return False
        
        return self.x == other.x and self.y == other.y
    
    def __ne__(self, other: 'Point') -> bool:
        """
        Check if two points are not equal.
        
        Args:
            other (Point): The other point.
        
        Returns:
            bool: True if the points are not equal, False otherwise.
        """
        return not self.__eq__(other)
    
    def __str__(self) -> str:
        """
        Return a string representation of the point.
        
        Returns:
            str: A string representation of the point.
        """
        if self.infinity:
            return "Point(infinity)"
        return f"Point({self.x}, {self.y})"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the point.
        
        Returns:
            str: A string representation of the point.
        """
        return self.__str__()


class EllipticCurve:
    """
    A class representing an elliptic curve over a finite field.
    
    The curve is defined by the equation y^2 = x^3 + ax + b (mod p).
    
    Attributes:
        a (int): The coefficient a in the curve equation.
        b (int): The coefficient b in the curve equation.
        p (int): The prime modulus of the finite field.
        generator (Point): The generator point of the curve.
        order (int): The order of the generator point.
    """
    
    def __init__(self, a: int, b: int, p: int, generator: Tuple[int, int] = None, order: int = None):
        """
        Initialize an elliptic curve.
        
        Args:
            a (int): The coefficient a in the curve equation.
            b (int): The coefficient b in the curve equation.
            p (int): The prime modulus of the finite field.
            generator (Tuple[int, int], optional): The generator point of the curve.
                Defaults to None.
            order (int, optional): The order of the generator point. Defaults to None.
        """
        self.a = a
        self.b = b
        self.p = p
        
        if generator is not None:
            self.generator = Point(generator[0], generator[1])
        else:
            self.generator = None
        
        self.order = order
    
    def is_on_curve(self, point: Union[Point, Tuple[int, int]]) -> bool:
        """
        Check if a point is on the curve.
        
        Args:
            point (Point or Tuple[int, int]): The point to check.
        
        Returns:
            bool: True if the point is on the curve, False otherwise.
        """
        if isinstance(point, tuple):
            point = Point(point[0], point[1])
        
        if point.infinity:
            return True
        
        # Check if the point satisfies the curve equation: y^2 = x^3 + ax + b (mod p)
        left = (point.y * point.y) % self.p
        right = (point.x * point.x * point.x + self.a * point.x + self.b) % self.p
        
        return left == right
    
    def add(self, p1: Point, p2: Point) -> Point:
        """
        Add two points on the curve.
        
        Args:
            p1 (Point): The first point.
            p2 (Point): The second point.
        
        Returns:
            Point: The sum of the two points.
        """
        # Handle the point at infinity
        if p1.infinity:
            return Point(p2.x, p2.y, p2.infinity)
        if p2.infinity:
            return Point(p1.x, p1.y, p1.infinity)
        
        # Handle the case where the points are inverses of each other
        if p1.x == p2.x and p1.y != p2.y:
            return Point(infinity=True)
        
        # Compute the slope of the line through the two points
        if p1.x == p2.x and p1.y == p2.y:
            # The points are the same, so compute the tangent line
            # slope = (3 * x^2 + a) / (2 * y) mod p
            numerator = (3 * p1.x * p1.x + self.a) % self.p
            denominator = (2 * p1.y) % self.p
            
            # Compute the modular inverse of the denominator
            denominator_inv = pow(denominator, self.p - 2, self.p)
            slope = (numerator * denominator_inv) % self.p
        else:
            # The points are different, so compute the secant line
            # slope = (y2 - y1) / (x2 - x1) mod p
            numerator = (p2.y - p1.y) % self.p
            denominator = (p2.x - p1.x) % self.p
            
            # Compute the modular inverse of the denominator
            denominator_inv = pow(denominator, self.p - 2, self.p)
            slope = (numerator * denominator_inv) % self.p
        
        # Compute the coordinates of the sum
        # x3 = slope^2 - x1 - x2 mod p
        x3 = (slope * slope - p1.x - p2.x) % self.p
        
        # y3 = slope * (x1 - x3) - y1 mod p
        y3 = (slope * (p1.x - x3) - p1.y) % self.p
        
        return Point(x3, y3)
    
    def multiply(self, point: Point, scalar: int) -> Point:
        """
        Multiply a point by a scalar.
        
        Args:
            point (Point): The point to multiply.
            scalar (int): The scalar to multiply by.
        
        Returns:
            Point: The product of the point and the scalar.
        """
        # Handle the point at infinity
        if point.infinity:
            return Point(infinity=True)
        
        # Handle the case where the scalar is 0
        if scalar == 0:
            return Point(infinity=True)
        
        # Handle negative scalars
        if scalar < 0:
            scalar = -scalar
            point = Point(point.x, (-point.y) % self.p)
        
        # Use the double-and-add algorithm for efficient multiplication
        result = Point(infinity=True)
        addend = Point(point.x, point.y)
        
        while scalar:
            if scalar & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            scalar >>= 1
        
        return result
    
    def __str__(self) -> str:
        """
        Return a string representation of the curve.
        
        Returns:
            str: A string representation of the curve.
        """
        return f"EllipticCurve(y^2 = x^3 + {self.a}x + {self.b} mod {self.p})"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the curve.
        
        Returns:
            str: A string representation of the curve.
        """
        return self.__str__()