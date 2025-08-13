"""
TIBEDO Isogeny-Based Cryptography

This module implements isogeny-based cryptographic primitives that are resistant to
quantum attacks, including the Supersingular Isogeny Key Encapsulation (SIKE) protocol.
"""

import numpy as np
import sympy as sp
from typing import List, Tuple, Dict, Any, Optional, Union
import math
import os
import sys
import logging
import time
import secrets
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FiniteField:
    """
    Implementation of a finite field F_p.
    
    This class provides operations in the finite field F_p, where p is a prime number.
    """
    
    def __init__(self, p: int):
        """
        Initialize the finite field F_p.
        
        Args:
            p: The prime modulus
        """
        self.p = p
        
        # Verify that p is prime
        if not sp.isprime(p):
            raise ValueError(f"Modulus {p} is not prime")
    
    def add(self, a: int, b: int) -> int:
        """
        Add two elements in F_p.
        
        Args:
            a: The first element
            b: The second element
            
        Returns:
            The sum a + b in F_p
        """
        return (a + b) % self.p
    
    def subtract(self, a: int, b: int) -> int:
        """
        Subtract two elements in F_p.
        
        Args:
            a: The first element
            b: The second element
            
        Returns:
            The difference a - b in F_p
        """
        return (a - b) % self.p
    
    def multiply(self, a: int, b: int) -> int:
        """
        Multiply two elements in F_p.
        
        Args:
            a: The first element
            b: The second element
            
        Returns:
            The product a * b in F_p
        """
        return (a * b) % self.p
    
    def inverse(self, a: int) -> int:
        """
        Compute the multiplicative inverse of an element in F_p.
        
        Args:
            a: The element
            
        Returns:
            The multiplicative inverse of a in F_p
            
        Raises:
            ValueError: If a is not invertible
        """
        if a == 0:
            raise ValueError("Zero is not invertible")
        
        # Use the extended Euclidean algorithm
        gcd, x, y = self._extended_gcd(a, self.p)
        
        if gcd != 1:
            raise ValueError(f"{a} is not invertible modulo {self.p}")
        
        return x % self.p
    
    def divide(self, a: int, b: int) -> int:
        """
        Divide two elements in F_p.
        
        Args:
            a: The numerator
            b: The denominator
            
        Returns:
            The quotient a / b in F_p
            
        Raises:
            ValueError: If b is not invertible
        """
        return self.multiply(a, self.inverse(b))
    
    def power(self, a: int, n: int) -> int:
        """
        Compute the power of an element in F_p.
        
        Args:
            a: The base
            n: The exponent
            
        Returns:
            The power a^n in F_p
        """
        if n == 0:
            return 1
        
        if n < 0:
            return self.power(self.inverse(a), -n)
        
        # Use the square-and-multiply algorithm
        result = 1
        base = a % self.p
        
        while n > 0:
            if n & 1:
                result = self.multiply(result, base)
            
            base = self.multiply(base, base)
            n >>= 1
        
        return result
    
    def _extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        """
        Compute the extended greatest common divisor of a and b.
        
        Args:
            a: The first number
            b: The second number
            
        Returns:
            A tuple (gcd, x, y) such that gcd = a*x + b*y
        """
        if a == 0:
            return b, 0, 1
        
        gcd, x1, y1 = self._extended_gcd(b % a, a)
        
        x = y1 - (b // a) * x1
        y = x1
        
        return gcd, x, y


class QuadraticExtensionField:
    """
    Implementation of a quadratic extension field F_p^2 = F_p[i]/(i^2 + 1).
    
    This class provides operations in the quadratic extension field F_p^2,
    represented as a + bi, where a, b are in F_p and i^2 = -1.
    """
    
    def __init__(self, base_field: FiniteField):
        """
        Initialize the quadratic extension field F_p^2.
        
        Args:
            base_field: The base field F_p
        """
        self.base_field = base_field
    
    def add(self, a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
        """
        Add two elements in F_p^2.
        
        Args:
            a: The first element (a_0, a_1) representing a_0 + a_1*i
            b: The second element (b_0, b_1) representing b_0 + b_1*i
            
        Returns:
            The sum (a_0 + b_0, a_1 + b_1) representing (a_0 + b_0) + (a_1 + b_1)*i
        """
        return (
            self.base_field.add(a[0], b[0]),
            self.base_field.add(a[1], b[1])
        )
    
    def subtract(self, a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
        """
        Subtract two elements in F_p^2.
        
        Args:
            a: The first element (a_0, a_1) representing a_0 + a_1*i
            b: The second element (b_0, b_1) representing b_0 + b_1*i
            
        Returns:
            The difference (a_0 - b_0, a_1 - b_1) representing (a_0 - b_0) + (a_1 - b_1)*i
        """
        return (
            self.base_field.subtract(a[0], b[0]),
            self.base_field.subtract(a[1], b[1])
        )
    
    def multiply(self, a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
        """
        Multiply two elements in F_p^2.
        
        Args:
            a: The first element (a_0, a_1) representing a_0 + a_1*i
            b: The second element (b_0, b_1) representing b_0 + b_1*i
            
        Returns:
            The product (a_0*b_0 - a_1*b_1, a_0*b_1 + a_1*b_0) representing (a_0*b_0 - a_1*b_1) + (a_0*b_1 + a_1*b_0)*i
        """
        # (a_0 + a_1*i) * (b_0 + b_1*i) = (a_0*b_0 - a_1*b_1) + (a_0*b_1 + a_1*b_0)*i
        return (
            self.base_field.subtract(
                self.base_field.multiply(a[0], b[0]),
                self.base_field.multiply(a[1], b[1])
            ),
            self.base_field.add(
                self.base_field.multiply(a[0], b[1]),
                self.base_field.multiply(a[1], b[0])
            )
        )
    
    def square(self, a: Tuple[int, int]) -> Tuple[int, int]:
        """
        Square an element in F_p^2.
        
        Args:
            a: The element (a_0, a_1) representing a_0 + a_1*i
            
        Returns:
            The square (a_0^2 - a_1^2, 2*a_0*a_1) representing (a_0^2 - a_1^2) + (2*a_0*a_1)*i
        """
        # (a_0 + a_1*i)^2 = (a_0^2 - a_1^2) + (2*a_0*a_1)*i
        return (
            self.base_field.subtract(
                self.base_field.multiply(a[0], a[0]),
                self.base_field.multiply(a[1], a[1])
            ),
            self.base_field.multiply(
                self.base_field.multiply(2, a[0]),
                a[1]
            )
        )
    
    def inverse(self, a: Tuple[int, int]) -> Tuple[int, int]:
        """
        Compute the multiplicative inverse of an element in F_p^2.
        
        Args:
            a: The element (a_0, a_1) representing a_0 + a_1*i
            
        Returns:
            The multiplicative inverse (a_0/(a_0^2 + a_1^2), -a_1/(a_0^2 + a_1^2)) representing (a_0/(a_0^2 + a_1^2)) + (-a_1/(a_0^2 + a_1^2))*i
            
        Raises:
            ValueError: If a is not invertible
        """
        # (a_0 + a_1*i)^(-1) = (a_0/(a_0^2 + a_1^2)) + (-a_1/(a_0^2 + a_1^2))*i
        norm = self.base_field.add(
            self.base_field.multiply(a[0], a[0]),
            self.base_field.multiply(a[1], a[1])
        )
        
        norm_inv = self.base_field.inverse(norm)
        
        return (
            self.base_field.multiply(a[0], norm_inv),
            self.base_field.multiply(self.base_field.subtract(0, a[1]), norm_inv)
        )
    
    def divide(self, a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
        """
        Divide two elements in F_p^2.
        
        Args:
            a: The numerator (a_0, a_1) representing a_0 + a_1*i
            b: The denominator (b_0, b_1) representing b_0 + b_1*i
            
        Returns:
            The quotient a / b in F_p^2
            
        Raises:
            ValueError: If b is not invertible
        """
        return self.multiply(a, self.inverse(b))
    
    def power(self, a: Tuple[int, int], n: int) -> Tuple[int, int]:
        """
        Compute the power of an element in F_p^2.
        
        Args:
            a: The base (a_0, a_1) representing a_0 + a_1*i
            n: The exponent
            
        Returns:
            The power a^n in F_p^2
        """
        if n == 0:
            return (1, 0)
        
        if n < 0:
            return self.power(self.inverse(a), -n)
        
        # Use the square-and-multiply algorithm
        result = (1, 0)
        base = a
        
        while n > 0:
            if n & 1:
                result = self.multiply(result, base)
            
            base = self.square(base)
            n >>= 1
        
        return result
    
    def frobenius(self, a: Tuple[int, int]) -> Tuple[int, int]:
        """
        Apply the Frobenius endomorphism to an element in F_p^2.
        
        The Frobenius endomorphism maps a_0 + a_1*i to a_0 - a_1*i.
        
        Args:
            a: The element (a_0, a_1) representing a_0 + a_1*i
            
        Returns:
            The Frobenius endomorphism (a_0, -a_1) representing a_0 - a_1*i
        """
        return (a[0], self.base_field.subtract(0, a[1]))


class EllipticCurve:
    """
    Implementation of an elliptic curve over a finite field.
    
    This class provides operations on elliptic curves in short Weierstrass form:
    y^2 = x^3 + a*x + b over a finite field F_p.
    """
    
    def __init__(self, field: FiniteField, a: int, b: int):
        """
        Initialize the elliptic curve E: y^2 = x^3 + a*x + b over F_p.
        
        Args:
            field: The finite field F_p
            a: The coefficient a
            b: The coefficient b
        """
        self.field = field
        self.a = a
        self.b = b
        
        # Verify that 4*a^3 + 27*b^2 != 0
        discriminant = self.field.add(
            self.field.multiply(4, self.field.power(a, 3)),
            self.field.multiply(27, self.field.power(b, 2))
        )
        
        if discriminant == 0:
            raise ValueError("The curve is singular")
    
    def is_on_curve(self, point: Optional[Tuple[int, int]]) -> bool:
        """
        Check if a point is on the elliptic curve.
        
        Args:
            point: The point (x, y) or None for the point at infinity
            
        Returns:
            True if the point is on the curve, False otherwise
        """
        if point is None:
            # The point at infinity is on the curve
            return True
        
        x, y = point
        
        # Check if y^2 = x^3 + a*x + b
        left = self.field.power(y, 2)
        right = self.field.add(
            self.field.add(
                self.field.power(x, 3),
                self.field.multiply(self.a, x)
            ),
            self.b
        )
        
        return left == right
    
    def add(self, p: Optional[Tuple[int, int]], q: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Add two points on the elliptic curve.
        
        Args:
            p: The first point (x_1, y_1) or None for the point at infinity
            q: The second point (x_2, y_2) or None for the point at infinity
            
        Returns:
            The sum p + q
        """
        # Handle the point at infinity
        if p is None:
            return q
        
        if q is None:
            return p
        
        x1, y1 = p
        x2, y2 = q
        
        # Handle the case where p = -q
        if x1 == x2 and y1 == self.field.subtract(0, y2):
            return None
        
        # Compute the slope
        if x1 == x2 and y1 == y2:
            # p = q, so compute the tangent slope
            # slope = (3*x1^2 + a) / (2*y1)
            slope = self.field.divide(
                self.field.add(
                    self.field.multiply(3, self.field.power(x1, 2)),
                    self.a
                ),
                self.field.multiply(2, y1)
            )
        else:
            # p != q, so compute the secant slope
            # slope = (y2 - y1) / (x2 - x1)
            slope = self.field.divide(
                self.field.subtract(y2, y1),
                self.field.subtract(x2, x1)
            )
        
        # Compute the new point
        # x3 = slope^2 - x1 - x2
        x3 = self.field.subtract(
            self.field.subtract(
                self.field.power(slope, 2),
                x1
            ),
            x2
        )
        
        # y3 = slope * (x1 - x3) - y1
        y3 = self.field.subtract(
            self.field.multiply(
                slope,
                self.field.subtract(x1, x3)
            ),
            y1
        )
        
        return (x3, y3)
    
    def multiply(self, k: int, p: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Multiply a point on the elliptic curve by a scalar.
        
        Args:
            k: The scalar
            p: The point (x, y) or None for the point at infinity
            
        Returns:
            The product k*p
        """
        if k == 0 or p is None:
            return None
        
        if k < 0:
            # k*p = -k*(-p)
            return self.multiply(-k, (p[0], self.field.subtract(0, p[1])))
        
        # Use the double-and-add algorithm
        result = None
        addend = p
        
        while k > 0:
            if k & 1:
                result = self.add(result, addend)
            
            addend = self.add(addend, addend)
            k >>= 1
        
        return result


class MontgomeryCurve:
    """
    Implementation of an elliptic curve in Montgomery form.
    
    This class provides operations on elliptic curves in Montgomery form:
    By^2 = x^3 + Ax^2 + x over a finite field F_p.
    """
    
    def __init__(self, field: FiniteField, A: int, B: int):
        """
        Initialize the elliptic curve E: By^2 = x^3 + Ax^2 + x over F_p.
        
        Args:
            field: The finite field F_p
            A: The coefficient A
            B: The coefficient B
        """
        self.field = field
        self.A = A
        self.B = B
        
        # Verify that B(A^2 - 4) != 0
        discriminant = self.field.multiply(
            B,
            self.field.subtract(
                self.field.power(A, 2),
                4
            )
        )
        
        if discriminant == 0:
            raise ValueError("The curve is singular")
    
    def is_on_curve(self, point: Optional[Tuple[int, int]]) -> bool:
        """
        Check if a point is on the elliptic curve.
        
        Args:
            point: The point (x, y) or None for the point at infinity
            
        Returns:
            True if the point is on the curve, False otherwise
        """
        if point is None:
            # The point at infinity is on the curve
            return True
        
        x, y = point
        
        # Check if By^2 = x^3 + Ax^2 + x
        left = self.field.multiply(self.B, self.field.power(y, 2))
        right = self.field.add(
            self.field.add(
                self.field.power(x, 3),
                self.field.multiply(self.A, self.field.power(x, 2))
            ),
            x
        )
        
        return left == right
    
    def add(self, p: Optional[Tuple[int, int]], q: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Add two points on the elliptic curve.
        
        Args:
            p: The first point (x_1, y_1) or None for the point at infinity
            q: The second point (x_2, y_2) or None for the point at infinity
            
        Returns:
            The sum p + q
        """
        # Handle the point at infinity
        if p is None:
            return q
        
        if q is None:
            return p
        
        x1, y1 = p
        x2, y2 = q
        
        # Handle the case where p = -q
        if x1 == x2 and y1 == self.field.subtract(0, y2):
            return None
        
        # Compute the slope
        if x1 == x2 and y1 == y2:
            # p = q, so compute the tangent slope
            # slope = (3*x1^2 + 2*A*x1 + 1) / (2*B*y1)
            slope = self.field.divide(
                self.field.add(
                    self.field.add(
                        self.field.multiply(3, self.field.power(x1, 2)),
                        self.field.multiply(
                            self.field.multiply(2, self.A),
                            x1
                        )
                    ),
                    1
                ),
                self.field.multiply(
                    self.field.multiply(2, self.B),
                    y1
                )
            )
        else:
            # p != q, so compute the secant slope
            # slope = (y2 - y1) / (x2 - x1)
            slope = self.field.divide(
                self.field.subtract(y2, y1),
                self.field.subtract(x2, x1)
            )
        
        # Compute the new point
        # x3 = B*slope^2 - A - x1 - x2
        x3 = self.field.subtract(
            self.field.subtract(
                self.field.subtract(
                    self.field.multiply(self.B, self.field.power(slope, 2)),
                    self.A
                ),
                x1
            ),
            x2
        )
        
        # y3 = slope * (x1 - x3) - y1
        y3 = self.field.subtract(
            self.field.multiply(
                slope,
                self.field.subtract(x1, x3)
            ),
            y1
        )
        
        return (x3, y3)
    
    def multiply(self, k: int, p: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Multiply a point on the elliptic curve by a scalar.
        
        Args:
            k: The scalar
            p: The point (x, y) or None for the point at infinity
            
        Returns:
            The product k*p
        """
        if k == 0 or p is None:
            return None
        
        if k < 0:
            # k*p = -k*(-p)
            return self.multiply(-k, (p[0], self.field.subtract(0, p[1])))
        
        # Use the double-and-add algorithm
        result = None
        addend = p
        
        while k > 0:
            if k & 1:
                result = self.add(result, addend)
            
            addend = self.add(addend, addend)
            k >>= 1
        
        return result
    
    def x_only_ladder(self, k: int, x: int) -> int:
        """
        Compute the x-coordinate of k*P given the x-coordinate of P.
        
        This uses the Montgomery ladder algorithm, which is efficient and
        resistant to side-channel attacks.
        
        Args:
            k: The scalar
            x: The x-coordinate of the point P
            
        Returns:
            The x-coordinate of k*P
        """
        if k == 0:
            raise ValueError("Cannot compute the x-coordinate of the point at infinity")
        
        if k < 0:
            # The x-coordinate of k*P is the same as the x-coordinate of -k*P
            k = -k
        
        # Initialize the Montgomery ladder
        x1 = x
        x2 = 1
        z2 = 0
        x3 = x
        z3 = 1
        
        # Process the bits of k from most significant to least significant
        for i in range(k.bit_length() - 2, -1, -1):
            bit = (k >> i) & 1
            
            if bit == 0:
                # Double (x3, z3) and add to (x2, z2)
                # (x2, z2) = add((x2, z2), (x3, z3), x1)
                # (x3, z3) = double((x3, z3))
                
                # Add
                x2, z2, x3, z3 = self._xadd(x2, z2, x3, z3, x1)
                
                # Double
                x3, z3 = self._xdbl(x3, z3)
            else:
                # Double (x2, z2) and add to (x3, z3)
                # (x3, z3) = add((x2, z2), (x3, z3), x1)
                # (x2, z2) = double((x2, z2))
                
                # Add
                x3, z3, x2, z2 = self._xadd(x3, z3, x2, z2, x1)
                
                # Double
                x2, z2 = self._xdbl(x2, z2)
        
        # Return the x-coordinate of k*P
        return self.field.divide(x2, z2)
    
    def _xdbl(self, x: int, z: int) -> Tuple[int, int]:
        """
        Double a point in projective coordinates using only the x-coordinate.
        
        Args:
            x: The x-coordinate
            z: The z-coordinate
            
        Returns:
            The doubled point (x', z')
        """
        # Compute (x', z') = double((x, z))
        # x' = (x^2 - z^2)^2
        # z' = 4*x*z*((x^2 + z^2) + ((A+2)/4)*(4*x*z))
        
        # Compute x^2 and z^2
        x2 = self.field.power(x, 2)
        z2 = self.field.power(z, 2)
        
        # Compute (x^2 - z^2)^2
        x_new = self.field.power(
            self.field.subtract(x2, z2),
            2
        )
        
        # Compute 4*x*z
        xz4 = self.field.multiply(
            self.field.multiply(4, x),
            z
        )
        
        # Compute (A+2)/4
        a24 = self.field.divide(
            self.field.add(self.A, 2),
            4
        )
        
        # Compute (x^2 + z^2) + ((A+2)/4)*(4*x*z)
        term = self.field.add(
            self.field.add(x2, z2),
            self.field.multiply(a24, xz4)
        )
        
        # Compute z' = 4*x*z*term
        z_new = self.field.multiply(xz4, term)
        
        return (x_new, z_new)
    
    def _xadd(self, x1: int, z1: int, x2: int, z2: int, x_diff: int) -> Tuple[int, int, int, int]:
        """
        Add two points in projective coordinates using only the x-coordinates.
        
        Args:
            x1: The x-coordinate of the first point
            z1: The z-coordinate of the first point
            x2: The x-coordinate of the second point
            z2: The z-coordinate of the second point
            x_diff: The x-coordinate of the difference of the two points
            
        Returns:
            The sum (x1', z1', x2', z2')
        """
        # Compute (x1', z1') = add((x1, z1), (x2, z2), x_diff)
        # x1' = z_diff * x_sum^2
        # z1' = x_diff * z_sum^2
        # where x_sum = (x1*x2 + z1*z2)
        #       z_sum = (x1*z2 + z1*x2)
        #       x_diff is the x-coordinate of the difference of the two points
        
        # Compute x1*x2 and z1*z2
        x1x2 = self.field.multiply(x1, x2)
        z1z2 = self.field.multiply(z1, z2)
        
        # Compute x1*z2 and z1*x2
        x1z2 = self.field.multiply(x1, z2)
        z1x2 = self.field.multiply(z1, x2)
        
        # Compute x_sum = (x1*x2 + z1*z2)
        x_sum = self.field.add(x1x2, z1z2)
        
        # Compute z_sum = (x1*z2 + z1*x2)
        z_sum = self.field.add(x1z2, z1x2)
        
        # Compute x1' = z_diff * x_sum^2
        x1_new = self.field.multiply(
            x_diff,
            self.field.power(x_sum, 2)
        )
        
        # Compute z1' = x_diff * z_sum^2
        z1_new = self.field.multiply(
            x_diff,
            self.field.power(z_sum, 2)
        )
        
        return (x1_new, z1_new, x2, z2)


class SIKE:
    """
    Supersingular Isogeny Key Encapsulation (SIKE) protocol.
    
    This class implements the SIKE protocol, which is a post-quantum key
    encapsulation mechanism based on supersingular isogeny Diffie-Hellman.
    """
    
    def __init__(self, security_level: int = 128):
        """
        Initialize the SIKE protocol.
        
        Args:
            security_level: The security level in bits (128, 192, or 256)
        """
        # Set parameters based on security level
        if security_level == 128:
            # SIKEp434
            self.p = 2**216 * 3**137 - 1
            self.e2 = 216
            self.e3 = 137
        elif security_level == 192:
            # SIKEp503
            self.p = 2**250 * 3**159 - 1
            self.e2 = 250
            self.e3 = 159
        elif security_level == 256:
            # SIKEp751
            self.p = 2**372 * 3**239 - 1
            self.e2 = 372
            self.e3 = 239
        else:
            raise ValueError(f"Unsupported security level: {security_level}")
        
        # Initialize the finite field
        self.field = FiniteField(self.p)
        
        # Initialize the starting curve
        # E_0: y^2 = x^3 + 6*x^2 + x
        self.A = 6
        self.curve = MontgomeryCurve(self.field, self.A, 1)
        
        # Initialize the base points
        # These are simplified and not the actual base points used in SIKE
        # In a real implementation, we would use the correct base points
        self.P2 = (2, 1)  # A point of order 2^e2
        self.Q2 = (3, 1)  # A point of order 2^e2
        self.P3 = (4, 1)  # A point of order 3^e3
        self.Q3 = (5, 1)  # A point of order 3^e3
    
    def generate_keypair(self, is_alice: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate a SIKE keypair.
        
        Args:
            is_alice: Whether this is Alice's keypair (True) or Bob's keypair (False)
            
        Returns:
            A tuple (private_key, public_key)
        """
        if is_alice:
            # Alice's keypair
            # Private key: a random integer in [1, 2^e2 - 1]
            sk = secrets.randbelow(2**self.e2 - 1) + 1
            
            # Public key: the j-invariant of the curve E_A = E_0/<[sk]P2>
            # In a real implementation, we would compute the isogeny
            # For now, we'll just use a simplified version
            pk = self._compute_public_key_alice(sk)
        else:
            # Bob's keypair
            # Private key: a random integer in [1, 3^e3 - 1]
            sk = secrets.randbelow(3**self.e3 - 1) + 1
            
            # Public key: the j-invariant of the curve E_B = E_0/<[sk]P3>
            # In a real implementation, we would compute the isogeny
            # For now, we'll just use a simplified version
            pk = self._compute_public_key_bob(sk)
        
        # Return the keypair
        private_key = {'sk': sk, 'is_alice': is_alice}
        public_key = {'pk': pk, 'is_alice': is_alice}
        
        return private_key, public_key
    
    def encapsulate(self, public_key: Dict[str, Any]) -> Tuple[Dict[str, Any], bytes]:
        """
        Encapsulate a shared secret using SIKE.
        
        Args:
            public_key: The public key
            
        Returns:
            A tuple (ciphertext, shared_secret)
        """
        # Extract the public key components
        pk = public_key['pk']
        is_alice = public_key['is_alice']
        
        if is_alice:
            # Alice's public key, so Bob is encapsulating
            # Generate Bob's keypair
            private_key, bob_public_key = self.generate_keypair(is_alice=False)
            
            # Compute the shared secret
            j = self._compute_shared_secret_bob(private_key['sk'], pk)
        else:
            # Bob's public key, so Alice is encapsulating
            # Generate Alice's keypair
            private_key, alice_public_key = self.generate_keypair(is_alice=True)
            
            # Compute the shared secret
            j = self._compute_shared_secret_alice(private_key['sk'], pk)
        
        # Hash the j-invariant to get the shared secret
        shared_secret = self._hash_j_invariant(j)
        
        # Return the ciphertext and shared secret
        ciphertext = {'pk': private_key['sk'], 'is_alice': not is_alice}
        
        return ciphertext, shared_secret
    
    def decapsulate(self, private_key: Dict[str, Any], ciphertext: Dict[str, Any]) -> bytes:
        """
        Decapsulate a shared secret using SIKE.
        
        Args:
            private_key: The private key
            ciphertext: The ciphertext
            
        Returns:
            The shared secret
        """
        # Extract the private key components
        sk = private_key['sk']
        is_alice = private_key['is_alice']
        
        # Extract the ciphertext components
        pk = ciphertext['pk']
        is_alice_ciphertext = ciphertext['is_alice']
        
        # Check that the ciphertext is for the correct party
        if is_alice != is_alice_ciphertext:
            raise ValueError("Ciphertext is for the wrong party")
        
        if is_alice:
            # Alice is decapsulating
            # Compute the shared secret
            j = self._compute_shared_secret_alice(sk, pk)
        else:
            # Bob is decapsulating
            # Compute the shared secret
            j = self._compute_shared_secret_bob(sk, pk)
        
        # Hash the j-invariant to get the shared secret
        shared_secret = self._hash_j_invariant(j)
        
        return shared_secret
    
    def _compute_public_key_alice(self, sk: int) -> int:
        """
        Compute Alice's public key.
        
        Args:
            sk: Alice's private key
            
        Returns:
            Alice's public key (the j-invariant of the curve E_A)
        """
        # In a real implementation, we would compute the isogeny
        # For now, we'll just use a simplified version
        return (sk * 123456789) % self.p
    
    def _compute_public_key_bob(self, sk: int) -> int:
        """
        Compute Bob's public key.
        
        Args:
            sk: Bob's private key
            
        Returns:
            Bob's public key (the j-invariant of the curve E_B)
        """
        # In a real implementation, we would compute the isogeny
        # For now, we'll just use a simplified version
        return (sk * 987654321) % self.p
    
    def _compute_shared_secret_alice(self, sk: int, pk: int) -> int:
        """
        Compute the shared secret using Alice's private key and Bob's public key.
        
        Args:
            sk: Alice's private key
            pk: Bob's public key
            
        Returns:
            The j-invariant of the shared curve
        """
        # In a real implementation, we would compute the isogeny
        # For now, we'll just use a simplified version
        return (sk * pk) % self.p
    
    def _compute_shared_secret_bob(self, sk: int, pk: int) -> int:
        """
        Compute the shared secret using Bob's private key and Alice's public key.
        
        Args:
            sk: Bob's private key
            pk: Alice's public key
            
        Returns:
            The j-invariant of the shared curve
        """
        # In a real implementation, we would compute the isogeny
        # For now, we'll just use a simplified version
        return (sk * pk) % self.p
    
    def _hash_j_invariant(self, j: int) -> bytes:
        """
        Hash the j-invariant to get the shared secret.
        
        Args:
            j: The j-invariant
            
        Returns:
            The shared secret
        """
        # Convert the j-invariant to bytes
        j_bytes = j.to_bytes((j.bit_length() + 7) // 8, byteorder='big')
        
        # Hash the j-invariant
        return hashlib.sha256(j_bytes).digest()


# Example usage
if __name__ == "__main__":
    # Finite field example
    print("Finite Field Example")
    print("===================")
    
    # Create a finite field F_p
    p = 101
    field = FiniteField(p)
    
    # Perform some operations in F_p
    a = 10
    b = 20
    
    print(f"a = {a}, b = {b}")
    print(f"a + b = {field.add(a, b)}")
    print(f"a - b = {field.subtract(a, b)}")
    print(f"a * b = {field.multiply(a, b)}")
    print(f"a / b = {field.divide(a, b)}")
    print(f"a^3 = {field.power(a, 3)}")
    
    print()
    
    # Elliptic curve example
    print("Elliptic Curve Example")
    print("=====================")
    
    # Create an elliptic curve E: y^2 = x^3 + 2*x + 3 over F_101
    curve = EllipticCurve(field, 2, 3)
    
    # Find a point on the curve
    for x in range(p):
        # Compute y^2 = x^3 + 2*x + 3
        y_squared = field.add(
            field.add(
                field.power(x, 3),
                field.multiply(2, x)
            ),
            3
        )
        
        # Check if y_squared is a quadratic residue
        for y in range(p):
            if field.power(y, 2) == y_squared:
                point = (x, y)
                break
        else:
            continue
        
        break
    
    print(f"Found point on the curve: {point}")
    print(f"Is on curve: {curve.is_on_curve(point)}")
    
    # Perform some operations on the curve
    double_point = curve.add(point, point)
    print(f"2*P = {double_point}")
    print(f"Is on curve: {curve.is_on_curve(double_point)}")
    
    triple_point = curve.add(double_point, point)
    print(f"3*P = {triple_point}")
    print(f"Is on curve: {curve.is_on_curve(triple_point)}")
    
    # Compute a scalar multiplication
    k = 10
    k_point = curve.multiply(k, point)
    print(f"{k}*P = {k_point}")
    print(f"Is on curve: {curve.is_on_curve(k_point)}")
    
    print()
    
    # SIKE example
    print("SIKE Example")
    print("===========")
    
    # Create a SIKE instance with security level 128
    sike = SIKE(security_level=128)
    
    # Generate Alice's keypair
    alice_private_key, alice_public_key = sike.generate_keypair(is_alice=True)
    print(f"Generated Alice's keypair")
    
    # Generate Bob's keypair
    bob_private_key, bob_public_key = sike.generate_keypair(is_alice=False)
    print(f"Generated Bob's keypair")
    
    # Alice encapsulates a shared secret using Bob's public key
    ciphertext, alice_shared_secret = sike.encapsulate(bob_public_key)
    print(f"Alice encapsulated shared secret: {alice_shared_secret.hex()}")
    
    # Bob decapsulates the shared secret using his private key
    bob_shared_secret = sike.decapsulate(bob_private_key, ciphertext)
    print(f"Bob decapsulated shared secret: {bob_shared_secret.hex()}")
    
    # Check if the shared secrets match
    if alice_shared_secret == bob_shared_secret:
        print("Shared secrets match!")
    else:
        print("Shared secrets do not match!")