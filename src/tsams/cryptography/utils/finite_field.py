"""
Finite Field implementation.

This module provides an implementation of finite fields,
which are used in the ECDLP solver.
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np


class FieldElement:
    """
    A class representing an element in a finite field.
    
    Attributes:
        value (int): The value of the element.
        field (FiniteField): The field that the element belongs to.
    """
    
    def __init__(self, value: int, field: 'FiniteField'):
        """
        Initialize a field element.
        
        Args:
            value (int): The value of the element.
            field (FiniteField): The field that the element belongs to.
        
        Raises:
            ValueError: If the value is not in the field.
        """
        if value < 0 or value >= field.order:
            raise ValueError(f"Value {value} is not in the field of order {field.order}")
        
        self.value = value
        self.field = field
    
    def __add__(self, other: 'FieldElement') -> 'FieldElement':
        """
        Add two field elements.
        
        Args:
            other (FieldElement): The other field element.
        
        Returns:
            FieldElement: The sum of the two field elements.
        
        Raises:
            ValueError: If the elements are from different fields.
        """
        if self.field != other.field:
            raise ValueError("Cannot add elements from different fields")
        
        return FieldElement((self.value + other.value) % self.field.order, self.field)
    
    def __sub__(self, other: 'FieldElement') -> 'FieldElement':
        """
        Subtract two field elements.
        
        Args:
            other (FieldElement): The other field element.
        
        Returns:
            FieldElement: The difference of the two field elements.
        
        Raises:
            ValueError: If the elements are from different fields.
        """
        if self.field != other.field:
            raise ValueError("Cannot subtract elements from different fields")
        
        return FieldElement((self.value - other.value) % self.field.order, self.field)
    
    def __mul__(self, other: 'FieldElement') -> 'FieldElement':
        """
        Multiply two field elements.
        
        Args:
            other (FieldElement): The other field element.
        
        Returns:
            FieldElement: The product of the two field elements.
        
        Raises:
            ValueError: If the elements are from different fields.
        """
        if self.field != other.field:
            raise ValueError("Cannot multiply elements from different fields")
        
        return FieldElement((self.value * other.value) % self.field.order, self.field)
    
    def __truediv__(self, other: 'FieldElement') -> 'FieldElement':
        """
        Divide two field elements.
        
        Args:
            other (FieldElement): The other field element.
        
        Returns:
            FieldElement: The quotient of the two field elements.
        
        Raises:
            ValueError: If the elements are from different fields.
            ZeroDivisionError: If the divisor is zero.
        """
        if self.field != other.field:
            raise ValueError("Cannot divide elements from different fields")
        
        if other.value == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        
        # Compute the modular inverse of the divisor
        inverse = other.inverse()
        
        return self * inverse
    
    def __pow__(self, exponent: int) -> 'FieldElement':
        """
        Raise a field element to a power.
        
        Args:
            exponent (int): The exponent.
        
        Returns:
            FieldElement: The field element raised to the power.
        """
        # Handle negative exponents
        if exponent < 0:
            return self.inverse() ** (-exponent)
        
        # Use the square-and-multiply algorithm for efficient exponentiation
        result = FieldElement(1, self.field)
        base = FieldElement(self.value, self.field)
        
        while exponent:
            if exponent & 1:
                result = result * base
            base = base * base
            exponent >>= 1
        
        return result
    
    def inverse(self) -> 'FieldElement':
        """
        Compute the multiplicative inverse of the field element.
        
        Returns:
            FieldElement: The multiplicative inverse of the field element.
        
        Raises:
            ZeroDivisionError: If the element is zero.
        """
        if self.value == 0:
            raise ZeroDivisionError("Cannot compute the inverse of zero")
        
        # Use Fermat's Little Theorem to compute the inverse
        # a^(p-1) ≡ 1 (mod p) for any non-zero a in the field
        # Therefore, a^(p-2) ≡ a^(-1) (mod p)
        return self ** (self.field.order - 2)
    
    def __eq__(self, other: 'FieldElement') -> bool:
        """
        Check if two field elements are equal.
        
        Args:
            other (FieldElement): The other field element.
        
        Returns:
            bool: True if the field elements are equal, False otherwise.
        """
        if not isinstance(other, FieldElement):
            return False
        
        return self.value == other.value and self.field == other.field
    
    def __ne__(self, other: 'FieldElement') -> bool:
        """
        Check if two field elements are not equal.
        
        Args:
            other (FieldElement): The other field element.
        
        Returns:
            bool: True if the field elements are not equal, False otherwise.
        """
        return not self.__eq__(other)
    
    def __str__(self) -> str:
        """
        Return a string representation of the field element.
        
        Returns:
            str: A string representation of the field element.
        """
        return str(self.value)
    
    def __repr__(self) -> str:
        """
        Return a string representation of the field element.
        
        Returns:
            str: A string representation of the field element.
        """
        return f"FieldElement({self.value}, {self.field})"


class FiniteField:
    """
    A class representing a finite field.
    
    Attributes:
        order (int): The order of the field.
        is_prime (bool): Whether the order of the field is prime.
    """
    
    def __init__(self, order: int):
        """
        Initialize a finite field.
        
        Args:
            order (int): The order of the field.
        
        Raises:
            ValueError: If the order is not a prime power.
        """
        self.order = order
        
        # Check if the order is a prime power
        self.is_prime = self._is_prime(order)
        
        if not self.is_prime:
            # Check if the order is a prime power
            factors = self._prime_factors(order)
            if len(set(factors)) != 1:
                raise ValueError(f"Order {order} is not a prime power")
    
    def _is_prime(self, n: int) -> bool:
        """
        Check if a number is prime.
        
        Args:
            n (int): The number to check.
        
        Returns:
            bool: True if the number is prime, False otherwise.
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
    
    def _prime_factors(self, n: int) -> List[int]:
        """
        Compute the prime factorization of a number.
        
        Args:
            n (int): The number to factorize.
        
        Returns:
            List[int]: The prime factors of the number.
        """
        factors = []
        
        # Check for divisibility by 2
        while n % 2 == 0:
            factors.append(2)
            n //= 2
        
        # Check for divisibility by odd numbers
        i = 3
        while i * i <= n:
            while n % i == 0:
                factors.append(i)
                n //= i
            i += 2
        
        # If n is a prime greater than 2
        if n > 2:
            factors.append(n)
        
        return factors
    
    def element(self, value: int) -> FieldElement:
        """
        Create a field element with the given value.
        
        Args:
            value (int): The value of the element.
        
        Returns:
            FieldElement: The field element.
        """
        return FieldElement(value % self.order, self)
    
    def zero(self) -> FieldElement:
        """
        Get the zero element of the field.
        
        Returns:
            FieldElement: The zero element.
        """
        return FieldElement(0, self)
    
    def one(self) -> FieldElement:
        """
        Get the one element of the field.
        
        Returns:
            FieldElement: The one element.
        """
        return FieldElement(1, self)
    
    def elements(self) -> List[FieldElement]:
        """
        Get all elements of the field.
        
        Returns:
            List[FieldElement]: All elements of the field.
        """
        return [FieldElement(i, self) for i in range(self.order)]
    
    def __eq__(self, other: 'FiniteField') -> bool:
        """
        Check if two fields are equal.
        
        Args:
            other (FiniteField): The other field.
        
        Returns:
            bool: True if the fields are equal, False otherwise.
        """
        if not isinstance(other, FiniteField):
            return False
        
        return self.order == other.order
    
    def __ne__(self, other: 'FiniteField') -> bool:
        """
        Check if two fields are not equal.
        
        Args:
            other (FiniteField): The other field.
        
        Returns:
            bool: True if the fields are not equal, False otherwise.
        """
        return not self.__eq__(other)
    
    def __str__(self) -> str:
        """
        Return a string representation of the field.
        
        Returns:
            str: A string representation of the field.
        """
        return f"FiniteField({self.order})"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the field.
        
        Returns:
            str: A string representation of the field.
        """
        return self.__str__()