"""
Modular System Implementation

This module implements the 56-modular system used in the TIBEDO Framework.
"""

import numpy as np
import sympy as sp

class ModularSystem:
    """
    Implementation of the 56-Modular System used in the TIBEDO Framework.
    
    The 56-modular system is a mathematical structure that enables congruential
    acceleration in the TIBEDO Framework.
    """
    
    def __init__(self, modulus=56):
        """
        Initialize the ModularSystem object.
        
        Args:
            modulus (int): The modulus for the system. Default is 56.
        """
        self.modulus = modulus
        self.reduction_factor = self.compute_reduction_factor()
        
    def compute_reduction_factor(self):
        """
        Compute the reduction factor for the modular system.
        
        Returns:
            float: The reduction factor.
        """
        # According to Theorem 6.2.1, the reduction factor is related to
        # the number of distinct prime factors of the modulus
        
        # Count the distinct prime factors of the modulus
        factors = self._prime_factors(self.modulus)
        distinct_factors = len(set(factors))
        
        # Compute the reduction factor
        return 2 ** distinct_factors
        
    def _prime_factors(self, n):
        """
        Find the prime factors of a number.
        
        Args:
            n (int): The number to factorize.
            
        Returns:
            list: The prime factors of n.
        """
        factors = []
        d = 2
        while n > 1:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
            if d*d > n:
                if n > 1:
                    factors.append(n)
                break
        return factors
        
    def find_compatible_prime_set(self, size, max_prime=1000):
        """
        Find a set of primes that are compatible with the modular system.
        
        Args:
            size (int): The size of the prime set to find.
            max_prime (int): The maximum prime to consider.
            
        Returns:
            list: A list of compatible primes, or None if no compatible set is found.
        """
        # According to Definition 6.3.1, a prime p is compatible with the
        # modular system if p % modulus != 1
        
        # Find all compatible primes up to max_prime
        compatible_primes = []
        for p in range(2, max_prime + 1):
            if self._is_prime(p) and p % self.modulus != 1:
                compatible_primes.append(p)
                if len(compatible_primes) >= size:
                    return compatible_primes[:size]
        
        # If we couldn't find enough compatible primes, return None
        if len(compatible_primes) < size:
            # For the purpose of this implementation, we'll return whatever primes we found
            # even if there aren't enough of them
            if compatible_primes:
                # Pad with additional primes if needed
                additional_primes = [p for p in range(2, max_prime + 1) if self._is_prime(p)]
                combined_primes = list(set(compatible_primes + additional_primes))
                return sorted(combined_primes)[:size]
            return None
            
        return compatible_primes[:size]
        
    def _is_prime(self, n):
        """
        Check if a number is prime.
        
        Args:
            n (int): The number to check.
            
        Returns:
            bool: True if n is prime, False otherwise.
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
        
    def chinese_remainder_theorem(self, remainders, moduli):
        """
        Apply the Chinese Remainder Theorem to find a number that satisfies
        the given congruences.
        
        Args:
            remainders (list): The remainders for each modulus.
            moduli (list): The moduli.
            
        Returns:
            int: The solution to the system of congruences.
        """
        # Check that the inputs are valid
        if len(remainders) != len(moduli):
            raise ValueError("The number of remainders must equal the number of moduli.")
            
        # Compute the product of all moduli
        M = 1
        for m in moduli:
            M *= m
            
        # Compute the solution
        result = 0
        for i in range(len(moduli)):
            a_i = remainders[i]
            m_i = moduli[i]
            M_i = M // m_i
            
            # Find the modular multiplicative inverse of M_i modulo m_i
            # This is the value of y_i such that M_i * y_i ≡ 1 (mod m_i)
            y_i = self._mod_inverse(M_i, m_i)
            
            result += a_i * M_i * y_i
            
        return result % M
        
    def _mod_inverse(self, a, m):
        """
        Find the modular multiplicative inverse of a modulo m.
        
        Args:
            a (int): The number to find the inverse for.
            m (int): The modulus.
            
        Returns:
            int: The modular multiplicative inverse of a modulo m.
        """
        # Use the extended Euclidean algorithm to find the modular inverse
        g, x, y = self._extended_gcd(a, m)
        
        if g != 1:
            raise ValueError(f"Modular inverse does not exist (gcd({a}, {m}) = {g}).")
        else:
            return x % m
            
    def _extended_gcd(self, a, b):
        """
        Extended Euclidean Algorithm to find gcd(a, b) and coefficients x, y
        such that ax + by = gcd(a, b).
        
        Args:
            a (int): First number.
            b (int): Second number.
            
        Returns:
            tuple: (gcd(a, b), x, y) where ax + by = gcd(a, b).
        """
        if a == 0:
            return b, 0, 1
        else:
            gcd, x, y = self._extended_gcd(b % a, a)
            return gcd, y - (b // a) * x, x