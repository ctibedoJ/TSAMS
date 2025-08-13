"""
TSC Solver Implementation

The TSC Solver integrates the Throw, Shot, and Catch phases to solve the
Elliptic Curve Discrete Logarithm Problem (ECDLP) in linear time.
"""

from .throw import ThrowPhase
from .shot import ShotPhase
from .catch import CatchPhase


class TSCSolver:
    """
    Implementation of the complete Throw-Shot-Catch (TSC) Algorithm.
    
    The TSC Solver integrates the Throw, Shot, and Catch phases to solve the
    Elliptic Curve Discrete Logarithm Problem (ECDLP) in linear time.
    """
    
    def __init__(self):
        """Initialize the TSCSolver object."""
        self.throw_phase = ThrowPhase()
        self.shot_phase = ShotPhase()
        self.catch_phase = CatchPhase()
        self.result = None
    
    def solve(self, P, Q, curve_params):
        """
        Solve the ECDLP using the TSC algorithm.
        
        Args:
            P (tuple): The coordinates of the base point P (x₁, y₁).
            Q (tuple): The coordinates of the point Q (x₂, y₂).
            curve_params (dict): The parameters of the elliptic curve.
                Should contain 'a', 'b', 'p', and 'n' (order of P).
                
        Returns:
            int: The discrete logarithm k such that Q = k*P.
        """
        # Execute the Throw Phase
        throw_state = self.throw_phase.initialize(P, Q, curve_params)
        
        # Execute the Shot Phase
        shot_state = self.shot_phase.execute(throw_state)
        
        # Execute the Catch Phase
        self.result = self.catch_phase.execute(shot_state, curve_params)
        
        # Return the discrete logarithm
        return self.result['discrete_logarithm']
    
    def get_detailed_result(self):
        """
        Get the detailed result of the TSC algorithm.
        
        Returns:
            dict: The detailed result, including the discrete logarithm,
                  verification status, and intermediate states.
        """
        if self.result is None:
            raise ValueError("The TSC algorithm has not been executed yet.")
        
        return self.result
    
    def get_performance_metrics(self):
        """
        Get performance metrics for the TSC algorithm execution.
        
        Returns:
            dict: Performance metrics including time complexity analysis.
        """
        if self.result is None:
            raise ValueError("The TSC algorithm has not been executed yet.")
        
        # Extract relevant parameters from the result
        final_state = self.result['final_state']
        bit_length = final_state.get('bit_length', 0)
        throw_depth = final_state.get('throw_depth', 0)
        shot_depth = final_state.get('shot_depth', 0)
        transformation_count = final_state.get('transformation_count', 0)
        
        # Calculate theoretical time complexity
        theoretical_complexity = bit_length  # O(n) where n is the bit length
        
        # Calculate actual operations performed
        actual_operations = throw_depth + transformation_count + 1  # +1 for the catch phase
        
        # Return the metrics
        return {
            'bit_length': bit_length,
            'theoretical_complexity': theoretical_complexity,
            'actual_operations': actual_operations,
            'throw_depth': throw_depth,
            'shot_depth': shot_depth,
            'transformation_count': transformation_count,
            'complexity_ratio': actual_operations / theoretical_complexity if theoretical_complexity > 0 else float('inf')
        }