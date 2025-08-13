#!/usr/bin/env python3
"""
Example script demonstrating the usage of the ECDLP solver.
"""

import argparse
import time
from tsams_cryptography.ecdlp import ClassicalECDLPSolver
from tsams_cryptography.utils.elliptic_curve import Point


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Solve the ECDLP for a given public key.')
    parser.add_argument('--curve', type=str, default='secp256k1',
                        help='The elliptic curve to use (default: secp256k1)')
    parser.add_argument('--public-key', type=str, required=True,
                        help='The public key point in the format "x,y"')
    parser.add_argument('--base-point', type=str, default=None,
                        help='The base point in the format "x,y" (default: curve generator)')
    parser.add_argument('--timeout', type=float, default=None,
                        help='The maximum time to spend on the computation in seconds')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel computation')
    parser.add_argument('--memory-optimized', action='store_true',
                        help='Use memory optimization')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='The maximum number of worker processes to use')
    args = parser.parse_args()
    
    # Parse the public key
    try:
        x, y = map(int, args.public_key.split(','))
        public_key = (x, y)
    except ValueError:
        print(f"Error: Invalid public key format: {args.public_key}")
        print("The public key should be in the format 'x,y'")
        return 1
    
    # Parse the base point
    base_point = None
    if args.base_point is not None:
        try:
            x, y = map(int, args.base_point.split(','))
            base_point = (x, y)
        except ValueError:
            print(f"Error: Invalid base point format: {args.base_point}")
            print("The base point should be in the format 'x,y'")
            return 1
    
    # Initialize the solver
    solver = ClassicalECDLPSolver(
        curve=args.curve,
        parallel=args.parallel,
        memory_optimized=args.memory_optimized,
        max_workers=args.max_workers
    )
    
    # Solve the ECDLP
    print(f"Solving ECDLP for public key {public_key} on curve {args.curve}...")
    start_time = time.time()
    try:
        private_key = solver.solve(public_key, base_point, args.timeout)
        elapsed_time = time.time() - start_time
        print(f"Private key: {private_key}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
    except TimeoutError:
        elapsed_time = time.time() - start_time
        print(f"Timeout after {elapsed_time:.2f} seconds")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())